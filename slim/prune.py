import torch
import torch.nn as nn
from .sparsegpt import SparseGPT
from .sparsegpt import Quantizer as SparseGPTQuantizer
from .layerwrapper import WrappedGPT
from .data import get_loaders
from .utils import get_layers_list, shift_zeros, find_layers, prune_nm
from .lora import add_lora
from slim.quantization.quantization import Quantizer as AutoQuantizer, QuantizedMatmul
import tqdm.auto as tqdm
from .jsq_utils import clip_matrix, generate_ss
from .smooth import smooth_layer
from huggingface_hub import hf_hub_download
import numpy as np
import gc
import jax
import jax.numpy as jnp
from mpax import create_qp, raPDHG
import numpy as np
from mpax.utils import TerminationStatus
from .mask_optim import block_wise_optimize_mask


def prepare_calibration_input(
        model,
        dataloader,
        nsamples=128
):
    """
    Prepare inputs for calibration.

    Args:
        model: torch.nn.Module - The model to calibrate
        dataloader: torch.utils.data.DataLoader - The dataloader to use for calibration

    Returns:
        inps: torch.Tensor - The input tensor for calibration
        outs: torch.Tensor - The output tensor for calibration
        attention_mask: torch.Tensor - The attention mask for calibration
    """
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False
    layers = get_layers_list(model)


    dtype = next(iter(model.parameters())).dtype
    torch.cuda.empty_cache()
    inps = torch.zeros((nsamples, model.config.max_position_embeddings, model.config.hidden_size), dtype=dtype, device="cpu")
    input_device = "cpu"
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.to(input_device)
            cache['i'] += 1
            for key in kwargs:
                cache[key] = kwargs[key]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0])
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    model.config.use_cache = use_cache
    del cache['i']
    return inps, outs, cache


def prune_magnitude(
        model,
        sparsity_ratio,
        prune_n=0,
        prune_m=0,
        quantize_weight=False,
        bitwidth=4,
        slim_quant=False,
        tiled_weight_quantization=False,
        weight_tile_size=256,
):
    """
    Prune a model using magnitude pruning and quantize weights using SLiM-Quant or AbsMax.

    Args:
        model: torch.nn.Module - The model to prune
        sparsity_ratio: float - The ratio of weights to prune
        prune_n: int - The number N in N:M pruning
        prune_m: int - The number M in N:M pruning
        quantize_weight: bool - Whether to quantize weights
        bitwidth: int - The bitwidth to use for quantization
        slim_quant: bool - Whether to use SLiM-Quant
        tiled_weight_quantization: bool - Whether to use block quantization
        weight_tile_size: int - The size of the blocks for block quantization

    Returns:
        None
    """
    layers = get_layers_list(model)
    progress_bar = tqdm.tqdm(range(len(layers)))

    if quantize_weight:
        quantizer = AutoQuantizer(
            "weight",
            num_bits=bitwidth,
            slim_quant=slim_quant,
            block_quantization=tiled_weight_quantization,
            block_dim=weight_tile_size,
        )
    else:
        quantizer = None

    for i in progress_bar:
        progress_bar.set_description(f"Layer {i}")
        layer = layers[i].cuda()
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = prune_nm(W_metric, prune_n, prune_m)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel() * sparsity_ratio)].cpu()
                W_mask = (W_metric <= thresh)

            W[W_mask] = 0
            subset[name].weight.data = W
            if quantizer is not None:
                quantized_weight = quantizer.quantize_weight(subset[name].weight.data)
                subset[name].weight.data = quantizer.dequantize_absmax(quantized_weight).to(torch.bfloat16)
                subset[name].register_buffer("quantization_scaling_factor", quantizer.scaling_factor)
                if not tiled_weight_quantization:
                    subset[name].scaling_factor = quantizer.scaling_factor
                else:
                    subset[name].scaling_factor = None
        layer = layer.cpu()

def prune_wanda(
        model,
        tokenizer,
        sparsity_ratio=0.5,
        prune_n=0,
        prune_m=0,
        quantize_weight=False,
        bitwidth=4,
        slim_quant=False,
        tiled_weight_quantization=False,
        weight_tile_size=256,
        shift_zero_metrics=True,
        lora_rank=0.,
        slim_lora=True,
        prune_lora=False,
        quantize_lora=False,
        lora_tile_size=256,
        separate_lora=True,
        nsamples=128,
        seed=0,
        calibration_dataset="c4",
        pad_lora=False,
        quantize_first=True,
        scale_important_weights=False,
        update_weights=False,
        use_qp_solver=False,
        double_precision=False,
        update_mask=True,
):
    """
    Prune a model using WANDA and quantize weights using SLiM-Quant or AbsMax and add low-rank adapter using SLiM or SVD.

    Args:
        model: torch.nn.Module - The model to prune
        tokenizer: transformers.Tokenizer - The tokenizer for the model
        sparsity_ratio: float - The ratio of weights to prune
        prune_n: int - The number N in N:M pruning
        prune_m: int - The number M in N:M pruning
        quantize_weight: bool - Whether to quantize weights
        bitwidth: int - The bitwidth to use for quantization
        slim_quant: bool - Whether to use SLiM-Quant
        tiled_weight_quantization: bool -
        weight_tile_size: int - The size of the blocks for block quantization
        shift_zero_metrics: bool - Whether to shift zero metrics
        lora_rank: float - The rank ratio for low-rank adapter
        slim_lora: bool - Whether to use SLiM for low-rank adapter
        prune_lora: bool - Whether to prune the low-rank adapter
        quantize_lora: bool - Whether to quantize the low-rank adapter
        lora_tile_size: int - The size of the blocks for block quantization of the low-rank adapter
        separate_lora: bool - Whether to separate the low-rank adapter
        nsamples: int - The number of samples to use for calibration
        seed: int - The seed to use for calibration
        calibration_dataset: str - The dataset to use for calibration
        pad_lora: bool - Whether to pad the LoRA weights
        quantize_first: bool - Whether to quantize the weights before or after pruning
        scale_important_weights: bool - Whether to scale the important weights
        update_weights: bool - Whether to update weights during pruning
        use_qp_solver: bool - Whether to use quadratic programming solver
        double_precision: bool - Whether to use double precision for calculations

    Returns:
        None
    """
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    dataloader, _ = get_loaders(
        calibration_dataset,
        nsamples=nsamples,
        seed=seed,
        seqlen=model.config.max_position_embeddings,
        tokenizer=tokenizer
    )
    with torch.no_grad():
        inps, outs, kwargs = prepare_calibration_input(model, dataloader, nsamples)

    if quantize_weight:
        quantizer = AutoQuantizer(
            "weight",
            num_bits=bitwidth,
            slim_quant=slim_quant,
            block_quantization=tiled_weight_quantization,
            block_dim=weight_tile_size,
        )
    else:
        quantizer = None

    layers = get_layers_list(model)

    progress_bar = tqdm.tqdm(range(len(layers)))

    for i in progress_bar:
        progress_bar.set_description(f"Layer {i} - Gathering data")
        layer = layers[i].cuda()

        subset = find_layers(layer)
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
            if update_weights:
                wrapped_layers[name].inputs = []
                wrapped_layers[name].outputs = []
                wrapped_layers[name].original_weight = subset[name].weight.data.clone().detach()

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
                if update_weights:
                    wrapped_layers[name].inputs.append(inp[0].clone().detach().cpu())
                    wrapped_layers[name].outputs.append(out.clone().detach().cpu())

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            with torch.no_grad():
                for key in kwargs:
                    if isinstance(kwargs[key], torch.Tensor):
                        kwargs[key] = kwargs[key].cuda()
                    if isinstance(kwargs[key], tuple):
                        kwargs[key] = tuple([k.cuda() for k in kwargs[key]])
                outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(outs[j].device)

        for h in handles:
            h.remove()

        for name in subset:
            progress_bar.set_description(f"Layer {i} - Pruning and Quantizing {name}")
            if shift_zero_metrics:
                wrapped_layers[name].scaler_row = shift_zeros(wrapped_layers[name].scaler_row)

            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                if hasattr(subset[name].weight, 'mask'):
                    W_mask = subset[name].weight.mask
                    del subset[name].weight.mask
                else:
                    W_mask = prune_nm(W_metric, prune_n, prune_m)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                # unstructured pruning
                indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            if lora_rank > 0.:
                lora_tile_size = lora_tile_size if (quantize_lora or pad_lora) else None
                add_lora(subset[name],
                         W_mask=W_mask,
                         rank_ratio=lora_rank,
                         slim_lora=slim_lora,
                         activations=wrapped_layers[name],
                         quantizer=quantizer,
                         prune_lora=prune_lora,
                         separate_lora=separate_lora,
                         lora_tile_size=lora_tile_size,
                         quantize_first=quantize_first,
                         scale_important_weights=scale_important_weights
                         )

                if quantizer is not None:
                    subset[name].register_buffer("quantization_scaling_factor", quantizer.scaling_factor)
                    if not tiled_weight_quantization:
                        subset[name].scaling_factor = quantizer.scaling_factor
                    else:
                        subset[name].scaling_factor = None

                if separate_lora:
                    def add_lora_hook(module, input, output):
                        if hasattr(module, "lora_quantizer"):
                            xl = QuantizedMatmul.apply(
                                input[0].to(module.lora_left.dtype) / torch.sqrt(module.lora_rank),
                                module.lora_left,
                                module.lora_quantizer
                            )
                            xlr = QuantizedMatmul.apply(
                                xl / torch.sqrt(module.lora_rank),
                                module.lora_right,
                                module.lora_quantizer
                            )
                            output += xlr

                        else:
                            output += torch.matmul(
                                torch.matmul(input[0].to(module.lora_left.dtype),
                                             module.lora_left / torch.sqrt(module.lora_rank)),
                                module.lora_right / torch.sqrt(module.lora_rank))


                    subset[name].lora_rank = torch.tensor(subset[name].lora_left.shape[1])
                    subset[name].lora_left = torch.nn.Parameter(subset[name].lora_left * torch.sqrt(subset[name].lora_rank))
                    subset[name].lora_right = torch.nn.Parameter(subset[name].lora_right * torch.sqrt(subset[name].lora_rank))
                    subset[name].register_forward_hook(add_lora_hook)


            else:
                if scale_important_weights:
                    # Get 1% of largest activations
                    metric = subset[name].scaler_row * subset[name].weight.data.abs().sum(dim=0)
                    important_weights = metric.topk(
                        int(0.01 * metric.numel()), largest=True, sorted=False)[1]
                else:
                    important_weights = None
                if quantize_first:
                    if quantizer is not None:
                        quantized_weight = quantizer.quantize_weight(subset[name].weight.data, important_weights)
                        subset[name].weight.data = quantizer.dequantize_absmax(quantized_weight).to(torch.bfloat16)
                        if quantizer is not None:
                            subset[name].register_buffer("quantization_scaling_factor", quantizer.scaling_factor)
                            if not tiled_weight_quantization:
                                subset[name].scaling_factor = quantizer.scaling_factor
                            else:
                                subset[name].scaling_factor = None
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
                    if quantizer is not None:
                        quantized_weight = quantizer.quantize_weight(subset[name].weight.data, important_weights)
                        subset[name].weight.data = quantizer.dequantize_absmax(quantized_weight).to(torch.bfloat16)
                        if quantizer is not None:
                            subset[name].register_buffer("quantization_scaling_factor", quantizer.scaling_factor)
                            if not tiled_weight_quantization:
                                subset[name].scaling_factor = quantizer.scaling_factor
                            else:
                                subset[name].scaling_factor = None
            if update_weights:
                def compute_error(weight):
                    with torch.no_grad():
                        errors = []
                        for (x, y) in zip(wrapped_layers[name].inputs, wrapped_layers[name].outputs):
                            y_hat = torch.matmul(x.cuda(), weight.t())
                            errors.append(torch.nn.functional.mse_loss(y_hat, y.cuda()).item())
                    return np.mean(errors)
                init_loss = compute_error(subset[name].weight.cuda())
                if use_qp_solver:
                    with torch.no_grad():
                        device = "cpu"
                        input_cov = torch.zeros((wrapped_layers[name].inputs[0].shape[-1], wrapped_layers[name].inputs[0].shape[-1]), device="cuda")
                        for x in wrapped_layers[name].inputs:
                            x = x.view(-1, x.shape[-1]).cuda()
                            input_cov += torch.matmul(x.t(), x) / len(wrapped_layers[name].inputs) / x.shape[0]
                            x = x.cpu()
                        def single_optimize(c_vector, G_matrix, h_vector, l_vector, u_vector, Q_matrix, A_matrix, b_vector, eps_abs=1e-2):
                            """Optimize a single QP problem."""
                            # print("Q:", Q_matrix)
                            # print("c:", c_vector)
                            # print("A:", A_matrix)
                            # print("b:", b_vector)
                            # print("G:", G_matrix)
                            # print("h:", h_vector)
                            # print("l:", l_vector)
                            # print("u:", u_vector)
                            # exit()
                            qp = create_qp(Q_matrix, c_vector, A_matrix, b_vector, G_matrix, h_vector, l_vector, u_vector, use_sparse_matrix=False)
                            solver = raPDHG(eps_abs=eps_abs, eps_rel=1e-2, verbose=False, iteration_limit=100_000)  # Set verbose=False for batch processing
                            result = solver.optimize(qp)

                            # Calculate objective value: 1/2 x' Q x + c' x
                            obj = 0.5 * jnp.dot(result.primal_solution, jnp.dot(Q_matrix, result.primal_solution)) + jnp.dot(c_vector, result.primal_solution)
                            return result.primal_solution, obj, result.termination_status
                        dtype = torch.float64 if double_precision else torch.float32
                        jax.config.update("jax_enable_x64", double_precision)
                        weight = wrapped_layers[name].original_weight.clone().to(dtype).to(device)
                        # For each row of weight (w), minimize
                        # w @ input_cov @ w + 0
                        # s.t. w[w_mask] = 0
                        Q_torch = input_cov.to(device).to(dtype)
                        Q = jnp.asarray(Q_torch)  # Shared Q matrix
                        n_params = Q.shape[0]
                        
                        c_torch = torch.zeros(n_params, device=device, dtype=dtype)
                        c = jnp.asarray(c_torch)

                        A_torch = torch.zeros(1, n_params, device=device, dtype=dtype)
                        b_torch = torch.zeros(1, device=device, dtype=dtype)
                        A = jnp.asarray(A_torch)  # Zero A matrix
                        b = jnp.asarray(b_torch)  # Zero b vector

                        G_torch_mini = torch.ones((1, n_params), device=device, dtype=dtype)
                        h_torch_mini = -torch.ones((1), device=device, dtype=dtype) * 1e5
                        G = jnp.asarray(G_torch_mini)
                        h = jnp.asarray(h_torch_mini)
                                

                        batch_optimize = jax.vmap(single_optimize, in_axes=(None, None, None, 0, 0, None, None, None, None))

                        # Process in mini-batches
                        all_solutions = []
                        all_objectives = []

                        batch_size = weight.shape[0]
                        eps_abs = 1e-2
                        mini_batch_size = min(batch_size, max(1, 128 * 12 * 1024 * 1024 // (weight.shape[1] ** 2)))
                        tuning_progress_bar = tqdm.tqdm(range(0, batch_size, mini_batch_size), desc="Tuning Progress")
                        skip_layer = False

                        for start_idx in tuning_progress_bar:
                            end_idx = min(start_idx + mini_batch_size, batch_size)
                            current_mini_batch_size = end_idx - start_idx
                            
                            l_torch_mini = torch.full((current_mini_batch_size, n_params), -float('inf'), device=device, dtype=dtype)
                            u_torch_mini = torch.full((current_mini_batch_size, n_params), float('inf'), device=device, dtype=dtype)
                            
                            # For each sample in mini-batch, randomly select variables to have equality constraints
                            for row in range(current_mini_batch_size):
                                eq_indices = W_mask[start_idx + row, :].to(device)
                                l_torch_mini[row, eq_indices] = -weight[start_idx+row, :][eq_indices].clone()
                                u_torch_mini[row, eq_indices] = -weight[start_idx+row, :][eq_indices].clone() + 1e-4

                            # Convert mini-batch tensors to JAX arrays
                            l_mini = jnp.asarray(l_torch_mini)
                            u_mini = jnp.asarray(u_torch_mini)

                            torch.cuda.empty_cache()  # Clear GPU memory if using CUDA

                            # Solve mini-batch of QP problems
                            tuning_progress_bar.set_description(f"Solving mini-batch {start_idx // mini_batch_size + 1}/{(batch_size + mini_batch_size - 1) // mini_batch_size}")
                            tuning_progress_bar.set_postfix({"eps_abs": eps_abs})
                            solutions_mini, objectives_mini, termination_status_mini = batch_optimize(c, G, h, l_mini, u_mini, Q, A, b, eps_abs)
                            converged_vals = [status == TerminationStatus.OPTIMAL for status in termination_status_mini]
                            if sum(converged_vals) / len(converged_vals) < 0.95:
                                print("QP solver did not converge")
                                print("Termination status:", termination_status_mini)
                                print("Skipping Layer")
                                skip_layer = True
                                break

                            solutions_mini = np.asarray(solutions_mini)
                            solutions_mini = torch.from_numpy(solutions_mini).to(dtype=torch.bfloat16).to(device)
                            # print("R", solutions_mini)
                            # print("W", weight[start_idx, :])
                            # print("W + R", weight[start_idx, :] + solutions_mini)
                            all_solutions.append(solutions_mini)
                            all_objectives.append(objectives_mini)
                        if not skip_layer:
                            all_solutions = torch.cat(all_solutions, dim=0)
                            best_weight = weight.to(torch.bfloat16) + all_solutions
                            best_weight[W_mask] = 0  ## set weights to zero
                        else:
                            best_weight = weight.to(torch.bfloat16)
                            best_weight[W_mask] = 0  ## set weights to zero
                        best_weight = best_weight.cuda()

                        #Delete all the variables
                        del G_torch_mini, h_torch_mini, l_torch_mini, u_torch_mini
                        del G, h, l_mini, u_mini
                        del c_torch, c, A_torch, b_torch, A, b
                        del Q_torch, Q
                        del weight, solutions_mini, all_solutions, all_objectives
                        torch.cuda.empty_cache()
                        gc.collect()


                else:
                    if update_mask:
                        tunable_layer = torch.nn.Linear(subset[name].weight.data.shape[1], subset[name].weight.data.shape[0], bias=False)
                        tunable_layer.weight.data = wrapped_layers[name].original_weight.data.clone().detach().cuda()
                        tunable_layer.init_mask = (subset[name].weight.data != 0).to(torch.bfloat16).cuda()

                        init_loss_, final_loss_ = block_wise_optimize_mask(
                            tunable_layer,
                            {},
                            wrapped_layers[name].inputs,
                            wrapped_layers[name].outputs,
                            num_epochs=4,
                            optimizer="adam",
                            verbose=False,
                        )

                        best_weight = tunable_layer.weight.data.clone().detach()
                        trainable_weight = best_weight

                        print("Average Mask Similarity: ", ((best_weight == 0) == W_mask).float().mean())
                    else:
                        trainable_weight = subset[name].weight



                    num_steps = 20
                    best_loss = torch.inf
                    best_weight = None
                    for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
                        weight = torch.nn.Parameter(trainable_weight.data.clone().cuda())
                        optimizer = torch.optim.Adam([weight], lr=lr)
                        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_steps)
                        tuning_progress_bar = tqdm.tqdm(range(num_steps), desc="Tuning Progress")
                        init_loss = None
                        for step in tuning_progress_bar:
                            avg_loss = 0
                            for (x, y) in zip(wrapped_layers[name].inputs, wrapped_layers[name].outputs):
                                output = x.cuda() @ weight.t()
                                loss = torch.nn.functional.mse_loss(output, y.cuda())
                                loss.backward()
                                avg_loss += loss.item()
                            avg_loss /= len(wrapped_layers[name].inputs)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            weight.data[W_mask] = 0  ## set weights to zero
                            if init_loss is None:
                                init_loss = avg_loss
                            tuning_progress_bar.set_postfix({"init_loss": init_loss, "loss": avg_loss, "lr": lr})
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                            best_weight = weight.data.clone().detach()
                
                final_loss = compute_error(best_weight)
                norm = compute_error(torch.zeros_like(best_weight))
                print(f"Init Loss: {init_loss / norm}, Final Loss: {final_loss / norm}")
                if final_loss < init_loss:
                    subset[name].weight.data = best_weight.to(subset[name].weight.dtype)
                else:
                    print("No improvement. Skipping update.")

                wrapped_layers[name].inputs = []
                wrapped_layers[name].outputs = []
                del wrapped_layers[name].original_weight
                gc.collect()

        progress_bar.set_description(f"Layer {i} - Evaluating Output")

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(outs[j].device)
        inps, outs = outs, inps

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(
        model, 
        tokenizer,
        sparsity_ratio=0.5,
        prune_n=0, 
        prune_m=0,
        nsamples=128,
        seed=0,
        quantize_weight=False,
        bitwidth=4,
        tiled_weight_quantization=False,
        weight_tile_size=256,
        calibration_dataset="c4"
):
    """
    Prune a model using SparseGPT and quantize weights using OPTQ (GPTQ).
    SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa

    Args:
        model: torch.nn.Module - The model to prune
        tokenizer: transformers.Tokenizer - The tokenizer for the model
        device: torch.device - The device to use for pruning
        sparsity_ratio: float - The ratio of weights to prune
        prune_n: int - The number N in N:M pruning
        prune_m: int - The number M in N:M pruning
        nsamples: int - The number of samples to use for calibration
        seed: int - The seed to use for calibration
        quantize_weight: bool - Whether to quantize weights
        bitwidth: int - The bitwidth to use for quantization
        tiled_weight_quantization: bool - Whether to use block quantization
        weight_tile_size: int - The size of the blocks for block
        calibration_dataset: str - The dataset to use for calibration

    Returns:
        None
    """
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    dataloader, _ = get_loaders(
        calibration_dataset,
        nsamples=nsamples,
        seed=seed,
        seqlen=model.config.max_position_embeddings,
        tokenizer=tokenizer
    )

    with torch.no_grad():
        inps, outs, kwargs = prepare_calibration_input(model, dataloader, nsamples)

    layers = get_layers_list(model)

    progress_bar = tqdm.tqdm(range(len(layers)))

    for i in progress_bar:
        progress_bar.set_description(f"Layer {i} - Gathering data")
        layer = layers[i].cuda()

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])
            if quantize_weight:
                gpts[name].quantizer = SparseGPTQuantizer()
                gpts[name].quantizer.configure(
                    bitwidth,
                    perchannel=tiled_weight_quantization,
                    sym=True,
                    mse=False,
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            for key in kwargs:
                if isinstance(kwargs[key], torch.Tensor):
                    kwargs[key] = kwargs[key].cuda()
                if isinstance(kwargs[key], tuple):
                    kwargs[key] = tuple([k.cuda() for k in kwargs[key]])

            outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(outs.device)

        for h in handles:
            h.remove()

        for name in gpts:
            progress_bar.set_description(f"Layer {i} - Pruning and Quantizing {name}")
            gpts[name].fasterprune(sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=weight_tile_size)
            if quantize_weight:
                subset[name].register_buffer("quantization_scaling_factor", 1. / gpts[name].quantizer.scale)
                if not tiled_weight_quantization:
                    subset[name].scaling_factor = 1. / gpts[name].quantizer.scale[0]
                else:
                    subset[name].scaling_factor = None
            gpts[name].free()

        progress_bar.set_description(f"Layer {i} - Evaluating Output")
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(outs[j].device)

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def quantize_model(
       model,
       bitwidth=4,
       slim_quant=False,
       weight_tiled_quantization=False,
       weight_tile_size=256,
):
    """
    Quantize the model using the AutoQuantizer class.

    Args:
        model: torch.nn.Module - The model to quantize
        bitwidth: int - The bitwidth to quantize the model to
        slim_quant: bool - Use SLiM-Quant
        weight_tiled_quantization: bool - Use block quantization
        weight_tile_size: int - The size of the block for block quantization

    Returns:
        None
    """
    quantizer = AutoQuantizer(
        "weight",
        num_bits=bitwidth,
        slim_quant=slim_quant,
        block_quantization=weight_tiled_quantization,
        block_dim=weight_tile_size,
    )
    layers = get_layers_list(model)

    progress_bar = tqdm.tqdm(range(len(layers)))
    
    for i in progress_bar:
        layer = layers[i]
        layer = layer.cuda()

        subset = find_layers(layer)
        
        for name in subset:
            progress_bar.set_description(f"Layer {i} - Quantizing {name}")

            quantized_weight = quantizer.dequantize_absmax(
                quantizer.quantize_weight(subset[name].weight.data)
            )
            
            subset[name].weight.data = quantized_weight.to(subset[name].weight.dtype)
            subset[name].register_buffer("quantization_scaling_factor", quantizer.scaling_factor)
            if not weight_tiled_quantization:
                subset[name].scaling_factor = quantizer.scaling_factor
            else:
                subset[name].scaling_factor = None
        layer = layer.cpu()


def joint_pq(
        model,
        tokenizer,
        prune_n=0,
        prune_m=0,
        nsamples=128,
        bitwidth=4,
        sparsity_ratio=0.5,
        weight_tile_size=256,
        mixing_factor=2.1,
        seed=0,
        calibration_dataset="c4",
        lora_rank=0.,
        slim_lora=True,
        prune_lora=False,
        quantize_lora=False,
        lora_tile_size=256,
        separate_lora=True,
        quantize_first=True, 
        pad_lora=False,
        scale_important_weights=False,
):
    """
    Prune and quantize a model using joint pruning and quantization.
    
    Args:
        model: torch.nn.Module - The model to prune and quantize
        tokenizer: transformers.Tokenizer - The tokenizer for the model
        prune_n: int - The number N in N:M pruning
        prune_m: int - The number M in N:M pruning
        nsamples: int - The number of samples to use for calibration
        bitwidth: int - The bitwidth to quantize the model to
        sparsity_ratio: float - The ratio of weights to prune
        weight_tile_size: int - The size of the blocks for block quantization
        mixing_factor: float - The mixing factor for WANDA
        seed: int - The seed to use for calibration
        calibration_dataset: str - The dataset to use for calibration
        lora_rank: float - The rank ratio for low-rank adapter
        slim_lora: bool - Whether to use SLiM for low-rank adapter
        prune_lora: bool - Whether to prune the low-rank adapter
        quantize_lora: bool - Whether to quantize the low-rank adapter
        lora_tile_size: int - The size of the blocks for block quantization of the low-rank adapter
        separate_lora: bool - Whether to separate the low-rank adapter
        quantize_first: bool - Whether to quantize the weights before or after pruning
        pad_lora: bool - Whether to pad the LoRA weights
        scale_important_weights: bool - Whether to scale the important weights
    
    Returns:
        None
    """

    dataloader, _ = get_loaders(calibration_dataset, nsamples=nsamples, seed=seed, seqlen=model.seqlen,
                                tokenizer=tokenizer)
    with torch.no_grad():
        inps, outs, kwargs = prepare_calibration_input(model, dataloader, nsamples)

    layers = get_layers_list(model)

    quantizer = AutoQuantizer(
            "weight",
            num_bits=bitwidth,
            slim_quant=False,
            block_quantization=True,
            block_dim=weight_tile_size,
        )

    progress_bar = tqdm.tqdm(range(len(layers)))

    for i in progress_bar:
        layer = layers[i].cuda()
        layer_name = f'model.layers.{i}'

        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        progress_bar.set_description(f"Layer {i} - Gathering data")
        act_scales = {}

        def stat_tensor(name, tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()

            if name in act_scales:
                act_scales[layer_name + '.' + name] = torch.max(act_scales[name], comming_max)
            else:
                act_scales[layer_name + '.' + name] = comming_max

        def add_batch(name):
            def tmp(_, inp, out):
                inp = clip_matrix(inp[0].data, True, 0, 1e-2)
                stat_tensor(name, inp)
                wrapped_layers[name].add_batch(inp, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            with torch.no_grad():
                for key in kwargs:
                    if isinstance(kwargs[key], torch.Tensor):
                        kwargs[key] = kwargs[key].cuda()
                    if isinstance(kwargs[key], tuple):
                        kwargs[key] = tuple([k.cuda() for k in kwargs[key]])
                outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(outs.device)
        for h in handles:
            h.remove()

        for name in subset:
            progress_bar.set_description(f"Layer {i} - Pruning {name}")
            weight = torch.abs(subset[name].weight.data)
            activation = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            ss = generate_ss(wrapped_layers[name].inp_sum / wrapped_layers[name].inp_num, subset[name].weight.data)
            W_metric = weight * activation
            W_metric = W_metric + mixing_factor * ss

            W_mask = (torch.zeros_like(W_metric) == 1)
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                # unstructured pruning
                indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(nsamples):
            with torch.no_grad():
                for key in kwargs:
                    if isinstance(kwargs[key], torch.Tensor):
                        kwargs[key] = kwargs[key].cuda()
                    if isinstance(kwargs[key], tuple):
                        kwargs[key] = tuple([k.cuda() for k in kwargs[key]])
                outs[j] = layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(outs.device)

        progress_bar.set_description(f"Layer {i} - Smoothing {name}")
        smooth_layer(layer_name, layer, act_scales, 0.5)

        for name in subset:
            if lora_rank > 0.:
                progress_bar.set_description(f"Layer {i} - Quantizing and Adding LoRA to {name}")
                lora_tile_size = lora_tile_size if (quantize_lora or pad_lora) else None
                add_lora(subset[name],
                            W_mask=subset[name].weight.data == 0,
                            rank_ratio=lora_rank,
                            slim_lora=slim_lora,
                            activations=wrapped_layers[name],
                            quantizer=quantizer,
                            prune_lora=prune_lora,
                            separate_lora=separate_lora,
                            lora_tile_size=lora_tile_size,
                            quantize_first=quantize_first,
                            scale_important_weights=scale_important_weights,
                            )

                if quantizer is not None:
                    subset[name].register_buffer("quantization_scaling_factor", quantizer.scaling_factor)
                    subset[name].scaling_factor = None

                if separate_lora:
                    def add_lora_hook(module, input, output):
                        if hasattr(module, "lora_quantizer"):
                            xl = QuantizedMatmul.apply(
                                input[0].to(module.lora_left.dtype) / torch.sqrt(module.lora_rank),
                                module.lora_left,
                                module.lora_quantizer
                            )
                            xlr = QuantizedMatmul.apply(
                                xl / torch.sqrt(module.lora_rank),
                                module.lora_right,
                                module.lora_quantizer
                            )
                            output += xlr

                        else:
                            output += torch.matmul(
                                torch.matmul(input[0].to(module.lora_left.dtype),
                                                module.lora_left / torch.sqrt(module.lora_rank)),
                                module.lora_right / torch.sqrt(module.lora_rank))


                    subset[name].lora_rank = torch.tensor(subset[name].lora_left.shape[1])
                    subset[name].lora_left = torch.nn.Parameter(subset[name].lora_left * torch.sqrt(subset[name].lora_rank))
                    subset[name].lora_right = torch.nn.Parameter(subset[name].lora_right * torch.sqrt(subset[name].lora_rank))
                    subset[name].register_forward_hook(add_lora_hook)
            else:
                if scale_important_weights:
                    # Get 1% of largest activations
                    metric = subset[name].scaler_row * subset[name].weight.data.abs().sum(dim=0)
                    important_weights = metric.topk(
                        int(0.01 * metric.numel()), largest=True, sorted=False)[1]
                else:
                    important_weights = None
                progress_bar.set_description(f"Layer {i} - Quantizing {name}")
                quantized_weight = quantizer.quantize_weight(subset[name].weight.data, important_weights)
                subset[name].weight.data = quantizer.dequantize_absmax(quantized_weight).to(torch.bfloat16)
                subset[name].register_buffer("quantization_scaling_factor", quantizer.scaling_factor)
                subset[name].scaling_factor = None

        inps, outs = outs, inps
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()


def prune_and_quantize(
        model,
        tokenizer,
        bitwidth=4,
        slim_quant=True,
        weight_tiled_quantization=False,
        weight_tile_size=256,
        prune_method="wanda",
        sparsity_ratio=0.5,
        sparsity_type="2:4",
        quantize_weight=False,
        nsamples=128,
        shift_zero_metrics=True,
        lora_rank=0.,
        slim_lora=True,
        prune_lora=False,
        quantize_lora=False,
        lora_tile_size=256,
        separate_lora=True,
        seed=0,
        joint_pq_mixing_factor=2.1,
        calibration_dataset="c4",
        pad_lora=False,
        scale_important_weights=False,
        mask_checkpoint=None,
        quant_type=None,
        update_weights=True,
        use_qp_solver=True,
        double_precision=False,
):
    """
    Prune and quantize a model and add low-rank adapter to it.
    Quantize using quant type

    Args:
        model: torch.nn.Module - The model to prune and quantize
        tokenizer: transformers.Tokenizer - The tokenizer for the model
        bitwidth: int - The bitwidth to quantize the model to
        slim_quant: bool - Use SLiM-Quant
        weight_tiled_quantization: bool - Use block quantization
        weight_tile_size: int - The size of the block for block quantization
        prune_method: str - The pruning method to use, one of "wanda", "magnitude", "sparsegpt", "joint_pq"
        sparsity_ratio: float - The ratio of weights to prune
        sparsity_type: str - The type of sparsity to use (unstructured, dense, N:M)
        quantize_weight: bool - Whether to quantize weights
        nsamples: int - The number of samples to use for calibration
        shift_zero_metrics: bool - Whether to shift zero metrics in Wanda
        lora_rank: float - The rank ratio for low-rank adapter
        slim_lora: bool - Whether to use SLiM for low-rank adapter
        prune_lora: bool - Whether to 2:4 prune the L low-rank adapter
        quantize_lora: bool - Whether to quantize the low-rank adapter
        lora_tile_size: int - The size of the block for block quantization of the low-rank adapter
        separate_lora: bool - Whether to separate the low-rank adapter
        seed: int - The seed to use for calibration
        joint_pq_mixing_factor: float - The mixing factor for joint pruning and quantization
        calibration_dataset: str - The dataset to use for calibration
        pad_lora: bool - Whether to pad the low-rank adapter to the quantization tile size (whithout quantizing)
        scale_important_weights: bool - Whether to scale the important weights before quantization,
        mask_checkpoint: str - The checkpoint to use for MaskLLM pruning
        quant_type: str - The quantization type to use
        update_weights: bool - Whether to update weights during pruning
        use_qp_solver: bool - Whether to use quadratic programming solver
        double_precision: bool - Whether to use double precision for calculations

    Returns:
        None
    """
    if sparsity_ratio == 0. or sparsity_type == "dense":
        if quantize_weight:
            print("Quantizing the dense model:")
            if lora_rank > 0:
                raise NotImplementedError("LoRA approximation not implemented for quantization only - "
                                          "Please use pruning with low sparsity ratio for quantization only.")
            quantize_model(model,
                           bitwidth,
                           slim_quant,
                           weight_tiled_quantization,
                           weight_tile_size,
                           )
        else:
            print("Using original dense model.")
    else:
        print("Sparsity Ratio: ", sparsity_ratio)
        print("Pruning Structure: ", sparsity_type)
        # Handling n:m sparsity
        prune_n, prune_m = 0, 0
        if sparsity_type != "unstructured":
            prune_n, prune_m = map(int, sparsity_type.split(":"))
            prune_n = prune_m - prune_n
            assert sparsity_ratio == prune_n / prune_m, \
                f"Sparsity ratio must be {prune_n / prune_m} for structured N:M sparsity"
        if prune_method in ["wanda", "maskllm"]:
            if prune_method == "wanda":
                pruning_name = "Wanda"
            else:
                pruning_name = "MaskLLM"
            if quantize_weight:
                if slim_quant:
                    quantization_method = "SLiM-Quant"
                else:
                    if weight_tiled_quantization:
                        quantization_method = "Tiled Group AbsMax"
                    else:
                        quantization_method = "AbsMax"
                print(F"Pruning the model with {pruning_name} and quantizing the weights using {quantization_method}.")
            else:
                print(f"Pruning the model with {pruning_name}.")
            if lora_rank > 0:
                if slim_lora:
                    print(f"Adding SLiM-LoRA approximation with rank ratio {lora_rank}.")
                else:
                    print(f"Adding Naive-LoRA approximation with rank ratio {lora_rank}.")
            if prune_lora and not (prune_n == 2 and prune_m == 4):
                raise NotImplementedError("Pruning LoRA is only supported for 2:4 sparsity ratio")
            if prune_method == "maskllm":
                assert mask_checkpoint is not None, "Mask checkpoint must be provided for MaskLLM pruning"
                assert prune_n == 2 and prune_m == 4, "MaskLLM pruning only supports 2:4 sparsity ratio"
                try:
                    downloaded_mask = hf_hub_download(repo_id=mask_checkpoint, filename="mask_compressed.npz")
                    mask_ckpt = np.load(downloaded_mask)
                    for k, v in mask_ckpt.items():
                        k_original = k.replace(".mask", "")
                        v = np.unpackbits(v)  # to bits
                        mask = torch.from_numpy(v).float()
                        param = dict(model.named_parameters()).get(k_original, None)
                        mask = mask.view(*param.shape)
                        param.mask = (mask == 0).bool()
                except FileNotFoundError:
                    raise FileNotFoundError("Mask checkpoint not found. Please provide a valid checkpoint.")
            prune_wanda(
                model,
                tokenizer,
                sparsity_ratio,
                prune_n,
                prune_m,
                quantize_weight,
                bitwidth,
                slim_quant,
                weight_tiled_quantization,
                weight_tile_size,
                shift_zero_metrics,
                lora_rank,
                slim_lora,
                prune_lora,
                quantize_lora,
                lora_tile_size,
                separate_lora,
                nsamples,
                seed,
                calibration_dataset,
                pad_lora,
                scale_important_weights=scale_important_weights,
                update_weights=update_weights,
                use_qp_solver=use_qp_solver,
                double_precision=double_precision,
            )
        elif prune_method == "magnitude":
            if scale_important_weights and quantize_weight:
                raise NotImplementedError("Scaling important weights not implemented for magnitude pruning and "
                                          "quantization")
            if lora_rank > 0:
                raise NotImplementedError("LoRA approximation not implemented for magnitude pruning")
            if quantize_weight:
                if slim_quant:
                    quantization_method = "SLiM-Quant"
                else:
                    if weight_tiled_quantization:
                        quantization_method = "Tiled Group AbsMax"
                    else:
                        quantization_method = "AbsMax"
                print(F"Pruning the model with Magnitude Pruning "
                      F"and quantizing the weights using {quantization_method}.")
            else:
                print("Pruning the model with Magnitude Pruning.")
            prune_magnitude(
                model,
                sparsity_ratio,
                prune_n,
                prune_m,
                quantize_weight,
                bitwidth,
                slim_quant,
                weight_tiled_quantization,
                weight_tile_size,
            )
        elif prune_method == "sparsegpt":
            if scale_important_weights and quantize_weight:
                raise NotImplementedError("Scaling important weights not implemented for magnitude pruning and "
                                          "quantization")
            if lora_rank > 0:
                raise NotImplementedError("LoRA approximation not implemented for SparseGPT")
            if slim_quant:
                raise NotImplementedError("SparseGPT can only support OPTQ (GPTQ) quantization")
            if quantize_weight:
                if weight_tiled_quantization:
                    quantization_method = "Group OPTQ (GPTQ)"
                else:
                    quantization_method = "OPTQ (GPTQ)"
                print(F"Pruning the model with SparseGPT and quantizing the weights using {quantization_method}.")
            else:
                print("Pruning the model with SparseGPT.")
            prune_sparsegpt(
                model,
                tokenizer,
                sparsity_ratio,
                prune_n,
                prune_m,
                nsamples,
                seed,
                quantize_weight,
                bitwidth,
                weight_tiled_quantization,
                weight_tile_size,
                calibration_dataset,
            )
        elif prune_method == "joint_pq":
            if weight_tiled_quantization is False:
                raise NotImplementedError("Joint pruning and quantization only supports block quantization")
            if slim_quant:
                raise NotImplementedError("Joint pruning and quantization only supports AbsMax")
            if quantize_weight is False:
                raise NotImplementedError("Joint pruning and quantization requires quantizing weights")
            joint_pq(
                model,
                tokenizer,
                prune_n,
                prune_m,
                nsamples,
                bitwidth,
                sparsity_ratio,
                weight_tile_size,
                joint_pq_mixing_factor,
                seed,
                calibration_dataset,
                lora_rank,
                slim_lora,
                prune_lora,
                quantize_lora,
                lora_tile_size,
                separate_lora,
                pad_lora,
                scale_important_weights,
            )
        else:
            raise NotImplementedError(f"Pruning method {prune_method} not implemented")