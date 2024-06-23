import torch
from compression.model_compression import static_prune_weight_reduction_dim_forward
from types import MethodType
import numpy as np
from transformers import AutoModelForCausalLM
from compression.quantization.model_quantizing import Quantizer as AutoQuantizer

hf_token = "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"


def density_ratio(x):
    return (x != 0).sum().float() / x.numel()


def get_layers_list(model):
    if hasattr(model, "model"):
        if hasattr(model.model, "layers"):
            layers = model.model.layers
        else:
            if hasattr(model.model, "decoder"):
                layers = model.model.decoder.layers
            else:
                raise NotImplementedError
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        raise NotImplementedError
    return layers


def shift_zeros(x):
    min_positive = x.clone().detach()
    min_positive[min_positive == 0] = 1
    min_positive = min_positive.min()
    return x + min_positive


def optimize_pruned_l(L, R, error_mat):
    sparse_L, mask_L = prune_row_wise(L, 2, 4)
    sparse_L = sparse_L.half()
    criterion = torch.nn.MSELoss()
    optimizer_L = optim.SGD(params=[sparse_L], lr=0.01, momentum=0.9)
    optimizer_R = optim.SGD(params=[R], lr=0.01, momentum=0.9)
    sparse_L.requires_grad = True
    R.requires_grad = True
    convergence_threshold = 5e-8
    e2e_iteration = 0
    max_iterations = 50
    e2e_convergence_threshold = 1e-8
    e2e_prev_loss = float('inf')

    while True:
        prev_loss = float('inf')
        iteration = 0
        while True:
            optimizer_L.zero_grad()
            error_mat_hat = sparse_L @ R
            loss = criterion(error_mat_hat, error_mat.half())
            loss.backward()
            optimizer_L.step()
            sparse_L_detached = sparse_L.detach()
            sparse_L_detached[mask_L] = 0.
            sparse_L = sparse_L_detached.requires_grad_()

            if abs(loss.item() - prev_loss) < convergence_threshold:
                print("L Converged in " + str(iteration))
                break
            prev_loss = loss.item()
            iteration += 1

            if iteration >= max_iterations:
                print("Maximum iterations reached.")
                break

        iteration = 0
        prev_loss = float('inf')

        while True:
            optimizer_R.zero_grad()
            error_mat_hat = sparse_L @ R
            loss = criterion(error_mat_hat, error_mat.half())
            loss.backward()
            optimizer_R.step()
            sparse_L_detached = sparse_L.detach()
            sparse_L_detached[mask_L] = 0.
            sparse_L = sparse_L_detached.requires_grad_()

            if abs(loss.item() - prev_loss) < convergence_threshold:
                print("R Converged in " + str(iteration))
                break
            prev_loss = loss.item()
            iteration += 1

            if iteration >= max_iterations:
                print("Maximum iterations reached.")
                break

        if abs(loss.item() - e2e_prev_loss) < e2e_convergence_threshold:
            print("e2e algo Converged in " + str(e2e_iteration))
            break
        e2e_prev_loss = loss.item()
        e2e_iteration += 1

        if e2e_iteration >= max_iterations:
            print("E2E Maximum iterations reached.")
            break

    sparse_L.requires_grad = False
    R.requires_grad = False

    return sparse_L, R


def add_lora(module,
             W_mask,
             rank_ratio=0.01,
             use_wanda=False,
             activations=None,
             use_randomized_svd=True,
             quantizer=None,
             bitwidth=8,
             use_std=False,
             max_bitwidth=8,
             pruned_L=False,
             separate_lora=True):
    if use_wanda and not any(activations.scaler_row == 0):
        if quantizer is None:
            W_metric = module.weight.data * (torch.sqrt(activations.scaler_row.reshape((1, -1))))
            new_weight = W_metric.clone().detach()
            new_weight[W_mask] = 0
            error_mat = W_metric - new_weight
        else:
            W_metric = module.weight.data * (torch.sqrt(activations.scaler_row.reshape((1, -1))))
            new_weight = module.weight.data
            new_weight[W_mask] = 0
            new_weight = quantizer.quantize_weight(new_weight, bitwidth, use_std=use_std, max_bitwidth=max_bitwidth)
            new_weight = quantizer.dequantize_absmax(new_weight) * (torch.sqrt(activations.scaler_row.reshape((1, -1))))
            error_mat = (W_metric - new_weight)
    else:
        new_weight = module.weight.data.clone().detach()
        new_weight[W_mask] = 0
        if quantizer is not None:
            new_weight = quantizer.quantize_weight(new_weight, bitwidth, use_std=use_std, max_bitwidth=max_bitwidth)
            new_weight = quantizer.dequantize_absmax(new_weight)
        error_mat = module.weight.data - new_weight
    # Use SVD on the error matrix to find the best low-rank approximation
    if use_randomized_svd:
        U, S, V = randomized_svd(error_mat.float(), rank_ratio)
    else:
        U, S, V = torch.svd(error_mat.float())
    rank = int(rank_ratio * min(error_mat.shape))

    if pruned_L:
        L = U[:, :rank].half()
        R = torch.diag_embed(S[:rank]).half() @ V[:, :rank].half().T
        sparse_L, R = optimize_pruned_l(L, R, error_mat)
        if separate_lora:
            lora_left = R.t()
            lora_right = sparse_L.half().t()
        else:
            low_rank_weight = sparse_L.half() @ R

    else:
        if separate_lora:
            lora_left = (torch.diag_embed(S[:rank]).half() @ V[:, :rank].half().T).t()
            lora_right = U[:, :rank].half().t()
        else:
            low_rank_weight = U[:, :rank].half() @ torch.diag_embed(S[:rank]).half() @ V[:, :rank].half().T
    if use_wanda and not any(activations.scaler_row == 0):
        denom = (torch.sqrt(activations.scaler_row.reshape((1, -1)))).half()
        if separate_lora:
            lora_left = lora_left / (denom.t())
        else:
            low_rank_weight /= denom
    if separate_lora:
        low_rank_weight = lora_right.t() @ lora_left.t()
    new_weight = module.weight.data - low_rank_weight
    new_weight[W_mask] = 0
    if quantizer is not None:
        new_weight = quantizer.quantize_weight(new_weight, bitwidth, use_std=use_std, max_bitwidth=max_bitwidth)
        new_weight = quantizer.dequantize_absmax(new_weight)

    if separate_lora:
        module.lora_left = torch.nn.Parameter(lora_left).half()
        module.lora_right = torch.nn.Parameter(lora_right).half()
        module.weight.data = new_weight.half()
    else:
        module.weight.data = (new_weight + low_rank_weight).half()


def randomized_svd(B, rank, redundancy=2):
    if rank < 1:
        output_rank = int(rank * min(B.size()))
        rank = int(min(rank * redundancy, 0.5) * min(B.size()))
    m, n = B.size()
    rand_matrix = torch.randn((n, rank)).to(B.device)  # short side by k
    Q, _ = torch.linalg.qr(B @ rand_matrix)  # long side by k
    smaller_matrix = (Q.transpose(0, 1) @ B)  # k by short side
    U_hat, S, V = torch.svd(smaller_matrix, False)
    U = (Q @ U_hat)
    return U[:, :output_rank], S[0:output_rank], V[:, :output_rank]


def accelerate_module(module, quantize=False, bitwidth=8):
    module.forward = MethodType(static_prune_weight_reduction_dim_forward, module)
    module.accelerate = True
    module.quantization_en = quantize
    module.weight.requires_grad = False
    if quantize:
        if bitwidth <= 8:
            module.dtype = torch.int8
        elif bitwidth <= 16:
            module.dtype = torch.int16
        else:
            module.dtype = torch.int32
        abs_max = module.weight.abs().max()
        module.weight_scaling_factor = (2. ** (bitwidth - 1) - 1) / abs_max
        module.weight.data = torch.round(module.weight.data * module.weight_scaling_factor).to(module.dtype)
    else:
        module.weight_scaling_factor = None
    module.qbitwidth = bitwidth
    module.sparse_index = None
    module.mask = None
    module.add_lora = False  # TODO: Fix


def remove_outlier(x, std_factor=2):
    """Remove outliers from a list."""
    mean = np.mean(x)
    std = np.std(x)
    return [e for e in x if (mean - std_factor * std < e < mean + std_factor * std)]


if __name__ == "__main__":
    pass

    # # Unit Test for randomized_svd
    # A = torch.randn(1000, 1000)
    # rank_ratio = 0.1
    # U, S, V = randomized_svd(A, rank_ratio)
    # error = A - U @ torch.diag_embed(S) @ V.T
    # print(error.norm() / A.norm())
    # U, S, V = torch.svd(A)
    # rank = int(rank_ratio * min(A.shape))
    # error = A - U[:, :rank] @ torch.diag_embed(S[:rank]) @ V[:, :rank].T
    # print(error.norm() / A.norm())

    # Unit test for acceleration
    layer = torch.nn.Linear(1024, 4096).cuda()
    input = torch.randn(512, 1024).cuda()
    from compression.pruning_kernels.sparse_backend import prune_tensor

    layer.weight.data = prune_tensor(layer.weight.data)[0]
    weight_fp16 = layer.weight.data.clone().detach()
    # print(layer.weight.data[0:8, 0:8])
    accelerate_module(layer, quantize=True)
    # print(layer.weight.data[0:8, 0:8])
    y_sparse = layer(input)
    y_dense = input @ weight_fp16.t()
    print(y_sparse.dtype)
    # print(y_dense[0:8, 0:8])
    # print(y_sparse[0:8, 0:8])
    print("SpMM Relative Error: ", ((y_dense - y_sparse).float().norm() / y_dense.float().norm()).item())


def find_layers(module, layers=[torch.nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def attach_input_quantization_hooks(model, num_bits=8):
    def input_quantization_pre_hook(module, input):
        quantized_input = module.quantizer.quantize(input[0])
        dequantized_input = module.quantizer.dequantize_input(quantized_input)
        print("Relative Error:", (torch.norm(input[0] - dequantized_input) / torch.norm(input[0])).item(),
              module.weight.shape)
        # input.data = module.quantizer.dequantize_input(quantized_input)

    layers = get_layers_list(model)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            subset[name].quantizer = AutoQuantizer("input", num_bits=num_bits)
            subset[name].register_forward_pre_hook(input_quantization_pre_hook)


def merge_lora(model):
    for name, module in model.named_modules():
        if hasattr(module, "lora_left") and hasattr(module, "lora_right"):
            module.weight.data += (module.lora_right.t() @ module.lora_left.t()).to(module.weight.device).to(
                module.weight.dtype)
            del module.lora_left
            del module.lora_right


def add_empty_lora(model, rank):
    for layer in get_layers_list(model):
        subset = find_layers(layer)
        for name in subset:
            layer_rank = int(min(subset[name].weight.shape) * rank)
            subset[name].lora_left = torch.nn.Parameter(
                torch.zeros((subset[name].weight.shape[1], layer_rank), device=subset[name].weight.device).half())
            subset[name].lora_right = torch.nn.Parameter(
                torch.zeros((layer_rank, subset[name].weight.shape[0]), device=subset[name].weight.device).half())

        def add_lora_hook(module, input, output):
            output += torch.matmul(torch.matmul(input[0].to(module.lora_left.dtype), module.lora_left),
                                   module.lora_right)

        subset[name].register_forward_hook(add_lora_hook)


def conv1d_to_linear(conv1d):
    input_dim = conv1d.weight.shape[0]
    output_dim = conv1d.weight.shape[1]
    bias = conv1d.bias is not None
    layer = torch.nn.Linear(input_dim, output_dim, bias=bias)
    layer.weight.data = conv1d.weight.data.T
    if bias:
        layer.bias.data = conv1d.bias.data
    return layer


def convert_flashattn_checkpoint_to_hf(checkpoint):
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
        for key in list(checkpoint.keys()):
            new_key = key.replace('model.', '')
            if "embeddings.word_embeddings.weight" in new_key:
                new_key = "transformer.wte.weight"
            if "embeddings.position_embeddings.weight" in new_key:
                new_key = "transformer.wpe.weight"
            if "layers" in new_key:
                new_key = new_key.replace("layers", "h")
            if "mixer" in new_key:
                new_key = new_key.replace("mixer", "attn")
            if "Wqkv" in new_key:
                new_key = new_key.replace("Wqkv", "c_attn")
            if "out_proj" in new_key:
                new_key = new_key.replace("out_proj", "c_proj")
            if "norm" in new_key:
                new_key = new_key.replace("norm", "ln_")
            if "fc1" in new_key:
                new_key = new_key.replace("fc1", "c_fc")
            if "fc2" in new_key:
                new_key = new_key.replace("fc2", "c_proj")
            if "wte" in new_key or "wpe" in new_key or "weight" not in new_key or "lora" in new_key or "lm_head" in new_key:
                checkpoint[new_key] = checkpoint.pop(key)
            else:
                checkpoint[new_key] = checkpoint.pop(key).T
            if "num-tokens" in new_key:
                del checkpoint[new_key]
    return checkpoint


def get_llm(model_name, cache_dir="llm_weights", local_checkpoint_dir="", device_map=None):
    kwargs = {}
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        token=hf_token,
        **kwargs
    )
    if device_map == None:
        layer_num_params = 0
        for param in model.parameters():
            layer_num_params += param.numel()
        model_size = layer_num_params * 2
        free_mem, total_mem = torch.cuda.mem_get_info()
        if model_size < 0.75 * free_mem:
            print("Loading model in GPU...")
            model = model.cuda()
        else:
            print("Model does not fit in GPUs. Loading model in CPU...")

    if local_checkpoint_dir != "":
        checkpoint = torch.load(local_checkpoint_dir, map_location="cpu")
        if "gpt" in model_name:
            checkpoint = convert_flashattn_checkpoint_to_hf(checkpoint)
        model.load_state_dict(checkpoint)

    for i, layer in enumerate(get_layers_list(model)):
        if hasattr(layer, "attn"):
            layer.attn.c_attn = conv1d_to_linear(layer.attn.c_attn)
            layer.attn.c_proj = conv1d_to_linear(layer.attn.c_proj)
        if hasattr(layer, "mlp"):
            if hasattr(layer.mlp, "c_fc"):
                layer.mlp.c_fc = conv1d_to_linear(layer.mlp.c_fc)
            if hasattr(layer.mlp, "c_proj"):
                layer.mlp.c_proj = conv1d_to_linear(layer.mlp.c_proj)

    model.seqlen = model.config.max_position_embeddings
    return model


def distribute_model(model, fill_ratio=1., reserved_input_size=None):
    model = model.cpu()
    torch.cuda.empty_cache()
    # model = model.float()
    input_size = 4 if model.seqlen == 2048 else 20 if model.seqlen == 4096 else 58 if reserved_input_size is None else reserved_input_size
    layer_num_params = 0
    layers = get_layers_list(model)
    for param in layers[0].parameters():
        layer_num_params += param.numel()
    layer_size = 2 * layer_num_params
    free_mem, total_mem = torch.cuda.mem_get_info()
    num_layers_per_gpu = int(free_mem * fill_ratio) // layer_size - input_size
    print(f"Transfering {min(torch.cuda.device_count() * num_layers_per_gpu, len(layers))} "
          f"layers out of {len(layers)} to GPU. Each GPU will hold {num_layers_per_gpu} layers.")

    def move_to_gpu_pre_hook(module, input):
        print("Moving data to device: ", module.device)
        input[0].data = input[0].data.half().to(module.device)

    def move_to_cpu_post_hook(module, input, output):
        output[0].data = output[0].data.cpu().float()

    def move_to_gpu_backward_pre_hook(module, grad_input):
        print("Moving grads to ", module.device)
        grad_input[0].data = grad_input[0].data.to(module.device)

    last_fit_layer = -1
    for i in range(torch.cuda.device_count()):
        for layer_num in range(num_layers_per_gpu):
            last_fit_layer += 1
            if (last_fit_layer == len(layers) - 1) or (
                    (i == torch.cuda.device_count() - 1) and (layer_num == num_layers_per_gpu - 1)):
                layers[last_fit_layer] = layers[last_fit_layer].half().cuda(i)
                layers[last_fit_layer].register_forward_hook(move_to_cpu_post_hook)
                # layers[last_fit_layer].register_full_backward_pre_hook(move_to_gpu_backward_pre_hook)
                # print("Appending backward hook to layer", last_fit_layer)
                break
            layers[last_fit_layer] = layers[last_fit_layer].half().cuda(i)
            if layer_num == 0:
                layers[last_fit_layer].device = torch.device(i)
                layers[last_fit_layer].register_forward_pre_hook(move_to_gpu_pre_hook)
        if last_fit_layer == len(layers) - 1:
            break
    # def cast_input_to_float_pre_hook(module, input):
    #     input[0].data = input[0].data.float()
    #
    # def cast_ouput_to_half_post_hook(module, input, output):
    #     output_half = output[0].half()
    #     if output_half.dim() == 2:
    #         output_half = output_half.unsqueeze(0)
    #     return output_half
    #
    for layer_num in range(last_fit_layer + 1, len(layers)):
        layers[layer_num] = layers[layer_num].float()
        # model.model.decoder.layers[layer_num].self_attn_layer_norm = model.model.decoder.layers[layer_num].self_attn_layer_norm.float()
        # model.model.decoder.layers[layer_num].self_attn_layer_norm.register_forward_pre_hook(cast_input_to_float_pre_hook)
        # model.model.decoder.layers[layer_num].self_attn_layer_norm.register_forward_hook(cast_ouput_to_half_post_hook)
        # model.model.decoder.layers[layer_num].final_layer_norm = model.model.decoder.layers[layer_num].final_layer_norm.float()
        # model.model.decoder.layers[layer_num].final_layer_norm.register_forward_pre_hook(cast_input_to_float_pre_hook)
        # model.model.decoder.layers[layer_num].final_layer_norm.register_forward_hook(cast_ouput_to_half_post_hook)
    if hasattr(model.model, "decoder"):  # OPT
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.float()
    elif hasattr(model.model, "layers"):  # LLaMA
        model.model.norm = model.model.norm.float()
    model.lm_head = model.lm_head.float()
    return model