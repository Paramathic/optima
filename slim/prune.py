import torch
import torch.nn as nn
from .sparsegpt import SparseGPT
from .sparsegpt import Quantizer as SparseGPTQuantizer
from .layerwrapper import WrappedGPT
from .data import get_loaders
from .utils import get_layers_list, shift_zeros, find_layers, prune_nm
from .lora import add_lora
from slim.quantization.quantization import Quantizer as AutoQuantizer, QuantizedMatmul
from transformers import LlamaForCausalLM
import tqdm.auto as tqdm
from .jsq_utils import clip_matrix, generate_ss
from .smooth import smooth_layer


def prepare_calibration_input(
        model,
        dataloader,
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
    use_cpu = model.device == torch.device("cpu")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_layers_list(model)


    if use_cpu:
        if hasattr(model.model, "decoder"): #OPT
            model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cuda()
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.cuda()
            if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
                model.model.decoder.project_out = model.model.decoder.project_out.cuda()
            if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.cuda()
        if hasattr(model.model, "layers"): #LLaMA
            model.model.embed_tokens = model.model.embed_tokens.cuda()

        layers[0] = layers[0].cuda()

    dtype = next(iter(model.parameters())).dtype
    torch.cuda.empty_cache()
    inps = torch.zeros((128, model.config.max_position_embeddings, model.config.hidden_size), dtype=dtype, device="cpu")
    input_device = "cpu"
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    is_llama = isinstance(model, LlamaForCausalLM)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.to(input_device)
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if is_llama:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(model.device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    if use_cpu:
        layers[0] = layers[0].cpu()
        if hasattr(model.model, "decoder"): #OPT
            model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
            if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
                model.model.decoder.project_out = model.model.decoder.project_out.cpu()
            if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.cpu()
        if hasattr(model.model, "layers"): #LLaMA
            model.model.embed_tokens = model.model.embed_tokens.cpu()
        torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    if is_llama:
        position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    if is_llama:
        return inps, outs, attention_mask, position_ids
    return inps, outs, attention_mask


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
        layer = layers[i]
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
                subset[name].weight.data = quantizer.dequantize_absmax(quantized_weight).half()
                if not tiled_weight_quantization:
                    subset[name].scaling_factor = quantizer.scaling_factor
                else:
                    subset[name].scaling_factor = None


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

    Returns:
        None
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    is_llama = isinstance(model, LlamaForCausalLM)

    dataloader, _ = get_loaders(
        calibration_dataset,
        nsamples=nsamples,
        seed=seed,
        seqlen=model.config.max_position_embeddings,
        tokenizer=tokenizer
    )
    with torch.no_grad():
        if is_llama:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader)
        else:
            inps, outs, attention_mask = prepare_calibration_input(model, dataloader)

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
        use_cpu = model.device == torch.device("cpu")
        cpu_only = False
        layer = layers[i].cuda() if use_cpu else layers[i]

        subset = find_layers(layer)
        try:
            if f"model.layers.{i}" in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{i}"]
                if is_llama:
                    inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
                        dev), position_ids.to(dev)
                else:
                    inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)
        except:
            pass
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            with torch.no_grad():
                if not cpu_only:
                    try:
                        if is_llama:
                            outs[j] = layer(inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask,
                                            position_ids=position_ids)[0].to(outs.device)
                        else:
                            outs[j] = layer(inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask)[0].to(
                                outs.device)
                    except:
                        print("Block does not fit in a single GPU. Doing all the computation in CPU.")
                        cpu_only = True

                        layer = layer.cpu().float()
                        if is_llama:
                            outs[j] = layer(inps[j].unsqueeze(0),
                                            attention_mask=attention_mask,
                                            position_ids=position_ids)[0]
                        else:
                            outs[j] = layer(inps[j].unsqueeze(0).float(), attention_mask=attention_mask.cpu() if attention_mask is not None else attention_mask)[0].half()
                else:
                    if is_llama:
                        outs[j] = layer(inps[j].unsqueeze(0),
                                        attention_mask=attention_mask,
                                        position_ids=position_ids)[0]
                    else:
                        outs[j] = layer(inps[j].unsqueeze(0).float(), attention_mask=attention_mask.cpu())[0].half()

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
                         )

                if quantizer is not None:
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
                subset[name].weight.data[W_mask] = 0  ## set weights to zero
                if quantizer is not None:
                    quantized_weight = quantizer.quantize_weight(subset[name].weight.data)
                    subset[name].weight.data = quantizer.dequantize_absmax(quantized_weight).half()
                    if quantizer is not None:
                        if not tiled_weight_quantization:
                            subset[name].scaling_factor = quantizer.scaling_factor
                        else:
                            subset[name].scaling_factor = None


        if cpu_only:
            layer = layer.float()

        progress_bar.set_description(f"Layer {i} - Evaluating Output")

        for j in range(nsamples):
            with torch.no_grad():
                if not cpu_only:
                    if is_llama:
                        outs[j] = layer(inps[j].unsqueeze(0).cuda(),
                                        attention_mask=attention_mask,
                                        position_ids=position_ids)[0].to(outs[j].device)
                    else:
                        outs[j] = layer(inps[j].unsqueeze(0).cuda(),
                                        attention_mask=attention_mask)[0].to(outs[j].device)
                else:
                    if is_llama:
                        outs[j] = layer(inps[j].unsqueeze(0).float(),
                                        attention_mask=attention_mask.cpu(),
                                        position_ids=position_ids)[0].half()
                    else:
                        outs[j] = layer(inps[j].unsqueeze(0).float(), attention_mask=attention_mask.cpu())[0].half()
        inps, outs = outs, inps

        if use_cpu:
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
    use_cache = model.config.use_cache
    model.config.use_cache = False

    is_llama = isinstance(model, LlamaForCausalLM)

    dataloader, _ = get_loaders(
        calibration_dataset,
        nsamples=nsamples,
        seed=seed,
        seqlen=model.config.max_position_embeddings,
        tokenizer=tokenizer
    )

    with torch.no_grad():
        if is_llama:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader)
        else:
            inps, outs, attention_mask = prepare_calibration_input(model, dataloader)

    layers = get_layers_list(model)

    progress_bar = tqdm.tqdm(range(len(layers)))

    for i in progress_bar:
        progress_bar.set_description(f"Layer {i} - Gathering data")
        use_cpu = model.device == torch.device("cpu")
        cpu_only = False
        layer = layers[i].cuda() if use_cpu else layers[i]

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
            if not cpu_only:
                try:
                    if is_llama:
                        outs[j] = layer(inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask,
                                        position_ids=position_ids)[0].to(outs.device)
                    else:
                        outs[j] = layer(inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask)[0].to(
                            outs.device)
                except:
                    print("Block does not fit in a single GPU. Doing all the computation in CPU.")
                    cpu_only = True

                    layer = layer.cpu().float()

                    if is_llama:
                        outs[j] = layer(inps[j].unsqueeze(0),
                                        attention_mask=attention_mask,
                                        position_ids=position_ids)[0]
                    else:
                        outs[j] = layer(inps[j].unsqueeze(0).float(), attention_mask=attention_mask.cpu())[0].half()
            else:
                if is_llama:
                    outs[j] = layer(inps[j].unsqueeze(0),
                                    attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0).float(), attention_mask=attention_mask.cpu())[0].half()

        for h in handles:
            h.remove()

        for name in gpts:
            progress_bar.set_description(f"Layer {i} - Pruning and Quantizing {name}")
            gpts[name].fasterprune(sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=weight_tile_size)
            if quantize_weight:
                if not tiled_weight_quantization:
                    subset[name].scaling_factor = 1. / gpts[name].quantizer.scale[0]
                else:
                    subset[name].scaling_factor = None
            gpts[name].free()

        if cpu_only:
            layer = layer.float()
        progress_bar.set_description(f"Layer {i} - Evaluating Output")
        for j in range(nsamples):
            with torch.no_grad():
                if not cpu_only:
                    if is_llama:
                        outs[j] = layer(inps[j].unsqueeze(0).cuda(),
                                        attention_mask=attention_mask,
                                        position_ids=position_ids)[0].to(outs[j].device)
                    else:
                        outs[j] = layer(inps[j].unsqueeze(0).cuda(),
                                        attention_mask=attention_mask)[0].to(outs[j].device)
                else:
                    if is_llama:
                        outs[j] = layer(inps[j].unsqueeze(0).float(),
                                        attention_mask=attention_mask.cpu(),
                                        position_ids=position_ids)[0].half()
                    else:
                        outs[j] = layer(inps[j].unsqueeze(0).float(), attention_mask=attention_mask.cpu())[0].half()

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        if use_cpu:
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

        subset = find_layers(layer)
        
        for name in subset:
            progress_bar.set_description(f"Layer {i} - Quantizing {name}")
            
            quantized_weight = quantizer.dequantize_absmax(
                quantizer.quantize_weight(subset[name].weight.data)
            )
            
            subset[name].weight.data = quantized_weight.to(subset[name].weight.dtype)


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
    
    Returns:
        None
    """

    is_llama = isinstance(model, LlamaForCausalLM)

    dataloader, _ = get_loaders(calibration_dataset, nsamples=nsamples, seed=seed, seqlen=model.seqlen,
                                tokenizer=tokenizer)
    with torch.no_grad():
        if is_llama:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader)
        else:
            inps, outs, attention_mask = prepare_calibration_input(model, dataloader)

    layers = get_layers_list(model)

    quantizer = AutoQuantizer(
            "weight",
            num_bits=bitwidth,
            slim_quant=False,
            block_quantization=True,
            block_dim=weight_tile_size,
        )

    for i in range(len(layers)):
        layer = layers[i]
        layer_name = f'model.layers.{i}'

        subset = find_layers(layer)

        try:
            if f"model.layers.{i}" in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{i}"]
                if is_llama:
                    inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
                        dev), position_ids.to(dev)
                else:
                    inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)
        except:
            pass

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        print(f"Get scales of layer {i} and pruning")
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
                if is_llama:
                    outs[j] = layer(inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask,
                                    position_ids=position_ids)[0].to(outs.device)
                else:
                    outs[j] = layer(inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask)[0].to(
                        outs.device)
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
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
                if is_llama:
                    outs[j] = layer(inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask,
                                    position_ids=position_ids)[0].to(outs.device)
                else:
                    outs[j] = layer(inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask)[0].to(
                        outs.device)

        print(f"smoothing layer {i}")
        smooth_layer(layer_name, layer, act_scales, 0.5)

        print(f"quantizing layer {i}")
        quantized_weight = quantizer.quantize_weight(subset[name].weight.data)
        subset[name].weight.data = quantizer.dequantize_absmax(quantized_weight).half()

        inps, outs = outs, inps

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
):
    """
    Prune and quantize a model and add low-rank adapter to it.

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
        if prune_method == "wanda":
            if quantize_weight:
                if slim_quant:
                    quantization_method = "SLiM-Quant"
                else:
                    if weight_tiled_quantization:
                        quantization_method = "Tiled Group AbsMax"
                    else:
                        quantization_method = "AbsMax"
                print(F"Pruning the model with Wanda and quantizing the weights using {quantization_method}.")
            else:
                print("Pruning the model with Wanda.")
            if lora_rank > 0:
                if slim_lora:
                    print(f"Adding SLiM-LoRA approximation with rank ratio {lora_rank}.")
                else:
                    print(f"Adding SVD LoRA approximation with rank ratio {lora_rank}.")
            if prune_lora and not (prune_n == 2 and prune_m == 4):
                raise NotImplementedError("Pruning LoRA is only supported for 2:4 sparsity ratio")
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
            )
        elif prune_method == "magnitude":
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
                calibration_dataset
            )
        elif prune_method == "joint_pq":
            if weight_tiled_quantization is False:
                raise NotImplementedError("Joint pruning and quantization only supports block quantization")
            if lora_rank > 0:
                raise NotImplementedError("LoRA approximation not implemented for Joint Pruning and Quantization")
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
                calibration_dataset
            )
        else:
            raise NotImplementedError(f"Pruning method {prune_method} not implemented")