import time
import heapq
import torch
import torch.nn as nn
from .sparsegpt import SparseGPT
from .sparsegpt import Quantizer as SparseGPTQuantizer
from .layerwrapper import WrappedGPT
from .data import get_loaders

from .ablate import AblateGPT

from .utils import add_lora, get_layers_list, shift_zeros, accelerate_module
from compression.quantization.model_quantizing import Quantizer as AutoQuantizer
from transformers import LlamaForCausalLM
from compression.ops import prune_row_wise


def find_layers(module, layers=[nn.Linear], name=''):
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


def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = get_layers_list(model)
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"Layer {i} Sparsity Ratio: {float(sub_count) / sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


def prepare_calibration_input(model, dataloader, device):
    use_cpu = model.device == torch.device("cpu")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_layers_list(model)

    try:
        if "model.embed_tokens" in model.hf_device_map:
            device = model.hf_device_map["model.embed_tokens"]
    except:
        pass

    if use_cpu:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cuda()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cuda()
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cuda()
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cuda()
        layers[0] = layers[0].cuda()

    dtype = next(iter(model.parameters())).dtype
    try:
        print("Memory usage before loading inputs: ", torch.cuda.memory_allocated(device) / 1024 ** 3, "GB")
        inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
        input_device = inps.device
    except:
        torch.cuda.empty_cache()
        print("Memory usage after failing loading inputs: ", torch.cuda.memory_allocated(device) / 1024 ** 3, "GB")
        print("Inputs don't fit in the GPU. Storing them in CPU...")
        inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device="cpu")
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
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    if use_cpu:
        layers[0] = layers[0].cpu()
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
        torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    if is_llama:
        position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    if is_llama:
        return inps, outs, attention_mask, position_ids
    return inps, outs, attention_mask


def save_bias_correction_data(model, layers, inps, outs, attention_mask, single_gpu_en, device, nsamples):
    layers_inps_outs = {}

    for i in range(len(layers)):
        if single_gpu_en:
            layer = layers[i].to(device)
        else:
            layer = layers[i]

        subset = find_layers(layer)
        sample_counter = 0
        try:
            if f"model.layers.{i}" in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)
        except:
            pass

        def save_layer_inp_out(module, input, output):
            if sample_counter == 0:
                layers_inps_outs[id(module)] = (
                torch.zeros((nsamples, input[0].squeeze().shape[0], input[0].squeeze().shape[1]),
                            dtype=next(iter(model.parameters())).dtype, device=torch.device("cpu")),
                torch.zeros((nsamples, output.squeeze().shape[0], output.squeeze().shape[1]),
                            dtype=next(iter(model.parameters())).dtype, device=torch.device("cpu")))

            layers_inps_outs[id(module)][0][sample_counter] = input[0].detach().clone().squeeze().to(
                torch.device("cpu"))
            layers_inps_outs[id(module)][1][sample_counter] = output.detach().clone().squeeze().to(torch.device("cpu"))

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(save_layer_inp_out))
        coeff = int(128 / nsamples)
        for j in range(nsamples):
            with torch.no_grad():
                outs[coeff * j] = layer(inps[coeff * j].unsqueeze(0), attention_mask=attention_mask)[0]
                sample_counter += 1
        for h in handles:
            h.remove()

        inps, outs = outs, inps

        if single_gpu_en:
            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()

    return layers_inps_outs


def update_outputs_for_bias_correction(model, layers, layers_inps_outs, single_gpu_en, device, alpha, nsamples):
    def update_layer_inp_out(module, input, output):
        layers_inps_outs[id(module)][1][sample_counter] -= output.detach().clone().squeeze().to(torch.device("cpu"))

    for i in range(len(layers)):
        if single_gpu_en:
            layer = layers[i].to(device)
        else:
            layer = layers[i]

        subset = find_layers(layer)
        sample_counter = 0
        try:
            if f"model.layers.{i}" in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)
        except:
            pass

        sample_counter = 0
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(update_layer_inp_out))
        print(nsamples)
        for j in range(nsamples):
            for name in subset:
                with torch.no_grad():
                    layer_input = layers_inps_outs[id(subset[name])][0][sample_counter].to(
                        subset[name].weight.data.device)
                    subset[name](layer_input)
                    del layer_input
                    torch.cuda.empty_cache()
            sample_counter += 1

        for h in handles:
            h.remove()

        correct_bias(subset, layers_inps_outs, alpha)

        if single_gpu_en:
            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()

    return layers_inps_outs


def correct_bias(subset, layers_outs, alpha):
    print("bias correction alpha: " + str(alpha))
    for name in subset:
        subset[name].bias.data += alpha * torch.mean(
            layers_outs[id(subset[name])][1].view(-1, layers_outs[id(subset[name])][1].shape[2]).to(
                subset[name].bias.data.device), dim=0)


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = get_layers_list(model)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W) == 1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel() * args.sparsity_ratio)].cpu()
                W_mask = (W_metric <= thresh)

            W[W_mask] = 0


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    is_llama = isinstance(model, LlamaForCausalLM)

    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    with torch.no_grad():
        if is_llama:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
        else:
            inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device)

    if args.quantize:
        quantizer = AutoQuantizer("weight")
    else:
        quantizer = None

    layers = get_layers_list(model)

    if args.bias_correction:
        raise NotImplementedError
        # layers_inps_outs = save_bias_correction_data(model, layers, inps, outs, attention_mask, single_gpu_en, device, args.bias_correction_nsamples)

        # with torch.no_grad():
        #     if is_llama:
        #         inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device,
        #                                                                              single_gpu_en)
        #     else:
        #         inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device, single_gpu_en)

    for i in range(len(layers)):
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
        for j in range(args.nsamples):
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

        for name in subset:
            print(f"pruning layer {i} name {name}")
            if args.shift_zero_metrics:
                wrapped_layers[name].scaler_row = shift_zeros(wrapped_layers[name].scaler_row)
            if args.quantize_before_pruning and quantizer is not None:
                quantized_weight = quantizer.dequantize_absmax(
                    quantizer.quantize_weight(
                        subset[name].weight.data,
                        args.bitwidth,
                        use_std=args.use_std_in_quantization,
                        max_bitwidth=args.max_bitwidth
                    )
                )
                W_metric = torch.abs(quantized_weight) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            else:
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                # print(prune_n, prune_m)
                # _, W_mask = prune_row_wise(W_metric, prune_n, prune_m)
                # print("New: ", torch.mean(W_mask.float()))
                # W_mask = W_mask == False
                # print(torch.mean(W_mask.float()))
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                # unstructured pruning
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            if args.lora_rank > 0.:
                add_lora(subset[name],
                         W_mask=W_mask,
                         rank_ratio=args.lora_rank,
                         use_wanda=args.wanda_in_lora,
                         activations=wrapped_layers[name],
                         use_randomized_svd=args.randomized_svd,
                         quantizer=quantizer,
                         bitwidth=args.bitwidth,
                         use_std=args.use_std_in_quantization,
                         max_bitwidth=args.max_bitwidth,
                         pruned_L=args.pruned_l,
                         separate_lora=args.separate_lora)

                if args.separate_lora:
                    def add_lora_hook(module, input, output):
                        output += torch.matmul(torch.matmul(input[0], module.lora_left),
                                               module.lora_right)

                    subset[name].register_forward_hook(add_lora_hook)


            else:
                subset[name].weight.data[W_mask] = 0  ## set weights to zero
                if (not args.quantize_before_pruning) and quantizer is not None:
                    quantized_weight = quantizer.quantize_weight(subset[name].weight.data,
                                                                         args.bitwidth,
                                                                         use_std=args.use_std_in_quantization,
                                                                         max_bitwidth=args.max_bitwidth)
                    subset[name].weight.data = quantizer.dequantize_absmax(quantized_weight)

        if cpu_only:
            layer = layer.float()

        for j in range(args.nsamples):
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

    if args.bias_correction:
        raise NotImplementedError
        # layers_inps_outs = update_outputs_for_bias_correction(model, layers, layers_inps_outs, single_gpu_en, device, args.bias_alpha, args.bias_correction_nsamples)

    if args.accelerate:
        for i in range(len(layers)):
            if layer.device == torch.device("cpu"):
                raise NotImplemented

            layer = layers[i]

            subset = find_layers(layer)
            for name in subset:
                accelerate_module(subset[name], args.quantize, args.bitwidth)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    use_cache = model.config.use_cache
    model.config.use_cache = False

    is_llama = isinstance(model, LlamaForCausalLM)

    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    with torch.no_grad():
        if is_llama:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, dev)
        else:
            inps, outs, attention_mask = prepare_calibration_input(model, dataloader, dev)

    layers = get_layers_list(model)


    print('Ready.')

    for i in range(len(layers)):
        use_cpu = model.device == torch.device("cpu")
        cpu_only = False
        layer = layers[i].cuda() if use_cpu else layers[i]

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])
            if args.quantize < 16:
                gpts[name].quantizer = SparseGPTQuantizer()
                gpts[name].quantizer.configure(
                    args.bitwidth, perchannel=False, sym=True, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
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
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()


        if cpu_only:
            layer = layer.float()

        for j in range(args.nsamples):
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

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_layers_list(model)

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    # position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask,
                                   prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def quantize_model(args, model):
    quantizer = AutoQuantizer("weight")
    layers = get_layers_list(model)
    
    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)
        
        for name in subset:
            print(f"Quantizing layer {i} name {name}")
            
            quantized_weight = quantizer.dequantize_absmax(
                quantizer.quantize_weight(
                    subset[name].weight.data,
                    args.bitwidth,
                    use_std=args.use_std_in_quantization,
                    max_bitwidth=args.max_bitwidth
                )
            )
            
            subset[name].weight.data = quantized_weight.to(subset[name].weight.dtype)

