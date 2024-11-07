import torch
import numpy as np
from .quantization import Quantizer as AutoQuantizer
import tqdm.auto as tqdm
from .utils import prune_nm, get_layers_list, find_layers


def prune_and_optimize_lora(
        L,
        R,
        num_iters=1000,
        lr_end_factor=1e-4
):
    """
    Prune L in LoRA and optimizer L and R to compensate for the pruning loss.

    Args:
        L: torch.Tensor, The left matrix in LoRA
        R: torch.Tensor, The right matrix in LoRA
        num_iters: int, The number of optimization iterations
        lr_end_factor: float, The factor to scale the learning rate by at the end of optimization

    Returns:
        torch.Tensor, The mask of the pruned elements in L
    """
    target = torch.matmul(L, R).float()
    target_norm = torch.norm(target).item()
    L_mask = prune_nm(L.t(), 2, 4).t()
    L[L_mask] = 0
    L_param = torch.nn.Parameter(L.float(), requires_grad=True)
    R_param = torch.nn.Parameter(R.float(), requires_grad=True)
    optimizer = torch.optim.Adam([L_param, R_param], lr=1e6 / min(L.shape[0], R.shape[1]) ** 2)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=lr_end_factor, total_iters=num_iters)
    progress_bar = tqdm.tqdm(range(num_iters))
    initial_error = torch.norm(torch.matmul(L, R).float() - target.float()) / target_norm
    for iter in progress_bar:
        output = torch.matmul(L_param, R_param)
        loss = torch.norm(output - target) / target_norm
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        L_param.data[L_mask] = 0
        progress_bar.set_description(
            'Iteration {} - Initial Loss: {:.2f} - Current Loss: {:.2f}, LR: {:.2e}'.format(
                iter + 1,
                initial_error.item(),
                loss.item(),
                scheduler.get_lr()[0]
            )
        )
    L.data = L_param.data.half()
    R.data = R_param.data.half()
    return L_mask


def quantize_lora(
        model,
        bitwidth=8,
        lora_tile_size=256
):
    """
    Quantize the LoRA matrices in a model.

    Args:
        model: nn.Module, The model to quantize
        bitwidth: int, The number of bits to quantize the LoRA matrices to
        lora_tile_size: int, The size of the

    Returns:
        None
    """

    quantizer = AutoQuantizer("weight")
    layers = get_layers_list(model)

    progress_bar = tqdm.tqdm(range(len(layers)))

    for i in progress_bar:
        layer = layers[i]

        subset = find_layers(layer)

        for name in subset:
            progress_bar.set_description(f"Layer {i} - Quantizing LoRA for {name}")

            quantized_lora_left = quantizer.dequantize_absmax(
                quantizer.quantize_weight(
                    subset[name].lora_left.data,
                    bitwidth,
                    slim_quant=False,
                    block_quantization=True,
                    block_dim=int(np.sqrt(lora_tile_size)),
                )
            )

            quantized_lora_right = quantizer.dequantize_absmax(
                quantizer.quantize_weight(
                    subset[name].lora_right.data,
                    bitwidth,
                    slim_quant=False,
                    block_quantization=True,
                    block_dim=int(np.sqrt(lora_tile_size)),
                )
            )

            subset[name].lora_left.data = quantized_lora_left.to(subset[name].weight.dtype)
            subset[name].lora_right.data = quantized_lora_right.to(subset[name].weight.dtype)



def add_lora(
        module,
        W_mask,
        rank_ratio=0.01,
        slim_lora=False,
        activations=None,
        quantizer=None,
        bitwidth=8,
        slim_quant=False,
        prune_lora=False,
        separate_lora=True,
        block_quantization=False,
        weight_tile_size=256,
        lora_tile_size=None
):
    """
    Add low-rank adapters to compensate for the compression loss.

    Args:
        module: nn.Module, The module to add the low-rank adapters to
        W_mask: torch.Tensor, The mask of the pruned weights
        rank_ratio: float, The ratio of the rank of the low-rank approximation to the number of rows in the weight matrix
        slim_lora: bool, Whether to use slim LoRA
        activations: torch.Tensor, The activations of the layer
        quantizer: Quantizer, The quantizer to use
        bitwidth: int, The number of bits to quantize the weights to
        slim_quant: bool, Whether to use slim quantization
        prune_lora: bool, Whether to prune the LoRA matrices
        separate_lora: bool, Whether to use separate LoRA matrices
        block_quantization: bool, Whether to use block quantization
        weight_tile_size: int, The size of the weight tiles
        lora_tile_size: int, The size of the LoRA tiles
    """
    if slim_lora and not any(activations.scaler_row == 0):
        if quantizer is None:
            W_metric = module.weight.data * (torch.sqrt(activations.scaler_row.reshape((1, -1))))
            new_weight = W_metric.clone().detach()
            new_weight[W_mask] = 0
            error_mat = W_metric - new_weight
        else:
            W_metric = module.weight.data * (torch.sqrt(activations.scaler_row.reshape((1, -1))))
            new_weight = module.weight.data
            new_weight[W_mask] = 0
            new_weight = quantizer.quantize_weight(new_weight,
                                                   bitwidth,
                                                   slim_quant=slim_quant,
                                                   block_quantization=block_quantization,
                                                   block_dim=int(np.sqrt(weight_tile_size)),
                                                   )
            new_weight = quantizer.dequantize_absmax(new_weight) * (torch.sqrt(activations.scaler_row.reshape((1, -1))))
            error_mat = (W_metric - new_weight)
    else:
        new_weight = module.weight.data.clone().detach()
        new_weight[W_mask] = 0
        if quantizer is not None:
            new_weight = quantizer.quantize_weight(new_weight,
                                                   bitwidth,
                                                   slim_quant=slim_quant,
                                                   block_quantization=block_quantization,
                                                   block_dim=int(np.sqrt(weight_tile_size)),
                                                   )
            new_weight = quantizer.dequantize_absmax(new_weight)
        error_mat = module.weight.data - new_weight

    # Use SVD on the error matrix to find the best low-rank approximation
    U, S, V = torch.svd(error_mat.float())

    rank = int(rank_ratio * min(error_mat.shape))

    if lora_tile_size is not None:
        tile_dim = int(np.sqrt(lora_tile_size))
        residue = rank % tile_dim
        if residue != 0:
            rank = rank + (tile_dim - residue)
        assert rank % tile_dim == 0

    if separate_lora:
        lora_left = (torch.diag_embed(S[:rank]).half() @ V[:, :rank].half().T).t()
        lora_right = U[:, :rank].half().t()
        if prune_lora:
            lora_left_mask = prune_and_optimize_lora(lora_left, lora_right)
    else:
        low_rank_weight = U[:, :rank].half() @ torch.diag_embed(S[:rank]).half() @ V[:, :rank].half().T
        if prune_lora:
            raise NotImplementedError
    if slim_lora and not any(activations.scaler_row == 0):
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
        new_weight = quantizer.quantize_weight(new_weight,
                                               bitwidth,
                                               slim_quant=slim_quant,
                                               block_quantization=block_quantization,
                                               block_dim=int(np.sqrt(weight_tile_size)),
                                               )
        new_weight = quantizer.dequantize_absmax(new_weight)

    if separate_lora:
        module.lora_left = torch.nn.Parameter(lora_left).half().contiguous()
        module.lora_right = torch.nn.Parameter(lora_right).half().contiguous()
        module.weight.data = new_weight.half().contiguous()
        if prune_lora:
            module.lora_left_mask = lora_left_mask
    else:
        module.weight.data = (new_weight + low_rank_weight).half()
