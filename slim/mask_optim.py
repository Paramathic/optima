import torch
import numpy as np
from tqdm import tqdm
import transformers
from types import MethodType
import gc


def compute_error(model, inps, outs, model_kwargs):
    errors = []
    with torch.no_grad():
        for inp, out in zip(inps, outs):
            model_out = model(inp.unsqueeze(0).cuda(), **model_kwargs)[0].squeeze(0)
            errors.append((torch.norm(out.cuda() - model_out) / torch.norm(out.cuda())).item())

    return np.mean(errors)

structured_masks = torch.tensor(
    [
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0]
    ], 
    dtype=torch.bfloat16,
    device='cuda',
    )


def generate_unstructured_mask(mask_params, hard=False, temperature=1.0):
    """
    This function is used to generate an unstructured mask from the mask parameters.
    """
    mask = torch.nn.functional.gumbel_softmax(mask_params, dim=1, hard=hard, tau=temperature)
    return mask

def generate_2_4_mask(mask_params, hard=False, temperature=1.0):
    """
    This function is used to generate a 2:4 mask from the mask parameters.
    """
    mask = torch.nn.functional.gumbel_softmax(mask_params, dim=1, hard=hard, tau=temperature)
    return mask


def masked_linear(module, input):
    """
    This function is used to perform the quantized matmul operation.
    Args:
        module: The module to perform the operation on.
        input: The input to the module.
    """
    if module.mask_params.shape == module.weight.shape:
        mask = generate_unstructured_mask(module.mask_params)
    else:
        mask = generate_2_4_mask(module.mask_params)
        mask = mask @ structured_masks
        mask = mask.reshape(module.weight.shape)
    masked_weight = module.weight * mask
    module.masked_weight = masked_weight
    output = torch.matmul(input, masked_weight.t())
    if module.bias is not None:
        output += module.bias   
    return output


def get_optimizer(optimizer, params, lr, **kwargs):
    """
    This function is used to get the optimizer for the model.
    Args:
        optimizer: The optimizer to use for the model.
        params: The parameters to optimize.
        lr: The learning rate to use for the optimizer.
    """
    if optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, **kwargs)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, **kwargs)
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, **kwargs)
    elif optimizer == "adafactor":
        optimizer = transformers.Adafactor(params, lr=lr, relative_step=False, **kwargs)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented")
    return optimizer


def is_2_4_mask(mask):
    """
    This function is used to check if the mask is a 2:4 mask.
    """
    # is_2_4 = torch.all(mask.reshape(-1, 4).mean(dim=1) == 0.5)
    # return is_2_4
    return True


def block_wise_optimize_mask(
        block,
        model_kwargs,
        input_list,
        output_list,
        num_epochs=3,
        compute_dtype=torch.bfloat16,
        optimizer="adam",
        verbose=True,
        val_set_size=128,
        checkpoint_name="/tmp/checkpoint.pt",
        mask_strength=4.0,
        reg_factor=False,
):
    """
    This function is used to optimize the parameters of a block of the model.
    Args:
        block: The block of the model to optimize.
        model_kwargs: The kwargs to pass to the model.
        input_list: The list of inputs to the model.
        output_list: The list of outputs from the model.
        num_epochs: The number of epochs to train for.
        compute_dtype: The dtype to use for the model.
        optimizer: The optimizer to use for the model.
        verbose: Whether to print verbose output.
        val_set_size: The size of the validation set.
        checkpoint_name: The name of the checkpoint to save the model to.
    """

    if val_set_size <= 1.0:
        val_set_size = int(len(input_list) * val_set_size)

    for param in block.parameters():
            param.requires_grad = False
    dtype = param.dtype
    device = param.device
    block = block.to(compute_dtype)
    
    for name, module in block.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = MethodType(masked_linear, module)
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
            

            if is_2_4_mask(module.init_mask):
                mask = module.init_mask.reshape(-1, 4)
                mask = mask @ structured_masks.t() * mask_strength
            else:
                mask = (2 * mask_strength * module.init_mask) - mask_strength
                
            module.mask_params = torch.nn.Parameter(mask.to(device).to(compute_dtype).clone().detach()).cuda()
            # module.mask_params.requires_grad = True    

    metric = torch.nn.MSELoss()

    torch.save(block.state_dict(), checkpoint_name)

    init_loss_exact = compute_error(block, input_list, output_list, model_kwargs)

    if reg_factor is True:
        sum_weights = 0
        for name, module in block.named_modules():
            if isinstance(module, torch.nn.Linear):
                sum_weights += module.masked_weight.abs().sum()
        reg_factor = init_loss_exact / sum_weights.item()
    else:
        reg_factor = 0.

    with torch.set_grad_enabled(True):
        print("Searching for the best learning rate.")
        average_losses = []
        lr_list = [1e2, 5e1, 1e1, 5e0, 1e0, 5e-1, 1e-1, 5e-2, 1e-2]
        for lr in lr_list:
            block_copy = block
            block_copy.load_state_dict(torch.load(checkpoint_name))
            mask_search_optimizer = get_optimizer(optimizer, block_copy.parameters(), lr)
            mask_search_scheduler = torch.optim.lr_scheduler.LinearLR(mask_search_optimizer,
                                                     start_factor=1.0,
                                                     end_factor=1e-2,
                                                     total_iters=num_epochs * len(input_list)) # We use the exact same scheduler as in the actual training
            losses = []
            for input, output in zip(input_list[:val_set_size], output_list[:val_set_size]):
                input = input.to(device).to(compute_dtype)
                output = output.to(device).to(compute_dtype)
                y = block_copy(input.unsqueeze(0), **model_kwargs)[0].squeeze(0)
                loss = metric(y, output)
                reg_val = torch.tensor(0., device=device, dtype=compute_dtype)
                if reg_factor != 0.:
                    for name, module in block.named_modules():
                        if isinstance(module, torch.nn.Linear):
                            reg_val -= module.masked_weight.abs().sum() * reg_factor
                total_loss = loss + reg_val
                total_loss.backward()
                mask_search_optimizer.step()
                mask_search_scheduler.step()
                mask_search_optimizer.zero_grad()
                losses.append(loss.item())
            average_loss = np.mean(losses[-val_set_size//2:])
            average_losses.append(average_loss)
        lr = lr_list[np.argmin(average_losses)]
        del block_copy, losses, average_losses, mask_search_optimizer


    params = block.parameters()
    optimizer = get_optimizer(optimizer, params, lr)

    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                     start_factor=1.0,
                                                     end_factor=1e-2,
                                                     total_iters=num_epochs * len(input_list))  

    progress_bar = tqdm(range(num_epochs * len(input_list)), disable=not verbose)
    losses = []
    init = True
    with torch.set_grad_enabled(True):
        for epoch in range(num_epochs):
            for input, output in zip(input_list, output_list):
                input = input.to(device).to(compute_dtype)
                output = output.to(device).to(compute_dtype)
                y = block(input.unsqueeze(0), **model_kwargs)[0].squeeze(0)
                loss = metric(y, output)
                norm = metric(y, torch.zeros_like(y))
                if init:
                    init_loss = loss.item() / norm.item()
                    init = False
                reg_val = torch.tensor(0., device=device, dtype=compute_dtype)
                if reg_factor != 0.:
                    for name, module in block.named_modules():
                        if isinstance(module, torch.nn.Linear):
                            reg_val -= module.masked_weight.abs().sum() * reg_factor
                total_loss = loss + reg_val
                total_loss.backward()
                losses.append(loss.item()  / norm.item())
                average_loss = np.mean(losses[-100:])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix({'loss': average_loss, "lr": lr, "reg": reg_val.item()})
    if verbose:
        print(f"Initial Loss: {init_loss:.2e} - Final Loss: {average_loss:.2e}")

    final_loss = compute_error(block, input_list, output_list, model_kwargs)

    if verbose:
        print("Inside *** Initial Loss:", init_loss_exact, "Final Loss:", final_loss)

    block = block.to(dtype)

    with torch.no_grad():
        mask_similarity = []
        for name, module in block.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.forward = MethodType(torch.nn.Linear.forward, module)
                if is_2_4_mask(module.init_mask):
                    # Set the maximum elemnt of mask params to 1 and the rest to 0
                    mask = torch.zeros_like(module.mask_params)
                    max_vals = torch.max(module.mask_params, dim=1)[0].repeat_interleave(6).reshape(-1, 6)
                    max_indices = module.mask_params == max_vals
                    mask[max_indices] = 1.
                    # for i in range(mask.shape[0]):
                    #     mask[i, torch.argmax(module.mask_params[i, :])] = 1
                    module.mask = mask @ structured_masks
                    module.mask = module.mask.reshape(module.weight.shape)
                else:
                    nnz = module.init_mask.sum()
                    top_weights, top_indices = torch.topk(module.mask_params.abs(), k=int(nnz))
                    module.mask = torch.zeros_like(module.weight)
                    module.mask[top_indices] = 1
                mask_similarity.append(torch.mean((module.init_mask == module.mask).float()).item())
                module.mask = module.mask.to(dtype).cpu()
                module.mask.requires_grad = False
                module.init_mask = module.init_mask.cpu()
                module.mask_params.data = module.mask_params.cpu()
                module.weight.data[torch.logical_not(module.mask.bool())] = 0.
                del module.mask_params
                del module.init_mask
                gc.collect()
                torch.cuda.empty_cache()
        print(f"Mask similarity: ", mask_similarity)

    return init_loss, average_loss