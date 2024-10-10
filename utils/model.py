from transformers import  AutoModelForCausalLM
from slim.utils import get_layers_list, find_layers
import numpy as np
import torch


def get_llm(model_name,
            cache_dir="llm_weights",
            device_map=None,
            local_files_only=False,
            hf_token="",
            ):
    """
    Load a model from transformers
    model_name: str, the name of the model to load
    cache_dir: str, the directory to save the model weights
    local_checkpoint_dir: str, the directory to load the model weights from
    device_map: dict, a dictionary mapping device names to device objects
    local_files_only: bool, whether to only load local files
    """
    kwargs = {}
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        token=hf_token,
        local_files_only=local_files_only,
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

    model.seqlen = model.config.max_position_embeddings
    return model


def add_empty_lora(
        model,
        lora_tile_size=None,
        lora_rank=0.01,
):
    layer_list = get_layers_list(model)
    for i in range(len(layer_list)):
        layer = layer_list[i]
        subset = find_layers(layer)
        for name in subset:
            layer_rank = int(min(subset[name].weight.shape) * lora_rank)
            if lora_tile_size is not None:
                tile_dim = int(np.sqrt(lora_tile_size))
                residue = layer_rank % tile_dim
                if residue != 0:
                    layer_rank = layer_rank + (tile_dim - residue)
                assert layer_rank % tile_dim == 0
            subset[name].lora_left = torch.nn.Parameter(
                torch.zeros((subset[name].weight.shape[1], layer_rank), device=subset[name].weight.device).half())
            subset[name].lora_right = torch.nn.Parameter(
                torch.zeros((layer_rank, subset[name].weight.shape[0]), device=subset[name].weight.device).half())

            def add_lora_hook(module, input, output):
                output += torch.matmul(
                    torch.matmul(input[0].to(module.lora_left.dtype) / torch.sqrt(module.lora_rank), module.lora_left),
                    module.lora_right) / torch.sqrt(module.lora_rank)

            subset[name].lora_rank = torch.tensor(layer_rank)
            subset[name].register_forward_hook(add_lora_hook)


def contigous_model(model):
    for layer in get_layers_list(model):
        for param in layer.parameters():
            param.data = param.data.contiguous()
    return model