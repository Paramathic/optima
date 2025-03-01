from slim.utils import get_layers_list, find_layers
import torch
import lm_eval


def get_llm(model_name,
            local_files_only=False,
            hf_token="",
            seqlen=2048
            ):
    """
    Load a model from transformers
    model_name: str, the name of the model to load
    local_files_only: bool, whether to only load local files
    hf_token: str, the huggingface token to use
    seqlen: int, the maximum sequence length to use
    """
    model_args = f"pretrained={model_name},dtype=half,local_files_only={local_files_only},low_cpu_mem_usage=True,token={hf_token}"
    lm_eval_model = lm_eval.api.registry.get_model("hf").create_from_arg_string(
        model_args,
        {
            "batch_size": None,
            "max_batch_size": None,
            "device": None,
        },
    )
    model = lm_eval_model._model
    model.config.max_position_embeddings = seqlen
    model.seqlen = seqlen
    return model, lm_eval_model


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
                tile_dim = lora_tile_size
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