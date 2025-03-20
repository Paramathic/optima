import torch
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test_24 import (
    marlin_24_quantize)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace, marlin_quantize)
from vllm.model_executor.layers.quantization.gptq_marlin_24 import (
    GPTQ_MARLIN_24_MAX_PARALLEL, GPTQ_MARLIN_24_MIN_THREAD_N)
from vllm import _custom_ops as ops
from types import MethodType


def slim_forward(module, input):
    if input.dim() == 3:
        bs, seqlen, d_in = input.shape
    else:
        bs, d_in = input.shape
        seqlen = 1
    # if seqlen == 1:
    xw = ops.gptq_marlin_24_gemm(input.view(-1, d_in),
                                 module.marlin_24_q_w_comp,
                                 module.marlin_24_meta,
                                 module.marlin_24_s,
                                 module.marlin_24_workspace.scratch,
                                 module.quant_type,
                                 bs*seqlen,
                                 module.d_out,
                                 d_in)
    if hasattr(module, "lora_left"):
        xl = torch.matmul(input.view(-1, d_in), module.lora_left)
        torch.addmm(xw, xl, module.lora_right, out=xw)
    if input.dim() == 3:
        output = xw.view(bs, seqlen, module.d_out)
    else:
        output = xw
    # else:
    #     # raise NotImplementedError("Not implemented for seqlen > 1")
    #     output = torch.matmul(input, module.weight.t())

    if not module.bias is None:
        output += module.bias
    return output


def compress_model(
        model,
        lora_rank=0.1,
        pad_lora=True,
        quantize_lora=False,
        lora_group_size=128,
        quant_type=scalar_types.uint4b8,
        group_size=-1,
        skip_layers=[],
):
    marlin_workspaces = {}
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            if layer in skip_layers:
                continue
            weight = layer.weight.data.clone().detach().cuda()
            # Create a MarlinWorkspace object for the layer
            d_out = weight.shape[0]
            if not d_out in marlin_workspaces:
                marlin_workspaces[d_out] = MarlinWorkspace(d_out, GPTQ_MARLIN_24_MIN_THREAD_N,
                                                        GPTQ_MARLIN_24_MAX_PARALLEL)
            layer.quant_type = quant_type
            layer.marlin_24_workspace = marlin_workspaces[d_out]
            layer.d_out = weight.shape[0]
            (marlin_24_w_ref, layer.marlin_24_q_w_comp, layer.marlin_24_meta,
             layer.marlin_24_s) = marlin_24_quantize(weight.t(), quant_type, group_size)

            # layer.weight.data[marlin_24_w_ref == 0] = 0
            # print(torch.norm(layer.weight - marlin_24_w_ref.float()) / torch.norm(layer.weight.float()))
            # print(layer.weight[0:8, 0:8])
            # print(marlin_24_w_ref[0:8, 0:8])
            if lora_rank > 0:
                rank = int(min(layer.weight.shape) * lora_rank)
                if pad_lora:
                    residue = rank % lora_group_size
                    if residue != 0:
                        # rank = rank + (lora_group_size - residue)
                        rank = rank - residue
                    assert rank % lora_group_size == 0
                lora_left = torch.randn([weight.shape[1], rank],
                                        device=weight.device,
                                        dtype=weight.dtype,
                                        ) / 100
                lora_right = torch.randn([rank, weight.shape[0]],
                                         device=weight.device,
                                         dtype=weight.dtype,
                                         ) / 100
                

            del layer.weight, weight, marlin_24_w_ref
            torch.cuda.empty_cache()

            if quantize_lora:
                raise NotImplementedError("Quantizing LoRA is not supported yet.")
            else:
                if lora_rank > 0:
                    layer.lora_left = lora_left
                    layer.lora_right = lora_right
                layer.forward = MethodType(slim_forward, layer)




if __name__ == "__main__":
    model = torch.nn.Linear(1024, 1024, dtype=torch.float16, device="cuda")
    compress_model(model)