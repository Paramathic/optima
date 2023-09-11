from .m_n_sparsity import Sparsify as MNSparsify
import torch.nn.functional as F
from types import MethodType
from src.utils import is_main_process


def sparse_linear_forward(module, input):
    input = MNSparsify.apply(input)
    return F.linear(input, module.weight, module.bias)

def sparse_linear_activation_forward(module, input):
    input = MNSparsify.apply(input)
    if not module.bias is None:
        return module.biased_act_fn(module.bias, F.linear(input, module.weight, None))
    else:
        return module.act_fn(F.linear(input, module.weight, module.bias))



def sparsify_inputs(model, skip_layers=[]):
    if is_main_process():
        print("Modifying Model to Sparse Inputs")
    known_modules = {"Linear", "LinearActivation"}
    for name, module in model.named_modules():
        if type(module).__name__ in known_modules:
            if module in skip_layers:
                if is_main_process():   
                    print("Skipping Module: ", module)
                continue
            
            if type(module).__name__ == "Linear":
                module.forward = MethodType(sparse_linear_forward, module)
            elif type(module).__name__ == "LinearActivation":
                module.forward = MethodType(sparse_linear_activation_forward, module)
