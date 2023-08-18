from .m_n_sparsity import Sparsify as MNSparsify
import torch.nn.functional as F
from types import MethodType


def sparse_linear_forward(module, input):
    input = MNSparsify.apply(input, module.m, module.n)
    return F.linear(input, module.weight, module.bias)


def sparsify_inputs(model, m=2, n=4):
    print("Modifying Model to Sparse Inputs")
    known_modules = {"Linear"}
    for name, module in model.named_modules():
        if type(module).__name__ in known_modules:
            module.m = m
            module.n = n
            if type(module).__name__ == "Linear":
                module.forward = MethodType(sparse_linear_forward, module)
