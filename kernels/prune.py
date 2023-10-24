from torch.utils.cpp_extension import load


pruner = load(name='pruner', sources=['./pruning/kernels/prune.cpp', './pruning/kernels/prune.cu'], build_directory="./pruning/kernels/build", verbose=True)