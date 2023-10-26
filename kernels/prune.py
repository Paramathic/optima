from torch.utils.cpp_extension import load
import os

base_path = "."

if not os.path.exists(base_path + "/pruning/kernels/build"):
    os.makedirs(base_path + "/pruning/kernels/build")

pruner = load(name='pruner', sources=[base_path + '/pruning/kernels/prune.cpp', base_path + '/pruning/kernels/prune.cu'], build_directory=base_path + "/pruning/kernels/build", verbose=True)
