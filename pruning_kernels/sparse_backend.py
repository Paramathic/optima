from torch.utils.cpp_extension import load
import os
import torch

base_path = __file__.replace("sparse_backend.py", "")

if torch.cuda.get_device_capability()[0] < 8:
    base_path += "cuda_cores"
    if not os.path.exists(f"{base_path}/build"):
        os.makedirs(f"{base_path}/build")
    pruner = load(name='pruner',
                  sources=[f'{base_path}/sparse_backend.cpp',
                           f'{base_path}/sparse_backend.cu'],
                  build_directory=f'{base_path}/build',
                  verbose=True,
                  )
else:
    base_path += "tensor_cores"
    if not os.path.exists(f"{base_path}/build"):
        os.makedirs(f"{base_path}/build")

    if not os.path.exists(base_path + "/libcusparse_lt"):
        os.system(
            "wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/"
            "libcusparse_lt-linux-x86_64-0.4.0.7-archive.tar.xz")
        os.system("tar -xf libcusparse_lt-linux-x86_64-0.4.0.7-archive.tar.xz")
        os.system(f"mv libcusparse_lt-linux-x86_64-0.4.0.7-archive {base_path}/libcusparse_lt")
        os.system("rm libcusparse_lt-linux-x86_64-0.4.0.7-archive.tar.xz")

    pruner = load(name='pruner',
                  sources=[f'{base_path}/sparse_backend.cpp',
                           f'{base_path}/sparse_backend.cu',
                           ],
                  extra_cflags=[
                      f'-L{base_path}/libcusparse_lt/lib',
                      '-lcusparse',
                      '-lcusparseLt',
                      '-ldl'
                  ],
                  extra_cuda_cflags=[
                      f'-L{base_path}/libcusparse_lt/lib',
                      '-lcusparse',
                      '-lcusparseLt',
                      '-ldl'
                  ],
                  extra_ldflags=[
                      f'-L{base_path}/libcusparse_lt/lib',
                      '-lcusparse',
                      '-lcusparseLt',
                      '-ldl'
                  ],
                  extra_include_paths=[
                      base_path + '/libcusparse_lt/include'
                  ],
                  build_directory=f'{base_path}/build',
                  with_cuda=True,
                  verbose=False)

    init_flag = pruner.init_cusparse_lt()
    assert init_flag == 0, "Failed to initialize CuSparseLT"


if __name__ == "__main__":
    if torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.int8
        bs = 32
        dim1 = 32
        dim2 = 32
        x = torch.randn(bs, dim1).to(dtype).cuda()
        weight = torch.randn(dim2, dim1).to(dtype).cuda()
        w_sparse_idx = pruner.setup_spmatmul(x, weight, False, True, False, True)
        y_sparse = pruner.spmatmul(x, w_sparse_idx, False)
        if dtype == torch.float16:
            y_dense = torch.matmul(x, weight.t()).cuda()
        elif dtype == torch.int8:
            y_dense = torch.matmul(x.float(), weight.t().float()).cuda()
            y_dense = y_dense.to(dtype)
        print("SPMM Relative Error: ", ((y_dense - y_sparse).float().norm() / y_dense.float().norm()).item())