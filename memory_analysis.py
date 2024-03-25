from kernels.sparse_backend import pruner
import argparse
import os
import torch


import subprocess

def get_allocated_memory():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
        output = output.decode('utf-8')
        allocated_memory = [int(x) for x in output.strip().split('\n')]
        return allocated_memory
    except subprocess.CalledProcessError:
        return None


parser = argparse.ArgumentParser()
parser.add_argument('--dim1', type=int, default=4096)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--dim2', type=int, default=-1)
parser.add_argument('--dense', action="store_true")
parser.add_argument('--full_compute', action="store_true")

args = parser.parse_args()

dim1 = args.dim1
dim2 = args.dim2 if args.dim2 > 0 else dim1
bs = args.batch_size

sparsity_string = "dense" if args.dense else "sparse"
print(f"Running memory analysis for matmul with dim1={dim1}, dim2={dim2}, batch_size={bs}")


def dense_memory(bs, dim1, dim2, full_compute=True):
    with torch.no_grad():
        init_memory_usage = get_allocated_memory()[0]
        x = torch.randn(bs, dim1).half().cuda()
        weight = torch.randn(dim1, dim2).half().cuda()
        y = torch.matmul(x, weight)
        grad_output = torch.randn(bs, dim2).half().cuda()
        grad_weight = torch.matmul(x.T, grad_output).cuda()
        grad_input = torch.matmul(grad_output, weight.T).cuda()
        exp_avg = torch.zeros_like(weight).half().cuda()
        exp_avg_sq = torch.zeros_like(weight).half().cuda()
        if not args.full_compute:
            del x, y, grad_output, grad_weight
        # Empty GPU cache
        torch.cuda.empty_cache()
        final_memory_usage = get_allocated_memory()[0]
        print("Allocated memory (in MiB) for each GPU:", final_memory_usage - init_memory_usage)
        return final_memory_usage - init_memory_usage

def spase_memory(bs, dim1, dim2, full_compute=True):
    with torch.no_grad():
        init_memory_usage = get_allocated_memory()[0]
        x = torch.randn(bs, dim1).half().cuda()
        weight = torch.randn(dim2, dim1).half().cuda()
        w_sparse_idx = pruner.setup_spmatmul(x, weight, False, True, False, True)
        y = pruner.spmatmul(x, w_sparse_idx, False)
        grad_output = torch.randn(bs, dim1).half().cuda()
        w_sparse_t_idx = pruner.setup_spmatmul(x, weight, False, False, False, True)
        mask = torch.zeros((dim2, dim1)).bool().cuda()
        del weight
        grad_weight = torch.zeros((dim1, dim2 // 2)).half().cuda()  # torch.matmul(x.T, grad_output).cuda()
        grad_input = pruner.spmatmul(grad_output, w_sparse_t_idx, False).cuda()
        exp_avg = torch.zeros((dim1, dim2 // 2)).half().cuda()
        exp_avg_sq = torch.zeros((dim1, dim2 // 2)).half().cuda()
        if not args.full_compute:
            del x, y, grad_output, grad_weight
        # Empty GPU cache
        torch.cuda.empty_cache()
        final_memory_usage = get_allocated_memory()[0]
        print("Allocated memory (in MiB) for each GPU:", final_memory_usage - init_memory_usage)
        return final_memory_usage - init_memory_usage


if args.dense:
    dense_memory(bs, dim1, dim2, args.full_compute)
else:
    spase_memory(bs, dim1, dim2, args.full_compute)




