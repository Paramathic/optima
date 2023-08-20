import torch
from .kernels.prune import pruner

def compute_remainder_sparsity(mat, m, n, init_sparsity=None):
    if init_sparsity is None:
        init_sparsity = 1 - (mat.sum() / mat.numel())
    reshaped_mat = mat.reshape(-1, n).to(torch.int8)
    reshaped_mat, _ = torch.sort(reshaped_mat, dim=1, descending=True)
    reshaped_mat[:, 0:m] = 0.
    final_sparsity = 1 - (reshaped_mat.sum() / reshaped_mat.numel())
    return final_sparsity, init_sparsity - final_sparsity


class Sparsify(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input):
        # assert (m, n, input.dtype) in [(1, 2, torch.float32), (2, 4, torch.float16)], f"Invalid Sparsity: m={m}, n={n}, input.dtype={input.dtype}"
        n = 2 if input.dtype == torch.float32 else 4
        assert input.shape[-1] % (n) == 0, f"Invalid Shape for m:n Sparsification: input.shape={input.shape}, n={n}"
        input_shape = input.shape
        sparse_input, mask = pruner.prune(input.reshape(-1, input.shape[-1])) #sparsify(input, m, n)
        sparse_input = sparse_input.reshape(input_shape)
        mask = mask.reshape(input_shape)
        ctx.save_for_backward(mask)
        return sparse_input

    @staticmethod
    def backward(ctx, grad_output):
        sparsity_mask = ctx.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[sparsity_mask] = 0.
        return grad_input


def sparsify(mat, m, n):
    reshaped_mat = mat.clone().reshape(-1, n)
    mask = torch.zeros_like(reshaped_mat, dtype=torch.bool)
    if (m, n) == (1, 2):
        _, indices = torch.topk(torch.abs(reshaped_mat), k=m, dim=1, sorted=False, largest=True)
        rows = (indices == 1).sum(dim=1)
        mask[:, 0] = rows
        mask[:, 1] = torch.logical_not(rows)
    elif (m, n) == (2, 4):
        _, indices = torch.topk(torch.abs(reshaped_mat), k=m, dim=1, sorted=False, largest=True)
        rows = torch.logical_not((indices == 0).sum(dim=1))
        mask[:, 0] = rows
        rows = torch.logical_not((indices == 1).sum(dim=1))
        mask[:, 1] = rows
        rows = torch.logical_not((indices == 2).sum(dim=1))
        mask[:, 2] = rows
        rows = torch.logical_not((indices == 3).sum(dim=1))
        mask[:, 3] = rows
    elif m < n / 2:
        _, indices = torch.topk(torch.abs(reshaped_mat), k=m, dim=1, sorted=False, largest=True)
        for i in range(n):
            rows = torch.logical_not((indices == i).sum(dim=1))
            mask[:, i] = rows
    else:
        _, indices = torch.topk(torch.abs(reshaped_mat), k=(n - m), dim=1, sorted=False, largest=False)
        for i in range(n):
            rows = (indices == i).sum(dim=1)
            mask[:, i] = rows
    reshaped_mat[mask] = 0.
    return reshaped_mat.reshape(mat.shape), mask.reshape(mat.shape)



if __name__ == '__main__':
    mat = torch.rand(8, 8)
    print(mat)
    mat, mask = sparsify(mat, 2, 4)
    print(mat)