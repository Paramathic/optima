import torch
from .kernels.prune import pruner


force_2_4 = False


def density_ratio(mat):
    return (mat != 0.).sum() / mat.numel()


def compute_remainder_sparsity(mat, m, n, init_sparsity=None):
    if init_sparsity is None:
        init_sparsity = 1 - (mat.sum() / mat.numel())
    reshaped_mat = mat.reshape(-1, n).to(torch.int8)
    reshaped_mat, _ = torch.sort(reshaped_mat, dim=1, descending=True)
    reshaped_mat[:, 0:m] = 0.
    final_sparsity = 1 - (reshaped_mat.sum() / reshaped_mat.numel())
    return final_sparsity, init_sparsity - final_sparsity


def prune_row_wise(input):
    if force_2_4:
        m, n = 2, 4
        dtype = input.dtype
        input = input.contiguous()
        input_shape = input.shape
        half_input = input.half()
        sparse_input, mask = pruner.prune(half_input.reshape(-1, input.shape[-1]))
        if dtype == torch.float32:
            sparse_input = input.clone().reshape(mask.shape)
            sparse_input[mask] = 0.
    else:
        n = 2 if input.dtype == torch.float32 else 4
        assert input.dtype in [torch.float16, torch.float32]
        assert input.shape[-1] % (n) == 0, f"Invalid Shape for m:n Sparsification: input.shape={input.shape}, n={n}"
        input = input.contiguous()
        input_shape = input.shape
        sparse_input, mask = pruner.prune(input.reshape(-1, input.shape[-1]))  # sparsify(input, m, n)
    sparse_input = sparse_input.reshape(input_shape)
    mask = mask.reshape(input_shape)
    return sparse_input, mask


def prune_column_wise(input, transpose=False):
    assert not (transpose and (input.dim() != 2))
    input_shape = input.shape
    input = input.reshape(-1, input.shape[-1])
    sparse_input, mask = prune_row_wise(input.t())
    if not transpose:
        sparse_input = sparse_input.t()
        mask = mask.t()
        sparse_input = sparse_input.reshape(input_shape)
        mask = mask.reshape(input_shape)
    return sparse_input, mask


class Sparsify(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input):
        sparse_input, mask = prune_row_wise(input)
        ctx.save_for_backward(mask)
        return sparse_input

    @staticmethod
    def backward(ctx, grad_output):
        sparsity_mask = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[sparsity_mask] = 0.
        return grad_input


class Matmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return torch.matmul(input, weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        grad_weight = torch.matmul(grad_output.t().to(dtype), input.to(dtype))
        grad_input = torch.matmul(grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        return grad_input, grad_weight


class DynamicPruneInputsMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight):
        sparse_input, mask = prune_row_wise(input)
        ctx.save_for_backward(sparse_input, weight, mask)
        return torch.matmul(sparse_input, weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, sparsity_mask = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        grad_weight = torch.matmul(grad_output.t().to(dtype), input.to(dtype))
        grad_input = torch.matmul(grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        grad_input[sparsity_mask] = 0.
        return grad_input, grad_weight


class ReductionDimDynamicPruneInputsMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight):
        sparse_input, mask = prune_row_wise(input)
        ctx.save_for_backward(sparse_input, weight, mask)
        return torch.matmul(sparse_input, weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, sparsity_mask = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        input = input.to(dtype)
        grad_output = grad_output.to(dtype)
        weight = weight.to(dtype)

        input2, _ = prune_column_wise(input, transpose=False)
        if density_ratio(input2) > 0.37:
            input = input2
        with open("density.csv", "a") as f:
            f.write(f"{density_ratio(input2)}\n")
        grad_weight = torch.matmul(grad_output.t(), input)

        grad_input = torch.matmul(grad_output, weight)
        grad_input = grad_input.reshape(input_shape)
        grad_input[sparsity_mask] = 0.
        return grad_input, grad_weight


class StaticPruneInputsMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight, mask=None):
        if mask is None:
            sparse_input, mask = prune_row_wise(input)
        else:
            sparse_input = input
            sparse_input[mask] = 0.
        ctx.save_for_backward(sparse_input, weight, mask)
        return torch.matmul(sparse_input, weight.t()), mask

    @staticmethod
    def backward(ctx, grad_output, grad_mask):
        input, weight, sparsity_mask = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        grad_weight = torch.matmul(grad_output.t().to(dtype), input.to(dtype))
        grad_input = torch.matmul(grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        grad_input[sparsity_mask] = 0.
        return grad_input, grad_weight, None


class ReductionDimStaticPruneInputsMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight, mask=None):
        if mask is None:
            sparse_input, mask = prune_row_wise(torch.rand_like(input))
        else:
            try:
                sparse_input = input
                sparse_input[mask] = 0.
            except:
                sparse_input = input
        ctx.save_for_backward(sparse_input, weight, mask)
        return torch.matmul(sparse_input, weight.t()), mask

    @staticmethod
    def backward(ctx, grad_output, grad_mask):
        input, weight, sparsity_mask = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        input = input.to(dtype)
        grad_output = grad_output.to(dtype)
        weight = weight.to(dtype)

        input, _ = prune_column_wise(input, transpose=False)
        grad_weight = torch.matmul(grad_output.t(), input)

        grad_input = torch.matmul(grad_output, weight)
        grad_input = grad_input.reshape(input_shape)
        grad_input[sparsity_mask] = 0.
        return grad_input, grad_weight, None


class DynamicPruneWeightMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight):
        sparse_weight, _ = prune_column_wise(weight)
        ctx.save_for_backward(input, sparse_weight)
        return torch.matmul(input, sparse_weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        grad_weight = torch.matmul(grad_output.t().to(dtype), input.to(dtype))

        weight, _ = prune_row_wise(weight)
        grad_input = torch.matmul(grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        return grad_input, grad_weight


class ReductionDimDynamicPruneWeightMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight):
        sparse_weight, _ = prune_row_wise(weight)
        ctx.save_for_backward(input, sparse_weight)
        return torch.matmul(input, sparse_weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        grad_weight = torch.matmul(grad_output.t().to(dtype), input.to(dtype))

        weight, _ = prune_column_wise(weight)
        grad_input = torch.matmul(grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        return grad_input, grad_weight


class StaticPruneWeightMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight, mask=None):
        if mask is None:
            sparse_weight, mask = prune_column_wise(weight)
        else:
            sparse_weight = weight
            sparse_weight[mask] = 0.
        ctx.save_for_backward(input, weight, mask)
        return torch.matmul(input, weight.t()), mask

    @staticmethod
    def backward(ctx, grad_output, grad_mask):
        input, weight, mask = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        grad_weight = torch.matmul(grad_output.t().to(dtype), input.to(dtype))
        grad_weight[mask] = 0.

        weight, _ = prune_row_wise(weight)
        grad_input = torch.matmul(grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        return grad_input, grad_weight, None


class DynamicPruneOutputGradMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return torch.matmul(input, weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        sparse_grad_output, _ = prune_row_wise(grad_output)
        grad_weight = torch.matmul(sparse_grad_output.t().to(dtype), input.to(dtype))
        grad_input = torch.matmul(sparse_grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        return grad_input, grad_weight


class ReductionDimDynamicPruneOutputGradMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return torch.matmul(input, weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        sparse_grad_output_transpose, _ = prune_column_wise(grad_output, transpose=True)
        grad_weight = torch.matmul(sparse_grad_output_transpose.to(dtype), input.to(dtype))
        sparse_grad_output, _ = prune_row_wise(grad_output)
        grad_input = torch.matmul(sparse_grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        return grad_input, grad_weight


class ReductionDimStaticPruneWeightMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight, mask=None):
        if mask is None:
            sparse_weight, mask = prune_row_wise(weight)
            weight.data = sparse_weight.data
        else:
            sparse_weight = weight
            sparse_weight[mask] = 0.
        ctx.save_for_backward(input, weight, mask)
        return torch.matmul(input, weight.t()), mask

    @staticmethod
    def backward(ctx, grad_output, grad_mask):
        input, weight, mask = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        grad_weight = torch.matmul(grad_output.t().to(dtype), input.to(dtype))
        grad_weight[mask] = 0.

        weight, _ = prune_column_wise(weight)

        grad_input = torch.matmul(grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        return grad_input, grad_weight, None


# def sparsify(mat, m, n):
#     reshaped_mat = mat.clone().reshape(-1, n)
#     mask = torch.zeros_like(reshaped_mat, dtype=torch.bool)
#     if (m, n) == (1, 2):
#         _, indices = torch.topk(torch.abs(reshaped_mat), k=m, dim=1, sorted=False, largest=True)
#         rows = (indices == 1).sum(dim=1)
#         mask[:, 0] = rows
#         mask[:, 1] = torch.logical_not(rows)
#     elif (m, n) == (2, 4):
#         _, indices = torch.topk(torch.abs(reshaped_mat), k=m, dim=1, sorted=False, largest=True)
#         rows = torch.logical_not((indices == 0).sum(dim=1))
#         mask[:, 0] = rows
#         rows = torch.logical_not((indices == 1).sum(dim=1))
#         mask[:, 1] = rows
#         rows = torch.logical_not((indices == 2).sum(dim=1))
#         mask[:, 2] = rows
#         rows = torch.logical_not((indices == 3).sum(dim=1))
#         mask[:, 3] = rows
#     elif m < n / 2:
#         _, indices = torch.topk(torch.abs(reshaped_mat), k=m, dim=1, sorted=False, largest=True)
#         for i in range(n):
#             rows = torch.logical_not((indices == i).sum(dim=1))
#             mask[:, i] = rows
#     else:
#         _, indices = torch.topk(torch.abs(reshaped_mat), k=(n - m), dim=1, sorted=False, largest=False)
#         for i in range(n):
#             rows = (indices == i).sum(dim=1)
#             mask[:, i] = rows
#     reshaped_mat[mask] = 0.
#     return reshaped_mat.reshape(mat.shape), mask.reshape(mat.shape)



if __name__ == '__main__':
    mat = torch.randn(456, 768).to('cuda').half()
    print(torch.sum(mat != 0.) / mat.numel())
    mat, mask = prune_row_wise(mat)
    print(torch.sum(mat != 0.) / mat.numel())
    mat, mask = prune_column_wise(mat)
    print(torch.sum(mat != 0.) / mat.numel())