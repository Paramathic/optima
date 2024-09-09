import torch

import triton
import triton.language as tl

@triton.jit
def quantize(x_vals, alpha, beta, q):
    """ Returns (x[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2] - betas[i, j]) / alphas[i, j]
        x_vals: 2D array of FP16
        alpha: 2D array of FP16
        beta: 2D array of FP16
        q: quantization bitwidth
       """
    if q == 4:
        max_val = 7 #TODO: Change to (2 ** (q - 1) - 1)
    else:
        max_val = 127

    if beta is not None:
        x_vals = (x_vals - beta) / alpha * max_val
    else:
        x_vals = x_vals / alpha * max_val
    x_vals = tl.clamp(x_vals, -(max_val+1), max_val)
    x_vals = x_vals.to(tl.int8)

    return x_vals


@triton.jit
def dequantize(x_vals, alpha, beta, q):
    """ Returns (x[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2] - betas[i, j]) / alphas[i, j]
        x_vals: 2D array of INT8
        alpha: 2D array of FP16
        beta: 2D array of FP16
        q: quantization bitwidth
       """
    if q == 4:
        max_val = 7
    else:
        max_val = 127

    x_vals = x_vals.to(tl.float16) / max_val * alpha
    if beta is not None:
        x_vals = x_vals + beta
    return x_vals


@triton.jit
def get_block_ptrs(x, block_size1, block_size2, row_size, i, j):
    """ Returns x[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2]
        x: 2D array of FP16
        block_size1: int
        block_size2: int
        row_size: int
        i: int
        j: int
       """
    offset1 = i * block_size1 + tl.arange(0, block_size1)
    offset2 = j * block_size2 + tl.arange(0, block_size2)
    x_ptrs = x + offset1[:, None] * row_size + offset2[None, :]
    return x_ptrs






if __name__ == "__main__":
    @triton.jit
    def quantize_e2e(x,
                     y,
                     alphas,
                     betas,
                     block_size1: tl.constexpr,
                     block_size2: tl.constexpr,
                     row_size: tl.constexpr,
                     col_size: tl.constexpr,
                     q):
        """ Returns (x[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2] - betas[i, j]) / alphas[i, j]
            x: Input 2D array of FP16
            y: Output 2D array of int8
            alphas: 2D array of FP16
            betas: 2D array of FP16
            block_size1: int
            block_size2: int
            q: quantization bitwidth
           """

        # Compute the start and end indices of the block
        i, j = tl.program_id(0), tl.program_id(1)
        #x_ptrs = get_block_ptrs(x, block_size1, block_size2, row_size, i, j)
        x_ptrs = tl.make_block_ptr(
            base=x,
            shape=(col_size, row_size),
            strides=(row_size, 1),
            offsets=(i*block_size1, j*block_size2),
            block_shape=(block_size1, block_size2),
            order=(1,0)
        )
        alphas_row_size = row_size // block_size2
        alpha = tl.load(alphas + i * alphas_row_size + j).to(tl.float16)
        if betas is not None:
            beta = tl.load(betas + i * alphas_row_size + j)
        else:
            beta = None

        # Load the input values
        x_vals = tl.load(x_ptrs)

        # Quantize the input values
        x_vals = quantize(x_vals, alpha, beta, q)

        # Store the quantized values
        #y_ptrs = get_block_ptrs(y, block_size1, block_size2, row_size, i, j)
        y_ptrs = tl.make_block_ptr(
            base=y,
            shape=(col_size, row_size),
            strides=(row_size, 1),
            offsets=(i*block_size1, j*block_size2),
            block_shape=(block_size1, block_size2),
            order=(1,0)
        )
        tl.store(y_ptrs, x_vals)
    # Allocate memory
    x = torch.randn(32, 32).cuda().half()
    y_triton = torch.empty_like(x, dtype=torch.int8).cuda()
    alphas = torch.ones(1, 1).cuda().half()
    betas = torch.ones(1, 1).cuda().half()
    q = 8

    # Launch the kernel
    block_size1 = x.shape[0] // alphas.shape[0]
    block_size2 = x.shape[1] // alphas.shape[1]
    y = torch.empty_like(x)
    for i in range(alphas.shape[0]):
        for j in range(alphas.shape[1]):
            block = x[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2]
            block_min = block.min()
            block_max = block.max()
            alphas[i, j] = (block_max - block_min)
            betas[i, j] = (block_max + block_min) / 2
            y[i*block_size1:(i+1)*block_size1, j*block_size2:(j+1)*block_size2] = ((block - betas[i, j]) / alphas[i, j] * 127).to(torch.int8)

    print(y)
    grid = lambda meta: (triton.cdiv(x.shape[0], block_size1), triton.cdiv(x.shape[1], block_size2))
    quantize_e2e[grid](x, y_triton, alphas, betas, block_size1, block_size2, x.shape[1], x.shape[0], q)

    print(y_triton)
    print("Relative Error: ", ((y - y_triton).float().norm() / y.float().norm()).item())