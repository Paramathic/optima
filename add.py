import torch, triton
import triton.language as tl

from utils import quantize, dequantize

@triton.jit
def add_kernel (
    A, A_alpha, A_beta, # First operand
    B, B_alpha, B_beta, # Second operand
    C, C_alpha, C_beta,  # Output
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_NUM_ROWS: tl.constexpr,
    BLOCK_NUM_COLS: tl.constexpr
):
    # Which block am I working on?
    i, j = tl.program_id(0), tl.program_id(1)

    # Calculate the pointers for our block
    A_ptrs = tl.make_block_ptr(
        base=A,
        shape=(M,N),
        strides=(N,1),
        offsets=(i*BLOCK_NUM_ROWS, j*BLOCK_NUM_COLS),
        block_shape=(BLOCK_NUM_ROWS, BLOCK_NUM_COLS),
        order=(1,0)
    )
    A_alpha_ptr = A_alpha + (i*BLOCK_NUM_COLS) + j
    A_beta_ptr = A_beta + (i*BLOCK_NUM_COLS) + j

    B_ptrs = tl.make_block_ptr(
        base=B,
        shape=(M,N),
        strides=(N,1),
        offsets=(i*BLOCK_NUM_ROWS, j*BLOCK_NUM_COLS),
        block_shape=(BLOCK_NUM_ROWS, BLOCK_NUM_COLS),
        order=(1,0)
    )
    B_alpha_ptr = B_alpha + (i*BLOCK_NUM_COLS) + j
    B_beta_ptr = B_beta + (i*BLOCK_NUM_COLS) + j

    C_ptrs = tl.make_block_ptr(
        base=C,
        shape=(M,N),
        strides=(N,1),
        offsets=(i*BLOCK_NUM_ROWS, j*BLOCK_NUM_COLS),
        block_shape=(BLOCK_NUM_ROWS, BLOCK_NUM_COLS),
        order=(1,0)
    )
    C_alpha_ptr = C_alpha + (i*BLOCK_NUM_COLS) + j
    C_beta_ptr = C_beta + (i*BLOCK_NUM_COLS) + j

    # Load the data
    _A = tl.load(A_ptrs)
    _A_alpha = tl.load(A_alpha_ptr)
    _A_beta = tl.load(A_beta_ptr)

    _B = tl.load(B_ptrs)
    _B_alpha = tl.load(B_alpha_ptr)
    _B_beta = tl.load(B_beta_ptr)

    # Dequantize
    _A = dequantize(_A, _A_alpha, _A_beta, 8)
    _B = dequantize(_B, _B_alpha, _B_beta, 8)

    _C = _A + _B

    # Calculate Alpha and Beta for C
    block_min = tl.min(_C)
    block_max = tl.max(_C)
    _C_alpha = block_max - block_min
    _C_beta = (block_max + block_min)/2

    # Make sure alpha and beta are going in as halves
    _C_alpha = _C_alpha.to(tl.float16)
    _C_beta = _C_beta.to(tl.float16)
    
    # Quantize C
    _q_C = quantize(_C, _C_alpha, _C_beta, 8)

    # Store C, C_alpha, C_beta into their respective locations
    tl.store(C_ptrs, _q_C)
    tl.store(C_alpha_ptr, _C_alpha)
    tl.store(C_beta_ptr, _C_beta)



def add (A, A_alpha, A_beta,
         B, B_alpha, B_beta,
         block_shape):
    
    # Setup all the C variables for the output of the kernel
    C = torch.zeros_like(A)
    C_alpha = torch.zeros_like(A_alpha)
    C_beta = torch.zeros_like(B_beta)

    grid = (A.shape[0]//block_shape[0], A.shape[1]//block_shape[1])

    add_kernel[grid](A, A_alpha, A_beta,
                     B, B_alpha, B_beta,
                     C, C_alpha, C_beta,
                     A.shape[0], A.shape[1],
                     block_shape[0], block_shape[1])
    
    return C, C_alpha, C_beta


if __name__ == '__main__':
    device = torch.device('cuda')

    M, N = 1024, 1024
    block_shape = (32, 32)
    ab_shape = (M//block_shape[0])* (N//block_shape[1])

    x = torch.randn((M, N), device=device).to(torch.int8)
    x_alpha = torch.randn(ab_shape, device=device, dtype=torch.half)
    x_beta = torch.randn(ab_shape, device=device, dtype=torch.half)

    y = torch.randn((M, N), device=device).to(torch.int8)
    y_alpha = torch.randn(ab_shape, device=device, dtype=torch.half)
    y_beta = torch.randn(ab_shape, device=device, dtype=torch.half)

    z, z_alpha, z_beta = add(x, x_alpha, x_beta,
                             y, y_alpha, y_beta,
                             block_shape)
    
    print(f'z: {z}, z_alpha: {z_alpha}, z_beta:{z_beta}')
