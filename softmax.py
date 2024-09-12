import torch
from utils import *


configs = [
    triton.Config({'num_warps': 1, 'num_ctas': 1}),
    triton.Config({'num_warps': 2, 'num_ctas': 1}),
    triton.Config({'num_warps': 4, 'num_ctas': 1}),
    triton.Config({'num_warps': 8, 'num_ctas': 1}),
]


@triton.autotune(
    configs=configs,
    key=['num_rows', 'num_cols'],
)


@triton.jit
def _softmax_fwd_fused(
    x,  # pointer to the input
    x_alphas,  # pointer to the x_alphas
    x_betas,  # pointer to the x_betas
    q,
    y,  # pointer to the output
    y_alphas,  # pointer to the y_alphas
    y_betas,  # pointer to the y_betas
    stride,  # how much to increase the pointer when moving by 1 row
    num_cols,  # number of columns in X
    num_rows,
    BLOCK_SIZE_COL: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    param_row = tl.program_id(0)
    row = BLOCK_SIZE_ROW * param_row
    params_col_cnt = num_cols // BLOCK_SIZE_COL
    # Y += row * stride
    # X += row * stride
    # Compute mean
    _sum = tl.zeros([BLOCK_SIZE_ROW, BLOCK_SIZE_COL], dtype=tl.float32)
    for param_col in range(0, tl.cdiv(num_cols, BLOCK_SIZE_COL)):
        col = param_col * BLOCK_SIZE_COL
        x_ptrs = tl.make_block_ptr(
            base=x,
            shape=(num_rows, num_cols),
            strides=(stride, 1),
            offsets=(row, col),
            block_shape=(BLOCK_SIZE_ROW, BLOCK_SIZE_COL),
            order=(1, 0),
        )
        data = tl.load(x_ptrs,
                       boundary_check=(0, 1),
                       padding_option="zero",
                       ).to(tl.int8)
        x_alpha = tl.load(x_alphas + param_row * params_col_cnt + param_col)
        if x_betas is not None:
            x_beta = tl.load(x_betas + param_row * params_col_cnt + param_col)
        else:
            x_beta = None
        data = dequantize(data, x_alpha, x_beta, q).to(tl.float32)
        _sum += tl.exp(data)
    sum = tl.sum(_sum, axis=1,  keep_dims=True)
    # Compute variance
    rsum = 1 / sum
    # Normalize and apply linear transformation
    for param_col in range(0, tl.cdiv(num_cols, BLOCK_SIZE_COL)):
        col = param_col * BLOCK_SIZE_COL
        x_ptrs = tl.make_block_ptr(
            base=x,
            shape=(num_rows, num_cols),
            strides=(stride, 1),
            offsets=(row, col),
            block_shape=(BLOCK_SIZE_ROW, BLOCK_SIZE_COL),
            order=(1, 0),
        )
        data = tl.load(x_ptrs,
                       boundary_check=(0, 1),
                       padding_option="zero",
                       ).to(tl.float32)
        x_alpha = tl.load(x_alphas + param_row * params_col_cnt + param_col)
        if x_betas is not None:
            x_beta = tl.load(x_betas + param_row * params_col_cnt + param_col)
        else:
            x_beta = None
        data = dequantize(data, x_alpha, x_beta, q).to(tl.float32)
        x_hat = tl.exp(data) * rsum
        output = (x_hat).to(tl.float16)

        y_alpha = tl.load(y_alphas + param_row * params_col_cnt + param_col)
        y_beta = tl.load(y_betas + param_row * params_col_cnt + param_col)
        output = quantize(output, y_alpha, y_beta, q)
        # Write output
        y_ptrs = tl.make_block_ptr(
            base=y,
            shape=(num_rows, num_cols),
            strides=(stride, 1),
            offsets=(row, col),
            block_shape=(BLOCK_SIZE_ROW, BLOCK_SIZE_COL),
            order=(1, 0),
        )
        tl.store(y_ptrs,
                 output,
                 boundary_check=(0, 1),
                 )


class Softmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, x_alphas, x_betas, y_alphas, y_betas, q):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        # Less than 64KB per feature: enqueue fused kernel
        # MAX_FUSED_SIZE = 65536 // x.element_size()
        # BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        BLOCK_SIZE = x.shape[0] // x_alphas.shape[0]
        BLOCK_SIZE_ROW = x.shape[1] // x_alphas.shape[1]
        # if N > BLOCK_SIZE:
        #     raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        grid_size = (triton.cdiv(M, BLOCK_SIZE_ROW), )
        # enqueue kernel
        _softmax_fwd_fused[grid_size](  #
            x_arg, x_alphas, x_betas, q, y, y_alphas, y_betas,  #
            x_arg.stride(0), N, M,  #
            BLOCK_SIZE_COL=BLOCK_SIZE, BLOCK_SIZE_ROW=BLOCK_SIZE_ROW)
        return y



softmax = Softmax.apply


if __name__ == "__main__":
    M = 4096
    block_size_row, block_size_col = 64, 64
    q = 8

    # for N in [512 * i for i in range(2, 32)]:
    #     device = "cuda"
    #     dtype = torch.float16
    #     # create data
    #     x_shape = (M, N)
    #     x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    #     x_alphas, x_betas = compute_quantization_params(x, block_size_row, block_size_col)
    #     x_quant = quantize_tensor(x, x_alphas, x_betas, q)
    #     x_dequant = dequantize_tensor(x_quant, x_alphas, x_betas, q)
    #     # forward pass
    #     y_ref = torch.nn.functional.softmax(x).to(dtype)
    #     y_alphas, y_betas = compute_quantization_params(y_ref, block_size_row, block_size_col)
    #     y_tri = softmax(x_quant, x_alphas, x_betas, y_alphas, y_betas, q)
    #     y_tri_dequantized = dequantize_tensor(y_tri, y_alphas, y_betas, q)
    #     # compare
    #     error = torch.norm(y_tri_dequantized.float() - y_ref.float()) / torch.norm(y_ref.float())
    #     assert error < 3e-2
    #     # assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)


    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[512 * i for i in range(2, 32)],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=['Triton', 'Torch'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='GB/s',
            plot_name='Softmax',
            args={'M': M, 'dtype': torch.float16},
        ))
    def bench_layer_norm(M, N, dtype, provider, device='cuda'):
        # create data
        x_shape = (M, N)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
        x.requires_grad_(False)
        quantiles = [0.5, 0.2, 0.8]

        if provider == "triton":
            x_alphas, x_betas = compute_quantization_params(x, block_size_row, block_size_col)
            x_quant = quantize_tensor(x, x_alphas, x_betas, q)
            def y_fwd():
                return softmax(x_quant, x_alphas, x_betas, x_alphas, x_betas, 8)
        if provider == "torch":
            def y_fwd():
                return torch.nn.functional.softmax(x).to(dtype)

        gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    bench_layer_norm.run(save_path='profiling_results/softmax', print_data=True)