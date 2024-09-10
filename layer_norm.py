import torch
from utils import *


@triton.jit
def _layer_norm_fwd_fused(
    x,  # pointer to the input
    y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    stride,  # how much to increase the pointer when moving by 1 row
    num_cols,  # number of columns in X
    num_rows,
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE_COL: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = BLOCK_SIZE_ROW * tl.program_id(0)
    # Y += row * stride
    # X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE_ROW, BLOCK_SIZE_COL], dtype=tl.float32)
    row_offset = tl.arange(0, BLOCK_SIZE_ROW)
    for col in range(0, num_cols, BLOCK_SIZE_COL):
        x_ptrs = tl.make_block_ptr(
            base=x,
            shape=(BLOCK_SIZE_ROW, BLOCK_SIZE_COL),
            strides=(stride, 1),
            offsets=(row, col),
            block_shape=(BLOCK_SIZE_ROW, BLOCK_SIZE_COL),
            order=(1, 0)
        )
        data = tl.load(x_ptrs).to(tl.float32)
        _mean += data
    mean = tl.sum(_mean, axis=1,  keep_dims=True) / num_cols
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE_ROW, BLOCK_SIZE_COL], dtype=tl.float32)
    for col in range(0, num_cols, BLOCK_SIZE_COL):
        x_ptrs = tl.make_block_ptr(
            base=x,
            shape=(BLOCK_SIZE_ROW, BLOCK_SIZE_COL),
            strides=(stride, 1),
            offsets=(row, col),
            block_shape=(BLOCK_SIZE_ROW, BLOCK_SIZE_COL),
            order=(1, 0)
        )
        data = tl.load(x_ptrs).to(tl.float32)
        data = data - mean
        _var += data * data
    var = tl.sum(_var, axis=1, keep_dims=True) / num_cols
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for col in range(0, num_cols, BLOCK_SIZE_COL):
        cols = col + tl.arange(0, BLOCK_SIZE_COL)
        mask = cols < num_cols
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x_ptrs = tl.make_block_ptr(
            base=x,
            shape=(BLOCK_SIZE_ROW, BLOCK_SIZE_COL),
            strides=(stride, 1),
            offsets=(row, col),
            block_shape=(BLOCK_SIZE_ROW, BLOCK_SIZE_COL),
            order=(1, 0)
        )
        data = tl.load(x_ptrs).to(tl.float32)
        x_hat = (data - mean) * rstd
        output = (x_hat * w + b).to(tl.float16)
        # Write output
        y_ptrs = tl.make_block_ptr(
            base=y,
            shape=(BLOCK_SIZE_ROW, BLOCK_SIZE_COL),
            strides=(stride, 1),
            offsets=(row, col),
            block_shape=(BLOCK_SIZE_ROW, BLOCK_SIZE_COL),
            order=(1, 0)
        )
        tl.store(y_ptrs, output)


class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        # Less than 64KB per feature: enqueue fused kernel
        # MAX_FUSED_SIZE = 65536 // x.element_size()
        # BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        BLOCK_SIZE = 512
        BLOCK_SIZE_ROW=1
        # if N > BLOCK_SIZE:
        #     raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        grid_size = (triton.cdiv(M, BLOCK_SIZE_ROW), )
        # enqueue kernel
        _layer_norm_fwd_fused[grid_size](  #
            x_arg, y, weight, bias,   #
            x_arg.stride(0), N, M, eps,  #
            BLOCK_SIZE_COL=BLOCK_SIZE, BLOCK_SIZE_ROW=BLOCK_SIZE_ROW, num_warps=num_warps, num_ctas=1)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y



layer_norm = LayerNorm.apply


if __name__ == "__main__":
    M = 4096
    for N in [512 * i for i in range(2, 32)]:
        eps = 1e-5
        device = "cuda"
        dtype = torch.float16
        # create data
        x_shape = (M, N)
        w_shape = (x_shape[-1], )
        weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
        dy = .1 * torch.randn_like(x)
        x.requires_grad_(True)
        # forward pass
        y_tri = layer_norm(x, w_shape, weight, bias, eps)
        y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
        # compare
        assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)


    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[512 * i for i in range(2, 32)],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=['Triton', 'Torch'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='GB/s',
            plot_name='Layer-Norm',
            args={'M': 4096, 'dtype': torch.float16},
        ))
    def bench_layer_norm(M, N, dtype, provider, eps=1e-5, device='cuda'):
        # create data
        x_shape = (M, N)
        w_shape = (x_shape[-1],)
        weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=False)
        bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=False)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
        x.requires_grad_(False)
        quantiles = [0.5, 0.2, 0.8]

        def y_fwd():
            if provider == "triton":
                return layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

            if provider == "torch":
                return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    bench_layer_norm.run(save_path='profiling_results/layer_norm', print_data=True)