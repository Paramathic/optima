import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
SparseSemiStructuredTensor._FORCE_CUTLASS = False
import numpy as np


@torch.compile
def to_sparse_semi_structured_compiled(mat):
    return to_sparse_semi_structured(mat)

start = torch.cuda.Event()
end = torch.cuda.Event()

b, d_in, d_out = 4096, 13824, 5120
rank = int(0.1 * min(d_in, d_out))

w = torch.randn(d_out, d_in, dtype=torch.float16, device='cuda')
l = torch.randn(d_in, rank, dtype=torch.float16, device='cuda')
r = torch.randn(rank, d_out, dtype=torch.float16, device='cuda')


steps = 100
start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

for i in range(steps):
    x = torch.randn(b, d_in, dtype=torch.float16, device='cuda')
    start_events[i].record()
    y = torch.matmul(x, w.t())
    end_events[i].record()

torch.cuda.synchronize()
times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
dense_time = np.median(times)

w = to_sparse_semi_structured_compiled(w)

for i in range(steps):
    x = torch.randn(b, d_in, dtype=torch.float16, device='cuda')
    start_events[i].record()
    y = torch.matmul(x, w.t())
    end_events[i].record()

torch.cuda.synchronize()
times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
sparse_time = np.median(times)


for i in range(steps):
    x = torch.randn(b, d_in, dtype=torch.float16, device='cuda')
    start_events[i].record()
    y = torch.matmul(torch.matmul(x, l), r)
    end_events[i].record()

torch.cuda.synchronize()
times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
low_rank_time = np.median(times)

print(f"Dense: {dense_time}ms")
print(f"Sparse: {sparse_time}ms")
print(f"Low rank: {low_rank_time}ms")
print(f"Sparse Speedup: {dense_time / sparse_time}")
print(f"Sparse+Low-Rank Speedup: {dense_time / (low_rank_time + sparse_time)}")