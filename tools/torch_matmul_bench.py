#!/usr/bin/env python3
"""D11 of docs/diffusion.md: torch on MPS at the shapes of one transformer block of the edit step, to hold cancho-gpu's `gemm_mma` (13 TFLOPS) and its attention kernels to what the same machine does through torch.

    torch_matmul_bench.py [--dtype bf16|f16]

Prints, in the dtype given (default both): the three products of a block ([4160, 4096] x [4096, 4096], [4160, 4096] x [4096, 12288], [4160, 12288] x [12288, 4096], the weights as torch's `F.linear` takes them: [n, k]) and
`scaled_dot_product_attention` for 32 heads of 128 with 4,160 queries and 8,320 keys. Median of 20 timed runs after 5 warm-up runs, each synchronised.
"""
import sys
import time

import torch
import torch.nn.functional as F

dts = {"bf16": torch.bfloat16, "f16": torch.float16}
want = [sys.argv[sys.argv.index("--dtype") + 1]] if "--dtype" in sys.argv else ["bf16", "f16"]


def bench(f, flops, label):
    for _ in range(5):
        f()
    torch.mps.synchronize()
    ts = []
    for _ in range(20):
        t = time.perf_counter()
        f()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t)
    ts.sort()
    print("   %-52s %8.2f ms  %6.2f TFLOPS" % (label, ts[10] * 1e3, flops / ts[10] / 1e12))


for name in want:
    dt = dts[name]
    print(name)
    for m, n, k in ((4160, 4096, 4096), (4160, 12288, 4096), (4160, 4096, 12288)):
        a = torch.randn(m, k, device="mps", dtype=dt)
        w = torch.randn(n, k, device="mps", dtype=dt)
        bench(lambda: F.linear(a, w), 2.0 * m * n * k, "linear [%d, %d] x [%d, %d]^T" % (m, k, n, k))
    q = torch.randn(1, 32, 4160, 128, device="mps", dtype=dt)
    kk = torch.randn(1, 32, 8320, 128, device="mps", dtype=dt)
    v = torch.randn(1, 32, 8320, 128, device="mps", dtype=dt)
    mask = torch.zeros(4160, 8320, device="mps", dtype=dt)
    bench(lambda: F.scaled_dot_product_attention(q, kk, v), 4.0 * 32 * 4160 * 8320 * 128, "sdpa 32 heads, 4160 q, 8320 keys, no mask")
    bench(lambda: F.scaled_dot_product_attention(q, kk, v, attn_mask=mask), 4.0 * 32 * 4160 * 8320 * 128, "sdpa with an additive mask")
