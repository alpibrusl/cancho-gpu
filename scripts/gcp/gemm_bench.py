#!/usr/bin/env python3
"""Step 5 of docs/cloud.md: a matmul microbenchmark on the real GPU, in TFLOPS.

    python3 scripts/gcp/gemm_bench.py --cancho-gpu build/cancho-gpu [--sizes 1024,2048,4096,8192] [--json FILE]

Three implementations of y = a w^T, a f16 [m, k], w f16 [n, k], square sizes:

  ours      the kernel `cancho-gpu emit tests/chx/ported/gemm_mma.chx` writes for
            nvidia-ada (wmma 16x16x16, f32 accumulate, f32 output), compiled by
            NVRTC (through cupy) and launched on the grid and threads `emit` reports
  cublas    cupy's matmul on f16 (cuBLAS GemmEx; f16 output)
  torch     torch.matmul on f16 (cuBLAS again, through torch's dispatcher; f16 output)

`cublas` and `torch` both reach cuBLAS: the second row is the check that the
first is not a wrapper artifact. Their output is f16 and ours f32 (the .chx says
so), a difference of 2 bytes per output element, small against 2*k flops per
element at these sizes. The "peak" is the BEST cuBLAS number measured on this
card in this run, so a fraction quoted from here is a fraction of what this
card was measured to do, not of the datasheet figure (which assumes clocks and
a power budget an L4 does not hold under load; docs/cloud.md). Our output is
checked against torch's before its time is believed. Median of --iters runs after
--warmup, CUDA events. Needs cupy and torch (remote_cuda.sh makes the venv)."""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile

import cupy as cp
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def cuda_include():
    """The CUDA toolkit's include dir (it has crt/mma.h). NOT the pip wheels':
    nvidia/cuda_runtime/include in the venv has mma.h without crt/, and cupy
    would otherwise find that one ("cannot open source file crt/mma.h")."""
    for d in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda", "/usr/local/cuda-12.9"):
        if d and os.path.exists(os.path.join(d, "include", "crt", "mma.h")):
            return os.path.join(d, "include")
    return None


def emit(binary, m, n, k):
    d = tempfile.mkdtemp()
    p = subprocess.run([binary, "emit", os.path.join(ROOT, "tests/chx/ported/gemm_mma.chx"), d,
                        f"m={m}", f"n={n}", f"k={k}"], capture_output=True, text=True, cwd=ROOT)
    if p.returncode:
        raise SystemExit(f"emit failed: {p.stderr.strip()}")
    cu = [f for f in os.listdir(d) if f.endswith(".cu")][0]
    src = open(os.path.join(d, cu)).read()
    launch = re.search(r": (\d+) threads, grid (\d+)x(\d+)", p.stdout)
    entry = re.search(r'extern "C" __global__ void (\w+)', src).group(1)
    return src, entry, int(launch.group(1)), (int(launch.group(2)), int(launch.group(3)))


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    cp.cuda.Device().synchronize()
    ms = []
    for _ in range(iters):
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record()
        fn()
        e.record()
        e.synchronize()
        ms.append(cp.cuda.get_elapsed_time(s, e))
    return statistics.median(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancho-gpu", required=True)
    ap.add_argument("--sizes", default="1024,2048,4096,8192")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--json")
    a = ap.parse_args()
    props = cp.cuda.runtime.getDeviceProperties(0)
    gpu = props["name"].decode()
    cc = f"sm_{props['major']}{props['minor']}"
    print(f"toolkit include for NVRTC: {cuda_include()}")
    print(f"{gpu} ({cc}), torch {torch.__version__}, cupy {cp.__version__}")
    if cc != "sm_89":
        print(f"NOTE: {cc} is not Ada; the emitted kernel is Ada's lowering (docs/cloud.md)")
    rows = []
    for m in [int(s) for s in a.sizes.split(",")]:
        n = k = m
        flops = 2.0 * m * n * k
        src, entry, threads, grid = emit(a.cancho_gpu, m, n, k)
        inc = cuda_include()
        opts = (("-I" + inc,) if inc else ()) + ("--std=c++17",)
        mod = cp.RawModule(code=src, options=opts)
        kern = mod.get_function(entry)
        ta = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
        tw = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.1
        ca, cw = cp.asarray(ta), cp.asarray(tw)
        y = cp.zeros((m, n), dtype=cp.float32)
        launch = lambda: kern(grid, (threads,), (ca, cw, y))
        launch()
        cp.cuda.Device().synchronize()
        want = torch.matmul(ta, tw.t()).float()
        got = torch.as_tensor(y, device="cuda")
        err = ((got - want).abs().max() / want.abs().max()).item()
        t_ours = timed(launch, a.warmup, a.iters)
        t_cublas = timed(lambda: cp.matmul(ca, cw.T), a.warmup, a.iters)
        t_torch = timed(lambda: torch.matmul(ta, tw.t()), a.warmup, a.iters)
        row = {"m": m, "n": n, "k": k, "max_rel_err_vs_torch": err}
        for name, ms in (("ours", t_ours), ("cublas", t_cublas), ("torch", t_torch)):
            row[name + "_ms"] = ms
            row[name + "_tflops"] = flops / (ms * 1e-3) / 1e12
        rows.append(row)
        print(f"{m:5d}^3  ours {row['ours_tflops']:6.1f}  cublas {row['cublas_tflops']:6.1f}  torch {row['torch_tflops']:6.1f} TFLOPS"
              f"   ours/cublas {row['ours_tflops'] / row['cublas_tflops']:.2f}"
              f"   max error vs torch {err:.1e} of peak" + ("   WRONG OUTPUT, ignore the speed" if err > 2e-2 else ""))
        del ca, cw, y, ta, tw
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
    peak = max(max(r["cublas_tflops"], r["torch_tflops"]) for r in rows)
    print(f"\nmeasured peak on this card (best cuBLAS above): {peak:.1f} TFLOPS")
    for r in rows:
        print(f"  {r['m']}^3: ours is {100 * r['ours_tflops'] / peak:.0f}% of that peak")
    if a.json:
        with open(a.json, "w") as h:
            json.dump({"gpu": gpu, "arch": cc, "torch": torch.__version__, "cupy": cp.__version__,
                       "measured_peak_tflops": peak, "rows": rows}, h, indent=1)
    return 0 if all(r["max_rel_err_vs_torch"] <= 2e-2 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
