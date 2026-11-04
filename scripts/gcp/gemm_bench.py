#!/usr/bin/env python3
"""Step `gemm_bench` of docs/cloud.md: a matmul microbenchmark on the real GPU, in TFLOPS,
with enough recorded to explain the numbers.

    python3 scripts/gcp/gemm_bench.py --cancho-gpu build/cancho-gpu \
        [--sizes 1024,2048,3072,4096,6144,8192] [--shapes 8192:8192:4096,...] [--json FILE]

Implementations of y = a w^T, a f16 [m, k], w f16 [n, k]:

  ours      the kernel `cancho-gpu emit tests/chx/ported/gemm_mma.chx` writes for nvidia-ada
            (wmma 16x16x16, f32 accumulate, f32 output), compiled by NVRTC through cupy and
            launched on the grid and threads `emit` reports
  cublas    cupy.matmul on f16 (cuBLAS; f16 output)
  torch     torch.matmul on f16 (cuBLAS through torch's dispatcher; f16 output)
  ours+swz  an EXPERIMENT, not the emitter: the same kernel with its block indices remapped
            (grouped ordering, 8 row-blocks per group) by a text edit of one line made in
            this script. It tests one hypothesis about the fall-off at large sizes (below)
            and changes nothing in the compiler.

For each shape and implementation the timed loop runs at least --min-seconds (3) and
nvidia-smi samples SM clock, power, temperature and the throttle-reason bits every 100 ms
meanwhile, so each row says what clock and power it was timed at. Per shape it also records
the kernel's grid, block, static shared memory, registers and local (spill) bytes as cupy
reports them and, with ptxas, ptxas -v's; the occupancy the CUDA occupancy API reports
(blocks per SM, through cupy's driver binding when it has one) and the number of waves. ncu is
not installed and not needed.

Hypothesis for the fall-off (ours/cuBLAS 0.97 at 2048^3, 0.89 at 4096^3, 0.62 at 8192^3 in the
third run, cuBLAS steady at 58.9 TFLOPS): the emitted grid is `grid i over m/bm, j over n/bn`
with i = blockIdx.x, the fastest-varying. Consecutive blocks take DIFFERENT 128-row tiles of `a`
against the same tile of `w`. A wave of ~120 blocks touches ~120 distinct `a` tiles of 128 x k f16
(2 MB each at k = 8192: 240 MB), far over the L4's 48 MB L2, so `a` comes from DRAM for every
column of blocks where cuBLAS rasterises for L2 reuse; and the single-buffered load-sync-compute
loop has nothing to hide that latency with. ours+swz is the test. Our output is checked against
torch's before its time is believed. Needs cupy and torch."""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

import cupy as cp
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
GID_LINE = "const uint gid = blockIdx.x, gid2 = blockIdx.y;"
SWIZZLE = ("const uint lin = blockIdx.x + gridDim.x * blockIdx.y; const uint gsz = 8u; const uint per = gsz * gridDim.y; "
           "const uint first = (lin / per) * gsz; const uint gs = min(gridDim.x - first, gsz); "
           "const uint gid = first + (lin % per) % gs, gid2 = (lin % per) / gs;")


def cuda_include():
    """The CUDA toolkit's include dir (it has crt/mma.h). NOT the pip wheels': the
    nvidia/cuda_runtime/include in the venv has mma.h without crt/, and cupy would
    otherwise find that one ("cannot open source file crt/mma.h")."""
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


class Sampler:
    """nvidia-smi at 100 ms for the duration of a with-block."""
    Q = "clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active"

    def __enter__(self):
        self.p = None
        self.f = tempfile.TemporaryFile("w+")
        try:
            self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=" + self.Q, "--format=csv,noheader,nounits", "-lms", "100"],
                                      stdout=self.f, stderr=subprocess.DEVNULL)
        except OSError:
            self.p = None
        return self

    def __exit__(self, *a):
        if self.p:
            self.p.terminate()
            self.p.wait()
        self.f.seek(0)
        rows = []
        for line in self.f.read().splitlines():
            c = [x.strip() for x in line.split(",")]
            try:
                rows.append((float(c[0]), float(c[1]), float(c[2]), float(c[3]), int(c[4], 16)))
            except (ValueError, IndexError):
                pass
        self.rows = rows

    def summary(self):
        if not self.rows:
            return {"samples": 0}
        sm, mem, pw, tmp, thr = zip(*self.rows)
        bits = 0
        for t in thr:
            bits |= t
        return {"samples": len(self.rows), "sm_mhz_median": statistics.median(sm), "sm_mhz_min": min(sm),
                "mem_mhz": statistics.median(mem), "power_w_median": statistics.median(pw), "power_w_max": max(pw),
                "temp_c_max": max(tmp), "throttle_reasons_or": hex(bits)}


def timed(fn, warmup, iters, min_seconds):
    for _ in range(warmup):
        fn()
    cp.cuda.Device().synchronize()
    ms, t0 = [], time.time()
    with Sampler() as smp:
        while len(ms) < iters or time.time() - t0 < min_seconds:
            s, e = cp.cuda.Event(), cp.cuda.Event()
            s.record()
            fn()
            e.record()
            e.synchronize()
            ms.append(cp.cuda.get_elapsed_time(s, e))
    return statistics.median(ms), smp.summary()


def kernel_info(kern, src, threads, grid, sms):
    info = {"grid": list(grid), "block": threads, "blocks_total": grid[0] * grid[1],
            "static_smem_bytes": kern.shared_size_bytes, "regs_cupy": kern.num_regs, "local_bytes_cupy": kern.local_size_bytes}
    try:
        from nvrtc_check import compile_one, find_nvrtc, include_dirs, ptxas_report
        lib, _ = find_nvrtc()
        ptx, _log = compile_one(lib, src.encode(), "compute_89", include_dirs())
        if ptx is not None:
            info["ptxas"] = ptxas_report(ptx)
    except Exception as e:  # noqa: BLE001 - diagnostic only
        info["ptxas_error"] = str(e)[:100]
    try:
        per_sm = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(kern.ptr, threads, 0)
        info["blocks_per_sm"] = per_sm
        info["resident_blocks"] = per_sm * sms
        info["waves"] = round(info["blocks_total"] / max(1, per_sm * sms), 2)
        info["occupancy_warps_pct"] = round(100.0 * per_sm * threads / 32 / 48, 1)  # Ada: 48 warps per SM
    except Exception as e:  # noqa: BLE001
        info["occupancy_error"] = str(e)[:100]
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancho-gpu", required=True)
    ap.add_argument("--sizes", default="1024,2048,3072,4096,6144,8192", help="square sizes m=n=k")
    ap.add_argument("--shapes", default="8192:8192:4096,8192:8192:2048,4096:4096:8192", help="m:n:k, non-square")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--min-seconds", type=float, default=3.0)
    ap.add_argument("--no-swizzle", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    props = cp.cuda.runtime.getDeviceProperties(0)
    gpu = props["name"].decode()
    sms = props["multiProcessorCount"]
    cc = f"sm_{props['major']}{props['minor']}"
    inc = cuda_include()
    print(f"toolkit include for NVRTC: {inc}")
    print(f"{gpu} ({cc}, {sms} SMs, L2 {props.get('l2CacheSize', 0) >> 20} MB), torch {torch.__version__}, cupy {cp.__version__}")
    if cc != "sm_89":
        print(f"NOTE: {cc} is not Ada; the emitted kernel is Ada's lowering (docs/cloud.md)")
    shapes = [(int(s),) * 3 for s in a.sizes.split(",") if s] + [tuple(int(x) for x in s.split(":")) for s in a.shapes.split(",") if s]
    opts = (("-I" + inc,) if inc else ()) + ("--std=c++17",)
    rows = []
    for m, n, k in shapes:
        flops = 2.0 * m * n * k
        src, entry, threads, grid = emit(a.cancho_gpu, m, n, k)
        variants = [("ours", src)]
        if not a.no_swizzle and GID_LINE in src:
            variants.append(("ours+swz", src.replace(GID_LINE, SWIZZLE)))
        ta = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
        tw = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.1
        ca, cw = cp.asarray(ta), cp.asarray(tw)
        want = torch.matmul(ta, tw.t()).float()
        row = {"m": m, "n": n, "k": k}
        for name, code in variants:
            kern = cp.RawModule(code=code, options=opts).get_function(entry)
            y = cp.zeros((m, n), dtype=cp.float32)
            launch = lambda: kern(grid, (threads,), (ca, cw, y))  # noqa: E731
            launch()
            cp.cuda.Device().synchronize()
            err = ((torch.as_tensor(y, device="cuda") - want).abs().max() / want.abs().max()).item()
            ms, smp = timed(launch, a.warmup, a.iters, a.min_seconds)
            row[name + "_ms"] = ms
            row[name + "_tflops"] = flops / (ms * 1e-3) / 1e12
            row[name + "_err"] = err
            row[name + "_clocks"] = smp
            if name == "ours":
                row["kernel"] = kernel_info(kern, code, threads, grid, sms)
            del y
        for name, fn in (("cublas", lambda: cp.matmul(ca, cw.T)), ("torch", lambda: torch.matmul(ta, tw.t()))):
            ms, smp = timed(fn, a.warmup, a.iters, a.min_seconds)
            row[name + "_ms"] = ms
            row[name + "_tflops"] = flops / (ms * 1e-3) / 1e12
            row[name + "_clocks"] = smp
        rows.append(row)
        c, o, ki = row["cublas_clocks"], row["ours_clocks"], row["kernel"]
        swz = "ours+swz_tflops" in row
        print(f"{m}x{n}x{k}: ours {row['ours_tflops']:.1f}  " + (f"ours+swz {row['ours+swz_tflops']:.1f}  " if swz else "")
              + f"cublas {row['cublas_tflops']:.1f}  torch {row['torch_tflops']:.1f} TFLOPS   ours/cublas {row['ours_tflops'] / row['cublas_tflops']:.2f}"
              + (f"  swz/cublas {row['ours+swz_tflops'] / row['cublas_tflops']:.2f}" if swz else "")
              + f"   err {row['ours_err']:.1e}" + ("  WRONG OUTPUT, ignore the speed" if row["ours_err"] > 2e-2 else ""))
        print(f"    clocks while timed: ours SM {o.get('sm_mhz_median')}/{o.get('sm_mhz_min')} MHz (median/min), {o.get('power_w_median')} W (max {o.get('power_w_max')}), "
              f"throttle {o.get('throttle_reasons_or')}, {o.get('samples')} samples | cublas SM {c.get('sm_mhz_median')} MHz, {c.get('power_w_median')} W, throttle {c.get('throttle_reasons_or')}")
        px = ki.get("ptxas", {})
        print(f"    kernel: grid {ki['grid']} = {ki['blocks_total']} blocks x {ki['block']} threads, static smem {ki['static_smem_bytes']} B, "
              f"regs {ki['regs_cupy']} (ptxas {px.get('registers')}, spill {px.get('spill_bytes')} B), blocks/SM {ki.get('blocks_per_sm')}, "
              f"warp occupancy {ki.get('occupancy_warps_pct')}%, waves {ki.get('waves')}")
        del ca, cw, ta, tw
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
    peak = max(max(r["cublas_tflops"], r["torch_tflops"]) for r in rows)
    print(f"\nmeasured peak on this card (best cuBLAS above): {peak:.1f} TFLOPS")
    for r in rows:
        print(f"  {r['m']}x{r['n']}x{r['k']}: ours is {100 * r['ours_tflops'] / peak:.0f}% of that peak"
              + (f", with swizzled blocks {100 * r['ours+swz_tflops'] / peak:.0f}%" if "ours+swz_tflops" in r else ""))
    if a.json:
        with open(a.json, "w") as h:
            json.dump({"gpu": gpu, "arch": cc, "sms": sms, "torch": torch.__version__, "cupy": cp.__version__,
                       "measured_peak_tflops": peak, "rows": rows}, h, indent=1)
    return 0 if all(r["ours_err"] <= 2e-2 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
