#!/usr/bin/env python3
"""Step `fp4_perf` of docs/cloud.md: what the NVFP4 lookup fix (PR #104) cost or saved on
the GPU. The fix replaced `__shfl_sync(0xffffffffu, fp4_lane, <nibble>)` by an index into
the `__constant__` 16-entry table `FP4_V[<nibble>]`; docs/device.md 5b left the cost of a
divergent `__constant__` read against the shuffle unmeasured.

    python3 scripts/gcp/fp4_perf.py --cancho-gpu build/cancho-gpu [--json FILE] [--dry-run]

For each case the kernel is emitted by the compiler (the NEW lowering) and the OLD one is
made from it by a text edit (`old_variant`): every `FP4_V[<expr>]` back to
`__shfl_sync(0xffffffffu, fp4_lane, <expr>)` and the `fp4_lane` definition line put back
(the two forms are in `git show 46670b5`). Both are compiled with NVRTC through cupy
(as gemm_bench.py does) and timed with CUDA events over weights rotated through enough
copies to exceed the L2 (an L4 has 48 MB), and the report is achieved GB/s (every buffer
once per launch: the weights dominate) and old/new time. The old variant HANGS when K is
not a multiple of 32, so the cases here have K a multiple of 32; each (case, variant) runs
in its own process with a timeout, and a timeout is reported as TIMEOUT, not waited on.

Two kinds of case, and the report says which is which. At the model's own K (5120, 17408)
the emitted rows kernels read the weights through `fp4_pair` and contain NO table lookup,
so old and new are the same program (ratio ~1, "no lookup": a control). The lookup is in the
generic path, which (rows, res, swiglu, batch families) was taken when K was not a multiple
of 512 (5152, 17440, odd multiples of 32), which is where the ratio meant something. The
compiler since takes `fp4_pair` there too, with a partial last round (docs/device.md 5b), so
on a compiler with that change these cases report `lookups` 0 and old == new (the same program):
the L4 numbers of run 20261105-115058-l4 are of the compiler before it, and the speed of the
new emission is NOT yet measured. To measure the table-index cost again, the case must be one the
compiler still sends to the generic path (find it with the `lookups` column). The `lookups` column is the number of
sites found in the emitted CUDA, so a compiler that changes the paths shows up there.

Needs cupy and numpy (the venv of the gemm_bench step); `--dry-run` needs neither. Exit 1
when a variant timed out, failed, or the two outputs disagree."""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHX = os.path.join(ROOT, "tests", "chx")
# (file, consts, role). n x k at the Qwen3.8-27B-like shapes of tests/cases.txt; the K = 5152 and
# 17440 are those K plus 32, the nearest odd multiples of 32 (before the partial-round step they took the lookup path).
CASES = [
    ("matvec_fp4_rows.chx", "n=17408 k=5120", "control: model K, no lookup"),
    ("matvec_fp4_rows.chx", "n=5120 k=17408", "control: model K, no lookup"),
    ("matvec_fp4_rows.chx", "n=17408 k=5152", "odd x32 K: was lookup path"),
    ("matvec_fp4_rows.chx", "n=5120 k=17440", "odd x32 K: was lookup path"),
    ("matvec_fp4_res.chx", "n=5120 k=17440", "odd x32 K: was lookup path"),
    ("matvec_fp4_swiglu.chx", "n=17408 k=5152", "odd x32 K: was lookup path"),
    ("matvec_fp4_batch.chx", "m=4 n=17408 k=5152", "odd x32 K: was lookup path"),
    ("matvec_fp4_res_batch.chx", "m=4 n=5120 k=17440", "odd x32 K: was lookup path"),
    ("matvec_fp4_rows.chx", "n=17408 k=96", "small K: 16-wide step, idle lanes (absolute GB/s)"),
    ("matvec_fp4_rows.chx", "n=17408 k=160", "small K: 16-wide step, idle lanes (absolute GB/s)"),
    ("matvec_fp4_rows.chx", "n=17408 k=256", "small K: 16-wide step, idle lanes (absolute GB/s)"),
    ("matvec_fp4_rows.chx", "n=17408 k=1056", "small K: 16-wide step, idle lanes (absolute GB/s)"),
]
LANE_DEF = "    const float fp4_lane = FP4_V[tid & 15u];\n"
# lines the emitter writes before the body; the lane definition goes after them
PREAMBLE = re.compile(r"^    (const uint (tid|gid\d?) = .*|__shared__ .*|.*= \(.*\)arena4;)$")


def old_variant(src):
    """The pre-#104 lowering of NEW emitted CUDA `src`. Returns (text, lookups replaced)."""
    out, i, n = [], 0, 0
    while True:
        j = src.find("FP4_V[", i)
        if j < 0:
            out.append(src[i:])
            break
        if src[j + 6:j + 9] == "16]":  # the table's own declaration
            out.append(src[i:j + 9])
            i = j + 9
            continue
        depth, e = 1, j + 6
        while e < len(src) and depth:
            depth += {"[": 1, "]": -1}.get(src[e], 0)
            e += 1
        if depth:
            raise ValueError("unbalanced FP4_V[ in the emitted CUDA")
        out.append(src[i:j] + "__shfl_sync(0xffffffffu, fp4_lane, " + src[j + 6:e - 1] + ")")
        i = e
        n += 1
    text = "".join(out)
    if n == 0:
        return text, 0
    lines = text.splitlines(keepends=True)
    k = next((x for x, l in enumerate(lines) if "__global__" in l), None)
    while k is not None and k < len(lines) and lines[k].strip() != "{":
        k += 1
    if k is None or k >= len(lines):
        raise ValueError("no kernel body found in the emitted CUDA")
    k += 1
    while k < len(lines) and PREAMBLE.match(lines[k].rstrip("\n")):
        k += 1
    lines.insert(k, LANE_DEF)
    return "".join(lines), n


def declarations(chx_path, consts):
    """[(name, dtype, elements, "in"|"out")] of the algo's `in`/`out` lines, in order."""
    env = {k: int(v) for k, v in (c.split("=") for c in consts.split())}
    res = []
    for line in open(chx_path):
        m = re.match(r"\s*(in|out)\s+(\w+):\s*(\w+)\[(.*)\]\s*$", line)
        if m:
            elems = 1
            for dim in m.group(4).split(","):
                elems *= eval(dim.strip().replace("/", "//"), {"__builtins__": {}}, env)  # noqa: S307 (our own .chx)
            res.append((m.group(2), m.group(3), elems, m.group(1)))
    return res


def emit(binary, chx, consts, timeout):
    d = tempfile.mkdtemp()
    p = subprocess.run([binary, "emit", os.path.join(CHX, chx), d] + consts.split(), capture_output=True, text=True,
                       cwd=ROOT, timeout=timeout)
    if p.returncode:
        raise RuntimeError("emit failed: " + p.stderr.strip()[:300])
    cu = [f for f in os.listdir(d) if f.endswith(".cu")]
    if not cu:
        raise RuntimeError("emit wrote no .cu")
    src = open(os.path.join(d, cu[0])).read()
    launch = re.search(r": (\d+) threads, grid (\d+)x(\d+)", p.stdout)
    entry = re.search(r'extern "C" __global__ void (\w+)', src)
    if not launch or not entry:
        raise RuntimeError("cannot read the launch or the entry point")
    return src, entry.group(1), int(launch.group(1)), (int(launch.group(2)), int(launch.group(3)))


def params(src):
    """[(cuda type, name)] of the entry point, in order."""
    sig = src[src.index("__global__"):]
    sig = sig[:sig.index(")")]
    return re.findall(r"(?:const\s+)?(\w+)\*\s*(?:__restrict__\s+)?p\d+_(\w+)", sig)


# ---- the worker: one variant, one process, on the GPU ------------------------------------
def worker(a):
    import cupy as cp
    import numpy as np
    spec = json.load(open(a.worker))
    inc = None
    for d in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda", "/usr/local/cuda-12.9"):
        if d and os.path.exists(os.path.join(d, "include", "crt", "mma.h")):
            inc = os.path.join(d, "include")
            break
    opts = (("-I" + inc,) if inc else ()) + ("--std=c++17",)
    kern = cp.RawModule(code=open(spec["cu"]).read(), options=opts).get_function(spec["entry"])
    rng = np.random.default_rng(1)
    dtypes = {"float": np.float32, "char": np.int8, "__half": np.float16}
    host = []
    for (cty, name), (_, _, elems, kind) in zip(spec["params"], spec["decls"]):
        dt = dtypes[cty]
        if kind == "out":
            host.append(np.zeros(elems, dt))
        elif dt == np.int8:
            # scales: E4M3 bytes near 1.0 (never NaN); the packed codes: anything
            host.append(np.full(elems, 0x38, dt) if name.endswith("s") else rng.integers(-128, 128, elems).astype(dt))
        else:
            host.append((rng.standard_normal(elems) * 0.1).astype(dt))
    nbytes = sum(h.nbytes for h in host)
    sets = max(2, min(8, -(-spec["flush_bytes"] // nbytes)))
    dev = [[cp.asarray(h) for h in host] for _ in range(sets)]
    ys = [i for i, d in enumerate(spec["decls"]) if d[3] == "out"]

    def launch(s):
        kern(tuple(spec["grid"]), (spec["threads"],), tuple(dev[s]))
    launch(0)
    cp.cuda.Device().synchronize()
    np.save(spec["y_out"], cp.asnumpy(dev[0][ys[0]]))
    for i in range(a.warmup):
        launch(i % sets)
    cp.cuda.Device().synchronize()
    ms, t0, i = [], time.time(), 0
    while len(ms) < a.iters or time.time() - t0 < a.min_seconds:
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record()
        launch(i % sets)
        e.record()
        e.synchronize()
        ms.append(cp.cuda.get_elapsed_time(s, e))
        i += 1
    med = statistics.median(ms)
    print(json.dumps({"ms": med, "ms_min": min(ms), "iters": len(ms), "bytes": nbytes, "copies": sets,
                      "gbps": nbytes / (med * 1e-3) / 1e9}))


def run_variant(a, cu, entry, threads, grid, plist, decls, tag):
    d = tempfile.mkdtemp()
    cu_path = os.path.join(d, tag + ".cu")
    open(cu_path, "w").write(cu)
    spec = {"cu": cu_path, "entry": entry, "threads": threads, "grid": list(grid), "params": plist, "decls": decls,
            "y_out": os.path.join(d, tag + ".npy"), "flush_bytes": int(a.flush_mb * 1e6)}
    sp = os.path.join(d, tag + ".json")
    json.dump(spec, open(sp, "w"))
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", sp, "--warmup", str(a.warmup), "--iters", str(a.iters),
           "--min-seconds", str(a.min_seconds)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout, start_new_session=True)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "detail": f"no answer in {a.timeout:.0f} s"}
    if p.returncode:
        return {"status": "ERROR", "detail": (p.stderr.strip().splitlines() or ["exit %d" % p.returncode])[-1][:200]}
    r = json.loads(p.stdout.strip().splitlines()[-1])
    r.update(status="OK", y=spec["y_out"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancho-gpu")
    ap.add_argument("--json")
    ap.add_argument("--dry-run", action="store_true", help="emit and transform only; no GPU, no cupy")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--timeout", type=float, default=120, help="seconds per (case, variant) process")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--min-seconds", type=float, default=2.0)
    ap.add_argument("--flush-mb", type=float, default=256, help="rotate weight copies until this much is touched between reuses")
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    if not a.cancho_gpu:
        ap.error("--cancho-gpu is required")
    np = None
    if not a.dry_run:
        try:
            import numpy as np  # noqa: F811
            import cupy  # noqa: F401
        except ImportError as e:
            print(f"fp4_perf needs numpy and cupy ({e}); use --dry-run without them")
            return 1
    rows, bad = [], 0
    print(f"{'case':54} {'lookups':>7} {'old GB/s':>9} {'new GB/s':>9} {'old/new time':>12}  role")
    for chx, consts, role in CASES:
        name = f"{chx[:-4]} {consts}"
        row = {"case": name, "role": role}
        try:
            new, entry, threads, grid = emit(a.cancho_gpu, chx, consts, 120)
            old, n = old_variant(new)
            decls = declarations(os.path.join(CHX, chx), consts)
            plist = params(new)
            if [nm for _, nm in plist] != [d[0] for d in decls]:
                raise RuntimeError(f"entry parameters {[nm for _, nm in plist]} do not match the declarations {[d[0] for d in decls]}")
            row["lookups"] = n
            if a.dry_run:
                print(f"{name:54} {n:7d} {'-':>9} {'-':>9} {'-':>12}  {role}")
                rows.append(row)
                continue
            rn = run_variant(a, new, entry, threads, grid, plist, decls, "new")
            ro = run_variant(a, old, entry, threads, grid, plist, decls, "old") if n else dict(rn)
            row.update(new=rn, old=ro)
            for r in (rn, ro):
                if r["status"] != "OK":
                    bad += 1
            if rn["status"] == "OK" and ro["status"] == "OK" and n:
                diff = float(abs(np.load(rn["y"]) - np.load(ro["y"])).max())
                row["max_abs_diff"] = diff
                if diff > 0:
                    bad += 1
                    row["detail"] = f"OUTPUTS DIFFER by {diff:g}"
            if rn["status"] == "OK" and ro["status"] == "OK":
                ratio = ro["ms"] / rn["ms"]
                row["old_over_new"] = ratio
                print(f"{name:54} {n:7d} {ro['gbps']:9.1f} {rn['gbps']:9.1f} {ratio:12.3f}  {role}" + ("  (same program)" if not n else "")
                      + (f"  {row['detail']}" if "detail" in row else ""))
            else:
                print(f"{name:54} {n:7d} old {ro.get('status')} {ro.get('detail', '')} | new {rn.get('status')} {rn.get('detail', '')}")
        except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as e:
            bad += 1
            row["error"] = str(e)
            print(f"{name:54} ERROR {e}")
        rows.append(row)
    if a.json:
        for r in rows:
            for v in ("new", "old"):
                if v in r:
                    r[v].pop("y", None)
        json.dump({"rows": rows}, open(a.json, "w"), indent=1)
    print(("\n%d problem(s)" % bad) if bad else "\nno variant hung, failed or disagreed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
