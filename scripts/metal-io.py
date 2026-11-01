#!/usr/bin/env python3
"""Gate 3 of docs/run.md section 7: the Metal device program on real hardware.
RUN ON A MAC (Apple Silicon). Not in scripts/gate.sh, and not run in the Linux
container this was written in (no Metal there); the numbers it prints are the
ones to paste into docs/run.md section 7.5.

    scripts/build-metal-device.sh
    python3 scripts/metal-io.py build/cancho-gpu-metal-device

1. `bwd_dx r=2 n=2 k=3` with dy = 1 2 / 3 4, w = 1 0 2 / 0 1 3, `--out dx`:
   within 2e-2 of its peak of the exact CPU answer 1.0 2.0 8.0 3.0 4.0 18.0.
2. The four backward kernels (bwd_dx, bwd_dw, bwd_db, bwd_sigmoid) over the
   interpreter's LCG inputs, against tests/kernels.py's Python references
   (taken from that file, not copied), EVERY element, to 2e-2 of the peak (the GPU
   may sum in another order, so equality is not the claim) -- and the largest
   difference is printed, in units of the peak, so it can be read.
"""
import ast
import math
import os
import subprocess
import sys
import tempfile

BIN = os.path.abspath(sys.argv[1])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 2e-2
failed = 0

# The references are tests/kernels.py's own functions (that file runs its checks
# on import, so its definitions are taken out of the syntax tree).
tree = ast.parse(open(os.path.join(ROOT, "tests", "kernels.py")).read())
ns = {"math": math}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("lcg", "sigmoid_", "bwd_dx", "bwd_dw", "bwd_db", "bwd_sigmoid"):
        exec(compile(ast.Module([node], []), "kernels.py", "exec"), ns)


def run(kernel, consts, files, out):
    with tempfile.TemporaryDirectory() as d:
        argv = [BIN, "run", os.path.join(ROOT, "tests", "chx", f"{kernel}.chx")] + consts
        for name, values in files:
            path = os.path.join(d, name)
            with open(path, "w") as f:
                f.write(" ".join(repr(v) for v in values) + "\n")
            argv += ["--in", name, path]
        r = subprocess.run(argv + ["--out", out], capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip()
    return [float(x) for x in r.stdout.split()], r.stderr


def compare(name, got, want):
    global failed
    if got is None:
        print(f"FAIL {name}: the device program refused: {want}")
        failed += 1
        return
    peak = max(abs(v) for v in want)
    worst = max(abs(a - b) for a, b in zip(got, want)) if len(got) == len(want) else float("inf")
    good = worst <= TOL * peak
    print(f"{'ok  ' if good else 'FAIL'} {name}: {len(got)} values, largest difference {worst:.3g} = {worst / peak:.2e} of the peak {peak:.4g}")
    failed += 0 if good else 1


got, err = run("bwd_dx", ["r=2", "n=2", "k=3"], [("dy", [1, 2, 3, 4]), ("w", [1, 0, 2, 0, 1, 3])], "dx")
compare("bwd_dx known answer 1 2 8 3 4 18", got, err if got is None else [1.0, 2.0, 8.0, 3.0, 4.0, 18.0])
if got is not None:
    print("     device printed:", *got)

CASES = [
    ("bwd_dx", dict(r=8, n=32, k=64), ["dy", "w"], lambda r, n, k: [r * n, n * k], "dx", ns["bwd_dx"]),
    ("bwd_dx", dict(r=3, n=5, k=4), ["dy", "w"], lambda r, n, k: [r * n, n * k], "dx", ns["bwd_dx"]),
    ("bwd_dw", dict(r=8, n=32, k=64), ["dyt", "xt"], lambda r, n, k: [n * r, k * r], "dw", ns["bwd_dw"]),
    ("bwd_dw", dict(r=256, n=32, k=64), ["dyt", "xt"], lambda r, n, k: [n * r, k * r], "dw", ns["bwd_dw"]),
    ("bwd_db", dict(r=64, n=32), ["ones", "dy"], lambda r, n: [r, r * n], "db", ns["bwd_db"]),
    ("bwd_sigmoid", dict(r=8, n=64), ["z", "dy"], lambda r, n: [r * n, r * n], "dz", ns["bwd_sigmoid"]),
]
for kernel, c, names, sizes, out, ref in CASES:
    tensors = ns["lcg"](sizes(**c))
    got, err = run(kernel, [f"{k}={v}" for k, v in c.items()], list(zip(names, tensors)), out)
    label = f"{kernel} " + " ".join(f"{k}={v}" for k, v in c.items())
    compare(label, got, err if got is None else ref(**c))
sys.exit(1 if failed else 0)
