#!/usr/bin/env python3
"""Does the Metal device program overtake the CPU interpreter as the fit grows?
docs/training.md section 6 step 3 left it open. Times one loss-and-gradient
evaluation of examples/train/lsq_objective.py (forward lsq_residual, backward
bwd_dw) at growing R x K, through `cancho-gpu run` and through
`cancho-gpu-metal-device`, and checks that both give the same loss.

    python3 scripts/bench_train_size.py [--cpu build/cancho-gpu] [--metal build/cancho-gpu-metal-device] [--reps 3]

One machine, medians, not a benchmark in the sense of benchmark.md."""
import argparse
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "train"))
import lsq_objective as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cpu", default=os.path.join(ROOT, "build", "cancho-gpu"))
ap.add_argument("--metal", default=os.path.join(ROOT, "build", "cancho-gpu-metal-device"))
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--sizes", default="16x3,256x16,1024x64,4096x128,16384x256")
args = ap.parse_args()

print(f"{'R x K':>12} {'cpu ms':>9} {'metal ms':>9} {'ratio':>7}  loss agreement")
for size in args.sizes.split(","):
    L.R, L.K = (int(v) for v in size.split("x"))
    L.WSTAR = [((i % 7) - 3) / 2.0 for i in range(L.K)]
    w = [0.1 * ((i % 5) - 2) for i in range(L.K)]
    row = {}
    for name, binary, target in (("cpu", args.cpu, "nvidia-ada"), ("metal", args.metal, "-")):
        times = []
        for _ in range(args.reps):
            t0 = time.time()
            try:
                loss, grad = L.loss_grad(os.path.abspath(binary), ROOT, w, target)
            except Exception as e:  # a refusal or a size limit is a result
                row[name] = (None, None, str(e)[:80])
                break
            times.append(1000 * (time.time() - t0))
        else:
            row[name] = (statistics.median(times), (loss, grad), "")
    c, m = row["cpu"], row["metal"]
    if c[0] is None or m[0] is None:
        print(f"{size:>12} cpu: {c[2] or 'ok'}  metal: {m[2] or 'ok'}")
        continue
    gd = max(abs(a - b) for a, b in zip(c[1][1], m[1][1])) / max(max(abs(v) for v in c[1][1]), 1e-30)
    print(f"{size:>12} {c[0]:9.0f} {m[0]:9.0f} {c[0] / m[0]:7.2f}  loss {abs(c[1][0] - m[1][0]) / c[1][0]:.1e} rel, grad {gd:.1e} of peak")
