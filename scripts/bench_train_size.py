#!/usr/bin/env python3
"""Does the Metal device program overtake the CPU interpreter as the fit grows?
docs/training.md section 6 step 3 left it open. Times one loss-and-gradient
evaluation of examples/train/lsq_objective.py (forward lsq_residual, backward
bwd_dw) at growing R x K, through `cancho-gpu run` and through
`cancho-gpu-metal-device`, and checks that both give the same loss.

    python3 scripts/bench_train_size.py [--cpu build/cancho-gpu] [--metal build/cancho-gpu-metal-device] [--reps 3]

and against in-process baselines doing the same loss and gradient (docs/training.md section 6
step 3): pure Python, numpy, and torch on the CPU and on MPS. The baselines hold the data in
memory (on the device for MPS) and per evaluation take `w` in and give the loss and gradient
back as Python floats, which is what an optimiser needs; they are float32 as the kernels are.

One machine, medians, not a benchmark in the sense of benchmark.md."""
import argparse
import os
import subprocess
import tempfile
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "train"))
import lsq_objective as L  # noqa: E402
import lsq_session  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cpu", default=os.path.join(ROOT, "build", "cancho-gpu"))
ap.add_argument("--metal", default=os.path.join(ROOT, "build", "cancho-gpu-metal-device"))
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--session", default=os.path.join(ROOT, "build", "cancho-gpu-session"))
ap.add_argument("--sizes", default="16x3,256x16,1024x64,4096x128,16384x256")
args = ap.parse_args()

def session_ms(r, k, w, reps):
    """(setup seconds, ms per request) of one resident session on the CPU interpreter: the data is read once, a request is `w`."""
    if not os.path.exists(args.session):
        return None
    d = tempfile.mkdtemp()
    try:
        path = lsq_session.write(d, r, k)
        line = " ".join(repr(v) for v in w) + "\n"
        t0 = time.perf_counter()
        proc = subprocess.Popen([os.path.abspath(args.session), path, "--target", "nvidia-ada"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        proc.stdin.write(line)
        proc.stdin.flush()
        first = proc.stdout.readline()
        setup = time.perf_counter() - t0
        if not first:
            return None
        ts = []
        for _ in range(reps):
            t1 = time.perf_counter()
            proc.stdin.write(line)
            proc.stdin.flush()
            proc.stdout.readline()
            ts.append(1000 * (time.perf_counter() - t1))
        proc.stdin.close()
        proc.wait(timeout=60)
        return setup, statistics.median(ts)
    finally:
        import shutil
        shutil.rmtree(d)


def baselines(x, t, w, reps):
    """ms per evaluation of loss 1/(2R)|e|^2 and gradient 1/R e^T x, e = x w - t, in process."""
    out = {}
    R, K = len(x), len(w)

    def med(fn):
        fn()  # warm up
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            r = fn()
            ts.append(1000 * (time.perf_counter() - t0))
        return statistics.median(ts), r

    if R * K <= 1 << 20:
        def pure():
            e = [sum(a * b for a, b in zip(row, w)) - ti for row, ti in zip(x, t)]
            return 0.5 * sum(v * v for v in e) / R, [sum(e[i] * x[i][c] for i in range(R)) / R for c in range(K)]
        out["python"] = med(pure)
    try:
        import numpy as np
        xn, tn = np.array(x, dtype=np.float32), np.array(t, dtype=np.float32)

        def npy():
            e = xn @ np.array(w, dtype=np.float32) - tn
            return float(0.5 * (e @ e) / R), (e @ xn / R).tolist()
        out["numpy"] = med(npy)
    except ImportError:
        pass
    try:
        import torch
        for dev in ("cpu", "mps"):
            if dev == "mps" and not torch.backends.mps.is_available():
                continue
            xt_ = torch.tensor(x, dtype=torch.float32, device=dev)
            tt_ = torch.tensor(t, dtype=torch.float32, device=dev)

            def run_torch(xt_=xt_, tt_=tt_, dev=dev):
                wt = torch.tensor(w, dtype=torch.float32, device=dev)
                e = xt_ @ wt - tt_
                loss = 0.5 * (e @ e) / R
                g = e @ xt_ / R
                if dev == "mps":
                    torch.mps.synchronize()
                return float(loss), g.tolist()
            out["torch-" + dev] = med(run_torch)
    except ImportError:
        pass
    return out


NAMES = ["python", "numpy", "torch-cpu", "torch-mps"]
print(f"{'R x K':>12} {'cancho cpu':>11} {'cancho mtl':>11} " + " ".join(f"{n:>10}" for n in NAMES) + "   (ms per evaluation)")
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
    sess = session_ms(L.R, L.K, w, max(args.reps, 5)) if L.R * L.K <= 1 << 20 else None
    if sess:
        print(f"{size:>12}  session: {sess[1]:.1f} ms per evaluation after {sess[0]:.2f} s of setup (the data is read once)")
    x, t = L.data()
    base = baselines(x, t, w, max(args.reps, 5))
    cell = lambda r: f"{r[0]:11.0f}" if r[0] is not None else f"{'refused':>11}"
    bc = lambda n: f"{base[n][0]:10.2f}" if n in base else f"{'-':>10}"
    print(f"{size:>12} {cell(row['cpu'])} {cell(row['metal'])} " + " ".join(bc(n) for n in NAMES))
    # agreement of every path that answered with the first baseline's loss and gradient
    ref = base.get("numpy") or base.get("python")
    notes = []
    for name, r in list(row.items()) + [(n, base[n]) for n in base]:
        if r[0] is None or ref is None:
            continue
        loss, grad = r[1]
        gd = max(abs(a - b) for a, b in zip(grad, ref[1][1])) / max(max(abs(v) for v in ref[1][1]), 1e-30)
        notes.append(f"{name} {abs(loss - ref[1][0]) / abs(ref[1][0]):.0e}/{gd:.0e}")
    print("             loss rel / grad of peak against numpy: " + ", ".join(notes))
