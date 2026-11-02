#!/usr/bin/env python3
"""Time one Adam step of the network of mlp_session.py: cancho-gpu's Metal training session against torch on the CPU and on MPS, the same model, data and optimiser, float32.

    python3 examples/train/mlp_bench.py [--sizes 4096x32x32,16384x64x128] [--steps 500] [--rounds 3] [--markdown]

Interleaved (each round runs every runtime once, in turn, and the median of the rounds is reported) because the machine's clocks move between runs (docs/benchmark.md).
cancho: one request of `steps` steps after a request of 20 (the first compiles nothing, but the GPU wakes up); torch: 20 warm-up steps, a synchronize, `steps` steps, a synchronize."""
import os
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import mlp_session as M  # noqa: E402

SESSION = os.environ.get("SESSION", os.path.join(ROOT, "build", "cancho-gpu-metal-session"))


def cancho(session, steps):
    p = subprocess.Popen([SESSION, session], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    p.stdin.write("20\n")
    p.stdin.flush()
    if not p.stdout.readline():
        raise SystemExit("the session did not answer: " + p.stderr.read()[:200])
    t0 = time.perf_counter()
    p.stdin.write(f"{steps}\n")
    p.stdin.flush()
    line = p.stdout.readline()
    dt = time.perf_counter() - t0
    p.stdin.close()
    p.wait()
    return dt / steps * 1000, float(line.split()[0])


def torch_run(device, r, k, hd, steps, lr):
    import torch
    x, t = M.data(r, k)
    w1, b1, w2, b2 = M.init(k, hd)
    X, T = torch.tensor(x, device=device), torch.tensor(t, device=device)
    P = [torch.tensor(a, device=device, requires_grad=True) for a in (w1, b1, w2, b2)]
    opt = torch.optim.Adam(P, lr=lr)

    def step():
        opt.zero_grad(set_to_none=True)
        h = torch.sigmoid(X @ P[0].T + P[1])
        e = h @ P[2] + P[3] - T
        loss = 0.5 * (e * e).sum() / r
        loss.backward()
        opt.step()
        return loss

    sync = torch.mps.synchronize if device == "mps" else (lambda: None)
    for _ in range(20):
        step()
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = step()
    sync()
    return (time.perf_counter() - t0) / steps * 1000, float(loss.detach())


def main():
    sizes = [tuple(int(v) for v in s.split("x")) for s in (sys.argv[sys.argv.index("--sizes") + 1] if "--sizes" in sys.argv else "4096x32x32,16384x64x128").split(",")]
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 500
    rounds = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 3
    try:
        import torch  # noqa: F401
        have_torch = True
    except ImportError:
        have_torch = False
    print("| model (samples x features x hidden) | parameters | cancho-gpu (Metal) | torch (CPU) | torch (MPS) | loss after 520 steps: cancho / torch |\n|---|---|---|---|---|---|")
    for r, k, hd in sizes:
        d = tempfile.mkdtemp()
        session = M.write(d, r, k, hd, 2000, lr=0.01)
        times = {"cancho": [], "cpu": [], "mps": []}
        loss = {}
        for _ in range(rounds):
            ms, loss["cancho"] = cancho(session, steps)
            times["cancho"].append(ms)
            if have_torch:
                for dev in ("cpu", "mps"):
                    ms, loss[dev] = torch_run(dev, r, k, hd, steps, 0.01)
                    times[dev].append(ms)
        med = {key: (statistics.median(v) if v else None) for key, v in times.items()}
        fmt = lambda v: "-" if v is None else f"{v:.3f} ms"
        n = hd * k + 2 * hd + 1
        print(f"| {r} x {k} x {hd} | {n} | {fmt(med['cancho'])} | {fmt(med['cpu'])} | {fmt(med['mps'])} | {loss['cancho']:.5f} / {loss.get('cpu', float('nan')):.5f} |")


if __name__ == "__main__":
    main()
