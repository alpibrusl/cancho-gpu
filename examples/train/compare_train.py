#!/usr/bin/env python3
"""Train the same ReLU network on the same data with torch (CPU and MPS), OpenNN (CPU) and cancho-gpu (a Metal session), and compare the time and what is learned
(docs/resident.md section 13).

    python3 examples/train/compare_train.py [--dir DIR] [--epochs 20] [--width 256] [--layers 3] [--batch 256] [--opennn BINARY] [--only cancho,torch-cpu,...]

The data (DIR/X.npy, y.npy: (ntrain + ntest) x 64 standard normal features and a label from a random teacher network with noise, the first --ntrain rows to train on, then --ntest to test) is made
if it is not there. The network is 64-width x layers-1 ReLU units, a sigmoid output and cross-entropy, Adam at 1e-3, minibatches of 256, 20 epochs. torch and cancho-gpu start from the same
Glorot weights and take the same minibatches in the same order (the data is shuffled once), so their loss curves can be compared; OpenNN starts from its own Glorot draw and shuffles itself (its own `dense`
driver of benchmarks/families/dense.cpp, built from its source). Every framework's test accuracy is computed here, from its parameters, by the same numpy code, except OpenNN's, which its driver prints (threshold 0.5)."""
import math
import os
import re
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import deep_session as D  # noqa: E402
import mp_session as MP  # noqa: E402

arg = lambda name, default: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
DIR = arg("--dir", os.path.join(ROOT, "build", "compare"))
EPOCHS = int(arg("--epochs", 20))
WIDTH = int(arg("--width", 256))
LAYERS = int(arg("--layers", 3))
BATCH = int(arg("--batch", 256))
LR = 1e-3
NTRAIN = int(arg("--ntrain", 100000))
NTEST = int(arg("--ntest", 20000))
ONLY = arg("--only", "cancho,cancho-mp,torch-cpu,torch-mps,torch-mps-amp,opennn").split(",")
OPENNN = arg("--opennn", os.environ.get("DENSE_OPENNN", ""))
SESSION = os.environ.get("SESSION", os.path.join(ROOT, "build", "cancho-gpu-metal-session"))
WIDTHS = [WIDTH] * LAYERS
NB = NTRAIN // BATCH


def make_data():
    os.makedirs(DIR, exist_ok=True)
    if os.path.exists(os.path.join(DIR, "X.npy")):
        return
    rng = np.random.RandomState(7)
    d, n, m = 64, NTRAIN, NTEST
    W = [rng.normal(size=(d, 48)) / math.sqrt(d), rng.normal(size=(48, 48)) / math.sqrt(48), rng.normal(size=(48, 1)) / math.sqrt(48)]
    X = rng.normal(size=(n + m, d)).astype(np.float32)
    h = X
    for w in W[:-1]:
        h = np.tanh(h @ w)
    s = (h @ W[-1])[:, 0]
    s = (s - np.median(s)) / s.std()
    y = (s + 0.35 * rng.normal(size=s.shape) > 0).astype(np.float32)
    np.save(os.path.join(DIR, "X.npy"), X)
    np.save(os.path.join(DIR, "y.npy"), y)
    for name, sl in (("train", slice(0, n)), ("test", slice(n, n + m))):
        np.savetxt(os.path.join(DIR, name + ".csv"), np.concatenate([X[sl], y[sl, None]], 1), delimiter=",", fmt="%.7g")


def test_metrics(ws, bs):
    X = np.load(os.path.join(DIR, "X.npy"))[NTRAIN:NTRAIN + NTEST].astype(np.float64)
    y = np.load(os.path.join(DIR, "y.npy"))[NTRAIN:NTRAIN + NTEST].astype(np.float64)
    z = D.forward([w.astype(np.float64) for w in ws], [b.astype(np.float64) for b in bs], X)
    return float(((z > 0.0) == (y > 0.5)).mean()), float(np.mean(np.logaddexp(0.0, z) - y * z))


def run_cancho(mp=False):
    W = MP if mp else D
    sess = W.write(DIR, os.path.join(DIR, "s_time"), WIDTHS, BATCH, EPOCHS * NB, lr=LR, train=NTRAIN)
    p = subprocess.Popen([SESSION, sess], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t0 = time.perf_counter()
    p.stdin.write("1\n")
    p.stdin.flush()
    if not p.stdout.readline():
        raise SystemExit("cancho session: " + p.stderr.read()[:300])
    setup = time.perf_counter() - t0
    secs = []
    for _ in range(EPOCHS):
        t0 = time.perf_counter()
        p.stdin.write(f"{NB}\n")
        p.stdin.flush()
        p.stdout.readline()
        secs.append(time.perf_counter() - t0)
    p.stdin.close()
    p.wait()
    # the accuracy run: the same session with every parameter in the answer, after each epoch
    dump = W.write(DIR, os.path.join(DIR, "s_dump"), WIDTHS, BATCH, EPOCHS * NB, lr=LR, train=NTRAIN, dump=True)
    p = subprocess.Popen([SESSION, dump], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dims = [64] + WIDTHS + [1]
    curve = []
    p.stdin.write("1\n")
    p.stdin.flush()
    p.stdout.readline()
    done = 1
    for e in range(EPOCHS):
        p.stdin.write(f"{(e + 1) * NB - done}\n")
        p.stdin.flush()
        done = (e + 1) * NB
        v = [float(a) for a in p.stdout.readline().split()]
        loss, rest, o = v[0], v[1:], 0
        ws, bs = [], []
        for k, n in zip(dims[:-1], dims[1:]):
            ws.append(np.array(rest[o:o + n * k]).reshape(n, k))
            o += n * k
            bs.append(np.array(rest[o:o + n]))
            o += n
        curve.append(test_metrics(ws, bs) + (loss,))
    p.stdin.close()
    p.wait()
    return secs, curve, setup


def run_torch(device, amp=False):
    import torch
    X = np.load(os.path.join(DIR, "X.npy"))[:NTRAIN]
    y = np.load(os.path.join(DIR, "y.npy"))[:NTRAIN]
    dims = [64] + WIDTHS + [1]
    ws0, bs0 = D.init(dims, 3)
    Xt = torch.tensor(X[:NB * BATCH], device=device)
    yt = torch.tensor(y[:NB * BATCH], device=device)
    P = [torch.tensor(a, device=device, requires_grad=True) for a in ws0 + bs0]
    nl = len(ws0)
    opt = torch.optim.Adam(P, lr=LR)

    def step(j):
        lo = j * BATCH
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            h = Xt[lo:lo + BATCH]
            for l in range(nl - 1):
                h = torch.relu(h @ P[l].T + P[nl + l])
            z = (h @ P[nl - 1].T + P[2 * nl - 1])[:, 0]
        z = z.float()
        loss = torch.nn.functional.softplus(z).mean() - (yt[lo:lo + BATCH] * z).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    sync = torch.mps.synchronize if device == "mps" else (lambda: None)
    secs, curve = [], []
    for e in range(EPOCHS):
        sync()
        t0 = time.perf_counter()
        for j in range(NB):
            loss = step(j)
        sync()
        secs.append(time.perf_counter() - t0)
        ws = [P[l].detach().cpu().numpy() for l in range(nl)]
        bs = [P[nl + l].detach().cpu().numpy() for l in range(nl)]
        curve.append(test_metrics(ws, bs) + (float(loss.detach()),))
    return secs, curve, 0.0


def run_opennn():
    if not OPENNN or not os.path.exists(OPENNN):
        return None
    t0 = time.perf_counter()
    r = subprocess.run([OPENNN, "train", os.path.join(DIR, "train.csv"), os.path.join(DIR, "test.csv"), str(EPOCHS), str(BATCH), str(WIDTH), str(LAYERS), "relu", "cpu", "fp32"], capture_output=True, text=True)
    wall = time.perf_counter() - t0
    out = r.stdout
    secs = [float(v) for v in re.search(r"epoch_order=([\d.,e-]+)", out).group(1).split(",")]
    acc = float(re.search(r"test_accuracy=([\d.e-]+)", out).group(1))
    auc = float(re.search(r"test_roc_auc=([\d.e-]+)", out).group(1))
    return secs, acc, auc, wall


def main():
    make_data()
    rows = []
    print(f"network 64-{'-'.join(map(str, WIDTHS))}-1 (ReLU, sigmoid output, cross-entropy), Adam {LR}, minibatches of {BATCH}, {EPOCHS} epochs of {NB * BATCH} samples; test set {NTEST}", flush=True)
    for name in ONLY:
        if name == "cancho":
            secs, curve, setup = run_cancho()
        elif name == "cancho-mp":
            secs, curve, setup = run_cancho(True)
        elif name == "torch-mps-amp":
            secs, curve, setup = run_torch("mps", True)
        elif name == "torch-cpu":
            secs, curve, setup = run_torch("cpu")
        elif name == "torch-mps":
            secs, curve, setup = run_torch("mps")
        elif name == "opennn":
            r = run_opennn()
            if r is None:
                print("opennn: no binary (--opennn or DENSE_OPENNN)")
                continue
            secs, acc, auc, wall = r
            rows.append((name, statistics.median(secs), sum(secs), acc, None, f"AUC {auc:.4f}"))
            print(f"{name}: {statistics.median(secs):.3f} s/epoch, test accuracy {acc:.4f}, AUC {auc:.4f}", flush=True)
            continue
        else:
            continue
        acc, tloss, last = curve[-1]
        rows.append((name, statistics.median(secs), sum(secs), acc, tloss, ""))
        print(f"{name}: {statistics.median(secs):.3f} s/epoch, total {sum(secs):.1f} s, test accuracy {acc:.4f}, test loss {tloss:.4f}", flush=True)
        print("   test accuracy by epoch: " + " ".join(f"{c[0]:.3f}" for c in curve), flush=True)
        best = max(c[0] for c in curve)
        rows[-1] = rows[-1][:5] + (f"best {best:.4f} at epoch {[c[0] for c in curve].index(best) + 1}",)
    print("\n| framework | s / epoch (median) | training time | test accuracy, last epoch | test loss | note |\n|---|---|---|---|---|---|")
    for name, med, tot, acc, tl, extra in rows:
        print(f"| {name} | {med:.3f} | {tot:.1f} s | {acc:.4f} | {'-' if tl is None else f'{tl:.4f}'} | {extra} |")


main()
