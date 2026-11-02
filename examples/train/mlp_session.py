#!/usr/bin/env python3
"""A two-layer network trained with Adam entirely on the device (docs/resident.md section 12): the data, the parameters and Adam's moments are resident,
a request is a count of steps, and the answer is the loss after them (and, with `--dump`, the parameters).

    python3 examples/train/mlp_session.py DIR R K HD STEPS [--dump] [--lr L] [--batch B] [--relu]     writes DIR/session and its files (numpy)
    echo 1000 | cancho-gpu-metal-session DIR/session                             1000 Adam steps, one wait; prints the loss

The model is mlp_fwd.chx (sigmoid hidden layer, or ReLU with --relu: mlp_fwd_relu.chx; linear output), the loss e^T e / 2R, the backward pass bwd_dw.chx and mlp_dzt.chx, the optimiser adam.chx on each of w1, b1, w2, b2.
The targets are a smooth function of the features, which a network of this size can fit."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
B1, B2, EPS = 0.9, 0.999, 1e-8


def block(n, cap):
    """The largest divisor of n that is at most cap."""
    return next(b for b in range(min(n, cap), 0, -1) if n % b == 0)


def tuned(r, hd):
    """(threads, bn) of bwd_dw2.chx, measured on an M4 Max: a result with many rows wants threadgroups of 32 of them and many threads for the long reduction."""
    return (512, 32) if hd >= 64 and r >= 8192 else None


def data(r, k, seed=1):
    """(x, t, init) with x [r, k], t [r] and the starting parameters, as float32 arrays."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1.0, 1.0, size=(r, k)).astype(np.float32)
    u = rng.normal(size=k) / math.sqrt(k)
    t = (np.sin(2.0 * (x @ u)) + 0.3 * x[:, 0] * x[:, 1 % k]).astype(np.float32)
    return x, t


def init(k, hd, seed=2):
    rng = np.random.RandomState(seed)
    w1 = (rng.normal(size=(hd, k)) / math.sqrt(k)).astype(np.float32)
    b1 = np.zeros(hd, dtype=np.float32)
    w2 = (rng.normal(size=hd) / math.sqrt(hd)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)
    return w1, b1, w2, b2


def schedule(steps, lr):
    """Per step: [lr / (1 - beta1^t), 1 / sqrt(1 - beta2^t)] in binary64 (the table of the Adam kernels)."""
    return [(lr / (1.0 - B1 ** t), 1.0 / math.sqrt(1.0 - B2 ** t)) for t in range(1, steps + 1)]


def put(path, values):
    """The shortest text that reads back as the same float32 (a 16384 x 64 matrix is 6 MB, under the 16 MiB a resident file may be)."""
    with open(path, "w") as f:
        f.write(" ".join(str(v) for v in np.asarray(values, dtype=np.float32).reshape(-1)) + "\n")


def write(directory, r_all, k, hd, steps, lr=0.01, dump=False, dw2=None, batch=None, act="sigmoid"):
    """With `batch` B the data (r_all samples) is a table of r_all / B minibatches and step j trains on batch j mod (r_all / B); the loss answered is that of the last minibatch."""
    suffix = "_relu" if act == "relu" else ""
    r = batch or r_all
    nb = r_all // r
    assert r * nb == r_all, "the batch must divide the samples"
    os.makedirs(directory, exist_ok=True)
    p = lambda name: os.path.join(directory, name)
    x, t = data(r_all, k)
    w1, b1, w2, b2 = init(k, hd)
    put(p("x"), x)
    put(p("xt"), np.concatenate([x[j * r:(j + 1) * r].T.reshape(-1) for j in range(nb)]))
    put(p("t"), t)
    put(p("ones"), np.ones(r))
    for name, v in (("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2)):
        put(p(name), v)
        put(p("z" + name), np.zeros_like(v))
    with open(p("sched"), "w") as f:
        f.write(" ".join(f"{a!r} {b!r}" for a, b in schedule(steps, lr)) + "\n")
    chx = lambda name: os.path.join(HERE, name) if os.path.exists(os.path.join(HERE, name)) else os.path.join(ROOT, "tests", "chx", name)
    if dw2 is None:
        dw2 = tuned(r, hd)
    # bwd_dw2 splits the reduction over more threadgroups when the result has many rows (hd >= 64); for fewer, bwd_dw.chx is faster (measured)
    gradient = "bwd_dw2.chx" if dw2 else "bwd_dw.chx"
    # bwd_dw2.chx with the schedule that suits this size
    if dw2:
        text = open(chx("bwd_dw2.chx")).read()
        text = "\n".join(f"schedule bwd_dw2 for {line.split()[3]} {{ threads {dw2[0]} bn {dw2[1]} bk 1 }}" if line.startswith("schedule bwd_dw2 for") else line for line in text.split("\n"))
        with open(p("bwd_dw2.chx"), "w") as f:
            f.write(text)
    bm = block(r, 32)
    lines = [f"# a {k}-{hd}-1 sigmoid network on {r} samples, Adam, lr {lr}",
             f"program fwd {chx('mlp_fwd' + suffix + '.chx')} r={r} k={k} hd={hd} bm={bm}",
             f"program dz  {chx('mlp_dzt' + suffix + '.chx')} r={r} hd={hd} bm={bm}",
             f"program gw1 {chx(gradient) if not dw2 else p(gradient)} r={r} n={hd} k={k}",
             f"program gb1 {chx(gradient) if not dw2 else p(gradient)} r={r} n={hd} k=1",
             f"program gw2 {chx('bwd_dw.chx')} r={r} n=1 k={hd}",
             f"program gb2 {chx('bwd_dw.chx')} r={r} n=1 k=1",
             f"program ls  {chx('bwd_dw.chx')} r={r} n=1 k=1"]
    sizes = {"w1": hd * k, "b1": hd, "w2": hd, "b2": 1}
    for name, n in sizes.items():
        lines.append(f"program a{name} {chx('adam.chx')} n={n} bn={block(n, 128)} gs={1.0 / r!r} b1={B1} b2={B2} eps={EPS}")
    if batch:
        lines += [f"table fwd.x {p('x')} {nb}", f"table fwd.t {p('t')} {nb}", f"table gw1.xt {p('xt')} {nb}"]
    else:
        lines += [f"resident fwd.x {p('x')}", f"resident fwd.t {p('t')}", f"resident gw1.xt {p('xt')}"]
    lines += [f"resident gb1.xt {p('ones')}", f"resident gb2.xt {p('ones')}"]
    for name in sizes:
        lines += [f"resident fwd.{name} {p(name)}", f"resident a{name}.w {p(name)}", f"resident a{name}.m {p('z' + name)}", f"resident a{name}.v {p('z' + name)}", f"table a{name}.s {p('sched')} {steps}"]
    lines += [f"resident dz.w2c {p('w2')}",
              "link fwd.ht -> dz.ht", "link fwd.e -> dz.e", "link dz.dzt -> gw1.dyt", "link dz.dzt -> gb1.dyt", "link fwd.e -> gw2.dyt", "link fwd.ht -> gw2.xt",
              "link fwd.e -> gb2.dyt", "link fwd.e -> ls.dyt", "link fwd.e -> ls.xt", "link gw1.dw -> aw1.grad", "link gb1.dw -> ab1.grad", "link gw2.dw -> aw2.grad", "link gb2.dw -> ab2.grad"]
    for name in sizes:
        lines += [f"carry a{name}.wo -> a{name}.w", f"carry a{name}.mo -> a{name}.m", f"carry a{name}.vo -> a{name}.v", f"carry a{name}.wo -> fwd.{name}"]
    lines += ["carry aw2.wo -> dz.w2c", "steps", f"answer ls.dw scale 0.5/{r}"]
    if dump:
        lines += [f"answer a{name}.wo" for name in sizes]
    with open(p("session"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return p("session")


def reference(r, k, hd, steps, lr=0.01, batch=None, act="sigmoid"):
    """The same training in binary64 numpy: (the loss before each step, the final parameters). With `batch`, step j uses minibatch j mod (r / batch)."""
    x_all, t_all = data(r, k)
    x_all, t_all = x_all.astype(np.float64), t_all.astype(np.float64)
    bsz = batch or r
    r_all, r = r, bsz
    w1, b1, w2, b2 = [a.astype(np.float64) for a in init(k, hd)]
    m = [np.zeros_like(a) for a in (w1, b1, w2, b2)]
    v = [np.zeros_like(a) for a in (w1, b1, w2, b2)]
    losses = []
    for step in range(1, steps + 1):
        lo = ((step - 1) % (r_all // r)) * r
        x, t = x_all[lo:lo + r], t_all[lo:lo + r]
        z = x @ w1.T + b1
        h = np.maximum(z, 0.0) if act == "relu" else 1.0 / (1.0 + np.exp(-z))
        e = h @ w2 + b2[0] - t
        losses.append(0.5 * float(e @ e) / r)
        dz = (e[:, None] * w2[None, :]) * ((h > 0.0) if act == "relu" else h * (1.0 - h))
        grads = [dz.T @ x, dz.sum(0), h.T @ e, np.array([e.sum()])]
        for i, (param, g) in enumerate(zip((w1, b1, w2, b2), grads)):
            g = g / r
            m[i] = m[i] + (g - m[i]) * (1.0 - B1)
            v[i] = v[i] * B2 + (1.0 - B2) * g * g
            param -= lr / (1.0 - B1 ** step) * m[i] / (np.sqrt(v[i]) / math.sqrt(1.0 - B2 ** step) + EPS)
    return losses, (w1, b1, w2, b2)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    lr = float(sys.argv[sys.argv.index("--lr") + 1]) if "--lr" in sys.argv else 0.01
    if "--lr" in sys.argv:
        a.remove(sys.argv[sys.argv.index("--lr") + 1])
    act = "relu" if "--relu" in sys.argv else "sigmoid"
    a = [x for x in a if x != "--relu"]
    batch = int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else None
    if batch:
        a.remove(sys.argv[sys.argv.index("--batch") + 1])
    print(write(a[0], int(a[1]), int(a[2]), int(a[3]), int(a[4]), lr=lr, dump="--dump" in sys.argv, batch=batch, act=act))
