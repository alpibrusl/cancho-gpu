#!/usr/bin/env python3
"""A deeper ReLU network for binary classification trained with Adam on minibatches entirely on the device (docs/resident.md section 13): D inputs, hidden layers of the given
widths, one sigmoid output, cross-entropy loss, a request is a count of steps, the answer the mean loss of the last minibatch (and, with `dump`, every parameter).

    python3 examples/train/deep_session.py DATA_DIR OUT_DIR WIDTHS BATCH EPOCHS [--lr L] [--train N] [--dump]       (DATA_DIR holds X.npy and y.npy)

The minibatches are consecutive blocks of the training rows, taken round and round (a table, `resident.md` section 12.2): the data is shuffled once, not every epoch."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
B1, B2, EPS = 0.9, 0.999, 1e-8


def block(n, cap):
    return next(b for b in range(min(n, cap), 0, -1) if n % b == 0)


def tuned_bm(batch, width):
    """Samples a threadgroup takes (measured on an M4 Max, docs/resident.md section 13): few of them, so that a minibatch is spread over many threadgroups."""
    return block(batch, 2 if width >= 256 else 16)


def init(dims, seed=3):
    """Glorot-uniform weights [n, k] and zero biases for each layer, float32."""
    rng = np.random.RandomState(seed)
    ws, bs = [], []
    for k, n in zip(dims[:-1], dims[1:]):
        lim = math.sqrt(6.0 / (k + n))
        ws.append(rng.uniform(-lim, lim, size=(n, k)).astype(np.float32))
        bs.append(np.zeros(n, dtype=np.float32))
    return ws, bs


def put(path, values):
    with open(path, "wb") as f:
        f.write(np.asarray(values, dtype="<f4").reshape(-1).tobytes())


def schedule(steps, lr):
    return [(lr / (1.0 - B1 ** t), 1.0 / math.sqrt(1.0 - B2 ** t)) for t in range(1, steps + 1)]


def write(data_dir, directory, widths, batch, steps, lr=1e-3, train=None, dump=False, seed=3, bm=None, threads=None):
    """Writes the session in `directory`; answers its path. `steps` is the length of the step table (the schedule of Adam's bias corrections)."""
    os.makedirs(directory, exist_ok=True)
    p = lambda name: os.path.join(directory, name)
    X = np.load(os.path.join(data_dir, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(data_dir, "y.npy")).astype(np.float32)
    n_all = train or len(X)
    nb = n_all // batch
    X, y = X[:nb * batch], y[:nb * batch]
    d = X.shape[1]
    dims = [d] + list(widths) + [1]
    nl = len(dims) - 1
    ws, bs = init(dims, seed)
    put(p("x"), X)
    put(p("xt"), np.concatenate([X[j * batch:(j + 1) * batch].T.reshape(-1) for j in range(nb)]))
    put(p("y"), y)
    put(p("ones"), np.ones(batch))
    put(p("sched"), np.array(schedule(steps, lr)).reshape(-1))
    for l in range(nl):
        put(p(f"w{l}"), ws[l])
        put(p(f"b{l}"), bs[l])
        put(p(f"zw{l}"), np.zeros_like(ws[l]))
        put(p(f"zb{l}"), np.zeros_like(bs[l]))
        if l < nl - 1:
            put(p(f"eye{l}"), np.eye(dims[l + 1]))
    bm = bm or tuned_bm(batch, max(widths))
    if threads is None and max(widths) >= 256:
        threads = 256  # wide layers: more threads on each of the many small threadgroups
    chx0 = lambda name: os.path.join(HERE, name) if os.path.exists(os.path.join(HERE, name)) else os.path.join(ROOT, "tests", "chx", name)

    def chx(name):
        # the same kernel with this size's threadgroup shape, when one is asked for
        if threads is None or name not in ("dense_fwd.chx", "out_fwd.chx", "hid_bwd.chx"):
            return chx0(name)
        text = open(chx0(name)).read()
        text = "\n".join(f"schedule {line.split()[1]} for {line.split()[3]} {{ threads {threads} }}" if line.startswith("schedule ") else line for line in text.split("\n"))
        with open(p(name), "w") as f:
            f.write(text)
        return p(name)
    L = ["# a ReLU network " + "-".join(map(str, dims)) + f", minibatches of {batch}, Adam {lr}"]
    H = nl - 1  # hidden layers: 0 .. H-1; the output layer is H
    for l in range(H):
        L.append(f"program f{l} {chx('dense_fwd.chx')} b={batch} k={dims[l]} n={dims[l + 1]} bm={bm}")
    L.append(f"program fo {chx('out_fwd.chx')} b={batch} k={dims[H]} bm={bm}")
    for l in reversed(range(H)):
        L.append(f"program q{l} {chx('hid_bwd.chx')} b={batch} n={dims[l + 1]} n2={dims[l + 2]} bm={bm}")
    for l in range(nl):
        L.append(f"program gw{l} {chx('bwd_dw.chx')} r={batch} n={dims[l + 1]} k={dims[l]}")
        L.append(f"program gb{l} {chx('bwd_dw.chx')} r={batch} n={dims[l + 1]} k=1")
    L.append(f"program ls {chx('bwd_dw.chx')} r={batch} n=1 k=1")
    tensors = []
    for l in range(nl):
        tensors.append((f"w{l}", dims[l + 1] * dims[l]))
        tensors.append((f"b{l}", dims[l + 1]))
    for name, n in tensors:
        L.append(f"program a{name} {chx('adam.chx')} n={n} bn={block(n, 128)} gs={1.0 / batch!r} b1={B1} b2={B2} eps={EPS}")
    f32 = " f32"
    L += [f"table f0.hin {p('x')} {nb}{f32}", f"table gw0.xt {p('xt')} {nb}{f32}", f"table fo.y {p('y')} {nb}{f32}"]
    L += [f"resident ls.xt {p('ones')}{f32}"] + [f"resident gb{l}.xt {p('ones')}{f32}" for l in range(nl)]
    for l in range(H):
        L.append(f"resident q{l}.eye {p(f'eye{l}')}{f32}")
    for l in range(nl):
        owner = f"f{l}" if l < H else "fo"
        L.append(f"resident {owner}.w {p(f'w{l}')}{f32}")
        L.append(f"resident {owner}.bias {p(f'b{l}')}{f32}")
        if l < H:
            L.append(f"resident f{l}.biasc {p(f'b{l}')}{f32}")
        if l >= 1:
            L.append(f"resident q{l - 1}.w {p(f'w{l}')}{f32}")
    for name, n in tensors:
        z = "zw" if name[0] == "w" else "zb"
        L += [f"resident a{name}.w {p(name)}{f32}", f"resident a{name}.m {p(z + name[1:])}{f32}", f"resident a{name}.v {p(z + name[1:])}{f32}", f"table a{name}.s {p('sched')} {steps}{f32}"]
    for l in range(H):
        L.append(f"link f{l}.h -> " + (f"f{l + 1}.hin" if l + 1 < H else "fo.hin"))
        L.append(f"link f{l}.h -> q{l}.h")
        if l >= 1:
            L.append(f"link f{l - 1}.ht -> gw{l}.xt")
    L.append(f"link f{H - 1}.ht -> gw{H}.xt")
    L.append(f"link fo.dz -> q{H - 1}.dzn")
    for l in reversed(range(H - 1)):
        L.append(f"link q{l + 1}.dz -> q{l}.dzn")
    for l in range(H):
        L += [f"link q{l}.dzt -> gw{l}.dyt", f"link q{l}.dzt -> gb{l}.dyt"]
    L += [f"link fo.dzt -> gw{H}.dyt", f"link fo.dzt -> gb{H}.dyt", "link fo.l -> ls.dyt"]
    for l in range(nl):
        L += [f"link gw{l}.dw -> aw{l}.grad", f"link gb{l}.dw -> ab{l}.grad"]
    for l in range(nl):
        owner = f"f{l}" if l < H else "fo"
        L += [f"carry aw{l}.wo -> aw{l}.w", f"carry aw{l}.mo -> aw{l}.m", f"carry aw{l}.vo -> aw{l}.v", f"carry aw{l}.wo -> {owner}.w"]
        if l >= 1:
            L.append(f"carry aw{l}.wo -> q{l - 1}.w")
        L += [f"carry ab{l}.wo -> ab{l}.w", f"carry ab{l}.mo -> ab{l}.m", f"carry ab{l}.vo -> ab{l}.v", f"carry ab{l}.wo -> {owner}.bias"]
        if l < H:
            L.append(f"carry ab{l}.wo -> f{l}.biasc")
    L += ["steps", f"answer ls.dw scale 1/{batch}"]
    if dump:
        for name, n in tensors:
            L.append(f"answer a{name}.wo")
    with open(p("session"), "w") as f:
        f.write("\n".join(L) + "\n")
    return p("session")


def forward(ws, bs, x):
    h = x
    for w, b in zip(ws[:-1], bs[:-1]):
        h = np.maximum(h @ w.T + b, 0.0)
    return (h @ ws[-1].T + bs[-1])[:, 0]


def accuracy(ws, bs, x, y):
    return float((((forward(ws, bs, x) > 0.0)) == (y > 0.5)).mean())


def reference(data_dir, widths, batch, steps, lr=1e-3, train=None, seed=3):
    """The same training in binary64 numpy: the mean loss of each step's minibatch (before its update) and the final parameters."""
    X = np.load(os.path.join(data_dir, "X.npy")).astype(np.float64)
    y = np.load(os.path.join(data_dir, "y.npy")).astype(np.float64)
    nb = (train or len(X)) // batch
    X, y = X[:nb * batch], y[:nb * batch]
    dims = [X.shape[1]] + list(widths) + [1]
    ws, bs = [a.astype(np.float64) for a in init(dims, seed)[0]], [a.astype(np.float64) for a in init(dims, seed)[1]]
    P = ws + bs
    m = [np.zeros_like(a) for a in P]
    v = [np.zeros_like(a) for a in P]
    losses = []
    for step in range(1, steps + 1):
        lo = ((step - 1) % nb) * batch
        x, t = X[lo:lo + batch], y[lo:lo + batch]
        hs = [x]
        for w, b in zip(ws[:-1], bs[:-1]):
            hs.append(np.maximum(hs[-1] @ w.T + b, 0.0))
        z = (hs[-1] @ ws[-1].T + bs[-1])[:, 0]
        losses.append(float(np.mean(np.logaddexp(0.0, z) - t * z)))
        dz = (1.0 / (1.0 + np.exp(-z)) - t)[:, None]
        gw, gb = [None] * len(ws), [None] * len(ws)
        for l in reversed(range(len(ws))):
            gw[l] = dz.T @ hs[l]
            gb[l] = dz.sum(0)
            if l > 0:
                dz = (dz @ ws[l]) * (hs[l] > 0.0)
        for i, (param, g) in enumerate(zip(ws + bs, gw + gb)):
            g = g / batch
            m[i] = m[i] + (g - m[i]) * (1.0 - B1)
            v[i] = v[i] * B2 + (1.0 - B2) * g * g
            param -= lr / (1.0 - B1 ** step) * m[i] / (np.sqrt(v[i]) / math.sqrt(1.0 - B2 ** step) + EPS)
    return losses, (ws, bs)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    opt = lambda name, default: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
    lr = float(opt("--lr", 1e-3))
    train = int(opt("--train", 0)) or None
    for v in (str(lr), opt("--train", "")):
        if v in a:
            a.remove(v)
    print(write(a[0], a[1], [int(w) for w in a[2].split(",")], int(a[3]), int(a[4]), lr=lr, train=train, dump="--dump" in sys.argv))
