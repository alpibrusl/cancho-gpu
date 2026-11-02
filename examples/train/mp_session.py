#!/usr/bin/env python3
"""The network of deep_session.py in mixed precision on the matrix unit (docs/resident.md section 14): D inputs, hidden layers of the given
widths, one sigmoid output, cross-entropy loss, a request is a count of steps, the answer the mean loss of the last minibatch (and, with `dump`, every parameter).

    python3 examples/train/mp_session.py DATA_DIR OUT_DIR WIDTHS BATCH EPOCHS [--lr L] [--train N] [--dump]       (DATA_DIR holds X.npy and y.npy)

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
    """Writes the session in `directory`; answers its path. Weights are kept in f32 (Adam's) and in f16 (what the matrix unit reads), activations and gradients of the
    layers in f16, products accumulated in f32. Every width and the batch must be multiples of 64, the input width of 32."""
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
    H = nl - 1
    bm = bm or (2 if max(widths) >= 1024 else 4 if max(widths) >= 512 else 8)  # rows an elementwise threadgroup takes: its staged tiles must fit in 32 KiB
    assert batch % 64 == 0 and all(w % 64 == 0 for w in widths) and d % 32 == 0
    ws, bs = init(dims, seed)
    put(p("x"), X)
    put(p("y"), y)
    put(p("ones"), np.ones(batch))
    put(p("sched"), np.array(schedule(steps, lr)).reshape(-1))
    for l in range(nl):
        put(p(f"w{l}"), ws[l])
        put(p(f"b{l}"), bs[l])
        put(p(f"zw{l}"), np.zeros_like(ws[l]))
        put(p(f"zb{l}"), np.zeros_like(bs[l]))
        if l < H:  # the f16 copy of the weights, binary16 words (the `f16` flag)
            with open(p(f"w{l}.h"), "wb") as f:
                f.write(ws[l].astype("<f2").reshape(-1).tobytes())
    chx0 = lambda name: next(q for q in (os.path.join(HERE, name), os.path.join(ROOT, "tests", "chx", name), os.path.join(ROOT, "tests", "chx", "ported", name)) if os.path.exists(q))
    chx = chx0
    L = ["# a mixed-precision ReLU network " + "-".join(map(str, dims)) + f", minibatches of {batch}, Adam {lr}"]
    L.append(f"program cast {chx('cast_h.chx')} r={batch} c={d} bm={bm}")
    for l in range(H):
        L.append(f"program mm{l} {chx('gemm_mma.chx')} m={batch} n={dims[l + 1]} k={dims[l]}")
        L.append(f"program ep{l} {chx('bias_relu_h.chx')} b={batch} n={dims[l + 1]} bm={bm}")
    L.append(f"program fo {chx('out_fwd_h.chx')} b={batch} k={dims[H]} bm={bm}")
    L.append(f"program ob {chx('out_bwd_h.chx')} b={batch} k={dims[H]} bm={bm}")
    for l in reversed(range(H)):
        L.append(f"program gw{l} {chx('gemm_mma_tn.chx')} m={dims[l + 1]} n={dims[l]} k={batch}")
        L.append(f"program gb{l} {chx('rowvec_mm.chx')} b={batch} k={dims[l + 1]} bk={block(dims[l + 1], 32)}")
        if l >= 1:
            L.append(f"program dh{l} {chx('gemm_mma_nn.chx')} m={batch} n={dims[l]} k={dims[l + 1]}")
            L.append(f"program sm{l} {chx('step_mask_h.chx')} b={batch} n={dims[l]} bm={bm}")
    L.append(f"program gwo {chx('rowvec_mm.chx')} b={batch} k={dims[H]} bk={block(dims[H], 32)}")
    L.append(f"program gbo {chx('rowvec_mm.chx')} b={batch} k=1 bk=1")
    L.append(f"program ls {chx('bwd_dw.chx')} r={batch} n=1 k=1")
    tensors = []
    for l in range(nl):
        tensors.append((f"w{l}", dims[l + 1] * dims[l]))
        tensors.append((f"b{l}", dims[l + 1]))
    for name, n in tensors:
        kern = "adam_h.chx" if name[0] == "w" and int(name[1:]) < H else "adam.chx"
        L.append(f"program a{name} {chx(kern)} n={n} bn={block(n, 128)} gs={1.0 / batch!r} b1={B1} b2={B2} eps={EPS}")
    f32 = " f32"
    L += [f"table cast.x {p('x')} {nb}{f32}", f"table fo.y {p('y')} {nb}{f32}", f"resident ls.xt {p('ones')}{f32}", f"resident gbo.v {p('ones')}{f32}"]
    for l in range(H):
        L.append(f"resident gb{l}.v {p('ones')}{f32}")
        L += [f"resident mm{l}.w {p(f'w{l}.h')} f16", f"resident ep{l}.bias {p(f'b{l}')}{f32}"]
        if l >= 1:
            L.append(f"resident dh{l}.w {p(f'w{l}.h')} f16")
    L += [f"resident fo.w {p(f'w{H}')}{f32}", f"resident fo.bias {p(f'b{H}')}{f32}", f"resident ob.w {p(f'w{H}')}{f32}"]
    for name, n in tensors:
        z = "zw" if name[0] == "w" else "zb"
        L += [f"resident a{name}.w {p(name)}{f32}", f"resident a{name}.m {p(z + name[1:])}{f32}", f"resident a{name}.v {p(z + name[1:])}{f32}", f"table a{name}.s {p('sched')} {steps}{f32}"]
    # forward
    L.append("link cast.y -> mm0.a")
    for l in range(H):
        L.append(f"link mm{l}.y -> ep{l}.z")
        if l + 1 < H:
            L.append(f"link ep{l}.h -> mm{l + 1}.a")
    L += [f"link ep{H - 1}.h -> fo.hin", f"link ep{H - 1}.h -> ob.h", "link fo.dz -> ob.dzo", "link fo.l -> ls.dyt", "link fo.dz -> gwo.v"]
    L += [f"link ep{H - 1}.h -> gwo.x", "link fo.dzh -> gbo.x"]
    # backward: dz of layer l is ob.dz (l = H-1) or sm{l+1}.dz
    for l in reversed(range(H)):
        dz = "ob.dz" if l == H - 1 else f"sm{l + 1}.dz"
        L += [f"link {dz} -> gw{l}.a", f"link {dz} -> gb{l}.x"]
        L.append(f"link {'cast.y' if l == 0 else f'ep{l - 1}.h'} -> gw{l}.w")
        if l >= 1:
            L += [f"link {dz} -> dh{l}.a", f"link dh{l}.y -> sm{l}.dh", f"link ep{l - 1}.h -> sm{l}.h"]
    for l in range(H):
        L += [f"link gw{l}.y -> aw{l}.grad", f"link gb{l}.out -> ab{l}.grad"]
    L += [f"link gwo.out -> aw{H}.grad", f"link gbo.out -> ab{H}.grad"]
    for l in range(nl):
        for nm, dest in ((f"w{l}", None), (f"b{l}", None)):
            pass
    for name, n in tensors:
        L += [f"carry a{name}.wo -> a{name}.w", f"carry a{name}.mo -> a{name}.m", f"carry a{name}.vo -> a{name}.v"]
    for l in range(H):
        L.append(f"carry aw{l}.wh -> mm{l}.w")
        if l >= 1:
            L.append(f"carry aw{l}.wh -> dh{l}.w")
        L.append(f"carry ab{l}.wo -> ep{l}.bias")
    L += [f"carry aw{H}.wo -> fo.w", f"carry aw{H}.wo -> ob.w", f"carry ab{H}.wo -> fo.bias"]
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
