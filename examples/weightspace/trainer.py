#!/usr/bin/env python3
"""The gradient-descent arm of W2 (docs/weight-space.md section 7): the 5 -> 64 -> 64 -> 3 tanh network of task.py trained with Adam on the CPU session path of cancho-gpu
(docs/resident.md sections 12 to 13: resident tables of minibatches, a request is a count of steps, parameters never leave the session), and the same forward as the harness's `forward`.

    python3 examples/weightspace/trainer.py ...       (see arms.py, which drives it)

The recipe is the M1 one (resident.md section 15's AdamW at 1e-3 with a 5% warm-up and a cosine decay; no weight decay here, which the 4.6 k weights do not need). The loss is e^T e / 2b on targets
standardised per axis (mean 0, standard deviation 1 on the training set), so that the metre-sized outputs do not make Adam's first steps tiny; the standardisation is folded into the last layer
exactly afterwards (task coordinates are the network's), so the trained network is an ordinary 5-64-64-3 tanh network in metres."""
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import task  # noqa: E402

B1, B2, EPS = 0.9, 0.999, 1e-8
ROOT = os.path.dirname(os.path.dirname(HERE))


def block(n, cap):
    return next(b for b in range(min(n, cap), 0, -1) if n % b == 0)


def put(path, values):
    with open(path, "wb") as f:
        f.write(np.asarray(values, dtype="<f4").reshape(-1).tobytes())


def schedule(steps, lr, warm=0.05):
    """Per step t: [lr_t / (1 - beta1^t), 1 / sqrt(1 - beta2^t)] (binary64), lr_t linear warm-up over `warm` of the steps then a cosine decay to 0."""
    w = max(1, int(round(warm * steps)))
    out = []
    for t in range(1, steps + 1):
        f = t / w if t <= w else 0.5 * (1.0 + math.cos(math.pi * (t - w) / max(1, steps - w)))
        out.append((lr * f / (1.0 - B1 ** t), 1.0 / math.sqrt(1.0 - B2 ** t)))
    return out


def forward_session(directory, params, x, rows=256):
    """The forward of a network on `rows` poses: the session path the harness's `forward` uses. Writes the session in `directory`; answers its path. Its answer line is the 3*rows outputs."""
    os.makedirs(directory, exist_ok=True)
    p = lambda name: os.path.join(directory, name)
    xb = np.zeros((rows, 5), dtype=np.float32)
    xb[:len(x)] = x
    put(p("x"), xb)
    for name in task.NAMES:
        put(p(name), params[name])
    put(p("eye"), np.eye(3))
    put(p("zt"), np.zeros((rows, 3)))
    bm = block(rows, 16)
    L = [f"program f0 {os.path.join(HERE, 'tanh_fwd.chx')} b={rows} k=5 n=64 bm={bm}",
         f"program f1 {os.path.join(HERE, 'tanh_fwd.chx')} b={rows} k=64 n=64 bm={bm}",
         f"program fo {os.path.join(HERE, 'lin_out.chx')} b={rows} k=64 n=3 bm={bm}",
         f"resident f0.hin {p('x')} f32", f"resident f0.w {p('w1')} f32", f"resident f0.bias {p('b1')} f32", f"resident f0.biasc {p('b1')} f32",
         f"resident f1.w {p('w2')} f32", f"resident f1.bias {p('b2')} f32", f"resident f1.biasc {p('b2')} f32",
         f"resident fo.w {p('w3')} f32", f"resident fo.bias {p('b3')} f32", f"resident fo.t {p('zt')} f32", f"resident fo.eye {p('eye')} f32",
         "link f0.h -> f1.hin", "link f1.h -> fo.hin", "input fo.t", "answer fo.y"]
    # `input fo.t` would need the request line to hold the targets; the targets are not needed for a forward, so keep them resident instead
    L = [l for l in L if not l.startswith("input")]
    with open(p("session"), "w") as f:
        f.write("\n".join(L) + "\n")
    return p("session")


def run_forward(session_bin, directory, params, x, rows=256):
    """Network outputs [len(x), 3] (float32 values as floats) computed by the CPU session."""
    s = forward_session(directory, params, x, rows)
    r = subprocess.run([session_bin, s, "--target", "nvidia-ada"], input="\n", capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError("cancho-gpu-session: " + r.stderr[:400])
    return np.array([float(v) for v in r.stdout.split()]).reshape(rows, 3)[:len(x)]


def train_session(directory, q, pos, init, batch, steps, lr=1e-3, dump=True, kernels=HERE):
    """Writes the training session (q [N, 5] radians, pos [N, 3] metres, N a multiple of `batch`). Answers (session path, (mean, std) of the targets). Step j trains on minibatch j mod N/batch."""
    os.makedirs(directory, exist_ok=True)
    p = lambda name: os.path.join(directory, name)
    n_all = len(q)
    nb = n_all // batch
    assert nb * batch == n_all
    mean, std = pos.mean(axis=0), pos.std(axis=0)
    t = ((pos - mean) / std).astype(np.float32)
    x = q.astype(np.float32)
    put(p("x"), x)
    put(p("xt"), np.concatenate([x[j * batch:(j + 1) * batch].T.reshape(-1) for j in range(nb)]))
    put(p("t"), t)
    put(p("ones"), np.ones(batch))
    put(p("eye3"), np.eye(3))
    put(p("eye64"), np.eye(64))
    put(p("sched"), np.array(schedule(steps, lr)).reshape(-1))
    for name in task.NAMES:
        put(p(name), init[name])
        put(p("z" + name), np.zeros_like(init[name]))
    bm = block(batch, 16)
    chx = lambda name: os.path.join(kernels, name)
    gw = os.path.join(ROOT, "tests", "chx", "bwd_dw.chx")
    ad = os.path.join(ROOT, "examples", "train", "adam.chx")
    dims = task.DIMS
    L = [f"# the W2 gradient arm: {'-'.join(map(str, dims))} tanh, minibatches of {batch}, Adam {lr}, {steps} steps",
         f"program f0 {chx('tanh_fwd.chx')} b={batch} k=5 n=64 bm={bm}",
         f"program f1 {chx('tanh_fwd.chx')} b={batch} k=64 n=64 bm={bm}",
         f"program fo {chx('lin_out.chx')} b={batch} k=64 n=3 bm={bm}",
         f"program q1 {chx('tanh_bwd.chx')} b={batch} n=64 n2=3 bm={bm}",
         f"program q0 {chx('tanh_bwd.chx')} b={batch} n=64 n2=64 bm={bm}",
         f"program gw1 {gw} r={batch} n=64 k=5", f"program gb1 {gw} r={batch} n=64 k=1",
         f"program gw2 {gw} r={batch} n=64 k=64", f"program gb2 {gw} r={batch} n=64 k=1",
         f"program gw3 {gw} r={batch} n=3 k=64", f"program gb3 {gw} r={batch} n=3 k=1",
         f"program ls {gw} r={batch} n=3 k=3"]
    sizes = {n: int(np.prod(task.SHAPES[n])) for n in task.NAMES}
    for name, n in sizes.items():
        L.append(f"program a{name} {ad} n={n} bn={block(n, 128)} gs={1.0 / batch!r} b1={B1} b2={B2} eps={EPS}")
    f32 = " f32"
    L += [f"table f0.hin {p('x')} {nb}{f32}", f"table gw1.xt {p('xt')} {nb}{f32}", f"table fo.t {p('t')} {nb}{f32}"]
    L += [f"resident gb1.xt {p('ones')}{f32}", f"resident gb2.xt {p('ones')}{f32}", f"resident gb3.xt {p('ones')}{f32}",
          f"resident fo.eye {p('eye3')}{f32}", f"resident q1.eye {p('eye64')}{f32}", f"resident q0.eye {p('eye64')}{f32}"]
    owner = {"1": "f0", "2": "f1", "3": "fo"}
    for i in "123":
        L += [f"resident {owner[i]}.w {p('w' + i)}{f32}", f"resident {owner[i]}.bias {p('b' + i)}{f32}"]
        if i != "3":
            L.append(f"resident {owner[i]}.biasc {p('b' + i)}{f32}")
    L += [f"resident q1.w {p('w3')}{f32}", f"resident q0.w {p('w2')}{f32}"]
    for name in task.NAMES:
        L += [f"resident a{name}.w {p(name)}{f32}", f"resident a{name}.m {p('z' + name)}{f32}", f"resident a{name}.v {p('z' + name)}{f32}", f"table a{name}.s {p('sched')} {steps}{f32}"]
    L += ["link f0.h -> f1.hin", "link f1.h -> fo.hin", "link f0.h -> q0.h", "link f1.h -> q1.h",
          "link f0.ht -> gw2.xt", "link f1.ht -> gw3.xt",
          "link fo.e -> q1.dzn", "link q1.dz -> q0.dzn",
          "link q0.dzt -> gw1.dyt", "link q0.dzt -> gb1.dyt", "link q1.dzt -> gw2.dyt", "link q1.dzt -> gb2.dyt",
          "link fo.et -> gw3.dyt", "link fo.et -> gb3.dyt", "link fo.et -> ls.dyt", "link fo.et -> ls.xt"]
    for i in "123":
        L += [f"link gw{i}.dw -> aw{i}.grad", f"link gb{i}.dw -> ab{i}.grad"]
    for i in "123":
        for kind in "wb":
            n = kind + i
            L += [f"carry a{n}.wo -> a{n}.w", f"carry a{n}.mo -> a{n}.m", f"carry a{n}.vo -> a{n}.v", f"carry a{n}.wo -> {owner[i]}.{'w' if kind == 'w' else 'bias'}"]
            if kind == "b" and i != "3":
                L.append(f"carry a{n}.wo -> {owner[i]}.biasc")
    L += ["carry aw3.wo -> q1.w", "carry aw2.wo -> q0.w"]
    L += ["steps", f"answer ls.dw scale 0.5/{batch}"]
    if dump:
        L += [f"answer a{name}.wo" for name in task.NAMES]
    with open(p("session"), "w") as f:
        f.write("\n".join(L) + "\n")
    return p("session"), (mean, std)


def fold(flat32, mean, std):
    """The trained parameters in the standardised target space -> the network in metres: w3 and b3 scaled by the standard deviation, b3 shifted by the mean (exact in binary64, then rounded to f32)."""
    p = task.unflatten(flat32)
    p["w3"] = p["w3"] * std[:, None]
    p["b3"] = p["b3"] * std + mean
    return {k: v.astype(np.float32) for k, v in p.items()}


def reference(q, pos, init, batch, steps, lr=1e-3):
    """The same training in binary64 numpy: (the loss before each step, the final parameters in the standardised space)."""
    mean, std = pos.mean(axis=0), pos.std(axis=0)
    t_all = ((pos - mean) / std).astype(np.float32).astype(np.float64)
    x_all = q.astype(np.float32).astype(np.float64)
    nb = len(q) // batch
    P = [init[n].astype(np.float64) for n in task.NAMES]
    m = [np.zeros_like(a) for a in P]
    v = [np.zeros_like(a) for a in P]
    sched = schedule(steps, lr)
    losses = []
    for step in range(1, steps + 1):
        lo = ((step - 1) % nb) * batch
        x, t = x_all[lo:lo + batch], t_all[lo:lo + batch]
        w1, b1, w2, b2, w3, b3 = P
        h1 = np.tanh(x @ w1.T + b1)
        h2 = np.tanh(h1 @ w2.T + b2)
        e = h2 @ w3.T + b3 - t
        losses.append(0.5 * float((e * e).sum()) / batch)
        d2 = (e @ w3) * (1 - h2 * h2)
        d1 = (d2 @ w2) * (1 - h1 * h1)
        grads = [d1.T @ x, d1.sum(0), d2.T @ h1, d2.sum(0), e.T @ h2, e.sum(0)]
        rate, isq = sched[step - 1]
        for i, g in enumerate(grads):
            g = g / batch
            m[i] = m[i] + (g - m[i]) * (1.0 - B1)
            v[i] = v[i] * B2 + (1.0 - B2) * g * g
            P[i] = P[i] - rate * m[i] / (np.sqrt(v[i]) * isq + EPS)
    return losses, dict(zip(task.NAMES, P))


def train(session_bin, directory, n, steps, batch=250, lr=1e-3, chunk=500, log=print):
    """Trains G(n): n poses of task.train_set, `steps` Adam steps. Answers (parameters in metres as f32 dict, seconds of training, last loss)."""
    q, pos = task.train_set(n)
    s, (mean, std) = train_session(directory, q, pos, task.init_params(), batch, steps, lr)
    t0 = time.time()
    proc = subprocess.Popen([session_bin, s, "--target", "nvidia-ada"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    done, last, flat = 0, None, None
    while done < steps:
        k = min(chunk, steps - done)
        proc.stdin.write(f"{k}\n")
        proc.stdin.flush()
        vals = [float(v) for v in proc.stdout.readline().split()]
        done += k
        last = (vals[0] + vals[4] + vals[8])      # trace of e^T e / 2b: the loss of the last minibatch (standardised units)
        flat = np.array(vals[9:])
        log(f"  G({n}) step {done}/{steps}  minibatch loss {last:.5f}  {time.time() - t0:.0f} s")
    proc.stdin.close()
    proc.wait()
    return fold(flat, mean, std), time.time() - t0, last
