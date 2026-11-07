"""A session that holds the optimiser alone (docs/optimizer.md section 4): the gradient of every micro-step is a table, so what is tested is the clip, the groups, the
schedule and the accumulation, exactly, against torch. `write` answers the path of the session file; a request of R repetitions answers the clip coefficient and, for every
parameter, its weights, first and second moments.

    write(dir, params, grads, accum=1, steps=1, lr=..., wd={group: wd}, max_norm=None, gs=1.0, half=False)
    params: [(name, array, group)];  grads: {name: array [steps * accum, *shape]}, the gradient (scaled by 1/gs) of each micro-step
"""
import os

import numpy as np

import schedule as S

HERE = os.path.dirname(os.path.abspath(__file__))


def block(n, cap=128):
    return next(b for b in range(min(n, cap), 0, -1) if n % b == 0)


def put(path, values, dtype="<f4"):
    with open(path, "wb") as f:
        f.write(np.asarray(values, dtype=dtype).reshape(-1).tobytes())


def write(directory, params, grads, accum=1, steps=1, lr=None, wd=None, max_norm=None, gs=1.0, b1=0.9, b2=0.999, eps=1e-8, kdir=None, sc=None, mut=()):
    """`kdir`: the directory the kernels are read from (the mutants' copy). `sc`: the accumulator's scale (default 1/accum)."""
    os.makedirs(directory, exist_ok=True)
    kdir = kdir or HERE
    p = lambda name: os.path.join(directory, name)
    reps = steps * accum
    lr = lr or S.constant(1e-3)
    wd = wd or {}
    sc = (1.0 / accum) if sc is None else sc
    sc = 1.0 if "no_div" in mut else sc
    maxn = repr(float(max_norm)) if max_norm is not None else "1.0e30"
    L = []
    prog = lambda name, kern, **kw: L.append(f"program {name} {os.path.join(kdir, kern)} " + " ".join(f"{k}={v}" for k, v in kw.items()))
    put(p("keep"), [1.0] * reps if "keep_always" in mut else S.keep(steps, accum))  # `mut`: deliberate faults for the mutant tests (tests/optim.py)
    put(p("zero1"), [0.0])
    groups = sorted({g for _, _, g in params})
    for i, group in enumerate(groups):
        w_d = wd.get(groups[(i + 1) % len(groups)] if "wd_swap" in mut else group, 0.0)
        rows = S.rows(steps, accum, lr, w_d, b1, b2)
        if "dec_peak" in mut:  # the decay at the rate of the first step, not of the step
            rows = [[r[0], r[1], 1.0 - lr(1) * w_d, r[3], r[4]] for r in rows]
        if "apply_always" in mut:
            rows = [[r[0], r[1], r[2], 1.0, 0.0] for r in rows]
        put(p(f"sched_{group}"), rows)
    for name, w, group in params:
        n = int(np.asarray(w).size)
        g = np.asarray(grads[name], dtype=np.float64)
        assert g.shape[0] == reps and int(np.prod(g.shape[1:])) == n, name
        put(p(name), w)
        put(p(name + ".z"), np.zeros(n))
        put(p(name + ".g"), g)
        bn = block(n)
        prog("acc_" + name, "gacc.chx", n=n, bn=bn, sc=repr(sc))
        L.append(f"resident acc_{name}.acc {p(name + '.z')} f32")
        L.append(f"table acc_{name}.grad {p(name + '.g')} {reps} f32")
        L.append(f"table acc_{name}.k {p('keep')} {reps} f32")
        L.append(f"carry acc_{name}.o -> acc_{name}.acc")
    prev = None
    for name, w, group in params:
        n = int(np.asarray(w).size)
        prog("sq_" + name, "gradsq.chx", n=n, bn=block(n), gs=repr(gs))
        if prev is None:
            L.append(f"resident sq_{name}.acc {p('zero1')} f32")
        else:
            L.append(f"link sq_{prev}.o -> sq_{name}.acc")
        L.append(f"link acc_{name}.o -> sq_{name}.grad")
        prev = name
    prog("clip", "clip_scale.chx", maxn=maxn)
    L.append(f"link sq_{prev}.o -> clip.ss")
    for name, w, group in params:
        n = int(np.asarray(w).size)
        prog("a_" + name, "adamw_g.chx", n=n, bn=block(n), gs=repr(gs), b1=repr(b1), ob1=repr(1.0 - b1), b2=repr(b2), ob2=repr(1.0 - b2), eps=repr(eps))
        L.append(f"resident a_{name}.w {p(name)} f32")
        L.append(f"resident a_{name}.m {p(name + '.z')} f32")
        L.append(f"resident a_{name}.v {p(name + '.z')} f32")
        L.append(f"table a_{name}.s {p('sched_' + group)} {reps} f32")
        L.append(f"link acc_{name}.o -> a_{name}.grad")
        L.append(f"link clip.cs -> a_{name}.cs")
        for x in "wmv":
            L.append(f"carry a_{name}.{x}o -> a_{name}.{x}")
    L.append("steps")
    L.append("answer clip.cs")
    for name, w, group in params:
        for x in "wmv":
            L.append(f"answer a_{name}.{x}o")
    with open(p("session"), "w") as f:
        f.write("\n".join(L) + "\n")
    return p("session")


def parse(params, numbers):
    """The answer of a request: (clip coefficient, {name: (w, m, v)}) as arrays."""
    out, o = {}, 1
    for name, w, _ in params:
        shape = np.asarray(w).shape
        n = int(np.prod(shape))
        trio = []
        for _ in range(3):
            trio.append(np.array(numbers[o:o + n]).reshape(shape))
            o += n
        out[name] = tuple(trio)
    return numbers[0], out
