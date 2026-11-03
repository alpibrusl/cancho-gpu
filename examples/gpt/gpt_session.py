#!/usr/bin/env python3
"""A GPT trained with AdamW on the device (docs/resident.md section 15; the plan is docs/roadmap-models.md M1): pre-LayerNorm decoder blocks with causal multi-head attention and a GELU
feed-forward, character (or any small-vocabulary) tokens, mixed precision on the matrix unit: f32 master weights and Adam moments, f16 copies of the matrices, f16 activations and
gradients of the pre-activations, f32 accumulation and residual stream. gpt_ref.py is its specification (binary64 numpy, hand-written backward, checked by finite differences).

    python3 examples/gpt/gpt_session.py ... see write()

A request is a count of optimiser steps; the answer is the mean loss of the last step's batch (and, with `grads`, the raw gradients of the first step, for the tests)."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "examples", "train"))
sys.path.insert(0, HERE)
import gpt_ref as R  # noqa: E402

B1, B2, EPS = 0.9, 0.95, 1e-8
LOSS_SCALE = 16.0  # the gradient of the logits is scaled by this and the optimiser divides it out


def block(n, cap):
    return next(b for b in range(min(n, cap), 0, -1) if n % b == 0)


def put(path, values, dtype="<f4"):
    with open(path, "wb") as f:
        f.write(np.asarray(values, dtype=dtype).reshape(-1).tobytes())


def schedule(steps, lr_fn):
    """Per step: [lr_t / (1 - beta1^t), 1 / sqrt(1 - beta2^t)] (binary64): the learning-rate schedule is in the table."""
    return [(lr_fn(t) / (1.0 - B1 ** t), 1.0 / math.sqrt(1.0 - B2 ** t)) for t in range(1, steps + 1)]


def write(directory, c, tokens, targets, steps, lr=1e-3, wd=0.1, seed=1, grads=False, lr_fn=None, params=None, dump=False, tune=None, serve=False, save=None):
    """Writes the session. `tokens` and `targets`: int arrays [steps_data, N] (a batch for each step of the table, taken round and round). `c`: a gpt_ref.Config.
    Answers the path of the session file. With `serve`, a forward-only session for generation (docs/resident.md section 16): a request is the `n` token ids of a window and
    an `n`-long one-hot selector of the position to read, the answer is that position's `vp` logits; `tokens` and `targets` are not used.
    With `save` (a path), the f32 master weights are `save` directives to that one file, in the order of `names(c)`, written on the request `checkpoint` (docs/resident.md section 19):
    read them back with `read_checkpoint`."""
    os.makedirs(directory, exist_ok=True)
    p = lambda name: os.path.join(directory, name)
    n, d, h, dh, t, nb, ll, vp, v = c.n, c.d, c.h, c.dh, c.t, c.b, c.layers, c.vp, c.v
    assert dh % 32 == 0 and d % 64 == 0 and t % 64 == 0 and n % 64 == 0 and vp % 64 == 0 and (vp + t) % 64 == 0 and t in (64, 128)
    nsteps = len(tokens)
    P = params or R.init_params(c, seed)
    bm = 1 if d >= 640 else 2 if d >= 384 else 4 if d >= 256 else 8  # rows an elementwise threadgroup takes: its staged tiles (the 4 d wide ones) must fit in 32 KiB
    lr_fn = lr_fn or (lambda s: lr)
    # constants and data
    put(p("ones"), np.ones(n))
    put(p("zeros_nd"), np.zeros((n, d)))
    put(p("pos"), np.tile(np.arange(t), nb) + vp)
    put(p("jrow_oh"), np.arange(vp + t))
    put(p("jrow_v"), np.arange(vp))
    put(p("maskv"), np.where(np.arange(vp) < v, 0.0, -1e9))
    put(p("mask"), np.triu(np.full((t, t), -1e9), 1))
    put(p("tok"), np.asarray(tokens, dtype=np.float32))
    put(p("tgt"), np.asarray(targets, dtype=np.float32))
    put(p("sched"), np.array(schedule(steps, lr_fn)).reshape(-1))
    # parameters: every matrix as an f32 master and an f16 copy, every vector as f32
    mats = ["emb", "wlm"] + [f"{k}{l}" for l in range(ll) for k in ("wqkv", "wo", "w1", "w2")]
    vecs = ["lnfg", "lnfb"] + [f"{k}{l}" for l in range(ll) for k in ("ln1g", "ln1b", "bqkv", "bo", "ln2g", "ln2b", "b1", "b2")]
    for name in mats + vecs:
        put(p(name), P[name])
        if not serve:
            put(p("z" + name), np.zeros_like(P[name]))
    for name in mats:
        put(p(name + ".h"), P[name], "<f2")
    # the kernels, with the schedules this shape needs
    # the threadgroup shapes measured best on an M4 Max for d = 256, context 128, 4096 tokens a step (28.0 -> 15.5 ms a step, docs/resident.md section 15): attention in
    # 32-row tiles, and the weight gradients (a reduction over all the tokens) in 32 x 32 tiles, so that there are enough threadgroups to fill the GPU
    tune = {"att": dict(bm=32), "dq": dict(bm=32), "tn": lambda m_, n_: dict(threads=128, warps="2 2", bm=32, bn=32, bk=32), **(tune or {})}

    def kernel(name, tag="", **sched):
        src = next(q for q in (os.path.join(HERE, name), os.path.join(ROOT, "examples", "train", name), os.path.join(ROOT, "tests", "chx", name), os.path.join(ROOT, "tests", "chx", "ported", name)) if os.path.exists(q))
        if not sched:
            return src
        text = open(src).read()
        out = []
        for line in text.split("\n"):
            if line.startswith("schedule ") and "apple-m-series" in line:
                head = line.split("{")[0]
                line = head + "{ " + " ".join(f"{k} {v_}" for k, v_ in sched.items()) + " }"
            out.append(line)
        dst = p(name.replace(".chx", tag + ".chx"))
        with open(dst, "w") as f:
            f.write("\n".join(out))
        return dst

    L = [f"# GPT: {ll} layers, d {d}, {h} heads, context {t}, batch {nb}, vocabulary {v} (padded to {vp}), AdamW {lr}"]
    prog = lambda name, kern, **kw: L.append(f"program {name} {kern} " + " ".join(f"{k}={v_}" for k, v_ in kw.items()))
    res = lambda target, path, flag="f32": L.append(f"resident {target} {path}" + ("" if flag == "" else f" {flag}"))
    link = lambda a, b: L.append(f"link {a} -> {b}")
    carry = lambda a, b: L.append(f"carry {a} -> {b}")
    tab = lambda target, path, blocks, flag="f32": L.append(f"table {target} {path} {blocks}" + ("" if flag == "" else f" {flag}"))
    att = dict(threads=128, warps="2 2", bm=64, bn=t, bk=32)
    att.update(tune.get("att", {}))
    sc_k = kernel("att_scores.chx", **att)
    dp_k = kernel("att_dp.chx", **att)
    pv_k = kernel("att_pv.chx", threads=128, warps="2 2", bm=64, bn=dh, bk=32)
    dq = dict(threads=128, warps="2 2", bm=64, bk=32)
    dq.update(tune.get("dq", {}))
    dq_k = kernel("att_dqkv.chx", **dq)
    gemm = kernel("gemm_mma.chx", **tune["nt"]) if "nt" in tune else kernel("gemm_mma.chx")
    gnn = kernel("gemm_mma_nn.chx", **tune["nn"]) if "nn" in tune else kernel("gemm_mma_nn.chx")
    def gtn_for(m_, n_):
        sch = tune.get("tn", lambda m__, n__: None)(m_, n_)
        return kernel("gemm_mma_tn.chx", tag=f"_{m_}_{n_}", **sch) if sch else kernel("gemm_mma_tn.chx")
    k = lambda f: kernel(f)

    # ---- forward
    prog("oh", k("onehot.chx"), r=n, c=vp + t, bm=bm, bc=block(vp + t, 128))
    if serve:
        L.append("input oh.tok")
    else:
        tab("oh.tok", p("tok"), nsteps)
    res("oh.pos", p("pos"))
    res("oh.jrow", p("jrow_oh"))
    prog("ex", gnn, m=n, n=d, k=vp + t)
    link("oh.oh", "ex.a")
    res("ex.w", p("emb.h"), "f16")
    x_in = "ex.y"
    for l in range(ll):
        s = str(l)
        prog("ln1_" + s, k("ln_fwd.chx"), r=n, c=d, bm=bm)
        link(x_in, f"ln1_{s}.x")
        res(f"ln1_{s}.gamma", p(f"ln1g{l}"))
        res(f"ln1_{s}.beta", p(f"ln1b{l}"))
        prog("qkv_" + s, gemm, m=n, n=3 * d, k=d)
        link(f"ln1_{s}.y", f"qkv_{s}.a")
        res(f"qkv_{s}.w", p(f"wqkv{l}.h"), "f16")
        prog("qb_" + s, k("bias_h.chx"), r=n, c=3 * d, bm=bm)
        link(f"qkv_{s}.y", f"qb_{s}.z")
        res(f"qb_{s}.bias", p(f"bqkv{l}"))
        prog("sc_" + s, sc_k, nb=nb, nh=h, t=t, d=d, dh=dh)
        link(f"qb_{s}.y", f"sc_{s}.qkv")
        prog("sm_" + s, k("att_softmax.chx"), nb=nb, nh=h, t=t, bm=bm, scale=1.0 / math.sqrt(dh))
        link(f"sc_{s}.s", f"sm_{s}.s")
        res(f"sm_{s}.mask", p("mask"))
        prog("pv_" + s, pv_k, nb=nb, nh=h, t=t, d=d, dh=dh)
        link(f"sm_{s}.p", f"pv_{s}.p")
        link(f"qb_{s}.y", f"pv_{s}.qkv")
        prog("oc_" + s, k("cast_h.chx"), r=n, c=d, bm=bm)
        link(f"pv_{s}.o", f"oc_{s}.x")
        prog("wo_" + s, gemm, m=n, n=d, k=d)
        link(f"oc_{s}.y", f"wo_{s}.a")
        res(f"wo_{s}.w", p(f"wo{l}.h"), "f16")
        prog("r1_" + s, k("res_bias.chx"), r=n, c=d, bm=bm)
        link(f"wo_{s}.y", f"r1_{s}.z")
        res(f"r1_{s}.bias", p(f"bo{l}"))
        link(x_in, f"r1_{s}.res")
        prog("ln2_" + s, k("ln_fwd.chx"), r=n, c=d, bm=bm)
        link(f"r1_{s}.y", f"ln2_{s}.x")
        res(f"ln2_{s}.gamma", p(f"ln2g{l}"))
        res(f"ln2_{s}.beta", p(f"ln2b{l}"))
        prog("f1_" + s, gemm, m=n, n=4 * d, k=d)
        link(f"ln2_{s}.y", f"f1_{s}.a")
        res(f"f1_{s}.w", p(f"w1{l}.h"), "f16")
        prog("gb_" + s, k("gelu_bias.chx"), r=n, c=4 * d, bm=bm)
        link(f"f1_{s}.y", f"gb_{s}.z")
        res(f"gb_{s}.bias", p(f"b1{l}"))
        prog("f2_" + s, gemm, m=n, n=d, k=4 * d)
        link(f"gb_{s}.h", f"f2_{s}.a")
        res(f"f2_{s}.w", p(f"w2{l}.h"), "f16")
        prog("r2_" + s, k("res_bias.chx"), r=n, c=d, bm=bm)
        link(f"f2_{s}.y", f"r2_{s}.z")
        res(f"r2_{s}.bias", p(f"b2{l}"))
        link(f"r1_{s}.y", f"r2_{s}.res")
        x_in = f"r2_{s}.y"
    xs = [None] * (ll + 1)  # the residual stream entering each layer, and the last
    xs[0] = "ex.y"
    for l in range(ll):
        xs[l + 1] = f"r2_{l}.y"
    prog("lnf", k("ln_fwd.chx"), r=n, c=d, bm=bm)
    link(xs[ll], "lnf.x")
    res("lnf.gamma", p("lnfg"))
    res("lnf.beta", p("lnfb"))
    prog("lm", gemm, m=n, n=vp, k=d)
    link("lnf.y", "lm.a")
    res("lm.w", p("wlm.h"), "f16")
    if serve:
        prog("sel", k("row_select.chx"), b=n, k=vp, bk=block(vp, 32))
        link("lm.y", "sel.x")
        L.append("input sel.v")
        L.append("answer sel.out")
        with open(p("session"), "w") as f:
            f.write("\n".join(L) + "\n")
        return p("session")
    prog("ce", k("ce.chx"), r=n, c=vp, bm=1 if vp >= 1024 else bm, scale=LOSS_SCALE)
    link("lm.y", "ce.lg")
    tab("ce.tgt", p("tgt"), nsteps)
    res("ce.jrow", p("jrow_v"))
    res("ce.maskv", p("maskv"))
    prog("ls", kernel("bwd_dw.chx"), r=n, n=1, k=1)
    link("ce.loss", "ls.dyt")
    res("ls.xt", p("ones"))

    # ---- backward
    def tn(name, a, b_, m, nn_, kk, out_grad=None):
        prog(name, gtn_for(m, nn_), m=m, n=nn_, k=kk)
        link(a, f"{name}.a")
        link(b_, f"{name}.w")

    def nnp(name, a, w_res, m, nn_, kk):
        prog(name, gnn, m=m, n=nn_, k=kk)
        link(a, f"{name}.a")
        if w_res:
            res(f"{name}.w", w_res[0], "f16")

    def colsum(name, x, cols):
        prog(name, k("rowvec_mm.chx"), b=n, k=cols, bk=block(cols, 32))
        res(f"{name}.v", p("ones"))
        link(x, f"{name}.x")

    def lnb(name, dy, x, gamma_file, dres, cols=d):
        prog(name, k("ln_bwd.chx"), r=n, c=d, bm=bm)
        link(dy, f"{name}.dy")
        link(x, f"{name}.x")
        res(f"{name}.gamma", gamma_file)
        if dres is None:
            res(f"{name}.dres", p("zeros_nd"))
        else:
            link(dres, f"{name}.dres")
        colsum(name + "_g", f"{name}.pg", d)
        colsum(name + "_b", f"{name}.dyh", d)

    def castn(name, src, cols):
        prog(name, k("cast_h.chx"), r=n, c=cols, bm=bm)
        link(src, f"{name}.x")

    tn("gwlm", "ce.dl", "lnf.y", vp, d, n)
    nnp("dlf", "ce.dl", [p("wlm.h")], n, d, vp)
    lnb("lnfb", "dlf.y", xs[ll], p("lnfg"), None)
    dx = "lnfb.dx"
    for l in reversed(range(ll)):
        s = str(l)
        castn("dxc_" + s, dx, d)
        tn("gw2_" + s, f"dxc_{s}.y", f"gb_{s}.h", d, 4 * d, n)
        colsum("gb2_" + s, f"dxc_{s}.y", d)
        nnp("dh_" + s, f"dxc_{s}.y", [p(f"w2{l}.h")], n, 4 * d, d)
        prog("gbk_" + s, k("gelu_bwd.chx"), r=n, c=4 * d, bm=bm)
        link(f"dh_{s}.y", f"gbk_{s}.dh")
        link(f"gb_{s}.u", f"gbk_{s}.u")
        tn("gw1_" + s, f"gbk_{s}.du", f"ln2_{s}.y", 4 * d, d, n)
        colsum("gb1_" + s, f"gbk_{s}.du", 4 * d)
        nnp("da2_" + s, f"gbk_{s}.du", [p(f"w1{l}.h")], n, d, 4 * d)
        lnb("ln2b_" + s, f"da2_{s}.y", f"r1_{s}.y", p(f"ln2g{l}"), dx)
        dx1 = f"ln2b_{s}.dx"
        castn("dx1c_" + s, dx1, d)
        tn("gwo_" + s, f"dx1c_{s}.y", f"oc_{s}.y", d, d, n)
        colsum("gbo_" + s, f"dx1c_{s}.y", d)
        nnp("do_" + s, f"dx1c_{s}.y", [p(f"wo{l}.h")], n, d, d)
        castn("doc_" + s, f"do_{s}.y", d)
        prog("dp_" + s, dp_k, nb=nb, nh=h, t=t, d=d, dh=dh)
        link(f"doc_{s}.y", f"dp_{s}.dout")
        link(f"qb_{s}.y", f"dp_{s}.qkv")
        prog("ds_" + s, k("att_dsoft.chx"), nb=nb, nh=h, t=t, bm=bm, scale=1.0 / math.sqrt(dh))
        link(f"dp_{s}.dp", f"ds_{s}.dp")
        link(f"sm_{s}.p", f"ds_{s}.p")
        prog("dq_" + s, dq_k, nb=nb, nh=h, t=t, d=d, dh=dh)
        link(f"ds_{s}.ds", f"dq_{s}.ds")
        link(f"sm_{s}.p", f"dq_{s}.p")
        link(f"doc_{s}.y", f"dq_{s}.dout")
        link(f"qb_{s}.y", f"dq_{s}.qkv")
        castn("dqc_" + s, f"dq_{s}.dqkv", 3 * d)
        tn("gwqkv_" + s, f"dqc_{s}.y", f"ln1_{s}.y", 3 * d, d, n)
        colsum("gbqkv_" + s, f"dqc_{s}.y", 3 * d)
        nnp("da1_" + s, f"dqc_{s}.y", [p(f"wqkv{l}.h")], n, d, 3 * d)
        lnb("ln1b_" + s, f"da1_{s}.y", xs[l], p(f"ln1g{l}"), dx1)
        dx = f"ln1b_{s}.dx"
    castn("dx0c", dx, d)
    tn("gemb", "oh.oh", "dx0c.y", vp + t, d, n)

    # ---- the optimiser
    grad_of = {"emb": "gemb.y", "wlm": "gwlm.y", "lnfg": "lnfb_g.out", "lnfb": "lnfb_b.out"}
    for l in range(ll):
        s = str(l)
        grad_of.update({f"wqkv{l}": f"gwqkv_{s}.y", f"bqkv{l}": f"gbqkv_{s}.out", f"wo{l}": f"gwo_{s}.y", f"bo{l}": f"gbo_{s}.out",
                        f"w1{l}": f"gw1_{s}.y", f"b1{l}": f"gb1_{s}.out", f"w2{l}": f"gw2_{s}.y", f"b2{l}": f"gb2_{s}.out",
                        f"ln1g{l}": f"ln1b_{s}_g.out", f"ln1b{l}": f"ln1b_{s}_b.out", f"ln2g{l}": f"ln2b_{s}_g.out", f"ln2b{l}": f"ln2b_{s}_b.out"})
    f16_users = {"emb": ["ex"], "wlm": ["lm", "dlf"]}
    for l in range(ll):
        s = str(l)
        f16_users.update({f"wqkv{l}": [f"qkv_{s}", f"da1_{s}"], f"wo{l}": [f"wo_{s}", f"do_{s}"], f"w1{l}": [f"f1_{s}", f"da2_{s}"], f"w2{l}": [f"f2_{s}", f"dh_{s}"]})
    f32_users = {"lnfg": ["lnf.gamma", "lnfb.gamma"], "lnfb": ["lnf.beta"]}
    for l in range(ll):
        s = str(l)
        f32_users.update({f"ln1g{l}": [f"ln1_{s}.gamma", f"ln1b_{s}.gamma"], f"ln1b{l}": [f"ln1_{s}.beta"], f"ln2g{l}": [f"ln2_{s}.gamma", f"ln2b_{s}.gamma"], f"ln2b{l}": [f"ln2_{s}.beta"],
                          f"bqkv{l}": [f"qb_{s}.bias"], f"bo{l}": [f"r1_{s}.bias"], f"b1{l}": [f"gb_{s}.bias"], f"b2{l}": [f"r2_{s}.bias"]})
    sizes = {name: int(np.asarray(P[name]).size) for name in mats + vecs}
    common = dict(gs=repr(1.0 / (n * LOSS_SCALE)), b1=B1, b2=B2, eps=EPS)
    for name in mats:
        prog("a_" + name, kernel("adamw_h.chx"), n=sizes[name], bn=block(sizes[name], 128), lwd=repr(lr * wd), **common)
    for name in vecs:
        prog("a_" + name, kernel("adam.chx", ) if False else os.path.join(ROOT, "examples", "train", "adam.chx"), n=sizes[name], bn=block(sizes[name], 128), **common)
    for name in mats + vecs:
        res(f"a_{name}.w", p(name))
        res(f"a_{name}.m", p("z" + name))
        res(f"a_{name}.v", p("z" + name))
        tab(f"a_{name}.s", p("sched"), steps)
        link(grad_of[name], f"a_{name}.grad")
        carry(f"a_{name}.wo", f"a_{name}.w")
        carry(f"a_{name}.mo", f"a_{name}.m")
        carry(f"a_{name}.vo", f"a_{name}.v")
    for name in mats:
        for u in f16_users[name]:
            carry(f"a_{name}.wh", f"{u}.w")
    for name in vecs:
        for u in f32_users[name]:
            carry(f"a_{name}.wo", u)
    L.append("steps")
    L.append(f"answer ls.dw scale 1/{n}")
    if grads:
        for name in mats + vecs:
            L.append(f"answer {grad_of[name]}")
    if dump:
        for name in mats + vecs:
            L.append(f"answer a_{name}.wo")
    if save:
        for name in mats + vecs:
            L.append(f"save a_{name}.wo {save}")
    with open(p("session"), "w") as f:
        f.write("\n".join(L) + "\n")
    return p("session")


def names(c):
    mats = ["emb", "wlm"] + [f"{k}{l}" for l in range(c.layers) for k in ("wqkv", "wo", "w1", "w2")]
    vecs = ["lnfg", "lnfb"] + [f"{k}{l}" for l in range(c.layers) for k in ("ln1g", "ln1b", "bqkv", "bo", "ln2g", "ln2b", "b1", "b2")]
    return mats, vecs


def parse_dump(c, params, numbers):
    """The parameters from the numbers a `dump` session answers after the loss."""
    mats, vecs = names(c)
    out, o = {}, 1
    for name in mats + vecs:
        shape = params[name].shape
        n = int(np.prod(shape))
        out[name] = np.array(numbers[o:o + n]).reshape(shape)
        o += n
    return out


def shapes(c):
    """The shape of every parameter, as gpt_ref.init_params makes them."""
    d = c.d
    out = {"emb": (c.vp + c.t, d), "wlm": (c.vp, d), "lnfg": (d,), "lnfb": (d,)}
    for l in range(c.layers):
        out.update({f"ln1g{l}": (d,), f"ln1b{l}": (d,), f"wqkv{l}": (3 * d, d), f"bqkv{l}": (3 * d,), f"wo{l}": (d, d), f"bo{l}": (d,),
                    f"ln2g{l}": (d,), f"ln2b{l}": (d,), f"w1{l}": (4 * d, d), f"b1{l}": (4 * d,), f"w2{l}": (d, 4 * d), f"b2{l}": (d,)})
    return out


def read_checkpoint(c, path):
    """The parameters (binary64 arrays) from the file `save` wrote: little-endian float32 words, every tensor in the order of `names(c)`, nothing else."""
    mats, vecs = names(c)
    sh = shapes(c)
    flat = np.fromfile(path, dtype="<f4")
    want = sum(int(np.prod(sh[k])) for k in mats + vecs)
    if flat.size != want:
        raise SystemExit(f"{path}: {flat.size} words, the model has {want}")
    out, o = {}, 0
    for k in mats + vecs:
        n = int(np.prod(sh[k]))
        out[k] = flat[o:o + n].astype(np.float64).reshape(sh[k])
        o += n
    return out


def checkpoint(proc, c):
    """Ask a session that has `save` directives for a checkpoint: the bytes written, one number for the file."""
    proc.stdin.write("checkpoint\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise SystemExit("cancho session: " + proc.stderr.read()[:300])
    return [int(v) for v in line.split()]
