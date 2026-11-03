#!/usr/bin/env python3
"""A ResNet-style CIFAR network trained with AdamW on the device (docs/resident.md section 18; the plan is docs/roadmap-models.md M4). cnn_ref.py is its specification (binary64
numpy, hand-written backward, finite differences). The data lives in a padded layout: every image has a border of zeros, the images are flattened to one [n (h+2)^2, c] matrix, and g
zero rows precede and follow it, so that a 3x3 convolution is nine products of that matrix, shifted by (dy-1)(h+2) + (dx-1) rows, with that tap's slice of the weights (conv3.chx).
Mixed precision as the GPT's: f32 master weights and Adam moments, f16 copies for the matrix unit, f16 activations and gradients of the pre-activations, f32 accumulation, BatchNorm
statistics and residual gradients.

A request is a count of optimiser steps; the answer is the mean loss of the last step's batch (and, with `grads`, the raw gradients of the first step; with `dump`, every parameter and
the running statistics, for the tests and the curves)."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import cnn_ref as R  # noqa: E402

B1, B2, EPS = 0.9, 0.999, 1e-8
LOSS_SCALE = 16.0  # the gradient of the logits is scaled by this and the optimiser divides it out
BN_EPS, BN_MOM = 1e-5, 0.1
G = 64  # guard rows on each side of a padded matrix (at least h + 3 for the widest stage)
VP = 64  # the classifier's output rows: the classes padded with masked logits, as the GPT's vocabulary


def block(n, cap):
    return next(b for b in range(min(n, cap), 0, -1) if n % b == 0)


def pick(n, cap, mult):
    """The largest divisor of n that is at most cap and a multiple of mult."""
    return next(b for b in range(min(n, cap), 0, -1) if n % b == 0 and b % mult == 0)


def put(path, values, dtype="<f4"):
    with open(path, "wb") as f:
        f.write(np.asarray(values, dtype=dtype).reshape(-1).tobytes())


def schedule(steps, lr_fn):
    """Per step: [lr_t / (1 - beta1^t), 1 / sqrt(1 - beta2^t)] (binary64): the learning-rate schedule is in the table."""
    return [(lr_fn(t) / (1.0 - B1 ** t), 1.0 / math.sqrt(1.0 - B2 ** t)) for t in range(1, steps + 1)]


def cin_padded(c):
    return -(-c.cin // 8) * 8


def names(c):
    return R.param_names(c)


def sizes(c):
    return {"w_" + n: (co, 3, 3, ci if n != "stem" else cin_padded(c)) for n, ci, co in c.layers()}


def pad_params(c, P):
    """The reference's parameters in the session's layout: the stem's input channels padded with zeros, every weight as [cout, 9 cin], the classifier padded to VP rows."""
    out = {}
    for n, ci, co in c.layers():
        w = P["w_" + n]
        if n == "stem":
            w = np.concatenate([w, np.zeros(w.shape[:3] + (cin_padded(c) - ci,))], axis=3)
        out["w_" + n] = w.reshape(co, -1)
        out["g_" + n], out["b_" + n] = P["g_" + n], P["b_" + n]
    out["wfc"] = np.concatenate([P["wfc"], np.zeros((VP - c.classes, P["wfc"].shape[1]))], axis=0)
    return out


def input_tables(c, images, directory):
    """images [S, N, h, h, cin] floats: every batch as two blocks of N/2 images, [N/2 h h, cin] f16 (the two halves of the batch), written to xa and xb."""
    half = c.n // 2
    for name, part in (("xa", images[:, :half]), ("xb", images[:, half:])):
        with open(os.path.join(directory, name), "wb") as f:
            for batch in part:
                f.write(batch.reshape(half * c.h * c.h, c.cin).astype("<f2").tobytes())


def write(directory, c, images, labels, steps, lr=2e-3, wd=0.01, seed=1, grads=False, lr_fn=None, params=None, dump=False, tune=None):
    """Writes the session and answers its path. `images`: [S, N, h, h, cin] (S batches, taken round and round); `labels`: [S, N] ints. `c`: a cnn_ref.Config."""
    os.makedirs(directory, exist_ok=True)
    p = lambda name: os.path.join(directory, name)
    n, nsteps = c.n, len(images)
    assert c.h + 3 <= G and images.shape[1:] == (n, c.h, c.h, c.cin)
    P = pad_params(c, params or R.init_params(c, seed))
    mats, vecs = names(c)
    layers = c.layers()
    lr_fn = lr_fn or (lambda s: lr)
    tune = tune or {}
    hs = [c.size_at(s) for s in range(len(c.widths))]
    rps = [n * (h + 2) * (h + 2) for h in hs]
    for r_ in rps:
        assert r_ % 16 == 0, "the batch must make every stage's padded rows a multiple of 16"
    # constants and data
    assert n % 2 == 0
    input_tables(c, images, directory)
    cmat = np.zeros((c.cin, cin_padded(c)))
    cmat[:, : c.cin] = np.eye(c.cin)
    put(p("cmat"), cmat)
    em = np.zeros((c.h + 2, c.h))
    em[np.arange(1, c.h + 1), np.arange(c.h)] = 1.0
    put(p("em"), em)
    put(p("tgt"), np.asarray(labels, dtype=np.float32))
    put(p("sched"), np.array(schedule(steps, lr_fn)).reshape(-1))
    put(p("ones"), np.ones(n))
    put(p("jrow_v"), np.arange(VP))
    put(p("maskv"), np.where(np.arange(VP) < c.classes, 0.0, -1e9))
    for s, h in enumerate(hs):
        hp = h + 2
        valid = np.zeros((hp, hp))
        valid[1:-1, 1:-1] = 1.0
        mk = np.tile(valid.reshape(-1), n)
        put(p(f"mk{s}"), mk)  # [rp, 1] and [1, rp] are the same bytes
        put(p(f"gv{s}"), valid.reshape(-1) / (h * h))
        if s > 0:  # the pooling from stage s-1 (h2 = 2h wide) to this one: pm [h+2, 2h+2], pt its transpose
            pm = np.zeros((hp, 2 * h + 2))
            for x in range(h):
                pm[x + 1, 2 * x + 1] = pm[x + 1, 2 * x + 2] = 0.25
            put(p(f"pm{s}"), pm)
            put(p(f"pt{s}"), pm.T)
    for name in mats + vecs:
        put(p(name), P[name])
        put(p("z" + name), np.zeros_like(P[name]))
    for name in mats:
        put(p(name + ".h"), P[name], "<f2")
    for nm, ci, co in layers:
        put(p(f"rm_{nm}"), np.zeros(co))
        put(p(f"rv_{nm}"), np.ones(co))

    def kernel(name, tag="", **sched):
        src = next(q for q in (os.path.join(HERE, name), os.path.join(ROOT, "examples", "gpt", name), os.path.join(ROOT, "examples", "train", name),
                               os.path.join(ROOT, "tests", "chx", name), os.path.join(ROOT, "tests", "chx", "ported", name)) if os.path.exists(q))
        if not sched:
            return src
        out = []
        for line in open(src).read().split("\n"):
            if line.startswith("schedule ") and "apple-m-series" in line:
                line = line.split("{")[0] + "{ " + " ".join(f"{k} {v}" for k, v in sched.items()) + " }"
            out.append(line)
        dst = p(name.replace(".chx", tag + ".chx"))
        with open(dst, "w") as f:
            f.write("\n".join(out))
        return dst

    L = [f"# CNN: widths {c.widths}, blocks {c.blocks}, image {c.h}, batch {n}, classes {c.classes} (padded to {VP}), AdamW {lr}"]
    prog = lambda name, kern, **kw: L.append(f"program {name} {kern} " + " ".join(f"{k}={v}" for k, v in kw.items()))
    res = lambda target, path, flag="f32": L.append(f"resident {target} {path}" + ("" if flag == "" else f" {flag}"))
    link = lambda a, b: L.append(f"link {a} -> {b}")
    carry = lambda a, b: L.append(f"carry {a} -> {b}")
    tab = lambda target, path, blocks, flag="f32": L.append(f"table {target} {path} {blocks}" + ("" if flag == "" else f" {flag}"))

    # ---- the per-layer facts
    info = {}
    for nm, ci, co in layers:
        s = 0 if nm == "stem" else int(nm[2:]) if nm.startswith("up") else int(nm[1])
        rp = rps[s]
        cinp = cin_padded(c) if nm == "stem" else ci
        rb = block(rp, max(8, 2048 // co))
        nb = rp // rb
        cb = block(nb, 256)
        bc = block(co, 32)
        bm_e = block(rp, max(1, 2048 // co))  # elementwise threadgroups: rows per instance
        bm_b = block(rp, max(1, 1024 // co))  # the backward's (more live tiles)
        cbm = pick(rp, 64, 16)
        info[nm] = dict(s=s, ci=ci, cin=cinp, co=co, rp=rp, wp=hs[s] + 2, rb=rb, nb=nb, cb=cb, bc=bc, bm_e=bm_e, bm_b=bm_b, cbm=cbm,
                        rs=pick(rp, 1024, 32), m=n * hs[s] * hs[s])
        put(p(f"ones_cb{cb}"), np.ones(cb))

    def conv_fwd(nm, xsrc):
        i = info[nm]
        t = tune.get("conv", {})
        k = kernel("conv3.chx", tag="_" + nm, **{**dict(threads=128, warps="2 2", bm=i["cbm"], bn=min(i["co"], 64), bk=min(i["cin"], 32)), **t})
        prog("cv_" + nm, k, rp=i["rp"], cin=i["cin"], cout=i["co"], wp=i["wp"], g=G)
        link(xsrc, "cv_" + nm + ".x")
        res("cv_" + nm + ".w", p("w_" + nm + ".h"), "f16")
        # the statistics
        prog("bp_" + nm, kernel("bn_part.chx"), rp=i["rp"], c=i["co"], rb=i["rb"])
        link("cv_" + nm + ".y", "bp_" + nm + ".z")
        res("bp_" + nm + ".v", p(f"mk{i['s']}"))
        prog("bf_" + nm, kernel("bn_fin.chx"), nb=i["nb"], c=i["co"], cb=i["cb"], bc=i["bc"], m=i["m"], eps=repr(BN_EPS), mom=repr(BN_MOM))
        link("bp_" + nm + ".s1", "bf_" + nm + ".s1")
        link("bp_" + nm + ".s2", "bf_" + nm + ".s2")
        res("bf_" + nm + ".ones", p(f"ones_cb{i['cb']}"))
        res("bf_" + nm + ".gamma", p("g_" + nm))
        res("bf_" + nm + ".beta", p("b_" + nm))
        res("bf_" + nm + ".rm", p("rm_" + nm))
        res("bf_" + nm + ".rv", p("rv_" + nm))
        carry("bf_" + nm + ".rmo", "bf_" + nm + ".rm")
        carry("bf_" + nm + ".rvo", "bf_" + nm + ".rv")

    def bn_apply(nm, resname=None):
        i = info[nm]
        prog("ap_" + nm, kernel("bn_apply_res.chx" if resname else "bn_apply.chx"), rp=i["rp"], c=i["co"], bm=i["bm_e"], g=G)
        link("cv_" + nm + ".y", "ap_" + nm + ".z")
        link("bf_" + nm + ".sc", "ap_" + nm + ".sc")
        link("bf_" + nm + ".sh", "ap_" + nm + ".sh")
        res("ap_" + nm + ".mk", p(f"mk{i['s']}"))
        if resname:
            link(resname, "ap_" + nm + ".res")
        return "ap_" + nm + ".a"

    # ---- forward
    prog("pk", kernel("pack.chx"), n2=n // 2, h=c.h, cin=c.cin, cp=cin_padded(c), g=G)
    tab("pk.xa", p("xa"), nsteps, "f16")
    tab("pk.xb", p("xb"), nsteps, "f16")
    res("pk.cmat", p("cmat"))
    res("pk.em", p("em"))
    act = "pk.o"
    acts = {}  # layer -> (input activation, residual activation or None)
    for s in range(len(c.widths)):
        if s > 0:
            prog(f"pool{s}", kernel("pool_fwd.chx"), n=n, h=hs[s - 1], c=c.widths[s - 1], g=G)
            link(act, f"pool{s}.a")
            res(f"pool{s}.pm", p(f"pm{s}"))
            act = f"pool{s}.o"
            conv_fwd(f"up{s}", act)
            acts[f"up{s}"] = (act, None)
            act = bn_apply(f"up{s}")
        elif True:
            conv_fwd("stem", "pk.o")
            acts["stem"] = ("pk.o", None)
            act = bn_apply("stem")
        for b in range(c.blocks[s]):
            a0 = act
            conv_fwd(f"s{s}b{b}a", a0)
            acts[f"s{s}b{b}a"] = (a0, None)
            u = bn_apply(f"s{s}b{b}a")
            conv_fwd(f"s{s}b{b}b", u)
            acts[f"s{s}b{b}b"] = (u, a0)
            act = bn_apply(f"s{s}b{b}b", a0)
    sl = len(c.widths) - 1
    cl = c.widths[-1]
    rcg = block((hs[sl] + 2) ** 2, 20)
    prog("gap", kernel("gap_fwd.chx"), n=n, h=hs[sl], c=cl, g=G, rc=rcg)
    link(act, "gap.a")
    res("gap.vm", p(f"gv{sl}"))
    prog("fc", kernel("gemm_mma.chx", threads=128, warps="2 2", bm=min(n, 64), bn=VP, bk=min(cl, 32)), m=n, n=VP, k=cl)
    link("gap.f", "fc.a")
    res("fc.w", p("wfc.h"), "f16")
    prog("ce", kernel("ce.chx"), r=n, c=VP, bm=block(n, 8), scale=LOSS_SCALE)
    link("fc.y", "ce.lg")
    tab("ce.tgt", p("tgt"), nsteps)
    res("ce.jrow", p("jrow_v"))
    res("ce.maskv", p("maskv"))
    prog("ls", kernel("bwd_dw.chx"), r=n, n=1, k=1)
    link("ce.loss", "ls.dyt")
    res("ls.xt", p("ones"))

    # ---- backward
    prog("gfc", kernel("gemm_mma_tn.chx", threads=128, warps="2 2", bm=VP, bn=min(cl, 64), bk=min(n, 32)), m=VP, n=cl, k=n)
    link("ce.dl", "gfc.a")
    link("gap.f", "gfc.w")
    prog("dfeat", kernel("gemm_mma_nn.chx", threads=128, warps="2 2", bm=min(n, 64), bn=min(cl, 64), bk=VP), m=n, n=cl, k=VP)
    link("ce.dl", "dfeat.a")
    res("dfeat.w", p("wfc.h"), "f16")
    prog("gapb", kernel("gap_bwd.chx"), n=n, h=hs[sl], c=cl, rc=rcg)
    link("dfeat.y", "gapb.gf")
    res("gapb.mc", p(f"gv{sl}"))
    gout = "gapb.gi"
    grad_of = {"wfc": "gfc.y"}

    def bwd_layer(nm, gout):
        """The backward of one convolution-BatchNorm layer given the gradient `gout` of its output; answers the name of the f32 gradient of its input (None for the stem)."""
        i = info[nm]
        xin, rsd = acts[nm]
        co = i["co"]
        rs = bool(rsd)
        prog("bq_" + nm, kernel("bn_bwd_part_res.chx" if rs else "bn_bwd_part.chx"), rp=i["rp"], c=co, rb=i["rb"], **({"g": G} if rs else {}))
        link(gout, "bq_" + nm + ".gr")
        link("cv_" + nm + ".y", "bq_" + nm + ".z")
        for k_ in ("sc", "sh", "mean", "istd"):
            link(f"bf_{nm}.{k_}", f"bq_{nm}.{k_}")
        res("bq_" + nm + ".v", p(f"mk{i['s']}"))
        if rs:
            link(rsd, "bq_" + nm + ".res")
        prog("bg_" + nm, kernel("bn_bwd_fin.chx"), nb=i["nb"], c=co, cb=i["cb"], bc=i["bc"])
        link("bq_" + nm + ".p1", "bg_" + nm + ".p1")
        link("bq_" + nm + ".p2", "bg_" + nm + ".p2")
        res("bg_" + nm + ".ones", p(f"ones_cb{i['cb']}"))
        prog("ba_" + nm, kernel("bn_bwd_apply_res.chx" if rs else "bn_bwd_apply.chx"), rp=i["rp"], c=co, bm=i["bm_b"], g=G, m=i["m"])
        link(gout, "ba_" + nm + ".gr")
        link("cv_" + nm + ".y", "ba_" + nm + ".z")
        for k_ in ("sc", "sh", "mean", "istd"):
            link(f"bf_{nm}.{k_}", f"ba_{nm}.{k_}")
        res("ba_" + nm + ".gamma", p("g_" + nm))
        link("bg_" + nm + ".dbeta", "ba_" + nm + ".dbeta")
        link("bg_" + nm + ".dgamma", "ba_" + nm + ".dgamma")
        res("ba_" + nm + ".mk", p(f"mk{i['s']}"))
        if rs:
            link(rsd, "ba_" + nm + ".res")
        # the weight gradient
        t = tune.get("dw", {})
        tm, tn = min(co, 64), min(i["cin"], 64)
        wr, wc = (2 if tm % 16 == 0 else 1), (2 if tn % 16 == 0 else 1)
        prog("dw_" + nm, kernel("conv3_dw.chx", tag="_" + nm, **{**dict(threads=32 * wr * wc, warps=f"{wr} {wc}", bm=tm, bn=tn, bk=32), **t}), rp=i["rp"], cin=i["cin"], cout=co, wp=i["wp"], g=G, rs=i["rs"])
        link("ba_" + nm + ".dz", "dw_" + nm + ".dz")
        link(xin, "dw_" + nm + ".x")
        ns = i["rp"] // i["rs"]
        k9 = 9 * i["cin"]
        prog("ws_" + nm, kernel("wsum.chx"), ns=ns, r=co, k=k9, bm=min(co, 16), bn=block(k9, 128))
        link("dw_" + nm + ".part", "ws_" + nm + ".part")
        grad_of["w_" + nm] = "ws_" + nm + ".dw"
        grad_of["g_" + nm] = "bg_" + nm + ".dgamma"
        grad_of["b_" + nm] = "bg_" + nm + ".dbeta"
        if nm == "stem":
            return None
        t = tune.get("dx", {})
        prog("dx_" + nm, kernel("conv3_dx.chx", tag="_" + nm, **{**dict(threads=128, warps="2 2", bm=i["cbm"], bn=min(i["cin"], 64), bk=min(co, 32)), **t}), rp=i["rp"], cin=i["cin"], cout=co, wp=i["wp"], g=G)
        link("ba_" + nm + ".dz", "dx_" + nm + ".dz")
        res("dx_" + nm + ".w", p("w_" + nm + ".h"), "f16")
        return "dx_" + nm + ".gx"

    for s in reversed(range(len(c.widths))):
        for b in reversed(range(c.blocks[s])):
            nb_, na_ = f"s{s}b{b}b", f"s{s}b{b}a"
            gx_b = bwd_layer(nb_, gout)
            gx_a = bwd_layer(na_, gx_b)
            prog(f"add_s{s}b{b}", kernel("add2.chx"), r=info[na_]["rp"], c=info[na_]["ci"], bm=info[na_]["bm_e"])
            link(gx_a, f"add_s{s}b{b}.a")
            link(f"ba_{nb_}.skip", f"add_s{s}b{b}.b")
            gout = f"add_s{s}b{b}.y"
        if s > 0:
            gx = bwd_layer(f"up{s}", gout)
            prog(f"poolb{s}", kernel("pool_bwd.chx"), n=n, h=hs[s - 1], c=c.widths[s - 1])
            link(gx, f"poolb{s}.gr")
            res(f"poolb{s}.pt", p(f"pt{s}"))
            gout = f"poolb{s}.gi"
        else:
            bwd_layer("stem", gout)

    # ---- the optimiser
    f16_users = {"wfc": ["fc", "dfeat"]}
    f32_users = {}
    for nm, _, _ in layers:
        f16_users["w_" + nm] = ["cv_" + nm] + ([] if nm == "stem" else ["dx_" + nm])
        f32_users["g_" + nm] = ["bf_" + nm + ".gamma", "ba_" + nm + ".gamma"]
        f32_users["b_" + nm] = ["bf_" + nm + ".beta"]
    sz = {name: int(np.asarray(P[name]).size) for name in mats + vecs}
    common = dict(gs=repr(1.0 / (n * LOSS_SCALE)), b1=B1, b2=B2, eps=EPS)
    for name in mats:
        prog("a_" + name, os.path.join(ROOT, "examples", "gpt", "adamw_h.chx"), n=sz[name], bn=block(sz[name], 128), lwd=repr(lr * wd), **common)
    for name in vecs:
        prog("a_" + name, os.path.join(ROOT, "examples", "train", "adam.chx"), n=sz[name], bn=block(sz[name], 128), **common)
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
        for nm, _, _ in layers:
            L.append(f"answer bf_{nm}.rmo")
            L.append(f"answer bf_{nm}.rvo")
    with open(p("session"), "w") as f:
        f.write("\n".join(L) + "\n")
    return p("session")


def parse_dump(c, numbers):
    """The parameters (as cnn_ref's) and the running statistics from the numbers a `dump` session answers after the loss."""
    mats, vecs = names(c)
    shapes = {"w_" + n: (co, 3, 3, ci) for n, ci, co in c.layers()}
    shapes.update({k + n: (co,) for n, _, co in c.layers() for k in ("g_", "b_")})
    shapes["wfc"] = (c.classes, c.widths[-1])
    out, o = {}, 1
    cp = cin_padded(c)
    for name in mats + vecs:
        shape = shapes[name]
        if name == "w_stem":
            raw = np.array(numbers[o:o + shape[0] * 9 * cp]).reshape(shape[0], 3, 3, cp)[..., : c.cin]
            o += shape[0] * 9 * cp
            out[name] = raw
        elif name == "wfc":
            out[name] = np.array(numbers[o:o + VP * shape[1]]).reshape(VP, shape[1])[: c.classes]
            o += VP * shape[1]
        else:
            k = int(np.prod(shape))
            out[name] = np.array(numbers[o:o + k]).reshape(shape)
            o += k
    running = {}
    for nm, _, co in c.layers():
        running[nm] = (np.array(numbers[o:o + co]), np.array(numbers[o + co:o + 2 * co]))
        o += 2 * co
    return out, running
