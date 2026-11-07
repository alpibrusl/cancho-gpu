#!/usr/bin/env python3
"""A ResNet-18-class backbone trained with AdamW on the device (docs/resident.md section 26; the plan is docs/roadmap-models.md M9, V1). resnet_ref.py is its specification (binary64
numpy, hand-written backward, torch float64 autograd and finite differences). This file writes the session, the way cnn_session.py writes the M4 CNN's; the new part is the geometry.

Every strided convolution is a stride-1 convolution of a *space-to-depth* image: a 2x2 block of pixels becomes one pixel of 4 c channels (s2d_fwd, a product with a 0/1 selection matrix, since a
window with a stride is not in the language). A 3x3 stride-2 convolution (padding 1) is then a 2x2 convolution with taps at (-1, 0), a 1x1 stride-2 convolution reads the (0,0) quarter of the
channels, and the 7x7 stride-2 stem (padding 3) is a 4x4 convolution with taps at (-2 .. +1) of the space-to-depth image: a tap may reach two pixels past an edge, which the one-pixel border of
the padded layout absorbs, because the next row's border cell is what a flat shifted read lands on. The weights are held in that layout (`embed`), with a mask for the entries that are not weights
(7x7 = 49 of 64 taps of the 4x4 window of 2x2 blocks); the mask multiplies the weight gradient so Adam never moves them. Max-pooling is nine shifted windows of the space-to-depth image
compared with `step`. The data are kept as in cnn_session.py (padded matrices with 3-row guards of `G` zero rows)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import cnn_session as C  # noqa: E402
import resnet_ref as R  # noqa: E402

G = 192  # guard rows: at least the widest shift, 2 wp + 2 (the stem's taps reach two rows and two columns) = 134 at 96 x 128
VP, B1, B2, EPS = C.VP, C.B1, C.B2, C.EPS
LOSS_SCALE, BN_EPS, BN_MOM = C.LOSS_SCALE, C.BN_EPS, C.BN_MOM
block, pick, put = C.block, C.pick, C.put
CP = 4  # the image's channels padded to 4, so that its space-to-depth image has 16 (a multiple of the matrix unit's 8)

# kind -> (kh, kw, oy, ox, kernel size K, pad) of the stride-1 window on the space-to-depth image (or the image itself)
GEOM = {"stem": (4, 4, 2, 2, 7, 3), "c3": (3, 3, 1, 1, 3, 1), "c3s2": (2, 2, 1, 1, 3, 1), "c1s2": (1, 1, 0, 0, 1, 0)}


def embed(real, kind, ones=False):
    """A weight [co, K, K, ci] in the layout of the stride-1 window: [co, kh kw cin]. For a strided kind the taps are spread over the 2x2 blocks' channel groups (2 py + px)."""
    co, K, _, ci = real.shape
    kh, kw, oy, ox, _, pad = GEOM[kind]
    if kind in ("c3", "c1s2"):
        return (np.ones_like(real) if ones else real).reshape(co, -1)
    cg = CP if kind == "stem" else ci
    out = np.zeros((co, kh, kw, 4 * cg))
    for a in range(kh):
        for b in range(kw):
            for py in range(2):
                for px in range(2):
                    ky, kx = 2 * (a - oy) + py + pad, 2 * (b - ox) + px + pad
                    if 0 <= ky < K and 0 <= kx < K:
                        out[:, a, b, (2 * py + px) * cg:(2 * py + px) * cg + ci] = 1.0 if ones else real[:, ky, kx, :]
    return out.reshape(co, -1)


def unembed(eff, kind, co, ci):
    """The inverse of embed on the entries that are weights."""
    kh, kw, oy, ox, K, pad = GEOM[kind]
    if kind in ("c3", "c1s2"):
        return eff.reshape(co, K, K, ci)
    cg = CP if kind == "stem" else ci
    e = eff.reshape(co, kh, kw, 4 * cg)
    out = np.zeros((co, K, K, ci))
    for a in range(kh):
        for b in range(kw):
            for py in range(2):
                for px in range(2):
                    ky, kx = 2 * (a - oy) + py + pad, 2 * (b - ox) + px + pad
                    if 0 <= ky < K and 0 <= kx < K:
                        out[:, ky, kx, :] = e[:, a, b, (2 * py + px) * cg:(2 * py + px) * cg + ci]
    return out


def conv_dims(kind, ci):
    """(cx, cin): the width of the matrix the convolution reads and the columns of it that it uses."""
    if kind == "stem":
        return 4 * CP, 4 * CP
    if kind == "c3":
        return ci, ci
    if kind == "c3s2":
        return 4 * ci, 4 * ci
    return 4 * ci, ci


def pad_params(c, P):
    out = {}
    for n, kind, ci, co in c.layers():
        out["w_" + n] = embed(P["w_" + n], kind)
        out["g_" + n], out["b_" + n] = P["g_" + n], P["b_" + n]
    out["wfc"] = np.concatenate([P["wfc"], np.zeros((VP - c.classes, P["wfc"].shape[1]))], axis=0)
    return out


def input_tables(c, images, directory):
    """images [S, N, H, W, cin]: every batch as two blocks of N/2 images, [N/2 H W, cin] f16, written to xa and xb."""
    half = c.n // 2
    for name, part in (("xa", images[:, :half]), ("xb", images[:, half:])):
        with open(os.path.join(directory, name), "wb") as f:
            for batch in part:
                f.write(batch.reshape(half * c.h * c.w, c.cin).astype("<f2").tobytes())


def warps_for(bm, bn):
    """The warp grid of an mma tile bm x bn: 2 x 2 where the atoms (8 x 8) allow it, fewer where the tile is narrow (a 32 x 8 tile on Metal does not split into 2 x 2 warps of whole atoms)."""
    wr, wc = (2 if bm % 16 == 0 else 1), (2 if bn % 16 == 0 else 1)
    return dict(threads=32 * wr * wc, warps=f"{wr} {wc}")


def wb_for(w):
    return next(b for b in range(min(w, 16), 0, -1) if w % b == 0)


def write(directory, c, images, labels, steps, lr=2e-3, wd=0.01, seed=1, grads=False, lr_fn=None, params=None, dump=False, tune=None, keep=False, opt=None):
    """Writes the session and answers its path. `images`: [S, N, H, W, cin] (S batches, taken round and round); `labels`: [S, N]. `c`: a resnet_ref.Config."""
    os.makedirs(directory, exist_ok=True)
    p = lambda name: os.path.join(directory, name)
    n, nsteps = c.n, len(images)
    assert images.shape[1:] == (n, c.h, c.w, c.cin) and n % 2 == 0
    P = pad_params(c, params or R.init_params(c, seed))
    mats, vecs = R.param_names(c)
    layers = c.layers()
    lr_fn = lr_fn or (lambda s: lr)
    tune = tune or {}
    # the resolutions: 'st' after the stem, then the four stages
    dims = {"st": (c.h // 2, c.w // 2)}
    dims.update({str(s): c.size_at(s) for s in range(4)})
    rps = {r: n * (h + 2) * (w + 2) for r, (h, w) in dims.items()}
    for r, rp in rps.items():
        assert rp % 16 == 0, f"the batch must make the padded rows of resolution {r} a multiple of 16"
        assert G >= 2 * (dims[r][1] + 2) + 2
    input_tables(c, images, directory)
    cmat = np.zeros((c.cin, CP))
    cmat[:, : c.cin] = np.eye(c.cin)
    put(p("cmat"), cmat)
    put(p("tgt"), np.asarray(labels, dtype=np.float32))
    put(p("sched"), np.array(C.schedule(steps, lr_fn)).reshape(-1))
    put(p("ones"), np.ones(n))
    put(p("jrow_v"), np.arange(VP))
    put(p("maskv"), np.where(np.arange(VP) < c.classes, 0.0, -1e9))
    for r, (h, w) in dims.items():
        valid = np.zeros((h + 2, w + 2))
        valid[1:-1, 1:-1] = 1.0
        put(p(f"mk_{r}"), np.tile(valid.reshape(-1), n))
        put(p(f"gv_{r}"), valid.reshape(-1) / (h * w))
    for wb in {wb_for(w) for _, w in dims.values()} | {wb_for(c.w // 2)}:
        for px in range(2):
            s = np.zeros((wb, 2 * wb))
            s[np.arange(wb), 2 * np.arange(wb) + px] = 1.0
            put(p(f"sel{px}_{wb}"), s)
            put(p(f"selt{px}_{wb}"), s.T)
    for name in mats + vecs:
        put(p(name), P[name])
        put(p("z" + name), np.zeros_like(P[name]))
    for name in mats:
        put(p(name + ".h"), P[name], "<f2")
    for nm, kind, ci, co in layers:
        put(p(f"rm_{nm}"), np.zeros(co))
        put(p(f"rv_{nm}"), np.ones(co))
        if kind == "stem":
            put(p("wmask_stem"), embed(np.zeros((co, 7, 7, c.cin)), kind, ones=True)[0:1])
        elif kind == "c3s2":
            put(p("wmask_" + nm), embed(np.zeros((co, 3, 3, ci)), kind, ones=True)[0:1])

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

    L = [f"# ResNet: widths {c.widths}, blocks {c.blocks}, image {c.h}x{c.w}, batch {n}, classes {c.classes} (padded to {VP}), AdamW {lr}"]
    prog = lambda name, kern, **kw: L.append(f"program {name} {kern} " + " ".join(f"{k}={v}" for k, v in kw.items()))
    res = lambda target, path, flag="f32": L.append(f"resident {target} {path}" + ("" if flag == "" else f" {flag}"))
    link = lambda a, b: L.append(f"link {a} -> {b}")
    carry = lambda a, b: L.append(f"carry {a} -> {b}")
    tab = lambda target, path, blocks, flag="f32": L.append(f"table {target} {path} {blocks}" + ("" if flag == "" else f" {flag}"))

    info = {}
    for nm, kind, ci, co in layers:
        r = "st" if nm == "stem" else nm[1]
        rp = rps[r]
        h, w = dims[r]
        cx, cin = conv_dims(kind, ci)
        kh, kw, oy, ox, _, _ = GEOM[kind]
        rb = block(rp, max(8, 2048 // co))
        nb = rp // rb
        cb = block(nb, 256)
        info[nm] = dict(kind=kind, r=r, cx=cx, cin=cin, co=co, rp=rp, wp=w + 2, rb=rb, nb=nb, cb=cb, bc=block(co, 32), bm_e=block(rp, max(1, 2048 // co)), bm_b=block(rp, max(1, 1024 // co)),
                        cbm=pick(rp, 64, 16), rs=pick(rp, 1024, 32 if rp % 32 == 0 else 16), m=n * h * w, kh=kh, kw=kw, oy=oy, ox=ox, ci=ci)
        put(p(f"ones_cb{cb}"), np.ones(cb))

    def conv_fwd(nm, xsrc):
        i = info[nm]
        t = tune.get("conv", {})
        k = kernel("convk.chx", tag="_" + nm, **{**warps_for(i["cbm"], min(i["co"], 64)), **dict(bm=i["cbm"], bn=min(i["co"], 64), bk=min(i["cin"], 32)), **t})
        prog("cv_" + nm, k, rp=i["rp"], cx=i["cx"], cin=i["cin"], cout=i["co"], wp=i["wp"], g=G, kh=i["kh"], kw=i["kw"], oy=i["oy"], ox=i["ox"])
        link(xsrc, "cv_" + nm + ".x")
        res("cv_" + nm + ".w", p("w_" + nm + ".h"), "f16")
        prog("bp_" + nm, kernel("bn_part.chx"), rp=i["rp"], c=i["co"], rb=i["rb"])
        link("cv_" + nm + ".y", "bp_" + nm + ".z")
        res("bp_" + nm + ".v", p(f"mk_{i['r']}"))
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

    def bn_apply(nm, resname=None, relu=True):
        i = info[nm]
        kern = "bn_apply_res.chx" if resname else "bn_apply.chx" if relu else "bn_apply_lin.chx"
        prog("ap_" + nm, kernel(kern), rp=i["rp"], c=i["co"], bm=i["bm_e"], g=G)
        link("cv_" + nm + ".y", "ap_" + nm + ".z")
        link("bf_" + nm + ".sc", "ap_" + nm + ".sc")
        link("bf_" + nm + ".sh", "ap_" + nm + ".sh")
        res("ap_" + nm + ".mk", p(f"mk_{i['r']}"))
        if resname:
            link(resname, "ap_" + nm + ".res")
        return "ap_" + nm + ".a"

    def s2d(name, src, r_out, cch):
        """The space-to-depth image of `src` (an activation at the resolution twice r_out's, cch channels)."""
        h, w = dims[r_out]
        wb = wb_for(w)
        prog(name, kernel("s2d_fwd.chx"), n=n, h=h, w=w, c=cch, g=G, wb=wb)
        link(src, name + ".x")
        res(name + ".sel0", p(f"sel0_{wb}"))
        res(name + ".sel1", p(f"sel1_{wb}"))
        return name + ".s"

    # ---- forward
    prog("pk", kernel("pack_hw.chx"), n2=n // 2, h=c.h, w=c.w, cin=c.cin, cp=CP, g=G)
    tab("pk.xa", p("xa"), nsteps, "f16")
    tab("pk.xb", p("xb"), nsteps, "f16")
    res("pk.cmat", p("cmat"))
    s_in = s2d("sin", "pk.o", "st", CP)
    conv_fwd("stem", s_in)
    act = bn_apply("stem")
    s_mp = s2d("smp", act, "0", c.widths[0])
    h0, w0 = dims["0"]
    prog("mp", kernel("maxpool_fwd.chx"), rp=rps["0"], c=c.widths[0], wp=w0 + 2, g=G, bm=block(rps["0"], min(16, max(1, 512 // c.widths[0]))))  # the nine staged candidates: bm 16 at 64 channels made 43,008 B > 32,768
    link(s_mp, "mp.s")
    res("mp.mk", p("mk_0"))
    act = "mp.a"
    acts = {"stem": (s_in, None)}
    block_in = {}
    for s in range(4):
        for b in range(c.blocks[s]):
            a0 = act
            first = b == 0 and s > 0
            nm_a, nm_b, nm_d = f"s{s}b{b}a", f"s{s}b{b}b", f"s{s}b{b}d"
            if first:
                sx = s2d(f"s2d_{s}", a0, str(s), c.widths[s - 1])
                block_in[s] = sx
                conv_fwd(nm_a, sx)
                acts[nm_a] = (sx, None)
                conv_fwd(nm_d, sx)
                acts[nm_d] = (sx, None)
                dsn = bn_apply(nm_d, relu=False)
            else:
                conv_fwd(nm_a, a0)
                acts[nm_a] = (a0, None)
                dsn = a0
            u = bn_apply(nm_a)
            conv_fwd(nm_b, u)
            acts[nm_b] = (u, dsn)
            act = bn_apply(nm_b, dsn)
    cl = c.widths[-1]
    hl, wl = dims["3"]
    rcg = block((hl + 2) * (wl + 2), min(20, max(1, 4096 // cl)))  # rc rows of cl f32 channels are staged: 12 KB at 512 channels (20 rows made 33,792 B > 32,768, section 26.2)
    prog("gap", kernel("gap2_fwd.chx"), n=n, h=hl, w=wl, c=cl, g=G, rc=rcg)
    link(act, "gap.a")
    res("gap.vm", p("gv_3"))
    prog("fc", kernel("gemm_mma.chx", **warps_for(min(n, 64), VP), bm=min(n, 64), bn=VP, bk=min(cl, 32)), m=n, n=VP, k=cl)
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
    prog("gfc", kernel("gemm_mma_tn.chx", **warps_for(VP, min(cl, 64)), bm=VP, bn=min(cl, 64), bk=min(n, 32)), m=VP, n=cl, k=n)
    link("ce.dl", "gfc.a")
    link("gap.f", "gfc.w")
    prog("dfeat", kernel("gemm_mma_nn.chx", **warps_for(min(n, 64), min(cl, 64)), bm=min(n, 64), bn=min(cl, 64), bk=VP), m=n, n=cl, k=VP)
    link("ce.dl", "dfeat.a")
    res("dfeat.w", p("wfc.h"), "f16")
    prog("gapb", kernel("gap2_bwd.chx"), n=n, h=hl, w=wl, c=cl, rc=rcg)
    link("dfeat.y", "gapb.gf")
    res("gapb.mc", p("gv_3"))
    grad_of = {"wfc": "gfc.y"}

    def bwd_layer(nm, gout):
        """The backward of one convolution-BatchNorm layer given the gradient `gout` of its output; answers the name of the f32 gradient of the convolution's input (None for the stem)."""
        i = info[nm]
        xin, rsd = acts[nm]
        co = i["co"]
        rs = bool(rsd) and not nm.endswith("d")
        lin = nm.endswith("d")
        sfx = "_lin" if lin else ""
        prog("bq_" + nm, kernel(("bn_bwd_part_res.chx" if rs else f"bn_bwd_part{sfx}.chx")), rp=i["rp"], c=co, rb=i["rb"], **({"g": G} if rs else {}))
        link(gout, "bq_" + nm + ".gr")
        link("cv_" + nm + ".y", "bq_" + nm + ".z")
        for k_ in (("mean", "istd") if lin else ("sc", "sh", "mean", "istd")):
            link(f"bf_{nm}.{k_}", f"bq_{nm}.{k_}")
        res("bq_" + nm + ".v", p(f"mk_{i['r']}"))
        if rs:
            link(rsd, "bq_" + nm + ".res")
        prog("bg_" + nm, kernel("bn_bwd_fin.chx"), nb=i["nb"], c=co, cb=i["cb"], bc=i["bc"])
        link("bq_" + nm + ".p1", "bg_" + nm + ".p1")
        link("bq_" + nm + ".p2", "bg_" + nm + ".p2")
        res("bg_" + nm + ".ones", p(f"ones_cb{i['cb']}"))
        prog("ba_" + nm, kernel("bn_bwd_apply_res.chx" if rs else f"bn_bwd_apply{sfx}.chx"), rp=i["rp"], c=co, bm=i["bm_b"], g=G, m=i["m"])
        link(gout, "ba_" + nm + ".gr")
        link("cv_" + nm + ".y", "ba_" + nm + ".z")
        for k_ in (("mean", "istd") if lin else ("sc", "sh", "mean", "istd")):
            link(f"bf_{nm}.{k_}", f"ba_{nm}.{k_}")
        res("ba_" + nm + ".gamma", p("g_" + nm))
        link("bg_" + nm + ".dbeta", "ba_" + nm + ".dbeta")
        link("bg_" + nm + ".dgamma", "ba_" + nm + ".dgamma")
        res("ba_" + nm + ".mk", p(f"mk_{i['r']}"))
        if rs:
            link(rsd, "ba_" + nm + ".res")
        t = tune.get("dw", {})
        tm, tn = min(co, 64), min(i["cin"], 64)
        wr, wc = (2 if tm % 16 == 0 else 1), (2 if tn % 16 == 0 else 1)
        kd = dict(rp=i["rp"], cx=i["cx"], cin=i["cin"], cout=co, wp=i["wp"], g=G, kh=i["kh"], kw=i["kw"], oy=i["oy"], ox=i["ox"], rs=i["rs"])
        prog("dw_" + nm, kernel("convk_dw.chx", tag="_" + nm, **{**dict(threads=32 * wr * wc, warps=f"{wr} {wc}", bm=tm, bn=tn, bk=32 if i["rs"] % 32 == 0 else 16), **t}), **kd)
        link("ba_" + nm + ".dz", "dw_" + nm + ".dz")
        link(xin, "dw_" + nm + ".x")
        ns = i["rp"] // i["rs"]
        k9 = i["kh"] * i["kw"] * i["cin"]
        masked = i["kind"] in ("stem", "c3s2")
        prog("ws_" + nm, kernel("wsum_mask.chx" if masked else "wsum.chx"), ns=ns, r=co, k=k9, bm=min(co, 16), bn=block(k9, 128))
        link("dw_" + nm + ".part", "ws_" + nm + ".part")
        if masked:
            res("ws_" + nm + ".mk", p("wmask_" + nm))
        grad_of["w_" + nm] = "ws_" + nm + ".dw"
        grad_of["g_" + nm] = "bg_" + nm + ".dgamma"
        grad_of["b_" + nm] = "bg_" + nm + ".dbeta"
        if nm == "stem":
            return None
        t = tune.get("dx", {})
        prog("dx_" + nm, kernel("convk_dx.chx", tag="_" + nm, **{**warps_for(i["cbm"], min(i["cin"], 64)), **dict(bm=i["cbm"], bn=min(i["cin"], 64), bk=min(co, 32)), **t}),
             rp=i["rp"], cx=i["cx"], cin=i["cin"], cout=co, wp=i["wp"], g=G, kh=i["kh"], kw=i["kw"], oy=i["oy"], ox=i["ox"])
        link("ba_" + nm + ".dz", "dx_" + nm + ".dz")
        res("dx_" + nm + ".w", p("w_" + nm + ".h"), "f16")
        return "dx_" + nm + ".gx"

    def s2d_back(name, gsrc, r_out, cch):
        h, w = dims[r_out]
        wb = wb_for(w)
        prog(name, kernel("s2d_bwd.chx"), n=n, h=h, w=w, c=cch, wb=wb)
        link(gsrc, name + ".gs")
        res(name + ".selt0", p(f"selt0_{wb}"))
        res(name + ".selt1", p(f"selt1_{wb}"))
        return name + ".gi"

    gout = "gapb.gi"
    for s in reversed(range(4)):
        for b in reversed(range(c.blocks[s])):
            nm_a, nm_b, nm_d = f"s{s}b{b}a", f"s{s}b{b}b", f"s{s}b{b}d"
            first = b == 0 and s > 0
            gx_b = bwd_layer(nm_b, gout)
            gx_a = bwd_layer(nm_a, gx_b)
            ia = info[nm_a]
            if first:
                gx_d = bwd_layer(nm_d, f"ba_{nm_b}.skip")
                prog(f"add_{s}b{b}", kernel("add2.chx"), r=ia["rp"], c=ia["cx"], bm=block(ia["rp"], max(1, 2048 // ia["cx"])))
                link(gx_a, f"add_{s}b{b}.a")
                link(gx_d, f"add_{s}b{b}.b")
                gout = s2d_back(f"s2db_{s}", f"add_{s}b{b}.y", str(s), c.widths[s - 1])
            else:
                last_guarded = s == 0 and b == 0
                prog(f"add_{s}b{b}", kernel("add2g.chx" if last_guarded else "add2.chx"), r=ia["rp"], c=ia["ci"], bm=ia["bm_e"], **({"g": G} if last_guarded else {}))
                link(gx_a, f"add_{s}b{b}.a")
                link(f"ba_{nm_b}.skip", f"add_{s}b{b}.b")
                gout = f"add_{s}b{b}.y"
    prog("mpb", kernel("maxpool_bwd.chx"), rp=rps["0"], c=c.widths[0], wp=w0 + 2, g=G, bm=block(rps["0"], 16))
    link(gout, "mpb.gr")
    link("mp.idx", "mpb.idx")
    gst = s2d_back("s2db_mp", "mpb.gs", "0", c.widths[0])
    bwd_layer("stem", gst)

    # ---- the optimiser
    f16_users = {"wfc": ["fc", "dfeat"]}
    f32_users = {}
    for nm, _, _, _ in layers:
        f16_users["w_" + nm] = ["cv_" + nm] + ([] if nm == "stem" else ["dx_" + nm])
        f32_users["g_" + nm] = ["bf_" + nm + ".gamma", "ba_" + nm + ".gamma"]
        f32_users["b_" + nm] = ["bf_" + nm + ".beta"]
    sz = {name: int(np.asarray(P[name]).size) for name in mats + vecs}
    common = dict(gs=repr(1.0 / (n * LOSS_SCALE)), b1=B1, b2=B2, eps=EPS)
    if opt:
        import gpt_opt
        gpt_opt.wire(L, p, mats, vecs, sz, grad_of, f16_users, f32_users, 1.0 / (n * LOSS_SCALE), B1, B2, EPS, steps, lr_fn, opt)
    else:
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
        for nm, _, _, _ in layers:
            L.append(f"answer bf_{nm}.rmo")
            L.append(f"answer bf_{nm}.rvo")
    with open(p("session"), "w") as f:
        f.write("\n".join(L) + "\n")
    return p("session")


def parse_flat(c, numbers, o=0, effective=False):
    """The parameters (resnet_ref's layout) from the numbers a session answers in the order mats then vecs, starting at `o`; with `effective` the gradient entries not masked are also unembedded."""
    mats, vecs = R.param_names(c)
    kinds = {n: (k, ci, co) for n, k, ci, co in c.layers()}
    out = {}
    for name in mats + vecs:
        if name == "wfc":
            out[name] = np.array(numbers[o:o + VP * c.widths[-1]]).reshape(VP, c.widths[-1])[: c.classes]
            o += VP * c.widths[-1]
        elif name.startswith("w_"):
            kind, ci, co = kinds[name[2:]]
            cx, cin = conv_dims(kind, ci)
            kh, kw = GEOM[kind][:2]
            k = co * kh * kw * cin
            out[name] = unembed(np.array(numbers[o:o + k]).reshape(co, -1), kind, co, ci)
            o += k
        else:
            co = kinds[name[2:]][2]
            out[name] = np.array(numbers[o:o + co])
            o += co
    return out, o


def parse_dump(c, numbers):
    """The parameters and the running statistics from the numbers a `dump` session answers after the loss."""
    out, o = parse_flat(c, numbers, 1)
    running = {}
    for nm, _, _, co in c.layers():
        running[nm] = (np.array(numbers[o:o + co]), np.array(numbers[o + co:o + 2 * co]))
        o += 2 * co
    return out, running
