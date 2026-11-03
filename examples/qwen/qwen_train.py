#!/usr/bin/env python3
"""LoRA fine-tuning of Qwen2 in a session of .chx kernels (docs/resident.md section 21; roadmap M7.2): the forward pass of qwen_session.py with the adapters, the backward pass, and AdamW on the adapters.
qwen_ref.py (`lora_loss_grads`) is the specification (binary64, hand-written backward, checked by finite differences).

    python3 examples/qwen/qwen_train.py WEIGHTS_DIR SESSION_DIR      # a session of Qwen2.5-0.5B on the fixed batch, rank 16

The adapters are per projection as in peft, y = x W^T + (alpha / r) (x A^T) B^T, but the projections the converter fused (q/k/v, gate/up) keep their adapters in one matrix each: Acat is the
r-row blocks of the projections' A one under another [n_proj r, in] and Bcat is block-diagonal [out, n_proj r] (zero off the blocks), so a fused projection and its three adapters are two products on the
matrix unit (`gemm_lora`: the base product and the 16-wide adapter product accumulate into one fragment). Mathematically this is exactly three separate adapters (the off-diagonal blocks are zero, stay zero:
the optimiser masks their gradient), and `split_grads` / `split_params` give the per-projection matrices back. The scale alpha / r is folded into the f16 copy of B (`Bs = s B`): with it the adapter's
forward is `mid Bs^T`, the gradient of mid is `dy Bs`, and the gradient of A is `(dy Bs)^T x` with no further factor; the gradient of B is `s dy^T mid`, whose factor s goes into B's gradient scale.

A quantised base (docs/resident.md section 27; roadmap M7.5): when the weights directory holds `NAME.q`, `NAME.s`, `NAME.g` for the four matrices of a block (quant.py pack) the session keeps those i8 and f32 files resident
(0.56 bytes a weight), and before each product that reads a matrix runs `dq_fp4.chx`, which dequantises it into an f16 scratch laid out as the f16 file would be; the products, the rest of the forward and the backward are
the unchanged ones. The scratch of a matrix kind is one buffer shared by every layer and by the forward and the backward (`alias`), so one layer's matrices are all it costs.

Mixed precision as in qwen_session.py: f16 weights and f16 operands of every product, f32 accumulation, f32 residual stream and gradient stream, f32 master adapters and Adam's moments with f16 copies.
The gradient of the logits is scaled by `loss_scale` (f16 range) and the optimiser divides it and the number of counted positions out."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import qwen_ref as R  # noqa: E402
import qwen_session as S  # noqa: E402

GROUPS = (("qkv", ("q", "k", "v")), ("o", ("o",)), ("gu", ("gate", "up")), ("dn", ("down",)))


def group_dims(c, r):
    """For each group: (in, out of each projection, rows of Acat = n_proj r, rows of Bcat = sum of the outs)."""
    out = {}
    for g, ps in GROUPS:
        ins = {R.proj_dims(c, p)[0] for p in ps}
        assert len(ins) == 1
        outs = [R.proj_dims(c, p)[1] for p in ps]
        out[g] = dict(inp=ins.pop(), outs=outs, r2=len(ps) * r, N=sum(outs), projs=ps)
    return out


def fuse(c, r, AB, l):
    """The fused matrices of layer l from the per-projection ones: {group: (Acat [r2, in], Bcat [N, r2] block-diagonal, mask [N, r2])}."""
    res = {}
    for g, gd in group_dims(c, r).items():
        A = np.concatenate([AB[f"l{l:02d}.{p}.A"] for p in gd["projs"]], 0)
        B = np.zeros((gd["N"], gd["r2"]))
        mask = np.zeros((gd["N"], gd["r2"]))
        o = 0
        for i, p in enumerate(gd["projs"]):
            out = gd["outs"][i]
            B[o:o + out, i * r:(i + 1) * r] = AB[f"l{l:02d}.{p}.B"]
            mask[o:o + out, i * r:(i + 1) * r] = 1.0
            o += out
        res[g] = (A, B, mask)
    return res


def split_params(c, r, fused, l):
    """The per-projection {name: matrix} of layer l from {group: (Acat, Bcat)}."""
    out = {}
    for g, gd in group_dims(c, r).items():
        A, B = fused[g][0], fused[g][1]
        o = 0
        for i, p in enumerate(gd["projs"]):
            out[f"l{l:02d}.{p}.A"] = A[i * r:(i + 1) * r]
            out[f"l{l:02d}.{p}.B"] = B[o:o + gd["outs"][i], i * r:(i + 1) * r]
            o += gd["outs"][i]
    return out


def fused_names(c, r):
    return [(l, g, ab) for l in range(c.layers) for g in group_dims(c, r) for ab in ("A", "B")]


def schedule(steps, lr_fn, b1, b2, first=0):
    """Per step: [lr_t / (1 - beta1^t), 1 / sqrt(1 - beta2^t)] (binary64): the learning-rate schedule is in the table. `first` steps were taken before (a session continued from a checkpoint)."""
    return [(lr_fn(t) / (1.0 - b1 ** t), 1.0 / math.sqrt(1.0 - b2 ** t)) for t in range(first + 1, first + steps + 1)]


def rope_bwd_source(c, bm):
    """The backward of the rotary embedding for all heads in one kernel: the gradients of the query heads (heads-major f32), of the key heads and of the value heads (token-major f32) become the f16
    fused gradient dqkv [n, d + 2 dkv] the q/k/v projection's backward reads. dx1 = dy1 cos + dy2 sin, dx2 = dy2 cos - dy1 sin (the inverse rotation)."""
    nh, hkv, d = c.h, c.hkv, c.d
    hf = c.dh // 2
    body = []

    def rot(src, row, col, dst_col, tag):
        body.extend([f"  let a{tag} = load {src}[{row}, {col} ; bm, hf]", f"  let e{tag} = load {src}[{row}, {col} + hf ; bm, hf]",
                     f"  let c{tag} = load cs[i * bm, 0 ; bm, hf]", f"  let s{tag} = load sn[i * bm, 0 ; bm, hf]",
                     f"  let p{tag} = &a{tag} * &c{tag}", f"  let q{tag} = &e{tag} * &s{tag}", f"  let lo{tag} = p{tag} + q{tag}",
                     f"  let u{tag} = e{tag} * c{tag}", f"  let w{tag} = a{tag} * s{tag}", f"  let hi{tag} = u{tag} - w{tag}",
                     f"  let lh{tag} = stage f16 lo{tag}", f"  let hh{tag} = stage f16 hi{tag}",
                     f"  store lh{tag} -> y[b * t + i * bm, {dst_col}]", f"  store hh{tag} -> y[b * t + i * bm, {dst_col} + hf]"])

    for h in range(nh):
        rot("dq", f"(b * {nh} + {h}) * t + i * bm", "0", f"{h * 2 * hf}", f"q{h}")
    for g in range(hkv):
        rot("dk", "b * t + i * bm", f"{g * 2 * hf}", f"{d + g * 2 * hf}", f"k{g}")
    for g in range(hkv):
        body.extend([f"  let v{g} = load dv[b * t + i * bm, {g * 2 * hf} ; bm, hf * 2]", f"  let vh{g} = stage f16 v{g}", f"  store vh{g} -> y[b * t + i * bm, {d + hkv * 2 * hf + g * 2 * hf}]"])
    return f"""// The backward of the rotary embedding of all {nh} query heads and {hkv} key heads, and the cast of the {hkv} value-head gradients, into the f16 fused gradient of the q/k/v projection
// (generated by qwen_train.rope_bwd_source; docs/resident.md section 21).
algo rope_bwd(nb, t, bm, hf)
  in  dq: f32[nb * {nh} * t, 2 * hf]
  in  dk: f32[nb * t, {hkv * 2} * hf]
  in  dv: f32[nb * t, {hkv * 2} * hf]
  in  cs: f32[t, hf]
  in  sn: f32[t, hf]
  out y:  f16[nb * t, {d + 2 * hkv * c.dh}]
{{
  grid b over nb, i over t / bm
""" + "\n".join(body) + """
}

schedule rope_bwd for apple-m-series { threads 128 }
schedule rope_bwd for nvidia-ada     { threads 128 }
"""


def write(directory, c, wdir, tokens, AB, r=16, alpha=32.0, steps=1, lr=1e-4, lr_fn=None, wd=0.0, betas=(0.9, 0.999), eps_adam=1e-8, loss_scale=16.0, train=True,
          grads=None, save=None, probes=(), tune=None, reuse=True, state=None, first_step=0, save_state=None, lean=False):
    """Writes the session over the batches `tokens` ([S, nb, t], taken round and round, one per step) and the adapters `AB` ({'l00.q.A': [r, in], ...}); answers the path of the session file.
    A request is a count of steps; the answer is the mean loss of the last step's batch (positions with a next token). With train=False no optimiser runs (the adapters stay as they are) and with
    `grads` (a path) the raw gradient of every fused adapter matrix is saved there on the request `checkpoint` (see `read_grads`). With train=True and `save` (a path) the f32 fused adapter matrices are
    saved on `checkpoint`. `probes`: program outputs ('rm1_0.y') to answer after the loss, for debugging. With `reuse` (the default) the temporaries that only one layer needs share one buffer across
    layers (`alias`, docs/resident.md section 21.4); the session file's checks refuse an alias that would overwrite a value a later program reads.
    A run longer than a session's table can hold (a table is a file of at most 512 MiB: 60 batches of 8 x 512) is several sessions: with `save_state` (a path) the first and second moments are saved
    beside the adapters (`<path>.m`, `<path>.v`, as `read_fused` reads them) and the next session starts from them with `state` ({(l, group, 'A'|'B'): (m, v)}) and `first_step` (the steps already
    taken, which the bias correction and the learning-rate schedule count from; docs/resident.md section 22).
    With `lean` the memory-lean variant of docs/resident.md section 23.4: the fused gate/up product is kept as f16 (`swiglu_h2` writes the copy) and the SwiGLU output the down projection
    reads is not kept but recomputed from it in the backward, so that a layer's f32 gate/up product and f16 SwiGLU output are no longer saved for every layer."""
    os.makedirs(directory, exist_ok=True)
    p = lambda name: os.path.join(directory, name)
    w = lambda name: os.path.abspath(os.path.join(wdir, name))
    n, d, h, hkv, dh, ff, v, t, nb, ll = c.n, c.d, c.h, c.hkv, c.dh, c.ff, c.v, c.t, c.nb, c.layers
    grp, dkv, hf = c.group, c.dkv, c.dh // 2
    tokens = np.asarray(tokens)
    if tokens.ndim == 2:
        tokens = tokens[None]
    nsteps = tokens.shape[0]
    assert tokens.shape[1:] == (nb, t)
    assert dh % 32 == 0 and d % 64 == 0 and t % 64 == 0 and n % 64 == 0 and (d + 2 * dkv) % 64 == 0 and ff % 64 == 0 and h % hkv == 0 and dh == d // h and r % 16 == 0
    scale = alpha / r
    QG = S.quant_group(wdir)
    quantised = QG > 0
    assert not quantised or reuse, "a quantised base needs the shared scratch (reuse)"
    G = group_dims(c, r)
    b1, b2 = betas
    lr_fn = lr_fn or (lambda s_: lr)
    tn = dict(bc=2304, ebm=1, abm=2, rbm=8, cbm=8, sbm=1, sbc=S.block(ff, 256), attn=dict(threads=128, warps="2 2", bm=64, bn=64, bk=32), ao=dict(threads=128, warps="2 2", bm=64 if dh <= 64 else 32, bk=32),
              gemm_sched={})
    tn.update(tune or {})
    emb = np.fromfile(w("emb.f16"), dtype="<f2").reshape(-1, d)[:v]
    # data: the embeddings of each step's batch (f16, the lookup is host work), the 1-based targets, the weight of each position (the last of a sequence has no next token)
    S.put(p("x0h"), emb[tokens.reshape(nsteps, -1)], "<f2")
    S.put(p("tgt"), np.stack([R.next_targets(tk).reshape(-1) + 1 for tk in tokens]))
    ang = np.arange(t)[:, None] * c.theta ** (-np.arange(hf) / hf)[None, :]
    S.put(p("cos"), np.cos(ang))
    S.put(p("sin"), np.sin(ang))
    S.put(p("mask"), np.triu(np.full((t, t), -1e9), 1))
    wcount = np.ones((nb, t))
    wcount[:, -1] = 0.0
    wcount = wcount.reshape(-1)
    ncount = float(wcount.sum())
    S.put(p("wcount"), wcount)
    S.put(p("wscale"), wcount * loss_scale)
    S.put(p("zeros_nd"), np.zeros((n, d)))
    bc = min(tn["bc"], -(-v // 128) * 128)
    vp = -(-v // bc) * bc
    lm_file = w("emb.f16" if c.tied else "head.f16")  # the output layer: the embedding table when tied, else a matrix of its own (docs/resident.md section 28)
    exact = vp == v and os.path.getsize(lm_file) == v * d * 2
    wlm = lm_file if exact else p("wlm.f16")
    if not exact:
        with open(wlm, "wb") as f:
            f.write(np.fromfile(lm_file, dtype="<f2").reshape(-1, d)[:v].tobytes())
            f.write(np.zeros((vp - v, d), dtype="<f2").tobytes())
    S.put(p("jrow"), np.arange(1, vp + 1))
    S.put(p("maskv"), np.where(np.arange(vp) < v, 0.0, -1e9))
    for name, src in (("rope_q.chx", S.rope_source("rope_q", 1, True)), ("rope_k.chx", S.rope_source("rope_k", grp, True)), ("bias_v.chx", S.rope_source("bias_v", grp, False)),
                      ("rope_bwd.chx", rope_bwd_source(c, tn["rbm"]))):
        for nm, a, b in tn.get("kmut", []):
            if nm == name:
                assert a in src, (name, a)
                src = src.replace(a, b)
        open(p(name), "w").write(src)
    # the adapters: per fused group the f32 masters, the f16 copy of A, the f16 copy of s B
    for l in range(ll):
        for g, (A, B, mask) in fuse(c, r, AB, l).items():
            q = f"{l:02d}_{g}"
            S.put(p(f"A{q}"), A)
            S.put(p(f"B{q}"), B)
            S.put(p(f"Ah{q}"), A, "<f2")
            S.put(p(f"Bh{q}"), scale * B, "<f2")
            if train:
                for nm, arr in (("A", A), ("B", B)):
                    m0, v0 = state[(l, g, nm)] if state else (np.zeros_like(arr), np.zeros_like(arr))
                    assert m0.shape == arr.shape and v0.shape == arr.shape, (l, g, nm, m0.shape, arr.shape)
                    S.put(p(f"zm{nm}{q}"), m0)
                    S.put(p(f"zv{nm}{q}"), v0)
                S.put(p(f"mA{q}"), np.ones_like(A))
                S.put(p(f"mB{q}"), mask)
    if train:
        S.put(p("sched"), np.array(schedule(steps, lr_fn, b1, b2, first_step)).reshape(-1))

    def kernel(name, nv=False, **sched):
        """The kernel file with its apple-m-series schedule replaced by `sched` (and, for a tile that is not a multiple of the CUDA atoms, the nvidia-ada one by a single warp over the same block, which is
        what the CPU interpreter's `--target nvidia-ada` runs in the tests)."""
        src = next(q for q in (p(name), os.path.join(HERE, name), os.path.join(ROOT, "examples", "train", name), os.path.join(ROOT, "tests", "chx", "ported", name), os.path.join(ROOT, "tests", "chx", name)) if os.path.exists(q))
        muts = [(a, b) for (nm, a, b) in tn.get("kmut", []) if nm == name]  # the tests build sessions with one deliberate bug in a kernel
        if not sched and not muts:
            return src
        text = open(src).read()
        for a, b in muts:
            assert a in text, (name, a)
            text = text.replace(a, b)
        out = []
        for line in text.split("\n"):
            if sched and line.startswith("schedule ") and "apple-m-series" in line:
                line = line.split("{")[0] + "{ " + " ".join(f"{k} {v_}" for k, v_ in sched.items()) + " }"
            elif sched and nv and line.startswith("schedule ") and "nvidia-ada" in line:
                line = line.split("{")[0] + "{ threads 32 warps 1 1 " + " ".join(f"{k} {v_}" for k, v_ in sched.items() if k in ("bm", "bn", "bk")) + " }"
            out.append(line)
        dst = p(name.replace(".chx", "_" + "_".join(str(x) for x in sched.values()).replace(" ", "") + ("_m" if muts else "") + ".chx"))
        open(dst, "w").write("\n".join(out))
        return dst

    L = [f"# Qwen2 LoRA training: {ll} layers, d {d}, {h} query heads over {hkv}, ff {ff}, vocabulary {v}, {nb} sequences of {t}, rank {r}, alpha {alpha}, {nsteps} batch(es), loss scale {loss_scale}"]
    prog = lambda name, kern, **kw: L.append(f"program {name} {kern} " + " ".join(f"{k}={v_}" for k, v_ in kw.items()))
    res = lambda target, path, flag="f32": L.append(f"resident {target} {path} {flag}")
    link = lambda a, b: L.append(f"link {a} -> {b}")
    carry = lambda a, b: L.append(f"carry {a} -> {b}")
    tab = lambda target, path, blocks, flag="f32": L.append(f"table {target} {path} {blocks} {flag}")
    eps = repr(c.eps)

    def sk(kind, **base_):
        """The schedule of a product with a 16 to 48 wide side; `tune['sk_<kind>']` ({'bm': .., 'bn': .., 'warps': ..}) overrides it."""
        out = dict(base_)
        out.update(tn.get("sk_" + kind, {}))
        return out

    gemm_s = lambda m, nn_, kk: tn["gemm_sched"].get((m, nn_, kk), {})
    gemm_f = lambda m, nn_, kk: kernel("gemm_mma.chx", **gemm_s(m, nn_, kk))

    def base(name, a, wfile, m, nn_, kk):
        prog(name, gemm_f(m, nn_, kk), m=m, n=nn_, k=kk)
        link(a, f"{name}.a")
        res(f"{name}.w", wfile, "f16")

    dims = {"wqkv": (d + 2 * dkv, d), "wo": (d, d), "wgu": (2 * ff, d), "wd": (d, ff)}
    if quantised:
        open(p("dq_fp4.chx"), "w").write(S.dq_text(QG))

    def dq_emit(l_, mat, phase):
        """The program that dequantises matrix `mat` of layer l_ into its f16 scratch (phase "f" for the forward, "b" for the backward), declared before the product that reads it; None for an f16 base."""
        if not quantised:
            return None
        n_, k_ = dims[mat]
        nm = f"dq{phase}_{mat}_{l_}"
        tl = S.dq_tile(n_, k_, QG, tn)
        prog(nm, kernel("dq_fp4.chx"), n=n_, k=k_, bn=tl["bn"], bk=tl["bk"])
        for part, ext, flag in (("wq", "q", "i8"), ("ws", "s", "i8"), ("wg", "g", "f32")):
            res(f"{nm}.{part}", w(f"l{l_:02d}.{mat}.{ext}"), flag)
        return nm

    def w_supply(target, l_, mat, nm):
        if nm is None:
            res(target, w(f"l{l_:02d}.{mat}.f16"), "f16")
        else:
            link(f"{nm}.wh", target)

    ahf = {}   # f16 users of each fused A / s B copy, for the optimiser's carry
    bhf = {}
    # ---- the embeddings of the step
    prog("wd", kernel("widen_h.chx"), r=n, c=d, bm=tn["ebm"])
    tab("wd.x", p("x0h"), nsteps, "f16")
    x_in = "wd.y"
    xs = [x_in]
    for l in range(ll):
        s = str(l)
        q = f"l{l:02d}."
        prog("rm1_" + s, kernel("rms_h.chx"), r=n, c=d, bm=tn["ebm"], eps=eps)
        link(x_in, f"rm1_{s}.x")
        res(f"rm1_{s}.gain", w(q + "ln1.f32"))

        def lora_mid(gname, a_src, cols_in):
            gd = G[gname]
            k = f"{l:02d}_{gname}"
            m = f"m{gname}_{s}"
            prog(m, kernel("gemm_mma.chx", nv=True, **sk("nt", threads=128, warps="2 2", bm=16, bn=gd["r2"], bk=32)), m=n, n=gd["r2"], k=cols_in)
            link(a_src, f"{m}.a")
            res(f"{m}.w", p(f"Ah{k}"), "f16")
            ahf.setdefault(k, []).append(f"{m}.w")
            prog(f"c{gname}_{s}", kernel("cast_h.chx"), r=n, c=gd["r2"], bm=8)
            link(f"{m}.y", f"c{gname}_{s}.x")

        def lora_gemm(name, gname, a_src, mat, nout, kin, kern="gemm_lora.chx"):
            gd = G[gname]
            k = f"{l:02d}_{gname}"
            nmq = dq_emit(l, mat, "f")
            prog(name, kernel(kern), m=n, n=nout, k=kin, r2=gd["r2"])
            link(a_src, f"{name}.a")
            w_supply(f"{name}.w", l, mat, nmq)
            link(f"c{gname}_{s}.y", f"{name}.mid")
            res(f"{name}.bs", p(f"Bh{k}"), "f16")
            bhf.setdefault(k, []).append(f"{name}.bs")

        lora_mid("qkv", f"rm1_{s}.y", d)
        lora_gemm("qkv_" + s, "qkv", f"rm1_{s}.y", "wqkv", d + 2 * dkv, d)
        for nm, kern, nh_, c0 in (("rq", "rope_q.chx", h, 0), ("rk", "rope_k.chx", hkv, d)):
            prog(f"{nm}_{s}", p(kern), nb=nb, nh=nh_, t=t, w=d + 2 * dkv, c0=c0, bm=tn["rbm"], hf=hf)
            link(f"qkv_{s}.y", f"{nm}_{s}.qkv")
            res(f"{nm}_{s}.bias", w(q + "bqkv.f32"))
            res(f"{nm}_{s}.cs", p("cos"))
            res(f"{nm}_{s}.sn", p("sin"))
        prog("rv_" + s, p("bias_v.chx"), nb=nb, nh=hkv, t=t, w=d + 2 * dkv, c0=d + dkv, bm=tn["rbm"], hf=hf)
        link(f"qkv_{s}.y", f"rv_{s}.qkv")
        res(f"rv_{s}.bias", w(q + "bqkv.f32"))
        prog("sc_" + s, kernel("att_s.chx", **tn["attn"]), nbh=nb * h, t=t, dh=dh)
        link(f"rq_{s}.y", f"sc_{s}.q")
        link(f"rk_{s}.y", f"sc_{s}.k")
        prog("sm_" + s, kernel("att_sm.chx"), nbh=nb * h, t=t, bm=tn["abm"], scale=repr(1.0 / math.sqrt(dh)))
        link(f"sc_{s}.s", f"sm_{s}.s")
        res(f"sm_{s}.mask", p("mask"))
        prog("ao_" + s, kernel("att_o.chx", **tn["ao"]), nbh=nb * h, t=t, dh=dh)
        link(f"sm_{s}.p", f"ao_{s}.pr")
        link(f"rv_{s}.y", f"ao_{s}.v")
        prog("uh_" + s, kernel("unheads.chx"), nb=nb, nh=h, t=t, dh=dh, bm=tn["rbm"])
        link(f"ao_{s}.o", f"uh_{s}.o")
        lora_mid("o", f"uh_{s}.y", d)
        lora_gemm("wo_" + s, "o", f"uh_{s}.y", "wo", d, d)
        prog("r1_" + s, kernel("res_add.chx"), r=n, c=d, bm=tn["ebm"])
        link(f"wo_{s}.y", f"r1_{s}.z")
        link(x_in, f"r1_{s}.res")
        prog("rm2_" + s, kernel("rms_h.chx"), r=n, c=d, bm=tn["ebm"], eps=eps)
        link(f"r1_{s}.y", f"rm2_{s}.x")
        res(f"rm2_{s}.gain", w(q + "ln2.f32"))
        lora_mid("gu", f"rm2_{s}.y", d)
        lora_gemm("gu_" + s, "gu", f"rm2_{s}.y", "wgu", 2 * ff, d)
        prog("sw_" + s, kernel("swiglu_h2.chx" if lean else "swiglu_h.chx"), r=n, f=ff, bm=tn["sbm"], bc=tn["sbc"])
        link(f"gu_{s}.y", f"sw_{s}.gu")
        lora_mid("dn", f"sw_{s}.y", ff)
        lora_gemm("dn_" + s, "dn", f"sw_{s}.y", "wd", d, ff)
        prog("r2_" + s, kernel("res_add.chx"), r=n, c=d, bm=tn["ebm"])
        link(f"dn_{s}.y", f"r2_{s}.z")
        link(f"r1_{s}.y", f"r2_{s}.res")
        x_in = f"r2_{s}.y"
        xs.append(x_in)
    prog("rmf", kernel("rms_h.chx"), r=n, c=d, bm=tn["ebm"], eps=eps)
    link(x_in, "rmf.x")
    res("rmf.gain", w("norm.f32"))
    base("lm", "rmf.y", wlm, n, vp, d)
    nq = vp // bc
    prog("cs", kernel("ce_stats.chx"), r=n, c=vp, bc=bc, nq=nq)
    link("lm.y", "cs.lg")
    tab("cs.tgt", p("tgt"), nsteps)
    res("cs.jrow", p("jrow"))
    res("cs.maskv", p("maskv"))
    prog("cc", kernel("ce_comb.chx"), r=n, nq=nq, bm=tn["cbm"])
    for o in ("cm", "cs", "ct", "ci"):
        link(f"cs.{o}", f"cc.{o}")
    prog("ls", kernel("bwd_dw.chx"), r=n, n=1, k=1)
    link("cc.loss", "ls.dyt")
    res("ls.xt", p("wcount"))
    # ---- the backward: the loss gradient and the head
    prog("cg", kernel("ce_grad.chx"), r=n, c=vp, bc=bc, nq=nq)
    link("lm.y", "cg.lg")
    tab("cg.tgt", p("tgt"), nsteps)
    res("cg.jrow", p("jrow"))
    res("cg.maskv", p("maskv"))
    link("cc.lse", "cg.lse")
    res("cg.wt", p("wscale"))
    prog("hd", kernel("gemm_mma_nn.chx", **gemm_s(n, d, vp)), m=n, n=d, k=vp)
    link("cg.dl", "hd.a")
    res("hd.w", wlm, "f16")
    prog("rmfb", kernel("rms_bwd.chx"), r=n, c=d, bm=tn["ebm"], eps=eps)
    link("hd.y", "rmfb.dy")
    link(xs[ll], "rmfb.x")
    res("rmfb.gain", w("norm.f32"))
    res("rmfb.dres", p("zeros_nd"))
    gdx = "rmfb.dx"
    gradout = {}   # (l, group) -> (gA output, gB output)
    for l in reversed(range(ll)):
        s = str(l)
        q = f"l{l:02d}."

        # --- down projection
        prog("gc_" + s, kernel("cast_h.chx"), r=n, c=d, bm=tn["ebm"])
        link(gdx, f"gc_{s}.x")

        def adapter_back(gname, dy16, x16, nout, nin, mat):
            """dmid, dx = dy W + dmid A (the name of its output), dB, dA for the fused group; `dy16`/`x16` are 'prog.out' names; the weight file is filled by the caller."""
            gd = G[gname]
            k = f"{l:02d}_{gname}"
            r2 = gd["r2"]
            dm = f"dm{gname}_{s}"
            prog(dm, kernel("gemm_mma_nn.chx", nv=True, **sk("nn", threads=128, warps="2 2", bm=16, bn=r2, bk=32)), m=n, n=r2, k=nout)
            link(dy16, f"{dm}.a")
            res(f"{dm}.w", p(f"Bh{k}"), "f16")
            bhf[k].append(f"{dm}.w")
            prog(f"dc{gname}_{s}", kernel("cast_h.chx"), r=n, c=r2, bm=8)
            link(f"{dm}.y", f"dc{gname}_{s}.x")
            dxn = f"dx{gname}_{s}"
            nmq = dq_emit(l, mat, "b")
            prog(dxn, kernel("dx_lora.chx"), m=n, n=nin, k=nout, r2=r2)
            link(dy16, f"{dxn}.dy")
            link(f"dc{gname}_{s}.y", f"{dxn}.dmid")
            w_supply(f"{dxn}.w", l, mat, nmq)
            res(f"{dxn}.a", p(f"Ah{k}"), "f16")
            ahf[k].append(f"{dxn}.a")
            gb, ga = f"gB{gname}_{s}", f"gA{gname}_{s}"
            prog(gb, kernel("gemm_mma_tn.chx", nv=True, **sk("tnb", threads=128, warps="2 2", bm=16, bn=r2, bk=32)), m=nout, n=r2, k=n)
            link(dy16, f"{gb}.a")
            link(f"c{gname}_{s}.y", f"{gb}.w")
            prog(ga, kernel("gemm_mma_tn.chx", nv=True, **sk("tna", threads=128, warps="2 2", bm=r2, bn=16, bk=32)), m=r2, n=nin, k=n)
            link(f"dc{gname}_{s}.y", f"{ga}.a")
            link(x16, f"{ga}.w")
            gradout[(l, gname)] = (f"{ga}.y", f"{gb}.y")
            return dxn

        sw_in = f"sw_{s}.y"
        if lean:
            prog("swr_" + s, kernel("swiglu_h16.chx"), r=n, f=ff, bm=tn["sbm"], bc=tn["sbc"])
            link(f"sw_{s}.gh", f"swr_{s}.gu")
            sw_in = f"swr_{s}.y"
        dxn = adapter_back("dn", f"gc_{s}.y", sw_in, d, ff, "wd")
        prog("sb_" + s, kernel("swiglu_bwd16.chx" if lean else "swiglu_bwd.chx"), r=n, f=ff, bm=tn["sbm"], bc=tn["sbc"])
        link(f"{dxn}.y", f"sb_{s}.da")
        link(f"sw_{s}.gh" if lean else f"gu_{s}.y", f"sb_{s}.gu")
        dxn = adapter_back("gu", f"sb_{s}.dgu", f"rm2_{s}.y", 2 * ff, d, "wgu")
        prog("rm2b_" + s, kernel("rms_bwd.chx"), r=n, c=d, bm=tn["ebm"], eps=eps)
        link(f"{dxn}.y", f"rm2b_{s}.dy")
        link(f"r1_{s}.y", f"rm2b_{s}.x")
        res(f"rm2b_{s}.gain", w(q + "ln2.f32"))
        link(gdx, f"rm2b_{s}.dres")
        ga_ = f"rm2b_{s}.dx"
        prog("gac_" + s, kernel("cast_h.chx"), r=n, c=d, bm=tn["ebm"])
        link(ga_, f"gac_{s}.x")
        dxn = adapter_back("o", f"gac_{s}.y", f"uh_{s}.y", d, d, "wo")
        prog("hdo_" + s, kernel("heads_h.chx"), nb=nb, nh=h, t=t, dh=dh, bm=tn["rbm"])
        link(f"{dxn}.y", f"hdo_{s}.y")
        prog("dpp_" + s, kernel("att_s.chx", **tn["attn"]), nbh=nb * h, t=t, dh=dh)
        link(f"hdo_{s}.o", f"dpp_{s}.q")
        link(f"rv_{s}.y", f"dpp_{s}.k")
        prog("dsf_" + s, kernel("att_dsoft.chx"), nbh=nb * h, t=t, bm=tn["abm"], scale=repr(1.0 / math.sqrt(dh)))
        link(f"dpp_{s}.s", f"dsf_{s}.dp")
        link(f"sm_{s}.p", f"dsf_{s}.p")
        prog("dqh_" + s, kernel("att_o.chx", **tn["ao"]), nbh=nb * h, t=t, dh=dh)
        link(f"dsf_{s}.ds", f"dqh_{s}.pr")
        link(f"rk_{s}.y", f"dqh_{s}.v")
        prog("dkk_" + s, kernel("att_dkv.chx"), nb=nb, nh=h, hkv=hkv, t=t, dh=dh)
        link(f"dsf_{s}.ds", f"dkk_{s}.ds")
        link(f"rq_{s}.y", f"dkk_{s}.q")
        prog("dvv_" + s, kernel("att_dkv.chx"), nb=nb, nh=h, hkv=hkv, t=t, dh=dh)
        link(f"sm_{s}.p", f"dvv_{s}.ds")
        link(f"hdo_{s}.o", f"dvv_{s}.q")
        prog("rb_" + s, p("rope_bwd.chx"), nb=nb, t=t, bm=tn["rbm"], hf=hf)
        link(f"dqh_{s}.o", f"rb_{s}.dq")
        link(f"dkk_{s}.dk", f"rb_{s}.dk")
        link(f"dvv_{s}.dk", f"rb_{s}.dv")
        res(f"rb_{s}.cs", p("cos"))
        res(f"rb_{s}.sn", p("sin"))
        dxn = adapter_back("qkv", f"rb_{s}.y", f"rm1_{s}.y", d + 2 * dkv, d, "wqkv")
        prog("rm1b_" + s, kernel("rms_bwd.chx"), r=n, c=d, bm=tn["ebm"], eps=eps)
        link(f"{dxn}.y", f"rm1b_{s}.dy")
        link(xs[l], f"rm1b_{s}.x")
        res(f"rm1b_{s}.gain", w(q + "ln1.f32"))
        link(ga_, f"rm1b_{s}.dres")
        gdx = f"rm1b_{s}.dx"
    # ---- the optimiser
    if train:
        gs = 1.0 / (ncount * loss_scale)
        for l in range(ll):
            for g in G:
                k = f"{l:02d}_{g}"
                sizes = {"A": G[g]["r2"] * G[g]["inp"], "B": G[g]["N"] * G[g]["r2"]}
                for ab in ("A", "B"):
                    nm = f"a{ab}{k}"
                    nel = sizes[ab]
                    prog(nm, kernel("adamw_l.chx"), n=nel, bn=S.block(nel, 128), gs=repr(gs * (scale if ab == "B" else 1.0)), b1=b1, b2=b2, eps=eps_adam, lwd=repr(lr * wd), osc=repr(scale if ab == "B" else 1.0))
                    res(f"{nm}.w", p(f"{ab}{k}"))
                    res(f"{nm}.m", p(f"zm{ab}{k}"))
                    res(f"{nm}.v", p(f"zv{ab}{k}"))
                    res(f"{nm}.mk", p(f"m{ab}{k}"))
                    tab(f"{nm}.s", p("sched"), steps)
                    link(gradout[(l, g)][0 if ab == "A" else 1], f"{nm}.grad")
                    carry(f"{nm}.wo", f"{nm}.w")
                    carry(f"{nm}.mo", f"{nm}.m")
                    carry(f"{nm}.vo", f"{nm}.v")
                    for u in (ahf if ab == "A" else bhf)[k]:
                        carry(f"{nm}.wh", u)
    if reuse:
        # What a layer computes and uses within the layer only (and, in the backward, everything but the adapters' gradients) lives in one buffer for all layers.
        fwd = ["qkv_{}.y", "sc_{}.s", "ao_{}.o", "wo_{}.y", "dn_{}.y"]
        bwd = ["gc_{}.y", "dxdn_{}.y", "sb_{}.dgu", "dxgu_{}.y", "rm2b_{}.dx", "gac_{}.y", "dxo_{}.y", "hdo_{}.o", "dpp_{}.s", "dsf_{}.ds", "dqh_{}.o", "dkk_{}.dk", "dvv_{}.dk", "rb_{}.y",
               "dxqkv_{}.y", "rm1b_{}.dx"]
        if lean:
            fwd += ["sw_{}.y", "gu_{}.y"]
            bwd.append("swr_{}.y")
        for slot in fwd:
            for l in range(1, ll):
                L.append(f"alias {slot.format(l - 1)} {slot.format(l)}")
        if quantised:
            # the scratch of a matrix kind: one buffer through the forward (layer 0 up) and then the backward (layer ll - 1 down)
            for mat in dims:
                for l in range(1, ll):
                    L.append(f"alias dqf_{mat}_{l - 1}.wh dqf_{mat}_{l}.wh")
                L.append(f"alias dqf_{mat}_{ll - 1}.wh dqb_{mat}_{ll - 1}.wh")
                for l in reversed(range(ll - 1)):
                    L.append(f"alias dqb_{mat}_{l + 1}.wh dqb_{mat}_{l}.wh")
        for slot in bwd:
            for l in reversed(range(ll - 1)):
                L.append(f"alias {slot.format(l + 1)} {slot.format(l)}")
    L.append("steps")
    L.append(f"answer ls.dw scale 1/{ncount:g}")
    for pr in probes:
        L.append(f"answer {pr}")
    if grads:
        for l in range(ll):
            for g in G:
                L.append(f"save {gradout[(l, g)][0]} {os.path.abspath(grads)} f32")
                L.append(f"save {gradout[(l, g)][1]} {os.path.abspath(grads)} f32")
    if save and train:
        for l in range(ll):
            for g in G:
                for ab in ("A", "B"):
                    L.append(f"save a{ab}{l:02d}_{g}.wo {os.path.abspath(save)} f32")
    if save_state and train:
        for o, ext in (("mo", "m"), ("vo", "v")):
            for l in range(ll):
                for g in G:
                    for ab in ("A", "B"):
                        L.append(f"save a{ab}{l:02d}_{g}.{o} {os.path.abspath(save_state)}.{ext} f32")
    open(p("session"), "w").write("\n".join(L) + "\n")
    return p("session")


def read_fused(c, r, path):
    """The f32 file `save` wrote for the gradients (A then B of each group, layer by layer) or for the adapters, as {(l, group, 'A'|'B'): array}."""
    flat = np.fromfile(path, dtype="<f4").astype(np.float64)
    G = group_dims(c, r)
    out, o = {}, 0
    for l in range(c.layers):
        for g, gd in G.items():
            for ab, shape in (("A", (gd["r2"], gd["inp"])), ("B", (gd["N"], gd["r2"]))):
                nel = shape[0] * shape[1]
                out[(l, g, ab)] = flat[o:o + nel].reshape(shape)
                o += nel
    if o != flat.size:
        raise SystemExit(f"{path}: {flat.size} words, the model has {o}")
    return out


def split_grads(c, r, raw, ncount, loss_scale, alpha=32.0):
    """The per-projection gradients ({'l00.q.A': [r, in], ...}) from the raw fused outputs: dA = (dy Bs)^T x needs only the loss scale and the count out; dB = s dy^T mid needs s as well."""
    s = alpha / r
    out = {}
    for l in range(c.layers):
        fused = {g: (raw[(l, g, "A")] / (ncount * loss_scale), raw[(l, g, "B")] * s / (ncount * loss_scale)) for g in group_dims(c, r)}
        out.update(split_params(c, r, fused, l))
    return out
