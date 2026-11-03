#!/usr/bin/env python3
"""Full fine-tuning of Qwen2 in a session of .chx kernels (docs/resident.md section 24; roadmap M7.4, part B): every parameter is trained, the tied embedding table included.
qwen_ref.py (`full_loss_grads`) is the specification (binary64, hand-written backward, checked by finite differences).

    python3 examples/qwen/qwen_full.py WEIGHTS_DIR SESSION_DIR      # writes the session of the fixed batch

What is new against qwen_train.py (LoRA):
* **The embedding lookup and its gradient need no gather and no atomics.** The ids of a step are an f32 `table`; `onehot_h` builds the f16 matrix `Y [tokens, vocabulary]` (a 1 where the column is the
  row's id) and the lookup is the product `Y E` (the same shape as the output layer's input gradient, `gemm_mma_nn`), the scatter-add of the gradient into the table rows is `Y^T dx` (`gemm_mma_tn`, the
  shape of the output layer's own weight gradient); the table's gradient is that plus the output layer's `dlogits^T h`, and `adamw_e` adds the two. With a vocabulary of 152 k these cost what the
  output layer costs (section 24.2).
* **Every weight has a gradient**: `dW = dy^T x` is `gemm_mma_tn` (f16 operands, f32 accumulation), the gains' and biases' are column sums (`rms_dg`, `colsum`, `colsum_h`).
* **The optimiser runs inside the backward**, right after the layer's last use of the weight (`adamw_f`: f32 master, two f32 moments, the f16 copy the matrix unit reads, carried to every program
  that reads it), so a layer's weight gradients live in buffers shared by all layers (`alias`); the table's two gradients wait for the end of the backward.
Mixed precision as in qwen_session.py and qwen_train.py: f16 weights and f16 operands of every product, f32 accumulation, f32 residual and gradient streams, f32 masters and moments."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import qwen_ref as R  # noqa: E402
import qwen_session as S  # noqa: E402
import qwen_train as T  # noqa: E402

def adam_block(n, cap):
    """The elements one threadgroup of the optimiser takes: the largest divisor of n up to cap that is a multiple of the 128 threads."""
    return next(b for b in range(min(n, cap) // 128 * 128, 0, -128) if n % b == 0)


def master_files(directory, c, wdir, vp):
    """The f32 masters of the matrices (from the f16 files the forward reads: the run starts from exactly those values) and the table's padded to `vp` rows; zero files for the moments, one for each size."""
    os.makedirs(directory, exist_ok=True)
    out, zeros = {}, {}

    def zfile(n):
        if n not in zeros:
            path = os.path.join(directory, f"zeros{n}")
            if not os.path.exists(path):
                with open(path, "wb") as f:
                    left = n * 4
                    z = bytes(1 << 24)
                    while left > 0:
                        f.write(z[:min(left, len(z))])
                        left -= len(z)
            zeros[n] = path
        return zeros[n]

    for name in R.names(c):
        kind = name.split(".")[-1]
        if kind in ("norm", "ln1", "ln2", "bqkv"):
            out[name] = (os.path.abspath(os.path.join(wdir, name + ".f32")), None)
            continue
        a = np.fromfile(os.path.join(wdir, name + ".f16"), dtype="<f2")
        if name == "emb":
            a = a[:c.v * c.d]
            a = np.concatenate([a, np.zeros((vp - c.v) * c.d, dtype="<f2")])
        path = os.path.join(directory, name + ".m32")
        a.astype("<f4").tofile(path)
        out[name] = (path, None)
    return out, zfile


STATE_EXT = (("w32", "wo"), ("m", "mo"), ("v", "vo"))


def state_paths(state, pname):
    """The three f32 files of one parameter in a saved-state directory: its master, first and second moment. A missing one is a refusal naming it (nothing is guessed: a zero moment would run, and be wrong)."""
    out = []
    for ext, _ in STATE_EXT:
        path = os.path.abspath(os.path.join(state, f"{pname}.{ext}"))
        if not os.path.exists(path):
            raise SystemExit(f"error[usage]: the state directory {state} has no {pname}.{ext} (a run is resumed from the directory a `save_state` run wrote, whole)")
        out.append(path)
    return out


def write(directory, c, wdir, tokens, steps=1, lr=1e-5, lr_fn=None, wd=0.0, betas=(0.9, 0.999), eps_adam=1e-8, loss_scale=16.0, save=None, tune=None, reuse=True, kmut=(), grads=None, state=None,
          first_step=0, save_state=None):
    """Writes the session over the batches `tokens` ([S, nb, t], one per step, taken in order; a request is a count of steps) from the f16 weights of `wdir`; answers the path of the session file.
    The answer is the mean loss of the last step's batch. With `save` (a directory) the request `checkpoint` writes the trained parameters there as a weights directory (f16 matrices, f32 vectors; what
    qwen_session.write and convert.py's readers take), so that its forward pass can be measured. With `grads` (a path) the raw f32 gradient of every parameter (the table's as its two parts) is saved there on
    `checkpoint` in the order of `grad_names` (the weight-gradient buffers are then not shared between layers).
    A run in several sessions (docs/resident.md section 24.7): with `save_state` (a directory) `checkpoint` also writes, for every parameter, the f32 master (`NAME.w32`; the table's padded to its
    chunked row count), the first (`NAME.m`) and the second moment (`NAME.v`), and the next session starts from them with `state` (that directory) and `first_step` (the steps already taken, which the
    bias correction and the learning-rate schedule count from); `wdir` of the next session is the `save` directory of the last, whose f16 matrices are the roundings of those masters. The batches
    `tokens` of a continued session are the ones after the steps already taken."""
    assert c.tied, "full fine-tuning is built for a tied embedding table (docs/resident.md section 24); an untied head (docs/resident.md section 28) is trained with adapters only"
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
    assert dh % 32 == 0 and d % 64 == 0 and t % 64 == 0 and n % 64 == 0 and (d + 2 * dkv) % 64 == 0 and ff % 64 == 0 and h % hkv == 0 and dh == d // h
    lr_fn = lr_fn or (lambda s_: lr)
    b1, b2 = betas
    tn = dict(bc=2304, ebm=1, abm=2, rbm=8, cbm=8, sbm=1, sbc=S.block(ff, 256), attn=dict(threads=128, warps="2 2", bm=64, bn=64, bk=32), ao=dict(threads=128, warps="2 2", bm=64 if dh <= 64 else 32, bk=32),
              gemm_sched={}, col=dict(bm=64, bn=64), adam_bn=1024)
    tn.update(tune or {})
    S.put(p("ids"), np.stack([tk.reshape(-1) + 1 for tk in tokens]))
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
    emb = np.fromfile(w("emb.f16"), dtype="<f2").reshape(-1, d)[:v]
    wlm = p("wlm.f16")
    with open(wlm, "wb") as f:
        f.write(emb.tobytes())
        f.write(np.zeros((vp - v, d), dtype="<f2").tobytes())
    S.put(p("jrow"), np.arange(1, vp + 1))
    S.put(p("maskv"), np.where(np.arange(vp) < v, 0.0, -1e9))
    S.put(p("ones_n"), np.ones((n, 16)), "<f2")
    for name, src in (("rope_q.chx", S.rope_source("rope_q", 1, True)), ("rope_k.chx", S.rope_source("rope_k", grp, True)), ("bias_v.chx", S.rope_source("bias_v", grp, False)),
                      ("rope_bwd.chx", T.rope_bwd_source(c, tn["rbm"]))):
        for nm, a, b in kmut:
            if nm == name:
                assert a in src, (name, a)
                src = src.replace(a, b)
        open(p(name), "w").write(src)
    masters, zfile = master_files(os.path.join(directory, "master"), c, wdir, vp) if not state else ({}, None)
    sched = np.array(T.schedule(steps, lr_fn, b1, b2, first_step)).reshape(-1)
    S.put(p("sched"), sched)

    def kernel(name, nv=False, **sched_):
        src = next(q for q in (p(name), os.path.join(HERE, name), os.path.join(ROOT, "examples", "train", name), os.path.join(ROOT, "tests", "chx", "ported", name), os.path.join(ROOT, "tests", "chx", name))
                   if os.path.exists(q))
        muts = [(a, b) for (nm, a, b) in kmut if nm == name]
        if not sched_ and not muts:
            return src
        text = open(src).read()
        for a, b in muts:
            assert a in text, (name, a)
            text = text.replace(a, b)
        out = []
        for line in text.split("\n"):
            if sched_ and line.startswith("schedule ") and "apple-m-series" in line:
                line = line.split("{")[0] + "{ " + " ".join(f"{k} {v_}" for k, v_ in sched_.items()) + " }"
            elif sched_ and nv and line.startswith("schedule ") and "nvidia-ada" in line:
                line = line.split("{")[0] + "{ threads 32 warps 1 1 " + " ".join(f"{k} {v_}" for k, v_ in sched_.items() if k in ("bm", "bn", "bk")) + " }"
            out.append(line)
        dst = p(name.replace(".chx", "_" + "_".join(str(x) for x in sched_.values()).replace(" ", "") + ("_m" if muts else "") + ".chx"))
        open(dst, "w").write("\n".join(out))
        return dst

    L = [f"# Qwen2 full fine-tuning: {ll} layers, d {d}, {h} query heads over {hkv}, ff {ff}, vocabulary {v}, {nb} sequences of {t}, {nsteps} batch(es), loss scale {loss_scale}"]
    prog = lambda name, kern, **kw: L.append(f"program {name} {kern} " + " ".join(f"{k}={v_}" for k, v_ in kw.items()))
    res = lambda target, path, flag="f32": L.append(f"resident {target} {path} {flag}")
    link = lambda a, b: L.append(f"link {a} -> {b}")
    carry = lambda a, b: L.append(f"carry {a} -> {b}")
    tab = lambda target, path, blocks, flag="f32": L.append(f"table {target} {path} {blocks} {flag}")
    eps = repr(c.eps)
    gemm_s = lambda m, nn_, kk: tn["gemm_sched"].get((m, nn_, kk), {})
    users = {}     # parameter name -> the 'program.param' names that read its f16 copy (or f32 value) and must receive the optimiser's output
    adams = []     # (name, program) in order, for the saves

    def user(pname, target):
        users.setdefault(pname, []).append(target)

    def fwd_gemm(name, a, pname, m, nn_, kk):
        prog(name, kernel("gemm_mma.chx", **gemm_s(m, nn_, kk)), m=m, n=nn_, k=kk)
        link(a, f"{name}.a")
        res(f"{name}.w", wlm if pname == "emb" else w(pname + ".f16"), "f16")
        user(pname, f"{name}.w")

    def bwd_nn(name, a, pname, m, nn_, kk):
        prog(name, kernel("gemm_mma_nn.chx", **gemm_s(m, nn_, kk)), m=m, n=nn_, k=kk)
        link(a, f"{name}.a")
        res(f"{name}.w", wlm if pname == "emb" else w(pname + ".f16"), "f16")
        user(pname, f"{name}.w")

    def wgrad(name, dy16, x16, mm, nn_):
        prog(name, kernel("gemm_mma_tn.chx", **gemm_s(mm, nn_, n)), m=mm, n=nn_, k=n)
        link(dy16, f"{name}.a")
        link(x16, f"{name}.w")
        return f"{name}.y"

    def adam(pname, grad, grad2=None, flat=None):
        """AdamW for the parameter file `pname`: the master in f32 (the f32 file itself for the vectors), the moments from a zero file, the gradient from the output `grad`."""
        if state:
            mfile, mom_m, mom_v = state_paths(state, pname)
        else:
            mfile, mom_m, mom_v = masters[pname][0], zfile(flat), zfile(flat)
        nel = flat
        nm = "ad_" + pname.replace(".", "_")
        kern = "adamw_e.chx" if grad2 else "adamw_f.chx"
        prog(nm, kernel(kern), n=nel, bn=adam_block(nel, tn["adam_bn"]), gs=repr(1.0 / (ncount * loss_scale)), b1=b1, b2=b2, eps=eps_adam, lwd=repr(lr * wd))
        res(f"{nm}.w", mfile)
        res(f"{nm}.m", mom_m)
        res(f"{nm}.v", mom_v)
        link(grad, f"{nm}.grad")
        if grad2:
            link(grad2, f"{nm}.grad2")
        tab(f"{nm}.s", p("sched"), steps)
        carry(f"{nm}.wo", f"{nm}.w")
        carry(f"{nm}.mo", f"{nm}.m")
        carry(f"{nm}.vo", f"{nm}.v")
        kind = pname.split(".")[-1]
        if kind in ("norm", "ln1", "ln2", "bqkv"):
            for u in users.get(pname, []):
                carry(f"{nm}.wo", u)
        else:
            for u in users.get(pname, []):
                carry(f"{nm}.wh", u)
        adams.append((pname, nm, grad, grad2))

    # ---- the lookup: the one-hot matrix of the step's ids times the table
    prog("oh", kernel("onehot_h.chx"), r=n, c=vp, bc=bc)
    tab("oh.ids", p("ids"), nsteps)
    res("oh.jrow", p("jrow"))
    prog("g0", kernel("gemm_mma_nn.chx", **gemm_s(n, d, vp)), m=n, n=d, k=vp)
    link("oh.y", "g0.a")
    res("g0.w", wlm, "f16")
    user("emb", "g0.w")
    x_in = "g0.y"
    xs = [x_in]
    for l in range(ll):
        s = str(l)
        q = f"l{l:02d}."
        prog("rm1_" + s, kernel("rms_h.chx"), r=n, c=d, bm=tn["ebm"], eps=eps)
        link(x_in, f"rm1_{s}.x")
        res(f"rm1_{s}.gain", w(q + "ln1.f32"))
        user(q + "ln1", f"rm1_{s}.gain")
        fwd_gemm("qkv_" + s, f"rm1_{s}.y", q + "wqkv", n, d + 2 * dkv, d)
        for nm, kern, nh_, c0 in (("rq", "rope_q.chx", h, 0), ("rk", "rope_k.chx", hkv, d)):
            prog(f"{nm}_{s}", p(kern), nb=nb, nh=nh_, t=t, w=d + 2 * dkv, c0=c0, bm=tn["rbm"], hf=hf)
            link(f"qkv_{s}.y", f"{nm}_{s}.qkv")
            res(f"{nm}_{s}.bias", w(q + "bqkv.f32"))
            user(q + "bqkv", f"{nm}_{s}.bias")
            res(f"{nm}_{s}.cs", p("cos"))
            res(f"{nm}_{s}.sn", p("sin"))
        prog("rv_" + s, p("bias_v.chx"), nb=nb, nh=hkv, t=t, w=d + 2 * dkv, c0=d + dkv, bm=tn["rbm"], hf=hf)
        link(f"qkv_{s}.y", f"rv_{s}.qkv")
        res(f"rv_{s}.bias", w(q + "bqkv.f32"))
        user(q + "bqkv", f"rv_{s}.bias")
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
        fwd_gemm("wo_" + s, f"uh_{s}.y", q + "wo", n, d, d)
        prog("r1_" + s, kernel("res_add.chx"), r=n, c=d, bm=tn["ebm"])
        link(f"wo_{s}.y", f"r1_{s}.z")
        link(x_in, f"r1_{s}.res")
        prog("rm2_" + s, kernel("rms_h.chx"), r=n, c=d, bm=tn["ebm"], eps=eps)
        link(f"r1_{s}.y", f"rm2_{s}.x")
        res(f"rm2_{s}.gain", w(q + "ln2.f32"))
        user(q + "ln2", f"rm2_{s}.gain")
        fwd_gemm("gu_" + s, f"rm2_{s}.y", q + "wgu", n, 2 * ff, d)
        prog("sw_" + s, kernel("swiglu_h.chx"), r=n, f=ff, bm=tn["sbm"], bc=tn["sbc"])
        link(f"gu_{s}.y", f"sw_{s}.gu")
        fwd_gemm("dn_" + s, f"sw_{s}.y", q + "wd", n, d, ff)
        prog("r2_" + s, kernel("res_add.chx"), r=n, c=d, bm=tn["ebm"])
        link(f"dn_{s}.y", f"r2_{s}.z")
        link(f"r1_{s}.y", f"r2_{s}.res")
        x_in = f"r2_{s}.y"
        xs.append(x_in)
    prog("rmf", kernel("rms_h.chx"), r=n, c=d, bm=tn["ebm"], eps=eps)
    link(x_in, "rmf.x")
    res("rmf.gain", w("norm.f32"))
    user("norm", "rmf.gain")
    prog("lm", kernel("gemm_mma.chx", **gemm_s(n, vp, d)), m=n, n=vp, k=d)
    link("rmf.y", "lm.a")
    res("lm.w", wlm, "f16")
    user("emb", "lm.w")
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
    # ---- the backward: the loss gradient, the output layer's gradients, the final norm
    prog("cg", kernel("ce_grad.chx"), r=n, c=vp, bc=bc, nq=nq)
    link("lm.y", "cg.lg")
    tab("cg.tgt", p("tgt"), nsteps)
    res("cg.jrow", p("jrow"))
    res("cg.maskv", p("maskv"))
    link("cc.lse", "cg.lse")
    res("cg.wt", p("wscale"))
    bwd_nn("hd", "cg.dl", "emb", n, d, vp)
    gE_head = wgrad("gEh", "cg.dl", "rmf.y", vp, d)
    prog("rmfb", kernel("rms_bwd.chx"), r=n, c=d, bm=tn["ebm"], eps=eps)
    link("hd.y", "rmfb.dy")
    link(xs[ll], "rmfb.x")
    res("rmfb.gain", w("norm.f32"))
    user("norm", "rmfb.gain")
    res("rmfb.dres", p("zeros_nd"))

    def colgrad(tag, dy, x, pname):
        """The gain's gradient (column sum of dy x rs) and its AdamW step."""
        prog("dg_" + tag, kernel("rms_dg.chx"), r=n, c=d, bm=tn["ebm"], eps=eps)
        link(dy, f"dg_{tag}.dy")
        link(x, f"dg_{tag}.x")
        prog("cl_" + tag, kernel("colsum.chx"), r=n, c=d, **tn["col"])
        link(f"dg_{tag}.u", f"cl_{tag}.x")
        adam(pname, f"cl_{tag}.y", flat=d)

    colgrad("f", "hd.y", xs[ll], "norm")
    gdx = "rmfb.dx"
    wg_aliases = []
    for l in reversed(range(ll)):
        s = str(l)
        q = f"l{l:02d}."
        prog("gc_" + s, kernel("cast_h.chx"), r=n, c=d, bm=tn["ebm"])
        link(gdx, f"gc_{s}.x")
        # --- down projection
        bwd_nn("dxdn_" + s, f"gc_{s}.y", q + "wd", n, ff, d)
        g_ = wgrad("gd_" + s, f"gc_{s}.y", f"sw_{s}.y", d, ff)
        adam(q + "wd", g_, flat=d * ff)
        prog("sb_" + s, kernel("swiglu_bwd.chx"), r=n, f=ff, bm=tn["sbm"], bc=tn["sbc"])
        link(f"dxdn_{s}.y", f"sb_{s}.da")
        link(f"gu_{s}.y", f"sb_{s}.gu")
        # --- gate / up
        bwd_nn("dxgu_" + s, f"sb_{s}.dgu", q + "wgu", n, d, 2 * ff)
        g_ = wgrad("ggu_" + s, f"sb_{s}.dgu", f"rm2_{s}.y", 2 * ff, d)
        adam(q + "wgu", g_, flat=2 * ff * d)
        prog("rm2b_" + s, kernel("rms_bwd.chx"), r=n, c=d, bm=tn["ebm"], eps=eps)
        link(f"dxgu_{s}.y", f"rm2b_{s}.dy")
        link(f"r1_{s}.y", f"rm2b_{s}.x")
        res(f"rm2b_{s}.gain", w(q + "ln2.f32"))
        user(q + "ln2", f"rm2b_{s}.gain")
        link(gdx, f"rm2b_{s}.dres")
        colgrad("2_" + s, f"dxgu_{s}.y", f"r1_{s}.y", q + "ln2")
        ga_ = f"rm2b_{s}.dx"
        # --- o projection
        prog("gac_" + s, kernel("cast_h.chx"), r=n, c=d, bm=tn["ebm"])
        link(ga_, f"gac_{s}.x")
        bwd_nn("dxo_" + s, f"gac_{s}.y", q + "wo", n, d, d)
        g_ = wgrad("go_" + s, f"gac_{s}.y", f"uh_{s}.y", d, d)
        adam(q + "wo", g_, flat=d * d)
        prog("hdo_" + s, kernel("heads_h.chx"), nb=nb, nh=h, t=t, dh=dh, bm=tn["rbm"])
        link(f"dxo_{s}.y", f"hdo_{s}.y")
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
        # --- q/k/v projection and its bias
        bwd_nn("dxqkv_" + s, f"rb_{s}.y", q + "wqkv", n, d, d + 2 * dkv)
        g_ = wgrad("gqkv_" + s, f"rb_{s}.y", f"rm1_{s}.y", d + 2 * dkv, d)
        adam(q + "wqkv", g_, flat=(d + 2 * dkv) * d)
        prog("cb_" + s, kernel("colsum_h.chx"), r=n, c=d + 2 * dkv, **tn["col"])
        link(f"rb_{s}.y", f"cb_{s}.x")
        adam(q + "bqkv", f"cb_{s}.y", flat=d + 2 * dkv)
        prog("rm1b_" + s, kernel("rms_bwd.chx"), r=n, c=d, bm=tn["ebm"], eps=eps)
        link(f"dxqkv_{s}.y", f"rm1b_{s}.dy")
        link(xs[l], f"rm1b_{s}.x")
        res(f"rm1b_{s}.gain", w(q + "ln1.f32"))
        user(q + "ln1", f"rm1b_{s}.gain")
        link(ga_, f"rm1b_{s}.dres")
        colgrad("1_" + s, f"dxqkv_{s}.y", xs[l], q + "ln1")
        gdx = f"rm1b_{s}.dx"
    # ---- the table: the input side's scatter-add, then AdamW on the sum of the two gradients
    prog("dx0", kernel("cast_h.chx"), r=n, c=d, bm=tn["ebm"])
    link(gdx, "dx0.x")
    prog("oh2", kernel("onehot_h.chx"), r=n, c=vp, bc=bc)
    tab("oh2.ids", p("ids"), nsteps)
    res("oh2.jrow", p("jrow"))
    gE_in = wgrad("gEi", "oh2.y", "dx0.y", vp, d)
    adam("emb", gE_head, gE_in, flat=vp * d)
    if reuse:
        # the layer-local temporaries share one buffer across layers (qwen_train.write's list, plus the weight gradients, each consumed by its optimiser step before the next layer's)
        fwd = ["qkv_{}.y", "sc_{}.s", "ao_{}.o", "wo_{}.y", "dn_{}.y"]
        bwd = ["gc_{}.y", "dxdn_{}.y", "sb_{}.dgu", "dxgu_{}.y", "rm2b_{}.dx", "gac_{}.y", "dxo_{}.y", "hdo_{}.o", "dpp_{}.s", "dsf_{}.ds", "dqh_{}.o", "dkk_{}.dk", "dvv_{}.dk", "rb_{}.y",
               "dxqkv_{}.y", "rm1b_{}.dx", "dg_2_{}.u", "dg_1_{}.u"]
        if not grads:
            bwd += ["gd_{}.y", "ggu_{}.y", "go_{}.y", "gqkv_{}.y", "cl_2_{}.y", "cl_1_{}.y", "cb_{}.y"]
        for slot in fwd:
            for l in range(1, ll):
                L.append(f"alias {slot.format(l - 1)} {slot.format(l)}")
        for slot in bwd:
            for l in reversed(range(ll - 1)):
                L.append(f"alias {slot.format(l + 1)} {slot.format(l)}")
        # the first lookup's one-hot matrix, the output gradient and the second one-hot matrix are the same 1.2 GB in turn
        L.append("alias oh.y cg.dl")
        L.append("alias cg.dl oh2.y")
    L.append("steps")
    L.append(f"answer ls.dw scale 1/{ncount:g}")
    if save:
        os.makedirs(save, exist_ok=True)
        for pname, nm, _, _ in adams:
            kind = pname.split(".")[-1]
            if kind in ("norm", "ln1", "ln2", "bqkv"):
                L.append(f"save {nm}.wo {os.path.abspath(save)}/{pname}.f32 f32")
            else:
                L.append(f"save {nm}.wh {os.path.abspath(save)}/{pname}.f16 f16")
    if save_state:
        os.makedirs(save_state, exist_ok=True)
        for pname, nm, _, _ in adams:
            for ext, o in STATE_EXT:
                L.append(f"save {nm}.{o} {os.path.abspath(save_state)}/{pname}.{ext} f32")
    if grads:
        for pname, nm, g1, g2 in adams:
            for g in (g1, g2):
                if g:
                    L.append(f"save {g} {os.path.abspath(grads)} f32")
    open(p("session"), "w").write("\n".join(L) + "\n")
    return p("session")


def grad_names(c):
    """The parameters in the order `write` saves their gradients (the table is last and has two parts, 'emb.head' and 'emb.in'): [(name, shape)]."""
    sh = R.shapes(c)
    out = [("norm", sh["norm"])]
    for l in reversed(range(c.layers)):
        q = f"l{l:02d}."
        out += [(q + "wd", sh[q + "wd"]), (q + "wgu", sh[q + "wgu"]), (q + "ln2", sh[q + "ln2"]), (q + "wo", sh[q + "wo"]), (q + "wqkv", sh[q + "wqkv"]), (q + "bqkv", sh[q + "bqkv"]), (q + "ln1", sh[q + "ln1"])]
    return out


def read_grads(c, path, loss_scale, ncount, vp):
    """The gradients ({name: array}, binary64; the table's [v, d]) from the file `grads` wrote, with the loss scale and the count divided out."""
    flat = np.fromfile(path, dtype="<f4").astype(np.float64) / (loss_scale * ncount)
    out, o = {}, 0
    for name, shape in grad_names(c):
        k = int(np.prod(shape))
        out[name] = flat[o:o + k].reshape(shape)
        o += k
    k = vp * c.d
    out["emb"] = (flat[o:o + k] + flat[o + k:o + 2 * k]).reshape(vp, c.d)[:c.v]
    if o + 2 * k != flat.size:
        raise SystemExit(f"{path}: {flat.size} words, the model has {o + 2 * k}")
    return out
