#!/usr/bin/env python3
"""docs/optimizer.md section 7: the sessions that train a model (the CNN and the Qwen LoRA; the GPT's is tests/optim.py) on the optimiser of examples/optim, `opt=` of their `write`.

    python3 tests/optim_sessions.py build/cancho-gpu-session [build/cancho-gpu-metal-session] [--mutants]

For each session:
1. The new path equals `torch.optim.AdamW` + `clip_grad_norm_` on the session's own gradients: one optimiser step with the clip active, a decay (the schedule's first rate, which is not the peak)
   and the schedule, the gradients being those the session saved (raw, so they are rescaled and masked here as the optimiser does). The weights after the step agree to f32 rounding.
2. With one micro-step, no clip and a constant rate the new path is the old one (`adamw_h` / `adam`, `adamw_l`): the losses of the steps and the weights agree.
3. (LoRA) Two micro-batches of 2 sequences make a step that equals one step of the batch of 4, with the clip active, in the regime where Adam is linear in the gradient (eps far above sqrt v).
`--mutants` damages the wiring (the mask, the clip, the loss scale, the f16 copy) and requires a check to fail. Needs numpy and torch; each is skipped without them."""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("optim", "cnn", "qwen", "gpt"):
    sys.path.insert(0, os.path.join(ROOT, "examples", sub))
try:
    import numpy as np
    import torch
except ImportError:
    print("skip: numpy and torch are needed")
    sys.exit(0)
import schedule as SCH  # noqa: E402

BINS = [os.path.abspath(a) for a in sys.argv[1:] if not a.startswith("--") and os.path.exists(a)]
EXTRA = lambda b: [] if "metal" in b else ["--target", "apple-m-series"]
bad = 0
COUNT = True


def say(good, text):
    global bad
    print(f"{'ok  ' if good else 'FAIL'} {text}" if COUNT else f"     {text}")
    bad += 0 if good or not COUNT else 1
    return good


def torch_step(params, grads, decay, lr, wd, max_norm, b1=0.9, b2=0.999, eps=1e-8):
    """One step of torch's AdamW (the decay groups: `decay[name]`) after clip_grad_norm_; the arrays are flat and given in float32. Answers the new arrays and the clip coefficient."""
    ts = {k: torch.nn.Parameter(torch.tensor(np.asarray(v, dtype=np.float32))) for k, v in params.items()}
    for k, t in ts.items():
        t.grad = torch.tensor(np.asarray(grads[k], dtype=np.float32))
    norm = torch.nn.utils.clip_grad_norm_(list(ts.values()), max_norm) if max_norm is not None else None
    opt = torch.optim.AdamW([{"params": [ts[k] for k in ts if decay[k]], "weight_decay": wd}, {"params": [ts[k] for k in ts if not decay[k]], "weight_decay": 0.0}], lr=lr, betas=(b1, b2), eps=eps)
    opt.step()
    return {k: t.detach().numpy().astype(np.float64) for k, t in ts.items()}, norm


def spread(a, b, ref):
    """The norm of a - b as a fraction of the norm of the move b - ref."""
    return float(np.sqrt(sum(float(np.sum((a[k] - b[k]) ** 2)) for k in a) / max(sum(float(np.sum((b[k] - ref[k]) ** 2)) for k in a), 1e-300)))


def run(binary, session, lines):
    r = subprocess.run([binary, session] + EXTRA(binary), input=lines, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise SystemExit(f"{binary}: {r.stderr[:400]}")
    return [np.array([float(v) for v in line.split()]) for line in r.stdout.splitlines()]


def mutate(session, mut):
    """A damaged copy of the generated session file: the wiring faults the checks must see."""
    if not mut:
        return
    text = open(session).read()
    lines = text.split("\n")
    out = []
    last = None
    for ln in lines:
        if mut == "no_clip" and ln.startswith("program clip "):
            ln = ln.split(" maxn=")[0] + " maxn=1.0e30"
        if mut == "no_mask" and ln.startswith("resident acc_") and ".acc " in ln:
            last = ln.split()[2]
        if mut == "no_mask" and ln.startswith("resident acc_") and ".mk " in ln:
            parts = ln.split()
            ones = last + ".ones"  # the mask of ones, as large as the accumulator
            np.ones(os.path.getsize(last) // 4, dtype="<f4").tofile(ones)
            parts[2] = ones
            ln = " ".join(parts)
        if mut == "no_osc" and ln.startswith("program aB"):
            ln = ln.split(" osc=")[0] + " osc=1.0"
        if mut == "loss_scale" and ln.startswith("program sq_"):
            parts = ln.split()
            ln = " ".join(f"gs={float(x[3:]) * 2}" if x.startswith("gs=") else x for x in parts)
        out.append(ln)
    open(session, "w").write("\n".join(out))
    assert "\n".join(out) != text, mut


# ---------------------------------------------------------------- the CNN
def cnn_checks(binary, mut=None):
    import cnn_ref as R
    import cnn_session as C
    d = tempfile.mkdtemp()
    c = R.Config(h=8, cin=3, widths=(16, 16, 32), blocks=(1, 1, 1), classes=10, n=16)
    rng = np.random.RandomState(5)
    imgs = rng.normal(size=(4, c.n, c.h, c.h, c.cin))
    lab = rng.randint(0, 10, size=(4, c.n))
    P = C.pad_params(c, R.init_params(c, 1))
    mats, vecs = C.names(c)
    names = mats + vecs
    gs = 1.0 / (c.n * C.LOSS_SCALE)
    lr, wd, max_norm = SCH.warmup_cosine(3e-3, 4, 10, 3e-4), 0.05, 0.05
    out = []
    # (1) one step, the clip active, against torch on the session's own gradients
    sess = C.write(os.path.join(d, "t"), c, imgs, lab, 1, grads=True, dump=True, lr_fn=lr, params=R.init_params(c, 1), opt=dict(accum=1, max_norm=max_norm, wd=wd))
    mutate(sess, mut)
    r = run(binary, sess, "1\n")[0]
    o, g, w1 = 1, {}, {}
    for nm in names:
        g[nm] = r[o:o + P[nm].size] * gs
        o += P[nm].size
    for nm in names:
        w1[nm] = r[o:o + P[nm].size]
        o += P[nm].size
    ref, norm = torch_step({k: np.asarray(P[k]).reshape(-1) for k in names}, g, {k: k in mats for k in names}, lr(1), wd, max_norm)
    err = max(float(np.max(np.abs(w1[k] - ref[k]))) for k in names)
    move = max(float(np.max(np.abs(ref[k] - np.asarray(P[k]).reshape(-1)))) for k in names)
    out.append(("CNN equals torch", err < 5e-7 * 1 and float(norm) > max_norm))
    say(out[-1][1], f"CNN, one step with clip {max_norm} (gradient norm {float(norm):.3f}), decay {wd} at the schedule's first rate {lr(1):.1e}: weights equal torch's AdamW + clip_grad_norm_ to {err:.1e} (the move is {move:.1e})")
    if mut:
        return out
    # (2) one micro-step, no clip, a constant rate: the old path (adamw_h, adam), the losses of 3 steps and the weights. With eps far above sqrt v Adam is linear in the gradient
    # (a gradient at the noise of f16 would flip sign between two runs that differ in the last bit), so what is compared is the gradient and the decay
    const = SCH.constant(30.0)
    eps0, C.EPS = C.EPS, 1e3
    res = []
    for label, opt in (("old", None), ("new", dict(accum=1, wd=wd))):
        s = C.write(os.path.join(d, "eq_" + label), c, imgs, lab, 3, lr=30.0, wd=0.02 / 30.0, lr_fn=const, dump=True, params=R.init_params(c, 1), opt=None if opt is None else dict(accum=1, wd=0.02 / 30.0))
        res.append(run(binary, s, "1\n1\n1\n"))
    C.EPS = eps0
    losses = max(abs(float(a[0]) - float(b[0])) / abs(float(a[0])) for a, b in zip(res[0], res[1]))
    tot = sum(P[k].size for k in names)
    old_w, new_w, w0 = res[0][-1][1:1 + tot], res[1][-1][1:1 + tot], np.concatenate([np.asarray(P[k]).reshape(-1) for k in names])
    wdiff = float(np.linalg.norm(old_w - new_w) / np.linalg.norm(old_w - w0))
    out.append(("CNN new equals old", losses < 1e-4 and wdiff < 1e-3))
    say(out[-1][1], f"CNN, one micro-step, no clip, a constant rate: the new path's 3 losses are the old path's to {losses:.1e} (relative), the weights to {wdiff:.1e} of the norm of the move")
    return out


def rn_checks(binary):
    """The ResNet session shares the CNN's optimiser block: its new path with one micro-step, no clip and a constant rate gives the old path's losses (3 steps; weights as in the CNN's check)."""
    import resnet_ref as R
    import rn_session as RN
    d = tempfile.mkdtemp()
    c = R.Config(h=64, w=96, widths=(8, 16, 16, 32), blocks=(1, 2, 1, 1), classes=10, n=8)
    rng = np.random.RandomState(5)
    imgs = rng.normal(size=(3, c.n, c.h, c.w, c.cin))
    lab = rng.randint(0, 10, size=(3, c.n))
    const = SCH.constant(30.0)
    eps0, RN.EPS = RN.EPS, 1e3
    res = []
    for label, opt in (("old", None), ("new", dict(accum=1, wd=0.02 / 30.0))):
        s = RN.write(os.path.join(d, label), c, imgs, lab, 3, lr=30.0, wd=0.02 / 30.0, lr_fn=const, params=R.init_params(c, 1), opt=opt)
        res.append(run(binary, s, "1\n1\n1\n"))
    RN.EPS = eps0
    losses = max(abs(float(a[0]) - float(b[0])) / abs(float(a[0])) for a, b in zip(res[0], res[1]))
    ok = losses < 1e-4
    say(ok, f"ResNet, one micro-step, no clip, a constant rate: the new path's 3 losses are the old path's to {losses:.1e} (relative)")
    return [("ResNet new equals old", ok)]


# ---------------------------------------------------------------- the Qwen LoRA
def lora_checks(binary, mut=None):
    import qwen_ref as R
    import qwen_session as S
    import qwen_train as T
    d = tempfile.mkdtemp()
    c = R.Config(layers=2, d=128, h=4, hkv=2, dh=32, ff=256, v=320, t=64, nb=2)
    c4 = R.Config(layers=2, d=128, h=4, hkv=2, dh=32, ff=256, v=320, t=64, nb=4)
    P = R.init_params(c, seed=1)
    P["emb"] = P["emb"] * 0.3
    toks = np.random.default_rng(0).integers(0, c.v, (4, c.nb, c.t))
    AB = R.init_lora(c, 16, 7, b_std=0.02)
    wdir = os.path.join(d, "w")
    S.write_files(wdir, c, P)
    extra = EXTRA(binary)
    ls = 64.0
    out = []

    def go(name, cfg, tk, steps, lines, opt, lr_fn, eps=1e-8, wd=0.0, grads=False, lrv=1e-4):
        base = os.path.join(d, name)
        os.makedirs(base, exist_ok=True)
        sess = T.write(os.path.join(base, "t"), cfg, wdir, tk, AB, r=16, loss_scale=ls, train=True, steps=steps, lr=lrv, lr_fn=lr_fn, wd=wd, eps_adam=eps, save=os.path.join(base, "ad.bin"),
                       grads=os.path.join(base, "gr.bin") if grads else None, tune=dict(bc=128), opt=opt)
        if opt:
            mutate(sess, mut)
        ss = S.Session(binary, sess, extra)
        got = [float(ss.ask(str(k))[0].split()[0]) for k in lines]
        ss.checkpoint()
        ss.close()
        return got, T.read_fused(cfg, 16, os.path.join(base, "ad.bin")), (T.read_fused(cfg, 16, os.path.join(base, "gr.bin")) if grads else None)

    flat = lambda fa: {f"{l}.{g}.{ab}": fa[(l, g, ab)] for (l, g, ab) in fa}
    # (1) one step, the clip active, against torch on the session's own gradients
    lr, wd, max_norm = SCH.warmup_cosine(3e-3, 4, 10, 3e-4), 0.1, 0.5
    _, fa, fg = go("t1", c, toks[:1], 1, [1], dict(accum=1, max_norm=max_norm, wd=wd), lr, wd=wd, grads=True)
    start = {}
    for l in range(c.layers):
        for g, (A, B, mask) in T.fuse(c, 16, AB, l).items():
            start[(l, g, "A")], start[(l, g, "B")] = A, B
    masks = {}
    for l in range(c.layers):
        for g, (A, B, mask) in T.fuse(c, 16, AB, l).items():
            masks[(l, g, "A")], masks[(l, g, "B")] = np.ones_like(A), mask
    # the session's normalisation: the sum of the positions with a next token (ncount)
    ncount = c.nb * (c.t - 1)
    scale = 32.0 / 16
    eff = {k: fg[k] * masks[k] * (1.0 / (ncount * ls)) * (scale if k[2] == "B" else 1.0) for k in fg}
    keys = sorted(start)
    ref, norm = torch_step({k: start[k].reshape(-1) for k in keys}, {k: eff[k].reshape(-1) for k in keys}, {k: True for k in keys}, lr(1), wd, max_norm)
    err = max(float(np.max(np.abs(fa[k].reshape(-1) - ref[k]))) for k in keys)
    move = max(float(np.max(np.abs(ref[k] - start[k].reshape(-1)))) for k in keys)
    off = max(float(np.abs(fa[(l, "qkv", "B")][:c.d, 16:]).max()) for l in range(c.layers))
    out.append(("LoRA equals torch", err < 5e-7 and float(norm) > max_norm and off == 0.0))
    say(out[-1][1], f"LoRA, one step with clip {max_norm} (gradient norm {float(norm):.3f}), decay {wd} at the schedule's first rate {lr(1):.1e}: adapters equal torch's AdamW + clip_grad_norm_ to {err:.1e} (the move is {move:.1e}); fused B off-diagonal {off:.0e}")
    if mut and mut != "no_osc":
        return out
    # (2) one micro-step, no clip, a constant rate: the old path (adamw_l, the constant lr wd)
    const = SCH.constant(100.0)
    lo, fo, _ = go("eq_old", c, toks[:3], 3, [1, 1, 1], None, const, eps=1e3, wd=0.02 / 100.0, lrv=100.0)
    ln, fn, _ = go("eq_new", c, toks[:3], 3, [1, 1, 1], dict(accum=1, wd=0.02 / 100.0), const, eps=1e3, wd=0.02 / 100.0, lrv=100.0)
    dl = max(abs(a - b) / abs(a) for a, b in zip(lo, ln))
    dw = spread({k: fn[k] for k in start}, {k: fo[k] for k in start}, start)
    out.append(("LoRA new equals old", dl < 1e-4 and dw < 1e-3))
    say(out[-1][1], f"LoRA, one micro-step, no clip, a constant rate: the new path's 3 losses are the old path's to {dl:.1e} (relative), the adapters to {dw:.1e} of the move")
    if mut:
        return out
    # (3) two micro-batches of 2 sequences against one batch of 4, the clip active, Adam in its linear regime (eps far above sqrt v)
    big = np.concatenate([toks[0], toks[1]], axis=0)[None]
    opt = dict(accum=2, max_norm=0.5, wd=0.0)
    lr_lin = SCH.constant(100.0)
    _, f_mic, _ = go("acc_mic", c, toks[:2], 2, [2], opt, lr_lin, eps=1e3)
    _, f_big, _ = go("acc_big", c4, big, 1, [1], dict(accum=1, max_norm=0.5, wd=0.0), lr_lin, eps=1e3)
    _, f_nc, _ = go("acc_noclip", c, toks[:2], 2, [2], dict(accum=2, max_norm=None, wd=0.0), lr_lin, eps=1e3)
    s0 = {k: v for k, v in start.items()}
    e, e_nc = spread({k: f_mic[k] for k in s0}, {k: f_big[k] for k in s0}, s0), spread({k: f_nc[k] for k in s0}, {k: f_big[k] for k in s0}, s0)
    out.append(("LoRA accumulation equals the big batch", e < 2e-2 and e_nc > 10 * e))
    say(out[-1][1], f"LoRA, one step of 2 micro-batches equals one step of the batch of 4 (clip 0.5): adapters differ by {e:.1e} of the move; without the clip {e_nc:.1e}")
    return out


MUTS = [("the clip not applied", "no_clip", "lora"), ("the mask on the gradient dropped", "no_mask", "lora"), ("the loss scale wrong in the norm", "loss_scale", "lora"), ("the f16 copy of B without alpha / r", "no_osc", "lora"), ("the clip not applied", "no_clip", "cnn"),
        ("the loss scale wrong in the norm", "loss_scale", "cnn")]


def main():
    global bad
    mutants = "--mutants" in sys.argv
    for binary in BINS:
        print(f"-- {os.path.basename(binary)}")
        if mutants:
            for label, mut, which in MUTS:
                global COUNT
                COUNT = False
                res = (cnn_checks if which == "cnn" else lora_checks)(binary, mut)
                COUNT = True
                failed = [n for n, ok in res if not ok]
                say(bool(failed), f"mutant, {which}: {label}: killed by {len(failed)} check(s)")
            continue
        res = cnn_checks(binary) + rn_checks(binary) + lora_checks(binary)
        print(f"optim sessions: {sum(ok for _, ok in res)}/{len(res)} ok")
    sys.exit(1 if bad else 0)


main()
