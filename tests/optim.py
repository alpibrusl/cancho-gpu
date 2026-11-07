#!/usr/bin/env python3
"""docs/optimizer.md (roadmap G6): global-norm clipping, AdamW decay groups, the warm-up + cosine schedule and gradient accumulation, held to torch.

    python3 tests/optim.py build/cancho-gpu-session [--mutants]

The CPU session runs the kernels of examples/optim/ on gradients given as tables (so the optimiser is tested alone) and torch float32 runs the same micro-steps:
1. the clip: `torch.nn.utils.clip_grad_norm_` then `torch.optim.AdamW` give the weights, first and second moments (eps 1e-3, so a scaled gradient shows) and the coefficient, with the norm above
   max_norm, below it (the coefficient is exactly 1) and with every gradient zero;
2. two parameter groups (decay 0.1 and 0) over 40 steps of a warm-up + cosine schedule, against AdamW's param groups; the schedule against torch's SequentialLR(LinearLR, CosineAnnealingLR);
3. accumulation: N = 4 micro-steps equal one step on the mean gradient (the big batch), and 5 windows with clipping equal torch's backward of loss / N, N times, then step;
`--mutants` makes one deliberate fault at a time (a copy of a kernel, or a flag of the generator) and requires the checks to fail.
Needs numpy and torch; without either the test says so and passes."""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "optim"))
try:
    import numpy as np
    import torch
    import optim_session as O
    import schedule as S
except ImportError as e:
    print(f"skip: {e}")
    sys.exit(0)

BINS = [os.path.abspath(a) for a in sys.argv[1:] if not a.startswith("-") and os.path.exists(a)] or [os.path.join(ROOT, "build", "cancho-gpu-session")]
BIN = BINS[0]
SHAPES = [("a", (6, 5), "d"), ("b", (7,), "n"), ("c", (3, 40), "d"), ("e", (1,), "n"), ("f", (17,), "d")]  # 17 and 7 are prime: the block is 17 and 7, the tail of the tile logic


def problem(seed, reps, scale=1.0):
    rng = np.random.default_rng(seed)
    params = [(n, rng.normal(size=s).astype(np.float32), g) for n, s, g in SHAPES]
    grads = {n: (rng.normal(size=(reps,) + s) * scale).astype(np.float32) for n, s, _ in SHAPES}
    return params, grads


def device(params, grads, d, reqs=None, **kw):
    session = O.write(d, params, grads, **kw)
    total = kw.get("steps", 1) * kw.get("accum", 1)
    r = subprocess.run([BIN, session], input=f"{total}\n", capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise SystemExit(f"{BIN}: {r.stderr[:300]}")
    return O.parse(params, [float(v) for v in r.stdout.split()])


def reference(params, grads, steps=1, accum=1, lr=None, wd=None, max_norm=None, gs=1.0, b1=0.9, b2=0.999, eps=1e-8):
    """torch float32: the loss / accum backward `accum` times (the gradient is added into .grad), clip_grad_norm_, AdamW.step, one rate a step."""
    lr = lr or S.constant(1e-3)
    wd = wd or {}
    ts = {n: torch.nn.Parameter(torch.tensor(w, dtype=torch.float32)) for n, w, _ in params}
    groups = sorted({g for _, _, g in params})
    opt = torch.optim.AdamW([{"params": [ts[n] for n, _, g in params if g == grp], "weight_decay": wd.get(grp, 0.0)} for grp in groups], lr=lr(1), betas=(b1, b2), eps=eps)
    coef = 1.0
    for t in range(1, steps + 1):
        for grp in opt.param_groups:
            grp["lr"] = lr(t)
        opt.zero_grad(set_to_none=True)
        for j in range(accum):
            for n, _, _ in params:
                g = torch.tensor(grads[n][(t - 1) * accum + j] * np.float32(gs), dtype=torch.float32) / accum
                ts[n].grad = g if ts[n].grad is None else ts[n].grad + g
        if max_norm is not None:
            total = torch.nn.utils.clip_grad_norm_(list(ts.values()), max_norm)
            coef = min(1.0, float(max_norm) / (float(total) + 1e-6))
        opt.step()
    states = {n: (ts[n].detach().numpy().astype(np.float64), opt.state[ts[n]]["exp_avg"].numpy().astype(np.float64), opt.state[ts[n]]["exp_avg_sq"].numpy().astype(np.float64)) for n, _, _ in params}
    return coef, states


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(float(np.max(np.abs(b))), 1e-30))


def compare(label, dev, ref, tol_w=1e-6, tol_mv=2e-6, tol_cs=1e-6):
    cs, got = dev
    ref_cs, want = ref
    errs = {x: max(rel(got[n][i], want[n][i]) for n in want) for i, x in enumerate("wmv")}
    ok = all(errs["w"] <= tol_w and errs[x] <= tol_mv for x in "mv") and errs["w"] <= tol_w and abs(cs - ref_cs) <= tol_cs * max(abs(ref_cs), 1e-30)
    print(f"{'ok  ' if ok else 'FAIL'} {label}: w {errs['w']:.1e} m {errs['m']:.1e} v {errs['v']:.1e} (relative to the largest value); clip {cs:.7g} torch {ref_cs:.7g}")
    return ok


def checks(kdir=None, mut=()):
    """Every check against torch; the list of (name, ok). `kdir` and `mut` are the faults of the mutant tests."""
    d = tempfile.mkdtemp()
    res = []
    kw = dict(kdir=kdir, mut=mut)
    out = lambda name, ok: res.append((name, ok))
    # 1. the clip, three cases (the gradients are the micro-step's after gs = 0.25: the norm is of the true gradient)
    for label, scale, maxn in (("norm above max_norm", 1.0, 0.5), ("norm below max_norm", 0.01, 5.0), ("zero gradient", 0.0, 0.5), ("norm near the 1e-6 of the formula", 1e-7, 3e-7)):
        params, grads = problem(1, 3, scale)
        args = dict(steps=3, lr=S.constant(1e-2), max_norm=maxn, gs=0.25, eps=1e-3, wd={"d": 0.1})
        dev = device(params, grads, os.path.join(d, "clip_" + label.split()[0]), **args, **kw)
        ref = reference(params, grads, **args)
        out("clip " + label, compare("clip, " + label, dev, ref))
        if "below" in label:
            out("clip below: coefficient exactly 1", dev[0] == 1.0)
    # 2. groups and the schedule
    lr = S.warmup_cosine(2e-2, 5, 40, 2e-3)
    params, grads = problem(2, 40)
    args = dict(steps=40, lr=lr, wd={"d": 0.3, "n": 0.0}, eps=1e-6)
    out("groups and schedule, 40 steps", compare("two groups, warm-up + cosine, 40 steps", device(params, grads, os.path.join(d, "grp"), **args, **kw), reference(params, grads, **args)))
    # the schedule against torch's own schedulers
    m = torch.nn.Parameter(torch.zeros(1))
    o = torch.optim.SGD([m], lr=2e-2)
    sch = torch.optim.lr_scheduler.SequentialLR(o, [torch.optim.lr_scheduler.LinearLR(o, start_factor=1 / 5, end_factor=1.0, total_iters=4), torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=35, eta_min=2e-3)], milestones=[5])
    worst = 0.0
    for t in range(1, 41):
        worst = max(worst, abs(o.param_groups[0]["lr"] - lr(t)))
        o.step()
        sch.step()
    print(f"{'ok  ' if worst < 1e-12 else 'FAIL'} schedule equals SequentialLR(LinearLR, CosineAnnealingLR) at all 40 steps: largest difference {worst:.1e}")
    out("schedule", worst < 1e-12)
    # 3a. N micro-steps equal one step on the mean gradient (the big batch)
    N = 4
    params, grads = problem(3, N)
    args = dict(steps=1, accum=N, lr=S.constant(1e-2), eps=1e-6)
    dev = device(params, grads, os.path.join(d, "acc"), **args, **kw)
    big = {n: g.mean(axis=0, keepdims=True).astype(np.float32) for n, g in grads.items()}
    ref = reference(params, big, steps=1, accum=1, lr=S.constant(1e-2), eps=1e-6)
    out("accumulation equals the big batch", compare(f"{N} micro-steps = one big batch", dev, ref))
    # 3b. five windows with the clip and decay against torch's accumulate-then-step
    params, grads = problem(4, 5 * N, 1.0)
    lr = S.warmup_cosine(1e-2, 2, 5, 1e-3)
    args = dict(steps=5, accum=N, lr=lr, wd={"d": 0.1}, max_norm=1.0, eps=1e-4)
    out("5 accumulated windows", compare("5 windows of 4 micro-steps, clip 1.0, decay, schedule", device(params, grads, os.path.join(d, "win"), **args, **kw), reference(params, grads, **args)))
    # 3c. the f16 copy the matrix unit reads is of the weights the next micro-step uses: in the middle of a window (repetition 4 of windows of 3) the weights are those after step 1 and so is the copy
    params, grads = problem(5, 6)
    session = O.write(os.path.join(d, "half"), params, grads, steps=2, accum=3, lr=S.constant(1e-2), wd={"d": 0.1}, eps=1e-3, half=True, **kw)
    r = subprocess.run([BIN, session], input="4\n", capture_output=True, text=True, timeout=600)
    nums = np.array([float(v) for v in r.stdout.split()])
    sizes = [int(np.prod(w.shape)) for _, w, _ in params]
    tail = nums[1 + 3 * sum(sizes):]
    wcat = np.concatenate([nums[1 + 3 * sum(sizes[:i]) + 0: 1 + 3 * sum(sizes[:i]) + sizes[i]] for i in range(len(sizes))])
    ok_h = tail.size == sum(sizes) and bool(np.array_equal(tail, wcat.astype(np.float16).astype(np.float64)))
    print(f"{'ok  ' if ok_h else 'FAIL'} the f16 copy in the middle of a window is the f16 rounding of the weights the next micro-step reads")
    out("f16 copy mid-window", ok_h)
    shutil.rmtree(d, ignore_errors=True)
    return res


def gpt_checks(kdir=None, mut=()):
    """4. A GPT (examples/gpt, 2 layers) trained by the new path: 3 optimiser steps of 2 micro-batches of 1 sequence equal 3 steps of one batch of 2 sequences (the big batch), with the
    clip active, a decay group and the schedule. The weights after the steps agree to the f16 rounding of the activations, and the clip really bit (the unclipped run differs)."""
    sys.path.insert(0, os.path.join(ROOT, "examples", "gpt"))
    import gpt_ref as GR
    import gpt_session as GS
    d = tempfile.mkdtemp()
    steps, accum = 3, 2
    small, big = GR.Config(v=65, vp=128, t=64, d=128, h=2, layers=2, b=1), GR.Config(v=65, vp=128, t=64, d=128, h=2, layers=2, b=2)
    rng = np.random.RandomState(9)
    tok = rng.randint(0, 65, size=(steps * accum, small.n))
    tgt = (tok * 3 + 1) % 65
    params = GR.init_params(small, 1)
    lr = S.warmup_cosine(3e1, 2, steps, 3.0)
    # (a) with no clip, one micro-step and a constant rate the new path is the old one (adamw_h and adam, a constant lwd): the same weights after 2 steps
    eps0 = GS.EPS
    GS.EPS = 1e3  # Adam moves every element by about lr whatever its gradient (the first step is lr sign(g)): a gradient at the noise of f16 flips sign between two runs that differ in the last bit, and the clip, which Adam absorbs, would not show. With an eps far above sqrt(v) the update is lr m / eps, linear in the accumulated gradient: what is compared is the gradient
    const = S.constant(30.0)
    eqs = []
    for name, opt_ in (("old", None), ("new", dict(accum=1, wd=1e-5))):
        sess = GS.write(os.path.join(d, "eq_" + name), small, tok[:2], tgt[:2], 2, params=params, lr=30.0, wd=1e-5, lr_fn=const, dump=True, opt=opt_)
        r = subprocess.run([BIN, sess] + ([] if "metal" in BIN else ["--target", "apple-m-series"]), input="2\n", capture_output=True, text=True, timeout=1200)
        eqs.append(GS.parse_dump(small, params, [float(v) for v in r.stdout.split()]))
    diff = max(float(np.max(np.abs(eqs[0][k] - eqs[1][k]))) for k in eqs[0])
    ok_eq = diff < 5e-6
    print(f"{'ok  ' if ok_eq else 'FAIL'} GPT, the new path with one micro-step, no clip and a constant rate equals adamw_h / adam after 2 steps: largest weight difference {diff:.1e}")

    def final(c, tk, tg, opt, st, name):
        sess = GS.write(os.path.join(d, name), c, tk, tg, st, params=params, lr_fn=lr, dump=True, opt=opt)
        r = subprocess.run([BIN, sess] + ([] if "metal" in BIN else ["--target", "apple-m-series"]), input=f"{len(tk)}\n", capture_output=True, text=True, timeout=1200)
        if r.returncode != 0:
            raise SystemExit(f"{BIN}: {r.stderr[:300]}")
        return GS.parse_dump(c, params, [float(v) for v in r.stdout.split()])

    def spread(a, b):
        return float(np.sqrt(sum(float(np.sum((a[k] - b[k]) ** 2)) for k in a) / sum(float(np.sum((b[k] - params[k]) ** 2)) for k in a)))  # the norm of the difference as a fraction of the norm of the move

    opt = dict(accum=accum, max_norm=0.3, wd=1e-5)
    micro = final(small, tok, tgt, opt, steps, "micro")
    bigrun = final(big, tok.reshape(steps, 2 * small.n), tgt.reshape(steps, 2 * small.n), dict(accum=1, max_norm=0.3, wd=1e-5), steps, "big")
    unclipped = final(small, tok, tgt, dict(accum=accum, max_norm=None, wd=1e-5), steps, "noclip")
    e, c_ = spread(micro, bigrun), spread(unclipped, bigrun)
    ok = e < 2e-2 and c_ > 10 * e
    print(f"{'ok  ' if ok else 'FAIL'} GPT, {steps} steps of {accum} micro-batches = {steps} steps of the big batch (clip 0.3, decay, schedule): weights differ by {e:.1e} of the norm of the move; without the clip {c_:.1e}")
    GS.EPS = eps0
    shutil.rmtree(d, ignore_errors=True)
    return [("GPT new path equals the old", ok_eq), ("GPT accumulation equals the big batch", ok)]


MUTANTS = [
    ("the clip factor without the square root", "clip_scale.chx", "  let rr  = rsqrt &sg\n  let nrm = sg * rr\n", "  let nrm = sg * 1.0\n", ()),
    ("the clip not applied to the gradient", "adamw_g.chx", "let g  = g1 * cb", "let g  = g1 + cb * 0.0", ()),
    ("the clip applied twice", "adamw_g.chx", "let g  = g1 * cb", "let g2 = g1 * &cb\n  let g  = g2 * cb", ()),
    ("the norm without the loss scale gs", "gradsq.chx", "let g  = g0 * gs", "let g  = g0 * 1.0", ()),
    ("the clip's 1e-6 dropped", "clip_scale.chx", "let den = nrm + 1.0e-6", "let den = nrm + 0.0", ()),
    ("the accumulation not divided", None, None, None, ("no_div",)),
    ("the accumulator never reset", None, None, None, ("keep_always",)),
    ("an update on every micro-step", None, None, None, ("apply_always",)),
    ("the decay of the wrong group", None, None, None, ("wd_swap",)),
    ("the decay at the first step's rate", None, None, None, ("dec_peak",)),
    ("the f16 copy of the update that was not applied", "adamw_gh.chx", "stage f16 wb", "stage f16 wn", ()),
    ("the decay not applied", "adamw_g.chx", "let wn = &w0 * db - q2 * rb", "let wn = &w0 + db * 0.0 - q2 * rb", ()),
]


def mutants():
    bad = 0
    for label, kernel, old, new, mut in MUTANTS:
        kdir = None
        if kernel:
            kdir = tempfile.mkdtemp()
            for f in os.listdir(os.path.join(ROOT, "examples", "optim")):
                if f.endswith(".chx"):
                    shutil.copy(os.path.join(ROOT, "examples", "optim", f), kdir)
            text = open(os.path.join(kdir, kernel)).read()
            assert old in text, (label, old)
            open(os.path.join(kdir, kernel), "w").write(text.replace(old, new))
            if kernel == "adamw_g.chx":
                shutil.copy(os.path.join(kdir, kernel), os.path.join(kdir, "adamw_gh.chx"))
        failed = [n for n, ok in checks(kdir, mut) if not ok]
        print(f"{'ok  ' if failed else 'FAIL'} mutant, {label}: killed by {len(failed)} check(s)" + (f" ({failed[0]})" if failed else ""))
        bad += 0 if failed else 1
    return bad


def main():
    if "--mutants" in sys.argv:
        sys.exit(1 if mutants() else 0)
    global BIN
    bad = 0
    for BIN in BINS:  # the CPU session, then the Metal one when given
        print(f"-- {os.path.basename(BIN)}")
        res = checks() + gpt_checks()
        print(f"optim: {sum(ok for _, ok in res)}/{len(res)} ok")
        bad += sum(not ok for _, ok in res)
    sys.exit(1 if bad else 0)


main()
