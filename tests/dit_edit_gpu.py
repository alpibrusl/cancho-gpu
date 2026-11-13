#!/usr/bin/env python3
"""D9e of docs/diffusion.md: one denoising step of Qwen-Image-2.1-Turbo's transformer **with a condition image** on a Metal GPU, as a prefix pass and a cached step, held to the float64 step of tools/dit_edit_ref.py.

    dit_edit_gpu.py DE_DIR WORK_DIR --weights DIR [--gp N] [--gs N] [--time] [--mutate cond-causal|prefix-sampled-t|no-cat]

DE_DIR is the output of `tools/dit_edit_ref.py` (the joint sequence after the front end, the rotary angles, the mask, the modulation, and the float64 residual after every block). `--weights` is the directory of f16 block
weights that tests/dit_big.py writes (head-permuted, so dit_rope16's rotate-half is the model's rotary embedding). Needs a Mac, the checkpoint, and `build/cancho-gpu-metal-session`. Not in `scripts/gate.sh`.

The design (docs/diffusion.md D9, read from `QwenImage21Transformer2DModel.forward` with a KV cache): the prefix, every row that is not a target token (the text rows and the condition image's tokens), is modulated from t = 0 and never
attends to a target token, so its keys and values are the same at every step. **The prefix pass** runs those rows through the 32 blocks with the block-causal mask and saves each block's keys (after the per-head norm and the rotary
embedding, f16) and values (f16). **A step** runs only the target's rows: per block the same programs as tests/dit_big.py, except that the keys and values are `dit_cat` of the cached prefix's and the target's own, so the
scores are [32 x N_target, N_prefix + N_target] (rows padded to a multiple of 64, both parts to the same count); the mask only hides padded keys (a target row sees the whole prefix and the whole target block).
The host does the front end (txt_in, img_in, the timestep embedding, the modulation, the rotary tables: in float64, from DE_DIR) and the back end (norm_out and proj_out on the target rows).
"""
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import dit_block_ref as ref  # noqa: E402
import dit_step_ref as step  # noqa: E402

arg = lambda name, default=None: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
de, work = sys.argv[1], sys.argv[2]
WDIR = arg("--weights")
GP, GS = int(arg("--gp", 32)), int(arg("--gs", 32))
TIME = "--time" in sys.argv
MUT = arg("--mutate")  # a deliberate host-side bug, which the comparison must refuse
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
CHX = os.path.join(ROOT, "tests/chx")
NEG = -1e9
NB, D, H, HD = 32, 4096, 32, 128
os.makedirs(work, exist_ok=True)
P = lambda n: os.path.join(work, n)
W = lambda n: os.path.join(WDIR, n)
f16 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f2")).tofile(path)
f32 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f4")).tofile(path)
ld = lambda n: np.load(os.path.join(de, n + ".npy"))

x0, rope, sel, allowed, mod, temb = ld("x0"), ld("rope"), ld("sel"), ld("allowed"), ld("mod"), ld("temb")
blocks = ld("blocks_f64")
want_out = ld("out_f64")
pre, tgt = np.nonzero(~sel)[0], np.nonzero(sel)[0]
NPR, NTG = len(pre), len(tgt)
pad64 = lambda n: (n + 63) // 64 * 64
NP = max(pad64(NPR), pad64(NTG))
print("prefix %d rows, target %d rows, both padded to %d; keys %d" % (NPR, NTG, NP, 2 * NP), flush=True)

# ---- data
xp = np.zeros((NP, D))
xp[:NPR] = x0[pre]
f32(P("xpre.f32"), xp)
xt = np.zeros((NP, D))
xt[:NTG] = x0[tgt]
f32(P("xtgt.f32"), xt)
for nm, rows, n in (("pre", pre, NPR), ("tgt", tgt, NTG)):
    cs, sn = np.ones((NP, 64)), np.zeros((NP, 64))
    cs[:n], sn[:n] = rope[rows][..., 0], rope[rows][..., 1]
    f32(P("cs_%s.f32" % nm), cs)
    f32(P("sn_%s.f32" % nm), sn)
f32(P("sel_pre.f32"), np.ones((NP, 1)) if MUT == "prefix-sampled-t" else np.zeros((NP, 1)))  # the prefix reads the t = 0 row of the modulation
f32(P("sel_tgt.f32"), np.ones((NP, 1)))
ap = allowed[np.ix_(pre, pre)].copy()
if MUT == "cond-causal":  # the condition image made causal instead of bidirectional inside itself
    ap = np.tril(np.ones_like(ap))
mp = np.full((NP, NP), NEG)
for i in range(NP):
    for_i = np.ones(NPR, dtype=bool) if i >= NPR else ap[i]
    mp[i, :NPR][for_i] = 0.0
f32(P("mask_pre.f32"), mp)
ms = np.full((NP, 2 * NP), NEG)
ms[:, :NPR] = 0.0
ms[:, NP:NP + NTG] = 0.0
f32(P("mask_step.f32"), ms)
for i, nm in enumerate(("s1", "g1", "s2", "g2")):
    f32(P("ms_%s.bin" % nm), mod[:, i * D:(i + 1) * D])


class Session:
    def __init__(self, name):
        self.name, self.lines = name, []

    def program(self, n, kernel, args):
        self.lines.append(f"program {n} {CHX}/{kernel}.chx {args}")

    def res(self, target, path, flag="f32"):
        self.lines.append(f"resident {target} {path} {flag}")

    def link(self, a, b):
        self.lines.append(f"link {a} -> {b}")

    def save(self, out, path, flag):
        self.lines.append(f"save {out} {path} {flag}")

    def run(self, probe_of, probe_m, probe_n):
        self.program("probe", "dit_rows2", f"m={probe_m} n={probe_n} bn=1024")
        self.link(probe_of, "probe.x")
        open(P(self.name + ".session"), "w").write("\n".join(self.lines) + "\nsteps\nanswer probe.y\n")
        p = subprocess.Popen([SESSION, P(self.name + ".session")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        reqs = []
        for line in (["1\n", "1\n"] if TIME else ["1\n"]) + ["checkpoint\n"]:
            t1 = time.perf_counter()
            p.stdin.write(line)
            p.stdin.flush()
            out = p.stdout.readline()
            reqs.append(time.perf_counter() - t1)
            if not out:
                raise SystemExit("session %s stopped: %s" % (self.name, p.stderr.read()[:800]))
        p.stdin.close()
        p.wait()
        return reqs[-2] if TIME else reqs[0]


def blocks_session(name, lo, hi, xfile, prefix):
    s = Session(name)
    suf = "pre" if prefix else "tgt"
    prev = None
    for b in range(lo, hi):
        n = lambda t: "b%d_%s" % (b, t)
        sel_f = P("sel_%s.f32" % suf)
        s.program(n("lnm1"), "dit_lnmod", f"s={NP} n={D}")
        if prev:
            s.link(prev, n("lnm1") + ".x")
        else:
            s.res(n("lnm1") + ".x", xfile)
        s.res(n("lnm1") + ".sc", P("ms_s1.bin")); s.res(n("lnm1") + ".sel", sel_f)
        for gg, wf in (("gq", "wq"), ("gk", "wk"), ("gv", "wv")):
            s.program(n(gg), "ported/gemm_mma", f"m={NP} n={D} k={D}")
            s.link(n("lnm1") + ".y", n(gg) + ".a"); s.res(n(gg) + ".w", W("%s%d.f16" % (wf, b)), "f16")
        for nm, src, wf in (("nq", "gq", "nq"), ("nk", "gk", "nk")):
            s.program(n(nm), "rmsnorm_rows", f"r={NP * H} n={HD} eps=0.000001")
            s.link(n(src) + ".y", n(nm) + ".x"); s.res(n(nm) + ".w", W("%s%d.f32" % (wf, b)))
        for rn, src, sc_ in (("rq", "nq", repr(1 / math.sqrt(HD))), ("rk", "nk", "1.0")):
            s.program(n(rn), "dit_rope16", f"s={NP} nh={H} h=64 sc={sc_}")
            s.link(n(src) + ".y", n(rn) + ".x"); s.res(n(rn) + ".cs", P("cs_%s.f32" % suf)); s.res(n(rn) + ".sn", P("sn_%s.f32" % suf))
        s.program(n("vh"), "dit_to_f16", f"s={NP} n={D}"); s.link(n("gv") + ".y", n("vh") + ".x")
        if prefix:
            s.save(n("rk") + ".y", P("kpre%d.f16" % b), "f16")
            s.save(n("vh") + ".y", P("vpre%d.f16" % b), "f16")
            kk, vv, t, maskf = n("rk") + ".y", n("vh") + ".y", NP, "mask_pre.f32"
        else:
            kk, vv, t, maskf = n("ck") + ".y", n("cv") + ".y", 2 * NP, "mask_step.f32"
            if MUT == "no-cat":  # the cached prefix left out: keys are the target's own twice
                s.program(n("ck"), "dit_cat", f"s={NP} n={D} bn=1024"); s.link(n("rk") + ".y", n("ck") + ".a"); s.link(n("rk") + ".y", n("ck") + ".b")
                s.program(n("cv"), "dit_cat", f"s={NP} n={D} bn=1024"); s.link(n("vh") + ".y", n("cv") + ".a"); s.link(n("vh") + ".y", n("cv") + ".b")
            else:
                s.program(n("ck"), "dit_cat", f"s={NP} n={D} bn=1024"); s.res(n("ck") + ".a", P("kpre%d.f16" % b), "f16"); s.link(n("rk") + ".y", n("ck") + ".b")
                s.program(n("cv"), "dit_cat", f"s={NP} n={D} bn=1024"); s.res(n("cv") + ".a", P("vpre%d.f16" % b), "f16"); s.link(n("vh") + ".y", n("cv") + ".b")
        s.program(n("sc"), "dit_scores_mma", f"nh={H} nq={NP} t={t} d={HD}")
        s.link(n("rq") + ".y", n("sc") + ".q"); s.link(kk, n("sc") + ".k")
        s.program(n("st"), "dit_stats_m", f"nh={H} nq={NP} t={t} qs=32 cs=64"); s.link(n("sc") + ".sc", n("st") + ".sc"); s.res(n("st") + ".mask", P(maskf))
        s.program(n("pv"), "dit_pv_mma", f"nh={H} nq={NP} t={t} d={HD}")
        s.link(n("sc") + ".sc", n("pv") + ".sc"); s.res(n("pv") + ".mask", P(maskf)); s.link(n("st") + ".m", n("pv") + ".m"); s.link(n("st") + ".z", n("pv") + ".z"); s.link(vv, n("pv") + ".v")
        s.program(n("c1"), "dit_to_f16", f"s={NP} n={D}"); s.link(n("pv") + ".o", n("c1") + ".x")
        s.program(n("go"), "ported/gemm_mma", f"m={NP} n={D} k={D}"); s.link(n("c1") + ".y", n("go") + ".a"); s.res(n("go") + ".w", W("wo%d.f16" % b), "f16")
        s.program(n("gr1"), "dit_gate_res", f"s={NP} n={D}")
        if prev:
            s.link(prev, n("gr1") + ".x")
        else:
            s.res(n("gr1") + ".x", xfile)
        s.link(n("go") + ".y", n("gr1") + ".y"); s.res(n("gr1") + ".g", P("ms_g1.bin")); s.res(n("gr1") + ".sel", sel_f)
        s.program(n("lnm2"), "dit_lnmod", f"s={NP} n={D}"); s.link(n("gr1") + ".z", n("lnm2") + ".x"); s.res(n("lnm2") + ".sc", P("ms_s2.bin")); s.res(n("lnm2") + ".sel", sel_f)
        for gg, wf in (("gg", "wg"), ("gp", "wp")):
            s.program(n(gg), "ported/gemm_mma", f"m={NP} n={3 * D} k={D}")
            s.link(n("lnm2") + ".y", n(gg) + ".a"); s.res(n(gg) + ".w", W("%s%d.f16" % (wf, b)), "f16")
        s.program(n("sw"), "dit_swiglu", f"s={NP} n={3 * D} bn=1024"); s.link(n("gg") + ".y", n("sw") + ".g"); s.link(n("gp") + ".y", n("sw") + ".u")
        s.program(n("gd"), "ported/gemm_mma", f"m={NP} n={D} k={3 * D}"); s.link(n("sw") + ".y", n("gd") + ".a"); s.res(n("gd") + ".w", W("wout%d.f16" % b), "f16")
        s.program(n("gr2"), "dit_gate_res", f"s={NP} n={D}")
        s.link(n("gr1") + ".z", n("gr2") + ".x"); s.link(n("gd") + ".y", n("gr2") + ".y"); s.res(n("gr2") + ".g", P("ms_g2.bin")); s.res(n("gr2") + ".sel", sel_f)
        prev = n("gr2") + ".z"
    outf = P("x_%s_%d.bin" % (suf, hi))
    s.save(prev, outf, "f32")
    secs = s.run(prev, NP, D)
    return outf, secs


tp = 0.0
x = P("xpre.f32")
for lo in range(0, NB, GP):
    x, secs = blocks_session("pre%d" % lo, lo, lo + GP, x, True)
    tp += secs
print("prefix pass: %.2f s (%s)" % (tp, "steady state, weights resident" if TIME else "first requests: includes each session's compile and upload"), flush=True)
ts = 0.0
x = P("xtgt.f32")
mx = 0.0
for lo in range(0, NB, GS):
    x, secs = blocks_session("stp%d" % lo, lo, lo + GS, x, False)
    ts += secs
    got = np.fromfile(x, dtype="<f4").reshape(NP, D)[:NTG].astype(np.float64)
    wb = blocks[lo + GS - 1][tgt].astype(np.float64)
    r = float(np.sqrt((wb ** 2).mean()))
    print("   after block %2d: target rows vs float64: mean abs error %.4g (%.3g of the rms %.3g), max %.4g" % (lo + GS, np.abs(got - wb).mean(), np.abs(got - wb).mean() / r, r, np.abs(got - wb).max()), flush=True)
print("cached step: %.2f s (%s)" % (ts, "steady state, weights resident" if TIME else "first requests"))
y = step.back(got, temb, np.ones(NTG, dtype=bool))
d = np.abs(y - want_out)
rms = float(np.sqrt((want_out ** 2).mean()))
print("step output [%d, 64] vs float64: mean abs error %.4g (%.3g of the rms %.3g), max %.4g" % (NTG, d.mean(), d.mean() / rms, rms, d.max()))
BOUND = 5e-3
ok = d.mean() / rms <= BOUND
print("%s the edit step on the GPU: output within %.0e of the rms of the float64 step" % ("ok  " if ok else "FAIL", BOUND))
sys.exit(0 if ok else 1)
