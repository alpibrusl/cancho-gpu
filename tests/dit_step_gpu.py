#!/usr/bin/env python3
"""D2d of docs/diffusion.md: the 32 transformer blocks of Qwen-Image-2.1-Turbo on a Metal GPU, as one session of 640 kernel programs, held to the float64 step and to torch's BF16 output.

    dit_step_gpu.py GOLDEN_DIR REF_DIR [--keep DIR] [--blocks N]

REF_DIR is the output of `python tools/dit_step_ref.py GOLDEN_DIR REF_DIR` (the float64 residual stream after every block). Needs a Mac, the checkpoint, D0's goldens, numpy, torch and safetensors (the venv
of the first image edits has them), `build/cancho-gpu-metal-session` (`scripts/build-metal-session.sh`), and about 15 GB of disk for the f16 weight files (`--keep DIR` says where; they are written once
per run). Not in `scripts/gate.sh`.

The block is D2c's (tests/dit_block_gpu.py has the layout and the reasons: 277 tokens padded to 320, f16 weights with q and k permuted within each head, the host's modulation, RoPE tables and mask),
repeated: the residual stream of one block is linked to the next block's inputs, on the device. The front end (img_in, txt_in, the timestep embedding, the modulation) and the back end (norm_out, proj_out)
are computed on the host in float64 (tools/dit_step_ref.py) in this slice: they are small (one 277 x 4096 product per layer) and not what D2d is about; D2e puts them on the GPU.

Reported: the residual after blocks 4, 8, 16, 24 and 32 against the float64 reference (the session answers those tensors), the step's output (norm_out and proj_out applied to the GPU's last residual) against
the float64 step and against torch's BF16 `step0_out`, and the session's time.
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import dit_block_ref as ref  # noqa: E402
import dit_step_ref as stepref  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gold, refdir = sys.argv[1], sys.argv[2]
keep = sys.argv[sys.argv.index("--keep") + 1] if "--keep" in sys.argv else None
NB = int(sys.argv[sys.argv.index("--blocks") + 1]) if "--blocks" in sys.argv else 32
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
CHX = os.path.join(ROOT, "tests/chx")
NEG = -1e9
D, H, HD, S, SP = 4096, 32, 128, 277, 320
perm = np.array(list(range(0, HD, 2)) + list(range(1, HD, 2)))
SHOW = [b for b in (4, 8, 16, 24, 32) if b <= NB]

_, _, mod0, rope, tmask = ref.goldens(gold)
x, temb, mod = stepref.front(gold)


def head_perm_rows(m):
    return m.reshape(H, HD, -1)[:, perm, :].reshape(H * HD, -1)


def f16(path, a):
    np.ascontiguousarray(a.astype("<f2")).tofile(path)


def f32(path, a):
    np.ascontiguousarray(a.astype("<f4")).tofile(path)


work = keep or tempfile.mkdtemp()
os.makedirs(work, exist_ok=True)
P = lambda n: os.path.join(work, n)
xp = np.zeros((SP, D))
xp[:S] = x
f32(P("x.f32"), xp)
sel = np.zeros((SP, 1))
sel[:S, 0] = tmask
f32(P("sel.f32"), sel)
for i, name in enumerate(("sc1", "g1", "sc2", "g2")):
    f32(P(name + ".f32"), mod[:, i * D:(i + 1) * D])
cs, sn = np.ones((SP, 64)), np.zeros((SP, 64))
cs[:S], sn[:S] = rope[..., 0], rope[..., 1]
f32(P("cs.f32"), cs)
f32(P("sn.f32"), sn)
idx = np.arange(SP)
tm = np.zeros(SP, dtype=bool)
tm[:S] = tmask
allowed = ((idx[:, None] >= idx[None, :]) | (tm[:, None] & tm[None, :])) & (idx < S)[None, :]
f32(P("mask.f32"), np.where(allowed, 0.0, NEG))

t0 = time.time()
for b in range(NB):
    t = lambda k: ref.weight("transformer_blocks.%d.%s" % (b, ref.BLOCK0[k])).float().numpy()
    f16(P("wq%d.f16" % b), head_perm_rows(t("q")))
    f16(P("wk%d.f16" % b), head_perm_rows(t("k")))
    for k, n in (("v", "wv"), ("o", "wo"), ("g", "wg"), ("p", "wp"), ("out", "wout")):
        f16(P("%s%d.f16" % (n, b)), t(k))
    f32(P("nq%d.f32" % b), t("nq")[perm][None])
    f32(P("nk%d.f32" % b), t("nk")[perm][None])
print("weights written: %d blocks in %.0f s" % (NB, time.time() - t0), flush=True)

prog = []


def program(name, kernel, args):
    prog.append(f"program {name} {CHX}/{kernel}.chx {args}")


def res(target, fname, flag="f32"):
    prog.append(f"resident {target} {P(fname)} {flag}")


def link(a, b):
    prog.append(f"link {a} -> {b}")


for b in range(NB):
    n = lambda s: "b%d_%s" % (b, s)
    program(n("lnm1"), "dit_lnmod", f"s={SP} n={D}")
    if b == 0:
        res(n("lnm1") + ".x", "x.f32")
    else:
        link("b%d_gr2.z" % (b - 1), n("lnm1") + ".x")
    res(n("lnm1") + ".sc", "sc1.f32"); res(n("lnm1") + ".sel", "sel.f32")
    for g, wf in (("gq", "wq"), ("gk", "wk"), ("gv", "wv")):
        program(n(g), "ported/gemm_mma", f"m={SP} n={D} k={D}")
        link(n("lnm1") + ".y", n(g) + ".a"); res(n(g) + ".w", "%s%d.f16" % (wf, b), "f16")
    for nm, src, wf in (("nq", "gq", "nq"), ("nk", "gk", "nk")):
        program(n(nm), "rmsnorm_rows", f"r={SP * H} n={HD} eps=0.000001")
        link(n(src) + ".y", n(nm) + ".x"); res(n(nm) + ".w", "%s%d.f32" % (wf, b))
    for rn, src in (("rq", "nq"), ("rk", "nk")):
        program(n(rn), "dit_rope", f"s={SP} nh={H} h=64")
        link(n(src) + ".y", n(rn) + ".x"); res(n(rn) + ".cs", "cs.f32"); res(n(rn) + ".sn", "sn.f32")
    program(n("sc"), "dit_scores", f"nh={H} nq={SP} t={SP} qs=32 ts=64 d={HD}")
    link(n("rq") + ".y", n("sc") + ".q"); link(n("rk") + ".y", n("sc") + ".k"); res(n("sc") + ".mask", "mask.f32")
    program(n("st"), "dit_stats", f"nh={H} nq={SP} t={SP} qs=32 cs=64"); link(n("sc") + ".sc", n("st") + ".sc")
    program(n("pv"), "dit_pv", f"nh={H} nq={SP} t={SP} qs=32 cs=64 dw=32 d={HD}")
    link(n("sc") + ".sc", n("pv") + ".sc"); link(n("st") + ".m", n("pv") + ".m"); link(n("st") + ".z", n("pv") + ".z"); link(n("gv") + ".y", n("pv") + ".v")
    program(n("c1"), "dit_to_f16", f"s={SP} n={D}"); link(n("pv") + ".o", n("c1") + ".x")
    program(n("go"), "ported/gemm_mma", f"m={SP} n={D} k={D}"); link(n("c1") + ".y", n("go") + ".a"); res(n("go") + ".w", "wo%d.f16" % b, "f16")
    program(n("gr1"), "dit_gate_res", f"s={SP} n={D}")
    if b == 0:
        res(n("gr1") + ".x", "x.f32")
    else:
        link("b%d_gr2.z" % (b - 1), n("gr1") + ".x")
    link(n("go") + ".y", n("gr1") + ".y"); res(n("gr1") + ".g", "g1.f32"); res(n("gr1") + ".sel", "sel.f32")
    program(n("lnm2"), "dit_lnmod", f"s={SP} n={D}"); link(n("gr1") + ".z", n("lnm2") + ".x"); res(n("lnm2") + ".sc", "sc2.f32"); res(n("lnm2") + ".sel", "sel.f32")
    for g, wf in (("gg", "wg"), ("gp", "wp")):
        program(n(g), "ported/gemm_mma", f"m={SP} n={3 * D} k={D}")
        link(n("lnm2") + ".y", n(g) + ".a"); res(n(g) + ".w", "%s%d.f16" % (wf, b), "f16")
    program(n("sw"), "dit_swiglu", f"s={SP} n={3 * D} bn=1024"); link(n("gg") + ".y", n("sw") + ".g"); link(n("gp") + ".y", n("sw") + ".u")
    program(n("gd"), "ported/gemm_mma", f"m={SP} n={D} k={3 * D}"); link(n("sw") + ".y", n("gd") + ".a"); res(n("gd") + ".w", "wout%d.f16" % b, "f16")
    program(n("gr2"), "dit_gate_res", f"s={SP} n={D}")
    link(n("gr1") + ".z", n("gr2") + ".x"); link(n("gd") + ".y", n("gr2") + ".y"); res(n("gr2") + ".g", "g2.f32"); res(n("gr2") + ".sel", "sel.f32")
prog.append("steps")
for b in SHOW:
    prog.append("answer b%d_gr2.z" % (b - 1))
open(P("step.session"), "w").write("\n".join(prog) + "\n")

t0 = time.time()
r = subprocess.run([SESSION, P("step.session")], input="1\n", capture_output=True, text=True, timeout=7200)
secs = time.time() - t0
if r.returncode != 0 or not r.stdout.strip():
    print("FAIL the session did not answer: status %d, %s" % (r.returncode, r.stderr.strip()[:800]))
    sys.exit(1)
nums = np.array(r.stdout.split(), dtype=np.float64).reshape(len(SHOW), SP, D)[:, :S]
print("session: %d programs, answered in %.1f s (Metal; includes the compile and the upload of %.1f GB of weights)" % (sum(1 for l in prog if l.startswith("program")), secs, NB * 0.4358))

blocks = np.load(os.path.join(refdir, "blocks_f64.npy")).astype(np.float64)
ok = True
for i, b in enumerate(SHOW):
    d = np.abs(nums[i] - blocks[b - 1])
    rms = float(np.sqrt((blocks[b - 1] ** 2).mean()))
    print("residual after block %2d: mean abs error %.4g (%.3g of the rms %.3g), max abs error %.4g" % (b, d.mean(), d.mean() / rms, rms, d.max()))
if NB == 32:
    y = stepref.back(nums[-1], temb, tmask)
    exact = np.load(os.path.join(refdir, "step_out_f64.npy"))
    want = np.load(os.path.join(gold, "step0_out.npy")).astype(np.float64)[0]
    rms = float(np.sqrt((want ** 2).mean()))
    for label, a, bb in (("GPU step vs float64 step", y, exact), ("GPU step vs torch BF16 (step0_out)", y, want), ("(float64 step vs torch BF16, for scale)", exact, want)):
        d = np.abs(a - bb)
        print("%-42s mean abs error %.4g (%.3g of the output rms %.3g), max abs error %.4g" % (label, d.mean(), d.mean() / rms, rms, d.max()))
    ok = float(np.abs(y - exact).mean()) / rms <= 1e-2
    print("%s the 32 blocks on the GPU" % ("ok  " if ok else "FAIL"))
sys.exit(0 if ok else 1)
