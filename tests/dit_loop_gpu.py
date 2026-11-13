#!/usr/bin/env python3
"""D3 of docs/diffusion.md: the eight denoising steps of Qwen-Image-2.1-Turbo on a Metal GPU in one session, the latents carried on the device from one step to the next (the Euler update is a kernel too) -- the front end (txt_in, img_in, the timestep embedding, the shared modulation), the 32 blocks, the back end
(norm_out, proj_out) and the update latents <- latents + (sigma_next - sigma) * prediction -- held to the float64 trajectory and to torch's BF16 latents after every step (D0's `latents_N`).

    dit_loop_gpu.py GOLDEN_DIR TRAJ_DIR [--keep DIR] [--steps N]

TRAJ_DIR is the output of `python tools/dit_traj_ref.py GOLDEN_DIR TRAJ_DIR` (the float64 trajectory; optional: without it only torch's BF16 latents are compared).

REF_DIR is the output of `python tools/dit_step_ref.py GOLDEN_DIR REF_DIR` (the float64 residual stream after every block). Needs a Mac, the checkpoint, D0's goldens, numpy, torch and safetensors (the venv
of the first image edits has them), `build/cancho-gpu-metal-session` (`scripts/build-metal-session.sh`), and about 15 GB of disk for the f16 weight files (`--keep DIR` says where; they are written once
per run). Not in `scripts/gate.sh`.

The block is D2c's (tests/dit_block_gpu.py has the layout and the reasons: 277 tokens padded to 320, f16 weights with q and k permuted within each head, the host's modulation, RoPE tables and mask),
repeated: the residual stream of one block is linked to the next block's inputs, on the device. D2d (tests/dit_step_gpu.py) had the front and back ends on the host in float64; here they are kernels too. What stays on the host: the timestep's sinusoid (a table of cos and sin, as RoPE's), the zero-padded text and latent
matrices, and the weights as f16 files.

Reported: the residual after blocks 4, 8, 16, 24 and 32 against the float64 reference (the session answers those tensors), the step's output (norm_out and proj_out applied to the GPU's last residual) against
the float64 step and against torch's BF16 `step0_out`, and the session's time.
"""
import math
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
gold, trajdir = sys.argv[1], sys.argv[2]
NSTEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 8
keep = sys.argv[sys.argv.index("--keep") + 1] if "--keep" in sys.argv else None
NB = 32
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
CHX = os.path.join(ROOT, "tests/chx")
NEG = -1e9
D, H, HD, S, SP = 4096, 32, 128, 277, 320
perm = np.array(list(range(0, HD, 2)) + list(range(1, HD, 2)))
SHOW = [b for b in (4, 8, 16, 24, 32) if b <= NB]

_, _, mod0, rope, tmask = ref.goldens(gold)


def head_perm_rows(m):
    return m.reshape(H, HD, -1)[:, perm, :].reshape(H * HD, -1)


def f16(path, a):
    np.ascontiguousarray(a.astype("<f2")).tofile(path)


def f32(path, a):
    np.ascontiguousarray(a.astype("<f4")).tofile(path)


work = keep or tempfile.mkdtemp()
os.makedirs(work, exist_ok=True)
P = lambda n: os.path.join(work, n)
sel = np.zeros((SP, 1))
sel[:S, 0] = tmask
f32(P("sel.f32"), sel)
cs, sn = np.ones((SP, 64)), np.zeros((SP, 64))
cs[:S], sn[:S] = rope[..., 0], rope[..., 1]
f32(P("cs.f32"), cs)
f32(P("sn.f32"), sn)
idx = np.arange(SP)
tm = np.zeros(SP, dtype=bool)
tm[:S] = tmask
allowed = ((idx[:, None] >= idx[None, :]) | (tm[:, None] & tm[None, :])) & (idx < S)[None, :]
f32(P("mask.f32"), np.where(allowed, 0.0, NEG))

ld = lambda n: np.load(os.path.join(gold, n + ".npy")).astype(np.float64)
latents, text = ld("step0_in")[0], ld("text_embeds")[0]
sigmas, timesteps = ld("sigmas"), ld("timesteps")
lat = np.zeros((SP, 64))
lat[21:S] = latents
f16(P("lat.f16"), lat)
f32(P("lat.f32"), lat)
f32(P("dts.f32"), np.array([sigmas[i + 1] - sigmas[i] for i in range(8)]))
txt = np.zeros((SP, D))
txt[:text.shape[0]] = text
f32(P("txt.f32"), txt)
f32(P("tw1.f32"), (ref.weight("txt_in.text_norm.weight").double().numpy() + 1)[None])
freqs = np.exp(-math.log(10000) * np.arange(128) / 128)
emb = np.zeros((8, 64, 256))
for i in range(8):
    args = np.array([timesteps[i] / 1000.0, 0.0])[:, None] * 1000.0 * freqs[None]
    emb[i, :2] = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
f16(P("emb.f16"), emb)
for name, key in (("wti", "txt_in.in_layer.weight"), ("wto", "txt_in.out_layer.weight"), ("wim", "img_in.weight"), ("wt1", "time_text_embed.timestep_embedder.linear_1.weight"),
                  ("wt2", "time_text_embed.timestep_embedder.linear_2.weight"), ("wmo", "modulation.1.weight"), ("wno", "norm_out.linear.weight"), ("wpj", "proj_out.weight")):
    f16(P(name + ".f16"), ref.weight(key).float().numpy())

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


program("tn", "rmsnorm_rows", f"r={SP} n={D} eps=0.000001"); res("tn.x", "txt.f32"); res("tn.w", "tw1.f32")
program("tnc", "dit_to_f16", f"s={SP} n={D}"); link("tn.y", "tnc.x")
program("ti", "ported/gemm_mma", f"m={SP} n={D} k={D}"); link("tnc.y", "ti.a"); res("ti.w", "wti.f16", "f16")
program("tg", "dit_gelu", f"s={SP} n={D} bn=1024"); link("ti.y", "tg.x")
program("to", "ported/gemm_mma", f"m={SP} n={D} k={D}"); link("tg.y", "to.a"); res("to.w", "wto.f16", "f16")
program("ii", "ported/gemm_mma", f"m={SP} n={D} k=64"); res("ii.a", "lat.f16", "f16"); res("ii.w", "wim.f16", "f16")
program("jn", "dit_add", f"s={SP} n={D} bn=1024"); link("to.y", "jn.a"); link("ii.y", "jn.b")
program("t1", "ported/gemm_mma", f"m=64 n={D} k=256"); prog.append("table t1.a %s 8 f16" % P("emb.f16")); res("t1.w", "wt1.f16", "f16")
program("ts", "dit_silu", f"s=64 n={D} bn=1024"); link("t1.y", "ts.x")
program("t2", "ported/gemm_mma", f"m=64 n={D} k={D}"); link("ts.y", "t2.a"); res("t2.w", "wt2.f16", "f16")
program("tsil", "dit_silu", f"s=64 n={D} bn=1024"); link("t2.y", "tsil.x")
program("mo", "ported/gemm_mma", f"m=64 n={4 * D} k={D}"); link("tsil.y", "mo.a"); res("mo.w", "wmo.f16", "f16")
program("ms", "dit_modsplit", f"m=64 n={D} bn=1024"); link("mo.y", "ms.md")
for b in range(NB):
    n = lambda s: "b%d_%s" % (b, s)
    program(n("lnm1"), "dit_lnmod", f"s={SP} n={D}")
    link("jn.z" if b == 0 else "b%d_gr2.z" % (b - 1), n("lnm1") + ".x")
    link("ms.s1", n("lnm1") + ".sc"); res(n("lnm1") + ".sel", "sel.f32")
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
    link("jn.z" if b == 0 else "b%d_gr2.z" % (b - 1), n("gr1") + ".x")
    link(n("go") + ".y", n("gr1") + ".y"); link("ms.g1", n("gr1") + ".g"); res(n("gr1") + ".sel", "sel.f32")
    program(n("lnm2"), "dit_lnmod", f"s={SP} n={D}"); link(n("gr1") + ".z", n("lnm2") + ".x"); link("ms.s2", n("lnm2") + ".sc"); res(n("lnm2") + ".sel", "sel.f32")
    for g, wf in (("gg", "wg"), ("gp", "wp")):
        program(n(g), "ported/gemm_mma", f"m={SP} n={3 * D} k={D}")
        link(n("lnm2") + ".y", n(g) + ".a"); res(n(g) + ".w", "%s%d.f16" % (wf, b), "f16")
    program(n("sw"), "dit_swiglu", f"s={SP} n={3 * D} bn=1024"); link(n("gg") + ".y", n("sw") + ".g"); link(n("gp") + ".y", n("sw") + ".u")
    program(n("gd"), "ported/gemm_mma", f"m={SP} n={D} k={3 * D}"); link(n("sw") + ".y", n("gd") + ".a"); res(n("gd") + ".w", "wout%d.f16" % b, "f16")
    program(n("gr2"), "dit_gate_res", f"s={SP} n={D}")
    link(n("gr1") + ".z", n("gr2") + ".x"); link(n("gd") + ".y", n("gr2") + ".y"); link("ms.g2", n("gr2") + ".g"); res(n("gr2") + ".sel", "sel.f32")
program("no", "ported/gemm_mma", f"m=64 n={D} k={D}"); link("tsil.y", "no.a"); res("no.w", "wno.f16", "f16")
program("nr", "dit_rows2", f"m=64 n={D} bn=1024"); link("no.y", "nr.x")
program("fl", "dit_lnmod", f"s={SP} n={D}"); link("b%d_gr2.z" % (NB - 1), "fl.x"); link("nr.y", "fl.sc"); res("fl.sel", "sel.f32")
program("pj", "ported/gemm_mma", f"m={SP} n=64 k={D}"); link("fl.y", "pj.a"); res("pj.w", "wpj.f16", "f16")
program("eu", "dit_euler", f"s={SP} n=64"); res("eu.lat", "lat.f32"); link("pj.y", "eu.pr"); res("eu.sel", "sel.f32"); prog.append("table eu.dt %s 8 f32" % P("dts.f32"))
program("euh", "dit_to_f16", f"s={SP} n=64"); link("eu.z", "euh.x")
prog.append("carry eu.z -> eu.lat")
prog.append("carry euh.y -> ii.a")
prog.append("steps")
prog.append("answer eu.z")
open(P("full.session"), "w").write("\n".join(prog) + "\n")

t0 = time.time()
r = subprocess.run([SESSION, P("full.session")], input="1\n" * NSTEPS, capture_output=True, text=True, timeout=7200)
secs = time.time() - t0
lines = [l for l in r.stdout.split("\n") if l.strip()]
if r.returncode != 0 or len(lines) != NSTEPS:
    print("FAIL the session answered %d of %d steps: status %d, %s" % (len(lines), NSTEPS, r.returncode, r.stderr.strip()[:800]))
    sys.exit(1)
got = [np.array(l.split(), dtype=np.float64).reshape(SP, 64)[21:S] for l in lines]
print("session: %d programs, %d steps in %.1f s (Metal; includes the compile and the upload of 14.3 GB of weights)" % (sum(1 for l in prog if l.startswith("program")), NSTEPS, secs))
f64 = np.load(os.path.join(trajdir, "latents_f64.npy")).astype(np.float64) if os.path.exists(os.path.join(trajdir, "latents_f64.npy")) else None
ok = True
for i in range(NSTEPS):
    torch_l = ld("latents_%d" % i)[0]
    rms = float(np.sqrt((torch_l ** 2).mean()))
    dt_ = np.abs(got[i] - torch_l)
    msg = "step %d: latents rms %.4g; GPU vs torch BF16: mean abs %.3g (%.3g of the rms), max %.3g" % (i, rms, dt_.mean(), dt_.mean() / rms, dt_.max())
    if f64 is not None and i < len(f64):
        d6 = np.abs(got[i] - f64[i])
        d7 = np.abs(torch_l - f64[i])
        msg += ";  GPU vs float64: mean abs %.3g (%.3g of the rms), max %.3g;  torch vs float64: mean abs %.3g, max %.3g" % (d6.mean(), d6.mean() / rms, d6.max(), d7.mean(), d7.max())
    print(msg)
    ok &= bool(np.isfinite(got[i]).all()) and dt_.mean() / rms < 0.05
print("%s the %d steps on the GPU (the mean error against torch's BF16 latents stays under 5%% of their rms)" % ("ok  " if ok else "FAIL", NSTEPS))
sys.exit(0 if ok else 1)
