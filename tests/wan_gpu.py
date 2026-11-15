#!/usr/bin/env python3
"""V1 of docs/diffusion.md section 6: blocks of Wan2.2-TI2V-5B's transformer on a Metal GPU, from models/wan_dit.model, held to the float64 reference of tools/wan_ref.py.

    wan_gpu.py WORK_DIR [--blocks B] [--head] [--frames F] [--size HxW] [--text T] [--seed S] [--fused] [--no-ref] [--mutate NAME]

Needs a Mac, the first shard of `Wan-AI/Wan2.2-TI2V-5B-Diffusers` in the Hugging Face cache (blocks 0 to 6 are whole in it), numpy and safetensors (the venv of the first image edits; `--bf16` also needs torch and diffusers), and `build/cancho-gpu-metal-session`.
Not in `scripts/gate.sh`. The text's tokens must be a multiple of 64; the latent's tokens (F x H/2 x W/2) are padded to one (the padded keys are hidden by the self-attention's mask, as in the edit's). `--head` adds the output layer (norm, `proj_out`) and compares the velocity [48, F, H, W].

The host does what D6 left to it: the patch embedding, the condition embedder (a timestep a token: the conditioning frame's tokens at t = 0), the text's projection, the rotary tables, and the f16 weight files, with the **channels of a head of q and k of the self-attention permuted even-first**
(and their norms' weights: an RMS norm over all 3,072 channels does not mind), so that `dit_rope16`'s rotate-half is the model's rotary rotation of adjacent pairs (tests/dit_block.py holds the permutation to the model's arithmetic). The GPU does the blocks.
`--mutate` makes one deliberate host-side error that the comparison must refuse: `no-permutation` (the weights' channels in the model's order).
"""
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import wan_ref as wr  # noqa: E402

arg = lambda name, default=None: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
work = sys.argv[1]
NB, F = int(arg("--blocks", 1)), int(arg("--frames", 2))
H, Wd = (int(v) for v in arg("--size", "16x16").split("x"))
T, SEED, MUT = int(arg("--text", 64)), int(arg("--seed", 0)), arg("--mutate")
D, NH, HD, FF = wr.D, wr.NH, wr.HD, wr.FFN
os.makedirs(work, exist_ok=True)
P = lambda n: os.path.join(work, n)
f16 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f2")).tofile(path)
f32 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f4")).tofile(path)
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")

lat, txt, t = wr.inputs(F, H, Wd, T, SEED)
S = len(t)
SP = (S + 63) // 64 * 64
assert T % 64 == 0, "the text's tokens (%d) must be a multiple of 64" % T
HEAD = "--head" in sys.argv
stages = {}
NOREF = "--no-ref" in sys.argv  # time only: the host's front end, no float64 blocks
want = wr.forward(lat, txt, t, 0 if NOREF else NB, stages, with_head=HEAD and not NOREF)
print("latent [48, %d, %d, %d]: %d tokens (padded to %d), text %d, %d blocks%s; float64 output rms %.4g" % (F, H, Wd, S, SP, T, NB, " and the output layer" if HEAD else "", np.sqrt((want ** 2).mean())), flush=True)
NEG = -1e9
pad = lambda a, fill: np.concatenate([a, np.full((SP - S,) + a.shape[1:], fill)], 0)

# ---- the weights: f16 files, q and k of the self-attention with the channels of each head permuted even-first
idx = np.empty(D, dtype=int)
for h in range(NH):
    for i in range(HD // 2):
        idx[h * HD + i], idx[h * HD + HD // 2 + i] = h * HD + 2 * i, h * HD + 2 * i + 1
perm = (lambda a: a) if MUT == "no-permutation" else (lambda a: a[idx])
vec = lambda a: np.asarray(a)[None]
for b in range(NB):
    p = "blocks.%d." % b
    for nm in "qk":
        f16(P("w%s%d.f16" % (nm, b)), perm(wr.W(p + "attn1.to_%s.weight" % nm)))
        f32(P("b%s%d.f32" % (nm, b)), vec(perm(wr.W(p + "attn1.to_%s.bias" % nm))))
        f32(P("n%s%d.f32" % (nm, b)), vec(perm(wr.W(p + "attn1.norm_%s.weight" % nm))))
    f16(P("wv%d.f16" % b), wr.W(p + "attn1.to_v.weight"))
    f32(P("bv%d.f32" % b), vec(wr.W(p + "attn1.to_v.bias")))
    f16(P("wo%d.f16" % b), wr.W(p + "attn1.to_out.0.weight"))
    f32(P("bo%d.f32" % b), vec(wr.W(p + "attn1.to_out.0.bias")))
    f32(P("n2w%d.f32" % b), vec(wr.W(p + "norm2.weight")))
    f32(P("n2b%d.f32" % b), vec(wr.W(p + "norm2.bias")))
    for nm in "qkv":
        f16(P("w2%s%d.f16" % (nm, b)), wr.W(p + "attn2.to_%s.weight" % nm))
        f32(P("b2%s%d.f32" % (nm, b)), vec(wr.W(p + "attn2.to_%s.bias" % nm)))
    f32(P("n2q%d.f32" % b), vec(wr.W(p + "attn2.norm_q.weight")))
    f32(P("n2k%d.f32" % b), vec(wr.W(p + "attn2.norm_k.weight")))
    f16(P("w2o%d.f16" % b), wr.W(p + "attn2.to_out.0.weight"))
    f32(P("b2o%d.f32" % b), vec(wr.W(p + "attn2.to_out.0.bias")))
    f16(P("wf1%d.f16" % b), wr.W(p + "ffn.net.0.proj.weight"))
    f32(P("bf1%d.f32" % b), vec(wr.W(p + "ffn.net.0.proj.bias")))
    f16(P("wf2%d.f16" % b), wr.W(p + "ffn.net.2.weight"))
    f32(P("bf2%d.f32" % b), vec(wr.W(p + "ffn.net.2.bias")))
    # the modulation: scale_shift_table + the timestep's six vectors, one row for the sampled t and one for t = 0 (the first frame's tokens)
    tab = wr.W(p + "scale_shift_table")[0]
    row_t, row_0 = tab + stages["proj"][S - 1], tab + stages["proj"][0]
    for k, nm in enumerate(("sh1", "sc1", "g1", "sh2", "sc2", "g2")):
        f32(P("m%d_%s.f32" % (b, nm)), np.stack([row_t[k], row_0[k]]))

# ---- the inputs
f32(P("x0.f32"), pad(stages["patch"], 0.0))
f16(P("ctx.f16"), stages["text"])
f32(P("cs.f32"), pad(stages["cos"], 1.0))
f32(P("sn.f32"), pad(stages["sin"], 0.0))
f32(P("sel.f32"), pad((t > 0).astype(np.float64)[:, None], 1.0))
f32(P("ones_s.f32"), np.ones((SP, 1)))
f32(P("ones_t.f32"), np.ones((T, 1)))
ms = np.zeros((SP, SP))
ms[:, S:] = NEG  # the padded keys
f32(P("mask_self.f32"), ms)
f32(P("mask_cross.f32"), np.zeros((SP, T)))
f32(P("ci_s.f32"), np.ones((SP, HD // 2)))
f32(P("si_s.f32"), np.zeros((SP, HD // 2)))
if HEAD:
    tab = wr.W("scale_shift_table")[0]
    temb_t, temb_0 = stages["temb"][S - 1], stages["temb"][0]
    f32(P("head_sh.f32"), np.stack([tab[0] + temb_t, tab[0] + temb_0]))
    f32(P("head_sc.f32"), np.stack([tab[1] + temb_t, tab[1] + temb_0]))
    f16(P("proj_out.f16"), wr.W("proj_out.weight"))
    f32(P("proj_out_b.f32"), vec(wr.W("proj_out.bias")))
f32(P("init_s.f32"), np.full((SP, 1), -3e4))
f32(P("ci_t.f32"), np.ones((T, HD // 2)))
f32(P("si_t.f32"), np.zeros((T, HD // 2)))

# ---- the session, from the model file
defs = ["CHX=" + os.path.join(ROOT, "tests/chx"), "WORK=" + work, "WDIR=" + work, "S=%d" % SP, "T=%d" % T, "NB=%d" % NB, "HEAD=%d" % int(HEAD), "ATT=%d" % int("--fused" in sys.argv)]
model = os.path.join(ROOT, "models/wan_dit.model")
import time  # noqa: E402

t0 = time.time()
r = subprocess.run([SESSION, model] + defs, input="1\ncheckpoint\n", capture_output=True, text=True, timeout=7200)
secs = time.time() - t0
if r.returncode != 0 or not os.path.exists(P("x_out.f32")):
    print("FAIL the session did not answer: status %d, %s" % (r.returncode, r.stderr.strip()[:800]))
    sys.exit(1)
print("session: %d blocks answered in %.1f s (Metal; includes the compile and the upload of %.2f GB of weights)" % (NB, secs, NB * 0.33), flush=True)
if HEAD:
    y = np.fromfile(P("head_out.f32"), dtype="<f4").reshape(SP, 192)[:S].astype(np.float64)
    got = y.reshape(F, H // 2, Wd // 2, 1, 2, 2, 48).transpose(6, 0, 3, 1, 4, 2, 5).reshape(48, F, H, Wd)
else:
    got = np.fromfile(P("x_out.f32"), dtype="<f4").reshape(SP, D)[:S].astype(np.float64)
if NOREF:
    print("no reference: output rms %.4g, finite %s" % (np.sqrt((got ** 2).mean()), bool(np.isfinite(got).all())))
    sys.exit(0)
rms = float(np.sqrt((want ** 2).mean()))
d = np.abs(got - want)
print("output %s vs float64: mean abs error %.4g (%.3g of the rms %.3g), max %.4g" % (list(got.shape), d.mean(), d.mean() / rms, rms, d.max()))
BOUND = 5e-4 if NB <= 7 else 4e-3  # a tenth of torch's BF16 floor, written before the first run: about 5e-3 after 1 to 7 blocks and 3.9e-2 for the whole model (tools/wan_ref.py --full --bf16)
ok = d.mean() / rms <= BOUND
print("%s the Wan block(s) on the GPU: output within %.0e of the rms of the float64 reference (torch's BF16 floor, tools/wan_ref.py --bf16, is about 5e-3 to 7e-3 after 1 to 7 blocks, 3.9e-2 for the whole model)" % ("ok  " if ok else "FAIL", BOUND))
sys.exit(0 if ok else 1)
