#!/usr/bin/env python3
"""V4a of docs/diffusion.md section 6: a video from a prompt, the denoiser and the decoder of FastWan2.2-TI2V-5B on a Metal GPU.

    wan_video_gpu.py WORK_DIR --weights FASTWAN_TRANSFORMER_DIR --text TEXT.npy [--frames F] [--size HxW] [--seed S] [--wdir DIR] [--fps N] [--cond COND.f32] [--base [--steps N] [--guidance G] [--neg NEG.npy]] [--noise N.npy]

F is the number of latent frames (the video has 1 + 4 (F - 1)), HxW the latent's size in pixels / 16 (22x40 is 640 x 352); `--text` is tests/wan_text_gpu.py's output (umT5's encoder, a GPU session of its own: V3; tools/wan_text.py, torch on the host, writes the same file); `--wdir` holds the f16 weight files of tests/wan_gpu.py's layout (written on first use, 10 GB); the output is WORK_DIR/video.gif and WORK_DIR/frames/.

The sampler is FastVideo's `DmdDenoisingStage` for this model (the distilled one needs no guidance and no negative prompt): the latents start as Gaussian noise; for t in (1000, 757, 522) the transformer gives the velocity v
of the noisy latents x, the clean latents are x0 = x - sigma(t) v, and, but after the last step, the next noisy latents are (1 - sigma') x0 + sigma' n with fresh noise n and sigma' = sigma(next t); the last x0 is the video's latent.
sigma(t) is the training scheduler's table (shift 8: sigma = 8 s / (1 + 7 s) for s = i / 1000, the table's entry nearest to t). The latents go to the decoder as x0 std + mean (tests/wan_vae_gpu.py, V2).
On the host: the noise, the sampler's arithmetic, the patch embedding, the condition embedder (a timestep's modulation tables), the text's projection, the rotary tables and the weight files. On the GPU: the 30 blocks and the output layer of the transformer
(models/wan_dit.model; one session whose weights stay on the device through the three steps, the step's inputs `reload`ed) and the decoder.

`--cond` is image to video (docs/wan-vae-encoder.md section 11): the file is `[48, H, W]` singles, the first frame's latent as `tests/wan_vae_enc_gpu.cho latent` writes it (already `(mean - latents_mean) / latents_std`). As diffusers'
`WanImageToVideoPipeline` does in the `expand_timesteps` branch (read, `pipeline_wan_i2v.py`): the tokens of frame 0 are given the timestep 0 (the session's per-token selector: the second row of every modulation table), and the latents
are mixed `(1 - mask) * cond + mask * latents` (mask 0 on frame 0, 1 elsewhere) before every call of the transformer and after the last. FastVideo's `DmdDenoisingStage` has no such code (it concatenates the image latent on the
channels, which this 48-channel model cannot take): that this mixing is right for the DISTILLED weights is not read from FastVideo and is not measured.

`--base` is the BASE model's sampler (docs/wan-vae-encoder.md section 12): `--weights` names the base model's transformer directory (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`, five float32 shards) and `--wdir` a directory of its own for the f16 files; flow matching with Euler steps
(`--steps`, default 30) over the sigmas of diffusers' `FlowMatchEulerDiscreteScheduler(shift=5)` (checked to 3e-8), and classifier-free guidance `--guidance` (default 5; 1 turns it off, and then `--neg` is not needed): `v = v_uncond + G (v_text - v_uncond)`, the unconditional
text being `--neg`, the empty prompt's umT5 states (`tools/wan_text.py "" NEG.npy`). Two calls of the transformer a step: the text enters the session as the 60 buffers `b*_yg.a`, `b*_zg.a`, reloaded when the context changes. `--noise` is the starting noise, `[48, F, H, W]`
(`.npy`), so that a run can be compared with another implementation's from the same noise.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import wan_ref as wr  # noqa: E402  (reads --weights)

arg = lambda name, default=None: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
work = sys.argv[1]
F = int(arg("--frames", 5))
H, Wd = (int(v) for v in arg("--size", "22x40").split("x"))
SEED = int(arg("--seed", 0))
WDIR = arg("--wdir", os.path.join(work, "w"))
FPS = int(arg("--fps", 16))
STEPS = [1000, 757, 522]
BASE = "--base" in sys.argv
D, NH, HD = wr.D, wr.NH, wr.HD
T = 512
os.makedirs(work, exist_ok=True)
os.makedirs(WDIR, exist_ok=True)
P = lambda n: os.path.join(work, n)
f16 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f2")).tofile(path)
f32 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f4")).tofile(path)
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
NB = 30
LH, LW = H, Wd  # the latent's size in pixels of the latent grid: [48, F, H, W]
S = F * (H // 2) * (Wd // 2)
SP = (S + 63) // 64 * 64
NEG = -1e9
vec = lambda a: np.asarray(a)[None]
COND = arg("--cond")
NP = (H // 2) * (Wd // 2)  # the tokens of one latent frame
cond = None
if COND:
    cond = np.fromfile(COND, dtype="<f4").astype(np.float64)
    assert cond.size == 48 * H * Wd, "the condition has %d values, the latent's frame needs %d" % (cond.size, 48 * H * Wd)
    cond = cond.reshape(48, 1, H, Wd)
    assert F >= 2


def mix(a):
    """(1 - mask) * cond + mask * a: frame 0 is the condition."""
    if cond is None:
        return a
    out = a.copy()
    out[:, :1] = cond
    return out


def sigma_table():
    """The training scheduler's table (FastVideo's FlowMatchEulerDiscreteScheduler, shift 8): `timesteps` and `sigmas`, 1000 entries."""
    ts = np.linspace(1, 1000, 1000, dtype=np.float32)[::-1].astype(np.float64)
    s = ts / 1000.0
    sig = 8.0 * s / (1 + 7.0 * s)
    return sig * 1000.0, sig


TS, SIG = sigma_table()
sigma_of = lambda t: float(SIG[int(np.argmin(np.abs(TS - t)))])

# ---- the weights, once: f16 files, q and k of the self-attention with the channels of each head permuted even-first (tests/wan_gpu.py)
if not os.path.exists(os.path.join(WDIR, "proj_out.f16")):
    idx = np.empty(D, dtype=int)
    for h in range(NH):
        for i in range(HD // 2):
            idx[h * HD + i], idx[h * HD + HD // 2 + i] = h * HD + 2 * i, h * HD + 2 * i + 1
    Wp = lambda n: os.path.join(WDIR, n)
    t0 = time.time()
    for b in range(NB):
        p = "blocks.%d." % b
        for nm in "qk":
            f16(Wp("w%s%d.f16" % (nm, b)), wr.W(p + "attn1.to_%s.weight" % nm)[idx])
            f32(Wp("b%s%d.f32" % (nm, b)), vec(wr.W(p + "attn1.to_%s.bias" % nm)[idx]))
            f32(Wp("n%s%d.f32" % (nm, b)), vec(wr.W(p + "attn1.norm_%s.weight" % nm)[idx]))
        f16(Wp("wv%d.f16" % b), wr.W(p + "attn1.to_v.weight"))
        f32(Wp("bv%d.f32" % b), vec(wr.W(p + "attn1.to_v.bias")))
        f16(Wp("wo%d.f16" % b), wr.W(p + "attn1.to_out.0.weight"))
        f32(Wp("bo%d.f32" % b), vec(wr.W(p + "attn1.to_out.0.bias")))
        f32(Wp("n2w%d.f32" % b), vec(wr.W(p + "norm2.weight")))
        f32(Wp("n2b%d.f32" % b), vec(wr.W(p + "norm2.bias")))
        for nm in "qkv":
            f16(Wp("w2%s%d.f16" % (nm, b)), wr.W(p + "attn2.to_%s.weight" % nm))
            f32(Wp("b2%s%d.f32" % (nm, b)), vec(wr.W(p + "attn2.to_%s.bias" % nm)))
        f32(Wp("n2q%d.f32" % b), vec(wr.W(p + "attn2.norm_q.weight")))
        f32(Wp("n2k%d.f32" % b), vec(wr.W(p + "attn2.norm_k.weight")))
        f16(Wp("w2o%d.f16" % b), wr.W(p + "attn2.to_out.0.weight"))
        f32(Wp("b2o%d.f32" % b), vec(wr.W(p + "attn2.to_out.0.bias")))
        f16(Wp("wf1%d.f16" % b), wr.W(p + "ffn.net.0.proj.weight"))
        f32(Wp("bf1%d.f32" % b), vec(wr.W(p + "ffn.net.0.proj.bias")))
        f16(Wp("wf2%d.f16" % b), wr.W(p + "ffn.net.2.weight"))
        f32(Wp("bf2%d.f32" % b), vec(wr.W(p + "ffn.net.2.bias")))
    f16(Wp("proj_out.f16"), wr.W("proj_out.weight"))
    f32(Wp("proj_out_b.f32"), vec(wr.W("proj_out.bias")))
    print("weights written to %s in %.0f s" % (WDIR, time.time() - t0), flush=True)

# ---- the constant inputs
text = np.load(arg("--text"))
assert text.shape[0] <= T and text.shape[1] == wr.CFG["text_dim"]
txt = np.zeros((T, text.shape[1]))
txt[: text.shape[0]] = text  # zero rows after the prompt's tokens, to 512, as the pipelines do
ctx = wr.text_embed(txt)
f16(P("ctx.f16"), ctx)
cos, sin = wr.rope_tables(F, H // 2, Wd // 2)
pad = lambda a, fill: np.concatenate([a, np.full((SP - S,) + a.shape[1:], fill)], 0)
f32(P("cs.f32"), pad(cos, 1.0))
f32(P("sn.f32"), pad(sin, 0.0))
sel = np.ones((SP, 1))
if cond is not None:
    sel[:NP] = 0.0  # the conditioning frame's tokens take the t = 0 row
f32(P("sel.f32"), sel)
f32(P("ones_s.f32"), np.ones((SP, 1)))
f32(P("ones_t.f32"), np.ones((T, 1)))
ms = np.zeros((SP, SP))
ms[:, S:] = NEG
f32(P("mask_self.f32"), ms)
f32(P("mask_cross.f32"), np.zeros((SP, T)))
f32(P("ci_s.f32"), np.ones((SP, HD // 2)))
f32(P("si_s.f32"), np.zeros((SP, HD // 2)))
f32(P("ci_t.f32"), np.ones((T, HD // 2)))
f32(P("si_t.f32"), np.zeros((T, HD // 2)))
tab_head = wr.W("scale_shift_table")[0]


def write_step_inputs(lat, t):
    """The inputs that change with the step: the patch embedding of the latents and the modulation tables of the timestep (row 0: the sampled timestep's; row 1: t = 0's, which the conditioning frame's tokens take; with no condition the selector never picks row 1)."""
    f32(P("x0.f32"), pad(wr.patchify(lat), 0.0))
    temb, proj = wr.time_embed(np.array([float(t), 0.0]))
    for b in range(NB):
        tab = wr.W("blocks.%d.scale_shift_table" % b)[0]
        row, row0 = tab + proj[0], tab + proj[1]
        for k, nm in enumerate(("sh1", "sc1", "g1", "sh2", "sc2", "g2")):
            f32(P("m%d_%s.f32" % (b, nm)), np.stack([row[k], row0[k]]))
    f32(P("head_sh.f32"), np.stack([tab_head[0] + temb[0], tab_head[0] + temb[1]]))
    f32(P("head_sc.f32"), np.stack([tab_head[1] + temb[0], tab_head[1] + temb[1]]))


RELOAD = ["b0_lnm1.x", "hd_ln.sh", "hd_ln.sc"] + ["b%d_%s" % (b, n) for b in range(NB) for n in ("lnm1.sh", "lnm1.sc", "gr1.g", "lnm2.sh", "lnm2.sc", "gr2.g")]
rng = np.random.default_rng(SEED)
lat = mix(np.load(arg("--noise")).astype(np.float64) if arg("--noise") else rng.standard_normal((48, F, H, Wd)))
write_step_inputs(lat, STEPS[0])
defs = ["CHX=" + os.path.join(ROOT, "tests/chx"), "WORK=" + work, "WDIR=" + WDIR, "S=%d" % SP, "T=%d" % T, "NB=%d" % NB, "HEAD=1"]
print("latent [48, %d, %d, %d]: %d tokens (padded to %d), text %d tokens (padded to %d); video %d frames of %d x %d" % (F, H, Wd, S, SP, text.shape[0], T, 1 + 4 * (F - 1), 16 * H, 16 * Wd), flush=True)
t_start = time.time()
proc = subprocess.Popen([SESSION, os.path.join(ROOT, "models/wan_dit.model")] + defs, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(P("session.err"), "w"), text=True, bufsize=1)


def ask(line):
    proc.stdin.write(line + "\n")
    proc.stdin.flush()
    out = proc.stdout.readline()
    if not out:
        raise SystemExit("FAIL the session ended: see " + P("session.err"))
    return out


base_x0 = None
if BASE:
    NSTEP, GS, SHIFT = int(arg("--steps", 30)), float(arg("--guidance", 5.0)), 5.0
    shf = lambda x: SHIFT * x / (1 + (SHIFT - 1) * x)
    sig_all = np.append(shf(np.linspace(1.0, shf(1.0 / 1000), NSTEP)), 0.0)  # diffusers' FlowMatchEulerDiscreteScheduler(shift=5).set_timesteps(NSTEP).sigmas: the shift is applied twice, as there
    ctx_text = ctx
    ctx_neg = None
    if GS != 1.0:
        assert arg("--neg"), "--guidance other than 1 needs --neg: the empty prompt's text (tools/wan_text.py \"\" NEG.npy)"
        nt = np.load(arg("--neg"))
        ntx = np.zeros((T, nt.shape[1]))
        ntx[: nt.shape[0]] = nt
        ctx_neg = wr.text_embed(ntx)
    CTXN = ["b%d_%s.a" % (b, n) for b in range(NB) for n in ("yg", "zg")]
    state = {"first": True, "ctx": "text"}

    def velocity(latn, t, which):
        """The transformer's velocity for `latn` at timestep t with the text `which` ("text" or "neg")."""
        write_step_inputs(latn, t)
        names = list(RELOAD) if not state["first"] else []
        if which != state["ctx"]:
            f16(P("ctx.f16"), ctx_text if which == "text" else ctx_neg)
            names += CTXN
            state["ctx"] = which
        if names:
            ask("reload " + " ".join(names))
        state["first"] = False
        ask("1")
        ask("checkpoint")
        y = np.fromfile(P("head_out.f32"), dtype="<f4").reshape(SP, 192)[:S].astype(np.float64)
        return y.reshape(F, H // 2, Wd // 2, 1, 2, 2, 48).transpose(6, 0, 3, 1, 4, 2, 5).reshape(48, F, H, Wd)

    for i in range(NSTEP):
        t0 = time.time()
        tt = sig_all[i] * 1000.0
        vc = velocity(lat, tt, "text")
        v = vc
        if GS != 1.0:
            vu = velocity(lat, tt, "neg")
            v = vu + GS * (vc - vu)
        lat = mix(lat + (sig_all[i + 1] - sig_all[i]) * v)
        print("step %d of %d: sigma %.4f -> %.4f, velocity rms %.4g, latent rms %.4g, %.1f s%s" % (i + 1, NSTEP, sig_all[i], sig_all[i + 1], np.sqrt((v ** 2).mean()), np.sqrt((lat ** 2).mean()), time.time() - t0, " (includes the upload of the weights)" if i == 0 else ""), flush=True)
    base_x0 = lat
x0 = base_x0
for i, t in enumerate([] if BASE else STEPS):
    t0 = time.time()
    if i > 0:
        write_step_inputs(lat, t)
        ask("reload " + " ".join(RELOAD))
    ask("1")
    ask("checkpoint")
    y = np.fromfile(P("head_out.f32"), dtype="<f4").reshape(SP, 192)[:S].astype(np.float64)
    v = y.reshape(F, H // 2, Wd // 2, 1, 2, 2, 48).transpose(6, 0, 3, 1, 4, 2, 5).reshape(48, F, H, Wd)
    sg = sigma_of(t)
    x0 = mix(lat - sg * v)
    print("step %d: t = %d, sigma = %.4f, velocity rms %.4g, x0 rms %.4g, %.1f s%s" % (i, t, sg, np.sqrt((v ** 2).mean()), np.sqrt((x0 ** 2).mean()), time.time() - t0, " (includes the upload of the weights)" if i == 0 else ""), flush=True)
    if i < len(STEPS) - 1:
        sn = sigma_of(STEPS[i + 1])
        lat = mix((1 - sn) * x0 + sn * rng.standard_normal(x0.shape))
proc.stdin.close()
proc.wait()
print("denoiser: %.1f s in all" % (time.time() - t_start), flush=True)

# ---- the decoder (V2) and the video
import glob  # noqa: E402

vcfg = json.load(open(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/*/vae/config.json"))[0]))
mean = np.array(vcfg["latents_mean"])[:, None, None, None]
std = np.array(vcfg["latents_std"])[:, None, None, None]
np.save(P("z.npy"), x0 * std + mean)
np.save(P("x0_latent.npy"), x0)
PY = sys.executable
t0 = time.time()
r = subprocess.run([PY, os.path.join(ROOT, "tests/wan_vae_gpu.py"), P("vae"), "--z", P("z.npy"), "--no-ref", "--save", P("video.npy")], capture_output=True, text=True, timeout=7200)
print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "FAIL decoder: " + r.stderr[-400:], flush=True)
if r.returncode != 0 or not os.path.exists(P("video.npy")):
    sys.exit(1)
vid = np.load(P("video.npy"))  # [3, T, H, W] in [-1, 1]
print("decoder: %.1f s; video %s" % (time.time() - t0, list(vid.shape)), flush=True)
from PIL import Image  # noqa: E402

if cond is not None:
    rm = lambda a: float(np.sqrt((a ** 2).mean()))
    print("condition: latent frame 0 against the condition %.3g of its rms (0 when the mixing holds); video frame k against frame 0, relative rms: %s" % (rm(x0[:, :1] - cond) / rm(cond), ", ".join("k=%d %.3f" % (k, rm(vid[:, k] - vid[:, 0]) / rm(vid[:, 0])) for k in (1, 4, 8, 16) if k < vid.shape[1])), flush=True)
os.makedirs(P("frames"), exist_ok=True)
frames = []
for k in range(vid.shape[1]):
    im = Image.fromarray(((vid[:, k] + 1) * 127.5).clip(0, 255).astype(np.uint8).transpose(1, 2, 0))
    im.save(P("frames/f%03d.png" % k))
    frames.append(im)
frames[0].save(P("video.gif"), save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0)
print("wrote %d frames to %s and %s" % (len(frames), P("frames"), P("video.gif")))
