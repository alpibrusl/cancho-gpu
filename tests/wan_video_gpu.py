#!/usr/bin/env python3
"""V4a of docs/diffusion.md section 6: a video from a prompt, the denoiser and the decoder of FastWan2.2-TI2V-5B on a Metal GPU.

    wan_video_gpu.py WORK_DIR --weights FASTWAN_TRANSFORMER_DIR --text TEXT.npy [--frames F] [--size HxW] [--seed S] [--wdir DIR] [--fps N]

F is the number of latent frames (the video has 1 + 4 (F - 1)), HxW the latent's size in pixels / 16 (22x40 is 640 x 352); `--text` is tools/wan_text.py's output (the umT5 encoder runs on the HOST, in torch: the one stage that
does not run on the GPU); `--wdir` holds the f16 weight files of tests/wan_gpu.py's layout (written on first use, 10 GB); the output is WORK_DIR/video.gif and WORK_DIR/frames/.

The sampler is FastVideo's `DmdDenoisingStage` for this model (the distilled one needs no guidance and no negative prompt): the latents start as Gaussian noise; for t in (1000, 757, 522) the transformer gives the velocity v
of the noisy latents x, the clean latents are x0 = x - sigma(t) v, and, but after the last step, the next noisy latents are (1 - sigma') x0 + sigma' n with fresh noise n and sigma' = sigma(next t); the last x0 is the video's latent.
sigma(t) is the training scheduler's table (shift 8: sigma = 8 s / (1 + 7 s) for s = i / 1000, the table's entry nearest to t). The latents go to the decoder as x0 std + mean (tests/wan_vae_gpu.py, V2).
On the host: the noise, the sampler's arithmetic, the patch embedding, the condition embedder (a timestep's modulation tables), the text's projection, the rotary tables and the weight files. On the GPU: the 30 blocks and the output layer of the transformer
(models/wan_dit.model; one session whose weights stay on the device through the three steps, the step's inputs `reload`ed) and the decoder.
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
f32(P("sel.f32"), np.ones((SP, 1)))  # every token at the sampled timestep (no conditioning frame: text to video)
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
    """The inputs that change with the step: the patch embedding of the latents and the modulation tables of the timestep (two identical rows: the sampled one and the unused t = 0 one)."""
    f32(P("x0.f32"), pad(wr.patchify(lat), 0.0))
    temb, proj = wr.time_embed(np.array([float(t)]))
    for b in range(NB):
        tab = wr.W("blocks.%d.scale_shift_table" % b)[0]
        row = tab + proj[0]
        for k, nm in enumerate(("sh1", "sc1", "g1", "sh2", "sc2", "g2")):
            f32(P("m%d_%s.f32" % (b, nm)), np.stack([row[k], row[k]]))
    f32(P("head_sh.f32"), np.stack([tab_head[0] + temb[0]] * 2))
    f32(P("head_sc.f32"), np.stack([tab_head[1] + temb[0]] * 2))


RELOAD = ["b0_lnm1.x", "hd_ln.sh", "hd_ln.sc"] + ["b%d_%s" % (b, n) for b in range(NB) for n in ("lnm1.sh", "lnm1.sc", "gr1.g", "lnm2.sh", "lnm2.sc", "gr2.g")]
rng = np.random.default_rng(SEED)
lat = rng.standard_normal((48, F, H, Wd))
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


x0 = None
for i, t in enumerate(STEPS):
    t0 = time.time()
    if i > 0:
        write_step_inputs(lat, t)
        ask("reload " + " ".join(RELOAD))
    ask("1")
    ask("checkpoint")
    y = np.fromfile(P("head_out.f32"), dtype="<f4").reshape(SP, 192)[:S].astype(np.float64)
    v = y.reshape(F, H // 2, Wd // 2, 1, 2, 2, 48).transpose(6, 0, 3, 1, 4, 2, 5).reshape(48, F, H, Wd)
    sg = sigma_of(t)
    x0 = lat - sg * v
    print("step %d: t = %d, sigma = %.4f, velocity rms %.4g, x0 rms %.4g, %.1f s%s" % (i, t, sg, np.sqrt((v ** 2).mean()), np.sqrt((x0 ** 2).mean()), time.time() - t0, " (includes the upload of the weights)" if i == 0 else ""), flush=True)
    if i < len(STEPS) - 1:
        sn = sigma_of(STEPS[i + 1])
        lat = (1 - sn) * x0 + sn * rng.standard_normal(x0.shape)
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

os.makedirs(P("frames"), exist_ok=True)
frames = []
for k in range(vid.shape[1]):
    im = Image.fromarray(((vid[:, k] + 1) * 127.5).clip(0, 255).astype(np.uint8).transpose(1, 2, 0))
    im.save(P("frames/f%03d.png" % k))
    frames.append(im)
frames[0].save(P("video.gif"), save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0)
print("wrote %d frames to %s and %s" % (len(frames), P("frames"), P("video.gif")))
