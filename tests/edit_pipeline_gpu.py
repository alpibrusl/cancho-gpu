#!/usr/bin/env python3
"""D9g of docs/diffusion.md: an image edit on a Metal GPU from a photograph and a prompt to the pixels, in the stages of D9: the vision tower (D9b), the language model reading the image (D9c), the VAE encoder (D9d), the transformer's prefix
pass and eight cached steps with the Euler update (D9e), the VAE decoder (D4c).

    edit_pipeline_gpu.py IMAGE WORK_DIR --prompt TEXT [--seed N] [--steps N] --weights DIR --text-weights DIR [--out PNG]

IMAGE is a photograph (scripts/fetch-test-photo.sh); WORK_DIR keeps every stage's files; `--weights` is the directory of f16 block weights of tests/dit_big.py (head-permuted) and `--text-weights` the directory of the text encoder's
(tests/text_img_gpu.py writes it on first use, 15.7 GB). Needs a Mac, the checkpoint, the venv of the first image edits, and `build/cancho-gpu-metal-session`. Not in `scripts/gate.sh`. The result is written to WORK_DIR/edited.png
and never committed.

What the GPU does: all five stages above. What the host still does (as in D6): the processor's tokenisation and the image's resize (the pipeline's own functions), the embedding lookup, the noise (torch's generator, seed N), the scheduler's
timesteps (the pipeline's own), the transformer's front end (txt_in, img_in, the timestep embedding, the modulation) and back end (norm_out, proj_out) in float64, the Euler update, the layout of the latents for the decoder, the
final clamp and the 8-bit PNG. There is no float64 reference of the whole edit (the transformer's alone would take hours); tools/edit_torch.py runs the real pipeline in BF16 on MPS from the same photograph, prompt and noise.
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
arg = lambda name, default=None: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
image, work = sys.argv[1], sys.argv[2]
PROMPT = arg("--prompt")
SEED = int(arg("--seed", 0))
NSTEPS = int(arg("--steps", 8))
WDIR, TWDIR = arg("--weights"), arg("--text-weights")
assert PROMPT and WDIR and TWDIR, "--prompt, --weights and --text-weights are required"
os.makedirs(work, exist_ok=True)
P = lambda n: os.path.join(work, n)
PY = sys.executable
timings = {}


def run(args, log, label):
    t0 = time.time()
    r = subprocess.run([PY] + args, capture_output=True, text=True, cwd=ROOT)
    open(P(log), "w").write(r.stdout + r.stderr)
    timings[label] = time.time() - t0
    print("%-34s %s in %.0f s" % (label, "ok" if r.returncode == 0 else "FAILED (%d)" % r.returncode, timings[label]), flush=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        sys.exit(1)


# ---- 0. the photograph, resized as the pipeline resizes it
from PIL import Image  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions, calculate_shift, retrieve_timesteps  # noqa: E402

t0 = time.time()
pil = Image.open(image)
if pil.mode != "RGBA":
    pil = pil.convert("RGBA")
ip = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
W0, H0 = calculate_dimensions(1024 * 1024, pil.size[0] / pil.size[1])[:2]
W0, H0 = W0 // 32 * 32, H0 // 32 * 32
resized = ip.resize(pil, width=W0, height=H0)
rgba = ip.preprocess(pil, width=W0, height=H0)[0].numpy().astype(np.float64)  # [4, H, W] in [-1, 1]
pix = np.asarray(resized.convert("RGB"))
np.save(P("pix.npy"), pix)
np.save(P("rgba.npy"), rgba)
Image.fromarray(pix).save(P("input_resized.png"))
GH, GW = H0 // 16, W0 // 16
print("photograph %s -> %d x %d (a %d x %d latent grid, %d tokens)" % (pil.size, W0, H0, GH, GW, GH * GW), flush=True)
timings["host prep"] = time.time() - t0

# ---- 1. the vision tower
run(["tests/vit_gpu.py", P("vit_work"), "--image", P("pix.npy"), "--no-ref", "--save-out", P("vis")], "1_vision.log", "vision tower (GPU)")
merged, deep = np.load(P("vis/merged.npy")).astype(np.float64), np.load(P("vis/deep.npy")).astype(np.float64)

# ---- 2. the language model reading the image
import text_img_ref as ti  # noqa: E402
import text_ref as tr  # noqa: E402

t0 = time.time()
ids, grid, _ = ti.build(PROMPT, pix.astype(np.float64))
assert grid[1:] == [GH, GW], "the processor resized the image: %s" % grid
types = (ids == ti.IMG_PAD).astype(int)
assert types.sum() == merged.shape[0], (types.sum(), merged.shape)
pos = ti.positions(types, grid[1:])
emb = tr._f.get_slice("model.language_model.embed_tokens.weight")
x_in = np.stack([emb[int(t):int(t) + 1][0].double().numpy() for t in ids])
pad_rows = np.nonzero(types)[0]
x_in[pad_rows] = merged
os.makedirs(P("tin"), exist_ok=True)
np.save(P("tin/ids.npy"), ids)
np.save(P("tin/positions.npy"), pos)
np.save(P("tin/embeds_in.npy"), x_in.astype(np.float32))
np.save(P("tin/deep_f64.npy"), deep)
timings["host: tokenise, embed, positions"] = time.time() - t0
run(["tests/text_img_gpu.py", P("tin"), TWDIR, "--no-ref", "--save", P("prompt_rows.npy")], "2_text.log", "language model, with the image (GPU)")
prompt = np.load(P("prompt_rows.npy")).astype(np.float64)  # [rows from drop_idx, 4096]
pad = types[14:].astype(bool)
assert len(prompt) == len(pad)

# ---- 3. the VAE encoder
run(["tests/vae_enc_gpu.py", P("venc_work"), "--image", P("rgba.npy"), "--no-ref", "--save-out", P("mean.npy")], "3_vae_enc.log", "VAE encoder (GPU)")
import vae_enc_ref as ve  # noqa: E402

mean = np.load(P("mean.npy")).astype(np.float64)  # [64, GH, GW]
cond = ve.normalise(mean).reshape(64, -1).T

# ---- 4. the transformer: a prefix pass and the steps
import dit_edit_ref as der  # noqa: E402
import dit_step_ref as step  # noqa: E402
from dit_edit_lib import Edit, statics  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402
import torch  # noqa: E402

der.SHAPES = [(GH, GW), (GH, GW)]
gen = torch.Generator().manual_seed(SEED)
noise = torch.randn((1, 1, 64, GH, GW), generator=gen, dtype=torch.float32)
latents = noise.view(1, 64, GH * GW).transpose(1, 2)[0].numpy().astype(np.float32)  # [tokens, 64], the pipeline's packing
np.save(P("noise_tokens.npy"), latents)
sched = FlowMatchEulerDiscreteScheduler.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", subfolder="scheduler")
cfg_sig = json.load(open(os.path.join(__import__("huggingface_hub").snapshot_download("Qwen/Qwen-Image-2.1-Turbo", allow_patterns=["model_index.json"]), "model_index.json")))["sample_sigmas"]
mu = calculate_shift(latents.shape[0], sched.config.get("base_image_seq_len", 256), sched.config.get("max_image_seq_len", 4096), sched.config.get("base_shift", 0.5), sched.config.get("max_shift", 1.15))
timesteps, nsteps = retrieve_timesteps(sched, NSTEPS, "cpu", sigmas=cfg_sig[:NSTEPS] if NSTEPS <= len(cfg_sig) else cfg_sig, mu=mu)
sigmas = sched.sigmas.numpy().astype(np.float64)
print("steps %d, mu %.4f, timesteps %s, sigmas %s" % (nsteps, mu, np.round(timesteps.numpy(), 1).tolist(), np.round(sigmas, 4).tolist()), flush=True)

t0 = time.time()
x, mod, temb, rope, sel, allowed, pad_exp = der.assemble(prompt, pad, cond, latents.astype(np.float64), float(timesteps[0]) / 1000.0)
pre, tgt = np.nonzero(~sel)[0], np.nonzero(sel)[0]
NPR, NTG = len(pre), len(tgt)
NP = max((NPR + 63) // 64 * 64, (NTG + 63) // 64 * 64)
print("joint sequence %d: prefix %d rows, target %d rows, padded to %d" % (len(x), NPR, NTG, NP), flush=True)
e = Edit(P("edit_work"), WDIR, NP, NPR, NTG, int(arg("--gp", 4)), int(arg("--gs", 2)))
e.static(*statics(rope, allowed, pre, tgt, NP))
timings["host: front end, step 0"] = time.time() - t0
t_prefix = t_steps = 0.0
for i in range(nsteps):
    ts = time.time()
    if i > 0:
        x, mod, temb, _, _, _, _ = der.assemble(prompt, pad, cond, latents.astype(np.float64), float(timesteps[i]) / 1000.0)
    e.modulation(mod)
    if i == 0:
        xp = np.zeros((NP, 4096))
        xp[:NPR] = x[pre]
        e.prefix(xp)
        t_prefix = e.secs["prefix"]
    xt = np.zeros((NP, 4096))
    xt[:NTG] = x[tgt]
    full = e.step(xt)
    t_steps += e.secs["step"]
    pred = step.back(full[:NTG], temb, np.ones(NTG, dtype=bool))  # [tokens, 64]
    latents = (latents.astype(np.float64) + (sigmas[i + 1] - sigmas[i]) * pred).astype(np.float32)
    print("step %d: t %.1f, %.0f s wall (sessions %.0f s); latents rms %.4f" % (i, float(timesteps[i]), time.time() - ts, e.secs["step"], float(np.sqrt((latents ** 2).mean()))), flush=True)
np.save(P("latents_final.npy"), latents)
timings["transformer: prefix + steps (sessions, first requests)"] = t_prefix + t_steps

# ---- 5. the VAE decoder
cfgv = json.load(open(os.path.join(__import__("huggingface_hub").snapshot_download("Qwen/Qwen-Image-2.1-Turbo", allow_patterns=["vae/config.json"]), "vae/config.json")))
mu_v, sd_v = np.array(cfgv["latents_mean"]), np.array(cfgv["latents_std"])
z = latents.astype(np.float64).T.reshape(64, GH, GW) * sd_v[:, None, None] + mu_v[:, None, None]
np.save(P("z.npy"), z)
gold = arg("--gold") or P("none")
run(["tests/vae_gpu.py", arg("--gold", os.path.join(ROOT, "tests")), "--keep", P("vae_work"), "--z", P("z.npy"), "--save-rgb", P("rgb.npy"), "--no-ref"], "5_vae_dec.log", "VAE decoder (GPU)")
rgb = np.load(P("rgb.npy"))
out = arg("--out", P("edited.png"))
Image.fromarray(rgb).save(out)
Image.fromarray(np.concatenate([pix, rgb], axis=1)).save(P("before_after.png"))
print("\nwrote %s (%d x %d) and %s" % (out, rgb.shape[1], rgb.shape[0], P("before_after.png")))
print("timings (s): " + ", ".join("%s %.0f" % (k, v) for k, v in timings.items()))
