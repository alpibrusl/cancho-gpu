#!/usr/bin/env python3
"""D6 of docs/diffusion.md: a text-to-image run on a Metal GPU from the prompt and the seed to the pixels, in three sessions (the text encoder of D5, the eight steps of D3, the VAE decoder of D4c), against the float64 pipeline and against torch's BF16 one.

    pipeline_gpu.py GOLDEN_DIR WORK_DIR

Needs everything the three scripts need (a Mac, the checkpoint, D0's goldens, the venv of the first edits, 18 + 15 + 1 GB of scratch disk, one after the other: each session's weight files are deleted when it ends). Not in `scripts/gate.sh`.

What this is and is not. The GPU does: the 36-layer text encoder, the 32-block transformer for eight steps with the Euler update, the VAE decoder. The host still does: the tokenisation (D0's token ids), the embedding lookup, the initial noise (torch's seeded random numbers, D0's `step0_in`: cancho-gpu
does not reproduce torch's generator), the rotary and timestep tables, laying the final latents out for the decoder (undoing their mean and std), the weights as f16 files, the final clamp and the 8-bit PNG. The text embeddings, the final latents and the image pass between the sessions through files
on the host. The float64 pipeline is the same three stages with the exact encoder's embeddings: the reference that says which image the prompt gives (torch's BF16 one starts from embeddings that are 2.1% off, D5).
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gold, work = sys.argv[1], sys.argv[2]
os.makedirs(work, exist_ok=True)
PY = sys.executable
P = lambda n: os.path.join(work, n)


def run(args, log):
    t0 = time.time()
    r = subprocess.run([PY] + args, capture_output=True, text=True, cwd=ROOT)
    open(P(log), "w").write(r.stdout + r.stderr)
    print("%-28s %s in %.0f s" % (log, "ok" if r.returncode == 0 else "FAILED (%d)" % r.returncode, time.time() - t0), flush=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        sys.exit(1)
    return r.stdout


def clean(d):
    subprocess.run(["rm", "-rf", d])


# 1. the float64 pipeline's text embeddings (the exact encoder) and the GPU's
run(["tools/text_ref.py", gold, P("text_ref")], "1_text_ref.log")
exact_text = np.load(P("text_ref/layers_f64.npy"))[35][14:]
np.save(P("text_f64.npy"), exact_text)
out = run(["tests/text_gpu.py", gold, P("text_ref"), "--keep", P("text_work"), "--save", P("text_gpu.npy")], "2_text_gpu.log")
clean(P("text_work"))
# 2. the float64 trajectory from the exact embeddings
run(["tools/dit_traj_ref.py", gold, P("traj_f64"), "--text", P("text_f64.npy")], "3_traj_f64.log")
# 3. the GPU's eight steps from the GPU's embeddings
out = run(["tests/dit_loop_gpu.py", gold, P("traj_f64"), "--keep", P("dit_work"), "--text", P("text_gpu.npy"), "--save-latents", P("lat_gpu.npy")], "4_dit_gpu.log")
clean(P("dit_work"))
# 4. decode: the final latents -> z (mean and std undone, as the pipeline does), on the GPU and in float64
snap = os.path.join(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1-Turbo/snapshots"))
cfg = json.load(open(os.path.join(snap, os.listdir(snap)[0], "vae/config.json")))
mean, std = np.array(cfg["latents_mean"]), np.array(cfg["latents_std"])


def to_z(lat):  # [256, 64] tokens x channels -> [64, 16, 16], times std plus mean
    return (lat.reshape(16, 16, 64).transpose(2, 0, 1)) * std[:, None, None] + mean[:, None, None]


lat_gpu = np.load(P("lat_gpu.npy")).astype(np.float64)
lat_f64 = np.load(P("traj_f64/latents_f64.npy"))[-1]
lat_torch = np.load(os.path.join(gold, "latents_7.npy")).astype(np.float64)[0]
chk = np.abs(to_z(lat_torch) - np.load(os.path.join(gold, "vae_in.npy")).astype(np.float64)[0, :, 0]).max()
print("check of the layout: torch's final latents through to_z differ from D0's vae_in by %.3g (max abs)" % chk)
np.save(P("z_gpu.npy"), to_z(lat_gpu))
np.save(P("z_f64.npy"), to_z(lat_f64))
run(["tests/vae_gpu.py", gold, "--keep", P("vae_work"), "--z", P("z_gpu.npy"), "--save-rgb", P("rgb_gpu.npy")], "5_vae_gpu.log")
clean(P("vae_work"))
run(["tests/vae_gpu.py", gold, "--keep", P("vae_work"), "--z", P("z_f64.npy"), "--save-rgb", P("rgb_f64_decoded_on_gpu.npy")], "6_vae_f64_latents.log")
clean(P("vae_work"))
# 5. the images
sys.path.insert(0, os.path.join(ROOT, "tools"))
import vae_ref as vr  # noqa: E402

img_f64 = np.clip(vr.decode(to_z(lat_f64)), -1, 1)
rgb = lambda a: ((a[:3] + 1) * 127.5).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
rgb_f64 = rgb(img_f64)
rgb_gpu = np.load(P("rgb_gpu.npy"))
rgb_torch = np.load(os.path.join(gold, "image.npy"))


def psnr(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


print()
print("final latents [256, 64] (rms %.3f): mean abs error" % np.sqrt((lat_f64 ** 2).mean()))
print("   GPU pipeline vs float64 pipeline:  %.4g (max %.4g)" % (np.abs(lat_gpu - lat_f64).mean(), np.abs(lat_gpu - lat_f64).max()))
print("   torch BF16   vs float64 pipeline:  %.4g (max %.4g)" % (np.abs(lat_torch - lat_f64).mean(), np.abs(lat_torch - lat_f64).max()))
print("   GPU pipeline vs torch BF16:        %.4g (max %.4g)" % (np.abs(lat_gpu - lat_torch).mean(), np.abs(lat_gpu - lat_torch).max()))
print("the 256 x 256 image, 8-bit RGB: mean abs difference (of 255) and PSNR")
for label, a, b in (("GPU pipeline vs float64 pipeline", rgb_gpu, rgb_f64), ("torch BF16 vs float64 pipeline", rgb_torch, rgb_f64), ("GPU pipeline vs torch BF16", rgb_gpu, rgb_torch)):
    print("   %-34s %.2f   %.1f dB" % (label, np.abs(a.astype(float) - b.astype(float)).mean(), psnr(a, b)))
from PIL import Image  # noqa: E402

Image.fromarray(rgb_gpu).save(P("image_gpu.png"))
Image.fromarray(rgb_f64).save(P("image_f64.png"))
Image.fromarray(rgb_torch).save(P("image_torch.png"))
Image.fromarray(np.concatenate([rgb_torch, rgb_f64, rgb_gpu], axis=1)).save(P("compare.png"))
print("images: %s (torch BF16 | float64 pipeline | GPU pipeline)" % P("compare.png"))
