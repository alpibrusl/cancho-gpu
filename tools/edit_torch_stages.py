#!/usr/bin/env python3
"""D11 of docs/diffusion.md: the real Qwen-Image-2.1-Turbo editing pipeline (diffusers, BF16 on MPS) timed stage by stage, the same photograph, prompt and noise as tests/edit_pipeline_gpu.py.

    edit_torch_stages.py IMAGE EDIT_DIR --prompt TEXT [--steps N] [--runs N]

EDIT_DIR is the work directory of tests/edit_pipeline_gpu.py (its `noise_tokens.npy` is the initial latents). The text encoder (vision tower and language model together), the VAE's encode and decode and every call of the transformer are wrapped with
a synchronising timer; the pipeline is run `runs` times in one process (default 2: the first includes the warm-up of MPS's kernels, the second is the warm one) and each run's stages are printed. Nothing is written but the printout.
"""
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from diffusers import QwenImage21Pipeline

arg = lambda name, default=None: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
image, edit_dir = sys.argv[1], sys.argv[2]
prompt, steps, runs = arg("--prompt"), int(arg("--steps", 8)), int(arg("--runs", 2))
noise = torch.tensor(np.load(os.path.join(edit_dir, "noise_tokens.npy")))[None].to(torch.bfloat16)
t0 = time.time()
pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", dtype=torch.bfloat16).to("mps")
print("load: %.1f s" % (time.time() - t0), flush=True)
log = {}


def timed(obj, name, label):
    f = getattr(obj, name)

    def g(*a, **k):
        torch.mps.synchronize()
        t = time.perf_counter()
        r = f(*a, **k)
        torch.mps.synchronize()
        log.setdefault(label, []).append(time.perf_counter() - t)
        return r
    setattr(obj, name, g)


timed(pipe.text_encoder, "forward", "text encoder (vision tower + language model)")
timed(pipe.vae, "encode", "VAE encode")
timed(pipe.vae, "decode", "VAE decode")
timed(pipe.transformer, "forward", "transformer call")
for r in range(runs):
    log.clear()
    torch.mps.synchronize()
    t = time.perf_counter()
    out = pipe(prompt=prompt, image=[Image.open(image)], num_inference_steps=steps, latents=noise.to("mps")).images[0]
    torch.mps.synchronize()
    total = time.perf_counter() - t
    print("run %d (%s): %.1f s total, image %s" % (r + 1, "first, warm-up included" if r == 0 else "warm", total, out.size))
    acc = 0.0
    for k, v in log.items():
        acc += sum(v)
        print("   %-44s %d call(s), %.2f s%s" % (k, len(v), sum(v), "" if len(v) == 1 else " (first %.2f, median %.2f)" % (v[0], sorted(v)[len(v) // 2])))
    print("   %-44s %.2f s (host: processor, scheduler, the rest)" % ("not in a wrapped stage", total - acc), flush=True)
