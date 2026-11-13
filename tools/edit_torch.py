#!/usr/bin/env python3
"""D9g of docs/diffusion.md: the real Qwen-Image-2.1-Turbo pipeline (diffusers, BF16 on MPS) editing the same photograph with the same prompt and the same noise as tests/edit_pipeline_gpu.py, to compare with.

    edit_torch.py IMAGE EDIT_DIR --prompt TEXT [--steps N]

EDIT_DIR is the work directory of tests/edit_pipeline_gpu.py: its `noise_tokens.npy` ([tokens, 64], the initial latents in the pipeline's packing, from torch's CPU generator) is handed to the pipeline as `latents`, so the two
runs start from the same numbers. Writes EDIT_DIR/torch_edited.png and EDIT_DIR/torch_latents_final.npy ([tokens, 64] float32) and prints the wall time of the call (one run, the first of the process: it includes warm-up).
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
prompt, steps = arg("--prompt"), int(arg("--steps", 8))
noise = torch.tensor(np.load(os.path.join(edit_dir, "noise_tokens.npy")))[None].to(torch.bfloat16)
pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", dtype=torch.bfloat16).to("mps")
final = {}


def cb(p, i, t, kw):
    final["latents"] = kw["latents"].float().cpu().numpy()[0]
    return kw


t0 = time.time()
out = pipe(prompt=prompt, image=[Image.open(image)], num_inference_steps=steps, latents=noise.to("mps"), callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"]).images[0]
secs = time.time() - t0
out.convert("RGB").save(os.path.join(edit_dir, "torch_edited.png"))
np.save(os.path.join(edit_dir, "torch_latents_final.npy"), final["latents"])
print("torch BF16 on MPS: %d steps, %.1f s for the call (the first of this process: warm-up included); image %s" % (steps, secs, out.size))
