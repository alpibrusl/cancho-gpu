"""D6b of docs/diffusion.md: how long each stage of Qwen-Image-2.1-Turbo takes in torch (diffusers) on MPS, in BF16, for the D0 prompt: the text encoder, the transformer (eight steps) and the VAE decoder, each timed with the GPU synchronised before and after.

    python tools/diffusion_time.py [--size 256] [--reps 7]

One warm-up call, then `reps` timed calls of the whole pipeline; each stage's time is the sum of its forward calls inside that pipeline call (use_kv_cache=False as D0). The median and the range over the reps are printed. The weights are resident (the model is loaded once).
"""
import argparse
import statistics
import time

import torch
from diffusers import QwenImage21Pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--prompt", default="A red bicycle leaning against a white wall, soft morning light.")
args = ap.parse_args()
pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", dtype=torch.bfloat16).to("mps")
acc = {"text": 0.0, "transformer": 0.0, "vae": 0.0}
steps = []


def timed(name, fn, collect=None):
    def wrapper(*a, **k):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        r = fn(*a, **k)
        torch.mps.synchronize()
        dt = time.perf_counter() - t0
        acc[name] += dt
        if collect is not None:
            collect.append(dt)
        return r
    return wrapper


pipe.text_encoder.forward = timed("text", pipe.text_encoder.forward)
pipe.transformer.forward = timed("transformer", pipe.transformer.forward, steps)
pipe.vae.decode = timed("vae", pipe.vae.decode)


def call():
    for k in acc:
        acc[k] = 0.0
    steps.clear()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    pipe(prompt=args.prompt, width=args.size, height=args.size, num_inference_steps=8, use_kv_cache=False, generator=torch.Generator("cpu").manual_seed(1))
    torch.mps.synchronize()
    return time.perf_counter() - t0, dict(acc), list(steps)


call()  # warm-up
runs = [call() for _ in range(args.reps)]
med = lambda xs: statistics.median(xs)
print("torch %s, MPS, BF16, %d x %d, %d timed runs after one warm-up" % (torch.__version__, args.size, args.size, args.reps))
for key in ("text", "transformer", "vae"):
    xs = [r[1][key] for r in runs]
    print("  %-12s median %.3f s  (min %.3f, max %.3f)" % (key, med(xs), min(xs), max(xs)))
xs = [r[0] for r in runs]
print("  %-12s median %.3f s  (min %.3f, max %.3f)  (includes the scheduler, the host's tokenisation and the image post-processing)" % ("pipeline", med(xs), min(xs), max(xs)))
st = [s for r in runs for s in r[2]]
print("  one transformer call: median %.3f s (%d calls)" % (med(st), len(st)))
