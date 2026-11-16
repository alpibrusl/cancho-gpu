#!/usr/bin/env python3
"""docs/text-to-image.md: a text-to-image run of Qwen-Image-2.1-Turbo on a Metal GPU from a FREE prompt to the pixels: the edit driver (tests/edit_pipeline_gpu.py, D9g) without a picture, so without the vision tower and the VAE encoder.

    t2i_pipeline_gpu.py WORK_DIR --prompt TEXT --weights DIR --text-weights DIR [--size WxH] [--seed N] [--steps N] [--out PNG] [--gp N] [--gs N] [--resident] [--unified] [--subprocess] [--force] [--mutate no-text|reversed]

`--weights` and `--text-weights` are the edit driver's (the head-permuted f16 transformer blocks of tests/dit_big.py; the language model's f16 weights of tests/text_img_gpu.py). `--size` is WxH in pixels, multiples of 32 (default 512x512; 1024x1024 is the largest
that is allowed without `--force`). The result is WORK_DIR/image.png (or `--out`); every stage's files stay in WORK_DIR, and nothing is committed.

What the GPU does: the language model over the text-only template (tests/text_img_gpu.py with no picture rows), the transformer's steps with the Euler update (tests/dit_edit_lib.py `Edit`, a prefix of the text rows only), the VAE decoder. What the host still does: the processor's
tokenisation, the embedding lookup, the noise (torch's generator, seed N), the scheduler's timesteps (the pipeline's own), the transformer's front and back ends in float64, the Euler update, the layout of the latents for the decoder, the clamp and the 8-bit PNG (as D9g).
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [q for q in sys.path if os.path.abspath(q or ".") != _here]  # as the edit driver: tests/kernels.py would answer transformers' optional `import kernels`
sys.path.insert(0, os.path.join(ROOT, "tools"))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("dit_edit_lib", os.path.join(_here, "dit_edit_lib.py"))
dit_edit_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dit_edit_lib)
Edit, statics = dit_edit_lib.Edit, dit_edit_lib.statics
arg = lambda name, default=None: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def refuse(why):
    print("t2i: " + why, file=sys.stderr)
    sys.exit(2)


if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
    refuse("usage: t2i_pipeline_gpu.py WORK_DIR --prompt TEXT --weights DIR --text-weights DIR [--size WxH] [--seed N] [--steps N] [--out PNG]")
work = sys.argv[1]
PROMPT = arg("--prompt")
SEED = int(arg("--seed", 0))
NSTEPS = int(arg("--steps", 8))
WDIR, TWDIR = arg("--weights"), arg("--text-weights")
if PROMPT is None:
    refuse("--prompt is required")
if not (WDIR and TWDIR):
    refuse("--weights and --text-weights are required")
for d in (WDIR, TWDIR):
    if not os.path.isdir(d):
        refuse("not a directory: %s" % d)
try:
    W0, H0 = (int(v) for v in arg("--size", "512x512").lower().split("x"))
except ValueError:
    refuse("--size is WxH in pixels, for example 512x512")
if W0 % 32 or H0 % 32 or W0 < 64 or H0 < 64:
    refuse("--size must be multiples of 32, at least 64: got %dx%d" % (W0, H0))
if W0 * H0 > 1024 * 1024 and "--force" not in sys.argv:
    refuse("--size %dx%d is more than 1024 x 1024 pixels, the largest run measured (--force goes past it)" % (W0, H0))
if not 1 <= NSTEPS <= 50:
    refuse("--steps is 1 to 50")
if arg("--mutate") not in (None, "no-text", "reversed"):
    refuse("--mutate is no-text or reversed")
os.makedirs(work, exist_ok=True)
P = lambda n: os.path.join(work, n)
PY = sys.executable
timings = {}
GH, GW = H0 // 16, W0 // 16


def run(args, log, label):
    """A stage script, run inside this process (as the edit driver)."""
    t0 = time.time()
    if "--subprocess" in sys.argv:
        r = subprocess.run([PY] + args, capture_output=True, text=True, cwd=ROOT)
        out, code = r.stdout + r.stderr, r.returncode
    else:
        import contextlib
        import io
        import runpy
        buf, code = io.StringIO(), 0
        old_argv, old_path = sys.argv, list(sys.path)
        sys.argv = [os.path.join(ROOT, args[0])] + args[1:]
        sys.path.insert(0, os.path.join(ROOT, "tests"))
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                runpy.run_path(os.path.join(ROOT, args[0]), run_name="__main__")
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        finally:
            sys.argv, sys.path[:] = old_argv, old_path
        out = buf.getvalue()
    open(P(log), "w").write(out)
    timings[label] = time.time() - t0
    print("%-34s %s in %.0f s" % (label, "ok" if code == 0 else "FAILED (%d)" % code, timings[label]), flush=True)
    if code != 0:
        print(out[-3000:])
        sys.exit(1)


from PIL import Image  # noqa: E402
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps  # noqa: E402

# ---- 1. the prompt through the processor (the pipeline's text-to-image template: no picture), the language model on the GPU
import text_img_ref as ti  # noqa: E402
import text_ref as tr  # noqa: E402
import vit_ref as vr  # noqa: E402

T2I_TEMPLATE = f"<|im_start|>system\n{ti.SYS}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
t0 = time.time()
from transformers import AutoProcessor  # noqa: E402

proc = AutoProcessor.from_pretrained(vr.CKPT + "processor")
text = PROMPT if PROMPT.strip() else " "  # the pipeline: "Qwen has no bos token, so an empty string leaves the encoder with nothing to read"
ids = proc(text=[T2I_TEMPLATE.format(text)], padding=True, return_tensors="pt")["input_ids"][0].numpy().astype(int)
if len(ids) > 512:
    refuse("the prompt is %d tokens with the template: at most 512 are run" % len(ids))
types = np.zeros(len(ids), dtype=int)
pos = ti.positions(types, (0, 0))  # all text: the same position on the three axes
emb = tr._f.get_slice("model.language_model.embed_tokens.weight")
x_in = np.stack([emb[int(t):int(t) + 1][0].double().numpy() for t in ids])
os.makedirs(P("tin"), exist_ok=True)
np.save(P("tin/ids.npy"), ids)
np.save(P("tin/positions.npy"), pos)
np.save(P("tin/embeds_in.npy"), x_in.astype(np.float32))
np.save(P("tin/deep_f64.npy"), np.zeros((3, 0, 4096)))
timings["host: tokenise, embed"] = time.time() - t0
print("prompt: %d tokens with the template (%d rows kept); image %d x %d (%d x %d latent grid, %d tokens)" % (len(ids), len(ids) - 14, W0, H0, GH, GW, GH * GW), flush=True)
run(["tests/text_img_gpu.py", P("tin"), TWDIR, "--no-ref", "--save", P("prompt_rows.npy")], "1_text.log", "language model (GPU)")
prompt = np.load(P("prompt_rows.npy")).astype(np.float64)  # [rows from drop_idx, 4096]
MUT = arg("--mutate")  # a deliberate host-side bug, which the comparison with torch must refuse
if MUT == "no-text":
    prompt = np.zeros_like(prompt)  # the prompt is not read
elif MUT == "reversed":
    prompt = prompt[::-1].copy()  # the text rows in the wrong order
elif MUT is not None:
    refuse("--mutate is no-text or reversed")
pad = np.zeros(len(prompt), dtype=bool)  # no picture rows

# ---- 2. the transformer: a prefix of the text rows only, then the steps
import dit_edit_ref as der  # noqa: E402
import dit_step_ref as step  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402
import torch  # noqa: E402

der.SHAPES = [(GH, GW)]  # the target only: no condition block
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
cond = np.zeros((0, 64))  # no condition picture
x, mod, temb, rope, sel, allowed, pad_exp = der.assemble(prompt, pad, cond, latents.astype(np.float64), float(timesteps[0]) / 1000.0)
pre, tgt = np.nonzero(~sel)[0], np.nonzero(sel)[0]
NPR, NTG = len(pre), len(tgt)
NP = max((NPR + 63) // 64 * 64, (NTG + 63) // 64 * 64)
print("joint sequence %d: prefix %d rows (text), target %d rows, padded to %d" % (len(x), NPR, NTG, NP), flush=True)
e = Edit(P("edit_work"), WDIR, NP, NPR, NTG, int(arg("--gp", 4)), int(arg("--gs", 2)), resident="--resident" in sys.argv, unified="--unified" in sys.argv)
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
e.close()
np.save(P("latents_final.npy"), latents)
timings["transformer: prefix + steps (sessions)"] = t_prefix + t_steps

# ---- 3. the VAE decoder
cfgv = json.load(open(os.path.join(__import__("huggingface_hub").snapshot_download("Qwen/Qwen-Image-2.1-Turbo", allow_patterns=["vae/config.json"]), "vae/config.json")))
mu_v, sd_v = np.array(cfgv["latents_mean"]), np.array(cfgv["latents_std"])
z = latents.astype(np.float64).T.reshape(64, GH, GW) * sd_v[:, None, None] + mu_v[:, None, None]
np.save(P("z.npy"), z)
run(["tests/vae_gpu.py", os.path.join(ROOT, "tests"), "--keep", P("vae_work"), "--z", P("z.npy"), "--save-rgb", P("rgb.npy"), "--no-ref"], "3_vae_dec.log", "VAE decoder (GPU)")
rgb = np.load(P("rgb.npy"))
out = arg("--out", P("image.png"))
Image.fromarray(rgb).save(out)
print("\nwrote %s (%d x %d)" % (out, rgb.shape[1], rgb.shape[0]))
print("timings (s): " + ", ".join("%s %.0f" % (k, v) for k, v in timings.items()))
