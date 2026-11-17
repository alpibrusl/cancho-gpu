#!/usr/bin/env python3
"""D9g of docs/diffusion.md: an image edit on a Metal GPU from a photograph and a prompt to the pixels, in the stages of D9: the vision tower (D9b), the language model reading the image (D9c), the VAE encoder (D9d), the transformer's prefix
pass and eight cached steps with the Euler update (D9e), the VAE decoder (D4c).

    edit_pipeline_gpu.py IMAGE WORK_DIR --prompt TEXT [--also IMAGE2 [--also IMAGE3 ...]] [--condition-area PIXELS] [--seed N] [--steps N] --weights DIR --text-weights DIR [--out PNG] [--resident] [--unified] [--subprocess]

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
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [q for q in sys.path if os.path.abspath(q or ".") != _here]  # python puts the script's directory first, and tests/kernels.py would answer transformers' optional `import kernels`
sys.path.insert(0, os.path.join(ROOT, "tools"))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("dit_edit_lib", os.path.join(_here, "dit_edit_lib.py"))  # by path: the `kernels` package is not installed, and tests/kernels.py would answer for it
dit_edit_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dit_edit_lib)
Edit, statics = dit_edit_lib.Edit, dit_edit_lib.statics
arg = lambda name, default=None: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
image, work = sys.argv[1], sys.argv[2]
images = [image] + [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--also" and i + 1 < len(sys.argv)]  # one or more pictures (docs/two-images.md); with one, the run is what it always was
COND_AREA = int(arg("--condition-area", 1024 * 1024))  # the area each condition picture is resized to (the pipeline: output_resolution^2)
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
    """A stage script, run inside this process (runpy, with its own argv and its output kept in a log), so that the interpreter and the heavy imports are paid once: as subprocesses the four stages cost
    about 12 s of start-up and imports (D9k). `--subprocess` on the command line runs them as separate processes, as D9g did."""
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
        sys.path.insert(0, os.path.join(ROOT, "tests"))  # what python does for a script: its own directory first
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


# ---- 0. the photographs, resized as the pipeline resizes them
from PIL import Image  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions, calculate_shift, retrieve_timesteps  # noqa: E402

sfx = lambda k: "" if k == 0 else "_%d" % k  # the first picture keeps the files' names of the one-picture run
t0 = time.time()
ip = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
infos = []
for k, path in enumerate(images):
    pil = Image.open(path)
    if pil.mode != "RGBA":
        pil = pil.convert("RGBA")
    Wk, Hk = calculate_dimensions(COND_AREA, pil.size[0] / pil.size[1])[:2]
    Wk, Hk = Wk // 32 * 32, Hk // 32 * 32
    resized = ip.resize(pil, width=Wk, height=Hk)
    rgba_k = ip.preprocess(pil, width=Wk, height=Hk)[0].numpy().astype(np.float64)  # [4, H, W] in [-1, 1]
    pix_k = np.asarray(resized.convert("RGB"))
    np.save(P("pix%s.npy" % sfx(k)), pix_k)
    np.save(P("rgba%s.npy" % sfx(k)), rgba_k)
    Image.fromarray(pix_k).save(P("input_resized%s.png" % sfx(k)))
    infos.append(dict(pix=pix_k, rgba=rgba_k, GH=Hk // 16, GW=Wk // 16, size=pil.size))
    print("photograph %d %s -> %d x %d (a %d x %d latent grid, %d tokens)" % (k + 1, pil.size, Wk, Hk, Hk // 16, Wk // 16, (Hk // 16) * (Wk // 16)), flush=True)
# the target: the pipeline sizes it from the LAST picture's aspect at output_resolution^2 (1024^2)
Wt, Ht = calculate_dimensions(1024 * 1024, infos[-1]["size"][0] / infos[-1]["size"][1])[:2]
Wt, Ht = Wt // 32 * 32, Ht // 32 * 32
GHt, GWt = Ht // 16, Wt // 16
pix = infos[0]["pix"]
print("target %d x %d (%d x %d latent grid, %d tokens)" % (Wt, Ht, GHt, GWt, GHt * GWt), flush=True)
# The attention keeps the scores of 32 heads: the prefix pass NP x NP of them (32 NP^2 elements), which the kernels index with 32 bits. Measured (docs/two-images.md section 7): conditions of 1024 x 1024 gave a picture with a green cast and no error
# when the step padded its target to NP rows (64 NP^2 elements, 4.4e9 at NP 8,256); the step now runs on the target's own rows over two segments of keys (docs/two-images.md section 8), so only the prefix pass is left: NP < 11,585.
# Refuse before any GPU work, with a margin of 128 text rows.
_np_est = max((sum(i["GH"] * i["GW"] for i in infos) + 128 + 63) // 64 * 64, (GHt * GWt + 63) // 64 * 64)
if 32 * _np_est * _np_est >= 2 ** 32:
    print("edit: the joint sequence would have about %d prefix rows: the prefix pass's attention scores (%.2g elements) pass what the kernels can index (2^32). Lower --condition-area (1024 x 1024 = 1048576 is the largest measured for two pictures)" % (_np_est, 32.0 * _np_est * _np_est), file=sys.stderr)
    sys.exit(2)
timings["host prep"] = time.time() - t0

# ---- 1. the vision tower
merged_l, deep_l = [], []
for k in range(len(infos)):  # one run per picture: the tower attends inside each picture only (cu_seqlens), checked in docs/two-images.md
    run(["tests/vit_gpu.py", P("vit_work%s" % sfx(k)), "--image", P("pix%s.npy" % sfx(k)), "--no-ref", "--save-out", P("vis%s" % sfx(k))], "1_vision%s.log" % sfx(k), "vision tower %d (GPU)" % (k + 1))
    merged_l.append(np.load(P("vis%s/merged.npy" % sfx(k))).astype(np.float64))
    deep_l.append(np.load(P("vis%s/deep.npy" % sfx(k))).astype(np.float64))
merged, deep = np.concatenate(merged_l, 0), np.concatenate(deep_l, 1)

# ---- 2. the language model reading the image
import text_img_ref as ti  # noqa: E402
import text_ref as tr  # noqa: E402

t0 = time.time()
many = len(infos) > 1
ids, grid, _ = ti.build(PROMPT, [i["pix"].astype(np.float64) for i in infos] if many else pix.astype(np.float64))
grids = grid if many else [grid]
for g, i in zip(grids, infos):
    assert g[1:] == [i["GH"], i["GW"]], "the processor resized the image: %s" % g
types = (ids == ti.IMG_PAD).astype(int)
assert types.sum() == merged.shape[0], (types.sum(), merged.shape)
pos = ti.positions(types, [g[1:] for g in grids] if many else grids[0][1:])
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
import vae_enc_ref as ve  # noqa: E402

cond_l = []
for k in range(len(infos)):
    run(["tests/vae_enc_gpu.py", P("venc_work%s" % sfx(k)), "--image", P("rgba%s.npy" % sfx(k)), "--no-ref", "--save-out", P("mean%s.npy" % sfx(k))], "3_vae_enc%s.log" % sfx(k), "VAE encoder %d (GPU)" % (k + 1))
    mean = np.load(P("mean%s.npy" % sfx(k))).astype(np.float64)  # [64, GH, GW]
    cond_l.append(ve.normalise(mean).reshape(64, -1).T)
cond = np.concatenate(cond_l, 0)

# ---- 4. the transformer: a prefix pass and the steps
import dit_edit_ref as der  # noqa: E402
import dit_step_ref as step  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402
import torch  # noqa: E402

der.SHAPES = [(i["GH"], i["GW"]) for i in infos] + [(GHt, GWt)]  # the conditions in order, then the target
gen = torch.Generator().manual_seed(SEED)
noise = torch.randn((1, 1, 64, GHt, GWt), generator=gen, dtype=torch.float32)
latents = noise.view(1, 64, GHt * GWt).transpose(1, 2)[0].numpy().astype(np.float32)  # [tokens, 64], the pipeline's packing
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
_scores = 32.0 * NP * NP * (1 if (NTG + 63) // 64 * 64 < NP else 2)  # the prefix pass's NP x NP scores; a step that cannot run on fewer rows than the prefix (NP = the target's) pads to 2 NP keys
if _scores >= 2 ** 32:  # the authoritative check (the one after stage 0 is an estimate)
    print("edit: NP = %d: the attention scores (%.2g elements) pass what the kernels can index (2^32): lower --condition-area" % (NP, _scores), file=sys.stderr)
    sys.exit(2)
print("joint sequence %d: prefix %d rows, target %d rows, padded to %d" % (len(x), NPR, NTG, NP), flush=True)
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
timings["transformer: prefix + steps (sessions, first requests)"] = t_prefix + t_steps

# ---- 5. the VAE decoder
cfgv = json.load(open(os.path.join(__import__("huggingface_hub").snapshot_download("Qwen/Qwen-Image-2.1-Turbo", allow_patterns=["vae/config.json"]), "vae/config.json")))
mu_v, sd_v = np.array(cfgv["latents_mean"]), np.array(cfgv["latents_std"])
z = latents.astype(np.float64).T.reshape(64, GHt, GWt) * sd_v[:, None, None] + mu_v[:, None, None]
np.save(P("z.npy"), z)
gold = arg("--gold") or P("none")
run(["tests/vae_gpu.py", arg("--gold", os.path.join(ROOT, "tests")), "--keep", P("vae_work"), "--z", P("z.npy"), "--save-rgb", P("rgb.npy"), "--no-ref"], "5_vae_dec.log", "VAE decoder (GPU)")
rgb = np.load(P("rgb.npy"))
out = arg("--out", P("edited.png"))
Image.fromarray(rgb).save(out)
if len(infos) == 1:
    Image.fromarray(np.concatenate([pix, rgb], axis=1)).save(P("before_after.png"))
else:  # the pictures and the result, side by side at the result's height
    tiles = [np.asarray(Image.fromarray(i["pix"]).resize((max(1, round(i["pix"].shape[1] * rgb.shape[0] / i["pix"].shape[0])), rgb.shape[0]))) for i in infos] + [rgb]
    Image.fromarray(np.concatenate(tiles, axis=1)).save(P("before_after.png"))
print("\nwrote %s (%d x %d) and %s" % (out, rgb.shape[1], rgb.shape[0], P("before_after.png")))
print("timings (s): " + ", ".join("%s %.0f" % (k, v) for k, v in timings.items()))
