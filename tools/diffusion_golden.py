"""D0 of docs/diffusion.md: run the reference (diffusers' QwenImage21Pipeline on MPS) once on a fixed small
input and write what each later stage is held to, as .npy files plus a manifest.

    python tools/diffusion_golden.py OUTDIR [--size 256] [--steps 8] [--seed 1] [--prompt TEXT]

No input photo: a prompt and a seed. Files written (float32 copies of the bf16 tensors, row-major):
  text_embeds        the transformer's `encoder_hidden_states` at step 0 (the text encoder's output)
  text_mask          its mask
  step0_in/step0_out the transformer's `hidden_states` in and its output at step 0
  block0_in/out      transformer_blocks[0]'s hidden_states in and out at step 0
  latents_N          the latents after step N
  vae_in/vae_out     the VAE decode's input (after the mean/std) and output
  image              the final image, uint8 HxWx3
The manifest (manifest.json) holds each file's shape, dtype and sha256; with --check it runs the reference a
second time and says whether the two runs agree bit for bit (MPS bf16 need not: the answer is recorded, not assumed).
use_kv_cache=False: the cache changes the rounding (the pipeline's docstring), so the goldens fix the plain path.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch
from diffusers import QwenImage21Pipeline


def run(args, outdir):
    os.makedirs(outdir, exist_ok=True)
    pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", dtype=torch.bfloat16).to("mps")
    cap = {}

    def put(name, t):
        cap[name] = t.detach().float().cpu().numpy()

    first = {"done": False}

    def tf_pre(mod, a, kw):
        if not first["done"]:
            put("step0_in", kw["hidden_states"])
            put("text_embeds", kw["encoder_hidden_states"])
            m = kw.get("encoder_hidden_states_mask")
            if m is not None:
                put("text_mask", m.float())

    def tf_post(mod, a, kw, out):
        if not first["done"]:
            put("step0_out", out[0])
            first["done"] = True

    b0 = {"done": False}

    def blk_pre(mod, a, kw):
        if not b0["done"]:
            h = kw["hidden_states"] if "hidden_states" in kw else a[0]
            put("block0_in", h)

    def blk_post(mod, a, kw, out):
        if not b0["done"]:
            put("block0_out", out[0] if isinstance(out, tuple) else out)
            b0["done"] = True

    pipe.transformer.register_forward_pre_hook(tf_pre, with_kwargs=True)
    pipe.transformer.register_forward_hook(tf_post, with_kwargs=True)
    pipe.transformer.transformer_blocks[0].register_forward_pre_hook(blk_pre, with_kwargs=True)
    pipe.transformer.transformer_blocks[0].register_forward_hook(blk_post, with_kwargs=True)

    orig_decode = pipe.vae.decode

    def decode(z, *a, **k):
        put("vae_in", z)
        r = orig_decode(z, *a, **k)
        put("vae_out", r[0] if isinstance(r, tuple) else r.sample)
        return r

    pipe.vae.decode = decode

    def cb(p, i, t, kw):
        put("latents_%d" % i, kw["latents"])
        return kw

    t0 = time.time()
    img = pipe(prompt=args.prompt, width=args.size, height=args.size, num_inference_steps=args.steps,
               use_kv_cache=False, generator=torch.Generator("cpu").manual_seed(args.seed),
               callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"]).images[0]
    cap["image"] = np.asarray(img.convert("RGB"), dtype=np.uint8)
    secs = time.time() - t0
    man = {}
    for k, v in sorted(cap.items()):
        np.save(os.path.join(outdir, k + ".npy"), v)
        man[k] = {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": hashlib.sha256(v.tobytes()).hexdigest()}
    del pipe
    torch.mps.empty_cache()
    return cap, man, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prompt", default="A red bicycle leaning against a white wall, soft morning light.")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    cap, man, secs = run(args, args.outdir)
    out = {"size": args.size, "steps": args.steps, "seed": args.seed, "prompt": args.prompt, "seconds": round(secs, 1),
           "torch": torch.__version__, "files": man}
    if args.check:
        cap2, man2, _ = run(args, args.outdir + ".again")
        same = {k: man[k]["sha256"] == man2[k]["sha256"] for k in man}
        diff = {k: float(np.abs(cap[k].astype(np.float32) - cap2[k].astype(np.float32)).max()) for k in man if not same[k]}
        out["two_runs_identical"] = all(same.values())
        out["max_abs_difference_where_not"] = diff
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "files"}, indent=1))
    for k, v in man.items():
        print("%-12s %-20s %s" % (k, v["shape"], v["sha256"][:12]))


main()
