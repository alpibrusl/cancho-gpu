#!/usr/bin/env python3
"""cancho-media: one command line for the image and video generators of this repository (docs/cli.md).

    scripts/media video "PROMPT" -o DIR [--frames N] [--size WxH] [--seed S] [--fps N]
    scripts/media edit  IMAGE "INSTRUCTION" -o DIR [--seed S] [--steps N]
    scripts/media image "PROMPT" -o DIR        (not wired: see `media image --help`)
    scripts/media models                       (the supported models per task; `--model NAME` on each task)
    scripts/media doctor                       (what is installed and what is missing)

It wraps the drivers that exist and does no GPU work itself: `video` is tools/wan_text.py (umT5 on the host) then tests/wan_video_gpu.py (FastWan2.2-TI2V-5B: three DMD steps and the VAE decoder, on the
Metal GPU), `edit` is tests/edit_pipeline_gpu.py (Qwen-Image-2.1-Turbo). Run it with the interpreter of the first image edits' venv (scripts/media does). One GPU job at a time: a lock file in the
cache directory refuses a second run. Limits are the ones measured (docs/diffusion.md V1b/V2b/V4a), not the model's: `--force` goes past them at your own risk of ending the session."""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.expanduser(os.environ.get("CANCHO_MEDIA_CACHE", "~/.cache/cancho-media"))
HF = os.path.expanduser("~/.cache/huggingface/hub")
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
MAX_TOKENS = 4000  # the transformer's attention is the three-kernel one here: measured only up to 1,100 tokens (V4a); the scores are 24 heads x S^2 x 4 B a block
MAX_PIXEL_FRAMES = 33 * 1280 * 704  # the decoder: 33 frames at 1280 x 704 in 21.6 s; 49 frames took 3 times as long (V2b)


# The models each task can use. A model is a name, the Hugging Face repositories it needs, and the driver that runs it; a new model is a new entry and its driver (docs/cli.md). Only these two exist.
MODELS = {
    "video": {"fastwan2.2-ti2v-5b": {"repo": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers", "also": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "note": "3 DMD steps, 640x352 to 1280x704, up to 33 frames measured"}},
    "edit": {"qwen-image-2.1-turbo": {"repo": "Qwen/Qwen-Image-2.1-Turbo", "also": None, "note": "8 steps, image + instruction"}},
    "image": {},  # no text-to-image driver for a free prompt yet
}
DEFAULT = {"video": "fastwan2.2-ti2v-5b", "edit": "qwen-image-2.1-turbo"}


def pick(task, name):
    """The registry entry for `name` (or the task's only model), or a refusal that lists what is supported."""
    have = MODELS[task]
    name = (name or DEFAULT.get(task, "")).lower()
    if name not in have:
        die("model %r is not supported for %s; supported: %s" % (name, task, ", ".join(have) or "none"))
    return name, have[name]


def die(msg, code=2):
    print("media: " + msg, file=sys.stderr)
    sys.exit(code)


def snapshot(repo, sub=""):
    g = sorted(glob.glob(os.path.join(HF, "models--" + repo.replace("/", "--"), "snapshots", "*", sub)))
    return g[-1] if g else None


def need_gpu():
    if sys.platform != "darwin":
        die("this runs on a Mac with a Metal GPU (cancho-gpu's session is Metal); not found on " + sys.platform)
    if not os.path.exists(SESSION):
        die("build/cancho-gpu-metal-session is missing: build it with `CANCHO=<pinned cancho> scripts/build-metal-session.sh` (cancho.toml names the compiler)")


class Lock:
    def __enter__(self):
        os.makedirs(CACHE, exist_ok=True)
        self.path = os.path.join(CACHE, "gpu.lock")
        if os.path.exists(self.path):
            try:
                pid = int(open(self.path).read())
                os.kill(pid, 0)
                die("another media job is running (pid %d): one GPU job at a time" % pid)
            except (ValueError, ProcessLookupError, PermissionError):
                pass  # stale lock
        open(self.path, "w").write(str(os.getpid()))
        return self

    def __exit__(self, *a):
        try:
            os.remove(self.path)
        except OSError:
            pass


def run(cmd, label):
    print("== " + label, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        die("%s failed (exit %d)" % (label, r.returncode), 1)
    print("== %s: %.0f s" % (label, time.time() - t0), flush=True)


def cmd_video(a):
    need_gpu()
    model, info = pick("video", a.model)
    w, h = (int(v) for v in a.size.lower().split("x"))
    if w % 32 or h % 32 or w < 64 or h < 64:
        die("--size WxH: both sides must be multiples of 32 (the latent is 16 times smaller and is cut into 2 x 2 patches); %s is not" % a.size)
    lf = (max(a.frames, 1) - 1 + 3) // 4 + 1  # latent frames; the video has 1 + 4 (lf - 1) frames
    frames = 1 + 4 * (lf - 1)
    if frames != a.frames:
        print("media: %d frames requested; the model makes 1 + 4k, so %d" % (a.frames, frames))
    tokens = lf * (h // 32) * (w // 32)
    if tokens > MAX_TOKENS and not a.force:
        die("%d frames at %dx%d are %d tokens: the denoiser's attention here is measured only up to about 1,100 (docs/diffusion.md V4a; the fused switch `ATT=1` is not wired into this driver). "
            "Use fewer frames or a smaller size (17 frames at 640x352 is 1,100 tokens), or --force" % (frames, w, h, tokens))
    if frames * w * h > MAX_PIXEL_FRAMES and not a.force:
        die("%d frames at %dx%d decode as one clip: measured up to 33 frames at 1280x704 (21.6 s); 49 took 66.6 s and more is untried (V2b). Use --force to try" % (frames, w, h))
    tdir = a.weights or snapshot(info["repo"], "transformer")
    if not tdir or not snapshot(info["also"]):
        die("weights not in the Hugging Face cache: `hf download FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` and `hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers` (or pass --weights DIR)")
    out = os.path.abspath(a.output)
    os.makedirs(out, exist_ok=True)
    wdir = os.path.expanduser(a.wdir)
    free = shutil.disk_usage(os.path.dirname(wdir) if os.path.isdir(os.path.dirname(wdir)) else CACHE if os.path.isdir(CACHE) else "/").free
    if not os.path.exists(os.path.join(wdir, "proj_out.f16")) and free < 14e9:
        die("the f16 weight files (10 GB) are written to %s on first use and only %.0f GB are free" % (wdir, free / 1e9))
    with Lock():
        txt = os.path.join(out, "text.npy")
        run([sys.executable, "tools/wan_text.py", a.prompt, txt], "text encoder (host, torch)")
        run([sys.executable, "tests/wan_video_gpu.py", out, "--weights", tdir, "--text", txt, "--frames", str(lf), "--size", "%dx%d" % (h // 16, w // 16),
             "--seed", str(a.seed), "--wdir", wdir, "--fps", str(a.fps)], "denoiser and decoder (GPU)")
    open(os.path.join(out, "prompt.txt"), "w").write(a.prompt + "\n")
    if shutil.which("ffmpeg"):
        mp4 = os.path.join(out, "video.mp4")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(out, "frames/f%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4])
        print("wrote " + mp4 if r.returncode == 0 else "media: ffmpeg failed; the gif and the frames are there")
    print("done: %s (video.gif, frames/, prompt.txt)" % out)


def cmd_edit(a):
    need_gpu()
    model, info = pick("edit", a.model)
    if not os.path.isfile(a.image):
        die("no such image: " + a.image)
    wd = a.weights or os.environ.get("CANCHO_MEDIA_EDIT_WEIGHTS") or os.path.join(CACHE, "qwen-edit-dit")
    tw = a.text_weights or os.environ.get("CANCHO_MEDIA_EDIT_TEXT_WEIGHTS") or os.path.join(CACHE, "qwen-edit-text")
    if not snapshot(info["repo"], "transformer"):
        die("Qwen-Image-2.1-Turbo is not in the Hugging Face cache (the repository's directory is empty): `hf download Qwen/Qwen-Image-2.1-Turbo` (about 50 GB)")
    if not os.path.isdir(wd):
        die("the transformer's f16 block weights (head-permuted, 14 GB, written by tests/dit_big.py: docs/diffusion.md D7) are not at %s: pass --weights DIR or set CANCHO_MEDIA_EDIT_WEIGHTS" % wd)
    out = os.path.abspath(a.output)
    os.makedirs(out, exist_ok=True)
    with Lock():
        run([sys.executable, "tests/edit_pipeline_gpu.py", os.path.abspath(a.image), os.path.join(out, "work"), "--prompt", a.instruction, "--seed", str(a.seed), "--steps", str(a.steps),
             "--weights", wd, "--text-weights", tw, "--out", os.path.join(out, "edited.png"), "--resident"], "image edit (GPU, five stages)")
    open(os.path.join(out, "prompt.txt"), "w").write(a.instruction + "\n")
    print("done: " + os.path.join(out, "edited.png"))


def cmd_models(a):
    for task, have in MODELS.items():
        print("%s:" % task)
        for name, info in have.items():
            print("  %-22s %s%s  [%s]" % (name, info["repo"], " (default)" if DEFAULT.get(task) == name else "", info["note"]))
        if not have:
            print("  (none supported yet)")


def cmd_image(a):
    pick("image", a.model)
    die("text to image from a free prompt is not wired. The generator of docs/diffusion.md D6 (tests/pipeline_gpu.py) runs the prompt of D0's golden (its token ids and noise are files), and the "
        "tokeniser, the noise and the text encoder for a prompt of your own were built only for the edit pipeline. Meanwhile `media edit` takes an image and an instruction", 3)


def cmd_doctor(a):
    ok = True

    def line(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print("%-4s %-34s %s" % ("ok" if good else "MISS", name, detail))
    line("macOS / Metal", sys.platform == "darwin")
    line("build/cancho-gpu-metal-session", os.path.exists(SESSION))
    line("FastWan transformer weights", snapshot("FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers", "transformer"))
    line("Wan2.2 VAE, umT5, tokenizer", snapshot("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "vae"))
    line("Qwen-Image-2.1-Turbo (for edit)", snapshot("Qwen/Qwen-Image-2.1-Turbo", "transformer"))
    line("ffmpeg (mp4 output)", shutil.which("ffmpeg"))
    for m in ("numpy", "torch", "transformers", "safetensors", "PIL"):
        try:
            __import__(m)
            line("python: " + m, True)
        except ImportError:
            line("python: " + m, False, "use the venv of the first image edits (scripts/media does)")
    print("free disk: %.0f GB" % (shutil.disk_usage("/").free / 1e9))
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(prog="media", description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="examples:\n  media video \"A red fox running through snow\" -o out/fox --frames 17 --size 640x352 --seed 1\n"
                                       "  media edit photo.jpg \"Make it a bright sunny afternoon.\" -o out/sunny\n  media doctor")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("video", help="a video from a prompt (FastWan2.2-TI2V-5B, 3 steps)")
    v.add_argument("prompt")
    v.add_argument("-o", "--output", required=True, help="directory for video.gif, video.mp4, frames/")
    v.add_argument("--frames", type=int, default=17, help="video frames; rounded up to 1 + 4k (default 17)")
    v.add_argument("--size", default="640x352", help="WxH pixels, multiples of 32 (default 640x352)")
    v.add_argument("--model", help="model name (`media models`; default %s)" % DEFAULT["video"])
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--fps", type=int, default=16)
    v.add_argument("--weights", help="FastWan transformer directory (default: the Hugging Face cache)")
    v.add_argument("--wdir", default=os.path.join(CACHE, "wan-f16"), help="where the 10 GB of f16 weight files are kept between runs")
    v.add_argument("--force", action="store_true", help="go past the measured limits")
    v.set_defaults(f=cmd_video)
    e = sub.add_parser("edit", help="edit an image from an instruction (Qwen-Image-2.1-Turbo)")
    e.add_argument("image")
    e.add_argument("instruction")
    e.add_argument("-o", "--output", required=True, help="directory for edited.png")
    e.add_argument("--model", help="model name (`media models`; default %s)" % DEFAULT["edit"])
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--steps", type=int, default=8)
    e.add_argument("--weights", help="head-permuted f16 transformer blocks (env CANCHO_MEDIA_EDIT_WEIGHTS)")
    e.add_argument("--text-weights", help="f16 text-encoder weights, written on first use (env CANCHO_MEDIA_EDIT_TEXT_WEIGHTS)")
    e.set_defaults(f=cmd_edit)
    i = sub.add_parser("image", help="text to image (not wired: says why)")
    i.add_argument("prompt")
    i.add_argument("-o", "--output")
    i.add_argument("--model", help="model name (`media models`)")
    i.set_defaults(f=cmd_image)
    sub.add_parser("models", help="list the supported models per task").set_defaults(f=cmd_models)
    d = sub.add_parser("doctor", help="check what is installed")
    d.set_defaults(f=cmd_doctor)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
