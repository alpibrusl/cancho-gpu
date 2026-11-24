#!/usr/bin/env python3
"""cancho-media: one command line for the image and video generators of this repository (docs/cli.md).

    scripts/media video "PROMPT" -o DIR [--frames N] [--size WxH] [--seed S] [--fps N] [--precise]
    scripts/media video "PROMPT" --image PICTURE -o DIR   (the base model continues the picture: docs/video-image.md)
    scripts/media edit  IMAGE "INSTRUCTION" -o DIR [--seed S] [--steps N]
    scripts/media image "PROMPT" -o DIR [--size WxH] [--seed N] [--steps N]
    scripts/media rewrite {video|edit} "PROMPT" [--image PICTURE [--also PICTURE2]]   (only the prompt rewriter, no GPU: what `--rewrite` would send)
    scripts/media models                       (the supported models per task; `--model NAME` on each task)
    scripts/media doctor                       (what is installed and what is missing)

It wraps the drivers that exist and does no GPU work itself: `video` is tests/wan_text_gpu.py (umT5's encoder, a GPU session; docs/diffusion.md V3) then tests/wan_video_gpu.py (FastWan2.2-TI2V-5B: three DMD steps and the VAE decoder, on the
Metal GPU), `edit` is tests/edit_pipeline_gpu.py (Qwen-Image-2.1-Turbo). Run it with the interpreter of the first image edits' venv (scripts/media does). One GPU job at a time: a lock file in the
cache directory refuses a second run. Limits are the ones measured (docs/diffusion.md V1b/V2b/V4a), not the model's: `--force` goes past them at your own risk of ending the session."""
import argparse
import glob
import os
import shlex
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.expanduser(os.environ.get("CANCHO_MEDIA_CACHE", "~/.cache/cancho-media"))
HF = os.path.expanduser("~/.cache/huggingface/hub")
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
MAX_TOKENS = 4000  # the transformer's attention is the three-kernel one here: measured only up to 1,100 tokens (V4a); the scores are 24 heads x S^2 x 4 B a block
MAX_PIXEL_FRAMES = 17 * 1024 * 576  # the decoder runs out of GPU memory above about this (docs/diffusion.md V2b): 17 frames of 1024 x 576 work, 17 of 1152 x 640 and 33 of 960 x 512 do not


# The models each task can use. A model is a name, the Hugging Face repositories it needs, and the driver that runs it; a new model is a new entry and its driver (docs/cli.md). Only these two exist.
MODELS = {
    "video": {"fastwan2.2-ti2v-5b": {"repo": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers", "also": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "note": "3 DMD steps, 640x352 to 1280x704, up to 33 frames measured"},
              "wan2.2-ti2v-5b-base": {"repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "also": None, "i2v": True, "note": "only with --image: 30 Euler steps with guidance, continues the picture (docs/video-image.md)"}},
    "edit": {"qwen-image-2.1-turbo": {"repo": "Qwen/Qwen-Image-2.1-Turbo", "also": None, "note": "8 steps, image + instruction"}},
    "image": {"qwen-image-2.1-turbo": {"repo": "Qwen/Qwen-Image-2.1-Turbo", "also": None, "note": "8 steps, a prompt, 512x512 to 1024x1024 measured"}},
}
BASE_VIDEO = "wan2.2-ti2v-5b-base"
# Wan2.2's own default negative prompt (its shared_config `sample_neg_prompt`, from memory of the repository, not read here): the model was trained with it; docs/video-image.md section 7
NEG_OFFICIAL = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
REWRITER = "qwen3.8:27b-mlx"  # the studio's default (docs/studio-rewrite.md D9): a local Ollama at 127.0.0.1:11434, nothing else
REWRITE_CLI = os.path.join(ROOT, "build/rewrite_cli")
REWRITE_PROMPTS = os.path.join(ROOT, "studio/rewrite")  # video.txt (a text-to-video prompt) and edit.txt (one picture); there is none for an image or a video from a picture
DEFAULT = {"video": "fastwan2.2-ti2v-5b", "edit": "qwen-image-2.1-turbo", "image": "qwen-image-2.1-turbo"}


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


def rewrite_prompt(kind, prompt, picture, model, seconds=600, picture2=None):
    """The prompt rewritten by the local model (docs/studio-rewrite.md), or the original with the reason it was not (the layer fails open, D4). Prints both. A missing binary is a refusal: the person asked for it."""
    if not os.path.isfile(REWRITE_CLI):
        die("--rewrite needs build/rewrite_cli: `cancho build studio/rewrite_cli.cho studio/rewrite.cho studio/files.cho --std -o build/rewrite_cli`")
    system = os.path.join(REWRITE_PROMPTS, ("edit2" if picture2 else kind) + ".txt")
    if not os.path.isfile(system):
        die("no system prompt for %s: %s" % (kind, system))
    pic = os.path.abspath(picture) if picture else "-"  # the rewriter's file reads want absolute paths
    r = subprocess.run([REWRITE_CLI, "ask", model, system, prompt, pic, str(seconds)] + ([os.path.abspath(picture2)] if picture2 else []), cwd=ROOT, capture_output=True, text=True, timeout=seconds + 60)
    lines = r.stdout.split("\n")
    head = lines[0] if lines else ""
    if r.returncode != 0 or len(lines) < 2 or not lines[1].strip():
        print("media: prompt not rewritten (%s): the original is used" % (head.partition(" ")[2] or (r.stdout.strip() or "no answer")), file=sys.stderr, flush=True)
        return prompt
    print("media: prompt rewritten by %s:\n  was: %s\n  now: %s" % (model, prompt, lines[1]), file=sys.stderr, flush=True)
    return lines[1]


def cmd_rewrite(a):
    if a.kind == "edit" and not a.image:
        die("rewrite edit needs --image PICTURE (the model reads it)")
    if a.kind == "video" and a.image:
        die("there is no rewriter template for a video from a picture (not measured); --image is for `rewrite edit`")
    if a.image and not os.path.isfile(a.image):
        die("no such image: " + a.image)
    if a.also and a.kind != "edit":
        die("--also is for `rewrite edit`")
    if len(a.also) > 1:
        die("the rewriter reads at most two pictures (image1 and --also once): the two-picture template is the one measured")
    if a.also and not os.path.isfile(a.also[0]):
        die("no such image: " + a.also[0])
    if not a.prompt.strip():
        die("the prompt is empty")
    new = rewrite_prompt(a.kind, a.prompt, a.image, a.rewrite_model, picture2=a.also[0] if a.also else None)
    print(new)
    sys.exit(0 if new != a.prompt else 1)


def picture_f32(path, w, h, dest):
    """The picture as [3, h, w] little-endian singles in [-1, 1]: resized to cover w x h and cropped at the centre (docs/video-image.md step 1)."""
    from PIL import Image
    import numpy as np
    try:
        im = Image.open(path).convert("RGB")
    except Exception as e:  # PIL raises many types for a file it cannot read
        die("cannot read the picture %s: %s" % (path, e))
    k = max(w / im.width, h / im.height)
    nw, nh = max(w, round(im.width * k)), max(h, round(im.height * k))
    im = im.resize((nw, nh), Image.LANCZOS)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    a = np.asarray(im.crop((x0, y0, x0 + w, y0 + h)), dtype=np.float32)
    np.ascontiguousarray((a / 127.5 - 1.0).transpose(2, 0, 1).astype("<f4")).tofile(dest)


def video_image(a, info, w, h, lf, frames):
    """media video --image: the base model continues the picture (docs/video-image.md)."""
    if lf < 2:
        die("--frames %d: frame 0 is the picture, ask for at least 2 (the model makes 1 + 4k, so 5)" % a.frames)
    if a.steps < 1 or a.guidance < 1:
        die("--steps must be at least 1 and --guidance at least 1 (1 turns guidance off)")
    if a.segments < 1:
        die("--segments must be at least 1")
    if a.segments > 2 and not a.force:
        die("--segments %d: the third clip of a chain degrades (smeared background, colour fringes; docs/video-chain.md section 4, measured on one picture); two are clean. --force to ask for more" % a.segments)
    if len(a.then) > a.segments - 1:
        die("--then given %d times but %d segments need at most %d (the first uses the prompt)" % (len(a.then), a.segments, a.segments - 1))
    base = snapshot(info["repo"])
    tdir = a.weights or snapshot(info["repo"], "transformer")
    enc = os.path.join(ROOT, "build/wan_vae_enc_gpu")
    if a.weights and not os.path.isfile(os.path.join(a.weights, "config.json")):
        die("--weights %s has no config.json: it must be the base model's transformer directory" % a.weights)
    if not base or not tdir or not snapshot(info["repo"], "vae"):
        die("the base model is not in the Hugging Face cache: `hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers` (about 30 GB; or pass --weights DIR for its transformer)")
    if not os.path.exists(enc):
        die("build/wan_vae_enc_gpu is missing: `cancho build tests/wan_vae_enc_gpu.cho tests/wan_enc/st.cho tests/wan_enc/codec.cho studio/files.cho --std -o build/wan_vae_enc_gpu` (docs/wan-vae-encoder.md section 10)")
    wdir = os.path.expanduser(a.wdir or os.path.join(CACHE, "wan-f16-base"))
    free = shutil.disk_usage(CACHE if os.path.isdir(CACHE) else "/").free
    if not os.path.exists(os.path.join(wdir, "proj_out.f16")) and free < 14e9:
        die("the base model's f16 weight files (10 GB) are written to %s on first use and only %.0f GB are free" % (wdir, free / 1e9))
    out = os.path.abspath(a.output)
    os.makedirs(out, exist_ok=True)
    vae = snapshot(info["repo"], "vae")
    segs = a.segments
    prompts = [a.prompt] + [t for t in a.then] + [a.prompt] * (segs - 1 - len(a.then))
    carry, ref, cors = a.image, None, []
    with Lock():
        for k in range(segs):
            sdir = out if segs == 1 else os.path.join(out, "seg%d" % (k + 1))
            os.makedirs(sdir, exist_ok=True)
            if segs > 1:
                print("== segment %d of %d (%d frames, seed %d)" % (k + 1, segs, frames, a.seed + k), flush=True)
            video_segment(a, w, h, lf, sdir, carry, prompts[k], a.seed + k, tdir, vae, enc, wdir)
            if segs == 1:
                break
            fs = sorted(glob.glob(os.path.join(sdir, "frames", "f*.png")))
            if ref is None:  # the first segment's own first frame is the picture as the model sees it: the colour reference of the whole video
                ref = frame_stats(fs[0])
            cor = ramp_correct(fs, ref, os.path.join(sdir, "corrected"), not a.no_correct)
            cors += cor if k == 0 else cor[1:]
            carry = cor[-1]  # the last frame is the clean one (docs/video-chain.md section 5)
    if segs == 1:
        open(os.path.join(out, "prompt.txt"), "w").write(a.prompt + "\n")
    else:
        join_segments(a, out, cors, prompts)
    mp4_and_done(a, out)


def video_segment(a, w, h, lf, sdir, image, prompt, seed, tdir, vae, enc, wdir):
    """One clip from a picture (docs/video-image.md); the GPU lock is the caller's."""
    work = os.path.join(sdir, "work")
    os.makedirs(work, exist_ok=True)
    pic, cond, txt, neg = (os.path.join(work, n) for n in ("picture.f32", "cond.f32", "text.npy", "neg.npy"))
    picture_f32(image, w, h, pic)
    run([enc, "prepare", os.path.join(vae, "diffusion_pytorch_model.safetensors"), work, ROOT, pic, str(h), str(w)], "picture encoder: prepare")
    run(["sh", "-c", "printf '1\\ncheckpoint\\n' | %s %s" % (shlex.quote(SESSION), shlex.quote(os.path.join(work, "enc.session")))], "picture encoder: run (GPU)")
    run([enc, "latent", os.path.join(vae, "config.json"), work, str(h), str(w), cond], "picture encoder: latent")
    tw = os.path.join(CACHE, "umt5-f16")
    run([sys.executable, "tests/wan_text_gpu.py", prompt, txt, "--wdir", tw], "text encoder (GPU)")
    run([sys.executable, "tests/wan_text_gpu.py", NEG_OFFICIAL if a.negative is None else a.negative, neg, "--wdir", tw], "negative text")
    run([sys.executable, "tests/wan_video_gpu.py", sdir, "--weights", tdir, "--text", txt, "--neg", neg, "--cond", cond, "--base", "--steps", str(a.steps), "--guidance", str(a.guidance),
         "--frames", str(lf), "--size", "%dx%d" % (h // 16, w // 16), "--seed", str(seed), "--wdir", wdir, "--fps", str(a.fps)], "base model, %d steps with guidance, and decoder (GPU)" % a.steps)


def frame_stats(path):
    """Per-channel mean and standard deviation of a picture (the colour reference of a chain)."""
    from PIL import Image
    import numpy as np
    a = np.asarray(Image.open(path).convert("RGB"), dtype=float).reshape(-1, 3)
    return a.mean(0), a.std(0)


def ramp_correct(frames, ref, dst, on):
    """The ramped colour correction of docs/video-chain.md: frame j of n has its per-channel mean and spread moved j/(n-1) of the way to the reference's, so the first frame (the previous segment's last,
    already matched) is untouched and the last is fully matched. Writes PNGs into `dst` and returns their paths; with `on` false it only returns the segment's own frames."""
    if not on:
        return list(frames)
    from PIL import Image
    import numpy as np
    os.makedirs(dst, exist_ok=True)
    rm, rs = ref
    out = []
    for j, f in enumerate(frames):
        w = j / (len(frames) - 1)
        a = np.asarray(Image.open(f).convert("RGB"), dtype=float)
        m, sd = a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0)
        o = (a - m) / (sd + 1e-6) * ((1 - w) * sd + w * rs) + ((1 - w) * m + w * rm)
        p = os.path.join(dst, os.path.basename(f))
        Image.fromarray(np.clip(o, 0, 255).astype(np.uint8)).save(p)
        out.append(p)
    return out


def join_segments(a, out, frames, prompts):
    """The chain's frames as frames/f000.png ... , video.gif and prompt.txt (one prompt a line) in `out`."""
    from PIL import Image
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    ims = []
    for i, f in enumerate(frames):
        im = Image.open(f).convert("RGB")
        im.save(os.path.join(out, "frames", "f%03d.png" % i))
        ims.append(im)
    ims[0].save(os.path.join(out, "video.gif"), save_all=True, append_images=ims[1:], duration=int(1000 / a.fps), loop=0)
    open(os.path.join(out, "prompt.txt"), "w").write("\n".join(prompts) + "\n")
    print("joined %d segments: %d frames" % (len(prompts), len(frames)))


def mp4_and_done(a, out):
    if shutil.which("ffmpeg"):
        mp4 = os.path.join(out, "video.mp4")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(out, "frames/f%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4])
        print("wrote " + mp4 if r.returncode == 0 else "media: ffmpeg failed; the gif and the frames are there")
    print("done: %s (video.gif, frames/, prompt.txt)" % out)


def cmd_video(a):
    if (a.segments != 1 or a.then or a.no_correct) and not a.image:
        die("--segments, --then and --no-correct are for a video from a picture (--image): they chain clips (docs/video-chain.md)")
    if a.rewrite and a.image:
        die("--rewrite is not available for a video from a picture: the rewriter has no template for it (not measured)")
    if a.precise and a.image:
        die("--precise is for a video from a prompt: the products with f32 operands were measured on the distilled model's three steps, not on the base model's (docs/diffusion.md section 6, V3c)")
    need_gpu()
    model, info = pick("video", a.model or (BASE_VIDEO if a.image else None))
    if a.image and not info.get("i2v"):
        die("--image needs the base model: %s is distilled and does not continue a picture (docs/wan-vae-encoder.md section 12); use --model %s" % (model, BASE_VIDEO))
    if info.get("i2v") and not a.image:
        die("%s is for --image (a picture to continue); for a video from a prompt alone use the default model" % model)
    if a.image and not os.path.isfile(a.image):
        die("no such image: " + a.image)
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
        die("%d frames at %dx%d decode as one clip and the decoder runs out of GPU memory above about 17 frames of 1024x576 or 33 of 640x352 (docs/diffusion.md V2b). Use fewer frames or a smaller size, or --force" % (frames, w, h))
    if a.image:
        return video_image(a, info, w, h, lf, frames)
    tdir = a.weights or snapshot(info["repo"], "transformer")
    if not tdir or not snapshot(info["also"]):
        die("weights not in the Hugging Face cache: `hf download FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` and `hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers` (or pass --weights DIR)")
    out = os.path.abspath(a.output)
    os.makedirs(out, exist_ok=True)
    wdir = os.path.expanduser(a.wdir or os.path.join(CACHE, "wan-f16"))
    free = shutil.disk_usage(os.path.dirname(wdir) if os.path.isdir(os.path.dirname(wdir)) else CACHE if os.path.isdir(CACHE) else "/").free
    if not os.path.exists(os.path.join(wdir, "proj_out.f16")) and free < 14e9:
        die("the f16 weight files (10 GB) are written to %s on first use and only %.0f GB are free" % (wdir, free / 1e9))
    orig = a.prompt
    with Lock():  # the rewriter's model (16 GB) is unloaded when it answers, so it never shares memory with the GPU job (docs/studio-rewrite.md D6)
        if a.rewrite:
            a.prompt = rewrite_prompt("video", a.prompt, None, a.rewrite_model)
        txt = os.path.join(out, "text.npy")
        run([sys.executable, "tests/wan_text_gpu.py", a.prompt, txt, "--wdir", os.path.join(CACHE, "umt5-f16")], "text encoder (GPU)")
        run([sys.executable, "tests/wan_video_gpu.py", out, "--weights", tdir, "--text", txt, "--frames", str(lf), "--size", "%dx%d" % (h // 16, w // 16),
             "--seed", str(a.seed), "--wdir", wdir, "--fps", str(a.fps)] + (["--f32"] if a.precise else []), "denoiser and decoder (GPU)")
    open(os.path.join(out, "prompt.txt"), "w").write(a.prompt + "\n")
    if a.prompt != orig:
        open(os.path.join(out, "prompt.orig.txt"), "w").write(orig + "\n")
    mp4_and_done(a, out)


def cmd_edit(a):
    if a.rewrite and len(a.also) > 1:
        die("--rewrite reads at most two pictures (the image and one --also): the two-picture template is the one measured")
    need_gpu()
    model, info = pick("edit", a.model)
    if not os.path.isfile(a.image):
        die("no such image: " + a.image)
    for extra in a.also:
        if not os.path.isfile(extra):
            die("no such image: " + extra)
    area = a.condition_area or (1024 * 1024 if not a.also else 768 * 768)
    prefix_rows = (len(a.also) + 1) * (area // 256) + 128  # the pictures' latent tokens and the text: the prefix pass keeps 32 x rows^2 attention scores, indexed with 32 bits
    if a.also and 32 * prefix_rows * prefix_rows >= 2 ** 32:
        die("--condition-area %d with %d pictures is about %d prefix rows: the attention scores pass what the kernels can index (at most about 11,500 rows; two pictures of 1024 x 1024 are 8,300). Lower it (docs/two-images.md)" % (area, len(a.also) + 1, prefix_rows))
    wd = a.weights or os.environ.get("CANCHO_MEDIA_EDIT_WEIGHTS") or os.path.join(CACHE, "qwen-edit-dit")
    tw = a.text_weights or os.environ.get("CANCHO_MEDIA_EDIT_TEXT_WEIGHTS") or os.path.join(CACHE, "qwen-edit-text")
    if not snapshot(info["repo"], "transformer"):
        die("Qwen-Image-2.1-Turbo is not in the Hugging Face cache (the repository's directory is empty): `hf download Qwen/Qwen-Image-2.1-Turbo` (about 50 GB)")
    if not os.path.isdir(wd):
        die("the transformer's f16 block weights (head-permuted, 14 GB, written by tests/dit_big.py: docs/diffusion.md D7) are not at %s: pass --weights DIR or set CANCHO_MEDIA_EDIT_WEIGHTS" % wd)
    out = os.path.abspath(a.output)
    os.makedirs(out, exist_ok=True)
    orig = a.instruction
    with Lock():
        if a.rewrite:
            a.instruction = rewrite_prompt("edit", a.instruction, a.image, a.rewrite_model, picture2=a.also[0] if a.also else None)
        run([sys.executable, "tests/edit_pipeline_gpu.py", os.path.abspath(a.image), os.path.join(out, "work"), "--prompt", a.instruction, "--seed", str(a.seed), "--steps", str(a.steps),
             "--weights", wd, "--text-weights", tw, "--out", os.path.join(out, "edited.png"), "--resident"]
            + sum((["--also", os.path.abspath(x)] for x in a.also), []) + (["--condition-area", str(area)] if a.also else []), "image edit (GPU, five stages)")
    open(os.path.join(out, "prompt.txt"), "w").write(a.instruction + "\n")
    if a.instruction != orig:
        open(os.path.join(out, "prompt.orig.txt"), "w").write(orig + "\n")
    print("done: " + os.path.join(out, "edited.png"))


def cmd_models(a):
    for task, have in MODELS.items():
        print("%s:" % task)
        for name, info in have.items():
            print("  %-22s %s%s  [%s]" % (name, info["repo"], " (default)" if DEFAULT.get(task) == name else "", info["note"]))
        if not have:
            print("  (none supported yet)")


def cmd_image(a):
    need_gpu()
    model, info = pick("image", a.model)
    try:
        w, h = (int(v) for v in a.size.lower().split("x"))
    except ValueError:
        die("--size is WxH in pixels, for example 1024x1024")
    if w % 32 or h % 32 or w < 64 or h < 64:
        die("--size must be multiples of 32, at least 64: got %dx%d" % (w, h))
    if w * h > 1024 * 1024:
        die("%dx%d is more than 1024 x 1024 pixels, the largest size measured (docs/text-to-image.md)" % (w, h))
    if not a.prompt.strip():
        die("the prompt is empty")
    wd = a.weights or os.environ.get("CANCHO_MEDIA_EDIT_WEIGHTS") or os.path.join(CACHE, "qwen-edit-dit")
    tw = a.text_weights or os.environ.get("CANCHO_MEDIA_EDIT_TEXT_WEIGHTS") or os.path.join(CACHE, "qwen-edit-text")
    if not snapshot(info["repo"], "transformer"):
        die("Qwen-Image-2.1-Turbo is not in the Hugging Face cache (the repository's directory is empty): `hf download Qwen/Qwen-Image-2.1-Turbo` (about 50 GB)")
    if not os.path.isdir(wd):
        die("the transformer's f16 block weights (head-permuted, 14 GB, written by tests/dit_big.py: docs/diffusion.md D7) are not at %s: pass --weights DIR or set CANCHO_MEDIA_EDIT_WEIGHTS" % wd)
    out = os.path.abspath(a.output)
    os.makedirs(out, exist_ok=True)
    with Lock():
        run([sys.executable, "tests/t2i_pipeline_gpu.py", os.path.join(out, "work"), "--prompt", a.prompt, "--seed", str(a.seed), "--steps", str(a.steps), "--size", "%dx%d" % (w, h),
             "--weights", wd, "--text-weights", tw, "--out", os.path.join(out, "image.png"), "--resident"], "text to image (GPU, three stages)")
    open(os.path.join(out, "prompt.txt"), "w").write(a.prompt + "\n")
    print("done: " + os.path.join(out, "image.png"))


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
    p.add_argument("--online", action="store_true", help="let the Hugging Face libraries contact the Hub (needed once, to fill the cache); the default is HF_HUB_OFFLINE=1, nothing leaves the machine")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("video", help="a video from a prompt (FastWan2.2-TI2V-5B, 3 steps)")
    v.add_argument("prompt")
    v.add_argument("-o", "--output", required=True, help="directory for video.gif, video.mp4, frames/")
    v.add_argument("--frames", type=int, default=17, help="video frames; rounded up to 1 + 4k (default 17)")
    v.add_argument("--size", default="640x352", help="WxH pixels, multiples of 32 (default 640x352)")
    v.add_argument("--model", help="model name (`media models`; default %s)" % DEFAULT["video"])
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--fps", type=int, default=16)
    v.add_argument("--precise", action="store_true", help="the transformer's products with f32 operands: the latents are within a hundredth of float64's, 2 to 3 s more (docs/diffusion.md section 6, V3c)")
    v.add_argument("--weights", help="FastWan transformer directory (default: the Hugging Face cache)")
    v.add_argument("--wdir", default=None, help="where the 10 GB of f16 weight files are kept between runs")
    v.add_argument("--image", help="a picture to continue: the first frame (uses the base model, 30 steps; docs/video-image.md)")
    v.add_argument("--steps", type=int, default=30, help="with --image: Euler steps (default 30)")
    v.add_argument("--guidance", type=float, default=3.5, help="with --image: classifier-free guidance (default 3.5, measured better than the model's 5 for a clip this short; 1 is off)")
    v.add_argument("--segments", type=int, default=1, help="with --image: chain this many clips, each starting from the previous one's last frame (default 1; two are clean, the third degrades: docs/video-chain.md); each is --frames long, the joined video has frames + (segments - 1) (frames - 1)")
    v.add_argument("--then", action="append", default=[], metavar="PROMPT", help="with --segments: the prompt of the next clip (repeat for more; the default is the first prompt again)")
    v.add_argument("--no-correct", action="store_true", help="with --segments: no colour correction toward the first picture (the colours drift darker and more saturated along a chain)")
    v.add_argument("--negative", default=None, help="with --image: the negative prompt (default Wan's own; an empty string for none)")
    v.add_argument("--force", action="store_true", help="go past the measured limits")
    v.add_argument("--rewrite", action="store_true", help="first have a local model (via Ollama) rewrite the prompt as a text-to-video prompt (not with --image); both are printed, the original is kept in prompt.orig.txt (docs/studio-rewrite.md)")
    v.add_argument("--rewrite-model", default=REWRITER, help="the Ollama model that rewrites (default %s)" % REWRITER)
    v.set_defaults(f=cmd_video)
    e = sub.add_parser("edit", help="edit an image from an instruction (Qwen-Image-2.1-Turbo)")
    e.add_argument("image")
    e.add_argument("instruction")
    e.add_argument("--also", action="append", default=[], metavar="IMAGE", help="another picture to combine with the first (repeat for more; docs/two-images.md): the instruction can refer to them")
    e.add_argument("--condition-area", type=int, default=None, help="pixels each picture is resized to (default 1048576 for one picture, 589824 = 768 x 768 for several; two pictures of 1048576 are the largest measured)")
    e.add_argument("-o", "--output", required=True, help="directory for edited.png")
    e.add_argument("--model", help="model name (`media models`; default %s)" % DEFAULT["edit"])
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--steps", type=int, default=8)
    e.add_argument("--rewrite", action="store_true", help="first have a local model (via Ollama; it sees the picture) rewrite the instruction (with --also, at most one: it reads both); both are printed, the original is kept in prompt.orig.txt (docs/studio-rewrite.md)")
    e.add_argument("--rewrite-model", default=REWRITER, help="the Ollama model that rewrites (default %s)" % REWRITER)

    e.add_argument("--weights", help="head-permuted f16 transformer blocks (env CANCHO_MEDIA_EDIT_WEIGHTS)")
    e.add_argument("--text-weights", help="f16 text-encoder weights, written on first use (env CANCHO_MEDIA_EDIT_TEXT_WEIGHTS)")
    e.set_defaults(f=cmd_edit)
    i = sub.add_parser("image", help="text to image from a prompt (Qwen-Image-2.1-Turbo)")
    i.add_argument("prompt")
    i.add_argument("-o", "--output", required=True, help="directory for image.png")
    i.add_argument("--model", help="model name (`media models`; default %s)" % DEFAULT["image"])
    i.add_argument("--size", default="1024x1024", help="WxH pixels, multiples of 32, at most 1024 x 1024 (default 1024x1024)")
    i.add_argument("--seed", type=int, default=0)
    i.add_argument("--steps", type=int, default=8)
    i.add_argument("--weights", help="head-permuted f16 transformer blocks (env CANCHO_MEDIA_EDIT_WEIGHTS)")
    i.add_argument("--text-weights", help="f16 text-encoder weights (env CANCHO_MEDIA_EDIT_TEXT_WEIGHTS)")
    i.set_defaults(f=cmd_image)
    r = sub.add_parser("rewrite", help="only the prompt rewriter (no GPU): print what --rewrite would use")
    r.add_argument("kind", choices=["video", "edit"])
    r.add_argument("prompt")
    r.add_argument("--image", help="for edit: the picture the model reads")
    r.add_argument("--also", action="append", default=[], metavar="IMAGE", help="for edit: the second picture (image2); at most one")
    r.add_argument("--rewrite-model", default=REWRITER, help="the Ollama model (default %s)" % REWRITER)
    r.set_defaults(f=cmd_rewrite)
    sub.add_parser("models", help="list the supported models per task").set_defaults(f=cmd_models)
    d = sub.add_parser("doctor", help="check what is installed")
    d.set_defaults(f=cmd_doctor)
    a = p.parse_args()
    if not a.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # inherited by the stages (subprocess): the cache is read, the Hub is not asked
    a.f(a)


if __name__ == "__main__":
    main()
