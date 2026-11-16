# cancho-media: one command line for images and video

`scripts/media` (the program is `tools/media.py`, Python standard library, run with the venv of the first image edits) wraps the generators that exist; it does no GPU work of its own and adds no kernel.

```
scripts/media doctor
scripts/media models
scripts/media video "A red fox running through fresh snow in a pine forest at sunrise, cinematic" -o out/fox --frames 17 --size 640x352 --seed 1
scripts/media edit photo.jpg "Make it a bright sunny afternoon." -o out/sunny --weights DIR --text-weights DIR
scripts/media image "A lighthouse on a rocky coast at dusk, photograph" -o out/lighthouse --size 1024x1024 --seed 2
```

* **`video`**: `tools/wan_text.py` (umT5 on the host, torch) then `tests/wan_video_gpu.py` (FastWan2.2-TI2V-5B, three DMD steps, the VAE decoder, Metal). Writes `video.gif`, `video.mp4` (if `ffmpeg` exists), `frames/`, `prompt.txt` in `-o DIR`. `--frames` is rounded up to 1 + 4k; `--size WxH` in pixels, multiples of 32. The 10 GB of f16 weight files are written once to `~/.cache/cancho-media/wan-f16` (`CANCHO_MEDIA_CACHE`, `--wdir`) and reused.
* **`edit`**: `tests/edit_pipeline_gpu.py --resident` (Qwen-Image-2.1-Turbo: vision tower, language model, VAE encoder, eight steps, decoder). Writes `edited.png`.
* **`--model NAME`** on `video`, `edit` and `image`; **`models`** lists them. The registry (`MODELS` in `tools/media.py`) holds, per task, the model's name and Hugging Face repositories; today `video`: `fastwan2.2-ti2v-5b` (default), `edit`: `qwen-image-2.1-turbo` (default), `image`: none. Any other name is refused with the supported list (run: `wan2.1`, `flux`, `foo`). Adding a model is a registry entry **and a driver for it**: the drivers fix the architecture's dimensions, kernels and sampler, so `--model` does not make another checkpoint run by itself.
* **`doctor`**: what is installed (platform, session binary, the two Wan and the Qwen snapshots, ffmpeg, the python modules, free disk).
* **Offline by default.** `media` sets `HF_HUB_OFFLINE=1` for every stage, so `transformers` and `huggingface_hub` read the local cache and do not ask the Hub whether a newer revision exists (they otherwise do, on every call, sending the repository and file names, never the prompt or the image). `--online` (before the subcommand: `media --online doctor`) lifts it, for filling the cache the first time. A file missing from the cache is an error, not a download. Measured 2026-11-16 with the cache full: `video` (17 frames, 640 x 352; 18 s GPU stage, 27 s in all) and `edit` (90 s) both ran offline to the end. Not measured: network traffic itself (no capture); the claim is the library's documented behaviour plus both runs succeeding with the Hub switched off.
* **One GPU job at a time**: a lock file (`gpu.lock` in the cache directory) refuses a second run.
* **Limits** are the measured ones, not the model's; `--force` goes past them: more than 4,000 tokens (the denoiser's attention is the three-kernel one here, measured only to 1,100 tokens: the fused `ATT=1` of docs/diffusion.md V1b is **not** wired into `tests/wan_video_gpu.py`), more than 17 x 1024 x 576 pixel-frames of decode: the decoder runs out of GPU memory above about that (17 frames of 1024 x 576 and 33 of 640 x 352 run; 17 of 1152 x 640 and 33 of 960 x 512 do not: V2b).

## What was run (2026-11-15, Apple M4 Max, 64 GB)

* `doctor`: all rows `ok`.
* **`video`, 17 frames, 640 x 352, seed 1**, the fox prompt above: **44 s of wall clock** for the command (text encoder 12 s; the GPU stage 32 s: writing the weight files 13 s, three steps 3.3 + 1.1 + 1.1 s, decoder 12.2 s), `video.mp4` and `video.gif` written. One run, a cold start of the weight files; the result was looked at, not compared with torch's pipeline.
* Refusals checked by hand: a size that is not a multiple of 32, 121 frames at 1280 x 704 (27,280 tokens: refused for the tokens first), a missing image, missing edit weights, a second job while the lock is held, `image` with an empty prompt or a bad size.

## What this does not do

* **`edit` was not run end to end here.** The Qwen-Image-2.1-Turbo snapshot is in the cache, but the transformer's head-permuted f16 block weights (14 GB, `tests/dit_big.py`, docs/diffusion.md D7) are not on disk, and `media` does not make them: it refuses with the path it looked at. Its argument handling and refusals were run; the command it issues is the one `tests/edit_pipeline_gpu.py` documents.
* **`image` (text to image from a free prompt) runs** (2026-11-16, `tests/t2i_pipeline_gpu.py`, docs/text-to-image.md): Qwen-Image-2.1-Turbo, 8 steps, `--size WxH` (multiples of 32, at most 1024 x 1024; default 1024 x 1024), `--seed`, `--steps`, the edit's two weight directories; writes `image.png` and `prompt.txt` in `-o DIR`. **Measured:** 512 x 512 in 27 s for the command (about 25 s of GPU), 1024 x 1024 about 75 s of GPU; against torch's BF16 pipeline 39.6 dB at 512 and 26.9 dB at 1024 (the distance docs/diffusion.md D7 documented; the GPU is the nearer to the exact image). Refusals: an empty prompt, a size that is not a multiple of 32 or is over 1024 x 1024, missing weights.
* No image-to-video, no video editing, no audio; the text encoder of `video` is torch on the host; large videos wait on the decoder's time chunking and the fused attention's wiring.
