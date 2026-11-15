# cancho-media: one command line for images and video

`scripts/media` (the program is `tools/media.py`, Python standard library, run with the venv of the first image edits) wraps the generators that exist; it does no GPU work of its own and adds no kernel.

```
scripts/media doctor
scripts/media models
scripts/media video "A red fox running through fresh snow in a pine forest at sunrise, cinematic" -o out/fox --frames 17 --size 640x352 --seed 1
scripts/media edit photo.jpg "Make it a bright sunny afternoon." -o out/sunny --weights DIR --text-weights DIR
scripts/media image "..."     # says why it does not run
```

* **`video`**: `tools/wan_text.py` (umT5 on the host, torch) then `tests/wan_video_gpu.py` (FastWan2.2-TI2V-5B, three DMD steps, the VAE decoder, Metal). Writes `video.gif`, `video.mp4` (if `ffmpeg` exists), `frames/`, `prompt.txt` in `-o DIR`. `--frames` is rounded up to 1 + 4k; `--size WxH` in pixels, multiples of 32. The 10 GB of f16 weight files are written once to `~/.cache/cancho-media/wan-f16` (`CANCHO_MEDIA_CACHE`, `--wdir`) and reused.
* **`edit`**: `tests/edit_pipeline_gpu.py --resident` (Qwen-Image-2.1-Turbo: vision tower, language model, VAE encoder, eight steps, decoder). Writes `edited.png`.
* **`--model NAME`** on `video`, `edit` and `image`; **`models`** lists them. The registry (`MODELS` in `tools/media.py`) holds, per task, the model's name and Hugging Face repositories; today `video`: `fastwan2.2-ti2v-5b` (default), `edit`: `qwen-image-2.1-turbo` (default), `image`: none. Any other name is refused with the supported list (run: `wan2.1`, `flux`, `foo`). Adding a model is a registry entry **and a driver for it**: the drivers fix the architecture's dimensions, kernels and sampler, so `--model` does not make another checkpoint run by itself.
* **`doctor`**: what is installed (platform, session binary, the two Wan and the Qwen snapshots, ffmpeg, the python modules, free disk).
* **One GPU job at a time**: a lock file (`gpu.lock` in the cache directory) refuses a second run.
* **Limits** are the measured ones, not the model's; `--force` goes past them: more than 4,000 tokens (the denoiser's attention is the three-kernel one here, measured only to 1,100 tokens: the fused `ATT=1` of docs/diffusion.md V1b is **not** wired into `tests/wan_video_gpu.py`), more than 17 x 1024 x 576 pixel-frames of decode: the decoder runs out of GPU memory above about that (17 frames of 1024 x 576 and 33 of 640 x 352 run; 17 of 1152 x 640 and 33 of 960 x 512 do not: V2b).

## What was run (2026-11-15, Apple M4 Max, 64 GB)

* `doctor`: all rows `ok`.
* **`video`, 17 frames, 640 x 352, seed 1**, the fox prompt above: **44 s of wall clock** for the command (text encoder 12 s; the GPU stage 32 s: writing the weight files 13 s, three steps 3.3 + 1.1 + 1.1 s, decoder 12.2 s), `video.mp4` and `video.gif` written. One run, a cold start of the weight files; the result was looked at, not compared with torch's pipeline.
* Refusals checked by hand: a size that is not a multiple of 32, 121 frames at 1280 x 704 (27,280 tokens: refused for the tokens first), a missing image, missing edit weights, a second job while the lock is held, `image`.

## What this does not do

* **`edit` was not run end to end here.** The Qwen-Image-2.1-Turbo snapshot is in the cache, but the transformer's head-permuted f16 block weights (14 GB, `tests/dit_big.py`, docs/diffusion.md D7) are not on disk, and `media` does not make them: it refuses with the path it looked at. Its argument handling and refusals were run; the command it issues is the one `tests/edit_pipeline_gpu.py` documents.
* **`image` (text to image from a free prompt) does not run**: `tests/pipeline_gpu.py` runs D0's golden prompt (token ids and noise are files). A driver for a prompt of one's own would need the tokenisation, the noise and the text encoder of the edit pipeline without an image; not built.
* No image-to-video, no video editing, no audio; the text encoder of `video` is torch on the host; large videos wait on the decoder's time chunking and the fused attention's wiring.
