# `media video --image`: a video that continues a picture (design, 2026-11-17)

> **Status: design, then built in the same PR; the measured result goes in section 5.** Claims are tagged **[read]**, **[measured]** or **[estimate]**. The pieces exist and were measured one by one (docs/wan-vae-encoder.md sections 10 to 12); this document is only the command that runs them in order.

## 1. What it does

`scripts/media video "PROMPT" --image PICTURE -o DIR [--frames N] [--size WxH] [--seed S] [--steps N] [--guidance G]`

The picture is the first frame; the prompt says what happens next. **It uses the base model (Wan2.2-TI2V-5B), not FastWan**: the distilled weights cut to another scene after frame 0 and the base model continues the picture (docs/wan-vae-encoder.md section 12, **[measured]** one picture, one seed). The price is 30 Euler steps with guidance (two transformer calls a step): **[measured]** 2.2 s a step at 17 frames of 640 x 352, so about 66 s for the sampler plus the decoder (11 s) and the one-time weight conversion.

## 2. The steps, in order (one GPU job; the lock is taken once)

1. **Picture to `[3, H, W]` floats in [-1, 1]** at the chosen `WxH`: resized to cover and cropped at the centre (no stretching), RGB, `x / 127.5 - 1`. This is the only new code, in `tools/media.py` (the wrapper that already imports PIL and numpy; no new file, no new language).
2. `build/wan_vae_enc_gpu prepare` then `build/cancho-gpu-metal-session <work>/enc.session` then `build/wan_vae_enc_gpu latent`: the first frame's latent `cond.f32`.
3. `tests/wan_text_gpu.py` twice (umT5's encoder on the GPU, `docs/diffusion.md` V3; `tools/wan_text.py`, torch on the host, until 2026-11-23): the prompt and the empty negative.
4. `tests/wan_video_gpu.py --base --cond cond.f32 --neg neg.npy --steps 30 --guidance 5`, with the base transformer, its own f16 directory (`~/.cache/cancho-media/wan-f16-base`) and `--size (h/16)x(w/16)`.
5. The mp4 as for `video`.

## 3. Refusals (each exits 2 with the reason and its remedy; none reaches a crash)

No such picture, or a file PIL cannot read; `--image` with `--model` naming FastWan (the model has no registered way to do this); a size that is not a multiple of 32; fewer than 2 frames (frame 0 is the picture); more than 4,000 tokens or the decoder's pixel limit (as `video`); `--steps` below 1; `--guidance` below 1; the base transformer, the VAE or the encoder binary (`build/wan_vae_enc_gpu`, with the `cancho build` line that makes it) missing; less than 14 GB free where the base weights' f16 files would be written.

## 4. What it does not do

Not wired into the studio. No UniPC sampler (Euler is built). **The base model at 17 frames and 640 x 352 degrades into artefacts after about frame 4, torch does the same (docs/wan-vae-encoder.md section 12)**: the command is honest about the model, it does not hide that; longer or larger clips are limited by the same measured token and decoder limits as `video`. The picture is never copied outside `DIR/work`.

## 5. Results

**[measured] 2026-11-17, one run, M4 Max.** The picture is a model-made one (`media image`, the fox prompt of docs/text-to-image.md, 640 x 352, seed 1; not a person's photo). `media video "…the fox turns its head" --image pic.png --frames 17 --size 640x352 --seed 1`: **93 s of wall clock** (picture encoder 1 + 0 + 0 s, umT5 8 s and 4 s, base model 30 steps with guidance and the decoder 80 s; the f16 files already existed). The encoder's latent is a real one (42,240 values, std 0.58). The script's own number for the mixing: frame 0 of the latent equals the condition (0).

**Looked at** (frames 0, 1, 4, 8, 16 beside the picture): **every frame is the same fox in the same snow and trees**, frame 0 is the picture; the fox hardly moves (it did not turn its head: the prompt's action is not followed at 17 frames). **From frame 1 a streak of saturated colour crosses the top of the picture** in some frames (frame 1, 4, 8; frame 16 is clean): the artefact docs/wan-vae-encoder.md section 12 measured in torch too at this size, not scored here. Relative rms of frame k against frame 0 (the script's number): k=1 0.096, 4 0.196, 8 0.252, 16 0.239 (the earlier fox picture gave 0.307 at k=1: another picture and prompt, so no comparison is made).

**Second run, the fox running** (prompt: the fox bolts out of the snow and runs, seed 3, 17 frames, 89 s; the video was sent to the owner): **the motion is there** (frame 16 against frame 0: 1.045 relative rms against 0.239 for the head-turn prompt): the fox stands, crouches, leaps and runs toward the camera. **Looked at: the first six frames are the picture's fox and scene; from about frame 8 the fox and the snow ball it kicks up deform and the background drifts into blur with bright artefacts**, so the quality of a clip is good for about half a second and degrades after: the base model's regime at this size (section 4), not scored.

**Provoked afterwards (2026-11-17):** a portrait picture (512 x 768) into 640 x 352: the centre crop works (frame 0 is the cropped lighthouse; 5 frames, frame 1 washed out, 0.78 against frame 0, a known artefact of very short base-model clips, one run). Refused with reason and exit 2: size 641x352, `--frames 1`, `--steps 0`, a file that is not a picture, the encoder binary missing. **Found and fixed:** `--weights` naming a directory that does not exist was only noticed by `wan_video_gpu.py` after the encoder and both text stages (exit 1, a traceback); it is now refused first (a `config.json` must be there). Not provoked: the base model absent from the Hugging Face cache, and disk space.

**Not shown:** one picture, prompt and seed; no torch reference of this run (the section 12 reference is for another picture); no run with a portrait or a different aspect (the crop was not exercised: the picture was already 640 x 352); the refusals for a missing base model, encoder binary or disk were not provoked (only a missing picture, `--image` with the distilled model, and the base model without `--image`).

## 6. In the studio (2026-11-17)

**API only.** `POST /api/jobs` with `"kind":"video"` and `"image":"/absolute/path"` (the same field an edit uses; a path on the server's machine) queues `media video ... --steps N --image PATH -- PROMPT`, so the base model continues the picture. Refused with 422 and a reason: `image` on an image job or not a string path, `frames` below 2, `rewrite` other than `off` (the rewriter has not been measured on this), `steps` on a video without a picture. The rewrite is forced off for these jobs. `studio/check.sh` holds the arguments, six refusals (nothing queued), the job page's parameter list and the OpenAPI text: **413 ok, 0 failed**. **The form (2026-11-17):** the picture field of `studio.cui` is now enabled for a video as well as an edit (empty: a video from the prompt alone); with a path the page sends it as `image` and the rewrite as `off`. **Changing "What to make" empties the path**, so a path typed for an edit is not sent by accident as a video's first frame (found while adding it: the old trace's second video would have sent the edit's path). The replay trace (44 steps, 88 checks) holds the body sent. Steps are not sent for a video (the server's default of 30 applies). **Not done:** the form was not looked at in a browser; **no real job was run through the studio's worker** (the checks use the fake worker; the argument list was only parsed by `media`, which refused the missing picture as it should); `/api/models` still lists only the distilled model (the base one is chosen by `--image`).
## 7. Why the clip degraded, and what helped (measured 2026-11-17)

**Question:** is the deterioration after about frame 8 the model's, or ours? **Measured:** the same picture, prompt (the running fox), noise (seed 3), 17 frames of 640 x 352, 30 steps, only the guidance and the negative prompt changed (the second text, `neg.npy`; the first is unchanged). Relative rms of frame k against frame 0, as the script prints it, and what the frames look like (**looked at**, frames 0, 4, 8, 12, 16):

| negative | guidance | k=1 | 4 | 8 | 16 | looked at |
|---|---|---|---|---|---|---|
| empty (as built until now) | 5 | 0.219 | 0.477 | 0.697 | 1.045 | fox deforms from frame 8, blur and bright artefacts |
| Wan's own | 5 | 0.284 | 0.498 | 0.764 | 1.143 | worse: white blobs, the fox dissolves |
| empty | 3.5 | 0.198 | 0.396 | 0.623 | 0.726 | coherent, but the fox's tail and legs bend wrongly by frame 16 |
| **Wan's own** | **3.5** | 0.220 | 0.396 | 0.592 | 0.595 | **a coherent fox runs to the end**, clean legs and tail |
| Wan's own | 2.5 | 0.135 | 0.276 | 0.459 | 0.582 | coherent, moves less (it mostly trots in place) |

**Reading:** **the guidance was the main cause** (the model's default 5 over-saturates a short clip); the negative prompt does nothing useful at 5 and tidies the anatomy at 3.5. One picture, one prompt, one seed, so the ranking of 3.5 and 2.5 is a taste for how much motion; the table is what happened, not a law. **Changed:** `media video --image` now defaults to guidance 3.5 and the negative prompt above (`--guidance`, `--negative ""` for none); the end-to-end run of the new defaults gave the same numbers as the experiment (0.220, 0.396, 0.592, 0.595). The negative's text was written from memory of Wan2.2's `shared_config.py`, **not re-read here**; it was not compared with the file. Not tried: guidance schedules, UniPC, longer clips, and other pictures (a different picture may want another guidance).
