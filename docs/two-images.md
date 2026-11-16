# Combining two (or more) pictures in one edit (design, 2026-11-17)

> **Status: built and measured for two pictures with conditions up to 960 x 960; **1024 x 1024 conditions give a wrong picture and are refused** (section 7); `media edit --also` runs it.** Claims are **[read]** (from `diffusers`' `transformer_qwenimage21.py` and `pipeline_qwenimage21.py` in the venv, and this repository's files; not run), **[reasoned]** or **[estimate]**. The owner asked (2026-11-16) for "combining two images in the input request".

## 1. What was read: the model takes N pictures

* **The pipeline** (`pipeline_qwenimage21.py`) **[read]**: `image` is a **list**. Each picture is resized to the area `output_resolution`^2 at its own aspect (`calculate_dimensions`), converted to RGBA, and gets its own entry in `img_shapes = [[(1, h_i/16, w_i/16) for each picture] + [target]]`. The text template repeats `<image1><|vision_start|><|image_pad|><|vision_end|> <image2><|vision_start|><|image_pad|><|vision_end|> ...` (a space between pictures), the processor takes the list of pictures in order, and `image_pad_mask` marks every picture's pad rows; the target's slots are appended. The latents of all the pictures (VAE mean, normalised, packed 2 x 2) are concatenated in order before the target's.
* **The transformer** (`QwenImage21Transformer2DModel`) **[read]**: `build_token_metadata` labels each token with its image block **from the block lengths in `img_shapes`** (not from runs of pad rows: "two condition images that happen to sit next to each other ... must stay separate blocks"). The mask (`build_qwenimage21_block_causal_mask`) is `(q_idx >= kv_idx) or same_image_block`: **the whole joint sequence is causal, and each image block (every condition picture and the target) is bidirectional inside itself**: so a later picture sees the earlier ones and the text before it; the target sees everything. `QwenImage21Rope`: text advances one position on all three axes; **each image block freezes the frame axis at the position reached by the preceding text**, lays its tokens on a height x width grid centred on zero, and advances the cursor by `max(height, width)` for the next block. With `causal_condition` (the default) the text and the conditions are modulated from t = 0 and their keys and values are cached after the first step.
* **This repository** **[read]**: `tools/dit_edit_ref.py` `rope_angles(shapes, pad_exp)` **already loops over a list of shapes** (conditions first, the target last) and `assemble` differs only in `block_len = [len(cond), ntgt]` and `np.repeat(np.arange(2), block_len)`: two blocks, written for one picture; `tools/text_img_ref.py` `positions(types, grid_hw)` takes one grid for all the picture runs of `types`; `build` takes one picture and the one-picture template; `tests/edit_pipeline_gpu.py` (D9g) does each stage once for one photograph (`vit_gpu.py --image`, `text_img_gpu.py`, `vae_enc_gpu.py --image`, `der.SHAPES = [(GH, GW), (GH, GW)]`).

## 2. What changes, stage by stage

| stage | for N pictures | size of the change |
|---|---|---|
| host: resize | each picture to the area of `--condition-area` (default `output_resolution`^2 as the pipeline) at its own aspect, sides rounded to 32: **a size per picture** | a loop |
| 1. vision tower | **one run per picture** (`tests/vit_gpu.py --image`, as it is). **[reasoned]** Qwen3-VL's tower attends inside each picture only (its `cu_seqlens` come from `image_grid_thw`), so separate runs equal the batched one; this is **checked** (section 4), not assumed. The merged rows and the three deepstack maps are concatenated in picture order. | a loop |
| 2. language model | the template of section 1 with N placeholders, the processor given the N pictures; `positions` with **one grid per picture run**; the embeddings of the pad rows from the concatenated tower rows; the deepstack added at the concatenated pad rows (`deep_f64` `[3, sum, 4096]`, which `text_img_gpu.py` already takes in this shape). The rows grow to about 60 + 1,024 per picture at 1024^2, padded to 64 | `build`, `positions`: a list instead of one |
| 3. VAE encoder | once per picture (`tests/vae_enc_gpu.py --image`); the normalised latents concatenated in order | a loop |
| 4. the sequence | `assemble` with `block_len = [h*w for (h, w) in SHAPES]` (conditions then target) and `ids = np.repeat(np.arange(len(SHAPES)), block_len)`; `pad` is the text encoder's pad mask over all pictures; the pad slots are filled in order by the concatenated conditions then the target's (`x[np.nonzero(pad_exp)] = img`): **already how it is written** | 3 lines |
| 5. the session | `Edit(work, wdir, NP, NPR, NTG, ...)`: the prefix is the text and **all** the condition rows, the same kernels | none expected; **memory and time are the open question (below)** |
| 6. decoder | the target's: unchanged | none |

**One picture must give what it gives today:** the generalised driver run with a single picture must produce **the same PNG as `tests/edit_pipeline_gpu.py` of main**, byte for byte, on the same photograph, prompt and seed (the regression gate).

## 3. The open question: size

**[estimate]** With each picture at 1024^2 (4,096 condition tokens) and a target of 1024^2, the joint sequence is about 12.4 k rows (two pictures); the prefix pass is 8.3 k rows (the edit's is 4.2 k) and each cached step has 4,096 query rows over 12.4 k keys: the scores of a block are 32 x 4,096 x 12,400 x 4 B = **6.5 GB** (4.4 GB with one picture), so the groups (`--gp`, `--gs`) shrink again; a step about 1.5 times the edit's. docs/diffusion.md D9 measured the one-picture edit at 8 to 9 s a step; so **[estimate] 12 to 15 s a step for two**, a prefill of about 15 s. **To be measured, not assumed:** whether the session fits in memory at 12.4 k rows, and the time. **The mitigation that stays inside the model:** conditions at a smaller area (the pipeline ties it to `output_resolution`, a parameter; each block's rotary positions are its own, so a smaller picture is a smaller block): 512^2 conditions are 1,024 tokens each. Whether the model uses a 512^2 condition as well as a 1024^2 one is a **quality question**, looked at, not scored.

## 4. The checks, written before the first run

1. **The transformer.** `tools/dit_edit_ref.py` with two pictures (`--torch32`): the float64 joint step against `QwenImage21Transformer2DModel` in float32: **mean abs error at most the one-picture run's (to be read from docs/diffusion.md D9e, 3.0e-4 of the rms)**; and the **mutants** that must fail: the two pictures given **one block id** (they would attend to each other bidirectionally: the docstring's warning), and the rotary angles with the second picture's frame position not advanced.
2. **The vision tower, once per picture against the batched one.** Torch's `visual` over both pictures at once (`pixel_values` and `image_grid_thw` of the processor) against the GPU's two runs: rows equal at the tower's own level (`tests/vit_gpu.py`'s bound), per picture; this is the check of the claim in section 2.
3. **The language model with two pictures.** The GPU's rows against `tools/text_img_ref.py` in float64 for a two-picture sequence: the bound of the one-picture run (**5% of the rms**, D9c), and the mutants of that stage (`pos1d`: 1-D positions; `no-deepstack`) must fail with two pictures as with one.
4. **The whole run against torch's BF16 pipeline**, `image=[a, b]`, the same prompt, size, seed and **noise** (the pipeline's `latents`): PSNR, with the **floor** measured as docs/text-to-image.md section 7 did (torch with its text against torch with the GPU's text rows), and **looked at**: the result uses both pictures (a thing from `a` in the scene of `b`, as asked).
5. **Refusals.** Zero pictures is the text-to-image path; more than the limit measured; a file that is not a picture; the same picture twice is allowed (not special-cased).

**Pictures for the tests:** never a user's photograph: **the two pictures the text-to-image driver made** (a fox in snow and a lighthouse at dusk, docs/text-to-image.md), kept out of the repository (`scratchpad/`, `.gitignore`: `*.png`); the prompt asks to put the fox beside the lighthouse.

## 5. What this will not show

One model, two pictures of 512^2 to 1024^2 and one prompt; no float64 run of the whole pipeline (as for the edit and the text-to-image: the transformer's alone takes hours); more than two pictures is **expected** to work by the same loops and is **not run**; the studio and `media edit` take one picture today and are **not changed until the checks above pass** (the studio's `/edit` form and the API's `image` would take a list; its own slice).

## 6. Order

(1) This document. (2) `assemble` generalised (3 lines) and check 1, with the two mutants. (3) `build` and `positions` for N pictures and check 3. (4) The driver as a loop over pictures, the regression gate (one picture = today's PNG), check 2 and the first two-picture run at 512^2 conditions. (5) 1024^2 conditions: memory and time, check 4. (6) Only then `media edit` with several `IMAGE`s and the studio.

## 7. Results (measured, 2026-11-17)

**Built.** `tools/dit_edit_ref.py` `assemble` for N pictures (`block_len` from `SHAPES`); `tools/text_img_ref.py` `template_n`, `build` and `positions` for a list; `tests/edit_pipeline_gpu.py IMAGE WORK --also IMAGE2 ... [--condition-area PIXELS]` (a loop over the pictures for the tower, the VAE encoder and the resize); `media edit IMAGE "instruction" --also IMAGE2 [--condition-area N]`.
**The regression gate passes: with one picture the driver's PNG is byte for byte the one `tests/edit_pipeline_gpu.py` of main writes** (`cmp`, same photograph, prompt and seed).

**Checks of section 4.**
1. **The transformer, two pictures of different shapes** (8 x 6 and 6 x 10 latent tokens, a target of 8 x 8, random rows): float64 against `QwenImage21Transformer2DModel` float32: **5.3e-6 of the rms** (mean abs 6.3e-6, max 1.2e-4). **Mutants refused:** the conditions as one block **4.7e-2** (8,800 times the baseline), the second picture's frame position not advanced **0.19**.
2. **The vision tower: one run per picture equals the batched run:** read in the library (`cu_seqlens`) and held in float64: the float64 language model over two pictures (a tower per picture, concatenated) against `Qwen3VLForConditionalGeneration` float32 with both pictures through the processor: **9.9e-5 of the rms** (192 image pads, 39 text rows). **Not measured on the GPU's f16 tower separately** (the run below measures the whole).
3. **The language model on the GPU with two pictures**, against the float64 run: text rows **6.3e-4 of the rms** (bound 5%); mutants **`pos1d` 0.137 and `no-deepstack` 0.117: both FAIL as required.**
4. **The whole run against torch's BF16 pipeline** (`image=[fox, lighthouse]`, the same prompt "Place the red fox sitting on the rocks in front of the lighthouse at dusk", seed and noise), the pictures the text-to-image driver made, a 1024 x 1024 target:
| conditions | tokens each | prefix rows (NP) | GPU against torch | floor (torch with the noise perturbed by 1e-3) | a step |
|---|---|---|---|---|---|
| 512 x 512 | 1,024 | 2,084 (4,096 padded) | **35.3 dB** | 45.4 dB | 7 s |
| 768 x 768 | 2,304 | 4,644 (4,672) | **31.1 dB** | not run | 9 s |
| 960 x 960 | 3,600 | 7,236 (7,296) | **20.8 dB** | 26.3 dB | 21 s |
| 1024 x 1024 | 4,096 | 8,228 (8,256) | **9.6 dB: wrong** | | 26 s |
**Looked at:** in every run the fox of the first picture sits on the rocks before the lighthouse of the second, and torch's does too. **At 960 x 960 the GPU's and torch's pictures are two good pictures of the same scene with the fox at a different size (the PSNR falls because the trajectory differs, not because one is corrupted; the floor of 26.3 dB says a perturbation of 1e-3 in the noise alone does that); the GPU is 5.5 dB below that floor, as at 512 (10 dB below), so "within the floor" is not claimed at any size.** At 512 the difference is in the fox's texture.

**A real limit, found by this work, and refused.** With conditions of **1024 x 1024 the GPU's picture has a dark green cast and no error is reported** (mean RGB 48, 48, 24 against torch's 90, 107, 116). It is **not** a quality problem of larger conditions (torch is right at that size; the GPU is right at 960). The step's attention keeps the scores of 32 heads for NP query rows over 2 NP keys (`dit_scores_mma`: `sc` is `f32[nh * nq, t]`): 64 NP^2 elements: **3.4e9 at NP 7,296 (right), 4.4e9 at NP 8,256 (wrong), against 2^32 = 4.29e9**. The reading is a 32-bit index in the kernels; **that is consistent with all four runs and was not proven** (no kernel was run on the matrix directly). The 768 and 512 runs have 1.4e9 and 0.3e9. The prefix pass has 32 NP^2 elements (2.2e9 at 8,256), under the limit. **Now refused before any GPU work:** the driver exits 2 when the estimate (conditions' tokens + 128 text rows, padded) or the real NP gives 64 NP^2 >= 2^32, and `media edit --also` refuses `--condition-area` over 921600. **The fix that would remove the limit, and halve the time (a step is 21 s at 960 where a step of 4,096 query rows over 11 k keys should be about 10 s): the step computes its queries for NP rows (the target's 4,096 padded up to the prefix's length) instead of the target's own 4,096, in `tests/dit_edit_lib.py`: not done.**

**Time (`media edit` with two pictures at 768 x 768, the default): 111 s for the command** (vision towers 5 + 4 s, language model 6 s, encoders 1 + 1 s, 8 steps and the prefix about 90 s, decoder 4 s). The estimate of section 3 (12 to 15 s a step at 1024^2) was wrong in both ways: the real step is 26 s and the picture is wrong there; at 768 it is 9 s.

**What this does not show:** one prompt, one seed, two generated pictures (never a user's photograph); more than two pictures is **not run**; the f16 tower per picture against the batched torch tower was not measured separately; the studio and its API take one picture (**not changed**); no float64 run of the whole pipeline.

**Commands**
```
scripts/media edit FIRST.png "Place the red fox sitting on the rocks in front of the lighthouse at dusk" --also SECOND.png -o out/combined --seed 3
```
