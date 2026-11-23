# Text to image from a free prompt, and what two condition images would need (design, 2026-11-16)

> **Status: built and measured for a free prompt (section 7); not yet in `media` or the studio (section 6 lists the order); two pictures are a plan (section 5).** Claims are tagged **[read]** (from the model's own pipeline file or this repository, not run), **[measured]** or **[estimate]**. This is slice T of the owner's request (2026-11-16): text to image, then combining two images. The first is built from the edit driver; the second only gets its plan (section 5).

## 1. What was read

`diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py` (the pipeline of `Qwen/Qwen-Image-2.1-Turbo`, in the venv) **[read]**:
* **One pipeline for both.** `image=None` is text to image; with `image` a **list** of pictures it is an edit. The prompt template of text to image is `<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n` (no picture, no vision tower); the hidden states are the last decoder layer's output **before** the final norm, from row `drop_idx` (the system message's tokens: 14) on. An empty prompt is replaced by one space.
* **The transformer's sequence** is the text rows, then one block of latent tokens for each condition picture, then the target's: `img_shapes = [[(1, h/16, w/16) for each condition picture] + [(1, H/16, W/16)]]`. With no picture it is `[[target]]`. `image_pad_mask` marks the picture slots; the target's are appended. **Nothing in the pipeline limits the number of condition pictures to one**: the text template repeats `<image1>...<image2>...` placeholders and the processor takes the list.
* **Sampling.** `sample_sigmas` from `model_index.json`, the scheduler's `mu` from the number of target tokens (`calculate_shift`), 8 steps in the repository's edit driver, **no classifier-free guidance** unless `true_cfg_scale > 1` and a negative prompt are given (the default is 1). With a condition the text and condition keys and values are step-independent (`causal_condition`) and cached after step 0.
* **Size.** `height`/`width` default to `output_resolution` (1024) and are rounded to a multiple of 32; a token covers 16 x 16 pixels, so 1024 x 1024 is 4,096 target tokens.

In this repository **[read]**: `tests/pipeline_gpu.py` (D6) runs text to image at **256 x 256 from D0's golden prompt** (its token ids, text embeddings and noise are files), which is why `media image` does not exist (`docs/cli.md`). `tests/edit_pipeline_gpu.py` (D9g) runs an **edit** of any photograph and prompt at about 1024 x 1024 in five stages (vision tower, language model reading the picture, VAE encoder, transformer with a prefix pass and cached steps, VAE decoder). `tools/dit_edit_ref.py` `assemble` builds the joint sequence for **one** condition picture (`block_len = [len(cond), ntgt]`, two block ids): its structure (a causal mask plus a bidirectional block for each picture, the rotary angles from `SHAPES`) is written for a list but not for more than one entry.

## 2. What text to image needs, from that

The edit driver **without** stages 1 and 3 (no vision tower, no VAE encoder) and with the rest narrowed:

| piece | for text to image | status |
|---|---|---|
| the prompt's ids | the processor with the text-only template (a `tools/text_img_ref.py` `build` without a picture) | to write: `build` takes a picture today |
| the text encoder on the GPU | `tests/text_img_gpu.py` with **no image rows** (no embeddings to scatter, no deepstack adds, positions 1-D), or `tests/text_gpu.py` (limited to 64 rows) | **to try**: text_img_gpu has only been run with a picture |
| the transformer | `der.assemble(prompt, pad=all False, cond=empty [0, 64], target, t)` with `SHAPES = [(GH, GW)]`: with a zero-length condition `np.repeat(np.arange(2), [0, ntgt])` gives the target one block id, the rotary angles take one shape, and the prefix is the text rows only | **to try**: never run with an empty condition |
| the session | `Edit(work, wdir, NP, NPR, NTG, ...)` with a prefix of about 20 rows (padded to 64) and 4,096 target rows: the same kernels | to try: never run with a prefix that short |
| the noise, sigmas, Euler update, the decoder | the edit driver's, unchanged | built |

**The driver:** `tests/t2i_pipeline_gpu.py WORK --prompt TEXT [--size WxH] [--seed N] [--steps N] --weights DIR --text-weights DIR [--out PNG]`, the edit driver's structure with those stages removed. It runs the real stages as scripts exactly as the edit driver does.

## 3. The checks, written before the first run

* **The text rows against torch.** The GPU encoder's rows for a free prompt against the pipeline's own `_get_qwen_prompt_embeds` (BF16 on MPS) for the same prompt: expected at the level D5 measured for the golden prompt (2.1% of the BF16 floor, `docs/diffusion.md` D5); **the bound is written now: 5% of the rms**, and a mutant that must fail: the system message's rows kept (`drop_idx` 0).
* **The image against torch.** The same prompt, size, seed and **noise** (the driver saves the noise tokens): the GPU's picture against the real pipeline's in BF16 on MPS, **PSNR**, with the **floor** measured as docs/wan-vae-encoder.md section 12 did (torch against itself under a small perturbation, here the text embeddings rounded differently): the GPU is held to **no further from torch than torch is from itself**; a pixel comparison of an 8-step run says "the same picture", not how exact the arithmetic is. **Looked at** besides: the picture is of what the prompt says.
* **Sizes.** 512 x 512 first (1,024 tokens), then 1024 x 1024 (4,096), the edit's size: **time and memory** are reported, not assumed.
* **Refusals.** An empty prompt (the pipeline's single space), a prompt over the encoder's row limit, a size that is not a multiple of 32, a missing weights directory: each exits non-zero with a reason, none reaches a crash.
* **A mutant** at the transformer: the target's rotary angles taken as for an edit (a condition block's shape in `SHAPES` that is not there) must change the picture beyond the floor.

## 4. What this will not show

One model, a handful of prompts, one machine; **no float64 reference of the whole run** (the transformer's alone takes hours: as for the edit); the host still does the tokenisation, the embedding lookup, the noise (torch's generator), the front and back ends of the transformer in float64, the Euler update and the PNG (as D9g); **text to image is not wired into `media` or the studio until the driver's checks pass**.

## 5. Two condition pictures (a plan, not built)

The model's pipeline takes a list **[read]**, so this is an extension of the edit driver, not of the model. Per stage, what changes for N pictures:
1. **Resize** each to the area `output_resolution`^2 at its own aspect (the pipeline does, per picture): sizes differ per picture.
2. **Vision tower** (`tests/vit_gpu.py`, one picture today): run once per picture (or batched with a block mask), each with its own grid.
3. **Language model** (`tests/text_img_gpu.py`): the template repeats `<imageK><|vision_start|><|image_pad|><|vision_end|>` for each; N sets of pad rows, **mRoPE positions per picture** (`tools/text_img_ref.py` `positions` takes one grid), the deepstack features scattered into the right rows.
4. **VAE encoder** (`tests/vae_enc_gpu.py`): once per picture.
5. **Transformer:** `der.assemble` with a list of conditions (`block_len`, `ids`, `SHAPES` one entry each); the prefix is the text and **N x 4,096** condition rows (at 1024^2 each), then the 4,096 target rows. **[estimate]** with two pictures the keys are about 12.4 k: the scores of a block are 32 x 4,096 x 12,400 x 4 B = 6.5 GB (4.4 GB with one), so the groups shrink again and a step costs about half as much more; **conditions at a smaller area** (the pipeline resizes to `output_resolution` squared, a parameter) would cut that.
6. **Whether the mask between condition blocks** is the one `assemble` writes (a later picture seeing an earlier one) is **to read in the transformer's code** (`causal_condition`) and held by the comparison with torch, not assumed.
The gates are the edit's: each stage against torch with the mutants that must fail, then the whole run against torch's picture within its floor. Its own document before any code.

## 6. Order

(1) This design is the PR's first commit. (2) The driver and its checks at 512 x 512. (3) 1024 x 1024 and the refusals. (4) Only then `media image` and the studio's `image` kind. (5) The two-picture plan gets its own document.

## 7. Results (measured, 2026-11-16)

**Built:** `tests/t2i_pipeline_gpu.py` (the edit driver without the vision tower and the VAE encoder). **Everything marked "to try" in section 2 worked the first time, with no change to any other file:** `tests/text_img_gpu.py` with no picture rows and an empty deepstack (`np.zeros((3, 0, 4096))`), `der.assemble` with an empty condition and `SHAPES = [(GH, GW)]`, `Edit` with a prefix of 21 to 26 rows.

**The prompt** "A red fox sitting in fresh snow in a pine forest at sunrise, soft light, photograph", seed 1: a photograph of a red fox sitting in snow before pine trees, **looked at: what the prompt says** (512 x 512 and 1024 x 1024).

| size | tokens | language model | prefix | 8 steps | decoder | all (without the one-time start) |
|---|---|---|---|---|---|---|
| 512 x 512 | 1,024 target, prefix 21 | 4 s | | 17 s (steps of about 1 s after the first) | 2 s | about 25 s |
| 1024 x 1024 | 4,096 target, prefix 26 | 2 s | | 66 s (steps of 7 s after the first, 18 s) | 4 s | about 75 s |

**Against torch's BF16 pipeline on MPS, the same prompt, size, seed and noise** (the real `QwenImage21Pipeline`, the noise tokens saved by the driver passed as `latents`), 8-bit RGB:
| | PSNR | mean abs (of 255) |
|---|---|---|
| 512: GPU against torch, torch given the GPU's text rows | 38.8 dB | 1.45 |
| 512: GPU against torch with its own text | **39.6 dB** | 1.20 |
| 512: **floor**: torch with its own text against torch with the GPU's text rows | 45.8 dB | 0.65 |
| 1024: GPU against torch with its own text | **26.9 dB** | 4.72 |
| 1024: GPU against torch, the GPU's text rows | 26.4 dB | |
| 1024: floor, as above | 36.0 dB | |
The same pictures, side by side and as a difference: composition, subject, pose and light are the same; the difference is texture. **This is a distance between two approximations, not an error of one:** `docs/diffusion.md` D7 measured at this size the GPU pipeline **48.5 dB from the float64 pipeline** and torch's BF16 **26.2 dB** from it (and the GPU against torch BF16 26.1 dB); 26.9 dB here is the same distance, torch's BF16 drift dominates. At 512 the 39.6 dB equals D9g's edit (39.1 dB). **The floor of text rounding (36 to 46 dB) is smaller than the GPU-to-torch distance: the GPU is not within it** and the claim is not made; the claim is the D7 one, that the GPU is the nearer to the exact picture. There is no float64 run of the whole pipeline here (the transformer's alone takes hours).

**Mutants refused** (512, against torch's picture): the prompt not read (text rows zero): **9.1 dB**; the text rows in reverse order: **19.5 dB**; the unmutated run is 39.6 dB.
**Refusals checked, each exits 2 with its reason and none reaches a crash:** no `--prompt`; a size that is not a multiple of 32; a size over 1024 x 1024 without `--force`; no `--weights`/`--text-weights`; a weights directory that does not exist; `--steps 0`; a bad `--mutate`; no work directory.

**Not shown:** more than one prompt and seed; the text rows against torch's `_get_qwen_prompt_embeds` as a separate number (the check of section 3 was done on the picture, not on the rows); an empty prompt (the driver sends one space as the pipeline does, not run); a prompt over 512 tokens (refused, not run); the time includes neither the start of the interpreter nor the first read of the 28 GB of weights; text rendering and long prompts were not tried.

## 8. The prefix pass on the text's own rows (measured, 2026-11-23)

**What section 7 ran was not what section 2 planned.** Section 2 has "a prefix of about 20 rows (padded to 64)"; the driver gave `Edit` one padding for both, `NP = max(pad64(NPR), pad64(NTG))`, so the 26 text rows went through the 32 blocks as 4,096 rows at 1024 x 1024, and a step, whose target is not shorter than `NP`, concatenated its keys to `2 NP` = 8,192 (4,070 of the first half masked).

**Built:** `Edit(..., npp=N)` (`tests/dit_edit_lib.py`): the rows of the prefix pass and of the cached keys and values, a multiple of 64 from `NPR` to `NP`; `NP` when it is not given, which is every caller but `tests/t2i_pipeline_gpu.py`, which passes `pad64(NPR)`. With `npp < NP` a step is the two-segment step of `docs/two-images.md` section 8 (`dit_scores_mma` twice, `dit_stats_o2`, `dit_pv_mma2`) with `t1 = npp` cached keys and `t2 = NS` of its own, whatever `NS` is; the masks are the same masks cut to those columns. Where that step cannot be used (`--unified`, `DIT_STATS` other than `online`, `DIT_STEP=padded`) `npp` is not used and the shapes are the old ones. An `npp` that is not such a multiple is refused (`edit: the prefix pass's rows ...`).

The prompt and seed of section 7, `scripts/media image ... --size WxH --seed 1` (`--resident`), wall time of the whole command, M4 Max, another process running CPU tests beside it:

| size | before | after | step 0 before, after | steps 1 to 7 | sessions (prefix + steps) |
|---|---|---|---|---|---|
| 1024 x 1024, two runs each | 77.1, 78.0 s | **61.1, 62.9 s** | 18 s (9 in the sessions), 10 s (8) | 7 s each, 6 s each | 67 and 66 s, 52 and 52 s |
| 512 x 512, one run each | 28.9 s | **22.7 s** | 9 s (3), 5 s (3) | 1 s each | 18 s, 14 s |

**The pictures are the same bytes:** the PNG after against the PNG before, 512 x 512 and 1024 x 1024 (both runs): maximum difference 0 of 255, 0 pixels differ. So is 512 x 512 without `--resident` (sessions of 4 and 2 blocks). **Two savings, not one:** the prefix pass (64 rows, not 4,096), and each step's attention over 4,160 keys, not 8,192, which is what takes a later step from 7 s to 6 s (the driver prints whole seconds).

**The edit driver is not touched:** it does not pass `npp`. Checked without a GPU: every session file and static table `Edit` writes (the 32 blocks' programs, links, aliases, masks, rotary tables, selectors), hashed, for prefixes of 4,130 rows over targets of 1,024 and 4,096 rows and 1,050 over 4,096, plain, `--resident` and `--unified`, is the same before and after the change (and with `DIT_STEP=padded`, with `npp` given, the same as before too).

**Not shown:** a full edit run after the change (none was run); `tests/dit_edit_gpu.py`'s comparison with float64 on the two-segment step with a short `t1` (the pictures being the same bytes as the concatenated step's is the evidence here); sizes other than these two; the times with nothing else running on the machine.

