# Image and video diffusion models in `.chx`

> **Status (D6, 2026-11-13): Qwen-Image-2.1-Turbo generates a 256 x 256 image on a Metal GPU in cancho-gpu from its prompt and seed, text encoder, eight denoising steps and VAE decoder, with the host still doing the tokenisation,
> the initial noise, the embedding lookup and some layout (§4, "D6 results"). The same run at 1024 x 1024 works ("D7") and, with the attention on the matrix unit ("D8"), takes 1.34 times torch's time on MPS (55 s against 41 s); image editing (the vision tower) is not done and video (§6) is not started.**
> The first target is an image model, Qwen-Image-2.1-Turbo (§2); video comes second (§6).
> Every number below is either read from the model's files (the command is given), read from this
> repository's own documents (named, not re-measured here), or marked **choice** or **open**.

## 1. Why this, and why in this order

cancho-gpu runs language models on a Metal GPU (`README.md`, `docs/qwen35.md`) and trains small
networks on it (`docs/resident.md`). The question of 2026-11-13 was whether it could also run the
image and video generators that were tried on this machine with PyTorch (MPS). The answer is *in
principle yes, and today no*: §4 lists what is missing.

Images come first because the image transformer is the same family of layers the language
models already use (RMSNorm, per-head QK norm, RoPE, SwiGLU, grouped projections), with one
new piece of structure (modulation) and one new size of problem (attention over thousands of
positions). Video adds a 3D autoencoder, an audio autoencoder and models four to five times
larger (§6); doing it before the image path is checked would test two unknowns at once.

The reference is **PyTorch with diffusers on MPS**, the implementation the first edits were made
with (Qwen-Image-2.1-Turbo: 8 steps, 33 s for a 576 x 1024 edit on this machine, measured by the
run's own log; those runs and their inputs are outside this repository and not committed). What a kernel computes is held to it, as `CLAUDE.md` requires for every numeric
kernel, with goldens written once by a Python tool (the same rule as the GGUF, sampler and
tokenizer goldens: Python is the oracle's author, never part of the run).

## 2. The model (measured)

`Qwen/Qwen-Image-2.1-Turbo`, 8 denoising steps, CFG 1, five parts (`model_index.json`).
Parameter counts are the sum of the tensor shapes in the safetensors headers of the
checkpoint; every tensor is BF16:

| Part | Parameters | Bytes (BF16) | What it is |
|---|---:|---:|---|
| `transformer` | 7.12 B (297 tensors) | 14.2 GB | `QwenImage21Transformer2DModel`: 32 layers, 32 heads of 128, hidden 4096, MLP ratio 3, patch size 1, 64 in/out channels, RoPE axes (16, 56, 56) |
| `text_encoder` | 8.77 B (750 tensors) | 17.5 GB | `Qwen3VLForConditionalGeneration`: text 36 layers, hidden 4096, 32 heads, 8 KV heads, head 128, MLP 12288, rope theta 5e6 with interleaved mRoPE (24, 20, 20); vision tower depth 27, hidden 1152, 16 heads, patch 16 |
| `vae` | 0.34 B (238 tensors) | 0.7 GB | `AutoencoderKLQwenImage21`: base dim 96, decoder base dim 144, dim multipliers (1, 2, 4, 8, 8), 88 rank-4 (2-D convolution) weights |
| `processor`, `scheduler` | none | none | tokenizer and image preprocessing; the 8 sigmas of the saved schedule |

The three weight parts sum to 32.4 GB, which is the size of the download (32.5 GB). They fit this
machine's 64 GB at BF16 with room for activations. **Choice:** the first target keeps BF16 (or
f16, as the language models do, `docs/inference.md`); no quantisation until the BF16 path agrees
with torch.

One block of the transformer (`transformer_blocks.N.*`), from the tensor names and shapes:
`attn.to_q`, `to_k`, `to_v`, `to_out.0` (each 4096 x 4096, no fusing in the file), `attn.norm_q`
and `norm_k` (128, per-head RMSNorm), `img_mlp.proj` and `gate_layer` (12288 x 4096) and
`out` (4096 x 12288), i.e. a gated MLP. Outside the blocks: `img_in` (64 -> 4096), `txt_in`
(an MLP, 4096 -> 4096 -> 4096, with a norm), `modulation.1` (4096 -> 16384, the timestep
embedding's projection), `norm_out`, `proj_out` (4096 -> 64). `causal_condition: true` is in the
config: **open** what it masks (§5, question 2).

## 3. What a denoising step needs, and what exists

"Exists" is read from the code (the `.chx` kernels in `tests/chx/`, `src/`) and from this repository's documents; the first
version of this section took the README's word and was **wrong in three places**, corrected here after reading the kernels.
It is still not re-run.

| Need | In cancho-gpu |
|---|---|
| RMSNorm, per-head QK norm, SwiGLU MLP, RoPE, f16 matmul on Metal | yes (`qk_norm.chx`, `rmsnorm_rows.chx`, `swiglu.chx`, `rope.chx`, the `matvec_*` family) |
| BF16 weights | **no dtype**: "the language has f16, f32 and i8 only" (`src/qwen08_load.cho`). A BF16 weight is widened to f32 on load (exact) or narrowed to f16 (not exact: BF16 has f32's range and f16 does not). For this model f32 doubles the transformer to 28.5 GB; f16 keeps 14.2 GB but must be checked for overflow against D0's goldens |
| Reading `.safetensors` | yes, in cancho: `src/safetensors.cho` and the Qwen2.5-0.5B loader. (`convert.py` is the 27B's route. Not a gap) |
| Attention over many positions | **partly, and not the part this model needs.** Every attention kernel is a *decode* step: the `g` query heads that share one key/value head against that head's cache. `attention.chx` (d = 256) and `attention_d64.chx` bound the scores tile by the 32 KiB of threadgroup memory; `attention_scores`, `attention_stats` and `attention_pvl` (online softmax over chunks of `cs` positions, "no ceiling on `t` but the cache") remove that bound for the keys. The README's "split-KV is not built" is out of date for them. A transformer of diffusion needs **thousands of queries at once**, so the queries must be tiled too; no kernel does that |
| Head dimension 128 | **no kernel**: the scale 1/sqrt(d) is a literal in each file (0.0625 for 256, 0.125 for 64) because "the surface has no scalar parameter"; d = 128 is a new file (1/sqrt(128)) |
| The attention mask | **block-causal, not bidirectional** (read from `transformer_qwenimage21.py`, `build_qwenimage21_block_causal_mask`): a position attends to every earlier position of the joint text-and-image sequence, and every image attends to all of its own tokens; padded prompt positions are not keys. The decode kernels take an additive mask, which can say that for one query row; a tiled kernel needs it per tile |
| Adaptive modulation (scale and gate from the timestep, per token) | **built in D2b**: `dit_lnmod.chx` and `dit_gate_res.chx` |
| 3 x 3 convolution | yes, as nine shifted matmuls of a padded NHWC matrix, in the CIFAR-10 CNN (852,000 parameters, `docs/resident.md` §18); "only small tiles fit" in 32 KiB. The VAE's channels (up to 8 x 96 = 768 encoder, 8 x 144 = 1152 decoder, from `vae/config.json`) at image resolution are far from that network. `conv_step.chx` is the 4-tap depthwise convolution of a gated-delta layer, not this |
| VAE decoder (residual blocks, up-sampling, the latent mean and std) | no |
| Vision tower (27-layer ViT, patch 16, mRoPE) for the text encoder | no |
| Flow-matching Euler step with the saved sigmas | no (a few lines, but it is not written) |

## 4. The plan: stages, each with a gate that can fail

Each stage ends when its gate passes; none starts before the one above it. A gate compares the
`.chx` kernels on the Metal GPU with the reference's tensors at the stage's output.

| Stage | What | Gate (written before the code) |
|---|---|---|
| D0 | **Built** (`tools/diffusion_golden.py`). Runs the reference on a fixed small input (256 x 256, one prompt, seed 1, 8 steps, `use_kv_cache=False`) and writes golden tensors: the text encoder's output, block 0's input and output, one full transformer step, the latents after each step, the VAE's input and output, the image. | Two runs give the same bytes. **Measured 2026-11-13 (torch 2.15.0, MPS, BF16): they do, for all 15 tensors** (`--check`). |
| D1 | **Built** (below). Block-causal attention with head dimension 128 for N queries and N keys (N from a few hundred up to 4,096 image tokens plus the prompt, §5, question 1): tiled over both, online softmax (the chunked kernels of §3 tile the keys only). | Against a float64 reference written from the mathematics (`tests/kernels.py` style), including the mutants that must fail it; then against D0's block goldens. |
| D2 | One transformer block, then the 32-layer step. Split: **D2a** the tolerance and a float64 block (built, below); **D2b** (built, below) each piece as a kernel against that reference (layer norm with modulation, per-head RMSNorm, 3-axis RoPE, the D1 attention, tanh-gated residual, SwiGLU, f16 matmuls with 4096 x 4096 and 12288 x 4096 weights); **D2c** (built, below) the block on the Metal GPU; **D2d** (built, below, with the front and back ends on the host) 32 blocks and the step; **D2e** (built, below) the front and back ends on the GPU. | One block: max abs error against D0 below the tolerance D2a measured. One step: the same. |
| D3 | **Built** (below). Eight steps with the saved schedule, from a fixed noise. | The latent after 8 steps against torch's, with the error reported per step. |
| D4 | The VAE decoder. Split: **D4a** a float64 decoder and the cost (built, below); **D4b** (built, below) its pieces as kernels, each against it; **D4c** (built, below) the decoder on the GPU. | The decoded image against D0's, in max abs error and PSNR. |
| D5 | **Built for text-to-image** (below; the vision tower, needed for editing, is not). The text encoder. | Hidden states against D0's. |
| D6 | **Built, and timed at 256 x 256 (D6b) and run at 1024 x 1024 (D7)** (below). End to end: prompt to PNG on the GPU, timed against torch MPS on the same machine, same seed. | A text-to-image run whose image matches torch's within the D3 and D4 tolerances. The time is a measurement, reported with the cases where cancho-gpu is slower, as `README.md` does for training. |

Editing (the beach, clothes and Ghibli tests in `imgtest/`) needs the vision tower and the
reference-image path; **choice:** text-to-image first, editing after D5's vision tower.

**Test images.** D0 to D3 and D6's text-to-image run need **no input photo**: a prompt and a seed. The editing
stages need one reference image. **Choice:** it will be a photograph with a licence that allows redistribution (public
domain or CC0, from Wikimedia Commons), added to `tests/` with its source URL, author and licence in a file beside it, and
checked in size. No photograph of a person taken by or sent by a user of this project goes into the repository.

### D6 results (measured): a text-to-image run on the Metal GPU

`python tests/pipeline_gpu.py GOLDEN_DIR WORK_DIR` runs the pipeline for D0's prompt ("A red bicycle leaning against a white wall, soft morning light."), seed 1, 256 x 256, 8 steps, as **three sessions** (the text encoder of D5, 792 programs; the eight steps of D3, 659; the VAE decoder of D4c, 161), passing the
text embeddings, the final latents and the layout through files on the host, and compares it with **the float64 pipeline** (the same three stages in float64 from the *exact* encoder's embeddings: `tools/text_ref.py`, `tools/dit_traj_ref.py --text`, `tools/vae_ref.py`) and with torch's BF16 one (D0).
**What the GPU does:** the 36-layer text encoder, the 32-block transformer for eight steps with the Euler update, the VAE decoder. **What the host still does:** the tokenisation (D0's 35 token ids), the embedding lookup, the initial noise (torch's seeded random numbers, D0's `step0_in`; cancho-gpu does not reproduce torch's generator),
the rotary and timestep tables, laying the final latents out for the decoder (undoing their mean and std; checked against D0's `vae_in` to 0.097, one BF16 step at that size), the weights as f16 files, the clamp and the 8-bit PNG.

| Final latents [256, 64] (rms 1.22) | Mean abs error | Max abs error |
|---|---|---|
| GPU pipeline vs float64 pipeline | **2.0e-3** | 0.077 |
| torch BF16 vs float64 pipeline | 2.2e-2 | 0.65 |
| GPU pipeline vs torch BF16 | 2.4e-2 | 0.73 |

| The 256 x 256 image, 8-bit RGB | Mean abs difference (of 255) | PSNR |
|---|---|---|
| **GPU pipeline vs float64 pipeline** | **0.49** | **43.2 dB** |
| torch BF16 vs float64 pipeline | 4.18 | 26.6 dB |
| GPU pipeline vs torch BF16 | 4.40 | 26.1 dB |

* **The GPU's image is about nine times closer to the exact image of the prompt than torch's BF16 image is** (mean difference 0.49 against 4.18 of 255; 43 dB against 27 dB), because the errors the earlier stages measured add up the same way: torch's text embeddings are 2.1% off (D5), its eight steps drift 1.1e-2
  (D3), its decoder adds 2.6e-3 (D4). The three images look the same (side by side: torch, float64, GPU).
* The numbers are one prompt, one seed and 256 x 256; **it is not a claim about the model's quality, nor a speed**: each session's wall time here is dominated by writing and uploading 14 to 16 GB of f16 weights (text 14 s, the steps 20 s, the decoder 7 s, cold, with the host's Python work), and **nothing is timed against torch MPS**, which D6's row in §4 asked for and is not done.
* **Not done:** timing; 1024 x 1024 (the attention scratch, 14x the work in the products, 1.7 x 10^13 FLOPs in the decoder: D2d and D4a give the sizes, none has run); editing, which needs the vision tower (D5); the host's list above, of which the tokenizer and the noise are the ones that stop it being "prompt and seed in, image out" with no host arithmetic.

### D8 results (measured): the attention on the matrix unit

The profile of D7 said the attention was 66% of a step (9.8 s of 14.8 s at 1024 x 1024), `dit_scores` on the generic f32 product at 0.64 TFLOPS, so four kernels (`tests/chx/dit_rope16.chx`, `dit_scores_mma.chx`, `dit_stats_m.chx`, `dit_pv_mma.chx`) put it on the matrix unit in f16, held by `tests/dit_attention_mma.py` to two binary64 references,
**every element** (the gate was written first and failed with "cannot read dit_rope16.chx"): one with the f16 roundings the kernels make (q times the scale after the rotation, k, v and the weights, tolerance 1e-4: **6.1e-5 and 2.3e-5** measured) and the exact attention (tolerance 5e-3: **4.1e-4 and 3.8e-4**); **nine mutants, all killed** (one was refused at first for a wrong text, fixed).
The accumulator of a matrix-unit product cannot be touched after it (no scaling, no mask), so the scale is folded into q (rounded to f16 with it) and the mask is added where the scores are read, in `dit_stats_m` and `dit_pv_mma`; the weights `exp(s + mask - m) / z` are formed in f32, rounded to f16 (`stage`) and multiplied by f16 v with `mma_nn`. The scores still go to an `[nh * nq, t]` f32 scratch (2.2 GB a block): an
attention that never stores them needs the softmax's running state carried through the loop, and a loop carries one tile and a matrix-unit accumulator cannot be rescaled: **not expressible in this language today**. The Metal schedule is bm 64, bn 64, bk 32; the CUDA one for `dit_pv_mma` first needed 68,608 B of static threadgroup memory against 49,152 (the golden was a refusal, which I read this time) and is now the same 64 x 64 x 32.

| Kernel alone, 4,160 rows, one block | before (D7) | on the matrix unit |
|---|---|---|
| scores | `dit_scores` 223 ms, 0.64 TFLOPS | `dit_scores_mma` **11.5 ms, 12.3 TFLOPS** |
| statistics | `dit_stats` 9.5 ms | `dit_stats_m` 17.9 ms (reads the mask) |
| weighted sum | `dit_pv` 73 ms, 1.95 TFLOPS | `dit_pv_mma` **24.9 ms, 5.7 TFLOPS** |
| the whole attention of a block (with the rotations to f16 and the cast of v) | 305 ms | **55 ms** |

| One transformer step at 1024 x 1024 (steady state, second request of each session; two runs gave the same) | torch MPS | before | **now** |
|---|---|---|---|
| step | 4.99 s | 14.78 s (2.96x) | **6.67 s (1.34x)**: front 0.03 s + 8 groups of 4 blocks at 0.83 s + back 0.002 s |
| the attention's part | | 9.8 s | 1.8 s |
| the three stages (text 0.19 s, eight steps, VAE 1.88 s) | 41.4 s | about 120 s | **about 55 s (1.34x)** |

The eight steps took 89 s of wall time with the ten sessions of each step starting and the host's files (155 s before). **cancho-gpu is now 1.34 times torch's time on MPS at 1024 x 1024**, and its step is 2.2 times faster than D7's. The step is now matrix products 4.6 s (69%), attention 1.8 s (27%), small kernels 0.25 s (4%). Not measured at 256 x 256 with the new attention (D6b's 0.497 s used the f32 one).

**What it cost in accuracy: nothing measurable.** At 256 x 256 the final latents are 1.39e-3 from the float64 trajectory (1.25e-3 with the f32 attention; torch 1.12e-2). At 1024 x 1024: final latents 1.37e-3 (1.73e-3 before; torch 4.0e-2), and the image **0.16 of 255 from the float64 pipeline, 49.3 dB** (0.18 and 48.5 dB before; torch 4.75, 26.2 dB).

**Tile schedules tried for `gemm_mma` at 4,160 x 4,096 x 4,096 (the products are 69% of the step): none beats the present one** (bm 64, bn 64, bk 32, 128 threads: 13.0 TFLOPS; bk 64, or 64 x 128, or 32 x 128: 12.4; 128 x 64 and 128 x 32: 12.5 on 4,224 rows; 128 x 128: 0.9 to 8.2 TFLOPS, register pressure). What is left to gain is in a different product kernel (double buffering is not expressible either), in the statistics (0.57 s a step; it re-reads the mask) and the weighted sum (0.8 s), and in the decoder (1.88 s against torch's 1.29 s).

**D8b: two schedules, measured (4,160 rows, one block, M4 Max).** `dit_pv_mma` read each score tile twice and formed its exponentials twice because `bn` 64 splits the 128 head columns in two; with one tile across them (`threads 256 warps 8 1 bm 64 bn 128 bk 32`) it takes **15.0 ms, from 24.9** (other shapes: 4 x 2 or 2 x 4 warps 16.0; `warps 2 2` or 4 x 1 at 128 threads 107 to 130 ms; bk 16 18.6; bk 64 and bm 128 are refused). `dit_stats_m` with `threads 64` instead of 256 takes **about 14.5 ms, from 17.2** (qs and cs variations are within the noise of about 1 ms; 13 to 14 ms at best). Step at 1024 x 1024: **6.25 s, from 6.67** (front 0.027 + 8 groups x 0.778 + back 0.003), so the pipeline is about 52 s, 1.26 times torch's 41 s (that total is the step sum plus D8's text and VAE times, not re-run end to end). Accuracy: the golden cases agree (420 of 420, four re-recorded because the apple-m-series lowering changed), the nine mutants are still killed, the GPU matches the interpreter (3e-7), and step 0 at 1024 x 1024 is 1.17e-3 from torch's BF16 (mean abs), as before. The NVIDIA schedules are unchanged and still not run.

### D7 results (measured): 1024 x 1024 (with the attention of D1; D8 above replaces its timing)

**The image** (cancho-gpu on the right; torch BF16 on the left; the prompt and seed of D0): `scratchpad/compare1024.png` is not in the repository (images are not committed). The scene is the same (a red bicycle with a basket against a cream wall, the shadow of a leafy branch, the kerb and a yellow line), and the two differ in small details (the kickstand).

**Why many sessions.** A session gives each program's outputs a buffer and keeps them: 32 blocks hold 32 attention score matrices of 32 x 4160 x 4160 x 4 B = 2.2 GB each (71 GB at 1024 x 1024), and a `table` (the way to read a different weight per repetition) is capped at the Metal session's 512 MiB byte pool, against 33.5 to 100 MB a layer. `tests/dit_big.py` therefore runs a
step as a **front** session, **8 group sessions of 4 blocks** and a **back** session, passing the residual stream, the modulation and the latents through binary files (`save`, `checkpoint`, `resident`). Its kernels are D3's; **run at 256 x 256 it reproduces D3's numbers to every digit shown** (the single-session run: step 7 against float64 1.25e-3), which is what says the splitting is right.
The VAE decoder fits in one session at 1024 x 1024 (161 programs). The 8-step run took 155 s of wall time in all (host work, ten sessions a step and their starts included); the weights, 14.3 GB, are written once (10 s) and each group uploads its own 1.75 GB.

**Against the float64 pipeline at 1024 x 1024** (`tools/dit_traj_ref.py`, 160 s a step, from the exact encoder's embeddings; the GPU pipeline here starts from the GPU encoder's):

| Latents [4096, 64], mean abs error against float64 | step 0 | step 3 | step 5 | step 6 | **step 7 (final)** |
|---|---|---|---|---|---|
| **GPU** | 1.0e-5 | 9.2e-5 | 3.3e-4 | 7.8e-4 | **1.7e-3** (max 0.20) |
| torch BF16 | 1.2e-3 | 3.8e-3 | 1.0e-2 | 2.1e-2 | **4.0e-2** (max 1.6) |

| The 1024 x 1024 image, 8-bit RGB | Mean abs difference (of 255) | PSNR |
|---|---|---|
| **GPU pipeline vs float64 pipeline** | **0.18** | **48.5 dB** |
| torch BF16 vs float64 pipeline | 4.75 | 26.2 dB |
| GPU pipeline vs torch BF16 | 4.82 | 26.1 dB |

* **The GPU's image is 26 times closer to the exact image of the prompt than torch's BF16 image** (mean difference 0.18 against 4.75 of 255; 48.5 dB against 26.2 dB), and its final latents 23 times closer. At 256 x 256 the factors were 9 and 10: torch's BF16 error grows with the number of tokens, the GPU's in f16 with f32 accumulation grows much less.

**Time** (each torch stage with `tools/diffusion_time.py --size 1024`, three timed runs; cancho-gpu with the **second** request of each session, the weights on the device, nothing else running on the machine; two runs of the whole step gave 14.78 s both times):

| Stage | torch MPS, BF16 | cancho-gpu | cancho-gpu / torch |
|---|---|---|---|
| text encoder | 0.171 s | 0.190 s (35 tokens: independent of the image size) | 1.11 |
| one transformer step | 4.99 s | **14.8 s** (front 0.03 s + 8 groups of 4 blocks at 1.844 s + back 0.002 s) | **2.96** |
| eight steps | 39.9 s | 118 s | 2.96 |
| VAE decode | 1.29 s | 1.88 s | 1.46 |
| **the three stages** | **41.4 s** | **about 120 s** | **about 2.9** |

The wall time of the eight steps, with the ten sessions of each step starting and the host's files, was 155 s (about 19 s a step): the sessions' starts and the files cost about 4.5 s a step on top of the 14.8 s.

**A correction.** The first version of this section gave 18.7 to 20.3 s a step and blamed a first measurement on the float64 job running beside it. **Both were wrong:** `tests/dit_big.py --time` took the *first* request of each session, which includes the session's own start (the device, the compile, the upload), as the steady state (an index off by one: `r[-3]` for `r[-2]`). It was
found because the kernels measured alone do not leave a gap: one block is 143 ms of matrix products + 305 ms of attention + 8.5 ms of the small kernels = 457 ms, a group of four 1.83 s, and the second request of a group is 1.844 s. The 256 x 256 timings of D6b were taken with `tests/session_time.py`, which reports the first request apart and the median of the others, and are not affected.

**cancho-gpu is about 2.9 times slower than torch on MPS at 1024 x 1024**, against 1.30 times at 256 x 256: the gap is the attention. The kernels alone at 4,160 rows (`tests/time_gemm.py 4160`, one block, times 32) add up to the step: the matrix products run at 12.1 to 13.0 TFLOPS (torch's whole step is 13.3 TFLOPS over its useful 66.5 TFLOP) and take **4.6 s (31%)**; the attention takes **9.8 s (66%)**:
`dit_scores` 223 ms a block at 0.64 TFLOPS (7.1 s in all), `dit_pv` 73 ms at 1.95 TFLOPS (2.3 s), `dit_stats` 9.5 ms (0.3 s); the small kernels (norms, RoPE, residuals, casts, SwiGLU) **0.27 s (2%)**, timed one by one (`rmsnorm_rows` per head is the largest, 2.1 ms a call). The chained attention in one session takes 305 ms, the sum of its parts. The data does not matter (random inputs gave the same times as zeros).
The matrix products alone would put cancho-gpu level with torch; **the attention is the thing to rewrite**: the scores on the matrix unit in f16 (they run on the generic f32 product, 0.64 TFLOPS against 12 to 13 for `mma`), and `dit_pv` likewise, which would take the attention from 9.8 s to a few seconds.

* **Not done:** that attention, image editing (the vision tower), the tokenizer and the noise on the device, and 1024 x 1024 in one session (it needs buffers given back, or an attention without the 2.2 GB of scores a block).

### D6b results (measured): the time of each stage against torch on MPS, 256 x 256

Both on this machine (Apple M4 Max, 64 GB), one stage at a time, the weights resident, the same prompt. torch: `python tools/diffusion_time.py` (diffusers, BF16, MPS, GPU synchronised around each stage's forward calls, one warm-up and seven timed pipeline runs, spreads under 1%). cancho-gpu:
`tests/text_gpu.py`, `tests/dit_loop_gpu.py` and `tests/vae_gpu.py` with `--time N` (the session is started once and sent N requests one at a time, `tests/session_time.py`; the answer is a probe of two rows so the size of the answer is not timed; 11 to 16 requests, spreads under 1%).
`tests/time_gemm.py` times kernels alone (a request of N repetitions in one command buffer, the difference between N = 200 and N = 1).

| Stage | torch MPS, BF16 | cancho-gpu, steady state | cancho-gpu / torch | cancho-gpu, first request (device, compile, upload) |
|---|---|---|---|---|
| text encoder (35 tokens; 64 padded rows on cancho-gpu) | 0.153 s | 0.190 s | 1.24 | 2.68 s |
| one transformer step (277 tokens; 320 padded rows on cancho-gpu) | 0.385 s | 0.497 s | 1.29 | 3.72 s |
| eight steps | 3.08 s | 3.98 s | 1.29 | |
| VAE decode (256 x 256) | 0.087 s | 0.132 s | 1.52 | 0.34 s |
| **the three stages** | **3.32 s** | **4.30 s** | **1.30** | 6.7 s |

**cancho-gpu is 30% slower than torch on MPS at 256 x 256** (the encoder 24%, the transformer 29%, the decoder 52%), which is the answer to D6's timing row; it is not ahead of it anywhere here. The torch pipeline call as a whole took 3.33 s (the stages' 3.32 s plus the scheduler, tokenisation and post-processing).

**Where the transformer step's 497 ms goes** (kernels alone, `tests/time_gemm.py`): the matrix products are **432 ms (87%)**: `gemm_mma.chx` runs the 4096 x 4096 products at 9.3 TFLOPS, the 12288-wide ones at 12.2 and 9.0 TFLOPS; the attention is **61 ms (12%)**: `dit_scores` takes 1.43 ms a block at
0.59 TFLOPS (the slowest kernel by far per FLOP), `dit_pv` 0.42 ms at 2.0 TFLOPS, `dit_stats` 0.06 ms; the rest (norms, RoPE, residuals, casts) is the last ~1%.

* **How much of the 29% is padding:** cancho-gpu runs 320 rows where torch runs 277, which is 15.5% more work in every product and in the attention. Divided out, the step would be about 430 ms against torch's 385: **about 12% slower per token**, which is the matrix products' rate (9 to 12 TFLOPS against torch's roughly 10 on the useful 3.96 TFLOP) and the scores kernel. The padding is the
  matrix unit's tile (64) and a choice of this slice, not of the hardware: a tile of 8 or 16 rows on the last block of a product is not written.
* **What the comparison is not:** one machine, one prompt, one size; a cancho-gpu step uses f16 operands with f32 accumulation and f32 intermediates against torch's BF16 throughout (different bytes moved); torch's attention is a fused kernel and cancho-gpu's materialises the scores (a 32 x 320 x 320 f32 scratch here, 2.1 GB per layer at 1024 x 1024). The
  cold first request (device, compile and upload of 15 GB) is not what torch's number includes (its model is already loaded).
* **Where the next gain is, from the profile:** the scores kernel (about 9% of the step, 0.59 TFLOPS against 9 to 12 for the products: a fused attention would remove it and the scratch), then the padding; the products themselves are within the range of torch's overall rate.

### D5 results (measured): the text encoder on the Metal GPU, with no new kernel

`python tools/text_ref.py GOLDEN_DIR [OUTDIR] [--torch32]` is the language model of Qwen3-VL (36 layers) in float64, written from the mathematics of its text model (the structure is in the tool's docstring), and `tests/text_gpu.py GOLDEN_DIR REF_DIR` runs it on the M4 Max as **one session of 792
programs**. D0 now also saves the templated prompt's token ids (**35 tokens**, 14 of them the chat template's system part, which the pipeline drops: 21 are kept) and `drop_idx`. Two facts read from `_get_qwen_prompt_embeds`: the pipeline takes the residual stream after the 36 layers **before the final RMSNorm** (it hooks `norm` to
return its input), and it keeps rows from `drop_idx` on.

| Comparison (the 21 kept rows x 4096, rms 12.1) | Mean abs error | Max abs error |
|---|---|---|
| float64 encoder vs **torch float32 on the CPU** (the same model) | **4.5e-5** | **5.3e-3** |
| torch float32 vs torch BF16 on MPS (D0's `text_embeds`) | 0.255 (2.1% of the rms) | 188 |
| **GPU encoder vs float64 encoder** | **5.4e-3 (4.5e-4 of the rms)** | **1.5** |
| GPU encoder vs torch BF16 | 0.255 | 189 |

* **The float64 encoder is the model** (it agrees with torch's float32 to 4.5e-5), and **torch's BF16 text encoder is 2.1% off the exact one on average, and by up to 188 at one element** (row 0, channel 1838: exact -100.4, torch +88.0, in a stream whose largest value is 13,518: the massive activations of the
  chat template's first tokens). **The GPU's encoder is about 47 times closer to the exact one than torch's BF16 run (mean).** Consequence for D6: torch's BF16 image is *not* the exact image of its prompt, since every later stage starts from these embeddings; **the GPU's image has to be compared with the float64 pipeline's, not only with torch's**.
* **No new kernel was needed:** the encoder is the DiT block with other constants. The rotary embedding of the text model is the rotate-half form with per-position tables (theta 5e6; mRoPE with the same position on its three axes is the ordinary one), so the DiT's channel permutation is not needed;
  grouped attention (32 query heads, 8 key/value heads) is `dit_scores` / `dit_stats` / `dit_pv` with the key and value projections' rows repeated four times (50 MB of f16 a layer); the causal mask is an input. The embedding lookup (35 rows of 151,936 x 4096), the rotary tables and the weights are the host's.
* **f16 range, for this prompt:** the largest values over the 36 layers, in float64: 165 (the first norm's output), 284 (the second's), 31 (attention), **6,409** (SwiGLU, the down projection's input), against f16's 65,504; the residual, in f32, peaks at 13,518. No overflow.
* **Time:** 2.7 s for the session with the compile of 792 programs and the upload of 15.7 GB (a cold start; the weight files, written in 10 s, were in the page cache), 35 tokens padded to 64: **no speed is claimed.**
* **Not done:** the vision tower (27 layers, patch 16, mRoPE, the deepstack mergers) that editing needs; the tokenizer on the device (the ids are D0's; cancho-gpu has a tokenizer that is checked against Hugging Face's for Qwen2.5, not run on this one); the whole pipeline end to end (D6).

### D4c results (measured): the VAE decoder on the Metal GPU

`tests/vae_gpu.py GOLDEN_DIR` (a Mac, the checkpoint and D0's goldens; not in `scripts/gate.sh`) runs the decoder of D0's 16 x 16 latent (`vae_in`) as **one session of 161 programs**: the pieces of D4b, the existing `conv3.chx` (64, 288, 576 and 1152 channels), `gemm_mma.chx` (the attention's scores),
`dit_stats.chx` and `dit_pv.chx` (one head of 1152 over the 324 bordered pixels, 384 rows) and one more kernel, `vae_norm` (the attention block's norm without the SiLU; in `tests/vae_pieces.py` with two mutants, **fourteen mutants in all, every one killed**). The shortcut's selection
matrices are built from DupUp3D's definition (output channel `o`, sub-position `(a, b)` reads input channel `((o ft + ft - 1) 4 + 2a + b) // reps`) and the first run agreed with the float64 decoder, which is the check that the rearrangement is right.

| Comparison (output [4, 256, 256], clamped, rms 0.822) | Mean abs error | Max abs error |
|---|---|---|
| **GPU decoder vs float64 decoder** | **2.5e-4 (3.0e-4 of the rms)** | **0.047** |
| GPU decoder vs torch BF16 on MPS (D0's `vae_out`) | 2.6e-3 | 0.39 |
| (float64 decoder vs torch BF16, D4a, for scale) | 2.6e-3 | 0.40 |

* **The GPU's decode is about ten times closer to the exact decoder than torch's own BF16 run (mean) and eight times at the worst pixel**; against torch it shows torch's BF16 noise. As 8-bit RGB the GPU's image differs from torch's by 0.65 of 255 on average (the two look the same).
* **What is and is not end to end:** this is D0's final latents decoded by cancho-gpu (the latents are torch's BF16 trajectory's, not D3's, and the image is the 256 x 256 reference one); the host still lays the latent out as a bordered f16 matrix (D6 will do it on the device, with the latents' mean and std), supplies the weights as f16 files and the selection matrices, and clamps to [-1, 1] (a kernel is
  not written for it: three lines). D3's latents on the GPU followed by this decoder is D6.
* **Time:** 1.3 s for the session at 256 x 256, with the compile of 161 programs and the upload of the weights (a cold start): **no speed is claimed**; the cost is 1.06 x 10^12 FLOPs here and 1.7 x 10^13 at 1024 x 1024 (D4a), and **nothing has run at 1024 x 1024**: the scores scratch there is 4,416 x 4,416 x 4 B = 78 MB, the largest activations are 1,052,704 rows of 144 f32 channels (606 MB).
* `vae_norm` is also in `tests/cases.txt` (the attention's size, 4,416 rows of 1,152) and `scripts/metal-kernels.sh` (bit-identical to the interpreter on the small case).

### D4b results (measured): the decoder's pieces as kernels

Ten new kernels (`vae_norm_silu`, `vae_bias`, `vae_bias_res`, `vae_pad16`, `vae_up2`, `vae_up2_epi`, `conv1`, `conv3_k16`, `vae_split16`, `vae_mask_add`), each held by `tests/vae_pieces.py` to a binary64 reference **written from the mathematics, every element, before the kernels** (it
failed with a missing file first), on a 6 x 6 image in the layout of the plan (bordered rows rounded up to 64, `w + 3` guard rows), with **twelve mutants, all killed**. A first mutant (an eps of 0.01 in the norm) survived: the test rows had a norm far above it; a row of tiny norm
and a zero interior row (whose result must be 0, not nan) were added, and it died. The worst error of every kernel is below its tolerance (f16 outputs: half an f16 step; f32: 2e-6 relative; the products 2e-5, for sums of 48 to 64 terms). Production shapes at 1024 x 1024 (rows 1,052,704 at the last block's 144 channels,
264,256 at the 512 x 512 source of the last upsample, 4,416 at the attention's 64 x 64, `conv3_k16` at 144 -> 144) emit CUDA and Metal and none is refused; the goldens are 408 of 408, none a refusal. On the M4 Max, `scripts/metal-kernels.sh` runs the small cases against the interpreter: the f32 and
f16 kernels are bit-identical or within 8.4e-7 (relative to the peak). Wired into `scripts/gate.sh`.

* **Decisions made while building:** the norm's `max(|x|, 1e-12)` is `|x|^2 + 1e-24` under a reciprocal square root (the same for any row with a real norm, 0 and not nan for a zero row); `sqrt(c)` and the attention's `1 / sqrt(c)` are constants of the kernel (`sc=`), as `eps` is; the nearest upsample is a grid over
  source rows and columns (the offsets are affine in the two coordinates, not in a flat row); the channel-selecting shortcut will be `conv1` with a host-made 0/1 matrix.
* **Not done:** the decoder itself (D4c): chaining these with the existing `conv3.chx` (channels 64, 288, 576, 1152), `gemm_mma.chx` (the attention's scores), `dit_stats.chx` and `dit_pv.chx` (nh = 1, d = 1152) into one session; the kernel that lays the latents out as the first bordered image and undoes the latents' mean and std; the host's selection matrices; and the comparison with D0's `vae_out`. No timing.

### D4a results (measured): the decoder in float64, what it costs, and the plan

`python tools/vae_ref.py GOLDEN_DIR` is the decoder of `AutoencoderKLQwenImage21` for one frame in float64 (the structure is in the tool's docstring, read from `autoencoder_kl_qwenimage21.py`), held to D0's `vae_out`:

| Comparison (output [4, 256, 256], clamped to [-1, 1], rms 0.82) | Mean abs error | Max abs error |
|---|---|---|
| float64 decoder vs **torch float32 on the CPU** (the same model) | **5.8e-7** | **8.8e-5** |
| torch float32 vs torch BF16 on MPS (D0's golden) | 2.6e-3 | 0.40 |
| float64 decoder vs torch BF16 | 2.6e-3 | 0.40 |

* **The float64 decoder is the model** (it agrees with torch's float32 to 6e-7), and **the decoder's BF16 noise floor is 2.6e-3 mean and 0.40 max** (0.3% of the output's rms; the worst pixel is far from the mean: a deep network of 1,152 channels in 8 bits). Per channel, the first three
  (colour) have a mean error of 3.0e-3 to 3.5e-3 and the fourth (the extra channel the pipeline drops) 2.9e-4. A decoder kernel is held to the float64 one first, and to the golden at this floor.
* **The structure, measured from the weights and the code:** a 1x1 convolution, `conv_in` (64 -> 1152), a mid block (residual, **attention** over all `h w` positions with one head of 1,152, residual), **five up blocks** (1152 -> 1152, 1152 -> 1152, 1152 -> 576, 576 -> 288, 288 -> 144) of three
  residual blocks each, the first four ending in a nearest-neighbour upsample by 2 and a 3 x 3 convolution, and each with a **shortcut** (DupUp3D: the block's input, channels repeated and rearranged into 2 x 2 positions, added to the output), then RMS norm, SiLU, `conv_out` (144 -> 4). Every convolution is 2-D
  (the temporal ones do nothing for one frame; the first three up blocks' `factor_t` 2 only changes which of two rearranged frames the first-chunk rule keeps). The RMS norm is over the channels of each pixel (`x / |x| * sqrt(C) * gamma`), not over a group.
* **What it costs:** **1.06 x 10^12 floating-point operations for a 16 x 16 latent (256 x 256 image), 1.7 x 10^13 for 64 x 64 (1024 x 1024)**, almost all in 3 x 3 convolutions: the three upsamplers' convolutions at the doubled resolution are 9% each at 256 x 256, and the first residual convolutions of
  the last three blocks 5% each (the largest single convolution is 98 GFLOP at 256 x 256, 1.6 TFLOP at 1024 x 1024). The attention's scores are 4 x h w x h w x 1152 (7.7 x 10^10 at 1024 x 1024: small). The activations at 1024 x 1024 are 1024 x 1024 x 144 x 2 B = 302 MB in f16 at the end.
  The README's best matrix product on this machine is 2.2 TFLOPS (512 x 512 x 512); the convolution kernel (`conv3`) has not been timed at this size, so **no time is claimed**: at that rate the 1024 x 1024 decode could not be under 8 s.
* **The plan for D4b (choices, not built):**
  * **Layout:** NHWC as a padded matrix of rows (one row a pixel, a border of zeros), f16, as the CIFAR network's (`docs/resident.md` section 18); `conv3.chx` (nine shifted products on the matrix unit; f16 in, f32 out, `bk` 32) covers the convolutions whose input channels are a multiple of 32
    (64, 288, 576, 1152). **The 144-channel ones (the last block's, and `conv_out`'s input) are not** (144 = 4.5 x 32): a variant with `bk` 16 and `bn` 48, and `conv_out`'s 4 output channels are padded to 8 with zero weights.
  * **Pieces that are new:** the RMS norm over channels with SiLU and the zero border (rows of up to 1,152 f32: it fits a threadgroup, like `dit_lnmod`); the bias, residual and border mask after a convolution; the nearest upsample (a grid over the input's rows and columns storing four windows: the offsets are affine in the two
    coordinates, not in a flat row index); the shortcut, which for `in == out` is that upsample of the input and otherwise **selects channels** (block 2 and 3 take the odd channels, or the row phase's, of the input): a 0/1 selection matrix as a 1 x 1 convolution, exact in f16 (each output is one input times 1, accumulated in f32), made by the host;
    and the mid block's attention (a matrix product for the scores with f16 q and k, the statistics and the weighted sum as `dit_stats.chx` and a `d` = 1152 version of `dit_pv.chx`: their tiles hold `d` columns, which needs a loop over `d`).
  * **Gates:** each piece against a float64 reference, every element, with mutants, as D1 to D3; then the decoder on the GPU against this tool and against D0's `vae_out`.

### D3 results (measured): the eight steps on the Metal GPU

`python tools/dit_traj_ref.py GOLDEN_DIR TRAJ_DIR` runs the eight steps in float64 (the transformer of `tools/dit_step_ref.py` and the Euler update `latents <- latents + (sigma_next - sigma) * prediction`, with D0's sigmas: 1, 0.978453, 0.95418,
0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0, which are the checkpoint's `sample_sigmas` unshifted; 14 s a step), and `tests/dit_loop_gpu.py GOLDEN_DIR TRAJ_DIR` runs them on the M4 Max as **one session of 659 programs**:
the 657 of D2e, `dit_euler` (a new kernel, in `tests/dit_block.py` with two mutants, both killed) and its f16 copy. The 14 GB of weights are uploaded once; each step reads its timestep embedding and its `dt` from a **table**, and the
latents go from one step to the next by **`carry`**, on the device. D0 now also saves the sigmas and timesteps.

| Step | latents rms | GPU vs float64: mean abs error (of the rms) | torch BF16 vs float64: mean abs error | GPU vs torch BF16 |
|---|---|---|---|---|
| 0 | 0.982 | 1.2e-5 (1.3e-5) | 1.13e-3 | 1.13e-3 |
| 1 | 0.960 | 1.9e-5 (2.0e-5) | 1.55e-3 | 1.55e-3 |
| 2 | 0.936 | 2.7e-5 (2.9e-5) | 1.89e-3 | 1.89e-3 |
| 3 | 0.910 | 4.0e-5 (4.4e-5) | 2.23e-3 | 2.24e-3 |
| 4 | 0.873 | 6.7e-5 (7.7e-5) | 2.44e-3 | 2.46e-3 |
| 5 | 0.802 | 1.9e-4 (2.4e-4) | 2.98e-3 | 3.07e-3 |
| 6 | 0.837 | 5.1e-4 (6.1e-4) | 5.02e-3 | 5.31e-3 |
| **7 (the final latents)** | 1.224 | **1.3e-3 (1.0e-3)**, max 0.039 | **1.1e-2**, max 0.237 | 1.2e-2, max 0.276 |

* **The GPU's trajectory stays close to the exact one at every step and ends about nine times closer to it than torch's BF16 run (mean) and six times at the worst element.** Against torch's BF16 latents the difference is torch's own error, as before.
* The GPU's error grows about 100-fold over the eight steps (1.2e-5 to 1.3e-3), as torch's does (1.1e-3 to 1.1e-2): the steps' sizes grow (the last one is 0.41 of the whole range) and each carries the previous steps' error. The final latents' relative error is 0.1% for the GPU.
* **Time:** 8 steps in 6.1 s, with the compile and the 14.3 GB upload included, at 320 tokens: **no speed is claimed.**
* **The check in the test is deliberately loose** (the mean error against torch's BF16 latents under 5% of their rms, and finite): it is torch's noise that sets it; the float64 comparison is what is reported.
* **Not done:** the VAE decoder (D4), the text encoder (D5), the image (D6); anything at 1024 x 1024.

### D2e results (measured): the whole step on the Metal GPU

Five more kernels (`dit_gelu`, `dit_silu`, `dit_add`, `dit_modsplit`, `dit_rows2`; in `tests/dit_block.py` with seven mutants, **all killed**: every kernel of D2b to D2e is now under one gate and 17 mutants) put the rest of the step on the device:
`txt_in` (RMSNorm with the zero-centred weight, matrix product, GELU, matrix product), `img_in`, the timestep embedding (matrix product, SiLU, matrix product), the shared modulation (SiLU, matrix product, split into the
four [2, n] inputs), and at the end `norm_out` (SiLU of the embedding, matrix product, `dit_lnmod` with that scale) and `proj_out`. `tests/dit_full_gpu.py` runs it as **one session of 657 programs**; the host provides only the
zero-padded text and latent matrices, the timestep's sinusoid (a table of cos and sin, like RoPE's) and the f16 weight files. The padded rows are zero and stay zero through the front end (the RMSNorm of a zero row is 0, GELU(0) = 0), so the
text rows and the image rows are added (`dit_add`) into the joint sequence.

| Comparison | Mean abs error | Max abs error |
|---|---|---|
| residual after block 4 / 8 / 16 / 24 / 32, against float64 | 1.1e-3 / 3.3e-3 / 6.3e-3 / 5.7e-3 / 1.1e-2 (5e-5 to 2.4e-4 of the rms) | 3.0 / 10.0 / 10.9 / 11.2 / 11.2 |
| **the step's output [277, 64], GPU (front end, blocks, back end) against float64** | **5.9e-4 (3.7e-4 of the output rms 1.61)** | **6.9e-3** |
| the step's output, GPU against torch BF16 (`step0_out`) | 9.6e-3 (6.0e-3 of the rms) | 0.132 |
| (float64 against torch BF16, for scale) | 9.8e-3 | 0.133 |

* **The whole step now needs no host arithmetic**, and it is still about **16 times closer to the exact step than torch's BF16 run (mean) and 19 times at the worst element**: D2d, with the front and back ends in float64 on the host, was 53 and 36 times. The difference is what the
  front end loses in f16 (the latents, the text rows after their RMSNorm, the activations between `txt_in`'s products): the residual after block 4 has a mean error of 1.1e-3 against D2d's 2.4e-4, and a **maximum of 3.0 against 0.059**. The residual's peak is 9,130 and f16's spacing there is 4 to 8, so
  a maximum error of 3 to 11 is the size of one f16 rounding of a massive activation; **I did not locate the element**, so that is the likely cause and not a shown one.
* **Time:** 4.3 s for the session, with the compile of 657 programs and the upload of 14.3 GB, a cold start at 320 tokens (D2d: 3.9 s): **no speed is claimed.**
* **A mistake of method, for the record:** the gate for these five kernels was written after the kernels (they compiled at the first try and the references were then written); the mutants, which are what shows the gate can fail, were written with it and all fail it. The earlier slices wrote the gate first.
* **Not done:** the eight steps and the scheduler (D3), at any size; 1024 x 1024.

### D2d results (measured): the 32 blocks on the Metal GPU

`python tools/dit_step_ref.py GOLDEN_DIR REF_DIR` is the whole transformer step in float64 (the front end: `img_in`, `txt_in`, the timestep embedding and the shared modulation; 32 blocks; `norm_out` and `proj_out`), and
`tests/dit_step_gpu.py GOLDEN_DIR REF_DIR` runs the 32 blocks as **one session of 640 programs** on the M4 Max, the residual stream of each block linked on the device to the next block's inputs. The front and back ends are on the host in float64 in this slice (D2e puts them on the GPU).
D0 now also saves the timestep (1.0 at step 0). Neither script is in `scripts/gate.sh` (a Mac, the checkpoint, 14 GB of scratch disk).

| Comparison | Mean abs error | Max abs error |
|---|---|---|
| residual after block 4, against float64 | 2.4e-4 (1.3e-5 of the rms 19.3) | 0.059 |
| after block 8 | 6.9e-4 (1.9e-5 of 37.1) | 1.16 |
| after block 16 | 1.6e-3 (3.8e-5 of 41.5) | 1.15 |
| after block 24 | 1.4e-3 (3.4e-5 of 41.5) | 1.73 |
| after block 32 | 3.1e-3 (6.7e-5 of 47) | 1.86 |
| **the step's output [277, 64], GPU against float64** | **1.9e-4 (1.2e-4 of the output rms 1.61)** | **3.7e-3** |
| the step's output, GPU against torch BF16 (D0's `step0_out`) | 9.8e-3 (6.1e-3 of the rms) | 0.133 |
| (float64 against torch BF16, for scale) | 9.8e-3 | 0.133 |

* **The GPU's step is about 53 times closer to the exact step than torch's own BF16 run (mean), and 36 times at the worst element**; against torch it shows torch's BF16 noise and nothing of its own. The residual's error grows with depth (2.4e-4 after
  4 blocks, 3.1e-3 after 32) the way the activations do (the residual's rms goes from 19 to 47) and stays at 7e-5 of it.
* **The f16 question of §5, answered for this step:** in float64, over the 32 blocks, the largest value of every tensor the GPU path holds in f16 is **21** (the input of q, k, v), **12.3** (gate, up), **59.3** (the attention output) and **4,923** (the output of
  SwiGLU, the input of the down projection), against f16's 65,504; the residual, kept in f32, peaks at 9,130. **f16 activations do not overflow for this prompt at this step**; they are one prompt and one step, not a bound for others (a different prompt, or a later
  step, has other activations): D3 sees the other seven steps, and a guard that refuses a tensor over 65,504 is cheap to add and is not built.
* **Time:** the session answered in **3.9 s**, including compiling 640 programs and uploading 13.9 GB of weights (the weight files, written in 14 s, were in the page cache). That is a cold start for one step at 320 tokens; **no speed is claimed**, and 4,352 tokens (1024 x 1024)
  have 14 times the work in every matrix product and 200 times in the scores.
* **The front end** reproduces D0's modulation to 0.016 (max abs; torch's is BF16).
* **Not done:** the front and back ends on the GPU (D2e); the eight steps (D3); anything at 1024 x 1024: the scores scratch there is 2.1 GB per layer, the weights' 13.9 GB are resident, and the session was not run at that size.

### D2c results (measured): block 0 on the Metal GPU

`tests/dit_block_gpu.py GOLDEN_DIR` (needs a Mac, the checkpoint, D0's goldens and the repo's session program; not in `scripts/gate.sh`) writes one **session of 20 kernel programs** and runs it on the M4 Max: `dit_lnmod`,
three `gemm_mma` (q, k, v), two `rmsnorm_rows`, two `dit_rope`, `dit_scores`, `dit_stats`, `dit_pv`, `dit_to_f16`, `gemm_mma` (o), `dit_gate_res`, `dit_lnmod`, two `gemm_mma` (gate, up), `dit_swiglu`, `gemm_mma` (down), `dit_gate_res`.
277 tokens are padded to 320 (the matrix unit's tile is 64); the weights are f16 files (436 MB), q and k permuted within each head. The session writer is Python, as the repository's others are (`examples/gpt/gpt_session.py`).

| Comparison (277 tokens x 4096) | Mean abs error | Max abs error |
|---|---|---|
| session vs the **float64 block** | **5.5e-5 (3.3e-6 of the rms 16.9)** | **0.072** |
| session vs torch BF16 on MPS (D0's golden) | 3.2e-3 (1.9e-4 of the rms) | 11.1 |
| (torch BF16 vs float64, D2a, for scale) | 3.2e-3 | 11.1 |

* **The GPU block is about 58 times closer to the exact block than torch's own BF16 run (mean), and 150 times at the worst element.** Against the golden it shows torch's BF16 noise and nothing of its own; the tolerance of
  D2a (float64 within 1e-3 of the rms; the golden within 4e-4 and 24) holds with room.
* **Time:** the session answered in **0.8 s**, which includes opening the device, compiling 20 programs and uploading 436 MB of weights. That is not a speed measurement of the block (one run, a cold start; 320 tokens, not 4,352); none is claimed.
* **The test can fail** (`--mutate`): the channels of q and k not permuted (mean error 0.39, 2.3% of the rms: refused), the selector inverted (0.25: refused). A third, keys not masked at the padded positions, **passes, and is equivalent here**:
  the padded rows are after every real token and the mask is causal, so no real query can see them; the mask for the padded keys matters for the padded queries only, which are dropped.
* **On CUDA** the `gemm_mma` schedule's block is 128 (`bm 128`), so the sequence would be padded to a multiple of 128 there; the CUDA text of these kernels is emitted and held by goldens and has not run (no NVIDIA machine, as the README says).
* **Not done:** 31 more blocks (the weights are 14.2 GB, and 436 MB x 32 does not go through one session file as written: D2d); the timestep embedding, `img_in`, `txt_in` and `norm_out`/`proj_out` around the blocks; activations of the later blocks (the f16 range question stays open until D2d measures them).

### D2b results (measured): the pieces between the matrix products

Four new kernels and one existing one, each held by `tests/dit_block.py` to a binary64 reference, every element (the gate was written first and failed with "cannot read
dit_lnmod.chx"):

| Kernel | What | Worst error, as a fraction of its tolerance |
|---|---|---|
| `dit_lnmod.chx` | layer norm without affine, times 1 + scale; the scale row (the t = 0 one for text, the sampled one for the target image) chosen per token by a selector; **f16 output** for the next matrix product | 0.48 (f16 half-step tolerance), with an outlier of 2,300 and a row of variance ~5e-7 so that eps is tested |
| `dit_rope.chx` | rotate-half RoPE with per-position tables, rows of 128 (a head), `nh` to a token | 0.06 (2e-6 relative) |
| `rmsnorm_rows.chx` (existing) | the per-head RMSNorm of q and k with a [1, 128] weight, eps 1e-6 | 0.19 |
| `dit_gate_res.chx` | x + tanh(gate) * y, the gate row chosen per token | 0.16 |
| `dit_swiglu.chx` | silu(g) * u, f16 output (**corrected in D2c:** the first version took a whole row; at n = 12288 that needs 73,728 B of threadgroup memory against Metal's 32,768, and its production-shape golden had been recorded as a *refusal* (`REFUSED`) that I did not read; it now takes `bn` columns to a threadgroup, and no `dit_*` golden is a refusal) | 0.63 (f16) |

* **The host's plan for RoPE, checked in pure Python:** the model rotates interleaved pairs (2j, 2j + 1); the kernel rotates halves. Permuting the output channels of `to_q` and `to_k`
  (and the two norm weights) within each head, even channels first, once at load, makes them the same up to that permutation, and leaves q . k and the RMSNorm unchanged
  (equal to 12 digits). The identity permutation in its place is refused by the gate.
* **Attention layout:** the D1 kernels now take q, k, v and give o **token-major** (`[tokens, 32 x 128]`, as the projections write them: a head's tile is a window at column
  `h * 128`), so no transpose sits between the projections and attention. The D1 gate and its five mutants were rerun on the new layout: all pass, all killed.
* **Mutants:** eight deliberate changes (the scale used without the 1, eps 1e-4, the selector ignored in two kernels, the second half's sign, position 0's tables for every token,
  no tanh, the sigmoid of u) plus the identity permutation: **all killed**. The eps mutant survived the first inputs, because a row's variance was always far above eps; the row
  of tiny variance was added for it.
* **On the GPU:** `scripts/metal-kernels.sh` runs all of them against the interpreter on the M4 Max: the f32 kernels within 2.7e-7 (relative to the output's peak), `dit_lnmod` within
  1.6e-4. The script now has a bound of 5e-4 for a case written `f16:name`: a one-bit difference of an f32 result flips an f16 rounding (half a step is 4.9e-4 of the value), so 1e-5
  would refuse a correct kernel. The whole script passes.
* **Goldens:** 376 of 376 agree (the D1 ones were recorded again for the new layout; no other directory changed).
* **Not done:** the matrix products. `tests/chx/ported/gemm_mma.chx` (f16 weights and activations, f32 accumulation) is the candidate, held at the block's shapes in D2c. The sequence length
  must be a multiple of its tile (64 on Apple), so the host pads the prompt; the padded rows must not be keys (the mask already says so).

### D2a results (measured): the tolerance, and a float64 block

`tools/dit_block_ref.py GOLDEN_DIR` computes block 0 in float64 from the checkpoint's weights, D0's block input, the shared modulation and the rotary frequencies
(D0 now also saves `block0_modulation`, `block0_rope` and `block0_target_mask`; the earlier hashes did not change, and two runs still agree bit for bit), and compares it
with the BF16 block output torch produced. The block's definition is in the tool's docstring, read from `transformer_qwenimage21.py`.

| Measurement | Result |
|---|---|
| float64 block against torch BF16 on MPS, all 277 x 4096 outputs | mean abs error 0.0032, **1.9e-4 of the output's rms** (16.9); largest abs error **11.1** |
| where the largest error is | token 19 (a text token), channel 2472: exact 2276.9, torch 2288.0; BF16's spacing at that size is 16. The block has **massive activations**: the input peaks at 2,352 against an rms of 16.9 |
| relative error per element | not meaningful near zero (a value that cancels has a huge relative error): the error is reported against the rms, and the largest in BF16 steps of the value is 0.7 of a step at the outlier |
| weights of block 0 narrowed from BF16 to f16 | **21,085 of 218,104,064 change (0.0097%)**; the block with f16 weights differs from the exact one by 1.1e-6 at most: **narrowing the weights is free** for this block |
| largest activation in block 0 | 2,369 (output); attention scores' q and k peak at 25.6; f16 holds 65,504: **no overflow in block 0**, which says nothing about blocks 1 to 31 (measure them in D2d before choosing f16 over f32 for the activations) |

**The tolerance, a choice from these numbers:** a kernel's block is held to the float64 block first (mean abs error at most 1e-3 of the output's rms: f16 matmuls
with f32 accumulation should sit far below torch's BF16 at 1.9e-4 ... and if they do not, that is the finding), and to torch's BF16 golden at the BF16 noise floor: mean abs
error at most 4e-4 of the rms and max abs error at most 24 (a step and a half of BF16 at the outlier). A kernel that is closer to float64 than torch's own BF16 is closer to
the truth, not wrong.

### D1 results (measured)

Three kernels, `dit_scores.chx`, `dit_stats.chx` and `dit_pv.chx` (`tests/chx/`), chained: for each head, a tile of `qs` queries against `ts` keys
(scores, scale 1/sqrt(128), additive mask), the row maximum and normaliser over chunks of `cs` keys, then the weighted sum of v. The mask is an input
(the host builds Qwen-Image 2.1's block-causal one); the kernels tile the queries as well as the keys, which the decode kernels of §3 did not.

* **Gate** `python3 tests/dit_attention.py build/cancho-gpu` (written before the kernels: it failed with "cannot read dit_scores.chx", then passed). Three
  cases through `cancho-gpu run` with real inputs, every output element against a binary64 reference that computes each row's softmax at once and builds
  the mask from `build_qwenimage21_block_causal_mask`'s definition: 2 heads of 32 text positions; 3 heads of 48 positions (5 text, two image blocks, two padded keys);
  2 heads of 64 with the row maximum in the last chunk. Max abs error **1.7e-7, 1.2e-7, 1.2e-7** (tolerance 2e-6).
* **Mutants** `--mutants`: five deliberate changes (the scale 1/16, the mask read transposed, the running maximum kept from the first chunk, the normaliser skipping
  the last chunk, v read from the first head's rows), **all killed**. A first version of the scale mutant survived because it changed the literal in the
  kernel's header comment, not the code; the mutant now names `s * 0.0883...`.
* **Production shape compiles:** 32 heads, 4,096 queries and keys, `qs` 32, `ts` and `cs` 64, `dw` 32 emit CUDA and Metal and pass `check` on both targets (the 32 KiB
  threadgroup budget holds); the goldens are `tests/cases.txt`'s six new cases.
* **On the GPU** (`scripts/metal-kernels.sh`, Apple M4 Max, 4 heads of 256 positions): worst relative difference from the interpreter **1.2e-7, 2.7e-7, 1.9e-7**.
* **Not measured:** speed at 4,096 positions; the scores scratch for 32 heads at that size is 32 x 4,096 x 4,096 x 4 B = 2.1 GB, which is why a fused kernel
  (no scratch) is a later step. The text mask golden of D0 is still missing. The three chained kernels were checked on 32 to 64 positions on the CPU and 256 on the GPU, not at 4,096.
* **Repo state:** the goldens agree 368 of 368 with a binary built from these sources by the compiler at hand (`5a5dd1f`, not the pinned `a49ea77`, `--ignore-compiler-rev`); the
  `build/cancho-gpu` left from 3 November disagreed on 29 older fp4 cases, which is a stale binary, not a change here.

### D0 results (measured)

`python tools/diffusion_golden.py OUTDIR --check` (the venv of the first edits; 5.4 s for one run, model load not included). The
files are 11 MB in all, so they are **not committed**; `golden/diffusion/manifest.json` holds each tensor's shape, dtype and
sha256, and the tool regenerates them. The hashes hold for this torch and this machine; a different torch or Mac may differ in the last
bits, and that is a reason to re-run `--check`, not a failure of a kernel.

| Tensor | Shape | |
|---|---|---|
| text embeddings | [1, 21, 4096] | 21 prompt tokens, hidden 4096 |
| block 0 in / out | [1, 277, 4096] | joint sequence: 21 text + 256 image tokens |
| step 0 in / out | [1, 256, 64] / [1, 277, 64] | the pipeline keeps the last 256 rows of the output |
| latents after each step | [1, 256, 64] | 16 x 16 tokens of 64 channels |
| VAE in / out | [1, 64, 1, 16, 16] / [1, 4, 1, 256, 256] | the decoder returns **4 channels** (`in_channels` 4: RGB plus one more, which the pipeline's postprocess drops); the VAE's tensors have a frame axis of 1 |

The 256 x 256 image is 256 tokens, which is the §5 count (16 x 16 pixels a token) measured, not only read. The text mask was
`None` for this single prompt, so no mask golden exists yet (a padded batch would need one).

## 5. Open questions, to answer by measuring before D1

1. **The number of positions** (answered by reading `pipeline_qwenimage21.py`, not by running it): the VAE
   compresses 16x spatially and the transformer takes latents unpatched, so one token covers 16 x 16
   pixels. A 1024 x 1024 image is 64 x 64 = **4,096 image tokens**, a 256 x 256 one 256, plus the prompt's tokens
   and, when editing, the reference image's. D0 measures the real sequence lengths and writes them here.
2. **What `causal_condition` does** (answered from `transformer_qwenimage21.py`): the text and reference-image tokens are
   modulated from timestep 0, so their activations do not depend on the step and can be cached; only the target image's
   tokens use the real timestep. The sequence is block-causal (§3). The pipeline's `use_kv_cache` relies on it and, by
   its own docstring, changes the rounding, so the same seed gives a different image with it on and off: **choice**,
   the goldens fix `use_kv_cache=False`.
3. **f16 or f32 for BF16 weights** (answered in §3: there is no BF16 dtype). Whether this
   model's activations stay in f16's range is measured against D0's goldens before D2; if
   not, the transformer goes up as f32 (28.5 GB) or the overflowing tensors stay wide.
4. **Memory.** The three parts are 32.4 GB; attention scratch for the N of question 1 is added
   to it. Measure the peak against the 64 GB of this machine.
5. **KV caching.** The model card says the pipeline reuses the text and reference-image context
   across steps ("prefix KV caching"). Whether D2 needs it to be fast, or only to be correct,
   is open; correctness comes first, so the first version recomputes.

## 6. Video, second

Not started; only the shape of the problem is recorded, from the model cards (read on 2026-11-13,
not measured here).

* **MiniMax-H3** (`MiniMaxAI/MiniMax-H3`, 498.5 GB in the repository): a 33 B-parameter
  dense transformer, a Qwen3-VL-32B text encoder, a visual VAE and a 32 kHz stereo audio VAE;
  4 to 15 s of video at 768 p by default. Its own examples use four GPUs. At BF16 it does not
  fit in this machine; the community repackagings are 12.5 to 40 GB for the transformer in
  formats (`int8_convrot`, `w4a8`) that are not NVFP4. **Open:** whether NVFP4, which cancho-gpu
  has (`docs/weight-space*.md`, README), can carry a 33 B transformer here (about 17 GB of weights
  at 4 bits, **an estimate**, plus the 32 B text encoder).
* **Lance** (`bytedance-research/Lance`, 30.9 GB, Apache-2.0): a 3 B language model, a
  Qwen2.5-VL vision tower and Wan2.2's VAE; 480 p, 12 fps. Its README requires CUDA and 40 GB of
  VRAM; it has not been tried on a Mac. It is the smaller road to video, and the VAE is the 3-D
  convolution that D4 does not cover.

Video stages (V0 onward) will be written when D3 has passed: they reuse D0's method, and the
choice between the two models above needs the memory numbers of §5, question 4.

## 7. What this document does not claim

No speed against torch, MLX or ComfyUI; no statement that BF16, bidirectional attention, a VAE or
the vision tower will fit the existing kernels; no result on any model but the Qwen2.5 and
Qwen3.8 ones the README reports. The stages are a plan, ordered so that each gate is cheap and
a wrong guess is found early.
