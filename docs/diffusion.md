# Image and video diffusion models in `.chx`

> **Status: design only. Nothing here is built, and no diffusion model has been run in cancho-gpu.**
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
| D4 | The VAE decoder. | The decoded image against D0's, in max abs error and PSNR. |
| D5 | The text encoder (text part first, the vision tower later: it is needed for editing, not for text-to-image). | Hidden states against D0's, per layer. |
| D6 | End to end: prompt to PNG on the GPU, timed against torch MPS on the same machine, same seed. | A text-to-image run whose image matches torch's within the D3 and D4 tolerances. The time is a measurement, reported with the cases where cancho-gpu is slower, as `README.md` does for training. |

Editing (the beach, clothes and Ghibli tests in `imgtest/`) needs the vision tower and the
reference-image path; **choice:** text-to-image first, editing after D5's vision tower.

**Test images.** D0 to D3 and D6's text-to-image run need **no input photo**: a prompt and a seed. The editing
stages need one reference image. **Choice:** it will be a photograph with a licence that allows redistribution (public
domain or CC0, from Wikimedia Commons), added to `tests/` with its source URL, author and licence in a file beside it, and
checked in size. No photograph of a person taken by or sent by a user of this project goes into the repository.

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
