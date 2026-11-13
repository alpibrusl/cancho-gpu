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
| Non-causal (bidirectional) attention | the decode kernels take an additive mask, so the mask can be all zeros; **open** whether the tiled kernel needs more |
| Adaptive modulation (shift, scale, gate from the timestep) | no |
| 3 x 3 convolution | yes, as nine shifted matmuls of a padded NHWC matrix, in the CIFAR-10 CNN (852,000 parameters, `docs/resident.md` §18); "only small tiles fit" in 32 KiB. The VAE's channels (up to 8 x 96 = 768 encoder, 8 x 144 = 1152 decoder, from `vae/config.json`) at image resolution are far from that network. `conv_step.chx` is the 4-tap depthwise convolution of a gated-delta layer, not this |
| VAE decoder (residual blocks, up-sampling, the latent mean and std) | no |
| Vision tower (27-layer ViT, patch 16, mRoPE) for the text encoder | no |
| Flow-matching Euler step with the saved sigmas | no (a few lines, but it is not written) |

## 4. The plan: stages, each with a gate that can fail

Each stage ends when its gate passes; none starts before the one above it. A gate compares the
`.chx` kernels on the Metal GPU with the reference's tensors at the stage's output.

| Stage | What | Gate (written before the code) |
|---|---|---|
| D0 | A tool that runs the reference on a small fixed input (a 256 x 256 image, one prompt, seed fixed) and writes golden tensors: the text encoder's output, each stage's input and output of one transformer block, one full transformer step, the VAE decoder's output. | The tool's own output is deterministic: two runs give the same bytes. The goldens are under `golden/diffusion/`, with their sizes in the document. |
| D1 | Bidirectional attention with head dimension 128 for N queries and N keys (N from a few hundred up to the sizes in §5, question 1): tiled over both, online softmax (the chunked kernels of §3 tile the keys only). | Against a float64 reference written from the mathematics (`tests/kernels.py` style), including the mutants that must fail it; then against D0's block goldens. |
| D2 | One transformer block, then the 32-layer step. | One block: max abs error against D0 below a tolerance **chosen from what BF16 torch vs float64 gives**, measured first, not guessed. One step: the same. |
| D3 | Eight steps with the saved schedule, from a fixed noise. | The latent after 8 steps against torch's, with the error reported per step. |
| D4 | The VAE decoder. | The decoded image against D0's, in max abs error and PSNR. |
| D5 | The text encoder (text part first, the vision tower later: it is needed for editing, not for text-to-image). | Hidden states against D0's, per layer. |
| D6 | End to end: prompt to PNG on the GPU, timed against torch MPS on the same machine, same seed. | A text-to-image run whose image matches torch's within the D3 and D4 tolerances. The time is a measurement, reported with the cases where cancho-gpu is slower, as `README.md` does for training. |

Editing (the beach, clothes and Ghibli tests in `imgtest/`) needs the vision tower and the
reference-image path; **choice:** text-to-image first, editing after D5's vision tower.

**Test images.** D0 to D3 and D6's text-to-image run need **no input photo**: a prompt and a seed. The editing
stages need one reference image. **Choice:** it will be a photograph with a licence that allows redistribution (public
domain or CC0, from Wikimedia Commons), added to `tests/` with its source URL, author and licence in a file beside it, and
checked in size. No photograph of a person taken by or sent by a user of this project goes into the repository.

## 5. Open questions, to answer by measuring before D1

1. **The number of positions.** How many image tokens does a 1024 x 1024 image give (the
   VAE's spatial factor, `patch_size` 1, the 64-channel latent)? It sets the attention size and
   the memory of D1. Measure it from the pipeline's latent shape; do not infer it from a
   resolution.
2. **What `causal_condition` masks.** Read `QwenImage21Transformer2DModel` in the pinned
   diffusers source (the `imgtest/.venv` has it) and put the answer here.
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
