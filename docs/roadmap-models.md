# Training and serving real models with cancho-gpu: the gaps, and a plan

> **Status: a plan (a design written before the code), with measured starting points.** Marks as elsewhere: **[measured]**, **[read]** (from source or documents, not run), **[open]**.
> Written after `resident.md` section 14: the training stack reaches 2 TFLOPS on the matrix unit and trains MLPs faster than torch on MPS. **M1 is built and met its exits (`resident.md` section 15)**; the rest below is still a plan.

## 1. What the machine can do, and what cancho-gpu can do now

The machine is an M4 Max, 64 GB, about 14 TFLOPS in float32 and more in f16 on the matrix unit, 546 GB/s **[read, Apple]**; the matrix-unit product `gemm_mma` runs at 2.2 TFLOPS in our generic tiling **[measured]**.
A training run holds, per parameter, an f32 master (4 B), Adam's two moments (8 B) and an f16 copy (2 B): **14 B**, plus activations. That bounds training on this machine to about 1 B parameters; **100 to 300 M parameters is the comfortable range, and what the plan aims at**.
Serving is bounded by weight bytes over bandwidth: a 4-bit 7 B model reads 3.5 GB a token, about 150 tokens a second at the bandwidth, and `cancho-gpu` reaches 39.9 on a 27 B model today (`benchmark.md`).

What exists **[measured]**: dense and mixed-precision MLP training with Adam, minibatch tables, binary files, 4 session directives (`resident`, `table`, `carry`, `steps`), ReLU, sigmoid, softplus, exp, log, rsqrt, row reductions; decode kernels (RoPE, KV cache, attention in two kernels, gated-delta, NVFP4 matvec) and runners for Qwen2 and Qwen3.8; an OpenAI-compatible server; a BPE tokenizer and sampler.
What does not: everything in section 2.

## 2. The gaps, by what each model needs

| gap | needed by | what it is | how, in the surface | size |
|---|---|---|---|---|
| **G1 embedding lookup and its gradient** | every LLM, speech | a table row chosen by a token id; the gradient adds into rows | forward: an **index-valued load** (a tile whose row offset is data); backward: **scatter-add**, which needs atomics or a sort-and-segment reduction. For a small vocabulary (characters: 65) a one-hot matrix and `mma`/`mma_tn` do both with no new operation | M (atomics: L) |
| **G2 normalisation, forward and backward** | transformers, CNNs | LayerNorm/RMSNorm/BatchNorm | `rowsum` gives the forward; the backward needs column sums, which a product with a row of ones gives (`rowvec_mm` already is that) | S |
| **G3 softmax and cross-entropy over a vocabulary** | LLMs | row softmax, its backward `p (dp - rowsum(dp p))`, the loss with an integer target | all `rowsum`/`exp`; the target needs G1's index-valued load (or a one-hot for small vocabularies) | S |
| **G4 attention for training** | transformers | causal scores, softmax, `P V` and the backward `dV = P^T dO`, `dP = dO V^T`, `dS = P (dP - rowsum(dP P))`, `dQ = dS K`, `dK = dS^T Q` | every product is an `mma` layout (now available); batched heads are a second grid axis; the causal mask is an additive table. A fused (flash-style) form needs a running max (**`max a b`** in the surface and loops carrying more than one tile, `kernels.md` section 4) | M; flash form L |
| **G5 activations** | all | GELU, SiLU, tanh and their derivatives | element-wise, a few lines each in the lowering and the interpreter | S |
| **G6 optimiser completeness** | all | AdamW decay, global-norm gradient clipping, the schedule | decay is element-wise; the clip is a reduction over every tensor (a kernel summing squares, then a scale in the Adam step); the schedule is already a `table` | S |
| **G7 dropout and random numbers** | regularised training | a deterministic counter-based RNG | a `rand` operation (Philox) in the surface, with a reference in Python | M |
| **G8 range of f16** | deep or large models | f16 overflows at 65504 | bf16 tiles on the matrix unit (does Metal on this OS take `bfloat`? **[open]**), or loss scaling | S to M |
| **G9 data beyond memory** | any real dataset | tables are loaded into the interpreter's binary64 memory (8 bytes an element) before being uploaded; integer ids need an integer dtype | upload tables to the device directly from the file in chunks, `i32` dtype, no interpreter copy | M |
| **G10 checkpoints** | training then serving | a session can only print text | `save P.PARAM FILE` (binary) and a reader in the runners. **Built** (`resident.md` section 18): `save P.OUT FILE [f32\|f16]` and the request `checkpoint`, atomic, with `fs_write` added to the session programs' authority by the owner's approval; the reader is `gpt_session.read_checkpoint` | S |
| **G11 convolution** | vision, speech front-ends | conv2d/conv1d, forward and both gradients | in NHWC with a zero-padded border, a convolution is a **loop over the kernel taps** of `mma` on shifted row windows of one flattened `[N*H*W, C]` matrix (a rectangular load at a computed row offset), and the gradients are `mma_nn` and `mma_tn` over the same taps: **no new operation**, only the kernels; strides above 1 and max-pooling need `max a b` | M |
| **G12 generic serving** | any open model | a decoder runner for any llama-family model (RMSNorm, RoPE, grouped-query attention, SwiGLU), from its files, with a prefill on the matrix unit | today's runners hardcode Qwen2 and Qwen3.8, and prefill takes 8 tokens at a time; a configuration-driven runner on `mma` | L |
| **G13 weight formats** | serving | bf16/f16 dense (trivial) and 4/8-bit groups (Q4_K_M, Q8_0, AWQ/GPTQ) | each is a dequantising matvec like `matvec_fp4`; NVFP4 exists | S per format |
| **G14 contexts beyond 1024** | long prompts | split-KV attention with its merge | G4's running max and multi-tile loops | L |
| **G15 audio** | speech | log-mel spectrogram | a short-time Fourier transform is a product with a precomputed cosine/sine matrix (no FFT needed at n_fft = 400), then the mel filter matrix: two `mma`; reading a wav is host code | S |

## 3. The milestones, each ending at a number

Each milestone is a separate change, designed before it is built, with gates that can fail, and its result is **compared with torch on this machine the way `resident.md` section 13 and 14 compare MLPs**: the same data, the same start, the same batches, the loss curves compared, then the time.

**M1. A character-level GPT trains. Built (`resident.md` section 15): the gradients of all 28 tensors within 8e-4 of a float64 backward, a 10.8 M-parameter GPT at 1.376 nats a character, 37 ms a step against torch MPS's 41.** (original plan follows) (G2 to G6, G10; no G1: 65 characters are a one-hot). A 6-layer, 384-wide, 6-head GPT on the Shakespeare characters (about 10 M parameters), context 256, as in nanoGPT **[read]**.
Exit: the loss after 100 steps equals a float64 numpy implementation's to 1e-3 (CPU session) and torch's to 3 digits; the validation loss falls under 1.6 nats; step time and total time are within 1.5 of torch on MPS, or better.
New kernels: LayerNorm forward/backward, causal attention forward/backward (non-fused first), GELU, the cross-entropy, AdamW. New language: `gelu`, `tanh` (G5), `max a b`.

**M2. Serve what M1 trained. Built (`resident.md` section 16): all 576 greedy tokens equal torch's, 650 tokens a second against torch MPS's 340 (recompute) and 273 (cached); the checkpoint is host-side, no key/value cache yet.** (original plan follows) (G10, G12 in its smallest form). `save`/load the checkpoint; a decoder runner for this GPT with a KV cache (the attention-decode kernels exist); sample text with the existing sampler.
Exit: the generated tokens equal torch's greedy tokens for the same weights; tokens a second against torch on MPS.

**M3. A real vocabulary and a real dataset. Built except for what is named (`resident.md` sections 17 and 18): a 2,048-token BPE, a 27 M-parameter GPT (81 ms a step against torch's 86) and, through the device-side `save` of section 18, a 59.95 M-parameter one (8 layers, d 768): 167.8 ms a step against torch MPS's 196.7 (15% faster) and 201.1 with mixed precision, validation loss 3.106 against 3.114 and 3.278, the curves within 0.013 nats at every point; no gather/scatter needed at that vocabulary. Not done: TinyStories (it needs a download), vocabularies above about 2,048, memory against torch, contexts above 128.** (original plan follows) (G1, G7, G9). Index-valued load and scatter-add (atomics, or the sort-and-segment form measured against them), device-direct tables, a BPE-tokenised corpus; a 30 to 60 M parameter GPT on TinyStories-scale text.
Exit: the loss curve agrees with torch's; tokens a second in training against torch; memory against torch.

**M4. A CIFAR-10 convolutional network** (G11, BatchNorm of G2, `max`). A ResNet-style network of about 1 M parameters. Exit: 90% test accuracy in the time torch needs on this machine, the curves agreeing.

**M5. Serve an open LLM** (G12, G13, G14 optional). A llama-family model of 1 to 3 B parameters (Llama-3.2, Qwen2.5) from its safetensors or GGUF files with a mixed-precision prefill. Exit: its tokens equal the reference's on a fixed prompt set (`scripts/model-check.sh` style) and its speed is reported against Ollama and MLX-LM as `benchmark.md` does.

**M6. Speech** (G11 for conv1d, G15). Whisper-tiny/base inference: log-mel, encoder, decoder with cross-attention. Exit: the transcript of a fixed clip equals the reference's, and the real-time factor against torch on MPS is reported; fine-tuning only after that.

## 4. What this plan does not do

* **Multi-GPU or distributed training**, and models beyond what 64 GB trains (1 B parameters at the very most).
* **CUDA**: every new operation is lowered for both targets and held by goldens, but no kernel here has run on NVIDIA. The numbers above are Apple's; an NVIDIA machine would measure the same milestones separately (the operand-layout text of `mma_nn`/`mma_tn` for CUDA is generated, not run).
* **A model zoo or a Python front end.** Models are written as sessions of `.chx` kernels (generated by scripts, as `mlp_session.py` is); a higher-level builder is a later question and is not needed to reach any milestone.

## 5. Order, and the first step

G5 and `max a b` (S, language), then G2 and G4 forward and backward as kernels held to numpy, then the GPT session (M1). M1 needs no new directive and no new dtype, so it can start at once; it is the first test of whether the kernel set reaches a transformer.
