<p align="center"><img src="docs/assets/cancho-gpu-logo.png" alt="cancho-gpu" width="220"></p>

# cancho-gpu

[![ci](https://github.com/alpibrusl/cancho-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/alpibrusl/cancho-gpu/actions/workflows/ci.yml)

**A GPU kernel compiler that says what it can do.** A compiler for typed tile kernels (the `.chx` language), written in [cancho](https://github.com/alpibrusl/cancho): it checks a program (every tile consumed exactly once, bounds, shapes, the threadgroup budget) and lowers it to **CUDA C** and **Metal**. The language began as an independent Rust implementation's and is defined here now ([`docs/chx.md`](docs/chx.md), [History](#history)). It links nothing but libc, and `cancho authority`, checked in CI, names what it may do: read and write files, print, allocate; never the network, never foreign code. The [project page](https://alpibrusl.github.io/cancho-gpu/) has the summary.

**Demo:** a recorded terminal session (a kernel compiled to Metal, a wrong one refused, Qwen3.8-27B generating on the GPU) is on [the project page](https://alpibrusl.github.io/cancho-gpu/) and in [`docs/demo.mp4`](docs/demo.mp4); `scripts/make-demo.sh` records it again.

**Status: it runs language models on a Metal GPU and it trains and fine-tunes models on one.** *Running:* Qwen2.5-0.5B (agreeing with Hugging Face transformers) and Qwen3.8-27B (agreeing with a reference implementation's golden log-probabilities; 27.4 tokens a second, 39.9 with the model's own draft head, against Ollama's 45.3 in the same benchmark run: [`docs/benchmark.md`](docs/benchmark.md)), and it serves the 27B over an OpenAI-compatible chat endpoint with streaming ([`docs/serving.md`](docs/serving.md)). *Training* (all of it in sessions on a Metal GPU, [`docs/resident.md`](docs/resident.md)): a small network with Adam equal to `torch.optim.Adam`'s (section 12); a GPT of up to 59.95 M parameters trained, served and checked against torch, with a byte-pair vocabulary of 2,048 tokens (sections 15 to 19); a 0.85 M-parameter ResNet that reaches 87.4% test accuracy on real CIFAR-10 without augmentation, against torch's 87.0% (section 18); and Qwen2.5-0.5B and 1.5B fine-tuned with LoRA, and the 0.5B with every parameter, with gradients and loss curves equal to peft's and torch's (sections 20 to 24, 27). The numbers against torch on MPS, with the cases where cancho-gpu is slower, are in [Where it stands](#where-it-stands-against-torch-and-others); what is missing is in [Known gaps](#known-gaps). A ResNet-18-class backbone for cancho-robot's vision is trained to the loss and gradients of torch's but has not been timed (section 26). A fine-tuning experiment on cancho-code's job (M8, [`docs/roadmap-models.md`](docs/roadmap-models.md) section 7) has its design, its training data and its two untuned baselines, which score near zero (0.012 of the oracle's points for the better arm); **no tuned model has been trained for it and no training result is in.** A 4-bit (NVFP4) base for LoRA is built on Qwen2.5-1.5B (QLoRA, M7.5 part A, section 27): the adapters win back 45.6% of the 0.0492-nat cost of the 4-bit base, at the same step time as the f16 base and 1.77 GB less memory; on the 7B (part B, section 28) the forward equals transformers' and a 90-step run takes the held-out loss from 2.1026 to 1.7556 at 5.34 s a step, with no gradient check at that size. *Images:* Qwen-Image-2.1-Turbo generates (1024 x 1024 in about 52 s against torch's 41 s, estimated) and **edits a photograph end to end on the GPU in 102 s against torch's 54 s**, the same edit as torch's on the one photograph compared ([`docs/diffusion.md`](docs/diffusion.md)). **Still open or unproven:** concurrent requests, AMD, and **CUDA beyond one card** (CUDA compiles and its kernels agree with the interpreter on one NVIDIA L4 in Google Cloud, `docs/cloud.md`, but nothing has been timed there except a GEMM and the NVFP4 matvec comparison is not yet run; no machine here has an NVIDIA GPU; Metal has run on Apple Silicon); every measurement is one machine (an Apple M4 Max), and the training comparisons are one run and one seed each. Attention is two kernels over 32 threadgroups for contexts up to 1024 positions in the decode runners (`docs/qwen35.md`; a longer context needs the online-softmax merge of split-KV, which is not built: `docs/kernels.md`), and training runs at a context of 128 (the GPT) or 512 (Qwen).

## What you get

* **Output that was checked against an independent implementation, and is held against numbers now.** CUDA C and Metal for the 433 cases of `tests/cases.txt` (403 emit, 30 are refused) are recorded in `golden/` and held to those files byte for byte (`tests/golden.sh`); they first agreed with the Rust implementation's output (218 cases at the commit in `golden/PROVENANCE`), which was the oracle until `docs/native.md` N2, and are regression goldens since: a change to one is read as text. What a kernel *computes* is held to a float64 reference written from the mathematics for **every numeric kernel family** (`tests/kernels.py`, `docs/chx.md` section 6; each has deliberate changes that its reference must catch, `--mutants`, and the diffusion, VAE and vision kernels have their own in `tests/dit_attention*.py`, `tests/vae_pieces.py` and `tests/vit_pieces.py`), and, on a Mac, to the GPU. A refusal names its rule ([`docs/rules.md`](docs/rules.md)).
* **A reference interpreter.** `--run` interprets a kernel on the CPU (binary32 arithmetic, sums in program order) and prints a summary; the summaries are goldens and agree with the float64 references to 1e-5 (142 cases are recorded with `--run`).
* **A way to drive a GPU.** A C shim and a device program compile the emitted kernel with NVRTC (CUDA) or `newLibraryWithSource` (Metal), upload the interpreter's inputs, launch, and compare ([`docs/device.md`](docs/device.md)).
* **A model reader.** `gguf` and `safetensors` read a header (names, shapes, offsets, metadata) and refuse hostile bytes with a rule tag; `tokenize` is a byte-level BPE tokenizer and `sample` a temperature/top-k/top-p sampler. Each prints a canonical text that an independent implementation prints too: a Python GGUF reader and a Python implementation of the sampler's algorithm (both written for this, both replacing the Rust tools that first made the goldens, and both reproducing those goldens exactly), the `safetensors` library for safetensors. Identical on every fixture and on real files: six GGUF models (0.27 to 5.6 GB), 2,180 + 1,199 safetensors tensors of Qwen3.8-27B. The sampler's picks and ids are identical to the independent implementation's on 2,269 rows and 100% of its probabilities bit-identical on one platform (cancho-gpu's own `exp` is correctly rounded and differs from the platform's `expf` in 0.6% of inputs, by a few ulps). **The tokenizer is held to Hugging Face `tokenizers`, and agrees with it** on the real Qwen2.5 and Qwen3.8 tokenizers: no record differs over 37 hand-written, 4,500 random Unicode-heavy and 800 real prose and code records. Getting there is the finding of issue [#51](https://github.com/alpibrusl/cancho-gpu/issues/51): while it was held to the Rust tokenizer it was written to match, it differed from the library on 3 of 37 and about half of 500 records (NFC without decomposition, a non-greedy newline rule, approximate character classes, the pre-tokenizer not read from the file, added tokens looked for after NFC) ([`docs/runtime.md`](docs/runtime.md)).
* **Decode kernels for an attention layer.** `rope.chx`, `kv_append.chx`, `attention.chx` and `attn_gate.chx` for Qwen3.8-27B's full-attention layers, held to a plain-Python model of what they compute (and, when recorded, identical to the reference compiler's output), and run once on a Metal GPU ([`docs/kernels.md`](docs/kernels.md)).
* **It runs a model.** `cancho-gpu-run` loads a Qwen2 checkpoint, tokenizes a prompt, runs every layer as compiled `.chx` kernels on a Metal GPU and samples the next token: Qwen2.5-0.5B gives the same 12 greedy tokens as transformers in float32 on three prompts, with every top-5 logit within 1e-4, in 6 to 12 ms a token (the spread between runs) ([`docs/inference.md`](docs/inference.md)).
* **It runs a 27B model.** `cancho-gpu-run35` runs Qwen3.8-27B (64 layers, 48 of them gated-delta, NVFP4 weights read from Ollama's store) as `.chx` kernels: on three prompts, eight steps each, the same token, the same top five and logit gaps within 1e-4 of the reference's, and twelve greedy tokens that are the reference runtime's, at 27.4 tokens a second, 39.9 with the model's own draft head (Ollama, which also uses one: 45.3 in the same benchmark run) ([`docs/qwen35.md`](docs/qwen35.md)).
* **It serves a model.** `cancho-gpu-serve` answers `POST /v1/chat/completions` (whole or streamed, with the model's chat template held byte for byte to jinja2 on 23 requests, and thinking mode as `reasoning_content`) from the 27B on the GPU, one request at a time ([`docs/serving.md`](docs/serving.md)).
* **Training on the device.** A session ([`docs/resident.md`](docs/resident.md)) compiles and uploads once and keeps the parameters, the optimiser's state and the data on the GPU: `steps` runs many steps for one request, `carry` feeds an output to the next step, `table` cycles through minibatches (section 12); `alias` lets outputs share a buffer, a rule the session file's checks enforce (21.4); `save` and `checkpoint` write parameters out as a binary file, atomically (the session programs' authority gained `fs_write` for this, by the owner's approval; the compiler's did not change) (section 19). A `resident` or `table` file can be binary (`f32`, `f16`; `i8` for a `resident` 4-bit base) and is then not capped at 16 MiB (a *text* file still is; a resident file larger than the session's byte pool is uploaded in chunks, section 24.3). The surface has `relu`, `step`, `tanh` and the matrix-unit products `mma`, `mma_nn`, `mma_tn` (sections 12.3 and 14); it still has no atomics, no index-valued load and no `max a b` (section 24.1, `docs/roadmap-models.md` section 8.2).
* **Models trained there.** Each is held to a float64 specification written for it and to torch or peft, and each section ends in what it does not show:
  * a **GPT** (pre-LayerNorm, GELU, AdamW) of 3.26 M, 10.8 M, 27.4 M and 59.95 M parameters, on characters and on a 2,048-token byte-pair vocabulary built here, then **served** from a forward-only session with all 576 greedy tokens equal to torch's (roadmap M1 to M3, sections 15 to 17, 19);
  * a **CNN**, a 0.85 M-parameter ResNet with BatchNorm, trained to **87.4% on real CIFAR-10** (no augmentation; torch's 87.0%; the roadmap's 90% is not met by any runtime here) (M4, section 18);
  * **Qwen2.5-0.5B and 1.5B**: the forward pass equal to transformers' (loss within 1.2e-4 and 5.2e-6), LoRA gradients at cosine 0.99986 and 0.99938 or better against peft, 300-step (0.5B) and 150-step (1.5B) LoRA runs whose loss curves equal peft float32's, and **full fine-tuning of the 0.5B** (the embedding table trained, resumable across sessions, a 300-step run against torch float32); the 1.5B full step fits in memory (four steps, no torch comparison); and a **4-bit NVFP4 base** for the 1.5B's LoRA (the new `i8` flag of `resident`, a dequantising kernel): forward 8.6e-5 from transformers' loss, 392 of 392 gradients at cosine 0.997738 or better against peft, a 150-step run 2.3e-4 from peft float32 on the same weights (M7 and M7.5 part A, sections 20 to 24, 27);
  * a **ResNet-18-class backbone** (11,181,632 parameters, at 96 x 128) for cancho-robot's vision, forward and backward equal to torch's, not yet timed (M9 V1, section 26).
  A fine-tuning **experiment** on cancho-code's job (M8) has its design, data pipeline (6,946 records, 2.78 M tokens, a contamination check) and two untuned baselines (`docs/roadmap-models.md` section 7, `docs/m8/`); no tuned model exists yet.
* **Batched decode and a sampler on the device.** `qsw decode` generates for up to 32 sequences a step with the sampled ids kept on the GPU: the same 48 tokens as transformers' greedy generation, 35.6 ms a step at 32 x 1,152 (43.0 ms with LoRA adapters); prefill, LoRA in the step, temperature sampling at 0.99 ms for 32 rows, and a detokeniser that matches Hugging Face's on 461 of 461 ([`docs/decode.md`](docs/decode.md), [`docs/sampler.md`](docs/sampler.md)). It is what [cancho-rl](https://github.com/alpibrusl/cancho-rl) samples with.
* **A complete optimiser.** Global-norm clipping, decay groups, a schedule and gradient accumulation as `.chx` kernels, held to torch and a binary64 version ([`docs/optimizer.md`](docs/optimizer.md)).
* **Images, and editing them.** Qwen-Image-2.1-Turbo (a 7.12 B diffusion transformer, the Qwen3-VL text encoder, a VAE decoder) from a prompt and seed to pixels, in kernels: at 1024 x 1024 the image is 0.16 of 255 from the exact float64 pipeline (49.3 dB; torch's BF16: 4.75, 26.2 dB), and a step takes 6.25 s against torch MPS's 4.99 s (the pipeline about 52 s against 41 s, estimated from the measured stages). **Editing a photograph** (the Qwen3-VL vision tower, the language model reading it, a VAE encoder, and the transformer with the photograph as a cached prefix) runs end to end on the GPU: a real public-domain photograph and "Make it a bright sunny afternoon." come out as the same edit as torch's (final latents 1.0% of their rms apart, 39.1 dB between the images), in **102 s against torch's 54 s** (torch's is a warm process with its 17 s of loading left out; ours includes loading and uploading 30 GB of weights, and with everything already loaded it would be about 70 s, an estimate: D11). It was 150 s: the weights stay on the device between the eight steps through a session request, `reload` (`resident.md` section 29), which took off 36 s with a bit-identical result; the four stages in one process took off 7 s more, and the softmax statistics read the scores once instead of twice (`dit_stats_c` and `dit_stats_comb`, D10) another 5 s. A second photograph of another shape (768 x 1376) with a prompt that changes the scene ran in 118 s and was looked at, not compared with torch. The transformer's step is 7.2 s against torch's 5.6, and the whole difference is two kernels (D11): our products run at 13.0 TFLOPS against torch's 14.4, and our attention takes 70 ms a block against torch's fused 41; the rest of the gap is the cold start of four sessions (about 32 s). One seed and no torch comparison on the second. Video is not started ([`docs/diffusion.md`](docs/diffusion.md), D9).
* **Weight-space editing, with its negative results.** A pre-registered pipeline for editing a model's weights: the published edit costs 0.0205 nats a token (+1.3% perplexity), a gradient search for the loop-fixing edit was not recovered, and the engine side (a greedy runner, the edit as a 262,844-byte patch, agreement to 16,510 positions) runs on Metal ([`docs/weight-space-w4.md`](docs/weight-space-w4.md), [`docs/weight-space-w4-r4.md`](docs/weight-space-w4-r4.md)).
* **An authority you can read.** The compiler performs `args`, `err_write`, `fs_read`, `fs_write`, `heap`, `io_write` and nothing else; the device program is the one that calls foreign code, and says so.

How it is built, what it cost and what the language made hard: [`docs/design.md`](docs/design.md).

## Requirements

- The **cancho** compiler at the commit pinned in `cancho.toml`; the revision is part of the contract.
- Rust, to build that compiler.
- To run the tests: `python3` (fixtures, fuzzing, the kernel reference) and a C compiler (the device plumbing).
- Nothing from the reference runtime. Every golden and `src/unicode_data.cho` are regenerated with `python3` (and `numpy` for the sampler, the `tokenizers` library for the tokenizer) and this compiler; the Rust is only for building the cancho compiler.

## Quick start

```sh
git clone https://github.com/alpibrusl/cancho
git clone https://github.com/alpibrusl/cancho-gpu && cd cancho-gpu

REV=$(sed -n 's/^cancho = "\([0-9a-f]*\)".*/\1/p' cancho.toml)    # the revision CI uses
(cd ../cancho && git checkout "$REV" && cargo build --release -p cancho)
export CANCHO=$PWD/../cancho/target/release/cancho

$CANCHO build                         # build/cancho-gpu
mkdir -p out
build/cancho-gpu emit tests/chx/softmax.chx out r=16 n=256
```

## Examples

**Emit a kernel** for every target the file has a schedule for:

```
$ build/cancho-gpu emit tests/chx/softmax.chx out r=16 n=256
out/softmax_16_256.cu: 256 threads, grid 1x1, 0 B of tiles
out/softmax_16_256.metal: 256 threads, grid 1x1, 0 B of tiles
```

**Check it, and run it on the CPU:**

```
$ build/cancho-gpu check tests/chx/softmax.chx r=4 n=64 --run
nvidia-ada: y [4, 64] = mean 0.0156, peak 0.0259, first [0.009054681, 0.009019844, 0.015272213, 0.016739702]
softmax: nvidia-ada ok
...
```

**Run a kernel on inputs you give it** (the CPU interpreter; `docs/run.md`). `dy` is `1 2 / 3 4` and `w` is `1 0 2 / 0 1 3`, so `dx = dy w`:

```
$ printf '1 2\n3 4\n' > dy; printf '1 0 2\n0 1 3\n' > w
$ build/cancho-gpu run tests/chx/bwd_dx.chx r=2 n=2 k=3 --target nvidia-ada --in dy dy --in w w --out dx
1.0 2.0 8.0 3.0 4.0 18.0
```

**A refusal names its rule:**

```
error[use-after-move]: `x.0` used after it was moved
```

**Read a model's header** (the same text the reference runtime's reader prints; `--sum` adds a checksum of each tensor's bytes):

```
$ build/cancho-gpu gguf tests/gguf/tiny.gguf --sum
gguf v3 tensors=6 kv=31 alignment=32
kv general.architecture str "llama"
kv arr.strs array n=5 [str "a", str "bb", str "ccc", str "", ...]
tensor blk.0.ffn.weight type=Q8_0 dims=[64] bytes=68 sum=40d93799
...
```

**Read a safetensors header, tokenize, sample** (held to the outputs in `golden/`; `docs/native.md` N2f says which are independent):

```
$ build/cancho-gpu safetensors tests/safetensors/mlx-like.safetensors
safetensors header=312 tensors=4 data=224
meta format "mlx"
tensor layer.biases F16 shape=[8,1] range=[144,160]
...
$ build/cancho-gpu tokenize tests/tokenizer/toy-pairs.json tests/tokenizer/corpus.txt
tokenizer ids=862
ids 7: 72 101 399 111 44 723 33
dec "Hello, world!"
...
$ build/cancho-gpu sample tests/sampler/vocab-2.f32 2 1.0 0.95 20 7
row 0 pick 1 nucleus 2: 1=3f6922c8 0=3db6e9c3
...
```

**Run a kernel on a GPU** against the interpreter (needs a GPU, and is not part of the gate):

```sh
CANCHO=$CANCHO scripts/build-device.sh        # CUDA: build/cancho-gpu-device
CANCHO=$CANCHO scripts/build-metal-device.sh  # Metal: build/cancho-gpu-metal-device
# real inputs and the device's whole answer, as `cancho-gpu run` takes them (docs/run.md section 7):
build/cancho-gpu-metal-device run tests/chx/bwd_dx.chx r=2 n=2 k=3 --in dy dy.txt --in w w.txt --out dx
```

**See what it is allowed to do** (checked by `tests/authority.sh`, in the gate):

```sh
$CANCHO authority --std src/*.cho
# performs: args, err_write, fs_read(""), fs_write(""), heap, io_write
# never touches: the network, signals, other programs, foreign code
```

## Checking it

```sh
CANCHO=$CANCHO scripts/gate.sh               # format, file budget, build, unit tests, goldens, refusals, device plumbing (also with `--in` / `--out`), GGUF, safetensors, tokenizer, sampler, kernels, authority
scripts/record-golden.sh                     # golden/ from this compiler; read `git diff golden/` (the numbers are held by tests/kernels.py, not by these files)
scripts/regen-gguf-golden.sh                 # golden/gguf/ from tools/gguf_dump.py (also regen-sampler-golden.sh, regen-safetensors-golden.sh, record-tokenizer-golden.sh)
python3 tests/oracles.py                     # the Python oracles still reproduce their goldens; python3 tests/tokenizer_hf.py: the tokenizer against Hugging Face on real tokenizers (0 differences)
CANCHO=$CANCHO scripts/metal-kernels.sh      # the 9b kernels on a Metal GPU against the interpreter (needs a Mac)
```

`tests/chx/ported/` holds the reference implementation's own `.chx` kernels, copied at the commit in `golden/PROVENANCE`; `tests/chx/` holds programs written here for the lowering paths those do not reach (`softmax.chx` is the first kernel that needs `exp` and `log`).

## Roadmap

The epic is [cancho#251](https://github.com/alpibrusl/cancho/issues/251); each slice is an issue here.

| Slice | | State |
|---|---|---|
| 1–7 | parser, checker, CUDA and Metal emitters, interpreter | **built** |
| 8 | driving a CUDA GPU (shim, device program, mock driver) | **built**; run on one NVIDIA L4 on the suite as it then was (compiles, 176 of 188 cases agree; `docs/cloud.md`); the cases added since, the diffusion kernels among them, have not been run there |
| 9a | runtime core: GGUF, safetensors, BPE tokenizer, sampler ([#5](https://github.com/alpibrusl/cancho-gpu/issues/5)) | **built**; the model files' tensors are read by the runners (`docs/qwen35.md`, `docs/inference.md`) |
| 9b | exp/log in `.chx`, transformer kernels ([#6](https://github.com/alpibrusl/cancho-gpu/issues/6)) | `exp`, `log`, `softmax`, rope, kv_append, attn_gate **built**; attention as one kernel up to 1364 positions, and as two kernels over 32 threadgroups (scores over positions, softmax and `v` over the head dimension) up to the 1024 positions the 27B runner is compiled for; the online-softmax merge for longer contexts is not built |
| run | a model on this GPU: Qwen2.5-0.5B ([`docs/inference.md`](docs/inference.md)) and Qwen3.8-27B ([`docs/qwen35.md`](docs/qwen35.md)) | **both run and agree with their references**; the 27B at 27.4 tokens a second (39.9 speculative), a 1024-position context |
| train | a network trained on the GPU with Adam: sessions with `steps`, `carry` and `table` ([`docs/resident.md`](docs/resident.md) section 12), `opt minimize --method adam` in cancho-opt | **built**; the loss equals a float64 reference's and `torch.optim.Adam`'s to 6 digits after 100 steps; 0.30 ms a step at 4096 x 32 x 32 (torch: 0.37 CPU, 0.87 MPS) |
| M1–M3 | a GPT trained, then served: 3.26 M to 59.95 M parameters, a 2,048-token BPE vocabulary, `save`/`checkpoint` (`resident.md` sections 15 to 17, 19) | **built**; at 59.95 M parameters 167.8 ms a step against torch MPS's 196.7; contexts above 128 and vocabularies above about 2,048 are not |
| M4 | a CIFAR-10 CNN (`resident.md` section 18) | **built, exit not met**: 87.4% on real CIFAR-10 against the 90% asked (torch: 87.0%); 17.6 ms a step against torch MPS's 10.6 |
| M7 | fine-tune a Qwen: forward, LoRA, a 300-step run against peft, 1.5B, full fine-tuning of the 0.5B (`resident.md` sections 20 to 24) | **built**; the speed clause of the exit is met for the 0.5B and 1.5B LoRA runs; the memory clause (under half of torch's) is not met for the 0.5B LoRA run (59% of torch float16 autocast's footprint) and is met narrowly for the 1.5B (48.5% by footprint, 51.3% by torch's driver number) (`resident.md` 22.3, 23.5) |
| M7.5 | a 4-bit base (QLoRA) and a 7B model | part A, the 1.5B prototype, **built and measured** (`resident.md` section 27: all four exits met, the step 1.001 to 1.006 times the f16 base's); part B, **the 7B ran** (`resident.md` section 28: the forward equals transformers' with the loss 8.8e-5 higher, a 90-step QLoRA run takes the held-out loss from 2.1026 to 1.7556 at 5.34 s a step and 21.65 GB; no gradient comparison with peft at that size, no run longer than 90 steps) |
| M8 | does fine-tuning a small model on cancho help cancho-code? (`roadmap-models.md` section 7, `docs/m8/`) | design, data and the two untuned baselines **built and measured**; **no tuned model trained, no result** |
| M9 | vision for cancho-robot: a ResNet-18-class backbone (`roadmap-models.md` section 8, `resident.md` section 26) | V1 **built** (equal to torch, not timed); V2 to V4 open |
| 9c | serving: OpenAI-compatible chat endpoint over the 27B, with streaming and the model's chat template ([`docs/serving.md`](docs/serving.md), [#7](https://github.com/alpibrusl/cancho-gpu/issues/7)) | **built**: one request at a time; batching and a prefix cache open |
| 10 | Metal device path ([#8](https://github.com/alpibrusl/cancho-gpu/issues/8)) | **built and run on Apple Silicon** |
| 11 | AMD ([#9](https://github.com/alpibrusl/cancho-gpu/issues/9)) | open |
| decode | a batched decode, prefill, LoRA in the step and an on-device sampler for cancho-rl ([`docs/decode.md`](docs/decode.md)) | **built**; same tokens as transformers, 35.6 ms a step at 32 x 1,152 |
| G6 | optimiser completeness: clipping, decay groups, schedule, accumulation ([`docs/optimizer.md`](docs/optimizer.md)) | **built**, held to torch |
| diffusion | Qwen-Image-2.1-Turbo: text to image ([`docs/diffusion.md`](docs/diffusion.md)) | **built to 1024 x 1024**, 1.26 times torch's time (estimated); **image editing built** (one real edit end to end: 102 s against torch's 54 s, the same edit); video not started |
| W4 | weight-space editing with gates ([`docs/weight-space-w4.md`](docs/weight-space-w4.md)) | pipeline and engine **built**; the edit searches **not recovered** |
| 12 | scale: model adapters, KV offload ([#10](https://github.com/alpibrusl/cancho-gpu/issues/10)) | open |

## Where it stands against torch and others

Everything below is **[measured]** and copied from the section named in the last column; read that section for the conditions. Common to all rows: **one machine** (an Apple M4 Max, 64 GB), torch 2.12 on MPS (or as marked), **one run and one seed each** (medians of the steps of that run), and torch as it is run by default (not compiled, no MPS graphs). Memory is the process's peak footprint unless marked *driver*, which is what torch reports about its own allocations. A lower time is better; "slower" rows are in the table on purpose.

| Workload | cancho-gpu (Metal) | Reference | Source |
|---|---|---|---|
| **Training: an Adam step, 4096 x 32 x 32, 1,089 parameters** | 0.301 ms | torch CPU 0.372 ms, MPS 0.870 ms | `resident.md` 12.1 |
| An Adam step, 16384 x 64 x 128 | 1.013 ms | torch CPU 1.333 ms, MPS 1.011 ms | `resident.md` 12.1 |
| An epoch of a 64-512-512-512-1 network, 200,000 samples | 0.198 s mixed precision (0.941 s in float32, which section 13 measured at 2.6 times torch MPS's time) | torch MPS float32 0.377 s, autocast 0.457 s; OpenNN CPU 1.550 s | `resident.md` 13, 14 |
| GPT, 10.80 M parameters, 4,096 tokens a step | 37.0 ms | torch MPS float32 41.3 ms, float16 autocast 44.5 ms | `resident.md` 15 |
| GPT, 27.4 M parameters (2,048-token vocabulary) | 81.0 ms | 86.0 ms, 91.7 ms | `resident.md` 17 |
| GPT, 59.95 M parameters | 167.8 ms | 196.7 ms, 201.1 ms | `resident.md` 19.1 |
| GPT decoding, 3.26 M parameters, one sequence | 650 tokens a second | torch MPS 340 (recomputing the window), 273 (key/value cache) | `resident.md` 16 |
| **CNN, 0.85 M-parameter ResNet, batch 128** | 17.6 ms a step: **1.7 times slower**; CIFAR-10 test accuracy 0.8740 | torch MPS float32 10.6 ms, 0.8697; autocast 11.0 ms, 0.8674 | `resident.md` 18.2, 18.1 |
| Qwen2.5-0.5B forward, 4,096 tokens | 382.1 ms; 16.35 GB | torch MPS float32 488 ms, 14.4 GB (*driver*); autocast 474 ms, 15.0 GB (*driver*) | `resident.md` 20 |
| Qwen2.5-0.5B LoRA step (rank 16, all seven projections) | 889.2 ms; 16.14 GB | float16 autocast 1,142.0 ms, 27.48 GB; float32 1,237.9 ms, 31.49 GB | `resident.md` 22.3 |
| Qwen2.5-1.5B forward, 4,096 tokens | 1,116 ms | float32 1,258 ms; autocast 1,188 ms | `resident.md` 23.2 |
| Qwen2.5-1.5B LoRA step | 2,467.3 ms; 19.38 GB | autocast 2,743.8 ms, 39.93 GB; float32 2,980.3 ms, 46.76 GB | `resident.md` 23.5 |
| Qwen2.5-1.5B LoRA step on a 4-bit (NVFP4) base | 2,483.1 ms; 17.52 GB | the same step on an f16 base 2,480.6 ms, 19.29 GB; no torch time (peft ran with activation checkpointing) | `resident.md` 27.5 |
| Qwen2.5-0.5B full fine-tuning step (every parameter) | 1,381.4 ms; 21.03 GB | torch float32 1,450.8 ms; 44.56 GB | `resident.md` 24.4 |
| Qwen2.5-1.5B full fine-tuning step | 4,191 ms; 44.27 GB (four steps) | no comparison: not run | `resident.md` 24.9 |
| ResNet-18-class backbone, 11,181,632 parameters | equal to torch's loss and gradients; **no time measured** | | `resident.md` 26.3, 26.4 |
| **Decoding Qwen3.8-27B** (NVFP4), tokens a second | 27.4; 39.9 with speculative decoding | **Ollama 45.3**, MLX-LM 29.6 | `benchmark.md` |
| Decoding Qwen2.5-0.5B (bf16) | 180.4 | **MLX-LM 347.2**; transformers on the CPU, float32 67.0 | `benchmark.md` |

What the table says, with the numbers the sections state: for the GPT and for Qwen2.5 fine-tuning cancho-gpu takes less time a step than torch on MPS (1.28 times faster than float16 autocast on the 0.5B LoRA step, 1.11 times on the 1.5B's, 1.05 times on the 0.5B's full fine-tuning), with the same loss curves to the digits in the sections, and uses less memory in the three fine-tuning rows that have a torch number (0.5B LoRA, 1.5B LoRA, 0.5B full); the 0.5B *forward* uses more. It is **slower** on the CNN (1.7 times), and its decoding is slower than Ollama's on the 27B (88% of it with speculation) and than MLX-LM's on the 0.5B model. The step-time rows are not a claim about torch tuned (compiled, with graphs), about another machine, or about NVIDIA. The LoRA and full fine-tuning rows are on the repository's own Markdown, Rust and cancho files, a measure of agreement with torch and not of how good a tuned model is.

**Against OpenNN.** [`docs/comparison.md`](docs/comparison.md) is a forward pass of one two-layer sigmoid network, **on a CPU** (a 4-vCPU container), and small: both agree with a binary64 reference to 1.4e-7, OpenNN's call takes about 2.5 microseconds in process, cancho-gpu's CPU interpreter takes milliseconds as two processes, and no ratio is claimed because they measure different things. There is no GPU in it and no training in it. The only training comparison with OpenNN is in `resident.md` sections 13 and 14 and is against OpenNN's **CPU** path (Eigen without BLAS, so not its best; the epoch times of the third row above); **no comparison against OpenNN's GPU path exists**, for the forward pass or for training.

## Known gaps

Assembled from the limits paragraphs of the sections cited; none of these is hidden by the table above.

* **CUDA has run on one NVIDIA L4, and on nothing else.** The CUDA text is generated, held by goldens, run against a mock driver and, in Google Cloud, compiled by NVRTC (253 of 253 kernels) and run against the interpreter on an L4 (176 of 188 cases agree, 0 mismatches; `docs/cloud.md`); no machine here has an NVIDIA GPU, the lowering is Ada's (no Hopper), and there is no AMD path. Every training and fine-tuning number is Metal on one Apple M4 Max (`docs/device.md`, `resident.md` 20, 21.5, 26.4).
* **One run, one seed.** Every comparison with torch is one run each; a difference of 0.01 nats between GPT runtimes is inside what another seed would move (`resident.md` 19.1). The Qwen runs use a held-out set of 7 files and 16,384 tokens, the 1.5B run is 150 steps (not 300), and the full fine-tuning runs are 40 and 300 steps at one untuned learning rate (22.6, 23.6, 24.6). Some step times were measured with another process on the machine, and the sections say which (18.1, 24.8).
* **M4's 90% is not met.** 87.4% on real CIFAR-10 without augmentation, torch's 87.0%; augmentation is not implemented and max-pooling was done as average pooling in the 0.85 M network (`resident.md` 18.1, 18.3).
* **The CNN is slower than torch, and the launch count is why it is likely to stay so without work.** 17.6 ms against 10.6 ms; the step is 186 launches, and the ResNet-18-class network's is **287**; the one fusion the surface allows would remove about 14% of them **[estimate]**, and BatchNorm's statistics cannot be taken in a convolution's epilogue (rule `type`). GPU time of the ResNet-18 backbone is **not measured** (V2 is the milestone that measures), the full-size network does not run on the CPU interpreter, the head is a classifier and not the regression head the pose estimator needs, and no robot or synthetic data exist yet (`resident.md` 18.2, 26.3, 26.4, `roadmap-models.md` 8.3 to 8.5).
* **The vocabulary costs, and has a ceiling.** The embedding is a one-hot product because the surface has no index-valued load, atomics or sorted segments (`error[offset]`): +14% of a GPT step at 2,048 tokens, and about 203 ms of a 1,380 ms Qwen full fine-tuning step (15%) at 151,936. A vocabulary above about 2,048 for the GPT is not built (`resident.md` 17, 24.1, 24.5). The tied embedding means the table's gradient is two products added (24.4).
* **Contexts.** The GPT trains at 128; Qwen at 512 (`t` a multiple of 64, the causal half computed and masked, not skipped); nothing longer was tried. The decode runners stop at 1,024 positions (256 for the 0.5B runner) and split-KV attention is not built (`resident.md` 15, 20; `docs/kernels.md`).
* **M7.3's memory clause is not met for the 0.5B.** The exit asked for the memory under half of torch's with equal curves, or the step within 1.2 times: the step clause is met, the memory is 59% of torch float16 autocast's footprint (51% of float32's). For the 1.5B it is met narrowly (48.5% by footprint, 51.3% by torch's driver number), and only after two changes that removed 3.1 and 5.7 GB (`resident.md` 22.3, 23.4, 23.5). The 0.5B forward pass uses more memory than torch's (16.35 against 14.4 GB, driver) (20).
* **Precision.** The loss scale is a static 16 (a value above 64 is silently wrong on the 0.5B batch: 256 overflows); the f16 gradients make Adam move near-zero rows by rounding noise: layer 0's q `B` is 11.3% off peft's after 300 steps (20.1% after 40), with the held-out loss equal to 6e-5; whether an f32 gradient there would remove it was not tried (`resident.md` 21.1, 21.2, 22.4). The 300-step full fine-tuning curves drift from torch's (held-out loss up to 5.1e-3 apart, against 9e-5 in 40 steps) and the cause is not isolated (24.8).
* **The 1.5B full fine-tune is four steps.** It fits (44.27 GB of a 64 GB machine, thin headroom), with no held-out loss, no comparison with torch and no gradient check at that size (`resident.md` 24.9).
* **The 4-bit base is one model with a full study and one with a short run.** QLoRA is measured on the 1.5B (one seed, 150 steps; the key projection's gradients are the least exact at cosine 0.9977; the quantiser is host code; group size 16 only was run end to end on the GPU; the CUDA session was not touched). 54.4% of the 4-bit base's loss cost remains after 150 steps (still falling slowly). The 7B (section 28) ran 90 steps once, with no gradient comparison with peft, no f16-base session to compare with and no run longer than 90 steps; no torch comparison exists for QLoRA (`resident.md` 27.4 to 27.6, 28.6).
* **M8 has no training result.** The untuned baselines are near zero (arm A, base model with the primer: 0.012 of the oracle's points, 0 builds finished; arm B, no primer: 0.000; Claude Haiku 5.5 on the same items: 0.99), so the tuned arms are read on the compile rate and the refusal tags (the pre-registered floor rule). The only training so far is a 30-step smoke LoRA in peft (not cancho-gpu) to show the data trains (`roadmap-models.md` 7.2, 7.3).
* **Sessions and files.** A text resident file is capped at 16 MiB (binary files are not); a `table` must fit the Metal session's 512 MiB byte pool, so a 300-step Qwen run is five or six sessions handing over the adapters and Adam's moments; the resident weights were uploaded twice until they were shared (`resident.md` 12.4, 22.1, 23.4, 24.3, 24.7).
* **Serving.** One request at a time; the native server serves the 27B only. Decoding is slower than Ollama's and than MLX-LM's on the 0.5B model (above).
* **Language.** No atomics, no index-valued load, no `max a b` (it is expressible as `step` and a product), no strided windows (a stride is a permutation), no generic `A^T B` for `f32` (the matrix-unit `mma_tn` exists), no dynamic loss scaling (`resident.md` 24.1; `roadmap-models.md` 8.2).

## Learn more

| | |
|---|---|
| [docs/design.md](docs/design.md) | how the compiler is built, what it cost, every difference from the Rust and why |
| [docs/training.md](docs/training.md) | whether training belongs here, what `.chx` can express of a backward pass (four kernels built), the plan |
| [docs/resident.md](docs/resident.md) | sessions: compile and upload once, one line per evaluation, on the CPU interpreter and on Metal; training on the device (sections 12 to 14: Adam, minibatches, binary files, mixed precision), a GPT (15 to 17, 19), a CNN (18), `save` and `checkpoint` (19), Qwen2.5 forward, LoRA and full fine-tuning against transformers, peft and torch (20 to 24), a design for a 4-bit base and 7B (25), and a ResNet-18-class backbone (26) |
| [docs/roadmap-models.md](docs/roadmap-models.md) | training and serving real models: the gaps in the language and the runners, the milestones M1 to M7 with what each reached, the design and baselines of M8 (fine-tuning for cancho-code) and of M9 (vision for cancho-robot) |
| [docs/m8/](docs/m8/MANIFEST.json) | the M8 training data's manifests: the 14 repositories' commits, the mix, the packed arrays, the contamination check and exclusions, the smoke run |
| [docs/native.md](docs/native.md) | why the CPU path is 130 times slower than the language (a software f32), the swap for the real `f32` type, and N1 built (the CPU path 23 times faster, goldens byte-identical), and N2, which ended the Rust oracle (`chx.md`, 41 numeric references, recorded goldens) |
| [docs/run.md](docs/run.md) | `cancho-gpu run`: a kernel on the CPU interpreter with inputs from files and its whole output printed; designed, built and measured |
| [docs/comparison.md](docs/comparison.md) | compared with OpenNN: a two-layer network's forward pass on a CPU, outputs and timings, and where this project is worse |
| [docs/decode.md](docs/decode.md), [docs/sampler.md](docs/sampler.md) | a batched decode of Qwen2.5-1.5B on the device and its sampler, the numbers and what they are held to |
| [docs/optimizer.md](docs/optimizer.md) | clipping, decay groups, the schedule and accumulation, held to torch |
| [docs/diffusion.md](docs/diffusion.md) | Qwen-Image-2.1-Turbo: design, every stage against float64 and torch, text to image and photo editing, the timings at 256 and 1024 and where an edit's 102 s go |
| [docs/weight-space.md](docs/weight-space.md), [docs/weight-space-w4.md](docs/weight-space-w4.md) | editing a model's weights: the gates, the results and the negative ones |
| [docs/rules.md](docs/rules.md) | every refusal rule and what triggers it |
| [docs/device.md](docs/device.md) | driving a GPU: the shim, the authority it reports, the mock, what stays unproven |
| [docs/runtime.md](docs/runtime.md) | reading a model: the GGUF and safetensors readers, the tokenizer and the sampler, where each differs from its oracle, how each is held to it (and where the tokenizer disagrees with Hugging Face) |
| [docs/inference.md](docs/inference.md) | running Qwen2.5-0.5B on a Metal GPU: what is on the GPU and what is not, how it is checked against transformers, speed, limits |
| [docs/qwen35.md](docs/qwen35.md) | running Qwen3.8-27B: the layers, the NVFP4 and gated-delta kernels, the weights from Ollama's store, how it is checked against the reference's golden log-probabilities, speed |
| [docs/benchmark.md](docs/benchmark.md) | cancho-gpu against MLX-LM, Ollama and transformers on the same checkpoints (`bench/bench.py`): what is measured, the numbers, and why not to compare them across runs |
| [docs/kernels.md](docs/kernels.md) | the decode kernels of an attention layer, what they are checked against, the context limit, why split-KV with its merge cannot be written yet, and the two-kernel attention the runner uses for speed |
| [The project page](https://alpibrusl.github.io/cancho-gpu/) | the summary |

## Layout

```
src/               the compiler (lexer, parser, elaboration, checker, lowering, interpreter) and the model reader (gguf, safetensors, bpe, sampler, unicode)
device/            the device programs (CUDA, Metal): the only code that calls the shim
shim/              canchogpu.c (CUDA, over dlopen) and canchogpu-metal.mm (Metal)
tests/             goldens, refusal fixtures, the mock driver, fixtures, fuzzing and references for each reader and kernel
golden/            what this compiler wrote for every case, recorded (and, in subdirectories, what each reader's oracle prints)
tools/             the oracle programs and the table generator (all Python: GGUF, sampler, safetensors, the tokenizer through Hugging Face, the Unicode tables)
scripts/           gate, golden recording, device builds
examples/          the models trained in sessions (train: MLPs, gpt, cnn: CIFAR-10 and the ResNet-18-class backbone, qwen: forward, LoRA, full fine-tuning, M8's data pipeline), each with its numpy specification, its session writer and its comparison with torch
bench/             the benchmark against MLX-LM, Ollama and transformers, and the OpenNN comparison
docs/              design, rules, device, runtime, the training sections (resident.md, roadmap-models.md, docs/m8/), and the project page
```

## Limitations

CUDA is verified against a mock driver, not a GPU: no kernel here has run on NVIDIA hardware, and the lowering is Ada's (no Hopper). There is no AMD path. Metal's speed is measured (`docs/benchmark.md`) and is 88% of Ollama's on the 27B with speculative decoding (60% without); the three-row weight kernel of speculative decoding is ALU-bound in f32 and closing the rest needs cheaper arithmetic (`docs/native.md` N4d). The runners serve one request at a time with a 1024-position context (256 for the 0.5B model). The tokenizer and sampler narrow to what the reference runtime does (a fixed split pattern, `qwen35` GGUFs only); the sampler's probabilities agree with its to a few ulps, not bit for bit; `attention.chx` is for head dimension 256. Training runs only in sessions on a Metal GPU, and the [Known gaps](#known-gaps) above are the limits of what was trained (one machine, one seed, a 128 or 512 context, no CUDA hardware run, the CNN slower than torch, no tuned model for M8). The surface has `relu`, `step` and `tanh` and the matrix-unit products, but no atomics, no index-valued load and no `max a b` (`docs/resident.md` 24.1). A *text* resident file is capped at 16 MiB (and so is a `--in` file of `cancho-gpu run`, `docs/run.md`); binary `f32`/`f16` resident and table files are not (`docs/resident.md` 12.4).

## History

The `.chx` kernel language began as [lex-gpu](https://github.com/alpibrusl/lex-gpu), a Rust compiler and inference runtime. cancho-gpu is the same language written in cancho, as the first compiler-sized program in it (cancho#251). For its first phase lex-gpu was kept as the **test oracle**: the goldens and the model-reading oracles came from it, never from this code, and every `.chx` case agreed with it byte for byte (218 cases, the commit in `golden/PROVENANCE`). That ended with [`docs/native.md`](docs/native.md) N2: the language is specified here ([`docs/chx.md`](docs/chx.md)), numbers are held to independent references, the goldens are regression records, and the sum order and the CPU path are this project's to choose. The readers' oracles were replaced by independent ones in N2f (a Python GGUF reader, Hugging Face `tokenizers`, a Python port of the sampler, Python's Unicode tables). Nothing here depends on lex-gpu or on Rust at run time.

## Contributing

Every change goes through `scripts/gate.sh` (the same as CI). A new lowering path gets a case in `tests/cases.txt` (recorded with `scripts/record-golden.sh`, its diff read) and, if it computes numbers, a float64 reference in `tests/kernels.py`; every refusal gets a rule tag and a fixture. Design before code, in `docs/`, with claims measured; a claim that turns out false is corrected in place. See [`CLAUDE.md`](CLAUDE.md).

## Licence

[EUPL-1.2](LICENSE).
