# Kolibri 1.1 on the GPU (design, 2026-11-25; K0 done: the model is read and a reference written from its code continues a prompt sensibly; nothing runs on the GPU yet)

What was **read** (the GGUF file's header and tensors with this repository's reader, the model's reference code, the converter that wrote the file) is marked **[read]**; what was run, **[measured]**, on one Apple M4 Max with 64 GB; arithmetic or judgment, **[reckoned]**. A claim that turns out false is corrected in place.

The model is `Aleph-Alpha/Kolibri-1.1`, from its 4-bit file `Kolibri-1.1-Q4_0.gguf` of `Aleph-Alpha/Kolibri-1.1-GGUF` (42,631,619,008 bytes; its SHA-256 is the repository's listing's). The line of work is the one `docs/audio.md` followed for Qwen3-ASR and Qwen3-TTS: a reference first, a design before each stage's code, each stage held to numbers with mutants.

## 1. What it is **[read]**

| | |
|---|---|
| layers | 50, 2,560 wide; RMS norms (eps 1e-6) before and after the attention and before and after the MLP |
| attention | 48 query heads over 4 key/value heads of 128; an RMS norm with a weight on each q and k head; scale 1/sqrt(128) |
| 40 "sliding" layers | rotary positions (base 10,000), a causal window of 513 keys |
| 10 "indexed" layers (4, 9, 14, ... 49) | no positions at all; keys and values rounded to NVFP4 (groups of 16, an E4M3 scale); a query attends to all visible keys while there are at most 2,184 of them, and past that to a selection (2,048 chosen by score from a pool, the first 8 and the last 128) |
| shared keys and values | 6 of the indexed layers have no k and v of their own: 9, 14 and 49 read layer 4's, 34, 39 and 44 read layer 29's |
| MLP of layers 0 and 1 | SwiGLU of 3,584 |
| MLP of layers 2 to 49 | 384 experts of 512 (SwiGLU), 6 chosen a token, plus one shared expert of 512 that every token takes. The router is a product to 384 logits; the 6 largest of `logits + bias` are chosen and each weighs `sigmoid(logit)`, not normalised |
| vocabulary | 128,000; byte-level BPE with Qwen2's split; the end token 127,906 |
| the file's formats | Q4_0 (blocks of 32 values: an f16 scale and 16 bytes, value `(nibble - 8) scale`) for everything but: the norms, the routers and their biases (F32); the `down` matrices of four MoE layers (the experts' and the shared one's) and of the two dense MLPs (Q4_1: `nibble scale + offset`); the output head (Q6_K) |

72.5 billion of the weights are the experts' (40.8 GB of the file); a token uses 6 of 384 in each of 48 layers, about 2.9 billion weights with the rest.

## 2. K0: a reference written from the model's code **[measured]**

There is no implementation here to compare with. llama.cpp reads this file only from a branch that is not merged, and it was not built. So the reference is `ref.py`, by hand and not in the repository (`~/.cache/cancho-media/kolibri-ref/`): about 150 lines of numpy written from `modeling_kolibri1_1.py` and ggml's three block formats, a token at a time with caches of the keys and values, the dense weights dequantised once and an expert's on demand.

* **It is not the vendor's implementation, and the same reader wrote it who will write the kernels.** A misreading of the model's code would be in both. What stands against that: the text.
* **"The capital of France is"** continues **" Paris.  \\nQuestion: What is the"** (greedy; the first choice 3.8 ahead of the second). **"Berlin ist die Hauptstadt von Deutschland. Die Stadt liegt an der Spree und hat"** continues **" rund 3,7 Millionen Einwohner. Berlin ist bekannt für seine Geschichte, seine Kultur und seine vielen Sehenswürdigkeiten wie das Brandenburger"**, 24 tokens. A model with a layer read wrong does not write this.
* **Kept for the stages:** for each of the 39 tokens of the German run and the 13 of the other, the stream after every layer (`stream.f32`, `[tokens, 50, 2560]`), the row after the final norm, the six experts each layer chose and their weights (`moe.json`), and the logits of every greedy choice (`logits.f32`).
* **Every expert is used**: 383 of the 384 are chosen somewhere in the 39 tokens' 1,872 choices. The six weights run from 0.004 to 0.98.
* **Time:** 7 s to dequantise the dense weights, then about 1.5 s a token on the CPU.
* **Not checked by it:** a context past 2,184 keys (the indexer is not written; the reference stops there), the chat template, sampling.

## 3. A decision measured before the design: the dense weights cannot be f16

Qwen's weights are BF16 and exact in f16, and every stage of `docs/audio.md` ran its products on f16 weights. Here a weight is a 4-bit number times an f16 scale: up to 15 significant bits where f16 has 11.

**[measured]** The reference with every matrix but the experts' rounded to f16, on the German run's 25 greedy choices: **the logits move by up to 0.79 (0.48 among a row's 50 largest; the largest logit is 22.9), and one of the 25 first choices changes.** So no weight of this model is rounded to f16: they stay 4-bit on the device or are f32.

## 4. What exists and what is missing **[read from `tests/chx/`, `models/`, `src/`]**

| piece | have | missing |
|---|---|---|
| a 4-bit weight read in a kernel | `dequant_fp4`, for NVFP4 (a table of 16 values, an E4M3 scale) | **Q4_0**: `(nibble - 8) scale` with an f16 scale a group of 32. One operation of the language, K1 |
| a product with 4-bit weights | `matvec_fp4_rows` and its family, fused into the product's reduction for NVFP4 | the same for Q4_0. First as two kernels (dequantise to an f32 scratch, then the f32 product); fused later if the time asks for it |
| choosing 6 experts of 384 | S3's `samp_rank` (how many entries are larger) gives the six | **reading a window of a weight file chosen by a value**: a load's offset is an index or a constant (`error[offset]`, measured). Not in the kernel language: in the session, which binds a program's input at an offset it reads from a buffer, checked against the file. K2 |
| Q4_1 (`nibble scale + offset`) | | no second operation: `(nibble - 8) scale + (offset + 8 scale)`, the second term a product of 16 or 112 columns with the sums of the input's groups |
| the attention | the cached step's kernels take the head counts as constants (`attention_scores_b`, `attention_pv_b`, `kv_write_b`) | `dec_post_n` is written for 16 + 8 + 8 heads: a form for 48 + 4 + 4, and one with no rotation |
| the window of 513 | a mask, as S1c's of 72 | |
| NVFP4 rounding of the indexed layers' keys and values | | a kernel of existing operations: the group's largest magnitude (`rowmax`), the E4M3 scale and the E2M1 magnitude as sums of `step` against a table of thresholds the host writes. Ties at a threshold go down, where the reference rounds to even |
| shared keys and values | a `link` in a model file | |
| the tokenizer | `cancho-gpu tokenize` reads a GGUF, but refuses this one: `only the qwen35 pre-tokenizer has been checked` | Qwen2's split is already in `src/bpe.cho` for `tokenizer.json`; the GGUF path has to take it, and be held to the reference tokenizer's ids |
| the head | | Q6_K decoded by the host into f32 (328 M values, 1.3 GB) |
| the sampler | S3's (temperature, the k largest, the penalty) | top-p, which the file's defaults name (0.99); the chat template |

## 5. Memory and time **[reckoned]**

* **The experts stay 4-bit: 40.8 GB.** The other matrices as f32 are 10.2 GB (per layer 38 M values; the embedding and the head 328 M each): **51 GB**, under Metal's 55.7 GB working set on this machine but with little room. Kept 4-bit and dequantised in the step they would be the file's 42.6 GB and cost about 150 MB of writes a layer. Which of the two is decided in K4 by running it, not here.
* **A token reads about 2.9 billion weights.** Dequantised to a scratch and then multiplied, the experts alone are 6 x 3 x 1.3 M values a layer, 94 MB written, 4.5 GB a token: tens of milliseconds at this machine's memory speed. With about 2,000 programs' launches (section 27 of `docs/audio.md`: 12 microseconds each) a token is **50 to 100 ms, 10 to 20 tokens a second, unverified**.

## 6. Stages, each a gate

* **K0. The reference** **[done, section 2]**.
* **K1. `dequant_q4` in the language**: the operation, its float64 test and mutants, the kernels `dq_q4` (a window of 4-bit weights to an f32 scratch) and the product after it, against the reference's own dequantisation of one expert, to the bit. Refused inside a product until it is fused, with a rule tag.
* **K2. A window chosen by a value, in the session**: a directive that binds an input at `index x stride` of a resident file, the index read from a buffer the step computed; the CPU session and the Metal one agree; an index outside the file is a refusal with a rule tag, never a read.
* **K3. One MoE layer**: the router, the six, the experts and the shared one, from the reference's row before the layer to its row after it; then the dense MLP of layers 0 and 1.
* **K4. One attention layer of each kind** (sliding with rotary; indexed with NVFP4 keys and values; one that reads another layer's), then **the 50 layers as one cached step**, teacher-forced on the reference's ids: the stream after every layer and the logits. Bound, as `docs/audio.md` S2: the arg max at every row whose two best are clear, and the differences over the 50 largest reported.
* **K5. The host**: the weights from the GGUF (the nibbles repacked, the scales, Q6_K, Q4_1's second term), the tokenizer, the prompt.
* **K6. Generation** with the sampler, stopping at the end token (`scripts/tts-session until`'s way), and the time a token.
* **K7. A command.** Which one (`scripts/media`, or the server of `docs/serving.md`) is decided when K6 runs.

**Not in this plan:** contexts past 2,184 tokens (the indexer and its pool), more than one sequence, tools, the vendor's FP8 checkpoint, any other quantisation of the file.

## 7. What needs a yes before it happens

* **Building llama.cpp's unmerged branch** for a second reference. Not done; the stages above do not wait for it, but until some implementation that is not this reader's agrees, "the reference" means section 2's.
* **Nothing else is downloaded.** The 42.6 GB file is in `~/.cache/cancho-media/kolibri-1.1-gguf/`.
