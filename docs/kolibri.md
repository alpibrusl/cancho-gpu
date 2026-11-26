# Kolibri 1.1 on the GPU (design, 2026-11-25; K0 done: the model is read and a reference written from its code continues a prompt sensibly; K1 done: its 4-bit weights are read in a kernel, one expert's matrix to the bit on the Metal GPU; K2 done: a session reads the experts a value names, six of a layer's 384 on the Metal GPU; K3 done: the MLP half of each of the 50 layers, the mixture of experts with its router, runs on the Metal GPU within 2.1e-6 of the reference; K4 done: the 50 layers run as one cached step on the Metal GPU, teacher-forced on the reference's 39 tokens: the stream within 8.5e-6, the logits within 3.9e-6, every first choice the reference's, 77 ms a token; K5 done: the host is a cancho program, its 41 GB of weight files the scratch script's to the byte, the tokenizer reads the model's file, and a text's next token on the device is the reference's; K6a done: the step feeds itself, and from both prompts the device's greedy tokens are the reference's, all 24 and all 8, at 74 ms a token; K6b done: the sampler (temperature, the 128 largest, top-p) is on the device, its 90 draws of two seeded runs the float64 definition's, 62 ms a token; no context past 640 positions, no chat template, no single command)

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
| a product with 4-bit weights | `matvec_fp4_rows` and its family, fused into the product's reduction for NVFP4 | the same for Q4_0. First as two kernels (dequantise to an f32 scratch, then the f32 product); fused later if the time asks for it **[corrected in section 8: the product takes the lazy dequantisation as it is, with no scratch, from the first day; what NVFP4 has and this has not is a path of its own in the reduction]** |
| choosing 6 experts of 384 | S3's `samp_rank` (how many entries are larger) gives the six | **reading a window of a weight file chosen by a value**: a load's offset is an index or a constant (`error[offset]`, measured). Not in the kernel language: in the session, which binds a program's input at an offset it reads from a buffer, checked against the file. K2 |
| Q4_1 (`nibble scale + offset`) | | no second operation: `(nibble - 8) scale + (offset + 8 scale)`, the second term a product of 16 or 112 columns with the sums of the input's groups |
| the attention | the cached step's kernels take the head counts as constants (`attention_scores_b`, `attention_pv_b`, `kv_write_b`) | `dec_post_n` is written for 16 + 8 + 8 heads: a form for 48 + 4 + 4, and one with no rotation |
| the window of 513 | a mask, as S1c's of 72 | |
| NVFP4 rounding of the indexed layers' keys and values | | a kernel of existing operations: the group's largest magnitude (`rowmax`), the E4M3 scale and the E2M1 magnitude as sums of `step` against a table of thresholds the host writes. Ties at a threshold go down, where the reference rounds to even **[corrected in section 11: `fp4_round` rounds ties to even as the reference does, each threshold written the way its tie goes]** |
| shared keys and values | a `link` in a model file | |
| the tokenizer | `cancho-gpu tokenize` reads a GGUF, but refuses this one: `only the qwen35 pre-tokenizer has been checked` | Qwen2's split is already in `src/bpe.cho` for `tokenizer.json`; the GGUF path has to take it, and be held to the reference tokenizer's ids |
| the head | | Q6_K decoded by the host into f32 (328 M values, 1.3 GB) |
| the sampler | S3's (temperature, the k largest, the penalty) | top-p, which the file's defaults name (0.99); the chat template |

## 5. Memory and time **[reckoned]**

* **The experts stay 4-bit: 40.8 GB.** The other matrices as f32 are 10.2 GB (per layer 38 M values; the embedding and the head 328 M each): **51 GB**, under Metal's 55.7 GB working set on this machine but with little room. Kept 4-bit and dequantised in the step they would be the file's 42.6 GB and cost about 150 MB of writes a layer. Which of the two is decided in K4 by running it, not here. **[K4: kept 4-bit, 41 GB of files on the device; it ran]**
* **A token reads about 2.9 billion weights.** Dequantised to a scratch and then multiplied, the experts alone are 6 x 3 x 1.3 M values a layer, 94 MB written, 4.5 GB a token: tens of milliseconds at this machine's memory speed. With about 2,000 programs' launches (section 27 of `docs/audio.md`: 12 microseconds each) a token is **50 to 100 ms, 10 to 20 tokens a second, unverified** **[measured in section 11: 77 ms a step with the head, 13 a second, before the embedding, the arg max and the sampler are on the device]**.

## 6. Stages, each a gate

* **K0. The reference** **[done, section 2]**.
* **K1. `dequant_q4` in the language** **[done, section 8]**: the operation, its float64 test and mutants, the kernels `dq_q4` (a window of 4-bit weights to an f32 scratch) and the product after it, against the reference's own dequantisation of one expert, to the bit. Refused inside a product until it is fused, with a rule tag. **[corrected in section 8: not refused; a product takes it]**
* **K2. A window chosen by a value, in the session** **[done, section 9]**: a directive that binds an input at `index x stride` of a resident file, the index read from a buffer the step computed; the CPU session and the Metal one agree; an index outside the file is a refusal with a rule tag, never a read.
* **K3. One MoE layer** **[done, section 10]**: the router, the six, the experts and the shared one, from the reference's row before the layer to its row after it; then the dense MLP of layers 0 and 1.
* **K4. One attention layer of each kind** **[done, section 11]** (sliding with rotary; indexed with NVFP4 keys and values; one that reads another layer's), then **the 50 layers as one cached step**, teacher-forced on the reference's ids: the stream after every layer and the logits. Bound, as `docs/audio.md` S2: the arg max at every row whose two best are clear, and the differences over the 50 largest reported.
* **K5. The host** **[done, section 12]**: the weights from the GGUF (the nibbles repacked, the scales, Q6_K, Q4_1's second term), the tokenizer, the prompt.
* **K6. Generation** **[split in section 13: K6a, the step feeding itself greedily, is done; the sampler, K6b, is done in section 14; contexts past 640 positions are K6c]** with the sampler, stopping at the end token (`scripts/tts-session until`'s way), and the time a token.
* **K7. A command.** Which one (`scripts/media`, or the server of `docs/serving.md`) is decided when K6 runs.

**Not in this plan:** contexts past 2,184 tokens (the indexer and its pool), more than one sequence, tools, the vendor's FP8 checkpoint, any other quantisation of the file.

## 7. What needs a yes before it happens

* **Building llama.cpp's unmerged branch** for a second reference. Not done; the stages above do not wait for it, but until some implementation that is not this reader's agrees, "the reference" means section 2's.
* **Nothing else is downloaded.** The 42.6 GB file is in `~/.cache/cancho-media/kolibri-1.1-gguf/`.

## 8. K1: `dequant_q4` **[measured, 2026-11-25, M4 Max]**

`dequant_q4 q s g G` in the kernel language (`docs/chx.md`): `(code - 8) * s[group] * g[row]`, with `q: i8[n, k/2]` (two codes a byte, low nibble first), `s: f16[n, k/G]`, `g: f32[n]`. ggml's Q4_0 is `G` 32 and `g` ones. It is `dequant_fp4`'s node with another table and an f16 scale: the same operands, shapes and laziness, about 40 lines in the parser, the elaboration, the checker, the interpreter and the lowering. **No recorded golden of any other kernel changed.**

Two kernels: `dq_q4` (a window of 4-bit weights to an f32 matrix) and `matvec_q4_rows` (`matvec_fp4_rows` with the new operation).

**One expert of the file** (layer 10's expert 137, the `gate` matrix, 512 x 2,560; the host's part done by hand: a block's 16 low nibbles then 16 high ones repacked two a byte in the matrix's order, the f16 scales as they are):

| | the CPU session | the Metal session |
|---|---|---|
| `dq_q4`'s 1,310,720 values against the reference's dequantisation | **equal to the bit** | **equal to the bit** |
| `dq_q4` then the f32 product, against the float64 product (peak 0.156) | 2.5e-7 | 3.0e-8 |
| `matvec_q4_rows`, the dequantisation inside the product | 2.5e-7 | 2.3e-8 |

* **The design was wrong about the product, in the right direction.** Section 6 planned a scratch matrix and a refusal inside a product. A product already takes any lazy tile element by element, and takes this one: `matvec_q4_rows` reads 0.74 MB where the scratch form writes and reads 5 MB. What it has not is NVFP4's own path in the reduction (wide loads, a scale formed once a group).
* **Time, one matrix of 512 x 2,560, from requests of 20,000 and 100,000 steps on the Metal GPU:** the fused product about 3 microseconds a step, a plain f32 product of the same matrix 7, the scratch form 18. **These are single runs of a session that does nothing else and the first two are near the clock's noise; what they say is that an expert's product is microseconds, under a program's launch (12).** A token's 864 expert products are then launches, about 10 ms **[reckoned]**.
* **Held to float64** (`tests/kernels.py`, in the gate): both kernels, two sizes each. **Mutants, by hand: five, four killed there.** The fifth (every tile's codes read from the first tile) cannot be seen with the interpreter's made-up bytes, `0x30 + i % 8`, which are the same in every tile of 16 bytes: **`tests/q4.sh`, new and in the gate, runs both kernels on codes that do not repeat, against exact arithmetic, and kills it.**
* **On the Metal GPU against the interpreter** (`scripts/metal-kernels.sh`'s cases, by hand): `dq_q4` 1.2e-7 of a peak of 1.98 with rows' scales that are not ones (three factors, one rounding apart); the products 1.3e-6, 8.4e-7 and 1.5e-6 of their peaks at 512 x 2,560, 2,560 x 512 and 6,144 x 2,560.
* **A refusal:** scales that are bytes (`error[type]: q4 values must be I8 and their scales F16`, `tests/reject/type-q4.chx`).
* **Not done:** Q4_1 and Q6_K (the host's, K5); the host's repacking is a scratch script here, not a program; CUDA and HIP texts are emitted and held by goldens but ran nowhere; no path of its own in the reduction, so a large product (the attention's 6,144 x 2,560) is the plain element-by-element form, whose time against NVFP4's is not measured.

## 9. K2: a table picked by a value **[measured, 2026-11-25, M4 Max]**

**The design, as it was found in the sources before the code:** a session already binds a parameter at an offset of a larger buffer. That is a `table` (`docs/resident.md` section 12): a file of blocks of which repetition `j` reads block `j mod S`, bound on Metal with `cgpu_arg_at`. Choosing an expert is the same thing with another rule for the block. So K2 is not a new directive but a clause:

```
table P.PARAM FILE BLOCKS [f32|f16|i8] by Q.OUT SLOT
```

The block is the whole number in slot `SLOT` of `Q`'s f32 output, read when `P` is about to run; `Q` is listed before `P`. **A value that is not a block of the table (past it, negative, a fraction, 1e30) is `error[session]` and nothing is read.** `src/session.cho` (the clause, its checks, the CPU step) and `device/metal-session-main.cho` (the read-back and the offset), about 150 lines.

* **On the device the value is read back**: one wait and one download for an output in a repetition, however many tables it picks (a layer's 18 share one). The kernels stay as they are: an expert's product is `matvec_q4_rows` on a parameter of 512 x 2,560, and never knows there are 384.
* **Binary tables of the same file are one buffer**, as resident files already were: a layer's six programs pick from one copy of its 384 experts, not six.
* **`i8` tables**, which the 4-bit codes need.

**Six experts of the file** (layer 10's `gate` matrices, all 384 as one table of 251 MB of codes and one of 31 MB of scales; the six the reference chose for token 20 of the German run, 203, 253, 118, 11, 240 and 367, given as a kernel's output; one input row; the Metal session):

| | |
|---|---|
| the six products against float64 (peaks 0.14 to 0.23) | worst **3.6e-8**, each |
| the session's whole process, with its upload of 283 MB | 0.07 s |

**`tests/pick.sh`** (16 checks; the CPU session in the gate, the Metal session by hand: all 16 on both): three programs each adding the block a slot names; the slot read and not the first value; a picked table beside one that steps; picks that change from step to step through a carried sum; a table of codes and one of scales picked together into `dq_q4`, both blocks exact; the four values that are not a block, refused; six refusals of the session file (picked by a later program, a slot past the output, by a parameter that is only read, by a program that is not there, a slot that is not whole, no slot).

**Mutants, by hand, each killed by that test:** the CPU session reading block 0 whatever the value (5 checks fail), reading the output's first value and not the slot (4), reading the last block for a value past the table (1), and the Metal session keeping the first repetition's values (the stepped case fails).

* **A crash on the way, fixed before any of this was measured:** the first form decoded the value into the interpreter's memory, where a device session's outputs have no place (their offset is -1); with one program it happened to run and with two it trapped. The value is now read from the downloaded bytes. The stepped case and the three-program case of `tests/pick.sh` on the Metal session are what hold it; they are by hand, so **nothing in the gate runs the device's read-back** (the gate has no GPU, and its mock driver computes nothing a pick could read).
* **Not done:** the time of the read-back in a real step (48 waits a token) is not measured: K4's; CUDA's and the other device sessions do not have the clause; a picked table is uploaded whole, so a table over the session's 512 MiB staging pool is refused as any binary table is (an expert file is 251 MB).

## 10. K3: a layer's MLP half, the mixture of experts and the dense one **[measured, 2026-11-25, M4 Max]**

**The design, before the code.** The half is from the row after a layer's attention (`mid`, which the reference of section 2 did not keep: its scratch copy now does, with the router's logits and the MLP's output before its norm, in `r3/`) to the row after the layer. It is a model file, `models/kolibri_mlp.model`, of kernels that exist and four that do not:

| step | kernel |
|---|---|
| the norm before | `rmsnorm_rows` |
| the router: 384 logits, f32 weights | `matvec_rows` |
| plus the biases | `step_add` |
| each entry's place among them | **`moe_rank`**: `samp_rank` with the ties broken by index, as the reference's stable sort breaks them. With `samp_rank` itself two equal logits would share a place, a slot would hold the sum of two indices and the next slot none: a wrong expert read without a refusal |
| the six indices and `sigmoid(logit)` of each | **`moe_pick`** |
| an expert: `gate` and `up` (512 x 2,560), the activation, `down` (2,560 x 512) | `matvec_q4_rows` twice, `swiglu`, `matvec_q4_rows`; the three matrices' codes and scales are tables of 384 blocks picked `by` the index's slot (K2) |
| the layer's sum | **`axpy_slot`**: `m + wt[j] y`, starting from the shared expert's output |
| a `down` matrix in Q4_1 (layers 0 to 5) | `code d + m` is `(code - 8) d + (m + 8 d)`: the first term is the Q4_0 product, the second is constant over a group of 32 columns, so it is **`group_sum`** (the activation's 16 or 112 group sums) and a `matvec_res` with the offsets `m + 8 d` as an f32 matrix of 2,560 x 16 (or x 112), which the host writes |
| the norm after and the residual | `rmsnorm_rows`, `step_add` |

* **Correct first, with the fewest new kernels**: an expert is five programs, a layer about 40, a token about 1,900 launches (23 ms at section 27 of `docs/audio.md`'s 12 microseconds) **[reckoned]** **[measured below as a request of its own: 1.2 ms a layer, 57 ms a token's 50 halves, which is over this]**. `gate`, `up` and the activation as one kernel, and `down` with its weight, would make it three; that is K6's, when a token is timed.
* **The offsets are not exact**: `m + 8 d` of two f16 values is rounded to f32 when their exponents are far apart, so a Q4_1 weight read this way can differ from the reference's `code d + m` in its last bit. Measured below, not assumed away.
* **The router is f32 here and float64 in the reference.** The six agree unless the sixth and seventh are closer than the products' rounding; the margin is measured on every choice of the run.
* **The bound**: the row after the layer against the reference's, relative to the row's peak, at 1e-4; the six indices the reference's own.
* **In the gate** (no GPU, no model): the four kernels against float64 (`tests/kernels.py`) with mutants, and `tests/moe.sh`, a small layer of the model file on the CPU session against awk's arithmetic, ties included.

**Every layer's half, every token of the German run** (the weights by hand from the file, a scratch script; the Metal session, a process a half):

| | |
|---|---|
| halves run: 50 layers x 39 tokens | 1,950 |
| the row after the layer against the reference's, of the row's peak | worst **2.1e-6** (bound 1e-4) |
| the MLP's output before its norm | worst 4.9e-7 |
| the six experts, in order, of 1,872 choices | **the reference's, 1,872 of 1,872** |
| their weights | within 9.4e-8 |
| the router's logits against the reference's float64 | within 1.8e-6 |
| layers 0 and 1 (dense, `down` in Q4_1) and 2 to 5 (experts' `down` in Q4_1) | within 5.7e-7: the offsets as `m + 8 d` cost nothing that shows |

* **The six agreed, but not by a wide margin everywhere**: the smallest gap between a sixth and a seventh logit in the run is 3.0e-6, and the device's logits are up to 1.8e-6 from the reference's. A token whose gap is under the products' rounding would take another sixth expert than the reference, with a weight that is almost the same; neither would be wrong. It did not happen in 1,872 choices.
* **Time, measured as requests of one half each** (the session answering the row's 2,560 numbers as text every time, so this is over what a half costs inside a step): **1.2 ms a mixture-of-experts half** (41 programs and the read-back of the six), **0.9 ms a dense one** (9 programs, three products of 3,584 x 2,560), from 200 and 1,000 requests, two runs each. A token's 50 halves are then at most 57 ms: **the 23 ms reckoned above was low**, and section 5's 10 to 20 tokens a second is at its lower end before the attention is counted. The session's whole process with a layer's 850 MB of tables is 0.14 s.
* **No staging limit was met**: section 9 said a picked table is bound by the session's 512 MiB pool. A layer's three tables of 251 MB are each under it, and together (850 MB) they ran.

**The four kernels** against float64 (`tests/kernels.py`, in the gate) and on the Metal GPU against the interpreter (`scripts/metal-kernels.sh`, by hand: 0, 6.3e-7, 0, 8.9e-8 of their peaks). **Mutants, by hand, eight, each killed by a number**: `moe_rank` counting the smaller entries, comparing a chunk only with itself; `moe_pick` taking the next place too, weighing by the logit, summing logits for indices; `axpy_slot` with the first slot's weight, reading every tile from the first; `group_sum` summing the first group. Two more were first written so that they did not compile, and were replaced.

**`tests/moe.sh`** (in the gate, the CPU session; the Metal session by hand: all 6 on both): a layer 64 wide with 128 experts of 64, three taken, every file text, against awk's double arithmetic of the definition, as a mixture of experts and as the dense MLP, each with `down` in Q4_0 and in Q4_1's form. Two pairs of router rows are equal, one across the chunks of 64 and one at the third place.

* **The tie mutant does what the design said it would**: with the ties not broken the three are 75, 0 and 120, the sums of the equal pairs' indices and an empty slot, read as experts without a refusal. `tests/kernels.py` cannot see it (made-up logits are never equal); `tests/moe.sh` kills it.
* **Nine mutants of the model file, by hand, each killed by that test**: every expert weighed by slot 0's weight; every expert picked by slot 0; the sum started from an expert and not the shared one; the places taken of the logits without their biases; the weights of the logits with them; the Q4_1 offsets multiplying the activation and not its group sums; the residual the normed row; `up` reading `gate`'s matrix; an expert's `down` scales picked by slot 0.
* **The test was too weak at first, and a mutant found it**: the made-up codes repeated with the block's size, so all 128 experts were one matrix and "every expert picked by slot 0" passed the Q4_0 case. The codes and scales no longer repeat in a block.
* **Not done:** the weights are still written by a scratch script, not a program (K5); the layer's two halves are not joined and nothing here is a step (K4); an expert is five programs where three would do; the time is of requests, not of a step; CUDA and HIP texts are emitted and held by goldens but ran nowhere, and their sessions have no `table ... by`.

## 11. K4: the attention of each kind, and the 50 layers as one cached step

**The design, before the code.** The attention half is from the row before a layer to the row after its attention (`mid`). Its shapes fit the cached step's kernels as they are: 4 key/value heads, 12 query heads each, 128 wide (`attention_scores_b`, whose scale is the literal 1/sqrt(128), `attention_pv_b`, `kv_write_b`). Two kernels are new:

| step | kernel |
|---|---|
| the norm before | `rmsnorm_rows` |
| q, k and v: one product of 7,168 x 2,560, the three matrices' rows one after another (a layer with no k and v of its own: q alone, 6,144) | `matvec_q4_rows` |
| each q and k head divided by its root mean square, times its kind's weight, rotated by the position | **`dec_post_g`**: `dec_post_n` with the head counts as parameters (48 and 4 here; `dec_post_n` is written for 16 and 8). An indexed layer has no positions: its cos table is ones and its sin table zeros, which leaves a head as it is, to the bit. A layer with no keys of its own is always an indexed one, so its 48 query heads need no rotation and are `rmsnorm_rows` of 48 rows **[corrected while writing the model file: this said the new kernel would run with no key heads]** |
| an indexed layer's new key and value rows rounded to NVFP4 | **`fp4_round`**: groups of 16; the scale is E4M3 of `max / 6`, the magnitudes E2M1. The reference rounds the whole cache at every token; a row's rounding depends on nothing but the row, so it is rounded once, when written |
| the caches, the scores with the mask, the softmax and the product with the values | `kv_write_b`, `attention_scores_b`, `attention_pv_b` |
| a layer that reads another's keys and values (9, 14, 49 read 4's; 34, 39, 44 read 29's) | a `link` to the other layer's caches |
| the output product (2,560 x 6,144), the norm after, the residual | `matvec_q4_rows`, `rmsnorm_rows`, `step_add` |

* **`fp4_round` rounds as the reference does, ties included**, which section 4 did not plan (it said ties would go down). A magnitude has seven thresholds; at 0.75, 1.75 and 3.5 the reference's round-half-to-even goes up and at the other four down, so each threshold is written as `step(m - t)` (up above it) or `1 - step(t - m)` (up at it). The E4M3 scale is a sum over a table of its 125 thresholds, their increments and which way each tie goes (`e4m3.f32`, three rows, the host's). Every partial sum of the increments is a multiple of 2^-9 under 448 and exact in f32.
* **What may still differ from the reference there: the divisions.** `max / 6` and `value / scale` are float32 divisions in the reference; a device that multiplies by a reciprocal is one unit in the last place off, which changes a rounding only when the quotient sits on a threshold. Measured below on every rounded value of the run, not assumed.
* **The window of 513** is a mask the host writes, a row a step. The run of 39 tokens never reaches it; **a mask that bites is tested on a small layer only**.
* **One model file, `models/kolibri_step.model`**, for the single layers and the whole step. `L0` and `NL` choose the layers; `TF=1` feeds every layer the reference's own row before it (each layer measured alone, a shared layer with keys the device made from the reference's rows), `TF=0` chains them (the step). The first layer's row comes from a table a step, here the embeddings of the reference's ids (teacher-forced); the embedding picked by the id the step chose is K6's. The MLP half is section 10's programs with the layer's number in their names: **the file repeats `models/kolibri_mlp.model`'s blocks**, since a model file cannot include another.
* **The head** is f32 (the file's Q6_K decoded by the host, 1.3 GB) under `matvec_rows`, after the final norm.
* **Weights stay 4-bit on the device** (section 5 left it to this stage): the attention's matrices through `matvec_q4_rows` as the experts', so the device holds about the file's 42 GB.
* **Bounds**: each layer alone, `mid` and the row after it against the reference's at 1e-4 of the row's peak; chained, the stream after every layer reported, and the logits as S2 of `docs/audio.md`: the arg max at every row whose two best are clear, the differences over the 50 largest reported.
* **In the gate** (no GPU, no model): the two kernels against float64 (`tests/kernels.py`) with mutants, and `tests/kstep.sh`: ten small layers of the model file on the CPU session against awk's arithmetic of the definition (dense and mixture layers, a rotary one whose window bites, an indexed one, one that reads another's keys).
* **Not in this stage**: contexts where the caches' tile no longer fits (the softmax kernel holds a head group's `12 x T` scores as one tile) **[measured below: refused past 640 positions]**, the embedding and the arg max on the device, the host in cancho.

**Measured, 2026-11-25, M4 Max, the Metal session.** The weights by hand from the file (a scratch script: 41 GB in `~/.cache/cancho-media/kolibri-k4w/`, the head decoded to f32), the run's tables from the reference's kept rows (its scratch copy now keeps q, the new key and value rows as cached, and the attention's output, in `r4/`; its stream is `r3/`'s to the byte). The session is driven a step at a time with a checkpoint after each, so every row of every step is compared. All differences are of the row's peak.

| the 39 tokens of the German run | each layer alone (`TF=1`), 1,950 layers | the 50 chained (`TF=0`) |
|---|---|---|
| q, 48 heads | 5.7e-7 | 1.0e-6 |
| the new key rows / value rows | 3.2e-7 / 4.6e-7 | 5.0e-7 / 8.5e-7 |
| the attention's output before its product | 1.0e-6 | 1.3e-6 |
| the row after the attention | 2.4e-6 | 3.0e-6 |
| the row after the layer | 1.8e-6 | **8.5e-6** (layer 49, the first token) |
| the NVFP4-rounded keys and values of layers 4, 19, 24, 29: 159,744 entries | **the reference's bits, all** | the reference's bits, all |
| the six experts, in order, of 1,872 choices | 1,872 | 1,872 |
| the logits, 25 rows of 128,000 (the reference kept its greedy choices' rows) | | arg max the reference's at **25 of 25** (the two best never closer than 0.31); within **2.9e-6** over a row's 50 largest and 3.9e-6 over all, the largest logit 22.9 |

* **The rounding agreed to the bit, and that is a measurement, not a property**: Metal's two divisions gave the reference's quotient on every one of these entries and on the thresholds of `tests/kstep.sh`. In the chain the rows that are rounded already differ from the reference's by 1e-6; a value that close to a threshold would round the other way, a whole step of one entry. It did not happen in 159,744.
* **Time**: **77 ms a step** with the head (median of 38 requests of one step: 77.5; three requests of 39 steps: 76.7 to 77.8), so **13 tokens a second**, in section 5's reckoned range. It is the same with caches of 576 positions as of 64 (75 ms): the caches are not what a step costs. The first request, which reads 41 GB and compiles 2,542 programs, is 18 to 25 s. This step takes its row from a table and answers six numbers: the embedding by the id, the arg max and a sampler are not in it.
* **A context limit, measured**: with caches of 2,240 positions the session is refused, `error[budget]`: the softmax kernel's scores for a head group, 12 x 2,240 floats, need 108,288 B of a threadgroup's 32,768. So this step holds **at most 640 positions** **[reckoned from the refusal's numbers; 576 ran]**. The sliding layers never need more than their window; the indexed ones do, and need the long form of the softmax (`attention_stats` and `attention_pvl`, which exist for 8,704 keys at other shapes). K6's.
* **The window never bites in this run** (39 tokens against 513). What holds the mask is the small model below, whose window is 2.

**The two kernels** against float64 (`tests/kernels.py`, in the gate) and on the Metal GPU against the interpreter (`scripts/metal-kernels.sh`, by hand: `dec_post_g` 1.8e-7, `fp4_round` 1.5e-7 of their peaks; the made-up table of that case is not E4M3's, so its scales are not exact sums). **Mutants, by hand, eleven, each killed by a number**: `dec_post_g` reading the keys from the queries, copying the keys as the values, the key rotation's sign, k normed with q's weight, the cosine twice; `fp4_round` with the peak over 4, every group as the row's first, the sign lost, the scale from nothing, every increment added, the third threshold at 1.

* **`tests/kernels.py` cannot see the rounding's edges**, as it could not see K3's ties: its made-up table is not E4M3's, no value there reaches the largest magnitude and none sits on a threshold. A mutant with the largest magnitude 5 survived it. So `tests/kstep.sh` first runs `fp4_round` alone on 80 values chosen for it (every magnitude threshold, at scale 1 and at 1.25; a scale on a threshold that goes down and one that goes up; a group under 2^-9 and one over 448) and requires awk's values exactly, and it runs two mutants itself: the largest magnitude 5 (9 of the 80 differ) and a value at 0.75 going down (3 differ).
* One more mutant was first written so that it did not compile, and was replaced.

**`tests/kstep.sh`** (in the gate, the CPU session, 8 s; the Metal session by hand: every check on both): layers 0 to 9 of the model file 64 wide, 4 query heads over 2 key/value heads, 64 experts of 32 with 2 taken, a head of 64, text files, four steps, a window of 2 keys. Against awk's double arithmetic of the definition: the row after the last layer at each step, and after the last step the row after each of the ten layers and the logits.

* **Sixteen mutants of the model file, by hand, each killed by that test**: an indexed layer rotated; its keys not rounded; its values not rounded; a sliding layer seeing every key (no window); an indexed layer seeing only the window; layer 9 reading layer 3's caches; a shared layer taking the values of the layer before; the attention's residual the normed row; the norm after the attention skipped; the caches not carried; the scores taken against the values; the head reading the row before the final norm; layer 5's `down` read as Q4_0; a shared layer's queries not normed; the cos and sin tables swapped; the keys written into the value cache.
* **Not done:** the weights and the run's tables are written by scratch scripts, not a program (K5); the step is teacher-forced: it does not embed the id it chose, take an arg max or sample (K6); contexts past 640 positions; 39 tokens of one text are all that was run; the reference is still this reader's numpy and no other implementation has agreed with it; the model file repeats `models/kolibri_mlp.model`'s blocks; CUDA and HIP texts are emitted and held by goldens but ran nowhere.

## 12. K5: the host, in cancho

**The design, before the code.** Until here the device's files came from scratch scripts in another language. K5 is the program that writes them, `tests/kolibri_gpu.cho` (built by `scripts/build-kolibri.sh` with this repository's GGUF reader, `src/gguf.cho`), and the tokenizer.

| | |
|---|---|
| `kolibri_gpu weights <gguf> <dir>` | every file `models/kolibri_step.model` reads as W, a tensor a chunk at a time: an F32 tensor copied; a Q4_0 or Q4_1 tensor as its codes repacked two a byte in the matrix's order, its f16 scales, and for Q4_1 the offsets `m + 8 d` as f32; q, k and v one after another; the head's Q6_K decoded to f32; the tables that depend on no weight (ones, the experts' indices, `fp4_round`'s E4M3 table). Which layers have a dense MLP, which have keys of their own and which matrices are Q4_1 are read from the file (its metadata and each tensor's type), not written into the program |
| `kolibri_gpu inputs <gguf> <ids file> <dir> <positions>` | a run's tables for the ids: the embedding rows dequantised from the file, the cos and sin of each position, the caches' `sel` and `keep`, the two masks (the window from the file's metadata), the empty caches |
| `kolibri_gpu argmax <f32 file>` | the index of the largest value, the value and its distance to the second: the next token from the saved logits, until the arg max is on the device (K6) |
| `cancho-gpu tokenize <gguf>` | takes `tokenizer.ggml.pre = qwen2`: the splitter already has Qwen2's pattern for a `tokenizer.json` (`src/bpe.cho`), the GGUF path refused everything but `qwen35` |

* **The bound for `weights` is the bytes**: every file equal to the scratch script's, `cmp`, all 41 GB. The same for `inputs` except the cos and sin, where numpy's float32 functions and this program's (binary64, rounded) may differ in a last bit: counted.
* **The bound for the tokenizer** is the ids of the reference tokenizer (Hugging Face `tokenizers` on the model's `tokenizer.json`) on text of several scripts, by hand, since the vocabulary is in the 42 GB file; in the gate, the toy fixtures with `pre` set to `qwen2`.
* **The exit**: a text, tokenized by `cancho-gpu`, its tables and the weights written by `kolibri_gpu`, the step on the Metal session, the arg max of the saved logits: the reference's next token, for both of section 2's prompts.
* **Every refusal has a rule tag** (`kolibri-usage`, `kolibri-read`, `kolibri-meta`, `kolibri-tensor`, `kolibri-ids`, `kolibri-write`) and no file reaches a trap: a tensor that is missing, of another type, with a width that is not a multiple of 32, or past the end of the file is refused before anything is written for it.
* **In the gate** (no model): `tests/kolibri_host.sh` writes a toy GGUF of three layers (a dense one, a mixture with its own keys, a mixture that reads another's) byte by byte with awk, runs the three commands and holds every output to awk's arithmetic of the formats; the refusals; mutants.
* **Not in this stage**: the chat template, the sampler, generation (K6).

**Measured, 2026-11-25, M4 Max.**

| | |
|---|---|
| `weights` on the 42.6 GB file | 1,200 files, 41 GB, in **35 s** |
| against the scratch script's files (SHA-256 of each; the two sets do not fit this disk together, so the script's were summed, removed, and these written) | **1,200 of 1,200 the same** |
| `inputs` for the 39 ids: the embedding rows, `sel`, `keep`, the two masks, the caches | the script's to the byte |
| its cos and sin tables, 2,496 values each | 180 and 116 differ, each by one unit in the last place (6e-8) |
| `cancho-gpu tokenize` on the file against Hugging Face `tokenizers` on the model's `tokenizer.json`: the two corpora of `tests/tokenizer/`, 537 lines of many scripts | **every line's ids and its decoding the same**; 0.27 s |
| the step from these files alone, the 39 tokens chained | the stream within 8.5e-6, the logits within 4.8e-6, 25 of 25 first choices, 1,872 of 1,872 experts, the rounded entries' bits all the reference's: section 11's numbers |
| "The capital of France is", tokenized, tabled, stepped, `argmax` | 9224, " Paris", 3.80 ahead: the reference's |
| "Berlin ist die Hauptstadt ... und hat" | 14255, " rund", 0.31 ahead: the reference's |

* **One file was not the script's at first: the head, by the sign of 4.6 million zeros.** `d (scale (q - 32))` with a whole-number product gives `+0` where ggml's `(d scale) q` and numpy's give `-0`. Equal as numbers; the program now multiplies in ggml's order and the bytes are the same.
* **The cos and sin are this program's, not numpy's**: the angle is the same float32 (the power, its inverse and the product rounded as the reference rounds them: all 64 powers agree), and its cosine is the binary64 one rounded, where numpy's float32 cosine is one unit off on 7% of the values. The step's numbers above are with these tables.
* **The tokenizer**: the GGUF path refused every `pre` but `qwen35`; it now takes `qwen2`, with Qwen2's pattern (already there for a `tokenizer.json`) and **no NFC**: this model's `tokenizer.json` has no normaliser and llama.cpp normalises nothing. The count differs by two (the file has 128,000 tokens, the JSON's vocabulary 127,998), which no id of the corpora shows.
* **`tests/kolibri_host.sh`** (in the gate): a toy GGUF of three layers written byte by byte with awk; the 78 files of `weights` and the 8 of `inputs` against awk's arithmetic of the formats (the copies, codes and scales to the byte, the rest as numbers), the list of files exactly the expected one, `argmax` (the first of equal values), and 20 refusals, each with its rule and status.
* **Mutants, by hand, twenty**: eighteen of the program's (the nibbles' order, the low and high halves, the scale's bytes, the offset `m + d`, a Q4_1 block read as 18 bytes, k and v emptying q's file, the head's shift, its scales unsigned, its second half's scales, an embedding row a block late, its codes not centred, the window a key long, a step seeing the next key, `sel` as `keep`, the sine as the cosine, the last of equal values, keys written for a layer that has none, the end-of-file check dropped) and two of the tokenizer's (Qwen3's pattern under `qwen2`; NFC under `qwen2`). **Nineteen are killed. The end-of-file check dropped survives**: a tensor past the end is still refused, with the same rule, by the short read after it; the check is the second of two.
* **The test said `ok` of a part that had failed, while it was being written** (a flag compared with itself); a part's `ok` now counts that part's own failures.
* **Not done:** the chat template, the sampler and generation (K6); `weights` reads the layers' kinds from the file but `models/kolibri_step.model` has Kolibri 1.1's written into it, and nothing compares the two (another file of this family would be refused by the session at a missing file, not by a message that says why); this model's own `tokenizer.json` is refused by `cancho-gpu tokenize` (its regex spells `\p{N}{1}`), and a `tokenizer.json` with no normaliser would still be NFC-normalised there; the weights are a second 41 GB beside the 42.6 GB file.

## 13. K6a: the step feeds itself (greedy generation)

**The design, before the code.** Section 6 gave K6 as generation with the sampler and the time a token. It is split, and this section is the first part: **the step takes the embedding of the id it chose and chooses the next by arg max, on the device**, so that a request of N steps is N tokens with no host in between. The sampler (the file's defaults are temperature 1, the 128 largest, top-p 0.99; S3's rank kernel compares every entry with every other, 16 billion comparisons a token at this vocabulary, so it needs another form) and contexts past 640 positions are K6b and K6c.

| step | how |
|---|---|
| the id of the step | `row_mix` of one number: the prompt's id while the prompt lasts (a table a step, with its mask), then the id the step before chose (a `carry`) |
| its embedding | `dq_q4` of one row: the embedding matrix's codes and scales are tables of 128,000 blocks picked `by` that id (K2). No one-hot product over the vocabulary, which is what the ASR step does with 152,000 columns |
| the arg max | the ASR step's five programs (`samp_max`, `samp_gmax`, `amax_cand`, the two again, `amax_id`): the first of equal maxima, as numpy's. `amax_id` with `v1 = V` gives the id from 0, which is what the pick reads |
| the ids of a request | `tok_acc`, saved as `ids_out.f32`; each request also answers its step's id, so that what asks can stop at the end token (`scripts/tts-session until`, section 28 of `docs/audio.md`) |

* **`models/kolibri_step.model` with `GEN=1`**; `GEN=0` is K4's teacher-forced step, its expansion unchanged (held by the recorded hash).
* **The host** (`tests/kolibri_gpu.cho`): `weights` also writes `emb.q4`, `emb.s`, `rank.f32`, `zmk.f32` and `one1.f32`; `inputs` takes a count of steps past the prompt and writes the id table, its mask and the collector's tables; `ids` prints a saved run's ids.
* **The bound**: from the German prompt's 15 ids, greedy, the 24 ids the reference chose, every one; from the other prompt its 8. Then the text, detokenized by `cancho-gpu`.
* **The time** of a step with everything in it, as section 11 measured it without.
* **In the gate**: `tests/kstep.sh` runs the small model with `GEN=1` from a prompt of two ids for four steps against awk (its embedding the 4-bit rows) **[corrected: this said five steps, and that the test would hold the first of equal maxima; its head's rows are all different, so it does not: `docs/audio.md` section 10's tests of these kernels do]**, and `tests/kolibri_host.sh` holds the new files.

**Measured, 2026-11-25, M4 Max, the Metal session**, every file written by `kolibri_gpu` and the ids by `cancho-gpu tokenize`:

| | |
|---|---|
| "Berlin ist die Hauptstadt ... und hat", 15 ids, one request of 38 steps | the 24 ids after the prompt are **the reference's 24** |
| detokenized | "... rund 3,7 Millionen Einwohner. Berlin ist bekannt für seine Geschichte, seine Kultur und seine vielen Sehenswürdigkeiten wie das Brandenburger" |
| "The capital of France is", 5 ids, 12 steps | **the reference's 8**: " Paris.  \nQuestion: What is the" |
| a step with the embedding by the id and the arg max in it | **74 ms** (median of 37 requests of one step: 74.3; three requests of 38 steps: 71.2 to 74.2): 13 tokens a second |
| stopping at a token: `scripts/tts-session until 4.6e1 38 ...` (the answer of a step that chose id 46, the full stop) | stops after 22 steps, the ids up to that one collected |

* **The step is no slower with the two additions** (section 11 measured 77 ms without them): the pick's read-back of one id and the arg max's seven programs are inside the noise.
* **The end token itself was never reached**: neither greedy run ends within what was run. What is measured is that a run stops at a chosen answer.
* **Mutants, by hand, fifteen, all killed**: eight of the model file (the id not fed back; the id from 1; the id from the logits and not the ranks; the first layer's residual the table's row; the collector keeping the step's own id; the prompt's mask ignored; the embedding's scales a fixed row's; the candidates every chunk's maximum) and seven of the host (the ranks growing; every step marked a prompt step; the id table reversed; `ids` an index late; the embedding's blocks another matrix's; the collector's selector its complement; the zeros past the prompt not written). An eighth of the host was no mutant: the change did nothing.
* **The test was too weak at first, and a mutant found it**, as in K3: the small model's head repeated every 31 rows, so every chosen id was under 31 and "the candidates are every chunk's maximum" (an arg max that sees the first 64 columns) survived. The head's rows are now all different, and the test requires its four ids to differ and to lie in both chunks. K4's sixteen mutants were run again on the changed test: all killed.
* **The tests as they are now**: `tests/kstep.sh` adds the generating run (the id each step answers, and the ids collected, exactly); `tests/kolibri_host.sh` holds 83 files of `weights` and 14 of `inputs` and has 25 refusals (section 12 gives its counts before this stage).
* **Not done:** the sampler (K6b); contexts past 640 positions (K6c); the chat template; a command that does all of it (K7: today it is five commands by hand); the tables are written for a number of steps fixed before the run.

## 14. K6b: the sampler

**The design, before the code.** The file's defaults (`generation_config.json`, and the model's card): temperature 1, the 128 largest, top-p 0.99; no repetition penalty. The Hugging Face order: the logits over the temperature; every entry below the 128th largest dropped (an entry equal to it stays); of the rest's softmax, sorted from the smallest, every entry whose running sum is at most 1 - 0.99 dropped; a draw from what is left.

**S3's kernels do not carry over.** `samp_rank` counts, for every entry, the larger ones: at 3,072 columns 9 million comparisons, at 128,000 16 billion a token. What makes it small again:

* **The 128 largest entries lie in the 128 chunks of 64 with the largest maxima** (there are 128 entries at least as large as the 128th largest chunk maximum, so the 128th largest entry is at least that). So: the 2,000 chunk maxima (`samp_max`, there already), their places with ties broken by index (**`rank_idx`**: `moe_rank` with the chunk's width a parameter, since 2,000 is not a multiple of 64), a 128 x 2,000 matrix with a one where chunk q has place r (**`rank_onehot`**), and its product with the logits as 2,000 x 64: **the 128 chunks, 8,192 candidates, gathered by a matrix product** **[corrected when it ran: 144 chunks and 9,216 candidates. `samp_cum` takes its columns 24 a threadgroup on this target and refused 128, `error[offset]`; more chunks than the 128 change nothing that is drawn]** (**`gather_chunks`**; a one times a value plus zeros is exact). The same product with 0 .. 1,999 gives each gathered chunk's number.
* **Everything after is on 8,192 candidates** **[9,216: see above]**: the temperature; `samp_rank` and `samp_keep` as they are (67 million comparisons; a candidate's count of larger candidates is its count of larger entries, since every larger entry is a candidate); top-p as a second, weighted count (**`topp_mass`**: for every candidate the sum of `exp(y - max)` over the kept candidates not larger than it, itself included; **`topp_keep`**: kept where that mass is over `(1 - p)` of the whole, which is the mass of the largest); then S3's draw by inverse CDF (`samp_sum`, `samp_cum`, `samp_pick`, `samp_local`, `samp_tok`) with **one uniform number a step**, a table the host writes from a seed, and the gathered chunks' numbers as `samp_tok`'s base, so that the id is the vocabulary's.
* **What is not Hugging Face's, said before it is measured**: (1) a tie exactly at the 128th value between a gathered chunk and one left out (two chunk maxima equal, one of them the 128th) drops an entry torch keeps; (2) entries of equal probability at the top-p edge are dropped or kept together, where torch's sort takes them one by one; (3) the draw maps a uniform number to an entry in the order of the gathered chunks, which is no library's order: the distribution is the same, the stream of draws is this sampler's own. So the reference is this sampler's definition in float64, not `torch.multinomial`.
* **The bound.** With the device's ids fed to the reference (teacher-forced, as S2 of `docs/audio.md` held a sampled run), at every step the reference's sampler, given the reference's logits and the step's uniform number, draws the device's id; a step where the two differ is looked at (a candidate at the top-p edge, a uniform number at a boundary of the cumulative sum) and counted, not explained away. And greedy as the limit: `K=1` must give K6a's ids.
* **In the gate**: the four kernels against float64 with mutants; `tests/kstep.sh` with the small model sampling (a vocabulary of 128, the 2 largest chunks, 5 kept, top-p 0.9) against awk's sampler, step by step **[as built: a vocabulary of 2,048, 24 chunks gathered of 32, temperature 2/3, 5 kept and top-p 0.6, and a second run with 4 kept and no top-p]**.

**Measured, 2026-11-26, M4 Max, the Metal session**, `GEN=1 SAMP=1` with the file's settings; the uniform numbers from `kolibri_gpu inputs ... <steps> <seed>` (a 31-bit linear congruential generator's upper 24 bits):

| | |
|---|---|
| the German prompt, 60 steps, seed 1 | "... und hat drei Millionen Einwohner. Die Einwohner sprechen Deutsch. Berlin ist für seine Geschichte bekannt. Berlin ist in zwölf Bezirke eingeteilt. Die Stadt wurde im Zweiten Weltkrieg stark zerstört und danach wieder aufgebaut. Im Jahr 1989 fiel die Berliner" |
| seed 2 | "... und hat rund 3,6 Millionen Einwohner. Sie ist das politische, kulturelle und wirtschaftliche Zentrum Deutschlands. Die Stadt ist bekannt für ihre vielen Sehenswürdigkeiten, darunter das Brandenburger Tor, den Reichstag, die Museumsinsel und die Siegessäule. Die" |
| the reference's sampler (this section's definition in float64, on the reference's logits, fed the device's ids) at each of 45 draws a run | **the device's id at 90 of 90** |
| `TOPK=1` | K6a's greedy 24 ids |
| a step, sampling | **62 ms** (medians of 37 requests: 61.7 and 62.8; requests of 38 steps: 62.3 and 63.9) |
| a step, greedy, measured alternately with it | 72 ms (72.0 and 72.2) |

* **Sampling is faster than the arg max**, by 10 ms a step, twice over: the arg max's programs over 2,000 chunks cost more than the whole draw over 144. Which of them was not looked for. `SAMP=1 TOPK=1` is the greedy step at the sampler's price.
* **On these runs top-p always cut first**: after both filters 1 to 106 candidates were left (medians 8 and 2), never 128, so the 128 largest alone decided nothing here. What holds top-k is the small model below.
* **No draw was near an edge**: the nearest a uniform number came to a boundary of the cumulative sum was 3e-4 of the mass, and a candidate's sum to the top-p threshold 7.5e-6 of the whole (float32 keeps 6e-8). A draw on an edge would differ from the reference and would not be wrong.
* **The stream of draws is this sampler's own**, as said above: not torch's, not llama.cpp's.

**The five kernels** against float64 (`tests/kernels.py`, in the gate) and on the Metal GPU against the interpreter (`scripts/metal-kernels.sh`, by hand: `rank_idx`, `rank_onehot` and `topp_keep` 0; `gather_chunks` 4.8e-7; `topp_mass` 8e-8 of their peaks). **Mutants, by hand, twelve, each killed by a number**; two were first written so that they did not compile, and were replaced.

**`tests/kstep.sh`** adds two sampling runs of the small model against awk's sampler: the id each of four steps draws, exactly, and at the last step the gathered chunks' cumulative masses and every chunk's count of columns before the target (`cum.f32`, `cnt.f32`, which `SAVE=1` writes).

* **The test was too weak three times, and mutants found each** (eleven of the model file, by hand; all killed in the end):
  * *Four draws say little of a distribution*: four mutants (no top-p; the column counted before top-p; the temperature left out; a fourth that turned out to change nothing, see below) drew the same four ids. Hence the masses and the counts.
  * *Counting a chunk's columns before top-p* differs only when a candidate top-p dropped lies before one it kept, in the gathered order, and the target falls in its mass. With one large logit a chunk that never happens (the dropped are the smallest, in the last chunks). The small model's head now has its large logits in pairs inside a chunk, and the last uniform number is chosen by the reference to fall in such a candidate's mass; the test fails if there is none.
  * *One more than TOPK kept* survived top-p (which dropped the extra one) and survived 5 kept without top-p (the fifth and sixth logits are equal, and an entry equal to the last kept stays: 5 already keeps 6). The second run keeps 4.
* **A mutant that was none**: top-p's shift read from the gathered chunks' maxima and not the row's maximum. Its first entry is the row's maximum, the chunks being gathered largest first. Not counted.
* **The host**: `weights` writes the sampler's five tables for the file's own top-k made a multiple of 24, `inputs` the uniform numbers. Six mutants, by hand, all killed by `tests/kolibri_host.sh` (88 files of `weights`, 15 of `inputs`, 26 refusals now).
* **Not done:** contexts past 640 positions (K6c); the file's two end tokens (127906 and 127901: the driver stops at one answer); `min_p` (0 in the file); a repetition penalty (none in the file); the chat template; a command (K7). The greedy arg max's 10 ms are not explained. On the real model top-k was never the filter that decided.
