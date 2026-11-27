# Kolibri 1.1 on the GPU (design, 2026-11-25; K0 done: the model is read and a reference written from its code continues a prompt sensibly; K1 done: its 4-bit weights are read in a kernel, one expert's matrix to the bit on the Metal GPU; K2 done: a session reads the experts a value names, six of a layer's 384 on the Metal GPU; K3 done: the MLP half of each of the 50 layers, the mixture of experts with its router, runs on the Metal GPU within 2.1e-6 of the reference; K4 done: the 50 layers run as one cached step on the Metal GPU, teacher-forced on the reference's 39 tokens: the stream within 8.5e-6, the logits within 3.9e-6, every first choice the reference's, 77 ms a token; K5 done: the host is a cancho program, its 41 GB of weight files the scratch script's to the byte, the tokenizer reads the model's file, and a text's next token on the device is the reference's; K6a done: the step feeds itself, and from both prompts the device's greedy tokens are the reference's, all 24 and all 8, at 74 ms a token; K6b done: the sampler (temperature, the 128 largest, top-p) is on the device, its 90 draws of two seeded runs the float64 definition's, 62 ms a token; K6c done: contexts to the 2,184 positions an indexed layer sees whole, 78 ms a token at any of them, every layer alone the reference's over 2,184 tokens but for 9 of 8.9 million 4-bit roundings that fall the other way on a tie and one router tie; the chain of 50 layers leaves the reference at the first such rounding, as the reference leaves itself when given it; K7 done: `scripts/kolibri "<text>"` answers through the model's chat template and stops at its end tokens, a greedy answer of 105 tokens with its reasoning the reference's token for token, 13 tokens a second; no indexer past 2,184, one question and one answer, no server; section 17: the plan from there to a model server for cancho-code and cancho-agent, and its first steps: 24 tokens a second, one session for many conversations; section 18: the server, `scripts/kolibri-serve`, which cancho-code's own agent reaches and whose 2,184 positions its requests fill; section 19: caches as rolls of positions and the indexer: 16,384 positions, cancho-code's primer fits and its second task takes 7.6 s, its first 12 minutes; section 20: a prompt 16 tokens a pass: the primer in 191 s where it took 723, a second task in 3.9 s; section 21: tools: the template's block, calls and tool turns, `tool_calls` read out of an answer, and cancho-agent's own tool loop answered; section 22: a path for Q4_0 in the product: one row three times cheaper, a pass of 16 in 191 ms where it took 257, a step 7 to 13% cheaper, and the indexed layers now most of a step; section 23: an indexed layer gathers the keys it chose and attends to those alone: a step 34 to 41 ms where it was 51 to 63, cancho-code's first request 119 s; section 24: the step's experts as two launches: a step 29 to 35 ms; section 25: the head as the file's codes, 410 MB where it was 1.3 GB of f32: a step 27 to 30 ms)

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
| the head | | Q6_K decoded by the host into f32 (328 M values, 1.3 GB) **[section 25: a byte a code and an f32 a group of 16, 410 MB]** |
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
* **K6. Generation** **[split in section 13: K6a, the step feeding itself greedily, is done; the sampler, K6b, is done in section 14; contexts past 640 positions, K6c, in section 15]** with the sampler, stopping at the end token (`scripts/tts-session until`'s way), and the time a token.
* **K7. A command** **[done, section 16: `scripts/kolibri`, a script of its own]**. Which one (`scripts/media`, or the server of `docs/serving.md`) is decided when K6 runs.

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
* **Not done:** the time of the read-back in a real step (48 waits a token) is not measured: K4's; CUDA's and the other device sessions do not have the clause; a picked table is uploaded whole, so a table over the session's 512 MiB staging pool is refused as any binary table is (an expert file is 251 MB) **[since section 20.3 a binary table goes up a chunk at a time and is not held to the pool]**.

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

* **The rounding agreed to the bit, and that is a measurement, not a property**: Metal's two divisions gave the reference's quotient on every one of these entries and on the thresholds of `tests/kstep.sh`. In the chain the rows that are rounded already differ from the reference's by 1e-6; a value that close to a threshold would round the other way, a whole step of one entry. It did not happen in 159,744 **[it does in a longer run: section 15 counts 9 in 8.9 million, and what one does to the chain]**.
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
| the ids of a request | `tok_acc`, saved as `ids_out.f32`; each request also answers its step's id, so that what asks can stop at the end token (`scripts/tts-session until`, section 28 of `docs/audio.md`) **[since 17.6 there is no collector: the answers are the ids]** |

* **`models/kolibri_step.model` with `GEN=1`**; `GEN=0` is K4's teacher-forced step, its expansion unchanged (held by the recorded hash).
* **The host** (`tests/kolibri_gpu.cho`): `weights` also writes `emb.q4`, `emb.s`, `rank.f32`, `zmk.f32` and `one1.f32`; `inputs` takes a count of steps past the prompt and writes the id table, its mask and the collector's tables; `ids` prints a saved run's ids. **[since 17.6 the ids and the mask are a request's numbers, the collector and `ids` are gone]**
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

## 15. K6c: contexts past 640 positions, to the 2,184 the indexed layers see whole

**The design, before the code.** Section 11 measured the limit: the softmax kernel (`attention_pv_b`) holds a head group's `12 x T` scores as one tile, and past 640 positions that tile is over a threadgroup's 32,768 B. Two things grow with the context in that step, and each has its own answer:

* **A sliding layer never sees more than its window, so its caches need not be longer than it: a ring.** Its caches hold `TW` positions, the window rounded up to a multiple of 64 (576 for 513), and step `s` writes slot `s mod TW`. Nothing in a sliding layer's arithmetic knows a key's slot: the rotary position is in the key and the query before the key is cached (`dec_post_g`), and a softmax does not depend on the order of its terms **[in exact arithmetic: the device sums in slot order, so past the first wrap the sums are in another order than the reference's; measured below]**. So the kernels stay the short ones, and only the host's tables change: `selw.f32` and `keepw.f32` (a one at slot `s mod TW`, and its complement), and `mask.f32` now over the slots: slot `c` is seen at step `s` when the position it holds, `s - ((s - c) mod TW)`, is not negative and less than a window back. A slot is overwritten only by a position `TW` later, when its own has left the window, since `TW` is at least the window. **A sliding layer's step then costs the same at any context.**
* **An indexed layer sees every key, so its softmax takes the long form**, which exists (`attention_stats` and `attention_pvl`, written for the ASR decoder's 8,704 keys): the scores as before (`attention_scores_b`, a chunk of 64 keys a threadgroup), then each query head's maximum and normaliser over chunks of 64 with one carried tile, then the product with the values over the same chunks. No tile grows with `T`. Its arithmetic is not the short form's to the bit (the maximum is `m + relu(c - m)`, and the sums go chunk by chunk).
* **One form, not two.** The short softmax is dropped from the indexed layers at every `T`, and the ring is there at every `T`, rather than a switch at 640: a second path would be one the gate's small model or the real runs do not take. So K4's to K6b's numbers are measured again with this file.
* **The limit that stays: 2,184 positions.** Past that an indexed layer's query attends to a selection of the keys (the indexer's 2,048, the first 8 and the last 128: section 1), which section 6 left out of this plan and the reference stops at. Until here nothing held a run to it. `kolibri_gpu inputs` now refuses more steps than `kolibri1.attention.indexer.top_k` + 136 (`kolibri-context`); **the 8 and the 128 are not in the GGUF: they are the model's configuration (`index_num_sink_tokens`, `index_num_local_tokens`) written into the program as constants**. With `T` 2,240 the device's caches hold the 2,184.
* **What grows with the context and stays**: an indexed layer's scores and product, linear in `T` (ten layers); the four indexed caches rewritten whole at every step (`kv_write_b`: 2 x 4 x T x 128 floats each, 9 MB a step at 2,240); the host's tables `sel.f32`, `keep.f32`, `maskg.f32`, each `steps x T` floats (20 MB each at 2,184 steps of 2,240). None is a limit under 2,184; all would be at the model's 262,144.
* **The bound.** The reference teacher-forced over 2,184 tokens of one text (a scratch copy that rounds a cached row once, when cached, instead of the whole cache at every token; its stream on the 39 tokens must be section 2's to the byte), the device over the same ids: the stream after every layer and the final row at every step, the arg max of the logits at every row whose two best are clear **[false as a bound for the chain, measured below: one 4-bit rounding that falls the other way, at step 173, and the chain is another run from there. What holds the kernels over the 2,184 steps is each layer alone on the reference's rows; the chain is reported, with a control]**. That passes the ring's wrap three times (576, 1,152, 1,728) and ends on the last position the model sees whole. And generation there: from prefixes of the text past 640, the device's greedy ids against the reference's.
* **In the gate**: `tests/kstep.sh` runs the small model once more with caches that wrap and chunk: chunks of 4 (a new constant `TS`, 64 by default), a window of 3 in a ring of 4, indexed caches of 16, 14 steps, against awk's arithmetic of the definition (which has no ring and no chunks). `tests/kolibri_host.sh`: the new tables against awk, and the refusal.

**Measured, 2026-11-26, M4 Max, the Metal session.** The text: 2,184 tokens of prose from the model's card (`cancho-gpu tokenize`'s ids), the tables from `kolibri_gpu inputs ... 2240 2184`. The reference's scratch copy gave section 2's stream and final rows on the 39 tokens to the byte, and took 66 minutes for the 2,184. A session is driven a step at a time with a checkpoint after each. All differences are of the row's peak.

| | |
|---|---|
| **each layer alone** (`TF=1`: a layer's row is the reference's row before it), 109,200 layer-steps | |
| the 40 sliding layers, whose ring is gone round three times | within **6.2e-6** at every step but one |
| the one: layer 11 at step 1,500, 2.9e-3 | the reference's router has its sixth and seventh experts 2.1e-7 apart there |
| the rounded keys and values of layers 4, 19, 24, 29: 8,945,664 entries | **9 are not the reference's**, each its neighbour on the 4-bit grid (steps 66, 173, 197, 443, 455, 834, 848, 1,060, 1,658; all nine values, no key) |
| layers 29, 34, 39, 44 (layer 29's caches, in which nothing fell the other way) | within **2.4e-6** at every step: the long softmax, to 2,184 keys |
| layers 4, 9, 14, 49, 19, 24 (caches with such an entry) | medians 2.5e-7 to 2.3e-6; 46 layer-steps over 1e-4, the worst 2.3e-2 (layer 24, the step after one) |
| **the 50 chained** (`TF=0`) | |
| steps 0 to 172 | the stream within 3.5e-6 |
| from step 173, where one value of layer 4 (-0.0341796875, exactly between two neighbours at the reference's scale) is rounded the other way | the last layer's row: median 1.7e-2 to step 300, 1.1e-1 at the end, the worst 2.9 |
| the arg max of the logits | the reference's at 2,100 of 2,184 steps, and at 2,068 of the 2,140 where the reference's two best are over 0.05 apart |
| **a control: the reference again, with that one entry set as the device rounded it** (700 steps) | it leaves itself as the device leaves it: the last layer's median 1.7e-2 to step 300 (the device's 1.7e-2), 6.1e-2 at 500 to 700 (5.9e-2); another arg max at 13 steps (the device at 14) |
| **generating** (`GEN=1`, greedy) from a prefix of 700 tokens, and of 1,300 | the 16 ids after it are the reference's, both |
| from a prefix of 2,160, to the last position | the first 11 of 24; the twelfth is where the reference's two best are 0.007 apart |
| **a step** at positions 200 to 576, 576 to 1,152, 1,152 to 1,728, 1,728 to 2,184 | 78.1, 78.0, 77.4, 77.4 ms (medians, the teacher-forced step with the head) |
| a generating step past a prefix of 700, 1,300, 2,160 | 79.9, 79.6, 79.6 ms |
| K6a's two greedy runs and K6b's two seeded sampling runs, again with this file | the reference's 24 and 8 ids; the 46 ids of each sampling run as before; a greedy step at 64 positions 76 ms |

* **The chain cannot be held to the reference past the first rounding that differs, and that is the model's arithmetic, not this step's.** Section 11 said a value near a threshold would round the other way and counted none in 159,744. Over 2,184 tokens it happens: the device's value differs from the reference's in its last bits, the reference's sits on a tie, and the cached entry is a whole grid step off, for every later query. After that the rows differ by 1e-3, more roundings and expert choices fall the other way, and the two runs are two runs. **The control says how much of the device's distance is this: all that was measured.** What it does not say is which of the two is right; neither is: both round a number that is on the tie to within the arithmetic's noise.
* **So a long greedy run is not reproducible across implementations of this model**, this one and the reference being two. It agreed for 16 tokens after 700 and after 1,300, and for 11 after 2,160.
* **The ring did what was reckoned**: the sliding layers' softmax sums in slot order past the first wrap, and stays within 6.2e-6.
* **The step's time does not grow to 2,184 positions** (78 ms at every stretch; the first 200 steps of the long run were measured while the reference still ran beside it and are left out). The first request is 19 to 30 s.
* **A slip of the measurement, not of the step**: the generating runs were first compared one position off and read as "none of the ids agree"; and the earlier German run seemed changed because the kept file it was compared with was a later, shorter run's. Both were the comparison's.

**Mutants, by hand.** Of the model file, twelve, each killed by `tests/kstep.sh`: the maximum and the normaliser swapped; the statistics of the scores of the layer whose keys are read; the statistics, and the product, over the first chunk only; the two kinds of cache exchanged; a sliding layer writing, and keeping, by the position; a sliding layer seeing every key; an indexed layer seeing the window; no carry; a shared layer with the values of the layer before; the scores against the values. **Five of the twelve are killed by a refusal of sizes (the run with caches that wrap has rings of 4 slots and positions of 16), not by a number.** Of the host, eleven, each killed by `tests/kolibri_host.sh`: one key more in the window; a slot never written seen; the slot the position (sel, keep); the ring rounded down; the ring's caches, mask and selector as wide as the positions; a step more taken; the last step refused; the first 8 keys not counted. K4's sixteen, K6a's eight and K6b's eleven of the model file, again on the changed test: all killed. No kernel is new.

* **Found and not fixed here: a path with `..` in it stops `kolibri_gpu` with an illegal instruction, and `cancho-gpu gguf` too.** No refusal, no rule tag. It is not this stage's (the earlier programs do the same), and it is every program's that opens a file, so it is its own change. **Fixed since** (`runtime.md` section 11): a refusal with the rule tag, in every program.
* **Not done:** contexts past 2,184 positions (the indexer: its pool, its 2,048 chosen keys; the reference has none either); the tables `sel.f32`, `keep.f32`, `maskg.f32` are `steps x positions`, 20 MB each here and too large long before 262,144; the file's two end tokens; the chat template; a command (K7). One text was run. The reference is still this reader's numpy.

## 16. K7: a command

**The design, before the code.** Until here a text's answer was five commands by hand. Section 6 left open which command; it is **`scripts/kolibri`, a script of its own beside `scripts/tts` and `scripts/asr`**, not `scripts/media` (whose jobs are pictures and sound through `build/media`) and not the server of `docs/serving.md` (an engine built around Qwen3.8's step, its draft head and its prefix cache: none of it is this model's). What it runs is what exists:

| step | by |
|---|---|
| the chat template | **`kolibri_gpu prompt <text file> <effort> [<system file>]`**, new: the model's own template (`tokenizer.chat_template`, the same text in `tokenizer_config.json`) written out for what this command takes: one user message, a system message or none, and the reasoning effort. `--raw` skips it |
| the ids | `cancho-gpu tokenize <gguf>`: it already splits the template's special tokens as Hugging Face does **[measured before the design: one templated prompt, 44 ids, the same]** |
| the run's tables | `kolibri_gpu inputs`, for the prompt and at most `--max` more steps (256 by default), no more than section 15's 2,184 |
| the steps | the Metal session on `models/kolibri_step.model`, `GEN=1`, sampling with the file's settings and a `--seed` (1 by default), or `--greedy`; the prompt as one request, then a step a request **[as built: the prompt but its last token; the last token's step chooses the first new one, so it is the first of the steps that are read]** |
| **stopping** | at the first answer that is one of **the file's two end tokens**, read from it by **`kolibri_gpu ends <gguf>`**, new **[as built it also prints the most steps of a run, 2,184, so that the script cuts `--max` to it and does not carry the number itself]** (`tokenizer.ggml.eos_token_id` 127906, `<|im_end|>`, and `tokenizer.ggml.padding_token_id` 127901, `<|endoftext|>`: the two of the model's `generation_config.json`), or after `--max`. `scripts/tts-session until` stops at one answer and compares it as text; this compares numbers |
| the answer's ids | `kolibri_gpu ids`, cut before the end token **[since 17.6: the session's answers themselves]** |
| the text | `cancho-gpu detokenize`, which prints a quoted line; **`kolibri_gpu say <file>`**, new, writes that line's bytes as they are |

* **The template, as far as this command goes.** A system block always (the template writes one whenever there is a reasoning sentence, and there always is): the system message if given, a blank line, `# Reasoning effort`, and one of four sentences (`none`, `low`, `medium`, `high`; the template's default is `high`, and so is the command's). Then the user's message and the assistant's opening; with `none` the template also writes an empty `<think>` block. Tools, earlier turns and tool responses are not written: **one question, one answer**.
* **Held to the template itself**: the cases' expected bytes are recorded once from the model's template through jinja2 (by hand, a scratch script; `golden/kolibri/`), and the gate compares `prompt`'s output with them byte for byte. The gate runs no jinja2.
* **A user's text is not escaped**, in the template or here: a message that contains `<|im_end|>` ends its own turn, as it does through Hugging Face. Said, not prevented.
* **The weights**: `~/.cache/cancho-media/kolibri-w` (or `KOLIBRI_WEIGHTS`), written by `kolibri_gpu weights` on the first run if not there: 41 GB, about 35 s. The GGUF is `KOLIBRI_GGUF` or the cache's.
* **What the gate can hold without the model or a GPU** (`tests/kolibri_cmd.sh`): `prompt` against the recorded bytes; `ends` and `say` on the toy GGUF and on made lines, with their refusals; and the script's own logic with a session that is a stub (it answers a list of ids): that it stops at either end token and at `--max`, cuts the answer there, and passes the right arguments. **Not the model's answer**: that is by hand, on the Metal GPU.
* **The bound, by hand**: the command's greedy answer to a question is the reference's from the same templated ids, up to the first step whose two best are close (section 15: a long greedy run is not reproducible); the templated ids are Hugging Face's `apply_chat_template` ids; a sampled answer ends at an end token and reads as an answer.

**Measured, 2026-11-26, M4 Max, the Metal GPU.** `scripts/kolibri` as it is, on Kolibri 1.1's file and the weights `kolibri_gpu weights` wrote:

| | |
|---|---|
| `--greedy --effort none "Was ist die Hauptstadt von Frankreich? Antworte in einem Satz."` | "Die Hauptstadt von Frankreich ist Paris." and the end token `<|im_end|>`: **the reference's 7 ids and its end token**, from the same 49 prompt ids |
| `--greedy` (effort high) on a sum in words, 78 prompt ids | a `<think>` block, then the sum worked and "**30 sheep are left.**": 105 ids and the end token, **the reference's 105 and its end token** (its two best as close as 0.053 on the way) |
| the same question sampled, `--seed 3` | another reasoning and another wording of the same answer, 137 tokens, ended by the end token |
| the prompt's ids, both prompts | Hugging Face `tokenizers`' ids for the template's text |
| the template's text, nine cases (four efforts, a system message, an empty one, an empty question, lines and other scripts) | the model's own template through jinja2, byte for byte |
| a run | 19 to 22 s to the first token (the 41 GB read, 2,500 programs compiled), then 13 tokens a second greedy (105 in 8 s) and 15 sampling (137 in 9 s); the script counts whole seconds |

* **An end token was reached for the first time**: K6a's and K6b's runs were raw continuations, which do not end. With the template every run above ended by itself.
* **The command prints the reasoning too**: what is between `<think>` and `</think>` is part of what the model writes, and nothing takes it out.
* **A hang, found by a mutant and removed.** The script first started the session on two named pipes and read its answers. A session that stops at once (a refused model file) left the script writing to a pipe nobody read, or waiting on one nobody would write: a mutant that asked one step too many did not fail the test, it stopped it. The session now runs in a subshell that says `exit <status>` after it, so a session's end is a line to read. (The stall the mutant showed was in the test's own stub, which looped on an answer that is no number; the pipe's race was found reading for it.)

**`tests/kolibri_cmd.sh`** (in the gate, 5 s, no model and no GPU): the nine template cases against the recorded bytes; `ends` on the toy GGUF (`tests/kolibri_toy.sh`, taken out of `tests/kolibri_host.sh` so that both use it; it has the two end tokens now); `say` on a line with a quote, a backslash, a newline, UTF-8 and a zero byte, and eleven lines it refuses; then the script with a stub tokenizer and a stub session: what it asks of the session and in which requests, stopping at each end token and at `--max`, an answer of nothing, the steps cut to what the model sees whole, the template, `--raw`, `--system`, `-f`, the seed, `-o`, weights written when not there, the lock taken, respected and taken over, and thirteen refusals. 60 checks.

**Mutants, by hand: 34, of which 33 killed.** Of the script, 22 (stopping at one end token only, either; the end token in the answer; a step too many for the prompt; `--greedy` sampling; the seed, the effort, the system message, `-f`, `--raw` dropped; the steps not cut; the lock not looked at, kept though its job is gone, left behind; the answer's ids a step late; the answer compared as text; the session's failure not looked at; a prompt too long not refused; the weights never written; a step fewer than `--max`; no checkpoint; an answer that is no id read as 0). Of the three subcommands, 12: a sentence exchanged, the blank line after the system message, the think block always and never, an empty system message dropped, the two end tokens exchanged, four of `say`'s escapes and its closing quote. **The survivor: `say` taking hexadecimal digits in capitals**, which `cancho-gpu detokenize` does not write; a case for it was added and the mutant is killed now.

* **Not done:** more than one question and one answer (earlier turns, tool calls and their responses are in the template and not in `prompt`); the answer is printed when the run is over, not as it is written; the reasoning is not separated from the answer; contexts past 2,184 positions; no server, and nothing of `docs/serving.md`'s speed work (this model's file has no draft head); the first token takes 19 s every run, since the weights are read and the programs compiled each time; `min_p` and a repetition penalty (neither set in the file). **The model's answers were read, not scored**: three prompts.
* **Still open from section 15**: a path with `..` in it stops `kolibri_gpu` and `cancho-gpu` with an illegal instruction. The script makes its own paths absolute before it passes them on, so the command does not reach it; a hand-typed call still does. **Fixed since** (`runtime.md` section 11).

## 17. From a command to a model server for cancho-code and cancho-agent (plan, 2026-11-26)

**What is asked**: Kolibri 1.1 behind an endpoint that cancho-code and cancho-agent can use, at a speed that is usable. Section 16's command is none of that: one question, 2,184 positions, 19 s before every answer. This section is the plan, with what was read and measured before it. Of it, S2 (17.5), S1 (17.6), S3 (section 18), S4 (section 19), S5 (section 20), S6 (section 21) and S7 (section 22: the Q4_0 path; the experts' two launches in the step are not) are built, and after them section 23 (the indexed layers), which the plan did not have, and section 24 (the experts' two launches in the step, S7's other half).

### 17.1 What the two clients send **[read from their sources, 2026-11-26]**

| | cancho-code (`src/cancho_agent.cho`, `src/ollama.cho`) | cancho-agent (`src/openai.cho`, `src/model.cho`) |
|---|---|---|
| route | Ollama's `POST /api/generate`, plain HTTP, a literal IPv4 address, port 11434 by default | OpenAI's `POST /v1/chat/completions` |
| request | `model`, `prompt` (its primer and the task as one text), `stream: false`, `think: false`, `options: {num_ctx, temperature: 0.2, num_predict}`. No `system`, no `messages`: **the server writes the chat template** | `model`, `max_tokens`, `messages` (system, user, assistant with `tool_calls`, tool with `tool_call_id`), `tools` (JSON schemas). No `stream`, no sampling option |
| answer read | `response`, `done_reason` (`length` is retried), `prompt_eval_count`, `eval_count`; one body, by `Content-Length` or chunked | `choices[0].message.content` and `.tool_calls[].function.{name, arguments}` (**arguments a JSON string**), `finish_reason`, `usage` |
| sizes | **the primer is 5,500 tokens by its documentation and was 10,666 Qwen tokens in its M8 runs**, prompts of about 11,000 with `num_ctx` 16,384; `num_predict` to 8,192 | a system message, a user message and at most four tool turns in its live program: hundreds of tokens |
| waits | 30 s for the response's head after the request is sent, 30 s between bytes, 1,800 s an attempt | 120 s in its live programs |
| what repeats | the primer, the same bytes at the head of every request | the conversation so far, a turn longer each time |

* **cancho-code never sends a tool or a second turn**: it reads a program out of the text. What it needs is a long prompt, most of it the same as the last request's.
* **cancho-agent's live programs call port 443 with TLS, hard-coded**: a local plain-HTTP server is not reachable from them as they are. That is a change in cancho-agent, not here, and it is not made by this plan. **[21.3: still so in its two live programs; its `model.ask` takes a port and plain TCP now, and the loop of one of them, in a scratch copy with that one line changed, is answered]**
* **A wait of 30 s for the head** is shorter than a first answer can take here. The server must send its status line and headers as soon as the request is read, and the body chunked when it is done.

### 17.2 What a step costs now **[measured, 2026-11-26, M4 Max; `CANCHOGPU_PROFILE`, the shim's own counters]**

One prompt of 44 tokens and 34 greedy tokens after it (`scripts/kolibri --greedy --effort none`), 78 steps:

| | |
|---|---|
| a step on the clock | about 78 ms (section 15; 82 tokens in 7 s again today) |
| of it, the host waiting for the GPU | **53.6 ms**, in 49 waits: one read-back a layer for the experts the router chose, and the last |
| of that, the GPU working | 47.2 ms |
| the rest, about 22 ms | the host between the waits: `on_device` looks through every table of the session for every parameter of every program, each step (2,559 programs, 2,064 tables) **[read, not timed alone]** |

The GPU's time by kernel (`CANCHOGPU_PROFILE=2`, every launch alone, 55 steps, 2,560 launches a step, 50.5 ms a step):

| | launches a step | each | of the step |
|---|---|---|---|
| an expert's `gate` and `up` (512 x 2,560, 4-bit) | 672 | 19 us | 26% |
| q, k and v (6,144 x 2,560) | 50 | 140 us | 14% |
| the attention's output (2,560 x 7,168 or 6,144) | 50 | 150 us | 15% |
| an expert's `down` | 336 | 12 us | 8% |
| the head (128,000 x 2,560, f32) | 1 | 2.6 ms | 5% |
| everything else: norms, adds, the router, the caches, the softmax | about 1,450 | 3 to 40 us | 32% |

* **No kernel is the cost: the count is.** **[wrong, section 20.1: a launch is 2 microseconds when timed in a stream of them, and the count is 5 ms of a step; what costs is the Q4_0 product's decoding of a weight at a time. The times of this table are of launches run alone, each with its own wait.]** A third of the step is 1,450 launches of 3 to 40 microseconds, and another third is 1,008 expert products of 12 to 19: at these sizes a launch costs what it costs whatever it computes (section 8 measured an expert's product at 3 microseconds when nothing else runs between).
* **The dense products are 30% and are one row each**: with several tokens in a pass they are read once for all of them.
* **A prompt costs what an answer costs**, 78 ms a token: the head is the only thing a prompt's step could leave out. **cancho-code's primer would take 7 to 14 minutes**, every request.

### 17.3 What stands between here and there

| | now | needed |
|---|---|---|
| positions | 2,184 (no indexer) | 16,384 for cancho-code as it is run. To there the indexer chooses 2,048 keys a query among all of them, plus the first 8 and the last 128; its pool, a second choice by blocks of 16, starts only past 16,384 keys and is left out |
| a run's tables | written by the host before the run, `steps x positions` for the selectors and the masks: 20 MB each at 2,184, **1 GB each at 16,384** | none that grows with the square: the selectors and masks computed in the step from the position | **[S4]**
| a conversation | one a process: the ids and the draws are files read at the start | new ids at any request, a position to go back to |
| the load | 19 s every run | once |
| a prompt's speed | 13 tokens a second | several tokens a pass; and the prefix of the last request not computed again |
| an answer's speed | 13 to 15 tokens a second | the host's 22 ms gone, fewer launches |
| the wire | none | `/api/generate`, `/v1/chat/completions`, tools in the template and tool calls read out of the answer |

### 17.4 Stages, each a gate

* **S1. A step that takes a conversation as it comes** **[built, 17.6, smaller than written here first]**. The prompt's ids, their mask and the sampler's draws are a request's numbers, not files of a run; a new request word **`seek N`** in both sessions puts the repetition count, and so every table's block, at `N`. A session then answers any number of conversations one after another, and can go back to a position while the sliding layers' ring still holds the window before it. `scripts/kolibri` runs on it. **Gate**: `tests/kstep.sh` against awk as it was, plus conversations one after another in one session and seeks back; `tests/seek.sh`; mutants. **By hand**: section 16's runs again, the same ids. **[first written with the selectors and the masks computed in the step from the position. Not needed for this: they depend on the positions and the steps only, not on the conversation, so one set of tables serves every request to 2,184. They are 1 GB each at 16,384, and that is S4's.]**
* **S2. The host's 22 ms** **[built, 17.5: it was 34]**. The tables of a parameter found once at setup, not at every step. **Gate**: `tests/session.py`'s cases unchanged. **Measured**: the step before and after.
* **S3. The server** **[built, section 18]**. `cancho-gpu-kolibri`: the session's engine in a program that listens (the code of `device/metal-session-main.cho` as a module, the HTTP of `device/serve-main.cho`), the model loaded once, one request at a time. `/api/generate` (the prompt as one user message of the template), `/v1/chat/completions` (system, user and assistant turns), `/health`; headers at once and the body chunked; `num_predict` and `max_tokens`; the reasoning apart from the answer. The last request's ids are kept: a request that starts with them continues from where they part (S1's seek). **Gate** (no model: the toy GGUF on the CPU session's engine): the routes, the template's turns against recorded bytes, the refusals, the prefix kept and not kept. **By hand**: cancho-code's own stub test pointed at it, and one real task.
* **S4. Past 2,184 positions** **[built, section 19: caches as rolls of positions, and the indexer; 16,384 positions, the pool not]**. The selectors and the masks computed in the step from the position (S1 left them tables); the indexer's choice of keys in the indexed layers; an indexed cache written a row at a time, not whole (9 MB a step at 2,240 would be 66 MB at 16,384, eight caches); the reference gets the indexer too. **Gate**: the small model with a choice of 3 keys among 16 against awk. **By hand**: the reference on one text past 2,184, layer by layer.
* **S5. A prompt several tokens a pass** **[built, section 20: 17 ms a token at 11,000 positions where a step is 60]**. The dense products as products of rows, the attention causal inside the pass (the kernels have a batch size that is 1 today), the experts still one launch a token each. Reckoned from 17.2: the dense third and the small launches shared, the experts not: **about 20 ms a token, 50 a second, unverified**. **Gate**: the pass against the step, to the bit where the sums allow.
* **S6. Tools.** The template's tool descriptions and tool turns, written out as section 16 wrote the rest; `tool_calls` read out of the answer's text; `finish_reason: "tool_calls"`. **Gate**: recorded bytes from the model's template; made answers parsed.
* **S7. Fewer launches a token.** An expert's `gate`, `up` and activation as one kernel and `down` with its weight (section 10 left it); the six experts' launches overlapped. Aimed at 17.2's two thirds. **[section 20.1: launches are not what a token costs. What S7 should be is a path for Q4_0 in the product's reduction, and the pass's two-launch experts in the step.]**

**Order**: S1 first, because S3 to S5 stand on it. S2 is small and can go with it. Then S3: with S1 to S3 cancho-agent's kind of request works **[false as written: its requests carry tools, which are S6; what S3 gave it is the size and the route]** and cancho-code's fits when its primer and task are under 2,184 tokens, which they are not. **cancho-code needs S4 and S5 as well**: until then its primer does not fit, and after S4 alone it would fit and take minutes.

**What "good" would be, said before it is measured**: cancho-code's median request against Qwen2.5-1.5B on this machine was 3.7 s, its longest 169 s (its M8 note; no figure for the 27B was found). Reckoned for Kolibri after S5 and S3's kept prefix: a new task of 300 tokens after a kept primer 6 s, an answer of 500 tokens 30 to 40 s. **The first request after a start still pays the whole primer once**: 2 to 4 minutes at S5's reckoned speed, unless the kept prefix is also written to a file, which is not in this plan. **[measured, section 20.3: 191 s for the primer's 11,085 tokens; a second task after it 3.9 s]**

**Not in this plan**: more than one request at a time; the indexer's pool (contexts past 16,384); streaming in cancho-code or cancho-agent (neither reads a stream); a draft head (the file has none); TLS; cancho-agent's hard-coded port.

### 17.5 S2, built first: the host's share was 34 ms, not 22 **[measured, 2026-11-26, M4 Max]**

`device/metal-session-main.cho`: `tables_by_program` lists each program's tables once after `setup`, and `on_device` looks through that list for a parameter instead of through all the session's tables. Nothing else changes: the same binding, the same read-back, the same order.

| a greedy step after a prompt of 43 tokens, 34 steps, a request each, two runs each | median | range |
|---|---|---|
| before (`dd14e2e`) | 75.8 and 75.7 ms | 74.8 to 78.3 |
| after | **42.2 and 40.7 ms** | 39.7 to 44.6 |

* **13 tokens a second become 24**, and the 34 ids are the same ones. A prompt's token costs the same 42 ms.
* **17.2 reckoned the host at 22 ms from a profiled run; it was 34.** The profiled run's 53.6 ms of waiting included the profile's own printing at each of 49 waits.
* **What holds it**: the gate has no Metal GPU, so by hand on the Metal session: `tests/kstep.sh`, `tests/pick.sh`, `tests/moe.sh` and `tests/session.py --metal` pass as before. **Mutants, by hand, four, all killed** by `tests/pick.sh` and `tests/kstep.sh` on the Metal session: every table in the first program's list; a table in the list of the program after its own; the parameter not compared; the last table left out.
* **Not done**: the CPU session (`src/session.cho`) binds its tables its own way and was not looked at; the 42 ms that are left were not profiled again.

### 17.6 S1, built: a session that takes one conversation after another **[measured, 2026-11-26, M4 Max]**

**What changed.**

* **`seek N`**, a request of both sessions (`docs/resident.md` section 32): the next repetition is repetition `N`. Kolibri's tables of positions (the selectors, the masks, the rotary's cos and sin) are read by the repetition's number, so `seek 0` is a new conversation and `seek p` is back at position `p`.
* **The step's id, its mask and its uniform number are the request's**: `N ID MASK`, and `N ID MASK U` when sampling (`input cur.xt`, `input cur.m`, `input spk.u` in `models/kolibri_step.model`, where `idtab.f32`, `mtab.f32` and `u.f32` were tables of a run). A prompt is a request a token.
* **The collector of ids is gone** (`tok_acc`, its three tables, `ids_out.f32`, `kolibri_gpu ids`): every request answers the id its step chose, and `scripts/kolibri` keeps those.
* **`kolibri_gpu draws <seed> <count>`** prints the numbers `inputs` wrote to `u.f32`, a line each with all 24 decimals; `inputs` takes no seed.
* **Nothing is zeroed between conversations, and nothing needs to be**: an indexed layer's mask shows the keys to the step's position, a sliding layer's the slots whose position is in the window, and both are rewritten before they are shown again.

**What going back costs, and its limit.** A sliding layer's ring holds its last `TW` positions (576 for a window of 513). After a run to position `e`, a step `p` can run again when its window's first key, `p - 512`, is still there: `p >= e - 63`. Further back the first keys of the window are written over, and the answer is another one. Nothing refuses it: the session does not know what a seek is for. **So today going back is free for 63 positions, and a new conversation is always free.** A longer ring, or a kept copy of it at a position, is S3's.

**The step after a seek is given its id.** What the step carries is the id the last step it ran chose, not the one of the step before `p`. The caller has it (it is the answer of step `p - 1`) and sends it with a mask of 1.

| on the Metal GPU, Kolibri 1.1, by hand | |
|---|---|
| section 16's three runs through `scripts/kolibri` (the German question, greedy; the sum, greedy; the sum, sampled with `--seed 3`) | the kept runs' ids: all 7, all 105, all 137 |
| the 83 numbers of seed 1 from `draws`, read as f32 | `u.f32`'s of a kept run, to the bit |
| **one session**: a prompt of 44 tokens and its 34 greedy ids; `seek 0`; another prompt of 46 and its 8; `seek 0`; the first again; then `seek 53`, into the first's answer | each the ids of its own run alone; from step 53 on, the answer's ids from its 11th |
| the first conversation, with the load | 18.0 s |
| **the second, in the same session** | **2.3 s** (46 + 8 steps) |
| a request of a step, 230 of them, prompts and answers alike | median 44.2 ms |

* **A second question no longer pays the 16 to 19 s of the load.** That is the whole gain here; a token costs what it cost (42 ms in 17.5's form, 44 with the request's numbers sent each time: one run of each, not compared further).
* **The expansions of the model file** for `GEN=1` differ from the earlier ones in these lines only: two tables become inputs (three when sampling), and the collector's seven lines go. The teacher-forced forms are the same bytes (`tests/model.sh`: their recorded sums unchanged; the three generating ones recorded again).

**In the gate** (CPU session; each also run by hand on the Metal session, passing):

* `tests/seek.sh`, new, 17 checks, exact.
* `tests/kstep.sh`: the generating and sampling runs through requests that carry their id, mask and number, against awk as before; then **in one session** the first prompt, `seek 0`, the first prompt again (awk's ids), `seek 0`, a second prompt (the ids of a session that starts with it, which must not be the first prompt's), `seek 2` and its third step given its id (the same ids from there); and on the model with a ring of 4 slots for a window of 3, after 14 steps: back two steps (the two again, exactly), back to 0 (the 14 again, exactly), and **back three, where the answer must differ**: the limit above, shown rather than assumed.
* `tests/kolibri_host.sh`: `draws` against awk's arithmetic to the last decimal; `tests/kolibri_cmd.sh`: the requests the script sends, with the seed's numbers in them.

**Mutants, by hand: 30, all killed.** Of the session's `seek`, nine, by `tests/seek.sh` (five of them also by `tests/kstep.sh`): the count one further; nothing put; any number of words; a negative number, a fraction and 10^9 taken; no refusal in a session with no steps; a seek that answers and is then read as steps too; a seek read as steps only. Of the model file, four, by `tests/kstep.sh`: the id and the mask exchanged on the request; the id always the request's; never the request's; the number drawn with always 0. Of `kolibri_gpu`, six, by `tests/kolibri_host.sh`: 25 bits of the state; 23 decimals; the seed not used; a number fewer; the state not carried; `inputs` still taking a seed. Of `scripts/kolibri`, eleven, by `tests/kolibri_cmd.sh`: the last prompt token not given as the request's; every new step given it; the prompt's ids without their mask; a prompt token too many, and too few; the numbers not read; the same number at every step; a greedy run sent numbers; the seed not passed; the end token in the answer; the answer an id short.

* **Not done**: the server (S3); a ring longer than the window, or a copy of it, for going back further; the tables are still written for a run's length by `scripts/kolibri` at each run, where a server writes them once for 2,184; a prompt is a request a token, which in a shell costs a process's round trip each (not measured apart).

## 18. S3: the server (design, 2026-11-26, written before the code; built, 18.1)

`cancho-gpu-kolibri` listens and answers with the model of a session that stays loaded. What it is made of, and the choices:

* **The engine is a session, and the server speaks to it in the session's own request lines.** The server's logic (`kolibri/kserve.cho`) is a machine that, for a request, says the next line for the session (`seek 40`, `1 325 1 0.51...`) and takes the id it answers, until it is done. It never touches a device. Two programs run it: `cancho-gpu-kolibri` on the Metal session's engine (the code of `device/metal-session-main.cho`, moved to a module), and `cancho-gpu-kolibri-cpu` on the CPU session's. **The gate runs the second**, on a session file that is no model (a table says the next id from the id and the position): everything of the server but the model's arithmetic, with no GPU.
* **The routes**: `POST /api/generate` (Ollama's: `prompt` is one user message of the template, `system` a system message), `POST /v1/chat/completions` (OpenAI's: `system`, `user` and `assistant` messages), `GET /health`, `GET /v1/models`, `GET /api/tags`. One request at a time; a connection carries one request.
* **The answer's head goes out before the model runs.** cancho-code's client waits 30 s for a response's head. So a request is read, checked and tokenised first (every refusal that can be known then is a 400 with its message); then `200` and `Transfer-Encoding: chunked` are sent, and the body is one chunk when the answer is whole. A failure after that can only be said in the body.
* **The template** (`kolibri/kchat.cho`, which `kolibri_gpu prompt` now uses too): section 16's, with earlier turns. An assistant turn before the last question is written without its reasoning, as the template does; a later system message as a turn of its own. Tools and tool turns are refused (S6) **[section 21: written, as the template writes them]**.
* **Reasoning is off unless asked**: effort `none` unless the request says `reasoning_effort`, `think` or `enable_thinking`; cancho-code sends `think: false`. With reasoning, what is before `</think>` is `reasoning_content` (`thinking` for Ollama) and the rest the answer.
* **The prefix kept.** The server remembers the ids the session has run. A request's ids are compared with them; the session goes back to where they part (`seek`) and runs only the rest. Going back is exact while the sliding layers' ring holds the window before that position (17.6). **The server's session has a ring as long as its positions (2,240 slots, not 576), so it never wraps and any position can be gone back to.** That costs a longer scan in 40 layers, to be measured **[18.1: 2 ms of a step's 46, and the long softmax in those layers, which the design did not foresee]** **[19.4: the ring is gone; a cache is a roll of positions, and any position is gone back to without it]**; it does not survive S4 (a ring of 16,384 slots in 40 layers), which will need a copy of the ring instead.
* **Sampling is the session's, set at start**: the temperature, the largest kept and top-p are constants of the model file. A request's `temperature` is not read. `--greedy` starts a session that takes the arg max. The request's `seed` chooses the draws; with none each request has its own.
* **What it holds to**: at most the 2,184 positions; a prompt that does not fit is a 400. `num_predict` / `max_tokens` are cut to the room left, and the answer then ends with `length`.
* **Not here**: streaming (neither client reads a stream), tools (S6), more than one request at a time, a key, TLS.

### 18.1 Built and measured **[2026-11-26, M4 Max, the Metal GPU]**

```sh
scripts/build-kolibri.sh && scripts/build-kolibri-serve.sh
scripts/kolibri-serve [--port N] [--greedy | --temperature T] [--api-key KEY] [--name MODEL] [--max-new N]

curl localhost:11434/api/generate -d '{"prompt":"Write three sentences about the sea.","think":false,"options":{"num_predict":60}}'
curl localhost:11434/v1/chat/completions -d '{"messages":[{"role":"user","content":"Und von Italien?"}]}'
```

`scripts/kolibri-serve` writes the session's tables once, for the 2,184 steps and a ring of 2,240 slots **[19.4: no ring]**, and starts `build/cancho-gpu-kolibri`.

| by hand, `--greedy` | |
|---|---|
| from start to listening | 12 s, the weights in the file cache |
| `/api/generate`, section 17.2's prompt (44 tokens) | section 16's 34 ids, the same text; 7.4 s, the first request's programs with it |
| `/v1/chat/completions`, section 16's German question | "Die Hauptstadt von Frankreich ist Paris.", 7 ids; 1.3 s, 28 of its 49 prompt tokens kept from the request before (the template's start) |
| the same with that answer and "Und von Italien?" after it | "Die Hauptstadt von Italien ist Rom."; 1.5 s, 45 of 70 kept |
| a step, from those requests | **46 ms** (28 steps in 1.29 s, 32 in 1.48, 2,089 in 96.6 s) |
| **cancho-code's own agent** (`scripts/cancho-agent.sh --provider ollama --port`, its build as it is), a file of 61 functions with one type error and its feedback: a prompt of 1,749 tokens | the first sample 96.6 s; **the second and third 20.1 s each, 1,748 of the 1,749 kept** |
| its answers | each cut at 436 new tokens, the room left of 2,184: the agent reads `length` and asks again with a higher bound, which there is no room for |
| started with `--temperature 0.2`: the sheep sum with `reasoning_effort: low`, and "What is 12 times 12?" with `think: true` | `content` "30 sheep." with `reasoning_content` "We need answer simple. 17+23=40-10=30."; `response` "144" with `thinking` "We need answer one line. 12*12=144." |
| the same agent on a file of two lines (135 tokens) | three samples of 14 tokens, 0.7 s each after the first; none has an `edition` line, so the agent takes none |

* **The wire holds for cancho-code**: its client took an answer whose body came 96 s after its head. A wait of 30 s for the head would have ended it.
* **cancho-code cannot use it yet, for the reason section 17 gave**: a real request of its own, without its primer, already fills the positions. With the primer the prompt is refused (a 400 that says so). Its three samples are also one answer three times when the server is greedy; they differ when it samples.
* **A ring as long as the positions costs little**: 46 ms a step against 17.6's 44, with the long softmax in the 40 sliding layers over 2,240 slots. The short softmax does not lower past 640 slots (its tile is 108 KB of a threadgroup's 32 at 2,240), so `models/kolibri_step.model` takes the long form in a sliding layer when `TW > SHORTW` (640); under it nothing changes (the recorded expansions are the same bytes).
* **The first request is slow once** (7.4 s for 78 steps, 3.8 s over the steps' own time). Not looked into.

**In the gate**, with no model and no GPU:

* `tests/kolibri_serve.sh`, new, 74 checks on `cancho-gpu-kolibri-cpu` with a session of two tables for a model, through a client written for it (`tests/httpc.cho`): the routes and the key; 26 refusals, each a 400 with its message and nothing asked of the session; **the prompts of the 25 requests of `tests/kolibri_chat/` are the model's own template's, id for id** (recorded through jinja2 by hand, `golden/kolibri_chat/`: turns, an assistant's reasoning kept and dropped, its `<think>` tags, a later system message, parts, every effort and the three ways to ask for one, Ollama's prompt and system, raw); answers against awk to the limit, to either end token, to the session's last position; what the session is asked when a request repeats, goes on, parts from the last; a ring of 16 for a window of 10 where going back 6 is taken and 7 is not; the numbers drawn with, at the steps' positions; a session that refuses.
* `tests/kstep.sh`: the sliding layers in the long softmax, the same rows as in the short one. `tests/kolibri_host.sh`: `inputs` with a ring asked for.

**Mutants, by hand: 26 of the server's logic, 25 killed by `tests/kolibri_serve.sh`.** Of what is kept: every id counted as kept; the start not held to the prompt's last token; the ring's edge a position further, and a position nearer; the ring not looked at when it has wrapped; a ring that has not wrapped treated as one that has; the ids past the prompt not remembered. Of an answer: the second end token not an end; a new id too many; a step past the last position; the last prompt step not read as the first new id; the prompt's ids sent without their mask; the numbers drawn with from the seed's first wherever the job starts; the seed not used. Of a request: `think: false` not turning reasoning off; a later system message written as a user's; raw, `max_completion_tokens`, the request's model name and the system of a generate request not read; what was kept said as the whole prompt. Of the wire: the chunk's length one over; the key not looked at; `/health` asking for the key; GET run on a route that takes POST. **The one not killed is no mutant**: an assistant turn at the index of the last question keeping its think block, where that index is a user's turn whenever there is one, and a request with none is refused. **Three survived the first run and showed the test's gaps**: no changed ending lay exactly at the ring's edge, no case had a ring that had not wrapped, and every chat answer of the stand-in ended at once (its template ends in a newline, whose id was an end), so `max_completion_tokens` cut nothing. The test now puts the edge on a distance it has, has the unwrapped case, and its stand-in ends after letters only.

* **Not done**:
  * **A request's `temperature`, `top_p`, `top_k` are not read**: they are the session's, set when it starts (`--temperature`, `--greedy`). cancho-code sends 0.2 and gets what the server was started with.
  * **No limit to the wait for a request**: a client that sends half a request holds the server. The Qwen server has the same gap.
  * **The answer's reasoning apart from its text is not in the gate**: its stand-in answers letters. By hand it is.
  * `created_at` and the durations of Ollama's answer are not written; `/api/chat` is not a route; no stream.
  * A failure of the session after the head is gone is a `200` with an error in its body, and the server stops.
  * `developer` is taken as `system`, which the template itself does not know: this server's reading.
  * The HTTP head's parsing is a second copy of `device/serve-main.cho`'s, not shared with it.

## 19. S4: past 2,184 positions (design, 2026-11-26, written before the code; S4a built, 19.4; S4b built, 19.6: 16,384 positions)

The server holds 2,184 positions and cancho-code needs 16,384 (section 17.1). Three things stand in the way, and they are taken in two steps: **S4a** makes a position cost what it should in memory and tables, at the arithmetic that exists; **S4b** adds the one piece of the model that is not written, the indexer.

### 19.1 What grows with the positions today **[read from `models/kolibri_step.model` and `kolibri_gpu inputs`; the sizes reckoned]**

| | at 2,240 | at 16,384 |
|---|---|---|
| `sel.f32`, `keep.f32`, `maskg.f32`: a row of the positions for every step | 20 MB each | **1.07 GB each** |
| the same three for the ring, when it is as long as the positions (section 18) | 20 MB each | 1.07 GB each |
| a cache rewritten whole at every step (`kv_write_b`: `c keep + new sel`), 88 of them | 4.6 MB each, 400 MB a step | 33.5 MB each, **2.9 GB a step** |
| a sliding layer's scores and softmax over the ring, 40 layers | 2,240 slots | 16,384 slots for a window of 513 |
| an indexed layer's scores and softmax over its keys, 10 layers | 2,240 | 16,384; and past 2,184 keys the model attends to a choice of them (S4b) |

Section 18 made the ring as long as the positions so that any position could be gone back to. That does not reach 16,384: the tables alone are 6.5 GB.

### 19.2 S4a: a cache is a roll of positions

**A cache holds one block a position, in the order of the positions, and the session writes and reads it by the repetition's number.** No ring, no selectors, no whole-cache write.

* **The layout changes from head-major to position-major**: `[positions, heads x 128]` where it was `[heads x positions, 128]`. Position `p` is then one block of the buffer, and the keys of positions `a` to `b` are the blocks `a` to `b`.
* **Three additions to the session**, none of them Kolibri's (`docs/resident.md`):
  * **`store P.OUT NAME BLOCKS`**: a *roll*, a device buffer of `BLOCKS` blocks of `P.OUT`'s size, zero at the start. At repetition `j` the output of `P` is block `j`: the program writes into the roll, and nothing is copied. A repetition past the last block is refused.
  * **`window Q.PARAM NAME`**: `Q.PARAM`, which is `n` blocks long, is the `n` blocks of the roll that end with the repetition's, or the first `n` while there are fewer. A sliding layer's keys are 576 blocks of it; an indexed layer's are all of them, which by the same rule is the roll from its start.
  * **`table ... hold`**: the block `min(j, BLOCKS - 1)` where a table takes `j mod BLOCKS`. A sliding layer's mask over its window depends on the position only while the window is not full: 576 rows of 576, then the last row for ever.
* **The step's new key and value are stored as they are**: `dec_post_g`'s `kn` and `vn` (or their rounding, in an indexed layer) are the roll's block. `kv_write_b` and its 88 launches a step go.
* **The attention kernels read a position-major cache**: `attention_scores_p`, `attention_pv_p`, `attention_pvl_p`, the three existing ones with their loads turned. `attention_stats` is unchanged (it reads scores).
* **An indexed layer's mask is computed** **[19.6: by the session's `choose`, which took `pos_mask`'s place]**: `pos_mask`, one launch a step for the ten layers: 0 on the keys to the step's position, -1e9 after it, from the position (a table of one number a step) and a row of 0, 1, 2, ...
* **A sliding layer scans 576 keys at any context**, in the order of their positions. So its softmax is the short form again, and section 18's long form for a long ring is taken out.
* **Going back is free to any position**, with no ring to have wrapped: a block after the position sought is hidden by the mask until the step that writes it again. The server's `--ring` and `--window` stay for a session that has a ring; Kolibri's has none.
* **What a run's files are then**: `cos.f32` and `sin.f32` (64 numbers a step), `pos.f32` (one a step), `iota.f32` (one a position), `maskw.f32` (576 x 576, whatever the context). 2.5 MB at 2,184 steps, 9.9 MB at 16,384.
* **The sums change their order** in a sliding layer (positions, not slots), so past the first 576 steps the numbers are not the ring's to the bit. Section 15 measured what that is: within 6e-6.
* **What S4a does not do**: an indexed layer still attends to every key, which is the model's arithmetic only to 2,184. `kolibri_gpu inputs` goes on refusing more steps than that. S4a is measured at 2,184 and its cost reckoned from there. **[S4b, 19.6, is the rest]**

**The gate.** A test of the three directives alone on both sessions, with small whole numbers (`tests/roll.sh`): a roll written and read back whole, a window that slides, `hold`, `seek` back over a roll, each refusal. `tests/kstep.sh` against the same awk, which has no ring and no roll: the rows of every step, generation, sampling, a seek back to any step (exact now, where section 17.6's test showed a limit). The three kernels and `pos_mask` in `tests/cases.txt`. Mutants of each.

**By hand, on the Metal GPU**: section 16's three runs (the same ids, where the sums' order allows: they are short); section 15's 2,184 tokens layer by layer against the reference's kept rows; a step's time at 64 and at 2,184 positions; the server's runs of section 18.1 again.

### 19.3 S4b: the indexer (first sketch; the design is 19.5)

Past 2,184 visible keys an indexed layer's query attends to 2,048 keys chosen by a score (the first 12 query heads against key head 0, both rounded, the ReLU of each product, averaged), the first 8 and the last 128. Choosing 2,048 of up to 16,384 by rank, with ties to the earlier key, is not an operation of the kernel language. Two ways are open, and which is taken is decided by measuring the first: the scores read back and the choice made by the host, one read a layer as the experts' choice is read today; or a rank counted on the device. The reference (`ref.py`) has no indexer either and must get one first. The pool, a second choice by blocks of 16 keys, begins past 16,384 keys and stays out.

### 19.4 S4a, built and measured **[2026-11-27, M4 Max, the Metal GPU]**

Built as 19.2 says, with one thing found on the way: the model file's own default (`T = 64`) could no longer be a session, since a sliding layer's window is 576 blocks of a roll of `T`; the default is 576 now, and `kolibri_gpu inputs` refuses fewer positions than the window rounded up.

| | before (`586bb7e`) | now |
|---|---|---|
| a run's tables at 2,184 steps and 2,240 positions, reckoned from their shapes | 81 MB with a ring of 576; 128 MB with section 18's ring of 2,240 | **2.5 MB** (and `x0.f32`, 22 MB, which only a teacher-forced run reads) |
| the same at 16,384 | 3.4 GB; 6.5 GB with the ring as long | 9.9 MB |
| programs a step (greedy) | 2,558 | 2,471 (no cache writers; one program for the indexed layers' mask) |
| a greedy step, a request each, at 2,184 positions of tables | 46 ms with section 18's ring; 44 with a ring of 576 (17.6) | **41.9 ms** (median of 230 requests) |
| a step at positions 100 to 500, and at the last 300 before 700, 1,300 and 2,160 | section 15: 78 ms at any of them, before 17.5 | 42.6 to 42.9, and 43.0 to 43.1 |
| going back | 63 positions, or any with section 18's ring | any position |

**By hand**, `scripts/kolibri` and the session:

| | |
|---|---|
| section 16's three runs (7 ids; 105 greedy; 137 sampled with `--seed 3`) | the kept runs' ids, all |
| **one session**: from section 15's prefixes of 700, 1,300 and 2,160 tokens, each after `seek 0`, greedy | the 16 ids after 700 and after 1,300 are the reference's; after 2,160 the first 11 of 24, the twelfth where the reference's two best are 0.007 apart: **section 15's result, to the id** |
| the prefixes | 37.7 s, 55.3 s, 92.4 s: 43 ms a token |
| the server (`scripts/kolibri-serve --greedy`) with cancho-code's agent on section 18.1's file of 1,749 tokens | 96.6 s, then 18.9 s and 18.8 s with 1,748 kept (20.1 before); the chat's second turn the same answer |

* **The sums' order did not show**: section 15's ids are the ones a ring gave, past three turns of it. Layer by layer against the reference's kept rows was not run again; the ids were.
* **A sliding layer costs the same at any context, and so does the whole step to 2,184**: what an indexed layer scans grows with `T`, and at 2,240 it is not seen in the step's time.
* **Not measured: the step at 16,384 positions.** `inputs` refuses more than 2,184 steps until the indexer is written, and a step with ten layers scanning 16,384 keys was not run with it switched off. That is S4b's first measurement.

**In the gate**: `tests/roll.sh` (33 checks, the three directives alone); `tests/kstep.sh` against the same awk: every earlier check on the rolls, and seeks back two steps, three (where 17.6's test showed another answer, and now must be the same), to the middle and to 0, each exact; `tests/kolibri_host.sh`: the six files as awk has them, the same mask whatever the run, a window of 65 scanned as 128; the four kernels in `tests/cases.txt`; `golden/model/`'s five sums recorded again (the expansion's changed lines are the caches' and nothing else: 10,868 lines to 10,431).

**Mutants, by hand: 33, all killed.** Of the session, twelve, by `tests/roll.sh` (eight of them also by `tests/kstep.sh`): a store that writes the block after the step's, and one that writes nothing; a window that ends a block early, that is always the roll's start, that starts at the step's block; a step past the roll's last block taken; a table that holds going round, and staying a block early; a stored output also answered; a reader listed before the roll; a window longer than the roll, and one that is no whole number of blocks. Of the model file, six, by `tests/kstep.sh`: a layer reading the values of the layer before; the keys stored as values; an indexed layer's keys stored before they are rounded; an indexed layer with the sliding mask; a sliding layer's mask going round; the position a step late. Of the kernels, eight, by `tests/kstep.sh`: the keys of another head, and the first chunk's for every chunk; the values of head 0 for every head, in both softmax forms, and the first chunk's for every chunk; the mask hiding the step's own key, showing the one after it, and no mask. Of `kolibri_gpu inputs`, seven, by `tests/kolibri_host.sh`: the mask showing a key before the window, and one after the step; the positions and the columns from 1; a mask row a step; a window longer than the positions taken; the window not rounded up.

* **Taken out again**: section 18's long softmax in a sliding layer (`SHORTW`) and `inputs`' ring argument, a day old: no sliding layer scans more than its window now. `kv_write_b`, `attention_scores_b`, `attention_pv_b` and `attention_pvl` stay for the models that use them.
* **Not done**: the saves of a step's new keys and values (`SAVE=2`'s `kn<i>.f32`, `vn<i>.f32`) are gone, since a stored output is not saved; S4b, and with it any context past 2,184.

### 19.5 S4b: the indexer (design, 2026-11-27, written before the code; built, 19.6)

**Measured first** (the Metal GPU, the machine also running the gate): with caches of 16,384 positions and every indexed layer scanning all of them, a greedy step is **52.5 ms** (median of 230 requests; 41.9 at 2,240), and the ids of section 17.6's conversations are the same. So ten layers scanning 16,384 keys cost about 10 ms a step, and nothing but the choice of keys is missing.

**The rule** (`Kolibri1_1Indexer.select_tokens`, read again). A query of an indexed layer that sees `n` keys, `n` over 2,048 + 8 + 128:

* the score of key `c` is the sum over the first 12 query heads of `relu(q_h . k_c)`, the query heads and key head 0 both rounded to NVFP4, summed in float32 and weighed by `1 / (12 sqrt(128))`;
* the first 8 keys and the last 128 are attended to whatever their score;
* of the others, the 2,048 with the highest score, a tie going to the earlier key;
* every other key is masked. With `n` at most 2,184 every key is attended to.
* **The pool is left out**: it narrows the candidates only when a query sees more than 16,384 keys. So the limit after S4b is 16,384 positions, refused past that.
* The 8, the 128, the 12 and the 16,384 are the model's configuration and not in the GGUF: constants of the model file, as section 15 has the first two.

**Where the choice is made: the host, behind a directive of the session.** Counting ranks on the device is 16,384 x 16,384 comparisons a layer, 2.7 billion a step. The host's is a selection over 16,384 numbers. And the session already reads a value back in the middle of a step to pick the experts.

* **`choose P.PARAM by Q.OUT TOP FIRST LAST`** (both sessions; `docs/resident.md`): `P.PARAM` is a mask the session writes when `P` is about to run in repetition `j`, from the f32 output `Q.OUT` of as many numbers, `Q` listed before `P`. Entry `c` is 0 when `c <= j` and one of: `c < FIRST`; `c > j - LAST`; among the `TOP` largest of `Q.OUT` over the other entries to `j`, a tie to the lower `c`. Every other entry is -1e9. While `j + 1 <= TOP + FIRST + LAST` every entry to `j` is 0 and `Q.OUT` is not read. A number that is not one (NaN) is never among the largest.
* **A kernel for the scores**, `index_score`: the first 12 rows of the rounded queries against head 0 of the keys' roll, the ReLU, the sum over the 12, a chunk of keys a threadgroup. And `fp4_round` on the layer's queries, which exists.
* **`choose` takes the place of `pos_mask`**, a day old: it writes the mask of every key to the position too, and one supplier is one form. `pos_mask.chx`, its table and its row of numbers go.
* **A device session waits for what is encoded before it writes a mask** (the buffer may still be read by the repetition before), as it does to read a picked value: ten waits a step, where the experts' choices are 48.

**The reference gets the indexer first** (`ref.py`, by hand, numpy from the model's code), run teacher-forced over 3,000 tokens: section 15's 2,184 and 816 more. Its first 2,184 steps must be section 15's kept rows to the byte. It keeps, past 2,184, each indexed layer's scores and its choice.

**The bound.**

* **In the gate**: `tests/choose.sh`, the directive alone on both sessions with whole numbers: the largest, the ties, the first and the last, the regime where nothing is chosen, each refusal. `tests/kstep.sh`: the awk reference gets the indexer (the 3 largest, the first 1 and the last 2 of 16 positions) and the 14 steps must be its rows; mutants.
* **By hand, on the Metal GPU**, against the reference's 816 steps past 2,184: each indexed layer's scores (float32 sums in another order: a tolerance); **the choice itself, key for key, and where it differs how close the two scores at the edge were**; each layer alone on the reference's rows; the greedy ids after 2,400, 2,700 and 2,984 tokens; a step's time at 3,000 and at 16,000 positions.

**What it will not be**: checked against the vendor's implementation (there is none here: section 2); tried past 3,000 tokens against the reference (1.8 s a token there); the pool.

### 19.6 S4b, built and measured **[2026-11-27, M4 Max, the Metal GPU]**

Built as 19.5 says. `choose` is a directive of both sessions (`docs/resident.md` section 34), `index_score.chx` the scores, and in `models/kolibri_step.model` each indexed layer rounds its queries, scores its keys and has its mask chosen. `pos_mask.chx` is gone. `kolibri_gpu inputs` takes 16,384 steps and refuses more (`kolibri-context`), and `scripts/kolibri` and `scripts/kolibri-serve` size their session for that.

**The reference** (`ref.py` with the indexer, by hand): 3,000 tokens teacher-forced in 71 minutes. **Its first 2,184 steps are section 15's kept rows to the byte**, the stream and the final rows. Past them it kept, for each of the 816 steps and each of the ten indexed layers, the scores and the choice.

**Each layer alone** (the Metal session teacher-forced on the reference's rows, a checkpoint a step from 2,150 on):

| over the 816 steps past 2,184 keys, ten indexed layers: 8,160 layer-steps | |
|---|---|
| the scores of the keys | **equal to the reference's to the bit at 8,149**; at the other 11 much of the row differs (worst 3.5e-3 of the row's largest) |
| the choice of 2,048 keys, the reference's rule on the device's scores | **the reference's choice key for key at 8,154**; at the other 6, 8 keys differ in all, of 16.7 million chosen |
| the row after a sliding layer | within 3.6e-6 of the reference's, at every step |
| the row after an indexed layer of layer 29's caches (29, 34, 39, 44) | within 7.4e-5; three of the four within 2.5e-6 |
| the row after an indexed layer of layer 4's, 19's or 24's caches | medians 2.4e-7 to 2.6e-6; 57 layer-steps over 1e-4, the worst 1.7e-3 |
| a step, every layer alone | 48.4 ms before 2,184 keys, 48.7 after |

* **The scores are sums of products of rounded numbers, and they come out the reference's to the last bit**, in another order of adding. That was not expected: 19.5 planned a tolerance.
* **The 11 layer-steps where a score row differs are a rounding that fell the other way**, as section 15 found in the cached keys: there a query's value sits on a tie of the 4-bit grid, and the device's value is the reference's but for its last bits. Six of the 11 moved the choice, by one or two keys.
* **The 57 rows over 1e-4 are section 15's kind**, in the layers whose caches have an entry rounded the other way (section 15 counted 46 such layer-steps before 2,184, the worst 2.3e-2). The layers of layer 29's caches have none.
* **The selection itself is not what this compared**: the device's choice was made again in numpy from its saved scores. What holds the session's own selection is `tests/choose.sh`, exact, against a sort.

**The 50 chained**, teacher-forced over the same 3,000 ids, the id each step would choose against the reference's:

| steps | the reference's choice at | where its two best are over 0.05 apart |
|---|---|---|
| 0 to 399 | 99.2% | 99.5% |
| 400 to 1,999, in four stretches | 95.5 to 96.8% | 96.4 to 96.9% |
| 2,000 to 2,183 | 92.4% | 93.4% |
| **2,184 to 2,399** | 95.4% | 95.8% |
| 2,400 to 2,699 | 89.3% | 90.8% |
| 2,700 to 2,999 | 88.0% | 89.0% |

* **To 2,184 this is section 15's count again** (2,102 of 2,184 here, 2,100 there; 2,072 of 2,140 clear ones, 2,068 there).
* **No step at 2,184**: the first 216 steps with the indexer agree as the 1,600 before them did. **Then it falls to 89%.** Section 15 showed that a chain leaves the reference at the first rounding that falls the other way and that the reference leaves itself as fast when given that rounding. Past 2,184 a second thing can widen the gap: two runs a little apart choose different keys. **Whether the fall is the first alone or both was not separated**: it needs section 15's control again, the reference run with the device's roundings, 70 minutes that were not spent.
* **Generating from a prefix** (greedy, the reference's forks): after 2,984 tokens the 16 ids are the reference's; after 2,700 the first 13 (the reference's margin at the fourteenth 1.4); after 2,400 the first 4 (0.56). Those margins are not ties: at these steps the device's chain is another run than the reference's, as the table says.
* **A step of the chain**: 50.5 ms before 2,184 keys, 52.7 after (the machine also running tests): the ten waits and the choice cost about 2 ms.

**In the gate**: `tests/choose.sh` (25 checks); `tests/kstep.sh`, whose awk reference has the indexer: 14 steps of which 8 choose 3 keys in each of two layers, held to 4e-5 (the largest difference is 2.1e-5; the indexer's first and last counts exchanged move a row by 3e-3 to 0.5); `tests/kolibri_host.sh`, `tests/kolibri_cmd.sh` for the new limit.

**Mutants, by hand: 22, of which 20 killed and two are no mutants.** Of the selection, by `tests/choose.sh`: one fewer of the largest; every entry equal to the edge taken, none, the later of equals; the entries at the start competing, and those at the end; the choice beginning a step late; the entries after the step attended to; the scores taken from another program; a step past the mask taken. Of the model file and `index_score.chx`, by `tests/kstep.sh`: the scores against the values; the first and last counts exchanged; a key more, a key fewer; the queries not rounded; a query head fewer; no ReLU; key head 1; the first chunk of keys for every chunk; the last query heads. **The two that survive change nothing**: the choice beginning a step early (with exactly as many candidates as are taken, all are taken either way), and the edge taken one rank lower (what lies above it is then the same set).

**cancho-code with its primer**, the server started with `--temperature 0.2`, cancho-code's own agent and build as they are (`scripts/cancho-agent.sh --provider ollama --primer guidelines.md --num-ctx 16384`, the primer the output of `cancho agent-guidelines`: 37,983 bytes, **10,950 tokens**):

| | |
|---|---|
| a file of two lines with a type error and its feedback: a prompt of 11,085 tokens, none kept | **723 s**; 21 new tokens; the file is `fn broken() -> [] int { return 1; }`, taken on the first sample |
| a second task after it (another file, a name not in scope): 11,108 tokens, **11,027 kept** | **7.6 s**; 29 new tokens; the file is `fn total(a: int, b: int) -> [] int { return a + b; }`, taken on the first sample |
| a token at these positions | 65 ms over the first request, 69 over the second's 110 steps |

* **This is the first time cancho-code's prompt fits**, and both answers were programs it took. Two tasks of two lines say that the wire and the context hold, not what the model is worth at cancho: that is cancho-code's own evaluation, which was not run.
* **The first request after a start is twelve minutes**, all of it the primer at a token a step. That is S5's. Every request after it that begins with the primer pays for its own task only.
* **A token costs more at 11,000 positions than at 3,000** (65 to 69 ms against 53): the indexed layers scan the 16,384 positions of their caches whatever is in them, and the choice is made over what is there.

* **Not done**: the control that would say whether the indexer widens a chain's drift from the reference (above); anything past 3,000 tokens against the reference, and a step's time at 16,000 positions; the pool, so nothing past 16,384; a NaN among the scores (`choose` puts it below every number: not tested); the answers' worth, here as in section 16.

## 20. S5: a prompt several tokens a pass (design, 2026-11-27, written before the code; built, 20.3)

A prompt's token costs what an answer's costs: 60 ms at 11,000 positions, twelve minutes for cancho-code's primer (section 19.6). This section is what was measured before the design, and the design.

### 20.1 Where a step's time goes **[measured, 2026-11-27, M4 Max, the Metal GPU]**

**A prompt of 10,900 tokens a request each, then 60 greedy steps**, with the shim's line a wait (`CANCHOGPU_PROFILE=1`, which costs about 3 ms a step itself):

| | |
|---|---|
| a step at positions 500, 4,000, 10,900 | 54, 59, 61 ms |
| waits a step | 60: one a layer for the experts the router chose, ten for the indexed layers' masks, the last |
| the host waiting, of the last 60 steps | 55.2 ms a step |
| of that, the GPU working | 47.6 ms |
| a layer that is not indexed (its MLP and the next layer's attention, between two waits) | 0.55 ms of the GPU |
| an indexed layer | 1.85 ms, and 0.37 before its mask is written |
| the head and the arg max | 2.8 ms |
| a wait itself, over the GPU's time in it | 0.125 ms, so 7.5 ms a step |

**Kernels alone**, as sessions of one kernel on zeros, 300 to 5,000 launches a request (a launch of a kernel of one element is 2.2 microseconds; the times below include one):

| | a launch |
|---|---|
| q, k and v for one row (`matvec_q4_rows`, 7,168 x 2,560) | 79 us |
| the same for 8 rows, 16, 32 (the weights read once, every row of x read at every weight) | 160, 570, 2,394 us: **20, 36, 75 us a row** |
| the attention's output product for one row (2,560 x 6,144), 8 rows, 16 | 73, 135, 1,019 us: 73, 17, 64 us a row |
| the same shape with NVFP4 weights (`matvec_fp4_rows`, which has a path of its own in the reduction) | 28 us |
| an expert's `gate` (512 x 2,560) in Q4_0, and in NVFP4; its `down` | 9.3, 4.7 and 6.2 us without the launch |
| a token's six experts as today, 30 launches | 192 us |
| the six as two launches (all `gate` and `up` products and the activation; all `down` products and the weighted sum) | 118 us |
| an indexed layer's four kernels over 16,384 keys (the scores of the choice, the attention's scores, the softmax's statistics, the product with the values), one query | 1.54 ms |
| the same four for 16 queries in a launch each | 6.67 ms: **0.42 ms a query** |

* **Section 17.2 was wrong about what a token costs: it is not the count of launches.** A launch is 2 microseconds, so a step's 2,471 are 5 ms of its 42. What costs is the Q4_0 product, which decodes a weight at a time (section 8 said so and did not measure it): the six experts' products are 60% of a layer's 0.55 ms, q/k/v and the output product another 28%. **S7 as section 17.4 has it (fewer launches) is worth 5 ms at most; a path for Q4_0 in the product's reduction, as NVFP4 has, is worth more** (2 to 2.8 times on these shapes by the table). That is not this stage. **[22.2: built. The products are three times cheaper and a step 7 to 13%: the shares of a layer said in this bullet were too high, and the rest of a layer was not measured.]**
* **Rows share the weights only in small tiles.** Eight rows at a weight cost 2 times one row; 32 rows cost 30 times. So a pass's dense products take their rows 8 a threadgroup.
* **An expert's product does not get cheaper in a pass**: each token has its own six. Fused into two launches the six cost 118 us where they cost 192.
* **The indexed layers' scan is 3.7 times cheaper a query when 16 queries share a launch.**
* **A wait costs 0.125 ms whatever it waits for**, and a pass has as many as a step.

**Reckoned from these for a pass of 16 tokens at 11,000 positions**, a token: the experts 5.7 ms (48 layers of 118 us), the indexed layers 4.2 ms, the dense products 1.8 ms, the waits and the small launches 1 ms: **about 13 ms where a step is 60, so the primer in about 2.5 minutes where it took 12. Unverified.** **[measured, 20.3: 17.2 ms a token, the primer's passes in 188 s]**

### 20.2 The design

**A pass is a second form of the step in the same session**: its programs take `PB` tokens (16) where the step's take one, read the same weights and write the same rolls. A prompt goes through passes while it has `PB` tokens and one more left, and through steps for the rest; its last token is always a step, which is what chooses the first new token.

**The session gets phases** (`docs/resident.md` section 35):

* **`phase NAME STRIDE`**: the programs after the line are the phase's; a request `NAME N ...` runs them alone, `N` times, and each time is `STRIDE` repetitions of the session. A request that starts with a count runs the programs before the first `phase` line, as before.
* **A roll is stored by one output in each phase.** In a phase of stride `S` the output is `S` blocks, the repetition's and the `S - 1` after it; a window is the blocks that end with the last of those.
* **`table ... span`**: the parameter is `S` blocks of the table, the repetition's and those after it (the rotary cos and sin of a pass's positions).
* **`choose`** in a phase of stride `S` writes `S` masks one after another, row `b` for repetition `j + b`, from `S` rows of scores.
* Links, carries, picks and choices stay inside a phase. `seek` is as it was: the count is the session's.

**The pass's programs** (`models/kolibri_step.model` with `PASS=1`; `P` before each name):

| | the step | the pass |
|---|---|---|
| the ids | the request's id or the carried one, its embedding row picked | `PB` ids of the request, a row picked each, then the rows as one matrix (`rows_cat`) |
| norms, residuals, the rotary, the rounding, the activation | kernels of `r` rows | the same kernels with `r = PB` |
| q/k/v, the attention's output, a dense layer's and the shared expert's products | `matvec_q4_rows` | **`matmul_q4_pass`**: the rows 8 a threadgroup, the weights' tile as it was |
| the new keys and values | `store` a block | `store` `PB` blocks |
| a sliding layer's mask | `maskw.f32`, a row a position, held | **`maskp.f32`**: `PB` rows over the window for each first position to `PTW - PB`, then one block held. The window scanned is `PTW`, the window and `PB - 1` more rounded up to chunks (576 for 513 and 16) |
| the attention | `attention_scores_p`, `attention_pv_p`, `attention_stats`, `attention_pvl_p`, `index_score` | the same with a third grid axis over the pass's rows (`_pb`, `_b`); `attention_stats` as it is, with `PB` times the heads |
| the router | `matvec_rows`, `moe_rank`, `moe_pick` | the same over rows (`matmul_rows_pass`, `moe_rank_b`, `moe_pick_b`): `PB x K` indices and weights, read back in one wait |
| a token's `K` experts | 5 programs an expert | **2 programs a token**: `expertK_gu` (the `K` gate and up products of row `b` and the activation) and `expertK_down` (the `K` down products, weighed, added to row `b` of the shared expert's output); then `rows_cat` |
| layer 49 and layer 48's MLP, the final norm, the head, the sampler | | not run: no cache is written after layer 48's attention, and a prompt's token chooses nothing |

* **`expertK_gu`, `expertK_down`, `expertK_down41` (layers 2 to 5, whose `down` is Q4_1) and `rows_catB` have as many parameters as `K` or `PB`**, which a kernel file cannot say: they are written by `scripts/gen-kolibri-kernels.sh` for `K` 2 and 6 and `PB` 4 and 16 (the gate's small model and Kolibri), and a test holds the files to the script.
* **The step keeps its kernels**: the fused expert kernels would take a third off its experts' time too, and that is left for the stage that takes the Q4_0 product on.
* **The server** sends a prompt's tokens as `pass 1 ID ... ID` while it can, and the rest as steps. What it keeps of the last request, and where it parts from it, are as they were.

**What a pass must equal.** After passes over a prefix, the caches are the step's, to the order of the sums in the products: so every step after them answers what it answers after steps. That is the bound.

* **In the gate**: `tests/phase.sh`, the directives alone on both sessions with whole numbers (a phase's programs alone, the stride, a roll written by both phases, `span`, a chosen mask of several rows, each refusal). `tests/kstep.sh`: the small model with `PB=4`: 14 positions as steps, and the same through passes and steps in each split a prompt can have; the rows after the last layer and the logits of every step after a pass against the steps', and the ids. Mutants of the session, the kernels, the model file and the server's split.
* **By hand, on the Metal GPU**: section 16's runs and section 19.6's prefixes through passes (the ids of the steps after them); a pass's time at 3,000 and 11,000 positions; cancho-code's two requests of section 19.6 again.

**What it will not be**: faster for an answer's tokens (a step is a step); checked a pass at a time against the reference (the steps after it are); a pass of another size than 16 measured.

### 20.3 Built and measured **[2026-11-27, M4 Max, the Metal GPU]**

Built as 20.2 says. The session has phases (`docs/resident.md` section 35, `src/phase.cho`); `models/kolibri_step.model` with `PASS=1` has the phase `pass`, 2,559 programs beside the step's 2,490; `scripts/gen-kolibri-kernels.sh` writes the eight kernels that depend on `K` or on the pass's size; `kolibri_gpu inputs` takes a pass's tokens and writes its four tables; the server takes `--pass N`, and `scripts/kolibri-serve` starts it with 16.

**Two things the larger session met, and what was done:**

* **The Metal shim's table of compiled kernels (4,096) was passed**: a device session now compiles each distinct kernel once. Of the 5,049 programs most are the layer before's, or the token before's.
* **An expert table (251 MB) no longer fit what was left of the server's 512 MiB of staging**, with twice the kernel texts held there. A binary table now goes up a chunk at a time, as a resident file does since `docs/resident.md` section 24, and is not held to the pool.

**Time**, the session of the server (16,384 positions), greedy:

| | steps | passes of 16 |
|---|---|---|
| a prompt's token at positions to 2,000 | 54 ms | **15.1 ms** (a pass 241 ms) |
| the same at 3,500 to 10,900 | 59 to 61 ms | **17.0 to 17.5 ms** (a pass 272 to 280 ms) |
| cancho-code's primer, 10,944 tokens | 723 s as section 19.6 ran it | **188 s** (684 passes) |
| a prompt of 3,000 tokens | 170 s | 52 s |
| start to listening | 12 s | 16 s |

Where a pass's 244 ms go at the early positions (`CANCHOGPU_PROFILE=1`): the GPU works 219 ms of them, in 57 waits. A layer that is not indexed is 3.4 ms, **0.21 ms a token** where a step's is 0.55 and 20.1 reckoned 0.175; an indexed layer is 6.9 ms, and 3.1 ms before its masks are written.

* **3.5 times, not the 4.6 reckoned.** The layer is a fifth over its reckoning, and past 2,184 keys the choice of 2,048 for 16 queries in ten layers adds 2 ms a token, which 20.1 did not count.
* **The experts are now most of a prompt's token** (0.118 ms of a layer's 0.21 by 20.1's probe), and they are what a pass does not share.

**cancho-code with its primer**, as in section 19.6 (its own agent and build, `--num-ctx 16384`, the primer's 10,950 tokens; the server through `scripts/kolibri-serve --temperature 0.2`):

| | section 19.6 | now |
|---|---|---|
| the first task: a prompt of 11,085 tokens, none kept | 723 s | **191 s**; 17 new tokens |
| the same prompt again (the agent's next samples) | | 1.2 s and 1.5 s, 11,084 kept |
| a second task: 11,108 tokens, 11,027 kept | 7.6 s | **3.9 s**; 29 new tokens, taken on the first sample |

* **Both files were fixed.** The first task's first two samples had no `edition` line and the agent threw them away; the third was `fn broken() -> [] int { return 0; }`. In 19.6 the first sample was taken. Three samples of one prompt at temperature 0.2 say nothing of passes: the prompt's caches were the first sample's for all three.
* A chat request on the same server after them (49 tokens, 28 kept): "Die Hauptstadt von Italien ist Rom.", 1.0 s.

**What a pass leaves in the caches.** In the gate, on the CPU session, a pass is the steps to the bit (below). On the Metal GPU it is another order of sums, and Kolibri turns the last bit of a sum into another token in two places: a 4-bit rounding on a tie (section 15) and a router's sixth and seventh logits on a tie (section 10).

| by hand, the same prompt as steps and as passes with the steps that are left | |
|---|---|
| 600 tokens, then 16 greedy ids | the same 16 |
| the row after every layer at the step of position 2,100 | layers 0 to 17 within 1.9e-6 of the row's peak; layer 18 1.5e-5; from layer 19 on 4e-3 to 1.2e-1 |
| the scores of layer 4's 2,100 cached keys at that step, in the three layers that read them (4, 9, 14) | **the steps' to the bit, all 2,100, in all three**: layer 4's rounded keys are the same keys |
| the same for layer 19's keys | 286 of 2,100 differ |
| position 177, a pass's row against its step's, layer by layer | within 2e-7 through layer 6; **2e-2 at layer 7** |
| the router of layer 7 there | its biased logits within 4.8e-7 of the step's, all 384. **The step's sixth and seventh are -0.3635528 and -0.3635530; the pass has both at -0.3635530, and the tie goes to the lower expert**: 180 where the step took 383, weights 0.085 and 0.098. The other five are the same, with the same weights |
| 3,000 tokens: the predictions of the last 8 prompt steps, then 16 greedy ids | 7 of 8 the same; the first greedy id another, then 9 the same |

* **A pass is the step's arithmetic within 5e-7, and the model amplifies that at a tie**: from the first token whose router is on one, the rows after it are another run's, as section 15 found between this device and the reference. Neither run is the wrong one.
* **Against the reference**, the six prefixes of sections 15 and 19.6, read through passes, then greedy:

| prefix | steps (sections 19.4, 19.6) | passes |
|---|---|---|
| 700, 1,300 | all 16 ids the reference's | all 16 |
| 2,160 | the first 11 of 24 (the reference's two best 0.007 apart at the twelfth) | the first 11 of 24 |
| 2,400 | the first 4 of 16 | the first 4 of 16 |
| 2,700 | the first 13 of 16 | the first 13 of 16 |
| 2,984 | all 16 | all 16 |

**Read through passes the device leaves the reference where it left it read through steps**, at every one of the six. The prefixes took 20 to 46 s where the 2,160 took 92.

**In the gate** (no model, no GPU):

* `tests/phase.sh`, new, 48 checks: the directives alone on the CPU session with whole numbers (the Metal session by hand, the same 48).
* `tests/kstep.sh`: the small model, with a layer more (eleven, so that the layer that reads layer 4's caches is not the last), with a pass of 4 beside the step, generating over 14 positions. Four splits of passes and steps (three passes then two steps; a step, a pass, a step, a pass, four steps; two passes, back to position 2, two passes over what is there; ten steps, back to 3, two passes over the steps' blocks): the ids of the steps after, and the last step's rows after layers 0, 4 and the last and its logits, **exactly** the all-steps run's. The last pass's four rows after three layers are its positions' steps' rows, exactly. And the eight generated kernels are what the script writes.
* `tests/kolibri_serve.sh`, 84 checks (10 new): the lines a prompt is sent as against awk, from wherever the session starts; a prompt of 4, 5 and 6 tokens; the numbers a sampling step draws with after passes; a session whose pass answers another id; `--pass` with a session that has no phase.
* `tests/kolibri_host.sh`: a pass's four tables against awk, with a window of 2 and of 65; five refusals.
* `tests/cases.txt`: the 18 new kernel files at the small model's sizes and at Kolibri's widths (30 cases; with fewer rows or positions where 16 rows of Kolibri's largest shapes are minutes in the emulators, which run every case); `golden/model/kolibri_step.pass.sha256`. The five recorded expansions of the model file without `PASS` are the same bytes.

**Mutants, by hand: 97, of which 95 killed and two are no mutants.**

* **Of the session's phases, 31**, by `tests/phase.sh` (two of them only on the Metal session: a CPU session has no read-back and no buffer of its own for a roll): a repetition that counts one; every program run whatever the request, and a phase's request running the session's own; a window that ends with the repetition's first block, and one that stays at the roll's start a repetition longer; a spanned table read in blocks of the parameter's size, and past its last block; a roll's last block not asked for; a chosen mask's rows all the first repetition's, all by the first row's scores, or only the first written; the scores read back only when the first row needs them; every phase's inputs taken, and its answers printed; a table of another phase's program read; the rules of the file, each taken out (two phases joined; a phase with no answer, a stride of 0, a request's word for a name, a name twice, no program, no `steps`; a roll stored twice in a phase, or with other blocks; a window of a roll its phase does not store; a stored output, a spanned parameter and a chosen mask that are no whole number of the stride; the second phase's store given a roll of its own).
* **Of the kernels, 28**, by `tests/kstep.sh`: in each of the batched kernels the first row's data for every row (the rows of a product, the residual, the mask, the queries, the scores, the maxima, the router's logits and places), the first tile's weights, key head 0 and value head 0 for every head, the first chunk of values, no ReLU, no biases; in the generated ones the first row for every token, an expert's matrix for another's, no activation, the first token's weights and shared expert, an expert left out, no Q4_1 offsets, another expert's group sums, two rows exchanged, a row twice.
* **Of the model file's pass, 17**, by `tests/kstep.sh`: a token's experts picked by the first token's slots, and by the slot after; a sliding layer's mask going round; keys stored as values; an indexed layer rotated, its keys not rounded, its indexer a key short and scoring against the values; a layer that reads another's caches reading the wrong layer's keys, and its values; the residual the normed row; a layer too few before the last that stores, and its MLP left out; the shared expert left out; the weights from the biased logits; a dense `down` without its offsets; the ids a slot late.
* **Of the server, 11**, by `tests/kolibri_serve.sh`: a pass that leaves no token for the last step, and one only when two are left; a pass's ids from the token after, and its first id for all; a pass that does not pass its positions' numbers by; a pass counted as one position; its ids not kept; its answer not looked at; `--pass 1` taken; passes whether asked or not.
* **Of `kolibri_gpu inputs`, 10**, by `tests/kolibri_host.sh`: the last block as the one before; the first token's mask for every token; a key more in the window; the token after seen; the scanned positions without the pass's tokens; a block fewer; zeros for the cos of a layer with no positions; a pass of 1, of 65, and one that scans more than the positions, each taken.
* **The two that survive change nothing**: a carry of another phase made again (its program did not run, so it copies what is there), and the server's count of where the session is set a pass short (the step that always follows a pass sets it).
* **Four gaps the mutants showed, closed**: a step at the last position was refused because the *phase's* spanned table ends there (no check ran a step that late: one does now); the small model's pass never ran a layer that reads another layer's caches, because with ten layers that layer is the last and a pass leaves it out (the runs with a pass have eleven layers now); a server that did not stop made the test wait for ever, where it must fail (it waits ten seconds); and the device's read-back of the scores for a mask whose first row needs none passed by a coincidence of the test's scores (a check with other counts, where the choice is not the first entries).

* **Not done**: `scripts/kolibri` (the command) still sends a prompt a token a request; a pass of another size than 16 was not measured, nor a pass at 16,000 positions; the step keeps its five programs an expert (the two-launch form would take a third off its experts); the Q4_0 product still decodes a weight at a time, which is now most of a token, a prompt's or an answer's; an indexed layer scans all 16,384 positions of its caches at any context; the first request after a start is still three minutes, and nothing writes the caches to a file; the Metal programs are not in the gate.

## 21. S6: tools (design, 2026-11-27, written before the code; built, 21.3)

cancho-agent sends `tools` and reads `tool_calls` (section 17.1); the server refuses both. This section is what was measured before the design, and the design.

### 21.1 What the model does with its own template **[measured, 2026-11-27, M4 Max, the Metal GPU, `--greedy`, effort none]**

The template of the model's file was run by hand on two requests (transformers' own compiling of it, a scratch script) and its text sent as a raw prompt to the server as it is:

| | |
|---|---|
| the tags | `<tool_call>` 127909, `</tool_call>` 127910, `<tool_response>` 127911, `</tool_response>` 127912: a token each, and the tokenizer writes them back as their text |
| a system message, a question that needs two tools, two tools (295 tokens) | two blocks and the end token, 54 tokens: `<tool_call>`, a line `{"name": "get_weather", "arguments": {` with the arguments over three lines, `}}`, `</tool_call>`; a newline; the same for `multiply` with 17 and 23 |
| the same with that turn as `tool_calls` and two tool messages after it (388 tokens) | "The weather in Zürich is currently about **4 °C** with **fog**. 17 multiplied by 23 equals **391**.", 31 tokens |
| the second request's prompt | **349 of its 388 tokens kept**: the template writes an assistant's calls as `{"name": "NAME", "arguments": ARGUMENTS}` between the tags and a newline between two, which is byte for byte what the model wrote when `arguments` is given back as the model's own bytes |

* **The model calls tools in the template's form without being held to it**: nothing here constrains what it writes. Two requests say that it can, not how often.
* **`arguments` as the model's bytes is what keeps the turn**: an `arguments` written again (one line, other spaces) would be another turn's ids, and the session would go back to the question.

### 21.2 The design

**The template with tools** (`kolibri/kchat.cho`; `kolibri/ktool.cho`, new, for what is JSON):

* **The tools block**, in the system block after the reasoning sentence: the template's text and one line a tool. A line is the request's `tools[k]`, whole, **as Python's `json.dumps(tool, ensure_ascii=False)` writes it**, which is transformers' `tojson`: the members in the order they came, `, ` and `: ` between, a string with `"`, `\`, and the bytes under 32 escaped (`\n`, `\r`, `\t`, `\b`, `\f`, else `\u00xx`) and every other character as itself, whatever escapes the request used.
  * **A number is written as it came.** Python reads and writes it again: `1e2` becomes `100.0`, `1.50` `1.5`, `-0` `0`. For such a number the prompt is not transformers' by those bytes. Said, not prevented.
  * **A member named twice is written twice**; Python keeps the last. Said, not prevented. (In an answer's call the first `name` and the first `arguments` are the ones read.)
  * A tool that is not an object is refused. What is in it is not looked at: the template does not either.
  * `tools` empty or null, or `tool_choice: "none"`: no block, and the answer is not read for calls. `tool_choice` absent, null or `"auto"`: the block. **Any other `tool_choice` is refused**: nothing here holds the model to a call.
* **An assistant's `tool_calls`**: after its content, for each call a newline (not before the first when the content is empty), then `<tool_call>`, `{"name": "NAME", "arguments": ARGUMENTS}`, `</tool_call>`. `ARGUMENTS` is the call's `arguments` as it is when a string, and written as a tool is when anything else. A call is `{function: {name, arguments}}` or `{name, arguments}`, as the template takes both. A name that is no string, or no `arguments`, is refused (the template would write half a call or stop).
* **A `tool` message**: `<tool_response>`, its content and `</tool_response>` on lines of their own; the first of a run opens a `user` turn and the last closes it. `tool_call_id` is not read: the template writes no id, and results are in the order given.
* **The last question** is the last user message that is not itself a `<tool_response>...</tool_response>` text, as the template has it (until now: the last user message). An assistant turn after it keeps its reasoning.
* **Held to the template itself**, as sections 16 and 18 did: requests in `tests/kolibri_chat/`, their prompts recorded by hand through the model's template (`golden/kolibri_chat/`), compared id for id in the gate.

**The answer's calls** (`ktool.calls`), for a request with a tools block:

* The answer's text, after its reasoning is taken off, is read for `<tool_call>` ... `</tool_call>`. **A block is a call when what is between is a JSON object with a `name` that is a string, not empty, and an `arguments` that is an object.** Anything else between the tags, and a `<tool_call>` that is never closed (the limit, an end token), stays in the text.
* **`arguments` is a JSON string of the model's bytes for that object, as they are** (21.1). `id` is `call_<the request's number>_<k>`: no two of a conversation's are the same.
* `content` is what is outside the calls, without the white space at its ends, and `null` when that is nothing. With no call the content is the answer as it was.
* **`finish_reason` is `tool_calls`** when the answer ended by itself and has a call; an answer cut by the limit is `length`, with the calls that were whole.
* Nothing stops the model after `</tool_call>`: it ends its turn itself (21.1).

**Not in it**: `/api/generate` has no tools (Ollama's are `/api/chat`'s, which is not a route); `parallel_tool_calls`, `strict`, a schema held to; a stream.

**The gate**: the template cases above; `kolibri_gpu calls <file>`, new, which prints the message `ktool` reads out of a made answer, on made answers and their refusals; and through the server with a stand-in session that answers a table's ids by position, so that an answer is any bytes: a call's body, its `finish_reason`, and the turn sent back. **By hand**: cancho-agent's request shape on the Metal GPU.

### 21.3 Built and measured **[2026-11-27, M4 Max, the Metal GPU]**

```sh
scripts/build-kolibri.sh && scripts/build-kolibri-serve.sh
scripts/kolibri-serve [--port N] [--greedy | --temperature T] ...

curl localhost:11434/v1/chat/completions -d '{"messages":[{"role":"user","content":"What is 17 times 23?"}],
  "tools":[{"type":"function","function":{"name":"multiply","parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}}}}}]}'
```

As designed, in `kolibri/ktool.cho` (new), `kolibri/kchat.cho` and `kolibri/kserve.cho`. The server's old refusals of tools, tool messages and tool calls are gone.

| by hand | |
|---|---|
| 21.1's two requests through `/v1/chat/completions`, `--greedy` | the first: `content` null, two `tool_calls` with the arguments as the model wrote them (three lines each), `finish_reason` `tool_calls`; the second, with those calls and two tool messages: 21.1's sentence, **349 of 388 tokens kept** |
| **cancho-agent's own tool loop** (`bin/live_tools.cho` with its `443, true` changed to a local port and plain TCP, in a scratch copy, built from its sources as they are: one tool `add`, a system message, 300 tokens at most), "What is 1234 plus 4321? Use the tool.", `--greedy` | step 1 a call `add` with `{"a": 1234, "b": 4321}`; step 2 "5555". 15 s for the two requests, the first after a start (211 tokens, 13.9 s); the second 0.8 s, 241 of its 263 tokens kept |
| the same loop on a server that samples (the file's settings), six questions | **six answered**: four with one call and the right sum ("5555", "17 plus 25 equals 42.", "891" to a question in German, "You have 42 apples."), one with **two calls one after the other** (5 plus 6, then 11 plus 100: "111"), and "What is the capital of Portugal?" with no call ("Lisbon.") |
| those twelve requests | the first 14.5 s; the others 0.7 to 2.9 s, 184 to 289 of their 200 to 310 tokens kept |

* **cancho-agent can use the server**: its request is taken, its tools are in the prompt as the model's template writes them, and its reading of `tool_calls`, `finish_reason` and `usage` finds what it looks for. Its two live programs still say port 443 and TLS (section 17.1); `src/model.cho`'s `ask` takes a port and plain TCP now. Nothing of cancho-agent was changed.
* **A turn sent back is the turn the session ran**: after a call the next request goes on from where the session is, with no step run again. The gate holds this with a made answer; the rows above are the model's.
* **The session goes on, it does not go back**: 21.2 said the second request "keeps" the turn; what the server asks first is the turn's own end token, at the position after the answer's last, with no `seek`.
* **Seven conversations say that it works, not how often.** Nothing holds the model to the form or to a tool's schema: a block that is no call comes back as text with `finish_reason` `stop`, and the client sees an answer with tags in it.
* **The first request after a start took 14 s for 211 tokens** (a pass is 0.25 s): the programs' first run, which section 18.1 saw as 3.8 s and did not look into. Still not looked into.

**In the gate**, with no model and no GPU:

* `tests/kolibri_serve.sh`: **the prompts of eleven more requests of `tests/kolibri_chat/` are the model's own template's, id for id** (recorded by hand through transformers' compiling of the template, `golden/kolibri_chat/`): the tools block with and without a system message and with each kind of JSON value, escapes written in other ways than Python writes them, characters outside ASCII and under 32; an assistant's calls with arguments as a string, as an object, as null, under `function` and beside `name`, with content, with none, and with a content of one newline; one tool answer, three in a row, tool answers first; a user message that is a tool answer, that only starts as one, that only ends as one; `tool_choice` `auto`, `none`, null; no tools. 36 requests in all. Then, with a session that answers a table's ids by position: 21.1's own answer of two calls after its own prompt gives the message recorded, `tool_calls` and the counts; the turn sent back is gone on from; an answer cut after its calls says `length` with the calls, and cut inside the second says `length` with the first and the rest as text; `tool_choice: none`, an answer with no call and one whose block is no call are text and `stop`; with reasoning, a block inside the reasoning is no call; Ollama's route never reads calls. 14 refusals of tools, calls and tool messages and two of a string with half a surrogate pair, each a 400. 109 checks (84 before).
* `tests/kolibri_cmd.sh`: `kolibri_gpu calls` on 22 made answers (`tests/kolibri_calls/`, `golden/kolibri_calls/`): calls with their keys in either order, braces and quotes inside their strings, a name with escapes; text before, between and after; and what is no call: arguments that are an array or a string, no arguments, no name, a name that is empty or a number, an array, no JSON, nothing, more than one object, a tag never closed, a tag inside a tag. 64 checks (60 before).

**Mutants, by hand: 72, of which 70 killed** by those two tests: 39 of `ktool` (each escape, the separators, each kind of value, what makes a block a call, where its arguments start and end, what stays text, the ids), 10 of the template (the block for an empty list, each half of what makes a user message a tool answer, the newline before a first call, where a run of tool answers opens and closes), 23 of the server (`tool_choice`, what is refused, a call under `function`, arguments as a string, whose calls are read, when an answer is read for calls and from which text, `finish_reason`, the ids, Ollama's route).

* **The two that survived were two checks that did nothing**, and are gone: that what is between the tags is an object (what is no object has no `name` to find), and trimming it (the JSON reader passes white space).
* **Two mutants were first written wrong** (one did not build; one compared a text's handle with a number it is always above, so it changed nothing) and were written again: both killed.

* **Not done**:
  * **Nothing holds the model to a call's form or to a schema**: no grammar, no retry. `tool_choice` other than `auto` and `none` is refused for that reason.
  * **A number of a tool is written as it came** and a member named twice twice, where transformers would write `100.0` for `1e2` and keep the last member (21.2).
  * A tool message's `tool_call_id` is not read, and nothing checks that a tool answer follows a call.
  * `/api/chat` (Ollama's route with tools) is not a route; no stream; `parallel_tool_calls` and `strict` are not read.
  * cancho-agent's own programs were not changed: the loop above ran from a scratch copy with one line changed. Its evaluation, if it has one, was not run.
  * The first request after a start (above).

## 22. S7: a path for Q4_0 in the product (2026-11-27; the design is 20.1's two sentences, the rest was measured as it was built)

Section 20.1 measured that a token's cost is the Q4_0 product, which decodes a weight at a time, and said what S7 should be: a path of its own in the product's reduction, as NVFP4 has. This section is that path, what it gave, and what it showed.

### 22.1 What is built

* **`dequant_q4` has a record** (`src/matmul.cho`, `src/lower.cho`), as `dequant_fp4` has: the codes' bytes, the group's scale, the row's. The two split reductions take it: **a lane reads a group's scale once, sums `x * (code - 8)` over its elements of the group, and multiplies the sum by the scale**, where before every element read its scale and its row's scale and multiplied by both. A byte of codes is two weights. No table, no helper, nothing a dialect has to have: the CUDA and the HIP are the same lines.
* **One row** (`lower`'s split reduction: `matvec_q4_rows`, the experts' kernels): 16 elements a lane a step, half a group, as NVFP4.
* **Several rows** (`batched_rows`: `matmul_q4_pass`): **2 elements a lane a step**, which is what was fastest (below). The fragment paths and the staging stay NVFP4's alone; where a Q4_0 product fits neither split reduction it takes the lazy expression, an element at a time, as before.
* **Nothing of a kernel file changed.** The 15 recorded cases with `dequant_q4` in a product are other text, and no other case of the 642.

### 22.2 Measured **[2026-11-27, M4 Max, the Metal GPU]**

**Kernels alone**, as 20.1 measured them (a launch, with its 2.2 us):

| | before | now |
|---|---|---|
| q, k and v for one row (7,168 x 2,560) | 79 us | **26 us** |
| the attention's output product for one row (2,560 x 6,144) | 73 us | **25 us** |
| an expert's `gate` (512 x 2,560); its `down` | 11.6, 8.4 us | 6.5, 7.7 us |
| a token's six experts as the step runs them, 30 launches | 190 us | 126 us |
| the six as the pass's two launches | 118 us | **46 us** |
| the same shape with NVFP4 weights, for scale | 28 us | 28 us |
| q, k and v for a pass's 16 rows | 303 us | **211 us** |
| the output product for a pass's 16 rows | 273 us | **190 us** |

* **One row is three times cheaper, and as cheap as NVFP4.**
* **A pass's rows were already sharing the decode**, 8 rows a weight, so they gain less: 1.4 times. Sixteen rows of q, k and v are 293 million products in 211 us, and one row's 18 million take 26: a pass's product is twice as dense in the GPU's time as a row's, and the weights' decode is no longer what either waits for.
* **The several-row form's speed depends on how many elements a lane takes a step, and not in order** (16 rows of q, k and v): 16 elements 442 us, 8 elements 283, 4 elements 425, **2 elements 211**; the element-at-a-time form before it 303. The same arithmetic written out with no loop and no array, 16 elements a step: 303 us, and 480 for the output product. **Why is not known**; 2 was taken because it was measured fastest on both shapes, on this GPU. One row is the same at 2, 4, 8, 16 and 32 (24 to 26 us).

**The model**, the session alone, a prompt read through passes and then steps on its ids (`--greedy`):

| | before | now |
|---|---|---|
| a pass of 16 at positions to 1,100 | 244 ms | **178 ms** |
| a pass at positions to 3,950 | 257 ms | **191 ms**: 12 ms a token |
| a step at 1,100 positions | 54.9 ms | **51.1 ms** |
| a step at 3,950 positions | 62.1 ms | **53.9 ms** |
| cancho-code's primer, 10,928 tokens through 683 passes | 188 s (20.3) | **143 s**: 215 ms a pass at the end |
| a step at 10,950 positions | 61 ms (20.1, with the shim's profile on) | 63 ms: **not cheaper, and not compared like for like** |

**A step's GPU time, from the shim's waits** (60 steps after 1,100 positions; 60 waits a step, 50 ms waited, 42 ms of the GPU):

| | before (20.1) | now |
|---|---|---|
| a layer that is not indexed | 0.55 ms | **0.40 ms** |
| an indexed layer, with what is before its mask | 2.2 ms | 2.45 ms |
| the head and the arg max | 2.8 ms | 2.8 ms |

* **A step is 7 to 13% cheaper, not twice.** 20.1 said the Q4_0 products were 88% of a plain layer's 0.55 ms (the experts 60%, q/k/v and the output 28%); **that was wrong**: with those products three times cheaper the layer is 0.40 ms. The kernels alone say the products were about 0.34 ms of it and are 0.18 now; the rest of a layer (the window's attention, the norms, the router, the choice) was never measured kernel by kernel.
* **What a step waits for now is the ten indexed layers: 24.5 of the GPU's 42 ms.** Each scans its caches' 16,384 positions at any context (20.1: 1.54 ms for its four kernels), to read the 2,048 the mask keeps. That is the next thing, not this stage: the kept keys gathered and read alone, or the scan cut to the positions there are.
* **The six experts as two launches in the step** would be 46 us where they are 126: 3.8 ms of a step's 51. Not built here.

**Against the reference** (section 15's forks, the prefixes read through passes, as 20.3): the device goes as far as it did or further at each of the six prefixes (16 of 16, 16 of 16, 11 of 24, 4, **16 of 16 where it was 13**, 16 of 16). Against the kernels before, the ids of 1,140 positions through passes and steps are the same, and a run to 3,996 parts at one position: the sums are in another order, and section 20.3's tie applies.

### 22.3 Found on the way: loading filled the server's pool **[measured]**

The first prompt of more than about 1,000 tokens stopped the server built with the new path with an illegal instruction. It was not the path:

* **Loading the model left 3% of the server's pool of words**: 96.9 of its 100 million before this stage, 99.96 with it. A session read, checked and lowered every program's kernel anew and kept all of it: 5,049 programs of about a hundred kernels. A request then had what was left, and one long enough ran out. **Before this stage the margin was 3 million words, by luck.**
* **A kernel is now compiled once for all the programs that are it with the same constants** (`src/session.cho`): each program still has its own globals. After loading: **10.8 million words where there were 96.9, and 118 MB of text where there were 367.** The time to listening is the same 16 s: the weights are what it reads.
* **A step at 10,950 positions is no cheaper than it was**, where one at 1,100 and at 3,950 is: something of a step grows with the positions, and it is not the products. Not looked into. **[23.3: it was in the indexed layers; with them gathering, a step there is 40.6 ms]**

### 22.4 Checks

* **The emitted CUDA and HIP, run on the emulators** (the gate): the five small cases with a Q4_0 product agree with the interpreter, as every case did before; `tests/kernels.py`'s float64 references hold. The 15 recorded cases are re-recorded, and no other of the 642 changed.
* **On the Metal GPU, by hand**: `tests/kstep.sh` passes and fails the same checks as before the path (it fails where a pass must be a step to the bit, which holds on the CPU session only: section 20.3); the real model, above.
* **Mutants, by hand: 13, of which 12 killed.** Nine of the path, each by the emulated CUDA of those five cases: the second weight of a byte its first, a code its nibble less 7, a group's sum without its scale, the sums scaled as NVFP4's halves (in each form), the two weights of a byte swapped, the row's scale left out, a group of 16, the record marked NVFP4. Four of the shared kernels, by `tests/phase.sh` and `tests/kstep.sh`: a key without the constants, a key without the file, the first phase for every shared program. **The one that survives, a shared program without its lowered text, is the device session's alone**: the gate's session is the interpreter's, which reads none. By hand the real model's 5,049 programs run from about a hundred lowered texts.

* **A pool that runs out is still a trap, in the server and in every program of this repository**, with no refusal and no rule tag. That is against the rule that no input reaches a crash; it is not fixed here, and it is not this model's alone.

## 23. S8: an indexed layer reads the keys it chose, and no others (2026-11-27; measured before the design, 23.1; built, 23.3)

Section 22 left the ten indexed layers as 24.5 of the GPU's 42 ms in a step. This section is what was measured of them, the design, and what was built.

### 23.1 An indexed layer's four kernels, alone **[measured, 2026-11-27, M4 Max, the Metal GPU]**

| a launch, with its 2.2 us | over 16,384 positions | over 2,048 |
|---|---|---|
| the scores the keys are chosen by (`index_score`) | 44 us | 9 us |
| the attention's scores (`attention_scores_p`) | 145 us | 30 us |
| the softmax's maximum and sum (`attention_stats`) | 417 us | 53 us |
| the product with the values (`attention_pvl_p`) | 940 us | 71 us |

* **Nine tenths of the layer is the softmax and the product over 16,384 positions of which at most 2,184 are not masked** (the 2,048 chosen, the first 8, the last 128): every other position's term is an `exp` of minus a thousand million, which is zero, and is computed.
* **Choosing needs every position; attending does not.** The scores of the choice are 44 us of the 1,550.

### 23.2 The design

* **`gather a ix`, a new operation of the kernel language** (`docs/chx.md`): the rows of a 2-d tile that a row of indices names. An index is a whole number in an f32, as every index a kernel has; one under 0, past the last row or no number reads row 0 or the last row, never outside the tile. In the interpreter and in the three dialects.
* **`choose P.PARAM by Q.OUT TOP FIRST LAST index`** (`docs/resident.md` section 34): the same choice as a list. The session writes the numbers of the entries the mask would leave at 0, rising, and -1 in what is left of the parameter's room, which must hold `TOP + FIRST + LAST`. In a phase of stride S, S rows.
* **`gather_kv.chx`**: the chosen positions' keys and values out of the two caches into `[IK, 512]` each, and a mask that is -1e9 where the list says -1. `IK` is 2,240: every key a layer can choose (2,184), rounded up to the 64 the attention's kernels take a chunk.
* **An indexed layer is then**: its scores of the choice over the caches' positions, as before; the session's list; the gather; and the attention's three kernels **over 2,240 rows, whatever the context**. The same kernels as before for a step; for a pass two new ones (`attention_scores_pg.chx`, `attention_pvl_pg.chx`) in which each of the 16 tokens has its own 2,240 rows.
* **The pass gathers as the step does**, though its indexed layers already cost less a token (20.1): with the step alone gathering, a pass's sums were in another order than the steps' and the gate's check that a pass is the steps to the bit failed at the seventh digit. **[measured before deciding: with the pass gathering too it holds again]**
* **What the choice is does not change**: the same positions, by the same rule, from the same scores. What changes is the order of the softmax's and the product's sums, so on the device the last bits.

### 23.3 Built and measured **[2026-11-27, M4 Max, the Metal GPU]**

As designed: `gather` in `src/parser.cho`, `src/elab.cho`, `src/check.cho`, `src/interp.cho`, `src/lower.cho`; the list in `src/choose.cho` and `src/session.cho`; three kernel files and a fourth for the cases (`gather_spread.chx`); `models/kolibri_step.model`. The gather of a step's 2,240 positions is 12 us alone, and the attention's three kernels over them 160.

**The model**, the session alone, as 22.2 measured it (`--greedy`):

| | before (22.2) | now |
|---|---|---|
| a step at 1,100 positions | 51.1 ms | **34.0 ms** |
| a step at 3,950 positions | 53.9 ms | **37.2 ms** |
| a step at 10,950 positions | 63 ms | **40.6 ms** |
| a pass of 16 at positions to 1,100 | 178 ms | **127 ms**: 8 ms a token |
| a pass at positions to 3,950 | 191 ms | **137 ms** |
| cancho-code's primer, 10,928 tokens through 683 passes | 143 s | **110 s**: 167 ms a pass at the end |

**cancho-code's own agent** on section 19.6's two tasks, the server sampling with the file's settings:

| | S5 (20.3) | now |
|---|---|---|
| the first request with its primer (11,087 tokens, none kept) | 191 s | **119 s** |
| a second task's first sample (11,104 tokens, 11,027 kept) | 3.9 s | **2.0 s** |
| an answer of 800 tokens | | 33.2 s: **41.5 ms a token** |

* **A step is a third cheaper at every context, and what section 22 could not explain is gone**: a step at 10,950 positions is 40.6 ms where it was 63. What grew with the positions was in the indexed layers.
* **A step still grows with the positions**, 34 to 41 ms from 1,100 to 10,950, and a pass 127 to 167 ms: the scores of the choice are over every position of the caches, and the session reads them back to choose. Not looked into further.
* **Against the reference** (section 15's forks through passes): the same as section 22 at each of the six prefixes (16 of 16, 16 of 16, 11 of 24, 4, 16 of 16, 16 of 16). Against section 22's kernels the ids are the same through 1,140 positions and part once in a run to 3,996 and once in one to 10,984: the sums' order, as before.
* **The two tasks' whole times are not compared with S5's**: the samples were other samples, of other lengths (800 tokens where the limit cut them).

### 23.4 Checks

* **`tests/kstep.sh`** (the gate, the CPU session): the small model with its indexed layers gathering, held to awk's arithmetic of the definition, which has no gather and no list; a pass is the steps to the bit, as before. On the Metal session by hand it passes and fails the same checks as before this stage.
* **`tests/choose.sh`**: the list at every step of five runs against the list awk makes of its own mask (rising order, -1 where there are fewer, room for exactly the entries and for more); three refusals, and a step past the scores. 34 checks (25 before).
* **Nine new kernel cases**, each on the CUDA and the HIP emulator against the interpreter: `gather_kv`, the pass's two kernels, and `gather_spread`, whose indices are all over the tile, under 0, past its last row and between whole numbers. Two refused kernels (indices of two rows; rows that are no f32). The six recorded expansions of the model file are re-recorded.
* **Mutants, by hand: 22, all killed.** Of the lowering, by the emulated CUDA: the next column, the next element's index, a row past the end not held, an index under 0 taken by its size, the index rounded. Of the interpreter: a row past the end the first (the emulator disagrees), the row one further, every column the first (the small model). Of the list: in falling order, padded with 0, the entries the mask leaves out, too little room taken, any eighth word taken for `index`, a mask written where a list was asked, and in a phase every row the first repetition's or written to the first row. Of the kernels and the model file: the gathered mask always 0, the values gathered from the keys, a pass's token reading the first token's keys or values, one fewer of the largest, no key at the end.

* **Not done**:
  * **The scores of the choice are still over every position of the caches**, and read back by the session to choose: what of a step still grows with the context.
  * The six experts as two launches in the step (22.2: 3.8 ms). **[section 24: built, 5.3 ms]**
  * `gather` takes f32 rows and one row of indices; nothing else was needed.
  * `attention_pvl_pb.chx` is no longer used by the model; its cases stay.
  * A pool that runs out is still a trap (22.3). The Metal programs are not in the gate.

## 24. S9: the step's experts as the pass's two programs (2026-11-27; measured before, 22.2; built)

Section 22.2 measured a token's six experts as the step ran them, 30 launches, at 126 us, and as the pass runs them, two launches (`expert6_gu.chx`, `expert6_down.chx` or `expert6_down41.chx`), at 46. The step now runs the pass's two, with one row: nothing but `models/kolibri_step.model` changed, and no kernel.

**After section 23, a step's waits** (60 steps after 1,100 positions, the shim's profile): 60 a step, each about 0.35 ms of the GPU whatever its layer, the head 2.8 ms; 23.5 ms of the GPU in 31.6 ms waited.

| the session alone, `--greedy` | before (23.3) | now |
|---|---|---|
| a step at 1,100 positions | 34.0 ms | **28.7 ms** |
| a step at 10,950 positions | 40.6 ms | **35.1 ms** |
| a pass of 16 at positions to 1,100 | 127 ms | 125 ms |
| cancho-code's primer through 683 passes | 110 s | 110 s |
| the programs of the session, with the pass | 5,049 | 3,689 |

* **5.3 ms a step, where 22.2 reckoned 3.8** from the kernels alone: 28 fewer launches a layer are also 28 fewer programs for the session to bind.
* **The ids are section 23's, every one**, through 1,140 positions and through 10,984: the two programs sum a token's experts in the order the step's `axpy_slot` chain did. On the CPU session the small model's rows are the same to the bit (`tests/kstep.sh`, unchanged, passes).
* **The server, by hand**: a chat answer, a prompt of 3,941 tokens (37.5 s; 120 new tokens after it kept in 4.3 s, 36 ms a token), and section 21's two tool calls.
* **Mutants of the wiring, by hand: 6, all killed** by `tests/kstep.sh`: the weights from the router's indices, `up` read from `gate`'s table, every `down` the first slot's, the row before the MLP where the shared expert's goes, the gate's scales in the other order of slots, the offsets of the first slot for every expert.
* **What a step is now**: 60 waits, which cost 8 ms beyond the GPU's own time, one a layer so that the session can bind the tables the router chose and ten for the indexed layers' choices; the head, 2.8 ms for 1.3 GB of f32; the rest spread evenly over the layers.
* **Not done**: `axpy_slot.chx`, `swiglu.chx` for an expert and the step's per-expert programs are no longer in the model (their cases stay); the waits; the head.

## 25. S10: the head as the file's codes (2026-11-27; measured before the design, 25.1; built, 25.2)

Section 24 left the head at 2.8 ms of a step: `matvec_rows` over 1.3 GB of f32, the file's Q6_K decoded by the host.

### 25.1 Measured before, and the design **[2026-11-27, M4 Max, the Metal GPU]**

| the head's product alone (128,000 x 2,560), a launch | |
|---|---|
| f32 weights, as it was | 2,530 us: 1.3 GB read a token, which is the memory's rate |
| the same shape with 4-bit codes (`matvec_q4_rows`), for a floor | 350 us |
| **a byte a code and an f32 scale for each 16 (built: 25.2)** | **760 us** |

* **The f32 head is bound by the bytes it reads**, not by its arithmetic. A Q6_K value is `d scale (q - 32)`: a 6-bit code, a byte of scale for each 16 values, an f16 `d` for each 256. A byte a code and `d scale` as an f32 a group is 410 MB where the f32 is 1,310, **and it is exact**: `d scale` has 19 bits and fits an f32; the value itself has 25 and did not (the f32 file was rounded in its last bit for some weights, as the reference's decode is).
* **The design**: `dequant_i8 q s g 16`, a third form of the lazy dequantisation beside NVFP4's and Q4_0's (`docs/chx.md`): codes that are signed bytes, an f32 scale a group, an f32 a row. The one-row split reduction takes its record as it takes Q4_0's (section 22): a lane reads a group's scale once and sums `x * code` over 16 bytes. No form for several rows (the lazy expression, an element at a time) and none for codes or scales a kernel computed (refused): the head is one row, loaded.
* **The host** writes `head.q`, `head.s` and `ones_v.f32` where it wrote `head.f32`; `kolibri_gpu head <gguf> <dir>` writes those three alone, and the two scripts run it on a directory of weights written before (0.6 s for the real file; `head.f32` stays there, read by nothing).

### 25.2 Built and measured

`src/` (parser, elaboration, checker, interpreter, lowering, the product), `tests/chx/matvec_i8_rows.chx` and `dq_i8.chx`, `tests/kolibri_gpu.cho`, the model file, the two scripts.

| the session alone, `--greedy`, the two builds one after the other, twice | before | now |
|---|---|---|
| a step at 1,100 positions | 29.0, 29.2 ms | **27.1, 27.4 ms** |
| a step at 10,950 positions | | 29.6 ms |
| the server: 120 new tokens after a prompt of 3,941 kept | | 3.5 s: 29 ms a token |

* **1.8 ms a step**, what the kernels alone said.
* **The ids are section 24's, every one**, through 1,140 positions and through 10,984, and against the reference the six prefixes are as they were (16 of 16, 16 of 16, 11 of 24, 4, 16 of 16, 16 of 16): the head's sums are in another order and more exact, and no first choice moved in these runs.
* **Sections 22 to 24 were timed with four stray processes on the machine**, `awk` loops left by a test run of the day before that had been stopped, each on a core for 32 hours. They were found here and stopped. **Each of those sections compares runs taken in the same state, so their differences stand; their figures are a few per cent high** (a step at 10,950 positions is 29.6 ms here where section 24, before the head, has 35.1: 1.8 ms of that is the head). The table above is two builds measured one after the other after they were gone.

**Checks.** `tests/kstep.sh`: the small model's head is codes and scales (the values it had, as whole numbers to 72 either way times one of two scales), held to awk's arithmetic; on the Metal session by hand it passes and fails what it did. `tests/kolibri_host.sh`: `head.q` and `head.s` of the toy file against awk's unpacking of Q6_K, and `head` alone against `weights`. `tests/kolibri_cmd.sh`: a directory without the codes gets them and nothing else is written again. Four kernel cases on both emulators; two refused kernels.

**Mutants, by hand: 12, of which 11 killed.** Of the product: the next element's code, the sum scaled as NVFP4's halves, the row's scale left out, a group of 8; of the interpreter: the scale of another group; of the host: a code as its 6 bits, a scale without its block's `d`, the second half of a block with the first's scales, a scale as an unsigned byte; of the model file: other row scales. **The one that survives in the gate: a code read as an unsigned byte in the emitted kernel.** The cases' bytes are 48 to 55, as every i8 case's are; a negative code reaches an emitted kernel only on the Metal session, by hand (`tests/kstep.sh`, the real model). A twelfth showed that the dequantisation's form as a tile of its own was reached by no case and needed by nothing: it is a refusal now.

**Not done**: the 60 waits of a step (next); a form of `dequant_i8` for several rows; a case with negative bytes in the gate (the cases' inputs are one generator's, whose i8 are 48 to 55).
