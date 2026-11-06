# A batched decode of Qwen2.5-1.5B on the device: `qsw decode`

> **Status: built and held to the float64 forward pass and to transformers.** A session of up to 32 sequences, one token each a step, that samples on the device and returns the ids of N steps from
> one request. On the real model it generates the **same 48 tokens as transformers' greedy generation** for two chat prompts, and a 28-layer step of 32 sequences with 1,152-position caches takes **35.6 ms** (43.0 ms with LoRA adapters).
> Written for the sampler of cancho-rl (`docs/design.md` sections 34 to 40). A prompt can be fed one token a step, or prefilled (section "Prefill" below).

```
qsw decode --config FILE --wdir DIR --out DIR --root DIR --tokens FILE --lens FILE [--uniforms FILE] [--pos0 FILE] [--kv-dir DIR] [--adapters DIR] [--logits]
```

| input | what |
|---|---|
| `--config` | as `qsw write`'s (`layers d h hkv dh ff v t nb steps`, `tied`, floats as the decimal integer of their binary64 bits): here `nb` is the number of sequences (at most 32), `t` the capacity of every cache (a multiple of 64), `steps` the length of the tables, `invt` = 1 / temperature (default 1.25, the 0.8 of R1) |
| `--wdir` | the converter's files (`emb.f16`, `norm.f32`, `lNN.*`) |
| `--tokens`, `--lens` | int32 `[nb, steps]` and `[nb]`: the prompt tokens (only the first `lens[r]` of a row are read) |
| `--uniforms` | binary64 `[steps, nb]`, the random number of each sequence and step (0.5 without it) |
| `--pos0` | int32 `[nb]`, the position of each sequence's first step (0 without it) |
| `--kv-dir` | start from the caches a prefill saved there (`kcacheNN`, `vcacheNN`) instead of empty ones |
| `--adapters` | apply the LoRA adapters of the files `qsw write` wrote there (`AhNN_<group>`, `BhNN_<group>`, f16, the scale folded into Bh), the same ones the prefill and the training read (section "LoRA" below) |
| `--logits` | the session answers the logits of the last step and then the ids drawn, instead of the ids of every step |

It writes the data files and `session` into `--out`. A request is a count N: the session runs N steps (a second request continues) and answers `[nb, steps]` ids, 1-based, slot j holding step j's draw.

## How a step is built (every program an existing or new `.chx` kernel; `session` text, `steps`, `carry` and `table`)

* The ids of the step, as a one-hot `[nb, vp]` (`onehot_h`), times the transposed embedding (`matvec_rows_f16h`, docs/kernels.md section 7), are the f32 residual stream. `vp` is the vocabulary rounded up to 1,536 columns.
* Per layer, 15 programs: `rms_h`; the fused q/k/v product; `dec_post` (the biases and the rotary embedding of the row's own position, q written in the layout attention reads); `kv_write_b` for the keys and for the values (the new row into each sequence's cache with the `sel`/`keep` trick, the cache `carry`ed to itself); `attention_scores_b` and `attention_pv_b` (docs/kernels.md section 8); `cast_h`; the o product; `res_add`; `rms_h`; the gate/up product; `swiglu_h`; the down product; `res_add`.
* After the layers: `rms_h`, the output product, the seven sampler kernels (docs/sampler.md), `tok_acc` (the ids into slot j of a carried buffer), `ids_mix` (the next step's input: the prompt's own token while it is inside the prompt, else the draw) and `carry` of it.
* The tables, a block a step: cos and sin of every sequence's position, the attention mask and the cache's `sel` and `keep` (so sequences at different positions share the kernels), the uniforms, the prompt tokens and their flag.

## Held to

* **The float64 forward pass** (`python3 tests/decode_dev.py build/qsw build/cancho-gpu-metal-session`, a random model of 2 layers with the real layer shapes and 1,024 words, 2 sequences, 12 steps teacher-forced, the logits of every step): the largest difference is **1.7e-4 of the largest logit**, the cosine 1.000000 at every step, the arg max agrees at all 24 clear-cut positions.
* **transformers** (`python3 tests/decode_hf.py build/qsw build/cancho-gpu-metal-session WEIGHTS`, Qwen2.5-1.5B-Instruct, float32 on the CPU): two chat prompts (42 and 48 tokens) fed one token a step and then free-run at `invt` 1e6 (an arg max): **all 24 generated tokens agree with `generate(do_sample=False)` for each prompt**; and teacher-forced on those sequences, **the arg max agrees at all 138 positions, the smallest cosine of the logits is 0.999998**, and the sampler drew the arg max of its own logits at all 138.
* The kernels: `dec_post`, `kv_write_b` and `ids_mix` against float64 references (`tests/kernels.py`), the interpreter on a Metal GPU (`scripts/metal-kernels.sh`), goldens; `qsw decode`'s refusals by rule and its tables (`tests/qsw.py`, in the gate).
* **Mutants of the decode test, each killed** (`decode_dev.py`, one layer): the rotary sign, the cache's `keep` read as `sel`, the attention mask subtracted, the prompt flag ignored.

## Measured (Apple M4 Max, the real 28 layers, f32 caches)

| sequences x cache | ms a step (a request of 8) | peak memory |
|---|---|---|
| 8 x 256 | 11.2 | 4.2 GB |
| **32 x 1,152** | **35.6** | 7.7 GB |

The session text and tables take 2 to 3 s to write (`qsw decode`) and the session loads and compiles in about 1 s. At 400 steps, 32 sequences take about 14 s. The costs are not broken down: the products (14.8 ms) and the attention (6.8 ms) measured alone, `docs/kernels.md` sections 7 and 8, are the larger parts, and every step rewrites the whole f32 cache (`kv_write_b`) twice a layer.

## Prefill (`qsw write --no-train --kv-export DIR`, `tests/prefill_hf.py`)

A prompt of 700 tokens costs 700 decode steps (about 25 s). The prefill does it in one pass: `qsw write --no-train --kv-export DIR` writes a **forward-only session of the training forward pass** (the same programs, with the adapters if the adapters file has any; `--no-train` and no `--save` are required) over the distinct prompts (`nb` of them, `t` the length of the caches, padded), **without the head, the loss, the backward pass or the optimiser**; instead, for every layer, `kv_import` turns the keys and the values (heads-major f16, written once for each query head of the group) into the layout of the decode caches, f32, **eight copies of each prompt's** (the 8 samples of R1's group), and the session saves them to `DIR/kcacheNN` and `DIR/vcacheNN` on a `checkpoint`. The decode then starts each sequence at the last token of its prompt: `--tokens` row r holds that token, `--lens` is 1, `--pos0` the prompt's length minus one, `--kv-dir DIR`; the number of sequences is eight times the prompts.

* **Held to transformers on the real model** (`python3 tests/prefill_hf.py build/qsw build/cancho-gpu-metal-session WEIGHTS`): two chat prompts (42 and 48 tokens), prefilled, then decoded as 16 sequences (8 of each): **all 16 generate the 24 greedy tokens of `generate(do_sample=False)`**. `tests/qsw.py` (in the gate) holds the shape of the prefill session (one `kv_import` program for the keys and one for the values of each layer, their saves, no head or backward pass) and the refusal of `--kv-export` without `--no-train`; `kv_import` is held to a float64 reference, to a Metal GPU (exact) and by goldens.
* **Measured** (4 prompts of 1,152 positions, the real 28 layers): the session writes in 0.8 s, a prefill request takes **1.18 s**, writing the 56 cache files (2.1 GB) on the `checkpoint` 1.4 s; the peak memory of the prefill is 24.8 GB (the forward pass of 4 x 1,152 tokens), so it does not run beside a decode session at the same time. About 4 s of prefill for an iteration, against about 25 s for the same prompts fed through decode steps.
* `kv_import` takes `ts` = 4 positions a threadgroup on Metal: its eight f32 copies are eight tiles of the threadgroup memory budget.

## LoRA (`qsw decode --adapters DIR`, `tests/decode_lora.py`)

With `--adapters`, each of a layer's four projections (the fused q/k/v, o, the fused gate/up, down) becomes five programs in place of one, and its consumers still read it by its own name: the base product (`<p>b_L`, the converter's matrix as before), the adapter's first product `mid = x Ah^T` (`<p>m_L`, `matvec_rows_f16h`, the same kernel, k the width of the input and n = r times the number of fused projections: 48, 16, 32, 16), its cast to f16 (`<p>c_L`, as the training does), the second product `mid Bh^T` (`<p>l_L`), and `res_add` of the two (`<p>_L`, f32). The files are the ones `qsw write` writes for the training session: Acat, and the block-diagonal Bcat with the scale alpha / r folded in, both f16, so a decode reads exactly what the prefill and the training forward read. The fused kernel of the training (`gemm_lora`) is not used: it is a 64-row tile and a decode has 32 rows at most.

* **Held to the float64 forward pass with adapters** (`python3 tests/decode_lora.py build/qsw build/cancho-gpu-metal-session`: the model and the check of `decode_dev.py`, adapters with a B of standard deviation 0.05 made by `qwen_ref.init_lora`): the largest difference over 12 steps is **1.8e-4 of the largest logit**, the cosine 1.000000, the arg max agrees at 24 of 24 positions; the same logits of the model without adapters are 0.145 of the largest logit away, so the test could not pass with the adapters left out. A mutant that leaves the adapter off the down projection only gives 6.3e-2 and fails.
* **Held in the gate** (`tests/qsw.py`): without `--adapters` the session has no adapter program; with it each projection has the five programs, reading `Ah`/`Bh` of its group, and the consumers are unchanged; `--adapters` without a directory is refused (`qsw.usage`).
* **Measured** (the real 28 layers, 32 sequences, 1,152 positions, adapters of rank 16 on the seven projections, requests of 8): **43.0 ms a step against 35.8 without adapters** (+7.2 ms, 20%); 400 steps take about 17 s. The 16 extra programs a layer are small products but each is a launch; fusing them into the base products would recover part of it (not done).
* **Not checked**: the adapted decode against transformers with peft on the real model (the float64 forward pass above is the oracle, and the prefill, which applies the same adapters, is held to transformers without them).

## Two traps met on the way

* **Qwen's `generation_config.json` has `repetition_penalty` 1.05, which transformers applies in greedy generation**: compared with it on, the decode diverged from `generate(do_sample=False)` at the first repeated word, with the logits identical; `repetition_penalty=1.0` makes the comparison an arg max.
* A script in `tests/` that imports transformers imports `kernels`, which `tests/kernels.py` shadows (and runs); `tests/decode_hf.py` removes its own directory from the path first.

## Limits

* Qwen2.5-1.5B's shapes only (`dec_post` is written for 12 query heads, 2 key/value heads of 128): other shapes are refused (`qsw.shape`).
* A decode without `--kv-dir` reads a prompt one token a step (positions from 0): fine for tests, slow for real prompts (use the prefill).
* The adapters are products added in each layer, not merged into the weights, so a decode follows the adapters of the training session without rewriting the 3 GB of weights; the price is the 7 ms of section "LoRA".
* The caches are f32 and every step rewrites them whole; f16 caches would roughly halve the attention and the rewrite.
* Temperature sampling only (docs/sampler.md); a sequence that draws its end token goes on; the host truncates.
* A request of more than `steps` repetitions wraps the tables.
