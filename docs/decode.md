# A batched decode of Qwen2.5-1.5B on the device: `qsw decode`

> **Status: built and held to the float64 forward pass and to transformers.** A session of up to 32 sequences, one token each a step, that samples on the device and returns the ids of N steps from
> one request. On the real model it generates the **same 48 tokens as transformers' greedy generation** for two chat prompts, and a 28-layer step of 32 sequences with 1,152-position caches takes **35.6 ms**.
> Written for the sampler of cancho-rl (`docs/design.md` sections 34 to 39); prompts are fed by decode steps (no prefill kernel yet).

```
qsw decode --config FILE --wdir DIR --out DIR --root DIR --tokens FILE --lens FILE [--uniforms FILE] [--pos0 FILE] [--logits]
```

| input | what |
|---|---|
| `--config` | as `qsw write`'s (`layers d h hkv dh ff v t nb steps`, `tied`, floats as the decimal integer of their binary64 bits): here `nb` is the number of sequences (at most 32), `t` the capacity of every cache (a multiple of 64), `steps` the length of the tables, `invt` = 1 / temperature (default 1.25, the 0.8 of R1) |
| `--wdir` | the converter's files (`emb.f16`, `norm.f32`, `lNN.*`) |
| `--tokens`, `--lens` | int32 `[nb, steps]` and `[nb]`: the prompt tokens (only the first `lens[r]` of a row are read) |
| `--uniforms` | binary64 `[steps, nb]`, the random number of each sequence and step (0.5 without it) |
| `--pos0` | int32 `[nb]`, the position of each sequence's first step (0 without it) |
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

## Two traps met on the way

* **Qwen's `generation_config.json` has `repetition_penalty` 1.05, which transformers applies in greedy generation**: compared with it on, the decode diverged from `generate(do_sample=False)` at the first repeated word, with the logits identical; `repetition_penalty=1.0` makes the comparison an arg max.
* A script in `tests/` that imports transformers imports `kernels`, which `tests/kernels.py` shadows (and runs); `tests/decode_hf.py` removes its own directory from the path first.

## Limits

* Qwen2.5-1.5B's shapes only (`dec_post` is written for 12 query heads, 2 key/value heads of 128): other shapes are refused (`qsw.shape`).
* **No prefill**: a prompt is read one token a step (positions from 0), so a prompt of 700 tokens costs 700 steps. A prefill that writes the caches, with `--pos0` for the sequences that start after it, is the next piece.
* The caches are f32 and every step rewrites them whole; f16 caches would roughly halve the attention and the rewrite.
* Temperature sampling only (docs/sampler.md); a sequence that draws its end token goes on; the host truncates.
* A request of more than `steps` repetitions wraps the tables.
