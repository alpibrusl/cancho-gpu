# Sampling a token on the device

> **Status: built and held to float64 numpy on the CPU interpreter and on a Metal GPU; 0.99 ms a step for 32 rows over the Qwen2.5-1.5B vocabulary.**
> Seven kernels (`tests/chx/samp_*.chx`) and a test (`tests/samp_dev.py`). Written for the batched decode step of cancho-rl (`docs/design.md` sections 34 to 37): the logits of a step must not go to the host, because a session answers numbers as text and 32 x 152,064 of them cost about 55 ms.

## 1. What it computes

Temperature sampling by inverse CDF. For a row of logits `l` (padding columns masked), `p = softmax(l / T)`, and with a uniform `u` in [0, 1) the token is the first column whose cumulative probability reaches `u`. The uniform is **data**: the surface has no random numbers and no integer type, so the caller supplies one per row and step (a request line numbers, or a `table`), and the sampler is a deterministic function of the logits and `u`. The temperature is the constant `invt = 1 / T`.

## 2. How (two levels of prefix sums, because the surface has no scan)

The vocabulary is `nq` chunks of 64 columns (152,064 = 2,376 x 64: the padded vocabulary of the Qwen2.5-1.5B), and the logits are read as `[r * nq, 64]`, the same row-major memory.

| kernel | what |
|---|---|
| `samp_max` | the maximum of each chunk of the masked logits |
| `samp_gmax` | the row maximum, written for every chunk (`[r, nq]`), the shift of the next two |
| `samp_sum` | the mass of each chunk, `sum exp((l - max) * invt)` |
| `samp_cum` | the cumulative mass over the chunks, as a product with the lower-triangular matrix of ones (`matmul_nt`; the triangular matrix, `nq x nq`, is read once for all rows) |
| `samp_pick` | `t = u * total`; the chunk is the number of chunks whose cumulative mass is below `t`; writes its one-hot and, for every chunk, the target left inside it, `t - (mass of the chunks before)` |
| `samp_local` | inside every chunk, the within-chunk cumulative mass (a product with the 64 x 64 triangular matrix) compared with that chunk's target: how many columns lie before the sample |
| `samp_tok` | the 1-based token (the id convention of `ce_stats` and `onehot_h`): the chosen chunk's `index * 64 + count + 1`; `onehot_h` then makes the embedding product's operand |

Every chunk computes its within-chunk count and only the chosen chunk's is used: the surface's offsets are affine, so there is no data-dependent window to read the chosen chunk alone. The work is about 1 GFLOP a step, mostly the triangular products.

## 3. Held to

`python3 tests/samp_dev.py build/cancho-gpu-session` (in the gate) and `... build/cancho-gpu-metal-session`: a session of the seven kernels on a small vocabulary (4 rows, 24 chunks, 100 padding columns whose logits are +100 so a missing mask shows) and, with `--big`, on the real size (32 rows, 151,936 real columns):

* **600 draws (4,800 at the real size) equal the float64 inverse CDF of the same logits**, except where the uniform lies within 1e-4 (relative) of a boundary of the cumulative distribution, where the f32 sums may fall on the other side (none did), and every token is in the real vocabulary;
* the same uniforms give the same tokens;
* **3,000 draws from one row against its softmax**: chi-squared 69.8 on 54 bins (small) and 79.2 on 64 bins (real size), against limits of 100.8 and 114.9; no padding column is ever drawn;
* the seven kernels against the interpreter on a Metal GPU (`scripts/metal-kernels.sh`, worst relative difference 1.4e-7), and goldens for the emitted Metal and CUDA.

**Mutants, each killed:** the token off by one, the chunk selection shifted, the local target without `t`, the temperature dropped in either pass, the padding mask ignored in the maximum (it survived a test whose padding logits were +8; with +100 the unmasked maximum underflows every real column, which is the test now).

## 4. Measured

`python3 tests/samp_dev.py --time build/cancho-gpu-metal-session` (Apple M4 Max): **32 rows x 152,064 columns, 0.99 ms a step** (a request of 20 repetitions, the answer the 32 ids).

## 5. Limits

* Temperature only: no top-p, no top-k, no repetition penalty (R1's sampler had none).
* `nq` must be a multiple of `G` and of `bn` (`G` 24 and `bn` 24 at the real size); the logits' padding columns must be masked by the caller's `mk`.
* The uniform of a row at a boundary of the f32 cumulative sums may pick the neighbouring token: a probability of about 1e-7 per draw, which the distribution test would not see.
* The uniforms, one per row and step, are a table or a request line: producing them (a seeded generator on the host, in cancho) and keeping them out of the policy's gradient are the caller's.
