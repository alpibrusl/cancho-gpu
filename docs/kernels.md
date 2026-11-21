# The decode kernels of an attention layer (slice 9b)

> **Status: four kernels written in `.chx`, held to lex-gpu's emitter and
> interpreter, and checked against a reference for what they compute; the
> long-context split-KV attention is not built, and §4 says precisely why. (§6: the
> runner has since split attention into two kernels over 32 threadgroups; that is
> speed, not a longer context.)**
> Issue [#6](https://github.com/alpibrusl/cancho-gpu/issues/6).

Slice 9b of cancho#251 asked for "the transformer kernel set (softmax, rope,
attention, KV cache)" in the `.chx` surface. `softmax.chx` was the first part
(with the `exp` and `log` unops, `design.md` §10). This is the rest of what one
**full-attention layer of Qwen3.8-27B** needs for one decode step. From the
checkpoint's `config.json` (read, not run): 64 layers, every fourth a full
attention layer, so 16 of them; head dimension 256; 24 query heads sharing 4
key/value heads (6 to a head); rotary on a quarter of each head (64 of 256
dimensions) with theta 10^7; and an output gate (`attn_output_gate`). The other
48 layers are linear attention (a gated delta rule with a short convolution):
not here.

## 1. The kernels

| File | What it computes | Shapes (Qwen3.8) |
|---|---|---|
| `tests/chx/rope.chx` | `[x1 c - x2 s, x2 c + x1 s, rest]`: the "rotate half" form on the first 2h dimensions of each row, the rest passed through | `r=24` (or 4) rows, `h=32`, `p=192` |
| `tests/chx/kv_append.chx` | one position of every head's key and value cache, `cache[i * tmax + pos] = new[i]` | `nkv=4`, `d=256`, `tmax`, `pos` |
| `tests/chx/attention.chx` | (`d` must be 256: see below) for each key/value head `i`, `softmax(q k^T / 16 + mask) v` for its `g` query heads over the first `t` cached positions | `nkv=4`, `g=6`, `d=256`, `t <= 1364` (§3) |
| `tests/chx/attn_gate.chx` | `o * sigmoid(gate)` | `r=24`, `n=256` |

`rope`'s tables are one position's, `[1, h]`, shared by every row; the angles are
a table lookup the host writes (lex-gpu's runtime does the same), not arithmetic a
kernel needs. `attention`'s `mask` is added to the scores, 0 for a position in
use and a large negative number for one that is not, so `t` is a capacity. The
`1/16` is `1/sqrt(256)`, written as a literal: the surface has no scalar
parameters, so a different head dimension is a different file, and **`d` must
be 256**: any other value is accepted and scaled wrongly, which the file's
comment says and the cases all respect (an earlier case with `d=32` shared the
reference's 0.0625 and so hid it). The queries'
and keys' RMSNorm per head is `rmsnorm.chx`, already there; the projections are
the existing matrix kernels.

## 2. How they are held to something

* **The emitters.** Every case of `tests/cases.txt` for these kernels (19 of the
  159 now) writes CUDA C and Metal held to the recorded goldens, or is refused where the
  record says so (`tests/golden.sh`). They were first agreed byte for byte with lex-gpu's
  Rust emitter (`851497d`, `golden/PROVENANCE`) and are regression records since (`docs/native.md` N2).
  A refusal can be partial: an over-budget attention is accepted for CUDA and refused for
  Metal, and the golden keeps the CUDA file written before the refusal
  (`scripts/record-golden.sh`).
* **The interpreter.** The `--run` summaries of the ten `--run` cases are goldens (first
  agreed with the Rust interpreter's) and agree with the float64 references of `tests/kernels.py`.
* **What they compute** (`tests/kernels.py`, in the gate). Byte-identity with
  lex-gpu says the two compilers agree, not that `attention.chx` is attention. The
  script runs each kernel with `--run` on the interpreter's own made-up inputs
  (an LCG, `design.md` §10), runs the same inputs through plain Python written
  from the mathematics in the table above, and requires the printed mean and
  peak (four decimals) and the first four values (1e-5) to agree, for both
  targets' summaries: 14 checks on each, all agree. They are not vacuous, with a caveat that was found by review: the summary
is a mean, a peak and four values, so a displaced row is invisible when the
four values are zeros. Mutating `attention.chx` (the scale changed to 0.07, the
output written to the wrong place, the mask's sign flipped) fails two checks
each, and dropping the mask is refused by the compiler itself (`error[leak]:
`mask.5` is never consumed`), the linear discipline doing what it is for. For
`kv_append` the `pos=17` case alone passed a mutation that wrote one row late;
two cases small enough that the four printed values are the whole cache
(`nkv=2 tmax=2 d=1 pos=1`, `nkv=4 tmax=1 d=1 pos=0`) now catch a transposed
placement (`kc[i + pos * nkv]`), which the larger cases cannot.

## 3. The limit that was found

`attention.chx` keeps one head's scores, `[g, t]` floats, in one tile. A tile
lives in threadgroup memory, which is capped at 32 KiB on Metal and 48 KiB (static)
on CUDA, and both compilers say so identically:

| `t` (g = 6, d = 256) | Metal | CUDA |
|---|---|---|
| up to 1364 | accepted | accepted |
| 1365 (32,784 B) | refused (`budget`) | accepted |
| 2047 (49,152 B) | refused | accepted |
| 2048 (49,176 B) | refused | refused |

Qwen3.8 has a 262,144-token context. A decode step over more than 1364
positions needs the cache split into chunks, each chunk's partial softmax
computed on its own threadgroup, and the partials merged: the split-KV scheme
lex-gpu's Rust-built kernels use. It cannot be written in `.chx` today.

## 4. What the surface lacks for split-KV attention

Tried, with both compilers answering the same each time:

* A chunk's statistics are `rowmax` and `rowsum`, which are **1-dimensional** tiles
  and can only be stored into a 1-dimensional parameter (`2 offsets for a window
  of 1 dimensions`), so the partial maxima land chunk-major and cannot be read
  back as the `[g, chunks]` tile that `rowmax` over the chunks needs.
* The merge needs `max(a, b)` of two tiles to carry a running maximum through a
  loop. `BinOp::Max` is in the IR, but the surface has no spelling for it
  (`max z mj` parses as a move of `max` followed by a stray `z`).
* The merge also needs the maximum, the normaliser and the output all carried,
  and a loop carries **one** tile (lex-gpu's `syntax.rs` says so).
* There is no transpose or reshape, so a 1-d statistic cannot be turned into the
  column a `[g, d]` output must be scaled by without going through memory.

Two ways forward, each a change to the *oracle's* repository as well as this one
(the surface is lex-gpu's, and this compiler is held to it): a surface `max a b`
(and a way to store a 1-d tile as a column) would let the merge be written in
three small kernels over a global maximum; or loops that carry more than one
tile would allow the one-pass online softmax. Neither was done here, because
changing the language is the maintainer's call and the second is a real design.

## 4b. What was built since (speed, not length)

`attention.chx` runs one threadgroup for each key/value head: 4 of a GPU's cores, 71 us a layer. The runner now uses
`tests/chx/attention_scores.chx` (a grid over the head and over slices of the cached positions: the scores,
scaled and masked, written to a scratch) and `attention_pv.chx` (a grid over the head and over slices of the head
dimension: the softmax of a head's scores and the product with `v`), 32 threadgroups each, 5 and 9 us, the same arithmetic
(`docs/native.md` N4c). That is not the split-KV of §3: no merge of partial maxima is needed, because the scores are
materialised, and for that reason `attention_pv` holds a head's `[g, t]` scores in one tile, **so its ceiling is the
same 1364 positions**. A context past that still needs the merge, and §4's list of what the surface lacks still stands.

A later route needs neither the merge nor a language change: `attention_stats.chx` and `attention_pvl.chx` take the statistics and the weighted sum a chunk of positions at a time, over the materialised scores, with `max(a, b)` spelt `a + relu(b - a)` (but start from the first chunk's maximum, not a sentinel), and run at 16,640 positions at g=4 (`docs/weight-space.md` 14.5). §4's list is still true of a one-pass split-KV merge.

## 5. Not measured

* **A GPU, once, for correctness.** `scripts/metal-kernels.sh` (needs an Apple
  GPU, so not in the gate) runs eight cases on the Metal device path of
  `docs/device.md` (rope at 24 and 3 rows, kv_append, attention at three sizes up
  to `t = 1364`, the Metal ceiling, and attn_gate at two) and compares every
  output with the interpreter's. On this machine's Apple M4 Max the worst
  relative difference is 2.4e-6 (attention at `t = 1364`); rope and kv_append are
  exactly equal. Not run on CUDA.
* **Speed.** The lowering reports `0 B of tiles` for all of them: each element
  is computed by one thread from device memory, with no staging in threadgroup
  memory. They are correct; whether they are fast is a separate measurement.
* **Real weights or activations.** The inputs are the interpreter's made-up
  numbers.
* The 48 linear-attention layers, and the decode loop that would use these.

## 6. NVFP4 for a batch: the matrix unit from registers

A decode step multiplies one vector by each weight matrix; a prefill chunk, or the tokens a speculative
step checks, multiplies several. The weights are what costs (13 GB read a token), so a batch of tokens
should cost about what one does. It did not:

* The generic path for a batch (`matmul_nt` over a dequantised weight with 2 to 16 rows of `x`) keeps one
  accumulator per row and multiplies every weight pair against every row. Measured in the running
  Qwen3.8-27B pipeline, with the small kernels left single-token, the matrix-vector products of a pass cost
  1.4x a plain token's at 2 rows, 2x at 3 and 3x at 4 (31 ms for one; 43, 62, 96): compute-bound, so
  checking four tokens would cost more than decoding three.
* `matvec_fp4_batch.chx`, `matvec_fp4_res_batch.chx` and `matvec_fp4_swiglu_batch.chx` take `frag` in their
  Apple schedule (`threads 256 bn 64 frag 1`), which selects a different lowering (`frag_rows` in
  `src/matmul.cho`): the matrix unit, fed from registers. A lane of an 8x8 simdgroup fragment holds two
  neighbouring columns of one row, which for the weights is one packed byte, so the codes go from memory to
  the fragment with no threadgroup memory, and the activations are the other operand (up to eight tokens, the
  missing columns zero). The sum over the reduction does not care in what order it is taken, so a lane loads
  a whole word of codes (eight, four 8x8 steps) at once and the activations' fragment is permuted the same
  way; the loop is unrolled so that eight words are in flight a lane. The row factor and the 2^14 of the
  half-precision decode are applied once, to the result.

The cost is flat in the number of rows (in isolation, 17408x5120: 220 to 235 us for 1, 2, 4 and 8 rows; in
the pipeline an 8-token pass of the whole model is about 63 ms, which is 8 ms a token for prefill against
42 for a decode step today). It is correct against the interpreter to about 1e-6 of the peak at 1, 3 and 8
rows (`scripts/metal-kernels.sh`) and against the plain-Python model (`tests/kernels.py`). It is not yet
as fast as the decode path for one row, which is why those kernels are separate files: the decode path is
unchanged.

`frag` is a schedule key the Rust does not have, so the Rust never held these cases: their expected output is
in `golden/` like the others, recorded from this compiler (`scripts/record-golden.sh`) and
reviewed as text; what they compute is held by the interpreter, `tests/kernels.py` and the device check.
A schedule that sets `frag` where the shapes do not fit (more than eight rows, a reduction that is not a
multiple of 32, a group size other than 16, CUDA) lowers through the generic path, so the key is never wrong,
only unused.

## 7. f16 weights for a batch of up to 32 rows: the same lowering, at the roof up to 16 rows (2026-11-06)

A batched decode step (cancho-rl `docs/design.md` sections 34 and 35) multiplies up to 32 rows of activations by the same f16 weights, so it should cost what reading the weights once costs. Measured on a Qwen2.5-1.5B step's products (28 layers of qkv 2048x1536, o 1536x1536, gate/up 17920x1536, down 1536x8960 and the output product 152064x1536, 3.09 GB; `scripts/r3_products.py` in cancho-rl, random weights, no attention; Apple M4 Max; the answer of the session is a tiny product's, so no time goes to printing): **one row takes 6.4 ms (483 GB/s), the roof of a step. `matvec_rows_f16_m` takes 6.6 ms at 4 rows, 14.0 at 8 and 47.9 at 32; `gemm_mma` 28.9 at 16 rows and 37.0 at 32.** Past about 4 rows the existing kernels are compute- and issue-bound, not bandwidth-bound.

* **What was added.** `tests/chx/matvec_rows_f16_frag.chx` (f32 activations) and `tests/chx/matvec_rows_f16h.chx` (f16 activations, what a decode step's norm kernel produces) have `frag` in their Apple schedule, which selects two new lowerings in `src/matmul.cho`, the f16 counterparts of `frag_rows` (section 6): `frag_rows_f16`, a simdgroup owning `frag` tiles of eight weight rows with the activations the other operand (eight tokens a fragment, `ceil(m / 8)` fragments sharing each weight fragment), and `frag_rows_f16_ks`, chosen when the block of weight rows is exactly one simdgroup's (`bn` = 8 `frag`): every simdgroup of the threadgroup takes a slice of the reduction and the partial sums are added through the scratch. A lane loads the eight consecutive f16 of a 32-wide block with one 16-byte load, and the columns of the fragment stand for p = (c / 2) 8 + 2 s + c % 2 for the weights and the activations alike (the sum does not care about its order). With f16 activations both operands of the matrix unit are `half`, as in the training products. Shapes or types that do not fit (more than 32 rows, a reduction that is not a multiple of 32, no fragments on the target, CUDA) lower through the generic path, so the key is never wrong, only unused.
* **Held to the interpreter on a Metal GPU** (`scripts/metal-kernels.sh`, ten new cases: m = 1, 5, 13, 32 and k from 64 to 8960, both kernels): worst relative difference **3.0e-6**; to the float64 reference by the CPU interpreter in `tests/kernels.py` (m = 3 and 32, both targets, the f16 inputs rounded for the f16 kernel); the emitted Metal and CUDA are goldens, and no existing golden moved.
* **Measured** (`bn` 16, `frag` 2, 256 threads; ms a step of the whole set of products): f16 activations **1 row 7.1, 4 rows 7.1, 8 rows 6.9, 16 rows 7.7, 32 rows 14.8 (208 GB/s)**; f32 activations 8 rows 7.2, 32 rows 16.1. So **up to 16 rows a step costs what reading the weights once costs, and 32 rows cost 2.3 times that: 3.2 times faster than `matvec_rows_f16_m` and 2.5 times faster than `gemm_mma`.** The tile of weight rows and the thread count were swept in the first (mismeasured) pass and changed little; the activation loads and the matrix-unit operations were each ablated and neither is the limit at 32 rows.
* **A mistake in the first version of this section, corrected.** It reported 69 ms at 32 rows, 8 times the roof, and explained the gap by the activations' traffic. The benchmark's session answered the output product, 32 x 152,064 numbers printed as text, which took about 55 ms of every step for every kernel; per shape the four layer products with this kernel took 19.9 ms in all. With a tiny answer the figures above are what the kernels do, and the hypothesis (activation traffic) was tested and did not hold: replacing the activation loads by a constant gave 65.7 against 69.2 ms of that mismeasured pass.

## 8. Decode attention for a batch of sequences, heads of 128 (2026-11-06)

`attention_scores_b.chx` and `attention_pv_b.chx` are `attention_scores.chx` and `attention_pv.chx` with a batch: `nb` sequences, each key/value head of each with its own cache of `tmax` positions (`k`, `v`: `[nb * nkv * tmax, d]`) and each sequence its own mask over the positions (`mask: [nb, t]`, so sequences of different lengths share one compiled kernel, `t` being a capacity). The grid has the sequence as an axis (`grid r over nb, h over nkv, sp over t / ts`; the offsets `r * (nkv * g) + h * g` are affine in the coordinates, which is all the surface asks). `d` must be 128: the scale 1/sqrt(128) = 0.08838834764831845 is a literal, as 1/16 is in `attention.chx` for 256. Qwen2.5-1.5B: `nkv` 2, `g` 6.

* **Held to** the CPU interpreter and a float64 reference (`tests/kernels.py`, both targets) and, on a Metal GPU, to the interpreter (`scripts/metal-kernels.sh`, four cases up to 4 sequences of 1,152 positions): worst relative difference **1.7e-6** (1.9e-7 for the scores). Goldens for the emitted Metal and CUDA at a small shape.
* **Measured** (cancho-rl `scripts/r3_attention.py`, Apple M4 Max, 28 layers of scores and product on random f32 caches, the caches read in full whatever the lengths, `ts` 64, `dw` 16): **8 sequences of 1,152 positions 1.73 ms a step (305 GB/s), 32 sequences 6.77 ms (312 GB/s)** for the 2.11 GB of keys and values. That is close to the bandwidth the products reach (section 7), so no further kernel work is indicated for the decode attention; f16 caches would halve it.

## 9. What limits the NVFP4 batch product, measured on the kernel alone (2026-11-21)

`docs/serving.md` 8.7 left the 16-row product at 9.5 x 10^12 multiply-adds a second against the 14.6 it took Ollama's
whole prefill to be, with two open points: why the emitted kernel collapses past a certain unroll, and what the
ceiling is. This section is the study. **Only step 1 of 9.5 is built; the rest of 9.5 is the design this leads to.**

**How it was measured.** The emitted Metal of `matvec_fp4_batch` at `m=16 n=17408 k=5120` (the golden), edited by
hand into variants and timed alone: 200 launches in a command buffer, GPU time from the command buffer's own clock,
best of four, random inputs, Apple M4 Max (40 cores), nothing else on the GPU. Every variant that computes the product
was held to the unedited kernel's output on the same inputs (checksum and three elements to six digits). **The harness
was a throwaway and is not in the repository** (a Swift driver and text edits of the golden); the numbers are one
session's. TFLOPS below is 2 x rows x n x k over the time.

**The ceiling of the matrix unit [measured].** A loop of `simdgroup_multiply_accumulate` on 8x8 fragments that stay in
registers, no memory, no decoding:

| operands | float | half |
|---|---|---|
| 1 weight fragment x 2 activation fragments | 13.0 to 13.6 | 14.6 |
| 2 x 2 | 13.9 | 15.0 |
| 2 x 4, 4 x 2 | 13.3 to 13.5 | 14.4 to 14.5 |
| 4 x 4 (16 accumulators) | 2.4 | 3.1 |

A plain `fma` loop in registers reaches 8.8 (float) and 13.4 (half). Ollama's prefill on this machine is 272 to
281 tok/s x 2 x 24.3 x 10^9 weights a token (the sum of `n k` over the products of a pass, read from the profile) =
13.2 to 13.7.

**Corrected the same day (9.6): this loop is not the GPU's ceiling.** This paragraph said the GPU does about 14 in
float and 15 in half, that a simdgroup holding 16 accumulators falls off a cliff, and that Ollama therefore runs at
the ceiling. MLX's own kernel, timed alone, does 14.7 to 14.8 with float fragments and 16 float accumulators a
simdgroup. What the loop measured is this loop; the cliff at 4 x 4 was its own, not the unit's. What stands: the
thread count per threadgroup does not matter from 64 to 256, and Ollama's whole prefill runs at about the rate of
its weight kernel.

**Where the 16-row kernel's time goes [measured, by ablation of the emitted kernel].**

| variant | us a launch | TFLOPS |
|---|---|---|
| as emitted (16 rows) | 301 to 304 | 9.4 |
| activations a constant (no loads of `x`) | 260 | 11.0 |
| no loads at all, the decoding kept | 242 | 11.8 |
| the matrix unit with trivially computed operands (the loop above, operands rewritten each step) | 239 | 11.9 |
| the matrix unit with operands left in registers | 210 to 219 | 13.0 to 13.6 |

The matrix unit is 71% of the launch. The rest is delivering operands: the activations' loads 41 us, the weights'
loads 18, decoding and writing the fragments 27.

**The unroll cliff, explained [measured].** With 4 or 8 blocks in flight the 16-row kernel took 878 us (3.3 TFLOPS)
where 2 take 304. The cause is the array of activations a lane loads ahead of use (`xs[Q][U][4][2]`: 64 floats a
thread at `U` = 4), not the weights and not the Metal compiler's scheduling: **loading each pair of activations where
it is used, with no array, takes 303 to 316 us at every unroll from 1 to 8**, and 32 rows then run at 9.6 to 9.8
TFLOPS (583 us) where they took 11 s a prefill before. `docs/serving.md` 8.7 named the compiler's register allocation
as the suspect; it is the registers, and the kernel's own array is what fills them.

**What was tried, on the product alone.** Us a row is the launch over its rows, the figure a prefill pays.

| variant | rows | us | TFLOPS | us a row |
|---|---|---|---|---|
| as emitted | 16 | 301 | 9.4 | 18.8 |
| loads at use, two weight tiles a simdgroup (`frag 2`, 128 threads) | 16 | 285 | 10.0 | 17.8 |
| the same | 24 | 395 | 10.8 | 16.5 |
| loads at use, one tile | 32 | 583 | 9.8 | 18.2 |
| the activations' fragments from device memory with `simdgroup_load` | 16 | 323 to 337 | 8.5 to 8.8 | |
| several activation fragments held and shared by the weight tiles | 16 | 334 to 349 | 8.2 to 8.5 | |
| the operands' roles swapped (a lane holds two columns of one token) | 16 | 303 | 9.4 | 18.9 |
| the same | 64 | 2,081 | 5.5 | 32.5 |
| **activations stored `[k, m]`, a lane's two tokens one `float2` load** | 16 | 265 | 10.8 | 16.6 |
| **the same** | 24 | 376 | 11.4 | 15.7 |
| **the same** | 32 | 517 | 11.0 | 16.2 |
| **the same** | 48 | 730 | 11.7 | 15.2 |
| **the same** | 64 | 963 | 11.9 | 15.0 |
| the same, activations f16 in memory, float fragments | 32 | 455 | 12.5 | 14.2 |
| the same | 48 | 667 | 12.8 | 13.9 |
| all half: f16 activations, half fragments and accumulators, added to float every 32 products | 32 | 484 | 11.8 | 15.1 |
| the same | 64 | 944 | 12.1 | 14.8 |
| Metal 4 tensor ops (`mpp::tensor_ops::matmul2d`, 16 x 32 x 16), half weights, float accumulator | 16 | 430 | 6.6 | 26.9 |
| the same | 64 | 1,252 to 1,610 | 7.1 to 9.1 | 19.6 to 25.2 |

* **The transposed store is the one that pays without changing a number.** A lane of an activation fragment holds two
  neighbouring tokens at one reduction index; row-major `[m, k]` puts them `k` apart, two loads, and `[k, m]` puts them
  side by side, one. The order of the sum is the emitted kernel's, so the result is the same bits. It also has no
  cliff up to 64 rows. **20% less time a row at 64 rows, 12% at 16.**
* **f16 activations in memory buy another 6 to 8% and are a decision**, not a step: each activation is rounded to 11
  bits, about 1e-4 of a product's value against the 1e-6 two orders of the sum differ by today. The weights and the
  accumulators stay float. All-half fragments are no faster than that and worse numerically.
* **The tensor ops are not a way past the ceiling here.** MLX uses them (`steel/gemm/nax.h`, read at dc5ea6e) only on
  GPU architecture generation 17 and later; this M4 Max is not, the API reports a lane layout of four 8x8 fragments
  (so it is the same matrix unit underneath), the same loop in registers reaches the same 13.6 to 14.9, and a real
  kernel through it is slower than ours because the accumulators are copied in and out of the operation's own tensors
  every call.
* MLX's own kernels for this product were not measured when this table was made. They are in 9.6, and they are
  faster than every row above. Why this compiler's staged path (`gemm_mma`, section 7) is far from MLX's staged
  kernel is still **[open]**.

### 9.5 What this leads to (design, not built)

1. **`frag_rows_wide` loads activations where they are used.** No array, any unroll, 32 rows no longer collapse. Same
   bits. Worth nothing at 16 rows on its own; it is what the rest stands on.
   **Built** (`src/matmul.cho`): the four goldens of more than eight rows moved and no other; a 651-token prefill
   gives the tokens of the build before in 4.30 s against 4.31 (alternating runs); the emitted 32-row kernel takes
   597 us alone (9.6 TFLOPS).
2. **A product whose activations are stored `[k, m]`.** The language has `matmul` and `matmul_nt` only, so this is a
   new spelling (`matmul_tt`: the left operand stored the other way round, as `mma_tn` and `mma_tt` already say for
   accumulations) with its rule in `docs/chx.md`, the interpreter, both targets' generic lowering, and the fragment
   lowering that makes the one `float2` load. What feeds a product (the norms, the output gate, the SwiGLU product
   itself) writes its rows transposed, or a small kernel transposes them (four launches a layer, 4.4 MB at most).
3. **A chunk of 32 or 64 rows** with those kernels: 15.0 to 16.2 us a row against 18.8, and half or a quarter of the
   launches a token of everything else. The delta layers' chunk kernels are generated straight-line code and grow
   with the rows. The cache's cut would become a multiple of the chunk, which costs a warm turn up to 63 tokens
   re-fed unless a prompt's remainder gets its own smaller sets: this is the part that needs designing, not typing.

**What it would give [estimate from the table and `docs/serving.md` 8.7's 81%].** Weight products 20% cheaper a row
and everything else unchanged: 153 to about 182 tok/s on a cold prompt; with f16 activations, about 192. Ollama's
281 is not reached by any variant in the table above. **Corrected (9.6): it is reached by MLX's kernels, which were
not in the table**, so steps 2 and 3 are not the route to it; 9.6 says what is.

### 9.6 How Ollama does this product, read and timed (2026-11-21)

**What was read.** Ollama v0.40.16 (the installed version) and MLX at 54a7051, the commit its `libmlx.dylib`
reports (`0.32.4-43-g54a7051`). Section 9 above read MLX at dc5ea6e, which is not what runs here.

**What was timed.** A C program against the installed `libmlxc.dylib` (`mlx_quantize` in `nvfp4`, then
`mlx_quantized_matmul`, several products an evaluation, best of six), `n=17408 k=5120`, random inputs, same machine.
The time is wall clock around the evaluation, so it includes MLX's own launch cost. **The program is a throwaway and
is not in the repository.**

| rows | bf16 activations: us | us a row | TFLOPS | f32 activations: us | us a row | TFLOPS |
|---|---|---|---|---|---|---|
| 8 | 130 | 16.3 | 11.0 | 138 | 17.2 | 10.4 |
| 16 | 219 | 13.7 | 13.0 | 251 | 15.7 | 11.4 |
| 32 | 417 | 13.0 | 13.7 | 482 | 15.1 | 11.9 |
| 64 | 817 | 12.8 | 14.0 | 957 | 15.0 | 11.9 |
| 256 | 3,113 | 12.2 | 14.7 | 3,720 | 14.5 | 12.3 |
| 651 | 8,428 | 13.0 | 13.8 | 9,757 | 15.0 | 11.9 |
| 2,048 | 24,698 | 12.1 | 14.8 | 29,767 | 14.5 | 12.3 |

f16 activations time as bf16. Ours is 301 us at 16 rows (18.8 a row), f32.

**What Ollama does [read].**

* **The whole prompt is one pass.** `prefillChunkSize` is 2,048 tokens; every weight product of a prefill has the
  prompt's length as its rows. The last prompt token is left for decode, and the prompt's rows never go through the
  output head.
* **Activations are bf16.** The model's embeddings and norms are stored BF16 (its manifest: 790 BF16 tensors), and the
  residual stream stays in that type. The weight product's output is bf16 too. The per-tensor scale of NVFP4 is not
  in the kernel: the Go wrapper multiplies the product's output by it afterwards (or fuses it into SwiGLU).
* **6 to 16 rows: `fp_qmv_mma`, a kernel in registers like ours**, with four differences. (1) A lane reads 32 weights
  of one output row as one 16-byte load and 16 activations of each of its two tokens as two 8-wide loads, and the 128
  reduction indices of a block are assigned to fragments in the order that makes those loads contiguous (index
  `16 c + f` for fragment `f`, column `c`); activations stay row-major. That is about one memory operation for every
  five products where ours does one or two a product. (2) Weights are decoded two at a time by shifts into a `half2`
  and scaled in half, which is exact for NVFP4 without the tensor scale; no table. (3) A threadgroup is 8 simdgroups
  that each take every eighth block of 128 and add their partial sums through threadgroup memory at the end.
  (4) Four accumulators a simdgroup (16 outputs x 16 tokens), 16 outputs a threadgroup.
* **More than 16 rows: `fp_qmm_t`, staged.** 128 threads decode a 64 x 32 tile of weights
  into threadgroup memory as half (`Fp4HalfLoader`, the same shifts, 8 or 16 values a thread) and copy a 64 x 32 tile
  of activations as stored; four simdgroups each hold 16 float accumulators (32 x 32 outputs) and load fragments from
  the tiles, two products a fragment loaded. The 64-wide tile is taken when the rows over 32, rounded up, are even
  (or beyond 2,048 rows); otherwise the tile is 32 x 32, which is the dip at 651 rows.
* The tensor-op kernels (`*_nax`) are compiled in and not selected on this GPU.

**What this says about our gap [from the two tables].**

* At the same f32 activations MLX's kernels take 15.7 us a row at 16 rows and 14.5 on a long prompt against our
  18.8. The transposed-store variant of the table above (15.0 at 64 rows) only equals that; `fp_qmv_mma`'s contiguous
  loads get there at 16 rows with activations left row-major, so **step 2 of 9.5 is not needed and should not be
  built as designed.**
* 16-bit activations are worth another 17% (14.5 to 12.1). The table above put that at 6 to 8% for our lowering.
* **[inferred]** The weight products of a pass at 12.1 to 13.0 us a row would be 3.3 to 3.5 ms a token (24.3 x 10^9
  weights at this one shape's rate), against Ollama's 3.6 to 3.7 for everything. So **its everything-else is a few
  tenths of a millisecond a token where ours is 1.3**; one pass over the prompt launches each kernel once, not once
  a chunk or a token. With MLX's kernel rate and our 1.3 ms we would be at about 210 tok/s, not 281.
* **Not done:** none of this is built. Not measured: `fp_qmv_mma`'s loads put into our lowering; what our 1.3 ms is
  made of launch by launch.
