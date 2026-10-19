# The decode kernels of an attention layer (slice 9b)

> **Status: four kernels written in `.chx`, held to lex-gpu's emitter and
> interpreter, and checked against a reference for what they compute; the
> long-context split-KV attention is not built, and §4 says precisely why.**
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
  159 now) writes CUDA C and Metal **byte-identical with lex-gpu's Rust
  emitter's**, or is refused where the Rust refuses (`tests/golden.sh`,
  goldens recorded from the Rust, lex-gpu `851497d`). A refusal can be partial:
  an over-budget attention is accepted for CUDA and refused for Metal, and the
  golden keeps the CUDA file the Rust wrote before it refused
  (`scripts/regen-golden.sh`, `tests/golden.sh` learned this here).
* **The interpreter.** The `--run` summaries of the ten `--run` cases are
  byte-identical with the Rust interpreter's.
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

`frag` is a schedule key the Rust does not have, so no oracle can hold these cases: their expected output is
`golden-self/`, recorded from this compiler (`scripts/record-self-golden.sh`, `tests/self-cases.txt`) and
reviewed as text; what they compute is held by the interpreter, `tests/kernels.py` and the device check.
A schedule that sets `frag` where the shapes do not fit (more than eight rows, a reduction that is not a
multiple of 32, a group size other than 16, CUDA) lowers through the generic path, so the key is never wrong,
only unused.

## 7. i32, bf16, `rowargmax` and an offset read from data

The draft head's gather and argmax were hand-written Metal (`head_ops.metal`) because the surface could not
say them. They are `.chx` now (`gather_bf16.chx`, `argmax_f32.chx`), through four additions:

* **`i32`** is an element type for parameters, loads, stores and `rowargmax` results. Arithmetic on an i32
  tile is refused (`type`); it is an id, not a number.
* **`bf16`** is a storage type: a load of a bf16 parameter gives an f32 tile (the bits are the top half of an
  f32, so widening is a shift), and storing into one is refused. The interpreter rounds bf16 inputs to nearest, ties to even.
* **`rowargmax x`** is the index of the largest of each row of a 2-d tile, an `i32` tile of `[rows]`; ties
  go to the **last** of the equals, as the CPU's greedy pick. It lowers like `rowmax`: a strip a thread
  (`>=`), a shuffle, then scratch, merging by (value, then larger index). Indices travel through float scratch,
  exact to 2^24, so a longer row is a `lowering` refusal.
* **`tok[0]` in an offset**: `load table[tok[0], 0 ; 1, hid]` reads element 0 of the i32 parameter `tok` when
  the kernel runs, so the row is data on the device. Subscripts are constants; the parameter must be i32
  (`type`) and the element inside it (`bounds`). The checker cannot bound such an offset; the interpreter
  does (`error[interp]`), the compiled kernel does not, so the writer of the ids owns that.

Held by: the interpreter and `tests/kernels.py` (including ties and several rows), `golden-self/` for both
dialects (no oracle has these), `scripts/metal-kernels.sh` (GPU against the interpreter, exact) and
`scripts/spec-check.sh` (same tokens with the hand-written file deleted).
