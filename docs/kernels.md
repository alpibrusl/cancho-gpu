# The decode kernels of an attention layer (slice 9b)

> **Status: four kernels written in `.lx`, held to lex-gpu's emitter and
> interpreter, and checked against a reference for what they compute; the
> long-context split-KV attention is not built, and §4 says precisely why.**
> Issue [#6](https://github.com/alpibrusl/cancho-gpu/issues/6).

Slice 9b of cancho#251 asked for "the transformer kernel set (softmax, rope,
attention, KV cache)" in the `.lx` surface. `softmax.lx` was the first part
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
| `tests/lx/rope.lx` | `[x1 c - x2 s, x2 c + x1 s, rest]`: the "rotate half" form on the first 2h dimensions of each row, the rest passed through | `r=24` (or 4) rows, `h=32`, `p=192` |
| `tests/lx/kv_append.lx` | one position of every head's key and value cache, `cache[i * tmax + pos] = new[i]` | `nkv=4`, `d=256`, `tmax`, `pos` |
| `tests/lx/attention.lx` | (`d` must be 256: see below) for each key/value head `i`, `softmax(q k^T / 16 + mask) v` for its `g` query heads over the first `t` cached positions | `nkv=4`, `g=6`, `d=256`, `t <= 1364` (§3) |
| `tests/lx/attn_gate.lx` | `o * sigmoid(gate)` | `r=24`, `n=256` |

`rope`'s tables are one position's, `[1, h]`, shared by every row; the angles are
a table lookup the host writes (lex-gpu's runtime does the same), not arithmetic a
kernel needs. `attention`'s `mask` is added to the scores, 0 for a position in
use and a large negative number for one that is not, so `t` is a capacity. The
`1/16` is `1/sqrt(256)`, written as a literal: the surface has no scalar
parameters, so a different head dimension is a different file, and **`d` must
be 256**: any other value is accepted and scaled wrongly, which the file's
comment says and the cases all respect (an earlier case with `d=32` shared the
reference's 0.0625 and so hid it). The queries'
and keys' RMSNorm per head is `rmsnorm.lx`, already there; the projections are
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
  lex-gpu says the two compilers agree, not that `attention.lx` is attention. The
  script runs each kernel with `--run` on the interpreter's own made-up inputs
  (an LCG, `design.md` §10), runs the same inputs through plain Python written
  from the mathematics in the table above, and requires the printed mean and
  peak (four decimals) and the first four values (1e-5) to agree, for both
  targets' summaries: 14 checks on each, all agree. They are not vacuous, with a caveat that was found by review: the summary
is a mean, a peak and four values, so a displaced row is invisible when the
four values are zeros. Mutating `attention.lx` (the scale changed to 0.07, the
output written to the wrong place, the mask's sign flipped) fails two checks
each, and dropping the mask is refused by the compiler itself (`error[leak]:
`mask.5` is never consumed`), the linear discipline doing what it is for. For
`kv_append` the `pos=17` case alone passed a mutation that wrote one row late;
two cases small enough that the four printed values are the whole cache
(`nkv=2 tmax=2 d=1 pos=1`, `nkv=4 tmax=1 d=1 pos=0`) now catch a transposed
placement (`kc[i + pos * nkv]`), which the larger cases cannot.

## 3. The limit that was found

`attention.lx` keeps one head's scores, `[g, t]` floats, in one tile. A tile
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
lex-gpu's Rust-built kernels use. It cannot be written in `.lx` today.

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
