# Native: cancho-gpu at the speed of its language, and without the Rust oracle

> **Status: a design. Nothing in it is built except the measurements of §1.**
> Written before the code; each step of §6 ends at a number. Claims are marked
> **[measured]** (with the command), **[read]** (from a document or the source) or
> **[open]**. It answers two questions that arrived tangled: *why is cancho-gpu
> slower than the language it is written in*, and *should cancho-gpu stay defined
> by lex-gpu's Rust*. They have separate answers, and the order in which they are
> acted on matters (§5).

## 1. What was measured

The least-squares loss and gradient at 4096 x 128 (about 1.05 million
multiply-adds an evaluation), the same loops in cancho and in C, so the time per
multiply-add is the language's and not an interpreter's. Each program returns its
result as its exit status, so the work cannot be removed, and the two languages
agree on it. `CANCHO=<compiler> bench/native/run.sh`, cancho `a49ea779` (the
compiler in cancho, `CANCHO_COMPILER=auto`), `clang -O2`, Apple M4 Max
**[measured]**:

| loops | cancho | C / other |
|---|---|---|
| plain f64, sums in the order written | 0.44 ns | 0.54 ns |
| plain f32 (the `f32` type), sums in the order written | 0.48 | 0.54 |
| f32, **summation order free** (8 accumulators; an axpy over a contiguous row) | **0.19** | **0.046** |
| numpy, in Python (BLAS, SIMD, threads), for scale | | 0.02 |
| **cancho-gpu's CPU interpreter** (`src/interp.cho`) | **63** | |
| the Metal kernels at 4096 x 128: 308 microseconds of GPU time for 1.05 million multiply-adds (`resident.md` §10) | | 0.29 (derived) |

Three things follow.

1. **The language is not slow.** Compiled cancho equals C when sums are taken in
   the order written, in f64 and in f32.
2. **The interpreter is 130 times slower than the same loop written plainly,**
   and the reason is in the source **[read]**: cancho-gpu was written when cancho
   had no `f32`, so each f32 operation is a binary64 operation followed by a
   rounding done in integers (`src/f32.cho`, `round`: `bits_of`, `bit_length`,
   shifts, ties to even, about thirty integer operations), twice per multiply-add
   (`src/interp.cho`, `dot`). **cancho has had the type since the compiler learned
   it, and cancho's own `docs/f32.md` lists this swap as step F4, the one not
   built** ([cancho#672](https://github.com/alpibrusl/cancho/issues/672)).
   `design.md` §4, which says "f32 without an f32 type", is stale and is corrected
   in place by this change.
3. **There is a second gap, to C and numpy, once the order of a sum is free:**
   0.19 ns in cancho against 0.046 in C ([cancho#671](https://github.com/alpibrusl/cancho/issues/671);
   the cause is a hypothesis, bounds checks that stop the loop vectoriser, and has not been looked at).
   Taking sums in the order written is what cancho's language guarantees and what the
   Rust fixes (§3), so this gap cannot be closed while that is binding.

## 2. Two decisions, not one

| | what it buys | what it costs | needs the other? |
|---|---|---|---|
| **Swap the emulation for the `f32` type** (cancho's F4) | the interpreter at about the speed of the plain loop: from 63 ns to about 0.5 | `edition 5` to `7` in 46 files; tensor storage from `[float]` to `[f32]` in 5 source files and 4 programs (`fm` has 22 accesses in `interp.cho`) | **no**: it changes no result, cancho's `f32.md` F1 gate showed every f32 operation bit-identical to the emulation over 10^8 pairs |
| **Drop the Rust as the oracle** | the summation order is the implementation's to choose, so SIMD and blocking are allowed (0.19 and below); `.chx` is defined here, not by a Rust parser; no repository to keep a differential against | the independent check of the whole compiler that 224 goldens give; for CUDA, which has no hardware here, the only one | no |

Neither depends on the other, and **doing the swap first uses the oracle as its
net**, which is the best use it will have: the 224 goldens (`golden/`, 6.6 MB, from
`tests/cases.txt`'s 247 cases) hold every printed byte of every kernel, so a swap that changes
one is caught by `tests/golden.sh` and not by someone's opinion. After it, the
oracle has done its work.

## 3. What the Rust gives today, exactly

**[read]** `CLAUDE.md`: "The oracle is the Rust. Output must be byte-identical with lex-gpu's
emitter; `golden/` is regenerated only from the Rust." Concretely:

* `golden/` and `tests/cases.txt`: CUDA C, Metal, and the `--run` summaries of 224 outputs,
  byte for byte, of which the f32 printing (`{:?}`) and the f32 rounding are part;
* `scripts/differential.sh` and `scripts/regen-golden.sh`, against a lex-gpu checkout;
* the **language**: `.chx` is whatever lex-gpu's parser accepts, and its rule tags are the Rust's;
* the **order of every sum** (`dot` is a sequential sum over `k`, because the Rust's is);
* the repository's description and `README`: "byte-identical to lex-gpu's Rust emitter on every test case".

## 4. Independence (N2): what replaces it

1. **A specification of `.chx`, owned here** (`docs/chx.md`): the grammar, the typing
   rules, the meaning of every operation, and what is and is not specified: *the f32
   operations are IEEE binary32; the order of a reduction is not specified* (a
   conforming implementation may take it in any order; results are compared with a
   tolerance). The tests already written to the language, not the Rust
   (`tests/reject/`'s 17 fixtures, with the rule tags), become its conformance suite.
2. **Independent references for values**, as `tests/kernels.py` (a Python float64
   reference for the 9b kernels) and `tests/session.py` (the fit against float64) already
   are: they check the *numbers* a kernel computes, from the mathematics. Extending
   them to every kernel (`tests/cases.txt` has 247 cases; `tests/kernels.py` holds the 9b kernels to a reference, and how
   many of the 247 are covered by one has not been counted) is the largest piece of work in N2 and is §6's step N2b.
3. **The device is an oracle too, where there is one:** `scripts/metal-io.py` (7 comparisons) and
   `scripts/metal-kernels.sh` (13 comparisons) run kernels on a Mac and compare them with the CPU path.
4. **Our own goldens, as regression and not as truth.** `scripts/record-self-golden.sh`
   and `tests/self-cases.txt` (13 cases) exist: the emitted CUDA C and Metal text and the
   summaries are recorded from this compiler and a change to them is reviewed, not
   assumed wrong.
5. **What is lost, said plainly.** For Metal and the CPU, little: they run and are
   compared. **For CUDA, which has no hardware here (`device.md`), the Rust goldens are
   the only independent evidence that the emitted CUDA C is right**, and nothing replaces it
   until an NVIDIA machine does (`NVRTC` accepting it and the kernels agreeing with the
   CPU). Until then the CUDA text is *regression-tested, not verified*, and the README says so.
6. **The rules change in one place:** `CLAUDE.md` ("The oracle is the Rust" becomes "the oracle
   is `docs/chx.md` and the independent references"), the repository description, `README`,
   `design.md`'s byte-identity sections; `golden/`, `regen-golden.sh` and `differential.sh`
   are deleted **after** N1 is merged.

## 5. Order

```
N1  swap the emulation for f32            (oracle kept as the net)         -> 63 ns to ~0.5 ns
N2  retire the Rust oracle                (after N1 is merged)             -> chx.md, references, self-goldens
N3  free the summation order, SIMD        (needs N2 and cancho#671)        -> 0.5 ns to ~0.05-0.2 ns
N4  the matrix-by-vector kernels on a GPU (independent of N1-N3)          -> 308 us toward ~10-30 us
N5  tensors from binary files; the shim's autorelease defect                -> sizes past 16 MiB; no leak
```

N4 and N5 do not depend on the rest and can run beside it. Each step is a separate change.

## 6. The steps

### N1. The swap (cancho's F4)

* `edition 5;` to `edition 7;` in every file, which changes `Split` in the three
  programs' `main` (`signals`, `exec` are new fields, released at once) and nothing else
  that was found writing `bench/native/lsq_f32.cho`.
* `fm` from `[float]` (binary64 holding binary32 values) to `[f32]`: tensors, `interp`,
  `runcmd`, `session`, `devcodec`, the device programs. `src/f32.cho` shrinks to what the
  type does not do (`round_to` for `f16`/`i8` storage, the printing of `{:?}` through
  `std.fmt32`, whose output is byte for byte Rust's `{:?}`, checked against all 2^32 patterns).
* **Gate:** nothing changes. `tests/golden.sh` (224 files), `tests/run.sh`,
  `tests/kernels.py`, `tests/session.py`, `scripts/metal-io.py` and the full
  `scripts/gate.sh` pass byte for byte. A swap that moves a byte is wrong.
* **Number:** `bench/native/` and `scripts/bench_train_size.py` again: the CPU session at
  4096 x 128 from 36.9 ms toward 1 ms. **Prediction [open]: 1 to 3 ns per multiply-add**, since
  the interpreter's loop does a few index computations a multiply-add that the plain loop does not.
  If it is above 5, the document says so and the loop is restructured (see N3).

### N2. Independence

Section 4. The pieces, in order: (a) `docs/chx.md`, written from the parser and checker
as they are (the surface does not change); (b) references for every kernel of
`tests/cases.txt` that computes numbers, with a tolerance per kernel and the *mutants*
(`tests/session.py --mutants` is the model) showing each reference fails when the kernel is
wrong; (c) self-goldens for the text; (d) the rules and descriptions; (e) deletions. N2 does
not start until N1 is merged and measured.

### N3. A free summation order

With the order unspecified, `matmul` and `matmul_nt` in `interp.cho` become blocked: eight
accumulators for a dot, an axpy over contiguous rows for the other shape. The micro-benchmark is
0.19 ns in cancho and 0.046 in C; **the target is what cancho#671 allows**, and the interpreter's
own overhead will sit above it. **Gate:** `tests/kernels.py` and the references of N2b, with the
tolerance that a reassociated f32 sum needs (a few units in the last place per accumulator, written
down per kernel, not guessed). Without cancho#671 this step still gives about 2.5x.

### N4. Matrix-by-vector kernels on a GPU

**[measured]** the three kernels of one least-squares evaluation take 12 microseconds of GPU time
at 16 x 3, 55 at 1024 x 64 and 308 at 4096 x 128: about 7 gigaflops, on hardware that does thousands;
reading `x` twice at memory bandwidth is about 10 microseconds. So the kernels, not the interface,
are 30 times off the roofline at the one size where it shows. The shim can already split the time:
`CANCHOGPU_PROFILE=2` times each kernel, and `CANCHOGPU_NOOP=` / `CANCHOGPU_SKIP=` launch a grid that does
no work or leave a kernel out **[read]**. Plan: (a) profile `lsq_residual` (`matmul_nt` of `[r, k]` by `[1, k]`) and `bwd_dw`
with `n = 1`; (b) hypothesis **[open]**: the schedule gives one threadgroup of 128 threads per output tile and
an `n = 1` product has one tile column, so the grid is `r / tile` groups, each reducing `k` serially;
(c) a variant that splits the reduction across the threadgroup (a `threadgroup` reduction), chosen by shape
at lowering; (d) the gate is the references plus `metal-kernels.sh` on the Mac, and the number is
the table in `resident.md` §10. The lowering is shared with CUDA, so a change moves its text; after N2
that is a reviewed diff and before it a Rust-golden change, which is why N4 is better after N2 for the
kernels it touches.

### N5. Interface and the shim

* **Tensors from binary files.** A resident file is text and capped at 16 MiB (about 800,000
  numbers), so no problem past 4096 x 128 can be tried through a session: the sizes at which a GPU
  would win are out of reach. A `resident NAME FILE --f32` form (little-endian IEEE words, read with
  `f32_of_bits` over four bytes, no parse) has no cap beyond memory and removes the parsing from setup (0.68 s at 4096 x 128). Own
  document first.
* **The shim's autorelease defect** (`resident.md` §10): about 1.9 KB leaked a request. Explicit
  retain and release where `g.cmd`, `g.enc`, `g.last` are set and cleared, and an autorelease pool per entry
  point; it also touches the Qwen runners, so their checks (`scripts/qwen35-check.sh` and the others that need
  weights) are the gate, and where weights are not to hand that is said.

## 7. What this does not do

* It does not make cancho-gpu faster than numpy on a CPU for dense algebra; that needs SIMD and
  threads, which the language does not have yet (cancho#671 is the first question). The honest target
  for the CPU path is *level with plain C, then within a small factor of vectorised C*.
* It does not claim the CUDA output is right after N2 (§4.5).
* It does not change `.chx`: the surface stays; N2 only says where it is defined.

## 8. Open

* The cause of cancho#671, and so how much of the 0.19-to-0.046 gap a language change closes.
* Whether `[f32]` storage with `f16` and `i8` tensors rounds as the emulation did (`round_to`
  is kept for them; the gate is the goldens).
* Whether the interpreter, after N1 and N3, is close enough to C that a code generator to
  cancho source (a third dialect, built with the cancho compiler) is not worth its build step.
  The decision is by the N3 number: past about 3 times the plain loop it is worth designing.
