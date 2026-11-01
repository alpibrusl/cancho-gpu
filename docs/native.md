# Native: cancho-gpu at the speed of its language, and without the Rust oracle

> **Status: a design; N1 is built and measured (§6, N1), the rest is not.**
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
2. **The interpreter was 130 times slower than the same loop written plainly,** (that figure is for the whole pass; see N1 for the split between
   the two causes) and the first reason is in the source **[read]**: cancho-gpu was written when cancho
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

### N1. The swap (cancho's F4): built

**What was done, in three changes, each held to the goldens:**

1. `f32.round` is cancho's conversion (`float_of32(f32_of(x))`) and the integer code is kept as `round_soft`, the oracle
   `tests/unit_test.cho` holds it to on 400,000 values (random bit patterns, exact ties and a binary64 step either side of
   each, subnormals, the edge of overflow); two deliberate breaks of `round` fail it, one of them a single binary64 ulp.
   `f32.cho` is `edition 6`; its `sqrt32` is renamed `sqrt_f32` because `sqrt32` is a builtin there.
2. `interp.dot` accumulates in the `f32` type: the operands are f32 values, so converting them is exact, and one
   hardware multiply and one hardware add are what two integer roundings computed.
3. **`read_global` and the `store` op walk a rank-1 or rank-2 window by rows with a running index** instead of calling `elem`
   for every element. This is not the f32 type and was not in the first version of this design.

**Measured** (`scripts/bench_train_size.py`, 4096 x 128, the CPU session, ms per evaluation of 1.05 million multiply-adds):

| | ms | ns per multiply-add |
|---|---|---|
| before | 36.9 | 35 |
| after (1) and (2): the `f32` type | 9.2, then 8.0 | 8.8, then 7.7 |
| after (3): rows, not `elem` | **1.62** | **1.5** |

The 256 x 16 and 1024 x 64 sizes went from 0.31 to 0.03 ms and from 4.5 to 0.27 ms. The goldens agree byte for byte after
every one of the three (218 of 218 and 8 of 8), `tests/kernels.py`, `tests/session.py`, `tests/run.sh`, the authority
reports and `scripts/gate.sh` pass, and `metal-io.py` (7), `metal-kernels.sh` (13) and the Metal session (35) are unchanged.

**Correction to §1 and to what I said when this was proposed.** I put the whole 63 ns on the software f32 rounding. It was not
all of it. **The type alone gave 4 times (36.9 to 9.2 ms); the rest, a further 5 times, was the interpreter computing a tensor index
from scratch for every element it loaded or stored** (a loop over dimensions, a helper loop for the product of the inner
dimensions, and an integer division and remainder each, through the pool): `sample` on the running session put 80% of its
samples in the one small function `elem` called from the load loop, and about 10% in the multiply-adds. The 63 ns of §1 is the
figure for the whole interpreter on a pass that loads, multiplies and stores; the multiply-add itself was never 63.

**Against the prediction (1 to 3 ns): met, 1.5 ns.** Plain compiled cancho is 0.46 ms for the same evaluation, so the
interpreter is now 3.5 times a plain loop, not 130. It is also level with the Metal session below 1024 x 64 and 3 times slower
than it at 4096 x 128 (0.50 ms).

**Not done, and not needed to meet the number:** `fm` is still `[float]` (binary64 holding binary32 values) and 44 files
are still `edition 5`. Moving the storage to `[f32]` would halve the memory a tensor takes and remove a conversion per load;
what it would buy has not been measured, and the remaining distance to the plain loop (3.5 times) is the place to look
first. `rank >= 3` windows still use `elem`; the kernels in `tests/cases.txt` that use them were not timed.

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
