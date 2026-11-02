# Native: cancho-gpu at the speed of its language, and without the Rust oracle

> **Status: N1, N2, N2f and N4 are built and measured (§6) and no Rust tool is left; N3 and N5 are not.**
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
   `design.md`'s byte-identity sections; `regen-golden.sh` and `differential.sh` are deleted **after** N1 is merged. **`golden/`
   is kept, not deleted (changed from the first draft of this section):** its 226 cases are byte-identical with what this compiler
   writes (re-recording changed none of them), so deleting them would throw away a regression net for nothing. They become
   regression records.

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

### N2. Independence: built

**What was done:**

* **`docs/chx.md`**, the specification of the language, first edition: the grammar (read off the parser), the types, the
  operations and their broadcasting, the linearity rules, the targets, and what is **deliberately unspecified** (the order of a
  reduction; the accuracy of the transcendental functions; the `i8` tie rule; NaN and infinity). It lists its own open points.
* **A float64 reference for every one of the 41 numeric kernel families** of `tests/cases.txt`. Twenty-nine already had one in
  `tests/kernels.py`; twelve did not (`gemm`, `gemm_mma`, `gemm_fp4`, `gemm_fp4_res`, `rmsnorm`, `silu_mul`, `softmax`, `rowops`, `stage`,
  and the plain `matvec`, `matvec_fp4`, `batch_fp4`): they were written, they all agreed with the kernels at the first try on both
  targets, and `kernels.py --mutants` makes one deliberate change to nine of them (a sign, a missing residual, a dropped slice of k,
  a reciprocal omitted) and requires each reference to fail. `kernels.py` now has 146 checks. Two of my first mutants were
  stopped by the type checker instead of the reference and were replaced. One reference needed a looser tolerance (`matvec`
  at k = 4096: 1.7e-5 on terms adding up to 254, about one f32 rounding step at that scale), stated at the call.
* **The goldens are regression records.** `golden-self/`, `self-cases.txt` and `record-self-golden.sh` are folded into one
  `golden/` (226 cases: 200 emit, 26 refused), `scripts/record-golden.sh` records them from this compiler, and **re-recording
  changed none of the 218 that had come from the Rust** (checked by `git diff`: only `golden/PROVENANCE` differs, the renamed
  `LEX_GPU`). `tests/golden.sh` no longer says "the Rust refuses".
* **Deleted:** `scripts/differential.sh`, `scripts/regen-golden.sh`, `scripts/record-self-golden.sh`, `tests/self-cases.txt`.
* **The rules and the public text:** `CLAUDE.md` ("There is no Rust oracle"), the README (what you get, requirements, checking it,
  history, contributing), `docs/design.md` (a banner and §9 corrected in place), `docs/kernels.md`, `docs/llms.txt`,
  `docs/index.html`, the header of `src/parser.cho`.

**Not done, and said plainly:**

* **N2f is done below; no Rust tool is left.**
* **CUDA stays weaker than before** (§4.5): its text is a regression record and nothing else, until there is an NVIDIA machine.
* **The tests in `tests/cases.txt` that compute nothing numeric** (the 26 refusals, the lowering-only cases) have no float64
  reference, as before; their evidence is the goldens and the reject fixtures.

### N2f. The readers' oracles: built

The GGUF reader, the tokenizer, the sampler and the Unicode tables were held to dumps made by lex-rt's Rust tools. Four results:

* **GGUF: retired.** `tools/gguf_dump.py` is an independent reader written from the format description (it `mmap`s a file, so a
  5 GB model is no cost). It reproduces the three goldens byte for byte, **and is identical to `cancho-gpu gguf` on six real
  models** (0.27 to 5.6 GB; 112 to 427 tensors; the Ollama store on this machine). `tools/gguf_dump.rs` is deleted.
* **Sampler: retired.** The sampler is an algorithm (temperature, top-k, top-p, a `xorshift64*` stream seeded through `splitmix64`),
  so its oracle is a second implementation: `tools/sampler_dump.py`, in Python with exact 64-bit arithmetic and `numpy.float32`,
  ported from the algorithm. **75 cases, 2,269 rows: every pick and nucleus id identical, and all 191,630 probabilities
  bit-identical** to the goldens the Rust made (on this platform; `numpy.exp` and the platform's `expf` agree here, and may not
  elsewhere). `tools/sampler_dump.rs` is deleted. This is a port of a design, not a check against another design, and the
  doc says so.
* **Tokenizer: the oracle is Hugging Face `tokenizers`, it disagreed with us, and it was fixed (issue #51).** `tools/tokenizer_dump.py` uses
  the library, which reads each `tokenizer.json`'s own normalizer, pre-tokenizer and decoder. While `cancho-gpu tokenize` was held to
  the Rust it differed from the library on the real Qwen2.5-0.5B and Qwen3.8-27B tokenizers in 3 of 37 hand-written records and in 234
  and 211 of 500 random ones. **The Rust oracle had agreed with us because we were written to match it** (`bpe.cho` said so). The
  causes, found by the first differing character of each record, each fixed: NFC had no decomposition (`Ω` U+2126, `Å` U+212B; the
  largest class, 107 of 211); `\s*[\r\n]+` did not end at the last newline of a whitespace run (`"  \n  \n x"`: 2228 2228 830 against 29773 830,
  on the model this repo serves); `\p{L}`, `\p{M}`, `\p{N}` were approximations (`Ⅷ`, `⁴`, `½`, Thai vowel marks); the pattern was not read
  from the file; and added tokens were looked for after NFC, so `<|im_start|>` followed by U+0338 lost the token (found by the very
  last record of the 500). **Result: 0 differences** on 37 + 500 + 1,000 records for both tokenizers (`tests/tokenizer_hf.py`, in the gate;
  the 1,000 from a seed the fixing was not tuned against), and on 3,000 further random and 800 real prose and code records (2.17
  million characters); the toy fixtures, made complete enough for the library to read, agree on all 537 records of each of three
  variants and their goldens are now the library's output; speed is unchanged (0.50 s against 0.71 s on the prose). The splitter is an
  exact emulation of the two Qwen regexes (the Qwen2 and Qwen3 patterns differ in `\p{M}`), read from the file, and any other
  pre-tokenizer is refused (`tok-pre-tokenizer`) instead of being split wrongly. Not implemented: the `(?i:` group's Unicode case folding.
* **Unicode tables: replaced.** `src/unicode_data.cho` is generated from Python's `unicodedata` (`tools/unicode_tables.py`, Unicode 16.0): the
  general categories L, M, N, White_Space, the combining classes, and the canonical decompositions and compositions. Regenerating
  reproduces the committed file. `tools/unicode_tables.rs` is deleted.

**Gates added:** `tests/oracles.py` (the Python oracles against their goldens, ~7 s; both fail when broken on purpose),
`tests/tokenizer_hf.py` (skips when the library or the tokenizer files are missing). `scripts/regen-gguf-golden.sh` and
`regen-sampler-golden.sh` need only python (and numpy); regenerating changed no golden. `scripts/gguf-real.sh` and
`tokenizer-real.sh` need no Rust checkout.

### N3. A free summation order

With the order unspecified, `matmul` and `matmul_nt` in `interp.cho` become blocked: eight
accumulators for a dot, an axpy over contiguous rows for the other shape. The micro-benchmark is
0.19 ns in cancho and 0.046 in C; **the target is what cancho#671 allows**, and the interpreter's
own overhead will sit above it. **Gate:** `tests/kernels.py` and the references of N2b, with the
tolerance that a reassociated f32 sum needs (a few units in the last place per accumulator, written
down per kernel, not guessed). Without cancho#671 this step still gives about 2.5x.

### N4. Matrix-by-vector kernels on a GPU: built

**The cause was simpler than the plan guessed, and the fix is in the kernels, not the compiler.** The three kernels of one least-squares
evaluation took 312 microseconds of GPU time at 4096 x 128 (about 7 gigaflops; the memory allows about 10 microseconds). The plan below
(§ as first written) guessed that the grid had one tile column and each threadgroup reduced serially. **There was no grid at all**: both
kernels, like the repository's decode `matvec`, were written without a `grid` statement, so each launched
**one threadgroup of 128 threads, one core of the 40 on this GPU** (`emit` prints it: `launch: 1 threadgroups x 128 threads`), and in
the forward product neighbouring threads read rows 128 floats apart. The language does not invent parallelism: a `.chx` kernel is
as parallel as its `grid` says.

**Measured, in isolation** (the shim's per-kernel profile, `CANCHOGPU_PROFILE=2`; Apple M4 Max; medians of 35; a small harness that
runs one kernel on random resident data):

| kernel at 4096 x 128 | before (no grid) | after |
|---|---|---|
| forward, `matmul_nt` of `[4096,128]` by `[1,128]` (`lsq_residual`) | 154 us | **9 us** with `grid i over r / bm`, `bm 32` to `128`, 64 to 256 threads (7 to 8 us) |
| backward, `bwd_dw`, `[1,4096]` by `[128,4096]^T` | 148 us | **11 us** with `grid j over k / bk`, `bk 1`, 64 to 256 threads (10 us) |
| the loss, `bwd_dw` with `k = 1` | 10 us | 10 us |

The schedule matters in a way a grid-less kernel hid. Forward: with few threads and a big block a threadgroup is slow (bm 256 on 32
threads: 43 us); 128 to 256 threads at any `bm` from 32 to 128 sits at the memory roofline. Backward: one output column per threadgroup
(`bk 1`, 128 groups) is the fastest at every thread count from 32; `bk 8` needs 256 threads to match it, and `bk 32` is 3 times slower
because the reduction over 4096 rows is then split over too few groups.

**What was changed:** `tests/chx/bwd_dw.chx` (a grid over `k / bk`, `bk 1` on both targets) and `examples/train/lsq_residual.chx` (a
constant `bm`, the rows one threadgroup takes, chosen by `lsq_session.block` as the largest of 128 .. 1 dividing `r`). No compiler change.
Re-recording the goldens changed exactly the three `bwd_dw` cases (six files: the kernel name now ends `_1`), and the interpreter's
summaries are identical, so the CPU computes the same numbers. The gate passes; on the Mac `metal-io.py` (7 of 7, `bwd_dw` to 1.6e-7 of the
peak), `metal-kernels.sh` (13 of 13) and the Metal session tests (35) pass.

**Result, end to end** (the Metal session, ms per evaluation, data resident):

| R x K | before N4 | after | CPU session (before, after) |
|---|---|---|---|
| 16 x 3 | 0.14 | 0.15 | 0.01, 0.01 |
| 256 x 16 | 0.15 | 0.15 | 0.03, 0.03 |
| 1024 x 64 | 0.20 | 0.17 | 0.22, 0.33 |
| 4096 x 128 | **0.49** | **0.20** | 1.4, 1.7 to 2.0 |

At 4096 x 128 the GPU time is now 30 microseconds of the 200, so what remains is the per-request cost (uploading `w`, three launches, one
wait, the read-back, the text on the pipe): about 0.15 ms at any size, visible in the 16 x 3 row. **Against the bar set earlier (as fast as
plain cancho): met.** Plain compiled cancho on one core is 0.46 ms at 4096 x 128, so the Metal session is 2.3 times faster; torch on MPS
takes 0.66 ms for the same evaluation (3.3 times slower than this); numpy on the CPU takes 0.02 ms (10 times faster, on a problem of 2 MFLOP that
a GPU is the wrong tool for). The fit through `opt --persistent` converges identically on both sessions.

**The cost, honestly:** the grid kernels make the CPU interpreter slower, about 20 to 40% at 4096 x 128 (1.4 ms to 1.7 to 2.0): `bwd_dw`
with `bk 1` runs 128 instances, each of which loads the whole of `dyt` (524,000 extra element copies). The NVIDIA schedule has the same
`bk` and **has not been run on an NVIDIA GPU**; it is a guess carried over, not a measurement.

**Not done, and not known:** the decode kernels of the model runners (`matvec`, `matvec_rows` and the rest) are written without a grid
too (`emit` shows `grid 1x1` at n = 4096), so the same effect may be costing them; they were tuned for a different shape (a split-K
reduction inside one threadgroup) and **were not measured or changed here**. That is the first thing to look at for the 25 tokens a
second, and the harness for it exists (`gpu_time.py` below).

The original plan, kept because the guess it records was wrong: *"(b) hypothesis [open]: the schedule gives one threadgroup of 128
threads per output tile and an n = 1 product has one tile column, so the grid is r / tile groups, each reducing k serially; (c) a variant
that splits the reduction across the threadgroup, chosen by shape at lowering."* Neither was needed: `n = 1` was not the cause, and the lowering
is fine.

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
