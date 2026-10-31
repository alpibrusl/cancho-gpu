# `cancho-gpu run`: a kernel with real inputs

> **Status: built and measured (§6), CPU interpreter only; Metal and CUDA are
> open.** The design below was written first and the gates before the code. Step 2
> of [`training.md`](training.md) §6 had stopped here: `check --run` and `emit --run` execute a kernel only over the
> interpreter's made-up tensors (`design.md` §10), so no program can hand a
> kernel parameters and read back the answer. This document says what the
> smallest change is, and the gates it must pass, before any of it is written.
> Claims are **[measured]** (with the command), **[read]** (from the source or a
> document) or **[open]**.

## 1. What is missing, measured

* `build/cancho-gpu` with no arguments prints its usage: `emit`, `check` (both
  with `--run`), `gguf`, `safetensors`, `tokenize`, `chat`, `sample`. None takes
  a tensor **[measured]**.
* `interp.inputs` fills every non-writable parameter from an LCG seeded 12345
  (an `i8` parameter from `0x30 + i % 8`, a writable one with zeros), and
  `interp.summary` prints, per writable parameter, a mean, a peak and the first
  four values (`src/interp.cho`) **[read]**. So there is no way in, and only
  four values of an output of thousands out.
* `cancho-gpu-device` uploads "the same inputs the interpreter used"
  (`device.md` §3) **[read]**: the same limit on the GPU path.

## 2. Not a flag on `--run`

The `--run` summary is held **byte for byte** to the Rust's, on both targets,
for 14 cases (`design.md` §10.2) **[read]**, and the goldens record it. A flag
that changed what `--run` reads or prints would put that contract at risk for a
feature the Rust does not have. So this is a **new subcommand**, `run`, and
`check --run` and `emit --run` are untouched: no golden changes, and
`tests/golden.sh` is the proof.

## 3. The command (a choice)

```
cancho-gpu run <file.chx> [name=value ...] [--target T] [--in NAME FILE ...] [--out NAME ...]
```

* **`--target T`** is `nvidia-ada` or `apple-m-series`. It is required when the
  file has a schedule for both, and implied when it has one (`usage` otherwise).
  The target is not decoration: the interpreter sums in f32 in program order, and
  the two targets cut a reduction differently, so `gemm_fp4`'s last digit
  already differs between them (`design.md` §10) **[read]**.
* **`--in NAME FILE`** gives a read-only parameter its values. The file is
  decimal numbers separated by white space, row-major, **exactly** the
  parameter's element count and at most 16 MiB (a text handle's length), each parsed by the parser that reads `name=value`
  constants (`f32.parse`, a binary64) and rounded **once** to the parameter's dtype
  (f32, f16 ties to even, or i8 half away from zero, as the interpreter's own
  inputs are). **Every
  read-only parameter must be given**: there is no silent fall-back to the LCG,
  because a kernel run on made-up numbers by mistake looks exactly like a
  kernel run on real ones.
* **`--out NAME`** prints every element of a writable parameter on one line,
  separated by single spaces, each as `{:?}` of its f32 (`f32.debug`, the
  shortest text that reads back, held to the Rust's printer by `design.md` §4)
  and a newline; elements are written one at a time, so the output has no size
  limit but the machine's. Several `--out` print one line each, in the order given.
  Writable parameters start at zero, as today.
* **Text, not binary.** The one consumer, `opt`, speaks text, and the printer
  and the parser already exist and are checked. A binary path (`sample` already
  reads `.f32` files) would be faster; **[open]**, add it when a measurement
  asks for it.
* **No new capability.** Reading input files is `fs_read` and printing is
  `io_write`, both already in the report that `tests/authority.sh` pins. Reading
  standard input would add one, so it is deliberately not offered: a caller
  that holds its parameters on a pipe (as `opt` does) writes them to a file
  first.
* **Refusals reuse the rules of `docs/rules.md`**, with no new tag: `usage`
  (a missing, unknown, doubled or wrongly-kinded `--in`/`--out` name, an
  ambiguous target, a token that is not a number, a count that is not the
  parameter's size), `io` (a file that cannot be read), `interp` (the tensors do
  not fit the float memory). Nothing is printed on a refusal.

## 4. Gates (written before the code; each must be able to fail)

1. **Same execution as `--run`.** For a set of kernels, write the LCG's tensors
   (`tests/kernels.py` already reproduces it) to files, run `cancho-gpu run`,
   and require the mean, the peak and the first four values of `--out` to equal
   what `check --run` prints, on both targets. This is what says `run` executes
   the same program.
2. **Whole-vector correctness.** Today's checks compare four values of an
   output (`tests/kernels.py`; `docs/kernels.md` records that a displaced row is
   invisible when those four are zeros). With `--out`, compare **every** element
   of the backward kernels (`bwd_dx`, `bwd_dw`, `bwd_db`, `bwd_sigmoid`) with the
   reference, to 1e-5.
3. **Real inputs, not the LCG.** Inputs that are not the interpreter's (a case
   with a known closed form: `dx = dy w` on small integers) give exactly the
   known answer.
4. **Refusals.** One case for each refusal of §3, each with its rule tag, exit
   status 1 and nothing on standard output. `tests/reject.sh` runs `emit` on
   `.chx` fixtures and cannot reach these, so a new `tests/run.sh` does, and
   fails on a refusal it has no case for.
5. **Mutants.** Reading `--in` files in the wrong parameter order, off by one
   element, or rounding to the wrong dtype must fail gate 1 or 2.
6. **Nothing else moved.** `tests/golden.sh` (the Rust's output), `tests/authority.sh`
   (the report), `tests/files.sh` (no file over 2,000 lines; the command goes in
   a new `src/run.cho`, not into `interp.cho` at 699 lines).
7. **End to end through `opt`.** A least-squares fit trained by `opt minimize
   --method bfgs` whose objective is a program that writes its parameters to
   files, calls `cancho-gpu run` for the forward and `bwd_dw` for the gradient,
   and prints the loss and the gradient. Required: it reaches the normal-equations
   solution, and the cost per evaluation (two process starts and an interpreter
   run) is **recorded, not assumed**. cancho-opt's `examples/train/` holds the
   `opt` side; this is the first example where the objective is a kernel.

## 5. What this does not do

* It does not run on a GPU. `cancho-gpu-device run` with the same flags is the
  second step (it needs a machine with Metal or CUDA; this container has
  neither) **[open]**.
* It is a CPU interpreter in f32 emulation: slow by construction
  (`design.md` §10.1, about 2M operations for `gemm_fp4` at 128 cubed). The
  gate-7 measurement says whether that is workable for a small fit and nothing
  about a large one.
* One point per call. Evaluating many points in one process, to amortise the
  start, is **[open]**.
* It adds no ReLU, no `AᵀB` product and no atomics; those remain the surface
  gaps of `training.md` §4.

## 6. Measured

The command is `src/runcmd.cho` (about 300 lines), with `cli.constant` made
public and `interp.convert` added; `interp.run` and the `--run` path are
untouched.

* **Gate 1, same execution as `--run`:** for `bwd_dx` (two shapes), `bwd_dw` (two,
  one with a batch of 256), `bwd_db`, `bwd_sigmoid`, `attn_gate` and `softmax`,
  `run` over the interpreter's own tensors gives the first four values of
  `check --run` exactly and its mean and peak to 6e-5 (`tests/kernels.py`,
  8 runs) **[measured: `python3 tests/kernels.py build/cancho-gpu`]**.
* **Gate 2, the whole vector:** all elements of the seven that have a reference
  agree with it to 1e-5; the largest difference seen is 2e-6 (`bwd_dw` with a
  batch of 256).
* **Gate 3 and 4:** `tests/run.sh`: the known answer of `dx = dy w` on both
  targets, a signed exponent, two `--out`, and **20 refusals**, each with its
  rule, exit status 1 and nothing on standard output (it is in `scripts/gate.sh`).
* **Gate 5, mutants:** inputs stored in reverse order, the minus sign dropped, and
  the kernel not executed are each caught (`tests/kernels.py` and `tests/run.sh`
  fail). Three mutants, not an exhaustive set.
* **Gate 6, nothing else moved:** `scripts/gate.sh` passes with the pinned
  compiler: 218 golden cases still agree with the Rust, the authority report is
  unchanged, `cancho fmt --check` is clean.
* **Gate 7, end to end** (`scripts/check_train.py`, needs a built cancho-opt `opt`,
  so not in `scripts/gate.sh`): `examples/train/lsq_objective.py` computes the loss
  and gradient of a 16-row, 3-feature least-squares fit with
  `examples/train/lsq_residual.chx` (forward) and `tests/chx/bwd_dw.chx` (backward,
  `n = 1`), and `opt minimize --method bfgs` drives it. It converges in **12
  evaluations** to within **1.1e-5** of the normal-equations solution in `w` and
  1.2e-10 in the loss. The tolerance is f32 arithmetic and `opt`'s default `gtol`
  of 1e-5, not slack. **Cost: 28 to 36 ms an evaluation over three runs** (two
  `cancho-gpu` processes and a Python one), 0.33 to 0.43 s for the fit, on this
  machine and not as a benchmark. With the gradient's sign flipped the same fit does not converge.

What this does **not** show: anything about a network larger than 32 parameters
(`opt`'s limit), a batch beyond a few hundred rows (the interpreter emulates f32
in integers), or any GPU. The two kernels are held to the interpreter, and the
interpreter to the Rust, only for the kernels whose cases are in
`tests/cases.txt`; `lsq_residual.chx` is an example and has no golden.
