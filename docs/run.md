# `cancho-gpu run`: a kernel with real inputs

> **Status: built and measured (§6) on the CPU interpreter. The device programs
> take the same flags (§7): the plumbing is built and measured against mock
> drivers on Linux; the Metal hardware gates (§7.2 items 3 and 4) were **run on an
> Apple M4 Max** (`scripts/metal-io.py`, `scripts/check_train.py --bin`; §7.5); CUDA
> has no hardware at all.** The design below was written first and the gates before the code. Step 2
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

## 7. The same flags on the device programs

> §7.1-7.3 were written before the code. Claims are **[measured]**, **[read]**
> or **[open]**; §7.4 says what was and was not run.

### 7.1 The design

`cancho-gpu-device` (CUDA) and `cancho-gpu-metal-device` (Metal) take

```
run <file.chx> [name=value ...] [--in NAME FILE ...] [--out NAME ...]
```

with the grammar, the file format (white-space separated numbers, exactly the
parameter's element count, at most 16 MiB, rounded once by `interp.convert`) and
the refusals (`usage`, `io`, no new rule) of §3. There is no `--target`: each
program has its one target, and `--target` is refused as `usage` like any
argument that is not `name=value`.

* **One reader, three callers.** The argument parser, the check of the names
  against the program, and the file reader leave `src/runcmd.cho` as `pub`
  functions that the compiler program, `device/main.cho` and
  `device/metal-main.cho` all call. The build scripts compile every `src/*.cho`
  but `main.cho`, so they see them **[read]**. The CPU `run` keeps its bytes: its
  24 cases (`tests/run.sh`) and `tests/kernels.py` are the proof.
* **Both tensor sets.** The device programs build the interpreter's tensors
  twice, `want` (the reference) and `have` (the upload). With `--in` the files
  are read once, into `want`, and the read-only tensors are copied to `have`, so
  the two cannot differ. Every read-only parameter must be given: no fall-back to
  the LCG.
* **`--out` alone is refused.** `--out` without `--in` falls under "every read-only
  parameter must be given", as on the CPU: a device answer on made-up inputs is
  what the program printed before.
* **`--out` prints the device's answer, after the comparison.** The comparison
  against the interpreter (2e-2 of the output's peak, unchanged) runs first; on a
  mismatch the refusal is `device-mismatch` and the `--out` lines are not
  printed. With `--out`, **stdout carries only the `--out` lines** (one per name,
  `f32.debug`, as `cancho-gpu run`), because the consumer is a program parsing
  it; the summaries, the `upload:` lines and the comparison go to standard error.
  Without `--out` stdout is what it was, byte for byte (`tests/device.sh`).
* **Authority.** `fileio.prefix` needs the `file_read` label next to the
  `fs_read("")` the device programs already hold. That widens the device programs'
  effect list by one label of a capability they already own; their `ffi(...)`
  declarations are unchanged, and `tests/authority.sh` (the compiler program's
  report) is unchanged **[measured: §7.4]**.
* **The device's answer is not bit-exact.** A GPU may sum in another order; the
  gate is "within 2e-2 of the peak", the same as the existing comparison, and the
  hardware gate below uses it too. That tolerance is wide enough to pass a wrong
  kernel whose error is small against the peak; the CPU `run` is the exact
  reference (§4 gates 1-2), the device run says only that the device agrees with
  it to that tolerance.

### 7.2 Gates (written before the code; each must be able to fail)

1. **Plumbing, no GPU.** The mocks run no kernel, so a run with `--in` must still
   refuse `device-mismatch` (that it can fail), **and** the bytes each mock
   received for each uploaded buffer must be the encoded real inputs. The mocks
   log, after the `launch` line, `sums <fnv1a-64 of each argument buffer>`, in
   argument order; `tests/device-io.sh` computes the f32 / f16 / i8 encoding of
   the file's numbers in Python and compares. CUDA runs against
   `tests/mock/libcuda_mock.c`; Metal is linked against
   `tests/mock/libmetalgpu_mock.c` in place of the Objective-C++ shim, which is
   the one thing the Linux run does not exercise.
2. **Refusals**, for both programs: a missing `--in`, a wrong count (too few, too
   many), a token that is not a number, an unknown name, a doubled name, `--in`
   of a writable parameter, `--out` of a read-only one, an unreadable file, a
   flag without its argument, `--target`. Each with its rule, exit status 1,
   nothing on stdout.
3. **Real hardware (a Mac, Apple Silicon).** `bwd_dx r=2 n=2 k=3` with
   `dy = 1 2 / 3 4`, `w = 1 0 2 / 0 1 3`, `--out dx`: values within 2e-2 of the
   peak of `1.0 2.0 8.0 3.0 4.0 18.0`; then the four backward kernels against
   `tests/kernels.py`'s references, every element. `scripts/metal-io.py`.
4. **End to end on the GPU.** `scripts/check_train.py` and
   `examples/train/lsq_objective.py` take the binary and its target as
   arguments; the 12-evaluation fit runs with `cancho-gpu-metal-device` and the
   cost per evaluation is recorded beside the CPU's 28 to 36 ms. Whether the GPU
   is faster is **[open]**, and for a 16-row fit each evaluation compiles a
   kernel and opens the device, so slower is the expected direction.
5. **Nothing else moved:** `scripts/gate.sh` with the pinned compiler.
6. **Mutants** for the input path (inputs stored in the wrong order, the sign
   dropped, the upload skipped) must each fail gate 1.

### 7.3 Known limits carried over, not fixed here

`opt` takes at most 32 parameters; the surface has no ReLU, no `AᵀB` and no
atomics (`training.md` §4); the order of a batch sum on a GPU is not pinned
(`training.md` §7).

### 7.4 Measured, and what was not run

Run in a Linux container (no GPU, no Mac), with the compiler at the pin
(`a4572ea`) for this repository and `0768094` for cancho-opt.

* **Gate 1, plumbing [measured: `tests/device-io.sh`, 90 checks, in
  `scripts/gate.sh`]:** for `bwd_dx`, `bwd_dw` and `gemm_fp4` (f16, i8 and f32
  inputs; 8192 numbers in the largest file), on both programs: the mock's zeros are
  refused as `device-mismatch` with nothing on stdout, **and** the FNV-1a 64 of every
  buffer the mock received equals that of the f32 / f16 / i8 encoding of the file's
  numbers, computed independently in Python. The reference summary shows the peak
  18.0 of the known answer, so the reference ran on the files. The one run that
  can pass (`dy` all zeros, so the answer and the mock's zeros agree) prints the
  device's `--out` line, and two `--out` print two lines.
  The Metal program is linked against `tests/mock/libmetalgpu_mock.c` **in place
  of `shim/canchogpu-metal.mm`**: the Objective-C++ shim and the Metal framework
  are not exercised here **[read]**. `tests/metal-device.sh` is untouched, is not
  in the gate, and cannot have used its mock on a Mac as written: the shim
  never loads `CANCHOGPU_METAL_FRAMEWORK` and `build-metal-device.sh` links the
  real shim **[read, not run]**.
* **Gate 2, refusals [measured]:** 16 per program (those of §3 that apply, a missing
  source file, and `--target`, which the device programs do not take), each with its rule and the specific
  message, exit status 1, nothing on stdout and nothing reaching the device.
* **Mutants [measured]:** six, each turned the gate red and was reverted: inputs
  stored in reverse order (6 checks fail), the minus sign dropped (6), the upload
  given the LCG instead of the files (3), the upload call skipped on CUDA (3) and
  on Metal (3), and `--out` printed before the comparison (6). The first two sit in
  the loader shared with the CPU command, so `tests/run.sh` catches them too.
* **Gate 5, nothing else moved [measured]:** `scripts/gate.sh` passes with the
  pinned compiler: 218 golden cases agree with the Rust, 8 self-recorded,
  `tests/authority.sh` unchanged (`args, err_write, file_read, fs_read, fs_write,
  heap, io_write`), `cancho fmt --check` clean, `tests/run.sh` 24 cases and
  `tests/kernels.py` unchanged. `tests/reject.sh` now lists `io` among the rules
  that need no fixture (it is raised through `err.fail` from the shared loader;
  `tests/run.sh` and `tests/device-io.sh` hold it). The device programs' effect
  list gained `file_read`; their `ffi(...)` declarations did not change.
* **Gate 4, the fit [measured on the CPU only]:** `scripts/check_train.py` now takes
  `--bin`, `--target` and `--runs`. With `build/cancho-gpu` it converges in 12
  evaluations, 1.1e-5 from the normal equations as before, at **40 to 46 ms an
  evaluation over three runs on this machine** (the 28 to 36 ms of §6 was another
  machine: only a comparison on one machine means anything). With a wrapper that
  stands in for a device program (`--target -`, no `--target` passed) the same fit
  runs, which proves the argument plumbing and nothing about a GPU.

**Run on Metal hardware (Apple M4 Max, Metal 4; the compiler at cancho `a49ea779`, built with `--ignore-compiler-rev` because the pin predates it and `std.linalg` is gone from the older checkout) [measured]:**

* **Gate 3 passes.** `scripts/build-metal-device.sh && python3 scripts/metal-io.py build/cancho-gpu-metal-device`:
  all seven comparisons agree, the largest difference 3.55e-7 of the peak (`bwd_dw r=256`),
  against the 2e-2 allowed; the known answer `1 2 8 3 4 18` comes back exactly.
* **Gate 4 passes, and the cost is known.** `OPT=../cancho-opt/build/opt python3 scripts/check_train.py
  --bin build/cancho-gpu-metal-device --runs 3`: BFGS converges in 12 evaluations, within
  1.1e-5 in `w` and 3.1e-10 in the loss (the CPU's numbers), and the flipped gradient does not reach the
  answer. **94 to 125 ms an evaluation**, against 40 to 46 ms on the CPU: slower, as expected for a
  16-row fit, since each evaluation starts two processes that each compile the kernel and open
  the device. One machine, three runs, not a benchmark.

**Not run, and so not claimed:**

* **CUDA: no hardware.** Its half is plumbing against a mock driver
  and nothing more.
* The Metal shim was not compiled here (macOS only), so `build-metal-device.sh`
  with the new sources is untested; the same sources link against the mock.
