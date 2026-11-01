# A resident objective: keeping the data on the device between evaluations

> **Status: the opt half (`--persistent`), the CPU session and the Metal session are
> built and measured (§6 steps 1 to 3, §9, §10, §11). §1-§8 were written
> before the code; §9 says where the build differs from them and what it measured.** It is
> step 4 of [`training.md`](training.md) §6 taken one level down: the question
> is no longer whether to train, but whether the *interface* between `opt` and
> a kernel is what stops a GPU from winning. Claims are marked **[measured]**
> (with the command), **[read]** (from the source or a document) or **[open]**.

## 1. The problem, measured

`training.md` §6 step 3 measured one loss-and-gradient evaluation of a
least-squares fit through `cancho-gpu run` (CPU interpreter) and
`cancho-gpu-metal-device` (Apple M4 Max), `python3 scripts/bench_train_size.py`:

| R x K | CPU interpreter | Metal device |
|---|---|---|
| 16 x 3 | 6 ms | 69 ms |
| 1024 x 64 | 137 ms | 206 ms |
| 4096 x 128 | 1017 ms | 1122 ms |

Metal never passes the CPU, and the kernel is not the reason (and both are far from numpy or torch, which
do this evaluation in 0.02 to 0.8 ms: `training.md` §6 step 3 has that table, which this section does not repeat): at 4096 x 128 the
forward kernel is about a megaflop, yet one process takes 0.33 s on the CPU and
0.375 s on Metal. Where the time goes **[measured, one machine, the gate running
beside it, so rough]**:

* **Starting a device process.** `cancho-gpu-metal-device run … r=16 k=3` takes
  38.6 ms; `cancho-gpu run` the same kernel, 2.0 ms; the device program given only
  its usage text, 2.8 ms. The difference, about 36 ms, is opening the device and
  compiling the kernel. An evaluation starts two such processes (forward, then
  backward), which is the 69 ms.
* **Text.** Every evaluation the objective writes `x`, `x^T` and `t` as decimal text
  and the program parses them: 0.15 s of Python writing at 4096 x 128, and a cap of
  16 MiB per input (`run.md` §3), above which both paths refuse. None of this
  changes between evaluations; only `w` does.

So the interface repeats, per evaluation, work whose result never changes: the
device open, the compile, the upload of the data. A GPU can only show what it does
if those happen once.

## 2. The idea

One process, started once, that holds the compiled kernels and the data on the
device, and answers one line per evaluation. Two halves, because the objective
protocol of `opt` starts a program per point (`cancho-opt` `docs/design.md` §4
**[read]**) and so cannot talk to a process that stays:

* **`opt` side:** a persistent objective, `opt minimize … --persistent -- PROGRAM ARG…`.
  The program is started once; for each point `opt` writes one line (the numbers of
  the point, as now) and reads one line back (the value, and for BFGS the gradient,
  as now). Closing its standard input ends the program.
* **Device side:** `cancho-gpu-session` (CPU interpreter; a program of its own, §9) and
  `cancho-gpu-metal-device session` (Metal), which read a *session file*, set
  everything up, then loop: one request line in, one answer line out.

Neither half is useful without the other, so §6 builds `opt`'s first.

## 3. The session file (a choice)

A text file, one directive a line, `#` comments:

```
program fwd lsq_residual.chx r=4096 k=128
program bwd bwd_dw.chx       r=4096 n=1 k=128
program sq  bwd_dw.chx       r=4096 n=1 k=1

resident fwd.x  x.txt
resident fwd.t  t.txt
resident bwd.xt xt.txt

link fwd.e -> bwd.dyt        # the same device buffer, no trip through the host
link fwd.e -> sq.dyt
link fwd.e -> sq.xt          # e^T e: bwd_dw with k=1 is the sum of squares

input  fwd.w                 # the request line, in this order
answer sq.dw  scale 0.5/4096    # the loss
answer bwd.dw scale 1/4096      # its gradient
```

* **`program NAME FILE.chx name=value…`:** the same arguments as `run`; the kernel
  is compiled once, at setup. A name is for the other directives only.
* **`resident P.PARAM FILE`:** read once, uploaded once, never again. The file
  format, element count and 16 MiB cap are `run.md` §3's.
* **`link P.OUT -> Q.PARAM`:** an output of one program is an input of another and
  stays on the device. Shapes must have the same element count (`e` is `[r,1]`, `dyt`
  is `[1,r]`: the same flat memory, as `lsq_objective.py` relies on today **[read]**).
  This is what removes the host round trip and the transposed copy of `e`.
* **`input P.PARAM`:** parameters filled from the request line, in directive order.
* **`answer P.OUT [scale a/b]`:** what is printed, in order, one number each. The
  scale is an exact rational applied on the host in f64 after the download, so the
  `1/R` and `1/2R` of the loss and gradient are not rounded into an f32 kernel.
* **Every parameter of every program must be resident, linked or an input**, or the
  session is refused before any device work (the `run` rule, "no fall-back to the
  LCG", carried over).

Programs run in directive order for each request. The loss costs one more kernel
launch, `bwd_dw` with `k=1` on `e` twice, so no kernel is added to the surface
(`bwd_dw` is `tests/chx/bwd_dw.chx` **[read]**).

**The request line** is the `input` parameters' numbers, white-space separated,
exactly the element counts, one newline; **the answer line** is the `answer`s'
numbers in the shortest form that reads back, one newline. This is the
`opt` objective protocol with one line instead of one process: for a gradient
method the line is `value g1 … gK`, which is what `opt`'s BFGS reads today **[read]**.
End of input ends the session, status 0.

## 4. The `opt` side (a choice)

`--persistent` changes only how the program is run: one `exec`, a pipe each way,
one request and one answer per point. The existing parsing of the answer, the
finiteness check and the timeout apply per request. New refusals, each with a rule
tag and a fixture in `cancho-opt`'s `scripts/check_cli.py`: the program exits before
answering (`opt.objective-failed`, with the point), answers with other than the
expected count (`opt.objective-output`), or does not answer within the timeout
(a per-evaluation limit already exists, one hour, and a program that exceeds it is `opt.objective-failed`; `--timeout-ms` makes it settable. **Built:** `cancho-opt` `docs/design.md` section 14, no new rule tag). Authority
is unchanged: `exec` and a clock. It is a separate change in a separate repository
and starts with its own section in `docs/design.md`.

## 5. Gates (written before the code; each must be able to fail)

1. **The session agrees with the pipeline it replaces.** On the CPU interpreter,
   `session` answers agree with a float64 reference written from the mathematics, the
   loss to 1e-6 relative and the gradient to 1e-5 of its peak. **Corrected after
   building it:** the first draft said "equal `lsq_objective.py` through `cancho-gpu run`
   bit for bit", which is false and was never going to be: the old pipeline sums the
   loss in f64 on the host and divides a gradient that was printed in shortest decimal,
   while the session sums in f32 on the interpreter and scales the exact f32; they
   differ by about 3e-8 relative. At 16x3, 256x16 and 1024x64 (4096x128 is not in the
   gate: its reference takes a minute in pure Python). *Mutant:* the `link` to `sq.xt`
   dropped (the loss then differs).
2. **Metal agrees with the CPU session** within 2e-2 of the peak, as `run.md`
   §7.1 (the device sums in another order), and **in practice** to the 1e-6 seen in
   `metal-io.py`; the gate prints the largest difference so it can be read.
3. **The fit.** `opt minimize --method bfgs --persistent` through the CPU session and
   the Metal session converges in the same 12 evaluations to the normal-equations
   solution within 1e-4, and the flipped-gradient program still fails
   (`check_train.py` has this check and keeps it).
4. **Resident means once.** A mock driver (the `device.md` §5 style, no hardware)
   counts compiles, uploads and launches: after *N* requests the compiles are the
   number of programs, the uploads of resident data are one each, and the launches
   are three per request. *Mutant:* the resident upload moved inside the loop fails
   the upload count.
5. **Refusals, each with a tag and a fixture** (`tests/reject/`-style, none reaches
   a trap): an unknown directive, a program name used twice, a parameter neither
   resident, linked nor input, a link between shapes of different element counts, a
   link from an input, a file that does not exist, a request with the wrong number of
   values, a value that is not a number, end of input in the middle of a line. A
   refusal, before or after setup, is `error[session]: ...` on standard error, status 1
   and nothing more on standard output (the convention of the other commands; the
   first draft said an answer line and status 2). The payload is not echoed.
6. **Measured, then the doc is corrected.** `scripts/bench_train_size.py` extended
   with `--session`: per-evaluation ms at the §1 sizes, CPU session and Metal
   session, one machine, medians. The same script now also times numpy and torch (CPU and MPS), and the session is read against
   them too. **Predictions, to be tested, not claims:** per evaluation under 10 ms on Metal at
   1024 x 64; the Metal session faster than the CPU session at 4096 x 128; **not** faster than
   torch-MPS's 0.8 ms floor or than numpy (this document does not expect it). If either is false, §1's explanation is wrong or incomplete
   and this document says so here.

## 6. Plan, each step ending at a number

1. **`opt --persistent`** in cancho-opt. **Done** (draft PR alpibrusl/cancho-opt#8, design section 14). Measured: the same
   answer byte for byte as a process per point for Nelder-Mead, BFGS and SPSA, and 159
   Rosenbrock evaluations in 0.02 to 0.03 s against 2.7 to 2.9 s (the 12-evaluation
   fit of `check_train.py` is not run through it yet: that needs step 2).
2. **`cancho-gpu-session`** on the interpreter. **Done** (§9). Measured: the 12-evaluation
   fit through `opt --persistent` takes 3.8 to 4.4 ms in all, against 0.39 s with a
   process per evaluation; one evaluation at 4096 x 128 is 66 ms against 1150.
3. **`cancho-gpu-metal-session`** **Done** (§10): gates 2, 3 (Metal) and 6; gate 4 (a mock driver counting uploads) is not built. Number: the
   table in §1 again, with a third column.
4. **Decide** whether `training.md` step 4 goes on (Adam in `opt`, mini-batches) or
   stops: on the numbers.

Each step is a separate change. Nothing here touches the byte-identity of `emit`,
`check` or `run` with lex-gpu's Rust (`CLAUDE.md`): the CPU session is a
program of its own, as the device programs are.

## 7. What this does not do

* It does not train on the GPU: the optimiser stays in `opt`, on the host, and
  receives `K + 1` numbers a step. A model with millions of parameters would still
  pay for that, and BFGS holds a K-by-K matrix (`training.md` §3).
* It does not use a mini-batch: resident data is fixed for the session. Changing
  it per request (SGD, Adam on batches) is a different protocol **[open]**.
* It does not fix determinism: the order of a device sum is the GPU's
  (`training.md` §2, **[open]**).
* CUDA: the session is for Metal and the CPU; the CUDA shim is unchanged and has no
  hardware (`device.md`).
* The inputs are still text, once: the 16 MiB cap on a resident file stands until a
  binary format is designed **[open]**.

## 8. Open

* Whether the host-side f64 `scale` should instead be a kernel constant once the
  surface can say a scalar times a tile.
* A transposing load or `matmul_tn` would remove `xt` from the resident files
  (`training.md` §4); it is a lex-gpu change.

## 9. The CPU session as built (measured)

> Where this differs from §1-§8, and what it measured. Everything below was run on an
> Apple M4 Max with cancho `a49ea779` built with `CANCHO_COMPILER=auto` (the compiler
> written in cancho alone, no fallback to the Rust one).

* **A program of its own, `cancho-gpu-session`** (`session/main.cho`,
  `scripts/build-session.sh`), not a subcommand of `cancho-gpu`. Reading standard input
  is an authority `cancho-gpu` does not have (`tests/authority.sh` pins its report), so
  the loop lives in a program whose report is pinned separately: `args, err_write,
  file_read, fs_read, heap, io_read, io_write`, no `fs_write`, no network, no foreign
  code (`tests/authority.sh`). The session file's reader and the setup are `src/session.cho`,
  which the Metal session will use too.
* **`cancho-gpu-session FILE [--target T]`.** The target is for every program of the file
  (default `nvidia-ada` where the file has it, else `apple-m-series`); the session file
  has none, so the same file serves the device programs, which have one target each.
* **The directives of §3 as written**, plus: `answer` prints every element of the
  output, and the scale is `(x * A) / B` in f64. A link copies on the host here (the
  CPU has no device), just before the program that reads it. Numbers are printed
  shortest-that-reads-back in f64.
* **Memory:** each request's allocations are given back (`mem.release_to`); 2,000 and 20,000
  requests peak at the same resident size (1,933,312 bytes), and the mutant that does not
  give it back is killed by that check.
* **Gates** (`tests/session.py`, in `scripts/gate.sh`): 36 checks pass: a hand-checked
  answer, the fit at three sizes against float64, the data read once (the files are
  deleted after the first request), 2,000 requests, the end of input, and 28 refusals.
  **Eight mutants** (`--mutants`) are killed: a link that copies nothing, the denominator
  of a scale ignored, a scale the wrong way round, a link copied after its reader, memory
  not given back, a last line without its newline dropped, too many numbers accepted, a
  link from a program to itself. Two survived the first set of cases and the cases were
  mended (memory: 2,000 requests did not show a leak, so the check now compares peak
  resident sizes; the self-link: the case was refused for another reason).
* **Measured** (`scripts/bench_train_size.py`; one evaluation, medians, in milliseconds):

  | R x K | process per evaluation | session | setup, once | pure Python | numpy |
  |---|---|---|---|---|---|
  | 16 x 3 | 5 | 0.0 | 0.00 s | 0.01 | <0.005 |
  | 256 x 16 | 25 | 0.4 | 0.01 s | 0.27 | 0.01 |
  | 1024 x 64 | 156 | 4.8 | 0.09 s | 3.8 | 0.01 |
  | 4096 x 128 | 1150 | 66 | 0.86 s | 36 | 0.02 |

  (16384 x 256 is refused: a resident file is over 16 MiB of text.) **The interpreter's
  own arithmetic is about twice a pure Python loop's, not thousands of times slower:**
  4096 x 128 is 1.05 million multiply-adds in 66 ms, about 63 ns each, where Python's
  list comprehension takes about 35. The rest of the 1150 ms was text and process starts
  (`training.md` §6 step 3). The session is **17 times** faster than a process per
  evaluation at 4096 x 128 and about 14 to 30 at the small sizes.
* **The fit** (`OPT=…/opt python3 scripts/check_train.py --session --bin build/cancho-gpu-session --runs 5`):
  BFGS with `--persistent` converges in the same 12 evaluations to the
  normal-equations solution (1.1e-5 in `w`, 1.8e-10 in the loss), the whole fit in **3.8 to
  4.4 ms**, process start included; the flipped-gradient program still fails.
* **Not done:** the Metal session (§6 step 3) and so gates 2, 4 and the Metal half of 6.
  The CPU session answers the question the design asked of the interface: with it the
  per-evaluation cost is the interpreter's arithmetic, and it is no longer the thing in the way.

## 10. The Metal session as built (measured)

> Apple M4 Max, Metal 4, cancho `a49ea779` built with `CANCHO_COMPILER=auto`.

* **`cancho-gpu-metal-session FILE`** (`device/metal-session-main.cho`, `scripts/build-metal-session.sh`),
  a program of its own like the CPU session, with the same file, the same reader and setup
  (`src/session.cho`, whose `open` now also lowers each program to Metal source when asked) and the
  byte codec of the device programs (`src/devcodec.cho`, moved out of `metal-main.cho`, which uses it
  too). Each kernel is compiled once; every parameter has a device buffer once; resident data is uploaded
  once; **a link is the same buffer for the output and for the input** (no copy; a link between different
  element types is refused, since the buffer cannot be both); a request uploads the inputs, encodes the
  kernels in order (`cgpu_launch_async`, each waiting for the one before it, as the shim documents) with
  **one** `cgpu_sync`, and reads the answers back. Launching with a wait for each kernel instead cost
  0.72 ms at 4096 x 128 and 0.40 ms at 16 x 3; one wait costs 0.49 and 0.14.
* **Gate** (`python3 tests/session.py build/cancho-gpu-metal-session --metal`): 35 checks pass, the
  numbers held to 1e-4 instead of 1e-6 (the device sums in another order; in practice the loss is within
  4.5e-7 of the CPU session's). The existing Metal checks are unchanged after the codec moved
  (`metal-io.py` 7 of 7, `metal-kernels.sh` 13 of 13). Mutants were not run on the Metal build.
* **Measured** (milliseconds per evaluation, medians of 20 and more; setup once in brackets):

  | R x K | CPU session | Metal session | compiled cancho, plain f64 loops |
  |---|---|---|---|
  | 16 x 3 | 0.01 (0.00 s) | 0.14 (0.04 s) | |
  | 256 x 16 | 0.31 (0.01) | 0.15 (0.05) | |
  | 1024 x 64 | 4.5 (0.09) | 0.20 (0.12) | |
  | 4096 x 128 | 36.9 (0.68) | 0.50 (0.68) | 0.46 |

  (The last column is a single loop in the language, `-O2` of the compiler in cancho; it equals C `-O2`
  at 0.54 and gives the same result: the language is not the slow part, the CPU interpreter is.) The fit
  through `opt --persistent` takes **41 to 44 ms** in all on the Metal session (about 35 of them opening the
  device and compiling), against 4 ms on the CPU session and 1.1 to 1.5 s through the per-process device path.
* **So is it as fast as plain cancho? Only at the largest size here, and it should be faster.** At 4096 x 128
  Metal takes 0.50 ms against 0.46 for one core of compiled cancho; below that, plain cancho wins
  (it scales with the work, 0.06 ms at 1024 x 64 by that rate, against Metal's 0.20 floor). The shim reports
  **the GPU time of the three kernels: 12 microseconds at 16 x 3, 55 at 1024 x 64 and 308 at 4096 x 128.**
  Those 308 are about 7 gigaflops on a chip that does thousands, and reading `x` twice at memory bandwidth
  should take about 10: the kernels (a `matmul_nt` of an `[r, k]` tile by a vector, and `bwd_dw` with `n = 1`)
  use the GPU badly for a matrix-by-vector shape. That, and not the interface, is what stands between this and the
  roofline, and it is open work. Larger problems cannot be tried through this interface: a resident file is text
  and is capped at 16 MiB (about 800,000 numbers).

### Known defects and what is not done

* **[Fixed afterwards: `native.md` N5.] The shim leaks about 1.9 KB a request** (peak resident size 29.5 MB after 2,000 requests, 63.4 MB after
  20,000). `shim/canchogpu-metal.mm` is built without ARC and keeps its command buffer and encoder in a C
  struct, so the autoreleased objects of each `commandBuffer` call are never released. The right fix is explicit
  retain and release where `g.cmd`, `g.enc` and `g.last` are set and cleared (about eight sites), plus an autorelease
  pool around each entry point; it also touches the Qwen runners, so it is its own change. The gate bounds it
  at 4 KB a request and says so.
* No mock-driver gate (§5.4) and no mutants for the Metal build.
* CUDA: none; there is no hardware (`device.md`).

## 11. After `native.md` N4: the kernels were single-threadgroup

§10's account of the Metal session ends with "the kernels ... use the GPU badly for a matrix-by-vector shape" and 308 microseconds of
GPU time at 4096 x 128. **The explanation was incomplete and the number is superseded.** The kernels ran on one threadgroup because they
had no `grid` (`native.md` N4). With a `grid` (in `bwd_dw.chx` and `lsq_residual.chx`) the three kernels take **30 microseconds** and the Metal
session **0.20 ms** an evaluation at 4096 x 128 (0.49 before), 2.3 times faster than plain compiled cancho (0.46 ms) and 3.3 times faster than torch on
MPS (0.66 ms). §10's table of the Metal session, and its sentence on the roofline, describe the grid-less kernels.
