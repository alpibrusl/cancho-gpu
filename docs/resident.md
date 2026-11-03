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
* The inputs are text, once, unless the directive ends in `f32` (section 12.4), which reads binary words and has no 16 MiB cap.

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
  code (`tests/authority.sh`). *(Widened in §19: `fs_write` and `file_write`, for `save`, with the owner's approval.)* The session file's reader and the setup are `src/session.cho`,
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

## 12. A training session: the optimiser on the device (design, written before the code)

§11 measured 0.20 ms an evaluation at 4096 x 128 through a Metal session, 3.3 times faster than torch on MPS. That is one gradient evaluation, and `opt` takes a step between two of them: the
parameters go up as text and the gradient comes back as text, the cost §10 found (about 0.7 microsecond a number, and the pipe holds a few hundred of them: `opt`'s `docs/design.md` section 14).
A model with thousands of parameters cannot be trained that way, and a small one pays the text for every step. **The remedy is to keep the optimiser's state on the device too and to run many
steps for one request.** Three directives, and one convention for the request line:

```
resident adam.w  w0.txt          # the initial parameters (and m, v: files of zeros)
carry    adam.wo -> adam.w       # after every repetition, the new w is the w of the next one
table    adam.t  sched.txt       # a file of S blocks of the parameter's size; repetition j uses block j mod S
steps                            # the first number of a request is how many times the programs run
```

* **`steps`:** a request line starts with a count `N` (a positive integer) followed by the `input` parameters' numbers. The programs run in order `N` times, the device runs them back to back with
  **one wait at the end** (the encoded launches of every repetition go to one command buffer), and the answers are those of the last repetition, one line as before. A session without `steps` is unchanged.
* **`carry P.OUT -> Q.PARAM`:** after each repetition the parameter `Q.PARAM` holds what `P.OUT` was given, for the next. `Q` is listed at or before `P`; the two have the same size and element type; `Q.PARAM` must be
  `resident` (its first value). On the CPU interpreter this is a copy at the end of the repetition; **on the device the two are one buffer**, so a carried program must be safe in place: element `i` of the output depends on
  the carried parameter only at `i` (an element-wise optimiser step is). That is the contract, and it is checked on the CPU only by a test that compares the two sessions. The answers read after the last repetition see the carried value.
* **`table P.PARAM FILE`:** a resident file that holds `S` consecutive blocks, each the size of the parameter. Repetition `j` (counted from the first request of the session, so a second request continues) reads block `j mod S`.
  This is how Adam's bias corrections (a block of two numbers a step, computed on the host in f64) reach the kernel, and how a minibatch is chosen (S batches of data) without a trip through the host.
* **Refusals:** a `carry` whose ends differ in size or type, whose destination is not resident, whose source is read-only or listed before the destination, or a table whose size is not a multiple of the parameter's, is `session`,
  with the line, before any device work. `steps` twice, or a count that is not a positive integer, is `session`.

**What the gate must be able to fail (`tests/session.py`).** (1) A kernel that adds one to a carried parameter: after `N` repetitions it holds `N` more, on the CPU and on the device, and a second request continues from it.
(2) A table of distinct blocks read in order, wrapping around. (3) The CPU session and the Metal session agree on every case. (4) Each refusal. (5) Mutants: the count off by one, the table index not advanced, the carry not applied, a carry applied twice.
**Then the training itself (`examples/train/mlp_*`):** a two-layer network and Adam written as `.chx` kernels trained for a few thousand steps, the loss and the parameters checked against a float64 implementation of the same
steps (and `torch.optim.Adam` on the same data), and the time of a step against torch on MPS and on the CPU. **Claims not made before they are measured:** that this is faster than torch for a model of any size
(the earlier measurements say torch is at a flat 0.8 ms a step on MPS; whether a step here is under that is the question).

### 12.1 As built (measured)

`session.cho` and the two session programs implement `steps`, `carry` and `table` as designed, with two refinements found while building: a `table` directive names its blocks (`table P.PARAM FILE BLOCKS`, so the file is checked against the
parameter's size and no pre-scan of the file is needed), and on the device **one output may feed several parameters** (the four Adam outputs feed both the next step and the forward kernels): the output and every parameter it is carried to are
one buffer, allocated by whichever is set up first. `tests/session.py` gained 20 checks (the accumulating kernel `tests/chx/step_add.chx` against a float32 simulation, the wrap of the table, continuation across requests, eight refusals), run on the CPU
and on Metal, and 5 CPU mutants; four device-only mutants (a table never advanced, one repetition whatever the count, a carry not shared, the table position not advanced) were built by hand and all are killed by `tests/session.py --metal`.

**The training** (`examples/train/mlp_session.py`, `tests/train.py`; kernels `mlp_fwd.chx`, `mlp_dzt.chx`, `adam.chx`, `bwd_dw2.chx` beside the existing `bwd_dw.chx`): a request is a count of Adam steps, the answer the loss after them.
(1) The CPU session trains an 8-4-1 sigmoid network: the loss after 1, 10 and 30 steps equals a float64 numpy implementation's to 6 digits (0.282091, 0.218097, 0.142954 against 0.142955) and the parameters to 2e-6; 15 steps and 15 give exactly what 30 give.
(2) The Metal session gives the CPU session's losses and, on a 32-32-1 network and 4096 samples, **the float64 reference's to 6 digits after 1, 10 and 100 steps, and `torch.optim.Adam` in float32 on the same data and start, 0.069565 and 0.069565 after 100**.
It does not claim more than 100 steps: after 1000 the device is at a loss of 0.00076 and the reference at 0.00108, as two float precisions part company on a trained network (both fit it).
(3) **Time of an Adam step** (`python3 examples/train/mlp_bench.py`, Apple M4 Max, torch 2.12, float32, interleaved rounds, medians of 3 of 400 steps; the cancho column is one request of 400 steps with one wait):

| samples x features x hidden | parameters | cancho-gpu (Metal session) | torch (CPU) | torch (MPS) |
|---|---|---|---|---|
| 4096 x 32 x 32 | 1,089 | 0.301 ms | 0.372 ms | 0.870 ms |
| 16384 x 64 x 128 | 8,449 | 1.013 ms | 1.333 ms | 1.011 ms |

**What this says, plainly:** a training step on the device is as fast as torch's best on this machine (CPU at the small size, MPS and CPU at the larger), not faster by a wide margin, and equal to it in the numbers. It is 3 to 5 times faster than the
path through `opt` could be (an evaluation of 0.2 ms plus the text of the parameters and gradient each way, and `opt`'s pipe holds a few hundred numbers). Where the step goes, by the shim's profile (`CANCHOGPU_PROFILE=2`, isolated, 16384 x 64 x 128): the two
reductions over all 16384 samples with n = 1 (`ls`, `gb2`: 76 us each, one threadgroup) and the forward kernel (300 us) are the largest now that the bias and weight gradients of the hidden layer take `bwd_dw2.chx`
(a grid over both axes of the result: those two took 807 and 754 us with `bwd_dw.chx`, whose grid is over the columns only, so a bias gradient with k = 1 was one threadgroup reducing over everything: the N4 mistake of `native.md` again, found by profiling the first
version, which was 5.1 ms a step; with the grid, 1.0). The 0.127 ms a step that one early run of the small size showed was not reproduced by the interleaved runs, and is not claimed.
**Not claimed:** speed for a model of any other size; that fusing the launches (11 a step) would not help (it would: the small size is launch-bound); anything about NVIDIA (the kernels compile for it and the CPU session runs them, but no GPU was available).

### 12.2 Minibatches through `table` (measured)

`mlp_session.write(..., batch=B)` makes the data, the targets and the transposed data tables of `r / B` minibatches (`table fwd.x`, `table fwd.t`, `table gw1.xt`), so step `j` trains on batch `j mod (r / B)`: epochs without a trip through the host, and no new directive.
`tests/train.py` holds it to the float64 reference (minibatches of 128 of 1024, 40 steps, so five laps of a table of 8): the CPU session to 1e-4 at steps 1, 2, 3, 10 and 40, and the Metal session to the CPU session's losses.
**What it buys** (16384 x 64 x 128, lr 0.003, Metal, 300 epochs each; the loss is on all the data, from the returned parameters): full batch, 300 steps, 1.01 ms a step, 0.28 s, loss **0.0217**; minibatches of 2048, 2400 steps, 0.39 ms a step, 0.93 s, loss **0.00054**.
Same passes over the data, 40 times lower loss for 3.3 times the time: the step is cheaper but there are eight times as many. (One learning rate for both and one run each: it shows that minibatches work through the table, not which batch size is best.)

### 12.3 A ReLU network

`relu` and its derivative `step` (1 where x > 0) are now surface operations (`chx.md`; `tests/chx/activations.chx` with a float64 reference and a mutant). `mlp_session.write(..., act="relu")` (or `--relu`) uses `mlp_fwd_relu.chx` and `mlp_dzt_relu.chx`:
the hidden layer is `relu(w1 x^T + b1)` and its backward is `step(ht)` in place of `ht (1 - ht)` (`ht > 0` exactly where the pre-activation is). `tests/train.py`: on 1024 samples of 16 features and 16 hidden units the CPU session's loss after 1, 10 and 40 steps equals the float64
reference's (0.299027, 0.104148, 0.013525 against 0.299027, 0.104149, 0.013525), the loss falls from 0.2990 to 0.0135, and the Metal session equals the CPU session to 1e-4. Not measured: the step time of the ReLU network (the kernels differ from the sigmoid ones in two element-wise operations).
`matmul_tn` is still not a surface operation: the transposed orientation of `mlp_fwd.chx` (a column of `ht` is a sample) makes it unnecessary here, at the price of a transposed copy of the data (`xt`).

### 12.4 Binary tensor files (`f32`)

`resident P.PARAM FILE f32` and `table P.PARAM FILE BLOCKS f32` read the file as little-endian IEEE binary32 words (`f16` likewise for an f16 parameter, section 14), exactly the parameter's (or the table's) element count and nothing else: no number is parsed, so the 16 MiB cap of a text file does
not apply (the file is staged in the session's byte pool, 256 MiB). The parameter must be f32. A file of the wrong size is `usage`, one that is not there `io`, a word after the file other than `f32` `session`.
`tests/session.py` (CPU and Metal) runs the accumulating session from binary files and requires the text session's answer, with non-dyadic values so that every byte of a word matters; two CPU mutants (a byte shifted wrongly, a longer file accepted).
`mlp_session.py --binary` writes the training data this way. **Measured** (Apple M4 Max): the 16384 x 64 x 128 session answers its first request in **0.08 s from binary files against 0.22 to 0.40 s from text**; a 32768 x 64 x 128 network (the data is 8 MiB binary, 22 MiB as text, which is refused)
trains at **1.27 ms a step**, loss 0.0346 after 320 steps. **A limit that was removed in the same change:** a device session allocated interpreter memory for every tensor of every program (32 million floats in all), so 65536 x 64 x 128 did not fit. It now gives a tensor interpreter memory only when the host reads or writes it (a resident file, an input, an answer; `interp.globals_lazy` and `materialise`): **65536 x 64 x 128 (4.2 million samples-features, a 16 MiB binary data file) starts in 0.38 s and trains at 2.48 ms a step**, loss 0.0476 after 320 steps. The CPU session still allocates everything, since it computes in that memory.

## 13. A deeper network against torch and OpenNN (measured)

`examples/train/compare_train.py` trains the same network on the same data with four runtimes and prints the time and what was learned. **The task:** binary classification of 64 standard-normal features by a label from a random two-hidden-layer tanh teacher plus noise
(`examples/train/compare_train.py` makes the data; a perfect classifier of the noiseless label gets 0.90 and the best a student can do on the noisy one is about 0.87). **The network:** 64-W-W-W-1, ReLU, a sigmoid output, cross-entropy, Adam at 1e-3,
minibatches of 256 (512 for W = 512), 12 epochs. Four runtimes on an Apple M4 Max, float32, all in this one invocation of the script, one run each:

* **cancho-gpu:** a Metal session (`deep_session.py`; the kernels `dense_fwd`, `out_fwd`, `hid_bwd`, `bwd_dw`, `adam`), the data a table of binary files, a request an epoch of steps.
* **torch 2.12** on the CPU and on MPS, a Python loop over the same minibatches, `torch.optim.Adam`, the data already on the device.
* **OpenNN** (checkout `ad4dd07`), CPU, built from its source with Apple clang and libomp by `bench/opennn/build_dense.sh`: **its own `dense` benchmark driver** (`benchmarks/families/dense.cpp`, Eigen without BLAS), `Adam` and `CrossEntropy` through its `Training` class. It draws its own Glorot weights and shuffles itself.

cancho-gpu and torch start from the same Glorot weights and take the same minibatches in the same order (the data is shuffled once; a minibatch table is read round and round), so their loss curves are comparable; all the test accuracies except OpenNN's are computed here from the parameters by the same numpy code.
Seconds an epoch (the median of the 12), and the test accuracy after the last epoch:

| network | cancho-gpu (Metal) | torch CPU | torch MPS | OpenNN CPU |
|---|---|---|---|---|
| 64-128-128-128-1, 100,000 samples | **0.144 s**, 0.8451 | 0.219 s, 0.8438 | 0.333 s, 0.8435 | 0.401 s, 0.8460 |
| 64-256-256-256-1, 100,000 samples | **0.158 s**, 0.8322 | 0.566 s, 0.8386 | 0.365 s, 0.8281 | 0.644 s, 0.8380 |
| 64-512-512-512-1, 200,000 samples, batch 512 | 0.942 s, 0.8446 | 0.859 s, 0.8534 | **0.369 s**, 0.8486 | 1.610 s, 0.8500 |

**What this says.** (1) **They learn the same thing.** The best test accuracy over the epochs is 0.857 to 0.858 at width 128, 0.857 to 0.859 at 256 and 0.867 to 0.868 at 512, reached at epoch 2 to 4 by cancho-gpu and torch alike, and every runtime then overfits down to the 0.83 to 0.85 above
(the last-epoch figures differ by the noise of that decline, not by anything the runtimes do differently). The test loss of cancho-gpu equals torch-MPS's at every width (0.3796 and 0.3799; 0.4832 and 0.4827 at 512) and torch-CPU's differs (0.5434 at 512): two float32 trajectories from one start part company, as in section 12.
(2) **Speed depends on the width.** At 128 and 256 units a step is small, and the GPU session wins: 1.5 times faster than torch's best at 128 and 2.3 times at 256, because it runs the whole epoch without returning to Python, with no launch overhead the loop has. At 512 units torch on MPS is **2.6 times faster than cancho-gpu**, whose
matrix products are the generic untiled kernels of the surface (no threadgroup-memory tiling, no use of the matrix unit) and which does twice the forward arithmetic (each layer in both orientations, since the surface has no transposing load) and transposes by a product with the identity.
At 512 cancho-gpu is level with torch's CPU path and 1.7 times faster than OpenNN's. (3) **OpenNN is the slowest at every width**, on a CPU through Eigen alone; its own benchmarks use MKL on x86, which this machine does not have, so this is not its best.
**Not claimed:** anything about another machine, another model, NVIDIA (every kernel here compiles for it and none has run), or a speed that holds when torch is tuned (compiled, with CUDA graphs); one run each, and the margins of 10 to 15% (width 512, cancho-gpu against torch CPU) are inside what the machine's clocks move between runs (`benchmark.md`).
Tuning that mattered: the threadgroup shape of the three new kernels (a minibatch spread over `bm` = 2 samples a threadgroup with 256 threads, from 16 samples and 128 threads) took the 512 network from 6.3 ms to 2.4 ms a step.
`tests/train.py` holds the kernels to a float64 numpy implementation of the same training (a 16-32-32-1 network, 40 steps: 0.720912, 0.635014, 0.485022, 0.15289 against 0.720912, 0.635014, 0.485024, 0.152891) on the CPU session, and the Metal session to the CPU session.

## 14. Mixed precision on the matrix unit (measured)

Section 13 left cancho-gpu 2.6 times behind torch on MPS at 512 units: its matrix products are the generic untiled `matmul_nt`, 0.3 TFLOPS, and each layer was computed in both orientations. The matrix-unit path (`mma`, `docs/chx.md`) reads f16 tiles from threadgroup memory
and accumulates in f32: `gemm_mma.chx` takes **121 microseconds for a 512 x 512 x 512 product, 2.2 TFLOPS, against 838 microseconds for the generic kernel**. Using it for training needed the other two layouts of a layer's products, so the language gained them:

* **`mma_nn`, `mma_tn`, `mma_tt`** (`docs/chx.md`): `mma` with the left tile stored `[k, m]` and/or the right tile stored `[k, n]` (Metal: `simdgroup_load` with or without its transpose flag; CUDA: the `wmma` fragment's `row_major` or `col_major`; the CUDA text is held by goldens and has not run). A forward product is `mma`, a layer's input gradient `mma_nn`,
  a weight gradient `mma_tn`: no transposed copy of anything. `gemm_mma_nn.chx` and `gemm_mma_tn.chx` are held to float64 references (`tests/kernels.py`) with mutants, **and run on the Metal GPU against numpy: the three layouts at 512 x 512 x 512 agree to 3e-5 of a peak of 40, in 121 to 124 us each.**
* **`f16` after a `resident` or `table` file** reads binary16 words (the same cap-free binary form as `f32`, section 12.4).

`examples/train/mp_session.py` is `deep_session.py` rewritten for them: master weights and Adam's moments in f32; an f16 copy of each weight matrix written by the Adam kernel (`adam_h.chx`) beside the f32 one; activations and the gradients of the pre-activations in f16; every product on the matrix unit with f32 accumulation;
a bias-and-ReLU epilogue (`bias_relu_h`), a ReLU-gradient mask (`step_mask_h`), a gridded bias gradient (`rowvec_mm`; `bwd_db.chx` has no grid). The surface has no conversion between dtypes but `stage f16`/`stage f32`, which is how the epilogues narrow and widen. Widths must be multiples of 64, the input width of 32.
`tests/train.py` (new checks) holds it to the float64 reference (0.770328, 0.685322, 0.596921 against 0.770288, 0.685168, 0.596911 at steps 1, 2 and 10: the f16 rounding), the Metal session to the CPU interpreter's emulation of the same arithmetic, and `tests/session.py` the `f16` files (with a mutant).

Seconds an epoch, the `compare_train.py` network of section 13, one invocation per width (torch's `amp` is `torch.autocast(float16)` on MPS), M4 Max:

| width | cancho-gpu f32 | **cancho-gpu mixed** | torch CPU | torch MPS | torch MPS autocast | OpenNN CPU |
|---|---|---|---|---|---|---|
| 128 | 0.144 | **0.084** | 0.231 | 0.335 | 0.423 | 0.378 |
| 256 | 0.157 | **0.105** | 0.569 | 0.370 | 0.438 | 0.639 |
| 512 | 0.941 | **0.198** | 0.851 | 0.377 | 0.457 | 1.550 |
| 1024 | - | 0.437 | 1.920 | 0.434 | 0.544 | - |

**Reading.** Mixed precision takes the 512 network from 0.94 to **0.20 s an epoch, 4.8 times, and 1.9 times faster than torch on MPS in float32 and 2.3 times faster than its autocast**; it is faster than torch's best at 128, 256 and 512 and level with it at 1024 (0.437 against 0.434; the 1024 network is 2.2 million parameters,
and there the untuned kernels and the elementwise epilogues, which write the pre-activations to memory, show). The learning is the same: the best test accuracy over the epochs is 0.8572 / 0.8561 / 0.8672 / 0.8678 for cancho-gpu mixed at 128 / 256 / 512 / 1024 against torch's 0.8565 / 0.8569 / 0.8666 / 0.8678, and the test loss agrees (0.3763 against 0.3799 at 128).
(The "f32" column is section 13's; the f16 rounding of the weights and activations changes the trajectory in the last digits.) **Not claimed:** anything about a model whose activations overflow f16 (no loss scaling was needed here and none is implemented), another machine, or NVIDIA (the CUDA text of the new layouts has only been generated, not run).

## 15. A GPT trained on the device (roadmap M1; measured)

`docs/roadmap-models.md` M1 asked for a character-level GPT trained in a session, held to a float64 specification, and compared with torch. It is built and it meets its exits.

**The model** (`examples/gpt/`): pre-LayerNorm decoder blocks, causal multi-head attention, a GELU (tanh form) feed-forward, an untied output layer, AdamW; the vocabulary is padded to 128 with the padding logits masked, and tokens enter as the one-hot rows of a `[vocabulary + positions]` embedding matrix, so an embedding needs no gather and its gradient no scatter.
`gpt_ref.py` is the specification: binary64 numpy with a hand-written backward, **checked against finite differences (28 tensors, worst relative difference 1e-6)**. `gpt_session.py` writes the session: about 110 programs for 2 layers (the forward and backward passes, each product on the matrix unit with f16 operands and f32 accumulation, the residual stream and the master weights and Adam's moments in f32).
Every kernel is `.chx`: `onehot`, `ln_fwd`, `ln_bwd`, `bias_h`, `res_bias`, `gelu_bias`, `gelu_bwd`, `att_scores`, `att_softmax`, `att_pv`, `att_dp`, `att_dsoft`, `att_dqkv`, `ce`, `adamw_h`, plus the existing `gemm_mma*`, `cast_h`, `rowvec_mm`, `adam`.

**What the language gained for it:** `tanh`; **a column `[rows, 1]` broadcasts along the columns** (a loaded column against a matrix: before, only a row vector or a reduction did); **a third grid axis** (a head of a sequence needs `b`, `h` and a row tile; the offsets must stay affine, so a division of a grid coordinate is not allowed, and the third axis is folded into the launch's second dimension and split in the kernel).
And of the sessions: `resident` and `table` files with the `f32`/`f16` flag now go **straight to the device** without passing through the interpreter's binary64 memory (a 10 M-parameter model's weights and moments did not fit it), the Metal shim holds 1,024 modules instead of 64.

**Correctness** (`tests/train.py`, on the CPU interpreter's emulation of the arithmetic and on the GPU): one step of a 2-layer, 2-head GPT, loss 4.16103 against the specification's 4.16103, **the gradients of all 28 tensors within 8e-4 of the float64 backward (7.8e-4 on the GPU)**; 25 AdamW steps on a learnable stream against a float64 AdamW: 4.2242, 3.1027, 1.8494, 0.2231 at steps 1, 5, 10, 25 against 4.2242, 3.1018, 1.8404, 0.2215.
Two things were found by this checking and are worth recording because the loss alone hid them: the first version's GELU read `tanh in * k` as `(tanh in) * k` (a call takes the next atom: the argument must be a name), a **2% error in the feed-forward's gradients that torch's own fp16 autocast did not have (0.14%) and that made the model learn visibly slower**, found by comparing intermediate activations with the specification, layer by layer;
and a loss scale of 1024 on the f16 gradients overflowed after 300 steps (NaN), so the scale is 16 (the errors were the same with and without it, so f16's subnormals were not the problem).

**Training on text** (`examples/gpt/compare_gpt.py`; the text is the Markdown documentation of this repository and of cancho, 4,000,000 ASCII characters, 96 distinct, no download; 90% to train and the last 10% to validate; AdamW at 1e-3 with a 5% warm-up and a cosine decay, batches of 32 sequences of 128 characters (4096 tokens a step), 1,500 steps, the same start (`init_params`) and the same batches for every runtime). Apple M4 Max:

| model | runtime | ms a step | tokens a second | validation loss after 1,500 steps |
|---|---|---|---|---|
| 4 layers, d 256, 4 heads, 3.26 M parameters | **cancho-gpu (Metal)** | **15.4** | **265,000** | 1.403 |
| | torch MPS (float32) | 16.0 | 257,000 | 1.400 |
| | torch MPS (float16 autocast) | 17.6 | 233,000 | 1.417 |
| | torch CPU | 69.5 | 59,000 | 1.400 |
| 6 layers, d 384, 6 heads, 10.80 M parameters | **cancho-gpu (Metal)** | **37.0** | **111,000** | 1.376 |
| | torch MPS (float32) | 41.3 | 99,000 | 1.380 |
| | torch MPS (float16 autocast) | 44.5 | 92,000 | 1.408 |

**Reading.** The loss curves agree at every point (validation loss 2.263 / 1.780 / 1.600 / 1.499 / 1.431 / 1.403 for cancho-gpu and 2.263 / 1.776 / 1.596 / 1.492 / 1.430 / 1.400 for torch at steps 250, 500, 750, 1000, 1250, 1500), so it learns the same thing; the 10.8 M model reaches 1.376 nats a character, under the roadmap's 1.6.
It is as fast as torch on MPS in float32 (3% faster at 3.3 M parameters, 10% at 10.8 M) and faster than torch's mixed precision, **after tuning**: the first working version took 28 ms a step; attention in 32-row tiles (the 64-row tiles of the first version left most of the GPU idle: 256 threadgroups of tiny work, 1.3 ms for the scores of a layer) and the weight gradients in 32 x 32 tiles (a reduction over 4096 tokens into a 256 x 256 result was 16 threadgroups) took it to 15.5.
**Not claimed:** the contexts above 128 (the attention kernels keep a head's whole row of scores as one tile, `bn = t`, which is why `t` is 64 or 128), a vocabulary of more than 128 (a real tokenizer needs the embedding lookup of the roadmap's G1), dropout, gradient clipping, anything about NVIDIA; one run each, and the time of cancho-gpu is measured in a separate session that answers only the loss (a parameter dump is 3 million numbers of text a request).

## 16. Serving what was trained (roadmap M2; measured)

`examples/gpt/serve_gpt.py train` trains the 3.26 M-parameter GPT of section 15 (1,500 steps, 25 s) in a Metal session, with the parameters in the answer, and saves them with the vocabulary as a checkpoint (`.npz`).
`serve_gpt.py generate` loads it into a **forward-only session** (`gpt_session.write(..., serve=True)`: the same kernels and the same f16 matrix-unit products as the forward of training, the optimiser and backward left out). A request is the 128 token ids of a window and a one-hot selector of the position to read, and the answer is that position's logits (`row_select.chx`: the selector times the f32 logits, so a request answers 128 numbers, not 16,384); the host takes the arg max (or samples).
Causal attention means the positions after the one read cannot influence it, so the window is the sequence so far followed by anything.

**Checkpoints are host-side** *(here; §19 builds the device-side `save` that this paragraph said only larger models would need)*. The roadmap named a `save` directive; none was needed to reach the exit: the training session already answers every parameter (`dump`), and a checkpoint is those numbers written by the host. A device-side `save` would only matter for models whose parameters do not fit a text answer, which M5's models (billions) will.
**No key/value cache.** Each token recomputes the 128-token window (about 40 launches). That is the *same algorithm* as the first torch column below, so the comparison with it is like for like; the cache is a better algorithm, and torch's own cached decoder is in the table as the stronger baseline. A cached decoder in the session needs a kernel that appends to a cache at a position taken from the request, which is the roadmap's M5 (G12).

**Correctness.** `tests/train.py`: the logits of one position of a 2-layer GPT within 3e-4 of the float64 forward on the CPU interpreter and 3.6e-4 on Metal, same arg max. On the trained checkpoint, six held-out prompts of 32 characters, 96 greedy tokens each (`serve_gpt.py generate`, M4 Max): **all 6 continuations are identical to torch's (float32 on MPS), 576 of 576 tokens**, and torch's recompute and torch's cached decoder agree with each other too. Teacher-forced (cancho's choice at each position of torch's own continuation) it is also 576/576, so the identity is not a lucky cascade.
f16 products could flip an arg max where two logits are within rounding; on this model and these prompts none does. That is a measurement on 576 tokens, not a guarantee, and the text is greedy-decoding's usual loop ("... that is not a second that is not ...").

**Speed** (tokens a second, one sequence, the first prompt excluded from torch's means for warm-up, steady-state values): 

| decoder | tokens a second |
|---|---|
| **cancho-gpu session** (recompute the window, request and answer as text on a pipe) | **650** |
| torch MPS float32, recompute the window | 340 |
| torch MPS float32, key/value cache | 273 |

cancho-gpu is 1.9x torch's same algorithm and 2.4x torch's cached decoder. Torch eager on MPS at batch 1 is bound by launching its ~100 small operations a token, and so is this (40 launches, one command buffer); neither number is compute. `torch.compile` and MPS graphs were not tried. Not claimed: anything for a window longer than 128 (the sliding window is not implemented), sampling quality, other models, NVIDIA.

## 17. A real vocabulary and a larger model (roadmap M3, first part; measured)

**The tokenizer and the data** (`examples/gpt/bpe.py`): a byte-pair encoding trained here, on text already on this machine and without a download: the ASCII of the Markdown, Rust and cancho files under the parent of this repository (each distinct file once), split into words, 1,951 merges learned on the first 8 M characters, **a vocabulary of 2,048 tokens and 15.5 M tokens** (90% to train, the last 10% to validate; a run sees 6 M of them, so no epoch repeats). TinyStories, the roadmap's data, needs a download and was not used; the numbers below are not comparable with its.

**Embedding without a new operation.** The roadmap expected G1 (an index-valued load and a scatter-add) here. At 2,048 tokens it is not needed: the one-hot product of section 15 is `[4096, 2048 + 128] x [2176, d]`, as large as the output layer's product, which every language model pays anyway. Two kernels had to stop holding a whole row in one tile (a threadgroup has 32 KiB): `onehot.chx` takes its columns in chunks of 128 (a second grid axis) and `ce.chx` takes one row a threadgroup (`bm=1`). Cost, measured on the 4-layer, d 256 model: **15.4 ms a step with 128 characters, 17.6 with 2,048 tokens** (+14%), and torch (which does gather the embedding) takes 17.8.

**Training** (`compare_gpt.py --bpe 2048`; 4,096 tokens a step, 1,500 steps, AdamW 1e-3, the same start and batches for every runtime, M4 Max):

| model | runtime | ms a step | tokens a second | validation loss (nats a token) |
|---|---|---|---|---|
| 4 layers, d 256, 4.24 M parameters | **cancho-gpu** | 17.6 | 233,000 | 3.257 |
| | torch MPS float32 | 17.8 | 230,000 | 3.248 |
| 8 layers, d 512, 8 heads, 27.4 M parameters | **cancho-gpu** | **81.0** | **50,600** | 2.989 |
| | torch MPS float32 | 86.0 | 47,600 | 2.982 |
| | torch MPS float16 autocast | 91.7 | 44,700 | 3.100 |

The validation curves agree at every point (27.4 M model at 250, 500, 750, 1000, 1250, 1500 steps: 4.250 / 3.712 / 3.423 / 3.202 / 3.060 / 2.989 for cancho-gpu, 4.280 / 3.706 / 3.411 / 3.199 / 3.057 / 2.982 for torch), 6% faster than torch float32 and 13% faster than its mixed precision. Two minutes for the whole run.

**What stops short of the roadmap's exit, and why.**
* **27 M parameters, not 30 to 60 M.** *(Removed in §19, which built the device-side `save` and trained and validated 60 M.)* A d 768, 8-layer (57 M) session is accepted and runs, but the curve needs the parameters in an answer, and a run's tensors must fit the interpreter's memory (32 M numbers): the answer path refuses above it ("the run's tensors do not fit the interpreter's memory"). The way past is a device-side `save` (G10 proper), which needs the session to have `fs_write`, a widening of `cancho-gpu-session`'s authority that is the owner's decision. Training itself has no such cap (tables and residents go straight to the device, section 15).
* **Vocabularies above about 2,048.** `ce` still holds a row of the logits in a tile; a larger vocabulary needs it in column chunks with a two-pass maximum and sum, and, past about 8 k tokens, the one-hot product stops being cheap and G1 (gather, and a scatter-add with atomics or a sort-and-segment) starts to pay. Not built, not measured.
* **Memory against torch** was not measured. The context is still 128.

## 18. A convolutional network trained on the device (roadmap M4; measured)

`docs/roadmap-models.md` M4 asked for a CIFAR-style ResNet of about 1 M parameters trained in a session and compared with torch. It is built, held to a float64 specification, and trained and timed against torch on MPS on a **synthetic** image set. CIFAR-10 is not on this machine (searched under the home directory and `/Users`: no `cifar*`, no `data_batch_1`, no torchvision or keras cache) and
downloading it needs the owner's permission, which has not been given; **no number below is a CIFAR-10 number**, and the 90% exit of the roadmap is open ("What is not done").

### 18.1 Design (written before the code)

**A convolution is nine products of one matrix, no new operation.** Images are NHWC and are kept as a *padded matrix*: each image has a border of one row and column of zeros, the `n` images are flattened to `[n (h+2)^2, c]`, and `g = 64` zero rows precede and follow the matrix.
The output pixel at row `p` of a 3x3 convolution reads the input rows `p + (dy-1)(h+2) + (dx-1)` for the nine taps `(dy, dx)`: nine windows of one matrix, each shifted by a constant number of rows, each multiplied by that tap's `[cout, cin]` slice of the weights `[cout, 9 cin]`
(`mma` on f16 tiles, f32 accumulation). A window that starts at a computed row offset is what `load x[g + i * bm + dy * wp + dx - wp - 1, ...]` already was: an affine offset in the grid coordinate and the loop indices. The three loops (`dy`, `dx`, the channels) are loops, not nine copies:
unrolled, the nine pairs of tiles each count against the 32 KiB of threadgroup memory (`budget`), and only small tiles fit. The rows of the output that are the border hold junk (they read across a row's end); the epilogue multiplies by a mask and writes zero there, so the next convolution sees a zero border.
The two gradients are the same loops: the input gradient is `mma_nn` of the output gradient's windows at the *opposite* shifts with the same weight slice, and the weight gradient of a tap is `mma_tn` of the output gradient and the input shifted by that tap, summed over the rows. The weight gradient has a reduction over every pixel of the batch,
so the rows are cut into pieces (a third grid axis) that a small kernel (`wsum`) adds.

**BatchNorm** is a reduction over the batch and the pixels of each channel: `bn_part` writes, for blocks of rows, the sums of the valid (non-border) rows of `z` and `z^2` (a product of a 0/1 row with the tile), `bn_fin` adds the blocks and finishes (`mean`, `1/std`, the scale `gamma/std` and shift, the running statistics),
`bn_apply` is the epilogue (`relu(z scale + shift [+ skip])` masked, written as f16 into the padded matrix of the next layer). Its backward is the same two levels (`bn_bwd_part`, `bn_bwd_fin`) and `bn_bwd_apply`, `dz = gamma/std (d - dbeta/m - xhat dgamma/m)`, with `d` the gradient of the ReLU's output times `step(pre-activation)`.
The residual variants add the skip (f16) in the forward and also write `d` as the gradient that goes around the block.

**Pooling is average pooling, not max-pooling, and downsampling is a product.** `max a b` was in the plan (G11) and is not built: a 2x2 *average* pool is a sum of two rows of the padded matrix times a constant matrix `pm` `[h/2 + 2, h + 2]` (0.25 at the two columns of each output pixel, zero rows for the border), so it needs no new operation,
and its backward is the transpose of that matrix. The global average is a product with a row of `1 / (h h)` on the valid pixels. Average pooling gives up some accuracy on CIFAR against max-pooling (the usual ResNet uses stride-2 convolutions and an average at the end, not max); `max` remains open and would also give max-pooling its backward (a mask) here.

**The network** (`cnn_ref.py`): a stem convolution, then stages of widths 32, 64, 128 with 1, 2, 2 residual blocks (`relu(bn(conv(relu(bn(conv(a))))) + a)`); each later stage begins with a 2x2 average pool and a convolution to its width; a global average; a linear classifier without bias (the classes padded to 64, the extra logits masked as the GPT's vocabulary is).
13 convolutions, **852,000 parameters**. Mixed precision as the GPT's: f32 master weights and Adam moments, f16 copies for the matrix unit, f16 activations and f16 gradients of the pre-activations (with the loss scale 16), f32 accumulation, BatchNorm statistics and residual gradients. AdamW; the decay applies to the convolutions and the classifier, not to BatchNorm's scale and shift.
**The input** is two tables of half batches (`[64 h h, 3]` f16 each: a session stages a file in a 256 MiB pool, so a table of a whole epoch has to stay under that), padded to 8 channels and to the border by one kernel (`pack`).

**Specification and checks.** `cnn_ref.py` is the network in binary64 numpy with a hand-written backward, checked against finite differences (28 tensors in the small configuration, worst relative difference 8e-7); with `QUANT` it rounds to binary16 where the device does, which is what the kernels are held to.
`tests/cnn.py` (in the gate, CPU session; Metal with a second argument): every kernel on its own against numpy from the same f16-representable inputs, two mutants, then the whole network.

### 18.2 Measured

**Correctness** (`tests/cnn.py`, 47 checks on the CPU interpreter and on Metal, none failing). (1) Every kernel on its own against numpy from the same f16-representable inputs: `conv3`, `conv3_dx`, `conv3_dw` with `wsum`, `bn_part`, `bn_fin` (mean, 1/std, scale, shift, and the running statistics with the unbiased variance), `bn_apply` and `bn_apply_res`
(including that the border and the guards stay exactly zero), `bn_bwd_part`/`_fin`/`_apply` with and without the skip, `pack`, `pool_fwd`, `pool_bwd`, `gap_fwd`, `gap_bwd`: all within 1e-4 relative (2e-3 where the result is f16), on both. Two mutants (a wrong tap shift in `conv3`, in `conv3_dx`) are caught.
(2) The whole network (7 convolutions, 8x8 images, 28 tensors) against `cnn_ref.py` rounded to binary16 where the device rounds: the loss 2.53417 (CPU) and 2.53422 (Metal) against 2.53397, and the gradients of all 28 tensors with a cosine of at least 0.996 and a relative error between 3e-2 and 1e-1.
**That error is a floor, not a bug, and it is the first thing a reader will ask about.** A ReLU whose input is within the rounding of f16 of zero falls on either side of it depending on the order of an f32 sum; one such element moves the gradient of its channel's shift and scale by its own gradient, and the number of such elements is proportional to the number of elements, so the relative error does not fall with
the batch (it was the same, 4 to 9%, with 16 and with 128 images and with the arithmetic emulated to binary16, where every f16 *activation* agreed with the specification to one rounding step; the first suspicion, a wrong BatchNorm backward, was ruled out by checking `dz`, `dbeta` and the input gradient layer by layer: the whole difference at the first
layer that showed it was one element, a ReLU on the other side of zero by 7e-5). This is why the kernels are tested on their own with exact inputs (1) and the network by cosine (2). The GPT had no such floor because GELU is smooth. (3) 25 AdamW steps on a learnable label: the loss falls from 2.02 to 0.70 (CPU) and 0.67 (Metal), and the Metal session follows the CPU session.

**Training against torch** (`examples/cnn/compare_cnn.py`; 20,480 training and 5,000 test images of a synthetic set, 10 classes of 32 x 32 x 3 textures: each class is a pair of gratings with jittered orientation and frequency, plus a class-independent distractor grating and Gaussian noise (`cnn_data.py`); not CIFAR-10 and not comparable with its numbers;
batch 128, 20 epochs = 3,200 steps, AdamW at 2e-3 with 3% warm-up and a cosine decay, weight decay 0.01 on convolutions and classifier; the same start (`init_params`), the same two shuffles of the images taken in turn, for every runtime; the test accuracy of every runtime is computed from its parameters and BatchNorm running statistics by the same torch code):

| runtime | test accuracy after epochs 5 / 10 / 15 / 20 | test loss at the end | ms a step (interleaved rounds, quiet machine) | images a second |
|---|---|---|---|---|
| **cancho-gpu (Metal)** | 0.748 / 0.833 / 0.899 / 0.911 | 0.349 | 17.6 | 7,300 |
| torch MPS float32 | 0.696 / 0.782 / 0.904 / 0.913 | 0.343 | **10.6** | 12,000 |
| torch MPS float16 autocast | 0.769 / 0.846 / 0.903 / 0.908 | 0.351 | 11.0 | 11,600 |

**Reading.** They learn the same thing: 0.911, 0.913 and 0.908 at the end and test losses of 0.349, 0.343 and 0.351 (the three differ from each other by about what two float32 runs differ, and the intermediate epochs differ by more, as ReLU networks' do: the accuracies swing by several points between epochs 7 and 13 in all three); the loss falls to 0.002 on the training batches. **cancho-gpu is 1.7 times slower than torch on MPS**
(17.6 ms against 10.6 and 11.0): the exit of the roadmap, "the time torch needs", is **not met**. Where the time goes (`CANCHOGPU_PROFILE=2`, a quiet moment, ms a step): the three convolutions' kernels 3.2 (forward), 3.1 (input gradient) and 3.7 + 1.9 (weight gradient and its sum), together about 12 ms at roughly 6 to 8 TFLOPS, and the rest in the other launches (186 a step in all, 40 of them the optimiser's, one per tensor): BatchNorm's two-level reductions
(`bn_fin`, `bn_bwd_fin` 3 ms together: each is a loop of 10 to 70 iterations on 1 to 4 threadgroups), the elementwise epilogues, the residual add. What would help, not done: BatchNorm's sums folded into the convolution's epilogue (the tile is in registers there), fewer partial blocks, the weight gradient and its sum in one kernel, larger convolution tiles
(the nine-tap loop needs one pair of tiles in threadgroup memory; the unrolled form could not hold bigger than 64 x 32 x 16).
The wall time of the 3,200 steps was 144 s for cancho-gpu, 92 s for torch float32 and 103 s autocast in the run that made the curves, **on a machine that another process was using at the same time** (45, 29 and 32 ms a step; the rounds of `bench_cnn.py`, which interleaves the runtimes and shows the busy rounds, give the table's numbers: the same session ran at 17.6 ms in its quiet rounds and 41 in its busy ones, torch at 10.6 and 29).

**Tuning that mattered** (before the table): the weight gradient of the 128-channel layers computed a 128 x 128 tile per threadgroup and took 9 ms a layer (the accumulator does not fit a threadgroup's registers); looping over 64 x 64 tiles (with a carried dummy tile and a `store` inside the loop, which the language allows) took it to 0.28, the step from 57 to 19 ms; and the unrolled nine taps
broke the 32 KiB budget at 64 x 32 x 32, which is why `conv3` is a nest of loops.

### 18.3 What is not done, and the limits

* **CIFAR-10 was not used.** It is not on this machine, and the download (`https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`, about 163 MB) needs the owner's permission. `cnn_data.cifar(directory)` reads the archive's python batches (checked only against the format, not on the real files), and `compare_cnn.py --data <cifar-10-batches-py>` runs the same comparison on it. **Accuracy was measured only on the synthetic set**;
  on CIFAR-10 a ResNet of this size is expected to reach about 90% with augmentation (crops, flips), which is **not implemented**: the batches are fixed shuffles taken in turn (a table is a file, and a file is staged in a 256 MiB pool: the two half-batch tables hold 153 MB each for 50,000 images, so one lap of CIFAR-10 fits and two do not).
* **Average pooling, not max-pooling**; `max a b` is not in the surface. Strides above 1 are not implemented either (the pool and a convolution do the same job here). Only 3x3, stride 1 convolutions with no bias (BatchNorm follows each), and a linear classifier with no bias.
* **The geometry**: the padded rows of every stage must be a multiple of 16 (a batch of 16 or any multiple of it with sides 32, 16, 8 and 4) and the guards of 64 rows cover sides up to 61; the batch must be even; `cin` is padded to a multiple of 8 for the matrix unit.
* **BatchNorm's variance is `E[z^2] - mean^2` from f32 block sums**, one pass, not Welford: fine for the unit-scale activations here, and would lose digits for a layer whose mean is many times its deviation.
* The zero border and the guard rows rely on a session's buffers **starting zeroed** (an output is written only where it is valid): true of the CPU interpreter, and of the Metal buffers in every run here, but not something the surface promises.
* CUDA: the kernels compile for `nvidia-ada` and have goldens (`tests/cases.txt`), and have never run on a GPU. Eval-mode accuracy is computed on the host from the dumped parameters, not by a forward-only session.

## 19. `save` and `checkpoint`: parameters out of a session as a file (roadmap G10 and M3; measured)

§17 stopped at 27 M parameters because the only way out of a session was a text answer, and an answer must fit the interpreter's memory (32 M numbers). This section is the way out: a **binary file written by the session program from the device buffer**, with no text and
no number passing through the interpreter.

**The directive.** `save PROGRAM.OUT FILE [f32|f16]` names an output of a kernel and a file. The output is written as little-endian IEEE words, `f32` unless the flag says `f16`; the flag must match the output's type (a mismatch is `usage`, as for `resident`,
so no conversion is done). Several `save` lines may name one file, **listed together**: the file holds their tensors one after another in the order of the lines, and is written once (`gpt_session.py` saves all 28 tensors of a 2-layer GPT, or 100 of an 8-layer one, to one
`ckpt.bin`, in the order of `names(c)`). The saved buffer is what the last repetition left, after any `carry` (the output of an `adamw` kernel is the new weight).

**The trigger.** A request line that is the word `checkpoint` and nothing else. It runs no program, takes no step, moves no table, and answers one line: the bytes written to each file, in the order the files are first named (`20` for one file of five floats, `20 20` for two files).
It is a word and not a flag on an ordinary request for three reasons: it cannot be mistaken for a count of steps or an input (neither parses as `checkpoint`); a session without `save` is unchanged (and a `checkpoint` there is refused, rule `session`); and **a request can never name a file**,
because the word has no argument (`checkpoint /tmp/x` is `session`). The path comes only from the session file, which is the owner's. A device session has finished every kernel of the last request before it reads the next line, so the buffers a checkpoint downloads are the state after the last step, with no extra wait.

**Atomically.** The file is written whole to `FILE.tmp` (beside it, so the rename stays on one filesystem), made durable (`file_sync`) and renamed over `FILE` (`fs_rename`): a reader sees the old file or the new one, never half of one. A failure (the directory is not there, the disk is full, the rename is refused)
leaves the old file as it was, removes the temporary file, and is the rule `io` with the path and the errno. A leftover `FILE.tmp` from a crash is truncated by the next checkpoint. Not claimed: that a power loss between the rename and the directory's own flush keeps the new name (the directory is not synced), or anything about a filesystem that does not make `rename` atomic.

**Refusals, all before any device work, all with the line of the session file** (`session` unless said): a `save` of a parameter the kernel only reads; a flag that is not `f32` or `f16`; an output of another type than the flag (`usage`); a path with two dots in a row or of 4,000 bytes or more (the runtime's `Fs` traps on those, and no input reaches a trap, so they are refused first);
a file that is also a `resident`, `table` or kernel file of the session (a save does not overwrite what the session reads; this compares the names as written, so two spellings of one path are not caught); saves to one file that are not listed together. At the checkpoint: a request with anything after the word, and a session with no `save`.

**The authority.** `cancho-gpu-session` and `cancho-gpu-metal-session` were `args, err_write, file_read, fs_read, heap, io_read, io_write` (`io_read` for the request lines): they could write nothing but standard output. They now also perform **`fs_write("")` and `file_write`**. This widening was **explicitly approved by the repository owner** for this change;
it is the last row `tests/authority.sh` pins for the CPU session (`args, err_write, file_read, file_write, fs_read, fs_write, heap, io_read, io_write`; no network, no foreign code), and a further change to that list needs the same approval. What the write can do is narrower than the label: the only paths written are the `save` paths of the session file (and their `.tmp` names), a request cannot name one, and the label `fs_write("")` is the broadest
prefix the language can say, so the report cannot show that; this section and the tests do. The Metal session is not in `tests/authority.sh` (its report includes `ffi("libmetalgpu")`, and the gate does not build it); it shares every line that writes (`src/ckpt.cho`).

**The code.** `src/session.cho` reads and checks the directive (`s_saves`); `src/ckpt.cho` is the request word, the temporary file, the CPU session's writer and the common pieces (`write_all`, `finish`); `device/metal-session-main.cho` downloads each buffer with the shim's `cgpu_download` into the byte pool
(one tensor at a time: the pool, 256 MiB, bounds the largest tensor and not the model) and writes it. The CPU session encodes from the interpreter's memory (`devcodec.encode`), so it can save only what the interpreter holds.

**Checks that can fail** (`tests/session.py`, CPU and Metal; `tests/train.py`):
(1) a save writes nothing until a checkpoint; the file holds exactly the output's words after 3, then after 7 steps, equal to a float32 simulation with non-dyadic values; the answer is the byte count; a checkpoint takes no step;
(2) two programs into one file in order, two files with one total each, an f16 output as binary16 words, an existing file replaced;
(3) a failed checkpoint (the temporary name taken by a directory; a directory that is not there; the rename refused because the name is a directory) is `io`, leaves the old file and no temporary file;
(4) 13 refusals of a session file and 4 of a request (a file named, a number after the word, the word in capitals, no save), none writing anything;
(5) a 2-layer GPT trained 25 AdamW steps with a `save` of all 28 tensors: the file is **bit for bit** what the same session answers as text with `dump` (CPU and Metal), and `read_checkpoint` restores every shape.
**Mutants:** 14 on the CPU session, all killed (the type check, the two path checks, the grouping, the resident-file check, the input check, the rename, the temporary file, the group boundary, the last byte, the total, the checkpoint word with an argument, a checkpoint without a save, a failed checkpoint's leftover file; with the 16 of §12 that is 30 of 30), and 4 built by hand on the device program
(`device/metal-session-main.cho`: the last byte of a download, every save downloading the first save's buffer, a wrong total, the word not recognised), all killed by `tests/session.py --metal`.
A first version of the failed-checkpoint test did not kill "a failed checkpoint leaves its temporary file" (its failure came before the temporary file existed); the case was added that fails at the rename.

### 19.1 The GPT of §17 at 60 M parameters (roadmap M3; measured)

`compare_gpt.py` now gets the curve from `save`: one session answers only the loss, the timed requests run the steps, and after each stretch an untimed `checkpoint` writes the parameters to a file that the float64 forward reads (`gpt_session.read_checkpoint`). Before, the curve cost a second, dump-answering session that could not exceed 32 M numbers.
`serve_gpt.py train` writes its checkpoint the same way (`ckpt.bin`, then the `.npz` that `generate` loads).

`python3 examples/gpt/compare_gpt.py --bpe 2048 --layers 8 --d 768 --heads 12 --steps 1500 --only cancho,torch-mps,torch-mps-amp` (Apple M4 Max, torch 2.12, the same start, the same batches and AdamW for all; the same 15.5 M-token BPE corpus as §17; 4,096 tokens a step; AdamW 1e-3, warm-up 5%, cosine; one run each; nothing else running):
**8 layers, d 768, 12 heads, 59.95 M parameters** (the roadmap's 30 to 60 M; the embedding and output layer are 2 x 2,176 x 768 of it).

| runtime | ms a step | tokens a second | 1,500 steps | validation loss at the end |
|---|---|---|---|---|
| **cancho-gpu (Metal session)** | **167.8** | **24,400** | 252 s | **3.106** |
| torch MPS (float32) | 196.7 | 20,800 | 295 s | 3.114 |
| torch MPS (float16 autocast) | 201.1 | 20,400 | 302 s | 3.278 |

Validation loss (nats a token, four held-out batches of 4,096 tokens, float64 forward) at steps 250, 500, 750, 1000, 1250, 1500: cancho-gpu **4.339 / 3.815 / 3.572 / 3.331 / 3.195 / 3.106**; torch float32 4.352 / 3.821 / 3.563 / 3.339 / 3.200 / 3.114; torch float16 autocast 4.408 / 3.910 / 3.671 / 3.481 / 3.353 / 3.278.
**Reading.** The curves of cancho-gpu and torch float32 agree to within 0.013 nats at every point and cross twice, so it learns the same thing; torch's mixed precision is behind throughout (as at 27 M in §17). cancho-gpu is **15% faster than torch float32** (167.8 against 196.7 ms) and 17% faster than its mixed precision: the margin at 27 M was 6%.
**No kernel schedule change was needed**: the schedules tuned at d 256 and d 512 (`bm 1` for the elementwise kernels from d 640, which `gpt_session.py` already did, attention and weight-gradient tiles of 32) run at d 768 as they are, and four steps of the first run already showed 155 ms. The 168 ms is not tuned for this size; the step at d 768 was not profiled.

**Limits, honestly.** One run each, so a difference of 0.01 nats between runtimes is inside what a different seed would move. The validation set is 16 K tokens drawn from the same corpus as the training text (a held-out tail, 10%), not TinyStories (it needs a download, so its numbers are not comparable). The loss reached, 3.1 nats a token, is a model after 6 M tokens, not a trained one; 1,500 steps is a measurement of speed and agreement, not of how good a 60 M model can be.
Memory against torch was not measured. The context is still 128, the vocabulary 2,048 (§17's other two limits stand). A checkpoint of this model is 239,794,176 bytes (59.95 M f32 words, 100 tensors in one file) and a `checkpoint` request took **0.04 s** (three requests: 0.039, 0.043, 0.046 s, M4 Max SSD; macOS's `fsync` does not force the drive's own cache, so this is durable to the operating system, not to the platter).
The `.bin` has no header: the shapes come from the configuration, so a file with the wrong configuration is caught only by its size (`read_checkpoint` checks the word count). Loading goes through the host: `read_checkpoint` gives the tensors and `gpt_session.write(params=...)` writes the per-tensor `resident` files of a new session (the optimiser's moments restart at zero); a directive that loads one combined file into the parameters is not built.

## 20. Qwen2.5-0.5B's forward pass on the matrix unit (roadmap M7.1; measured)

`docs/roadmap-models.md` section 6 (M7) asked, as its first step, for the forward pass of Qwen2.5-0.5B over a batch of 4,096 tokens (8 sequences of 512) in a Metal session on the matrix unit, held to Hugging Face transformers: the loss within 2e-3, the arg max of the logits equal at 99% of the positions, the speed against torch. It is built and it meets all three. Gradients and LoRA are M7.2 and are not here.

**What was built** (`examples/qwen/`; every kernel is `.chx`, **no change to the language was needed**):
`convert.py` (the safetensors as binary files), `qwen_ref.py` (the specification: binary64 numpy written from the architecture, no framework), `qwen_session.py` (the session writer), `data.py` and `batch_8x512.npy` (the fixed batch: the Markdown, Rust and cancho files of this repository and of the `cancho` checkout tokenized with Qwen's own tokenizer, eight windows of 512 spread over the stream; the tokens are committed because the corpus changes as the repositories do; no download),
`hf_reference.py` (transformers' answers), `compare_qwen.py` (the exits), `torch_forward.py` (the baseline), and `tests/qwen.py` (in the gate).

* **The converter.** The matrices go out as f16 (q, k, v fused into one `[1152, 896]` matrix with a `[1152]` bias; gate and up fused into `[9728, 896]`; the tied embedding `[151936, 896]`), the norm gains and biases as f32. Of the 493,961,216 matrix values the largest is 1.664 against f16's 65,504: **none overflows**; **117 non-zero values are below 2^-25 and become zero**, and 1,449,082 become f16 subnormals (each with an absolute error of at most 2^-25 = 3e-8). A weight that does not fit f16 is a refusal (`tests/qwen.py` writes one and requires it).
* **The kernels**, per block 16 launches: `rms_h` (RMSNorm to f16), the fused q/k/v `gemm_mma`, `rope_q`, `rope_k` and `bias_v` (generated by `qwen_session.rope_source`: bias, the rotary rotation of the "rotate half" pairs when asked, f16, and the heads-major layout below), `att_s` (scores), `att_sm` (mask, softmax), `att_o` (probabilities times values), `unheads`, `gemm_mma` for o, `res_add`, `rms_h`, the fused gate/up `gemm_mma`, `swiglu_h`, `gemm_mma` for down, `res_add`. After the last block: `rms_h`, the head product `[4096, 896] x [896, 152064]`, and the loss as **`ce_stats` and `ce_comb`**.
  * **RoPE.** A position is `i * bm + (row of the tile)` within a sequence, so the grid is (sequence, head, row tile) and the cosine and sine tables are `[512, 32]` windows at `i * bm`: every offset is affine and no index is divided.
  * **Grouped-query attention** (14 query heads over 2 of 64; a head's group is `h / 7`, a division the surface forbids). Two designs were built and measured. **(A) a loop over the 7 queries of a group** inside a kernel gridded over (sequence, key/value head, row tile), with a store in the loop body (a `for` with a dummy carried tile, which the surface accepts): the scores of the 112 (sequence, head) pairs take **1.0 ms** at best (threads 256, `warps 1 8`, `bm 8`, `bn 512`, `bk 16`; 1.4 ms at `warps 1 4`; 36 ms at `bm 16`, `warps 2 2`, where the 16 x 512 accumulator spills) against **0.33 ms** for (B). **(B) is chosen: the key/value heads are written once per query head of their group by the kernel that rotates them** (seven stores of the same tile, so no copy kernel and no extra pass: `rope_k` and `bias_v` together take 1.3 ms of the 376 for all 24 layers), and q, k and v are all kept **heads-major**, `[sequence x head x position, 64]`. Then every attention kernel has one grid coordinate `p = sequence x 14 + head`, every offset is affine in it, and the scores can be **tiled over key columns in ordinary 64 x 64 tiles** (grid `p, i, j`), with the rows written to memory as f32 (117 MB a layer) and the softmax taking whole rows of 512 (`att_sm`, `bm 2`). The threadgroup memory of 32 KiB is why the 16 x 512 tile of (A) needs `bk 16` and why (B)'s tiles are small.
  * **The loss over 151,936 columns, three kernels and no `max a b`.** The vocabulary is rounded up to 152,064 = 66 chunks of 2,304 (the output matrix gets 128 zero rows, whose logits are masked by `-1e9`: a tile must hold a multiple of the 128 threads, and 151,936 = 128 x 1187 with 1187 prime has no usable divisor). `ce_stats` (one threadgroup per row and chunk: `rowmax`, `exp`, `rowsum`, the logit of the target by the `relu(1 - |j - t|)` selection of `ce.chx`, and the 1-based column of the largest logit) writes four `[4096, 66]` tables; `ce_comb` (a row of 66) combines them: `log(sum_q s_q exp(m_q - M)) + M - sum_q t_q` and the arg max. The 2.5 GB of logits are never read back.
  * **Causality.** The mask is an additive table, and the full 512 x 512 scores are computed (about 0.18 TFLOP of the 4.2 of the pass is above the diagonal). A loop bound that depends on the row tile (`for c in 0 .. i + 1`, to skip the tiles above the diagonal) is refused: *an extent is a constant, not an index*.
* **The session** (`qwen_session.write`): 24 x 16 + 4 programs, the weights `resident` as f16 files, the embedded batch `x0` as one f32 file (the lookup is host work: no gather is needed for a forward pass), `steps`, `answer cc.loss` and `answer cc.am` (8,192 numbers), and, for the comparison, `save` of the residual stream after every block (24 x 14.7 MB, written on the request `checkpoint`). Mixed precision as in sections 14 and 15: f16 weights and f16 operands of every product, f32 accumulation and residual stream; the softmax probabilities are rounded to f16 for the product with the values.

**Exits** (Apple M4 Max, `python3 examples/qwen/compare_qwen.py WORK --torch --spec`; transformers 5.2 float32 on the CPU, torch 2.12; the batch is the committed one):

| | cancho-gpu session | transformers float32 | exit |
|---|---|---|---|
| mean loss over the 4,088 positions with a next token | **2.463544** | 2.463422 | within 2e-3: **+1.2e-4** |
| arg max of the logits equal at | **99.83%** (7 of 4,096 differ) | | at least 99% |
| the 7 that differ: the gap between transformers' two largest logits | 0.0007, 0.0007, 0.0016, 0.0019, 0.0036, 0.0039, 0.0068 | | near-ties |
| the binary64 specification | loss 2.463514 (+9.2e-5), arg max equal at **100.00%** | | |
| residual stream after block k, relative L2 | 1.1e-3 (k = 1), 3.4e-4 (4 to 12), 3.9e-4 (16), 5.3e-4 (20), 9.2e-4 (22), **1.1e-3 (24)** | | |

The error does not grow with depth: it is set in the first two blocks (1.1e-3 and 1.2e-3 relative L2; block 1 holds Qwen's well-known outlier channels, which are large in the residual stream, so f16 operands round them coarsely) and is flat after. Relative to the *largest* value of the stream the worst is 1.4e-2 at block 22, from a few outlier elements of a stream whose largest value is in the thousands.
**Speed** (4,096 tokens, loss included, the logits made):

| runtime | ms a forward | tokens a second | peak memory |
|---|---|---|---|
| **cancho-gpu (Metal session)** | **382.1** (median of 5 requests; the first, 765 ms, compiles and warms) | **10,720** | 16.35 GB (`/usr/bin/time -l`, peak footprint) |
| torch MPS float32 | 488 | 8,395 | 14.4 GB (driver) |
| torch MPS float16 autocast | 474 | 8,650 | 15.0 GB (driver) |

**1.28 times torch's float32 and 1.24 times its float16 autocast** (the plan expected parity). The pass is 4.2 TFLOP, so 11.1 TFLOPS. **Where the GPU time goes** (`CANCHOGPU_PROFILE=2`, one request, 376 ms of kernels for the 382 of the request): gate/up product 132.0 ms (13.0 TFLOPS), head 88.4 (12.6), down 71.3 (12.0), q/k/v 16.3, o 13.0; then `att_sm` 11.3, `ce_stats` 10.8, `swiglu_h` 9.3, `att_o` 8.4, `att_s` 7.9, `res_add` 3.1, `rms_h` 1.9, the rotary and unheads kernels 2.9, `ce_comb` 0.01. **The five products are 85% of it.**
**The tile schedule of the products got its own measurement** (isolated, medians; TFLOPS at `[4096, 9728, 896]` / `[4096, 896, 4864]` / the head): the default `bm 64 bn 64 bk 32`, 128 threads, `warps 2 2` is the best, **13.1 / 12.2 / 12.6**; `bk 64` 11.5 / 11.9 / 12.1; `bn 128` with 256 threads 10.9 / 11.3 / 11.5; `bm 128 bn 64` 10.5 / 11.4 / 12.2; and **every 128 x 128 or 64 x 128 or 128 x 64 tile with 128 or 256 threads falls to 0.6 to 1.7 TFLOPS** (the accumulator no longer fits the registers). So no schedule change was made, and the products are within a few percent of what these kernels can reach.

**Bugs found** (the per-block comparison with transformers is what localises an error; none of these reached the numbers above):
1. **A resident file larger than the session's byte pool killed the process with status 132 and no message** (the Qwen output matrix is 272,498,688 bytes in f16, the pool was 256 MiB). That is an input reaching a trap. `fileio.prefix` now refuses a file that does not fit what is left of the pool (`usage`, naming the file and its size; found by this work, so `tests/session.py --metal` has the case), and the Metal session's pool is **512 MiB** (`device/metal-session-main.cho`), the bound on the largest tensor a device session stages. The CPU session's pool is unchanged at 256 MiB, and the CPU session does not run the real model (its interpreter memory is 32 M numbers), only the tests.
2. **A tile must hold a multiple of the threads** (`error[offset]: 320 does not divide by 128`, found on the 320-column vocabulary of the test model): the cross-entropy's chunks could not be cut from 151,936 = 128 x 1187 (1187 is prime) either, so the vocabulary is padded and masked (above).
3. **`store &x` of a staged f16 tile moves it** (`use-after-move`): the key/value copies stage their tile once per store.
4. The first comparison used a batch from a corpus that included a sibling checkout another process was editing; the transformers reference moved by 0.021 nats between two runs of `hf_reference.py`. The tokens are committed (`batch_8x512.npy`) for that reason.

**What the tests hold** (`tests/qwen.py`, 25 checks with the Metal session given (the gate runs it on the CPU session)): the converter on a two-layer fake checkpoint in bf16 (fused rows in order, f32 vectors, an overflow refused, flushes counted); the cross-entropy over 3 chunks with targets in every chunk and a padded, masked tail (the padding columns hold 50, so a missing mask fails) against numpy; a 2-layer Qwen-shaped model with random weights (4 query heads over 2 of 32, biases, rotary, causal mask; 2 sequences of 64): the loss of each of the 128 positions, the arg max and the residual stream after each block against `qwen_ref.py` to 3e-3 (observed 5e-4); and **eight mutants, all killed**: six of the specification (rotary pairing adjacent elements, no q/k/v bias, a head reading key/value head `h % 2`, no causal mask, gate and up exchanged, RMSNorm with the mean subtracted: each is 0.2 to 1.0 from the session, so the check would fail on a kernel that did that) and two of the kernels (the rotation's sign; the copy for the second head of a group written to the first). The specification itself was checked against transformers on the real weights (table above, and block by block to 1e-6 of the largest value on an earlier batch).

**Limits, honestly.**
* **Memory is not smaller than torch's, it is 14% larger** (16.35 against 14.4 GB): the session keeps the output of every program of every layer (about 512 MB a layer: 117 MB of f32 scores and 59 MB of probabilities, 159 MB of the gate/up product, then 2.5 GB of logits and 1 GB of weights), because buffers are not reused across programs. A forward pass needs one layer's at a time (about 5 GB with the logits and weights) **[derived, not built]**; M7.2 needs the saved activations anyway, so this is where the plan's memory win has to come from, and it is not shown here.
* Only `t = 512` and 8 sequences were run on the real model. `t` must be a multiple of 64 and the scores are an `[t, t]` f32 table per head and sequence (117 MB a layer at 512, 470 MB at 1,024, which was not tried). The causal half is computed and masked, not skipped.
* One batch, one run of each timing; the torch numbers are medians of 8. The arg max differs from transformers' at 7 positions, all within 0.007 of a tie; the specification's own arg max agrees with transformers' at every position, so the rounding of f16 products accounts for them.
* torch MPS's float32 and float16 times are for `model(input_ids, labels)` under `no_grad`, with the logits made (the same work); its peak is the driver's allocation, ours the process's peak footprint, which counts the same unified memory.
* Metal only: the CPU interpreter cannot hold the real model, and nothing here has run on NVIDIA (the kernels compile for it and the tiny model runs through the CPU session with `--target nvidia-ada`).

## 18.1 The same comparison on real CIFAR-10 (measured)

`compare_cnn.py --epochs 30 --train 50000 --test 10000 --laps 1 --data build/cifar/cifar-10-batches-py`: the 50,000 training and 10,000 test images of CIFAR-10 (downloaded once, with the owner's permission, from the University of Toronto's `cifar-10-python.tar.gz`, 163 MB), the 0.85 M-parameter ResNet of section 18, no augmentation, batch 128, 30 epochs (11,700 steps), AdamW 2e-3, the same start and batches for every runtime. One run each; **another agent used the GPU during it, so the step times are rough** (the interleaved rounds of section 18 are the better timing).

| runtime | ms a step | images a second | training time | test accuracy at the end | best |
|---|---|---|---|---|---|
| cancho-gpu | 17.4 | 7,338 | 204 s | 0.8740 | 0.8743 |
| torch MPS float32 | 10.7 | 11,997 | 125 s | 0.8697 | 0.8699 |
| torch MPS float16 autocast | 11.1 | 11,512 | 130 s | 0.8674 | 0.8674 |

**M4's exit of 90% test accuracy is not met by any runtime**: without augmentation this network stops at 87%, and torch gets the same. The three agree on the accuracy (0.874 / 0.870 / 0.867) and their test losses cross each other from epoch to epoch as ReLU networks' do, so the learning is equivalent; the time to the final accuracy is 1.6x torch's, as in section 18. Reaching 90% needs augmentation (random crops and flips) or a wider network, neither of which was built.
