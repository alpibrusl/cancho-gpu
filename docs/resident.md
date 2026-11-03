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

## 18.1 The same comparison on real CIFAR-10 (measured)

`compare_cnn.py --epochs 30 --train 50000 --test 10000 --laps 1 --data build/cifar/cifar-10-batches-py`: the 50,000 training and 10,000 test images of CIFAR-10 (downloaded once, with the owner's permission, from the University of Toronto's `cifar-10-python.tar.gz`, 163 MB), the 0.85 M-parameter ResNet of section 18, no augmentation, batch 128, 30 epochs (11,700 steps), AdamW 2e-3, the same start and batches for every runtime. One run each; **another agent used the GPU during it, so the step times are rough** (the interleaved rounds of section 18 are the better timing).

| runtime | ms a step | images a second | training time | test accuracy at the end | best |
|---|---|---|---|---|---|
| cancho-gpu | 17.4 | 7,338 | 204 s | 0.8740 | 0.8743 |
| torch MPS float32 | 10.7 | 11,997 | 125 s | 0.8697 | 0.8699 |
| torch MPS float16 autocast | 11.1 | 11,512 | 130 s | 0.8674 | 0.8674 |

**M4's exit of 90% test accuracy is not met by any runtime**: without augmentation this network stops at 87%, and torch gets the same. The three agree on the accuracy (0.874 / 0.870 / 0.867) and their test losses cross each other from epoch to epoch as ReLU networks' do, so the learning is equivalent; the time to the final accuracy is 1.6x torch's, as in section 18. Reaching 90% needs augmentation (random crops and flips) or a wider network, neither of which was built.

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
1. **A resident file larger than the session's byte pool killed the process with status 132 and no message** (a resident file is no longer limited by the pool: it goes up in chunks, section 24.3; a table still is) (the Qwen output matrix is 272,498,688 bytes in f16, the pool was 256 MiB). That is an input reaching a trap. `fileio.prefix` now refuses a file that does not fit what is left of the pool (`usage`, naming the file and its size; found by this work, so `tests/session.py --metal` has the case), and the Metal session's pool is **512 MiB** (`device/metal-session-main.cho`), the bound on the largest tensor a device session stages. The CPU session's pool is unchanged at 256 MiB, and the CPU session does not run the real model (its interpreter memory is 32 M numbers), only the tests.
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


## 21. LoRA on Qwen2.5-0.5B: the backward pass and the adapters (roadmap M7.2; measured)

`docs/roadmap-models.md` section 6 asked, as its second step, for the gradient of every LoRA adapter of Qwen2.5-0.5B on the matrix unit, equal to peft's autograd, and the adapter update. It is built and it meets its exits; the full 300-step curve comparison is M7.3 and is not here.
**Rank 16, alpha 32, on all seven linear layers of the 24 blocks** (q, k, v, o, gate, up, down: 8.8 M adapter parameters), embeddings (tied with the head), norms and biases frozen. The language gained nothing; the session gained one directive (`alias`, 21.4).

**What was built** (`examples/qwen/`): `qwen_ref.py` grew `lora_forward`, `lora_loss_grads` (the hand-written binary64 backward) and `fd_check` (finite differences); `qwen_train.py` writes the training session; `peft_reference.py` (peft 0.21 on transformers 5.2, float32 on the CPU: gradients, and AdamW steps); `compare_lora.py` (the exits); new kernels `rms_bwd`, `swiglu_bwd`, `heads_h`, `att_dsoft`, `att_dkv`, `ce_grad`, `gemm_lora`, `dx_lora`, `adamw_l`, `widen_h` and `ce_comb` with a third output (`lse`); generated `rope_bwd`. Tests: `tests/qwen.py` (gradients, optimiser and mutants, in the gate), `tests/session.py` (the alias).

### 21.1 Design
* **Adapters are per projection, stored fused.** `y = x W^T + (alpha/r)(x A^T) B^T`, A `[r, in]` initialised `U(-1/sqrt(in), 1/sqrt(in))` as peft does, B = 0. The converter fused q/k/v and gate/up, so a group's adapters are one `Acat [n_proj r, in]` and one block-diagonal `Bcat [out, n_proj r]`: **mathematically exactly the separate adapters** (the off-diagonal blocks start at 0 and the optimiser's gradient mask keeps them 0; a test requires them to be exactly 0 after training). The block-diagonal wastes 2 to 3 times the (tiny) adapter FLOPs of q/k/v and gate/up. `split_grads` gives back the per-projection matrices, which is what the comparison with peft uses.
* **The adapter product is accumulated into the base product's fragment** (`gemm_lora`: a second 16-wide `mma` loop after the `k` loop; `dx_lora` for the input gradient `dy W + dmid A`), so there is no extra pass over `[4096, N]` and no epilogue (a fragment can only be accumulated into, stored or carried). `alpha/r` is folded into the f16 copy of B (`Bs = s B`, exact for s = 2): then `dmid = dy Bs`, `dA = dmid^T x` with no factor, and `dB = s dy^T mid` takes its `s` in the optimiser's gradient scale. `adamw_l` writes `f16(s w)` as B's copy.
* **Backward kernels.** RMSNorm: `dx = rs u - x rs^3 mean(u x)` + the residual gradient (no gain gradient: frozen). RoPE: one generated kernel does the inverse rotation of all 14 query heads and 2 key heads and casts the value gradients, into the f16 fused `dqkv`. Attention: `dP = dO V^T` is `att_s` on the heads-major `dO` (`heads_h`), `att_dsoft`, `dQ = dS K` is `att_o`, `dK`/`dV`: see below. SwiGLU: `silu'(g) = s + g s (1-s)`. Loss: `ce_comb` also writes the row's log-sum-exp and `ce_grad` (same grid as `ce_stats`) writes `(exp(logit - lse) - onehot) w` as f16 over the 152,064 padded columns (`w` = loss scale, 0 for the last position of a sequence); the head's `dx = dl E` is `gemm_mma_nn` on the frozen matrix (K = 152,064).
* **GQA backward: a product, measured against a sum kernel.** The 7 query heads of a group are contiguous in the heads-major layout, so `dK_g = sum over 7 t consecutive rows of ds^T q` is **one `mma_tn` reduction over 3,584 rows** (`att_dkv`, offsets `(b nh + g 7) t + c bk`, affine): the group sum costs no kernel and no per-head buffer. The alternative (a per-query-head product, then a kernel summing 7 heads) measured, isolated, on the real shape: product 0.93 ms (`bm 64`), **0.81 ms (`bm 32`, used)**, 0.96 (`bm 16`); per-head 0.63 + sum 0.03 = 0.66 ms. The sum kernel is 18% faster per kernel and would save about 7 ms of a 890 ms step (0.8%) at the price of a 14-head f32 buffer and a launch; the product was kept for that reason.
* **Loss scale.** Gradients of the logits are `(p - onehot) w` with `w` = 16 (the default), divided out with the count of positions (4,088) by the optimiser. Measured (all 336 matrices, random B): scales 1, 16, 64 give the same cosines (0.99985) and errors; **256 overflows f16 somewhere in the backward (inf), 512 and 1024 give NaN/zero cosines**. There is no dynamic scaling or overflow check: a scale above 64 on this batch is wrong, silently.
* **Data.** The embeddings of a step are an f16 `table` prepared by the host (7.3 MB a step, read straight to the device), widened by `widen_h`; targets are an f32 table; the learning-rate schedule is a `table` of AdamW's two per-step numbers.

### 21.2 Exits (Apple M4 Max; real weights, the fixed batch of 8 x 512; `python3 examples/qwen/compare_lora.py WORK`)
Gradients against peft's float32 autograd, 24 layers x 7 projections x (A, B). peft's own initial state has B = 0, where **every A gradient is exactly zero** (ours too), so only B's are tested there; the second scenario draws B from N(0, 0.02^2) so that all 336 matrices have non-zero gradients. Smallest cosine / largest relative norm error over the 24 layers:

| projection | A cos | A rel | B cos | B rel |
|---|---|---|---|---|
| q | 0.999862 | 1.67e-2 | 0.999858 | 1.70e-2 |
| k | 0.999929 | 1.24e-2 | 0.999920 | 1.39e-2 |
| v | 0.999966 | 8.54e-3 | 0.999945 | 1.06e-2 |
| o | 0.999948 | 1.02e-2 | 0.999945 | 1.06e-2 |
| gate | 0.999932 | 1.18e-2 | 0.999924 | 1.24e-2 |
| up | 0.999951 | 9.94e-3 | 0.999950 | 1.01e-2 |
| down | 0.999941 | 1.09e-2 | 0.999941 | 1.09e-2 |

**Exit (cosine at least 0.995): met by a factor of 40 in the margin: 336 of 336 matrices at 0.99986 or better**; with B = 0 the 168 B matrices are at 0.99984 (largest relative error 1.8e-2). The relative error is 0.4 to 1.7% and falls with depth (0.35% to 0.6% in layers 16 to 23, 1% to 1.7% in layers 0 to 7), as in M7.1: the f16 operands are coarsest where Qwen's outlier channels are. This is the f16-rounding noise of section 18, not a bug (the same kernels equal the float64 specification to 2e-3 on the test model). The binary64 specification on the real weights against peft: smallest cosine 0.999985, largest relative error 6.2e-3.
**Loss** at B = 0: 2.463544 (M7.1's value exactly; transformers 2.463422, +1.2e-4); with the random B 2.793587 against peft's 2.793492 (+9.5e-5).
**A short run, 40 AdamW steps** (lr 2e-4, 5-step warm-up then cosine, 8 different batches of 8 x 512 round and round, A as peft initialises it, B = 0, betas 0.9/0.999, no decay; peft float32 on the CPU: 214 s): the loss of both at every step agrees to **1.6e-4 at most (mean 5e-5)**, 1.9702 to 1.2554 in both (e.g. step 11 1.68053 / 1.68061, step 40 1.25537 / 1.25537). The adapters after the 40 steps: cosine against peft's at least 0.977 over the 336 matrices; every matrix but two is within 1.6% in relative norm; **layer 0's q B is 21% off (cosine 0.977) and layer 8's 2.3%**: Adam divides by the square root of the second moment, so elements whose true gradient is about zero (here output rows 95 to 127, the lowest-frequency rotary pairs of one head in layer 0) take steps of size lr whose sign is set by rounding noise. Not a defect of the gradients (their cosine is 0.9999) but a property of Adam that the f16 gradients make visible; the curves are not affected.

**Speed and memory** (4,096 tokens a step, forward + backward + AdamW, the loss included; quiet machine, medians; `compare_lora.py WORK --time --torch`): 

| runtime | ms a step | tokens a second | peak memory |
|---|---|---|---|
| **cancho-gpu (Metal session)** | **886.7** (runs 888.5, 886.7, 886.2) | **4,620** | **15.67 GB** (peak footprint) |
| torch MPS float16 autocast (re-measured, same shape) | 1,157 | 3,540 | 26.0 GB (driver) |
| torch MPS float32 | 1,220 | 3,358 | 27.2 GB |

**1.31 times torch's float16 autocast speed and 60% of its memory** (the roadmap's exit for M7.3 is parity or half the memory; this is M7.2's measurement, not that exit). The step is 8.4 TFLOP, so 9.5 TFLOPS. Timings on a busy machine were 1,040 to 1,100 ms; the table is from a quiet one.

### 21.3 Where the time goes
Matrix products are about 75% of a step at 12 to 13 TFLOPS: gate/up forward 140 ms and input gradient 148, head 88 and 100, down 73 and 66, q/k/v and o about 15 each way. The rest: attention 88 ms (`att_dkv` 23, `att_dsoft` 21, `att_o` and `att_s` 33 for both directions, `att_sm` 11), `swiglu_bwd` 15, `ce_stats`/`ce_grad` 20, the 16 to 48-wide adapter products about 40 (after `bm 16` tiles: they have too few threadgroups for the long reductions over 4,096 tokens; 4% of the step from that tuning), the optimiser 1.

### 21.4 `alias`: buffer reuse (memory)
Without reuse a step kept the output of every program of every layer: **30.3 GB** peak footprint (the backward's temporaries are as large as the forward's: 117 MB of f32 `dP` and 80 MB of `dact` per layer). New session directive **`alias P.OUT Q.OUT`**: `Q.OUT` lives in the buffer of `P.OUT` (chains: `alias A B`, `alias B C`: one buffer). It is safe only if nothing reads `P.OUT` once Q's program has begun, and **the session file's checks enforce that statically (rule `session`, with the line)**: both outputs of kernels, same type and size, P's program before Q's, every `link` from a member of the group goes to a program listed before Q's, an output joins a group once, and no member is `answer`ed, `save`d or `carry`ed (those are read after the programs ran). The CPU interpreter ignores it (separate memory); the Metal session allocates the root's buffer once (`device/metal-session-main.cho`).
`qwen_train.write` aliases, across layers, the forward's `qkv`, scores, attention output and the two projections' f32 outputs, and 16 temporaries of the backward (the `dP`, `dS`, `dact`, `dgu`, `dx` of every projection, the gradient stream...): **15.6 GB**, losses bit-identical. `tests/session.py` (CPU and Metal): a chain of six programs with three aliases answers the unaliased answer, 14 refusals (a reader at or after the overwriting program, backwards, itself, an input, an answered/saved/carried output, twice, one side, a size mismatch...), and on the device a chain of 24 programs with 32 MB outputs peaks at 340 MB aliased against 1,079 MB; four mutants of the checks, all killed by `--mutants`. A first version of the reader check also counted the new member's own readers and refused every legal chain: the test of the chain found it.

### 21.5 Bugs, limits
* **The Metal shim's tables were too small**: 1,024 modules, 1,024 functions, 4,096 buffers; the step has 1,353 programs (error `too many modules`). Now 4,096 / 4,096 / 16,384 (`shim/canchogpu-metal.mm`).
* A gradient test at peft's initial state tests nothing for A (zero); the checks use a random B too, and `tests/qwen.py` asserts that A's gradients are exactly 0 at B = 0.
* The constant on the left of an operator is refused (`1.0 - s`); `s - 1.0` is the form. A skinny tile on the CUDA schedule needs its own line (the CPU tests run `--target nvidia-ada`).
* Weights are resident twice (the forward and backward programs each upload their copies: +1.3 GB; there is no way to link a resident). No causal skipping (the upper triangle of `dP`, `dS`, `dK`, `dV` is computed and masked), `t` = 512 only, 8 sequences, one machine, nothing run on NVIDIA. The static loss scale (above). Block-diagonal waste. The run is 40 steps of M7.2, not M7.3's curve. peft ran in float32 on the CPU, torch's timings are MPS.


## 22. A 300-step fine-tuning run against peft (roadmap M7.3; measured)

`docs/roadmap-models.md` section 6 asked, as its third step, for a fine-tuning run of 300 steps on a real corpus, the same batches and the same initial adapters in cancho-gpu and in peft, the loss curves equal and the held-out loss falling, with ms a step and memory against torch. It is built. **Against peft in float32 the curves are equal (largest difference of the training loss 1.2e-3, mean 5e-5; of the held-out loss 6e-5), the held-out loss falls from 2.4522 to 1.8699, the step is 889 ms against torch MPS's 1,142 (float16 autocast) and 1,238 (float32), and the memory is 16.1 GB against 25.9 to 29.9 (59% to 54%: not under half).** The exit is met by its speed clause, not by its memory clause (22.3). torch's float16 autocast, run as the baseline of the roadmap runs it, does *not* produce the same curve (22.2).

### 22.1 What was run
`examples/qwen/`: `finetune_data.py` (the corpus), `finetune.py` (`train`, `eval`, `compare`: cancho-gpu's side and the comparison), `peft_finetune.py` (peft on MPS, `--dtype float16|float32`, `--loss-scale`), `adam_noise.py` (22.4), `generate.py` (22.5); `qwen_train.write` gained `state`, `first_step` and `save_state` (below); `tests/qwen.py` gained a check of the handover.
* **Model and optimiser.** Qwen2.5-0.5B, rank-16 LoRA, alpha 32, on all seven linear layers of the 24 blocks (8.8 M parameters); A initialised as peft does (`qwen_ref.init_lora`, seed 7), B = 0, the same arrays for both runtimes. AdamW (betas 0.9/0.999, eps 1e-8, no decay), lr 2e-4, linear warm-up over 30 steps then cosine to 0 (one `lr_schedule` for both), 300 steps of 8 sequences of 512 (4,096 tokens, 1,228,800 in all, **no batch repeated**). cancho-gpu: f16 weights and operands, f32 accumulation, loss scale 16 (the M7.2 session); peft: transformers 5.2, peft 0.21, torch 2.12 on MPS, the model loaded as float16 with float32 adapters under `torch.autocast` (the roadmap's baseline), and as float32.
* **The corpus** (`finetune_data.py`, no download): the Markdown, Rust and cancho files of this machine's `cancho` and `cancho-gpu` checkouts (4,169 distinct contents, 80 MB; identical contents kept once, files over 100 KB left out), tokenized with Qwen's own tokenizer (`tokenizers`, from the Hugging Face cache) with the end-of-text token after each file, in the order of their content hash, packed into sequences of 512. **The split is by file name** (a base name hashing into 1 of 20 goes to the held-out side with all its copies: the same document in two checkouts, edited, is a near-duplicate and would measure memorisation). Train: the first 225 files of the train side (124 cancho, 54 Rust, 47 Markdown) make the 2,400 sequences; held-out: **32 sequences = 16,384 tokens from 7 files** (4 cancho, 2 Markdown, 1 Rust), 4 batches of 8. That is a narrow held-out set; it is what a 16 k-token measure of "unseen files of this kind" is. The tokens depend on the files the machine has, so they are not committed (sha1 of the int32 arrays: train `72f23a2d9f25`, held-out `fb2bb7a45830`); `corpus.txt` lists the files.
* **Five sessions of 60 steps.** A session's `table` is one file of at most 512 MiB (the byte pool, section 20), and a step's embeddings are 7.3 MB of f16, so a table holds 60 batches and the 300 distinct batches (2.2 GB) do not fit one. Each session of 60 saves the adapters **and Adam's two moments** with `save` / `checkpoint` (the `.wo`, `.mo` and `.vo` outputs of `adamw_l`) and the next one starts from them (`state`) with the step count (`first_step`: the bias correction and the learning-rate table count from it). The handover is **exact**: `tests/qwen.py` runs 6 steps as 3 + 3 and as 6 on the CPU interpreter and requires equal losses and adapters (they are, to 0.0), and two mutants (moments dropped; step count restarted) end 2.6e-3 and 7.9e-3 from the single session's adapters and 0.23 and 0.43 in loss. On the real model a 6-step run as 3 + 3 and as 6 gave bit-identical losses and adapters. A session start (weights up, kernels built, first request) is a few seconds and is not in the ms a step.
* **Held-out loss** every 30 steps. cancho-gpu: each checkpoint's adapters in a separate session after the run (the forward and backward programs with no optimiser, four requests of one batch each); peft: inline under `no_grad` (not in its step time). Both are the mean loss over the 4 batches.

### 22.2 The curves
Training loss (the loss of the batch the step trains on, before the step; batches differ in difficulty, so the curve is jagged by construction and only the columns are comparable):

| step | cancho-gpu | peft float32 | peft f16 autocast | peft f16 autocast, loss scale 1024 |
|---|---|---|---|---|
| 1 | 1.52868 | 1.52865 | 1.52896 | 1.52896 |
| 30 | 1.25509 | 1.25509 | 1.28860 | 1.25533 |
| 60 | 2.68567 | 2.68560 | 2.72444 | 2.68617 |
| 90 | 1.78401 | 1.78405 | 1.81635 | 1.78422 |
| 120 | 2.94940 | 2.94937 | 3.02508 | 2.94974 |
| 150 | 3.17197 | 3.17199 | 3.23906 | 3.17196 |
| 180 | 2.69026 | 2.69026 | 2.76878 | 2.69083 |
| 210 | 1.01507 | 1.01519 | 1.05001 | 1.01521 |
| 240 | 1.24932 | 1.24926 | 1.28005 | 1.24949 |
| 270 | 2.17968 | 2.17962 | 2.25287 | 2.17990 |
| 300 | 2.68905 | 2.68898 | 2.75562 | 2.68918 |

Held-out loss (the 32 sequences of unseen files):

| step | cancho-gpu | peft float32 | peft f16 autocast | peft f16 autocast, loss scale 1024 |
|---|---|---|---|---|
| 0 | 2.45223 | 2.45222 | 2.45214 | 2.45214 |
| 30 | 2.21175 | 2.21175 | 2.25219 | 2.21184 |
| 60 | 2.07885 | 2.07887 | 2.12128 | 2.07910 |
| 90 | 2.02850 | 2.02849 | 2.07565 | 2.02862 |
| 120 | 1.96763 | 1.96765 | 2.01687 | 1.96785 |
| 150 | 1.94124 | 1.94119 | 1.99472 | 1.94121 |
| 180 | 1.90546 | 1.90551 | 1.95844 | 1.90574 |
| 210 | 1.87807 | 1.87809 | 1.93683 | 1.87822 |
| 240 | 1.87588 | 1.87589 | 1.93218 | 1.87600 |
| 270 | 1.87050 | 1.87049 | 1.92422 | 1.87071 |
| 300 | **1.86992** | **1.86992** | 1.92317 | 1.87011 |

**The held-out loss falls from 2.4522 to 1.8699 (-0.582), monotonically at every checkpoint, in both runtimes.** Largest |difference| of the loss over the 300 steps (`finetune.py compare`):

| pair | training loss: largest (mean) | held-out: largest |
|---|---|---|
| cancho-gpu vs peft float32 | **1.2e-3 (5e-5)**, at step 242; three steps are above 2e-4 (209, 217, 242), the 99th percentile is 1.6e-4 | **6e-5** |
| cancho-gpu vs peft float16 autocast | 0.132 (0.052), step 139 | 0.059 |
| peft float16 autocast vs peft float32 | 0.132 (0.052) | 0.059 |
| cancho-gpu vs peft float16 autocast with a loss scale of 1024 | 1.9e-3 (2.9e-4) | 1.9e-4 |
| peft float32 vs the same with scale 1024 | 1.9e-3 (3.0e-4) | 1.9e-4 |

**torch's float16 autocast is the one that is off**, by as much as cancho-gpu is not: with no loss scaling its gradients are damaged (the adapters after 300 steps are at cosine 0.62 and a relative error of 0.9 from float32 peft's in the last layers, median over the 336 matrices 18%) and it learns less (1.923 against 1.870); the same run with the loss multiplied by 1024 before the backward pass and the gradients divided by it before the optimiser (the static scale the session has, 16) follows float32 to the table's last column. One run each. So "the curves are equal" holds against peft float32 and against scaled float16, and not against the unscaled autocast of the roadmap's baseline, which is a different (worse) training run; the speed baseline is unchanged by that.
cancho-gpu's adapters after the 300 steps against float32 peft's, per matrix: **smallest cosine 0.9936, median relative norm error 0.16%**, largest 11.3% (layer 0's q B, 22.4), then 1.8% (layer 8's q B), 1.1% (layer 0's k B and layer 1's q B), 0.9% (layer 0's o B).

### 22.3 Speed and memory, and the exit
Per-step wall time of the training request (forward, backward, AdamW, loss; the first request of each session, which compiles and warms, left out) and, for torch, of `zero_grad`, forward, backward and `step` with `torch.mps.synchronize()`; the held-out evaluations are not timed. Medians of 295 and 298 steps:

| runtime | ms a step | tokens a second | peak memory: footprint (driver) |
|---|---|---|---|
| **cancho-gpu (Metal session)** | **889.2** (mean 888.9, range 876.7 to 903.5) | **4,606** | **16.14 GB** (five sessions: 16.11 to 16.14) |
| torch MPS float16 autocast | 1,142.0 (5th to 95th percentile 1,138.7 to 1,146.7) | 3,587 | 27.48 GB (25.92) |
| torch MPS float32 | 1,237.9 (1,234.0 to 1,242.6) | 3,309 | 31.49 GB (29.90) |

(Footprint is `/usr/bin/time -l`'s peak memory footprint of the process, the same measure for all three; torch's "driver" is `torch.mps.driver_allocated_memory()` over its first 30 steps, the number of `roadmap-models.md`'s table. cancho-gpu's includes the 440 MB table of the session.) The machine was quiet for the GPU (the spread above is ±1.5%; the only other job was the CPU-only `tests/session.py --mutants` during the cancho-gpu run, and a pre-run of 6 steps gave 887.6 ms), so the rounds of section 18 were not interleaved. The M7.2 measurement (886.7 ms) is reproduced to 0.3%; the peak is 0.5 GB above M7.2's because of the table of 60 batches (M7.2 had 8).
**Exit: "within 1.2x of torch's step time OR the memory under half of torch's with equal curves".**
* **Step time: met, with room.** cancho-gpu takes 0.78 of torch float16 autocast's time (it is **1.28 times faster**) and 0.72 of float32's (1.39 times faster), and the curves are equal to float32 peft's.
* **Memory: not met.** 16.14 GB is 59% of torch float16 autocast's footprint (62% of its driver number) and 51% of float32's (54%). It is a 41 to 49% saving, not the half the roadmap hoped for; the activations of a step (about 10 GB) and the 2.5 GB logits dominate and the forward and backward programs still hold the weights twice (+1.3 GB, section 21.5).
The step is 8.4 TFLOP, so 9.4 TFLOPS.

### 22.4 The layer-0 q B (the weak spot of section 21.2), isolated
`adam_noise.py`: the 40-step run of section 21.2 again (lr 2e-4, 5-step warm-up, 8 batches round and round, here from this corpus), peft float32 on the CPU and the session, the gradient of four matrices recorded at **every step** in both (peft's by a hook before `opt.step()`, the session's by `save` after each step), then Adam in float64 fed those gradients. The loss agreed to 1.7e-4 at most over the 40 steps, and **the weak spot is back: layer 0's q B ends 20.1% from peft's in relative norm (cosine 0.980)**, layer 8's 2.4%, layer 12's 0.45%. Relative error of the final matrix:

| matrix | peft's own against float64 AdamW fed peft's gradients | the session's own against float64 AdamW fed the session's gradients | float64 AdamW on the session's gradients against on peft's | the session's against peft's |
|---|---|---|---|---|
| layer 0 q B | 1.5e-7 | 6.4e-6 | **0.201** | 0.201 (cos 0.980) |
| layer 8 q B | 1.4e-7 | 6.5e-6 | 0.024 | 0.024 |
| layer 12 q B | 1.4e-7 | 6.5e-6 | 0.0045 | 0.0045 |
| layer 0 q A | 1.6e-7 | 4.7e-7 | 1.2e-3 | 1.2e-3 |

* **It is not the optimiser, in either runtime.** peft's float32 Adam equals a float64 Adam on the same gradients to 1.5e-7 (so torch's Adam shows no noise of its own), and `adamw_l` equals one to 6.4e-6 on its own gradients. The whole 20% is reproduced by feeding the two *gradient sequences* to the same float64 Adam: **it is the difference between the two runtimes' gradients that Adam amplifies.**
* **Why those rows.** At step 1 (identical weights) the session's gradient of this matrix is 1.55e-2 from peft's in relative norm (cosine 0.99988: the f16 noise of section 21.2), with an rms of 3.1e-5. 168 of its 896 rows have a gradient under 1e-3 of the largest row's (rms 3.9e-7 at most): in those rows the mean relative error of the final matrix is 0.30 against 0.02 in the others, and for the eight worst rows (87, 95, 114, 119, 120, 121, 124, 127) the signal is 4e-8 to 1.2e-7 (rms over the run) while the session's gradient differs from peft's by 1.6e-7 to 7.7e-7: **noise 3 to 20 times the signal**. The error of an f16-operand product is set by the matrix's overall magnitude, not the row's, so a row whose true gradient is small gets an absolute error that is not; Adam's step is `lr m / (sqrt(v) + 1e-8)`, and 1e-7 to 1e-6 is 10 to 100 times its epsilon, so those rows move by about lr per step in a direction set by the noise. (Layer 8's q B has one such row and layer 12's none: 1.0e-6 of difference against a signal of 1.5e-4 and 1.8e-4.)
* **It does not matter for the loss.** Held-out loss (float32, MPS) of peft's adapters after the 40 steps: 2.28009; with layer 0's q B replaced by the session's: **2.28009 (+0.00000)**; with layers 0, 8 and 12's: 2.28009; with *every* matrix from the session: 2.28003 (-0.00006). The directions that are noise in those rows are directions the model does not use. After the 300 steps the same layer is 11.3% off and the held-out losses are equal to 6e-5 (22.2).
The run is also what the section 21 text guessed ("not a defect of the gradients but a property of Adam that the f16 gradients make visible"), now measured; what is **not** shown is that a wider gradient (an f32 `dy^T mid`) would remove it, which was not tried.

### 22.5 The fine-tuned model writes (generation)
`generate.py`: the 336 adapters of the 300-step run are merged into the f16 weights (`f16(W + (alpha/r) B A)` per projection, the fused q/k/v and gate/up as the converter wrote them), and 8 held-out prompts (the first 64 tokens of the 8 sequences of the first held-out batch) are continued by 32 tokens, greedily, by the M7.1 forward session with the window recomputed for each token (as `serve_gpt.py`): the 8 windows are the 8 sequences of a batch, the new token is the arg max the session answers at the last prompt position (the mask is causal, so what follows does not matter), and **each new token is a new session** (the request line cannot carry a 4,096 x 896 embedded batch; 0.9 s a token including the start, 29 s for the 32). transformers (float32, CPU) is given the same merged f16 values and generates the same way.
* **Against transformers on the same weights: 8 of 8 continuations are identical for all 32 tokens, for the base and for the merged model.**
* The merged model's held-out loss through the forward session is **1.86993** against 1.86992 for the unmerged adapters (and 2.45223 for the base): the merge to f16 loses nothing measurable.
* The continuations changed in the direction of the corpus. Prompt 0 (a source comment about overflow checks ending "the same gap `docs/"): the base continues `mandelbrot.cho` has.\n//\n// The `mandelbrot` function is...`, the fine-tuned model ``threads.md`'s own section 2.1 says\n// `cancho run --threads 1`'s is: the same program, the`` (a docs path of this repository's kind; the text was `overflow-cost.md`). Prompt 3 (a cancho function `name_seen` ending in `var at = 0`): the base continues `;\n var seen = 0;\n for i in 0..limit {` (Rust syntax, which cancho lacks), the fine-tuned model `;\n while at < len(name) {\n if name[at] != '.' {...` (cancho's `while` loop, `[]`-indexing and `{}` blocks). The share of the 256 generated tokens equal to the held-out text's is 11.3% for the base and 12.1% for the fine-tuned model: a handful of tokens, not significant, and a reminder that greedy continuation of 32 tokens is a weak measure of anything but style. The evidence beyond the loss number is that the text moved toward this corpus's idiom and that two implementations agree on every token of it.

### 22.6 Tests, bugs, limits
* **`tests/session.py --mutants`, run to the end with no pipe (the M7.2 run had only the 4 `alias` mutants alone): 34 of 34 mutants killed (the 30 earlier and the 4 of `alias`: an alias onto the buffer of an output a later program still reads, backwards, of an answered output, an output aliased twice), 145 checks `ok`, none failing, exit 0.** The first complete run reported one `FAIL`: the text of the mutant "a link from a program to itself is accepted" (`if sp >= dp {`) occurs twice in `src/session.cho` since `alias` added its own copy of that test, so the mutant was applied to the first occurrence and the harness (rightly) refused an ambiguous edit. The mutant now includes the line after it and is unique; this is a bug of the test, found only by running the whole thing.
* New check: the handover between sessions (`tests/qwen.py`, `restart_check`: 3 checks, equality and two mutants that must fail; in the gate).
* **Limits.** One run of each runtime (the 300-step curve is one seed, one data order; the timings are medians over 295 to 298 steps of one run). The held-out set is 7 files and 16 k tokens. cancho-gpu's held-out numbers come from separate sessions after the run, peft's from inline evaluations; both use the same 4 batches. peft's reference is MPS float32, not the CPU of section 21 (the 40-step run of 22.4 is the CPU one). The float16 autocast baseline was run as the roadmap's table runs it (no loss scaling) and then once with a loss scale of 1024, whose step time (1,178 ms median, 1,314 mean) is *not* reported as a speed: it ran at the same time as the CPU job of 22.4. A 60-step session limit (a 512 MiB table) means five session starts in a 300-step run; a longer run needs the embedding gather (M7.4's G1) or a larger pool. Generation recomputes the window and starts a session per token. Contexts above 512, other models, other machines, NVIDIA: not done.


## 23. Qwen2.5-1.5B with LoRA (roadmap M7.4, part A; measured)

`docs/roadmap-models.md` section 6 asked, as the first half of M7.4, for the same fine-tuning on Qwen2.5-1.5B (3.1 GB of f16 weights, activations "21 GB at most" at 4 x 512), and for the memory clause that M7.3 missed. Full fine-tuning of the 0.5B model (G1, the embedding gather and scatter-add) is part B, section 24.

### 23.1 Design (written before the numbers)
* **The model.** `config.json` of Qwen/Qwen2.5-1.5B (read, from the Hugging Face cache; nothing downloaded): 28 layers, hidden 1,536, **12 query heads over 2 key/value heads of 128 dimensions** (groups of 6; the 0.5B has 14 over 2 of 64, groups of 7), intermediate 8,960, vocabulary 151,936 (padded to 152,064 as before), tied embeddings, rope theta 1e6, rms eps 1e-6, biases on q/k/v. 1,543,569,408 matrix values (233 M of them the tied embedding) = 3.09 GB as f16; non-embedding 1.31 B.
* **What should change in the code: nothing but a tile.** `qwen_session.py` and `qwen_train.py` already read every dimension from the configuration (`qwen_ref.Config`, `from_hf`), the converter reads the checkpoint's config, and no kernel names 64 or 896 outside its schedule. The one risk, from section 20's tile measurements ("every 64 x 128 tile falls to 0.6 to 1.7 TFLOPS: the accumulator no longer fits the registers"), is `att_o`, whose output tile is `[bm, dh]` and was 64 x 64; with `dh` = 128 it is 64 x 128. `att_dkv` has the same `dh` columns at `bm 32`. **Prediction: `att_o` needs `bm 32` (a 32 x 128 accumulator) and everything else runs as it is.**
* **Sessions.** A step's embeddings are an f16 `table` of 4,096 x 1,536 x 2 B = 12.6 MB, so the 512 MiB table of a session holds 35 steps at most (60 for the 0.5B): runs are sessions of 30 handing the adapters and Adam's moments over (section 22.1).
* **Work.** A forward pass is 13.0 TFLOP (2 x 4,096 x 1.543 B for the products, 0.36 for the attention), a LoRA step about 26 TFLOP; at the 11 to 12 TFLOPS of section 20 that is 1.1 s and 2.3 to 2.4 s. torch's baseline is expected near 1.2x slower as in M7.3.
* **Memory.** The 0.5B step peaks at 15.6 GB with `alias`. Scaling the pieces (the f32 gate/up product 4,096 x 17,920 is 293 MB a layer, against 159 MB) gives about 2 times for the 1.5B, so **about 30 GB (measured: 27.98 GB, 23.4); torch's footprint is expected near 2 times its 0.5B 26 to 31 GB (measured: 40 to 47 GB, 23.5), so the half-of-torch clause is out of reach without work on where the memory goes (23.4: it was reached, narrowly)**.
* **Run.** The corpus, split, schedule (linear warm-up over a tenth of the steps, cosine, lr 2e-4), optimiser and initial adapters (`init_lora` seed 7) of section 22, **150 steps** (not 300: a step is 2.5 s and torch's 2.8 s; three torch runs plus ours is the budget) in 5 sessions of 30, held-out loss every 30 steps; peft on MPS in float32 and float16 autocast, the latter with a **loss scale of 1024** (section 22.2 showed the unscaled run is off) and, once, unscaled. The corpus is rebuilt from this machine's files: the held-out tokens are those of M7.3 (sha1 `fb2bb7a45830`), the training tokens are not (`4f21275189df` against `72f23a2d9f25`: the checkouts changed).

### 23.2 The forward pass (`python3 examples/qwen/compare_qwen.py WORK --torch --spec`, with `QWEN_MODEL` and `QWEN_WEIGHTS` naming the 1.5B)
`data.py` takes the model from `QWEN_MODEL` (default: the 0.5B) and the scripts take the converted weights from `QWEN_WEIGHTS`; the converter, the specification, both session writers and the comparison scripts needed nothing else (the 0.5B path and its tests are unchanged). The fixed batch of 8 x 512 (the committed tokens), transformers 5.2 float32 on the CPU:

| | cancho-gpu session | transformers float32 | exit |
|---|---|---|---|
| mean loss over the 4,088 positions with a next token | **2.237675** | 2.237670 | within 2e-3: **+5.2e-6** |
| arg max of the logits equal at | **99.93%** (3 of 4,096 differ) | | at least 99% |
| the 3 that differ: transformers' gap between its two largest logits | 0.0005, 0.0006, 0.0008 | | near-ties |
| the binary64 specification | loss 2.237743 (+7.3e-5), arg max equal at 100.00%; the session is -6.9e-5 from it | | |
| residual stream after block k, relative L2 | 1.7e-3 (k = 1), 2.4e-4 (2), 1.4e-4 to 1.5e-4 (3 to 14), 1.7e-4 to 2.9e-4 (20 to 26), 7.5e-4 (27), **8.3e-4 (28)** | | |

The error is set in the first block (1.7e-3, Qwen's outlier channels again) and falls to 1.4e-4; only the last two blocks, where the residual stream's largest values appear, raise it (relative to the largest value, 1.1e-2 at block 27).
**Speed** (4,096 tokens, loss included): cancho-gpu **1,116 ms** (median of 5 requests; first 1,881), **3,669 tokens a second**; torch MPS float32 1,258 ms (3,257), float16 autocast 1,188 ms (3,449): **1.13 and 1.06 times faster**; 13.0 TFLOP in 1.116 s is **11.7 TFLOPS**.
**The prediction held, with a measured cost of missing it.** With the 0.5B's `att_o` schedule (`bm 64`: a 64 x 128 accumulator) the forward took 1,314 ms, `att_o` alone 219 ms (7.7 ms a layer, 14 times what `att_s` takes for the same output); `bm 32` takes 0.63 ms a layer (17.6 ms in all) and `bm 16` 0.73, and the forward falls to 1,117 ms. `qwen_session.py` and `qwen_train.py` now choose `bm 32` when the head has more than 64 dimensions. Nothing else changed: the five products are 91% of the 1,117 ms of kernels (1,015 ms for 12.6 TFLOP: 12.4 TFLOPS).

### 23.3 The gradients and the step (`compare_lora.py WORK`, peft 0.21 float32 on the CPU)
Rank 16, alpha 32, all seven projections of the 28 blocks (18.5 M adapter parameters; 392 matrices). Smallest cosine / largest relative norm error over the 28 layers:

| projection | A cos | A rel | B cos | B rel |
|---|---|---|---|---|
| q | 0.999506 | 3.18e-2 | 0.999564 | 2.95e-2 |
| k | 0.999382 | 3.52e-2 | 0.999566 | 2.95e-2 |
| v | 0.999954 | 9.67e-3 | 0.999948 | 1.02e-2 |
| o | 0.999977 | 6.89e-3 | 0.999973 | 7.31e-3 |
| gate | 0.999962 | 8.67e-3 | 0.999964 | 8.52e-3 |
| up | 0.999958 | 9.28e-3 | 0.999968 | 7.99e-3 |
| down | 0.999974 | 7.22e-3 | 0.999973 | 7.34e-3 |

**Exit (cosine at least 0.995): 392 of 392 matrices at 0.99938 or better** (with B = 0, the 196 B matrices at 0.999507; every A gradient exactly 0 in both). The loss is peft's to +5.5e-6 (B = 0) and +6.4e-5 (random B). The error is concentrated in **layer 0** (3.5% relative; 0.3% to 0.9% from layer 1 on, 0.3% in the last layers) and in q and k, as the 0.5B's was (1.7%): the layer-0 residual stream holds the outlier channels, and the f16 operands round them. The mutants of the gradient (section 21) run on a model with 128-dimensional heads too (`tests/qwen.py`, `wide_head_check`: forward, all 14 adapter gradients on the CPU and the GPU and a rotation-sign mutant, killed).
**The step: 2,460 ms (3 runs 2,460.2, 2,457.3, 2,462.8), 1,665 tokens a second, 28.0 GB peak footprint**; the step is 26 TFLOP, so 10.6 TFLOPS. Where it goes (`CANCHOGPU_PROFILE=2`): the four large LoRA products and their input gradients are 1,548 ms (63%: gate/up forward 515 and input gradient 526, down 263 and 244), the head 150 and 186, the q/k/v and o products and their input gradients 206, the adapters' own narrow products about 100, attention 150 (`att_dkv` 52, `att_o` 34, `att_s` 31, `att_dsoft` 21, `att_sm` 11), `swiglu` 53, the cross-entropy 20, norms and residuals 20.

### 23.4 Where the memory goes, and what was done about it
`examples/qwen/session_memory.py SESSION` adds up what a session file declares (every program output, aliases sharing a buffer, every resident file) from the kernels' declared shapes. For the step above it gives **20.57 GB of outputs + 6.82 GB of resident declarations (3.64 GB of distinct files) = 27.4 GB, against the process's 27.98 GB measured**: the accounting is the measure. The outputs, per kind summed over the 28 layers: the f32 gate/up product 8.22 GB (`gu.y`, 293 MB a layer: it is read by the backward), the logits 2.49 (f32, 4,096 x 152,064), the SwiGLU output 2.06 (f16), the probabilities 1.41, the two f32 residual streams 1.41, the logits' gradient 1.25 (f16), the f16 RMSNorm outputs 0.71, the heads-major q, k, v 0.35 each and the attention output 0.35; the rest is under 0.15 each. So **the gate/up product is 40% of the activations and the logits 12%**; attention is small because the scores are aliased across layers (section 21.4).
Two changes, each held to the accounting before it was believed:
* **A resident file that two programs read is one buffer** (`device/metal-session-main.cho`; the weights of a layer are taken by its forward and by its backward, and the session uploaded each program's copy: section 21.5's "+1.3 GB" is 3.2 GB here). A read-only parameter whose file and size match one already given a buffer takes it; a carry's destination is never shared (it is written). Predicted -3.18 GB (6.82 declared - 3.64 distinct); **measured 27.98 -> 24.84 GB (-3.14)**, step time unchanged (2,455 ms). `tests/session.py --metal`: 12 programs reading one 32 MB file hold 675 MB, reading 12 copies 1,044 MB, with the same answers; and a program whose resident file is carried to leaves another program reading the same file unchanged (growing, 1.5 and 1.5, against the mutant that shares carried buffers: 2.0 and 2.0 and so on, which the check fails on).
* **The memory-lean step** (`qwen_train.write(..., lean=True)`, `--lean`): `swiglu_h2` writes the SwiGLU output and an f16 copy of the gate/up product, so the f32 product is needed only inside its layer and is aliased across layers (293 MB, not 8.2 GB), the backward reads the f16 copy (`swiglu_bwd16`), and the SwiGLU output the down projection's weight gradient needs is recomputed there from the copy (`swiglu_h16`, 0.7 ms a layer) instead of being kept for every layer. Predicted -5.8 GB (-7.9 +4.1 -2.0); **measured 24.84 -> 19.11 GB (-5.73)**, for +15 ms (2,470 ms a step, +0.6%). The price is that gate and up are rounded to f16 before the SiLU (they already were f16 operands of the products that follow): the gradients are unchanged at the table's precision (`compare_lora.py --lean`: 392 matrices, smallest cosine 0.999383, largest relative error 3.51e-2, against 0.999382 and 3.52e-2), and the 150-step run below uses it. `tests/qwen.py`: the lean gradients equal the specification's on the CPU and the GPU, and two mutants of its kernels are killed.
What is **not** done: the logits (2.49 + 1.25 GB) are still made whole, and the f16 copy of the gate/up product (4.1 GB) is still kept for every layer (recomputing the product would cost a third of the forward).

### 23.5 A 150-step fine-tuning run against peft (`finetune.py WORK train --steps 150 --segment 30 --lean`; `peft_finetune.py WORK --dtype ... --steps 150`)
The setup of section 22.1 with 150 steps in 5 sessions of 30, held-out loss every 30 steps; cancho-gpu's session is the lean one with the shared weights. Training loss (the loss of the batch the step trains on, before the step) and held-out loss (the 32 sequences of 7 unseen files, the same as M7.3's):

| step | cancho-gpu | peft float32 | peft f16 autocast, scale 1024 | peft f16 autocast, unscaled |
|---|---|---|---|---|
| 1 | 1.44162 | 1.44158 | 1.44144 | 1.44144 |
| 30 | 1.05672 | 1.05668 | 1.05689 | 1.07406 |
| 60 | 2.43601 | 2.43604 | 2.43653 | 2.46233 |
| 90 | 1.80558 | 1.80554 | 1.80546 | 1.82972 |
| 120 | 3.40720 | 3.40727 | 3.40723 | 3.45285 |
| 150 | 3.17892 | 3.17895 | 3.17864 | 3.21097 |
| held-out 0 | 2.24870 | 2.24869 | 2.24880 | 2.24880 |
| held-out 30 | 1.97617 | 1.97615 | 1.97640 | 1.99798 |
| held-out 60 | 1.86170 | 1.86167 | 1.86192 | 1.88434 |
| held-out 90 | 1.81625 | 1.81623 | 1.81636 | 1.84870 |
| held-out 120 | 1.77977 | 1.77976 | 1.77982 | 1.80743 |
| held-out 150 | **1.77148** | **1.77146** | 1.77166 | 1.79928 |

**The held-out loss falls from 2.2487 to 1.7715 (-0.477), monotonically at every checkpoint, in every runtime.** Largest |difference| over the 150 steps (`finetune.py compare`): **cancho-gpu against peft float32: training loss 3.7e-4 (mean 4e-5, step 80), held-out 3e-5**; against float16 autocast with the loss scale of 1024: 2.0e-3 (mean 2.6e-4), held-out 2.3e-4; **peft float16 autocast with no loss scale is again the one that is off** (training loss 0.059 from float32's, held-out 0.032: it ends at 1.799). The adapters after 150 steps against float32 peft's: smallest cosine 0.9870 (layer 0's q B), median relative norm error 0.21%, largest 16.2% (layer 0's q B), then 9.9% (layer 0's k B), 5.3% (layer 15's q B), 4.3% (layer 0's gate B): the weak spot of section 22.4 (Adam amplifying the two runtimes' gradient difference on near-zero rows); torch's scaled float16 has the same rows off (11.0% and 8.3% at layer 0's k and q B) from float32's. Not isolated again here.

| runtime | ms a step | tokens a second | peak memory: footprint (driver) |
|---|---|---|---|
| **cancho-gpu (Metal session), lean** | **2,467.3** (mean of 145 timed steps 2,469, range 2,459 to 2,496) | **1,660** | **19.38 GB** (all five sessions) |
| torch MPS float16 autocast, scale 1024 | 2,743.8 | 1,493 | 39.93 GB (37.75) |
| torch MPS float16 autocast, unscaled | 2,742.2 | 1,494 | 39.93 GB (37.75) |
| torch MPS float32 | 2,980.3 | 1,374 | 46.76 GB (45.51) |

The measures are those of section 22.3 (footprint: `/usr/bin/time -l`'s peak for every process; driver: `torch.mps.driver_allocated_memory()` over the first 30 steps); the machine ran nothing else for the GPU or the CPU during the four runs (5th to 95th percentile of the cancho-gpu steps 2,461 to 2,485 ms), one run of each.
**Exit "within 1.2x of torch's step time OR the memory under half of torch's with equal curves":**
* **Step time: met.** cancho-gpu takes 0.90 of torch float16 autocast's time (**1.11 times faster**) and 0.83 of float32's (**1.21 times**), with the curves equal to float32 peft's. The margin over torch is smaller than the 0.5B's (1.28 to 1.39 times): the 1.5B's attention is 128-wide and torch's products are as efficient at this size.
* **Memory: met this time, narrowly against float16 autocast.** 19.38 GB is **41.4% of torch float32's footprint and 48.5% of float16 autocast's**; by torch's *driver* numbers it is 42.6% and **51.3%**, so against float16 autocast the clause holds on the footprint measure of section 22.3 and not on the driver measure. Without the two changes of 23.4 it would not hold: 27.98 GB is 70% of float16 autocast's. M7.3's 0.5B figure (16.1 GB, 59%) stays as it was.

### 23.6 Tests, bugs, limits
* `tests/qwen.py` (in the gate): `wide_head_check` (heads of 128: forward, gradients, a mutant), the lean gradients and two lean-kernel mutants. `tests/session.py --metal`: `share_checks` (above). The gate does not run the Metal checks (they need a Mac): they were run here.
* **Found by this work.** (1) `att_o` at `bm 64` with 128-wide heads costs 14 times what it should (23.2): a throughput bug that no test sees, only the profile; it is now chosen by the head size. (2) A fragment cannot be staged (`error[type]: staged tile is a matrix fragment`): the first form of the lean step wrote the gate/up product as f16 from the product's own kernel; the copy had to come from `swiglu_h2`, after the f32 product. (3) The f32 gate/up product, the logits and the doubly uploaded weights were 8.2 + 2.5 + 3.2 GB of the 28; a plan that scaled the 0.5B's 15.6 GB by two (23.1: "about 30 GB") was right about the total (27.98) and wrong about what could be cut.
* **Limits.** One run of each runtime, 150 steps (not 300), one seed and one data order; the held-out set is 7 files and 16 k tokens; the training tokens of this corpus are not M7.3's (the checkouts changed), so the 0.5B and 1.5B curves are not the same data. torch's peak memory is the footprint of a process that also holds the evaluation's activations. `tests/qwen.py`'s lean checks are on a 2-layer model. The weights are resident once now but the forward and backward programs are still separate programs (1,500 programs a step). The f16 gate/up copy (4.1 GB) and the logits are the next memory. Contexts above 512, other machines and NVIDIA: not done.


## 24. Full fine-tuning of Qwen2.5-0.5B: the embedding table trained (roadmap M7.4, part B; measured)

`docs/roadmap-models.md` section 6 asked, as the second half of M7.4, for every parameter of the 0.5B model trained (14 bytes a parameter: 6.9 GB plus activations), which needs G1: the embedding lookup and the scatter-add of its gradient into the tied 151,936-row table. It is built, and a 40-step run **equals torch's float32 AdamW** (held-out loss within 9e-5 at every checkpoint). The compiler needed no change.

### 24.1 Design (written before the code): three ways to G1, chosen by what the kernel surface accepts
* **Index-valued load and atomics are not in the surface, measured.** A kernel that loads a row of the table at an id it has just loaded (`let t = load ids[i, 0 ; 1, 1]` and `load e[t, 0 ; 1, d]`) is refused by the compiler: `error[offset]: t is neither an index nor an integer constant`. Offsets are affine in the grid coordinates and loop counters, never data; and there is no atomic in the `.chx` grammar (nothing found in `src/` or `docs/`, section 13's reading; `docs/training.md`). So the roadmap's two options — atomics, or the host sorting each step's ids into segments (the kernels would still have to address a table row by a segment's data) — cannot be written; and the compiler is out of scope.
* **What the surface does accept: a one-hot matrix and the matrix unit.** Section 17 already noted that "past about 8 k tokens the one-hot product stops being cheap"; at 152 k columns it is a 4,096 x 152,064 f16 matrix `Y` (1.25 GB; a 1 where the column is the row's id, built by `onehot_h` with the `relu(1 - |j - id|)` selection of `ce_grad`, 5 ms), and **the lookup is the product `Y E`** (`gemm_mma_nn`, m 4,096, n 896, k 152,064: exactly the shape of the output layer's input gradient `dlogits E`, which the LoRA step already runs in 99 ms) and **the scatter-add of the gradient into the rows is `Y^T dx`** (`gemm_mma_tn`, m 152,064, n 896, k 4,096: exactly the shape of the output layer's weight gradient `dlogits^T h`). Both are exact for the one-hot operand (a 0 or a 1 in f16), accumulate in f32, repeated ids add (the lookup of a repeated token and the sum of its gradients are what a product does), and need no sort, no atomics, no host work. **Predicted cost: two products of the output layer's own size, about 100 ms and 85 ms of a step of about 1.4 s (15%), and 1.25 GB of `Y`, shared with the output layer's gradient by `alias` (the three are 4,096 x 152,064 f16 in turn).** The host-sorted form would cost nothing on the device and is not expressible; this one is measured below.
* **The rest of full fine-tuning is the LoRA step without the adapters**: `dW = dy^T x` is `gemm_mma_tn` for every weight (f16 operands, f32 accumulation), the gains' and biases' gradients are column sums (`rms_dg` then `colsum`, and `colsum_h` of the f16 fused q/k/v gradient: one threadgroup per column block, the rows summed by a product with a row of ones in f32), and AdamW (`adamw_f`, and `adamw_e` for the table, which adds its two gradients) runs on f32 masters with two f32 moments and writes the f16 copy the products read, carried into every program that reads it. The optimiser steps sit inside the backward, right after a layer's last use of the weight, so a layer's weight gradients share buffers across layers (`alias`).
* **Two limits of the runtime, found by the plan before the code.** The table's f32 master (152,064 x 896 x 4 = 545 MB) and its zero moments are resident files above the 512 MiB byte pool of section 20's bug 1, and their gradients are outputs above it to `save` (24.3). The honest options were a larger pool or chunks; chunks were built (the pool stays 512 MiB, a chunk 128 MiB).
* **Run.** The corpus and held-out set of section 23.5, 40 steps of 8 x 512 (no batch repeated), AdamW (0.9, 0.999, 1e-8, no weight decay), lr 2e-5, linear warm-up over 4 steps then cosine to 0, torch float32 on MPS from the **same starting weights** (the f16 values of the converted checkpoint, as float32: `torch_full.load_weights`), loss scale 16 in the session (as section 21). One session: a step's ids are an f32 `table` of 4,096 numbers, so there is no 512 MiB table of embeddings to hand over.

### 24.2 What was built (`examples/qwen/`)
`qwen_ref.py`: `full_loss_grads` (the binary64 hand-written backward of every parameter, the table's two parts added: `np.add.at` of the embedding gradient into the ids' rows) and `fd_check_full`; `qwen_full.py` (the session writer: lookup, forward, loss, backward with the optimiser, `save` of the trained weights as a weights directory, `grads` for the comparison); kernels `onehot_h`, `rms_dg`, `colsum`, `colsum_h`, `adamw_f`, `adamw_e`; `torch_full.py` (torch's side) and `finetune_full.py` (`train`, `eval`, `grads`, `compare`); in the runtime `device/metal-session-main.cho` (chunked resident upload and save), `shim/canchogpu-metal.mm` and its mock (`cgpu_download_at`). The checkpoint is a weights directory (`save` of each tensor to its own `.f16` / `.f32` file), so its forward pass is measured by the LoRA session with zero adapters (the held-out loss) and `qwen_session.write` reads it as it is.

### 24.3 The 512 MiB limits, lifted where they were hit
* **A resident file larger than the byte pool is uploaded in chunks** (`cgpu_upload_at`, 128 MiB at a time; the old refusal `error[usage] ... more than the session's byte pool holds` stays for tables, which are read whole). `tests/session.py --metal`: a 640 MB f32 file whose element k is k mod 8191 (a chunk is 4,096 mod 8191, so a misplaced chunk shows at its boundaries) comes up with the elements on both sides of both boundaries right; the mutant that uploads every chunk at offset 0 returns zeros there and fails the check. The previous test (a 600 MB file is refused) is replaced by it.
* **A `save` of an output larger than the pool was a trap** (status 132, no message: `devcodec.reserve` handed out a slice beyond the pool; found by the gradient comparison below, whose table gradient is 545 MB in f32). `cgpu_download_at` (new in the shim and its mock) reads it a chunk at a time; `tests/session.py --metal` saves a 640 MB output and reads it back whole, the probes and every 65,537th element equal. This is a defect of M7.1's session that nothing exercised: no earlier save was above 272 MB.

### 24.4 Exits
**The gradients** of all 170 tensors (24 blocks x 7 tensors + the final norm + the table) at step 1 against torch float32 autograd on the same batch (`finetune_full.py WORK grads`; the session built without buffer sharing so that every gradient survives; table gradient = the output layer's part plus the lookup's part). Smallest cosine and largest relative norm error over the layers:

| tensor | smallest cosine | largest relative error |
|---|---|---|
| table (tied; both parts) | 0.999997 | 2.35e-3 |
| final norm gain | 0.999998 | 2.11e-3 |
| ln1 (gain, 24) | 0.999902 (layer 4) | 3.05e-2 (layer 5) |
| ln2 (gain, 24) | 0.999406 (layer 3) | 6.04e-2 (layer 21) |
| wqkv | 0.999923 | 1.24e-2 |
| bqkv | 0.999962 | 9.09e-3 |
| wo | 0.999931 | 1.18e-2 |
| wgu | 0.999408 (layer 3) | 4.48e-2 (layer 21) |
| wd | 0.999792 | 2.05e-2 |

**170 of 170 tensors at cosine 0.99940 or better**, the table's at 0.999997: the lookup's scatter-add and the output layer's gradient together equal autograd's. (The binary64 specification equals finite differences for all 16 tensors of the test model to 4.5e-7, `tests/qwen.py`.) The gains of layers 3 and 21 and their gate/up matrices are the least equal (4% to 6%): the layers where Qwen2.5-0.5B's massive activations sit, as in sections 20 and 21.
**The run** (`finetune_full.py WORK train --steps 40 --lr 2e-5 --every 10`, `eval`, `torch_full.py WORK --steps 40 --lr 2e-5`, `compare`). Training loss (the loss of the batch the step trains on, before the step):

| step | cancho-gpu | torch float32 | difference |
|---|---|---|---|
| 1 | 1.52868 | 1.52865 | +0.00003 |
| 2 | 3.13157 | 3.12777 | +0.00380 |
| 3 | 3.01735 | 3.01494 | +0.00241 |
| 5 | 2.68914 | 2.68747 | +0.00167 |
| 10 | 2.16750 | 2.16733 | +0.00017 |
| 20 | 1.47437 | 1.47439 | -0.00002 |
| 30 | 1.12571 | 1.12595 | -0.00024 |
| 40 | 1.28143 | 1.28150 | -0.00007 |
| held-out 0 | 2.45223 | 2.45222 | +0.00001 |
| held-out 10 | 2.22771 | 2.22777 | -0.00005 |
| held-out 20 | 2.12334 | 2.12344 | -0.00009 |
| held-out 30 | 2.08439 | 2.08438 | +0.00001 |
| held-out 40 | **2.07058** | **2.07056** | +0.00002 |

**The held-out loss falls from 2.4522 to 2.0706 (-0.382) in both, to within 9e-5 at every checkpoint**; the training loss differs by 4.5e-4 on average over the 40 steps and by 3.8e-3 at most, **at step 2** (the first steps, where Adam's update is `lr` times the sign of the gradient for every element, including those whose gradient is rounding noise in one runtime or the other: the two runtimes' weights differ most just after them, and the difference then falls to under 8e-4 from step 6 on). The step-2 difference is a hypothesis about Adam, not isolated here.
**Speed and memory** (medians; torch under `/usr/bin/time -l` for its peak footprint, the driver number over its first 3 steps):

| runtime | ms a step | tokens a second | peak memory: footprint (driver) |
|---|---|---|---|
| **cancho-gpu (Metal session)** | **1,381.4** (5th to 95th percentile 1,378.8 to 1,392.1) | **2,965** | **21.03 GB** |
| torch MPS float32 | 1,450.8 (1,446.2 to 1,456.0) | 2,823 | 44.56 GB (41.37) |

cancho-gpu is **1.05 times faster** and holds **47% of torch's footprint** (51% of its driver number). Against the 0.5B LoRA step (889 ms, 16 GB) full fine-tuning is 55% slower and 5 GB larger. The weights, masters and moments are 6.9 GB; the rest, about 14 GB, is the activations and temporaries of section 21.4.

### 24.5 Where the time goes, and what G1 costs (`CANCHOGPU_PROFILE=2`, one step of 1,380 ms)
The products with the vocabulary: **the lookup `Y E` 99 ms** (the other `gemm_mma_nn` of that shape is the output layer's input gradient; the two together are 197.8 ms), **the scatter-add `Y^T dx` 84 ms** (the other `gemm_mma_tn` of that shape is the output layer's weight gradient; together 168.9 ms), the two `onehot_h` 10.6 ms and the table's AdamW 9.4 ms (a 545 MB master, two moments and two gradients: 5 GB of traffic): **G1 costs about 203 ms of the 1,380, 15%, the prediction (100 + 85 + 10 + 9)**, and the output layer's own head product 88 ms and its cross-entropy 20 ms are not in it. The other weight gradients, `dW = dy^T x`, are 134 + 74 + 23 + 17 ms (gate/up, down, q/k/v, o), as large as the forward products (131 + 71 + 16 + 13) — LoRA does not pay them. The 24 optimiser steps of the matrices take 11.9 ms (the layer's 8.7 M-parameter gate/up: 0.5 ms), the bias column sums 11.6 ms. A faster G1 would need a gather the surface does not have; the product could shrink only with fewer columns (a vocabulary restricted to the ids of the batch is a gather again).

### 24.6 Tests, bugs, limits
* `tests/qwen.py` (in the gate; 20 new checks): the full specification against finite differences (16 tensors, 4.5e-7); the kernels against numpy on the CPU interpreter and the GPU (the one-hot matrix with ids in every chunk and a repeated id; both column sums, with the sum of the first tile alone shown to be far away; `rms_dg`; `adamw_f`, `adamw_e`); 6 AdamW steps on all 16 tensors of a 2-layer model (losses 1.8e-4 from a float64 AdamW's on the specification, every tensor within 5% of its movement); **four mutants, all killed**: the one-hot row reading the first row's id, the table's second gradient dropped, the bias column sum over the first tile only, the gain's gradient adding the row's rsqrt.
* The bias comparison leaves out the low-frequency rotary pairs of the key bias: their gradient is 1e-6 to 1e-4 against 1e-2 for the others (rotating by a frequency that goes to 0 makes the bias invisible to the scores), and Adam moves them by rounding noise, the mechanism of section 22.4. An earlier comment of this work said the key bias had no gradient at all; it has one (2.8e-4 rms on the real model, torch's and ours equal to three digits) — the rotation breaks the invariance — and the comment was corrected.
* **Limits.** One run of each, 40 steps, one seed; lr 2e-5 (not tuned); torch float32 only (no float16 baseline for full fine-tuning); the weights start from the checkpoint's f16 values in both, not from its bf16 ones (117 values flush to zero, 1.4 M become f16 subnormals: section 20). The loss scale is 16 and static (section 21.1). The checkpoint is f16 for the matrices, so the saved model carries the f16 rounding of the masters; the masters were not saved (corrected: they and both moments are, since section 24.7, and a run in several sessions equals one session byte for byte). 0.5B only (corrected by section 24.9: the 1.5B does fit without the lean step, at 44.27 GB, with thin headroom; 7 GB of state for the 0.5B, 21.6 GB for the 1.5B). Contexts above 512, other machines and NVIDIA: not done. The gate runs these checks on the CPU interpreter; the Metal ones were run here.

### 24.7 Resumable across sessions (roadmap M7.4c; measured)
Section 24.6 left it open: "the masters are not saved (a longer run in several sessions would have to)". They are now, with both moments, and a chain of sessions equals one session **byte for byte**.
* **What was built.** `qwen_full.write(..., save_state=DIR)` makes the request `checkpoint` also write, for every parameter, `NAME.w32` (the f32 master; the table's with its padding rows), `NAME.m` and `NAME.v` (the moments), all f32, beside the weights directory of `save`; `write(..., state=DIR, first_step=K)` starts from them (the masters and moments are the optimiser programs' resident files, the bias correction and the learning-rate table count from step `K+1`) and its `wdir` is the previous `save` directory, whose f16 matrices are the roundings of those masters. A missing state file is a refusal naming it (`error[usage]`, never a silent zero moment). `finetune_full.py train --segment N` runs the steps in sessions of `N` (each starts from the last one's directory; the older one is deleted), and `--resume` continues an interrupted run from the last finished segment (`WORK/full_run.json`). The state is 12 bytes a parameter on disk (5.9 GB for the 0.5B model), the checkpoint 1 GB.
* **The test** (`tests/qwen.py`, `full_restart_check`, in the gate): a 2-layer model, 6 AdamW steps with a warm-up and cosine schedule, as 3 + 3 sessions against one of 6: the 6 losses, the 16 saved parameters and their 48 state files (master, both moments) are identical byte for byte, on the CPU interpreter and on the Metal session. **Three mutants, all killed** (on the interpreter; every one of the 64 files differs): the second session starts with zero moments (largest value difference 2.5e-2, losses 3.7e-2), with the step count restarted (1.6e-2, 2.2e-1), and with masters rounded to f16, which is what a restart from the f16 checkpoint alone would give (4.9e-4, losses 1.5e-4: small, and not zero).
* **On the real model.** 4 steps of Qwen2.5-0.5B as 2 + 2 sessions equal 4 in one (losses and weights, bit for bit, Metal). **300 steps as six sessions of 50 equal 300 steps in one session: all 300 training losses are identical and all 171 files of the step-300 checkpoint are identical byte for byte** (the checkpoints at steps 30 and 150 compared in one file each). The handover is exact on this machine's GPU; nothing in the argument makes that a property of Metal (the reductions' order is unspecified, `chx.md` section 5), so on another device the test would need a tolerance.

### 24.8 A 300-step full fine-tuning run against torch float32 (measured)
Setup of 24.1 with 300 steps: the corpus and held-out set of 24.4 (300 batches of 8 x 512, no repeat: 1.23 M tokens; the 32 held-out sequences), AdamW (0.9, 0.999, 1e-8, no weight decay), **lr 2e-5** (not tuned), warm-up 30 steps then cosine to 0, loss scale 16, held-out loss every 30 steps; cancho-gpu as six sessions of 50 (24.7), torch float32 on MPS from the same f16-valued starting weights.

| step | cancho-gpu train | torch f32 train | difference | held-out cancho-gpu | held-out torch | difference |
|---|---|---|---|---|---|---|
| 0 | | | | 2.45223 | 2.45222 | +0.00001 |
| 1 | 1.52868 | 1.52865 | +0.00003 | | | |
| 2 | 3.14678 | 3.14435 | +0.00243 | | | |
| 30 | 1.18783 | 1.18701 | +0.00082 | 2.11419 | 2.11374 | +0.00045 |
| 60 | 2.60359 | 2.60412 | -0.00052 | 2.02589 | 2.02705 | -0.00116 |
| 90 | 1.77194 | 1.77215 | -0.00021 | 1.94564 | 1.94556 | +0.00008 |
| 120 | 3.59324 | 3.59278 | +0.00046 | 1.89300 | 1.89289 | +0.00011 |
| 150 | 3.21514 | 3.21502 | +0.00012 | 1.90289 | 1.90135 | +0.00154 |
| 180 | 2.42823 | 2.42671 | +0.00152 | 1.83330 | 1.82819 | +0.00511 |
| 210 | 0.90075 | 0.90829 | -0.00754 | 1.77509 | 1.77404 | +0.00104 |
| 240 | 1.19487 | 1.19658 | -0.00172 | 1.76778 | 1.76686 | +0.00092 |
| 270 | 2.00933 | 2.00938 | -0.00005 | 1.76158 | 1.76066 | +0.00092 |
| 300 | 2.71471 | 2.71438 | +0.00033 | **1.76018** | **1.75926** | +0.00092 |

**The held-out loss falls from 2.4522 to 1.7602 (-0.692) in cancho-gpu and to 1.7593 in torch; it differs by 5.1e-3 at most (step 180) over the eleven checkpoints and by 9.2e-4 at the end.** The training loss differs by 1.4e-3 on average over 300 steps (largest 2.9e-2, **at step 5**; means over steps 1-10: 7.0e-3, 11-100: 5.5e-4, 101-200: 1.8e-3, 201-300: 1.3e-3; the largest after step 10 is 2.1e-2). That is looser than the 40-step run's 9e-5 (24.4): **the two runtimes' weights drift apart as the steps accumulate** (f16 products with f32 accumulation against float32 everywhere), they do not agree to the digit, and what is shown is that the curves overlap to about 0.3% of the loss and end in the same place. The cause of the drift is not isolated (the chain of sessions is not it: 24.7 shows it equals one session exactly). Held-out loss rises between steps 120 and 150 in both runtimes (1.893 -> 1.903, 1.893 -> 1.901): it is the same data order and the learning rate at its largest, not a defect of either.

**Speed and memory.** cancho-gpu: **21.06 GB** peak footprint (all six sessions 21.05 to 21.06), **1,387 ms a step** (median of the first 190 steps; 5th percentile of all 1,383), 2,953 tokens a second. torch float32: **42.07 GB** footprint (driver 38.88 GB), and 1,455 ms a step at its 10th percentile, median 1,707. **The step times of this run were disturbed by other work on the machine** (from about step 198 on, the steps of both runtimes show bursts of 2.7 s and a base of 1.65 to 1.7 s; `ps` showed a Spotify renderer process at about 140% CPU and other sessions' Python jobs, which I did not control, and my own numpy work ran during cancho-gpu's steps 214 to 218). So the median over all 300 steps (cancho-gpu 1,389.5, torch 1,706.9) and the second run of cancho-gpu as one session (median 1,602, same cause) are **not** the speed; the undisturbed figures are cancho-gpu's 1,383 to 1,390 and torch's 1,455 (24.4: 1,381 and 1,451, measured on a quiet machine), **1.05 times faster**, and **50% of torch's footprint** (21.06 / 42.07). One run of each; the disturbance is why the torch median is not quoted as torch's speed.

### 24.9 Does the 1.5B full fine-tune fit? (measured, four steps)
* **State arithmetic.** 1,543,714,304 parameters (233 M of them the table), 14 bytes each (f32 master 4, moments 8, f16 copy 2) = **21.6 GB** of state. The session as written is not the lean one of 23.4 (the f32 gate/up product, 8.2 GB, is kept for every layer). `session_memory.py` on the session (`m75/mem15.py`): **43.66 GB of program outputs** (the state's carried outputs 18.5 GB of f32 and 3.1 of f16, the gate/up product 8.2, the logits 2.5, the SwiGLU output 2.1, the probabilities and the residual streams 1.4 each, the one-hot matrix 1.2, the table's two gradients 0.9 each). The same accounting on the 0.5B session gives 20.50 GB against a measured 21.03 (+0.53); **prediction 44.2 GB**. The machine has 64 GB; the default Metal limit on it is about three quarters of that, 48 GB **[read, not measured: `iogpu.wired_limit_mb` is 0, the default]**, so the prediction fits with 4 GB to spare.
* **Measured** (`finetune_full.py WORK train --steps 4 --lr 2e-5 --every 4 --segment 2` with `QWEN_WEIGHTS` the 1.5B's, two sessions of two steps, the handover of 24.7 on the 1.5B too): **peak footprint 44.27 GB** (the prediction within 0.1 GB), **4,191 ms a step** (4,203 and 4,179), 977 tokens a second, step-1 loss 1.44162 (equal to the LoRA run's step 1 in 23.5, as it must: the same forward pass), then 2.883, 2.771, 2.512. It runs, on an otherwise ordinary desktop, with about 20 GB of the 64 not used by it.
* **What this does not show.** Four steps are a feasibility check, not a fine-tuning run: no held-out loss, **no comparison with torch** (not run: torch float32 would hold 16 bytes a parameter, 24.7 GB, next to activations that were 34 GB for the 0.5B and would be about 80 GB at 1.5B by that ratio **[estimate]**, so it does not fit this machine), no gradient check against autograd at 1.5B. The step is 1.70 times the 1.5B LoRA step (2,467 ms) and 3.03 times the 0.5B full step (1,381). Headroom is thin: another process holding a few GB would make the allocation fail, and the lean step (23.4: -5.7 GB for the LoRA session) is not ported to `qwen_full.py`, which would give back about 5.7 GB **[estimate]**.

## 25. A 4-bit base (QLoRA) and a 7B model: a design with probes (roadmap M7.5; nothing built)

> **Status: design, written before any kernel.** Marks: **[measured]** (a probe or a run, with its command), **[read]** (from source or documents), **[estimate]** (arithmetic from measured numbers, to be replaced by a measurement), **[open]**. Nothing was downloaded and no kernel was added: the probes are five refused and two accepted `.chx` files (`experimental/m75/`, `experimental/m75/probes.sh`), a host quantiser (`examples/qwen/quant.py`) and sessions of existing kernels.

### 25.1 What is asked, and what a 7B model changes
Qwen2.5-7B **[read: its model card; the parameter count below reproduces its 7.6156 B]**: 28 layers, hidden 3,584, 28 query heads over 4 key/value heads of 128 (a group of 7, as the 0.5B has), SwiGLU of 18,944, vocabulary 152,064, **untied** output layer. Parameters by the formula of `qwen_ref.shapes` plus a separate head: 6.526 B in the blocks, 0.545 B in the embedding and 0.545 B in the head, 7.616 B in all **[measured: arithmetic, `m75/mem7.py`]**. Against the 1.5B (1.310 B in blocks) the blocks are 5.0 times larger.
Every shape passes the session writer's assertions (a 7B-shaped LoRA session was written with dummy files: heads of 128, `d`, `ff` and the fused widths 4,608 and 37,888 divide by 64 and 128, the vocabulary pads to the same 152,064) **[measured]**. What is not there: **the untied head** (`convert.py` writes one `emb.f16` that is the input table and the output layer, `qwen_session.py` and `qwen_ref.py` read it so; the 7B needs a `head.f16` and the session's `wlm` to be it) **[read]**, and a 7B reference to compare with **[open]**.

### 25.2 What the surface can express (probes; `experimental/m75/probes.sh`)
| probe | result [measured] |
|---|---|
| an `i8` tile staged to f32 or f16 (`i8_stage_f32.chx`) | refused: `error[type]: staged tile is quantised (I8); dequant it with its scales first` |
| arithmetic on an `i8` tile (`i8_arithmetic.chx`: `a * 0.0625`) | refused, `type`, the same advice |
| `stage i8` (`stage_i8.chx`) | refused, `syntax`: `i8 is not a staging dtype (f16, f32)` |
| an `i8` tile stored into an f32 tensor | refused, `type` |
| an f32 tile stored into an `i8` tensor (`f32_store_i8.chx`) | refused, `narrowing`: `use `convert``. **There is no `convert` in the grammar** (`chx.md` section 3): the message names an operation that does not exist. A kernel cannot produce `i8` at all: the encoder is host code |
| shifts, `%`, bit operations | **not in the grammar** [read: `chx.md` section 3; an `i8` is not arithmetic, so the nibble tricks (`b - 16 round(b / 16)`) cannot be reached either] |
| `dequant_fp4 q s g G` -> `stage f16` -> `mma` (`tests/chx/ported/gemm_fp4.chx`) | compiles on both targets and runs on the interpreter for G = 16, 32 and 64 (64 needs `bk 64`; G = 128 at `bm 64 bk 128` is refused, `budget`: 49,152 B against 32,768) |
| the same for the backward product `dx = dy W` (`gemm_mma_nn_fp4.chx`: `mma_nn` with the dequantised `W` stored `[out, in]`, the row scale on the out dimension) | compiles on both targets; against numpy `dy @ W_hat` on the interpreter: **3.1e-7 of a peak 2.67** |
| `dequant_fp4` -> `stage f16` -> `store` to an f16 tensor (`dequant_to_f16.chx`: the layer's weights into a scratch matrix) | compiles on both targets, runs on the interpreter |
| the host packing of `quant.py` (`pack_nvfp4`: two codes a byte, low nibble first; e4m3 scale bytes; an f32 per row) against the real `dequant_fp4` (`quant.py validate`, through `cancho-gpu run gemm_fp4`) | **16,384 of 16,384 weights equal** once rounded to f16, at G = 16, 32 and 64; the mutant with the two nibbles of a byte swapped differs at 15,186 of 16,384 and fails the check (`validate` is in the gate) |
| an `i8` file as a `resident` of a session (`resident dq.wq FILE i8`) | refused by the session: `error[session]: line 2: resident PROGRAM.PARAM FILE [f32\|f16]`. **No quantised weights can enter a session today** |

**So the only 4-bit unpack in the surface is `dequant_fp4`, whose codebook is fixed: E2M1 (0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives) times an E4M3 scale per G values times an f32 per row.** Plain int4 and NF4 are not expressible (no `i8` arithmetic, no table lookup by data: 24.1's `error[offset]`). The question is therefore only whether that format is good enough, and it is measured below. A generic `dequant_lut q s codebook G` atom would express NF4 and int4 at the price of a language change (the maintainer's call, as `kernels.md` section 4 says of `max`); 25.3 says it is not worth it.

### 25.3 The format: NVFP4 as `dequant_fp4` reads it, measured against the alternatives
`quant.py errors` (relative Frobenius error of the weights, layers 0, 14 and 27 of the 1.5B, all four matrices) and `quant.py fake` + the existing forward session (`finetune_full.heldout`: the held-out loss of section 23.5's 32 sequences with the adapters at zero, i.e. **the loss QLoRA starts from**; the weights are dequantised on the host to f16 and run through the unchanged forward pass; the embedding table and the norms stay f16/f32):

| format | bits a weight | weight error | held-out loss | against f16 |
|---|---|---|---|---|
| f16 (the base of sections 22 and 23) | 16 | 0 | 2.2487 | |
| **NVFP4, G = 16 (`dequant_fp4`)** | 4.52 | 0.0946 | 2.2979 | +0.049 |
| **NVFP4, G = 32** | 4.27 | 0.1017 | 2.2968 | +0.048 |
| NF4, block 64, f16 absmax (QLoRA's) | 4.25 | 0.0936 | 2.2997 | +0.051 |
| int4, group 64, f16 absmax | 4.25 | 0.1135 | 2.3453 | +0.097 |
| int8, group 64 | 8.25 | 0.0063 | 2.2491 | +0.0004 |
(E2M1 with an f16 scale per 64: error 0.1065; NVFP4 G = 64: 0.1068, 4.15 bits.) One evaluation of 16 k tokens each, one model: **NVFP4 loses what NF4 loses (0.049 against 0.051 nats) and half of what int4 does**; the weight error orders G = 16 and G = 32 the other way round from the loss (0.0946 < 0.1017 but 2.2979 > 2.2968): the weight error is a rough proxy and the two NVFP4 groups are not distinguished by this measurement. **Choice: NVFP4 with G = 32 (4.27 bits, 0.53 B a weight) or 16 (4.52 bits)**; the surface's `frag` decode path wants 16, the training tile (`bk 32`) takes 32. QLoRA recovers part of this loss by training the adapters on the quantised base; how much is the prototype's measurement (25.6), not known here.

### 25.4 Two ways to use it, and the one chosen
* **(a) Dequantise inside the product** (`gemm_fp4` and the `mma_nn` form above): every row-block of the grid dequantises the same weight tile again: with 2,048 rows and `bm 64`, 32 times; the cost per weight is ALU on the critical path of the matrix unit **[estimate, not measured: it cannot be, an `i8` file cannot be a resident]**.
* **(b) Dequantise a layer into an f16 scratch, then use the existing products unchanged.** One elementwise kernel (`dequant_to_f16.chx`, a grid over tiles), run before a layer's forward products and again before its backward products, writing the layer's matrices into buffers aliased across layers (`alias`, 21.4). Every kernel of sections 20 to 23, their tests and their tile choices stay as they are; the f16 scratch is one layer (233 M weights, 0.47 GB at 7B). Its cost is memory traffic: **a copy kernel over a 7B-sized matrix (18,944 x 3,584 f16, read and written: 271 MB) takes 0.69 ms on this GPU, 391 GB/s [measured: `m75/probe/bw.session`, `CANCHOGPU_PROFILE=2`, three runs 0.694, 0.710, 0.693 ms; the machine was shared]**. A dequantisation moves 0.5625 B read and 2 B written a weight against the copy's 4 B: a 7B block's 233 M weights take about 1.5 ms **[estimate: the copy's rate, ALU unmeasured]**, 56 times a step (28 layers, forward and backward): **about 85 ms of a step of about 6 s, 1.4%** **[estimate]**. (b) is chosen: it costs a kernel, a session flag and a converter, not a new tiling, and its overhead is small enough that (a)'s only advantage, no scratch, is not needed.

### 25.5 Memory for 7B (4 sequences of 512 = 2,048 tokens, rank 16 on every linear layer, the lean step of 23.4)
`session_memory.py` on a 7B-shaped lean LoRA session written with dummy files (`m75/mem7.py`) **[measured accounting]**; on the 1.5B lean session at 8 x 512 it gives 14.85 GB of outputs + 3.65 GB of distinct files = 18.5 GB against the measured 19.38 (+4.7%), and that offset is added below.

| item | f16 base | NVFP4 base (G = 16) |
|---|---|---|
| blocks | 13.05 GB | 3.67 GB (6.526 B x 0.5625 B) |
| embedding table + head (f16, frozen, untied) | 2.18 GB | 2.18 GB |
| norms, biases, adapters (40.4 M parameters: master, two moments, f16 copy, 14 B: 0.57 GB) | 0.1 GB + 0.57 | the same |
| activations and temporaries (program outputs, 4 x 512) | 14.70 GB | 14.70 GB |
| of which: f16 copy of gate/up 4.35, probabilities 1.64, residual streams 1.64, f32 logits 1.25, their gradient 0.62, the rest 5.2 | | |
| f16 scratch for one layer's dequantised weights | | +0.5 GB |
| accounting offset (1.5B calibration) | +0.9 | +0.9 |
| **total** | **about 31 GB** | **about 22 GB** |
At 8 x 512 the activations are 28.29 GB: **about 44.6 GB** (f16) and **about 36 GB** (NVFP4) **[estimate]**. So **a 7B LoRA fits this machine without quantisation at 4 x 512**: 4 bits buys 9 GB, which is the difference between 4 x 512 and 8 x 512 (or a longer context), **not** the possibility of 7B. The activations, not the weights, are the larger part from the 1.5B on (a 7B step keeps 14.7 GB of them for 28 layers). Metal's default limit is about 48 GB **[read, 24.9]**.

### 25.6 Time for 7B, from the measured 0.5B and 1.5B
A LoRA step costs `4 x tokens x P` FLOPs (`roadmap-models.md` section 6; P the parameters used in products, the head included). Measured: 0.5B, 889 ms for 8.1 TFLOP = **9.1 TFLOPS** (22.3); 1.5B, 2,467 ms for 25.3 TFLOP = **10.25 TFLOPS** (23.5). 7B at 2,048 tokens: 4 x 2,048 x 7.07 B (blocks and head; the lookup is a host table) = **57.9 TFLOP**, which at 9.1 to 10.25 TFLOPS is **5.65 to 6.4 s a step, 320 to 360 tokens a second** (8 x 512: twice that) **[estimate]**; the NVFP4 base adds the dequantisation of 25.4, about 85 ms, 5.7 to 6.5 s. The wider matrices (3,584 x 18,944) may run nearer 13 TFLOPS as the 0.5B's gate/up product does (22.3: 13.0), which would put it nearer 5 s; the estimate does not assume it. There is **no torch number for 7B on this machine**: torch's 1.5B at 4 x 512 took 1,388 ms (section 6), 4.6 times fewer FLOPs, so about 6.4 s for 7B in float16 autocast **[estimate]**, with torch's memory at 20.5 GB for the 1.5B scaling past 64 GB for the 7B (its weights alone are 15.2 GB, its activations scale like ours) **[estimate]**; bitsandbytes does not run on MPS. A comparison would need the 7B run in torch float16: a request below.

### 25.7 The prototype that needs no download: the cached 1.5B, quantised locally
Weights: `quant.py` packs the 1.5B's f16 blocks (the cached bf16 safetensors via `convert.py`) into three files per matrix, `q` (i8 bytes), `s` (e4m3 bytes) and `g` (f32 per row), 0.74 GB for the blocks against 2.62 GB. **Its exit, in order:**
1. **The base equals the specification's.** The kernel path's dequantised weights equal `quant.py`'s host dequantisation (`validate` above, at scale): the forward loss of the fixed batch within 2e-3 of the **fake-quant** forward (25.3: the existing session over the dequantised f16 files), which is itself held to transformers on the same weights as M7.1 was.
2. **The gradients equal autograd's on the dequantised base**: the 392 adapter gradients at cosine 0.995 or better against peft float32 loaded with the fake-quant weights (the reference code of 23.3 unchanged: it is a plain f16 model).
3. **A 150-step run** (the setting of 23.5) whose training loss differs from peft float32's on the fake-quant base by under 1e-3 on average, with the held-out loss reported against the f16 base's 1.7715 (from 25.3's 2.2979 start): **the number QLoRA is for**, how much of the 0.049 the adapters win back.
4. **ms a step and memory** against 23.5's f16 base (2,467 ms, 19.38 GB): predicted 2,480 to 2,500 ms (+0.5 to 1.3%: 28 layers x 2 x 46.8 M weights x 2.56 B / 391 GB/s = 17 ms, plus launches) and about 17 GB (the blocks 2.62 -> 0.74 GB, +0.19 GB scratch) **[estimates]**.
Exit clause: 1 to 3 pass and the step is within 1.2 times of the f16 base's time; both numbers stated whichever way they fall. It would say nothing of 7B beyond the shapes, which it shares with the 1.5B (heads of 128).

### 25.8 The list of gaps, in order of what blocks what
1. **`resident` takes only f32 and f16** (measured, 25.2): an `i8` flag, the upload of bytes (the 24.3 chunking exists), and `tests/session.py` checks with a mutant. Runtime and shim, not the compiler. **Blocks everything.**
2. **The dequantise kernel and its session wiring**: `dequant_to_f16.chx` exists as a probe (it compiles and runs on the interpreter); needed are its tile schedule on Metal, the programs per layer with `alias`, and its test against `quant.py` (bitwise, the packing is validated) with mutants (scale of the wrong group, nibble order).
3. **The converter/session for quantised bases**: `convert.py --quant nvfp4:G`, `qwen_session.write(..., quant=)` reading `q`/`s`/`g` in place of `w.f16` for the four matrices of a block; `qwen_ref`'s specification loaded from the dequantisation.
4. **The untied head** (7B only): `head.f16` in the converter, the manifest and the session writers; `qwen_ref.head` takes it. Small, and a test on a random untied 2-layer model.
5. **A 7B reference**: transformers + peft in float32 on the CPU needs about 30 GB of host memory next to nothing else; on MPS in float16, 15 GB plus activations. Neither exists here yet **[open]**.
6. **The f16 range of 7B** (roadmap G8): Qwen2.5-7B's massive activations are larger than the 0.5B's; the residual streams are f32 here, the product operands f16. Whether any overflows 65,504 is not known; the converter's check covers the weights only **[open]**; a measurement of the maximum on the forward pass is the first thing to do with the 7B files.
7. **Compiler/surface, not required by the plan above**: a `dequant_lut` atom (NF4/int4, 25.2; unneeded after 25.3), the error message that offers `convert` (25.2), G = 128 groups in the training tile (`budget`), an `i8` `store` from a computed tile (an on-device encoder). Listed, not asked for.

### 25.9 What needs the owner (nothing was done)
* **A download of Qwen2.5-7B** (`Qwen/Qwen2.5-7B`, bf16 safetensors, four shards, about 15.2 GB **[read, the model card; not verified]**) into the Hugging Face cache; its tokenizer is the one already used. Needed for 25.5's measurements and for every 7B number above; a base model, not the Instruct one, as the 0.5B and 1.5B runs used. Optional: its `config.json` alone (a few KB) would confirm 25.1's shapes first.
* A decision on **torch's float16 7B baseline** (15 GB of weights on MPS plus its activations; likely over what is comfortable on the shared machine) or acceptance of the estimate of 25.6.
* A decision on the **runtime change of gap 1** (a new `resident` flag in `session/` and the shim) before the prototype of 25.7, which needs nothing downloaded.

### 25.10 Limits of this section
One model (the 1.5B) and one evaluation (16 k tokens) for the format comparison; no QLoRA run, so the 0.049-nat cost of the 4-bit base is the starting cost and not the final one; every 7B memory and time figure is an estimate built on the 0.5B and 1.5B measurements (the accounting is within 5% of a measured footprint at 1.5B, 24.9 within 0.1 GB at full fine-tuning; the time assumes the 7B's matrices run at the rate of the smaller models' and that the dequantisation overhead of 25.4 holds); the bandwidth figure is a copy of one matrix on a shared machine; the CUDA path was not touched (`dequant_fp4` is lowered for it and held by goldens only).

## 26. A ResNet-18-class backbone for cancho-robot's vision (roadmap M9, V1; measured)

`docs/roadmap-models.md` section 8 chooses the job and the milestones; this section is V1: a ResNet-18 of the usual structure (7x7 stride-2 stem, 3x3 stride-2 max-pool, four stages of two residual blocks with stride-2 3x3 convolutions and 1x1 shortcuts, global average, a linear head) at a robot-sized input, **3 x 96 x 128**, forward, backward and an AdamW step in a resident session, held to a numpy specification and to torch. **11,181,632 parameters** (the torchvision count with a 10-class head and no bias), 0.444 GMAC a frame.

### 26.1 Design (written before the code)

**A stride is a permutation.** A window with a step is not in the language (26.2), but a permutation is a product with a 0/1 matrix, which `pool_fwd` already did. *Space to depth* (`s2d_fwd`) turns each 2x2 block of pixels into one pixel of 4c channels (channel group 2 py + px holds pixel (2Y+py, 2X+px)); then a 3x3 stride-2 convolution with padding 1 is a stride-1 **2x2** convolution of that image with taps (-1, 0), a 1x1 stride-2 convolution is a 1x1 convolution that reads only the first c of the 4c channels (the kernel takes `cx`, the matrix's width, and `cin`, the columns used), and the 7x7 stride-2 stem (padding 3) is a **4x4** stride-1 convolution with taps (-2 .. +1), on the space-to-depth image of the 4-channel padded frame (16 channels, a multiple of the matrix unit's atom).
The weights are held in that layout (`rn_session.embed`: 49 of the 64 taps of the 4x4 window of 2x2 blocks are weights, the others zeros; 9 of 16 for the 2x2), and the weight gradient is multiplied by a 0/1 mask (`wsum_mask`) so Adam never moves a zero. The count of parameters is the network's, not the layout's.
**Taps that reach two pixels past an edge** need no wider border: a flat read `p + (dy-oy) wp + (dx-ox)` that leaves the image lands on the neighbouring row's border cell (or the next image's border row, or a guard), all zero. Reach 2 on either side is absorbed by the existing one-pixel border; reach 3 would read a pixel. This is checked against an explicit zero-padded convolution of the image (26.3).
`convk` is `conv3` with the window (`kh`, `kw`) and the shift (`oy`, `ox`) as constants; `convk_dx`, `convk_dw` are its gradients; `s2d_bwd` is the transpose permutation. Guards are 192 rows (the widest shift is `2 wp + 2` = 134 at the stem).
**Max-pool without `max a b`.** In space-to-depth coordinates the nine pixels of a 3x3 stride-2 window (padding 1) are nine (row shift, channel group) pairs of one matrix: the same shifted windows a convolution reads. The maximum of the running `m` and the next candidate `v` is `s v - (s - 1) m`, `s = step(v - m)`: exact (one product is exactly 0), strict `>` keeps the first maximum in row-major order (torch's rule), and the index of the winner is updated by the same `s`. The padding is zero where torch's is -inf, which is the same because the input is a ReLU's output; a window that ties the padding with real zeros sends its gradient to a border cell, discarded, where torch sends it to a real zero whose ReLU gradient is 0. The backward is a gather, `[idx == k] = relu(d+1) - 2 relu(d) + relu(d-1)` with `d = idx - k`, row by row (`maxpool_bwd`), the border marked `idx = -1` so it never matches. Loops carry one tile, so the nine candidates are unrolled by `gen_rn_kernels.py`.
**The shortcut's BatchNorm has no ReLU**: `bn_apply_lin`, `bn_bwd_part_lin`, `bn_bwd_apply_lin` are generated from the ReLU kernels by removing it.
**The input** is the frame as a plain NHWC table (`pack_hw`, which also stores the pixels one column in by an offset instead of the shifting matrix `pack` used, which at 128 columns would have been 66 KB), then `s2d_fwd`. No host-side rearrangement.
`resnet_ref.py` is the network in binary64 numpy with the hand-written backward, `resnet_torch.py` the same network in torch (no torchvision, no download), `rn_session.py` the session writer, `gen_rn_kernels.py` the generator of the new kernels, `bench_rn.py` the launch count, timing and profile.

### 26.2 What the surface can and cannot say (probes, `tests/probes.py`, in the gate; 12 checks)

See `roadmap-models.md` 8.2 for the table. The refusals met, by tag: `syntax` (a strided window, a unary minus, `abs`), `offset` (a product of coordinates in an offset; a grid that does not divide), `type` (arithmetic or `rowsum` on a fragment: BatchNorm's statistics cannot be taken in the convolution's epilogue), `carry` (a loop carrying a maximum and its index), `shape` (a literal on the left of a tile), and, only at full size, `budget` twice (max-pool at 64 channels with 16 rows: 43,008 B; the global average at 512 channels with 15 rows: 33,792 B; limit 32,768). The probes that compute (`max_select`) are compared with numpy bit for bit.

### 26.3 Measured

**Correctness** (`tests/resnet.py`, in the gate; 44 checks on the CPU interpreter, none failing; the same checks on Metal, none failing, run once when the GPU was idle).
1. The specification equals torch float64 autograd: loss 3.08402765 on both, the worst gradient difference 2.3e-14 of the tensor's peak, on the 11,181,632-parameter network at 96x128, and on a 3,276-parameter one.
2. Every new kernel against numpy on f16-representable inputs: `convk`, `convk_dx`, `convk_dw` with `wsum_mask` in the four windows (3x3; the 2x2 and 1x1 of a stride-2 convolution; the 4x4 of the stem) against an explicit zero-padded convolution of the image and its adjoint, `s2d_fwd`/`s2d_bwd` (two block widths), `maxpool_fwd`/`maxpool_bwd` against the image-level max-pool **with ties** (index and gradient) and on random ReLU outputs, the non-square global average and packing, the guarded sum, and the three BatchNorm kernels without the ReLU. **Nine mutants** (a dropped column shift in each of the three convolution kernels, the mask read from the wrong columns, swapped column parities in both permutations, a wrong sign in the index update, a candidate that can never win, a wrong index match in the backward) are each caught.
3. The whole network at 64 x 96, widths (8,16,16,32), blocks (1,2,1,1), batch 8, 14 convolutions, 43 tensors: loss 2.77023 (CPU), 2.76995 (Metal), the binary16-rounding specification 2.76991, **torch float32 2.77017**; the gradients' cosine with the specification at least 0.9985 (CPU) and 0.9972 (Metal), relative error at most 6.4e-2 and 8.1e-2; with torch float32 the cosine at least 0.9893 and 0.9855 and the relative error at most 1.6e-1 and 1.7e-1. **As in section 18.2, that error is the binary16 floor** (a ReLU within the rounding of zero falls on either side), not a bug: the kernels are held tight one by one (2) and the network by cosine.
4. **The full network at 96 x 128, batch 8, on Metal** (run once, `rn_session` with `grads`): loss **2.78119** against the specification's 2.78114 and **torch float32's 2.78022** (3.5e-4 relative); gradient cosine with the specification at least 0.9943 (relative error 0.108 at most), with torch at least 0.9905 (0.140 at most).
5. 25 AdamW steps on a label the images carry: the loss falls 2.979 to 1.094 (CPU) and 1.051 (Metal), and the Metal session follows the CPU session after 1, 5, 10 and 25 steps within 5%.

**Launches** (`bench_rn.py launches`): **287 programs a step** against the M4 CNN's 186 (reproduced): optimiser 61, convolutions 79, BatchNorm 120, the rest 27. Proposals in `roadmap-models.md` 8.4.

### 26.4 What is not done, and the limits

* **GPU timing and the profile are TODO.** `bench_rn.py time` (interleaved rounds against torch MPS float32 and autocast) and `profile` exist and refuse to start while a Metal session or a finetune is alive; when they were to be run another agent's 7B finetune (`~/jobs/m75`) was running, so **no ms-a-step number is claimed**. A first profile attempt hung because the profile's stderr filled an unread pipe (fixed: it is written to a file). V2 is the milestone that measures.
* **The full-size network does not run on the CPU interpreter**: a session's float pool is 32 M (`session/main.cho`), and batch 8 at 96x128 needs more (`error[interp]`). The CPU checks are at reduced width and size; the full size was checked on Metal, once.
* The batch must make every stage's padded rows a multiple of 16 (at 96x128, a multiple of 8); widths multiples of 8 (a 32x8 or 8x64 fragment does not split into 2x2 warps on Metal: `warps_for` chooses the grid); `wb` (the space-to-depth column block) divides the width.
* The head is a classifier (the loss M4 had), not the regression head V3 needs; the input is 3-channel, not the `Camera` capability's greyscale; there is no evaluation-mode forward session (BatchNorm's running statistics are produced and dumped). **BatchNorm's variance is `E[z^2] - mean^2`**, one pass, as in section 18.
* The zero border and guards rely on a session's buffers starting zeroed, as section 18 does; `idx`'s guard (0) and the gradient guard (0) rely on it too.
* Not measured: speed, memory, any real or synthetic robot data, int8, the Raspberry Pi. CUDA: the new kernels compile for `nvidia-ada` and have goldens, and have never run on a GPU.
