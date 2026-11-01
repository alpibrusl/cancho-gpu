# A resident objective: keeping the data on the device between evaluations

> **Status: a design, nothing built. Written before the code; the gates in §5
> are written before the program they judge and must be able to fail.** It is
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

Metal never passes the CPU, and the kernel is not the reason: at 4096 x 128 the
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
* **Device side:** `cancho-gpu session` (CPU interpreter) and
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
(`opt.objective-timeout`; the README speaks of a clock "for its timeout", but a search of `lib/nm/objective.cho` and `docs/design.md` for the word finds nothing, so whether a per-evaluation timeout exists today is **[open]** and this may be a new tag). Authority
is unchanged: `exec` and a clock. It is a separate change in a separate repository
and starts with its own section in `docs/design.md`.

## 5. Gates (written before the code; each must be able to fail)

1. **The session agrees with the pipeline it replaces.** On the CPU interpreter,
   `session` answers equal the loss and gradient of `lsq_objective.py` through
   `cancho-gpu run` **bit for bit** (same interpreter, same f32 sums; the host's f64
   scale is the only new arithmetic and the gate recomputes it). At 16x3, 256x16 and
   4096x128. *Mutant:* the `link` to `sq.xt` dropped (the loss then differs).
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
   refusal after setup is an answer line `error: <tag>` and exit status 2; the payload
   is never echoed.
6. **Measured, then the doc is corrected.** `scripts/bench_train_size.py` extended
   with `--session`: per-evaluation ms at the §1 sizes, CPU session and Metal
   session, one machine, medians. **Predictions, to be tested, not claims:** per
   evaluation under 10 ms on Metal at 1024 x 64; the Metal session faster than the CPU
   session at 4096 x 128. If either is false, §1's explanation is wrong or incomplete
   and this document says so here.

## 6. Plan, each step ending at a number

1. **`opt --persistent`** in cancho-opt: design section, gates, then code. Number: the
   existing 12-evaluation fit gives the same answer through a persistent *CPU*
   program (a ten-line Python loop), and evaluations per second with the process
   start gone.
2. **`cancho-gpu session`** on the interpreter: the session file reader in `src/`
   (shared, `pub`, as `runcmd.cho` is), the loop, gates 1, 3 (CPU) and 5. Runs on
   Linux with no hardware. Number: ms per evaluation at the §1 sizes with the process
   start and the data parsing gone.
3. **`cancho-gpu-metal-device session`**, gates 2, 3 (Metal), 4 and 6. Number: the
   table in §1 again, with a third column.
4. **Decide** whether `training.md` step 4 goes on (Adam in `opt`, mini-batches) or
   stops: on the numbers.

Each step is a separate change. Nothing here touches the byte-identity of `emit`,
`check` or `run` with lex-gpu's Rust (`CLAUDE.md`): `session` is a new subcommand,
as `run` was.

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

* Whether `opt` already has a per-evaluation timeout to extend, or whether
  `opt.objective-timeout` is new (§4).
* Whether the host-side f64 `scale` should instead be a kernel constant once the
  surface can say a scalar times a tile.
* A transposing load or `matmul_tn` would remove `xt` from the resident files
  (`training.md` §4); it is a lex-gpu change.
