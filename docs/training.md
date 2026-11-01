# Training: whether cancho-gpu should, and what a first step costs

> **Status: a design, plus steps 1 to 3 of §6 built (step 2 on the CPU interpreter, step 3 on Metal): four backward kernels
> (`tests/chx/bwd_*.chx`), held to the Rust's output and to a reference
> that is itself checked against finite differences. Recommendation: do not
> add training to cancho-gpu yet; steps 2 and 3 are measured (§6), and step 4
> is the decision they feed.** Claims are marked **[measured]** (with
> the command), **[read]** (from a document or the source, not run) or
> **[open]**.

## 1. The question

cancho-gpu compiles typed tile kernels and, on top of the compiler, runs two
language models for inference (`inference.md`, `qwen35.md`, `serving.md`). It
has no backward pass, optimiser or loss **[read]**: `grep -i -E
"train|backward|grad"` over `docs/` and `README.md` finds nothing.

OpenNN (Artelnics) is a C++ library that trains dense, recurrent, convolutional
and transformer models on CPU and CUDA, with mini-batch SGD with momentum and
BFGS among its optimisers **[read, from opennn.net and its reference; versions
and benchmarks not checked]**. The question was whether cancho-gpu should grow
that, and whether the sibling projects (cancho-ml, cancho-opt) are a better home.

## 2. Why not inside cancho-gpu, as it stands

* **The oracle does not carry over.** The rule is byte-identity with lex-gpu's
  emitter (`CLAUDE.md`). A new backward kernel emits through the same path, so
  its *text* can still be held to the Rust, but "it computes the gradient" has
  no Rust oracle. It needs an external reference (plain Python for the first
  kernels, as `tests/kernels.py` does for attention; autograd for anything
  larger) and a tolerance, which is a different kind of claim from the rest of
  the repository.
* **Determinism.** A weight gradient sums over the batch. The order of that sum
  on a GPU decides the last bits, and nothing in the compiler pins it **[open]**.
  cancho-ml's contract (exact sums, a stored summation order) is the opposite of
  what a GPU training loop gives for free.
* **Priorities.** The 27B decodes at 25 tok/s against Ollama's 43
  (`benchmark.md`); CUDA has run only against a mock driver; split-KV
  attention is unbuilt. Training competes with those.

## 3. Where the pieces belong

| Piece | Home | Reason |
|---|---|---|
| Forward and backward kernels | cancho-gpu | it is a kernel compiler |
| Optimisers (SGD, Adam, L-BFGS) | cancho-opt | BFGS, Nelder-Mead and SPSA are there, each held to scipy or Qiskit |
| A small network as a model type | cancho-ml, later | cancho-ml's `docs/status.md` lists "neural networks, GPU" as *not planned in v1*; its design leaves room for "a later cancho-gpu backend" |

cancho-opt runs the objective as an external program: it writes the point on
standard input and reads the value (and, for BFGS, the gradient) from standard
output **[read, its README]**. A training step has that shape (parameters in,
loss and gradient out), so a cancho-gpu program can be the objective **once it can
be given the parameters. When this was first written it could not**: neither
`check --run` nor `cancho-gpu-device` takes input, only the interpreter's own
made-up tensors (`device.md` §3). `cancho-gpu run` ([`run.md`](run.md)) now does
on the CPU interpreter, and a fit with kernels as the objective is measured in §6
step 2. The device path still cannot **[read, device.md]**. The limits: BFGS holds an n-by-n matrix, so this is
for small, full-batch problems; and text on a pipe per evaluation will dominate
for anything with millions of parameters **[open]**.

## 4. What `.chx` can express of a backward pass

The surface has `matmul` (A B), `matmul_nt` (A B^T), `rowsum`, `rowmax`,
`rsqrt`, `sigmoid`, `softplus`, `exp`, `log`, and `+ - * /` (`src/parser.cho`,
the call and matmul cases **[read]**). Probes, compiled and interpreted with
`check ... --run` against the pinned compiler (`a4572ea`), both targets:

| Backward of | Written as | Result |
|---|---|---|
| `y = x w^T` w.r.t. `x` | `dx = matmul dy w` | **[measured]** compiles, runs |
| `y = x w^T` w.r.t. `w` | `dw = matmul_nt dyt xt`, the host supplying `dY^T` and `X^T` | **[measured]** compiles, runs; with `r=256` too |
| bias gradient | `db = matmul ones dy`, a row of ones | **[measured]** compiles, runs |
| sigmoid | `dz = dy * (s - &s * &s)` | **[measured]** compiles, runs |
| ReLU | needs `max` or a compare | **[measured]** refused: no such operation (`syntax`) |

The probes are now `tests/chx/bwd_dx.chx`, `bwd_dw.chx`, `bwd_db.chx` and
`bwd_sigmoid.chx`, and *the values are right* **[measured]**: `tests/kernels.py`
(in the gate) first checks its own formulas against central differences of
`L = sum(g * sigmoid(x w^T + b))` (dx, dw, db, 1e-7, on r=3 n=4 k=5), then holds
each kernel's `--run` summary to those formulas on both targets (6 runs,
r up to 256). Byte-identity with the Rust's CUDA C, Metal and `--run` summaries
holds for the 10 new cases (`golden/`, lex-gpu `851497d`). Mutants: `s - q`
changed to `s + q` in `bwd_sigmoid` and `matmul a b` to `matmul a a` in
`bwd_dx` both fail the check. What this does not cover: only the interpreter's
f32 sums, not a GPU's, and nothing about the *order* of a batch sum on a device
(§7).

Findings that shaped the table:

* **There is no transposed-A product** (`A^T B`). `dW = dY^T X` is expressible
  only by the host laying out transposed copies, an extra pass over memory the
  forward does not need. A `matmul_tn`, or a transposing load, is a lex-gpu
  change (the Rust is the oracle, `CLAUDE.md`), not a cancho-gpu one.
* **Tiles are consumed once.** `s - s * s` is refused (`moved-while-borrowed`);
  `s - &s * &s` is accepted. A backward pass that reuses forward activations
  will spend borrows the way `softmax.chx` does.
* **A constant has no shape of its own:** `1 - s` is refused
  (`a constant needs something to take its shape from`), hence `s - s*s` above.
* **ReLU has no derivative to write** because it has no forward. SiLU and
  softplus are expressible (`sigmoid` is the softplus derivative), and are what
  Qwen uses.
* **No column reduction** (`colsum`); the ones-row product above stands in for
  it at the cost of a matmul.
* **No atomics** found in `src/` or `docs/` **[read; the grammar was not read in
  full]**. Without them a batch reduction has to live inside one kernel's
  reduction, which bounds the batch a kernel can sum over **[open]**.

## 5. Comparison with OpenNN, honestly

No benchmark against OpenNN is possible today: cancho-gpu cannot train, and
OpenNN's strengths (a full training workflow, mixed precision, multi-GPU) are
the part cancho-gpu lacks. What can be compared fairly is inference of a small
dense network:

1. A small MLP forward pass in both, the same weights, outputs compared.
2. Latency of that forward pass, medians of several runs, in `bench/`'s style
   (`benchmark.md`: same machine, one runtime at a time, load guard).
3. Refusals: what each accepts that it should not, counted from
   `tests/reject/`-style fixtures.

Item 1 first: it needs only `matvec`-family kernels that exist. OpenNN's version
and build, its CUDA path and its own numbers are **[open]** until someone runs
it.

## 6. Plan, each step ending at a number

1. **Reference check. Done** (§4): the four kernels, their cases and goldens
   from the Rust, and `tests/kernels.py` checks that agree to 1e-5 on the
   interpreter and fail on a mutant.
2. **A model trained through cancho-opt. Done on the CPU interpreter**
   ([`run.md`](run.md) §6). `opt` side (cancho-opt `examples/train/`): a convex fit
   with BFGS (15 evaluations, within 7.6e-13 of a Newton solve), a 2-2-1 sigmoid
   network on XOR (**1 of 8 starts solves it; 7 of 8 report `converged` at a local
   minimum**), SPSA on a noisy loss, and Nelder-Mead on a ReLU network. Kernel
   side: `cancho-gpu run` (a new subcommand, so `--run` stays byte-identical with
   the Rust's) gives a kernel real inputs and prints a whole output, and a
   least-squares fit whose loss and gradient are `lsq_residual.chx` and
   `bwd_dw.chx` reaches the normal-equations solution in **12 BFGS evaluations**,
   within 1.1e-5 in `w`, at **28 to 36 ms an evaluation** over three runs (process starts
   dominate; §3's worry is confirmed for small problems and unmeasured for large).
3. **The same on a Metal GPU** through `cancho-gpu-metal-device`, and the cost of
   the pipe per evaluation measured. **Done on an Apple M4 Max** ([`run.md`](run.md) §7.5).
   The four backward kernels agree with `tests/kernels.py`'s references to 3.6e-7 of
   the peak (`scripts/metal-io.py`), and the fit reaches the same solution in the same
   12 BFGS evaluations. **94 to 125 ms an evaluation** over three runs, against 40 to 46 ms
   on the CPU: the device path is *slower* for a problem this size, two processes each
   compiling the kernel and opening the device per evaluation. §3's worry is confirmed
   for small problems on a GPU too. **Growing the problem does not flip it** [measured,
`python3 scripts/bench_train_size.py`, one machine, medians of 3, one evaluation, both
paths including the Python that writes the inputs]:

   | R x K | CPU interpreter | Metal device | CPU / Metal |
   |---|---|---|---|
   | 16 x 3 | 6 ms | 69 ms | 0.08 |
   | 256 x 16 | 15 ms | 80 ms | 0.18 |
   | 1024 x 64 | 137 ms | 206 ms | 0.66 |
   | 4096 x 128 | 1017 ms | 1122 ms | 0.91 |
   | 16384 x 256 | refused by both: an input over 16 MiB of text | | |

   The loss agrees to 1.5e-10 (relative) and the gradient to 5.4e-7 of its peak at every size.
   Metal closes on the CPU but does not pass it, and the reason is not the GPU: at 4096 x 128 the
   residual kernel alone is 0.33 s on the CPU and 0.375 s on Metal, about a megaflop of work, and
   the Python writing the inputs is 0.15 s more. The text interface (every number printed and parsed
   per evaluation, capped at 16 MiB) is the cost, so a GPU gains nothing from this protocol.
   Only a binary or resident interface (parameters kept on the device between evaluations) could
   show what the hardware does; that is the decision for step 4.
4. **Decide.** Only here: Adam in cancho-opt, `matmul_tn` in lex-gpu, a model
   type in cancho-ml, or stop.

Each step is a separate change. Steps 2 and 3 touch cancho-opt and need access
to it; this document touches only cancho-gpu.

## 7. Open

* The device programs' `--in` / `--out` on **CUDA hardware**: an NVIDIA GPU (it has
  only ever run against a mock driver). Metal is done (§6 step 3).
* A binary or device-resident objective interface: with text on a pipe the GPU cannot win
  up to the 16 MiB input cap (§6 step 3); what it would do without that is unmeasured.
* cancho-opt takes the program by the name given without searching `PATH`
  (found writing its examples); the objective of a future cancho-gpu driver is
  given as an absolute path.
* Whether the summation order of a batch gradient can be pinned on a GPU, and
  at what cost.
* A `matmul_tn` or transposing load (a lex-gpu change).
* ReLU (`max`) in the surface.
* Atomics, or another way to reduce across thread groups.
* OpenNN's current numbers, from running it rather than reading its pages.
