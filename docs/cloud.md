# CUDA on a real GPU, in Google Cloud

> **Status: written and checked against a stub `gcloud` (`tests/gcp.sh`, in the
> gate), and run against Google on a real NVIDIA L4 (`g2-standard-8`,
> europe-west1-b, and one run in us-central1-a) in runs on 2026-11-04 and 2026-11-05: NVRTC compiles
> every golden kernel, the device cases agree with the interpreter, the two NVFP4
> shuffle bugs were seen on the card and are fixed (below), and the GEMM was
> timed, and the lookup's cost was measured.** One card. Results are in "What the L4 measured" below and `docs/device.md` §6.

`scripts/gcp/cuda_test.sh` rents one NVIDIA GPU, runs the CUDA checks of
`docs/device.md` §6 on it, copies the results to `results/gcp/<time>-<gpu>/`
and deletes the VM. It is modelled on lex-gpu's `scripts/gcp/nvidia_test.sh`,
whose cost guard rails it keeps.

```sh
gcloud auth login                                  # once; the scripts cannot prompt
DRY_RUN=1 scripts/gcp/cuda_test.sh                 # read every command first; nothing runs
GCP_PROJECT=<project> scripts/gcp/cuda_test.sh     # L4, EU zones first
GCP_PROJECT=<project> BASELINES=1 scripts/gcp/cuda_test.sh   # + torch/peft LoRA baselines
GCP_PROJECT=<project> JOB='scripts/build-device.sh && build/cancho-gpu-device run tests/chx/ported/gemm_mma.chx m=256 n=256 k=128' scripts/gcp/cuda_test.sh
scripts/gcp/sweep.sh                               # anything left running? (DELETE=1 deletes)
```

## Steps

On the VM, `scripts/gcp/remote_cuda.sh` runs these against the **committed**
`HEAD` (`git archive`; uncommitted work is not tested). Each is logged to
`~/results/<step>.log` and is independent: a failure does not stop the others,
and a step whose prerequisite failed says `SKIP`. `summary.tsv` has the verdicts.

| step | what | answers |
|---|---|---|
| `machine` | `nvidia-smi`, driver, CUDA, CPU, machine type | what was this run on |
| `toolchain` | clang (>= 15), the compiler **installed from a release** (sha256-checked; built from source at the pin only as a fallback), `cancho build` | prerequisite of the rest |
| `gate` | `scripts/gate.sh`'s own lines, one at a time | does everything GPU-free pass on Linux |
| `nvrtc` | every `golden/*/*.cu` under the **real NVRTC**, `compute_89`, then `ptxas -v` | does the emitted CUDA compile, with per-case errors, registers and spills |
| `device_cases` | `tests/cases.txt` through `cancho-gpu-device` on the GPU against the interpreter | do the numbers agree at the program's threshold (2e-2 of the output's peak), per case |
| `gemm_bench` | our emitted `gemm_mma` vs cuBLAS (cupy) vs torch: 1024..8192 cubed incl. 3072 and 6144, and m=n=8192 with k=4096, 2048 and 4096^2 x 8192; clocks, power and throttle bits per shape at 100 ms; grid, smem, registers, spills, occupancy, waves | TFLOPS, as a fraction of **this card's measured** cuBLAS peak, and why it falls off |
| `shuffle_repro` | the two NVFP4 lookup-shuffle bugs the CUDA emulator found (PR #103), on the GPU: `scripts/gcp/shuffle_repro.txt` through `cancho-gpu-device`, 60 s and one process per case, `nvidia-smi` between cases | are they seen on hardware (**FAIL = a bug was seen**): 7 PASS / 3 MISMATCH / 40 TIMEOUT before PR #104, 50 PASS after |
| `fp4_perf` | the NVFP4 matvec kernels at K = 5152 and 17440 (the model's K as a control), `FP4_V[nibble]` against the old full-mask `__shfl_sync` made by a text edit of the emitted CUDA (`scripts/gcp/fp4_perf.py`), each variant in its own process with a timeout | achieved GB/s and old/new time (**not yet run**) |
| `sandbox_linux` | `tests/weightspace.py` verbose, then `audit_probe.py`: the network calls of its failing check, bare and sandboxed, with the audit events seen | why "a network call is refused" fails on Linux |
| `job` | `$JOB`, alone, after `machine` and `toolchain` | anything else |
| `baselines` | `BASELINES=1`: torch + peft LoRA, Qwen2.5-0.5B and 1.5B, 4x512 | the torch numbers on NVIDIA to put beside ours |

Notes on what each does and does not say:

* **`gate`** runs the Linux gate unit by unit (parsed from `gate.sh`, so it
  cannot drift). `tests/sampler.sh` is listed in `KNOWN_OPEN`: **the Linux
  sampler golden has a known open mismatch**, so its failure is reported
  `KNOWN-OPEN` and does not fail the step, and an unexpected pass is flagged so
  the entry gets removed. Nothing else is excused. The device path in it is the
  mock driver (`tests/device.sh`); that is plumbing, not a GPU.
* **`nvrtc`** needs no GPU to answer; it is the layer lex-gpu runs on a Mac in
  Docker. It uses the goldens (the gate holds them equal to what the compiler
  writes). `ptxas` states register counts and spills directly, which Metal
  never does.
* **`device_cases`** skips the `--run` lines of `cases.txt` and cases with no
  CUDA lowering recorded. Large cases whose **CPU reference** is too slow end as
  `TIMEOUT` (`CASE_TIMEOUT`, default 300 s), counted apart from `MISMATCH`
  (`error[device-mismatch]`) and `ERROR` (`error[device]`: driver, NVRTC or launch).
  The device program has no timing (`docs/device.md` §5), hence `gemm_bench`.
* **`gemm_bench`** compares output against torch's before believing a time. Both
  baselines are cuBLAS (`cupy.matmul`, `torch.matmul`; f16 output against our f32,
  which the `.chx` says), so the second is the check on the first. "Peak" is the
  best cuBLAS TFLOPS **measured in the same run**; clocks and power are sampled to
  `gemm-clocks.csv` because an L4 is power-capped and its sustained clock is not
  its boost clock. Quote ours as a fraction of that, not of the datasheet.
* **`baselines`** downloads public Hugging Face weights on the VM (allowed here:
  these runs carry no user data). `examples/qwen/torch_baseline.py` names `mps`;
  the step runs a `sed`-translated copy (`torch_baseline_cuda.py`, kept in the
  results) and refuses to run if any `mps` is left. Only the torch side exists
  on CUDA: cancho-gpu's training sessions are Metal-only, so this gives the
  denominator, not a comparison.

**`shuffle_repro` in detail.** The list is copied from the emulator branch's
`tests/cuda-emu-bugs.txt` (seven reproducers: six deadlocks, one wrong value) with
the emulator-only columns dropped, plus a sweep of the same families at
K = 16 mod 32 and `batch_fp4` counts that are not a multiple of 32 or 256, and
controls (K a multiple of 32) that must PASS. The emulator's own 22 findings
(21 deadlocks, 1 wrong value) were not recorded as a list, only as families and
`k` values, so this sweep is that grid, **not** those exact 22. Statuses: PASS,
MISMATCH, TIMEOUT (killed after 60 s), ERROR (`error[device]`), REFUSED (the
compiler's own rule; not a finding), WEDGED (the GPU stopped answering
`nvidia-smi` or logged an Xid after a case: recorded, the step stops). A PASS does
not clear a bug the CUDA guide calls undefined; a hang may leave the context
unusable, which is why every case is its own process.

**`gemm_bench` and the fall-off.** In the third run ours/cuBLAS was 0.97 at 2048^3,
0.89 at 4096^3, 0.62 at 8192^3 with cuBLAS steady at 58.9 TFLOPS, so the card was
not throttling. What the emitter writes (`golden/gemm_mma_*`, `tests/chx/ported/gemm_mma.chx`):
one 128x128 tile of `y` per block, 256 threads in 2x4 warps, `bk` = 64 (the schedule
line `threads 256 warps 2 4 bm 128 bn 128 bk 64`), tiles of `a` and `w` staged through
36,864 B of static shared memory, a single buffer (`__syncthreads`, load, `__syncthreads`,
mma; no `cp.async`, no double buffering), and `grid i over m/bm, j over n/bn` lowered to
`gid = blockIdx.x, gid2 = blockIdx.y`, i.e. consecutive blocks take different row tiles
of `a`. Hypothesis (to be read from the run's numbers, not assumed): a wave of ~120
resident blocks touches ~120 distinct `a` tiles of 128 x k f16, 2 MB each at k = 8192,
over the L4's 48 MB L2, so `a` streams from DRAM again for every column of blocks,
while cuBLAS orders blocks for L2 reuse. The step records what would confirm or refute it:
per-shape SM clock, power and throttle bits (timed loops of at least 3 s, sampled every
100 ms); grid, block, static smem, registers and spills (cupy and ptxas); blocks per SM,
warp occupancy and waves from the occupancy API (`ncu` is not installed and not
needed); the shapes with k = 4096 / 2048 / 8192 (a short k shrinks each block's reads);
and **`ours+swz`**, the same kernel with its block indices remapped to a grouped order
by a text edit inside `gemm_bench.py` (an experiment; the emitter is unchanged). If
`ours+swz` closes the gap at 8192, the cause is block order, and the fix is a
rasterisation key in the schedule (not expressible today: the schedule keys are
`threads warps bm bn bk`, and the grid order follows the `grid` line). Other things that
are expressible without a compiler change: a larger tile (`bm 128 bn 256 bk 32` is 30,720 B,
under the 48 KB static limit in `src/dialect.cho`; `bk 64` at 128x256 would be 55 KB and is
refused, the 99 KB opt-in needing dynamic shared memory the shim does not set) and a different
warp split; not expressible: software pipelining or `cp.async`.

**Measured in run 4 (one run; the L4 is a single card on a single day).** The L4 is
**power-capped** (72 W): in the 4096x4096x8192 shape ours ran at an SM clock of about
**735 MHz against 922 MHz for cuBLAS**, with throttle reason `0x4` (SW power cap), so a
part of the gap to cuBLAS is that ours draws more power per flop, not only its memory
traffic. The block-order experiment helps ours at large sizes: **8192^3 from 47% to
67% of the 60.6 TFLOPS measured peak, 6144^3 from 55% to 70%**, and slightly hurts at
2048 to 4096 (**79% to 69% at 4096^3**). That supports block order (L2 reuse) as a
cause of the large-size fall-off without proving it, and says a fixed grouped order is
not a free win: it would want to depend on the size. The first run of this step
(run 4) also printed `blocks/SM None`: the occupancy call returned nothing; it now falls
back from cupy's binding to the driver call through ctypes to a labelled estimate, and
prints the errors instead of `None`.

**`sandbox_linux` and the gate on the VM.** The test reported `FAIL a network call is
refused (None)` with the program run to completion. `audit_probe.py` tries the same
three calls (`socket.create_connection`, `urllib.request.urlopen`,
`importlib.import_module('_socket')`) bare and through `sandbox.run`, with a recording
audit hook installed ahead of the sandbox's own in the child, and prints which events
(`import`, `socket.getaddrinfo`, `socket.connect`, ...) arrived and what the sandbox
decided; the verdict is for the next run to read. The gate on the VM skips `tests/gcp.sh`
(the shipped tree is a `git archive`, no `.git`, so it exits 128), and says so:
`SKIPPED on VM` in the log and a `gate:tests/gcp.sh SKIPPED-ON-VM` row in `summary.tsv`.

**The emulator.** Another branch (`cuda-emu`) builds a CUDA-on-CPU mock driver.
When it lands, its entry point (`tests/cuda-emu.sh`, `EMU_ENTRY`) is run by
`remote_cuda.sh`'s `emulator` step beside the real results, and it will be in
`gate.sh`. **TODO (fill from the emulator's report):** what the emulator can
catch (races? out-of-bounds? numerics of the mma atoms? launch geometry?) and
what only this run can (real NVRTC acceptance, `ptxas` spills, warp-level
behaviour, timing, driver/context behaviour). Until then: the real GPU is the
only thing that settles anything the mock driver cannot run, which is every
kernel.

## Knobs

`cuda_test.sh`, environment (the header lists them all):

| | |
|---|---|
| `GCP_PROJECT` | billing project; else `gcloud config get-value project`. **Printed either way;** a taken one waits `CONFIRM_SECS` (10 s) unless `YES=1` |
| `GPU` | `l4` (default, `g2-standard-8`, Ada **sm_89**, exactly the lowering target), `a100`, `h100`. For the last two a loud note is printed: **the lowering is Ada's only**; NVRTC compiles for the card's arch but tiles, atoms and schedules are Ada's, so a pass is not an Ampere/Hopper result |
| `DRY_RUN=1` | print every `gcloud` command, run none (needs no login) |
| `SPOT=0` | on-demand instead of Spot |
| `MAX_RUN` | hard cap on the VM's life, default `90m` |
| `KEEP=1` | leave the VM; the delete command is printed |
| `ZONES` | zones in order; default EU first, then US, then Asia (no user data on these runs, which is why leaving the EU is allowed; pin a region by setting it) |
| `STEPS`, `BASELINES=1`, `JOB` | which steps; opt-in baselines; one command alone |
| `IMAGE_FAMILY` | boot from `cancho-gpu-l4` if it exists in the project (`build_image.sh`), else the stock Deep Learning image |
| `CANCHO_RELEASE` | the compiler to install: `latest` (default), a tag or sha, or `source`. The VM downloads `cancho-rust-<sha>-linux-x86_64.tar.gz` and its `.sha256` from the public `alpibrusl/cancho` releases (so on the VM, not via `gh` and scp), verifies the hash, and **falls back to building the pinned commit** if the download or the check fails. Rust is installed only on that fallback. `CANCHO_FLAVOUR=rust` (default: what CI builds, reads `$CLANG`) or `standalone` (built by cancho; reads `/usr/bin/clang`; untested here) |
| `CANCHO_REPO` | for the source fallback or `CANCHO_RELEASE=source`: git URL (default `https://github.com/alpibrusl/cancho`) **or a local checkout**, from which the pinned commit is shipped (for a private repository) |
| `OUT_DIR`, `CASE_TIMEOUT`, `JOBS`, `SIZES`, `DTYPES`, `BASE_MODELS`, `KNOWN_OPEN`, `TORCH_INDEX` | results dir; step 4/5/7 settings |

**Pinned or not.** `cancho.toml` pins a commit with no release, so the usual
release differs from the pin. The script prints pin and release rev, runs
project-mode `build`/`test` through a wrapper that adds `--ignore-compiler-rev`
only when they differ, and writes `compiler NOT-PINNED-ignore-compiler-rev` to
`summary.tsv` (and `compiler.txt`). Such a run is not a pinned-compiler result;
quote it as "compiler `<rev>`". Only `CANCHO_RELEASE=source` (or a release equal
to the pin) gives `PINNED`.

## Cost guard rails

* **Spot by default** (`--provisioning-model=SPOT`); exit status **75** means Google
  took the VM back: no result, retry.
* **`--max-run-duration` (`MAX_RUN`, 90m) and `--instance-termination-action DELETE`**:
  GCE deletes the VM when the cap expires, even if the script is killed.
* **A cleanup trap that finds the VM by name** in the project, not by a remembered
  zone, on exit, error, SIGINT, SIGTERM and SIGHUP. A signal ends the script after
  cleaning (lex-gpu's trap runs the cleanup and then lets the script continue), and the
  remote command is waited for in the background so the trap is not deferred behind it.
* **Region loop** with `instances bulk create --count 1`: one question per region,
  never more than one VM.
* **Label `purpose=cancho-gpu-test`** on everything; `scripts/gcp/sweep.sh` lists
  (exit 1) or, with `DELETE=1`, deletes leftovers, and never touches other VMs.
* **Results come home even when the run fails**; the project is printed before
  anything is created.

`tests/gcp.sh` (in the gate) checks these against a stub `gcloud`: the create is
Spot with the cap and DELETE, `SPOT=0` drops only Spot, the VM is deleted on a
normal exit, a failing run, preemption, SIGTERM and SIGHUP, a preemption exits 75,
results are copied after a failure, `DRY_RUN` calls no `gcloud`, a missing project
is refused. It cannot show that real `gcloud` accepts the flags.

## Cost per run: an estimate from memory, to be checked

**These are from memory of Google's price list, not looked up; check the
current prices (and your region's, EU is dearer than `us-central1`) before
relying on them.** Roughly, Spot L4 `g2-standard-8` is about $0.3 an hour and
on-demand about $0.85; A100 40 GB `a2-highgpu-1g` about $1.2 Spot and $3.7
on-demand; H100 `a3-highgpu-1g` several times that. The 150 GB `pd-ssd` adds a few
cents an hour; the image, if built, a dollar or two a month.

| run | wall time (guess; the first run will tell) | Spot L4 |
|---|---|---|
| default steps | 45-90 min (compiler build, the Linux gate, 180 cases with CPU references, the benchmark) | well under $1 |
| `BASELINES=1` | + 20-40 min and ~10 GB of downloads | about $1 |
| `MAX_RUN` hit | 1.5 h | about $0.5 |

The cap is the guard: even a stuck run costs at most `MAX_RUN` of the VM.

## The image: optional

`scripts/gcp/build_image.sh` builds `cancho-gpu-l4` on a Spot CPU VM (toolchain,
the compiler at the pin, the venv with torch and cupy, optionally the Qwen
weights with `WITH_BASELINE_LIBS=1`). It is **not needed for a first run**: the
compiler is a downloaded release binary (about 1 MB), so only clang and a few apt
packages remain (a minute); building it from source, the fallback, is minutes plus
Rust. Neither is measured yet. It
pays when the ~4 GB of wheels and ~4 GB of weights are downloaded at the GPU's
rate on every run, or when the scripts are being iterated on.

## How this differs from lex-gpu's scripts

| | lex-gpu | here | why |
|---|---|---|---|
| steps | `cargo test`, Ollama baseline, Qwen golden/profile sweeps | toolchain, gate, NVRTC, device cases, GEMM vs cuBLAS, opt-in baselines | cancho-gpu is a compiler + a device program, not a runtime |
| project | required | required, or gcloud's default **printed, with a pause** | billing goes there |
| `DRY_RUN` | no | yes | a login-free way to read what would run |
| signals | trap re-enters the script | trap cleans and exits 130/143/129 | a handled signal does not end bash |
| gate | the suite | `gate.sh` split into units, a known-open list | one failure must not hide the rest |
| compiler | Rust workspace builds itself | built at the pin from `CANCHO_REPO` (needs `clang`) | cancho's backend shells out to clang |
| image | needed (Ollama, 20 GB models) | optional | see above |
| tests | none | `tests/gcp.sh` in the gate | the guard rails are the part that costs money when wrong |

## First real-hardware finding (2026-11-04, L4, driver 580, CUDA 12.9, Ubuntu 24.04)

The `nvrtc` step, the first thing here to run against a real toolchain:
**246 of 246 golden kernels compile under NVRTC 12.9 for `compute_89`** and
assemble with `ptxas` for sm_89. **One spills:** `batch_fp4_b=16_n=128_k=4096`
uses **255 registers and spills 1172 B** (the next heaviest,
`batch_fp4_b=2_n=200_k=256`, uses 168 and spills nothing). That is the cap
ptxas can give a thread, so the kernel's register pressure, not the card, sets
its speed; tracked in the issue titled "CUDA: batch_fp4 kernel spills 1172 B at
255 registers on sm_89". Only compilation was measured: no kernel ran in that
run, because the toolchain step failed (below), so `device_cases` and
`gemm_bench` had no `cancho-gpu`.

That run also found two defects in these scripts, since fixed: the compiler
built from the source archive (no `.git`) did not know its commit and the
project refused it, so the source fallback now builds with `CANCHO_REV=<pin>`
(`summary.tsv`: `PINNED-by-CANCHO_REV`) and a release binary is used only if its
`--version` reports its rev; and the local driver, edited while running, read
garbage from its own file and skipped the final copy of the results, so its body
is now one function called on the last line and the results are also copied by
the cleanup trap. (`tests/gcp.sh` reproduces the second against the old script.)

**Second run (same day):** toolchain installed the release binary in 85 s
(`NOT-PINNED`, as designed) and `nvrtc` passed again; `gate`, `device_cases` and
`gemm_bench` failed on two VM-side defects, fixed since: cancho runs a plain
`clang` by name, which an image with only `clang-NN` lacks (the script now makes
`clang` resolve to a version >= 15 and prints `clang --version` into
`machine.txt`), and `cupy.RawModule` found the pip wheel's `mma.h` (no `crt/`)
instead of the toolkit's (`gemm_bench.py` now passes the toolkit include first).
Which gate units then fail for a real reason is for the next run to say.

Run 4 also showed `shuffle_repro` and `sandbox_linux` never ran: defined, never
dispatched by the main sequence. `tests/gcp.sh` now fails if any name in the default
`STEPS` or any `s_*` function is not run by `main` (and I watched it fail by deleting the
line), and `sandbox_linux` is dispatched last, after `shuffle_repro`, so it runs even if
that step wedged the GPU.

## What the L4 measured (2026-11-04 and 2026-11-05)

All on one NVIDIA L4 (Ada, `sm_89`, 58 SMs, `g2-standard-8`, europe-west1-b, driver 580,
CUDA 12.9). One card on two days: a measurement, not a distribution. The runs are in
`results/gcp/` of the `cancho-gpu-cloud` checkout (not in this repository).

**The NVFP4 shuffle bugs, before and after the fix (PR #104).**

| | run | `shuffle_repro` (50 cases) | `device_cases` | `nvrtc` |
|---|---|---|---|---|
| before | `20261104-212332-l4` | 7 PASS, 3 MISMATCH, 40 TIMEOUT; the step took 2413 s | 169 PASS, 0 MISMATCH, 12 ERROR of 181 | 246 of 246 compile |
| after (main `46670b5`) | `20261105-083519-l4` | **50 PASS**, 0 MISMATCH, 0 TIMEOUT; the step took 20 s | 176 PASS, 0 MISMATCH, 12 ERROR of 188 | 253 of 253 compile |

The 12 `device_cases` errors are the same before and after: the interpreter's
`the run's tensors do not fit the interpreter's memory` on the largest shapes (the CPU
reference, not the GPU), which is why that step is reported FAIL; the 7 added cases are
the former reproducers, now in `tests/cases.txt`. The 40 hangs were the full-mask
`__shfl_sync` in divergent code; the fix is a table index (`docs/device.md` 5b). This is
a check that the fixed form does not hang or misread on this card, not a proof (the
CUDA guide calls the old form undefined). What the table index costs against the shuffle is measured next.

**What the table index costs (`fp4_perf`, run `20261105-115058-l4`, us-central1-a, main `99ad9e7`).**
Old (`__shfl_sync`, made by a text edit of the emitted CUDA) against new (`FP4_V[nibble]`),
GB/s of weights read; the ratio is old time over new time, so below 1 the lookup is slower.
No variant hung, failed or disagreed with the other.

| case | lookups | old | new | old/new |
|---|---|---|---|---|
| `matvec_fp4_rows` n=17408 k=5120, n=5120 k=17408 (the model's K; control) | 0 | 224, 219 | 224, 219 | 1.000: the same program |
| `matvec_fp4_rows` n=17408 k=5152, n=5120 k=17440 | 1 | 67.4, 60.8 | 50.6, 48.3 | 0.75, 0.79 |
| `matvec_fp4_res` n=5120 k=17440 | 1 | 59.8 | 47.8 | 0.80 |
| `matvec_fp4_swiglu` n=17408 k=5152 | 2 | 64.5 | 50.0 | 0.78 |
| `matvec_fp4_batch` m=4 n=17408 k=5152 | 1 | 54.9 | 46.0 | 0.84 |
| `matvec_fp4_res_batch` m=4 n=5120 k=17440 | 1 | 51.6 | 43.2 | 0.84 |

At the model's own K the fix costs nothing (those kernels read through `fp4_pair` and have no
lookup). On the generic path, which a K that is an odd multiple of 32 takes, the table index
is **16 to 25% slower** than the shuffle it replaces, the serialised divergent `__constant__`
read; but the old form hung for K not a multiple of 32 and gave wrong values below a full
warp, so there it had no correct baseline. That path runs at about 50 GB/s against about
220 GB/s for `fp4_pair`, so the gap between the two paths is larger than the lookup's cost.
One card, one run, a text-edit variant of the old form.

Since this run the compiler takes `fp4_pair` for those K as well (docs/device.md 5b, "Why the generic path was about 4x slower"): the K = 5152 and 17440 rows above describe the compiler before it and are **not yet re-measured** on the L4.

**The GEMM against cuBLAS (`gemm_bench`, run `20261104-212332-l4`; fractions of the 63.0
TFLOPS this run measured for cuBLAS, the best of cuBLAS and torch over the shapes).**
`gemm_mma` as emitted reaches **79 to 81%** of it at 2048, 3072 and 4096 cubed (49.7 to
51.0 TFLOPS), **54 to 57%** at 6144 and 8192 cubed (35.6 and 33.7), and 50% at
m=n=8192, k=4096 (31.8). With the block-order experiment of `gemm_bench.py` (`ours+swz`,
a text edit; the emitter is unchanged) it is **70 to 73%** at 6144 and 8192 cubed (45.8
and 45.4; 69% at m=n=8192, k=4096) and slightly lower, 70 to 78%, at 3072 and 4096 cubed:
as in run 4 above, a fixed grouped order helps the large shapes and costs a little at the
mid ones. The card is **power-capped at about 72 W** and its clock falls under load: the SM
clock while ours ran was 1395 MHz at 2048 cubed, 1035 MHz at 4096 and 750 MHz at 6144
(cuBLAS ran at lower clocks too, throttle reason `0x4`, the software power cap), so
a part of the gap is power per flop and not only memory traffic. Another run the same day
(`20261104-205215-l4`, run 4 above) measured the same shape of curve with a 60.6 TFLOPS
peak. The fractions are of this card's measured cuBLAS peak, not the datasheet's.

## What this does not establish

Beyond what the section above measured: any card but one L4 (the A100 and H100 paths were
never run, and the lowering is Ada's); the compiler built from source at the pin (the runs
used the release binary, `NOT-PINNED` in `summary.tsv`, as designed); the Spot preemption
path against Google (it is tested against the stub only); every number in the cost table
above, which is from memory; and the 12 device cases the interpreter
cannot hold. What did turn out to work on the real thing, which this section once listed
as unverified: `gcloud` accepted the flag set, the
torch/cupy combination installed (`cu126` wheels, driver 580), and `cupy.RawModule`
compiled the emitted kernel.
