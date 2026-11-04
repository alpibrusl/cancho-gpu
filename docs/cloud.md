# CUDA on a real GPU, in Google Cloud

> **Status: written and checked against a stub `gcloud` (`tests/gcp.sh`, in the
> gate); never run against Google, and nothing here has touched a GPU.** The
> README's "CUDA has never run on hardware" stays true until the first run
> below is made and its results are written into `docs/device.md` §7.

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
| `gemm_bench` | our emitted `gemm_mma` vs cuBLAS (cupy) vs torch, 1024..8192 cubed | TFLOPS, as a fraction of **this card's measured** cuBLAS peak |
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
| `MAX_RUN` | hard cap on the VM's life, default `2h` |
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
* **`--max-run-duration` (`MAX_RUN`, 2h) and `--instance-termination-action DELETE`**:
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
| `MAX_RUN` hit | 2 h | about $0.6 |

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

## What this does not establish

Not run against Google; not run on a GPU. In particular unverified: that
`gcloud` accepts the flag set (it is lex-gpu's, which ran), that the stock image has a driver for its kernel,
that the pinned cancho builds on the VM and how long it takes, the pip
torch/cupy combination (`TORCH_INDEX` is cu126; the driver is newer), that
`cupy.RawModule` compiles the emitted kernel as the shim's NVRTC does, and every
number above in the cost table. The first run is also the first test of
`device_cases`'s expectation that the 181 cases are quick enough on 4 workers.
