# Driving a GPU from cancho

> **Status: built (cancho#251 slice 8) and verified against a mock
> driver, §7. CUDA: **run on a real NVIDIA L4** (Google Cloud, `docs/cloud.md`:
> NVRTC compiles every golden kernel, 176 of 188 `device_cases` agree with the
> interpreter, and the two NVFP4 shuffle bugs of §5b were seen there and are gone
> after the fix) after being *executed* on the CPU by an emulating
> driver (§5b, measured in §7), which found a real bug; this repository's own
> machines have no NVIDIA GPU. Metal: **run on real hardware** (Mac Studio, Apple Silicon, slice 10,
> issue #8) — first real Metal run proved reach and numerical agreement.

cancho's `docs/gpu.md` §6 left one question open: *"can a cancho
program drive a GPU at all, with no new backend?"* This is the answer
for CUDA. It is a question about **reach**, not speed: the kernels are
the ones this compiler already emits; what is new is a cancho program
that compiles one with NVRTC, uploads inputs, launches it and reads the
result back — and compares that against the reference interpreter
(`docs/design.md` §10).

## 1. Why a C shim, and how thin

cancho reaches C through `extern fn` under an `Ffi` capability, with
three limits that decide the shape (cancho `docs/opaque-pointers.md`,
`docs/foreign-linking.md`, `examples/tls_nb/gaps/`):

1. **No floats cross.** An extern parameter is an `int`, a `bool`, a
   `c_ptr`, or a byte slice.
2. **A byte slice crosses as a pointer *and* a length**, and only safely
   as the last parameter (`g7`, `g14`).
3. **`c_ptr` names only an extern's own parameters and results** — an
   ordinary function cannot take or return one, so a handle could not be
   passed around a cancho program.

The CUDA driver API needs the opposite of all three:
`cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream, void **params,
void **extra)` takes an array of pointers to arguments, and `CUdeviceptr`
and `CUfunction` are values a program holds. No signature in it can be
declared from cancho directly.

So a shim of about 380 lines of C, `shim/canchogpu.c`, owns every handle
and hands cancho **integers**: a buffer is `3`, a function `0`. Every
entry point takes `int64_t`s and at most one trailing byte buffer, and
answers an `int64_t` (≥ 0 a handle or a count, < 0 a failure whose text
`cgpu_error` copies out):

| Entry | Does |
|---|---|
| `cgpu_open()` | `dlopen` the driver (`libcuda.so.1`) and NVRTC, `cuInit`, device 0, a context |
| `cgpu_compile(src)` | NVRTC to PTX for the device's own `compute_XY`, then `cuModuleLoadData`: a module |
| `cgpu_function(module, name)` | `cuModuleGetFunction`: a function |
| `cgpu_alloc(bytes)` | `cuMemAlloc`: a buffer |
| `cgpu_upload(buf, bytes)` / `cgpu_download(buf, out)` | `cuMemcpyHtoD` / `DtoH` of a whole buffer |
| `cgpu_arg(buf)` | append a buffer to the next launch's argument list |
| `cgpu_launch(fn, gx, gy, threads)` | `cuLaunchKernel` with the arguments appended, then synchronise; clears the list |
| `cgpu_error(out)` | the last failure's text |
| `cgpu_close()` | everything freed, context destroyed |

`cgpu_arg` is how the `void **params` array is built without cancho ever
holding a pointer: the shim keeps it.

The shim `dlopen`s rather than links, as lex-gpu's own `lex-cuda` does,
so the program builds on any machine and a missing driver is an answer
(`error[device]: …`), not a link failure. `CANCHOGPU_LIBCUDA` and
`CANCHOGPU_LIBNVRTC` name other libraries, which is what the mock (§5)
uses.

## 2. The authority

Every entry is declared under `Ffi("libcanchogpu")`, and the device program
narrows its `Ffi` to exactly that, so `cancho authority` reports
`ffi("libcanchogpu")` and every symbol by name — and `unbounded`, honestly:
cancho's `docs/gpu.md` §6 predicted it would "report `ffi("libcuda")`
and nothing about the device". It does, one library over. Narrowing a
GPU capability is cancho's `gpu.md` §3 (`Gpu(device)`), which this
measures the need for but does not build.

## 3. The program

`cancho-gpu-device run <file.chx> [name=value ...] [--in NAME FILE ...] [--out NAME ...]`,
a second binary (`device/`), so the compiler itself still links nothing but
libc. Without `--in`/`--out` the inputs are the interpreter's made-up ones; with
them, the files' (design, gates and measurements in [`run.md`](run.md) §7):

1. Parse, elaborate, check and lower for `nvidia-ada`, exactly as
   `emit` does; run the reference interpreter over the made-up inputs.
2. Open the device, compile the emitted source, find the entry.
3. Upload every parameter with the **same** inputs the interpreter used,
   encoded as the device holds them: f32 as IEEE bits, f16 as half bits,
   I8 as bytes. cancho has no float bit casts in either direction for
   f32 or f16; `f32.cho` computes the bit patterns in integers.
4. Launch on the program's grid with the schedule's threads.
5. Download the writable parameters, decode them, and print the same
   summary line `--run` prints, prefixed `device:`, then the worst
   difference against the interpreter.

The interpreter is the reference, as it is in lex-gpu's own tests: the
GPU sums in a different order (split-K, fragments), so equality is not
the claim. **The refusal threshold** is a difference of 2e-2 of the
output's peak magnitude -- the same number lex-gpu uses as its log-prob
tolerance against an f32 reference, borrowed rather than derived;
past it, `error[device-mismatch]`.

## 4. What is attempted

* **Metal.** It is Objective-C (`MTLDevice`, `newLibraryWithSource`),
  reached through `objc_msgSend` with a different signature per call —
  a shim like this one, in Objective-C++, built and **run on real hardware**
  (Mac Studio, Apple Silicon). The shim (`shim/canchogpu-metal.mm`) provides
  the same C interface as the CUDA shim: integer handles in, int64 status
  out, at most one trailing byte buffer. Entry points: `cgpu_open`
  (`MTLCreateSystemDefaultDevice`), `cgpu_compile` (`newLibraryWithSource`),
  `cgpu_function` (`newComputePipelineStateWithFunction`), `cgpu_alloc`
  (`newBufferWithLength`), `cgpu_upload` (`memcpy` + `didModifyRange`),
  `cgpu_download` (`memcpy` from buffer contents), `cgpu_arg` (record buffer),
  `cgpu_launch` (`dispatchThreadgroups` + `waitUntilCompleted`), `cgpu_error`,
  `cgpu_close`. The device program (`device/metal-main.cho`) mirrors the CUDA
  path for `apple-m-series` target.

## 5. What is not attempted

* **CUDA on real hardware.** This repository's machines have none.
* **Timing.** Reach first.
* **Hopper**: NVRTC is told the device's own architecture, but the
  lowering is the Ada one; lex-gpu picks a Hopper target from the
  compute capability and this does not.

## 5. How it is checked without a GPU

A mock driver and a mock NVRTC (`tests/mock/`, C) stand in for the real
ones. They cannot run a kernel, so they check the **plumbing** — the part
the C-to-cancho boundary can get wrong:

* NVRTC's "PTX" is the source itself; the mock driver writes, per
  launch, the entry it was asked for, the grid, the block, how many
  arguments arrived and each buffer's size into a log, and refuses an
  entry name the source does not define.
* Memory is host memory: an upload followed by a download must give back
  the same bytes. The mock "kernel" writes nothing, so a writable
  output reads back as zeros and the comparison against the interpreter
  must fail with `device-mismatch` — which is itself the test that the
  comparison can fail.

`tests/device.sh` runs that for every kernel of lex-gpu at test sizes and
checks the log against what the program is: entry name, grid, threads,
one argument per parameter, each buffer the parameter's size. And it
runs once with no driver at all, which must be `error[device]`, not a
trap.

## 5b. How it is checked without a GPU, by running it

The mock (§5) checks the plumbing and runs nothing. `tests/emu/` is a driver
and an NVRTC that **run the emitted CUDA on the CPU**, selected through the
same `CANCHOGPU_LIBCUDA` / `CANCHOGPU_LIBNVRTC` names, so the unmodified device
program and shim run end to end and the result is held to the interpreter at
the program's own threshold (§3).

* **`libnvrtc_emu.c`** compiles the source with the system C++ compiler
  (`c++ -std=c++17 -O1 -ffp-contract=fast`, NVRTC's default `--fmad=true`; no
  strict aliasing; clang's alignment sanitizer in trap mode) against
  `tests/emu/include/{cuda_fp16.h,mma.h}`, which are `tests/emu/cuda_emu.h`. The
  "PTX" it returns is the path of the shared object. Compile errors come back
  in NVRTC's log format (`kernel.cu(12): error: ...`), and a static
  `__shared__` over 48 KB is refused at compile time, as ptxas does.
* **`cuda_emu.h`** is CUDA as C++: `threadIdx` and friends, `__global__`,
  `__constant__`, `__shared__` (a static in its own section, so it can be sized
  and poisoned), `__half` (binary16, round to nearest even, `_Float16`),
  `float4`/`uint4` (aligned, and their copies check it), `min`/`max`, `rsqrtf`,
  `__int_as_float`; `__syncthreads` as a barrier; `__shfl_sync`, `_down_`, `_up_`,
  `_xor_` as warp collectives; and `nvcuda::wmma` with the documented 16x16x16
  semantics for both operand layouts (`fill`, `load`, `mma`, `store`, and the
  element-wise `x[]`/`num_elements`), where `load`/`mma`/`store` are warp
  collectives that require all 32 lanes, one pointer and `ldm`, 256-bit
  alignment and `ldm` a multiple of 16 bytes.
* **Threads are fibers** (`ucontext`) on one OS thread, switching only where CUDA
  synchronises (a barrier, a shuffle, a wmma call); a block runs to completion
  before the next. A thread therefore runs alone between synchronisation points
  and the *order* the scheduler picks decides what an unsynchronised program
  does. **Two schedules run, and the bytes they leave in global memory must be
  identical:** schedule 0 runs blocks and threads in order with shared memory
  filled `0xFF`; schedule 1 shuffles blocks and threads with a seed
  (`CANCHOGPU_EMU_SEED`, default 1, printed in the finding) and fills shared
  memory `0x00`. A difference is `emu-race:` on standard error: a race, or a read
  of `__shared__` nobody wrote. Everything is deterministic given the seed.
* **`libcuda_emu.c`**: module load is `dlopen`; each allocation is its own
  `MAP_SHARED` mapping with an inaccessible page on each side and the buffer
  flush against the upper one, so one byte past the end faults (reported with
  the buffer it overran) and a canary below and past the end is checked after
  the launch. A launch runs in a forked child (a crash, a hang or a deadlock is
  an error, not the end of the test) with a timeout. The launch is checked as
  the driver would: at most 1024 threads, grid y at most 65535, no dynamic
  shared memory.
* **What it reports**, as `error[device]: launching the kernel: ...` (hard) or as
  an `emu-warn:` line (legal on hardware in practice, undefined in the guide):
  deadlock (a barrier or collective not everyone reaches); a warp collective whose
  mask names a lane that exited or does not exist, or whose lanes disagree; a
  barrier reached at different lines or after some threads exited; an illegal or
  misaligned access; a write outside a buffer; a result that depends on the
  schedule.

`tests/cuda-emu.sh` (in the gate) runs every case of `tests/cases.txt` that
emits through it, and fails on a mismatch past the program's 2e-2, **or past 2e-3
of the peak** (the worst measured is 4.4e-4, §7, so this catches an error ten
times smaller than the program's own check), on any hard finding, and on
`emu-race`. A case whose interpreter cannot hold its tensors
(`error[interp]`) is listed in `tests/cuda-emu-known.txt` as `noref`, and
`tests/cuda_emu_big.py` runs those at full size without a reference. The
real bugs found (§5b: both NVFP4 lookup bugs, now fixed) were kept in
`tests/cuda-emu-bugs.txt` while open; their reproducers are cases of
`tests/cases.txt` now and must agree. `tests/cuda_static.py` checks the goldens without a compiler: static
shared at most 48 KB (the 99 KB Ada opt-in needs dynamic shared memory and
`cuFuncSetAttribute`, which the shim does not do), the header's `tg mem`
matches what is declared, threads at most 1024, nothing sm_90/sm_100 (clusters,
TMA, wgmma, tcgen05, fp4 types) and no inline PTX, one `__global__` entry whose
parameters are all `__restrict__` pointers. `tests/cuda_emu_selftest.py`
shows the emulator can fail: kernels written to be wrong (a missing barrier, an
uninitialised shared read, a write past either end, a deadlock, a misaligned load,
a partial-warp shuffle, a misaligned or partial wmma) must each be found, and the
right ones must give the right numbers (shuffle semantics, half rounding, all four
wmma layout pairs against a direct product). `tests/cuda-emu.sh` also breaks five
emitted kernels on purpose (`CANCHOGPU_EMU_MUTATE`): a barrier removed, a
store transposed, a loop bound overrun, a shared tile moved off 32 bytes, `expf`
swapped for `exp2f`; each is refused for its reason.

**What it cannot catch** (so a GPU run still has to):

* *NVRTC accepting the source.* The emulator accepts a superset (any arithmetic
  type converts to `__half`; `min`/`max` of any two types) and implements no
  more of `cuda_fp16.h` than the emitter uses. `scripts/cuda_check.sh` or a real
  NVRTC is the only check.
* *Hardware numerics.* Tensor-core accumulation order and truncation (the emulator
  uses `fmaf` in k order), `expf`/`tanhf`/`rsqrtf` ulp (the CPU libm's, not CUDA's),
  FMA contraction choices (the C++ compiler's, not NVRTC's).
* *Registers and resources.* "Too many resources requested for launch" (a
  block of T threads gets 65536 / T registers each: 64 at 1024 threads; the
  emitter's blocks are at most 256), local-memory spills, `__launch_bounds__`, occupancy, shared-memory
  bank conflicts, performance.
* *The memory model.* Blocks run one at a time and a thread runs alone between
  synchronisation points, so a missing `__threadfence`/`volatile`, visibility
  across blocks and warp-synchronous reductions that rely on lockstep are not
  modelled; the two schedules find a race only if the order changes the bytes.
* *Lane mapping inside a wmma fragment.* Every lane holds the whole tile, so
  element-wise code (`x[t] += y[t]`, which the emitter writes) is right and code
  that depends on which lane holds which element would be wrongly accepted.
* *Uninitialised locals*, integer division by zero and out-of-range shifts (the CPU
  answers differently from the GPU; the alignment sanitizer is the only UB check).
* *The real driver*: `cuCtxCreate` flags, error codes, `cuLaunchKernel`'s
  checks beyond the ones listed, JIT time.
* *Sizes nobody ran.* It executes what it is given: the cases of
  `tests/cases.txt`, and the sweep of §7. A size outside both is not covered.

## 6. What remains unproven

* **CUDA on real hardware: checked on one L4, not beyond it.** Measured on a
  real NVIDIA L4 (Ada, `g2-standard-8`, `scripts/gcp/cuda_test.sh`, `docs/cloud.md`):
  the emitted CUDA *compiles* under NVRTC 12.9 for `compute_89` (253 of 253
  golden kernels in the last run, 246 of 246 in the first; lex-gpu checks this
  with `scripts/cuda_check.sh` on its own emitter's output, ours is the same
  bytes, §9 of `design.md`); a real driver accepts these calls; and 176 of the
  188 `tests/cases.txt` cases agree with the interpreter at the program's
  threshold (0 mismatches; the other 12 are the interpreter's `tensors do not
  fit` memory error on the largest shapes, which is the CPU reference failing,
  not the GPU, and is unchanged between runs). The two bugs below, which on
  the same L4 hung 40 of 50 reproducer cases with `K` not a multiple of 32
  (7 PASS / 3 MISMATCH / 40 TIMEOUT, run `20261104-212332-l4`), are **gone after
  the fix of PR \#104: 50 of 50 PASS** (run `20261105-083519-l4`, main `46670b5`;
  the step took 20 s, from 2413 s). Still unproven: a PASS on a GPU does not
  clear what the CUDA guide calls undefined; the numerics of kernels the
  interpreter cannot hold (the 12); any card but an L4; and what the emulator
  cannot see, listed at the end of §5b. One card on two days, not a sample.
* **Metal timing:** Reach is proved; speed is not yet measured.

## 7. Measured

### CUDA (mock)

`tests/device.sh`, in the gate, on every one of lex-gpu's six kernels:

* The "driver" was asked for the entry the golden `.cu` defines, on the
  grid and block `emit` reports, with **one argument per parameter whose
  size is the upload's**, in order; NVRTC was asked for `compute_89`.
* The mock's zeros were refused as `device-mismatch` every time, so the
  comparison can fail.
* With no driver, `error[device]: opening the device: no CUDA driver:
  libcuda.so: cannot open shared object file: …`, exit 1.

Watched fail: binding one argument fewer than the kernel takes was
caught on all six kernels (the sizes no longer match the uploads). The
first version of the check counted arguments from the kernel's
signature and missed it; it now compares the sizes, and the shim points
unused argument slots at a zero, so a short list can never make a
reader walk into stack garbage.

The bit patterns (§3) are checked against Python's `struct` packing of
the same values (`test_ieee_bit_patterns_for_the_device`): normal,
subnormal, the largest finite, infinity and `-0.0`, for f32 and f16.

What `cancho authority` says about the device program — §2's
prediction, exactly:

```
UNBOUNDED: this program calls foreign code. ...
performs
    args
    err_write
    ffi("libcanchogpu")    <- unbounded
    fs_read("")
    heap
    io_write
unbounded by
    libcanchogpu:cgpu_alloc
    libcanchogpu:cgpu_arg
    ...
```

and the compiler itself, unchanged: `args, err_write, fs_read(""),
fs_write(""), heap, io_write`, bounded.


### CUDA (emulator)

`scripts/build-emu.sh`, then `tests/cuda-emu.sh`, `tests/cuda_static.py` and
`tests/cuda_emu_selftest.py`, all in the gate. Measured on an M-series Mac, CPU
only:

* **Every emitting case, run.** 246 of the 272 cases of `tests/cases.txt` emit a
  kernel (the other 26 are refused, as `golden/` records). **234 run end to end through the
  unmodified device program and shim on the emulator and agree with the
  interpreter** (worst difference over all 275 outputs: 4.4e-4 of the peak, one
  half-precision ulp in `pack_hw`; 115 outputs are bit-identical), with no
  deadlock, no illegal or misaligned access, no write outside a buffer, no
  `emu-warn`, and the two schedules leaving identical bytes. The other 12 are the
  model-sized cases whose tensors the interpreter cannot hold (`error[interp]`,
  `noref` in `tests/cuda-emu-known.txt`): `tests/cuda_emu_big.py` runs them at
  full size with random inputs and the checks that need no reference, and
  **11 of them run clean at full size** (the largest, `matvec_rows n=151936` and `matvec_fp4_rows n=248320`, 31040 blocks, in 102 s and 165 s for both schedules); the 12th, `gemm_fp4_res m=512 n=5120 k=17408` (45 G multiply-adds), was run at `m=128` (same `n`, `k` and kernel; 409 s) and is clean; the `m=512` grid did not finish in half an hour and was stopped. A second full run under another seed (`CANCHOGPU_EMU_SEED=7`)
  also passes (234 agree, 12 `noref`, no finding). About two minutes cold, a quarter of that with the compile cache.
* **No hardware-free check found a bug at the sizes the goldens record.** The
  bugs are at other sizes (below); a green `tests/cuda-emu.sh` says nothing
  about a size nobody ran.
* **It can go red.** 20 self-tests (`tests/cuda_emu_selftest.py`): right kernels
  give right numbers (shuffle down/up/xor/idx with widths, `__half` rounding at
  ties and overflow, all four wmma layout pairs against a direct product), and
  each wrong kernel is found: a missing `__syncthreads` and a read of
  unwritten `__shared__` (`emu-race`, seed printed), a write one element past
  and one before a buffer, a barrier and a shuffle waiting on each other, a
  misaligned vector load, a full-mask shuffle in a partial warp and in divergent
  code, a misaligned wmma pointer, wmma by half a warp, an ill-formed kernel
  (NVRTC's `kernel.cu(5): error:` format), a static `__shared__` of 12289
  floats, a 1025-thread block, grid y 65536, the static checks on seven
  hand-broken sources. And five emitted kernels broken on purpose
  (`CANCHOGPU_EMU_MUTATE`, in `tests/cuda-emu.sh`): the barriers of `rmsnorm`
  removed (`device-mismatch`), `gemm_mma`'s store transposed (`device-mismatch`),
  `rmsnorm`'s loop bound past the buffer (`past the end of buffer`), `gemm_mma`'s
  `v6` tile moved 16 bytes (`256-bit`), `softmax`'s `expf` made `exp2f`
  (`device-mismatch`); each is refused.
* **Static checks** (`tests/cuda_static.py`, 246 kernels, no findings): most
  static `__shared__` is 36864 B (`gemm_mma`), against 49152 B; threads per block
  at most 256; no sm_90/sm_100 construct, no inline PTX, two includes, one entry
  with only `__restrict__` pointer parameters, and each header's `tg mem` is
  what the source declares. **Note for the Ada budget:** the headers say "of
  101376 B" (the 99 KB opt-in), but nothing above 48 KB static can compile and
  the shim opts in to nothing; no kernel is near it, so the larger number is
  unused, not wrong.
* **Sizes the goldens do not have**: `tests/cuda_emu_sweep.py 30 3` perturbs
  every family's constants (the values its cases use, halved, doubled, moved by
  1, 16, 32): 2299 variants (seed 3), of which 1007 ran clean on the emulator and agreed, 1268 the compiler refused by its own rules (`offset` 883, `unbound` 158, `bounds` 98, `lowering` 30, and the like), 1 was too slow for the emulator's 60 s, 1 too big for the interpreter, and **22 failed, all in the NVFP4 kernels (21 deadlocks, 1 wrong answer)**. The compiler refuses most variants by its own rules
  (divisibility, bounds: cheap); of those it accepts, every family outside the NVFP4 ones runs
  clean, including the attention, convolution, normalisation, rope, softmax,
  fp16 matvec and wmma families.

**Two real bugs in the emitted CUDA, both fixed.** Both were the NVFP4 nibble
lookup `__shfl_sync(0xffffffffu, fp4_lane, nibble)` (`src/lower.cho`
`dequant_fp4`, through `dialect.shuffle`), a full-mask shuffle wherever a
dequantised weight is read. The CUDA emitter now indexes the `__constant__`
16-entry table instead (`FP4_V[nibble]`, `dialect.fp4_lookup`); Metal keeps its
simdgroup shuffle of `fp4_lane`, whose lanes all run it. The reproducers are
cases of `tests/cases.txt` (the `tests/cuda-emu-bugs.txt` list is gone) and agree
on the emulator. The record of what they were:

1. **A per-lane loop whose trip count differs between lanes.**
   `for (p0 = lane; p0 < K; p0 += 32)` with `K = 16 (mod 32)` (legal: the scale
   group is 16) gives lanes 0..15 one more trip than lanes 16..31, so the shuffle is
   called with a mask naming lanes that have left the loop. Undefined in CUDA
   (`shfl.sync` waits for the lanes it names; on sm_70+ those lanes next reach a
   *different* shuffle). The emulator: deadlock. **On a real NVIDIA L4 it hung:**
   40 of 50 `shuffle_repro` cases timed out, `K` = 64, 96, 128 passed
   (`results/gcp/20261104-212332-l4/shuffle_repro.log`). Seen in `batch_fp4`,
   `matvec_fp4`, `matvec_fp4_rows`, `matvec_fp4_res`, `matvec_fp4_swiglu`,
   `matvec_fp4_batch`, `matvec_fp4_res_batch`, `matvec_fp4_swiglu_batch`, at
   `k` = 16, 48, 80, 112, 144, 240, 272, 4080, 5104, 5136, 17392, 17424.
   Every golden had `K` a multiple of 32 and Qwen's `K` are too, which is why
   nothing recorded or run showed it.
2. **The element-per-thread fallback with a count that is not a multiple of 32.**
   `if (e < count) { ... __shfl_sync(0xffffffffu, ...) }`: in the last pass part of
   a warp is active; the lookup for nibble `v` reads lane `v`, which may be one of
   the inactive lanes. Wrong values: `batch_fp4 b=36 n=129 k=224` (count
   4644 = 18 x 256 + 36) was off by 0.78 against a peak of 9.09 under the
   emulator.

**Checked on the L4 (2026-11-05, main `46670b5`, run `20261105-083519-l4`):**
`shuffle_repro` is 50 PASS, 0 MISMATCH, 0 TIMEOUT (it was 7 / 3 / 40 before the
fix), and `device_cases` is 176 PASS, 0 MISMATCH, 12 ERROR of 188 (the 12 the
interpreter's memory, as before). A pass does not clear what the CUDA guide
calls undefined for the old form; it shows the fixed form does not hang or
misread on this card.
A `__constant__` read with a divergent index serialises within a warp; the
lookup's cost against the shuffle is measured by the `fp4_perf` step
(`scripts/gcp/fp4_perf.py`, `docs/cloud.md`), **not yet run**. Note what it
will find: the `matvec_fp4_rows`, `_res` and `_swiglu` kernels at the model's
own `K` (5120, 17408) read the weights through `fp4_pair` and contain no table
lookup at all (checked on the emitted CUDA), so the fix cannot change them; the
lookup is in the generic path, which those families take when `K` is an odd
multiple of 32 (5152, 17440), a `K` Qwen does not have.

### Metal (real hardware)

First real Metal run on Mac Studio (Apple Silicon GPU):

* **Reach proved:** The Metal shim (`shim/canchogpu-metal.mm`, Objective-C++)
  successfully loads the Metal framework, compiles emitted `.metal` source
  with `newLibraryWithSource`, creates compute pipelines, allocates buffers,
  uploads data, launches kernels with `dispatchThreadgroups`, and reads
  results back — all through the same C interface (`cgpu_*`) as the CUDA shim.
* **Numerical agreement:** All tested kernels produce results within the
  2e-2 tolerance against the CPU reference interpreter:
  - `softmax.chx r=4 n=64`: worst difference 5.59e-9 of peak 0.0259
  - `softmax.chx r=1 n=8`: worst difference 1.49e-8 of peak 0.1968
  - `softmax.chx r=16 n=256`: worst difference 4.66e-9 of peak 0.0063
  - `gemm_fp4.chx m=128 n=128 k=64`: worst difference 1.43e-6 of peak 6.7963
  - `silu_mul.chx r=4 n=64`: worst difference 1.49e-8 of peak 0.1537
* **Performance:** Total execution time (CPU reference + Metal GPU + comparison)
  for `softmax.chx r=4 n=64`: ~0.036s; for `softmax.chx r=16 n=256`: ~0.039s.
  (Timing includes both reference and device execution; GPU-only timing
  not yet isolated.)

What `cancho authority` says about the Metal device program:

```
UNBOUNDED: this program calls foreign code. ...
performs
    args
    err_write
    ffi("libmetalgpu")    <- unbounded
    fs_read("\")
    heap
    io_write
unbounded by
    libmetalgpu:cgpu_alloc
    libmetalgpu:cgpu_arg
    ...
```

and the compiler itself, unchanged: `args, err_write, fs_read("\"),
fs_write("\"), heap, io_write`, bounded.
