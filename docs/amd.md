# AMD: a HIP target (design, 2026-11-24; H1 built: `emit` writes HIP text for 156 cases, never compiled; H2 built: the shim drives a HIP runtime, held to a mock; H3 built: those cases run on a CPU emulator at 64 lanes, and it corrected section 2; H4 and H5 built: the shuffling kernels and the matrix products emit too, 500 cases in all, 441 agreeing with the interpreter on the emulator; nothing run on AMD hardware)

Issue [#9](https://github.com/alpibrusl/cancho-gpu/issues/9), slice 11 of the roadmap. Today `amd-cdna3` is a row of `src/target.cho` (`simd_width` 64, 1,024 threads, 64 KiB of threadgroup memory) with no emitter, no shim and no case: a file with a schedule for it type-checks and `emit` writes nothing for it.

What was **read** (ROCm's documentation, DigitalOcean's pages for the AMD Developer Cloud, this repository) is marked **[read]**; what was **counted** in this repository on `5473bdb` is marked **[counted]**; a judgment is **[reckoned]**. Nothing here was run on an AMD GPU, and no ROCm is installed on this project's machines. A claim marked **[unconfirmed]** was not found in any page read and has to be settled by the first run.

## 1. What AMD offers, as read on 2026-11-24

* **The run-time compiler exists and has NVRTC's shape.** hipRTC: `hiprtcCreateProgram`, `hiprtcCompileProgram`, `hiprtcGetProgramLogSize` / `hiprtcGetProgramLog`, `hiprtcGetCodeSize` / `hiprtcGetCode`, `hiprtcDestroyProgram`, in `libhiprtc.so` (its own library since ROCm 7.0). It has no standard C or C++ headers by default, and it needs `--gpu-architecture` when no supported GPU is present. **[read]** ([hipRTC](https://rocm.docs.amd.com/projects/HIP/en/docs-7.14.1/how-to/hip_rtc.html))
* **The module interface exists and has the driver API's shape.** `hipModuleLoadData`, `hipModuleGetFunction`, `hipModuleUnload`, `hipModuleLaunchKernel`. **[read]** ([module management](https://rocm.docs.amd.com/projects/HIP/en/docs-7.14.1/reference/hip_runtime_api/modules/module_management.html)) One page for the Fortran binding says `kernelParams` of `hipModuleLaunchKernel` is not implemented and that the arguments go through `extra`; that was not checked against the current C interface. **[unconfirmed]**
* **The kernel language is CUDA C with these differences** ([language extensions](https://rocm.docs.amd.com/projects/HIP/en/docs-7.14.1/reference/cpp_language_extensions.html)) **[read]**:
  * `warpSize` is **64** on gfx9 (Instinct) and 32 on gfx10 and later (Radeon, the Ryzen AI processors).
  * `__shfl`, `__shfl_up`, `__shfl_down`, `__shfl_xor` and their `_sync` forms are there; a `_sync` mask is 64 bits wide.
  * **There is no warp matrix interface** (`nvcuda::wmma`). The Instinct cards have matrix units; HIP's kernel language does not name them.
  * `__half` in general kernel code is not documented (it appears only for warp reductions, behind a macro). Whether `hip/hip_fp16.h` gives `__half`, `__half2`, `__half22float2` and `__floats2half2_rn` under hipRTC is **[unconfirmed]**.
  * The largest thread block and the limit of `__shared__` memory are not stated. **[unconfirmed]** The row's 1,024 and 64 KiB are lex-gpu's, of unrecorded origin.
* **Where it runs.** ROCm 7.14.1's matrix lists Instinct, Radeon RX 7000 and 9000 and the Ryzen AI Max 300 processors, on Ubuntu 24.04 and 26.04 and on Windows 11. **[read]** ([compatibility](https://rocm.docs.amd.com/en/docs-7.14.1/compatibility/compatibility-matrix.html))
* **What can be rented.** The AMD Developer Cloud (run by DigitalOcean) has one MI300X (192 GB) at $2.59 an hour and eight at $20.72. Its quick-start image is Ubuntu 24.04 with ROCm 6.4.0, on a page dated 2025-06-10; a bare Ubuntu image takes any ROCm. **[read]** ([features](https://docs.digitalocean.com/products/amd/details/features/), [pricing](https://docs.digitalocean.com/products/amd/details/pricing/), [setup](https://docs.digitalocean.com/products/amd/getting-started/recommended-amd-droplet-setup/)) The AMD AI Developer Program's credit was reported as $100, expiring 30 days after it is activated, from search results only (amd.com did not answer); at the listed price that is about 38 hours of one card. **[reckoned, on an unconfirmed figure]** The clock starts at activation, so the credit is asked for when section 4's slice H5 is ready to run, not before.

## 2. What the CUDA text assumes, counted

Of the 552 directories of `golden/`, 489 hold a `.cu` file (the rest are refusals, or Metal only). In those 489 **[counted]**:

| the kernel text | files |
|---|---|
| names a warp shuffle (`__shfl_down_sync`, 2,113 calls, and `__shfl_sync`) | 293 |
| names `nvcuda::wmma` | 29 |
| names neither | 169 |
| names `__half` | 178 |
| names `__half2`, `__half22float2` or `__floats2half2_rn` (the NVFP4 decode) | 68 |
| divides a thread index by a literal 32 | 346 |
| includes `<cuda_fp16.h>` and uses `__int_as_float` (every file, from the preamble) | 489 |

So three things stand between the CUDA text and an Instinct card, and each is a property of the lowering, not of a spelling:

1. **The width of a simdgroup is the literal 32** in `src/lower.cho` (the reductions, the warp grids, the mask `0xffffffffu` in `src/dialect.cho`). On gfx9 a wave has 64 lanes. A kernel whose threads only *group themselves* by `tid / 32u` and meet at `__syncthreads` does not care how wide the hardware's wave is; a kernel that shuffles does: `__shfl_down_sync(m, v, 16)` on a 64-lane wave pairs lane 16 with lane 0 as intended, and lane 32 with lane 16 where the lowering meant lane 32 to be lane 0 of the next simdgroup. **[reckoned]** 293 files. **Corrected by H3 (section 7): the pairing does go as described, and it does no harm.** Lane 16 does read lane 32, but lane 16's value after that step is one no later step reads: the trees deliver a simdgroup's sum to its lane 0, and lane 0 of each 32-lane half reads only its own half. Measured on the emulator at 64 lanes, every shuffling case that has a reference agrees with the interpreter once the mask is 64 bits wide. What does break the CUDA text is the mask alone: `0xffffffffu` does not name lanes 32 to 63.
2. **The matrix unit has no name in HIP.** 29 files. The product has to be lowered another way (section 3.3).
3. **Halves.** 178 files, all through `<cuda_fp16.h>`. Unconfirmed either way; if `__half` is missing, the dialect spells it `_Float16` (clang's, which hipRTC's compiler is) with explicit conversions.

169 files (35% of the CUDA text) need none of 1 and 2.

## 3. Decisions

### 3.1 A third dialect, not the CUDA text

Issue #9 offered two ways: reuse `nvidia-ada`'s CUDA C through ROCm's compatibility, or a HIP dialect. **A dialect**, `dialect.hip()`, for the reasons section 2 counts: 60% of the CUDA text shuffles at width 32 and would be *silently wrong* on a 64-lane wave, which is the one failure this project does not accept. The dialect is CUDA's spelling (most of the 35 places where `src/dialect.cho` asks `is_cuda` are shared) and differs where section 1 says the language does. `amd-cdna3` emits `<kernel>.hip`. **Corrected by H3: the reason given is not the one that holds.** The shuffles are not silently wrong at 64 lanes (section 7); a dialect is still what is needed, for smaller reasons: the 64-bit mask, the header, the infinity, the table's storage class, and `wmma`, which has no HIP spelling at all.

### 3.2 The width comes from the target row

`target.simd_width` is already in the table (32, 32, 32, 64) and the checker already reads it for the budget. The lowering reads it too, in place of the literal: lanes per simdgroup, the halving steps of a reduction (a 64-lane tree has one more step), the shuffle mask (`0xffffffffffffffffull` at 64). For the two existing targets the emitted text must not change by a byte; the goldens of CI are what holds that. This is the largest change of the slice list and the one most likely to be wrong, which is why it comes after an emulator can run its output (H3 before H4).

**Corrected by H3: not needed for a right answer.** Section 7 measured that the 32-lane trees are right on a 64-lane wave, so H4 is the mask and lifting the refusal, and the emitted trees stay as they are. Reading the width from the row would make a reduction use a whole wave (one tree of 64 where there are now two of 32), which is a question of speed for a card to answer, not a slice before one.

### 3.3 The matrix product without `wmma`

`mma` on `amd-cdna3` lowers to register accumulators and plain multiply-adds per lane: the `frag` tiles become per-thread `float` arrays, the product a loop the compiler vectorises as it can. It will be slower than the card's matrix units by an amount nobody here can state; it is correct, and it makes the 29 cases run. Reaching the matrix units (through inline assembly or a vendor intrinsic, if hipRTC's compiler accepts one) is a later question that only a card can answer.

### 3.4 The shim is the CUDA shim with other names

`shim/canchogpu.c` holds a table of fourteen driver functions and seven NVRTC functions behind `dlopen`. HIP's have the same shapes (section 1). A second table in the same file, chosen at `cgpu_open` by which library opens (`CANCHOGPU_LIBHIP`, `CANCHOGPU_LIBHIPRTC`), not a second shim: the handle bookkeeping, the argument list and the error path are the 380 lines that were hard to get right, and `docs/device.md` section 1's entry points do not change. If `kernelParams` is refused by the real library, the shim packs the arguments into `extra`; the mock accepts both and logs which.

### 3.5 Checked without a card, as CUDA was

`docs/device.md` sections 5 and 5b, repeated: a mock `libamdhip64`/`libhiprtc` for the plumbing, and the emulator of `tests/emu/` run **at 64 lanes** for the arithmetic. The emulator's warp width becomes a parameter; at 64 with the *CUDA* text it must report the wrong pairing of section 2 item 1 on a shuffling kernel, which is the test that the emulator can see the bug the dialect exists to avoid. The emulator found two real bugs in the CUDA path before any card did; it is the reason to expect the first hour on an MI300X to be spent on things section 1 marks unconfirmed and not on the lowering. **Corrected by H3:** the emulator at 64 lanes refuses the CUDA text for its mask, and with the mask widened it finds nothing wrong, on any case (section 7). The test kept in the gate is those two facts.

## 4. Slices

| | what | done when |
|---|---|---|
| H1 | `dialect.hip()`, `amd-cdna3` on the command line, `emit` writes `.hip` for a kernel that neither shuffles nor multiplies matrices; any other is refused `error[lowering]` naming what the target lacks | goldens recorded for the 169; the other two targets' goldens unchanged |
| H2 | the HIP table in the shim, the mock libraries, `tests/device.sh` for the target | the mock's log agrees with each program; no library gives `error[device]` |
| H3 | the emulator at a width from the target; the test of what 64 lanes do to the CUDA text | H1's cases agree with the interpreter under the emulator (section 7) |
| H4 | the 64-bit shuffle mask in the HIP dialect, and the shuffling kernels no longer refused (as designed: the lowering's width from the target row; section 7 measured that the trees need no change) | the shuffling kernels emit and agree under the emulator at 64; the two targets' goldens unchanged (section 8) |
| H5 | the product without `wmma` | the 29 emit and agree under the emulator (section 9) |
| H6 | (prepared, section 10) the first run on an MI300X: hipRTC compiles every golden `.hip`, the device cases against the interpreter, the unconfirmed items of section 1 answered in this document | a `results/` directory and this document corrected in place |
| H7 | a model on the card (Qwen3.8-27B through `cancho-gpu-serve`), with its speed | after H6, and only then is a number published |

H1 to H5 need no AMD hardware and no credit. H6 is one script on the pattern of `scripts/gcp/cuda_test.sh` (`docs/cloud.md`), with its cost guard rails.

## 5. H1, built

`cancho-gpu emit` lowers a file's `amd-cdna3` schedule to `<dir>/<kernel>.hip`, and `check` checks it. The dialect (`dialect.hip()` of `src/dialect.cho`) is CUDA's spelling with a different preamble: `#include <hip/hip_fp16.h>`, the infinity as clang's `__builtin_inff()` (hipRTC has no standard headers), and the NVFP4 table as `static const` where CUDA says `__constant__`. After lowering, `src/lower.cho` refuses for this target a kernel that uses `mma` and one whose text shuffles, each `error[lowering]` saying which slice lifts it (`tests/reject/lowering-mma-hip.chx`, `tests/reject/lowering-shuffle-hip.chx`). (H4, section 8, lifted the second and H5, section 9, the first; their fixtures are gone, and `tests/reject/lowering-wave-hip.chx` holds what is still refused: threads that are not whole waves.)

**Measured.** 76 kernel files of `tests/chx/` and `examples/` carry an `amd-cdna3` schedule, a copy of their `nvidia-ada` one, and their 156 cases of `tests/cases.txt` each record a `.hip` file in `golden/`: none names a shuffle or `wmma`, 42 name `__half`. The other goldens changed only by gaining the interpreter's `amd-cdna3:` lines in 70 `RUN.txt` files (92 lines added, none removed); no `.cu` or `.metal` file changed.

**Correction to section 2's 169.** 156 of them are recorded, not 169. Section 2 counted cases; a schedule belongs to a kernel file, and `emit` refuses the whole file when one target is refused, so a file gets a schedule only if *every* one of its cases can be lowered. 12 of the 169 are cases of files that shuffle at other sizes, and one (`examples/cnn/bn_fin.chx`) has a schedule of 32 threads, which is less than one 64-lane wave. All 13 wait for H4.

**Not done in H1.** The text has not been through any compiler: no hipRTC is here, and the emulator learns HIP in H3 (section 7). `cancho-gpu run --target` and a session still name only the two older targets.

## 6. H2, built

`shim/canchogpu.c` drives HIP through the entry points it already had (`docs/device.md` section 1), as 3.4 decided: HIP's functions are loaded into the driver's slots under their own names (`hipInit`, `hipModuleLoadData`, `hipModuleGetFunction`, `hipMalloc`, `hipFree`, `hipMemcpyHtoD`, `hipMemcpyDtoH`, `hipDeviceSynchronize`, `hipModuleLaunchKernel`; `hiprtcCreateProgram`, `hiprtcCompileProgram`, `hiprtcGetCodeSize`, `hiprtcGetCode`, the two log functions, `hiprtcDestroyProgram`). Where the shapes differ:

| | CUDA | HIP |
|---|---|---|
| a context | `cuCtxCreate` | none (HIP's is deprecated): `hipSetDevice(0)` |
| the architecture given to the compiler | the device's `compute_XY` | none; `CANCHOGPU_HIP_ARCH` names one for a machine without a supported GPU |
| the include path | where `cuda_fp16.h` is, under `CUDA_HOME` and others | where `hip/hip_fp16.h` is, under `ROCM_PATH`, `HIP_PATH`, `/opt/rocm` |
| an error's text | `cuGetErrorString(r, &s)` | `s = hipGetErrorString(r)` |
| a launch's arguments | `kernelParams` | `kernelParams`, or one packed buffer through `extra` when `CANCHOGPU_HIP_ARGS=extra` (section 1's unconfirmed item: both are written, the first run chooses) |

One entry point is new: `cgpu_backend()` answers 1 when the process would open HIP and 0 for CUDA, and starts nothing. HIP when `CANCHOGPU_LIBHIP` names a library; CUDA when `CANCHOGPU_LIBCUDA` does or a CUDA driver opens; else HIP if `libamdhip64.so` opens. `cancho-gpu-device` asks it and compiles the program for `amd-cdna3` in the HIP dialect, so the text and the runtime cannot disagree.

**Checked** (`tests/device.sh`, in the gate), against `tests/mock/libamdhip64_mock.c` and `libhiprtc_mock.c`: 4 kernels, each with its arguments both ways. hipRTC is given a program named `kernel.hip` and no option; the runtime sees the golden `.hip`'s entry, the grid and threads `emit` reports, one argument a parameter, each buffer the upload's size; the mock's zeros are refused `error[device-mismatch]`. And: the architecture reaches hipRTC only when asked for; a shuffling kernel is refused `error[lowering]` before any library call; no runtime, and a runtime without hipRTC, are `error[device]`.

**What the mock cannot say.** It has the signatures this document read, so it agrees with the shim by construction: a wrong signature, a missing symbol in the real `libamdhip64.so`, or a real refusal of `kernelParams` shows only on H6's card. Passing a device pointer as a 64-bit integer is right for x86-64 and AArch64 and is not what the C standard promises.

## 7. H3, built

The emulator of `tests/emu/` (`docs/device.md` section 5b) has a lane width: `EMU_WARP`, 32 or 64, which `cuda_emu.h` uses for `warpSize`, for which threads form a wave, for a shuffle's default width and for the width of a mask. Two libraries carry HIP's names over the existing two: `libhiprtc_emu.c` is `libnvrtc_emu.c` built with `EMU_HIP` (64 lanes unless `CANCHOGPU_EMU_WARP=32`, `hip/hip_fp16.h` from this repository, no `wmma`, a program named `kernel.hip`), and `libamdhip64_emu.c` is `libcuda_emu.c` with a launch that takes its arguments by `kernelParams` or by `extra`. `cancho-gpu-device` reaches them through the shim of section 6 with `CANCHOGPU_LIBHIP` and `CANCHOGPU_LIBHIPRTC`, so the text that runs is the text `emit` writes for `amd-cdna3`.

**Measured** (`tests/hip-emu.sh`, in the gate; one Apple M4 Max, the system's clang):

* **The 156 cases of H1: 134 agree with the interpreter, 22 have no reference** (the interpreter cannot hold their tensors; they are `noref` lines of `tests/cuda-emu-known.txt`), none disagrees, and neither schedule of any case differs from the other. The worst difference is 4.3e-4 of the peak, one half-precision step, as on the CUDA side. 31 of the 134 use `__half`.
* **The emulator's own width**: one hand-written kernel that sums a wave with a six-step tree gives 2,080 and 6,176 for 128 threads at 64 lanes, and the doubled sums of four half-waves at 32 (`tests/emu/wave_selftest.c`).
* **Three mutants of emitted HIP are refused**: `expf` for `exp2f` (`device-mismatch`), a store one row past its buffer (the guard page), and a `wmma` fragment (it does not compile, as HIP has none).
* **The CUDA text at 64 lanes.** As emitted it is refused at the first shuffle: the mask `0xffffffffu` does not name lane 32. With that one literal widened to `0xffffffffffffffffull` and nothing else changed, **all 291 cases whose CUDA text shuffles and names no `wmma` were run: 259 agree with the interpreter, 32 have no reference, none disagrees and none draws a warning** (`HIP_EMU_CUDA64=1 tests/hip-emu.sh` repeats it; two of them are in the gate).

That last line is the opposite of what sections 2 and 3.1 reckoned, and they are corrected in place. The reason, once seen: every shuffle the lowering writes is a step of a tree that brings a 32-lane simdgroup's result to its first lane, and on a 64-lane wave the lower half's tree never reads above lane 31 for a value that is used, while the upper half's tree reads lanes 32 to 63 exactly as it would as a wave of its own. The lanes that do read across the middle (16 to 31) hold partial sums nobody reads.

**What it means for the slices.** H4 becomes small: the HIP dialect writes the 64-bit mask and the refusal of shuffling kernels goes; the trees do not change. That is a claim about the emulator's model of a 64-lane shuffle, which is ROCm's documentation as read in section 1 (a shuffle's default width is `warpSize`, a lane past the width keeps its own value); H6 is where a card says whether the model is the hardware.

**What the emulator cannot say.** It is clang on a CPU with this repository's header, so it says nothing of hipRTC: whether `hip/hip_fp16.h` gives the half types, what a 32-bit mask does on a real 64-lane wave (the emulator refuses it; the documentation read says the mask is 64 bits and not what a short one does), the limit of `__shared__` memory (the emulator uses the row's 64 KiB), or `kernelParams` against `extra`. The 22 cases without a reference were not run at full size as their CUDA counterparts are by `tests/cuda_emu_big.py`.

## 8. H4, built

`dialect.hip()` writes a shuffle's mask as `0xffffffffffffffffull`, and `src/lower.cho` no longer refuses a kernel whose text shuffles; the steps of the trees are the ones the CUDA text has, as section 7 measured they can be. `mma` is still refused for this target (H5).

**Measured.**

* 83 more kernel files carry an `amd-cdna3` schedule (a copy of their `nvidia-ada` one), 159 schedules in all. **454 cases of `tests/cases.txt` now record a `.hip` file, 298 more than after H1**; 286 of the 454 files name a shuffle. `golden/` changed by those 298 new files and by 119 `amd-cdna3:` lines of the interpreter added to 99 `RUN.txt` files; no line was removed and no `.cu` or `.metal` file changed.
* **Under the emulator at 64 lanes (`tests/hip-emu.sh`, in the gate): 399 agree with the interpreter, 55 have no reference, none disagrees**, no case draws a warning and no case's two schedules differ. The worst difference is 7.1e-4 of the peak (`vit_ln`, s=1024 n=4608), the same figure the CUDA text of that case gives. The run takes about three and a half minutes on eight jobs here.
* Two more mutants are refused: the shuffling `rmsnorm` without its barriers (`device-mismatch`), and with its mask narrowed back to 32 bits (the upper half of a wave is not named).

**What still has no HIP text.** The cases that use `mma` (H5); `examples/cnn/bn_fin.chx`, whose schedule of 32 threads is less than one 64-lane wave; five cases that are recorded refusals (their CUDA text is in `golden/` only because it was written before another target refused the file); and the kernel files that no case of `tests/cases.txt` exercises, which were given no schedule because nothing here would check it.

**Not done.** `cancho-gpu run --target` and a session still name only the two older targets. The 55 cases without a reference were not run at full size.

## 9. H5, built

`mma` lowers for `amd-cdna3` as 3.3 decided, and by the smallest change that does it: the HIP text is the CUDA text with `nvcuda::wmma` renamed to `cgm`, a namespace the kernel's own preamble defines (`dialect.matrix_includes`, 23 lines of HIP). It gives the calls the lowering already writes (`fragment`, `fill_fragment`, `load_matrix_sync`, `mma_sync`, `store_matrix_sync`, `x[]` and `num_elements`) on plain registers. A 16 x 16 tile belongs to 32 consecutive threads, as under CUDA; thread *l* of them holds row *l*/2 and columns 8(*l* mod 2) to +7 of an accumulator (8 floats), that row of an `a` (16 floats) and those columns of a `b` (128 floats), so its share of a product is 8 sums of 16 multiply-adds in its own registers, with no shuffle and no barrier. A warp of a schedule's grid is those 32 threads, so a wave of 64 holds two, and the grid check counts 32 a warp for this target.

**Measured.** 13 more kernel files carry an `amd-cdna3` schedule (172 schedules in all) and their **29 cases record a `.hip` file: 483 cases in all**, every case of `tests/cases.txt` that records CUDA text except `bn_fin` and the five recorded refusals of section 8. `golden/` gained those 29 files and 9 interpreter lines in 8 `RUN.txt` files; nothing else changed. **Under the emulator at 64 lanes: 424 of the 483 agree with the interpreter, 59 have no reference, none disagrees** (the CUDA text's own figures are 430 and 59 of 489); of the 29 matrix cases 25 agree and 4 have no reference, the worst at 7.8e-5 of the peak. Two more mutants are refused: the product with a subtraction for its addition, and a thread taking the wrong columns.

**After `main` was merged in (2026-11-25).** `main` had gained kernels for speech while these slices were open; seven of their files were given an `amd-cdna3` schedule (179 in all) and one existing kernel had gained a case. The figures on the merged tree: **500 cases record a `.hip` file, and under the emulator 441 agree with the interpreter, 59 have no reference, none disagrees.** The figures above and in sections 5, 7 and 8 are those of each slice when it was built.

**What it costs, reckoned and not measured.** A thread reads 144 values to make 8 where `wmma` reads 16 for 8 (each value of `b` is read by 16 threads, of `a` by 2), and it holds 152 floats a tile pair in registers. So the matrix products on an AMD card will be slow by this lowering, by an amount only H6 can state; the card's own matrix units stay unreached (3.3).

**Not done.** `mma` of an f32 operand is still refused for this target with CUDA's message, though this lowering would not need the refusal. The four cases without a reference were not run at full size.

## 10. H6, prepared and not run

The run on a card is two scripts, written and tested without a card and without an account. **Nothing was rented, no `doctl` was run, and no credit was activated.**

* `scripts/amd/remote_hip.sh` runs on any machine with an AMD GPU and ROCm. Its steps: the machine (ROCm's version, the card, the two libraries); the toolchain (the install functions of `scripts/gcp/remote_cuda.sh`, read as a library: one path for both vendors); **`hiprtc`**, every golden `.hip` through the real hipRTC (`scripts/amd/hiprtc_check.c`), with how many of the failures name `__half`; **`wave`**, one hand-written kernel on the card (`tests/emu/wave_selftest.c` with `real`), which prints the card's `warpSize` and says whether a shuffle there is the one section 7's emulator models; **`device_cases`**, every case that emits HIP through the card against the interpreter, with the launch's arguments by `kernelParams` and by `extra` if the first case fails that way. `results/answers.txt` gathers what section 1 marks unconfirmed.
* `scripts/amd/hip_test.sh` creates one `gpu-mi300x1-192gb` Droplet with `doctl`, ships `git archive HEAD`, runs the script above over ssh, brings `results/` home and deletes the Droplet. `DRY_RUN=1` prints every command and runs none.

**Read on 2026-11-25, in DigitalOcean's documentation, and never used here:** the size slug `gpu-mi300x1-192gb`; the MI300X on demand in Atlanta (ATL1) only, for which `atl1` is this document's guess at the slug; $2.59 an hour, billed by the second with a minimum of 60 seconds; "Powering off a GPU Droplet does not stop billing. To end billing, destroy the GPU Droplet."; `doctl compute droplet list` shows GPU Droplets only with `--gpus`. The image is **[unconfirmed]**: one page names `gpu-amd-base` (Ubuntu 24.04, ROCm 10.0) and is the newest, another names `digitaloceanai-rocmjupyter` (ROCm 6.4.0), a third says Ubuntu 22.04; the script takes `IMAGE`, and `doctl compute image list-distribution` on the account is what settles it. Whether an account needs approval or a quota for a GPU Droplet was not found.

**The guard rails, and where they end.** DigitalOcean has no deletion after a time limit of its own, unlike Google's `--max-run-duration`. So every rail is the launcher's process: the Droplet is deleted on exit, on a failed run and on SIGINT, SIGTERM and SIGHUP, found by name among the Droplets tagged `cancho-gpu-test`, and a watchdog deletes it after `MAX_RUN` minutes (90). **If that process is killed outright, or its machine sleeps or loses power, nothing deletes the Droplet and it bills at $2.59 an hour until someone does**: `scripts/amd/hip_test.sh sweep` lists what is left and `DELETE=1` deletes it. At the reported $100 of credit that is about 38 hours, and a forgotten Droplet spends all of it in a day and a half.

**Checked** (`tests/amd-cloud.sh`, in the gate): the launcher against a stub `doctl`, `ssh` and `scp` (a dry run runs nothing; create, run, delete in that order; deletion when the remote run fails, on SIGTERM and at the time cap; refusal before anything is created without a key or with a bad cap; `sweep`), and `remote_hip.sh` with the emulator's libraries in place of AMD's (3 kernels compile, the wave is 64 lanes, 3 cases agree). **What that cannot show:** that DigitalOcean accepts these commands, that the image has what the script expects (`/opt/rocm`, a C compiler, `apt`), that the toolchain installs there, or anything about hipRTC. The first run will be spent on those, which is why its first invocation should be `STEPS='machine toolchain' MAX_RUN=20`.

**What the first run costs** is not known: no run of this length was timed on any card here. The cap bounds it: `MAX_RUN=90` is at most $3.89 an attempt at the price read.

## 11. What this does not claim

No AMD GPU has run a kernel of this project, no AMD compiler has read the HIP text of sections 5, 8 and 9 (a CPU compiler has), and the shim of section 6 has called only a mock and an emulator. The dialect's differences are read from documentation, not observed. Speed on AMD is unknown, and without the matrix units (3.3) the matrix products will be slower than the card allows. Radeon and the Ryzen AI processors (32-lane waves, where the CUDA width holds) are not a target here: a row for them is one line once H1 to H3 exist, and a machine to run it on is what is missing.
