# AMD: a HIP target (design, 2026-11-24; H1 built: `emit` writes HIP text for 156 cases, never compiled; nothing run on AMD hardware)

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

1. **The width of a simdgroup is the literal 32** in `src/lower.cho` (the reductions, the warp grids, the mask `0xffffffffu` in `src/dialect.cho`). On gfx9 a wave has 64 lanes. A kernel whose threads only *group themselves* by `tid / 32u` and meet at `__syncthreads` does not care how wide the hardware's wave is; a kernel that shuffles does: `__shfl_down_sync(m, v, 16)` on a 64-lane wave pairs lane 16 with lane 0 as intended, and lane 32 with lane 16 where the lowering meant lane 32 to be lane 0 of the next simdgroup. **[reckoned]** 293 files.
2. **The matrix unit has no name in HIP.** 29 files. The product has to be lowered another way (section 3.3).
3. **Halves.** 178 files, all through `<cuda_fp16.h>`. Unconfirmed either way; if `__half` is missing, the dialect spells it `_Float16` (clang's, which hipRTC's compiler is) with explicit conversions.

169 files (35% of the CUDA text) need none of 1 and 2.

## 3. Decisions

### 3.1 A third dialect, not the CUDA text

Issue #9 offered two ways: reuse `nvidia-ada`'s CUDA C through ROCm's compatibility, or a HIP dialect. **A dialect**, `dialect.hip()`, for the reasons section 2 counts: 60% of the CUDA text shuffles at width 32 and would be *silently wrong* on a 64-lane wave, which is the one failure this project does not accept. The dialect is CUDA's spelling (most of the 35 places where `src/dialect.cho` asks `is_cuda` are shared) and differs where section 1 says the language does. `amd-cdna3` emits `<kernel>.hip`.

### 3.2 The width comes from the target row

`target.simd_width` is already in the table (32, 32, 32, 64) and the checker already reads it for the budget. The lowering reads it too, in place of the literal: lanes per simdgroup, the halving steps of a reduction (a 64-lane tree has one more step), the shuffle mask (`0xffffffffffffffffull` at 64). For the two existing targets the emitted text must not change by a byte; the goldens of CI are what holds that. This is the largest change of the slice list and the one most likely to be wrong, which is why it comes after an emulator can run its output (H3 before H4).

### 3.3 The matrix product without `wmma`

`mma` on `amd-cdna3` lowers to register accumulators and plain multiply-adds per lane: the `frag` tiles become per-thread `float` arrays, the product a loop the compiler vectorises as it can. It will be slower than the card's matrix units by an amount nobody here can state; it is correct, and it makes the 29 cases run. Reaching the matrix units (through inline assembly or a vendor intrinsic, if hipRTC's compiler accepts one) is a later question that only a card can answer.

### 3.4 The shim is the CUDA shim with other names

`shim/canchogpu.c` holds a table of fourteen driver functions and seven NVRTC functions behind `dlopen`. HIP's have the same shapes (section 1). A second table in the same file, chosen at `cgpu_open` by which library opens (`CANCHOGPU_LIBHIP`, `CANCHOGPU_LIBHIPRTC`), not a second shim: the handle bookkeeping, the argument list and the error path are the 380 lines that were hard to get right, and `docs/device.md` section 1's entry points do not change. If `kernelParams` is refused by the real library, the shim packs the arguments into `extra`; the mock accepts both and logs which.

### 3.5 Checked without a card, as CUDA was

`docs/device.md` sections 5 and 5b, repeated: a mock `libamdhip64`/`libhiprtc` for the plumbing, and the emulator of `tests/emu/` run **at 64 lanes** for the arithmetic. The emulator's warp width becomes a parameter; at 64 with the *CUDA* text it must report the wrong pairing of section 2 item 1 on a shuffling kernel, which is the test that the emulator can see the bug the dialect exists to avoid. The emulator found two real bugs in the CUDA path before any card did; it is the reason to expect the first hour on an MI300X to be spent on things section 1 marks unconfirmed and not on the lowering.

## 4. Slices

| | what | done when |
|---|---|---|
| H1 | `dialect.hip()`, `amd-cdna3` on the command line, `emit` writes `.hip` for a kernel that neither shuffles nor multiplies matrices; any other is refused `error[lowering]` naming what the target lacks | goldens recorded for the 169; the other two targets' goldens unchanged |
| H2 | the HIP table in the shim, the mock libraries, `tests/device.sh` for the target | the mock's log agrees with each program; no library gives `error[device]` |
| H3 | the emulator at a width from the target; the test that 64 lanes break the CUDA text | H1's cases agree with the interpreter under the emulator |
| H4 | the lowering's width from the target row: reductions and shuffles at 64 | the 293 emit and agree under the emulator at 64; the two targets' goldens unchanged |
| H5 | the product without `wmma` | the 29 emit and agree under the emulator |
| H6 | the first run on an MI300X: hipRTC compiles every golden `.hip`, the device cases against the interpreter, the unconfirmed items of section 1 answered in this document | a `results/` directory and this document corrected in place |
| H7 | a model on the card (Qwen3.8-27B through `cancho-gpu-serve`), with its speed | after H6, and only then is a number published |

H1 to H5 need no AMD hardware and no credit. H6 is one script on the pattern of `scripts/gcp/cuda_test.sh` (`docs/cloud.md`), with its cost guard rails.

## 5. H1, built

`cancho-gpu emit` lowers a file's `amd-cdna3` schedule to `<dir>/<kernel>.hip`, and `check` checks it. The dialect (`dialect.hip()` of `src/dialect.cho`) is CUDA's spelling with a different preamble: `#include <hip/hip_fp16.h>`, the infinity as clang's `__builtin_inff()` (hipRTC has no standard headers), and the NVFP4 table as `static const` where CUDA says `__constant__`. After lowering, `src/lower.cho` refuses for this target a kernel that uses `mma` and one whose text shuffles, each `error[lowering]` saying which slice lifts it (`tests/reject/lowering-mma-hip.chx`, `tests/reject/lowering-shuffle-hip.chx`).

**Measured.** 76 kernel files of `tests/chx/` and `examples/` carry an `amd-cdna3` schedule, a copy of their `nvidia-ada` one, and their 156 cases of `tests/cases.txt` each record a `.hip` file in `golden/`: none names a shuffle or `wmma`, 42 name `__half`. The other goldens changed only by gaining the interpreter's `amd-cdna3:` lines in 70 `RUN.txt` files (92 lines added, none removed); no `.cu` or `.metal` file changed.

**Correction to section 2's 169.** 156 of them are recorded, not 169. Section 2 counted cases; a schedule belongs to a kernel file, and `emit` refuses the whole file when one target is refused, so a file gets a schedule only if *every* one of its cases can be lowered. 12 of the 169 are cases of files that shuffle at other sizes, and one (`examples/cnn/bn_fin.chx`) has a schedule of 32 threads, which is less than one 64-lane wave. All 13 wait for H4.

**Not done in H1.** The text has not been through any compiler: no hipRTC is here, and the emulator learns HIP in H3. `cancho-gpu run --target` and a session still name only the two older targets.

## 6. What this does not claim

No AMD GPU has run a kernel of this project, and no compiler has read the HIP text of section 5. The dialect's differences are read from documentation, not observed. Speed on AMD is unknown, and without the matrix units (3.3) the matrix products will be slower than the card allows. Radeon and the Ryzen AI processors (32-lane waves, where the CUDA width holds) are not a target here: a row for them is one line once H1 to H3 exist, and a machine to run it on is what is missing.
