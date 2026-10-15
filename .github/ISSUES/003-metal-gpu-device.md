# Issue #3: Metal GPU Device Implementation (Slice 10, Issue #8)

## Summary
Implement Metal GPU device support for cancho-gpu to enable running kernels on Apple Silicon GPUs. This is slice 10 of the roadmap and addresses issue alpibrusl/cancho-gpu#8.

## Background
The current cancho-gpu implementation supports CUDA GPUs through `shim/canchogpu.c` and the `device/` directory. We need to add equivalent Metal GPU support for Apple Silicon Macs.

## Tasks

### 1. Metal Shim Implementation (`shim/canchogpu-metal.c`)
- [ ] Create `shim/canchogpu-metal.c` following the same pattern as `shim/canchogpu.c`
- [ ] Implement Metal device initialization: `MTLCreateSystemDefaultDevice`
- [ ] Implement Metal shader compilation: `newLibraryWithSource` for emitted .metal source
- [ ] Implement compute pipeline creation: `newComputePipelineState`
- [ ] Implement buffer management: `newBufferWithLength` (alloc), `didChangeRange` (upload), `getBytesContents` (download)
- [ ] Implement kernel dispatch: `dispatchThreadgroups`, `synchronize`
- [ ] Follow the same interface pattern: integer handles in, int64 status out, at most one trailing byte buffer

### 2. Device Directory Extension
- [ ] Extend `device/` to support `apple-m-series` target
- [ ] Mirror the CUDA path structure for Metal
- [ ] Add Metal-specific device initialization and kernel execution
- [ ] Ensure the same interface as CUDA: run kernels and compare against CPU interpreter

### 3. Mock Testing
- [ ] Create mock Metal framework shim for testing (following `tests/device.sh` pattern)
- [ ] Verify mock implementation works before real GPU testing

### 4. Real GPU Testing
- [ ] Test with real Metal GPU: `cancho-gpu-device run tests/chx/softmax.chx r=4 n=64`
- [ ] Test other cases: rmsnorm, gemm_fp4, silu_mul
- [ ] Compare against CPU interpreter using `device-mismatch` on any difference

### 5. Performance Measurement
- [ ] Measure kernel wall time vs CPU interpreter
- [ ] Record numbers in `docs/device.md`
- [ ] Correct any false claims in documentation

## Validation
- [ ] Run the full gate: `CANCHO=<path> scripts/gate.sh`
- [ ] Expected: "140 of 140 golden cases agree", "gate: ok"

## PR #11 Validation
- [ ] Run new softmax kernel on Metal GPU across case sizes (r=1..16, n=8..256)
- [ ] Compare against interpreter's f32 output with same tolerance as docs/device.md
- [ ] Report results as PR comments on #11
- [ ] Update PR status to ready-for-review when oracle patch lands in lex-gpu

## Deliverables
- [ ] `shim/canchogpu-metal.c` + its mock
- [ ] `device/` extension for apple-m-series target
- [ ] Updated `docs/device.md` with real measurement numbers
- [ ] PR comments on #11 with validation results

## Dependencies
- Requires PR #11 to be merged (exp/log unops support)
- Requires lex-gpu oracle patch to be merged and goldens regenerated

## Rules Compliance
- Design before code: update `docs/device.md` with Metal-specific design
- Goldens regenerated only from Rust (scripts/regen-golden.sh)
- No source file over 2,000 lines
- Run `scripts/gate.sh` before calling anything done
- cancho-gpu links only libc; device binary may call shim

## Related Issues
- alpibrusl/cancho-gpu#8 (First real Metal run)
- alpibrusl/cancho-gpu#11 (exp/log unops support)
- alpibrusl/cancho#251 (cancho LLM-serving stack epic)