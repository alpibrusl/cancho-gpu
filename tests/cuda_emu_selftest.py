#!/usr/bin/env python3
"""The emulator against kernels written to be wrong (docs/device.md section
5b): the emulator is only worth its green results if it can go red. Each case
here is a small CUDA kernel run through build/emu/libnvrtc_emu.so and
libcuda_emu.so with the driver API, expecting either the right numbers or a
specific finding -- a race between schedules, an uninitialised shared read,
an overrun, a deadlock, a misaligned access, a wmma contract broken.

    scripts/build-emu.sh && python3 tests/cuda_emu_selftest.py
"""
import struct
import sys
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "emu"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emupy import compile_kernel, run, Result  # noqa: E402


def f32s(b):
    return list(array("f", b))


def bytes_f32(xs):
    return array("f", xs).tobytes()


HDR = "#include <cuda_fp16.h>\n#include <mma.h>\ntypedef unsigned int uint;\n"
tests = []


def case(f):
    tests.append(f)
    return f


@case
def warp_reduce_and_barrier_are_right():
    src = HDR + """
extern "C" __global__ void k(const float* __restrict__ x, float* __restrict__ y) {
    __shared__ float part[8];
    const uint tid = threadIdx.x;
    float v = x[blockIdx.x * 256 + tid];
    for (uint o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (tid % 32 == 0) part[tid / 32] = v;
    __syncthreads();
    if (tid == 0) { float s = 0; for (uint i = 0; i < 8; i++) s += part[i]; y[blockIdx.x] = s; }
}"""
    xs = [float(i % 7) for i in range(512)]
    r, outs = run(src, "k", (2, 1), 256, [bytes_f32(xs), bytes_f32([0, 0])])
    assert r.rc == 0 and not r.stderr, (r.err, r.stderr)
    assert f32s(outs[1]) == [sum(xs[:256]), sum(xs[256:])], f32s(outs[1])


@case
def shuffles_follow_the_documented_semantics():
    src = HDR + """
extern "C" __global__ void k(float* __restrict__ o) {
    const uint t = threadIdx.x;
    float v = (float)t;
    o[0 * 32 + t] = __shfl_down_sync(0xffffffffu, v, 3);
    o[1 * 32 + t] = __shfl_down_sync(0xffffffffu, v, 3, 8);
    o[2 * 32 + t] = __shfl_sync(0xffffffffu, v, 5);
    o[3 * 32 + t] = __shfl_sync(0xffffffffu, v, 5, 8);
    o[4 * 32 + t] = __shfl_xor_sync(0xffffffffu, v, 1);
    o[5 * 32 + t] = __shfl_up_sync(0xffffffffu, v, 2);
}"""
    r, outs = run(src, "k", (1, 1), 32, [bytes_f32([0.0] * 192)])
    assert r.rc == 0, r.err
    o = f32s(outs[0])
    for t in range(32):
        seg8 = t // 8 * 8
        assert o[t] == (t + 3 if t + 3 < 32 else t)
        assert o[32 + t] == (t + 3 if t + 3 < seg8 + 8 else t)
        assert o[64 + t] == 5
        assert o[96 + t] == seg8 + 5
        assert o[128 + t] == (t ^ 1)
        assert o[160 + t] == (t - 2 if t - 2 >= 0 else t)


@case
def a_missing_barrier_is_a_race():
    src = HDR + """
extern "C" __global__ void k(float* __restrict__ y) {
    __shared__ float s[256];
    const uint t = threadIdx.x;
    s[t] = (float)t;
    /* __syncthreads() is missing */
    y[t] = s[255 - t];
}"""
    r, _ = run(src, "k", (1, 1), 256, [bytes_f32([0.0] * 256)])
    assert "emu-race" in r.stderr and "seed 1" in r.stderr, (r.rc, r.stderr)


@case
def the_same_kernel_with_the_barrier_is_clean():
    src = HDR + """
extern "C" __global__ void k(float* __restrict__ y) {
    __shared__ float s[256];
    const uint t = threadIdx.x;
    s[t] = (float)t;
    __syncthreads();
    y[t] = s[255 - t];
}"""
    r, outs = run(src, "k", (1, 1), 256, [bytes_f32([0.0] * 256)])
    assert r.rc == 0 and not r.stderr, (r.err, r.stderr)
    assert f32s(outs[0]) == [255.0 - t for t in range(256)]


@case
def an_uninitialised_shared_read_is_found():
    src = HDR + """
extern "C" __global__ void k(float* __restrict__ y) {
    __shared__ float s[64];
    const uint t = threadIdx.x;
    if (t < 32) s[t] = 1.0f;
    __syncthreads();
    y[t] = s[t];   /* t >= 32 reads what nobody wrote */
}"""
    r, _ = run(src, "k", (1, 1), 64, [bytes_f32([0.0] * 64)])
    assert "emu-race" in r.stderr, (r.rc, r.stderr)


@case
def a_write_past_the_end_is_an_illegal_access():
    src = HDR + """
extern "C" __global__ void k(float* __restrict__ y) {
    y[threadIdx.x] = 1.0f;   /* 65 threads over 64 floats */
}"""
    r, _ = run(src, "k", (1, 1), 65, [bytes_f32([0.0] * 64)])
    assert r.rc != 0 and "past the end of buffer #" in r.err, (r.rc, r.err)


@case
def a_write_before_the_start_is_found():
    src = HDR + """
extern "C" __global__ void k(float* __restrict__ y) {
    if (threadIdx.x == 0) y[-1] = 1.0f;
}"""
    r, _ = run(src, "k", (1, 1), 32, [bytes_f32([0.0] * 64)])
    assert r.rc != 0 and "before" in r.err, (r.rc, r.err)


@case
def a_barrier_nobody_else_reaches_is_a_deadlock():
    src = HDR + """
extern "C" __global__ void k(float* __restrict__ y) {
    float v = (float)threadIdx.x;
    if (threadIdx.x < 16) v += __shfl_down_sync(0xffffffffu, v, 1);   /* waits for lanes 16..31 */
    else __syncthreads();                                              /* which wait here */
    y[threadIdx.x] = v;
}"""
    r, _ = run(src, "k", (1, 1), 32, [bytes_f32([0.0] * 32)])
    assert r.rc != 0 and "deadlock" in r.err, (r.rc, r.err)


@case
def a_misaligned_vector_load_traps():
    src = HDR + """
extern "C" __global__ void k(const float* __restrict__ x, float* __restrict__ y) {
    const uint4 v = *reinterpret_cast<const uint4*>(&x[1]);   /* 4 bytes in */
    y[0] = (float)v.x;
}"""
    r, _ = run(src, "k", (1, 1), 1, [bytes_f32([0.0] * 16), bytes_f32([0.0] * 4)])
    assert r.rc != 0 and ("misaligned" in r.err), (r.rc, r.err)


@case
def a_shuffle_in_divergent_code_with_a_full_mask_is_flagged():
    src = HDR + """
extern "C" __global__ void k(float* __restrict__ y) {
    float v = 1.0f;
    if (threadIdx.x < 8) v = __shfl_down_sync(0xffffffffu, v, 1);
    y[threadIdx.x] = v;
}"""
    r, _ = run(src, "k", (1, 1), 32, [bytes_f32([0.0] * 32)])
    # Lanes 8..31 exit without arriving: legal on the hardware only by luck.
    assert "emu-warn" in r.stderr and "exited" in r.stderr, (r.rc, r.err, r.stderr)


@case
def a_partial_warp_with_a_full_mask_is_flagged():
    src = HDR + """
extern "C" __global__ void k(float* __restrict__ y) {
    y[threadIdx.x] = __shfl_down_sync(0xffffffffu, 1.0f, 1);
}"""
    r, _ = run(src, "k", (1, 1), 16, [bytes_f32([0.0] * 16)])
    assert "emu-warn" in r.stderr and "last warp" in r.stderr, (r.rc, r.err, r.stderr)


def half_bits(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


@case
def half_conversion_is_round_to_nearest_even():
    src = HDR + """
extern "C" __global__ void k(const float* __restrict__ x, unsigned short* __restrict__ y) {
    __half h = __half(x[threadIdx.x]);
    y[threadIdx.x] = *reinterpret_cast<unsigned short*>(&h);
}"""
    xs = [1.0, 1.0 + 2**-11, 1.0 + 3 * 2**-11, 65519.0, 65520.0, 6.1e-5, 1e-8, -0.1, 3.14159274, 2049.0, 2051.0]
    r, outs = run(src, "k", (1, 1), len(xs), [bytes_f32(xs), bytes(2 * len(xs) + (-2 * len(xs)) % 16)])
    assert r.rc == 0, r.err
    got = list(struct.unpack("<%dH" % len(xs), outs[1][: 2 * len(xs)]))
    want = []
    for x in xs:
        try:
            want.append(half_bits(x))
        except OverflowError:
            want.append(0x7C00)
    assert got == want, (got, want)


def wmma_kernel(la, lb):
    # D[16x16] = A[16x16] * B[16x16] with A loaded as `la` and B as `lb`.
    return HDR + """
extern "C" __global__ void k(const __half* __restrict__ a, const __half* __restrict__ b, float* __restrict__ d) {
    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, __half, %s> fa;
    fragment<matrix_b, 16, 16, 16, __half, %s> fb;
    fragment<accumulator, 16, 16, 16, float> fc;
    fill_fragment(fc, 0.0f);
    load_matrix_sync(fa, a, 16);
    load_matrix_sync(fb, b, 16);
    mma_sync(fc, fa, fb, fc);
    store_matrix_sync(d, fc, 16, mem_row_major);
}""" % (la, lb)


def halves(xs):
    return b"".join(struct.pack("<e", x) for x in xs)


@case
def wmma_layouts_follow_the_documentation():
    import random
    rnd = random.Random(7)
    A = [[float(rnd.randint(-4, 4)) for _ in range(16)] for _ in range(16)]
    B = [[float(rnd.randint(-4, 4)) for _ in range(16)] for _ in range(16)]
    flat = lambda M: [x for row in M for x in row]
    T = lambda M: [list(c) for c in zip(*M)]
    for la in ("row_major", "col_major"):
        for lb in ("row_major", "col_major"):
            # memory holds the matrix in its own layout: col_major is the transpose, row by row
            ma = flat(A) if la == "row_major" else flat(T(A))
            mb = flat(B) if lb == "row_major" else flat(T(B))
            r, outs = run(wmma_kernel("nvcuda::wmma::" + la, "nvcuda::wmma::" + lb), "k", (1, 1), 32,
                          [halves(ma), halves(mb), bytes(1024)])
            assert r.rc == 0 and not r.stderr, (la, lb, r.err, r.stderr)
            want = flat([[sum(A[i][k] * B[k][j] for k in range(16)) for j in range(16)] for i in range(16)])
            assert f32s(outs[2]) == want, (la, lb)


@case
def wmma_with_a_misaligned_pointer_is_refused():
    src = wmma_kernel("nvcuda::wmma::row_major", "nvcuda::wmma::row_major").replace("load_matrix_sync(fa, a, 16)", "load_matrix_sync(fa, a + 8, 16)")
    r, _ = run(src, "k", (1, 1), 32, [halves([0.0] * 512), halves([0.0] * 256), bytes(1024)])
    assert r.rc != 0 and "256-bit" in r.err, (r.rc, r.err)


@case
def wmma_called_by_part_of_a_warp_is_found():
    src = wmma_kernel("nvcuda::wmma::row_major", "nvcuda::wmma::row_major").replace("    mma_sync(fc, fa, fb, fc);", "    if (threadIdx.x < 16) mma_sync(fc, fa, fb, fc);")
    r, _ = run(src, "k", (1, 1), 32, [halves([0.0] * 256), halves([0.0] * 256), bytes(1024)])
    assert r.rc != 0 or "emu-warn" in r.stderr, (r.rc, r.err, r.stderr)


@case
def an_ill_formed_kernel_gets_nvrtcs_log_format():
    rc, log, _ = compile_kernel(HDR + 'extern "C" __global__ void k(float* y) {\n  y[0] = nope;\n}\n')
    assert rc != 0 and "kernel.cu(5): error:" in log and "nope" in log, (rc, log)


@case
def static_shared_over_48k_is_refused_at_compile_time():
    src = HDR + 'extern "C" __global__ void k(float* y) {\n  __shared__ float s[12289];\n  s[threadIdx.x] = 1.0f; __syncthreads(); y[0] = s[3];\n}\n'
    rc, log, _ = compile_kernel(src)
    assert rc != 0 and "too much shared data" in log, (rc, log)
    ok = src.replace("12289", "12288")
    rc, log, _ = compile_kernel(ok)
    assert rc == 0, log


@case
def launch_limits_are_enforced():
    src = HDR + 'extern "C" __global__ void k(float* y) { y[0] = 1.0f; }\n'
    r, _ = run(src, "k", (1, 1), 1025, [bytes(16)])
    assert r.rc != 0 and "1024" in r.err, (r.rc, r.err)
    r, _ = run(src, "k", (1, 65536), 32, [bytes(16)])
    assert r.rc != 0 and "65535" in r.err, (r.rc, r.err)


@case
def a_grid_runs_every_block_with_its_own_shared_memory():
    src = HDR + """
extern "C" __global__ void k(float* __restrict__ y) {
    __shared__ float s;
    if (threadIdx.x == 0) s = (float)(blockIdx.x + 10 * blockIdx.y);
    __syncthreads();
    if (threadIdx.x == 1) y[blockIdx.y * gridDim.x + blockIdx.x] = s;
}"""
    r, outs = run(src, "k", (4, 3), 32, [bytes_f32([-1.0] * 12)])
    assert r.rc == 0 and not r.stderr, (r.err, r.stderr)
    assert f32s(outs[0]) == [float(x + 10 * y) for y in range(3) for x in range(4)]


@case
def the_static_checks_can_fail():
    import tempfile
    import cuda_static
    good = """// Generated by cancho-gpu from a .chx program. Do not edit by hand.
// kernel : k
// target : nvidia-ada
// launch : 1 threadgroups x 256 threads
// tg mem : 0 B tiles + 32 B scratch of 101376 B

#include <cuda_fp16.h>
extern "C" __global__ void k(
    const float* __restrict__ p0_x
)
{
    __shared__ float scratch[8];
}
"""
    def findings(src):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "case" / "k.cu"
            f.parent.mkdir()
            f.write_text(src)
            return cuda_static.check(f)[2]
    assert findings(good) == [], findings(good)
    bad = {
        "static __shared__": good.replace("scratch[8]", "scratch[13000]"),
        "threads per block": good.replace("x 256 threads", "x 2048 threads"),
        "header says": good.replace("0 B tiles + 32 B scratch", "0 B tiles + 64 B scratch"),
        "inline PTX": good.replace("__shared__ float scratch[8];", '__shared__ float scratch[8]; asm volatile("bar.sync 0;");'),
        "wgmma": good.replace("__shared__ float scratch[8];", "__shared__ float scratch[8]; wgmma.mma_async();"),
        "includes <cuda_bf16.h>": good.replace("cuda_fp16.h", "cuda_bf16.h"),
        "not a __restrict__ pointer": good.replace("const float* __restrict__ p0_x", "float p0_x"),
    }
    for want, src in bad.items():
        got = findings(src)
        assert any(want in g for g in got), (want, got)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    bad = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"{len(tests) - bad} of {len(tests)} emulator self-tests pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
