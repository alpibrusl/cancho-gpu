// What limits an NVFP4 matrix-vector product on Apple GPUs (issue #24).
//
//     clang++ -O2 -fobjc-arc -framework Metal -framework Foundation bw.mm -o bw && ./bw
//
// Hand-written Metal kernels over the 17408x5120 gate projection of Qwen3.8-27B
// (random weights): the shape the emitter's kernel runs in ~200 us. Every
// variant is checked against a CPU reference and timed over many launches in
// one command buffer. `copy` is the ceiling: the same bytes summed.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string.h>

static const int N = 17408, K = 5120;

static const char *SRC = R"MSL(
#include <metal_stdlib>
using namespace metal;
constant uint N = 17408, K = 5120;
inline float2 fp4_pair(uint b) {
    const uint w = b | (b << 12u);
    return float2(as_type<half2>(((w & 0x00070007u) << 9u) | ((w & 0x00080008u) << 12u)));
}
inline float fp8_e4m3(uint b) {
    const uint e = (b >> 3u) & 0xFu, m = b & 7u;
    const uint bits = select(((e + 120u) << 23u) | (m << 20u), as_type<uint>(float(m) * 0.001953125f), e == 0u);
    return as_type<float>(bits | ((b & 0x80u) << 24u));
}
// one 32-bit word of codes (8 codes = 4 bytes) against 8 x values
inline float dot8(uint w, float4 xa, float4 xb) {
    float2 p0 = fp4_pair(w & 0xFFu), p1 = fp4_pair((w >> 8) & 0xFFu);
    float2 p2 = fp4_pair((w >> 16) & 0xFFu), p3 = fp4_pair(w >> 24);
    return xa.x * p0.x + xa.y * p0.y + xa.z * p1.x + xa.w * p1.y
         + xb.x * p2.x + xb.y * p2.y + xb.z * p3.x + xb.w * p3.y;
}

// A: the emitter's kernel: a simdgroup a row, a lane 16 values (8 bytes) a step, byte loads.
kernel void kA(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]],
               device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]],
               device float *y [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
               uint3 tg [[threadgroup_position_in_grid]]) {
    const uint o = tid / 32u, lane = tid % 32u, row = tg.x * 8u + o;
    float s = 0.0f;
    const float rs = wg[row];
    for (uint p0 = lane * 16u; p0 < K; p0 += 512u) {
        const float sg = fp8_e4m3((uint)(uchar)ws[row * (K / 16) + p0 / 16u]) * rs;
        float run = 0.0f;
        for (uint u = 0; u < 16u; u += 2u) {
            const uint bq = (uint)(uchar)wq[row * (K / 2) + p0 / 2u + u / 2u];
            const float2 w = fp4_pair(bq);
            run += x[p0 + u] * w.x + x[p0 + u + 1] * w.y;
        }
        s += run * (sg * 16384.0f);
    }
    for (uint d = 16u; d > 0; d /= 2) s += simd_shuffle_down(s, d);
    if (lane == 0) y[row] = s;
}

// B: the same, one uint2 load of weights and four float4 of x a step.
kernel void kB(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]],
               device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]],
               device float *y [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
               uint3 tg [[threadgroup_position_in_grid]]) {
    const uint o = tid / 32u, lane = tid % 32u, row = tg.x * 8u + o;
    float s = 0.0f;
    const float rs = wg[row];
    device const uint2 *q = (device const uint2 *)(wq + row * (K / 2));
    device const float4 *x4 = (device const float4 *)x;
    for (uint p0 = lane * 16u; p0 < K; p0 += 512u) {
        const float sg = fp8_e4m3((uint)(uchar)ws[row * (K / 16) + p0 / 16u]) * rs;
        const uint2 w = q[p0 / 16u];
        const float4 a = x4[p0 / 4u], b = x4[p0 / 4u + 1], c = x4[p0 / 4u + 2], d = x4[p0 / 4u + 3];
        s += (dot8(w.x, a, b) + dot8(w.y, c, d)) * (sg * 16384.0f);
    }
    for (uint d = 16u; d > 0; d /= 2) s += simd_shuffle_down(s, d);
    if (lane == 0) y[row] = s;
}

// C: a uint4 (32 values) a lane a step, scales as one uchar2, all loads of a step first,
// and U steps unrolled so that U * 16 bytes are in flight a lane.
template <int U>
inline void kernel_c(device const float *x, device const char *wq, device const char *ws,
                     device const float *wg, device float *y, uint tid, uint row) {
    const uint lane = tid % 32u;
    float s = 0.0f;
    const float rs = wg[row];
    device const uint4 *q = (device const uint4 *)(wq + row * (K / 2));
    device const float4 *x4 = (device const float4 *)x;
    constexpr uint STEP = 32u * 32u;   // values a simdgroup covers a step
    for (uint base = 0; base < K; base += STEP * U) {
        uint4 w[U];
        uchar2 sc[U];
        for (int u = 0; u < U; u++) {
            const uint p0 = base + u * STEP + lane * 32u;
            w[u] = q[p0 / 32u];
            sc[u] = *(device const uchar2 *)(ws + row * (K / 16) + p0 / 16u);
        }
        for (int u = 0; u < U; u++) {
            const uint p0 = base + u * STEP + lane * 32u;
            const float s0 = fp8_e4m3(sc[u].x) * rs * 16384.0f, s1 = fp8_e4m3(sc[u].y) * rs * 16384.0f;
            float4 xv[8];
            for (int i = 0; i < 8; i++) xv[i] = x4[p0 / 4u + i];
            s += s0 * (dot8(w[u].x, xv[0], xv[1]) + dot8(w[u].y, xv[2], xv[3]))
               + s1 * (dot8(w[u].z, xv[4], xv[5]) + dot8(w[u].w, xv[6], xv[7]));
        }
    }
    for (uint d = 16u; d > 0; d /= 2) s += simd_shuffle_down(s, d);
    if (lane == 0) y[row] = s;
}
kernel void kC1(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]],
                device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]],
                device float *y [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
                uint3 tg [[threadgroup_position_in_grid]]) {
    kernel_c<1>(x, wq, ws, wg, y, tid, tg.x * (256 / 32) + tid / 32);
}
kernel void kC5(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]],
                device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]],
                device float *y [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
                uint3 tg [[threadgroup_position_in_grid]]) {
    // K / (32 * 32) = 5 steps: all of a row's loads in flight at once
    kernel_c<5>(x, wq, ws, wg, y, tid, tg.x * (256 / 32) + tid / 32);
}
// D: C1 with x staged once a threadgroup in threadgroup memory (K floats = 20 KiB).
kernel void kD(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]],
               device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]],
               device float *y [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
               uint3 tg [[threadgroup_position_in_grid]]) {
    threadgroup float4 xs[K / 4];
    for (uint i = tid; i < K / 4; i += 256) xs[i] = ((device const float4 *)x)[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint lane = tid % 32u, row = tg.x * 8u + tid / 32u;
    float s = 0.0f;
    const float rs = wg[row];
    device const uint4 *q = (device const uint4 *)(wq + row * (K / 2));
    uint4 w[5];
    uchar2 sc[5];
    for (int u = 0; u < 5; u++) {
        const uint p0 = u * 1024u + lane * 32u;
        w[u] = q[p0 / 32u];
        sc[u] = *(device const uchar2 *)(ws + row * (K / 16) + p0 / 16u);
    }
    for (int u = 0; u < 5; u++) {
        const uint p0 = u * 1024u + lane * 32u;
        const float s0 = fp8_e4m3(sc[u].x) * rs * 16384.0f, s1 = fp8_e4m3(sc[u].y) * rs * 16384.0f;
        s += s0 * (dot8(w[u].x, xs[p0 / 4], xs[p0 / 4 + 1]) + dot8(w[u].y, xs[p0 / 4 + 2], xs[p0 / 4 + 3]))
           + s1 * (dot8(w[u].z, xs[p0 / 4 + 4], xs[p0 / 4 + 5]) + dot8(w[u].w, xs[p0 / 4 + 6], xs[p0 / 4 + 7]));
    }
    for (uint d = 16u; d > 0; d /= 2) s += simd_shuffle_down(s, d);
    if (lane == 0) y[row] = s;
}
// E: C5 with 4 rows a simdgroup (more independent work a thread), 2 simdgroups a threadgroup.
kernel void kE(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]],
               device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]],
               device float *y [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
               uint3 tg [[threadgroup_position_in_grid]]) {
    const uint lane = tid % 32u;
    for (uint r = 0; r < 4; r++) {
        const uint row = (tg.x * 8u + tid / 32u) * 4u + r;
        if (row < N) kernel_c<5>(x, wq, ws, wg, y, tid, row);
    }
}
// W: the text the emitter writes with wide loads on (src/matmul.cho), serial `run` chain.
kernel void kW(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]],
               device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]],
               device float *y [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
               uint3 tg [[threadgroup_position_in_grid]]) {
    const uint o = tid / 32u, lane = tid % 32u, row = tg.x * 8u + o;
    float s = 0.0f;
    const float rs = wg[row];
    for (uint p0 = lane * 16u; p0 < K; p0 += 512u) {
        const float sg = fp8_e4m3((uint)(uchar)ws[row * 320u + p0 / 16u]) * rs;
        const uint2 w = *(const device uint2 *)(&wq[row * 2560u + p0 / 2u]);
        const float4 x0 = *(const device float4 *)(&x[p0]), x1 = *(const device float4 *)(&x[p0 + 4u]);
        const float4 x2 = *(const device float4 *)(&x[p0 + 8u]), x3 = *(const device float4 *)(&x[p0 + 12u]);
        float run = 0.0f;
        { const float2 q = fp4_pair((w.x >> 0u) & 0xFFu); run += x0.x * q.x + x0.y * q.y; }
        { const float2 q = fp4_pair((w.x >> 8u) & 0xFFu); run += x0.z * q.x + x0.w * q.y; }
        { const float2 q = fp4_pair((w.x >> 16u) & 0xFFu); run += x1.x * q.x + x1.y * q.y; }
        { const float2 q = fp4_pair((w.x >> 24u) & 0xFFu); run += x1.z * q.x + x1.w * q.y; }
        { const float2 q = fp4_pair((w.y >> 0u) & 0xFFu); run += x2.x * q.x + x2.y * q.y; }
        { const float2 q = fp4_pair((w.y >> 8u) & 0xFFu); run += x2.z * q.x + x2.w * q.y; }
        { const float2 q = fp4_pair((w.y >> 16u) & 0xFFu); run += x3.x * q.x + x3.y * q.y; }
        { const float2 q = fp4_pair((w.y >> 24u) & 0xFFu); run += x3.z * q.x + x3.w * q.y; }
        s += run * (sg * 16384.0f);
    }
    for (uint d = 16u; d > 0; d /= 2) s += simd_shuffle_down(s, d);
    if (lane == 0) y[row] = s;
}
// R rows a simdgroup, the per-row code of A (byte loads) and of B (uint2 / float4):
// is it the rows a simdgroup, not the loads, that matter?
template <int R, bool WIDE>
inline void rows_r(device const float *x, device const char *wq, device const char *ws,
                   device const float *wg, device float *y, uint tid, uint tgx) {
    const uint lane = tid % 32u;
    for (uint r = 0; r < R; r++) {
        const uint row = (tgx * 8u + tid / 32u) * R + r;
        float s = 0.0f;
        const float rs = wg[row];
        for (uint p0 = lane * 16u; p0 < K; p0 += 512u) {
            const float sg = fp8_e4m3((uint)(uchar)ws[row * (K / 16) + p0 / 16u]) * rs;
            if (WIDE) {
                const uint2 w = *(device const uint2 *)(wq + row * (K / 2) + p0 / 2u);
                device const float4 *x4 = (device const float4 *)x;
                s += (dot8(w.x, x4[p0 / 4u], x4[p0 / 4u + 1]) + dot8(w.y, x4[p0 / 4u + 2], x4[p0 / 4u + 3])) * (sg * 16384.0f);
            } else {
                float run = 0.0f;
                for (uint u = 0; u < 16u; u += 2u) {
                    const uint bq = (uint)(uchar)wq[row * (K / 2) + p0 / 2u + u / 2u];
                    const float2 w = fp4_pair(bq);
                    run += x[p0 + u] * w.x + x[p0 + u + 1] * w.y;
                }
                s += run * (sg * 16384.0f);
            }
        }
        for (uint d = 16u; d > 0; d /= 2) s += simd_shuffle_down(s, d);
        if (lane == 0) y[row] = s;
    }
}
#define ROWS_KERNEL(NAME, R, WIDE) \
kernel void NAME(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]], \
                 device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]], \
                 device float *y [[buffer(4)]], uint tid [[thread_index_in_threadgroup]], \
                 uint3 tg [[threadgroup_position_in_grid]]) { rows_r<R, WIDE>(x, wq, ws, wg, y, tid, tg.x); }
ROWS_KERNEL(kA2, 2, false)
ROWS_KERNEL(kA4, 4, false)
ROWS_KERNEL(kB2, 2, true)
ROWS_KERNEL(kB4, 4, true)
ROWS_KERNEL(kB8, 8, true)
// F: the ceiling: sum the weight and scale bytes with uint4 loads, nothing else.
kernel void kF(device const char *wq [[buffer(1)]], device const char *ws [[buffer(2)]],
               device float *y [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
               uint3 tg [[threadgroup_position_in_grid]]) {
    const uint lane = tid % 32u, row = tg.x * 8u + tid / 32u;
    device const uint4 *q = (device const uint4 *)(wq + row * (K / 2));
    uint a = 0;
    uint4 w[5];
    for (int u = 0; u < 5; u++) w[u] = q[u * 32 + lane];
    for (int u = 0; u < 5; u++) a += w[u].x ^ w[u].y ^ w[u].z ^ w[u].w;
    a += (uint)ws[row * (K / 16) + lane];
    for (uint d = 16u; d > 0; d /= 2) a += simd_shuffle_down(a, d);
    if (lane == 0) y[row] = (float)a;
}
)MSL";

static float fp4v[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6};
static float e4m3(unsigned b) {
    unsigned e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? m * 0.001953125f : ldexpf(1 + m / 8.0f, (int)e - 7);
    return b & 0x80 ? -v : v;
}

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&err];
        if (!lib) { NSLog(@"%@", err); return 1; }
        size_t nq = (size_t)N * K / 2, ns = (size_t)N * K / 16;
        std::vector<unsigned char> wq(nq), ws(ns);
        std::vector<float> x(K), wg(N);
        srand(1);
        for (auto &b : wq) b = rand();
        for (auto &b : ws) b = 0x30 + (rand() & 0x1f);   // plausible E4M3 scales, no NaN
        for (auto &v : x) v = (rand() / (float)RAND_MAX) - 0.5f;
        for (auto &v : wg) v = 1.0f / 16384.0f;
        std::vector<float> ref(N);
        for (int r = 0; r < N; r++) {
            double acc = 0;
            for (int p = 0; p < K; p++) {
                unsigned byte = wq[(size_t)r * K / 2 + p / 2];
                unsigned code = (p & 1) ? byte >> 4 : byte & 15;   // low nibble first
                acc += x[p] * fp4v[code] * e4m3(ws[(size_t)r * K / 16 + p / 16]) * wg[r];
            }
            ref[r] = (float)acc;
        }
        id<MTLBuffer> bx = [dev newBufferWithBytes:x.data() length:K * 4 options:0];
        // 16 copies of the matrix, used in turn, so that no launch finds its weights in the
        // system cache (one copy is 50 MB).
        const int COPIES = 16;
        std::vector<id<MTLBuffer>> bqs, bss;
        for (int c = 0; c < COPIES; c++) {
            bqs.push_back([dev newBufferWithBytes:wq.data() length:nq options:0]);
            bss.push_back([dev newBufferWithBytes:ws.data() length:ns options:0]);
        }
        id<MTLBuffer> bg = [dev newBufferWithBytes:wg.data() length:N * 4 options:0];
        id<MTLBuffer> by = [dev newBufferWithLength:N * 4 options:0];
        // 24 distinct copies of the weights would not fit the caches either way; the
        // matrix is 47 MB, far past the system-level cache, so repeats read DRAM.
        const double bytes = (double)nq + ns;
        struct V { const char *name; const char *fn; int rowsPerTG; int threads; } vs[] = {
            {"A emitter (byte loads)", "kA", 8, 256}, {"B uint2/float4", "kB", 8, 256}, {"W emitter text, wide on", "kW", 8, 256}, {"A2 scalar, 2 rows/simdgroup", "kA2", 16, 256}, {"A4 scalar, 4 rows/simdgroup", "kA4", 32, 256}, {"B2 wide, 2 rows/simdgroup", "kB2", 16, 256}, {"B4 wide, 4 rows/simdgroup", "kB4", 32, 256}, {"B8 wide, 8 rows/simdgroup", "kB8", 64, 256},
            {"C1 uint4, 1 step", "kC1", 8, 256}, {"C5 uint4, 5 in flight", "kC5", 8, 256},
            {"D C5 + x in threadgroup mem", "kD", 8, 256}, {"E C5, 4 rows a simdgroup", "kE", 32, 256},
            {"F ceiling (sum the bytes)", "kF", 8, 256}};
        for (auto &v : vs) {
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:v.fn]] error:&err];
            if (!ps) { NSLog(@"%s: %@", v.name, err); continue; }
            auto launch = [&](int reps) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:ps];
                [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bg offset:0 atIndex:3]; [e setBuffer:by offset:0 atIndex:4];
                for (int i = 0; i < reps; i++) {
                    [e setBuffer:bqs[i % COPIES] offset:0 atIndex:1]; [e setBuffer:bss[i % COPIES] offset:0 atIndex:2];
                    [e dispatchThreadgroups:MTLSizeMake(N / v.rowsPerTG, 1, 1) threadsPerThreadgroup:MTLSizeMake(v.threads, 1, 1)];
                }
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                return ([cb GPUEndTime] - [cb GPUStartTime]);
            };
            memset(by.contents, 0, N * 4);
            launch(3);
            double t = launch(200) / 200;
            float *y = (float *)by.contents;
            double worst = 0, peak = 0;
            for (int r = 0; r < N; r++) { worst = fmax(worst, fabs(y[r] - ref[r])); peak = fmax(peak, fabs(ref[r])); }
            printf("%-30s %7.1f us  %6.1f GB/s  max err %.2e (peak %.2f)%s\n", v.name, t * 1e6, bytes / t / 1e9, worst, peak,
                   strcmp(v.fn, "kF") == 0 ? "  [not a matvec]" : "");
        }
    }
    return 0;
}
