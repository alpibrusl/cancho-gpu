// Why three rows through an NVFP4 matvec cost more than one (docs/native.md N4d): batch.mm's `w` kernels plus the
// experiments that located it. Run like batch.mm. Findings (M4 Max, 17408 x 5120, weights rotated through 16 copies):
//   * M=1 142 us, M=2 159 us, M=3 about 210-246 us (`w3_*`, `x1_3_*`, `y3_*`): the cliff is at the third token.
//   * x4_*: x read from one row for all three tokens (the compiler then does one dot, so these OUTPUTS ARE WRONG): 169 us.
//     The time is therefore not the weights and not x's traffic but the number of f32 multiply-adds (3 per weight).
//   * h3_*: half x, half products (timing only; the numbers it prints are garbage, and half accumulation would not
//     keep the tokens): M=3 as fast as M=1 (166 vs 170 us). So the kernel is ALU-bound in f32, and the only way
//     to the one-row speed is cheaper arithmetic per weight, which changes the arithmetic.
// NVFP4 matrix times M vectors (the shape of a prefill chunk or a speculative
// check), by hand: how close to one pass over the weights can it get? (Issue 26.)
//
//     clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation batch.mm -o batch && ./batch
//
// Variant `kM<M,R>`: a simdgroup owns R rows; a lane keeps the 16 values of x
// of its slice, for all M tokens, in registers while it streams the R rows'
// weights past them, so the x traffic is paid once for R rows, not once for each.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

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
template <int M, int R>
inline void batch_rows(device const float *x, device const char *wq, device const char *ws,
                       device const float *wg, device float *y, uint tid, uint tgx) {
    const uint lane = tid % 32u;
    const uint row0 = (tgx * 8u + tid / 32u) * R;
    float acc[R][M];
    float rs[R];
    for (int r = 0; r < R; r++) { rs[r] = wg[row0 + r] * 16384.0f; for (int m = 0; m < M; m++) acc[r][m] = 0.0f; }
    for (uint p0 = lane * 16u; p0 < K; p0 += 512u) {
        float4 xv[M][4];
        for (int m = 0; m < M; m++) {
            device const float4 *x4 = (device const float4 *)(x + m * K + p0);
            for (int i = 0; i < 4; i++) xv[m][i] = x4[i];
        }
        for (int r = 0; r < R; r++) {
            const uint row = row0 + r;
            const uint2 w = *(device const uint2 *)(wq + row * (K / 2) + p0 / 2u);
            const float sg = fp8_e4m3((uint)(uchar)ws[row * (K / 16) + p0 / 16u]) * rs[r];
            float2 q[8];
            q[0] = fp4_pair(w.x & 0xFFu); q[1] = fp4_pair((w.x >> 8) & 0xFFu);
            q[2] = fp4_pair((w.x >> 16) & 0xFFu); q[3] = fp4_pair(w.x >> 24);
            q[4] = fp4_pair(w.y & 0xFFu); q[5] = fp4_pair((w.y >> 8) & 0xFFu);
            q[6] = fp4_pair((w.y >> 16) & 0xFFu); q[7] = fp4_pair(w.y >> 24);
            for (int m = 0; m < M; m++) {
                float d = 0.0f;
                for (int i = 0; i < 4; i++) {
                    d += xv[m][i].x * q[2 * i].x + xv[m][i].y * q[2 * i].y
                       + xv[m][i].z * q[2 * i + 1].x + xv[m][i].w * q[2 * i + 1].y;
                }
                acc[r][m] += d * sg;
            }
        }
    }
    for (int r = 0; r < R; r++)
        for (int m = 0; m < M; m++) {
            float s = acc[r][m];
            for (uint d = 16u; d > 0; d /= 2) s += simd_shuffle_down(s, d);
            if (lane == 0) y[m * N + row0 + r] = s;
        }
}
#define K_(NAME, M, R) kernel void NAME(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]], \
    device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]], device float *y [[buffer(4)]], \
    uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) { batch_rows<M, R>(x, wq, ws, wg, y, tid, tg.x); }
K_(k1_1, 1, 1) K_(k1_2, 1, 2) K_(k1_4, 1, 4)
K_(k2_1, 2, 1) K_(k2_2, 2, 2) K_(k2_4, 2, 4)
K_(k3_2, 3, 2) K_(k3_4, 3, 4)
K_(k4_1, 4, 1) K_(k4_2, 4, 2) K_(k4_4, 4, 4) K_(k4_8, 4, 8)
K_(k6_2, 6, 2) K_(k6_4, 6, 4)
K_(k8_1, 8, 1) K_(k8_2, 8, 2) K_(k8_4, 8, 4)

// V1: decode the R rows' weights of the slice first (R * 16 registers), then run the tokens past them,
// loading each token's x slice (L1) once for the R rows: fewer live registers than holding all of x.
template <int M, int R>
inline void batch_rows_w(device const float *x, device const char *wq, device const char *ws,
                         device const float *wg, device float *y, uint tid, uint tgx) {
    const uint lane = tid % 32u;
    const uint row0 = (tgx * 8u + tid / 32u) * R;
    float acc[R][M];
    float rs[R];
    for (int r = 0; r < R; r++) { rs[r] = wg[row0 + r] * 16384.0f; for (int m = 0; m < M; m++) acc[r][m] = 0.0f; }
    for (uint p0 = lane * 16u; p0 < K; p0 += 512u) {
        float2 q[R][8];
        float sg[R];
        for (int r = 0; r < R; r++) {
            const uint row = row0 + r;
            const uint2 w = *(device const uint2 *)(wq + row * (K / 2) + p0 / 2u);
            sg[r] = fp8_e4m3((uint)(uchar)ws[row * (K / 16) + p0 / 16u]) * rs[r];
            q[r][0] = fp4_pair(w.x & 0xFFu); q[r][1] = fp4_pair((w.x >> 8) & 0xFFu);
            q[r][2] = fp4_pair((w.x >> 16) & 0xFFu); q[r][3] = fp4_pair(w.x >> 24);
            q[r][4] = fp4_pair(w.y & 0xFFu); q[r][5] = fp4_pair((w.y >> 8) & 0xFFu);
            q[r][6] = fp4_pair((w.y >> 16) & 0xFFu); q[r][7] = fp4_pair(w.y >> 24);
        }
        for (int m = 0; m < M; m++) {
            device const float4 *x4 = (device const float4 *)(x + m * K + p0);
            const float4 a = x4[0], b = x4[1], c = x4[2], d4 = x4[3];
            for (int r = 0; r < R; r++) {
                const float d = a.x * q[r][0].x + a.y * q[r][0].y + a.z * q[r][1].x + a.w * q[r][1].y
                              + b.x * q[r][2].x + b.y * q[r][2].y + b.z * q[r][3].x + b.w * q[r][3].y
                              + c.x * q[r][4].x + c.y * q[r][4].y + c.z * q[r][5].x + c.w * q[r][5].y
                              + d4.x * q[r][6].x + d4.y * q[r][6].y + d4.z * q[r][7].x + d4.w * q[r][7].y;
                acc[r][m] += d * sg[r];
            }
        }
    }
    for (int r = 0; r < R; r++)
        for (int m = 0; m < M; m++) {
            float s = acc[r][m];
            for (uint d = 16u; d > 0; d /= 2) s += simd_shuffle_down(s, d);
            if (lane == 0) y[m * N + row0 + r] = s;
        }
}
#define KW_(NAME, M, R) kernel void NAME(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]], \
    device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]], device float *y [[buffer(4)]], \
    uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) { batch_rows_w<M, R>(x, wq, ws, wg, y, tid, tg.x); }
KW_(w1_2, 1, 2) KW_(w2_2, 2, 2) KW_(w3_2, 3, 2) KW_(w3_4, 3, 4) KW_(w4_2, 4, 2) KW_(w4_4, 4, 4) KW_(w6_4, 6, 4) KW_(w8_2, 8, 2) KW_(w8_4, 8, 4) KW_(w8_8, 8, 8)


// ---- experiments for M=3 ----
// X1: like w, with the 16-term sum as four independent partial sums
template <int M, int R>
inline void batch_rows_x1(device const float *x, device const char *wq, device const char *ws,
                         device const float *wg, device float *y, uint tid, uint tgx) {
    const uint lane = tid % 32u;
    const uint row0 = (tgx * 8u + tid / 32u) * R;
    float acc[R][M];
    float rs[R];
    for (int r = 0; r < R; r++) { rs[r] = wg[row0 + r] * 16384.0f; for (int m = 0; m < M; m++) acc[r][m] = 0.0f; }
    for (uint p0 = lane * 16u; p0 < K; p0 += 512u) {
        float2 q[R][8];
        float sg[R];
        for (int r = 0; r < R; r++) {
            const uint row = row0 + r;
            const uint2 w = *(device const uint2 *)(wq + row * (K / 2) + p0 / 2u);
            sg[r] = fp8_e4m3((uint)(uchar)ws[row * (K / 16) + p0 / 16u]) * rs[r];
            q[r][0] = fp4_pair(w.x & 0xFFu); q[r][1] = fp4_pair((w.x >> 8) & 0xFFu);
            q[r][2] = fp4_pair((w.x >> 16) & 0xFFu); q[r][3] = fp4_pair(w.x >> 24);
            q[r][4] = fp4_pair(w.y & 0xFFu); q[r][5] = fp4_pair((w.y >> 8) & 0xFFu);
            q[r][6] = fp4_pair((w.y >> 16) & 0xFFu); q[r][7] = fp4_pair(w.y >> 24);
        }
        for (int m = 0; m < M; m++) {
            device const float4 *x4 = (device const float4 *)(x + m * K + p0);
            const float4 a = x4[0], b = x4[1], c = x4[2], d4 = x4[3];
            for (int r = 0; r < R; r++) {
                const float d0 = a.x * q[r][0].x + a.y * q[r][0].y + a.z * q[r][1].x + a.w * q[r][1].y;
                const float d1 = b.x * q[r][2].x + b.y * q[r][2].y + b.z * q[r][3].x + b.w * q[r][3].y;
                const float d2 = c.x * q[r][4].x + c.y * q[r][4].y + c.z * q[r][5].x + c.w * q[r][5].y;
                const float d3 = d4.x * q[r][6].x + d4.y * q[r][6].y + d4.z * q[r][7].x + d4.w * q[r][7].y;
                acc[r][m] += ((d0 + d1) + (d2 + d3)) * sg[r];
            }
        }
    }
    for (int r = 0; r < R; r++)
        for (int m = 0; m < M; m++) {
            float s = acc[r][m];
            for (uint d = 16u; d > 0; d /= 2) s += simd_shuffle_down(s, d);
            if (lane == 0) y[m * N + row0 + r] = s;
        }
}
#define KX1_(NAME, M, R) kernel void NAME(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]], \
    device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]], device float *y [[buffer(4)]], \
    uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) { batch_rows_x1<M, R>(x, wq, ws, wg, y, tid, tg.x); }
KX1_(x1_1_2, 1, 2) KX1_(x1_2_2, 2, 2) KX1_(x1_3_2, 3, 2) KX1_(x1_3_4, 3, 4) KX1_(x1_3_1, 3, 1)

// X1: like w, with the 16-term sum as four independent partial sums
template <int M, int R>
inline void batch_rows_x4(device const float *x, device const char *wq, device const char *ws,
                         device const float *wg, device float *y, uint tid, uint tgx) {
    const uint lane = tid % 32u;
    const uint row0 = (tgx * 8u + tid / 32u) * R;
    float acc[R][M];
    float rs[R];
    for (int r = 0; r < R; r++) { rs[r] = wg[row0 + r] * 16384.0f; for (int m = 0; m < M; m++) acc[r][m] = 0.0f; }
    for (uint p0 = lane * 16u; p0 < K; p0 += 512u) {
        float2 q[R][8];
        float sg[R];
        for (int r = 0; r < R; r++) {
            const uint row = row0 + r;
            const uint2 w = *(device const uint2 *)(wq + row * (K / 2) + p0 / 2u);
            sg[r] = fp8_e4m3((uint)(uchar)ws[row * (K / 16) + p0 / 16u]) * rs[r];
            q[r][0] = fp4_pair(w.x & 0xFFu); q[r][1] = fp4_pair((w.x >> 8) & 0xFFu);
            q[r][2] = fp4_pair((w.x >> 16) & 0xFFu); q[r][3] = fp4_pair(w.x >> 24);
            q[r][4] = fp4_pair(w.y & 0xFFu); q[r][5] = fp4_pair((w.y >> 8) & 0xFFu);
            q[r][6] = fp4_pair((w.y >> 16) & 0xFFu); q[r][7] = fp4_pair(w.y >> 24);
        }
        for (int m = 0; m < M; m++) {
            device const float4 *x4 = (device const float4 *)(x + p0);
            const float4 a = x4[0], b = x4[1], c = x4[2], d4 = x4[3];
            for (int r = 0; r < R; r++) {
                const float d0 = a.x * q[r][0].x + a.y * q[r][0].y + a.z * q[r][1].x + a.w * q[r][1].y;
                const float d1 = b.x * q[r][2].x + b.y * q[r][2].y + b.z * q[r][3].x + b.w * q[r][3].y;
                const float d2 = c.x * q[r][4].x + c.y * q[r][4].y + c.z * q[r][5].x + c.w * q[r][5].y;
                const float d3 = d4.x * q[r][6].x + d4.y * q[r][6].y + d4.z * q[r][7].x + d4.w * q[r][7].y;
                acc[r][m] += ((d0 + d1) + (d2 + d3)) * sg[r];
            }
        }
    }
    for (int r = 0; r < R; r++)
        for (int m = 0; m < M; m++) {
            float s = acc[r][m];
            for (uint d = 16u; d > 0; d /= 2) s += simd_shuffle_down(s, d);
            if (lane == 0) y[m * N + row0 + r] = s;
        }
}
#define KX4_(NAME, M, R) kernel void NAME(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]], \
    device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]], device float *y [[buffer(4)]], \
    uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) { batch_rows_x4<M, R>(x, wq, ws, wg, y, tid, tg.x); }
KX4_(x4_3_2, 3, 2) KX4_(x4_3_4, 3, 4) KX4_(x4_2_2, 2, 2)

// Y: M=3, x of the lane's 16 columns in 12 named float4 registers, R rows streamed past them
template <int R>
inline void rows_y3(device const float *x, device const char *wq, device const char *ws,
                    device const float *wg, device float *y, uint tid, uint tgx) {
    const uint lane = tid % 32u;
    const uint row0 = (tgx * 8u + tid / 32u) * R;
    float acc0[R], acc1[R], acc2[R], rs[R];
    for (int r = 0; r < R; r++) { rs[r] = wg[row0 + r] * 16384.0f; acc0[r] = 0.0f; acc1[r] = 0.0f; acc2[r] = 0.0f; }
    for (uint p0 = lane * 16u; p0 < K; p0 += 512u) {
        device const float4 *xa = (device const float4 *)(x + p0);
        device const float4 *xb = (device const float4 *)(x + K + p0);
        device const float4 *xc = (device const float4 *)(x + 2 * K + p0);
        const float4 a0 = xa[0], a1 = xa[1], a2 = xa[2], a3 = xa[3];
        const float4 b0 = xb[0], b1 = xb[1], b2 = xb[2], b3 = xb[3];
        const float4 c0 = xc[0], c1 = xc[1], c2 = xc[2], c3 = xc[3];
#pragma clang loop unroll(full)
        for (int r = 0; r < R; r++) {
            const uint row = row0 + r;
            const uint2 w = *(device const uint2 *)(wq + row * (K / 2) + p0 / 2u);
            const float sg = fp8_e4m3((uint)(uchar)ws[row * (K / 16) + p0 / 16u]) * rs[r];
            const float2 q0 = fp4_pair(w.x & 0xFFu), q1 = fp4_pair((w.x >> 8) & 0xFFu), q2 = fp4_pair((w.x >> 16) & 0xFFu), q3 = fp4_pair(w.x >> 24);
            const float2 q4 = fp4_pair(w.y & 0xFFu), q5 = fp4_pair((w.y >> 8) & 0xFFu), q6 = fp4_pair((w.y >> 16) & 0xFFu), q7 = fp4_pair(w.y >> 24);
            acc0[r] += (a0.x * q0.x + a0.y * q0.y + a0.z * q1.x + a0.w * q1.y + a1.x * q2.x + a1.y * q2.y + a1.z * q3.x + a1.w * q3.y
                      + a2.x * q4.x + a2.y * q4.y + a2.z * q5.x + a2.w * q5.y + a3.x * q6.x + a3.y * q6.y + a3.z * q7.x + a3.w * q7.y) * sg;
            acc1[r] += (b0.x * q0.x + b0.y * q0.y + b0.z * q1.x + b0.w * q1.y + b1.x * q2.x + b1.y * q2.y + b1.z * q3.x + b1.w * q3.y
                      + b2.x * q4.x + b2.y * q4.y + b2.z * q5.x + b2.w * q5.y + b3.x * q6.x + b3.y * q6.y + b3.z * q7.x + b3.w * q7.y) * sg;
            acc2[r] += (c0.x * q0.x + c0.y * q0.y + c0.z * q1.x + c0.w * q1.y + c1.x * q2.x + c1.y * q2.y + c1.z * q3.x + c1.w * q3.y
                      + c2.x * q4.x + c2.y * q4.y + c2.z * q5.x + c2.w * q5.y + c3.x * q6.x + c3.y * q6.y + c3.z * q7.x + c3.w * q7.y) * sg;
        }
    }
    for (int r = 0; r < R; r++) {
        float s0 = acc0[r], s1 = acc1[r], s2 = acc2[r];
        for (uint d = 16u; d > 0; d /= 2) { s0 += simd_shuffle_down(s0, d); s1 += simd_shuffle_down(s1, d); s2 += simd_shuffle_down(s2, d); }
        if (lane == 0) { y[row0 + r] = s0; y[N + row0 + r] = s1; y[2 * N + row0 + r] = s2; }
    }
}
#define KY_(NAME, R) kernel void NAME(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]], \
    device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]], device float *y [[buffer(4)]], \
    uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) { rows_y3<R>(x, wq, ws, wg, y, tid, tg.x); }
KY_(y3_1, 1) KY_(y3_2, 2) KY_(y3_4, 4) KY_(y3_8, 8)

inline half2 fp4_pair_h(uint b) {
    const uint w = b | (b << 12u);
    return as_type<half2>(((w & 0x00070007u) << 9u) | ((w & 0x00080008u) << 12u));
}
// H: M=3 with half x and half products (timing only: x is the f32 buffer read as half)
template <int R>
inline void rows_h3(device const float *x, device const char *wq, device const char *ws,
                    device const float *wg, device float *y, uint tid, uint tgx) {
    const uint lane = tid % 32u;
    const uint row0 = (tgx * 8u + tid / 32u) * R;
    float acc0[R], acc1[R], acc2[R], rs[R];
    for (int r = 0; r < R; r++) { rs[r] = wg[row0 + r] * 16384.0f; acc0[r] = 0.0f; acc1[r] = 0.0f; acc2[r] = 0.0f; }
    device const half *xh = (device const half *)x;
    for (uint p0 = lane * 16u; p0 < K; p0 += 512u) {
        device const half4 *xa = (device const half4 *)(xh + p0);
        device const half4 *xb = (device const half4 *)(xh + K + p0);
        device const half4 *xc = (device const half4 *)(xh + 2 * K + p0);
        const half4 a0 = xa[0], a1 = xa[1], a2 = xa[2], a3 = xa[3];
        const half4 b0 = xb[0], b1 = xb[1], b2 = xb[2], b3 = xb[3];
        const half4 c0 = xc[0], c1 = xc[1], c2 = xc[2], c3 = xc[3];
#pragma clang loop unroll(full)
        for (int r = 0; r < R; r++) {
            const uint row = row0 + r;
            const uint2 w = *(device const uint2 *)(wq + row * (K / 2) + p0 / 2u);
            const float sg = fp8_e4m3((uint)(uchar)ws[row * (K / 16) + p0 / 16u]) * rs[r];
            const half2 q0 = fp4_pair_h(w.x & 0xFFu), q1 = fp4_pair_h((w.x >> 8) & 0xFFu), q2 = fp4_pair_h((w.x >> 16) & 0xFFu), q3 = fp4_pair_h(w.x >> 24);
            const half2 q4 = fp4_pair_h(w.y & 0xFFu), q5 = fp4_pair_h((w.y >> 8) & 0xFFu), q6 = fp4_pair_h((w.y >> 16) & 0xFFu), q7 = fp4_pair_h(w.y >> 24);
            half2 s0 = a0.xy * q0 + a0.zw * q1 + a1.xy * q2 + a1.zw * q3 + a2.xy * q4 + a2.zw * q5 + a3.xy * q6 + a3.zw * q7;
            half2 s1 = b0.xy * q0 + b0.zw * q1 + b1.xy * q2 + b1.zw * q3 + b2.xy * q4 + b2.zw * q5 + b3.xy * q6 + b3.zw * q7;
            half2 s2 = c0.xy * q0 + c0.zw * q1 + c1.xy * q2 + c1.zw * q3 + c2.xy * q4 + c2.zw * q5 + c3.xy * q6 + c3.zw * q7;
            acc0[r] += float(s0.x + s0.y) * sg; acc1[r] += float(s1.x + s1.y) * sg; acc2[r] += float(s2.x + s2.y) * sg;
        }
    }
    for (int r = 0; r < R; r++) {
        float s0 = acc0[r], s1 = acc1[r], s2 = acc2[r];
        for (uint d = 16u; d > 0; d /= 2) { s0 += simd_shuffle_down(s0, d); s1 += simd_shuffle_down(s1, d); s2 += simd_shuffle_down(s2, d); }
        if (lane == 0) { y[row0 + r] = s0; y[N + row0 + r] = s1; y[2 * N + row0 + r] = s2; }
    }
}
#define KH_(NAME, R) kernel void NAME(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]], \
    device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]], device float *y [[buffer(4)]], \
    uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) { rows_h3<R>(x, wq, ws, wg, y, tid, tg.x); }
KH_(h3_1, 1) KH_(h3_2, 2) KH_(h3_4, 4)
// X2: M=3 as a 2-token pass and a 1-token pass over the same decoded weights, one row at a time per iteration,
// the weights decoded to half pairs once
template <int R>
inline void batch_rows_x2(device const float *x, device const char *wq, device const char *ws,
                         device const float *wg, device float *y, uint tid, uint tgx) {
    const uint lane = tid % 32u;
    const uint row0 = (tgx * 8u + tid / 32u) * R;
    float acc[R][3];
    float rs[R];
    for (int r = 0; r < R; r++) { rs[r] = wg[row0 + r] * 16384.0f; for (int m = 0; m < 3; m++) acc[r][m] = 0.0f; }
    for (uint p0 = lane * 16u; p0 < K; p0 += 512u) {
        float4 xv[3][4];
        for (int m = 0; m < 3; m++) {
            device const float4 *x4 = (device const float4 *)(x + m * K + p0);
            for (int i = 0; i < 4; i++) xv[m][i] = x4[i];
        }
        for (int r = 0; r < R; r++) {
            const uint row = row0 + r;
            const uint2 w = *(device const uint2 *)(wq + row * (K / 2) + p0 / 2u);
            const float sg = fp8_e4m3((uint)(uchar)ws[row * (K / 16) + p0 / 16u]) * rs[r];
            float d[3] = {0.0f, 0.0f, 0.0f};
            for (int i = 0; i < 4; i++) {
                const uint word = (i < 2) ? w.x : w.y;
                const uint sh = (i & 1) * 16u;
                const float2 a = fp4_pair((word >> sh) & 0xFFu);
                const float2 b = fp4_pair((word >> (sh + 8u)) & 0xFFu);
                for (int m = 0; m < 3; m++)
                    d[m] += xv[m][i].x * a.x + xv[m][i].y * a.y + xv[m][i].z * b.x + xv[m][i].w * b.y;
            }
            for (int m = 0; m < 3; m++) acc[r][m] += d[m] * sg;
        }
    }
    for (int r = 0; r < R; r++)
        for (int m = 0; m < 3; m++) {
            float s = acc[r][m];
            for (uint d = 16u; d > 0; d /= 2) s += simd_shuffle_down(s, d);
            if (lane == 0) y[m * N + row0 + r] = s;
        }
}
#define KX2_(NAME, R) kernel void NAME(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]], \
    device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]], device float *y [[buffer(4)]], \
    uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) { batch_rows_x2<R>(x, wq, ws, wg, y, tid, tg.x); }
KX2_(x2_1, 1) KX2_(x2_2, 2) KX2_(x2_4, 4)

// Matrix unit, from registers: a simdgroup owns T tiles of 8 rows. A lane of an 8x8 simdgroup
// fragment holds two neighbouring columns of one row (row fm, columns fn, fn+1), so for the weights
// (rows x k) that is one packed byte: no threadgroup memory. The sum over k does not care in what order
// k is taken, so a lane reads a whole uint (8 codes, 4 steps of 8x8) contiguously and the k each
// column stands for is permuted the same way for the activations (the B fragment).
#include <metal_simdgroup_matrix>
template <int T>
inline void mma_rows(device const float *x, device const char *wq, device const char *ws,
                     device const float *wg, device float *y, uint tid, uint tgx, int M) {
    const uint lane = tid % 32u;
    const uint qid = lane / 4u;
    const uint fm = (qid & 4u) + ((lane / 2u) % 4u);        // fragment row
    const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;     // fragment column (and the next)
    const uint g = fn / 2u;                                  // which 8 k of the 32 this lane's columns stand for
    const uint rowbase = (tgx * 8u + tid / 32u) * 8u * T;
    simdgroup_float8x8 C[T];
    for (int t = 0; t < T; t++) C[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint blk = 0; blk < K / 32u; blk++) {
        // activations: B(k, token): this lane holds k-row fm, tokens fn and fn+1
        const uint kk = blk * 32u + (fm / 2u) * 8u + (fm % 2u);
        float xs[4][2];
        for (int s = 0; s < 4; s++) {
            xs[s][0] = (int)fn < M ? x[fn * K + kk + 2u * s] : 0.0f;
            xs[s][1] = (int)(fn + 1u) < M ? x[(fn + 1u) * K + kk + 2u * s] : 0.0f;
        }
        for (int t = 0; t < T; t++) {
            const uint row = rowbase + t * 8u + fm;
            const uint w = *(device const uint *)(wq + row * (K / 2u) + blk * 16u + g * 4u);
            const float sg = fp8_e4m3((uint)(uchar)ws[row * (K / 16u) + blk * 2u + g / 2u]);
            for (int s = 0; s < 4; s++) {
                simdgroup_float8x8 A, B;
                const float2 p = fp4_pair((w >> (8 * s)) & 0xFFu) * sg;
                A.thread_elements()[0] = p.x; A.thread_elements()[1] = p.y;
                B.thread_elements()[0] = xs[s][0]; B.thread_elements()[1] = xs[s][1];
                simdgroup_multiply_accumulate(C[t], A, B, C[t]);
            }
        }
    }
    for (int t = 0; t < T; t++) {
        const uint row = rowbase + t * 8u + fm;
        const float rs = wg[row] * 16384.0f;
        if ((int)fn < M) y[fn * N + row] = C[t].thread_elements()[0] * rs;
        if ((int)(fn + 1u) < M) y[(fn + 1u) * N + row] = C[t].thread_elements()[1] * rs;
    }
}
#define KM_(NAME, T, M) kernel void NAME(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]], \
    device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]], device float *y [[buffer(4)]], \
    uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) { mma_rows<T>(x, wq, ws, wg, y, tid, tg.x, M); }
KM_(m1_1, 1, 1) KM_(m2_1, 1, 2) KM_(m3_1, 1, 3) KM_(m4_1, 1, 4) KM_(m8_1, 1, 8)
KM_(m1_2, 2, 1) KM_(m2_2, 2, 2) KM_(m4_2, 2, 4) KM_(m8_2, 2, 8)
KM_(m4_4, 4, 4) KM_(m8_4, 4, 8)
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
        const int MAXM = 8;
        size_t nq = (size_t)N * K / 2, ns = (size_t)N * K / 16;
        std::vector<unsigned char> wq(nq), ws(ns);
        std::vector<float> x((size_t)MAXM * K), wg(N, 1.0f / 16384.0f);
        srand(1);
        for (auto &b : wq) b = rand();
        for (auto &b : ws) b = 0x30 + (rand() & 0x1f);
        for (auto &v : x) v = (rand() / (float)RAND_MAX) - 0.5f;
        const int COPIES = 16;
        std::vector<id<MTLBuffer>> bqs, bss;
        for (int c = 0; c < COPIES; c++) {
            bqs.push_back([dev newBufferWithBytes:wq.data() length:nq options:0]);
            bss.push_back([dev newBufferWithBytes:ws.data() length:ns options:0]);
        }
        id<MTLBuffer> bx = [dev newBufferWithBytes:x.data() length:x.size() * 4 options:0];
        id<MTLBuffer> bg = [dev newBufferWithBytes:wg.data() length:N * 4 options:0];
        id<MTLBuffer> by = [dev newBufferWithLength:(size_t)MAXM * N * 4 options:0];
        // the first 256 rows, every token, in double precision
        std::vector<double> ref((size_t)MAXM * 256);
        for (int m = 0; m < MAXM; m++)
            for (int r = 0; r < 256; r++) {
                double acc = 0;
                for (int p = 0; p < K; p++) {
                    unsigned byte = wq[(size_t)r * K / 2 + p / 2];
                    unsigned code = (p & 1) ? byte >> 4 : byte & 15;
                    acc += x[(size_t)m * K + p] * fp4v[code] * e4m3(ws[(size_t)r * K / 16 + p / 16]) * wg[r];
                }
                ref[m * 256 + r] = acc;
            }
        struct V { const char *fn; int M, R; } vs[] = {{"w1_2",1,2},{"w3_4",3,4},{"h3_1",3,1},{"h3_2",3,2},{"h3_4",3,4}};
        for (auto &v : vs) {
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:v.fn]] error:&err];
            if (!ps) { NSLog(@"%s: %@", v.fn, err); continue; }
            const int rpt = v.fn[0] == 'm' ? v.R : 8 * v.R;
            auto launch = [&](int reps) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:ps];
                [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bg offset:0 atIndex:3]; [e setBuffer:by offset:0 atIndex:4];
                for (int i = 0; i < reps; i++) {
                    [e setBuffer:bqs[i % COPIES] offset:0 atIndex:1]; [e setBuffer:bss[i % COPIES] offset:0 atIndex:2];
                    [e dispatchThreadgroups:MTLSizeMake(N / rpt, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                return [cb GPUEndTime] - [cb GPUStartTime];
            };
            memset(by.contents, 0, (size_t)MAXM * N * 4);
            launch(3);
            double t = launch(100) / 100;
            float *y = (float *)by.contents;
            double worst = 0, peak = 0;
            for (int m = 0; m < v.M; m++)
                for (int r = 0; r < 256; r++) { worst = fmax(worst, fabs(y[m * N + r] - ref[m * 256 + r])); peak = fmax(peak, fabs(ref[m * 256 + r])); }
            printf("%s M=%d R=%d  %7.1f us  %6.1f GB/s of weights  %5.2f us a token   max err %.1e (peak %.2f)\n", v.fn, v.M, v.R, t * 1e6,
                   (double)(nq + ns) / t / 1e9, t * 1e6 / v.M, worst, peak);
        }
    }
    return 0;
}
