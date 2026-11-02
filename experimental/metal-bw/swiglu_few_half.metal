// An experiment, not wired in (docs/native.md N4e): the speculative check's gate and up projections in half precision, hand-written, with the buffers, grid and threadgroup of matvec_fp4_swiglu_few.chx.
// ALU-bound in float32 (N4d); here the activations are rounded to half and the products of a lane's 16 weights are formed and added in half, then scaled and accumulated in float.
// NOT the arithmetic of the plain path: tokens may differ from plain greedy decoding. Same buffers, grid and threadgroup as matvec_fp4_swiglu_few.chx (3 rows, 32 weight rows a threadgroup).
#include <metal_stdlib>
using namespace metal;

#define K 5120u
#define N 17408u

// two fp4 codes of a byte as exact halves (the code's value, 0 to 6): the bit trick yields value * 2^-14, scaled back by 2^14 in half, which is exact
inline half2 fp4h(uint b) {
    const uint w = b | (b << 12u);
    return as_type<half2>(((w & 0x00070007u) << 9u) | ((w & 0x00080008u) << 12u)) * half(16384.0h);
}
inline float fp8_e4m3(uint b) {
    const uint e = (b >> 3u) & 0xFu, m = b & 7u;
    const uint bits = select(((e + 120u) << 23u) | (m << 20u), as_type<uint>(float(m) * 0.001953125f), e == 0u);
    return as_type<float>(bits | ((b & 0x80u) << 24u));
}

// y[t, n] = silu(x[t] . gate[n]) * (x[t] . up[n]) for 3 tokens t and 32 rows n a threadgroup; a simdgroup takes 4 rows, a lane 16 of every 512 columns
kernel void swiglu_few_h(device const float *x [[buffer(0)]],
                         device const char *gq [[buffer(1)]], device const char *gs [[buffer(2)]], device const float *gg [[buffer(3)]],
                         device const char *uq [[buffer(4)]], device const char *us [[buffer(5)]], device const float *ug [[buffer(6)]],
                         device float *y [[buffer(7)]],
                         uint tid [[thread_index_in_threadgroup]], uint3 tgpos [[threadgroup_position_in_grid]]) {
    const uint lane = tid % 32u;
    const uint row0 = tgpos.x * 32u + (tid / 32u) * 4u;
    float ag[4][3], au[4][3];
    for (int r = 0; r < 4; r++)
        for (int m = 0; m < 3; m++) { ag[r][m] = 0.0f; au[r][m] = 0.0f; }
    for (uint p0 = lane * 16u; p0 < K; p0 += 512u) {
        half2 xh[3][8];
        for (int m = 0; m < 3; m++) {
            device const float4 *x4 = (device const float4 *)(x + m * K + p0);
            for (int i = 0; i < 4; i++) {
                const float4 v = x4[i];
                xh[m][2 * i] = half2(v.xy);
                xh[m][2 * i + 1] = half2(v.zw);
            }
        }
        for (int r = 0; r < 4; r++) {
            const uint row = row0 + r;
            const uint2 wg = *(device const uint2 *)(gq + row * (K / 2u) + p0 / 2u);
            const uint2 wu = *(device const uint2 *)(uq + row * (K / 2u) + p0 / 2u);
            const float sgg = fp8_e4m3((uint)(uchar)gs[row * (K / 16u) + p0 / 16u]) * gg[row];
            const float sgu = fp8_e4m3((uint)(uchar)us[row * (K / 16u) + p0 / 16u]) * ug[row];
            half2 qg[8], qu[8];
            qg[0] = fp4h(wg.x & 0xFFu); qg[1] = fp4h((wg.x >> 8) & 0xFFu); qg[2] = fp4h((wg.x >> 16) & 0xFFu); qg[3] = fp4h(wg.x >> 24);
            qg[4] = fp4h(wg.y & 0xFFu); qg[5] = fp4h((wg.y >> 8) & 0xFFu); qg[6] = fp4h((wg.y >> 16) & 0xFFu); qg[7] = fp4h(wg.y >> 24);
            qu[0] = fp4h(wu.x & 0xFFu); qu[1] = fp4h((wu.x >> 8) & 0xFFu); qu[2] = fp4h((wu.x >> 16) & 0xFFu); qu[3] = fp4h(wu.x >> 24);
            qu[4] = fp4h(wu.y & 0xFFu); qu[5] = fp4h((wu.y >> 8) & 0xFFu); qu[6] = fp4h((wu.y >> 16) & 0xFFu); qu[7] = fp4h(wu.y >> 24);
            for (int m = 0; m < 3; m++) {
                half2 sg2 = xh[m][0] * qg[0], su2 = xh[m][0] * qu[0];
                for (int i = 1; i < 8; i++) { sg2 += xh[m][i] * qg[i]; su2 += xh[m][i] * qu[i]; }
                ag[r][m] += float(sg2.x + sg2.y) * sgg;
                au[r][m] += float(su2.x + su2.y) * sgu;
            }
        }
    }
    for (int r = 0; r < 4; r++)
        for (int m = 0; m < 3; m++) {
            float g = ag[r][m], u = au[r][m];
            for (uint d = 16u; d > 0; d /= 2) { g += simd_shuffle_down(g, d); u += simd_shuffle_down(u, d); }
            if (lane == 0) y[m * N + row0 + r] = g / (1.0f + precise::exp(-g)) * u;
        }
}
