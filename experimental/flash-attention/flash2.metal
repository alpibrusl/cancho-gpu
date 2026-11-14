#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
// flash attention with RS atom rows (8 query rows each) per simdgroup (docs/diffusion.md D16): K and V fragments loaded once serve RS multiplications.
constant constexpr uint D = 128;
constant uint NH [[function_constant(0)]];
constant uint NQ [[function_constant(1)]];
constant uint T  [[function_constant(2)]];
#ifndef BK
#define BK 16
#endif
#ifndef RS
#define RS 2
#endif
#ifndef NSG
#define NSG 4
#endif
#ifndef PAD
#define PAD 8
#endif
kernel void flash(device const half *q [[buffer(0)]], device const half *k [[buffer(1)]], device const half *v [[buffer(2)]],
                  device const float *mask [[buffer(3)]], device float *o [[buffer(4)]],
                  uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                  uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint LS = D + PAD;
    threadgroup half ks[BK * LS];
    threadgroup half vs[BK * LS];
    const uint h = tg.x, qb = tg.y;
    const uint row0 = qb * (NSG * 8 * RS) + sg * 8 * RS;
    const uint ld = NH * D;
    const ushort qid = lane / 4;
    const ushort fm = (qid & 4) + ((lane / 2) % 4);
    const ushort fn = (qid & 2) * 2 + (lane % 2) * 2;
    simdgroup_matrix<half, 8, 8> qf[RS][D / 8];
    for (uint r = 0; r < RS; ++r) for (uint kk = 0; kk < D / 8; ++kk) simdgroup_load(qf[r][kk], q + (size_t)(row0 + r * 8) * ld + h * D + kk * 8, ld);
    simdgroup_matrix<float, 8, 8> of[RS][D / 8];
    for (uint r = 0; r < RS; ++r) for (uint n = 0; n < D / 8; ++n) of[r][n] = simdgroup_matrix<float, 8, 8>(0.0f);
    float m_run[RS], l_run[RS];
    for (uint r = 0; r < RS; ++r) { m_run[r] = -1e30f; l_run[r] = 0.0f; }
    for (uint c = 0; c < T / BK; ++c) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < BK * D / 8; e += NSG * 32) {
            const uint r = e / (D / 8), cc = (e % (D / 8)) * 8;
            *(threadgroup uint4 *)(ks + r * LS + cc) = *(device const uint4 *)(k + (size_t)(c * BK + r) * ld + h * D + cc);
            *(threadgroup uint4 *)(vs + r * LS + cc) = *(device const uint4 *)(v + (size_t)(c * BK + r) * ld + h * D + cc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<float, 8, 8> sf[RS][BK / 8];
        for (uint r = 0; r < RS; ++r) for (uint j = 0; j < BK / 8; ++j) sf[r][j] = simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kk = 0; kk < D / 8; ++kk) {
            for (uint j = 0; j < BK / 8; ++j) {
                simdgroup_matrix<half, 8, 8> kt;
                simdgroup_load(kt, ks + (j * 8) * LS + kk * 8, LS, ulong2(0, 0), true);
                for (uint r = 0; r < RS; ++r) simdgroup_multiply_accumulate(sf[r][j], qf[r][kk], kt, sf[r][j]);
            }
        }
        simdgroup_matrix<half, 8, 8> pf[RS][BK / 8];
        float alpha[RS];
        for (uint r = 0; r < RS; ++r) {
            float mx = -1e30f;
            for (uint j = 0; j < BK / 8; ++j) {
                thread auto &el = sf[r][j].thread_elements();
#ifndef NOMASK
                const uint col = c * BK + j * 8 + fn;
                el[0] += mask[(size_t)(row0 + r * 8 + fm) * T + col];
                el[1] += mask[(size_t)(row0 + r * 8 + fm) * T + col + 1];
#endif
                mx = max(mx, max(el[0], el[1]));
            }
            mx = max(mx, simd_shuffle_xor(mx, 1));
            mx = max(mx, simd_shuffle_xor(mx, 8));
            const float m_new = max(m_run[r], mx);
            alpha[r] = fast::exp(m_run[r] - m_new);
            float ls = 0.0f;
            for (uint j = 0; j < BK / 8; ++j) {
                thread auto &el = sf[r][j].thread_elements();
                const float p0 = fast::exp(el[0] - m_new), p1 = fast::exp(el[1] - m_new);
                ls += p0 + p1;
                thread auto &pe = pf[r][j].thread_elements();
                pe[0] = half(p0);
                pe[1] = half(p1);
            }
            ls += simd_shuffle_xor(ls, 1);
            ls += simd_shuffle_xor(ls, 8);
            l_run[r] = l_run[r] * alpha[r] + ls;
            m_run[r] = m_new;
            for (uint n = 0; n < D / 8; ++n) {
                thread auto &oe = of[r][n].thread_elements();
                oe[0] *= alpha[r];
                oe[1] *= alpha[r];
            }
        }
        for (uint j = 0; j < BK / 8; ++j) {
            for (uint n = 0; n < D / 8; ++n) {
                simdgroup_matrix<half, 8, 8> vt;
                simdgroup_load(vt, vs + (j * 8) * LS + n * 8, LS);
                for (uint r = 0; r < RS; ++r) simdgroup_multiply_accumulate(of[r][n], pf[r][j], vt, of[r][n]);
            }
        }
    }
    for (uint r = 0; r < RS; ++r) {
        const float inv = 1.0f / l_run[r];
        for (uint n = 0; n < D / 8; ++n) {
            thread auto &oe = of[r][n].thread_elements();
            oe[0] *= inv;
            oe[1] *= inv;
            simdgroup_store(of[r][n], o + (size_t)(row0 + r * 8) * ld + h * D + n * 8, ld);
        }
    }
}
