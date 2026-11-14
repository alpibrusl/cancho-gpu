#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// A hand-written flash attention for the edit step's shape, to measure what a register-resident kernel does on this machine
// (docs/diffusion.md D16). q, k, v: half [rows, NH*128] token-major; mask: float [NQ, T] additive; o: float [NQ, NH*128].
// Threadgroup: 4 simdgroups, each owns 8 query rows; a key chunk of BK keys goes through threadgroup memory (shared by the 4 simdgroups).
constant constexpr uint D = 128;
constant uint NH [[function_constant(0)]];
constant uint NQ [[function_constant(1)]];
constant uint T  [[function_constant(2)]];
#ifndef BK
#define BK 32
#endif

kernel void flash(device const half *q [[buffer(0)]], device const half *k [[buffer(1)]], device const half *v [[buffer(2)]],
                  device const float *mask [[buffer(3)]], device float *o [[buffer(4)]],
                  uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                  uint tid [[thread_index_in_threadgroup]]) {
    #ifndef PAD
#define PAD 8
#endif
    constexpr uint LS = D + PAD;
    threadgroup half ks[BK * LS];
    threadgroup half vs[BK * LS];
    const uint h = tg.x, qb = tg.y;
    const uint row0 = qb * 32 + sg * 8;
    const uint ld = NH * D;
    // the element layout of an 8x8 simdgroup_matrix: this lane holds (fm, fn) and (fm, fn + 1)
    const ushort qid = lane / 4;
    const ushort fm = (qid & 4) + ((lane / 2) % 4);
    const ushort fn = (qid & 2) * 2 + (lane % 2) * 2;
    simdgroup_matrix<half, 8, 8> qf[D / 8];
    for (uint kk = 0; kk < D / 8; ++kk) simdgroup_load(qf[kk], q + (size_t)row0 * ld + h * D + kk * 8, ld);
    simdgroup_matrix<float, 8, 8> of[D / 8];
    for (uint n = 0; n < D / 8; ++n) of[n] = simdgroup_matrix<float, 8, 8>(0.0f);
    float m_run = -1e30f, l_run = 0.0f;
    for (uint c = 0; c < T / BK; ++c) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < BK * D / 8; e += 128) {            // 16-byte pieces: BK x D halves of k and of v
            const uint r = e / (D / 8), cc = (e % (D / 8)) * 8;
            *(threadgroup uint4 *)(ks + r * LS + cc) = *(device const uint4 *)(k + (size_t)(c * BK + r) * ld + h * D + cc);
            *(threadgroup uint4 *)(vs + r * LS + cc) = *(device const uint4 *)(v + (size_t)(c * BK + r) * ld + h * D + cc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<float, 8, 8> sf[BK / 8];
        for (uint j = 0; j < BK / 8; ++j) sf[j] = simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kk = 0; kk < D / 8; ++kk) {
            for (uint j = 0; j < BK / 8; ++j) {
                simdgroup_matrix<half, 8, 8> kt;
                simdgroup_load(kt, ks + (j * 8) * LS + kk * 8, LS, ulong2(0, 0), true);   // k^T: the tile is stored [key, d]
                simdgroup_multiply_accumulate(sf[j], qf[kk], kt, sf[j]);
            }
        }
        // mask, chunk maximum of this lane's two elements per atom, then across the four lanes that share the row
        float mx = -1e30f;
        for (uint j = 0; j < BK / 8; ++j) {
            thread auto &el = sf[j].thread_elements();
            const uint col = c * BK + j * 8 + fn;
            #ifndef NOMASK
            el[0] += mask[(size_t)(row0 + fm) * T + col];
            el[1] += mask[(size_t)(row0 + fm) * T + col + 1];
#endif
            mx = max(mx, max(el[0], el[1]));
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        const float m_new = max(m_run, mx);
        #ifdef FASTEXP
        const float alpha = fast::exp(m_run - m_new);
#else
        const float alpha = exp(m_run - m_new);
#endif
        float ls = 0.0f;
        simdgroup_matrix<half, 8, 8> pf[BK / 8];
        for (uint j = 0; j < BK / 8; ++j) {
            thread auto &el = sf[j].thread_elements();
            #ifdef FASTEXP
            const float p0 = fast::exp(el[0] - m_new), p1 = fast::exp(el[1] - m_new);
#else
            const float p0 = exp(el[0] - m_new), p1 = exp(el[1] - m_new);
#endif
            ls += p0 + p1;
            thread auto &pe = pf[j].thread_elements();
            pe[0] = half(p0);
            pe[1] = half(p1);
        }
        ls += simd_shuffle_xor(ls, 1);
        ls += simd_shuffle_xor(ls, 8);
        l_run = l_run * alpha + ls;
        m_run = m_new;
        for (uint n = 0; n < D / 8; ++n) {
            thread auto &oe = of[n].thread_elements();
            oe[0] *= alpha;
            oe[1] *= alpha;
        }
        for (uint j = 0; j < BK / 8; ++j) {
            for (uint n = 0; n < D / 8; ++n) {
                simdgroup_matrix<half, 8, 8> vt;
                simdgroup_load(vt, vs + (j * 8) * LS + n * 8, LS);
                simdgroup_multiply_accumulate(of[n], pf[j], vt, of[n]);
            }
        }
    }
    const float inv = 1.0f / l_run;
    for (uint n = 0; n < D / 8; ++n) {
        thread auto &oe = of[n].thread_elements();
        oe[0] *= inv;
        oe[1] *= inv;
        simdgroup_store(of[n], o + (size_t)row0 * ld + h * D + n * 8, ld);
    }
}
