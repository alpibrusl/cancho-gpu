// Two hand-written Metal kernels for the draft head, which the surface cannot say yet: it has no index-valued
// load (the embedding row of a token that is data on the device) and no argmax. They let a draft run with no
// round trip to the CPU: the token one step chose is the next step's input, and the verify's.
#include <metal_stdlib>
using namespace metal;

#define HID 5120
#define VOCAB 248320

// dst[i] = float(table[tok[0]][i]) for a bf16 table of VOCAB rows of HID.
kernel void gather_bf16(device const ushort *table [[buffer(0)]],
                        device const int *tok [[buffer(1)]],
                        device float *dst [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < HID) {
        ulong base = (ulong)(uint)tok[0] * HID;
        dst[i] = as_type<float>(((uint)table[base + i]) << 16);
    }
}

// out[0] = the index of the largest of VOCAB logits (the highest index among equals, as the CPU's greedy pick takes it). One threadgroup of 1024.
kernel void argmax_f32(device const float *x [[buffer(0)]],
                       device int *out [[buffer(1)]],
                       uint t [[thread_position_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    float best = -INFINITY;
    int bi = 0;
    for (uint i = t; i < VOCAB; i += 1024) {
        float v = x[i];
        if (v >= best) {
            best = v;
            bi = (int)i;
        }
    }
    for (uint o = 16; o > 0; o >>= 1) {
        float ov = simd_shuffle_down(best, o);
        int oi = simd_shuffle_down(bi, o);
        if (ov > best || (ov == best && oi > bi)) {
            best = ov;
            bi = oi;
        }
    }
    threadgroup float sv[32];
    threadgroup int si[32];
    if (lane == 0) {
        sv[sg] = best;
        si[sg] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        best = lane < 32 ? sv[lane] : -INFINITY;
        bi = lane < 32 ? si[lane] : 0;
        for (uint o = 16; o > 0; o >>= 1) {
            float ov = simd_shuffle_down(best, o);
            int oi = simd_shuffle_down(bi, o);
            if (ov > best || (ov == best && oi > bi)) {
                best = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            out[0] = bi;
        }
    }
}
