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

// Half weights: the half2 the decode trick produces (value * 2^-14, exact) is multiplied by the
// group's scale in half (an E4M3 times an E2M1 has 5 significant bits: exact), and goes to the matrix
// unit as a half operand against the activations' float fragment: no per-weight float conversion.
template <int T>
inline void mma_rows_h(device const float *x, device const char *wq, device const char *ws,
                       device const float *wg, device float *y, uint tid, uint tgx, int M) {
    const uint lane = tid % 32u;
    const uint qid = lane / 4u;
    const uint fm = (qid & 4u) + ((lane / 2u) % 4u);
    const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
    const uint g = fn / 2u;
    const uint rowbase = (tgx * 8u + tid / 32u) * 8u * T;
    simdgroup_float8x8 C[T];
    for (int t = 0; t < T; t++) C[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint blk0 = 0; blk0 < K / 32u; blk0 += 8u) {
        float xs[8][4][2];
        for (uint u = 0; u < 8u; u++) {
            const uint kk = (blk0 + u) * 32u + (fm / 2u) * 8u + (fm % 2u);
            for (uint s = 0; s < 4u; s++) {
                xs[u][s][0] = (int)fn < M ? x[fn * K + kk + 2u * s] : 0.0f;
                xs[u][s][1] = (int)(fn + 1u) < M ? x[(fn + 1u) * K + kk + 2u * s] : 0.0f;
            }
        }
        for (int t = 0; t < T; t++) {
            const uint row = rowbase + t * 8u + fm;
            uint w[8];
            half sgv[8];
            for (uint u = 0; u < 8u; u++) {
                const uint blk = blk0 + u;
                w[u] = *(device const uint *)(wq + row * (K / 2u) + blk * 16u + g * 4u);
                sgv[u] = (half)fp8_e4m3((uint)(uchar)ws[row * (K / 16u) + blk * 2u + g / 2u]);
            }
            for (uint u = 0; u < 8u; u++)
                for (uint s = 0; s < 4u; s++) {
                    simdgroup_half8x8 A;
                    simdgroup_float8x8 B;
                    const uint b = (w[u] >> (8u * s)) & 0xFFu;
                    const uint ww = b | (b << 12u);
                    const half2 q = as_type<half2>(((ww & 0x00070007u) << 9u) | ((ww & 0x00080008u) << 12u)) * sgv[u];
                    A.thread_elements()[0] = q.x; A.thread_elements()[1] = q.y;
                    B.thread_elements()[0] = xs[u][s][0]; B.thread_elements()[1] = xs[u][s][1];
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
#define KH_(NAME, T, M) kernel void NAME(device const float *x [[buffer(0)]], device const char *wq [[buffer(1)]], \
    device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]], device float *y [[buffer(4)]], \
    uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) { mma_rows_h<T>(x, wq, ws, wg, y, tid, tg.x, M); }
KH_(h1_1, 1, 1) KH_(h3_1, 1, 3) KH_(h8_1, 1, 8) KH_(h8_2, 2, 8)

// All half: the activations as half too (prepared once a layer), so a step is a half2 multiply by the scale,
// two half loads and a half x half matrix multiply with float accumulation.
template <int T>
inline void mma_rows_hh(device const half *x, device const char *wq, device const char *ws,
                        device const float *wg, device float *y, uint tid, uint tgx, int M) {
    const uint lane = tid % 32u;
    const uint qid = lane / 4u;
    const uint fm = (qid & 4u) + ((lane / 2u) % 4u);
    const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
    const uint g = fn / 2u;
    const uint rowbase = (tgx * 8u + tid / 32u) * 8u * T;
    simdgroup_float8x8 C[T];
    for (int t = 0; t < T; t++) C[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint blk0 = 0; blk0 < K / 32u; blk0 += 8u) {
        half xs[8][4][2];
        for (uint u = 0; u < 8u; u++) {
            const uint kk = (blk0 + u) * 32u + (fm / 2u) * 8u + (fm % 2u);
            for (uint s = 0; s < 4u; s++) {
                xs[u][s][0] = (int)fn < M ? x[fn * K + kk + 2u * s] : (half)0.0h;
                xs[u][s][1] = (int)(fn + 1u) < M ? x[(fn + 1u) * K + kk + 2u * s] : (half)0.0h;
            }
        }
        for (int t = 0; t < T; t++) {
            const uint row = rowbase + t * 8u + fm;
            uint w[8];
            half sgv[8];
            for (uint u = 0; u < 8u; u++) {
                const uint blk = blk0 + u;
                w[u] = *(device const uint *)(wq + row * (K / 2u) + blk * 16u + g * 4u);
                sgv[u] = (half)fp8_e4m3((uint)(uchar)ws[row * (K / 16u) + blk * 2u + g / 2u]);
            }
            for (uint u = 0; u < 8u; u++)
                for (uint s = 0; s < 4u; s++) {
                    simdgroup_half8x8 A, B;
                    const uint b = (w[u] >> (8u * s)) & 0xFFu;
                    const uint ww = b | (b << 12u);
                    const half2 q = as_type<half2>(((ww & 0x00070007u) << 9u) | ((ww & 0x00080008u) << 12u)) * sgv[u];
                    A.thread_elements()[0] = q.x; A.thread_elements()[1] = q.y;
                    B.thread_elements()[0] = xs[u][s][0]; B.thread_elements()[1] = xs[u][s][1];
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
#define KHH_(NAME, T, M) kernel void NAME(device const half *x [[buffer(0)]], device const char *wq [[buffer(1)]], \
    device const char *ws [[buffer(2)]], device const float *wg [[buffer(3)]], device float *y [[buffer(4)]], \
    uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) { mma_rows_hh<T>(x, wq, ws, wg, y, tid, tg.x, M); }
KHH_(g1_1, 1, 1) KHH_(g3_1, 1, 3) KHH_(g8_1, 1, 8) KHH_(g8_2, 2, 8)
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
        std::vector<__fp16> xh(x.size()); for (size_t i = 0; i < x.size(); i++) xh[i] = (__fp16)x[i];
        id<MTLBuffer> bxh = [dev newBufferWithBytes:xh.data() length:xh.size() * 2 options:0];
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
        struct V { const char *fn; int M, R; } vs[] = {
            {"k1_1", 1, 1}, {"k1_2", 1, 2}, {"k1_4", 1, 4},
            {"k2_1", 2, 1}, {"k2_2", 2, 2}, {"k2_4", 2, 4}, {"k3_2", 3, 2}, {"k3_4", 3, 4},
            {"k4_1", 4, 1}, {"k4_2", 4, 2}, {"k4_4", 4, 4}, {"k4_8", 4, 8},
            {"k6_2", 6, 2}, {"k6_4", 6, 4}, {"k8_1", 8, 1}, {"k8_2", 8, 2}, {"k8_4", 8, 4},
            {"w1_2", 1, 2}, {"w2_2", 2, 2}, {"w3_2", 3, 2}, {"w3_4", 3, 4}, {"w4_2", 4, 2}, {"w4_4", 4, 4}, {"w6_4", 6, 4}, {"w8_2", 8, 2}, {"w8_4", 8, 4}, {"w8_8", 8, 8},
            {"m1_1", 1, 64}, {"m2_1", 2, 64}, {"m3_1", 3, 64}, {"m4_1", 4, 64}, {"m8_1", 8, 64}, {"m1_2", 1, 128}, {"m2_2", 2, 128}, {"m4_2", 4, 128}, {"m8_2", 8, 128}, {"m4_4", 4, 256}, {"m8_4", 8, 256},
            {"h1_1", 1, 64}, {"h3_1", 3, 64}, {"h8_1", 8, 64}, {"h8_2", 8, 128},
            {"g1_1", 1, 64}, {"g3_1", 3, 64}, {"g8_1", 8, 64}, {"g8_2", 8, 128}};
        for (auto &v : vs) {
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:v.fn]] error:&err];
            if (!ps) { NSLog(@"%s: %@", v.fn, err); continue; }
            const int rpt = (v.fn[0] == 'm' || v.fn[0] == 'h' || v.fn[0] == 'g') ? v.R : 8 * v.R;
            auto launch = [&](int reps) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:ps];
                [e setBuffer:(v.fn[0] == 'g' ? bxh : bx) offset:0 atIndex:0]; [e setBuffer:bg offset:0 atIndex:3]; [e setBuffer:by offset:0 atIndex:4];
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
