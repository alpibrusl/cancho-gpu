edition 5;
module dialect;

// How a target spells what every backend needs: `lex_msl::dialect`,
// ported. `d` is `msl()` or `cuda()`; the lowering asks here for every
// spelling that differs and nowhere else, which is the seam the Rust
// draws and the reason one lowering serves both targets.

import mem;
import text;
import kinds;

pub fn msl() -> [] int {
    return 0;
}

pub fn cuda() -> [] int {
    return 1;
}

fn is_cuda(d: int) -> [] bool {
    return d == 1;
}

pub fn barrier[&m, &t](m: &!m [int], t: &!t [byte], d: int) -> [] int {
    if is_cuda(d) {
        return text.lit(m, t, "__syncthreads();");
    }
    return text.lit(m, t, "threadgroup_barrier(mem_flags::mem_threadgroup);");
}

pub fn shuffle[&m, &t](m: &!m [int], t: &!t [byte], d: int, value: int, src: int) -> [] int {
    if is_cuda(d) {
        return text.f2(m, t, "__shfl_sync(0xffffffffu, $, $)", value, src);
    }
    return text.f2(m, t, "simd_shuffle($, $)", value, src);
}

pub fn shuffle_down[&m, &t](m: &!m [int], t: &!t [byte], d: int, value: int, delta: int) -> [] int {
    if is_cuda(d) {
        return text.f2(m, t, "__shfl_down_sync(0xffffffffu, $, $)", value, delta);
    }
    return text.f2(m, t, "simd_shuffle_down($, $)", value, delta);
}

pub fn scalar_lit(d: int, dt: int) -> [] &static [byte] {
    if is_cuda(d) {
        if dt == 0 {
            return "__half";
        }
        if dt == 1 {
            return "float";
        }
        return "char";
    }
    if dt == 0 {
        return "half";
    }
    if dt == 1 {
        return "float";
    }
    return "char";
}

pub fn scalar[&m, &t](m: &!m [int], t: &!t [byte], d: int, dt: int) -> [] int {
    return text.lit(m, t, scalar_lit(d, dt));
}

pub fn shared_ptr[&m, &t](m: &!m [int], t: &!t [byte], d: int, ty: int) -> [] int {
    if is_cuda(d) {
        return text.f1(m, t, "$*", ty);
    }
    return text.f1(m, t, "threadgroup $*", ty);
}

pub fn shared_array[&m, &t](m: &!m [int], t: &!t [byte], d: int, ty: int, name: int, n: int) -> [] int {
    if is_cuda(d) {
        return text.f3(m, t, "__shared__ $ $[$];", ty, name, text.num(m, t, n));
    }
    return text.f3(m, t, "threadgroup $ $[$];", ty, name, text.num(m, t, n));
}

pub fn shared_array_aligned[&m, &t](m: &!m [int], t: &!t [byte], d: int, ty: int, name: int, n: int, align: int) -> [] int {
    if is_cuda(d) {
        return text.f4(m, t, "__shared__ __align__($) $ $[$];", text.num(m, t, align), ty, name, text.num(m, t, n));
    }
    return shared_array(m, t, d, ty, name, n);
}

pub fn exp[&m, &t](m: &!m [int], t: &!t [byte], d: int, x: int) -> [] int {
    if is_cuda(d) {
        return text.f1(m, t, "expf($)", x);
    }
    return text.f1(m, t, "precise::exp($)", x);
}

pub fn log[&m, &t](m: &!m [int], t: &!t [byte], d: int, x: int) -> [] int {
    if is_cuda(d) {
        return text.f1(m, t, "logf($)", x);
    }
    return text.f1(m, t, "precise::log($)", x);
}

pub fn rsqrt[&m, &t](m: &!m [int], t: &!t [byte], d: int, x: int) -> [] int {
    if is_cuda(d) {
        return text.f1(m, t, "rsqrtf($)", x);
    }
    return text.f1(m, t, "precise::rsqrt($)", x);
}

pub fn fabs[&m, &t](m: &!m [int], t: &!t [byte], d: int, x: int) -> [] int {
    if is_cuda(d) {
        return text.f1(m, t, "fabsf($)", x);
    }
    return text.f1(m, t, "fabs($)", x);
}

pub fn fmax[&m, &t](m: &!m [int], t: &!t [byte], d: int, a: int, b: int) -> [] int {
    if is_cuda(d) {
        return text.f2(m, t, "fmaxf($, $)", a, b);
    }
    return text.f2(m, t, "max($, $)", a, b);
}

pub fn convert[&m, &t](m: &!m [int], t: &!t [byte], d: int, dt: int, x: int) -> [] int {
    if is_cuda(d) && dt == kinds.f16() {
        return text.f1(m, t, "__float2half($)", x);
    }
    return text.f2(m, t, "$($)", scalar(m, t, d, dt), x);
}

pub fn includes[&m, &t](m: &!m [int], t: &!t [byte], d: int) -> [] int {
    if is_cuda(d) {
        return text.lit(m, t, "\n#include <cuda_fp16.h>\n\ntypedef unsigned int uint;\ntypedef unsigned char uchar;\n#ifndef INFINITY\n#define INFINITY __int_as_float(0x7f800000)\n#endif\n\n");
    }
    return text.lit(m, t, "\n#include <metal_stdlib>\nusing namespace metal;\n\n");
}

// The most threadgroup memory this backend can declare: CUDA caps a
// static `__shared__` declaration at 48 KiB.
pub fn max_static_shared(d: int) -> [] int {
    if is_cuda(d) {
        return 48 * 1024;
    }
    return 4611686018427387904;
}

// The vector a cooperative copy of `dtype` moves through, and its width:
// `(ty, n)`, or `n == 0` for none.
pub fn copy_vector(d: int, dt: int) -> [] (&static [byte], int) {
    if dt != kinds.f16() {
        return ("", 0);
    }
    if is_cuda(d) {
        return ("uint4", 8);
    }
    return ("half4", 4);
}

pub fn vector_store_shared[&m, &t](m: &!m [int], t: &!t [byte], d: int, ty: int, lvalue: int, value: int) -> [] int {
    if is_cuda(d) {
        return text.f3(m, t, "*reinterpret_cast<$*>(&$) = $;", ty, lvalue, value);
    }
    return text.f3(m, t, "*(threadgroup $*)(&$) = $;", ty, lvalue, value);
}

// Lines decoding one group of sixteen NVFP4 codes into halves.
pub fn fp4_group_store[&m, &t](m: &!m [int], t: &!t [byte], d: int, dst: int, w: int, sc: int) -> [] int {
    let out = mem.list(m);
    if is_cuda(d) {
        mem.push(m, out, text.lit(m, t, "__half2 h[8];"));
        mem.push(m, out, text.lit(m, t, "for (uint b = 0; b < 8u; ++b) {"));
        mem.push(m, out, text.f2(m, t, "    const float2 v = fp4_pair(((b < 4u ? $.x : $.y) >> (8u * (b & 3u))) & 0xFFu);", w, w));
        mem.push(m, out, text.f2(m, t, "    h[b] = __floats2half2_rn(v.x * $, v.y * $);", sc, sc));
        mem.push(m, out, text.lit(m, t, "}"));
        mem.push(m, out, text.f1(m, t, "*reinterpret_cast<uint4*>(&$) = *reinterpret_cast<uint4*>(&h[0]);", dst));
        mem.push(m, out, text.f1(m, t, "*reinterpret_cast<uint4*>(&$ + 8) = *reinterpret_cast<uint4*>(&h[4]);", dst));
        return out;
    }
    mem.push(m, out, text.f1(m, t, "threadgroup half4* row = (threadgroup half4*)(&$);", dst));
    mem.push(m, out, text.lit(m, t, "for (uint b = 0; b < 4u; ++b) {"));
    mem.push(m, out, text.f2(m, t, "    const uint wv = (b < 2u ? $.x : $.y) >> ((b & 1u) * 16u);", w, w));
    mem.push(m, out, text.lit(m, t, "    const float2 lo = fp4_pair(wv & 0xFFu), hi = fp4_pair((wv >> 8u) & 0xFFu);"));
    let l = mem.of2(m, sc, sc);
    mem.push(m, l, sc);
    mem.push(m, l, sc);
    mem.push(m, out, text.fmt(m, t, "    row[b] = half4(half(lo.x * $), half(lo.y * $), half(hi.x * $), half(hi.y * $));", l));
    mem.push(m, out, text.lit(m, t, "}"));
    return out;
}

// Whether a lane's run of loads is spelled as explicit wide loads.
pub fn wide_loads(d: int) -> [] bool {
    return is_cuda(d);
}

pub fn vector_load[&m, &t](m: &!m [int], t: &!t [byte], d: int, ty: int, lvalue: int) -> [] int {
    if is_cuda(d) {
        return text.f2(m, t, "*reinterpret_cast<const $*>(&$)", ty, lvalue);
    }
    return text.f2(m, t, "*(const device $ *)(&$)", ty, lvalue);
}

pub fn fp4_preamble[&m, &t](m: &!m [int], t: &!t [byte], d: int) -> [] int {
    if is_cuda(d) {
        return text.lit(m, t, "__constant__ float FP4_V[16] = {\n    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,\n    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};\n__device__ __forceinline__ float2 fp4_pair(uint b) {\n    const uint w = b | (b << 12u);\n    const uint bits = ((w & 0x00070007u) << 9u)\n                    | ((w & 0x00080008u) << 12u);\n    __half2 h;\n    memcpy(&h, &bits, sizeof(h));\n    return __half22float2(h);\n}\n__device__ __forceinline__ float fp8_e4m3(uint b) {\n    const uint e = (b >> 3u) & 0xFu, m = b & 7u;\n    const float sub = float(m) * 0.001953125f;\n    uint subbits;\n    memcpy(&subbits, &sub, sizeof(subbits));\n    const uint bits = (e == 0u) ? subbits : (((e + 120u) << 23u) | (m << 20u));\n    const uint out = bits | ((b & 0x80u) << 24u);\n    float r;\n    memcpy(&r, &out, sizeof(r));\n    return r;\n}\n\n");
    }
    return text.lit(m, t, "constant float FP4_V[16] = {\n    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,\n    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};\ninline float2 fp4_pair(uint b) {\n    const uint w = b | (b << 12u);\n    return float2(as_type<half2>(((w & 0x00070007u) << 9u)\n                               | ((w & 0x00080008u) << 12u)));\n}\ninline float fp8_e4m3(uint b) {\n    const uint e = (b >> 3u) & 0xFu, m = b & 7u;\n    const uint bits = select(((e + 120u) << 23u) | (m << 20u),\n                             as_type<uint>(float(m) * 0.001953125f), e == 0u);\n    return as_type<float>(bits | ((b & 0x80u) << 24u));\n}\n\n");
}

// The entry point, through the opening brace and the declarations that
// say where this thread is. `params` is a list of `[ty, name, writable]`.
pub fn entry_point[&m, &t](m: &!m [int], t: &!t [byte], d: int, name: int, params: int, scalars: bool, gid2: bool) -> [] int {
    let lines = mem.list(m);
    if is_cuda(d) {
        mem.push(m, lines, text.f1(m, t, "extern \"C\" __global__ void $(\n", name));
        var i = 0;
        while i < mem.size(m, params) {
            let p = mem.get(m, params, i);
            var cv = text.empty();
            if m[p + 2] == 0 {
                cv = text.lit(m, t, "const ");
            }
            mem.push(m, lines, text.f3(m, t, "    $$* __restrict__ $,\n", cv, m[p], m[p + 1]));
            i = i + 1;
        }
        if scalars {
            mem.push(m, lines, text.lit(m, t, "    const uint* __restrict__ scalars,\n"));
        }
        var s = text.joined(m, t, lines, "");
        // Trailing comma: Metal ends its list with the index parameters,
        // CUDA has none to end with.
        if text.ends_with(t, s, ",\n") {
            s = text.cat(m, t, text.sub(s, 0, text.size(s) - 2), text.lit(m, t, "\n"));
        }
        var tail = text.lit(m, t, ")\n{\n    const uint tid = threadIdx.x;\n    const uint gid = blockIdx.x;\n");
        if gid2 {
            tail = text.lit(m, t, ")\n{\n    const uint tid = threadIdx.x;\n    const uint gid = blockIdx.x, gid2 = blockIdx.y;\n");
        }
        return text.cat(m, t, s, tail);
    }
    mem.push(m, lines, text.f1(m, t, "kernel void $(\n", name));
    var i = 0;
    while i < mem.size(m, params) {
        let p = mem.get(m, params, i);
        var cv = text.empty();
        if m[p + 2] == 0 {
            cv = text.lit(m, t, "const ");
        }
        mem.push(m, lines, text.f4(m, t, "    device $$* $ [[buffer($)]],\n", cv, m[p], m[p + 1], text.num(m, t, i)));
        i = i + 1;
    }
    if scalars {
        mem.push(m, lines, text.n1(m, t, "    constant uint* scalars [[buffer($)]],\n", mem.size(m, params)));
    }
    mem.push(m, lines, text.lit(m, t, "    uint tid [[thread_index_in_threadgroup]],\n"));
    mem.push(m, lines, text.lit(m, t, "    uint3 tgpos [[threadgroup_position_in_grid]])\n{\n"));
    if gid2 {
        mem.push(m, lines, text.lit(m, t, "    const uint gid = tgpos.x, gid2 = tgpos.y;\n"));
    } else {
        mem.push(m, lines, text.lit(m, t, "    const uint gid = tgpos.x;\n"));
    }
    return text.joined(m, t, lines, "");
}

// ---------------------------------------------------------------------
// The matrix unit: CUDA `wmma` (16x16x16), Metal `simdgroup_matrix` (8x8)
// ---------------------------------------------------------------------

pub fn atom(d: int) -> [] int {
    if is_cuda(d) {
        return 16;
    }
    return 8;
}

pub fn matrix_includes[&m, &t](m: &!m [int], t: &!t [byte], d: int) -> [] int {
    if is_cuda(d) {
        return text.lit(m, t, "#include <mma.h>\n\n");
    }
    return text.lit(m, t, "#include <metal_simdgroup_matrix>\n\n");
}

pub fn frag_decl[&m, &t](m: &!m [int], t: &!t [byte], d: int, name: int, fm: int, fnn: int) -> [] int {
    if is_cuda(d) {
        return text.f3(m, t, "nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> $[$][$];", name, text.num(m, t, fm), text.num(m, t, fnn));
    }
    return text.f3(m, t, "simdgroup_matrix<float, 8, 8> $[$][$];", name, text.num(m, t, fm), text.num(m, t, fnn));
}

pub fn frag_fill[&m, &t](m: &!m [int], t: &!t [byte], d: int, frag: int, value: int) -> [] int {
    if is_cuda(d) {
        return text.f2(m, t, "nvcuda::wmma::fill_fragment($, $);", frag, value);
    }
    return text.f2(m, t, "$ = simdgroup_matrix<float, 8, 8>($);", frag, value);
}

// Lines adding `a b^T` into the accumulators `c[fm][fnn]`.
pub fn frag_mma[&m, &t](m: &!m [int], t: &!t [byte], d: int, c: int, a: int, b: int, k: int, lda: int, ldb: int, fm: int, fnn: int, a_row: int, b_row: int) -> [] int {
    let out = mem.list(m);
    let (sk, sa, sb) = (text.num(m, t, k), text.num(m, t, lda), text.num(m, t, ldb));
    let (sm, sn) = (text.num(m, t, fm), text.num(m, t, fnn));
    mem.push(m, out, text.lit(m, t, "{"));
    if is_cuda(d) {
        mem.push(m, out, text.f1(m, t, "    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> fa[$];", sm));
        mem.push(m, out, text.f1(m, t, "    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> fb[$];", sn));
        mem.push(m, out, text.f1(m, t, "    for (uint kk = 0; kk < $u; kk += 16u) {", sk));
        mem.push(m, out, text.f5(m, t, "        for (uint i = 0; i < $u; ++i) nvcuda::wmma::load_matrix_sync(fa[i], $ + ($ + i * 16u) * $u + kk, $u);", sm, a, a_row, sa, sa));
        mem.push(m, out, text.f5(m, t, "        for (uint j = 0; j < $u; ++j) nvcuda::wmma::load_matrix_sync(fb[j], $ + ($ + j * 16u) * $u + kk, $u);", sn, b, b_row, sb, sb));
        mem.push(m, out, text.f1(m, t, "        for (uint i = 0; i < $u; ++i)", sm));
        mem.push(m, out, text.f3(m, t, "            for (uint j = 0; j < $u; ++j) nvcuda::wmma::mma_sync($[i][j], fa[i], fb[j], $[i][j]);", sn, c, c));
    } else {
        mem.push(m, out, text.f2(m, t, "    simdgroup_matrix<half, 8, 8> fa[$], fb[$];", sm, sn));
        mem.push(m, out, text.f1(m, t, "    for (uint kk = 0; kk < $u; kk += 8u) {", sk));
        mem.push(m, out, text.f5(m, t, "        for (uint i = 0; i < $u; ++i) simdgroup_load(fa[i], $ + ($ + i * 8u) * $u + kk, $u);", sm, a, a_row, sa, sa));
        mem.push(m, out, text.f5(m, t, "        for (uint j = 0; j < $u; ++j) simdgroup_load(fb[j], $ + ($ + j * 8u) * $u + kk, $u, ulong2(0, 0), true);", sn, b, b_row, sb, sb));
        mem.push(m, out, text.f1(m, t, "        for (uint i = 0; i < $u; ++i)", sm));
        mem.push(m, out, text.f3(m, t, "            for (uint j = 0; j < $u; ++j) simdgroup_multiply_accumulate($[i][j], fa[i], fb[j], $[i][j]);", sn, c, c));
    }
    mem.push(m, out, text.lit(m, t, "    }"));
    mem.push(m, out, text.lit(m, t, "}"));
    return out;
}

pub fn frag_store[&m, &t](m: &!m [int], t: &!t [byte], d: int, frag: int, ptr: int, ldm: int) -> [] int {
    if is_cuda(d) {
        return text.f3(m, t, "nvcuda::wmma::store_matrix_sync($, $, $u, nvcuda::wmma::mem_row_major);", ptr, frag, text.num(m, t, ldm));
    }
    return text.f3(m, t, "simdgroup_store($, $, $u);", frag, ptr, text.num(m, t, ldm));
}

pub fn frag_add_loaded[&m, &t](m: &!m [int], t: &!t [byte], d: int, frag: int, ptr: int, ldm: int) -> [] int {
    let out = mem.list(m);
    mem.push(m, out, text.lit(m, t, "{"));
    if is_cuda(d) {
        mem.push(m, out, text.lit(m, t, "    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> rf;"));
        mem.push(m, out, text.f2(m, t, "    nvcuda::wmma::load_matrix_sync(rf, $, $u, nvcuda::wmma::mem_row_major);", ptr, text.num(m, t, ldm)));
        mem.push(m, out, text.f1(m, t, "    for (int t = 0; t < rf.num_elements; ++t) $.x[t] += rf.x[t];", frag));
    } else {
        mem.push(m, out, text.lit(m, t, "    simdgroup_matrix<float, 8, 8> rf;"));
        mem.push(m, out, text.f2(m, t, "    simdgroup_load(rf, $, $u);", ptr, text.num(m, t, ldm)));
        mem.push(m, out, text.f2(m, t, "    simdgroup_multiply_accumulate($, simdgroup_matrix<float, 8, 8>(1.0f), rf, $);", frag, frag));
    }
    mem.push(m, out, text.lit(m, t, "}"));
    return out;
}

pub fn store_align(d: int) -> [] int {
    if is_cuda(d) {
        return 8;
    }
    return 1;
}

pub fn default_pad(d: int) -> [] int {
    if is_cuda(d) {
        return 8;
    }
    return 0;
}
