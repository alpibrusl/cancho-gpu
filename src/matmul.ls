edition 5;
module matmul;

// Matrix products on the threads, not the matrix units: the `MatMulNT`
// and `MatMul` arms of `lex_msl::program::Gen::op` and its
// `batched_rows`, ported -- for the operands a `.lx` file can produce.
// A lazy NVFP4 dequantisation is the one packed weight format the
// surface has (`dequant_fp4`); the other packings of the Rust
// (`Pairs`, `Six`, ternary) come from builders the surface has no
// syntax for, and are not here.
//
// A dequantisation record (`Dq`) is
// `[v, s, row, cols, group, packed, qe, qc, qview]`: the per-element
// value, the per-group scale (templated on the group index), the
// row-only factor (templated on the row, or -1), and for NVFP4 the byte
// template `qe`, the row length in bytes `qc` and the window of the
// codes (or -1).

import mem;
import text;
import err;
import kinds;
import ir;
import dialect;
import gen;

pub fn dq[&m](m: &!m [int], v: int, s: int, row: int, cols: int, group: int, qe: int, qc: int, qview: int) -> [] int {
    let r = mem.grab(m, 9);
    m[r] = v;
    m[r + 1] = s;
    m[r + 2] = row;
    m[r + 3] = cols;
    m[r + 4] = group;
    m[r + 5] = 1;
    m[r + 6] = qe;
    m[r + 7] = qc;
    m[r + 8] = qview;
    return r;
}

fn n[&m, &t](m: &!m [int], t: &!t [byte], x: int) -> [] int {
    return text.num(m, t, x);
}

// Consecutive elements per lane per step in a split-K reduction: the
// largest of `default`, 8, 4, 2 that tiles the reduction evenly. (The
// Rust reads `LEX_VEC` first; this port has no environment, so the
// default is the answer, as it is when the variable is unset.)
fn split_k_vec(kd: int, lanes: int, want: int) -> [] int {
    if want > 0 && kd % (lanes * want) == 0 {
        return want;
    }
    if 8 <= want && kd % (lanes * 8) == 0 {
        return 8;
    }
    if 4 <= want && kd % (lanes * 4) == 0 {
        return 4;
    }
    if 2 <= want && kd % (lanes * 2) == 0 {
        return 2;
    }
    return 1;
}

// Whether an NVFP4 run can be loaded wide, and from what: answers
// `[weight lvalue, input lvalue]`, or -1. Only when the dialect asks and
// the alignment follows from the shape.
fn wide_fp4[&m, &t](m: &!m [int], t: &!t [byte], g: int, qe: int, qc: int, kd: int, vec: int, x: int) -> [] int {
    if !dialect.wide_loads(gen.dia(m, g)) || vec != 16 || qc % 8 != 0 || kd % 16 != 0 {
        return 0 - 1;
    }
    if m[x] != gen.a_lazy() {
        return 0 - 1;
    }
    let wq = gen.lvalue(t, qe);
    if wq < 0 {
        return 0 - 1;
    }
    let xa = gen.lvalue(t, m[x + 1]);
    if xa < 0 {
        return 0 - 1;
    }
    let dq_ok = param_dtype_of(m, t, g, wq) == kinds.i8();
    let dx_ok = param_dtype_of(m, t, g, xa) == kinds.f32();
    if dq_ok && dx_ok {
        return mem.rec2(m, wq, xa);
    }
    return 0 - 1;
}

// The dtype of the parameter an lvalue `pN_name[..]` reads, or -1.
fn param_dtype_of[&m, &t](m: &!m [int], t: &!t [byte], g: int, lv: int) -> [] int {
    let us = text.index_of(t, lv, '_', 1);
    if us < 2 {
        return 0 - 1;
    }
    var i = 0;
    var k = 1;
    while k < us {
        i = i * 10 + text.at(t, lv, k) - '0';
        k = k + 1;
    }
    let p = ir.p_params(m, gen.prog(m, g));
    if i >= mem.size(m, p) {
        return 0 - 1;
    }
    return ir.param_dtype(m, mem.get(m, p, i));
}

// The dequantisation a reduction may use, after the Rust's filter.
fn usable_dq[&m](m: &!m [int], d: int, kd: int, vec: int) -> [] bool {
    return vec > 1 && m[d + 3] == kd && m[d + 4] % vec == 0 && vec % 2 == 0;
}

pub fn lower[&m, &t](m: &!m [int], t: &!t [byte], g: int, x: int, o: int) -> [] int {
    let nt = m[o] == kinds.op_matmul_nt();
    let (a, b, acc) = (m[o + 1], m[o + 2], m[o + 3]);
    let ta = gen.arg_ty(m, t, g, a);
    if ta < 0 {
        return 0 - 1;
    }
    let tb = gen.arg_ty(m, t, g, b);
    if tb < 0 {
        return 0 - 1;
    }
    let mm = mem.get(m, ir.shape(m, ta), 0);
    let kd = mem.get(m, ir.shape(m, ta), 1);
    var nn = mem.get(m, ir.shape(m, tb), 1);
    if nt {
        nn = mem.get(m, ir.shape(m, tb), 0);
    }
    let ops = gen.operands(m, t, g, mem.of2(m, a, b), gen.flags2(m, false, false), mm * nn);
    if ops < 0 {
        return 0 - 1;
    }
    let (oa, ob) = (mem.get(m, ops, 0), mem.get(m, ops, 1));
    let av = gen.read(m, t, oa, text.f1(m, t, "i * $u + p", n(m, t, kd)));
    var bv = gen.read(m, t, ob, text.f1(m, t, "p * $u + j", n(m, t, nn)));
    if nt {
        bv = gen.read(m, t, ob, text.f1(m, t, "j * $u + p", n(m, t, kd)));
    }
    let threads = gen.threads(m, g);
    let weight = gen.lazy(m, g, b) >= 0;
    if nt && weight && mm > 1 && mm <= 16 && nn < threads {
        return batched_rows(m, t, g, x, acc, mm, nn, kd, b, ops);
    }
    let outs = mm * nn;
    var lanes = 1;
    while lanes * 2 <= threads / outs && lanes * 2 <= 32 {
        lanes = lanes * 2;
    }
    if lanes < 2 {
        let name = gen.declare_reg(m, t, g, x, ir.tile(m, acc, mem.of2(m, mm, nn), kinds.reg()));
        let body = mem.list(m);
        mem.push(m, body, text.f2(m, t, "const uint i = e / $u, j = e % $u;", n(m, t, nn), n(m, t, nn)));
        mem.push(m, body, text.lit(m, t, "float s = 0.0f;"));
        mem.push(m, body, text.f3(m, t, "for (uint p = 0; p < $u; ++p) s += $ * $;", n(m, t, kd), av, bv));
        mem.push(m, body, text.f2(m, t, "$[k] = $;", name, dialect.convert(m, t, gen.dia(m, g), acc, text.lit(m, t, "s"))));
        return gen.owned(m, t, g, outs, body);
    }
    // Fewer outputs than threads: a power-of-two group of lanes per
    // output, the reduction split across them and combined by shuffles.
    let resv = gen.staged(m, g);
    gen.need_scratch(m, g, resv + outs);
    if gen.staged(m, g) == 0 {
        gen.barrier(m, t, g);
    }
    gen.line_lit(m, t, g, "{");
    gen.depth_in(m, g);
    gen.line(m, t, g, text.f2(m, t, "const uint o = tid / $u, lane = tid % $u;", n(m, t, lanes), n(m, t, lanes)));
    gen.line_lit(m, t, g, "float s = 0.0f;");
    gen.line(m, t, g, text.f1(m, t, "if (o < $u) {", n(m, t, outs)));
    gen.depth_in(m, g);
    gen.line(m, t, g, text.f2(m, t, "const uint i = o / $u, j = o % $u;", n(m, t, nn), n(m, t, nn)));
    var d = 0 - 1;
    if nt {
        let dv = gen.dq_of(m, g, ir.var_of(b));
        if dv != 0 {
            d = dv;
        }
    }
    var want = 8;
    if d >= 0 {
        want = 16;
    }
    let vec = split_k_vec(kd, lanes, want);
    if d >= 0 && !usable_dq(m, d, kd, vec) {
        d = 0 - 1;
    }
    if d >= 0 {
        let gsz = m[d + 4];
        var rs = text.empty();
        if m[d + 2] >= 0 {
            gen.line(m, t, g, text.f1(m, t, "const float rs = float($);", gen.at_index_lit(m, t, m[d + 2], "j")));
            rs = text.lit(m, t, " * rs");
        }
        gen.line(m, t, g, text.f3(m, t, "for (uint p0 = lane * $u; p0 < $u; p0 += $u) {", n(m, t, vec), n(m, t, kd), n(m, t, lanes * vec)));
        gen.line(m, t, g, text.f2(m, t, "    const uint grp = j * $u + p0 / $u;", n(m, t, kd / gsz), n(m, t, gsz)));
        gen.line(m, t, g, text.f2(m, t, "    const float sg = $$;", gen.at_index_lit(m, t, m[d + 1], "grp"), rs));
        let (qe, qc) = (m[d + 6], m[d + 7]);
        let wide = wide_fp4(m, t, g, qe, qc, kd, vec, oa);
        if wide >= 0 {
            let (wq, xa) = (m[wide], m[wide + 1]);
            let w = gen.at_index(m, t, wq, text.f1(m, t, "j * $u + p0 / 2u", n(m, t, qc)));
            let u2 = text.lit(m, t, "uint2");
            gen.line(m, t, g, text.f1(m, t, "    const uint2 wq = $;", dialect.vector_load(m, t, gen.dia(m, g), u2, w)));
            var q = 0;
            while q < 4 {
                let xq = gen.at_index(m, t, xa, text.f2(m, t, "i * $u + p0 + $u", n(m, t, kd), n(m, t, 4 * q)));
                let f4 = text.lit(m, t, "float4");
                gen.line(m, t, g, text.f2(m, t, "    const float4 x$ = $;", n(m, t, q), dialect.vector_load(m, t, gen.dia(m, g), f4, xq)));
                q = q + 1;
            }
            gen.line_lit(m, t, g, "    float run = 0.0f;");
            var bb = 0;
            while bb < 8 {
                var word = text.lit(m, t, "wq.x");
                if bb >= 4 {
                    word = text.lit(m, t, "wq.y");
                }
                let qq = 2 * bb / 4;
                let cc = 2 * bb % 4;
                let l = mem.of6(m, word, n(m, t, 8 * (bb % 4)), n(m, t, qq), lane_name(m, t, cc), n(m, t, qq), lane_name(m, t, cc + 1));
                gen.line(m, t, g, text.fmt(m, t, "    { const float2 w = fp4_pair(($ >> $u) & 0xFFu); run += x$.$ * w.x + x$.$ * w.y; }", l));
                bb = bb + 1;
            }
            gen.line_lit(m, t, g, "    s += run * (sg * 16384.0f);");
        } else {
            let byte = gen.at_index(m, t, qe, text.f1(m, t, "j * $u + p0 / 2u + u / 2u", n(m, t, qc)));
            let a1 = gen.read(m, t, oa, text.f1(m, t, "i * $u + p + 1u", n(m, t, kd)));
            gen.line_lit(m, t, g, "    float run = 0.0f;");
            gen.line(m, t, g, text.f4(m, t, "    for (uint u = 0; u < $u; u += 2u) { const uint p = p0 + u; const uint bq = (uint)(uchar)($); const float2 w = fp4_pair(bq); run += $ * w.x + $ * w.y; }", n(m, t, vec), byte, av, a1));
            gen.line_lit(m, t, g, "    s += run * (sg * 16384.0f);");
        }
        gen.line_lit(m, t, g, "}");
    } else if vec > 1 {
        gen.line(m, t, g, text.f3(m, t, "for (uint p0 = lane * $u; p0 < $u; p0 += $u) {", n(m, t, vec), n(m, t, kd), n(m, t, lanes * vec)));
        gen.line(m, t, g, text.f3(m, t, "    for (uint u = 0; u < $u; ++u) { const uint p = p0 + u; s += $ * $; }", n(m, t, vec), av, bv));
        gen.line_lit(m, t, g, "}");
    } else {
        gen.line(m, t, g, text.f4(m, t, "for (uint p = lane; p < $u; p += $u) s += $ * $;", n(m, t, kd), n(m, t, lanes), av, bv));
    }
    gen.depth_out(m, g);
    gen.line_lit(m, t, g, "}");
    let dn = dialect.shuffle_down(m, t, gen.dia(m, g), text.lit(m, t, "s"), text.lit(m, t, "d"));
    gen.line(m, t, g, text.f2(m, t, "for (uint d = $u; d > 0; d /= 2) s += $;", n(m, t, lanes / 2), dn));
    gen.line(m, t, g, text.f2(m, t, "if (o < $u && lane == 0) scratch[$ + o] = s;", n(m, t, outs), n(m, t, resv)));
    gen.depth_out(m, g);
    gen.line_lit(m, t, g, "}");
    gen.barrier(m, t, g);
    let name = gen.declare_reg(m, t, g, x, ir.tile(m, acc, mem.of2(m, mm, nn), kinds.reg()));
    let cv = dialect.convert(m, t, gen.dia(m, g), acc, text.f1(m, t, "scratch[$ + e]", n(m, t, resv)));
    return gen.owned1(m, t, g, outs, text.f2(m, t, "$[k] = $;", name, cv));
}

fn lane_name[&m, &t](m: &!m [int], t: &!t [byte], i: int) -> [] int {
    if i == 0 {
        return text.lit(m, t, "x");
    }
    if i == 1 {
        return text.lit(m, t, "y");
    }
    if i == 2 {
        return text.lit(m, t, "z");
    }
    return text.lit(m, t, "w");
}

// `out[i, j] = sum_p a[i, p] * b[j, p]` for a few rows `i`: each
// simdgroup owns `r` weight rows and an accumulator per (row, token).
fn batched_rows[&m, &t](m: &!m [int], t: &!t [byte], g: int, x: int, acc: int, mm: int, nn: int, kd: int, b: int, ops: int) -> [] int {
    let simd = 32;
    var groups = gen.threads(m, g) / simd;
    if groups < 1 {
        groups = 1;
    }
    let r = (nn + groups - 1) / groups;
    let lanes = simd;
    var d = 0 - 1;
    let dv = gen.dq_of(m, g, ir.var_of(b));
    if dv != 0 {
        d = dv;
    }
    var want = 8;
    if d >= 0 {
        want = 16;
    }
    let vec = split_k_vec(kd, lanes, want);
    if d >= 0 && !usable_dq(m, d, kd, vec) {
        d = 0 - 1;
    }
    let (oa, ob) = (mem.get(m, ops, 0), mem.get(m, ops, 1));
    var step = 1;
    var pre = text.empty();
    let weights = mem.list(m);
    if d >= 0 {
        let byte = gen.at_index(m, t, m[d + 6], text.f1(m, t, "j * $u + p0 / 2u + u / 2u", n(m, t, m[d + 7])));
        step = 2;
        pre = text.f1(m, t, "const float2 bq = fp4_pair((uint)(uchar)($)); ", byte);
        mem.push(m, weights, text.lit(m, t, "(bq.x * sgr[rr])"));
        mem.push(m, weights, text.lit(m, t, "(bq.y * sgr[rr])"));
    } else {
        mem.push(m, weights, gen.read(m, t, ob, text.f1(m, t, "j * $u + p", n(m, t, kd))));
    }
    let resv = gen.staged(m, g);
    gen.need_scratch(m, g, resv + mm * nn);
    if gen.staged(m, g) == 0 {
        gen.barrier(m, t, g);
    }
    var v = vec;
    if v < 1 {
        v = 1;
    }
    gen.line_lit(m, t, g, "{");
    gen.depth_in(m, g);
    gen.line(m, t, g, text.f2(m, t, "const uint sgid = tid / $u, lane = tid % $u;", n(m, t, simd), n(m, t, simd)));
    let unroll = r * mm * step <= 128;
    let deferred = unroll && r * mm <= 16 && d >= 0;
    var ws = weights;
    if deferred {
        ws = mem.list(m);
        var i = 0;
        while i < mem.size(m, weights) {
            mem.push(m, ws, text.replace_lit(m, t, mem.get(m, weights, i), " * sgr[rr]", text.empty()));
            i = i + 1;
        }
    }
    gen.line(m, t, g, text.f2(m, t, "float s[$][$];", n(m, t, r), n(m, t, mm)));
    if unroll {
        var rr = 0;
        while rr < r {
            let parts = mem.list(m);
            var i = 0;
            while i < mm {
                mem.push(m, parts, text.f2(m, t, "s[$][$] = 0.0f;", n(m, t, rr), n(m, t, i)));
                i = i + 1;
            }
            gen.line(m, t, g, text.joined(m, t, parts, " "));
            rr = rr + 1;
        }
    } else {
        gen.line(m, t, g, text.f2(m, t, "for (uint rr = 0; rr < $u; ++rr) for (uint i = 0; i < $u; ++i) s[rr][i] = 0.0f;", n(m, t, r), n(m, t, mm)));
    }
    gen.line(m, t, g, text.f3(m, t, "for (uint p0 = lane * $u; p0 < $u; p0 += $u) {", n(m, t, v), n(m, t, kd), n(m, t, lanes * v)));
    gen.depth_in(m, g);
    if d >= 0 {
        let gsz = m[d + 4];
        gen.line(m, t, g, text.f2(m, t, "float sgr[$], mgr[$];", n(m, t, r), n(m, t, r)));
        var rs = text.empty();
        if m[d + 2] >= 0 {
            rs = text.f1(m, t, " * float($)", gen.at_index_lit(m, t, m[d + 2], "j"));
        }
        rs = text.cat(m, t, rs, text.lit(m, t, " * 16384.0f"));
        let sgrp = gen.at_index_lit(m, t, m[d + 1], "grp");
        if unroll {
            var rr = 0;
            while rr < r {
                let l = mem.of7(m, n(m, t, r), text.f1(m, t, "$u", n(m, t, rr)), n(m, t, nn - 1), n(m, t, kd / gsz), n(m, t, gsz), n(m, t, rr), sgrp);
                mem.push(m, l, rs);
                mem.push(m, l, n(m, t, rr));
                gen.line(m, t, g, text.fmt(m, t, "{ const uint j = min(sgid * $u + $, $u); const uint grp = j * $u + p0 / $u; sgr[$] = $$; mgr[$] = 0.0f; }", l));
                rr = rr + 1;
            }
        } else {
            let l = mem.of7(m, n(m, t, r), n(m, t, r), text.lit(m, t, "rr"), n(m, t, nn - 1), n(m, t, kd / gsz), n(m, t, gsz), text.lit(m, t, "rr"));
            mem.push(m, l, sgrp);
            mem.push(m, l, rs);
            mem.push(m, l, text.lit(m, t, "rr"));
            gen.line(m, t, g, text.fmt(m, t, "for (uint rr = 0; rr < $u; ++rr) { const uint j = min(sgid * $u + $, $u); const uint grp = j * $u + p0 / $u; sgr[$] = $$; mgr[$] = 0.0f; }", l));
        }
    }
    var wide = 0 - 1;
    if d >= 0 && unroll {
        wide = wide_fp4(m, t, g, m[d + 6], m[d + 7], kd, vec, oa);
    }
    if wide >= 0 {
        let (wq, xa, qc) = (m[wide], m[wide + 1], m[d + 7]);
        gen.line(m, t, g, text.f1(m, t, "uint2 wq[$];", n(m, t, r)));
        var rr = 0;
        while rr < r {
            let w = gen.at_index(m, t, wq, text.f1(m, t, "j * $u + p0 / 2u", n(m, t, qc)));
            let ld = dialect.vector_load(m, t, gen.dia(m, g), text.lit(m, t, "uint2"), w);
            gen.line(m, t, g, text.f5(m, t, "{ const uint j = min(sgid * $u + $u, $u); wq[$] = $; }", n(m, t, r), n(m, t, rr), n(m, t, nn - 1), n(m, t, rr), ld));
            rr = rr + 1;
        }
        var q = 0;
        while q < 4 {
            gen.line_lit(m, t, g, "{");
            gen.depth_in(m, g);
            var i = 0;
            while i < mm {
                let xq = gen.at_index(m, t, xa, text.f2(m, t, "$u + p0 + $u", n(m, t, i * kd), n(m, t, 4 * q)));
                let ld = dialect.vector_load(m, t, gen.dia(m, g), text.lit(m, t, "float4"), xq);
                gen.line(m, t, g, text.f2(m, t, "const float4 x$ = $;", n(m, t, i), ld));
                i = i + 1;
            }
            rr = 0;
            while rr < r {
                var half = 0;
                while half < 2 {
                    let byte = 2 * q + half;
                    var word = text.lit(m, t, "x");
                    if byte >= 4 {
                        word = text.lit(m, t, "y");
                    }
                    var c0 = text.lit(m, t, "x");
                    var c1 = text.lit(m, t, "y");
                    if half == 1 {
                        c0 = text.lit(m, t, "z");
                        c1 = text.lit(m, t, "w");
                    }
                    let sums = mem.list(m);
                    i = 0;
                    while i < mm {
                        let l = mem.of6(m, n(m, t, rr), n(m, t, i), n(m, t, i), c0, n(m, t, i), c1);
                        mem.push(m, sums, text.fmt(m, t, "s[$][$] += x$.$ * w0 + x$.$ * w1;", l));
                        i = i + 1;
                    }
                    let l = mem.of6(m, n(m, t, rr), word, n(m, t, 8 * (byte % 4)), n(m, t, rr), n(m, t, rr), text.joined(m, t, sums, " "));
                    gen.line(m, t, g, text.fmt(m, t, "{ const float2 w = fp4_pair((wq[$].$ >> $u) & 0xFFu); const float w0 = w.x * sgr[$], w1 = w.y * sgr[$]; $ }", l));
                    half = half + 1;
                }
                rr = rr + 1;
            }
            gen.depth_out(m, g);
            gen.line_lit(m, t, g, "}");
            q = q + 1;
        }
    } else {
        if deferred {
            gen.line(m, t, g, text.f2(m, t, "float run[$][$];", n(m, t, r), n(m, t, mm)));
            var rr = 0;
            while rr < r {
                let parts = mem.list(m);
                var i = 0;
                while i < mm {
                    mem.push(m, parts, text.f2(m, t, "run[$][$] = 0.0f;", n(m, t, rr), n(m, t, i)));
                    i = i + 1;
                }
                gen.line(m, t, g, text.joined(m, t, parts, " "));
                rr = rr + 1;
            }
        }
        gen.line(m, t, g, text.f2(m, t, "for (uint u = 0; u < $u; u += $u) {", n(m, t, v), n(m, t, step)));
        gen.depth_in(m, g);
        gen.line_lit(m, t, g, "const uint p = p0 + u;");
        let decl = mem.list(m);
        var k = 0;
        while k < mem.size(m, ws) {
            mem.push(m, decl, text.f2(m, t, "const float w$ = $;", n(m, t, k), mem.get(m, ws, k)));
            k = k + 1;
        }
        gen.line(m, t, g, text.f2(m, t, "float xa[$][$];", n(m, t, mm), n(m, t, step)));
        k = 0;
        while k < step {
            if unroll {
                var i = 0;
                while i < mm {
                    let rd = gen.read(m, t, oa, text.f1(m, t, "p + $u", n(m, t, i * kd + k)));
                    gen.line(m, t, g, text.f3(m, t, "xa[$][$] = $;", n(m, t, i), n(m, t, k), rd));
                    i = i + 1;
                }
            } else {
                let rd = gen.read(m, t, oa, text.f2(m, t, "i * $u + p + $u", n(m, t, kd), n(m, t, k)));
                gen.line(m, t, g, text.f3(m, t, "for (uint i = 0; i < $u; ++i) xa[i][$] = $;", n(m, t, mm), n(m, t, k), rd));
            }
            k = k + 1;
        }
        if unroll {
            var rr = 0;
            while rr < r {
                gen.line_lit(m, t, g, "{");
                gen.depth_in(m, g);
                gen.line(m, t, g, text.f3(m, t, "const uint j = min(sgid * $u + $u, $u);", n(m, t, r), n(m, t, rr), n(m, t, nn - 1)));
                let lit_rr = text.f1(m, t, "[$]", n(m, t, rr));
                let decl_lit = mem.list(m);
                k = 0;
                while k < mem.size(m, decl) {
                    mem.push(m, decl_lit, text.replace_lit(m, t, mem.get(m, decl, k), "[rr]", lit_rr));
                    k = k + 1;
                }
                let pre_lit = text.replace_lit(m, t, pre, "[rr]", lit_rr);
                gen.line(m, t, g, text.cat(m, t, pre_lit, text.joined(m, t, decl_lit, " ")));
                var accn = text.lit(m, t, "s");
                if deferred {
                    accn = text.lit(m, t, "run");
                }
                var i = 0;
                while i < mm {
                    gen.line(m, t, g, text.f4(m, t, "$[$][$] += $;", accn, n(m, t, rr), n(m, t, i), terms(m, t, mem.size(m, ws), n(m, t, i))));
                    i = i + 1;
                }
                gen.depth_out(m, g);
                gen.line_lit(m, t, g, "}");
                rr = rr + 1;
            }
        } else {
            gen.line(m, t, g, text.f1(m, t, "for (uint rr = 0; rr < $u; ++rr) {", n(m, t, r)));
            gen.depth_in(m, g);
            gen.line(m, t, g, text.f2(m, t, "const uint j = min(sgid * $u + rr, $u);", n(m, t, r), n(m, t, nn - 1)));
            gen.line(m, t, g, text.cat(m, t, pre, text.joined(m, t, decl, " ")));
            gen.line(m, t, g, text.f2(m, t, "for (uint i = 0; i < $u; ++i) s[rr][i] += $;", n(m, t, mm), terms(m, t, mem.size(m, ws), text.lit(m, t, "i"))));
            gen.depth_out(m, g);
            gen.line_lit(m, t, g, "}");
        }
        gen.depth_out(m, g);
        gen.line_lit(m, t, g, "}");
        if deferred {
            var rr = 0;
            while rr < r {
                let parts = mem.list(m);
                var i = 0;
                while i < mm {
                    let l = mem.of5(m, n(m, t, rr), n(m, t, i), n(m, t, rr), n(m, t, i), n(m, t, rr));
                    mem.push(m, parts, text.fmt(m, t, "s[$][$] += run[$][$] * sgr[$];", l));
                    i = i + 1;
                }
                gen.line(m, t, g, text.joined(m, t, parts, " "));
                rr = rr + 1;
            }
        }
    }
    gen.depth_out(m, g);
    gen.line_lit(m, t, g, "}");
    if unroll {
        var rr = 0;
        while rr < r {
            var i = 0;
            while i < mm {
                let sv = text.f2(m, t, "s[$][$]", n(m, t, rr), n(m, t, i));
                let dn = dialect.shuffle_down(m, t, gen.dia(m, g), sv, text.lit(m, t, "d"));
                gen.line(m, t, g, text.f4(m, t, "for (uint d = $u; d > 0; d /= 2) s[$][$] += $;", n(m, t, lanes / 2), n(m, t, rr), n(m, t, i), dn));
                i = i + 1;
            }
            rr = rr + 1;
        }
        gen.line_lit(m, t, g, "if (lane == 0) {");
        gen.depth_in(m, g);
        rr = 0;
        while rr < r {
            let parts = mem.list(m);
            var i = 0;
            while i < mm {
                mem.push(m, parts, text.f4(m, t, "scratch[$ + $u + j] = s[$][$];", n(m, t, resv), n(m, t, i * nn), n(m, t, rr), n(m, t, i)));
                i = i + 1;
            }
            gen.line(m, t, g, text.f4(m, t, "{ const uint j = sgid * $u + $u; if (j < $u) { $ } }", n(m, t, r), n(m, t, rr), n(m, t, nn), text.joined(m, t, parts, " ")));
            rr = rr + 1;
        }
        gen.depth_out(m, g);
        gen.line_lit(m, t, g, "}");
    } else {
        let dn = dialect.shuffle_down(m, t, gen.dia(m, g), text.lit(m, t, "s[rr][i]"), text.lit(m, t, "d"));
        gen.line(m, t, g, text.f4(m, t, "for (uint rr = 0; rr < $u; ++rr) for (uint i = 0; i < $u; ++i) for (uint d = $u; d > 0; d /= 2) s[rr][i] += $;", n(m, t, r), n(m, t, mm), n(m, t, lanes / 2), dn));
        let l = mem.of6(m, n(m, t, r), n(m, t, r), n(m, t, nn), n(m, t, mm), n(m, t, resv), n(m, t, nn));
        gen.line(m, t, g, text.fmt(m, t, "if (lane == 0) for (uint rr = 0; rr < $u; ++rr) { const uint j = sgid * $u + rr; if (j < $u) for (uint i = 0; i < $u; ++i) scratch[$ + i * $u + j] = s[rr][i]; }", l));
    }
    gen.depth_out(m, g);
    gen.line_lit(m, t, g, "}");
    gen.barrier(m, t, g);
    let name = gen.declare_reg(m, t, g, x, ir.tile(m, acc, mem.of2(m, mm, nn), kinds.reg()));
    let cv = dialect.convert(m, t, gen.dia(m, g), acc, text.f1(m, t, "scratch[$ + e]", n(m, t, resv)));
    return gen.owned1(m, t, g, mm * nn, text.f2(m, t, "$[k] = $;", name, cv));
}

// `xa[i][0] * w0 + xa[i][1] * w1 + ...`
fn terms[&m, &t](m: &!m [int], t: &!t [byte], count: int, i: int) -> [] int {
    let parts = mem.list(m);
    var k = 0;
    while k < count {
        mem.push(m, parts, text.f3(m, t, "xa[$][$] * w$", i, n(m, t, k), n(m, t, k)));
        k = k + 1;
    }
    return text.joined(m, t, parts, " + ");
}
