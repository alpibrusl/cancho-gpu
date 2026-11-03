#!/usr/bin/env python3
"""Writes the kernels of the ResNet-18-class backbone (docs/resident.md section 26) that the M4 CNN did not have: the convolution with any window and any shift (`convk`, with its two
gradients), the weight gradient with a mask, space-to-depth and its gradient, 3x3 stride-2 max-pooling and its gradient, the global average and the packing of an image for a non-square
frame, and the linear (no ReLU) variants of BatchNorm's kernels, which the shortcut convolution's BatchNorm needs. The text it writes is committed beside it; run it after changing it.

The shape of the idea (resident.md section 26.1): a stride-2 convolution is a stride-1 convolution of the *space-to-depth* image (each 2x2 block of pixels becomes one pixel of 4 c
channels), which is a permutation; a window with a stride is not in the language (`tests/probes.py`), a permutation by a product with a constant 0/1 matrix is."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def write(name, text):
    with open(os.path.join(HERE, name + ".chx"), "w") as f:
        f.write(text)


def sched(n, threads=128):
    return f"\nschedule {n} for apple-m-series {{ threads {threads} }}\nschedule {n} for nvidia-ada     {{ threads {threads} }}\n"


MMA = "schedule {n} for apple-m-series {{ threads 128 warps 2 2 bm 64 bn 32 bk 32 }}\nschedule {n} for nvidia-ada     {{ threads 128 warps 2 2 bm 64 bn 32 bk 32 }}\n"


def shift_expr(sign):
    """The text of (dy - oy) wp + (dx - ox), or its negative, with binary operators only (the grammar has no unary minus)."""
    return "dy * wp + dx - oy * wp - ox" if sign > 0 else "oy * wp + ox - dy * wp - dx"


def convk():
    write("convk", f"""// A kh x kw convolution, stride 1, on the padded image matrix (docs/resident.md section 26), the M4 CNN's `conv3` with the window and the shift as constants. The input x is [g + rp + g, cx]
// (rp = n (h+2) (w+2) rows, g zero rows on each side, g >= max shift); the output at row p reads the rows p + (dy - oy) wp + (dx - ox) for the taps (dy, dx), each multiplied by that tap's
// [cout, cin] slice of w [cout, kh kw cin]. Only the first cin of the cx columns of x are read, which is how a 1x1 stride-2 convolution reads the (0,0) quarter of a space-to-depth image.
// A tap may reach two pixels past an edge: the border of one pixel plus the neighbouring row's border absorb it (resident.md section 26.1). Rows of y that are border are junk; the epilogue masks.
algo convk(rp, cx, cin, cout, wp, g, kh, kw, oy, ox)
  in  x: f16[rp + 2 * g, cx]
  in  w: f16[cout, kh * kw * cin]
  out y: f32[rp, cout]
  tile bm, bn, bk
{{
  grid i over rp / bm, j over cout / bn
  let c = zeros f32[bm, bn] @frag
  let c = for dy in 0 .. kh with c {{
    let c = for dx in 0 .. kw with c {{
      let c = for p in 0 .. cin / bk with c {{
        let at = load x[g + i * bm + {shift_expr(1)}, p * bk ; bm, bk] @shared
        let wt = load w[j * bn, kw * dy * cin + dx * cin + p * bk ; bn, bk] @shared
        yield mma c at wt
      }}
      yield c
    }}
    yield c
  }}
  store c -> y[i * bm, j * bn]
}}

{MMA.format(n="convk")}""")
    write("convk_dx", f"""// The gradient of convk with respect to its input: gx[p] = sum over taps of dz[p - shift(tap)] W_tap (`mma_nn`: the right operand is read as stored). It is [rp, cx] wide and only the first cin
// columns are computed (the others stay zero in a session's zeroed buffer): the 1x1 stride-2 convolution's gradient lands in the (0,0) quarter of the space-to-depth image's gradient.
algo convk_dx(rp, cx, cin, cout, wp, g, kh, kw, oy, ox)
  in  dz: f16[rp + 2 * g, cout]
  in  w:  f16[cout, kh * kw * cin]
  out gx: f32[rp, cx]
  tile bm, bn, bk
{{
  grid i over rp / bm, j over cin / bn
  let c = zeros f32[bm, bn] @frag
  let c = for dy in 0 .. kh with c {{
    let c = for dx in 0 .. kw with c {{
      let c = for p in 0 .. cout / bk with c {{
        let at = load dz[g + i * bm + {shift_expr(-1)}, p * bk ; bm, bk] @shared
        let wt = load w[p * bk, kw * dy * cin + dx * cin + j * bn ; bk, bn] @shared
        yield mma_nn c at wt
      }}
      yield c
    }}
    yield c
  }}
  store c -> gx[i * bm, j * bn]
}}

{MMA.format(n="convk_dx")}""")
    write("convk_dw", """// The gradient of convk with respect to w, in pieces (conv3_dw's, with the window as constants): for the tap (dy, dx) and the block r of `rs` rows, part = dz^T x_shifted (`mma_tn`), tile by
// tile of the [cout, cin] result of that tap; `wsum` or `wsum_mask` adds the pieces. A loop must carry a tile, so it carries `u`, which is written to the unused output `junk`.
algo convk_dw(rp, cx, cin, cout, wp, g, kh, kw, oy, ox, rs)
  in  dz: f16[rp + 2 * g, cout]
  in  x:  f16[rp + 2 * g, cx]
  out part: f32[rp / rs * cout, kh * kw * cin]
  out junk: f32[1, 8]
  tile bm, bn, bk
{
  grid dy over kh, dx over kw, r over rp / rs
  let u = zeros f32[1, 8]
  let u = for jc in 0 .. cout / bm with u {
    let u = for jn in 0 .. cin / bn with u {
      let c = zeros f32[bm, bn] @frag
      let c = for p in 0 .. rs / bk with c {
        let a = load dz[g + r * rs + p * bk, jc * bm ; bk, bm] @shared
        let b = load x[g + r * rs + p * bk + dy * wp + dx - oy * wp - ox, jn * bn ; bk, bn] @shared
        yield mma_tn c a b
      }
      store c -> part[r * cout + jc * bm, dy * kw * cin + dx * cin + jn * bn]
      yield u
    }
    yield u
  }
  store u -> junk[0, 0]
}

schedule convk_dw for apple-m-series { threads 128 warps 2 2 bm 64 bn 64 bk 32 }
schedule convk_dw for nvidia-ada     { threads 128 warps 2 2 bm 64 bn 64 bk 32 }
""")
    write("wsum_mask", """// dw = (the sum of the `ns` pieces convk_dw wrote) * mk: part is [ns * r, k], mk is [1, k], out is [r, k]. The mask is 1 where the weight is a weight of the network and 0 where the
// space-to-depth layout holds a zero that must stay zero (a 7x7 kernel in a 4x4 window of 2x2 blocks has 192 - 147 of them per channel pair): Adam moves a weight by a gradient only.
algo wsum_mask(ns, r, k, bm, bn)
  in  part: f32[ns * r, k]
  in  mk:   f32[1, k]
  out dw:   f32[r, k]
{
  grid i over r / bm, j over k / bn
  let acc = zeros f32[bm, bn]
  let acc = for q in 0 .. ns with acc {
    let t = load part[q * r + i * bm, j * bn ; bm, bn]
    yield acc + t
  }
  let m = load mk[0, j * bn ; 1, bn]
  store acc * m -> dw[i * bm, j * bn]
}
""" + sched("wsum_mask"))


def s2d():
    write("s2d_fwd", """// Space to depth: each 2x2 block of pixels of the padded image X [g + n (2h+2) (2w+2) + g, c] becomes one pixel of the padded image S [g + n (h+2) (w+2) + g, 4 c], the channel group
// (2 py + px) holding the pixel (2Y + py, 2X + px). A permutation, written as the product with a 0/1 selection matrix (sel0, sel1: [wb, 2 wb], row j has its 1 at column 2j + px): the language has
// no strided window. One threadgroup takes a block of wb output pixels of one row Y of one image: the 2 wb input pixels of each of the two input rows. The output's border is never written.
algo s2d_fwd(n, h, w, c, g, wb)
  in  x:    f16[n * (2 * h + 2) * (2 * w + 2) + 2 * g, c]
  in  sel0: f32[wb, 2 * wb]
  in  sel1: f32[wb, 2 * wb]
  out s:    f16[n * (h + 2) * (w + 2) + 2 * g, 4 * c]
{
  grid b over n, y over h, xb over w / wb
  let r0 = load x[g + b * (2 * h + 2) * (2 * w + 2) + (2 * y + 1) * (2 * w + 2) + 2 * xb * wb + 1, 0 ; 2 * wb, c]
  let r1 = load x[g + b * (2 * h + 2) * (2 * w + 2) + (2 * y + 2) * (2 * w + 2) + 2 * xb * wb + 1, 0 ; 2 * wb, c]
  let f0 = stage f32 r0
  let f1 = stage f32 r1
  let a0 = load sel0
  let a1 = load sel1
  let t00 = matmul &a0 &f0
  let t01 = matmul &a1 f0
  let t10 = matmul a0 &f1
  let t11 = matmul a1 f1
  let h00 = stage f16 t00
  let h01 = stage f16 t01
  let h10 = stage f16 t10
  let h11 = stage f16 t11
  store h00 -> s[g + b * (h + 2) * (w + 2) + (y + 1) * (w + 2) + xb * wb + 1, 0]
  store h01 -> s[g + b * (h + 2) * (w + 2) + (y + 1) * (w + 2) + xb * wb + 1, c]
  store h10 -> s[g + b * (h + 2) * (w + 2) + (y + 1) * (w + 2) + xb * wb + 1, 2 * c]
  store h11 -> s[g + b * (h + 2) * (w + 2) + (y + 1) * (w + 2) + xb * wb + 1, 3 * c]
}
""" + sched("s2d_fwd"))
    write("s2d_bwd", """// The gradient of s2d_fwd (its transpose, the inverse permutation): a block of wb pixels of the row Y of the gradient of S [n (h+2) (w+2), 4 c] (f32, no guard) times the transposed selection
// matrices (selt0, selt1: [2 wb, wb]) is the gradient of 2 wb pixels of the two input rows 2Y+1 and 2Y+2. The border of the result is never written (zero).
algo s2d_bwd(n, h, w, c, wb)
  in  gs:   f32[n * (h + 2) * (w + 2), 4 * c]
  in  selt0: f32[2 * wb, wb]
  in  selt1: f32[2 * wb, wb]
  out gi:   f32[n * (2 * h + 2) * (2 * w + 2), c]
{
  grid b over n, y over h, xb over w / wb
  let a00 = load gs[b * (h + 2) * (w + 2) + (y + 1) * (w + 2) + xb * wb + 1, 0 ; wb, c]
  let a01 = load gs[b * (h + 2) * (w + 2) + (y + 1) * (w + 2) + xb * wb + 1, c ; wb, c]
  let a10 = load gs[b * (h + 2) * (w + 2) + (y + 1) * (w + 2) + xb * wb + 1, 2 * c ; wb, c]
  let a11 = load gs[b * (h + 2) * (w + 2) + (y + 1) * (w + 2) + xb * wb + 1, 3 * c ; wb, c]
  let p0 = load selt0
  let p1 = load selt1
  let u00 = matmul &p0 a00
  let u01 = matmul &p1 a01
  let u10 = matmul p0 a10
  let u11 = matmul p1 a11
  let r0 = u00 + u01
  let r1 = u10 + u11
  store r0 -> gi[b * (2 * h + 2) * (2 * w + 2) + (2 * y + 1) * (2 * w + 2) + 2 * xb * wb + 1, 0]
  store r1 -> gi[b * (2 * h + 2) * (2 * w + 2) + (2 * y + 2) * (2 * w + 2) + 2 * xb * wb + 1, 0]
}
""" + sched("s2d_bwd"))


# the 9 candidates of a 3x3 stride-2 window with padding 1, row-major, in space-to-depth coordinates (dY, py) for the row and (dX, px) for the column: rows 2y-1, 2y, 2y+1 of the image
AX = [(-1, 1), (0, 0), (0, 1)]


def cand(k):
    (dy, py), (dx, px) = AX[k // 3], AX[k % 3]
    return dy, dx, 2 * py + px


def rowexpr(dy, dx, sign):
    """g + i * bm + dy wp + dx (sign +1) or - dy wp - dx (sign -1) without unary minus."""
    a, b = dy * sign, dx * sign
    return "g + i * bm" + ("" if a == 0 else " + wp" if a > 0 else " - wp") + ("" if b == 0 else " + 1" if b > 0 else " - 1")


def maxpool():
    t = """// 3x3 stride-2 max-pooling (padding 1) of the space-to-depth image S [g + rp + g, 4 c] (resident.md section 26): the window of the pooled pixel (y, x) is the nine pixels
// 2y-1 .. 2y+1 by 2x-1 .. 2x+1 of the image, which in S are nine (shift, channel group) pairs of the padded matrix, read as shifted windows exactly as a convolution's taps. There is no
// `max a b`: the maximum of the running m and the next candidate v is s v - (s - 1) m with s = step(v - m), which is exactly v or m (one product is exactly zero), and the same s
// updates the index k of the winner. Strict `>` keeps the first maximum in row-major order, which is torch's rule. Padding is zero, which equals -inf here because the input is a ReLU's
// output (>= 0), and a window that ties the zero padding with real zeros sends its gradient to a border cell, which is discarded: the value there was 0, so its ReLU's gradient was 0.
// a is the pooled map (f16, guarded, border rows zero); idx the winner's index 0..8 (f32, guarded), -1 on the border rows so the backward never matches them.
algo maxpool_fwd(rp, c, wp, g, bm)
  in  s:  f16[rp + 2 * g, 4 * c]
  in  mk: f32[rp, 1]
  out a:  f16[rp + 2 * g, c]
  out idx: f32[rp + 2 * g, c]
{
  grid i over rp / bm
"""
    for k in range(9):
        dy, dx, grp = cand(k)
        t += f"  let v{k} = load s[{rowexpr(dy, dx, 1)}, {grp} * c ; bm, c]\n  let f{k} = stage f32 v{k}\n"
        if k == 0:
            t += "  let m0 = f0 * 1.0\n  let ix0 = zeros f32[bm, c]\n"
            continue
        t += f"""  let d{k} = &f{k} - &m{k - 1}
  let s{k} = step d{k}
  let u{k} = &s{k} - 1
  let p{k} = &s{k} * f{k}
  let q{k} = &u{k} * m{k - 1}
  let m{k} = p{k} - q{k}
  let w{k} = u{k} * ix{k - 1}
  let e{k} = s{k} * {k}
  let ix{k} = e{k} - w{k}
"""
    t += """  let mm = load mk[i * bm, 0 ; bm, 1]
  let o  = m8 * &mm
  let h  = stage f16 o
  store h -> a[g + i * bm, 0]
  let xm = ix8 * &mm
  let xn = xm + mm
  let xo = xn - 1
  store xo -> idx[g + i * bm, 0]
}
""" + sched("maxpool_fwd")
    write("maxpool_fwd", t)
    t = """// The gradient of maxpool_fwd, as a gather: the row q of the gradient of S, in the channel group of the candidate k, is the pooled gradient of the output p = q - shift(k) where that output's
// winner was k. [winner == k] is the triangle relu(d+1) - 2 relu(d) + relu(d-1), d = idx - k, exactly 1 at 0 and 0 at every other integer (and at -1, the border's mark). gr and idx are
// guarded like the forward's output (the shifted windows start before row 0); gs is [rp, 4 c], f32, no guard, and its border rows hold sums over valid outputs that the consumer (s2d_bwd) never reads.
algo maxpool_bwd(rp, c, wp, g, bm)
  in  gr:  f32[rp + 2 * g, c]
  in  idx: f32[rp + 2 * g, c]
  out gs:  f32[rp, 4 * c]
{
  grid i over rp / bm
"""
    groups = {}
    for k in range(9):
        groups.setdefault(cand(k)[2], []).append(k)
    for grp in range(4):
        acc = None
        for k in groups[grp]:
            dy, dx, _ = cand(k)
            r = rowexpr(dy, dx, -1)
            t += f"""  let ix{k} = load idx[{r}, 0 ; bm, c]
  let dd{k} = ix{k} - {k}
  let pa{k} = &dd{k} + 1
  let ra{k} = relu pa{k}
  let rb{k} = relu &dd{k}
  let pc{k} = dd{k} - 1
  let rc{k} = relu pc{k}
  let tw{k} = rb{k} * 2
  let tr{k} = ra{k} - tw{k}
  let eq{k} = tr{k} + rc{k}
  let gg{k} = load gr[{r}, 0 ; bm, c]
  let c{k} = eq{k} * gg{k}
"""
            if acc is None:
                acc = f"c{k}"
            else:
                t += f"  let sm{k} = {acc} + c{k}\n"
                acc = f"sm{k}"
        t += f"  store {acc} -> gs[i * bm, {grp} * c]\n"
    t += "}\n" + sched("maxpool_bwd")
    write("maxpool_bwd", t)


def other():
    write("gap2_fwd", """// Global average pooling of n padded images h x w: the mean over the valid pixels of each, f16 [n, c], summed rc rows at a time. vm [1, (h+2)(w+2)] is 1 / (h w) on a valid row, 0 on a border row.
algo gap2_fwd(n, h, w, c, g, rc)
  in  a:  f16[n * (h + 2) * (w + 2) + 2 * g, c]
  in  vm: f32[1, (h + 2) * (w + 2)]
  out f:  f16[n, c]
{
  grid b over n
  let z = zeros f32[1, c]
  let s = for q in 0 .. (h + 2) * (w + 2) / rc with z {
    let t  = load a[g + b * (h + 2) * (w + 2) + q * rc, 0 ; rc, c]
    let tf = stage f32 t
    let v  = load vm[0, q * rc ; 1, rc]
    let r  = matmul v tf
    yield z + r
  }
  let hh = stage f16 s
  store hh -> f[b, 0]
}
""" + sched("gap2_fwd"))
    write("gap2_bwd", """// The gradient of gap2_fwd: each image's row of gf [n, c] spread over its valid pixels, times 1 / (h w) (mc [(h+2)(w+2), 1], zero on the border), rc rows at a time.
algo gap2_bwd(n, h, w, c, rc)
  in  gf: f32[n, c]
  in  mc: f32[(h + 2) * (w + 2), 1]
  out gi: f32[n * (h + 2) * (w + 2), c]
{
  grid b over n, q over (h + 2) * (w + 2) / rc
  let r = load gf[b, 0 ; 1, c]
  let m = load mc[q * rc, 0 ; rc, 1]
  let t = matmul m r
  store t -> gi[b * (h + 2) * (w + 2) + q * rc, 0]
}
""" + sched("gap2_bwd"))
    write("pack_hw", """// The input images into the padded, guarded layout conv reads, for a frame h x w (pack.chx's, not square, and without its shifting matrix): two tables of n2 images each (the two halves of a batch,
// [n2 h w, cin] f16), one row y of one image a threadgroup. A row of w pixels times cmat [cin, cp] pads its channels to cp, and is stored one column in (an offset of one row of the matrix),
// between the zero border columns; the border rows and the guards are never written.
algo pack_hw(n2, h, w, cin, cp, g)
  in  xa:   f16[n2 * h * w, cin]
  in  xb:   f16[n2 * h * w, cin]
  in  cmat: f32[cin, cp]
  out o:    f16[2 * n2 * (h + 2) * (w + 2) + 2 * g, cp]
{
  grid b over n2, y over h
  let la = load xa[b * h * w + y * w, 0 ; w, cin]
  let fa = stage f32 la
  let c1 = load cmat
  let ra = matmul fa &c1
  let ha = stage f16 ra
  store ha -> o[g + b * (h + 2) * (w + 2) + (y + 1) * (w + 2) + 1, 0]
  let lb = load xb[b * h * w + y * w, 0 ; w, cin]
  let fb = stage f32 lb
  let rb = matmul fb c1
  let hb = stage f16 rb
  store hb -> o[g + (n2 + b) * (h + 2) * (w + 2) + (y + 1) * (w + 2) + 1, 0]
}
""" + sched("pack_hw"))
    write("add2g", """// y = a + b (f32), written into a guarded matrix (g zero rows on each side): the gradient that goes around a block meets the one through it, for a consumer (maxpool_bwd) whose windows start before row 0.
algo add2g(r, c, bm, g)
  in  a: f32[r, c]
  in  b: f32[r, c]
  out y: f32[r + 2 * g, c]
{
  grid i over r / bm
  let x = load a[i * bm, 0 ; bm, c]
  let z = load b[i * bm, 0 ; bm, c]
  store x + z -> y[g + i * bm, 0]
}
""" + sched("add2g"))
    # the linear BatchNorm (no ReLU), from the committed ReLU kernels: the shortcut convolution's BatchNorm. Without the ReLU the pre-activation is not needed (d = gr), so sc and sh go.
    drop = "  let s  = load sc\n  let b  = load sh\n  let y0 = &zz * s + b\n  let y  = y0 * 1.0\n  let st = step y\n  let d  = gg * st\n"
    for src, dst, edits in (
        ("bn_apply", "bn_apply_lin", [("  let r  = relu y\n", "  let r  = y * 1.0\n"), ("a = f16(relu(z sc + sh) * mk)", "a = f16((z sc + sh) * mk), no ReLU")]),
        ("bn_bwd_part", "bn_bwd_part_lin", [(drop, "  let d  = gg * 1.0\n"), ("  in  sc:   f32[1, c]\n  in  sh:   f32[1, c]\n", ""), ("d = gr * step(z sc + sh)", "d = gr (no ReLU)")]),
        ("bn_bwd_apply", "bn_bwd_apply_lin", [(drop, "  let d  = gg * 1.0\n"), ("  in  sc:   f32[1, c]\n  in  sh:   f32[1, c]\n", ""), ("d = gr * step(z sc + sh)", "d = gr (no ReLU)")]),
    ):
        text = open(os.path.join(HERE, src + ".chx")).read()
        for x, y in edits:
            assert x in text or x.startswith("d = gr"), (src, x)
            text = text.replace(x, y)
        text = text.replace(src, dst)
        write(dst, "// (generated by gen_rn_kernels.py from " + src + ".chx: the same kernel without the ReLU)\n" + text)


if __name__ == "__main__":
    convk()
    s2d()
    maxpool()
    other()
