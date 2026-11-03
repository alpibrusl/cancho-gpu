#!/usr/bin/env python3
"""Writes the convolution kernels of the CNN (docs/resident.md section 18). Taps are unrolled here because the offsets of a window must be affine in the grid coordinates and
loop indices, and a tap's shift (dy - 1) wp + (dx - 1) is not affine in a flattened tap number. The text it writes is committed beside it; run it after changing it."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TAPS = [(dy, dx) for dy in range(3) for dx in range(3)]


def shift(dy, dx, sign):
    """The text of (dy - 1) wp + (dx - 1) (or its negative) as the sum of terms the constant-expression grammar has (no unary minus)."""
    a, b = (dy - 1) * sign, (dx - 1) * sign
    return ("" if a == 0 else " + wp" if a > 0 else " - wp") + ("" if b == 0 else " + 1" if b > 0 else " - 1")


SCHED = "schedule {n} for apple-m-series {{ threads 128 warps 2 2 bm 64 bn 32 bk 32 }}\nschedule {n} for nvidia-ada     {{ threads 128 warps 2 2 bm 64 bn 32 bk 32 }}\n"


def conv3():
    return f"""// 3x3 convolution, stride 1, zero padding, NHWC, on the matrix unit (docs/resident.md section 18). The input x is a padded image matrix [g + rp + g, cin]: rp = n (h+2) (w+2)
// rows (the images with a border of zeros, flattened), and g zero rows on each side (g >= wp + 1, wp = w + 2). The output at row p reads the input rows p + (dy-1) wp + (dx-1):
// nine shifted windows of one matrix, each multiplied by that tap's [cout, cin] slice of w [cout, 9 cin] (tap t = 3 dy + dx). The rows of y that are border are junk: the epilogue
// masks them. The taps are loops (dy, dx, then the channels): unrolled they would each keep their own tiles in threadgroup memory and the budget would allow only small ones.
algo conv3(rp, cin, cout, wp, g)
  in  x: f16[rp + 2 * g, cin]
  in  w: f16[cout, 9 * cin]
  out y: f32[rp, cout]
  tile bm, bn, bk
{{
  grid i over rp / bm, j over cout / bn
  let c = zeros f32[bm, bn] @frag
  let c = for dy in 0 .. 3 with c {{
    let c = for dx in 0 .. 3 with c {{
      let c = for p in 0 .. cin / bk with c {{
        let at = load x[g + i * bm + dy * wp + dx - wp - 1, p * bk ; bm, bk] @shared
        let wt = load w[j * bn, 3 * dy * cin + dx * cin + p * bk ; bn, bk] @shared
        yield mma c at wt
      }}
      yield c
    }}
    yield c
  }}
  store c -> y[i * bm, j * bn]
}}

{SCHED.format(n="conv3")}"""


def conv3_dx():
    return f"""// The gradient of conv3 with respect to its input: dx[p] = sum over taps of dz[p - shift(t)] W_t, with W_t the [cout, cin] slice of w that conv3 used for tap t (`mma_nn`: the right
// operand is read in its stored layout). dz is a guarded padded matrix like conv3's x, zero on the border rows (so a border pixel contributes nothing); the rows of dx that are
// border are junk and masked by the consumer.
algo conv3_dx(rp, cin, cout, wp, g)
  in  dz: f16[rp + 2 * g, cout]
  in  w:  f16[cout, 9 * cin]
  out gx: f32[rp, cin]
  tile bm, bn, bk
{{
  grid i over rp / bm, j over cin / bn
  let c = zeros f32[bm, bn] @frag
  let c = for dy in 0 .. 3 with c {{
    let c = for dx in 0 .. 3 with c {{
      let c = for p in 0 .. cout / bk with c {{
        let at = load dz[g + i * bm + wp + 1 - dy * wp - dx, p * bk ; bm, bk] @shared
        let wt = load w[p * bk, 3 * dy * cin + dx * cin + j * bn ; bk, bn] @shared
        yield mma_nn c at wt
      }}
      yield c
    }}
    yield c
  }}
  store c -> gx[i * bm, j * bn]
}}

{SCHED.format(n="conv3_dx")}"""


def conv3_dw():
    return """// The gradient of conv3 with respect to w, in pieces: for the tap (dy, dx) and the block r of `rs` rows, part = dz^T x_shifted summed over those rows (`mma_tn`), tile by tile of the
// [cout, cin] result of that tap (loops over the tiles: a 128 x 128 accumulator does not fit a threadgroup's registers and made this kernel 30 times slower); `wsum` adds the pieces.
// Three grid axes: dy, dx and r (the rows are split so that there are enough threadgroups). A loop must carry a tile, so it carries `u`, which is written to the unused output `junk`.
algo conv3_dw(rp, cin, cout, wp, g, rs)
  in  dz: f16[rp + 2 * g, cout]
  in  x:  f16[rp + 2 * g, cin]
  out part: f32[rp / rs * cout, 9 * cin]
  out junk: f32[1, 8]
  tile bm, bn, bk
{
  grid dy over 3, dx over 3, r over rp / rs
  let u = zeros f32[1, 8]
  let u = for jc in 0 .. cout / bm with u {
    let u = for jn in 0 .. cin / bn with u {
      let c = zeros f32[bm, bn] @frag
      let c = for p in 0 .. rs / bk with c {
        let a = load dz[g + r * rs + p * bk, jc * bm ; bk, bm] @shared
        let b = load x[g + r * rs + p * bk + dy * wp + dx - wp - 1, jn * bn ; bk, bn] @shared
        yield mma_tn c a b
      }
      store c -> part[r * cout + jc * bm, dy * 3 * cin + dx * cin + jn * bn]
      yield u
    }
    yield u
  }
  store u -> junk[0, 0]
}

schedule conv3_dw for apple-m-series { threads 128 warps 2 2 bm 64 bn 64 bk 32 }
schedule conv3_dw for nvidia-ada     { threads 128 warps 2 2 bm 64 bn 64 bk 32 }
"""


def write(name, text):
    with open(os.path.join(HERE, name + ".chx"), "w") as f:
        f.write(text)


def sched(n, threads=128):
    return f"\nschedule {n} for apple-m-series {{ threads {threads} }}\nschedule {n} for nvidia-ada     {{ threads {threads} }}\n"


def pre_activation(res, rows, tag):
    """The text that computes y = z sc + sh (+ res) for the rows of the instance, with zz kept for later use."""
    t = f"""  let gg = load gr[i * {rows}, 0 ; {rows}, c]
  let zz = load z[i * {rows}, 0 ; {rows}, c]
  let s  = load sc
  let b  = load sh
  let y0 = &zz * s + b
"""
    if res:
        t += f"""  let rr = load res[g + i * {rows}, 0 ; {rows}, c]
  let rf = stage f32 rr
  let y  = y0 + rf
"""
    else:
        t += "  let y  = y0 * 1.0\n"
    return t


def apply(res):
    n = "bn_apply_res" if res else "bn_apply"
    t = f"""// a = f16(relu(z sc + sh{' + res' if res else ''}) * mk), written into the guarded padded matrix the next convolution reads (rows g .. g + rp): mk is 1 on a valid row and 0 on a border row, so
// the border stays zero.{' res is the block input (the skip connection), f16 in the same guarded layout.' if res else ''}
algo {n}(rp, c, bm, g)
  in  z:  f32[rp, c]
  in  sc: f32[1, c]
  in  sh: f32[1, c]
  in  mk: f32[rp, 1]
{'  in  res: f16[rp + 2 * g, c]' + chr(10) if res else ''}  out a:  f16[rp + 2 * g, c]
{{
  grid i over rp / bm
  let zz = load z[i * bm, 0 ; bm, c]
  let s  = load sc
  let b  = load sh
  let y0 = zz * s + b
"""
    if res:
        t += """  let rr = load res[g + i * bm, 0 ; bm, c]
  let rf = stage f32 rr
  let y  = y0 + rf
"""
    else:
        t += "  let y  = y0 * 1.0\n"
    t += """  let r  = relu y
  let m  = load mk[i * bm, 0 ; bm, 1]
  let o  = r * m
  let h  = stage f16 o
  store h -> a[g + i * bm, 0]
}
""" + sched(n)
    write(n, t)


def bwd_part(res):
    n = "bn_bwd_part_res" if res else "bn_bwd_part"
    t = f"""// BatchNorm backward, the first level. d = gr * step(z sc + sh{' + res' if res else ''}) is the gradient of the BatchNorm output (gr is the gradient of the ReLU's output, step its derivative); for each
// block of rb rows, the sums over the valid rows of d (the shift's gradient) and of d * xhat (the scale's), xhat = (z - mean) istd. bn_bwd_fin adds the blocks.
algo {n}(rp, c, rb{', g' if res else ''})
  in  gr:   f32[rp, c]
  in  z:    f32[rp, c]
  in  sc:   f32[1, c]
  in  sh:   f32[1, c]
  in  mean: f32[1, c]
  in  istd: f32[1, c]
  in  v:    f32[1, rp]
{'  in  res:  f16[rp + 2 * g, c]' + chr(10) if res else ''}  out p1:   f32[rp / rb, c]
  out p2:   f32[rp / rb, c]
{{
  grid i over rp / rb
""" + pre_activation(res, "rb", "") + """  let st = step y
  let d  = gg * st
  let mu = load mean
  let is = load istd
  let xc = zz - mu
  let xh = xc * is
  let dx = &d * xh
  let vv = load v[0, i * rb ; 1, rb]
  let a1 = matmul &vv d
  store a1 -> p1[i, 0]
  let a2 = matmul vv dx
  store a2 -> p2[i, 0]
}
""" + sched(n)
    write(n, t)


def bwd_apply(res):
    n = "bn_bwd_apply_res" if res else "bn_bwd_apply"
    t = f"""// BatchNorm backward, the second level: dz = gamma istd (d - dbeta / m - xhat dgamma / m) on the valid rows, f16, into the guarded padded matrix the convolution's backward reads
// (the border and the guard stay zero){'; and d itself, the gradient that goes around the block on the skip connection, as f32' if res else ''}. d and xhat are as in bn_bwd_part.
algo {n}(rp, c, bm, g, m)
  in  gr:   f32[rp, c]
  in  z:    f32[rp, c]
  in  sc:   f32[1, c]
  in  sh:   f32[1, c]
  in  mean: f32[1, c]
  in  istd: f32[1, c]
  in  gamma: f32[1, c]
  in  dbeta: f32[1, c]
  in  dgamma: f32[1, c]
  in  mk:   f32[rp, 1]
{'  in  res:  f16[rp + 2 * g, c]' + chr(10) if res else ''}  out dz:   f16[rp + 2 * g, c]
{'  out skip: f32[rp, c]' + chr(10) if res else ''}{{
  grid i over rp / bm
""" + pre_activation(res, "bm", "") + f"""  let st = step y
  let d  = gg * st
  let mu = load mean
  let is = load istd
  let xc = zz - mu
  let xh = xc * &is
  let ga = load gamma
  let db = load dbeta
  let dg = load dgamma
  let k  = ga * is
  let t1 = {'&d' if res else 'd'} - db * (1 / m)
  let t2 = xh * dg * (1 / m)
  let t3 = t1 - t2
  let t4 = t3 * k
  let mm = load mk[i * bm, 0 ; bm, 1]
  let t5 = t4 * mm
  let h  = stage f16 t5
  store h -> dz[g + i * bm, 0]
""" + ("  store d -> skip[i * bm, 0]\n" if res else "") + "}\n" + sched(n)
    write(n, t)


for r in (False, True):
    apply(r)
    bwd_part(r)
    bwd_apply(r)
for name, text in (("conv3", conv3()), ("conv3_dx", conv3_dx()), ("conv3_dw", conv3_dw())):
    write(name, text)
