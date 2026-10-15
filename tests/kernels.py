#!/usr/bin/env python3
"""What the 9b kernels compute, against a reference written from the
mathematics (docs/kernels.md), not from the kernels.

    kernels.py <cancho-gpu>

Each kernel is run on the CPU with `--run` over the interpreter's made-up
inputs (an LCG seeded 12345, in parameter order: docs/design.md section 10),
which prints, for every output, its mean, its peak and its first four values.
The same inputs go through plain Python (binary64) and the summaries must
agree: the mean and peak to the four decimals they are printed to, the first
four values to 1e-5 (the kernel sums in f32, in order). The goldens already
hold the kernels' text and summaries byte-for-byte against lex-gpu; this is
what says the summaries are *right*.
"""
import math
import re
import subprocess
import sys

BIN = sys.argv[1]


def lcg(sizes):
    seed = 12345
    tensors = []
    for n in sizes:
        out = []
        for _ in range(n):
            seed = (seed * 1664525 + 1013904223) % 4294967296
            out.append((seed >> 9) / 8388608.0 - 0.5)
        tensors.append(out)
    return tensors


def run(kernel, consts):
    r = subprocess.run([BIN, "check", f"tests/chx/{kernel}.chx"] + consts + ["--run"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{kernel} {consts}: {r.stderr.strip()}")
    # {output: {target: (mean, peak, first four)}}: both targets are printed.
    out = {}
    for line in r.stdout.splitlines():
        m = re.match(r"(nvidia-ada|apple-m-series): (\w+) \[[^\]]*\] = mean (-?[\d.]+), peak (-?[\d.]+), first \[(.*)\]", line)
        if m:
            out.setdefault(m.group(2), {})[m.group(1)] = (float(m.group(3)), float(m.group(4)), [float(x) for x in m.group(5).split(", ")])
    return out


def summary(values):
    return sum(values) / len(values), max(abs(x) for x in values), values[:4]


def check(name, per_target, want):
    mean, peak, first = summary(want)
    ok = True
    for target, (gm, gp, gf) in sorted(per_target.items()):
        good = abs(gm - mean) <= 6e-5 and abs(gp - peak) <= 6e-5 and all(abs(a - b) <= 1e-5 for a, b in zip(gf, first))
        print(f"{'ok  ' if good else 'FAIL'} {name} [{target}]: kernel mean {gm} peak {gp}; reference mean {mean:.4f} peak {peak:.4f}")
        if not good:
            print(f"     first four: kernel {gf}, reference {first}")
        ok &= good
    return ok and len(per_target) == 2


def rope(r, h, p):
    x, c, s = lcg([r * (2 * h + p), h, h])
    y = []
    for row in range(r):
        b = row * (2 * h + p)
        y += [x[b + j] * c[j] - x[b + h + j] * s[j] for j in range(h)]
        y += [x[b + h + j] * c[j] + x[b + j] * s[j] for j in range(h)]
        y += x[b + 2 * h:b + 2 * h + p]
    return y


def attention(nkv, g, t, d, tmax, scale=0.0625):
    q, k, v, mask = lcg([nkv * g * d, nkv * tmax * d, nkv * tmax * d, t])
    out = []
    for i in range(nkv):
        for r in range(g):
            qr = q[(i * g + r) * d:(i * g + r + 1) * d]
            s = []
            for pos in range(t):
                kr = k[(i * tmax + pos) * d:(i * tmax + pos + 1) * d]
                s.append(sum(a * b for a, b in zip(qr, kr)) * scale + mask[pos])  # 1/sqrt(d), a literal in the kernel
            mx = max(s)
            e = [math.exp(x - mx) for x in s]
            z = sum(e)
            w = [x / z for x in e]
            for col in range(d):
                out.append(sum(w[pos] * v[(i * tmax + pos) * d + col] for pos in range(t)))
    return out


def kv_append(nkv, tmax, d, pos):
    kn, vn = lcg([nkv * d, nkv * d])
    kc = [0.0] * (nkv * tmax * d)
    vc = [0.0] * (nkv * tmax * d)
    for i in range(nkv):
        kc[(i * tmax + pos) * d:(i * tmax + pos + 1) * d] = kn[i * d:(i + 1) * d]
        vc[(i * tmax + pos) * d:(i * tmax + pos + 1) * d] = vn[i * d:(i + 1) * d]
    return kc, vc


def attn_gate(r, n):
    o, gate = lcg([r * n, r * n])
    return [a / (1 + math.exp(-b)) for a, b in zip(o, gate)]


def matvec(n, k, extra=None):
    x, w = lcg([k, n * k])[:2]
    third = lcg([k, n * k] + ([n] if extra else []))[2] if extra else None
    y = [sum(x[j] * w[i * k + j] for j in range(k)) for i in range(n)]
    if extra == "bias" or extra == "res":
        y = [a + b for a, b in zip(y, third)]
    return y


def f16(x):
    import struct
    return struct.unpack("<e", struct.pack("<e", x))[0]


def matvec16(n, k, extra=None):
    x, w = lcg([k, n * k])[:2]
    third = lcg([k, n * k, n])[2] if extra else None
    y = [sum(x[j] * f16(w[i * k + j]) for j in range(k)) for i in range(n)]
    if extra:
        y = [a + b for a, b in zip(y, third)]
    return y


def kv_write(nkv, tmax, d):
    c, nw, sel, keep = lcg([nkv * tmax * d, nkv * d, tmax, tmax])
    return [c[(i * tmax + r) * d + col] * keep[r] + nw[i * d + col] * sel[r] for i in range(nkv) for r in range(tmax) for col in range(d)]


def swiglu(n):
    g, u = lcg([n, n])
    return [a / (1 + math.exp(-a)) * b for a, b in zip(g, u)]


def e2m1(code):
    v = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)[code & 7]
    return -v if code & 8 else v


def e4m3(b):
    e, m = b >> 3 & 15, b & 7
    v = m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
    return -v if b & 128 else v


def fp4_weights(n, k):
    """The interpreter's made-up NVFP4 weights: code bytes and scale bytes 0x30 + i % 8."""
    w = []
    for row in range(n):
        for c in range(k):
            byte = 48 + (row * (k // 2) + c // 2) % 8
            code = byte & 15 if c % 2 == 0 else byte >> 4
            scale = e4m3(48 + (row * (k // 16) + c // 16) % 8)
            w.append((code, scale))
    return w


def matvec_fp4(n, k, residual=False):
    sizes = [k, n] + ([n] if residual else [])
    x, g = lcg(sizes)[:2]
    r = lcg(sizes)[2] if residual else None
    w = fp4_weights(n, k)
    y = [sum(x[c] * e2m1(w[row * k + c][0]) * w[row * k + c][1] * g[row] for c in range(k)) for row in range(n)]
    return [a + b for a, b in zip(y, r)] if residual else y


def matvec_fp4_swiglu(n, k):
    x, g1, g2 = lcg([k, n, n])
    w = fp4_weights(n, k)
    def dot(row, g):
        return sum(x[c] * e2m1(w[row * k + c][0]) * w[row * k + c][1] * g[row] for c in range(k))
    return [silu_(dot(row, g1)) * dot(row, g2) for row in range(n)]


def silu_(x):
    return x / (1 + math.exp(-x))


def conv_step(c):
    x, cs, cw = lcg([c, 3 * c, 4 * c])
    y = [silu_(cs[j] * cw[j] + cs[c + j] * cw[c + j] + cs[2 * c + j] * cw[2 * c + j] + x[j] * cw[3 * c + j]) for j in range(c)]
    return y, cs[c:] + x


def qk_norm(nk, rows):
    (c,) = lcg([rows * 128])
    def norm(r, scale):
        v = c[r * 128:(r + 1) * 128]
        rs = 1 / math.sqrt(sum(a * a for a in v) / 128 + 1e-6)
        return [a * rs * scale for a in v]
    return ([x for r in range(nk) for x in norm(r, 0.0078125)],
            [x for r in range(nk) for x in norm(nk + r, 0.08838834764831845)])


def softplus_(x):
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


def gates(h):
    a, b, aa, dtb = lcg([h] * 4)
    g = [math.exp(-aa[i] * softplus_(a[i] + dtb[i])) for i in range(h)]
    bt = [1 / (1 + math.exp(-x)) for x in b]
    return [g[i] for i in range(h) for _ in range(128)], [bt[i] for i in range(h) for _ in range(128)]


def qnorm(r, n):
    x, w = lcg([r * 2 * n, n])
    out = []
    for i in range(r):
        v = x[i * 2 * n:i * 2 * n + n]
        rs = 1 / math.sqrt(sum(a * a for a in v) / n + 1e-6)
        out += [v[j] * rs * w[j] for j in range(n)]
    return out


def attn_gate_q(r, n):
    o, qg = lcg([r * n, r * 2 * n])
    return [o[i * n + j] / (1 + math.exp(-qg[i * 2 * n + n + j])) for i in range(r) for j in range(n)]


def delta_step(nk):
    s, q, k, v, g, bt = lcg([nk * 384 * 128, nk * 128, nk * 128, nk * 384, nk * 3 * 128, nk * 384])
    so, y = [0.0] * len(s), [0.0] * len(v)
    for i in range(nk):
        kb, qb = k[i * 128:(i + 1) * 128], q[i * 128:(i + 1) * 128]
        for j in range(3):
            gj = g[(i * 3 + j) * 128:(i * 3 + j + 1) * 128]
            for r in range(128):
                row = i * 384 + j * 128 + r
                sg = [s[row * 128 + c] * gj[c] for c in range(128)]
                kvm = sum(sg[c] * kb[c] for c in range(128))
                e = (v[row] - kvm) * bt[row]
                sn = [sg[c] + kb[c] * e for c in range(128)]
                so[row * 128:(row + 1) * 128] = sn
                y[row] = sum(sn[c] * qb[c] for c in range(128))
    return so, y


def rmsnorm_rows(r, n):
    x, w = lcg([r * n, n])
    out = []
    for i in range(r):
        v = x[i * n:(i + 1) * n]
        rs = 1 / math.sqrt(sum(a * a for a in v) / n + 1e-6)
        out += [v[j] * rs * w[j] for j in range(n)]
    return out


def gated_norm(h, n):
    x, z, w = lcg([h * n, h * n, n])
    out = []
    for i in range(h):
        v = x[i * n:(i + 1) * n]
        rs = 1 / math.sqrt(sum(a * a for a in v) / n + 1e-6)
        out += [v[j] * rs * w[j] * silu_(z[i * n + j]) for j in range(n)]
    return out


ok = True
for r, h, p in ((24, 32, 192), (3, 4, 2)):
    ok &= check(f"rope r={r} h={h} p={p}", run("rope", [f"r={r}", f"h={h}", f"p={p}"])["y"], rope(r, h, p))
for nkv, g, t, d, tmax in ((2, 6, 64, 256, 96), (3, 2, 16, 256, 20)):
    got = run("attention", [f"nkv={nkv}", f"g={g}", f"t={t}", f"d={d}", f"tmax={tmax}"])["o"]
    ok &= check(f"attention nkv={nkv} g={g} t={t} d={d} tmax={tmax}", got, attention(nkv, g, t, d, tmax))
for nkv, tmax, d, pos in ((4, 64, 256, 17), (2, 8, 16, 0), (2, 2, 1, 1), (4, 1, 1, 0)):
    got = run("kv_append", [f"nkv={nkv}", f"tmax={tmax}", f"d={d}", f"pos={pos}"])
    kc, vc = kv_append(nkv, tmax, d, pos)
    ok &= check(f"kv_append kc nkv={nkv} tmax={tmax} d={d} pos={pos}", got["kc"], kc)
    ok &= check(f"kv_append vc nkv={nkv} tmax={tmax} d={d} pos={pos}", got["vc"], vc)
for r, n in ((24, 256), (3, 7)):
    ok &= check(f"attn_gate r={r} n={n}", run("attn_gate", [f"r={r}", f"n={n}"])["y"], attn_gate(r, n))
for kernel, n, k, extra in (("matvec_rows", 64, 96, None), ("matvec_bias", 64, 96, "bias"), ("matvec_res", 64, 96, "res")):
    ok &= check(f"{kernel} n={n} k={k}", run(kernel, [f"n={n}", f"k={k}"])["y"], matvec(n, k, extra))
for kernel, extra in (("matvec_rows_f16", None), ("matvec_bias_f16", "bias"), ("matvec_res_f16", "res")):
    ok &= check(f"{kernel} n=64 k=96", run(kernel, ["n=64", "k=96"])["y"], matvec16(64, 96, extra))
ok &= check("kv_write nkv=2 tmax=8 d=4", run("kv_write", ["nkv=2", "tmax=8", "d=4"])["y"], kv_write(2, 8, 4))
(nw,) = lcg([2 * 4])
want = [nw[i * 4 + col] if r == 0 else 0.0 for i in range(2) for r in range(8) for col in range(4)]
ok &= check("kv_row nkv=2 tmax=8 d=4", run("kv_row", ["nkv=2", "tmax=8", "d=4"])["y"], want)
ok &= check("swiglu n=100", run("swiglu", ["n=100"])["y"], swiglu(100))
got = run("attention_d64", ["nkv=2", "g=7", "t=32", "d=64", "tmax=40"])["o"]
ok &= check("attention_d64 nkv=2 g=7 t=32 d=64 tmax=40", got, attention(2, 7, 32, 64, 40, 0.125))
ok &= check("matvec_fp4_rows n=16 k=64", run("matvec_fp4_rows", ["n=16", "k=64"])["y"], matvec_fp4(16, 64))
ok &= check("matvec_fp4_swiglu n=16 k=64", run("matvec_fp4_swiglu", ["n=16", "k=64"])["y"], matvec_fp4_swiglu(16, 64))
ok &= check("matvec_fp4_res n=16 k=64", run("matvec_fp4_res", ["n=16", "k=64"])["y"], matvec_fp4(16, 64, True))
got = run("conv_step", ["c=512"])
want_y, want_ns = conv_step(512)
ok &= check("conv_step y c=512", got["y"], want_y)
ok &= check("conv_step ns c=512", got["ns"], want_ns)
got = run("qk_norm", ["nk=2", "rows=6"])
want_q, want_k = qk_norm(2, 6)
ok &= check("qk_norm qn nk=2", got["qn"], want_q)
ok &= check("qk_norm kn nk=2", got["kn"], want_k)
got = run("gates", ["h=3"])
want_g, want_b = gates(3)
ok &= check("gates g h=3", got["g"], want_g)
ok &= check("gates bt h=3", got["bt"], want_b)
got = run("delta_step", ["nk=2"])
want_so, want_y = delta_step(2)
ok &= check("delta_step so nk=2", got["so"], want_so)
ok &= check("delta_step y nk=2", got["y"], want_y)
ok &= check("qnorm r=3 n=8", run("qnorm", ["r=3", "n=8", "eps=1e-6"])["y"], qnorm(3, 8))
ok &= check("attn_gate_q r=3 n=8", run("attn_gate_q", ["r=3", "n=8"])["y"], attn_gate_q(3, 8))
ok &= check("rmsnorm_rows r=3 n=16", run("rmsnorm_rows", ["r=3", "n=16", "eps=1e-6"])["y"], rmsnorm_rows(3, 16))
ok &= check("gated_norm h=3 n=16", run("gated_norm", ["h=3", "n=16", "eps=1e-6"])["y"], gated_norm(3, 16))
sys.exit(0 if ok else 1)
