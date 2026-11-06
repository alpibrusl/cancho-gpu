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
hold the kernels' text and summaries byte for byte (they were first recorded
from lex-gpu's Rust emitter and are regression goldens now, `docs/chx.md` section 6); this is what says the
summaries are *right*.

    kernels.py <cancho-gpu> [--new] [--mutants]

`--new` runs only the references of `docs/native.md` N2b (the kernels that had none); `--mutants` makes one deliberate
change to a copy of each of those kernels and requires its reference to fail (`CHX_ROOT` points `run` at the copy).
"""
import math
import os
import re
import subprocess
import sys
import tempfile

BIN = sys.argv[1]
CHX_ROOT = os.environ.get("CHX_ROOT", "tests/chx")


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
    r = subprocess.run([BIN, "check", f"{CHX_ROOT}/{kernel}.chx"] + consts + ["--run"], capture_output=True, text=True)
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


def check(name, per_target, want, need=2, tol=1e-5):
    mean, peak, first = summary(want)
    ok = True
    for target, (gm, gp, gf) in sorted(per_target.items()):
        good = abs(gm - mean) <= 6e-5 and abs(gp - peak) <= 6e-5 and all(abs(a - b) <= tol for a, b in zip(gf, first))
        print(f"{'ok  ' if good else 'FAIL'} {name} [{target}]: kernel mean {gm} peak {gp}; reference mean {mean:.4f} peak {peak:.4f}")
        if not good:
            print(f"     first four: kernel {gf}, reference {first}")
        ok &= good
    return ok and len(per_target) == need


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


def matvec_fp4_batch(m, n, k, kind=None):
    """The batched kernels: every row of x against the same NVFP4 weights; `kind` res adds a residual
    r[m, n] (the last input), swiglu takes a second weight (its own f32 scale) for the up projection."""
    sizes = [m * k, n] + ([n] if kind == "swiglu" else []) + ([m * n] if kind == "res" else [])
    ins = lcg(sizes)
    x, g = ins[0], ins[1]
    w = fp4_weights(n, k)
    def dot(t, row, scale):
        return sum(x[t * k + c] * e2m1(w[row * k + c][0]) * w[row * k + c][1] * scale[row] for c in range(k))
    out = []
    for t in range(m):
        for row in range(n):
            v = dot(t, row, g)
            if kind == "res":
                v += ins[2][t * n + row]
            elif kind == "swiglu":
                v = silu_(v) * dot(t, row, ins[2])
            out.append(v)
    return out


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


def qk_l2norm(nk, rows):
    """qk_norm for Hugging Face's l2norm: x * rsqrt(sum x^2 + 1e-6), the queries times 128^-1/2 (T6, docs/weight-space.md 14.9)."""
    (c,) = lcg([rows * 128])
    def norm(r, scale):
        v = c[r * 128:(r + 1) * 128]
        rs = 1 / math.sqrt(sum(a * a for a in v) + 1e-6)
        return [a * rs * scale for a in v]
    return ([x for r in range(nk) for x in norm(r, 0.08838834764831845)],
            [x for r in range(nk) for x in norm(nk + r, 1.0)])


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


def sigmoid_(x):
    return 1 / (1 + math.exp(-x))


def grad_formulas_agree(r=3, n=4, k=5, h=1e-6):
    """The reference below is only a reference if its formulas are the
    gradients of the forward they claim to be. Finite differences of
    L = sum(g * y), y = sigmoid(x w^T + b), on a small case: dx, dw, db and
    the sigmoid's own derivative against central differences (binary64)."""
    x, w, b, g = lcg([r * k, n * k, n, r * n])

    def loss(x, w, b):
        tot = 0.0
        for i in range(r):
            for j in range(n):
                z = sum(x[i * k + c] * w[j * k + c] for c in range(k)) + b[j]
                tot += g[i * n + j] * sigmoid_(z)
        return tot

    dz = []
    for i in range(r):
        for j in range(n):
            z = sum(x[i * k + c] * w[j * k + c] for c in range(k)) + b[j]
            s = sigmoid_(z)
            dz.append(g[i * n + j] * (s - s * s))
    dx = [sum(dz[i * n + j] * w[j * k + c] for j in range(n)) for i in range(r) for c in range(k)]
    dw = [sum(dz[i * n + j] * x[i * k + c] for i in range(r)) for j in range(n) for c in range(k)]
    db = [sum(dz[i * n + j] for i in range(r)) for j in range(n)]
    ok = True
    for name, params, grads in (("x", x, dx), ("w", w, dw), ("b", b, db)):
        for idx in range(len(params)):
            up, dn = list(params), list(params)
            up[idx] += h
            dn[idx] -= h
            args = {"x": x, "w": w, "b": b}
            args[name] = up
            lu = loss(args["x"], args["w"], args["b"])
            args[name] = dn
            ld = loss(args["x"], args["w"], args["b"])
            if abs((lu - ld) / (2 * h) - grads[idx]) > 1e-7:
                ok = False
    print(f"{'ok  ' if ok else 'FAIL'} backward formulas against central differences (r={r} n={n} k={k})")
    return ok


def bwd_dx(r, n, k):
    dy, w = lcg([r * n, n * k])
    return [sum(dy[i * n + j] * w[j * k + c] for j in range(n)) for i in range(r) for c in range(k)]


def bwd_dw(r, n, k):
    dyt, xt = lcg([n * r, k * r])
    return [sum(dyt[j * r + i] * xt[c * r + i] for i in range(r)) for j in range(n) for c in range(k)]


def bwd_db(r, n):
    ones, dy = lcg([r, r * n])
    return [sum(ones[i] * dy[i * n + j] for i in range(r)) for j in range(n)]


def bwd_sigmoid(r, n):
    z, dy = lcg([r * n, r * n])
    out = []
    for a, g in zip(z, dy):
        s = sigmoid_(a)
        out.append(g * (s - s * s))
    return out


def run_full(kernel, consts, names, sizes, out, target="nvidia-ada"):
    """`cancho-gpu run` over the interpreter's own tensors, written to files in
    parameter order (docs/run.md gates 1 and 2): the whole of `out`, and the
    first four values / mean / peak that `check --run` printed for the same
    program, which must be the same numbers."""
    tensors = lcg(sizes)
    with tempfile.TemporaryDirectory() as d:
        argv = [BIN, "run", f"tests/chx/{kernel}.chx"] + consts + ["--target", target]
        for name, values in zip(names, tensors):
            path = os.path.join(d, name)
            with open(path, "w") as f:
                f.write(" ".join(repr(v) for v in values) + "\n")
            argv += ["--in", name, path]
        r = subprocess.run(argv + ["--out", out], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"run {kernel} {consts}: {r.stderr.strip()}")
    return [float(x) for x in r.stdout.split()]


def check_full(name, kernel, consts, names, sizes, out, want=None):
    got = run_full(kernel, consts, names, sizes, out)
    summary_got = run(kernel, consts)[out]["nvidia-ada"]
    mean, peak, first = summary(got)
    # Gate 1: the same execution as `check --run`.
    same = abs(summary_got[0] - mean) <= 6e-5 and abs(summary_got[1] - peak) <= 6e-5 and summary_got[2] == first
    print(f"{'ok  ' if same else 'FAIL'} run == check --run: {name} ({len(got)} values)")
    ok = same
    # Gate 2: every element against the reference, not four.
    if want is not None:
        worst = max(abs(a - b) for a, b in zip(got, want)) if len(got) == len(want) else float("inf")
        good = worst <= 1e-5
        print(f"{'ok  ' if good else 'FAIL'} whole output: {name}, largest difference {worst:.2e}")
        ok &= good
    return ok


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
for kern, kind in (("matvec_fp4_batch", None), ("matvec_fp4_res_batch", "res"), ("matvec_fp4_swiglu_batch", "swiglu")):
    for m_ in (1, 3):
        ok &= check(f"{kern} m={m_} n=64 k=128", run(kern, [f"m={m_}", "n=64", "k=128"])["y"], matvec_fp4_batch(m_, 64, 128, kind))
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
ok &= grad_formulas_agree()
for r, n, k in ((8, 32, 64), (1, 5, 7)):
    ok &= check(f"bwd_dx r={r} n={n} k={k}", run("bwd_dx", [f"r={r}", f"n={n}", f"k={k}"])["dx"], bwd_dx(r, n, k))
for r, n, k in ((8, 32, 64), (256, 32, 64)):
    ok &= check(f"bwd_dw r={r} n={n} k={k}", run("bwd_dw", [f"r={r}", f"n={n}", f"k={k}"])["dw"], bwd_dw(r, n, k))
ok &= check("bwd_db r=64 n=32", run("bwd_db", ["r=64", "n=32"])["db"], bwd_db(64, 32))
ok &= check("bwd_sigmoid r=8 n=64", run("bwd_sigmoid", ["r=8", "n=64"])["dz"], bwd_sigmoid(8, 64))
for r, n, k in ((8, 32, 64), (1, 5, 7)):
    ok &= check_full(f"bwd_dx r={r} n={n} k={k}", "bwd_dx", [f"r={r}", f"n={n}", f"k={k}"], ["dy", "w"], [r * n, n * k], "dx", bwd_dx(r, n, k))
for r, n, k in ((8, 32, 64), (256, 32, 64)):
    ok &= check_full(f"bwd_dw r={r} n={n} k={k}", "bwd_dw", [f"r={r}", f"n={n}", f"k={k}"], ["dyt", "xt"], [n * r, k * r], "dw", bwd_dw(r, n, k))
ok &= check_full("bwd_db r=64 n=32", "bwd_db", ["r=64", "n=32"], ["ones", "dy"], [64, 64 * 32], "db", bwd_db(64, 32))
ok &= check_full("bwd_sigmoid r=8 n=64", "bwd_sigmoid", ["r=8", "n=64"], ["z", "dy"], [8 * 64, 8 * 64], "dz", bwd_sigmoid(8, 64))
ok &= check_full("attn_gate r=3 n=7", "attn_gate", ["r=3", "n=7"], ["o", "gate"], [21, 21], "y", attn_gate(3, 7))
ok &= check_full("softmax r=4 n=64", "softmax", ["r=4", "n=64"], ["x"], [256], "y")
# ---- the speculative check's multi-row kernels (docs/qwen35.md): each is the single-row kernel's mathematics over m rows ----------------

def conv_step3(c):
    x, cs, cw = lcg([3 * c, 3 * c, 4 * c])
    seq = [cs[0:c], cs[c:2 * c], cs[2 * c:3 * c], x[0:c], x[c:2 * c], x[2 * c:3 * c]]
    y = []
    for t in range(3):
        y += [silu_(sum(seq[t + k][j] * cw[k * c + j] for k in range(4))) for j in range(c)]
    return y, [v for t in range(3) for v in seq[t + 1] + seq[t + 2] + seq[t + 3]]


def gates_rows(m, h):
    a, b, aa, dtb = lcg([m * h, m * h, h, h])
    g = [math.exp(-aa[i] * softplus_(a[t * h + i] + dtb[i])) for t in range(m) for i in range(h)]
    bt = [1 / (1 + math.exp(-x)) for x in b]
    return [x for v in g for x in [v] * 128], [x for v in bt for x in [v] * 128]


def qk_norm_rows(m, nk, rows):
    (c,) = lcg([m * rows * 128])
    def norm(r, scale):
        v = c[r * 128:(r + 1) * 128]
        rs = 1 / math.sqrt(sum(a * a for a in v) / 128 + 1e-6)
        return [a * rs * scale for a in v]
    return ([x for t in range(m) for r in range(nk) for x in norm(t * rows + r, 0.0078125)],
            [x for t in range(m) for r in range(nk) for x in norm(t * rows + nk + r, 0.08838834764831845)])


def matvec16_m(m, n, k):
    x, w = lcg([m * k, n * k])
    return [sum(x[t * k + j] * f16(w[i * k + j]) for j in range(k)) for t in range(m) for i in range(n)]


def matvec16h(m, n, k):
    x, w = lcg([m * k, n * k])
    return [sum(f16(x[t * k + j]) * f16(w[i * k + j]) for j in range(k)) for t in range(m) for i in range(n)]


def attention_scores(nkv, g, t, d, tmax):
    q, k, mask = lcg([nkv * g * d, nkv * tmax * d, t])
    return [sum(q[(i * g + r) * d + c] * k[(i * tmax + pos) * d + c] for c in range(d)) * 0.0625 + mask[pos]
            for i in range(nkv) for r in range(g) for pos in range(t)]


def attention_pv(nkv, g, t, d, tmax):
    sc, v = lcg([nkv * g * t, nkv * tmax * d])
    out = []
    for i in range(nkv):
        for r in range(g):
            s = sc[(i * g + r) * t:(i * g + r + 1) * t]
            mx = max(s)
            e = [math.exp(x - mx) for x in s]
            z = sum(e)
            for col in range(d):
                out.append(sum(e[pos] / z * v[(i * tmax + pos) * d + col] for pos in range(t)))
    return out


def attention_stats(nkv, g, t):
    (sc,) = lcg([nkv * g * t])
    m, z = [], []
    for r in range(nkv * g):
        row = sc[r * t:(r + 1) * t]
        mx = max(row)
        m.append(mx)
        z.append(sum(math.exp(x - mx) for x in row))
    return m, z


def dec_post(r):
    y, bias, cs, sn = lcg([r * 16 * 128, 16 * 128, r * 64, r * 64])
    q, kn, vn = [], [], []
    for i in range(r):
        def rot(slot):
            a1 = [y[(i * 16 + slot) * 128 + c] + bias[slot * 128 + c] for c in range(64)]
            a2 = [y[(i * 16 + slot) * 128 + 64 + c] + bias[slot * 128 + 64 + c] for c in range(64)]
            return ([a1[c] * cs[i * 64 + c] - a2[c] * sn[i * 64 + c] for c in range(64)] + [a2[c] * cs[i * 64 + c] + a1[c] * sn[i * 64 + c] for c in range(64)])
        for h in range(12):
            q += rot(h)
        for h in range(2):
            kn += rot(12 + h)
        for h in range(2):
            vn += [y[(i * 16 + 14 + h) * 128 + c] + bias[(14 + h) * 128 + c] for c in range(128)]
    return q, kn, vn


def kv_write_b(nb, nkv, tmax, d):
    c, nw, sel, keep = lcg([nb * nkv * tmax * d, nb * nkv * d, nb * tmax, nb * tmax])
    return [c[((r * nkv + h) * tmax + pos) * d + col] * keep[r * tmax + pos] + nw[(r * nkv + h) * d + col] * sel[r * tmax + pos]
            for r in range(nb) for h in range(nkv) for pos in range(tmax) for col in range(d)]


def ids_mix(r):
    s, tf, m = lcg([r, r, r])
    return [m[i] * tf[i] + (1 - m[i]) * s[i] for i in range(r)]


def attention_scores_b(nb, nkv, g, t, d, tmax):
    q, k, mask = lcg([nb * nkv * g * d, nb * nkv * tmax * d, nb * t])
    out = []
    for r in range(nb):
        for h in range(nkv):
            for i in range(g):
                row = (r * nkv + h) * g + i
                for pos in range(t):
                    out.append(sum(q[row * d + c] * k[((r * nkv + h) * tmax + pos) * d + c] for c in range(d)) * 0.08838834764831845 + mask[r * t + pos])
    return out


def attention_pv_b(nb, nkv, g, t, d, tmax):
    sc, v = lcg([nb * nkv * g * t, nb * nkv * tmax * d])
    out = []
    for r in range(nb):
        for h in range(nkv):
            for i in range(g):
                row = (r * nkv + h) * g + i
                s = sc[row * t:(row + 1) * t]
                mx = max(s)
                e = [math.exp(x - mx) for x in s]
                z = sum(e)
                for col in range(d):
                    out.append(sum(e[pos] / z * v[((r * nkv + h) * tmax + pos) * d + col] for pos in range(t)))
    return out


def attention_pvl(nkv, g, t, d, tmax):
    sc, m, z, v = lcg([nkv * g * t, nkv * g, nkv * g, nkv * tmax * d])
    out = []
    for i in range(nkv):
        for r in range(g):
            row = i * g + r
            e = [math.exp(sc[row * t + pos] - m[row]) / z[row] for pos in range(t)]
            for col in range(d):
                out.append(sum(e[pos] * v[(i * tmax + pos) * d + col] for pos in range(t)))
    return out


def delta_step3(nk):
    s, q, k, v, g, bt = lcg([nk * 384 * 128, 3 * nk * 128, 3 * nk * 128, 30720, 3 * nk * 3 * 128, 3 * nk * 384])
    so = [[0.0] * len(s) for _ in range(3)]
    y = [0.0] * (3 * nk * 384)
    for i in range(nk):
        for j in range(3):
            for r in range(128):
                row = i * 384 + j * 128 + r
                state = s[row * 128:(row + 1) * 128]
                for t in range(3):
                    kb = k[(t * nk + i) * 128:(t * nk + i + 1) * 128]
                    qb = q[(t * nk + i) * 128:(t * nk + i + 1) * 128]
                    gj = g[(t * nk * 3 + i * 3 + j) * 128:(t * nk * 3 + i * 3 + j + 1) * 128]
                    sg = [state[c] * gj[c] for c in range(128)]
                    e = (v[t * 10240 + 4096 + row] - sum(sg[c] * kb[c] for c in range(128))) * bt[t * nk * 384 + row]
                    state = [sg[c] + kb[c] * e for c in range(128)]
                    so[t][row * 128:(row + 1) * 128] = state
                    y[t * nk * 384 + row] = sum(state[c] * qb[c] for c in range(128))
    return so, y


# ---- N2b: the kernels that had no reference (docs/native.md) -------------------------------------------------------------------------

def gemm_ref(m, n, k, wide_a=False):
    """y = a w^T over f16 inputs, accumulated wide (the kernel's f32 sum differs in the last bits)."""
    a, w = lcg([m * k, n * k])
    a = [x if wide_a else f16(x) for x in a]
    w = [f16(x) for x in w]
    return [sum(a[i * k + c] * w[j * k + c] for c in range(k)) for i in range(m) for j in range(n)]


def gemm_nn_ref(m, n, k):
    """y = a w with w stored [k, n]: the mma_nn layout."""
    a, w = lcg([m * k, k * n])
    a = [f16(x) for x in a]
    w = [f16(x) for x in w]
    return [sum(a[i * k + c] * w[c * n + j] for c in range(k)) for i in range(m) for j in range(n)]


def gemm_tn_ref(m, n, k):
    """y = a^T w with a stored [k, m] and w stored [k, n]: the mma_tn layout."""
    a, w = lcg([k * m, k * n])
    a = [f16(x) for x in a]
    w = [f16(x) for x in w]
    return [sum(a[c * m + i] * w[c * n + j] for c in range(k)) for i in range(m) for j in range(n)]


def gemm_fp4_ref(m, n, k, residual=False):
    """NVFP4 weights dequantised (code * block scale * the row's f32 scale), rounded to f16 by `stage f16`, against f16 activations."""
    ins = lcg([m * k, n] + ([m * n] if residual else []))
    x = [f16(v) for v in ins[0]]
    g = ins[1]
    w = fp4_weights(n, k)
    out = []
    for i in range(m):
        for j in range(n):
            acc = sum(x[i * k + c] * f16(e2m1(w[j * k + c][0]) * w[j * k + c][1] * g[j]) for c in range(k))
            out.append(acc + ins[2][i * n + j] if residual else acc)
    return out


def rmsnorm_ref(n, eps):
    x, w = lcg([n, n])
    rs = 1 / math.sqrt(sum(a * a for a in x) / n + eps)
    return [x[j] * rs * w[j] for j in range(n)]


def silu_mul_ref(n):
    g, u = lcg([n, n])
    return [a / (1 + math.exp(-a)) * b for a, b in zip(g, u)]


def softmax_ref(r, n):
    x = lcg([r * n])[0]
    out = []
    for i in range(r):
        v = x[i * n:(i + 1) * n]
        mx = max(v)
        e = [math.exp(a - mx) for a in v]
        out += [a / sum(e) for a in e]
    return out


def rowops_ref(r, n):
    x, b = lcg([r * n, n])
    out = []
    for i in range(r):
        v = x[i * n:(i + 1) * n]
        mx = max(v)
        z = [softplus_(a - mx) + softplus_(b[j]) for j, a in enumerate(v)]
        s = [1 / math.sqrt(a) for a in z]
        ss = sum(s)
        out += [2 * (s[j] * z[j]) / ss for j in range(n)]
    return out


def stage_ref(m, k, bk):
    """The kernel adds the k / bk column slices of `a` into an [m, bk] accumulator and stores it at the origin of y."""
    a = lcg([m * k])[0]
    out = [0.0] * (m * k)
    for i in range(m):
        for c in range(bk):
            out[i * k + c] = sum(a[i * k + p * bk + c] for p in range(k // bk))
    return out


def new_checks():
    ok = True
    for m, n, k in ((128, 128, 64), (48, 16, 32)):
        ok &= check(f"gemm m={m} n={n} k={k}", run("ported/gemm", [f"m={m}", f"n={n}", f"k={k}"])["y"], gemm_ref(m, n, k))
    ok &= check("gemm_mma m=128 n=128 k=64", run("ported/gemm_mma", ["m=128", "n=128", "k=64"])["y"], gemm_ref(128, 128, 64))
    ok &= check("gemm_mma_nn m=128 n=128 k=64", run("ported/gemm_mma_nn", ["m=128", "n=128", "k=64"])["y"], gemm_nn_ref(128, 128, 64))
    ok &= check("gemm_mma_tn m=128 n=128 k=64", run("ported/gemm_mma_tn", ["m=128", "n=128", "k=64"])["y"], gemm_tn_ref(128, 128, 64))
    ok &= check("gemm_fp4 m=128 n=128 k=64", run("ported/gemm_fp4", ["m=128", "n=128", "k=64"])["y"], gemm_fp4_ref(128, 128, 64))
    ok &= check("gemm_fp4_res m=128 n=128 k=64", run("ported/gemm_fp4_res", ["m=128", "n=128", "k=64"])["y"], gemm_fp4_ref(128, 128, 64, True))
    ok &= check("rmsnorm n=4096 eps=1e-5", run("ported/rmsnorm", ["n=4096", "eps=1e-5"])["y"], rmsnorm_ref(4096, 1e-5))
    ok &= check("silu_mul", run("ported/silu_mul", [])["y"], silu_mul_ref(16384))
    ok &= check("softmax r=4 n=64", run("softmax", ["r=4", "n=64"])["y"], softmax_ref(4, 64))
    for r, n in ((4, 1024), (300, 7)):
        ok &= check(f"rowops r={r} n={n}", run("rowops", [f"r={r}", f"n={n}"])["y"], rowops_ref(r, n))
    # k = 4096 terms summed in order in f32: measured 1.7e-5 from the binary64 sum on terms that add up to 254 in absolute value, about
    # one f32 rounding step at that scale (n * eps would allow 0.06), so 1e-4 here and 1e-5 for the shorter sums.
    ok &= check("matvec n=64 k=4096", run("matvec", ["n=64", "k=4096"])["y"], matvec(64, 4096), tol=1e-4)
    ok &= check("matvec_fp4 n=64 k=4096", run("matvec_fp4", ["n=64", "k=4096"])["y"], matvec_fp4(64, 4096))
    ok &= check("batch_fp4 b=4 n=64 k=512", run("batch_fp4", ["b=4", "n=64", "k=512"])["y"], matvec_fp4_batch(4, 64, 512))
    # `stage` has a different tile on each target (bk 16 on apple-m-series, 32 on nvidia-ada): each is held to its own.
    got = run("stage", ["m=8", "k=64"])["y"]
    for target, bk in (("apple-m-series", 16), ("nvidia-ada", 32)):
        ok &= check(f"stage m=8 k=64 bk={bk}", {target: got[target]}, stage_ref(8, 64, bk), need=1)
    got = run("conv_step3", ["c=512"])
    want_y, want_n = conv_step3(512)
    ok &= check("conv_step3 y c=512", got["y"], want_y)
    ok &= check("conv_step3 n1 c=512", got["n1"], want_n[0:1536])
    ok &= check("conv_step3 n2 c=512", got["n2"], want_n[1536:3072])
    ok &= check("conv_step3 n3 c=512", got["n3"], want_n[3072:4608])
    got = run("gates_rows", ["m=3", "h=5"])
    want_g, want_b = gates_rows(3, 5)
    ok &= check("gates_rows g m=3 h=5", got["g"], want_g)
    ok &= check("gates_rows bt m=3 h=5", got["bt"], want_b)
    got = run("qk_norm_rows", ["m=3", "nk=2", "rows=6"])
    want_q, want_k = qk_norm_rows(3, 2, 6)
    ok &= check("qk_norm_rows qn m=3 nk=2", got["qn"], want_q)
    ok &= check("qk_norm_rows kn m=3 nk=2", got["kn"], want_k)
    ok &= check("matvec_rows_f16_m m=3 n=64 k=96", run("matvec_rows_f16_m", ["m=3", "n=64", "k=96"])["y"], matvec16_m(3, 64, 96))
    ok &= check("matvec_rows_f16_frag m=3 n=64 k=96", run("matvec_rows_f16_frag", ["m=3", "n=64", "k=96"])["y"], matvec16_m(3, 64, 96))
    ok &= check("matvec_rows_f16_frag m=32 n=32 k=256", run("matvec_rows_f16_frag", ["m=32", "n=32", "k=256"])["y"], matvec16_m(32, 32, 256))
    ok &= check("matvec_rows_f16h m=3 n=64 k=96", run("matvec_rows_f16h", ["m=3", "n=64", "k=96"])["y"], matvec16h(3, 64, 96))
    ok &= check("matvec_rows_f16h m=32 n=32 k=256", run("matvec_rows_f16h", ["m=32", "n=32", "k=256"])["y"], matvec16h(32, 32, 256))
    got = run("delta_step3", ["nk=2"])
    want_so, want_y = delta_step3(2)
    for t in range(3):
        ok &= check(f"delta_step3 so{t + 1} nk=2", got[f"so{t + 1}"], want_so[t])
    ok &= check("delta_step3 y nk=2", got["y"], want_y)
    ok &= check("attention_scores nkv=2 g=6 t=64 ts=16", run("attention_scores", ["nkv=2", "g=6", "t=64", "ts=16", "d=256", "tmax=96"])["sc"], attention_scores(2, 6, 64, 256, 96))
    ok &= check("attention_pv nkv=2 g=6 t=64 dw=32", run("attention_pv", ["nkv=2", "g=6", "t=64", "dw=32", "d=256", "tmax=96"])["o"], attention_pv(2, 6, 64, 256, 96))
    got = run("attention_stats", ["nkv=2", "g=4", "t=96", "cs=16"])
    want_m, want_z = attention_stats(2, 4, 96)
    ok &= check("attention_stats m nkv=2 g=4 t=96 cs=16", got["m"], want_m)
    ok &= check("attention_stats z nkv=2 g=4 t=96 cs=16", got["z"], want_z)
    got = run("dec_post", ["r=3"])
    want_q, want_k, want_v = dec_post(3)
    ok &= check("dec_post q r=3", got["q"], want_q)
    ok &= check("dec_post kn r=3", got["kn"], want_k)
    ok &= check("dec_post vn r=3", got["vn"], want_v)
    ok &= check("kv_write_b nb=2 nkv=2 tmax=32 ts=16", run("kv_write_b", ["nb=2", "nkv=2", "tmax=32", "ts=16", "d=128"])["y"], kv_write_b(2, 2, 32, 128))
    ok &= check("ids_mix r=8", run("ids_mix", ["r=8"])["y"], ids_mix(8))
    ok &= check("attention_scores_b nb=2 nkv=2 g=3 t=32 ts=16", run("attention_scores_b", ["nb=2", "nkv=2", "g=3", "t=32", "ts=16", "d=128", "tmax=40"])["sc"], attention_scores_b(2, 2, 3, 32, 128, 40))
    ok &= check("attention_pv_b nb=2 nkv=2 g=3 t=32 dw=32", run("attention_pv_b", ["nb=2", "nkv=2", "g=3", "t=32", "dw=32", "d=128", "tmax=40"])["o"], attention_pv_b(2, 2, 3, 32, 128, 40))
    ok &= check("attention_pvl nkv=2 g=4 t=96 cs=16 dw=64", run("attention_pvl", ["nkv=2", "g=4", "t=96", "cs=16", "dw=64", "d=256", "tmax=100"])["o"], attention_pvl(2, 4, 96, 256, 100))
    got = run("activations", ["r=4", "n=8"])
    (xs,) = lcg([32])
    ok &= check("activations relu r=4 n=8", got["y"], [max(v, 0.0) for v in xs])
    ok &= check("activations step r=4 n=8", got["d"], [1.0 if v > 0.0 else 0.0 for v in xs])
    ok &= check("activations tanh r=4 n=8", got["t"], [math.tanh(v) for v in xs])
    got = run("colbcast", ["r=8", "c=16"])
    x, sc, tt = lcg([128, 8, 16])
    ok &= check("colbcast r=8 c=16", got["y"], [x[i * 16 + j] * sc[i] + tt[j] for i in range(8) for j in range(16)])
    ok &= checks08()
    return ok


# ---- T3 (docs/weight-space.md section 14): the kernels at Qwen3.5-0.8B's shapes -----------------------------------------------------
# One value head per key head (nk = 16 of each, 128 wide), conv width 6144 = 3 * 2048, qk_norm rows 16 + 16 + 16 = 48, gates h = 16,
# attention at g = 4 (8 query heads over 2 key/value heads), q_proj as 8 heads of [query 256, gate 256], k as 2 heads.

def delta_rows(nk, s, q, k, v, g, bt):
    """The rule on flat inputs (delta_step1's parameters): head i's [128, 128] state, its own g, beta, v, and key head i's q, k."""
    so, y = [0.0] * len(s), [0.0] * len(v)
    for i in range(nk):
        kb, qb, gi = k[i * 128:(i + 1) * 128], q[i * 128:(i + 1) * 128], g[i * 128:(i + 1) * 128]
        for r in range(128):
            row = i * 128 + r
            sg = [s[row * 128 + c] * gi[c] for c in range(128)]
            e = (v[row] - sum(sg[c] * kb[c] for c in range(128))) * bt[row]
            sn = [sg[c] + kb[c] * e for c in range(128)]
            so[row * 128:(row + 1) * 128] = sn
            y[row] = sum(sn[c] * qb[c] for c in range(128))
    return so, y


def delta_step1(nk):
    return delta_rows(nk, *lcg([nk * 128 * 128, nk * 128, nk * 128, nk * 128, nk * 128, nk * 128]))


def delta_step1_vs_spec(nk=3):
    """delta_rows against the specification (examples/qwen35/gdn_ref.py, numpy: HF's recurrent form, one token, nk heads of dk = dv = 128), so
    that the rule is read from the mathematics a second time and not only from the kernel it checks. The spec's state is [dk, dv], the
    kernel's [dv, dk]; the spec scales q by dk^-1/2 itself (the kernel's qk_norm does) and takes the log of the decay and one beta a head, so
    the kernel's beta, constant over a head's rows, is the spec's. Returns the largest difference."""
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "qwen35"))
    import gdn_ref
    rng = np.random.RandomState(7)
    S = rng.normal(size=(nk, 128, 128)) * 0.3  # [head, value row, key col]
    q, k, v = rng.normal(size=(nk, 128)), rng.normal(size=(nk, 128)), rng.normal(size=(nk, 128))
    decay, beta = rng.uniform(0.3, 1.0, size=nk), rng.uniform(0.1, 0.9, size=nk)
    so, y = delta_rows(nk, S.ravel().tolist(), q.ravel().tolist(), k.ravel().tolist(), v.ravel().tolist(),
                       np.repeat(decay, 128).tolist(), np.repeat(beta, 128).tolist())
    o, S1 = gdn_ref.recurrent((q * 128 ** 0.5)[:, None, :], k[:, None, :], v[:, None, :], np.log(decay)[:, None], beta[:, None], S0=S.transpose(0, 2, 1))
    return max(abs(o[:, 0, :].ravel() - np.array(y)).max(), abs(S1.transpose(0, 2, 1).ravel() - np.array(so)).max())


def checks08():
    ok = True
    # delta_step1: the whole of both outputs at nk = 2 and at the model's nk = 16, and the specification's own reading of the rule
    for nk in (2, 16):
        names, sizes = ["s", "q", "k", "v", "g", "bt"], [nk * 128 * 128, nk * 128, nk * 128, nk * 128, nk * 128, nk * 128]
        want_so, want_y = delta_step1(nk)
        ok &= check_full(f"delta_step1 so nk={nk}", "delta_step1", [f"nk={nk}"], names, sizes, "so", want_so)
        ok &= check_full(f"delta_step1 y nk={nk}", "delta_step1", [f"nk={nk}"], names, sizes, "y", want_y)
    try:
        import numpy  # noqa: F401
        worst = delta_step1_vs_spec()
        good = worst <= 1e-9
        print(f"{'ok  ' if good else 'FAIL'} delta_step1 reference == examples/qwen35/gdn_ref.py (HF's recurrent rule), 3 heads, largest difference {worst:.2e}")
        ok &= good
    except ImportError:
        print("skip delta_step1 against examples/qwen35/gdn_ref.py: numpy is not installed")
    # the other kernels, at the shapes of the 0.8B (their small cases are above and in the 27B's list)
    got = run("conv_step", ["c=6144"])
    want_y, want_ns = conv_step(6144)
    ok &= check("conv_step y c=6144", got["y"], want_y)
    ok &= check("conv_step ns c=6144", got["ns"], want_ns)
    got = run("qk_norm", ["nk=16", "rows=48"])
    want_q, want_k = qk_norm(16, 48)
    ok &= check("qk_norm qn nk=16 rows=48", got["qn"], want_q)
    ok &= check("qk_norm kn nk=16 rows=48", got["kn"], want_k)
    for nk_, rows_ in ((2, 6), (16, 48)):
        got = run("qk_l2norm", [f"nk={nk_}", f"rows={rows_}"])
        want_q, want_k = qk_l2norm(nk_, rows_)
        ok &= check(f"qk_l2norm qn nk={nk_} rows={rows_}", got["qn"], want_q)
        ok &= check(f"qk_l2norm kn nk={nk_} rows={rows_}", got["kn"], want_k)
    got = run("gates", ["h=16"])
    want_g, want_b = gates(16)
    ok &= check("gates g h=16", got["g"], want_g)
    ok &= check("gates bt h=16", got["bt"], want_b)
    ok &= check("gated_norm h=16 n=128", run("gated_norm", ["h=16", "n=128", "eps=1e-6"])["y"], gated_norm(16, 128))
    for r in (8, 2):
        ok &= check(f"rope r={r} h=32 p=192", run("rope", [f"r={r}", "h=32", "p=192"])["y"], rope(r, 32, 192))
        ok &= check(f"qnorm r={r} n=256", run("qnorm", [f"r={r}", "n=256", "eps=1e-6"])["y"], qnorm(r, 256))
    ok &= check("attn_gate_q r=8 n=256", run("attn_gate_q", ["r=8", "n=256"])["y"], attn_gate_q(8, 256))
    ok &= check("rmsnorm_rows r=1 n=1024", run("rmsnorm_rows", ["r=1", "n=1024", "eps=1e-6"])["y"], rmsnorm_rows(1, 1024))
    ok &= check("swiglu n=3584", run("swiglu", ["n=3584"])["y"], swiglu(3584))
    # attention at g = 4: the cache of the 0.8B is two heads of 256; t = 2046 is the most that Metal's 32 KiB lets one tile hold at g = 4 (cases.txt)
    for t, tmax in ((64, 96), (2046, 2048)):
        got = run("attention", ["nkv=2", "g=4", f"t={t}", "d=256", f"tmax={tmax}"])["o"]
        ok &= check(f"attention nkv=2 g=4 t={t} d=256 tmax={tmax}", got, attention(2, 4, t, 256, tmax))
    ok &= check("attention_scores nkv=2 g=4 t=256 ts=32", run("attention_scores", ["nkv=2", "g=4", "t=256", "ts=32", "d=256", "tmax=2048"])["sc"], attention_scores(2, 4, 256, 256, 2048))
    ok &= check("attention_pv nkv=2 g=4 t=256 dw=32", run("attention_pv", ["nkv=2", "g=4", "t=256", "dw=32", "d=256", "tmax=2048"])["o"], attention_pv(2, 4, 256, 256, 2048))
    # the matrix-vector products: f16 weights (the speed path), f32 weights (the exact widening of bf16), at the layer shapes. 1024 to 3584 terms
    # summed in order in f32 held to 1e-5 (measured: all of them pass; the 1e-4 of matvec n=64 k=4096 above was for terms adding up to 254).
    # The head's n = 248,320 rows do not fit the interpreter at k = 1024 (nor k = 128; the 27B's head is a known `noref`,
    # tests/cuda-emu-known.txt), so the head is run at k = 32 and emitted at k = 1024 in cases.txt.
    for n, k in ((3584, 1024), (6144, 1024), (248320, 32)):
        ok &= check(f"matvec_rows_f16 n={n} k={k}", run("matvec_rows_f16", [f"n={n}", f"k={k}"])["y"], matvec16(n, k), tol=1e-5)
    for k in (2048, 3584):
        ok &= check(f"matvec_res_f16 n=1024 k={k}", run("matvec_res_f16", ["n=1024", f"k={k}"])["y"], matvec16(1024, k, "res"), tol=1e-5)
    ok &= check("matvec_rows n=3584 k=1024", run("matvec_rows", ["n=3584", "k=1024"])["y"], matvec(3584, 1024), tol=1e-5)
    ok &= check("matvec_res n=1024 k=3584", run("matvec_res", ["n=1024", "k=3584"])["y"], matvec(1024, 3584, "res"), tol=1e-5)
    return ok


# One deliberate change to each kernel above, in a copy: its reference must fail. (file under CHX_ROOT, text, replacement, what it breaks)
MUTANTS = [
    ("ported/gemm.chx", "yield acc + matmul_nt at wt", "yield acc - matmul_nt at wt", "gemm accumulates with the wrong sign"),
    ("ported/gemm_mma_nn.chx", "yield mma_nn acc at wt", "yield mma acc at wt", "gemm_mma_nn reads its right tile as [n, k]"),
    ("ported/gemm_mma_tn.chx", "let at = load a[p * bk, i * bm ; bk, bm] @shared", "let at = load a[p * bk, j * bm ; bk, bm] @shared", "gemm_mma_tn takes the wrong columns of a"),
    ("ported/gemm_mma.chx", "for p in 0 .. k / bk with acc", "for p in 0 .. k / bk - 1 with acc", "gemm_mma leaves out the last slice of k"),
    ("ported/gemm_fp4.chx", "for p in 0 .. k / bk with acc", "for p in 0 .. k / bk - 1 with acc", "gemm_fp4 leaves out the last slice of k"),
    ("ported/gemm_fp4_res.chx", "  let acc = add_window acc r[i * bm, j * bn]\n", "", "gemm_fp4_res leaves out the residual"),
    ("ported/rmsnorm.chx", "let r  = rsqrt t", "let r  = t", "rmsnorm multiplies by the mean square, not its reciprocal root"),
    ("ported/silu_mul.chx", "store a * u -> y", "store a + u -> y", "silu_mul adds where it multiplies"),
    ("softmax.chx", "let q  = e / s", "let q  = e * s", "softmax multiplies by the sum"),
    ("rowops.chx", "let o  = h / ss", "let o  = h * ss", "rowops multiplies by the sum"),
    ("stage.chx", "yield acc + st", "yield acc - st", "stage subtracts"),
    ("conv_step3.chx", "store t0 * g0 -> y[0, i * bc]", "store t0 + g0 -> y[0, i * bc]", "conv_step3 adds the gate to the first row's output"),
    ("delta_step3.chx", "let e2_2 = d2_2 * b2_2", "let e2_2 = d2_2 + b2_2", "delta_step3 adds beta to the third token's error"),
    ("attention_scores.chx", "store sd + mk -> sc[i * g, sp * ts]", "store sd - mk -> sc[i * g, sp * ts]", "attention_scores subtracts the mask"),
    ("attention_pv.chx", "let p  = e / z", "let p  = e * z", "attention_pv multiplies by the sum"),
    ("attention_stats.chx", "yield m1 + relu (cm - &m1)", "yield m1 + relu (&m1 - cm)", "attention_stats keeps the minimum, not the maximum"),
    ("attention_stats.chx", "let sh = sm - &mx", "let sh = sm", "attention_stats leaves the maximum out of the normaliser"),
    ("attention_stats.chx", "for c in 1 .. t / cs with m1", "for c in 1 .. t / cs - 1 with m1", "attention_stats leaves out the last chunk of the maximum"),
    ("attention_pvl.chx", "let p  = e / zz", "let p  = e * zz", "attention_pvl multiplies by the sum"),
    ("attention_pvl.chx", "let sh = sm - mm", "let sh = sm + mm", "attention_pvl shifts the scores the wrong way"),
    ("attention_pvl.chx", "for c in 0 .. t / cs with a0", "for c in 1 .. t / cs with a0", "attention_pvl leaves out the first chunk"),
    ("activations.chx", "store relu &a -> y", "store step &a -> y", "relu is a step"),
    ("colbcast.chx", "let e = a * b", "let e = a + b", "colbcast adds the column where it multiplies"),
    ("activations.chx", "store tanh &a -> t", "store relu &a -> t", "tanh is a relu"),
    ("gates_rows.chx", "let ng = z1 - mm", "let ng = z1 + mm", "gates_rows decays upwards"),
    ("qk_norm_rows.chx", "let qx = q * qr", "let qx = q + qr", "qk_norm_rows adds the reciprocal root"),
    # T6: qk_norm with Hugging Face's epsilon
    ("qk_l2norm.chx", "store qx * 0.08838834764831845 -> qn", "store qx -> qn", "qk_l2norm leaves the queries' 1/sqrt(128) out"),
    ("qk_l2norm.chx", "store k * kr -> kn", "store k * kr * 0.08838834764831845 -> kn", "qk_l2norm scales the keys"),
    ("qk_l2norm.chx", "let kr = rsqrt ke", "let kr = ke", "qk_l2norm multiplies the keys by the sum, not its reciprocal root"),
    # T3: the one-value-head delta step
    ("delta_step1.chx", "let e0  = d0 * b0", "let e0  = d0 + b0", "delta_step1 adds beta to the error"),
    ("delta_step1.chx", "let sg0 = s0 * g0", "let sg0 = s0 + g0", "delta_step1 adds the decay to the state"),
    ("delta_step1.chx", "let g0  = load g[i, 0 ; 1, 128]", "let g0  = load g[0, 0 ; 1, 128]", "delta_step1 decays every head by the first head's gate"),
    ("delta_step1.chx", "let v0  = load v[i * 128 + r * br ; br]", "let v0  = load v[r * br ; br]", "delta_step1 reads every head's values from the first head"),
]


def mutants():
    import shutil
    ok = True
    for path, old, new, what in MUTANTS:
        d = tempfile.mkdtemp()
        try:
            shutil.copytree("tests/chx", os.path.join(d, "chx"))
            target = os.path.join(d, "chx", path)
            text = open(target).read()
            if text.count(old) != 1:
                print(f"FAIL mutant '{what}': the text it changes occurs {text.count(old)} times in {path}")
                ok = False
                continue
            open(target, "w").write(text.replace(old, new, 1))
            r = subprocess.run([sys.executable, __file__, BIN, "--new"], capture_output=True, text=True, env=dict(os.environ, CHX_ROOT=os.path.join(d, "chx")))
            fam = path.split("/")[-1].replace(".chx", "")
            failed_lines = [l for l in r.stdout.splitlines() if l.startswith("FAIL") and fam in l]
            if r.returncode != 0 and failed_lines:
                print(f"ok   mutant killed: {what}")
            elif r.returncode != 0:
                print(f"ok?  the change stopped the run without a FAIL line for {fam}: {what} ({(r.stderr or r.stdout)[-120:].strip()!r})")
            else:
                print(f"FAIL mutant '{what}' SURVIVED: its reference does not notice")
                ok = False
        finally:
            shutil.rmtree(d)
    return ok


if "--mutants" in sys.argv:
    sys.exit(0 if mutants() else 1)
if "--new" in sys.argv:
    sys.exit(0 if new_checks() else 1)
ok &= new_checks()
sys.exit(0 if ok else 1)
