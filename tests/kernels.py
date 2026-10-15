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
    r = subprocess.run([BIN, "check", f"tests/lx/{kernel}.lx"] + consts + ["--run"], capture_output=True, text=True)
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


def attention(nkv, g, t, d, tmax):
    q, k, v, mask = lcg([nkv * g * d, nkv * tmax * d, nkv * tmax * d, t])
    out = []
    for i in range(nkv):
        for r in range(g):
            qr = q[(i * g + r) * d:(i * g + r + 1) * d]
            s = []
            for pos in range(t):
                kr = k[(i * tmax + pos) * d:(i * tmax + pos + 1) * d]
                s.append(sum(a * b for a, b in zip(qr, kr)) * 0.0625 + mask[pos])  # d is 256: tests/lx/attention.lx
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
sys.exit(0 if ok else 1)
