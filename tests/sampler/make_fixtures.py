#!/usr/bin/env python3
"""Writes the sampler fixtures of tests/sampler/: logits files (little-endian
f32 rows) and `cases.txt`, one case a line:

    <file> <vocab> <temperature> <top_p> <top_k> <seed>

    python3 tests/sampler/make_fixtures.py

A case's rows are drawn in order from one sampler, as in generation. The
inputs cover ordinary logits, flat rows (ties, in index order), peaks, the
values a float can hold that a softmax must survive (infinities, NaN, signed
zeros, subnormals, the largest), and `top_k` on both sides of the 64 where
lex-rt changes algorithm.
"""
import math
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
state = [20261015]


def rnd(n):
    state[0] = (state[0] * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (state[0] >> 33) % n


def unit():
    return (rnd(1 << 30) + 0.5) / (1 << 30)


def normal():
    return math.sqrt(-2 * math.log(unit())) * math.cos(2 * math.pi * unit())


def pack(rows):
    return b"".join(struct.pack("<f", x) for r in rows for x in r)


def write(name, rows):
    with open(os.path.join(HERE, name), "wb") as f:
        f.write(pack(rows))


def main():
    cases = []
    ordinary = [[normal() * 3 for _ in range(1000)] for _ in range(40)]
    write("ordinary-1000.f32", ordinary)
    flat = [[0.5] * 300 for _ in range(20)]
    write("flat-300.f32", flat)
    ties = [[float(rnd(4)) for _ in range(500)] for _ in range(20)]
    write("ties-500.f32", ties)
    peaked = [[(30.0 if i == rnd(200) else normal()) for i in range(200)] for _ in range(20)]
    write("peaked-200.f32", peaked)
    specials = []
    base = [normal() for _ in range(40)]
    for kind in range(24):
        r = list(base)
        r[3] = [float("inf"), float("-inf"), float("nan"), -0.0, 0.0, 1e-45, 3.4028235e38, -3.4028235e38][kind % 8]
        r[7] = [float("-inf"), 1e-38, float("nan"), 0.0, -0.0, -1e-45, 1e30, -1e30][(kind // 3) % 8]
        if kind % 5 == 0:
            r = [float("-inf")] * 40
        if kind % 7 == 0:
            r[0] = float("-inf")
        specials.append(r)
    write("specials-40.f32", specials)
    tiny = [[normal()] for _ in range(10)]
    write("vocab-1.f32", tiny)
    two = [[normal(), normal()] for _ in range(20)]
    write("vocab-2.f32", two)
    for k_name, v in (("65", 65), ("5000", 5000)):
        write(f"ordinary-{k_name}.f32", [[normal() * 2 for _ in range(v)] for _ in range(12)])
    mild = [[normal() * 0.3 for _ in range(2000)] for _ in range(15)]
    write("mild-2000.f32", mild)
    for temp in ("1.0", "0.7", "0.01", "5", "0", "1e-3"):
        for top_p in ("0.95", "1.0", "0.3", "0.0"):
            cases.append(f"ordinary-1000.f32 1000 {temp} {top_p} 20 {rnd(1000)}")
    for k in ("1", "2", "20", "63", "64", "65", "66", "300", "999", "1000", "1001", "100000", "0"):
        cases.append(f"ordinary-1000.f32 1000 1.0 0.95 {k} 42")
        cases.append(f"mild-2000.f32 2000 1.3 1.0 {k} 7")
    for seed in ("0", "1", "42", "43", "1099511627776", "4611686018427387903"):
        cases.append(f"ordinary-1000.f32 1000 1.0 0.95 20 {seed}")
    cases += [
        "flat-300.f32 300 1.0 0.95 20 1", "flat-300.f32 300 1.0 1.0 300 2", "flat-300.f32 300 0 1.0 20 3",
        "ties-500.f32 500 1.0 0.95 20 4", "ties-500.f32 500 0.5 1.0 100 5", "ties-500.f32 500 0 1.0 20 6",
        "peaked-200.f32 200 1.0 0.95 20 8", "peaked-200.f32 200 0.3 0.5 5 9",
        "specials-40.f32 40 1.0 0.95 20 10", "specials-40.f32 40 0.7 1.0 70 11", "specials-40.f32 40 0 1.0 20 12",
        "specials-40.f32 40 1.0 0.9 1 13", "vocab-1.f32 1 1.0 0.95 20 14", "vocab-2.f32 2 1.0 0.95 20 15",
        "vocab-2.f32 2 0.5 1.0 1 16", "ordinary-65.f32 65 1.0 0.95 64 17", "ordinary-65.f32 65 1.0 0.95 65 18",
        "ordinary-5000.f32 5000 1.0 0.95 70 19", "ordinary-5000.f32 5000 2.0 1.0 5000 20",
    ]
    with open(os.path.join(HERE, "cases.txt"), "w") as f:
        f.write("\n".join(cases) + "\n")


main()
