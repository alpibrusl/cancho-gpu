#!/usr/bin/env python3
"""docs/two-images.md section 8: `dit_stats_o2` and `dit_pv_mma2`, the softmax statistics and the weighted sum of v over TWO segments of keys (an edit's step: the cached prefix's keys, then the target's), against `dit_stats_o` and `dit_pv_mma` on the same scores laid side by side.

    dit_attention_seg.py <cancho-gpu> [--mutants]

The scores (random, f32), the mask (0 or -1e9, every row with a valid key in each segment) and the f16 values are generated once; the one-segment kernels run on the whole [nh * nq, t1 + t2] matrix, the two-segment ones on its two halves. The maximum must be **exactly** equal, the
normaliser to 1e-6 relative (the chunks are visited in the same order) and the weighted sum to 1e-5 absolute. Two shapes (the segments of unequal length, in both orders). `--mutants`: deliberate changes to a copy of each kernel, which the comparison must fail.
"""
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile

BIN = os.path.abspath(sys.argv[1])
MUTANTS = "--mutants" in sys.argv
CHX_ROOT = os.environ.get("CHX_ROOT", "tests/chx")
NEG = -1e9
D = 128


def f16r(v):
    return struct.unpack("<e", struct.pack("<e", v))[0]


def run(root, kernel, consts, ins, outs, tmp):
    args = [BIN, "run", f"{root}/{kernel}.chx"] + consts + ["--target", "apple-m-series"]
    for name, vals in ins.items():
        path = os.path.join(tmp, f"{kernel}_{name}.txt")
        with open(path, "w") as f:
            f.write(" ".join(repr(float(x)) for x in vals))
        args += ["--in", name, path]
    for o in outs:
        args += ["--out", o]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip()
    return [[float(x) for x in line.split()] for line in r.stdout.strip().split("\n")], ""


def case(root, nh, nq, t1, t2, seed):
    rnd = random.Random(seed)
    t = t1 + t2
    sc = [rnd.gauss(0, 3) for _ in range(nh * nq * t)]  # [nh * nq, t]
    mask = [NEG if rnd.random() < 0.3 else 0.0 for _ in range(nq * t)]
    for q in range(nq):  # a valid key in each segment of every row
        mask[q * t + (q % t1)] = 0.0
        mask[q * t + t1 + (q % t2)] = 0.0
    v = [f16r(rnd.gauss(0, 1)) for _ in range(t * nh * D)]  # [t, nh * d]
    split = lambda a, rows, cols: ([a[r * t + c] for r in range(rows) for c in range(t1)], [a[r * t + c] for r in range(rows) for c in range(t1, t)])
    sc1, sc2 = split(sc, nh * nq, t)
    mk1, mk2 = split(mask, nq, t)
    v1, v2 = v[: t1 * nh * D], v[t1 * nh * D:]
    with tempfile.TemporaryDirectory() as tmp:
        o, err = run(root, "dit_stats_o", [f"nh={nh}", f"nq={nq}", f"t={t}", "qs=16", "cs=64"], {"sc": sc, "mask": mask}, ["m", "z"], tmp)
        if o is None:
            return False, "dit_stats_o: " + err
        m, z = o
        o, err = run(root, "dit_stats_o2", [f"nh={nh}", f"nq={nq}", f"t1={t1}", f"t2={t2}", "qs=16", "cs=64"], {"sc1": sc1, "mask1": mk1, "sc2": sc2, "mask2": mk2}, ["m", "z"], tmp)
        if o is None:
            return False, "dit_stats_o2: " + err
        m2, z2 = o
        dm = max(abs(a - b) for a, b in zip(m, m2))
        dz = max(abs(a - b) / max(abs(b), 1e-30) for a, b in zip(z2, z))
        if len(m2) != len(m) or not (dm == 0.0 and dz <= 1e-6):
            return False, "dit_stats_o2 differs from dit_stats_o: maximum %.3g (tolerance 0), normaliser %.3g relative (1e-6)" % (dm, dz)
        o, err = run(root, "dit_pv_mma", [f"nh={nh}", f"nq={nq}", f"t={t}", f"d={D}"], {"sc": sc, "mask": mask, "m": m, "z": z, "v": v}, ["o"], tmp)
        if o is None:
            return False, "dit_pv_mma: " + err
        want = o[0]
        o, err = run(root, "dit_pv_mma2", [f"nh={nh}", f"nq={nq}", f"t1={t1}", f"t2={t2}", f"d={D}"],
                     {"sc1": sc1, "mask1": mk1, "sc2": sc2, "mask2": mk2, "m": m, "z": z, "v1": v1, "v2": v2}, ["o"], tmp)
        if o is None:
            return False, "dit_pv_mma2: " + err
        got = o[0]
    e = max(abs(a - b) for a, b in zip(got, want))
    return len(got) == len(want) and e <= 1e-5, "nh=%d nq=%d segments %d + %d: statistics equal (maximum exactly, normaliser to %.1g), weighted sum %.3g from the one-segment kernel's (tolerance 1e-5)" % (nh, nq, t1, t2, dz, e)


def gate(root):
    ok = True
    for nh, nq, t1, t2, seed in ((2, 128, 128, 64, 1), (2, 128, 64, 192, 2)):
        good, msg = case(root, nh, nq, t1, t2, seed)
        print(f"{'ok  ' if good else 'FAIL'} {msg}")
        ok &= good
    return ok


if not MUTANTS:
    sys.exit(0 if gate(CHX_ROOT) else 1)

EDITS = [
    ("dit_stats_o2", "let r  = relu (cm - &mx)", "let r  = cm - &mx", "the second segment's maximum free to fall as well as rise"),
    ("dit_stats_o2", "let zs = zz * al", "let zs = zz + al", "the sum so far added to, not multiplied by, the rescaling factor, in the second segment"),
    ("dit_stats_o2", "let mf, zf = for c in 0 .. t2 / cs with mx, zz {", "let mf, zf = for c in 0 .. t2 / cs with m1, z1 {", "the second segment restarted from the first chunk's statistics"),
    ("dit_stats_o2", "load sc2[h * nq + qb * qs, c * cs ; qs, cs]", "load sc2[qb * qs, c * cs ; qs, cs]", "every head of the second segment read from the first head's rows"),
    ("dit_stats_o2", "store zf -> z[h * nq + qb * qs]", "store zz -> z[h * nq + qb * qs]", "the normaliser of the first segment stored"),
    ("dit_pv_mma2", "load v2[p * bk, h * d + j * bn ; bk, bn] @shared", "load v2[p * bk, j * bn ; bk, bn] @shared", "the second segment's v always read from the first head"),
    ("dit_pv_mma2", "load mask2[i * bm, p * bk ; bm, bk]", "load mask2[0, p * bk ; bm, bk]", "the second segment's mask always read from the first query rows"),
    ("dit_pv_mma2", "load sc2[h * nq + i * bm, p * bk ; bm, bk]", "load sc2[i * bm, p * bk ; bm, bk]", "every head's scores of the second segment read from the first head"),
    ("dit_pv_mma2", "let acc = for p in 0 .. t2 / bk with acc {", "let acc = for p in 0 .. t2 / bk with acc0 {", "the second segment accumulated from zero, dropping the first"),
]
survived = 0
for kernel, old, new, what in EDITS:
    with tempfile.TemporaryDirectory() as tmp:
        for f in os.listdir("tests/chx"):
            if f.endswith(".chx") and f.startswith("dit_"):
                shutil.copy(os.path.join("tests/chx", f), tmp)
        path = os.path.join(tmp, kernel + ".chx")
        text = open(path).read()
        if old not in text:
            print(f"INVALID {kernel}: '{old}' not found")
            survived += 1
            continue
        open(path, "w").write(text.replace(old, new, 1))
        dead = not gate(tmp)
        print(f"{'killed  ' if dead else 'SURVIVED'} {kernel}: {what}")
        survived += 0 if dead else 1
sys.exit(1 if survived else 0)
