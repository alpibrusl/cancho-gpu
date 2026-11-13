#!/usr/bin/env python3
"""D9e of docs/diffusion.md: `dit_cat` (row concatenation of two f16 matrices, an edit's keys and values) against its definition on every element, and two mutants that must fail.

    dit_cat.py <cancho-gpu> [--mutants]
"""
import os
import random
import shutil
import subprocess
import sys
import tempfile

BIN = os.path.abspath(sys.argv[1])
MUTANTS = "--mutants" in sys.argv
CHX_ROOT = os.environ.get("CHX_ROOT", "tests/chx")


def f16r(v):
    import struct
    return struct.unpack("<e", struct.pack("<e", v))[0]


def gate(root):
    s, n, bn = 6, 256, 128
    r = random.Random(4)
    a = [f16r(r.uniform(-8, 8)) for _ in range(s * n)]
    b = [f16r(r.uniform(-8, 8)) for _ in range(s * n)]
    with tempfile.TemporaryDirectory() as tmp:
        args = [BIN, "run", f"{root}/dit_cat.chx", f"s={s}", f"n={n}", f"bn={bn}", "--target", "apple-m-series"]
        for name, vals in (("a", a), ("b", b)):
            path = os.path.join(tmp, name + ".txt")
            open(path, "w").write(" ".join(repr(x) for x in vals))
            args += ["--in", name, path]
        args += ["--out", "y"]
        out = subprocess.run(args, capture_output=True, text=True)
        if out.returncode != 0:
            print("FAIL dit_cat:", out.stderr.strip()[:200])
            return False
        got = [float(x) for x in out.stdout.split()]
    want = a + b
    ok = len(got) == len(want) and all(abs(x - y) <= 1e-6 * max(1.0, abs(y)) for x, y in zip(got, want))  # the interpreter prints 8 significant digits
    print("%s dit_cat: y = [a; b] on every element of %d rows of %d" % ("ok  " if ok else "FAIL", 2 * s, n))
    return ok


if not MUTANTS:
    sys.exit(0 if gate(CHX_ROOT) else 1)
survived = 0
for old, new, what in (("store x2 -> y[s + i, j * bn]", "store x2 -> y[i, j * bn]", "both halves written to the first rows"), ("store x1 -> y[i, j * bn]", "store x1 -> y[s + i, j * bn]", "a and b swapped")):
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(os.path.join("tests/chx", "dit_cat.chx"), tmp)
        text = open(os.path.join(tmp, "dit_cat.chx")).read()
        if old not in text:
            print("INVALID", what)
            survived += 1
            continue
        open(os.path.join(tmp, "dit_cat.chx"), "w").write(text.replace(old, new, 1))
        dead = not gate(tmp)
        print("%s dit_cat: %s" % ("killed  " if dead else "SURVIVED", what))
        survived += 0 if dead else 1
sys.exit(1 if survived else 0)
