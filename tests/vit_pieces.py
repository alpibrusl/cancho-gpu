#!/usr/bin/env python3
"""D9b of docs/diffusion.md: the pieces of the Qwen3-VL vision tower that the text-to-image kernels did not have, each against a float64 reference written from the mathematics, every element.

    vit_pieces.py <cancho-gpu> [--mutants]

* `vit_ln`            LayerNorm with weight and bias (eps 1e-6, biased variance), f16 out, at n = 64 and at the tower's n = 1,152 and 4,608;
* `vit_bias_gelu`     gelu_tanh(x + b), f16 out (the blocks' MLP);
* `vit_bias_gelu_erf` the exact gelu(x + b) = 0.5 v (1 + erf(v / sqrt 2)) through Abramowitz and Stegun's erf, f16 out (the mergers), over -9 to 9 (erf saturates) and checked at the error of the approximation, not at f16 only.

Mutants: one deliberate change to a copy of each kernel, which the gate must fail.
"""
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile

BIN = os.path.abspath(sys.argv[1])
MUTANTS = "--mutants" in sys.argv
CHX_ROOT = os.environ.get("CHX_ROOT", "tests/chx")


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


def close_f16(got, want):  # an f16 result: half a unit in the last place of f16 (4.9e-4 relative) and a floor of 1e-3 for values near zero (as the other piece tests)
    worst = max(abs(a - b) / (4.9e-4 * abs(b) + 1e-3) for a, b in zip(got, want))
    return len(got) == len(want) and worst <= 1.0, worst


def rnd(seed, n, lo=-1.0, hi=1.0):
    r = random.Random(seed)
    return [r.uniform(lo, hi) for _ in range(n)]


def t_ln(root, tmp, s, n, seed):
    x = rnd(seed, s * n, -3, 3)
    for j in range(n):
        x[0 * n + j] = 20.0 + x[0 * n + j] * 0.1  # a mean 70 times the spread: the variance is taken about the mean (an offset of 1000 over 0.017 is a stress of f32 rowsum itself, not of this kernel: 0.4% off, as torch float32 would be)
        x[1 * n + j] = 2.5  # a constant row: variance 0, eps decides, the result is the bias
    w, b = rnd(seed + 1, n, 0.5, 1.5), rnd(seed + 2, n)
    want = []
    for i in range(s):
        row = x[i * n:(i + 1) * n]
        mu = sum(row) / n
        var = sum((v - mu) ** 2 for v in row) / n
        want += [(v - mu) / math.sqrt(var + 1e-6) * w[j] + b[j] for j, v in enumerate(row)]
    o, err = run(root, "vit_ln", [f"s={s}", f"n={n}"], {"x": x, "w": w, "b": b}, ["y"], tmp)
    return (False, err) if o is None else close_f16(o[0], want)


def gelu_tanh(v):
    return 0.5 * v * (1 + math.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v ** 3)))


def gelu_erf(v):
    return 0.5 * v * (1 + math.erf(v / math.sqrt(2)))


def t_gelu(root, tmp, kernel, fn, extra):
    s, n = 8, 128
    x, b = rnd(40, s * n, -9, 9), rnd(41, n, -1, 1)
    for j in range(n):
        x[j] = 0.0  # a row of zeros plus the bias: gelu(b)
        x[n + j] = -b[j]  # exactly zero after the bias
    want = [fn(x[i * n + j] + b[j]) for i in range(s) for j in range(n)]
    o, err = run(root, kernel, [f"s={s}", f"n={n}", "bn=128"] + extra, {"x": x, "b": b}, ["y"], tmp)
    return (False, err) if o is None else close_f16(o[0], want)


TESTS = [
    ("vit_ln n=64", lambda r, t: t_ln(r, t, 4, 64, 1)),
    ("vit_ln n=1152", lambda r, t: t_ln(r, t, 3, 1152, 2)),
    ("vit_ln n=4608", lambda r, t: t_ln(r, t, 3, 4608, 3)),
    ("vit_bias_gelu", lambda r, t: t_gelu(r, t, "vit_bias_gelu", gelu_tanh, [])),
    ("vit_bias_gelu_erf", lambda r, t: t_gelu(r, t, "vit_bias_gelu_erf", gelu_erf, ["m1=-1"])),
]


def gate(root):
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in TESTS:
            good, msg = fn(root, tmp)
            shown = msg if isinstance(msg, str) else f"worst error {msg:.3g} of its tolerance"
            print(f"{'ok  ' if good else 'FAIL'} {name}: {shown}")
            ok &= good
    return ok


if not MUTANTS:
    sys.exit(0 if gate(CHX_ROOT) else 1)

EDITS = [
    ("vit_ln", "let t  = vr + 0.000001", "let t  = vr + 0.01", "an eps of 0.01"),
    ("vit_ln", "let o2 = o1 + bb", "let o2 = o1", "no bias"),
    ("vit_ln", "let o1 = xn * ww", "let o1 = xn", "no weight"),
    ("vit_ln", "let vr = vs * (1 / n)", "let vr = vs * (1 / (n - 1))", "the unbiased variance"),
    ("vit_bias_gelu", "let vv = xx + bb", "let vv = xx", "no bias"),
    ("vit_bias_gelu", "let t1 = v3 * 0.044715", "let t1 = v3 * 0.04", "a wrong cubic coefficient"),
    ("vit_bias_gelu_erf", "let z  = &vv * 0.7071067811865476", "let z  = &vv * 0.5", "z = v / 2 instead of v / sqrt 2"),
    ("vit_bias_gelu_erf", "let sg = s1 - 1", "let sg = s1", "the sign of erf lost (always +)"),
    ("vit_bias_gelu_erf", "let cd = cc + 0.254829592", "let cd = cc + 0.25", "a coefficient of the erf polynomial truncated"),
    ("vit_bias_gelu_erf", "let g1 = g0 * 0.5", "let g1 = g0", "no 0.5"),
]
survived = 0
for kernel, old, new, what in EDITS:
    with tempfile.TemporaryDirectory() as tmp:
        for f in os.listdir("tests/chx"):
            if f.endswith(".chx") and f.startswith("vit_"):
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
