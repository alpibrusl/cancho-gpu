#!/usr/bin/env python3
"""D1 of docs/diffusion.md: block-causal attention for a diffusion transformer (head dimension 128, N queries against N keys),
as three kernels chained through `cancho-gpu run`, against a float64 reference written from the mathematics, **every element**.

    dit_attention.py <cancho-gpu> [--mutants]

The kernels are `dit_scores.chx` (the scores of a tile of queries against a tile of keys, scaled by 1/sqrt(128), plus an additive mask),
`dit_stats.chx` (the row maximum and normaliser over chunks of keys) and `dit_pv.chx` (the weighted sum of v). Chained they are
softmax(q k^T / sqrt(128) + mask) v for every head, with the queries tiled as well as the keys (the kernels of the decode path tile only the keys).

The mask is built here from the definition in diffusers' `build_qwenimage21_block_causal_mask` (transformer_qwenimage21.py): position q may attend to
kv when `q >= kv` or both are in the same image block (an image id >= 0), and kv is a valid key. The additive form is 0 for allowed and -1e9 for not.
The reference does not use the kernels' decomposition: it computes the whole row's softmax at once, in binary64.

Cases: 2 heads, 32 positions, 8-row query tiles (text only, causal); 3 heads, 48 positions of 5 text + two image blocks (block-causal) with two
padded keys; one case where the row maximum is in a late chunk. Tolerance 2e-6 on every element (the kernels sum in f32).

`--mutants` makes one deliberate change to a copy of each kernel and requires the gate to fail (`CHX_ROOT` points at the copy): the scale of 1/16 for
1/sqrt(128), a mask read at the wrong row, a running maximum that keeps the first chunk's, a normaliser that skips the last chunk, a value tile read
from the wrong head.
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
NEG = -1e9
TOL = 2e-6


def block_causal_mask(image_ids, key_valid):
    n = len(image_ids)
    m = []
    for q in range(n):
        for kv in range(n):
            same = image_ids[q] == image_ids[kv] and image_ids[q] >= 0
            ok = (q >= kv or same) and key_valid[kv]
            m.append(0.0 if ok else NEG)
    return m


def reference(nh, n, d, q, k, v, mask):
    out = []
    scale = 1.0 / math.sqrt(d)
    for h in range(nh):
        for i in range(n):
            s = [sum(q[(h * n + i) * d + c] * k[(h * n + j) * d + c] for c in range(d)) * scale + mask[i * n + j] for j in range(n)]
            mx = max(s)
            e = [math.exp(x - mx) for x in s]
            z = sum(e)
            for col in range(d):
                out.append(sum(e[j] / z * v[(h * n + j) * d + col] for j in range(n)))
    return out


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
    lines = r.stdout.strip().split("\n")
    return [[float(x) for x in line.split()] for line in lines], ""


def case(root, name, nh, n, d, qs, ts, cs, dw, image_ids, key_valid, late_max=False):
    rnd = random.Random(7 + n + nh)
    q = [rnd.uniform(-1, 1) for _ in range(nh * n * d)]
    k = [rnd.uniform(-1, 1) for _ in range(nh * n * d)]
    v = [rnd.uniform(-1, 1) for _ in range(nh * n * d)]
    if late_max:  # the largest score of row 0 is in the last chunk
        for c in range(d):
            k[(n - 1) * d + c] = q[c] * 3.0
    mask = block_causal_mask(image_ids, key_valid)
    want = reference(nh, n, d, q, k, v, mask)
    with tempfile.TemporaryDirectory() as tmp:
        sc, err = run(root, "dit_scores", [f"nh={nh}", f"nq={n}", f"t={n}", f"qs={qs}", f"ts={ts}", f"d={d}"], {"q": q, "k": k, "mask": mask}, ["sc"], tmp)
        if sc is None:
            return False, f"dit_scores: {err}"
        mz, err = run(root, "dit_stats", [f"nh={nh}", f"nq={n}", f"t={n}", f"qs={qs}", f"cs={cs}"], {"sc": sc[0]}, ["m", "z"], tmp)
        if mz is None:
            return False, f"dit_stats: {err}"
        o, err = run(root, "dit_pv", [f"nh={nh}", f"nq={n}", f"t={n}", f"qs={qs}", f"cs={cs}", f"dw={dw}", f"d={d}"], {"sc": sc[0], "m": mz[0], "z": mz[1], "v": v}, ["o"], tmp)
        if o is None:
            return False, f"dit_pv: {err}"
    got = o[0]
    if len(got) != len(want):
        return False, f"{len(got)} outputs, expected {len(want)}"
    worst = max(abs(a - b) for a, b in zip(got, want))
    return worst <= TOL, f"max abs error {worst:.3g} over {len(want)} elements"


def gate(root):
    ok = True
    cases = [
        ("text only, causal: nh=2 n=32 qs=8", 2, 32, 128, 8, 16, 16, 32, [-1] * 32, [True] * 32, False),
        ("text 5 + two image blocks, 2 padded keys: nh=3 n=48", 3, 48, 128, 16, 16, 16, 64, [-1] * 5 + [0] * 20 + [1] * 23, [True] * 3 + [False] * 2 + [True] * 43, False),
        ("row maximum in the last chunk: nh=2 n=64", 2, 64, 128, 16, 32, 16, 32, [-1] * 4 + [0] * 60, [True] * 64, True),
    ]
    for name, nh, n, d, qs, ts, cs, dw, ids, kv, late in cases:
        good, msg = case(root, name, nh, n, d, qs, ts, cs, dw, ids, kv, late)
        print(f"{'ok  ' if good else 'FAIL'} {name}: {msg}")
        ok &= good
    return ok


if not MUTANTS:
    sys.exit(0 if gate(CHX_ROOT) else 1)

# mutants: (kernel, old text, new text, what)
EDITS = [
    ("dit_scores", "s * 0.08838834764831845", "s * 0.0625", "the scale is 1/16, not 1/sqrt(128)"),
    ("dit_scores", "load mask[qb * qs, sp * ts ; qs, ts]", "load mask[sp * ts, qb * qs ; qs, ts]", "the mask is read transposed"),
    ("dit_stats", "yield m1 + relu (cm - &m1)", "yield m1", "the running maximum keeps the first chunk's"),
    ("dit_stats", "for c in 0 .. t / cs with z0", "for c in 0 .. t / cs - 1 with z0", "the normaliser skips the last chunk"),
    ("dit_pv", "load v[h * t + c * cs, dc * dw ; cs, dw]", "load v[c * cs, dc * dw ; cs, dw]", "v is read from the first head's rows"),
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
