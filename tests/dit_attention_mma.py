#!/usr/bin/env python3
"""D8 of docs/diffusion.md: the attention of the diffusion transformer on the matrix unit, in f16 (D1's kernels ran on the generic f32 product at 0.64 TFLOPS against 12 to 13 for `mma`; at 1024 x 1024 they are 66% of a step).

    dit_attention_mma.py <cancho-gpu> [--mutants]

Four kernels chained through `cancho-gpu run`: `dit_rope16` (the rotary embedding of dit_rope.chx, times a constant, written as f16: q gets 1/sqrt(128), k gets 1), `dit_scores_mma` (q k^T on the matrix unit from f16 q and k, f32 accumulation, the scores stored in f32),
`dit_stats_m` (the row maximum and normaliser of scores + mask; D10 adds the same in one pass, `dit_stats_c` and `dit_stats_comb`, which the weighted sum is given and which are held to `dit_stats_m`'s) and `dit_pv_mma` (the weights exp(s + mask - m) / z rounded to f16 and multiplied by f16 v on the matrix unit). v is rounded to f16 by `dit_to_f16.chx`. The mask is added where it is read (stats and the product), since a matrix-unit accumulator cannot be touched after the product.

Two references, in binary64, **every element**:
 1. the same arithmetic with the f16 roundings the kernels make (q * scale after the rotation, k, v, and the weights p), whose difference from the kernels is the f32 accumulation and the rounding ties: tolerance 1e-4 absolute (the outputs are of the order of 0.3);
 2. the exact attention with none of them, which bounds what the f16 costs: tolerance 5e-3 absolute.
Cases: 2 heads of 128 positions (text only, causal); 3 heads of 192 (5 text, two image blocks, padded keys). `--mutants`: one deliberate change to a copy of each kernel, which gate 1 must fail.
"""
import math
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
SCALE = 1.0 / math.sqrt(D)


def f16r(v):
    try:
        return struct.unpack("<e", struct.pack("<e", v))[0]
    except OverflowError:
        return math.copysign(65504.0, v)


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


def block_causal(ids, valid):
    n = len(ids)
    return [0.0 if ((q >= kv) or (ids[q] == ids[kv] and ids[q] >= 0)) and valid[kv] else NEG for q in range(n) for kv in range(n)]


def rope_ref(x, cs, sn, n, nh):  # x [n * nh * 128], rotate-half per head, tables [n, 64]
    out = [0.0] * len(x)
    for t in range(n):
        for h in range(nh):
            b = (t * nh + h) * D
            for j in range(64):
                a, c = x[b + j], x[b + 64 + j]
                out[b + j] = a * cs[t * 64 + j] - c * sn[t * 64 + j]
                out[b + 64 + j] = c * cs[t * 64 + j] + a * sn[t * 64 + j]
    return out


def attention_ref(nh, n, q, k, v, mask, f16):
    """q, k, v after the rotation, token-major [n * nh * 128]."""
    if f16:
        q = [f16r(a * SCALE) for a in q]
        k = [f16r(a) for a in k]
        v = [f16r(a) for a in v]
        scale = 1.0
    else:
        scale = SCALE
    out = [0.0] * (n * nh * D)
    for h in range(nh):
        for i in range(n):
            s = [sum(q[(i * nh + h) * D + c] * k[(j * nh + h) * D + c] for c in range(D)) * scale + mask[i * n + j] for j in range(n)]
            mx = max(s)
            e = [math.exp(x - mx) for x in s]
            z = sum(e)
            p = [f16r(x / z) if f16 else x / z for x in e]
            for col in range(D):
                out[(i * nh + h) * D + col] = sum(p[j] * v[(j * nh + h) * D + col] for j in range(n))
    return out


def case(root, name, nh, n, ids, valid, seed):
    rnd = random.Random(seed)
    q = [rnd.uniform(-1, 1) for _ in range(n * nh * D)]
    k = [rnd.uniform(-1, 1) for _ in range(n * nh * D)]
    v = [rnd.uniform(-1, 1) for _ in range(n * nh * D)]
    ang = [rnd.uniform(-3, 3) for _ in range(n * 64)]
    cs, sn = [math.cos(a) for a in ang], [math.sin(a) for a in ang]
    mask = block_causal(ids, valid)
    qr, kr = rope_ref(q, cs, sn, n, nh), rope_ref(k, cs, sn, n, nh)
    want1, want2 = attention_ref(nh, n, qr, kr, v, mask, True), attention_ref(nh, n, qr, kr, v, mask, False)
    with tempfile.TemporaryDirectory() as tmp:
        o, err = run(root, "dit_rope16", [f"s={n}", f"nh={nh}", "h=64", f"sc={SCALE!r}"], {"x": q, "cs": cs, "sn": sn}, ["y"], tmp)
        if o is None:
            return False, "dit_rope16 q: " + err
        q16 = o[0]
        o, err = run(root, "dit_rope16", [f"s={n}", f"nh={nh}", "h=64", "sc=1.0"], {"x": k, "cs": cs, "sn": sn}, ["y"], tmp)
        if o is None:
            return False, "dit_rope16 k: " + err
        k16 = o[0]
        o, err = run(root, "dit_to_f16", [f"s={n}", f"n={nh * D}"], {"x": v}, ["y"], tmp)
        if o is None:
            return False, "dit_to_f16: " + err
        v16 = o[0]
        o, err = run(root, "dit_scores_mma", [f"nh={nh}", f"nq={n}", f"t={n}", f"d={D}"], {"q": q16, "k": k16}, ["sc"], tmp)
        if o is None:
            return False, "dit_scores_mma: " + err
        sc = o[0]
        o, err = run(root, "dit_stats_m", [f"nh={nh}", f"nq={n}", f"t={n}", "qs=32", "cs=64"], {"sc": sc, "mask": mask}, ["m", "z"], tmp)
        if o is None:
            return False, "dit_stats_m: " + err
        m_two, z_two = o
        # the same statistics in one pass over the scores (D10): the pipeline's pair of kernels, and what the weighted sum below is given
        o, err = run(root, "dit_stats_c", [f"nh={nh}", f"nq={n}", f"t={n}", "qs=16", "cs=64"], {"sc": sc, "mask": mask}, ["mc", "zc"], tmp)
        if o is None:
            return False, "dit_stats_c: " + err
        mc, zc = o
        o, err = run(root, "dit_stats_comb", [f"nh={nh}", f"nq={n}", f"nc={n // 64}", "qs=32"], {"mc": mc, "zc": zc}, ["m", "z"], tmp)
        if o is None:
            return False, "dit_stats_comb: " + err
        m, z = o
        # the one-pass statistics against the two-pass ones: the maximum exactly, the normaliser to the order of the f32 sums
        dm = max(abs(a - b) for a, b in zip(m, m_two))
        dz = max(abs(a - b) / max(abs(b), 1e-30) for a, b in zip(z, z_two))
        if dm != 0.0 or dz > 1e-5:
            return False, "the one-pass statistics differ from the two-pass ones: maximum %.3g, normaliser %.3g (relative; tolerance 0 and 1e-5)" % (dm, dz)
        # and the online version, one kernel carrying the maximum and the normaliser together (D14), held to the same
        o, err = run(root, "dit_stats_o", [f"nh={nh}", f"nq={n}", f"t={n}", "qs=16", "cs=64"], {"sc": sc, "mask": mask}, ["m", "z"], tmp)
        if o is None:
            return False, "dit_stats_o: " + err
        dm = max(abs(a - b) for a, b in zip(o[0], m_two))
        dz = max(abs(a - b) / max(abs(b), 1e-30) for a, b in zip(o[1], z_two))
        if dm != 0.0 or dz > 1e-5:
            return False, "the online statistics differ from the two-pass ones: maximum %.3g, normaliser %.3g (relative; tolerance 0 and 1e-5)" % (dm, dz)
        o, err = run(root, "dit_pv_mma", [f"nh={nh}", f"nq={n}", f"t={n}", f"d={D}"], {"sc": sc, "mask": mask, "m": m, "z": z, "v": v16}, ["o"], tmp)
        if o is None:
            return False, "dit_pv_mma: " + err
    got = o[0]
    e1 = max(abs(a - b) for a, b in zip(got, want1))
    e2 = max(abs(a - b) for a, b in zip(got, want2))
    return len(got) == len(want1) and e1 <= 1e-4 and e2 <= 5e-3, "max abs error %.3g against the f16-rounded reference (tolerance 1e-4), %.3g against the exact one (5e-3), over %d elements" % (e1, e2, len(want1))


def gate(root):
    ok = True
    for name, nh, n, ids, valid, seed in (("text only, causal: nh=2 n=128", 2, 128, [-1] * 128, [True] * 128, 1),
                                          ("text 5 + two image blocks, padded keys: nh=3 n=192", 3, 192, [-1] * 5 + [0] * 80 + [1] * 107, [True] * 3 + [False] * 2 + [True] * 187, 2)):
        good, msg = case(root, name, nh, n, ids, valid, seed)
        print(f"{'ok  ' if good else 'FAIL'} {name}: {msg}")
        ok &= good
    return ok


if not MUTANTS:
    sys.exit(0 if gate(CHX_ROOT) else 1)

EDITS = [
    ("dit_rope16", "let hi  = d + e", "let hi  = d - e", "the sign of the second half"),
    ("dit_rope16", "let lo2 = lo * sc", "let lo2 = lo * 1", "no scale on the first half"),
    ("dit_scores_mma", "load k[j * bn, h * d + p * bk ; bn, bk]", "load k[j * bn, p * bk ; bn, bk]", "k always read from the first head"),
    ("dit_scores_mma", "store acc -> sc[h * nq + i * bm, j * bn]", "store acc -> sc[i * bm, j * bn]", "every head written to the first head's rows"),
    ("dit_stats_m", "let s0 = f0 + k0", "let s0 = f0", "no mask in the first chunk's maximum"),
    ("dit_stats_m", "let sh = s2 - &mx", "let sh = sm - &mx", "no mask in the normaliser"),
    ("dit_stats_c", "let s1 = s0 + k0", "let s1 = s0 - k0", "the mask subtracted in the chunks' statistics"),
    ("dit_stats_c", "let sh = s1 - &cm", "let sh = s1", "the chunk's own maximum not subtracted"),
    ("dit_stats_c", "store cz -> zc[c * nh * nq + h * nq + qb * qs]", "store cz -> zc[h * nq + qb * qs]", "every chunk's sum written to the first chunk's place"),
    ("dit_stats_o", "let zs = z1 * al", "let zs = z1 + al", "the sum so far added to, not multiplied by, the rescaling factor"),
    ("dit_stats_o", "yield mn, zs + cz", "yield zs + cz, mn", "the two carries swapped"),
    ("dit_stats_o", "let r  = relu (cm - &m1)", "let r  = cm - &m1", "the maximum free to fall as well as rise"),
    ("dit_stats_o", "let s2 = sm + mk", "let s2 = sm - mk", "the mask subtracted after the first chunk"),
    ("dit_stats_comb", "let d  = cm - &mx", "let d  = cm + &mx", "a chunk's sum rescaled by the wrong sign"),
    ("dit_stats_comb", "yield m1 + relu (cm - &m1)", "yield m1 + (cm - &m1)", "the maximum free to fall as well as rise"),
    ("dit_pv_mma", "let s2 = sm + mk", "let s2 = sm + sm", "the mask replaced by the scores"),
    ("dit_pv_mma", "let ph = stage f16 pp", "let ph = stage f16 e", "the weights not divided by the normaliser"),
    ("dit_pv_mma", "load v[p * bk, h * d + j * bn ; bk, bn] @shared", "load v[p * bk, j * bn ; bk, bn] @shared", "v always read from the first head"),
]
survived = 0
for kernel, old, new, what in EDITS:
    with tempfile.TemporaryDirectory() as tmp:
        for f in os.listdir("tests/chx"):
            if f.endswith(".chx") and (f.startswith("dit_") or f.startswith("conv")):
                shutil.copy(os.path.join("tests/chx", f), tmp)
        shutil.copytree("tests/chx/ported", os.path.join(tmp, "ported"))
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
