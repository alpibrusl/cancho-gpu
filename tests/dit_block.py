#!/usr/bin/env python3
"""D2b of docs/diffusion.md: the pieces of a Qwen-Image-2.1 transformer block that are not attention and not a matrix product, as kernels, each against a float64
reference written from the mathematics of the block (tools/dit_block_ref.py has the whole block), every element.

    dit_block.py <cancho-gpu> [--mutants]

* `dit_lnmod.chx`   y = layernorm(x) * (1 + scale), no affine, eps 1e-6; the scale row is the target image's (row 0) or the t = 0 one (row 1) by a per-token selector; y is f16 (the input of the next matrix product).
* `dit_rope.chx`    the rotate-half form of the rotary embedding with per-position tables [s, 64], on rows of 128 (a head each, nh to a token).
* `rmsnorm_rows.chx` (existing) the per-head RMSNorm of q and k with a [1, 128] weight, eps 1e-6.
* `dit_gate_res.chx` x + tanh(gate) * y, the gate row chosen as above.
* `dit_swiglu.chx`  silu(g) * u, written as f16 (the input of the output projection).
* `dit_to_f16.chx`  f32 to f16 (the attention output, before the output projection).
* D2e: `dit_gelu` (tanh approximation, f16), `dit_silu` (f16), `dit_add`, `dit_modsplit` (the shared modulation's four blocks of [2, n]), `dit_rows2`.
And the host's plan, as a pure-Python check (no kernel): the model rotates *interleaved* pairs (2j, 2j + 1) of a head; the kernel rotates halves (j, j + 64). Permuting the output
channels of to_q and to_k (and the norm weights) within each head, even channels first, makes the two the same up to that one permutation, which leaves q . k and the RMSNorm
unchanged. `permutation` below checks that on random data.

f16 outputs are compared with a tolerance of half a step of f16 (relative 4.9e-4) plus 1e-3 absolute; f32 outputs with 2e-6 relative to the larger of 1 and the value.
`--mutants` makes one deliberate change to a copy of each kernel and requires the gate to fail.
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


def close(got, want, f16):
    worst = 0.0
    for a, b in zip(got, want):
        err = abs(a - b)
        tol = (4.9e-4 * abs(b) + 1e-3) if f16 else 2e-6 * max(1.0, abs(b))
        worst = max(worst, err / tol)
    return len(got) == len(want) and worst <= 1.0, worst


def rnd(seed, n, lo=-1.0, hi=1.0):
    r = random.Random(seed)
    return [r.uniform(lo, hi) for _ in range(n)]


def selector(s):
    return [1.0 if i >= s // 4 else 0.0 for i in range(s)]  # the first quarter are text tokens, the rest the target image


def pick(sel, row0, row1, i, j):
    return row0[j] if sel[i] > 0.5 else row1[j]


def t_lnmod(root, tmp):
    s, n = 12, 256
    x = rnd(1, s * n, -2, 2)
    x[3 * n + 17] = 2300.0  # a massive activation, as block 0 has
    for j in range(n):  # a row whose variance (about 5e-7) is comparable to eps, so that eps is tested
        x[5 * n + j] *= 1e-3
    sc = rnd(2, 2 * n, -0.5, 0.5)
    sel = selector(s)
    want = []
    for i in range(s):
        row = x[i * n:(i + 1) * n]
        mu = sum(row) / n
        var = sum((v - mu) ** 2 for v in row) / n
        for j in range(n):
            want.append((row[j] - mu) / math.sqrt(var + 1e-6) * (1 + pick(sel, sc[:n], sc[n:], i, j)))
    o, err = run(root, "dit_lnmod", [f"s={s}", f"n={n}"], {"x": x, "sc": sc, "sel": sel}, ["y"], tmp)
    return (False, err) if o is None else close(o[0], want, True)


def t_rope(root, tmp):
    s, nh, h = 6, 3, 64
    x = rnd(3, s * nh * 2 * h)
    ang = rnd(4, s * h, -3, 3)
    cs, sn = [math.cos(a) for a in ang], [math.sin(a) for a in ang]
    want = []
    for t in range(s):
        for hd in range(nh):
            b = (t * nh + hd) * 2 * h
            want += [x[b + j] * cs[t * h + j] - x[b + h + j] * sn[t * h + j] for j in range(h)]
            want += [x[b + h + j] * cs[t * h + j] + x[b + j] * sn[t * h + j] for j in range(h)]
    o, err = run(root, "dit_rope", [f"s={s}", f"nh={nh}", f"h={h}"], {"x": x, "cs": cs, "sn": sn}, ["y"], tmp)
    return (False, err) if o is None else close(o[0], want, False)


def t_rmsnorm(root, tmp):
    r, n = 24, 128
    x = rnd(5, r * n, -3, 3)
    w = rnd(6, n, 0.5, 1.5)
    want = []
    for i in range(r):
        row = x[i * n:(i + 1) * n]
        rr = 1 / math.sqrt(sum(v * v for v in row) / n + 1e-6)
        want += [row[j] * rr * w[j] for j in range(n)]
    o, err = run(root, "rmsnorm_rows", [f"r={r}", f"n={n}", "eps=0.000001"], {"x": x, "w": w}, ["y"], tmp)
    return (False, err) if o is None else close(o[0], want, False)


def t_gate(root, tmp):
    s, n = 12, 256
    x = rnd(7, s * n, -2, 2)
    y = rnd(8, s * n, -3, 3)
    g = rnd(9, 2 * n, -2, 2)
    sel = selector(s)
    want = [x[i * n + j] + math.tanh(pick(sel, g[:n], g[n:], i, j)) * y[i * n + j] for i in range(s) for j in range(n)]
    o, err = run(root, "dit_gate_res", [f"s={s}", f"n={n}"], {"x": x, "y": y, "g": g, "sel": sel}, ["z"], tmp)
    return (False, err) if o is None else close(o[0], want, False)


def t_swiglu(root, tmp):
    s, n = 6, 384
    g = rnd(10, s * n, -4, 4)
    u = rnd(11, s * n, -2, 2)
    want = [v / (1 + math.exp(-v)) * w for v, w in zip(g, u)]
    o, err = run(root, "dit_swiglu", [f"s={s}", f"n={n}", "bn=128"], {"g": g, "u": u}, ["y"], tmp)
    return (False, err) if o is None else close(o[0], want, True)


def t_cast(root, tmp):
    s, n = 5, 256
    x = rnd(16, s * n, -100, 100)
    o, err = run(root, "dit_to_f16", [f"s={s}", f"n={n}"], {"x": x}, ["y"], tmp)
    return (False, err) if o is None else close(o[0], x, True)


def t_gelu(root, tmp):
    s, n = 4, 256
    x = rnd(17, s * n, -6, 6)
    want = [0.5 * v * (1 + math.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v ** 3))) for v in x]
    o, err = run(root, "dit_gelu", [f"s={s}", f"n={n}", "bn=128"], {"x": x}, ["y"], tmp)
    return (False, err) if o is None else close(o[0], want, True)


def t_silu(root, tmp):
    s, n = 4, 256
    x = rnd(18, s * n, -8, 8)
    want = [v / (1 + math.exp(-v)) for v in x]
    o, err = run(root, "dit_silu", [f"s={s}", f"n={n}", "bn=128"], {"x": x}, ["y"], tmp)
    return (False, err) if o is None else close(o[0], want, True)


def t_add(root, tmp):
    s, n = 4, 256
    a, b = rnd(19, s * n, -5, 5), rnd(20, s * n, -5, 5)
    o, err = run(root, "dit_add", [f"s={s}", f"n={n}", "bn=128"], {"a": a, "b": b}, ["z"], tmp)
    return (False, err) if o is None else close(o[0], [u + v for u, v in zip(a, b)], False)


def t_modsplit(root, tmp):
    m, n = 64, 256
    md = rnd(21, m * 4 * n, -3, 3)
    want = [[md[r * 4 * n + c * n + j] for r in range(2) for j in range(n)] for c in range(4)]
    o, err = run(root, "dit_modsplit", [f"m={m}", f"n={n}", "bn=128"], {"md": md}, ["s1", "g1", "s2", "g2"], tmp)
    if o is None:
        return False, err
    results = [close(o[c], want[c], False) for c in range(4)]
    return all(r[0] for r in results), max(r[1] for r in results)


def t_rows2(root, tmp):
    m, n = 64, 256
    x = rnd(22, m * n, -3, 3)
    o, err = run(root, "dit_rows2", [f"m={m}", f"n={n}", "bn=128"], {"x": x}, ["y"], tmp)
    return (False, err) if o is None else close(o[0], x[:2 * n], False)


def permutation(bad=False):
    """The host plan: interleaved RoPE on the model's channels == rotate-half RoPE on channels permuted even-first; q . k and the RMSNorm are unchanged."""
    hd, h = 128, 64
    q, k, w = rnd(12, hd), rnd(13, hd), rnd(14, hd, 0.5, 1.5)
    ang = rnd(15, h, -3, 3)
    c, s_ = [math.cos(a) for a in ang], [math.sin(a) for a in ang]
    perm = list(range(hd)) if bad else list(range(0, hd, 2)) + list(range(1, hd, 2))

    def norm(v, wt):
        r = 1 / math.sqrt(sum(a * a for a in v) / len(v) + 1e-6)
        return [a * r * b for a, b in zip(v, wt)]

    def inter(v):  # the model: pairs (2j, 2j + 1) as complex numbers times (cos + i sin)
        out = [0.0] * hd
        for j in range(h):
            out[2 * j] = v[2 * j] * c[j] - v[2 * j + 1] * s_[j]
            out[2 * j + 1] = v[2 * j] * s_[j] + v[2 * j + 1] * c[j]
        return out

    def halves(v):  # the kernel
        return [v[j] * c[j] - v[h + j] * s_[j] for j in range(h)] + [v[h + j] * c[j] + v[j] * s_[j] for j in range(h)]

    qm, km = inter(norm(q, w)), inter(norm(k, w))
    qp, kp = halves(norm([q[i] for i in perm], [w[i] for i in perm])), halves(norm([k[i] for i in perm], [w[i] for i in perm]))
    dot_m = sum(a * b for a, b in zip(qm, km))
    dot_p = sum(a * b for a, b in zip(qp, kp))
    same_q = max(abs(qp[i] - qm[perm[i]]) for i in range(hd))
    ok = abs(dot_m - dot_p) < 1e-9 and same_q < 1e-9
    return ok, f"q.k {dot_m:.12g} vs {dot_p:.12g}, rotated q differs by {same_q:.2g} after the permutation"


TESTS = [("dit_lnmod (with an outlier of 2300)", t_lnmod), ("dit_rope, rotate-half with per-position tables", t_rope), ("rmsnorm_rows as the per-head norm", t_rmsnorm),
         ("dit_gate_res", t_gate), ("dit_swiglu", t_swiglu), ("dit_to_f16", t_cast), ("dit_gelu (tanh approximation)", t_gelu), ("dit_silu", t_silu), ("dit_add", t_add),
         ("dit_modsplit (four [2, n] blocks of a [64, 4n] product)", t_modsplit), ("dit_rows2", t_rows2)]


def gate(root):
    ok = True
    good, msg = permutation()
    print(f"{'ok  ' if good else 'FAIL'} the host's channel permutation: {msg}")
    ok &= good
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
    ("dit_lnmod", "let g  = sc + 1", "let g  = sc + 0", "the scale is used as scale, not 1 + scale"),
    ("dit_lnmod", "let t  = vr + 0.000001", "let t  = vr + 0.0001", "eps 1e-4 for 1e-6"),
    ("dit_lnmod", "let a  = df * sl", "let a  = df * 0.5", "the selector is ignored: halfway between the two modulation rows"),
    ("dit_rope", "let hi = d + e", "let hi = d - e", "the sign of the second half"),
    ("dit_rope", "let c  = load cs[t, 0 ; 1, h]", "let c  = load cs[0, 0 ; 1, h]", "the tables of position 0 for every token"),
    ("dit_gate_res", "let th = tanh gt", "let th = gt", "no tanh on the gate"),
    ("dit_gate_res", "let a  = df * sl", "let a  = df * 0.5", "the selector is ignored in the gate"),
    ("dit_swiglu", "let sg = sigmoid &gt", "let sg = sigmoid &ut", "the sigmoid of u, not of g"),
    ("dit_to_f16", "load x[i, 0 ; 1, n]", "load x[0, 0 ; 1, n]", "row 0 for every row"),
    ("dit_gelu", "let t1 = v3 * 0.044715", "let t1 = v3 * 0.04", "the cubic term's coefficient 0.04"),
    ("dit_gelu", "let th = tanh t3", "let th = t3", "no tanh"),
    ("dit_silu", "let a  = vv * sg", "let a  = vv + sg", "v + sigmoid(v)"),
    ("dit_add", "store x + y ->", "store x - y ->", "a - b"),
    ("dit_modsplit", "load md[0, 2 * n + j * bn ; 2, bn]", "load md[0, 3 * n + j * bn ; 2, bn]", "scale2 read from gate2's columns"),
    ("dit_modsplit", "store b -> g1[0, j * bn]", "store b -> g1[0, 0]", "gate1 written at column 0 for every chunk"),
    ("dit_rows2", "load x[0, j * bn ; 2, bn]", "load x[1, j * bn ; 2, bn]", "rows 1 and 2, not 0 and 1"),
]
good, _ = permutation(bad=True)
print(f"{'killed  ' if not good else 'SURVIVED'} permutation: the identity in place of the even-first permutation is refused")
survived = 1 if good else 0
for kernel, old, new, what in EDITS:
    with tempfile.TemporaryDirectory() as tmp:
        for f in os.listdir("tests/chx"):
            if f.endswith(".chx") and (f.startswith("dit_") or f == "rmsnorm_rows.chx"):
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
