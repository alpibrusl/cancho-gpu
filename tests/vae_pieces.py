#!/usr/bin/env python3
"""D4b of docs/diffusion.md: the pieces of the VAE decoder of Qwen-Image-2.1 as kernels, each against a float64 reference written from the mathematics (tools/vae_ref.py has the whole decoder), every element.

    vae_pieces.py <cancho-gpu> [--mutants]

The layout (a choice, docs/resident.md section 18's): an image of h x w pixels with c channels is a matrix of rows, one row a pixel of the image *with a border of one zero pixel*, (h + 2) x (w + 2) rows
in raster order, followed by zero rows up to a multiple of 64 (R rows: the matrix unit's tile), and, for an f16 input of a convolution, g = w + 3 zero guard rows on each side. The rows of a convolution's
output that are border are junk (they read across a row's end) and are zeroed by the next piece, which multiplies by an interior mask [R, 1].

* `vae_norm_silu`  silu(rmsnorm(x)) over the channels of each row (x / max(|x|, 1e-12) * sqrt(c) * gamma), times the mask, as f16 with the guards.
* `vae_bias`, `vae_bias_res`  (x + bias) * mask, and the same plus a residual (f32).
* `vae_pad16`      f32 to f16 with the guards.
* `vae_up2`        the nearest upsample by 2 (h x w to 2h x 2w), f32 in the bordered layout, f16 with guards out.
* `vae_up2_epi`    after the upsample's convolution: conv + bias + the shortcut's four sub-positions (s00, s01, s10, s11), written at the doubled resolution.
* `conv1`          a 1 x 1 convolution (a product) over the guarded f16 rows; `conv3_k16`  the 3 x 3 convolution for channels that are a multiple of 16 and not of 32 (the decoder's 144).
* `vae_split16`    the attention's qkv product [R, 3c] to q * 1/sqrt(c) and k as f16 and v as f32;  `vae_mask_add`  the additive key mask on the scores.

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
H = Wd = 6  # the test image: 6 x 6 pixels, bordered 8 x 8 = 64 rows (a multiple of 64)
WP = Wd + 2
RP = (H + 2) * WP
R = (RP + 63) // 64 * 64
G = WP + 1


def run(root, kernel, consts, ins, outs, tmp, sched=None):
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


def close(got, want, f16=False, tol=2e-6):
    worst = 0.0
    for a, b in zip(got, want):
        t = (4.9e-4 * abs(b) + 1e-3) if f16 else tol * max(1.0, abs(b))
        worst = max(worst, abs(a - b) / t)
    return len(got) == len(want) and worst <= 1.0, worst


def rnd(seed, n, lo=-1.0, hi=1.0):
    r = random.Random(seed)
    return [r.uniform(lo, hi) for _ in range(n)]


def interior(p, hh=H, ww=Wd):
    y, x = divmod(p, ww + 2)
    return 1 <= y <= hh and 1 <= x <= ww and p < (hh + 2) * (ww + 2)


MASK = [1.0 if interior(p) else 0.0 for p in range(R)]


def guarded(rows, c):  # [R * c] -> [(R + 2G) * c] with zero guards
    return [0.0] * (G * c) + rows + [0.0] * (G * c)


def t_norm_silu(root, tmp):
    c = 64
    x = rnd(1, R * c, -3, 3)
    for j in range(c):  # a row of tiny norm (eps matters: |x|^2 is about 0.02) and a zero interior row (the result must be 0, not nan)
        x[11 * c + j] *= 0.01
        x[20 * c + j] = 0.0
    gamma = rnd(2, c, 0.5, 1.5)
    want = [0.0] * (G * c)
    for p in range(R):
        row = x[p * c:(p + 1) * c]
        n = max(math.sqrt(sum(v * v for v in row)), 1e-12)
        for j in range(c):
            v = row[j] / n * math.sqrt(c) * gamma[j]
            want.append(v / (1 + math.exp(-v)) * MASK[p])
    want += [0.0] * (G * c)
    o, err = run(root, "vae_norm_silu", [f"r={R}", f"c={c}", f"g={G}", f"sc={math.sqrt(c)!r}"], {"x": x, "gamma": gamma, "mask": MASK}, ["y"], tmp)
    return (False, err) if o is None else close(o[0], want, True)


def t_bias(root, tmp):
    c = 64
    x, b = rnd(3, R * c, -3, 3), rnd(4, c)
    want = [(x[p * c + j] + b[j]) * MASK[p] for p in range(R) for j in range(c)]
    o, err = run(root, "vae_bias", [f"r={R}", f"c={c}"], {"x": x, "b": b, "m": MASK}, ["z"], tmp)
    return (False, err) if o is None else close(o[0], want)


def t_bias_res(root, tmp):
    c = 64
    x, b, res = rnd(5, R * c, -3, 3), rnd(6, c), rnd(7, R * c, -3, 3)
    want = [(x[p * c + j] + b[j]) * MASK[p] + res[p * c + j] for p in range(R) for j in range(c)]
    o, err = run(root, "vae_bias_res", [f"r={R}", f"c={c}"], {"x": x, "b": b, "m": MASK, "res": res}, ["z"], tmp)
    return (False, err) if o is None else close(o[0], want)


def t_pad16(root, tmp):
    c = 64
    x = rnd(8, R * c, -50, 50)
    o, err = run(root, "vae_pad16", [f"r={R}", f"c={c}", f"g={G}"], {"x": x}, ["y"], tmp)
    return (False, err) if o is None else close(o[0], guarded(x, c), True)


def up_layout(h, w):
    wp2 = 2 * w + 2
    rp2 = (2 * h + 2) * wp2
    return wp2, rp2, (rp2 + 63) // 64 * 64, wp2 + 1


def t_up2(root, tmp):
    c = 32
    x = [0.0] * (R * c)
    r = random.Random(9)
    for p in range(R):
        if interior(p):
            for j in range(c):
                x[p * c + j] = r.uniform(-4, 4)
    wp2, rp2, R2, g2 = up_layout(H, Wd)
    want = [0.0] * ((R2 + 2 * g2) * c)
    for yy in range(H):
        for xx in range(Wd):
            src = (yy + 1) * WP + xx + 1
            for a in range(2):
                for b in range(2):
                    dst = g2 + (2 * yy + 1 + a) * wp2 + 2 * xx + 1 + b
                    want[dst * c:(dst + 1) * c] = x[src * c:(src + 1) * c]
    o, err = run(root, "vae_up2", [f"h={H}", f"w={Wd}", f"c={c}", f"r={R}", f"r2={R2}", f"g2={g2}"], {"x": x}, ["y"], tmp)
    return (False, err) if o is None else close(o[0], want, True)


def t_up2_epi(root, tmp):
    c = 32
    wp2, rp2, R2, g2 = up_layout(H, Wd)
    cv, b = rnd(10, R2 * c, -2, 2), rnd(11, c)
    s = [rnd(12 + k, R * c, -2, 2) for k in range(4)]
    want = [0.0] * (R2 * c)
    for yy in range(H):
        for xx in range(Wd):
            src = (yy + 1) * WP + xx + 1
            for a in range(2):
                for bb in range(2):
                    dst = (2 * yy + 1 + a) * wp2 + 2 * xx + 1 + bb
                    for j in range(c):
                        want[dst * c + j] = cv[dst * c + j] + b[j] + s[2 * a + bb][src * c + j]
    o, err = run(root, "vae_up2_epi", [f"h={H}", f"w={Wd}", f"c={c}", f"r={R}", f"r2={R2}"], {"cv": cv, "b": b, "s00": s[0], "s01": s[1], "s10": s[2], "s11": s[3]}, ["z"], tmp)
    return (False, err) if o is None else close(o[0], want)


def f16r(v):  # a value representable in f16 (so that the reference and the kernel multiply the same numbers)
    import struct
    return struct.unpack("<e", struct.pack("<e", v))[0]


def t_conv1(root, tmp):
    cin, cout = 64, 48
    x = guarded([f16r(v) for v in rnd(20, R * cin, -2, 2)], cin)
    w = [f16r(v) for v in rnd(21, cout * cin, -1, 1)]
    want = []
    for p in range(R):
        for o in range(cout):
            want.append(sum(x[(G + p) * cin + k] * w[o * cin + k] for k in range(cin)))
    o, err = run(root, "conv1", [f"r={R}", f"cin={cin}", f"cout={cout}", f"g={G}"], {"x": x, "w": w}, ["y"], tmp)
    return (False, err) if o is None else close(o[0], want, tol=2e-5)


def t_conv3_k16(root, tmp):
    cin, cout = 48, 48
    xr = [0.0] * (R * cin)
    r = random.Random(22)
    for p in range(R):
        if interior(p):
            for j in range(cin):
                xr[p * cin + j] = f16r(r.uniform(-2, 2))
    x = guarded(xr, cin)
    w = [f16r(v) for v in rnd(23, cout * 9 * cin, -0.5, 0.5)]
    want = []
    for p in range(R):
        for o in range(cout):
            acc = 0.0
            for dy in range(3):
                for dx in range(3):
                    q = G + p + (dy - 1) * WP + dx - 1
                    acc += sum(x[q * cin + k] * w[o * 9 * cin + (3 * dy + dx) * cin + k] for k in range(cin))
            want.append(acc)
    o, err = run(root, "conv3_k16", [f"rp={R}", f"cin={cin}", f"cout={cout}", f"wp={WP}", f"g={G}"], {"x": x, "w": w}, ["y"], tmp)
    if o is None:
        return False, err
    sel = [i for p in range(R) if interior(p) for i in range(p * cout, (p + 1) * cout)]  # the border rows of a convolution are junk
    return close([o[0][i] for i in sel], [want[i] for i in sel], tol=2e-5)


def t_split16(root, tmp):
    c = 64
    qkv = rnd(24, R * 3 * c, -3, 3)
    sc = 1 / math.sqrt(c)
    q = [qkv[p * 3 * c + j] * sc for p in range(R) for j in range(c)]
    k = [qkv[p * 3 * c + c + j] for p in range(R) for j in range(c)]
    v = [qkv[p * 3 * c + 2 * c + j] for p in range(R) for j in range(c)]
    o, err = run(root, "vae_split16", [f"r={R}", f"c={c}", f"sc={sc!r}"], {"qkv": qkv}, ["q", "k", "v"], tmp)
    if o is None:
        return False, err
    rs = [close(o[0], q, True), close(o[1], k, True), close(o[2], v)]
    return all(x[0] for x in rs), max(x[1] for x in rs)


def t_mask_add(root, tmp):
    n = 64
    sc = rnd(25, n * n, -5, 5)
    mk = [0.0 if interior(p) else -1e9 for p in range(n)]
    want = [sc[i * n + j] + mk[j] for i in range(n) for j in range(n)]
    o, err = run(root, "vae_mask_add", [f"n={n}"], {"sc": sc, "mk": mk}, ["z"], tmp)
    return (False, err) if o is None else close(o[0], want, tol=1e-6)


TESTS = [("vae_norm_silu (norm over channels, SiLU, mask, f16 with guards)", t_norm_silu), ("vae_bias", t_bias), ("vae_bias_res", t_bias_res), ("vae_pad16", t_pad16),
         ("vae_up2 (nearest, bordered layout)", t_up2), ("vae_up2_epi (conv + bias + four shortcut sub-positions)", t_up2_epi), ("conv1 (1 x 1)", t_conv1),
         ("conv3_k16 (3 x 3, 48 channels)", t_conv3_k16), ("vae_split16 (q scaled, k, v)", t_split16), ("vae_mask_add", t_mask_add)]


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
    ("vae_norm_silu", "let t  = ss + 0.000000000000000000000001", "let t  = ss + 0.01", "an eps of 0.01 in the norm"),
    ("vae_norm_silu", "let y1 = xn * sc", "let y1 = xn * 1", "no sqrt(c) factor"),
    ("vae_norm_silu", "let b  = a * m", "let b  = a * 1", "no border mask"),
    ("vae_bias", "let s  = x + bb", "let s  = x - bb", "bias subtracted"),
    ("vae_bias_res", "let t  = s * mk", "let t  = s * 1", "no border mask on the residual path"),
    ("vae_pad16", "store ah -> y[g + i, 0]", "store ah -> y[i, 0]", "no guard offset"),
    ("vae_up2", "y[g2 + (2 * i + 2) * (2 * w + 2) + 2 * j + 1, 0]", "y[g2 + (2 * i + 2) * (2 * w + 2) + 2 * j + 2, 0]", "the (1, 0) copy one column to the right"),
    ("vae_up2_epi", "load s10[(i + 1) * (w + 2) + j + 1, 0 ; 1, c]", "load s01[(i + 1) * (w + 2) + j + 1, 0 ; 1, c]", "the (1, 0) shortcut read from the (0, 1) one"),
    ("conv1", "load x[g + i * bm, p * bk ; bm, bk]", "load x[g + i * bm + 1, p * bk ; bm, bk]", "rows shifted by one"),
    ("conv3_k16", "3 * dy * cin + dx * cin + p * bk", "3 * dy * cin + dy * cin + p * bk", "the taps' column index uses dy twice"),
    ("vae_split16", "let qs = a * sc", "let qs = a * 1", "q not scaled"),
    ("vae_mask_add", "store a + m -> z[i, 0]", "store a - m -> z[i, 0]", "the mask subtracted"),
]
survived = 0
for kernel, old, new, what in EDITS:
    with tempfile.TemporaryDirectory() as tmp:
        for f in os.listdir("tests/chx"):
            if f.endswith(".chx") and (f.startswith("vae_") or f.startswith("conv")):
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
