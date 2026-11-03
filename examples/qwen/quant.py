#!/usr/bin/env python3
"""Host-side 4-bit and 8-bit quantisers for the base weights of a QLoRA run (docs/resident.md section 25; roadmap M7.5, a design with probes, no kernel written).

    python3 examples/qwen/quant.py errors WEIGHTS [--layers 0,14,27]          # the weight error of each format on the matrices of those layers, and the bits a weight costs
    python3 examples/qwen/quant.py validate                                    # the packing below against the real `dequant_fp4` of the interpreter (cancho-gpu run gemm_fp4)
    python3 examples/qwen/quant.py fake WEIGHTS OUT FORMAT                     # a weights directory whose matrices are quantised and dequantised (f16): the forward pass of an existing session measures the loss
    python3 examples/qwen/quant.py pack WEIGHTS OUT nvfp4:G                    # a weights directory whose four matrices of every block are the files `NAME.q`, `NAME.s`, `NAME.g` that a QLoRA session reads (section 27)
    python3 examples/qwen/quant.py dequant QWEIGHTS OUT                        # the f16 weights directory those files stand for, decoded from their bytes (what the session's dequantise kernel produces)

Formats: nvfp4:G (E2M1 codes, an E4M3 scale per G values, one f32 per row: exactly what the surface's `dequant_fp4` reads; G = 16, 32, 64), int4:G (symmetric absmax, f16 scale, levels -8..7),
nf4:G (QLoRA's 16 levels, absmax in f16), e2m1:G (the E2M1 codebook with an f16 scale per group: the codebook of nvfp4 without its 8-bit scale), int8:G.
The quantisers are numpy (binary64). They are the *measurement* of what a format costs in accuracy; the surface can read only nvfp4 (section 25.2)."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import qwen_ref as R  # noqa: E402
from nvfp4 import E2M1, E4M3, round_e4m3, round_codebook, groups, nvfp4, pack_nvfp4, pack_rows, unpack_rows  # noqa: E402,F401

NF4 = np.array([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])




def int4(w, g=64):
    gw = groups(w, g)
    s = (np.abs(gw).max(2) / 7).astype(np.float16).astype(np.float64)
    s = np.where(s == 0, 1.0, s)
    q = np.clip(np.round(gw / s[:, :, None]), -8, 7)
    return (q * s[:, :, None]).reshape(w.shape)


def nf4(w, g=64):
    gw = groups(w, g)
    s = np.abs(gw).max(2).astype(np.float16).astype(np.float64)
    s = np.where(s == 0, 1.0, s)
    return (NF4[round_codebook(gw / s[:, :, None], NF4)] * s[:, :, None]).reshape(w.shape)


def e2m1_f16(w, g=64):
    gw = groups(w, g)
    s = (np.abs(gw).max(2) / 6).astype(np.float16).astype(np.float64)
    s = np.where(s == 0, 1.0, s)
    z = gw / s[:, :, None]
    return (np.sign(z) * E2M1[round_codebook(np.abs(z), E2M1)] * s[:, :, None]).reshape(w.shape)


def int8(w, g=64):
    gw = groups(w, g)
    s = (np.abs(gw).max(2) / 127).astype(np.float16).astype(np.float64)
    s = np.where(s == 0, 1.0, s)
    return (np.clip(np.round(gw / s[:, :, None]), -127, 127) * s[:, :, None]).reshape(w.shape)


def quantise(w, fmt):
    kind, _, g = fmt.partition(":")
    g = int(g or 64)
    return {"nvfp4": lambda: nvfp4(w, g)[0], "int4": lambda: int4(w, g), "nf4": lambda: nf4(w, g), "e2m1": lambda: e2m1_f16(w, g), "int8": lambda: int8(w, g)}[kind]()


def bits(fmt, k=1536):
    """Bits a weight costs: the code, the group's scale, and for nvfp4 the row's f32 over a row of k."""
    kind, _, g = fmt.partition(":")
    g = int(g or 64)
    return {"nvfp4": 4 + 8 / g + 32 / k, "int4": 4 + 16 / g, "nf4": 4 + 16 / g, "e2m1": 4 + 16 / g, "int8": 8 + 16 / g}[kind]


def validate(mutants=True):
    """The packing against the interpreter: a 128 x 128 matrix through `gemm_fp4` with the identity as x gives its transpose; it must equal the numpy dequantisation rounded to f16 (the kernel stages the dequantised tile as f16), but for a ties-to-even difference of the f32 product."""
    n, k = 128, 128
    w = np.random.default_rng(3).standard_normal((n, k)) * 0.05
    w[5, 17] = 0.9
    w[9] = 0.0
    for g, swap in [(16, False), (32, False), (64, False)] + ([(16, True)] if mutants else []):
        q, s, row, deq = pack_nvfp4(w, g, swap)
        src = open(os.path.join(ROOT, "tests", "chx", "ported", "gemm_fp4.chx")).read()
        if g != 16:
            src = src.replace("k / 16", f"k / {g}").replace("bk / 16", f"bk / {g}").replace("dequant_fp4 q s g 16", f"dequant_fp4 q s g {g}")
            src = src.replace("bm 64 bn 128 bk 32", f"bm 64 bn 128 bk {max(32, g)}").replace("bm 128 bn 128 bk 64", f"bm 128 bn 128 bk {max(64, g)}")
        d = tempfile.mkdtemp()
        open(f"{d}/g.chx", "w").write(src)
        np.savetxt(f"{d}/x", np.eye(k), fmt="%d")
        np.savetxt(f"{d}/q", q.reshape(-1), fmt="%d")
        np.savetxt(f"{d}/s", s.reshape(-1), fmt="%d")
        np.savetxt(f"{d}/g", row, fmt="%.9g")
        out = subprocess.run([os.path.join(ROOT, "build", "cancho-gpu"), "run", f"{d}/g.chx", f"m={k}", f"n={n}", f"k={k}", "--target", "apple-m-series", "--in", "x", f"{d}/x", "--in", "wq", f"{d}/q", "--in", "ws", f"{d}/s",
                              "--in", "wg", f"{d}/g", "--out", "y"], capture_output=True, text=True)
        if out.returncode:
            raise SystemExit(f"gemm_fp4 G={g}: {out.stderr[:300]}")
        got = np.array(out.stdout.split(), dtype=np.float32).astype(np.float64).reshape(k, n).T  # y[i, j] = w_hat[j, i]; the text is the shortest form of an f32
        want = deq.astype(np.float16).astype(np.float64)
        ne = int(np.count_nonzero(got != want))
        if swap:
            print(f"mutant (nibbles swapped): {ne} of {n * k} weights differ from the dequantisation, so the check fails")
            assert ne > n * k // 2, "the nibble-order mutant was not caught"
            continue
        print(f"G={g}: the interpreter's dequant_fp4 and the numpy packing agree: {n * k - ne} of {n * k} weights equal once rounded to f16 (the kernel stages the tile as f16); the {ne} others differ by "
              f"{np.abs(got - want).max():.1e} at most (one f16 step is {np.spacing(np.float16(np.abs(deq).max())):.1e} at the largest weight {np.abs(deq).max():.2f}: the product is formed in f32 there, in binary64 here)")
        assert ne <= n * k // 1000, "packing disagrees with dequant_fp4"


def errors(wdir, layers):
    c = R.config_of(wdir)
    sh = R.shapes(c)
    fmts = ["nvfp4:16", "nvfp4:32", "nvfp4:64", "e2m1:64", "int4:64", "int4:128", "nf4:64", "nf4:128", "int8:64"]
    print(f"{'format':10s} {'bits/w':>7s}  " + "  ".join(f"{k:>7s}" for k in ("wqkv", "wo", "wgu", "wd")) + "   all (weighted)")
    tot = {f: [0.0, 0.0] for f in fmts}
    per = {f: {} for f in fmts}
    for l in layers:
        for kind in ("wqkv", "wo", "wgu", "wd"):
            name = f"l{l:02d}.{kind}"
            w = np.fromfile(os.path.join(wdir, name + ".f16"), dtype="<f2").astype(np.float64).reshape(sh[name])
            for f in fmts:
                e = quantise(w, f) - w
                a, b = float((e * e).sum()), float((w * w).sum())
                tot[f][0] += a
                tot[f][1] += b
                p = per[f].setdefault(kind, [0.0, 0.0])
                p[0] += a
                p[1] += b
    for f in fmts:
        print(f"{f:10s} {bits(f, c.d):7.3f}  " + "  ".join(f"{np.sqrt(per[f][k][0] / per[f][k][1]):7.4f}" for k in ("wqkv", "wo", "wgu", "wd")) + f"   {np.sqrt(tot[f][0] / tot[f][1]):.4f}")
    print(f"(relative Frobenius error of the weights, layers {layers} of {wdir})")


def fake(wdir, out, fmt):
    c = R.config_of(wdir)
    os.makedirs(out, exist_ok=True)
    sh = R.shapes(c)
    for name in sh:
        kind = name.split(".")[-1]
        ext = "f32" if kind in ("ln1", "ln2", "norm", "bqkv") else "f16"
        src, dst = os.path.join(wdir, f"{name}.{ext}"), os.path.join(out, f"{name}.{ext}")
        if kind in ("wqkv", "wo", "wgu", "wd"):
            w = np.fromfile(src, dtype="<f2").astype(np.float64).reshape(sh[name])
            quantise(w, fmt).astype("<f2").tofile(dst)
        else:
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
    import shutil
    shutil.copy(os.path.join(wdir, "manifest.json"), out)
    print(f"{out}: {fmt}, {len(sh)} tensors")


MATRICES = ("wqkv", "wo", "wgu", "wd")


def pack(wdir, out, fmt):
    """The quantised weights directory: every matrix of a block as `.q`, `.s`, `.g` (nvfp4:G only: it is the one format the surface reads); the other files and the manifest as they are, with the format added."""
    kind, _, g = fmt.partition(":")
    if kind != "nvfp4" or not g.isdigit():
        raise SystemExit(f"{fmt}: only nvfp4:G (G = 16, 32 or 64) can be read by a session")
    G = int(g)
    c = R.config_of(wdir)
    sh = R.shapes(c)
    os.makedirs(out, exist_ok=True)
    for name in sh:
        base = name.split(".")[-1]
        ext = "f32" if base in ("ln1", "ln2", "norm", "bqkv") else "f16"
        src = os.path.join(wdir, f"{name}.{ext}")
        if base in MATRICES:
            w = np.fromfile(src, dtype="<f2").reshape(sh[name])
            q, s, r = pack_rows(w, G)
            for suffix, arr in (("q", q), ("s", s), ("g", r)):
                arr.tofile(os.path.join(out, f"{name}.{suffix}"))
        else:
            dst = os.path.join(out, f"{name}.{ext}")
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
    man = json.load(open(os.path.join(wdir, "manifest.json")))
    man["quant"] = dict(format="nvfp4", group=G)
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(f"{out}: nvfp4:{G}, {len(sh)} tensors")


def dequant(qdir, out):
    man = json.load(open(os.path.join(qdir, "manifest.json")))
    G = man["quant"]["group"]
    c = R.config_of(qdir)
    sh = R.shapes(c)
    os.makedirs(out, exist_ok=True)
    for name in sh:
        base = name.split(".")[-1]
        ext = "f32" if base in ("ln1", "ln2", "norm", "bqkv") else "f16"
        dst = os.path.join(out, f"{name}.{ext}")
        if base in MATRICES:
            n, k = sh[name]
            q = np.fromfile(os.path.join(qdir, name + ".q"), dtype=np.uint8).reshape(n, k // 2)
            s = np.fromfile(os.path.join(qdir, name + ".s"), dtype=np.uint8).reshape(n, k // G)
            r = np.fromfile(os.path.join(qdir, name + ".g"), dtype="<f4")
            unpack_rows(q, s, r, G).astype("<f2").tofile(dst)
        else:
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(os.path.join(qdir, f"{name}.{ext}")), dst)
    man.pop("quant")  # the result is an f16 directory
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "errors":
        ls = [int(x) for x in (sys.argv[sys.argv.index("--layers") + 1] if "--layers" in sys.argv else "0,14,27").split(",")]
        errors(sys.argv[2], ls)
    elif cmd == "validate":
        validate()
    elif cmd == "pack":
        pack(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "dequant":
        dequant(sys.argv[2], sys.argv[3])
    elif cmd == "fake":
        fake(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit("usage: quant.py errors WEIGHTS | validate | fake WEIGHTS OUT FORMAT | pack WEIGHTS OUT nvfp4:G | dequant QWEIGHTS OUT")
