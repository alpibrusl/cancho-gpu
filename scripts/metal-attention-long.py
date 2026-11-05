#!/usr/bin/env python3
"""Long-context attention on a Metal GPU, the three kernels chained (docs/weight-space.md section 14.5, T4).

RUN ON A MAC (Apple Silicon). Not in scripts/gate.sh (no Metal there); `tests/kernels.py` holds each kernel to its reference
on the CPU and `tests/cuda-emu.sh` runs the CUDA the same three emit.

    scripts/build-metal-device.sh
    python3 scripts/metal-attention-long.py build/cancho-gpu-metal-device

attention_scores -> attention_stats -> attention_pvl on the device, each reading what the one before wrote (`--in` / `--out`),
at Qwen3.5-0.8B's shape (2 key heads, g = 4, d = 256; one key head at 16,640, for the size of the input files), held to the full softmax attention in float64 (numpy) over the first
`used` positions of a cache of `t`: the rest are masked by -1e9. The cases are lengths past the old ceiling (2,048 at g = 4),
the cap check's 16,544, a cache mostly masked (position 0 alone in the first chunk) and scores all below -88 (the case a
`-3e38` start of the running maximum gets wrong: attention_stats.chx).
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

BIN = os.path.abspath(sys.argv[1])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NKV, G, D, TS, CS, DW = 2, 4, 256, 64, 64, 32
TOL = 2e-5  # of the peak: sums of up to 16,640 terms in float32
failed = 0


def device(kernel, consts, ins, outs, d):
    argv = [BIN, "run", os.path.join(ROOT, "tests", "chx", f"{kernel}.chx")] + consts
    for name, arr in ins:
        path = os.path.join(d, name)
        np.savetxt(path, arr.reshape(1, -1), fmt="%d" if np.issubdtype(arr.dtype, np.integer) else "%.9g")
        argv += ["--in", name, path]
    for name in outs:
        argv += ["--out", name]
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{kernel} {consts}: {r.stderr.strip()[-400:]}")
    return [np.array(line.split(), dtype=np.float64) for line in r.stdout.strip().splitlines()]


def case(t, used, shift, seed, nkv=NKV):
    global failed
    rng = np.random.default_rng(seed)
    # k and v are -1, 0 or 1: `--in` files are at most 16 MiB of text (docs/run.md section 7), and 8,704 x 256 x 2 numbers are
    # over that unless they are short. q is wider so that the softmax is not flat.
    q = (3 * rng.standard_normal((nkv * G, D))).astype(np.float32)
    k = rng.integers(-1, 2, (nkv, t, D)).astype(np.int8)
    v = rng.integers(-1, 2, (nkv, t, D)).astype(np.int8)
    mask = np.where(np.arange(t) < used, shift, -1e9).astype(np.float32)
    # float64 reference: softmax(q k^T / 16 + mask) v over all t (the masked ones weigh exp(-1e9) = 0)
    want = np.zeros((nkv * G, D))
    for i in range(nkv):
        s = q[i * G:(i + 1) * G].astype(np.float64) @ k[i].astype(np.float64).T * 0.0625 + mask.astype(np.float64)
        e = np.exp(s - s.max(axis=1, keepdims=True))
        want[i * G:(i + 1) * G] = (e / e.sum(axis=1, keepdims=True)) @ v[i].astype(np.float64)
    with tempfile.TemporaryDirectory() as d:
        c = [f"nkv={nkv}", f"g={G}", f"t={t}", f"d={D}", f"tmax={t}"]
        (sc,) = device("attention_scores", c + [f"ts={TS}"], [("q", q), ("k", k), ("mask", mask)], ["sc"], d)
        sc32 = sc.astype(np.float32)
        m, z = device("attention_stats", [f"nkv={nkv}", f"g={G}", f"t={t}", f"cs={CS}"], [("sc", sc32)], ["m", "z"], d)
        (o,) = device("attention_pvl", c + [f"cs={CS}", f"dw={DW}"],
                      [("sc", sc32), ("m", m.astype(np.float32)), ("z", z.astype(np.float32)), ("v", v)], ["o"], d)
    peak = np.abs(want).max()
    worst = np.abs(o - want.reshape(-1)).max()
    good = bool(worst <= TOL * peak) and bool(np.isfinite(o).all())
    print(f"{'ok  ' if good else 'FAIL'} t={t} used={used} shift={shift}: largest difference {worst:.3g} = {worst / peak:.2e} of the peak {peak:.4g}")
    failed += 0 if good else 1


case(2048, 2048, 0.0, 1)
case(4096, 3000, 0.0, 2)
case(8704, 8704, 0.0, 3)
case(8704, 1, 0.0, 4)
case(8704, 8704, -300.0, 5)
case(16640, 16640, 0.0, 6, nkv=1)
sys.exit(1 if failed else 0)
