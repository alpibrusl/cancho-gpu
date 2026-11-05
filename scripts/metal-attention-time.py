#!/usr/bin/env python3
"""Time the three long-attention kernels on a Metal GPU at Qwen3.5-0.8B's shape (docs/weight-space.md 14.5).

RUN ON A MAC. Not in the gate. Time only (random data, no correctness: scripts/metal-attention-long.py is that).

    scripts/metal-attention-time.py <T> [<T> ...]       # e.g. 2048 4096 8704 16640

Builds the shim as a dynamic library and the kernels' Metal source with `build/cancho-gpu emit` in a temporary directory, then
launches scores, stats and pvl in one command buffer with one wait (the runner's pattern), 200 times after 5 warm-ups. The
figure includes the commit and the wait, which a layer in a token's command buffer shares with the rest of the token, so it is an
upper bound on what one attention layer adds.
"""
import ctypes
import glob
import os
import random
import struct
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NKV, G, D = 2, 4, 256
D_ = tempfile.mkdtemp()
subprocess.run(["clang", "-fobjc-arc", "-fno-exceptions", "-fno-objc-exceptions", "-shared", "-fPIC",
                os.path.join(ROOT, "shim", "canchogpu-metal.mm"), "-o", D_ + "/libmetalgpu.dylib",
                "-framework", "Metal", "-framework", "Foundation"], check=True)
L = ctypes.CDLL(D_ + "/libmetalgpu.dylib")
for f in ("cgpu_open", "cgpu_compile", "cgpu_function", "cgpu_alloc", "cgpu_upload", "cgpu_arg", "cgpu_launch_async", "cgpu_sync"):
    getattr(L, f).restype = ctypes.c_int64
assert L.cgpu_open() >= 0


def emit(kernel, consts):
    subprocess.run([os.path.join(ROOT, "build", "cancho-gpu"), "emit", os.path.join(ROOT, "tests", "chx", kernel + ".chx"), D_] + consts,
                   check=True, capture_output=True)


def kern(prefix, grid, threads):
    path = [p for p in glob.glob(f"{D_}/{prefix}_*.metal")][0]
    src = open(path).read().encode()
    m = L.cgpu_compile(src, len(src))
    assert m >= 0
    name = os.path.basename(path)[:-6].encode()
    fn = L.cgpu_function(m, name, len(name))
    assert fn >= 0
    os.remove(path)
    return fn, grid, threads


def buf(n):
    b = L.cgpu_alloc(n * 4)
    assert b >= 0
    data = (struct.pack("1000f", *[random.uniform(-.5, .5) for _ in range(1000)]) * (n // 1000 + 1))[: n * 4]
    L.cgpu_upload(b, data, len(data))
    return b


def launch(kk, bufs):
    fn, (gx, gy), th = kk
    for b in bufs:
        assert L.cgpu_arg(b) >= 0
    assert L.cgpu_launch_async(fn, gx, gy, th) >= 0


for T in map(int, sys.argv[1:]):
    emit("attention_scores", [f"nkv={NKV}", f"g={G}", f"t={T}", "ts=64", f"d={D}", f"tmax={T}"])
    emit("attention_stats", [f"nkv={NKV}", f"g={G}", f"t={T}", "cs=64"])
    emit("attention_pvl", [f"nkv={NKV}", f"g={G}", f"t={T}", "cs=64", "dw=32", f"d={D}", f"tmax={T}"])
    ks, kt, kp = kern("attention_scores", (NKV, T // 64), 512), kern("attention_stats", (NKV, 1), 256), kern("attention_pvl", (NKV, D // 32), 256)
    q, k, v, mask, sc, m, z, o = buf(NKV * G * D), buf(NKV * T * D), buf(NKV * T * D), buf(T), buf(NKV * G * T), buf(NKV * G), buf(NKV * G), buf(NKV * G * D)

    def once(which):
        if "s" in which:
            launch(ks, [q, k, mask, sc])
        if "t" in which:
            launch(kt, [sc, m, z])
        if "p" in which:
            launch(kp, [sc, m, z, v, o])
        assert L.cgpu_sync() >= 0

    for which, label in (("stp", "all three"), ("s", "scores"), ("t", "stats"), ("p", "pvl")):
        for _ in range(5):
            once(which)
        t0 = time.perf_counter()
        for _ in range(200):
            once(which)
        print(f"t={T:6d} {label:9s} {(time.perf_counter() - t0) / 200 * 1e6:6.0f} us")
