#!/usr/bin/env python3
"""D6b of docs/diffusion.md: how fast `gemm_mma.chx` runs at the shapes of the transformer's block (320 rows: 277 tokens padded to the matrix unit's tile of 64), on the Metal GPU, and the
attention's three kernels, each alone in a session with the data resident: a request "N" runs the program N times in one command buffer with one wait (docs/resident.md section 12), so the time of N = 200
less that of N = 1, over 199, is one run's. The inputs are zeros by default (RANDOM_DATA=1: random numbers: the time of a matrix product does depend on the values, through the power the unit draws).

    time_gemm.py [ROWS]     (ROWS, a multiple of 64, defaults to 320; 4160 is 1024 x 1024)
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_time  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
CHX = os.path.join(ROOT, "tests/chx")
tmp = tempfile.mkdtemp()


RANDOM = os.environ.get("RANDOM_DATA") == "1"  # random inputs instead of zeros: the matrix unit's power, and so its clock, depends on the data


def zeros(name, n, dt):
    path = os.path.join(tmp, name)
    if RANDOM:
        (np.random.default_rng(1).standard_normal(n) * 0.5).astype(dt).tofile(path)
    else:
        np.zeros(n, dtype=dt).tofile(path)
    return path


def bench(label, kernel, args, residents, flops, shape):
    lines = ["program p %s/%s.chx %s" % (CHX, kernel, args)]
    for name, n, dt, flag in residents:
        lines.append("resident p.%s %s %s" % (name, zeros(name, n, dt), flag))
    lines.append("program probe %s/%s.chx m=%d n=%d bn=%d" % (CHX, "dit_rows2h" if kernel in H16 else "dit_rows2", shape[0], shape[1], next(b for b in (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1) if shape[1] % b == 0)))
    lines.append("link p.%s -> probe.x" % residents_out[kernel])
    lines.append("steps")
    lines.append("answer probe.y")
    open(os.path.join(tmp, "s.session"), "w").write("\n".join(lines) + "\n")
    import subprocess
    out = []
    for n in (1, 20 if M > 1000 else 200):
        p = subprocess.Popen([SESSION, os.path.join(tmp, "s.session")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        p.stdin.write("1\n")
        p.stdin.flush()
        p.stdout.readline()  # warm-up request
        t0 = time.perf_counter()
        p.stdin.write("%d\n" % n)
        p.stdin.flush()
        p.stdout.readline()
        out.append(time.perf_counter() - t0)
        p.stdin.close()
        p.wait()
    per = (out[1] - out[0]) / 19 if M > 1000 else (out[1] - out[0]) / 199
    print("%-46s %8.1f us a run  %6.2f TFLOPS" % (label, per * 1e6, flops / per / 1e12))
    return per


H16 = {"dit_lnmod", "dit_to_f16", "dit_swiglu"}  # kernels whose output is f16: the probe widens it
residents_out = {"ported/gemm_mma": "y", "dit_scores": "sc", "dit_stats": "m", "dit_pv": "o"}
M = int(sys.argv[1]) if len(sys.argv) > 1 else 320
gemms = {}
for label, n, k in (("q, k, v, o projection  [%d x 4096] x [4096 x 4096]" % M, 4096, 4096), ("gate, up projection  [%d x 4096] x [4096 x 12288]" % M, 12288, 4096), ("down projection  [%d x 12288] x [12288 x 4096]" % M, 4096, 12288)):
    gemms[label] = bench(label, "ported/gemm_mma", "m=%d n=%d k=%d" % (M, n, k), [("a", M * k, np.float16, "f16"), ("w", n * k, np.float16, "f16")], 2.0 * M * n * k, (M, n))
a = gemms[list(gemms)[0]]
b = gemms[list(gemms)[1]]
c = gemms[list(gemms)[2]]
print("the matrix products of one block: 4 x %.0f + 2 x %.0f + 1 x %.0f us = %.1f ms; 32 blocks: %.0f ms (a step: see the table)" % (a * 1e6, b * 1e6, c * 1e6, (4 * a + 2 * b + c) * 1e3, 32 * (4 * a + 2 * b + c) * 1e3))
H, SP = 32, M
sc = bench("dit_scores  32 heads, %d x %d, d 128" % (M, M), "dit_scores", "nh=%d nq=%d t=%d qs=32 ts=64 d=128" % (H, SP, SP), [("q", SP * H * 128, np.float32, "f32"), ("k", SP * H * 128, np.float32, "f32"), ("mask", SP * SP, np.float32, "f32")], 2.0 * H * SP * SP * 128, (H * SP, SP))
st = bench("dit_stats", "dit_stats", "nh=%d nq=%d t=%d qs=32 cs=64" % (H, SP, SP), [("sc", H * SP * SP, np.float32, "f32")], 0.0, (H * SP, 1))
pv = bench("dit_pv", "dit_pv", "nh=%d nq=%d t=%d qs=32 cs=64 dw=32 d=128" % (H, SP, SP), [("sc", H * SP * SP, np.float32, "f32"), ("m", H * SP, np.float32, "f32"), ("z", H * SP, np.float32, "f32"), ("v", SP * H * 128, np.float32, "f32")], 2.0 * H * SP * SP * 128, (SP, H * 128))
print("attention of one block: %.1f ms; 32 blocks: %.0f ms" % ((sc + st + pv) * 1e3, 32 * (sc + st + pv) * 1e3))

# the small kernels of a block, at the same row count (counts per block in the label)
f32_ = np.float32
D = 4096
smalls = [
    ("dit_lnmod  x2", "dit_lnmod", "s=%d n=%d" % (M, D), [("x", M * D, f32_, "f32"), ("sc", 2 * D, f32_, "f32"), ("sel", M, f32_, "f32")], (M, D), 2, "y"),
    ("rmsnorm_rows (per head q, k)  x2", "rmsnorm_rows", "r=%d n=128 eps=0.000001" % (M * 32), [("x", M * D, f32_, "f32"), ("w", 128, f32_, "f32")], (M * 32, 128), 2, "y"),
    ("dit_rope  x2", "dit_rope", "s=%d nh=32 h=64" % M, [("x", M * D, f32_, "f32"), ("cs", M * 64, f32_, "f32"), ("sn", M * 64, f32_, "f32")], (M * 32, 128), 2, "y"),
    ("dit_to_f16  x1", "dit_to_f16", "s=%d n=%d" % (M, D), [("x", M * D, f32_, "f32")], (M, D), 1, "y"),
    ("dit_gate_res  x2", "dit_gate_res", "s=%d n=%d" % (M, D), [("x", M * D, f32_, "f32"), ("y", M * D, f32_, "f32"), ("g", 2 * D, f32_, "f32"), ("sel", M, f32_, "f32")], (M, D), 2, "z"),
    ("dit_swiglu  x1", "dit_swiglu", "s=%d n=%d bn=1024" % (M, 3 * D), [("g", M * 3 * D, f32_, "f32"), ("u", M * 3 * D, f32_, "f32")], (M, 3 * D), 1, "y"),
]
total = 0.0
for label, kernel, args, res_, shape, count, out in smalls:
    residents_out[kernel] = out
    per = bench(label.split("  x")[0], kernel, args, res_, 0.0, shape)
    total += per * count
print("the small kernels of one block: %.1f ms; 32 blocks: %.0f ms" % (total * 1e3, 32 * total * 1e3))

# the attention on the matrix unit (docs/diffusion.md D8)
residents_out.update({"dit_rope16": "y", "dit_scores_mma": "sc", "dit_stats_m": "m", "dit_pv_mma": "o"})
H16.add("dit_rope16")
r16 = bench("dit_rope16  (x2 per block)", "dit_rope16", "s=%d nh=32 h=64 sc=0.08838834764831845" % M, [("x", M * D, f32_, "f32"), ("cs", M * 64, f32_, "f32"), ("sn", M * 64, f32_, "f32")], 0.0, (M * 32, 128))
s2 = bench("dit_scores_mma  32 heads, %d x %d, d 128" % (M, M), "dit_scores_mma", "nh=32 nq=%d t=%d d=128" % (M, M), [("q", M * D, np.float16, "f16"), ("k", M * D, np.float16, "f16")], 2.0 * 32 * M * M * 128, (32 * M, M))
t2 = bench("dit_stats_m", "dit_stats_m", "nh=32 nq=%d t=%d qs=32 cs=64" % (M, M), [("sc", 32 * M * M, f32_, "f32"), ("mask", M * M, f32_, "f32")], 0.0, (32 * M, 1))
p2 = bench("dit_pv_mma", "dit_pv_mma", "nh=32 nq=%d t=%d d=128" % (M, M), [("sc", 32 * M * M, f32_, "f32"), ("mask", M * M, f32_, "f32"), ("m", 32 * M, f32_, "f32"), ("z", 32 * M, f32_, "f32"), ("v", M * D, np.float16, "f16")], 2.0 * 32 * M * M * 128, (M, D))
att = s2 + t2 + p2 + 2 * r16 + 0.0002  # + the cast of v (dit_to_f16, about 0.2 ms)
print("attention of one block on the matrix unit (two rope16, the cast of v, scores, stats, pv): %.1f ms; 32 blocks: %.0f ms" % (att * 1e3, 32 * att * 1e3))
