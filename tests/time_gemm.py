#!/usr/bin/env python3
"""D6b of docs/diffusion.md: how fast `gemm_mma.chx` runs at the shapes of the transformer's block (320 rows: 277 tokens padded to the matrix unit's tile of 64), on the Metal GPU, and the
attention's three kernels, each alone in a session with the data resident: a request "N" runs the program N times in one command buffer with one wait (docs/resident.md section 12), so the time of N = 200
less that of N = 1, over 199, is one run's. The inputs are zeros (the matrix unit's time does not depend on the values).

    time_gemm.py
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


def zeros(name, n, dt):
    path = os.path.join(tmp, name)
    np.zeros(n, dtype=dt).tofile(path)
    return path


def bench(label, kernel, args, residents, flops):
    lines = ["program p %s/%s.chx %s" % (CHX, kernel, args)]
    for name, n, dt, flag in residents:
        lines.append("resident p.%s %s %s" % (name, zeros(name, n, dt), flag))
    lines.append("steps")
    lines.append("answer p.%s" % residents_out[kernel])
    open(os.path.join(tmp, "s.session"), "w").write("\n".join(lines) + "\n")
    import subprocess
    out = []
    for n in (1, 200):
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
    per = (out[1] - out[0]) / 199
    print("%-46s %8.1f us a run  %6.2f TFLOPS" % (label, per * 1e6, flops / per / 1e12))
    return per


residents_out = {"ported/gemm_mma": "y", "dit_scores": "sc", "dit_stats": "m", "dit_pv": "o"}
M = 320
gemms = {}
for label, n, k in (("q, k, v, o projection  [320 x 4096] x [4096 x 4096]", 4096, 4096), ("gate, up projection  [320 x 4096] x [4096 x 12288]", 12288, 4096), ("down projection  [320 x 12288] x [12288 x 4096]", 4096, 12288)):
    gemms[label] = bench(label, "ported/gemm_mma", "m=%d n=%d k=%d" % (M, n, k), [("a", M * k, np.float16, "f16"), ("w", n * k, np.float16, "f16")], 2.0 * M * n * k)
a = gemms[list(gemms)[0]]
b = gemms[list(gemms)[1]]
c = gemms[list(gemms)[2]]
print("the matrix products of one block: 4 x %.0f + 2 x %.0f + 1 x %.0f us = %.1f ms; 32 blocks: %.0f ms (a step is 497 ms)" % (a * 1e6, b * 1e6, c * 1e6, (4 * a + 2 * b + c) * 1e3, 32 * (4 * a + 2 * b + c) * 1e3))
H, SP = 32, M
sc = bench("dit_scores  32 heads, 320 x 320, d 128", "dit_scores", "nh=%d nq=%d t=%d qs=32 ts=64 d=128" % (H, SP, SP), [("q", SP * H * 128, np.float32, "f32"), ("k", SP * H * 128, np.float32, "f32"), ("mask", SP * SP, np.float32, "f32")], 2.0 * H * SP * SP * 128)
st = bench("dit_stats", "dit_stats", "nh=%d nq=%d t=%d qs=32 cs=64" % (H, SP, SP), [("sc", H * SP * SP, np.float32, "f32")], 0.0)
pv = bench("dit_pv", "dit_pv", "nh=%d nq=%d t=%d qs=32 cs=64 dw=32 d=128" % (H, SP, SP), [("sc", H * SP * SP, np.float32, "f32"), ("m", H * SP, np.float32, "f32"), ("z", H * SP, np.float32, "f32"), ("v", SP * H * 128, np.float32, "f32")], 2.0 * H * SP * SP * 128)
print("attention of one block: %.1f ms; 32 blocks: %.0f ms" % ((sc + st + pv) * 1e3, 32 * (sc + st + pv) * 1e3))
