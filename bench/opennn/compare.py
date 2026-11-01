#!/usr/bin/env python3
"""A two-layer sigmoid network, forward pass only, in OpenNN and in cancho-gpu on
the same weights and inputs (docs/comparison.md has what this does and does not
show).

    python3 bench/opennn/compare.py --opennn build/forward_opennn --gpu build/cancho-gpu [--reps 200] [--runs 7]

Data are multiples of 1/1024, so they are exact in f32 and in the text both
programs read. The reference is the same network in binary64. Prints one table and
writes bench/opennn/results.json."""
import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KERNEL = os.path.join(ROOT, "bench", "opennn", "dense_sigmoid.chx")
K, H = 28, 64
SIZES = (1, 256)


def lcg(seed):
    s = seed
    while True:
        s = (s * 1664525 + 1013904223) % 4294967296
        yield (s >> 9) % 2048 - 1024  # -1024 .. 1023


def tensor(gen, n, scale):
    return [next(gen) / 1024.0 * scale for _ in range(n)]


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def reference(x, w1, b1, w2, b2, r):
    out = []
    for i in range(r):
        h = [sigmoid(sum(x[i * K + k] * w1[j * K + k] for k in range(K)) + b1[j]) for j in range(H)]
        out.append(sigmoid(sum(h[j] * w2[j] for j in range(H)) + b2[0]))
    return out


def write(path, values):
    with open(path, "w") as f:
        f.write(" ".join(repr(v) for v in values) + "\n")


def gpu(binary, directory, r, files):
    """The two layers through `cancho-gpu run`; the hidden activations go through a
    file, as text, between them. Returns the outputs and the wall time of the two
    processes together."""
    p = lambda n: os.path.join(directory, n)
    t0 = time.perf_counter()
    h = subprocess.run([binary, "run", KERNEL, f"r={r}", f"n={H}", f"k={K}", "--target", "nvidia-ada",
                        "--in", "x", p("x.txt"), "--in", "w", p("w1.txt"), "--in", "b", p("b1.txt"), "--out", "y"],
                       capture_output=True, text=True, check=True).stdout
    with open(p("h.txt"), "w") as f:
        f.write(h)
    y = subprocess.run([binary, "run", KERNEL, f"r={r}", "n=1", f"k={H}", "--target", "nvidia-ada",
                        "--in", "x", p("h.txt"), "--in", "w", p("w2.txt"), "--in", "b", p("b2.txt"), "--out", "y"],
                       capture_output=True, text=True, check=True).stdout
    return [float(v) for v in y.split()], time.perf_counter() - t0


def opennn(binary, directory, r, reps, threads):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    out = subprocess.run([binary, directory, str(r), str(K), str(H), str(reps)], capture_output=True, text=True, check=True, env=env).stdout.split()
    got = [float(v) for v in open(os.path.join(directory, "opennn_out.txt")).read().split()]
    return got, float(out[1]), float(out[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opennn", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--runs", type=int, default=7)
    a = ap.parse_args()
    opn, gpub = os.path.abspath(a.opennn), os.path.abspath(a.gpu)
    results = {"machine": {"cpus": os.cpu_count(), "platform": platform.platform(), "load_at_start": os.getloadavg()[0]}, "k": K, "h": H, "cases": []}
    for r in SIZES:
        gen = lcg(7 + r)
        x, w1, b1 = tensor(gen, r * K, 1.0), tensor(gen, H * K, 0.5), tensor(gen, H, 0.25)
        w2, b2 = tensor(gen, H, 0.5), tensor(gen, 1, 0.25)
        want = reference(x, w1, b1, w2, b2, r)
        with tempfile.TemporaryDirectory() as d:
            for name, v in (("x", x), ("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2)):
                write(os.path.join(d, name + ".txt"), v)
            g, _ = gpu(gpub, d, r, None)
            walls = [gpu(gpub, d, r, None)[1] for _ in range(a.runs)]
            o1, o1_min, o1_med = opennn(opn, d, r, a.reps, 1)
            on, on_min, on_med = opennn(opn, d, r, a.reps, os.cpu_count())
        err = lambda got: max(abs(p - q) for p, q in zip(got, want))
        case = {
            "rows": r,
            "max_abs_error_vs_binary64": {"cancho-gpu": err(g), "opennn": err(o1)},
            "max_abs_difference_between": max(abs(p - q) for p, q in zip(g, o1)),
            "opennn_us": {"1_thread": {"min": o1_min, "median": o1_med}, f"{os.cpu_count()}_threads": {"min": on_min, "median": on_med}},
            "cancho_gpu_interpreter_wall_ms": {"min": min(walls) * 1000, "median": statistics.median(walls) * 1000, "runs": a.runs},
        }
        results["cases"].append(case)
        print(f"rows {r}: error against binary64: cancho-gpu {err(g):.2e}, OpenNN {err(o1):.2e}; difference between them {case['max_abs_difference_between']:.2e}")
        print(f"        OpenNN calculate_outputs, 1 thread:  min {o1_min:.1f} us, median {o1_med:.1f} us ({a.reps} calls)")
        print(f"        OpenNN calculate_outputs, {os.cpu_count()} threads: min {on_min:.1f} us, median {on_med:.1f} us")
        print(f"        cancho-gpu run, two processes, CPU interpreter: min {min(walls) * 1000:.0f} ms, median {statistics.median(walls) * 1000:.0f} ms ({a.runs} runs)")
    with open(os.path.join(ROOT, "bench", "opennn", "results.json"), "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
