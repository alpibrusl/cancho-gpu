#!/usr/bin/env python3
"""A least-squares fit whose loss and gradient are computed by cancho-gpu kernels,
as an `opt` objective (docs/run.md gate 7).

    opt minimize --method bfgs --x0 0,0,0 -- /abs/python3 examples/train/lsq_objective.py /abs/cancho-gpu /abs/cancho-gpu-checkout [TARGET]

TARGET is the `--target` to pass: `nvidia-ada` (the default) for `cancho-gpu run`, or
`-` for none, which is what `cancho-gpu-metal-device` and `cancho-gpu-device` take
(each has its one target, docs/run.md section 7).

Reads w (K numbers) on standard input. Forward: `lsq_residual.chx` gives
e = x w^T - t. Backward: `tests/chx/bwd_dw.chx` with n = 1 gives e^T x, the
transposes supplied here (the surface has no A^T B, docs/training.md section 4).
Prints the loss 1/(2R) |e|^2 and its gradient 1/R e^T x on one line.

`opt` gives a program no environment but PATH and HOME, and does not search PATH,
so the paths are arguments."""
import os
import subprocess
import sys
import tempfile

R, K = 16, 3
WSTAR = [1.5, -2.0, 0.5]


def data():
    """x, t: a fixed LCG, and t = x w* plus a deterministic perturbation, so the
    least-squares optimum is not w* and has to be solved for."""
    seed = 2024
    def draw():
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) % 4294967296
        return (seed >> 9) / 8388608.0 - 0.5
    x = [[draw() for _ in range(K)] for _ in range(R)]
    t = [sum(a * b for a, b in zip(row, WSTAR)) + 0.1 * draw() for row in x]
    return x, t


def write(path, values):
    with open(path, "w") as f:
        f.write(" ".join(repr(v) for v in values) + "\n")


def run(binary, kernel, consts, ins, out, target="nvidia-ada"):
    argv = [binary, "run", kernel] + ([] if target == "-" else ["--target", target]) + consts
    for name, path in ins:
        argv += ["--in", name, path]
    r = subprocess.run(argv + ["--out", out], capture_output=True, text=True, check=True)
    return [float(v) for v in r.stdout.split()]


def loss_grad(binary, root, w, target="nvidia-ada"):
    x, t = data()
    with tempfile.TemporaryDirectory() as d:
        p = lambda name: os.path.join(d, name)
        write(p("x"), [v for row in x for v in row])
        write(p("w"), w)
        write(p("t"), t)
        e = run(binary, os.path.join(root, "examples/train/lsq_residual.chx"), [f"r={R}", f"k={K}"],
                [("x", p("x")), ("w", p("w")), ("t", p("t"))], "e", target)
        write(p("et"), e)  # dy^T: [1, R]
        write(p("xt"), [x[i][c] for c in range(K) for i in range(R)])  # x^T: [K, R]
        dw = run(binary, os.path.join(root, "tests/chx/bwd_dw.chx"), [f"r={R}", "n=1", f"k={K}"],
                 [("dyt", p("et")), ("xt", p("xt"))], "dw", target)
    return 0.5 * sum(v * v for v in e) / R, [v / R for v in dw]


if __name__ == "__main__":
    target = sys.argv[3] if len(sys.argv) > 3 else "nvidia-ada"
    loss, grad = loss_grad(sys.argv[1], sys.argv[2], [float(v) for v in sys.stdin.read().split()], target)
    print(repr(loss), *map(repr, grad))
