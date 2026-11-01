#!/usr/bin/env python3
"""The least-squares fit of lsq_objective.py as a session (docs/resident.md): the data is read once,
the forward kernel's residual feeds the backward kernel on the device, and a request is the
parameters, a line of K numbers, answered with the loss and the gradient, a line of K + 1.

    python3 examples/train/lsq_session.py DIR R K        write DIR/session, x, xt, t (the fit of lsq_objective.py at that size)
    cancho-gpu-session DIR/session < parameters          answer a line for each line

The loss is e^T e (bwd_dw with k = 1 on the residual twice), scaled by 1/2R on the host; the
gradient is bwd_dw's e^T x scaled by 1/R."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lsq_objective as L  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))


def block(r):
    """The rows one threadgroup of the forward kernel takes: the largest of 128, 64, ..., 1 that divides r."""
    return next(b for b in (128, 64, 32, 16, 8, 4, 2, 1) if r % b == 0)


def write(directory, r, k):
    L.R, L.K = r, k
    L.WSTAR = [((i % 7) - 3) / 2.0 if k > 3 else v for i, v in enumerate([1.5, -2.0, 0.5] + [0.0] * max(0, k - 3))][:k]
    x, t = L.data()
    os.makedirs(directory, exist_ok=True)
    p = lambda name: os.path.join(directory, name)
    L.write(p("x"), [v for row in x for v in row])
    L.write(p("xt"), [x[i][c] for c in range(k) for i in range(r)])
    L.write(p("t"), t)
    with open(p("session"), "w") as f:
        f.write(f"""# the least-squares fit at {r} x {k}
program fwd {os.path.join(ROOT, 'examples/train/lsq_residual.chx')} r={r} k={k} bm={block(r)}
program bwd {os.path.join(ROOT, 'tests/chx/bwd_dw.chx')} r={r} n=1 k={k}
program sq  {os.path.join(ROOT, 'tests/chx/bwd_dw.chx')} r={r} n=1 k=1

resident fwd.x  {p('x')}
resident fwd.t  {p('t')}
resident bwd.xt {p('xt')}

link fwd.e -> bwd.dyt
link fwd.e -> sq.dyt
link fwd.e -> sq.xt

input fwd.w

answer sq.dw  scale 0.5/{r}
answer bwd.dw scale 1/{r}
""")
    return p("session")


if __name__ == "__main__":
    print(write(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
