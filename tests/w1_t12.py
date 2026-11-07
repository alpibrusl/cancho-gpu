#!/usr/bin/env python3
"""docs/weight-space.md section 14.15: the pure parts of examples/weightspace/w1_t12.py. No model, no GPU, no numpy needed.

    python3 tests/w1_t12.py

  1. projection: 200 problems x 7,829 tokens x 10 ms a token = 4.35 hours, linear in each factor; non-positive input is refused `t12-input`.
  2. feasibility: the thresholds 10 and 24 hours, inclusive.
  3. job_lines drops the timing lines and groups by job; first_mismatch finds the first differing step, a length difference and equality.
  4. The CLI exits 2 with `t12-input` on a missing patch."""
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, os.path.join(ROOT, "examples", "weightspace"))
import w1_t12 as T  # noqa: E402

fails = 0


def check(ok, text):
    global fails
    print(("ok   " if ok else "FAIL ") + text)
    if not ok:
        fails += 1


check(abs(T.projection(7829, 10.0) - 200 * 7829 * 10 / 3.6e6) < 1e-12 and abs(T.projection(7829, 10.0) - 4.3494) < 1e-3, "projection: 4.35 hours at 10 ms a token")
check(abs(T.projection(1000, 2.0, 100) * 2 - T.projection(1000, 2.0, 200)) < 1e-12, "projection is linear in the number of problems")
for bad in ((0, 1.0), (1, 0.0), (-1, 1.0), (1, 1.0, 0)):
    try:
        T.projection(*bad)
        ok = False
    except T.T12Error as e:
        ok = e.rule == "t12-input"
    check(ok, f"projection{bad} is refused with t12-input")
check([T.feasibility(h) for h in (0.1, 10.0, 10.01, 24.0, 24.5)] == ["feasible", "feasible", "marginal", "marginal", "not feasible"], "feasibility thresholds")
out = "job 0 n=3\nids 0 1,2\ntiming: x=1\nstep 0 5\njob 1 n=3\nids 1 4\nstep 0 6\n"
jl = T.job_lines(out)
check(len(jl) == 2 and "timing: x=1" not in jl[0] and jl[1][0] == "job 1 n=3", "job_lines groups by job and drops timing")
check(T.first_mismatch([[1, 2, 3]], [[1, 2, 3]]) is None, "equal id lists: no mismatch")
check(T.first_mismatch([[1, 2, 3], [4, 5]], [[1, 2, 3], [4, 6]]) == (1, 1), "first differing step")
check(T.first_mismatch([[1, 2]], [[1, 2, 3]]) == (0, 2), "a length difference is a mismatch at the shorter length")
check(T.first_mismatch([[1]], [[1], [2]]) == (1, 0), "a missing job is a mismatch")
err = io.StringIO()
with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
    code = T.main(["throughput", "--patch", "/no/such/patch", "--out", "/nonexistent/o.json"])
check(code == 2 and "error[t12-input]" in err.getvalue(), "the CLI refuses a missing patch with t12-input, exit 2")
sys.exit(1 if fails else 0)
