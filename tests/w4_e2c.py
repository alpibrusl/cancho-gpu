#!/usr/bin/env python3
"""docs/weight-space-w4.md section 14: the pure parts of examples/weightspace/w4_e2c.py on planted data. No model, no torch.
Needs numpy; without it this prints a skip and exits 0 (the CI runner has none).

    python3 tests/w4_e2c.py

  1. find_loop: a planted tail of period 37 and span 1700 is found (smallest period, not a multiple); a text with no repeating tail, a
     tail shorter than the minimum span and a period under 8 are not loops; cut_of refuses when 256 tokens do not remain.
  2. sample_step: the penalty removes a seen token, top-k and top-p cut the support, the uniform picks by the renormalized cumulative,
     and the same uniform gives the same token.
  3. match_prefix, per_loop, gains: planted records give the exit share, the paired gains and their order.
  4. calibration and recommend: the rule fires on planted gains in the published order and fails on a wrong order, on a near-edited
     x1e-1 and on an interval that includes 0; recommend picks the smallest alpha with half the gain and the best inside the KL gate.
  5. Records: malformed records are refused with `w4e-records`; score on a directory of planted records; the CLI exits 2 on a missing
     directory."""
import contextlib
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, os.path.join(ROOT, "examples", "weightspace"))
try:
    import numpy as np
except ImportError:
    print("skip: numpy is not installed")
    sys.exit(0)
import w4_e2c as E  # noqa: E402

fails = 0


def check(ok, text):
    global fails
    print(("ok   " if ok else "FAIL ") + text)
    if not ok:
        fails += 1


def refuses(rule, f):
    try:
        f()
    except E.E2cError as e:
        return e.rule == rule
    return False


# 1
rng = np.random.default_rng(1)
period = [int(x) for x in rng.integers(0, 1000, 37)]
ids = [int(x) for x in rng.integers(2000, 3000, 500)] + period * 46      # 1702 periodic tokens
fl = E.find_loop(ids)
check(fl is not None and fl[0] == 37 and fl[1] >= 1702, f"the planted loop of period 37 is found: {fl}")
check(E.find_loop([int(x) for x in rng.integers(0, 50000, 4000)]) is None, "random text has no loop")
check(E.find_loop([int(x) for x in rng.integers(2000, 3000, 500)] + period * 20) is None, "a tail under the minimum span is not a loop")
check(E.find_loop([5, 6] * 1500) == (8, 3000), "a period of 2 is reported as 8, the smallest period the rule allows (a multiple of the true one)")
check(E.find_loop([5, 6, 7] * 400 + [9]) is None, "a text whose tail is not periodic is not a loop")
check(E.find_loop([5, 6, 7, 8, 9, 10, 11, 12] * 300) == (8, 2400), "period 8 is accepted")
c = E.cut_of(len(ids), fl[1])
check(c == len(ids) - fl[1] + 1000 and c + 256 <= len(ids), f"the cut is 1,000 tokens into the loop: {c}")
check(E.cut_of(2000, 1200) is None, "a loop with fewer than 256 tokens after the cut gets no cut")

# 2
V = 50
z = np.zeros(V)
z[3], z[4], z[5] = 10.0, 9.0, 8.0
seen = np.zeros(V, dtype=bool)
check(E.sample_step(z, seen, 0.0) == 3, "u = 0 draws the top token")
seen[3] = True
check(E.sample_step(z, seen, 0.0) == 4, "the presence penalty (1.5) demotes a seen token below the next (gap 1.0 is under 1.5)")
draws = {E.sample_step(z, np.zeros(V, dtype=bool), u) for u in np.linspace(0, 0.9999, 400)}
check(draws <= {3, 4, 5} and 3 in draws, f"top-p 0.95 keeps the head of the distribution: {sorted(draws)}")
z2 = np.zeros(V)
z2[:30] = np.linspace(5, 4.9, 30)
d2 = {E.sample_step(z2, np.zeros(V, dtype=bool), u) for u in np.linspace(0, 0.9999, 2000)}
check(max(d2) < 20, f"top-k 20 never draws outside the 20 best: max id {max(d2)}")
check(E.sample_step(z, np.zeros(V, dtype=bool), 0.5) == E.sample_step(z, np.zeros(V, dtype=bool), 0.5), "the same uniform, the same token")
check(E.sample_step(z, np.zeros(V, dtype=bool), 0.9999999) in (3, 4, 5), "u near 1 stays inside the support")

# 3
check(E.match_prefix([1, 2, 3], [1, 2, 4]) == 2 and E.match_prefix([1, 2], [1, 2, 3]) == 2 and E.match_prefix([], [1]) == 0, "match_prefix")


def rec(i, ms, nll, nllp=None):
    return {"id": i, "nll": nll, "nll_period": nll if nllp is None else nllp, "samples": [{"m": m} for m in ms]}


base = [rec(i, [256] * 4, 0.1) for i in range(10)]
mod = [rec(i, [256, 256, 30, 256] if i < 5 else [256] * 4, 0.3) for i in range(10)]
g, ids_ = E.gains(mod, base)
check(np.isclose(g["E"].mean(), 0.125) and np.allclose(g["N"], 0.2) and len(ids_) == 10, f"paired gains: E {g['E'].mean():.3f}, N {g['N'].mean():.3f}")
pl = E.per_loop(mod)
check(pl[0][0] == 0.25 and pl[9][0] == 0.0 and pl[0][4] == 0.25, "per_loop: exit share and the share leaving within 64")
lo, hi = E.bootstrap(g["E"])
check(lo <= 0.125 <= hi and E.bootstrap(g["E"]) == (lo, hi), "bootstrap interval contains the mean and is reproducible")

# 4
good = {"edited": (0.26, 0.1, 0.4), "e19": (0.16, 0.0, 0.3), "x1e-1": (0.01, -0.1, 0.1)}
ok, cl = E.calibration(good, "E")
check(ok and len(cl) == 3, "the published order (+26, +16, +1) passes")
check(not E.calibration({**good, "e19": (0.30, 0.0, 0.4)}, "E")[0], "e19 above the edit fails clause a")
check(not E.calibration({**good, "x1e-1": (0.12, 0.0, 0.2)}, "E")[0], "x1e-1 above half of e19 fails clause b")
check(not E.calibration({**good, "x1e-1": (-0.1, -0.2, 0.0)}, "E")[0], "x1e-1 far below the base fails clause b")
check(not E.calibration({**good, "edited": (0.26, -0.01, 0.5)}, "E")[0], "an interval that includes 0 fails clause c")
G = {"edited": 0.20, "a1": 0.02, "a3": 0.11, "a6": 0.19, "a9": 0.25}
check(E.recommend(G)[:2] == ("a3", "a6"), f"R1 = smallest alpha with half the gain (a3), R2 = best inside the KL gate (a6, not a9): {E.recommend(G)[:2]}")
G2 = {"edited": 0.20, "a1": 0.01, "a3": 0.02, "a6": 0.03, "a9": 0.04}
check(E.recommend(G2)[:2] == ("a9", "a6"), "when no alpha reaches half the edit, the two largest gains")
check(refuses("w4e-records", lambda: E.recommend({"edited": 1.0, "a1": 0.5})), "fewer than two alphas is refused")

# 5
check(refuses("w4e-records", lambda: E.check_record({"id": 1})), "a record without fields is refused")
check(refuses("w4e-records", lambda: E.check_record(rec(1, [300], 0.1))), "a match prefix over 256 is refused")
check(refuses("w4e-records", lambda: E.check_record(rec(1, [3], float("nan")))), "a NaN NLL is refused")
check(refuses("w4e-records", lambda: E.check_record(rec(1, [], 0.1))), "no samples is refused")
with tempfile.TemporaryDirectory() as d:
    def put(name, rows):
        with open(os.path.join(d, f"records_{name}.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
    check(refuses("w4e-records", lambda: E.score(d)), "a directory without a base is refused")
    put("base", base)
    for name, share, nll in (("edited", 8, 0.5), ("e19", 5, 0.4), ("x1e-1", 0, 0.1), ("a1", 0, 0.11), ("a3", 5, 0.2), ("a6", 7, 0.45), ("a9", 9, 0.6)):
        put(name, [rec(i, [10 if i < share else 256] * 4, nll + 0.001 * i) for i in range(10)])
    res = E.score(d)
    check(res["informative"] and res["metric"] == "E", f"planted records in the published order are informative on E: {res['calibration']['E']['passes']}")
    check(res["recommend"]["R1"] == "a3" and res["recommend"]["R2"] == "a6", f"recommendation {res['recommend']}")
    check("| edited |" in E.render(res), "render prints a row a model")
    with open(os.path.join(d, "records_bad.jsonl"), "w") as f:
        f.write("{not json\n")
    check(refuses("w4e-records", lambda: E.score(d)), "an unreadable records file is refused")
err = io.StringIO()
with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
    code = E.main(["score", "--dir", "/no/such/directory"])
check(code == 2 and "w4e-" not in "" and "error[w4e-records]" in err.getvalue(), f"the CLI exits 2 with a tagged refusal on a missing directory: {err.getvalue().strip()[:80]}")
# no input reaches a panic: random bytes as a records file
with tempfile.TemporaryDirectory() as d:
    r = np.random.default_rng(7)
    bad = 0
    for _ in range(60):
        with open(os.path.join(d, "records_base.jsonl"), "wb") as f:
            f.write(bytes(r.integers(0, 256, int(r.integers(1, 200)), dtype=np.uint8)))
        try:
            E.score(d)
            bad += 1
        except E.E2cError:
            pass
    check(bad == 0, "60 random byte strings as a records file: only tagged refusals")
sys.exit(1 if fails else 0)
