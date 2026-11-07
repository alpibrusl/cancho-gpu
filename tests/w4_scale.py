#!/usr/bin/env python3
"""docs/weight-space-w4-r4.md section 9: examples/weightspace/w4_scale.py's pure parts on hand-made tables and planted edits. No model, no GPU.
Needs numpy; without it this prints a skip and exits 0.

    python3 tests/w4_scale.py

  1. The grid: 18 configurations in a fixed order, bad fractions and budgets refused.
  2. The edit: (1, all columns) is edit_p itself; the norm is f times the full norm for the top-k only; the columns are the ranked top k.
  3. The selection: largest train E, ties by lower KL bound, then smaller f, then smaller k; eligibility is the KL bound <= 0.05.
  4. The verdict: the three clauses, each alone failing; the control at the planted configuration.
  5. Refusals carry their rule tags and nothing else escapes."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, os.path.join(ROOT, "examples", "weightspace"))
try:
    import numpy as np
except ImportError:
    print("skip: numpy is not installed")
    sys.exit(0)
import w4_grad_lib as G  # noqa: E402
import w4_scale as C  # noqa: E402

fails = 0


def expect(name, ok, detail=""):
    global fails
    print("%s %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  " + str(detail)))
    fails += 0 if ok else 1


def rule(fn):
    try:
        fn()
    except (C.ScaleError, G.GradError) as e:
        return e.rule
    except Exception as e:   # anything else is a panic
        return "unexpected %s: %s" % (type(e).__name__, e)
    return "no refusal"


cf = C.configs()
expect("1a the grid has 18 configurations, budgets outer", len(cf) == 18 and cf[0] == (0.1, 16) and cf[-1] == (1.0, 64) and len(set(cf)) == 18)
expect("1b names are unique and stable", len({C.config_name(f, k) for f, k in cf}) == 18 and C.config_name(0.5, 16) == "f0.5_k16")
expect("1c a zero budget or a negative fraction is refused", rule(lambda: C.configs(budgets=(0,))) == "w4s-config" and rule(lambda: C.configs(fractions=(-1.0,))) == "w4s-config")

rng = np.random.default_rng(5)
O, I = 16, 40
w = rng.normal(size=(O, I))
g = rng.normal(size=(O, I))
u = G.unit(rng.normal(size=O))
cols = np.sort(rng.choice(I, 12, replace=False))
ranked = C.rank_columns(g, cols)
nrm = np.linalg.norm(g[:, ranked], axis=0)
expect("2a ranking is by descending column norm over the same set", set(ranked) == set(cols) and np.all(np.diff(nrm) <= 0))
full = G.edit_p(w, u, cols, 3.0)
e11 = C.edit_fk(w, u, ranked, 3.0, 1.0, 12)
expect("2b (1, all) is edit_p", np.abs(full - e11).max() < 1e-12, np.abs(full - e11).max())
e5 = C.edit_fk(w, u, ranked, 3.0, 0.5, 5)
nz = np.flatnonzero(np.any(e5 != 0, axis=0))
expect("2c the top-k columns only, at f times the full coefficient", set(nz) == set(ranked[:5]) and np.allclose(e5[:, nz], 0.5 * full[:, nz]))
expect("2d the norm shrinks as f and the kept share", abs(np.linalg.norm(e5) - 0.5 * np.linalg.norm(full[:, ranked[:5]])) < 1e-12)
ur = G.random_unit(O, 15)
ce = C.ctrl_edit(e5, w, ur)
expect("2e the control has the candidate's columns and norm and a different direction",
       set(np.flatnonzero(np.any(ce != 0, axis=0))) == set(nz) and abs(np.linalg.norm(ce) - np.linalg.norm(e5)) < 1e-9 and G.edit_cosine(ce, e5) < 0.9)

kl_hi = {"a": 0.03, "b": 0.05, "c": 0.0501, "d": 0.01}
expect("3a eligibility is <= 0.05", C.eligible(kl_hi) == ["a", "b", "d"])
fk = {"a": (0.2, 64), "b": (0.5, 32), "d": (0.1, 16)}
expect("3b the largest train E wins", C.select({"a": 0.5, "b": 0.9, "d": 0.2}, kl_hi, fk) == "b")
expect("3c a tie goes to the lower KL bound", C.select({"a": 0.9, "b": 0.9, "d": 0.2}, kl_hi, fk) == "a")
kl2 = {"a": 0.03, "b": 0.03, "d": 0.03}
expect("3d then the smaller f, then the smaller k", C.select({"a": 1.0, "b": 1.0, "d": 1.0}, kl2, fk) == "d"
       and C.select({"a": 1.0, "b": 1.0}, kl2, {"a": (0.2, 64), "b": (0.2, 32)}) == "b")
expect("3e an empty set is refused", rule(lambda: C.select({}, kl_hi, fk)) == "w4s-none")
expect("3f a NaN is refused", rule(lambda: C.select({"a": float("nan")}, kl_hi, fk)) == "w4s-input" and rule(lambda: C.eligible({"a": float("nan")})) == "w4s-finite")

v = C.verdict(0.9, 0.04, 0.0)
expect("4a all three clauses give FOUND", v["found"] and v["effect"] and v["kl"] and v["specific"])
expect("4b each clause alone fails the verdict", not C.verdict(0.49, 0.04, 0.0)["found"] and not C.verdict(0.9, 0.0501, 0.0)["found"] and not C.verdict(0.9, 0.04, 0.1)["found"])
expect("4c a non-finite input is refused", rule(lambda: C.verdict(float("inf"), 0.04, 0.0)) == "w4s-finite")
expect("5a bad budget and fraction in edit_fk are refused", rule(lambda: C.edit_fk(w, u, ranked, 3.0, 0.5, 13)) == "w4s-config"
       and rule(lambda: C.edit_fk(w, u, ranked, 3.0, 0.0, 4)) == "w4s-config")
expect("5b an empty candidate has no control", rule(lambda: C.ctrl_edit(np.zeros((O, I)), w, ur)) == "w4s-degenerate")
sys.exit(1 if fails else 0)
