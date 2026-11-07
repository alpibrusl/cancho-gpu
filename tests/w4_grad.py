#!/usr/bin/env python3
"""docs/weight-space-w4-r4.md: examples/weightspace/w4_grad_lib.py and w4_grad.py on planted cases and a toy model. No download, no GPU.
Needs numpy; without it this prints a skip and exits 0 (the CI runner has none). The torch part runs only when torch is importable.

    python3 tests/w4_grad.py

  1. Directions: a planted rank-1 gradient is found by GU1 and C1, a planted mean by GU2, a planted dominant axis by GU3; sign never matters.
  2. Columns: C1 to C4 find a planted support; the edit forms P and G match their formulas entry by entry and hit the target norm.
  3. The nulls: derangements have no fixed point and are a pure function of the seed; sign flips would not be a null for a rank-1 signal.
  4. The split, the sums, the table (minimum over halves) and the rule on hand-made tables; FOUND and SPECIFIC.
  5. Refusals with their rule tags; random garbage never raises anything but the tagged refusal.
  6. (torch) the capture on a toy model with the layer layout of the real one: the weight gradient equals a finite difference, the hook sums
     equal the autograd gradient, the log-odds loss equals its definition, the layer finder skips a vision tower, bad pairs are refused."""
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
import w4_search_lib as S  # noqa: E402

fails = 0


def expect(name, ok, detail=""):
    global fails
    print("%s %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  " + str(detail)))
    fails += 0 if ok else 1


def raises(rule, fn):
    try:
        fn()
    except G.GradError as e:
        return e.rule == rule or "wrong rule " + e.rule
    except Exception as e:   # anything else is a panic
        return "unexpected %s: %s" % (type(e).__name__, e)
    return "no refusal"


rng = np.random.default_rng(11)
O, I, K = 24, 80, 8
u0 = S.unit(rng.normal(size=O))
cols0 = np.sort(rng.choice(I, K, replace=False))


def planted(strength=5.0, noise=0.2, seed=0):
    """A per-loop gradient: strength * u0 (v^T) on cols0 plus noise."""
    r = np.random.default_rng(seed)
    v = np.zeros(I)
    v[cols0] = np.sign(r.normal(size=K)) * (1.0 + np.abs(r.normal(size=K)))
    return (strength * np.outer(u0, v) + noise * r.normal(size=(O, I))).astype(np.float32)


def sums_of(gs, extra_m=None, extra_axis=None, ax_strength=0.0):
    s = G.Sums(O, I)
    for j, g in enumerate(gs):
        m = g.sum(1).astype(np.float64) if extra_m is None else extra_m
        q = np.zeros((O, O)) if extra_axis is None else ax_strength * np.outer(extra_axis, extra_axis)
        q = q + g.astype(np.float64) @ g.T.astype(np.float64)
        s.add(g, m, q, np.ones(I) * 4.0, 4)
    return s


# ---- 1. directions
gs = [planted(seed=j) for j in range(6)]
s = sums_of(gs)
d = G.directions(s)
expect("1a GU1 finds the planted rank-1 gradient", S.abscos(d["GU1"], u0) > 0.99, S.abscos(d["GU1"], u0))
expect("1b every direction is a unit vector", all(abs(np.linalg.norm(v) - 1) < 1e-9 for v in d.values()))
expect("1c the sign is not identified", abs(S.abscos(-d["GU1"], u0) - S.abscos(d["GU1"], u0)) < 1e-12)
s2 = sums_of(gs, extra_m=u0 * 3.0)
expect("1d GU2 is the normalised sum of the output gradients", S.abscos(G.directions(s2)["GU2"], u0) > 0.999)
ax = S.unit(rng.normal(size=O))
s3 = sums_of(gs, extra_axis=ax, ax_strength=1e6)
expect("1e GU3 follows a dominant second-moment axis while GU1 still finds the gradient",
       S.abscos(G.directions(s3)["GU3"], ax) > 0.99 and S.abscos(G.directions(s3)["GU1"], u0) > 0.99)

# ---- 2. columns and forms
w = rng.normal(size=(O, I))
rules = G.column_rules(s, w, u0, K)
expect("2a C1 and C2 find the planted support", set(rules["C1"]) == set(cols0) and set(rules["C2"]) == set(cols0), (rules["C1"], cols0))
expect("2b every rule returns K sorted distinct columns", all(len(r) == K and (np.diff(r) > 0).all() for r in rules.values()))
w2 = w.copy()
w2[:, cols0] += 8.0 * np.outer(u0, np.sign(rng.normal(size=K)))
expect("2c C3 finds columns of W aligned with u", set(G.column_rules(s, w2, u0, K)["C3"]) == set(cols0))
expect("2d the share of the gradient along u is near 1 for the planted one and near the chance for a random one",
       G.along(s.g, u0) > 0.95 and G.along(s.g, S.unit(rng.normal(size=O))) < 0.5)
tn = 2.5
ep = G.edit_p(w, u0, cols0, tn)
expect("2e form P: norm, support, formula", abs(np.linalg.norm(ep) - tn) < 1e-9 and set(np.flatnonzero(np.abs(ep).sum(0))) == set(cols0)
       and np.allclose(ep[:, cols0], -(tn / np.linalg.norm(np.outer(u0, u0 @ w[:, cols0]))) * np.outer(u0, u0 @ w[:, cols0])))
g64 = s.g.astype(np.float64)
eg = G.edit_g(g64, u0, cols0, tn)
expect("2f form G: norm, support, formula; descent on the loss (first-order change is negative)",
       abs(np.linalg.norm(eg) - tn) < 1e-9 and set(np.flatnonzero(np.abs(eg).sum(0))) == set(cols0) and float((g64 * eg).sum()) < 0)
expect("2g the forms do not depend on the sign of u", np.allclose(G.edit_p(w, -u0, cols0, tn), ep) and np.allclose(G.edit_g(g64, -u0, cols0, tn), eg))
expect("2h the published form with the true u and columns has edit cosine 1", abs(G.edit_cosine(ep, G.edit_p(w, u0, cols0, 7.0)) - 1) < 1e-12)
ur = G.random_unit(O, 15)
expect("2i the random control is a fixed unit vector per layer", abs(np.linalg.norm(ur) - 1) < 1e-12 and np.allclose(ur, G.random_unit(O, 15))
       and not np.allclose(ur, G.random_unit(O, 19)))

# ---- 3. nulls
dg = G.derangement(30, 5)
expect("3a a derangement is a permutation with no fixed point, a pure function of the seed",
       sorted(dg) == list(range(30)) and not (dg == np.arange(30)).any() and np.array_equal(dg, G.derangement(30, 5)) and not np.array_equal(dg, G.derangement(30, 6)))
expect("3b a derangement of one loop is refused", raises("w4g-loops", lambda: G.derangement(1, 0)) is True)
# a rank-1 direction shared by the loops survives sign flips (why they are not the null), and a mismatched loss is a different gradient
signed = [(-1) ** j * g for j, g in enumerate(gs)]
expect("3c sign flips of rank-1 gradients return the same direction (the reason for the target permutation)",
       S.abscos(G.directions(sums_of(signed))["GU1"], u0) > 0.95)
mismatch = [planted(strength=5.0, seed=50 + j) for j in range(6)]
r2 = np.random.default_rng(9)
u1 = S.unit(r2.normal(size=O))
mm = [np.outer(5.0 * u1, g[cols0[0]:cols0[0] + 1].ravel()[:1].repeat(I) + 1.0).astype(np.float32) for g in mismatch]
expect("3d an estimator fed gradients along another direction does not return u0", S.abscos(G.directions(sums_of(mm))["GU1"], u0) < 0.9)
ux = S.abscos(rng.normal(size=1000), rng.normal(size=1000))
expect("3e random unit vectors in 1,024 dimensions have a small |cos| with any direction", float(S.random_cosines(rng.normal(size=1024), 200, G.SEED).max()) < 0.2 and ux < 0.2)

# ---- 4. split, table, rule, behaviour
tr, te = G.split_loops(7)
expect("4a split: even positions train, odd test, a partition", list(tr) == [0, 2, 4, 6] and list(te) == [1, 3, 5])
expect("4b a pair's table entry is the minimum over the halves",
       G.merge_halves({"train": {15: {"rows": {"GU1/p/C1": {"cos": 0.9, "overlap": 50}}}}, "test": {15: {"rows": {"GU1/p/C1": {"cos": 0.6, "overlap": 40}}}}},
                      (15,), ("GU1",), ("C1",), ("p",)) == {("GU1/p", "C1"): {15: (0.6, 40)}})
tab = {("GU1/p", "C1"): {15: (0.95, 50), 19: (0.92, 49)}}
bt = {("GU1/p", 15): True, ("GU1/p", 19): True}
expect("4c RECOVERED on the registered thresholds", G.decide(tab, bt, (15, 19))["status"] == "RECOVERED")
tab2 = {("GU1/p", "C1"): {15: (0.95, 50), 19: (0.92, 47)}}
expect("4d 47 of 64 in one layer is not RECOVERED", G.decide(tab2, bt, (15, 19))["status"] != "RECOVERED")
tab3 = {("GU1/p", "C1"): {15: (0.75, 5), 19: (0.72, 5)}}
expect("4e PARTIAL (A): cos >= 0.7 in both layers and beats the null", G.decide(tab3, bt, (15, 19))["status"] == "PARTIAL")
expect("4f a direction that does not beat its null is NOT RECOVERED", G.decide(tab3, {("GU1/p", 15): True, ("GU1/p", 19): False}, (15, 19))["status"] == "NOT RECOVERED")
tab4 = {("GU1/p", "C1"): {15: (0.55, 35), 19: (0.52, 33)}}
expect("4g PARTIAL (B): cos >= 0.5 and overlap >= 32", G.decide(tab4, bt, (15, 19))["status"] == "PARTIAL")
tab5 = {("GU1/p", "C1"): {15: (0.3, 3), 19: (0.2, 1)}}
expect("4h NOT RECOVERED on low numbers", G.decide(tab5, bt, (15, 19))["status"] == "NOT RECOVERED")
nm = {("GU1/p", 15): 0.4, ("GU1/p", 19): 0.4}
expect("4i beats: the best |cos| of the direction against its null maximum", G.beats_of(tab4, nm, (15, 19)) == {("GU1/p", 15): True, ("GU1/p", 19): True}
       and G.beats_of(tab5, nm, (15, 19)) == {("GU1/p", 15): False, ("GU1/p", 19): False})
expect("4j FOUND: E >= 0.5 and KL upper bound <= 0.05", G.found(0.5, 0.05) and not G.found(0.49, 0.01) and not G.found(0.9, 0.0501))
expect("4k SPECIFIC: the control's E at most half the candidate's", G.specific(0.8, 0.4) and not G.specific(0.8, 0.41))

# ---- 5. refusals
expect("5a a zero gradient has no singular direction", raises("w4g-degenerate", lambda: G.top_left_sv(np.zeros((O, I)))) is True)
expect("5b NaN in a gradient is refused", raises("w4g-finite", lambda: G.Sums(O, I).add(np.full((O, I), np.nan), np.zeros(O), np.zeros((O, O)), np.zeros(I), 1)) is True)
expect("5c a wrong shape is refused", raises("w4g-shape", lambda: G.Sums(O, I).add(np.zeros((O, I + 1)), np.zeros(O), np.zeros((O, O)), np.zeros(I), 1)) is True)
expect("5d the split of too few loops is refused", raises("w4g-loops", lambda: G.split_loops(3)) is True)
expect("5e a non-positive target norm is refused", raises("w4g-scale", lambda: G.edit_p(w, u0, cols0, 0.0)) is True)
expect("5f duplicate or out-of-range columns are refused", raises("w4g-columns", lambda: G.edit_p(w, u0, [1, 1], 1.0)) is True and raises("w4g-columns", lambda: G.edit_p(w, u0, [I], 1.0)) is True)
expect("5g an empty sum has no rms", raises("w4g-loops", lambda: G.Sums(O, I).rms()) is True)
expect("5h a derangement of fewer than two loops is refused", raises("w4g-loops", lambda: G.derangement(0, 1)) is True)
expect("5i a degenerate edit (u orthogonal to every selected column of W) is refused", raises("w4g-degenerate", lambda: G.edit_p(np.zeros((O, I)), u0, cols0, 1.0)) is True)
bad, r = 0, np.random.default_rng(3)
for _ in range(80):
    x = r.choice([np.nan, np.inf, 0.0, 1.0, -1.0], size=(int(r.integers(1, 5)), int(r.integers(1, 5))))
    for fn in (lambda: G.top_left_sv(x), lambda: G.top_eigvec(x), lambda: G.unit(x.ravel()), lambda: G.edit_g(x, x[:, 0], [0], 1.0),
               lambda: G.column_rules(sums_of([planted()]), x, x[:, 0], 2)):
        try:
            fn()
        except G.GradError:
            pass
        except Exception as e:   # noqa: BLE001
            bad += 1
            print("   panic:", type(e).__name__, e)
expect("5j random small matrices with NaN, inf, zeros: only tagged refusals", bad == 0, bad)

# ---- 6. the capture on a toy model (torch)
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("skip: torch is not installed (the capture part)")
    sys.exit(1 if fails else 0)
import w4_grad as W  # noqa: E402

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
V, D, F, NL = 11, 6, 9, 4


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(D, F, bias=False)
        self.mlp.down_proj = nn.Linear(F, D, bias=False)

    def forward(self, x):
        return x + self.mlp.down_proj(torch.tanh(self.mlp.up_proj(x)))


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.layers = nn.ModuleList([Block() for _ in range(NL)])


class Out:
    def __init__(self, logits):
        self.logits = logits


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList([Block() for _ in range(NL + 3)])   # longer, but a vision tower
        self.lm_head = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, use_cache=False, logits_to_keep=0):
        x = self.model.embed(input_ids)
        for b in self.model.layers:
            x = b(x)
        lg = self.lm_head(x)
        return Out(lg[:, -logits_to_keep:] if logits_to_keep else lg)


toy = Toy()
ids = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
first = 5
layers = W.find_layers(toy)
expect("6a the layer finder takes the decoder layers and skips a longer vision tower", len(layers) == NL and layers is toy.model.layers)
pairs = [(1, "mlp.down_proj"), (3, "mlp.down_proj")]
res = W.grad_job(toy, layers, pairs, ids, first, ("p", "o"))


def loss_of(kind):
    lg = toy(torch.tensor([ids]), use_cache=False).logits[0, first:-1]
    t = torch.tensor(ids[first + 1:])
    lp = torch.log_softmax(lg, -1)[torch.arange(len(t)), t]
    return lp.mean() if kind == "p" else (lp - torch.log1p(-lp.exp())).mean()


for key in pairs:
    for kind in ("p", "o"):
        wt = layers[key[0]].mlp.down_proj.weight
        g = res[kind][key]["g"]
        fd = np.zeros(3)
        idx = [(0, 0), (2, 5), (5, 8)]
        for n, (i, j) in enumerate(idx):
            h = 1e-6
            with torch.no_grad():
                wt[i, j] += h
                lp_ = float(loss_of(kind))
                wt[i, j] -= 2 * h
                lm_ = float(loss_of(kind))
                wt[i, j] += h
            fd[n] = (lp_ - lm_) / (2 * h)
        got = np.array([g[i, j] for i, j in idx])
        expect(f"6b layer {key[0]} loss {kind}: the weight gradient equals a finite difference", np.allclose(got, fd, rtol=1e-3, atol=1e-6), (got, fd))
        expect(f"6c layer {key[0]} loss {kind}: the hook-side sum g^T a equals the autograd gradient", res[kind][key]["check"] < 1e-5, res[kind][key]["check"])
        expect(f"6d layer {key[0]} loss {kind}: Q is g^T g (symmetric PSD), m its row sum, n the positions", np.allclose(res[kind][key]["q"], res[kind][key]["q"].T)
               and np.linalg.eigvalsh(res[kind][key]["q"]).min() > -1e-9 and res[kind][key]["n"] == len(ids) and res[kind][key]["m"].shape == (D,))
expect("6e the reported loss equals the definition", abs(res["p"][pairs[0]]["loss"] - float(loss_of("p"))) < 1e-6 and abs(res["o"][pairs[0]]["loss"] - float(loss_of("o"))) < 1e-6)
expect("6f the model's parameters end without gradient requirement", not any(p.requires_grad for p in toy.parameters()))
expect("6g the first layers of the pair see a gradient only through the later ones: layer 1's input a has all positions", res["p"][pairs[0]]["r"].shape == (F,))


def ref(fn):
    try:
        fn()
    except G.GradError as e:
        return e.rule
    except Exception as e:   # noqa: BLE001
        return "panic " + type(e).__name__
    return "no refusal"


expect("6h a layer out of range is refused", ref(lambda: W.module_of(layers, 9, "mlp.down_proj")) == "w4g-pair")
expect("6i a module path that does not exist is refused", ref(lambda: W.module_of(layers, 0, "mlp.nothing")) == "w4g-pair")
expect("6j a pair that is not layer:module is refused", ref(lambda: W.parse_pairs(["15-mlp"])) == "w4g-pair" and ref(lambda: W.parse_pairs([])) == "w4g-pair")
expect("6k a job with nothing to score is refused", ref(lambda: W.grad_job(toy, layers, pairs, ids, len(ids) - 1)) == "w4g-job")
expect("6l a model without decoder layers is refused", ref(lambda: W.find_layers(nn.Linear(2, 2))) == "w4g-model")
import io
import contextlib
err = io.StringIO()
with contextlib.redirect_stderr(err):
    code = W.main(["analyse", "--work", "/no/such/dir", "--out", "/tmp/x.json"])
expect("6m the CLI exits 2 with a tagged refusal on a missing work directory", code == 2 and "error[w4g-loops]" in err.getvalue(), err.getvalue()[:100])
sys.exit(1 if fails else 0)
