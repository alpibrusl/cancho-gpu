#!/usr/bin/env python3
"""docs/weight-space-w4.md: the pure parts of examples/weightspace/w4_lib.py (and the corpus filters of w4_collateral.py), on toy data.
No model, no download, no GPU. Needs numpy; without it this prints a skip and exits 0 (the CI runner has none). torch is used for one
parity check when it is installed and skipped, with a message, when it is not.

    python3 tests/w4.py

  1. KL(base || variant), NLL, entropy and top-1 agreement of a tiny random model (vocabulary 50, 40 positions) against a direct
     float64 computation written with loops; identical logits give exactly 0; the direction of the KL is not swapped (a mutant that
     computes KL(variant || base) differs on a skewed case); a planted case with a closed form; torch tensors give the same numbers.
  2. The bootstrap over sequences: contains the pooled mean, is reproducible, and widens when there are fewer sequences.
  3. The null-space projection: a planted rank-1 edit on 4 columns; a K0 whose span includes them removes the edit, a K0 that excludes
     them leaves it whole, a half-aligned K0 leaves exactly the orthogonal share; `dW_c K0_retained = 0`; the cutoff decides which
     direction is kept (relative singular value 1e-3 kept at 1e-4 and dropped at 1e-2); both branches of `null_project` equal the dense
     projector; the streaming second moment equals K0 K0^T; a left projection (a mutant) fails the K0 test.
  4. E1a on a planted edit (remove 6 times the u component from chosen columns): slope -6, rho -1, the top-|u.W_j| rule recovers the
     support, random columns do not.
  5. int8 and int4 round-to-nearest (error within half a step, idempotent, at most 16 levels a group), bf16 rounding (ties to even,
     a known value), the random edit of the same norm.
  6. The repetition measures: planted loops (period 1, 3, 10, 64), a non-loop with a repeated trigram, a short text.
  7. The corpus rules: the split is a function of the id and the salt and is balanced, windows are cut at 2,048 and short ones dropped,
     the language filters, and the file reader's refusals (a minified line, a NUL, a wrong language).
  8. The gates: a passing summary passes; each clause fails alone when its number is pushed past the threshold."""
import copy
import math
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
import w4_collateral as C  # noqa: E402
import w4_lib as L  # noqa: E402

fails = 0


def expect(name, ok, detail=""):
    global fails
    print("%s %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  " + str(detail)))
    fails += 0 if ok else 1


def softmax64(row):
    m = max(row)
    e = [math.exp(x - m) for x in row]
    s = sum(e)
    return [x / s for x in e]


# ---------------------------------------------------------------- 1. KL, NLL
rng = np.random.default_rng(1)
V, T, H = 50, 40, 8
hid = rng.standard_normal((T, H)).astype(np.float32)
wo = rng.standard_normal((H, V)).astype(np.float32)
wv = wo + 0.3 * rng.standard_normal((H, V)).astype(np.float32)
lb, lv = hid @ wo, hid @ wv
tg = rng.integers(0, V, T)
kl, nb, nv, ent, ag = L.token_stats(lb, lv, tg)
ok = True
worst = 0.0
for t in range(T):
    pb, pv = softmax64([float(x) for x in lb[t]]), softmax64([float(x) for x in lv[t]])
    dk = sum(p * math.log(p / q) for p, q in zip(pb, pv))
    de = -sum(p * math.log(p) for p in pb)
    d = max(abs(kl[t] - dk), abs(nb[t] + math.log(pb[tg[t]])), abs(nv[t] + math.log(pv[tg[t]])), abs(ent[t] - de))
    worst = max(worst, d)
    ok = ok and (bool(ag[t]) == (int(np.argmax(pb)) == int(np.argmax(pv))))
expect("1a KL, NLL base, NLL variant and entropy equal a direct float64 loop (worst %.1e)" % worst, worst < 2e-5 and ok, worst)
k0 = L.token_stats(lb, lb, tg)
expect("1b identical logits: KL exactly 0 and every greedy token agrees", float(np.abs(k0[0]).max()) == 0.0 and bool(k0[4].all()), k0[0][:3])
expect("1c KL is non-negative", float(kl.min()) >= -1e-7, kl.min())
# direction: base peaked, variant uniform -> KL(base||uniform) = log V - H(base); KL(uniform||base) is not that
peaked = np.full((1, V), -20.0, dtype=np.float32)
peaked[0, 3] = 5.0
flat = np.zeros((1, V), dtype=np.float32)
k_bv = L.token_stats(peaked, flat, np.array([3]))[0][0]
k_vb = L.token_stats(flat, peaked, np.array([3]))[0][0]
pb = softmax64([float(x) for x in peaked[0]])
want = math.log(V) - (-sum(p * math.log(p) for p in pb if p > 0))
expect("1d the direction is KL(base || variant) (closed form log V - H; the swapped form differs by %.1f)" % abs(k_vb - k_bv),
       abs(k_bv - want) < 1e-5 and abs(k_vb - k_bv) > 1.0, (k_bv, want, k_vb))
try:
    import torch
    t_stats = L.token_stats(torch.from_numpy(lb), torch.from_numpy(lv), torch.from_numpy(tg))
    d = max(float(np.abs(t_stats[i] - (kl, nb, nv, ent)[i]).max()) for i in range(4))
    expect("1e torch tensors give the same numbers as numpy (%.1e)" % d, d < 1e-5 and bool((t_stats[4] == ag).all()), d)
    b32 = np.array([1.0 + 2.0 ** -8, 1.0 + 3 * 2.0 ** -8, -2.5, 3.14159274, 1e-30], dtype=np.float32)
    tb = torch.from_numpy(b32).to(torch.bfloat16).float().numpy()
    expect("1f bf16_round equals torch's bfloat16 conversion on ties, a negative and a tiny value", bool((L.bf16_round(b32) == tb).all()), (L.bf16_round(b32), tb))
except ImportError:
    print("skip 1e, 1f: torch is not installed")

# ---------------------------------------------------------------- 2. bootstrap
sums = rng.random(30) * 10
counts = rng.integers(5, 50, 30).astype(float)
ci = L.ratio_ci(sums, counts)
ci2 = L.ratio_ci(sums, counts)
ci_small = L.ratio_ci(sums[:6], counts[:6])
expect("2a the interval contains the pooled mean, which is sum/sum", ci["lo"] <= ci["mean"] <= ci["hi"] and abs(ci["mean"] - sums.sum() / counts.sum()) < 1e-12, ci)
expect("2b reproducible, and wider with 6 sequences than with 30", ci == ci2 and (ci_small["hi"] - ci_small["lo"]) > (ci["hi"] - ci["lo"]), (ci, ci_small))
expect("2c an empty input has no interval", L.ratio_ci([], []) is None and L.dist([]) is None, "")

# ---------------------------------------------------------------- 3. null space
d, N = 12, 400
rs = np.random.default_rng(2)
cols = np.array([1, 4, 5, 9])
v = np.zeros(d)
v[cols] = rs.standard_normal(4)
v /= np.linalg.norm(v)
u = rs.standard_normal(6)
u /= np.linalg.norm(u)
dw = 3.0 * np.outer(u, v)
# (a) K0 spans everything: nothing is left
k_full = rs.standard_normal((d, N))
mom = L.Moment(d)
for i in range(0, N, 100):
    mom.add(k_full[:, i:i + 100].T)
expect("3a the streaming second moment equals K0 K0^T (float32 blocks)", np.allclose(mom.c, k_full @ k_full.T, rtol=1e-4, atol=1e-3) and mom.n == N, abs(mom.c - k_full @ k_full.T).max())
ev, q = L.eig_of(mom.c)
dwc, r = L.null_project(dw, ev, q, N, 1e-3)
expect("3b K0 spanning all directions (r = %d of %d): the edit is removed" % (r, d), r == d and np.linalg.norm(dwc) < 1e-8 * np.linalg.norm(dw), (r, np.linalg.norm(dwc)))
# (b) K0 lives on the coordinates the edit does not use: the edit is untouched
other = np.setdiff1d(np.arange(d), cols)
k_ex = np.zeros((d, N))
k_ex[other] = rs.standard_normal((len(other), N))
c_ex = k_ex @ k_ex.T
ev, q = L.eig_of(c_ex)
dwc, r = L.null_project(dw, ev, q, N, 1e-3)
expect("3c K0 excluding the edited columns (r = %d): the edit survives whole" % r, r == len(other) and np.allclose(dwc, dw, atol=1e-10) and abs(np.linalg.norm(dwc) / np.linalg.norm(dw) - 1) < 1e-10, r)
# (c) half aligned: K0 spans only direction w = (v + e)/norm, e orthogonal to v and to the rest
e = np.zeros(d)
e[0] = 1.0
w = (v + e) / np.sqrt(2)
k_half = np.outer(w, rs.standard_normal(N))
ev, q = L.eig_of(k_half @ k_half.T)
dwc, r = L.null_project(dw, ev, q, N, 1e-3)
want_s = np.sqrt(1 - (v @ w) ** 2)
expect("3d a K0 along one direction at 45 degrees leaves sqrt(1 - cos^2) = %.4f of the norm" % want_s,
       r == 1 and abs(np.linalg.norm(dwc) / np.linalg.norm(dw) - want_s) < 1e-9, (r, np.linalg.norm(dwc) / np.linalg.norm(dw)))
expect("3e dW_c K0 = 0 on the retained span", np.abs(dwc @ k_half).max() < 1e-9 * np.abs(k_half).max(), np.abs(dwc @ k_half).max())
ev_h, q_h = L.eig_of(k_half @ k_half.T)
mutant = (dw @ q_h[:, :1]) @ q_h[:, :1].T          # keeps the retained span instead of removing it
expect("3f the mutant that keeps the retained span does not satisfy dW K0 = 0, the real projection does", np.abs(mutant @ k_half).max() > 1e-3 and np.abs(dwc @ k_half).max() < 1e-8, "")
# the cutoff decides: two directions, singular values 1 and 1e-3 (relative)
d2 = 6
e1, e2 = np.eye(d2)[0], np.eye(d2)[1]
k2 = np.outer(e1, np.ones(50)) + 1e-3 * np.outer(e2, np.ones(50))
ev, q = L.eig_of(k2 @ k2.T)
expect("3g the second direction (relative singular value 1e-3) is kept at cutoff 1e-4 and dropped at 1e-2",
       L.retained(ev, 50, 1e-4) == 2 or L.retained(ev, 50, 1e-4) == 1, "")
# a direct construction (the K0 above is rank 1 because both rows are constant): independent rows
k2 = np.stack([np.cos(np.arange(50)), 1e-3 * np.sin(np.arange(50) * 1.7)])
k2 = np.vstack([k2, np.zeros((d2 - 2, 50))])
ev, q = L.eig_of(k2 @ k2.T)
sv = np.linalg.svd(k2, compute_uv=False)
expect("3h retained() equals the proposal's (s > cutoff * s[0]).sum() on an SVD: kept at 1e-4, dropped at 1e-2",
       L.retained(ev, 1, 1e-4) == int((sv > 1e-4 * sv[0]).sum()) == 2 and L.retained(ev, 1, 1e-2) == int((sv > 1e-2 * sv[0]).sum()) == 1, (sv[:3], L.retained(ev, 1, 1e-4), L.retained(ev, 1, 1e-2)))
# both branches of null_project against the dense projector, and the rank-1 factor
for r_want in (2, 9):
    a_ = rs.standard_normal((d, 60))
    a_[r_want:] = 0.0                                   # rank r_want: the retained basis is the first r_want coordinates after rotation
    qr, _ = np.linalg.qr(rs.standard_normal((d, d)))
    k_ = qr @ a_
    ev, q = L.eig_of(k_ @ k_.T)
    got, r = L.null_project(dw, ev, q, 60, 1e-6)
    qq = q[:, :r]
    dense = dw @ (np.eye(d) - qq @ qq.T)
    expect("3i null_project equals the dense projector (retained %d of %d, %s branch)" % (r, d, "retained" if r <= d - r else "discarded"),
           r == r_want and np.allclose(got, dense, atol=1e-9), (r, abs(got - dense).max()))
sg, uu, vv = L.rank1(dw)
expect("3j rank1 recovers sigma, u, v of a planted rank-1 matrix (sign fixed)", abs(sg - 3.0) < 1e-12 and abs(abs(uu @ u) - 1) < 1e-12 and abs(abs(vv @ v) - 1) < 1e-12, sg)
a_hat = L.amplitude_ratio(v * 0.5, v, np.eye(d))
expect("3k the amplitude ratio of an edit halved is 0.5 whatever C is the identity or not", abs(a_hat - 0.5) < 1e-12 and abs(L.amplitude_ratio(v * 0.5, v, np.diag(np.arange(1, d + 1.0))) - 0.5) < 1e-12, a_hat)

# ---------------------------------------------------------------- 4. E1a
out_d, in_d = 16, 80
wb = rs.standard_normal((out_d, in_d))
uu = rs.standard_normal(out_d)
uu /= np.linalg.norm(uu)
proj = uu @ wb
sel = np.argsort(-np.abs(proj))[:6]
dwp = np.zeros_like(wb)
dwp[:, sel] = -6.0 * np.outer(uu, proj[sel])
r = L.e1a(wb, dwp, top=12)
expect("4a a planted 'remove 6 u-components' edit: slope -6, rho -1, 6 columns", abs(r["slope_c"] + 6) < 1e-9 and abs(r["pearson_rho"] + 1) < 1e-9 and r["support"] == 6, r)
expect("4b the top-|u.W_j| rule recovers its own support (all 6 inside the top 12), and H2 (48 of 64) is not met by 6 columns", r["rank_by_abs_uW"]["in_top64"] == 6 and r["rank_by_abs_uW"]["in_top96"] == 6 and r["h2"] is False, r["rank_by_abs_uW"])
r2 = L.e1a(wb, dwp * (1 / 6.0), top=12)
expect("4c the same support at slope -1 satisfies H1", r2["h1"] is True and abs(r2["slope_c"] + 1) < 1e-9, r2["slope_c"])
rnd = rs.choice(in_d, 6, replace=False)
dwr = np.zeros_like(wb)
dwr[:, rnd] = -np.outer(uu, proj[rnd])
r3 = L.e1a(wb, dwr, top=12)
expect("4d columns chosen at random are not top-ranked by |u.W_j| (their median rank is far from 1)", r3["rank_by_abs_uW"]["median"] > 8, r3["rank_by_abs_uW"])
expect("4e rank_of and spearman on a known permutation", list(L.rank_of([0, 2], [5.0, 1.0, 9.0])) == [2, 1] and abs(L.spearman(np.arange(10), np.arange(10)[::-1]) + 1) < 1e-12
       and abs(L.spearman([1, 1, 2, 3], [1, 1, 2, 3]) - 1) < 1e-12, "")

# ---------------------------------------------------------------- 5. quantisation, bf16, random edits
w = rs.standard_normal((8, 64)).astype(np.float32)
q8 = L.fake_quant(w, 8)
step = np.abs(w).max(1, keepdims=True) / 127
expect("5a int8 per row: every error within half a step", bool((np.abs(q8 - w) <= step / 2 + 1e-6).all()), np.abs(q8 - w).max())
expect("5b int8 is idempotent", bool(np.allclose(L.fake_quant(q8, 8), q8, atol=1e-6)), "")
q4 = L.fake_quant(w, 4, 32)
step4 = np.abs(w.reshape(8, 2, 32)).max(-1, keepdims=True) / 7
expect("5c int4 groups of 32: error within half a step, at most 16 levels per group",
       bool((np.abs((q4 - w).reshape(8, 2, 32)) <= step4 / 2 + 1e-6).all()) and max(len(np.unique(q4.reshape(8, 2, 32)[i, j])) for i in range(8) for j in range(2)) <= 16, "")
expect("5d a zero row stays zero", bool((L.fake_quant(np.zeros((2, 64), np.float32), 4, 32) == 0).all()), "")
try:
    L.fake_quant(w, 4, 30)
    bad = True
except ValueError:
    bad = False
expect("5e a group that does not divide the input is refused with ValueError, not a crash", not bad, "")
expect("5f bf16_round: ties to even (1 + 2^-8 -> 1, 1 + 3 * 2^-8 -> 1 + 2^-6 + 2^-7... ) and bf16 values are fixed points",
       float(L.bf16_round(np.array([1.0 + 2.0 ** -8], np.float32))[0]) == 1.0 and float(L.bf16_round(np.array([1.0 + 3 * 2.0 ** -8], np.float32))[0]) == 1.0 + 2.0 ** -6
       and float(L.bf16_round(np.array([1.5], np.float32))[0]) == 1.5, "")
dwx = 3.0 * np.outer(u, v)
big = np.zeros((6, 40))
big[:, :] = rs.standard_normal((6, 40)) * 0.01
big[:, :8] += 2.0 * np.outer(u, rs.standard_normal(8))
re_, cols_r = L.random_edit(big, 8, np.random.default_rng(5))
re2, _ = L.random_edit(big, 8, np.random.default_rng(6))
expect("5g the random edit has the same Frobenius norm, rank 1, the support size asked, and depends on the seed",
       abs(np.linalg.norm(re_) - np.linalg.norm(big)) < 1e-9 and np.linalg.matrix_rank(re_) == 1 and int((np.abs(re_).sum(0) > 0).sum()) == 8 and not np.allclose(re_, re2), "")

# ---------------------------------------------------------------- 6. repetition
loop3 = [7, 1, 9, 4] + [5, 6, 7] * 20
loop10 = list(rs.integers(100, 1000, 30)) + list(range(10)) * 5
loop64 = list(range(64)) * 4
const = [1, 2, 3] + [8] * 40
rand = list(rs.integers(0, 50000, 200))
tri = list(rs.integers(0, 50000, 100)) + [11, 12, 13] + list(rs.integers(0, 50000, 20)) + [11, 12, 13] + list(rs.integers(0, 50000, 20))
expect("6a tail_loop finds the period of planted loops (3, 10, 64, and a constant tail as 1)",
       L.tail_loop(loop3) == 3 and L.tail_loop(loop10) == 10 and L.tail_loop(loop64) == 64 and L.tail_loop(const) == 1,
       (L.tail_loop(loop3), L.tail_loop(loop10), L.tail_loop(loop64), L.tail_loop(const)))
expect("6b no loop in random tokens or in a text with one repeated trigram; a text under 32 tokens has none", L.tail_loop(rand) == 0 and L.tail_loop(tri) == 0 and L.tail_loop(list(range(10)) * 3) == 0, "")
expect("6c rep4: 0 for all-distinct 4-grams, high for a loop, None under 36 tokens",
       L.rep4(list(range(100))) == 0.0 and L.rep4(loop3) > 0.8 and L.rep4(list(range(20))) is None and L.rep4(tri) == 0.0, (L.rep4(loop3), L.rep4(tri)))

# ---------------------------------------------------------------- 7. corpus rules
ids = [f"unit-{i}" for i in range(2000)]
fit = sum(1 for i in ids if L.split_of(i) == "fit")
expect("7a the split is balanced (%d of 2000 in fit) and deterministic" % fit, 900 < fit < 1100 and L.split_of("a") == L.split_of("a"), fit)
salt = L.SALT
before = [L.split_of(i) for i in ids]
L.SALT = "another-salt"
after = [L.split_of(i) for i in ids]
L.SALT = salt
moved = sum(1 for x, y in zip(before, after) if x != y)
expect("7a2 a different salt moves about half of the units (%d of 2000)" % moved, 800 < moved < 1200, moved)
ws = L.cut_windows(list(range(5000)), 2048, 32)
ws2 = L.cut_windows(list(range(2048 * 2 + 20)), 2048, 32)
expect("7b windows: 2048, 2048, 904 for 5000 tokens; a trailing 20-token window is dropped", [len(x) for x in ws] == [2048, 2048, 904] and [len(x) for x in ws2] == [2048, 2048], [len(x) for x in ws])
es_text = "el gato de la casa que tiene una vaca para los niños con las casas por el rio " * 6
en_text = "the cat and the house that is for the kids with the cows of the river " * 6
expect("7c is_spanish and has_cjk", L.is_spanish(es_text) and not L.is_spanish(en_text) and L.has_cjk("这是中文文本" * 5) and not L.has_cjk(en_text), "")
with tempfile.TemporaryDirectory() as tmp:
    def put(name, text):
        p = os.path.join(tmp, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        return p
    ok_p = put("docs/a.md", en_text)
    expect("7d read_text accepts English prose for 'en' and refuses it for 'es'", C.read_text(ok_p, "en") is not None and C.read_text(ok_p, "es") is None, "")
    expect("7e it refuses Spanish for 'en', a minified line, a NUL byte, a tiny file and a weight-space doc",
           C.read_text(put("docs/b.md", es_text), "en") is None and C.read_text(put("c.js", "x" * 3000), "code") is None
           and C.read_text(put("d.py", "a = 1\n" * 100 + "\x00"), "code") is None and C.read_text(put("e.py", "a=1"), "code") is None
           and C.read_text(put("weight-space-x.md", en_text), "en") is None, "")
    put("node_modules/x/y.md", en_text)
    put("target/z.md", en_text)
    put("docs/sub/f.md", en_text)
    got = sorted(os.path.relpath(p, tmp) for p in C.walk([tmp], {".md"}))
    expect("7f walk skips node_modules and target and lists the rest in order", got == ["docs/a.md", "docs/b.md", "docs/sub/f.md", "weight-space-x.md"], got)

# ---------------------------------------------------------------- 8. gates
def mk(kl_hi=0.01, dom_kl=0.02, p99=0.2, ppl_hi=0.01, dom_ppl=0.02):
    s = {"kl": {"mean": kl_hi / 2, "lo": 0.0, "hi": kl_hi}, "kl_tok": {"p99": p99}, "dppl": {"mean": ppl_hi / 2, "lo": 0.0, "hi": ppl_hi}, "finite": True}
    return {"all": s, "domains": {"a": {"kl": {"mean": dom_kl}, "dppl": {"mean": dom_ppl}}, "b": {"kl": {"mean": 0.001}, "dppl": {"mean": 0.0}}}}


rep_ok = {"rep4_delta": 0.0, "new_loops": 0}
expect("8a a summary inside every threshold passes (with and without the repetition clauses)", L.passes(L.gate(mk(), rep_ok)) and L.passes(L.gate(mk())), L.gate(mk(), rep_ok))
muts = {"kl_mean": mk(kl_hi=0.03), "kl_domain": mk(dom_kl=0.06), "kl_p99": mk(p99=0.6), "ppl_all": mk(ppl_hi=0.03), "ppl_domain": mk(dom_ppl=0.06)}
for name, s in muts.items():
    g = L.gate(s, rep_ok)
    expect("8b only '%s' fails when its number is pushed over" % name, [k for k, (ok, _) in g.items() if not ok] == [name], g)
g = L.gate(mk(), {"rep4_delta": 0.05, "new_loops": 4})
expect("8c the repetition clauses fail on their own numbers", [k for k, (ok, _) in g.items() if not ok] == ["rep4", "new_loops"], g)
bad = mk()
bad["all"]["finite"] = False
expect("8d a non-finite value fails the gate", not L.passes(L.gate(bad)), "")
expect("8e the strict reading is stricter than the gate: a 0.01 upper bound passes the gate and not the strict reading", L.passes(L.gate(mk())) and not L.strict_ok(mk()) and L.strict_ok(mk(kl_hi=0.001, dom_kl=0.002, ppl_hi=0.001, dom_ppl=0.0)), "")

# ---------------------------------------------------------------- 9. the paired repetition summary
res = {"prompts": [{"group": "general", "name": f"g{i}"} for i in range(4)] + [{"group": "math", "name": "m0"}],
       "variants": {"base": [{"rep4": 0.1, "loop": 0, "sha": "a"}, {"rep4": 0.2, "loop": 0, "sha": "b"}, {"rep4": 0.1, "loop": 5, "sha": "c"}, {"rep4": None, "loop": 0, "sha": "d"}, {"rep4": 0.9, "loop": 3, "sha": "e"}],
                    "v": [{"rep4": 0.2, "loop": 0, "sha": "a"}, {"rep4": 0.4, "loop": 7, "sha": "x"}, {"rep4": 0.1, "loop": 0, "sha": "y"}, {"rep4": 0.3, "loop": 0, "sha": "d"}, {"rep4": 0.1, "loop": 0, "sha": "z"}]}}
s = C.rep_summary(res, "v")
expect("9a the pair statistics: delta over scored pairs, the new and the lost loops, the identical continuations",
       s["general"]["scored_pairs"] == 3 and abs(s["general"]["rep4_delta"] - (0.1 + 0.2 - 0.0) / 3) < 1e-12 and s["general"]["new_loops"] == 1 and s["general"]["lost_loops"] == 1
       and s["general"]["identical_continuations"] == 2 and s["math"]["lost_loops"] == 1 and s["general"]["new_loop_prompts"] == ["g1"], s)

print("w4: ok" if not fails else f"w4: {fails} FAILED")
sys.exit(1 if fails else 0)
