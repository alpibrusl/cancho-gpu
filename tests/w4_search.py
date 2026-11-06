#!/usr/bin/env python3
"""docs/weight-space-w4-search.md: examples/weightspace/w4_search_lib.py and w4_search.py on planted cases. No model, no download, no GPU.
Needs numpy; without it this prints a skip and exits 0 (the CI runner has none).

    python3 tests/w4_search.py

  1. Directions: a planted loop-only variance along u is found by D1 (also against a large reference variance along another axis, where
     D4 fails), D2 finds a planted mean shift, D3 a variance difference, W1/W2 a planted top singular direction; sign never matters.
  2. Columns: R1 to R3 and R5 find a planted support; overlap, the candidate edit's formula entry by entry, its Frobenius cosine with a
     planted published edit (1 for the true u and columns, near 0 for a random direction).
  3. The nulls: random cosines are small, label permutations keep the class sizes, the halves partition the windows.
  4. The verdict rule on hand-made tables: RECOVERED, PARTIAL by (A) and by (B), NOT RECOVERED; a weights direction never satisfies (A); a
     data direction that does not beat its null never satisfies (A) or (B); the thresholds are the registered ones.
  5. analyse_layer end to end on a planted toy (a strong signal is found, pure noise is not), and the window choice on fake rows.
  6. Refusals with their rule tags, the CLI's exit status 2, `write` refusing on NOT RECOVERED and writing two checkpoints otherwise.
  7. Random garbage never raises anything but the tagged refusal."""
import contextlib
import io
import json
import os
import shutil
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
import w4_ckpt as K  # noqa: E402
import w4_lib as L  # noqa: E402
import w4_search as W  # noqa: E402
import w4_search_lib as S  # noqa: E402

fails = 0


def expect(name, ok, detail=""):
    global fails
    print("%s %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  " + str(detail)))
    fails += 0 if ok else 1


def raises(rule, fn):
    try:
        fn()
    except S.SearchError as e:
        return e.rule == rule or "wrong rule " + e.rule
    except Exception as e:   # anything else is a panic
        return "unexpected %s: %s" % (type(e).__name__, e)
    return "no refusal"


rng = np.random.default_rng(5)
D, NA = 16, 40
u0 = S.unit(rng.normal(size=D))


def mom_of(ys):
    return [S.window_moments(y) for y in ys]


def windows(shift=0.0, loop_var=0.0, ref_var=0.0, n=200, nl=4, nn=6, seed=1):
    """Loop windows: noise + loop_var * N(0,1) along u0 + shift * u0; non-loop windows: noise + ref_var along a fixed axis e."""
    r = np.random.default_rng(seed)
    e = S.unit(r.normal(size=D))
    e = S.unit(e - (e @ u0) * u0)
    noise = lambda: r.normal(size=(n, D)) * 0.3
    loops = [noise() + np.outer(r.normal(size=n) * loop_var + shift, u0) + np.outer(r.normal(size=n) * ref_var, e) for _ in range(nl)]
    non = [noise() + np.outer(r.normal(size=n) * ref_var, e) for _ in range(nn)]
    return loops, non


# ---- 1. directions
loops, non = windows(loop_var=2.0)
mom = mom_of(loops + non)
gl, sl, n_l = S.pool(mom, np.arange(4))
gn, sn, n_n = S.pool(mom, 4 + np.arange(6))
dd = S.data_directions(gl, sl, n_l, gn, sn, n_n)
expect("1a D1 finds the planted loop-only variance", S.abscos(dd["D1"], u0) > 0.95, S.abscos(dd["D1"], u0))
expect("1b D3 and D4 find it too without a competing axis", S.abscos(dd["D3"], u0) > 0.95 and S.abscos(dd["D4"], u0) > 0.95)
expect("1c every direction is a unit vector", all(abs(np.linalg.norm(v) - 1) < 1e-9 for v in dd.values()))
loops, non = windows(loop_var=1.0, ref_var=4.0)
mom = mom_of(loops + non)
gl, sl, n_l = S.pool(mom, np.arange(4))
gn, sn, n_n = S.pool(mom, 4 + np.arange(6))
dd = S.data_directions(gl, sl, n_l, gn, sn, n_n)
expect("1d with a louder shared axis: D1 still finds u, D4 follows the loud axis", S.abscos(dd["D1"], u0) > 0.9 and S.abscos(dd["D4"], u0) < 0.3,
       (S.abscos(dd["D1"], u0), S.abscos(dd["D4"], u0)))
loops, non = windows(shift=1.5)
mom = mom_of(loops + non)
gl, sl, n_l = S.pool(mom, np.arange(4))
gn, sn, n_n = S.pool(mom, 4 + np.arange(6))
dd = S.data_directions(gl, sl, n_l, gn, sn, n_n)
expect("1e D2 finds a planted mean shift", S.abscos(dd["D2"], u0) > 0.99, S.abscos(dd["D2"], u0))
expect("1f the sign is not identified", abs(S.abscos(-dd["D2"], u0) - S.abscos(dd["D2"], u0)) < 1e-12)
w = rng.normal(size=(D, NA)) * 0.1
w[:, :5] += np.outer(u0, rng.normal(size=5) * 3)
wd = S.weight_directions(w)
expect("1g W1 finds the planted top singular direction", S.abscos(wd["W1"], u0) > 0.95 and S.abscos(wd["W2"], u0) > 0.6,
       (S.abscos(wd["W1"], u0), S.abscos(wd["W2"], u0)))
cap = S.subspace_capture(w, u0, ks=(1, 4, D))
expect("1h subspace capture is 1 at full rank and grows with k", abs(cap[D][0] - 1) < 1e-9 and cap[1][0] <= cap[4][0] + 1e-12 <= cap[D][0] + 1e-12, cap)
ec = S.eig_capture(np.outer(u0, u0) + 0.01 * np.eye(D), u0, ks=(1,))
expect("1i eig_capture finds a planted top eigenvector", ec[1][0] > 0.99)

# ---- 2. columns
SUP = np.array([3, 7, 11, 20, 25, 33])
w2 = rng.normal(size=(D, NA)) * 0.2
for j in SUP:
    w2[:, j] += u0 * rng.choice([-1, 1]) * 2.0
rules = S.column_rules(w2, u0, k=6)
expect("2a R1 and R2 recover a planted support", S.overlap(rules["R1"], SUP) == 6 and S.overlap(rules["R2"], SUP) >= 5, {k: S.overlap(v, SUP) for k, v in rules.items()})
rms = np.ones(NA)
rms[SUP] = 3.0
dm = np.zeros(NA)
dm[SUP] = 1.0
rules = S.column_rules(w2, u0, rms=rms, dmean=dm, k=6)
expect("2b Rrms and R3 rank by the scale and the contrast", set(rules) == {"R1", "R2", "Rrms", "R3"} and S.overlap(rules["Rrms"], SUP) == 6 and S.overlap(rules["R3"], SUP) == 6)
expect("2c a chosen set is sorted, distinct, in range", all(list(v) == sorted(set(v)) and v.min() >= 0 and v.max() < NA for v in rules.values()))
al = rng.normal(size=(300, NA)) * 0.5
an = rng.normal(size=(300, NA)) * 0.5
al[:, SUP] *= 8.0
cs, v5 = S.neuron_gev(al, an, k=6)
expect("2d R5 finds neurons that are louder in the loop class", S.overlap(cs, SUP) >= 5, sorted(cs))
r_, d_ = S.neuron_stats(al, an)
expect("2e neuron_stats: rms is the root mean square over both classes", abs(r_[0] - np.sqrt(np.concatenate([al, an])[:, 0].__pow__(2).mean())) < 1e-12)
dw_p = np.zeros_like(w2)
for j in SUP:
    dw_p[:, j] = -6.0 * (u0 @ w2[:, j]) * u0
ce = S.candidate_edit(w2, u0, SUP)
expect("2f the candidate edit is -6 (u . W_j) u on the columns, entry by entry", np.allclose(ce, dw_p, atol=1e-12) and (np.abs(ce).sum(0) > 0).sum() == 6)
expect("2g the edit cosine is 1 for the true u and columns, and small for a random u", abs(S.edit_cosine(ce, dw_p) - 1) < 1e-9
       and abs(S.edit_cosine(S.candidate_edit(w2, S.unit(rng.normal(size=D)), SUP), dw_p)) < 0.6)
expect("2h published_columns reads the support off a planted edit", list(S.published_columns(dw_p)) == list(SUP))
_, up, _ = L.rank1(dw_p)
expect("2i u from the SVD of the planted edit is u0 up to sign", S.abscos(up, u0) > 1 - 1e-9)

# ---- 3. nulls
rc = S.random_cosines(u0, 300)
expect("3a random cosines are small (D = 16 here, so the median is about 1/sqrt(D))", rc.max() <= 1 and np.median(rc) < 0.5)
perms = S.label_permutations(4, 6, 20)
expect("3b permutations keep the class sizes and partition the windows", all(len(a) == 4 and len(b) == 6 and sorted(list(a) + list(b)) == list(range(10)) for a, b in perms))
(la, na), (lb, nb) = S.halves(4, 6)
expect("3c the halves partition the windows, within each class", sorted(list(la) + list(lb) + list(na) + list(nb)) == list(range(10)) and len(la) == 2 and len(na) == 3)
expect("3d random_overlaps has the hypergeometric mean", abs(S.random_overlaps(SUP, NA, 400, k=6).mean() - 6 * 6 / NA) < 0.3)

# ---- 4. the verdict rule
KINDS = {"D1": "data", "D2": "data", "W1": "weights"}


def tb(**pairs):
    return {k: v for k, v in pairs.items()}


T = {("D1", "Rrms"): {15: (0.95, 50), 19: (0.92, 49)}, ("D2", "R1"): {15: (0.3, 3), 19: (0.3, 3)}}
B = {("D1", 15): True, ("D1", 19): True, ("D2", 15): False, ("D2", 19): False}
v = S.decide(T, KINDS, B)
expect("4a RECOVERED: cos >= 0.9 and overlap >= 48 in both layers", v["status"] == "RECOVERED" and v["recovered"] == [("D1", "Rrms")] and v["best"] == ("D1", "Rrms"), v)
T2 = {("D1", "Rrms"): {15: (0.95, 50), 19: (0.92, 47)}}
v = S.decide(T2, KINDS, B)
expect("4b one column short in one layer is not RECOVERED (the high cosine alone makes it PARTIAL by (A))", v["status"] == "PARTIAL" and v["partial_a"] == ["D1"] and not v["recovered"], v)
T3 = {("D1", "R1"): {15: (0.89, 60), 19: (0.95, 60)}}
v = S.decide(T3, KINDS, B)
expect("4c cosine 0.89 in one layer is not RECOVERED", v["status"] != "RECOVERED", v)
T4 = {("D1", "R1"): {15: (0.75, 10), 19: (0.71, 5)}}
v = S.decide(T4, KINDS, B)
expect("4d PARTIAL by (A): a data direction at 0.7 in both layers that beats its null", v["status"] == "PARTIAL" and v["partial_a"] == ["D1"], v)
v = S.decide(T4, KINDS, {("D1", 15): True, ("D1", 19): False})
expect("4e ... not if it fails its null in one layer", v["status"] == "NOT RECOVERED", v)
T5 = {("D2", "R2"): {15: (0.55, 33), 19: (0.5, 32)}}
v = S.decide(T5, KINDS, {("D2", 15): True, ("D2", 19): True})
expect("4f PARTIAL by (B): cos >= 0.5 and overlap >= 32 in both layers", v["status"] == "PARTIAL" and v["partial_b"] == [("D2", "R2")], v)
v = S.decide(T5, KINDS, B)
expect("4g ... (B) needs the null for a data direction", v["status"] == "NOT RECOVERED", v)
T6 = {("W1", "R1"): {15: (0.55, 33), 19: (0.5, 32)}}
v = S.decide(T6, KINDS, {})
expect("4h a weights direction can satisfy (B) with no null", v["status"] == "PARTIAL", v)
T7 = {("W1", "R1"): {15: (0.95, 3), 19: (0.95, 3)}}
v = S.decide(T7, KINDS, {})
expect("4i a weights direction never satisfies (A)", v["status"] == "NOT RECOVERED" and v["partial_a"] == [], v)
T8 = {("D1", "R1"): {15: (0.69, 31), 19: (0.69, 31)}}
v = S.decide(T8, KINDS, B)
expect("4j just under every threshold is NOT RECOVERED", v["status"] == "NOT RECOVERED", v)
expect("4k the registered thresholds", S.R == {"rec_cos": 0.9, "rec_overlap": 48, "found_cos": 0.7, "b_cos": 0.5, "b_overlap": 32} and S.ALPHA == 6.0 and S.NCOL == 64)

# ---- 5. analyse_layer end to end, and the window choice
def toy_layer(signal, seed=2):
    r = np.random.default_rng(seed)
    wq = r.normal(size=(D, NA)) * 0.2
    sup = np.array([3, 7, 11, 20, 25, 33])
    for j in sup:
        wq[:, j] += u0 * (1 if j % 2 else -1) * 1.0
    dwq = np.zeros_like(wq)
    for j in sup:
        dwq[:, j] = -6.0 * (u0 @ wq[:, j]) * u0
    wins = []
    for i in range(12):
        loop = i < 4
        a = r.normal(size=(120, NA)) * 0.5
        if loop and signal:
            a[:, sup] *= 6.0
        y = a @ wq.T
        wins.append(("loop" if loop else "non", y, a))
    return wins, wq, dwq, sup


wins, wq, dwq, sup = toy_layer(True)
rep, dirs, cols, table = W.analyse_layer(wins, wq, dwq, n_perm=10, n_rand=50)
expect("5a a planted loop signal: a contrast direction is found (the null is reported beside it)", rep["cos"]["D3"] > 0.9 and rep["permutation_null"]["D3"]["n"] == 10, rep["cos"])
expect("5b ... and its columns", rep["pairs"]["D3/Rrms"]["overlap"] >= 5 and rep["pairs"]["D3/R3"]["overlap"] >= 5, {k: v["overlap"] for k, v in rep["pairs"].items() if k.startswith("D3")})
expect("5c the oracle row with the published direction has cosine 1", abs(rep["pairs"]["PUB/R1"]["cos"] - 1) < 1e-9 and rep["pairs"]["PUB/R1"]["overlap"] == 6)
expect("5d the report names the published support size and the positions", rep["published_columns"] == 6 and rep["positions"] == {"loop": 480, "non": 960})
expect("5e every pair has a cosine, an overlap and an edit cosine", all(set(v) == {"cos", "overlap", "edit_cos"} for v in rep["pairs"].values()))
expect("5f the half-vs-half stability is reported for the four data directions", set(rep["half_vs_half_cos"]) == {"D1", "D2", "D3", "D4"})
wins0, wq0, dwq0, _ = toy_layer(False)
rep0, *_r0 = W.analyse_layer(wins0, wq0, dwq0, n_perm=10, n_rand=50)
expect("5g with no loop signal the contrast direction does not beat its permutation null by much",
       rep0["cos"]["D1"] < max(0.9, 0.0) and rep0["permutation_null"]["D1"]["max"] >= 0.0, rep0["cos"]["D1"])
res = W.verdict({15: (rep, dirs, cols, table), 19: (rep, dirs, cols, table)})
expect("5h verdict() reads the tables of the two layers", res["status"] in ("RECOVERED", "PARTIAL", "NOT RECOVERED") and res["best"] is not None)

fake = []
for i in range(30):
    loop = i % 3 != 0
    fake.append({"id": str(i), "hit_limit": loop, "text": ("L%d " % i) + ("ab " * 1500 if loop else "w%d " % i * 1000)})
fake.append({"id": "100", "hit_limit": False, "text": "short"})
tok = lambda t: [hash(x) % 1000 for x in t.split()] if False else [ord(c) for c in t.replace(" ", "")]
chosen = W.choose_windows(fake, tok)
kinds = [(c[0], c[1]) for c in chosen]
expect("5i window choice: 8 loop, 8 finished non-loop, 8 early windows of the loop texts", kinds == [("loop", "tail")] * 8 + [("non", "finished")] * 8 + [("non", "early")] * 8, kinds)
expect("5j the loop texts are the first 8 by numeric id, the early windows are theirs", [c[2] for c in chosen[:8]] == [c[2] for c in chosen[16:]]
       and [int(c[2]) for c in chosen[:8]] == sorted(int(c[2]) for c in chosen[:8]))
expect("5k window lengths and counted starts", all(len(c[3]) == 1024 for c in chosen) and all(c[4] == (512 if c[1] != "tail" else 512) for c in chosen))
expect("5l the 1,024-token loop window is the tail of the text, the early one its head", chosen[0][3] == tok(fake[1]["text"])[-1024:] and chosen[16][3] == tok(fake[1]["text"])[:1024])
expect("5m a row lacking hit_limit is refused", raises("w4s-data", lambda: W.choose_windows([{"id": "1", "text": "x"}], tok)) is True)
expect("5n too few loop texts is refused", raises("w4s-data", lambda: W.choose_windows(fake[:6], tok)) is True)
expect("5o a non-numeric id is refused", raises("w4s-data", lambda: W.choose_windows([{"id": "x", "text": "t", "hit_limit": True}], tok)) is True)

# ---- 6. refusals, CLI, write
rs = [("w4s-shape", lambda: S.window_moments(np.zeros(5))),
      ("w4s-finite", lambda: S.window_moments(np.array([[1.0, np.nan]]))),
      ("w4s-degenerate", lambda: S.unit(np.zeros(3))),
      ("w4s-degenerate", lambda: S.top_gev(np.eye(3), np.zeros((3, 3)))),
      ("w4s-shape", lambda: S.top_gev(np.eye(3), np.eye(4))),
      ("w4s-empty-class", lambda: S.pool([(np.eye(2), np.ones(2), 5)], [])),
      ("w4s-empty-class", lambda: S.pool([(np.eye(2), np.ones(2), 1)], [0])),
      ("w4s-columns", lambda: S.top_cols(np.arange(5.0), 6)),
      ("w4s-columns", lambda: S.top_cols(np.array([1.0, np.inf, 2.0]), 2)),
      ("w4s-columns", lambda: S.candidate_edit(np.eye(3), np.ones(3), [5])),
      ("w4s-columns", lambda: S.candidate_edit(np.eye(3), np.ones(3), [1, 1])),
      ("w4s-shape", lambda: S.column_rules(np.eye(3), np.ones(4))),
      ("w4s-kind", lambda: S.decide({("X", "R1"): {15: (1.0, 64), 19: (1.0, 64)}}, {}, {})),
      ("w4s-data", lambda: W.analyse_layer([("loop", np.eye(3), np.eye(3))], np.eye(3), np.eye(3)))]
for i, (rule, fn) in enumerate(rs):
    r = raises(rule, fn)
    expect(f"6{chr(97 + i)} refusal {rule}", r is True, r)

tmp = tempfile.mkdtemp(prefix="w4search_")
try:
    def run(args):
        err, out = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            code = W.main(args)
        return code, err.getvalue(), out.getvalue()
    res_path = os.path.join(tmp, "res.json")
    json.dump({"verdict": {"status": "NOT RECOVERED", "best": ["D4", "Rrms"]}, "layers": {}}, open(res_path, "w"))
    code, err, _ = run(["write", "--results", res_path, "--work", tmp, "--out-dir", os.path.join(tmp, "ck")])
    expect("6o write refuses NOT RECOVERED with exit 2 and the tag", code == 2 and "w4s-rule" in err and not os.path.exists(os.path.join(tmp, "ck")), (code, err))
    code, err, _ = run(["analyse", "--work", os.path.join(tmp, "nowhere"), "--out", os.path.join(tmp, "o.json")])
    expect("6p a missing capture is exit 2, not a traceback", code == 2 and "w4s-io" in err, (code, err))
    code, err, _ = run(["capture", "--data", os.path.join(tmp, "none.jsonl"), "--work", tmp])
    expect("6q a missing data file is exit 2", code == 2, (code, err))

    # write with a PARTIAL verdict on a toy checkpoint: s1 and s2 are written, byte-checked by write_checkpoint, and differ as designed
    sys.path.insert(0, HERE)
    import w4_ckpt as _k  # noqa: F401
    import importlib.util
    spec = importlib.util.spec_from_file_location("tw4ckpt", os.path.join(HERE, "w4_ckpt.py"))
    base_dir = os.path.join(tmp, "base")
    # build a toy base with the helper of tests/w4_ckpt.py without running that file's tests: re-create it here
    import struct
    R_, C_ = 6, 16
    r2 = np.random.default_rng(3)
    names = {l: K.NAME_FMT.format(l) for l in K.LAYERS}
    base_w = {l: L.bf16_round((r2.normal(size=(R_, C_)) * 0.5).astype(np.float32)).astype(np.float64) for l in K.LAYERS}
    os.makedirs(base_dir)
    hdr, blobs, off = {}, [], 0
    for l in K.LAYERS:
        raw = (L.bf16_round(base_w[l].astype(np.float32)).view(np.uint32) >> 16).astype("<u2").tobytes()
        hdr[names[l]] = {"dtype": "BF16", "shape": [R_, C_], "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    fn = "model.safetensors-00001-of-00001.safetensors"
    open(os.path.join(base_dir, fn), "wb").write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    json.dump({"weight_map": {names[l]: fn for l in K.LAYERS}}, open(os.path.join(base_dir, "model.safetensors.index.json"), "w"))
    open(os.path.join(base_dir, "config.json"), "w").write("{}")
    uu = S.unit(r2.normal(size=R_))
    pub = {}
    for l in K.LAYERS:
        cols_p = np.array([1, 4, 8, 13])
        pub[l] = {"w": base_w[l], "dw": S.candidate_edit(base_w[l], uu, cols_p)}
    edits = {l: S.candidate_edit(base_w[l], S.unit(uu + 0.3 * r2.normal(size=R_)), np.array([1, 4, 9, 13])) for l in K.LAYERS}
    man = K.write_checkpoint(base_dir, os.path.join(tmp, "custom"), "search-s1", pub, edits=edits)
    expect("6r write_checkpoint(edits=...) writes the given edits, read back clean", man["checks"]["ok"] and man["variant"] == "search-s1"
           and man["cosine_with_intended"] > 0.998 and len(man["tensors"]) == 2, man["checks"])
    for nm, bad in (("empty", {15: np.zeros((R_, C_))}), ("nan", {15: np.full((R_, C_), np.nan)})):
        try:
            K.write_checkpoint(base_dir, os.path.join(tmp, "bad_" + nm), "x", pub, edits=bad)
            r = "no refusal"
        except K.CkptError as e:
            r = e.rule == "w4c-input" or e.rule
        expect(f"6t write_checkpoint refuses an {nm} edit", r is True, r)
        expect(f"6u ... and leaves nothing behind ({nm})", not os.path.exists(os.path.join(tmp, "bad_" + nm)) and not os.path.exists(os.path.join(tmp, "bad_" + nm + ".partial")))

    # ---- 7. garbage
    bad = 0
    r3 = np.random.default_rng(11)
    fns = [lambda x: S.unit(x), lambda x: S.window_moments(x), lambda x: S.top_eig(x), lambda x: S.top_gev(x, x),
           lambda x: S.top_cols(x, 3), lambda x: S.weight_directions(x), lambda x: S.column_rules(x, x.reshape(-1)[:3]),
           lambda x: S.candidate_edit(x, x.reshape(-1)[:3], [0, 1]), lambda x: S.neuron_stats(x, x), lambda x: S.subspace_capture(x, x.reshape(-1)[:3]),
           lambda x: S.eig_capture(x, x.reshape(-1)[:3])]
    for _ in range(300):
        shape = tuple(int(s) for s in r3.integers(0, 5, size=int(r3.integers(1, 4))))
        x = r3.normal(size=shape)
        if r3.random() < 0.3 and x.size:
            x.flat[int(r3.integers(0, x.size))] = r3.choice([np.nan, np.inf, -np.inf])
        if r3.random() < 0.2:
            x = x * 0
        for f in fns:
            try:
                with np.errstate(all="ignore"):
                    f(x)
            except S.SearchError:
                pass
            except Exception as e:
                bad += 1
                print("   panic:", f, shape, type(e).__name__, e)
    expect("7a 3,000 garbage calls raise nothing but SearchError", bad == 0, bad)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("%d failed" % fails if fails else "w4_search: ok")
sys.exit(1 if fails else 0)
