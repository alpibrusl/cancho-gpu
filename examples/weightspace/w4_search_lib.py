"""docs/weight-space-w4-search.md: the pure parts of the W4 edit search (numpy only; no file, no model, no network).

Everything here is held by tests/w4_search.py on planted cases. The model-facing script is w4_search.py. Every refusal is a
SearchError with a rule tag `w4s-*`; no input reaches anything else."""
import numpy as np

import w4_lib as L

SEED = L.SEED + 4
RIDGE = 1e-4
ALPHA = 6.0
NCOL = 64


class SearchError(Exception):
    """A refusal; `rule` is its tag."""

    def __init__(self, rule, msg):
        super().__init__(f"{rule}: {msg}")
        self.rule = rule


def _mat(x, name, cols=None):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1 or (cols is not None and x.shape[1] != cols):
        raise SearchError("w4s-shape", f"{name} has shape {list(x.shape)}")
    if not np.isfinite(x).all():
        raise SearchError("w4s-finite", f"{name} has a NaN or an infinity")
    return x


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    n = float(np.linalg.norm(x))
    if not np.isfinite(n) or n == 0:
        raise SearchError("w4s-degenerate", "a zero or non-finite vector has no direction")
    return x / n


def abscos(a, b):
    """|cos| of two vectors (sign is not identified)."""
    return float(abs(unit(a) @ unit(b)))


# ------------------------------------------------------------------ moments of a window, of a labelling

def window_moments(y):
    """(sum y y^T, sum y, n) of the rows of `y` [n, d], float64."""
    y = _mat(y, "a window of outputs")
    return y.T @ y, y.sum(0), y.shape[0]


def pool(moments, idx):
    """The pooled (G, s, n) of the windows `idx` of a list of window_moments."""
    if len(idx) == 0:
        raise SearchError("w4s-empty-class", "a class has no window")
    g = sum(moments[i][0] for i in idx)
    s = sum(moments[i][1] for i in idx)
    n = sum(moments[i][2] for i in idx)
    if n < 2:
        raise SearchError("w4s-empty-class", "a class has fewer than 2 positions")
    return g, s, n


# ------------------------------------------------------------------ directions

def top_gev(a, b, ridge=RIDGE):
    """Unit top eigenvector of A v = lambda (B + ridge tr(B)/d I) v, A and B symmetric PSD [d, d]. Whitens with a Cholesky factor."""
    a, b = _mat(a, "A"), _mat(b, "B")
    d = a.shape[0]
    if a.shape != (d, d) or b.shape != (d, d):
        raise SearchError("w4s-shape", f"A {list(a.shape)} and B {list(b.shape)} are not the same square size")
    br = (b + b.T) / 2
    tr = float(np.trace(br))
    if tr <= 0:
        raise SearchError("w4s-degenerate", "the reference second moment has no energy")
    br = br + ridge * tr / d * np.eye(d)
    try:
        lc = np.linalg.cholesky(br)
    except np.linalg.LinAlgError:
        raise SearchError("w4s-degenerate", "the regularised reference moment is not positive definite")
    m = np.linalg.solve(lc, np.linalg.solve(lc, (a + a.T) / 2).T)   # L^-1 A L^-T
    w, q = np.linalg.eigh((m + m.T) / 2)
    v = np.linalg.solve(lc.T, q[:, -1])
    return unit(v)


def top_eig(c, absolute=False):
    """Unit eigenvector of a symmetric matrix with the largest eigenvalue (the largest |eigenvalue| when `absolute`)."""
    c = _mat(c, "C")
    if c.shape[0] != c.shape[1]:
        raise SearchError("w4s-shape", f"C {list(c.shape)} is not square")
    w, q = np.linalg.eigh((c + c.T) / 2)
    i = int(np.argmax(np.abs(w))) if absolute else int(np.argmax(w))
    return unit(q[:, i])


def data_directions(gl, sl, nl, gn, sn, nn, ridge=RIDGE):
    """D1 to D4 of docs/weight-space-w4-search.md 3.1 from pooled moments of the loop (l) and non-loop (n) positions."""
    cl, cn = gl / nl, gn / nn
    return {"D1": top_gev(cl, cn, ridge), "D2": unit(sl / nl - sn / nn), "D3": top_eig(cl - cn, absolute=True), "D4": top_eig(cl)}


def weight_directions(w):
    """W1: top left singular vector of W; W2: the same after scaling its columns to unit norm."""
    w = _mat(w, "W")
    u1 = np.linalg.svd(w, full_matrices=False)[0][:, 0]
    nrm = np.linalg.norm(w, axis=0)
    u2 = np.linalg.svd(w / np.maximum(nrm, 1e-30), full_matrices=False)[0][:, 0]
    return {"W1": unit(u1), "W2": unit(u2)}


def subspace_capture(w, u, ks=(1, 4, 16, 64)):
    """{k: (|| P_k u ||, sqrt(k/d))}: the share of u in the span of W's top-k left singular vectors, and its chance value."""
    w = _mat(w, "W")
    u = unit(u)
    if u.size != w.shape[0]:
        raise SearchError("w4s-shape", f"u has {u.size} entries, W has {w.shape[0]} rows")
    q = np.linalg.svd(w, full_matrices=False)[0]
    d = w.shape[0]
    return {int(k): (float(np.linalg.norm(q[:, :k].T @ u)), float(np.sqrt(k / d))) for k in ks if 1 <= k <= q.shape[1]}


def eig_capture(c, u, ks=(1, 4, 16, 64)):
    """{k: (|| P_k u ||, sqrt(k/d))}: the share of u in the span of the top-k eigenvectors of a symmetric matrix `c`."""
    c = _mat(c, "C")
    u = unit(u)
    if c.shape[0] != c.shape[1] or u.size != c.shape[0]:
        raise SearchError("w4s-shape", f"C {list(c.shape)} and u ({u.size}) do not match")
    w, q = np.linalg.eigh((c + c.T) / 2)
    q = q[:, ::-1]
    return {int(k): (float(np.linalg.norm(q[:, :k].T @ u)), float(np.sqrt(k / c.shape[0]))) for k in ks if 1 <= k <= c.shape[0]}


# ------------------------------------------------------------------ columns

def top_cols(score, n=NCOL):
    """The n columns with the largest score, as a sorted int array (ties by index, so the result is deterministic)."""
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 1 or n < 1 or n > score.size or not np.isfinite(score).all():
        raise SearchError("w4s-columns", f"cannot choose {n} of {score.size} scores")
    return np.sort(np.argsort(-score, kind="stable")[:n])


def neuron_stats(a_loop, a_non):
    """(rms over all positions, |mean_L - mean_N|) per neuron from [n, 3584] activations."""
    al, an = _mat(a_loop, "loop activations"), _mat(a_non, "non-loop activations", np.asarray(a_loop).shape[1])
    allx = np.concatenate([al, an])
    return np.sqrt((allx ** 2).mean(0)), np.abs(al.mean(0) - an.mean(0))


def column_rules(w, u, rms=None, dmean=None, k=NCOL):
    """{rule: 64 columns} for a unit direction u: R1, R2 (weights only), and Rrms, R3 when rms / dmean are given."""
    w = _mat(w, "W")
    u = unit(u)
    if u.size != w.shape[0]:
        raise SearchError("w4s-shape", f"u has {u.size} entries, W has {w.shape[0]} rows")
    proj = np.abs(u @ w)
    out = {"R1": top_cols(proj, k), "R2": top_cols(proj / np.maximum(np.linalg.norm(w, axis=0), 1e-30), k)}
    if rms is not None:
        out["Rrms"] = top_cols(proj * np.asarray(rms), k)
    if dmean is not None:
        out["R3"] = top_cols(proj * np.asarray(dmean), k)
    return out


def neuron_gev(a_loop, a_non, ridge=RIDGE, k=NCOL):
    """R5: the k (64) largest |v_j| of the top generalised eigenvector in the neuron space, plus v."""
    al, an = _mat(a_loop, "loop activations"), _mat(a_non, "non-loop activations")
    v = top_gev(al.T @ al / al.shape[0], an.T @ an / an.shape[0], ridge)
    return top_cols(np.abs(v), k), v


def overlap(cols, published):
    return int(np.intersect1d(np.asarray(cols), np.asarray(published)).size)


def published_columns(dw):
    return np.flatnonzero(np.abs(np.asarray(dw, dtype=np.float64)).sum(0) > 0)


def candidate_edit(w, u, cols, alpha=ALPHA):
    """dW[:, j] = -alpha (u . W_j) u on `cols`, zero elsewhere (float64, [out, in])."""
    w = _mat(w, "W")
    u = unit(u)
    if u.size != w.shape[0]:
        raise SearchError("w4s-shape", f"u has {u.size} entries, W has {w.shape[0]} rows")
    cols = np.asarray(cols, dtype=np.int64)
    if cols.size == 0 or cols.min() < 0 or cols.max() >= w.shape[1] or np.unique(cols).size != cols.size:
        raise SearchError("w4s-columns", "the column list is empty, out of range or has duplicates")
    d = np.zeros_like(w)
    d[:, cols] = -alpha * np.outer(u, u @ w[:, cols])
    return d


def edit_cosine(d1, d2):
    a, b = float(np.linalg.norm(d1)), float(np.linalg.norm(d2))
    return float((d1 * d2).sum() / (a * b)) if a > 0 and b > 0 else 0.0


# ------------------------------------------------------------------ nulls

def random_cosines(u_pub, n=200, seed=SEED):
    """|cos| of n random unit vectors with u_pub."""
    rng = np.random.default_rng(seed)
    u = unit(u_pub)
    return np.abs(unit_rows(rng.standard_normal((n, u.size))) @ u)


def unit_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)


def random_overlaps(published, ncols, n=200, seed=SEED, k=NCOL):
    rng = np.random.default_rng(seed)
    return np.array([overlap(rng.choice(ncols, size=k, replace=False), published) for _ in range(n)])


def label_permutations(n_loop, n_non, n_perm=50, seed=SEED):
    """n_perm random relabellings of n_loop + n_non windows: arrays (loop indices, non-loop indices)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        p = rng.permutation(n_loop + n_non)
        out.append((np.sort(p[:n_loop]), np.sort(p[n_loop:])))
    return out


def halves(n_loop, n_non):
    """Alternate windows within each class: ((loop A, non A), (loop B, non B)); windows are indexed loop first, then non-loop."""
    la, lb = np.arange(0, n_loop, 2), np.arange(1, n_loop, 2)
    na, nb = n_loop + np.arange(0, n_non, 2), n_loop + np.arange(1, n_non, 2)
    return (la, na), (lb, nb)


# ------------------------------------------------------------------ the success rule (docs/weight-space-w4-search.md 4)

R = {"rec_cos": 0.9, "rec_overlap": 48, "found_cos": 0.7, "b_cos": 0.5, "b_overlap": 32}


def decide(table, kinds, beats, layers=(15, 19), r=R):
    """`table[(direction, rule)][layer] = (|cos|, overlap)`; `kinds[direction]` is 'data' or 'weights'; `beats[(direction, layer)]` is
    whether a data direction's |cos| exceeds its permutation null. Returns {"status", "recovered", "partial_a", "partial_b", "best"}.
    `best` ranks pairs by (min |cos|, min overlap) over the layers."""
    def mn(pair, i):
        return min(table[pair][l][i] for l in layers)

    pairs = [p for p in table if all(l in table[p] for l in layers)]
    for p in pairs:
        if kinds.get(p[0]) not in ("data", "weights"):
            raise SearchError("w4s-kind", f"direction {p[0]!r} is neither data nor weights")
    rec = [p for p in pairs if mn(p, 0) >= r["rec_cos"] and mn(p, 1) >= r["rec_overlap"]]
    dirs = sorted({p[0] for p in pairs})
    pa = [d for d in dirs if kinds[d] == "data" and all(beats.get((d, l), False) for l in layers)
          and min(max(table[p][l][0] for p in pairs if p[0] == d) for l in layers) >= r["found_cos"]]
    pb = [p for p in pairs if mn(p, 0) >= r["b_cos"] and mn(p, 1) >= r["b_overlap"]
          and (kinds[p[0]] == "weights" or all(beats.get((p[0], l), False) for l in layers))]
    status = "RECOVERED" if rec else "PARTIAL" if (pa or pb) else "NOT RECOVERED"
    best = max(pairs, key=lambda p: (mn(p, 0), mn(p, 1), p)) if pairs else None
    return {"status": status, "recovered": sorted(rec), "partial_a": pa, "partial_b": sorted(pb), "best": best}
