"""docs/weight-space-w4-r4.md: the pure parts of the W4 gradient search (numpy only; no file, no model, no network).

Everything here is held by tests/w4_grad.py on planted cases. The model-facing script is w4_grad.py. Every refusal is a GradError
with a rule tag `w4g-*`; no input reaches anything else."""
import numpy as np

import w4_lib as L
import w4_search_lib as S

SEED = L.SEED + 5
SEED_CTRL = L.SEED + 6
NCOL = 64
DIRS = ("GU1", "GU2", "GU3")
RULES = ("C1", "C2", "C3", "C4")
LOSSES = ("p", "o")
HALVES = ("train", "test")


class GradError(Exception):
    """A refusal; `rule` is its tag."""

    def __init__(self, rule, msg):
        super().__init__(f"{rule}: {msg}")
        self.rule = rule


def _finite(x, name):
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise GradError("w4g-finite", f"{name} has a NaN, an infinity or a non-number")
    return x


def split_loops(n):
    """(train, test) positions of n loops in list order: train the even positions, test the odd ones (docs 2)."""
    if not isinstance(n, int) or n < 4:
        raise GradError("w4g-loops", f"{n} loops cannot be split into two halves of at least two")
    idx = np.arange(n)
    return idx[0::2], idx[1::2]


# ------------------------------------------------------------------ sums over loops

class Sums:
    """The sums of docs 3.1 over a set of loops for one layer and one loss: G [out, in], m [out], Q [out, out], r [in], n."""

    def __init__(self, out, inn):
        self.g = np.zeros((out, inn), dtype=np.float32)
        self.m = np.zeros(out, dtype=np.float64)
        self.q = np.zeros((out, out), dtype=np.float64)
        self.r = np.zeros(inn, dtype=np.float64)
        self.n = 0
        self.loops = 0

    def add(self, g, m, q, r, n):
        g = _finite(g, "G")
        if g.shape != self.g.shape or np.shape(m) != self.m.shape or np.shape(q) != self.q.shape or np.shape(r) != self.r.shape:
            raise GradError("w4g-shape", f"a loop's sums have shapes {np.shape(g)}, {np.shape(m)}, {np.shape(q)}, {np.shape(r)}")
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise GradError("w4g-shape", "a loop with no positions")
        self.g += g
        self.m += _finite(m, "m")
        self.q += _finite(q, "Q")
        self.r += _finite(r, "r")
        self.n += int(n)
        self.loops += 1

    def rms(self):
        if self.n < 1:
            raise GradError("w4g-loops", "no loop was added")
        return np.sqrt(self.r / self.n)


# ------------------------------------------------------------------ directions and columns

def unit(x):
    try:
        return S.unit(x)
    except S.SearchError as e:
        raise GradError("w4g-degenerate", str(e))


def top_left_sv(g):
    """The top left singular vector of g [out, in] (unit, float64): the top eigenvector of g g^T."""
    g = _finite(g, "G")
    if g.ndim != 2 or not np.any(g):
        raise GradError("w4g-degenerate", "a zero or non-matrix gradient has no singular direction")
    g32 = np.ascontiguousarray(g, dtype=np.float32)
    c = (g32 @ g32.T).astype(np.float64)
    return np.linalg.eigh((c + c.T) / 2)[1][:, -1]


def top_eigvec(q):
    """The eigenvector of the largest eigenvalue of a symmetric matrix (uncentred second moment)."""
    q = _finite(q, "Q")
    if q.ndim != 2 or q.shape[0] != q.shape[1] or not np.any(q):
        raise GradError("w4g-degenerate", "a zero or non-square second moment has no dominant direction")
    return np.linalg.eigh((q + q.T) / 2)[1][:, -1]


def directions(sums):
    """{GU1, GU2, GU3} for a Sums (docs 3.2), unit float64."""
    return {"GU1": top_left_sv(sums.g), "GU2": unit(sums.m), "GU3": top_eigvec(sums.q)}


def column_rules(sums, w, u, k=NCOL):
    """{C1..C4}: k of the columns (docs 3.3). C1 needs no u."""
    w = np.asarray(_finite(w, "W"), dtype=np.float64)
    if w.shape != sums.g.shape:
        raise GradError("w4g-shape", f"W is {w.shape}, the gradient {sums.g.shape}")
    u = unit(u)
    if u.size != w.shape[0]:
        raise GradError("w4g-shape", f"u has {u.size} entries, W has {w.shape[0]} rows")
    g = sums.g
    proj_w = np.abs(u @ w)
    try:
        return {"C1": S.top_cols(np.linalg.norm(g.astype(np.float64), axis=0), k),
                "C2": S.top_cols(np.abs(u @ g.astype(np.float64)), k),
                "C3": S.top_cols(proj_w, k),
                "C4": S.top_cols(proj_w * sums.rms(), k)}
    except S.SearchError as e:
        raise GradError("w4g-columns", str(e))


def along(g, u):
    """||u^T G|| / ||G||: the share of the gradient's Frobenius norm that lies along u (chance sqrt(1/out))."""
    g = np.asarray(g, dtype=np.float64)
    n = float(np.linalg.norm(g))
    return float(np.linalg.norm(unit(u) @ g) / n) if n > 0 else 0.0


# ------------------------------------------------------------------ the nulls

def derangement(n, seed):
    """A permutation of range(n) with no fixed point (docs 3.4: loop j is scored on the continuation of loop pi(j))."""
    if not isinstance(n, int) or n < 2:
        raise GradError("w4g-loops", "a derangement needs at least two loops")
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


# ------------------------------------------------------------------ one layer, one half

def analyse_half(by_loss, w, u_pub, cols_pub, k=NCOL):
    """For one layer and one half: `by_loss` = {loss: Sums}. Returns {"dirs": {dir/loss: u}, "cols": {dir/loss: {rule: cols}},
    "rows": {dir/loss/rule: {"cos", "overlap"}}}."""
    dirs, cols, rows = {}, {}, {}
    for loss, sums in by_loss.items():
        for name, u in directions(sums).items():
            key = f"{name}/{loss}"
            dirs[key] = u
            cols[key] = column_rules(sums, w, u, k)
            for rule, cs in cols[key].items():
                rows[f"{key}/{rule}"] = {"cos": S.abscos(u, u_pub), "overlap": S.overlap(cs, cols_pub)}
    return {"dirs": dirs, "cols": cols, "rows": rows}


def merge_halves(per_half, layers, dirs=DIRS, rules=RULES, losses=LOSSES):
    """The table of docs 4.1: `per_half[half][layer]` = analyse_half's output. Returns {(dir/loss, rule): {layer: (min cos, min overlap)}}."""
    table = {}
    for loss in losses:
        for d in dirs:
            for rule in rules:
                key = f"{d}/{loss}/{rule}"
                for layer in layers:
                    rs = [per_half[h][layer]["rows"][key] for h in HALVES]
                    table.setdefault((f"{d}/{loss}", rule), {})[layer] = (min(r["cos"] for r in rs), min(r["overlap"] for r in rs))
    return table


def beats_of(table, null_max, layers):
    """{(dir/loss, layer): bool}: the pair's best min-over-halves |cos| in the layer exceeds that direction's null maximum
    (`null_max[(dir/loss, layer)]`, the maximum over the random cosines and, for GU1 and GU2, the sign-flip permutations)."""
    out = {}
    for (dk, _rule), by in table.items():
        for layer in layers:
            best = max(v[layer][0] for (d2, _r), v in table.items() if d2 == dk)
            out[(dk, layer)] = bool(best > null_max[(dk, layer)])
    return out


def decide(table, beats, layers):
    """The rule of docs 4.1 (w4_search_lib.decide; every direction is data)."""
    kinds = {dk: "data" for (dk, _r) in table}
    return S.decide(table, kinds, beats, layers=tuple(layers))


# ------------------------------------------------------------------ the candidate edits (docs 3.5)

def _scaled(d, target):
    n = float(np.linalg.norm(d))
    if not np.isfinite(target) or target <= 0:
        raise GradError("w4g-scale", f"the target norm {target} is not positive")
    if n == 0 or not np.isfinite(n):
        raise GradError("w4g-degenerate", "the unscaled edit is zero")
    return d * (target / n)


def _check_cols(cols, ncol):
    cols = np.asarray(cols, dtype=np.int64)
    if cols.ndim != 1 or cols.size == 0 or cols.min() < 0 or cols.max() >= ncol or np.unique(cols).size != cols.size:
        raise GradError("w4g-columns", "the column list is empty, out of range or has duplicates")
    return cols


def edit_p(w, u, cols, target_norm):
    """Form P: dW[:, j] = -alpha (u . W_j) u on `cols`, alpha chosen so that ||dW||_F = target_norm."""
    w = np.asarray(_finite(w, "W"), dtype=np.float64)
    u = unit(u)
    if u.size != w.shape[0]:
        raise GradError("w4g-shape", f"u has {u.size} entries, W has {w.shape[0]} rows")
    cols = _check_cols(cols, w.shape[1])
    d = np.zeros_like(w)
    d[:, cols] = -np.outer(u, u @ w[:, cols])
    return _scaled(d, target_norm)


def edit_g(g, u, cols, target_norm):
    """Form G: dW[:, j] = -s (u . G_j) u on `cols` (the rank-1, column-restricted descent step on the loss), ||dW||_F = target_norm."""
    g = np.asarray(_finite(g, "G"), dtype=np.float64)
    u = unit(u)
    if u.size != g.shape[0]:
        raise GradError("w4g-shape", f"u has {u.size} entries, G has {g.shape[0]} rows")
    cols = _check_cols(cols, g.shape[1])
    d = np.zeros_like(g)
    d[:, cols] = -np.outer(u, u @ g[:, cols])
    return _scaled(d, target_norm)


def random_unit(n, layer, seed=SEED_CTRL):
    return unit(np.random.default_rng([seed, layer]).standard_normal(n))


def edit_cosine(d1, d2):
    return S.edit_cosine(np.asarray(d1, dtype=np.float64), np.asarray(d2, dtype=np.float64))


# ------------------------------------------------------------------ behaviour (docs 4.2)

FOUND_E, FOUND_KL = 0.5, 0.05


def found(e_share, kl_hi):
    """FOUND: E >= 0.5 on the held-out loops and the upper end of the mean-KL interval <= 0.05 nats."""
    if not (np.isfinite(e_share) and np.isfinite(kl_hi)):
        raise GradError("w4g-finite", "E or the KL bound is not finite")
    return bool(e_share >= FOUND_E and kl_hi <= FOUND_KL)


def specific(e_cand, e_ctrl):
    """SPECIFIC: the random-direction control's E is at most half of the candidate's."""
    return bool(e_ctrl <= 0.5 * e_cand)
