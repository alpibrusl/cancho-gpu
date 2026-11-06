"""docs/weight-space-w4.md: the pure parts of the W4 collateral experiments (numpy only; torch is touched only when an array is a torch tensor).

Everything here is held by tests/w4.py against direct float64 computations and planted cases. The model-facing script is
w4_collateral.py. No file or network is read here except what a function is handed."""
import hashlib

import numpy as np

SEED = 20261106
SALT = "w4-20261106"
CUTOFFS = (1e-2, 1e-3, 1e-4)


# ------------------------------------------------------------------ logits -> KL, NLL

def _is_torch(x):
    return hasattr(x, "detach") and hasattr(x, "dim")


def log_softmax(x):
    """Stable log-softmax over the last axis, for a numpy array or a torch tensor (same dtype in, same dtype out)."""
    if _is_torch(x):
        import torch
        return x - torch.logsumexp(x, dim=-1, keepdim=True)
    m = x.max(axis=-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))


def token_stats(logits_base, logits_var, targets):
    """Per position of [T, V] logits and [T] next-token ids: (KL(base||variant) in nats, NLL base, NLL variant, base entropy,
    greedy tokens agree), as float64 numpy arrays (the last as bool). The sums run over the whole vocabulary."""
    lpb, lpv = log_softmax(logits_base), log_softmax(logits_var)
    if _is_torch(lpb):
        import torch
        pb = lpb.exp()
        kl = (pb * (lpb - lpv)).sum(-1)
        ent = -(pb * lpb).sum(-1)
        tg = torch.as_tensor(targets, device=lpb.device).long().unsqueeze(-1)
        nb = -lpb.gather(-1, tg).squeeze(-1)
        nv = -lpv.gather(-1, tg).squeeze(-1)
        agree = lpb.argmax(-1) == lpv.argmax(-1)
        cv = lambda t: t.cpu().double().numpy()
        return cv(kl), cv(nb), cv(nv), cv(ent), agree.cpu().numpy()
    pb = np.exp(lpb)
    kl = (pb * (lpb - lpv)).sum(-1)
    ent = -(pb * lpb).sum(-1)
    t = np.asarray(targets).astype(np.int64)[:, None]
    nb = -np.take_along_axis(lpb, t, -1)[:, 0]
    nv = -np.take_along_axis(lpv, t, -1)[:, 0]
    agree = lpb.argmax(-1) == lpv.argmax(-1)
    return kl.astype(np.float64), nb.astype(np.float64), nv.astype(np.float64), ent.astype(np.float64), agree


def dist(x):
    """mean, p50, p90, p99, max of a 1-D array (None for an empty one)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    q = np.percentile(x, [50, 90, 99])
    return {"mean": float(x.mean()), "p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]), "max": float(x.max()), "n": int(x.size)}


def ratio_ci(sums, counts, n_boot=2000, seed=SEED):
    """The pooled mean sum(sums)/sum(counts) and its 95% percentile interval, resampling whole sequences (their positions are
    not independent). `sums[i]` and `counts[i]` are one sequence's total and number of positions."""
    s, c = np.asarray(sums, dtype=np.float64), np.asarray(counts, dtype=np.float64)
    if s.size == 0 or c.sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, s.size, size=(n_boot, s.size))
    means = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(s.sum() / c.sum()), "lo": float(lo), "hi": float(hi)}


# ------------------------------------------------------------------ the gates (docs/weight-space-w4.md 4.3)

GATE = {"kl_mean_hi": 0.02, "kl_domain_mean": 0.05, "kl_tok_p99": 0.5, "ppl_hi": 0.02, "ppl_domain": 0.05,
        "rep4_delta": 0.03, "new_loops": 3}
STRICT = {"kl_mean_hi": 0.002, "ppl_hi": 0.003, "kl_domain_mean": 0.01}


def gate(summary, rep=None, g=GATE):
    """{clause: (ok, detail)} for one variant's summary (as w4_collateral writes it). `summary["all"]` and `summary["domains"][d]`
    hold `kl` (the pooled mean and its interval), `kl_tok` (the token distribution), `dppl` (exp(mean d) - 1 and its interval) and
    `finite`; `rep` is {"rep4_delta": paired mean difference on the general prompts, "new_loops": count} or None."""
    out = {}
    a = summary["all"]
    out["kl_mean"] = (a["kl"]["hi"] <= g["kl_mean_hi"], f"upper bound {a['kl']['hi']:.5f} vs {g['kl_mean_hi']}")
    worst = max(((d, s["kl"]["mean"]) for d, s in summary["domains"].items() if s.get("kl")), key=lambda x: x[1], default=(None, 0.0))
    out["kl_domain"] = (worst[1] <= g["kl_domain_mean"], f"worst domain {worst[0]} {worst[1]:.5f} vs {g['kl_domain_mean']}")
    out["kl_p99"] = (a["kl_tok"]["p99"] <= g["kl_tok_p99"], f"token p99 {a['kl_tok']['p99']:.5f} vs {g['kl_tok_p99']}")
    out["ppl_all"] = (a["dppl"]["hi"] <= g["ppl_hi"], f"upper bound {a['dppl']['hi']:.5f} vs {g['ppl_hi']}")
    wd = max(((d, s["dppl"]["mean"]) for d, s in summary["domains"].items() if s.get("dppl")), key=lambda x: x[1], default=(None, 0.0))
    out["ppl_domain"] = (wd[1] <= g["ppl_domain"], f"worst domain {wd[0]} {wd[1]:.5f} vs {g['ppl_domain']}")
    if rep is not None:
        out["rep4"] = (rep["rep4_delta"] <= g["rep4_delta"], f"paired delta {rep['rep4_delta']:.5f} vs {g['rep4_delta']}")
        out["new_loops"] = (rep["new_loops"] <= g["new_loops"], f"{rep['new_loops']} vs {g['new_loops']}")
    out["finite"] = (bool(a.get("finite", True)), "no NaN or infinity, equal position counts")
    return out


def passes(gates):
    return all(ok for ok, _ in gates.values())


def strict_ok(summary):
    a = summary["all"]
    worst = max((s["kl"]["mean"] for s in summary["domains"].values() if s.get("kl")), default=0.0)
    return a["kl"]["hi"] <= STRICT["kl_mean_hi"] and a["dppl"]["hi"] <= STRICT["ppl_hi"] and worst <= STRICT["kl_domain_mean"]


# ------------------------------------------------------------------ streaming second moment, null space

class Moment:
    """sum of a a^T over positions, in float64, fed by float32 blocks or by Gram matrices computed elsewhere (on a device)."""

    def __init__(self, d):
        self.c = np.zeros((d, d), dtype=np.float64)
        self.n = 0

    def add(self, a):
        a = np.asarray(a, dtype=np.float32)
        self.c += (a.T @ a).astype(np.float64)
        self.n += a.shape[0]

    def add_gram(self, g, n):
        self.c += np.asarray(g, dtype=np.float64)
        self.n += int(n)


def eig_of(c):
    """Eigenvalues (descending, clipped at 0) and eigenvectors (columns) of a symmetric float64 matrix."""
    w, q = np.linalg.eigh((c + c.T) / 2)
    order = np.argsort(w)[::-1]
    return np.clip(w[order], 0.0, None), q[:, order]


def retained(eigvals, n, cutoff):
    """How many directions have singular value of K0 above `cutoff` times the largest: s_i = sqrt(n * lambda_i), as in
    `(s > cutoff * s[0]).sum()` of the proposal's sketch."""
    s = np.sqrt(n * np.asarray(eigvals, dtype=np.float64))
    if s.size == 0 or s[0] == 0:
        return 0
    return int((s > cutoff * s[0]).sum())


def null_project(dw, eigvals, eigvecs, n, cutoff):
    """(dw P, r) with P = I - Q_r Q_r^T the projector off the r retained directions, applied to the input side of dw
    ([out, in]). Uses whichever of the retained or the discarded basis is smaller."""
    dw = np.asarray(dw, dtype=np.float64)
    r = retained(eigvals, n, cutoff)
    d = eigvecs.shape[1]
    if r <= d - r:
        q = eigvecs[:, :r]
        return dw - (dw @ q) @ q.T, r
    q = eigvecs[:, r:]
    return (dw @ q) @ q.T, r


def rank1(dw):
    """sigma, u, v of the top singular triplet of dw (signs fixed so that the largest |u| entry is positive)."""
    u, s, vt = np.linalg.svd(np.asarray(dw, dtype=np.float64), full_matrices=False)
    u1, v1 = u[:, 0], vt[0]
    if u1[np.argmax(np.abs(u1))] < 0:
        u1, v1 = -u1, -v1
    return float(s[0]), u1, v1


def amplitude_ratio(vec_c, vec, c):
    """sqrt( vec_c^T C vec_c / vec^T C vec ): the share of the edit's amplitude on tokens whose input second moment is C."""
    den = float(vec @ c @ vec)
    return float(np.sqrt(max(float(vec_c @ c @ vec_c), 0.0) / den)) if den > 0 else float("nan")


# ------------------------------------------------------------------ bf16, quantisation, random edits

def bf16_round(x):
    """float32 -> the nearest bfloat16 (ties to even) -> float32."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).copy()
    u += ((u >> 16) & 1) + 0x7FFF
    u &= 0xFFFF0000
    return u.view(np.float32)


def fake_quant(w, bits, group=None):
    """Round-to-nearest symmetric quantisation of [out, in] weights, returned as float32: per output row (group None) or per
    groups of `group` along the input axis; scale = absmax / (2^(bits-1) - 1)."""
    w = np.asarray(w, dtype=np.float32)
    o, i = w.shape
    g = i if group is None else group
    if i % g:
        raise ValueError(f"input size {i} is not a multiple of the group {g}")
    x = w.reshape(o, i // g, g)
    qmax = float(2 ** (bits - 1) - 1)
    scale = np.abs(x).max(-1, keepdims=True) / qmax
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    q = np.clip(np.rint(x / scale), -qmax - 1, qmax)
    return (q * scale).reshape(o, i).astype(np.float32)


def random_edit(dw, support_size, rng):
    """A rank-1 edit of the same Frobenius norm as `dw`: `support_size` random columns carrying dw's top right-singular vector's
    nonzero values in random order, and a random unit output direction."""
    dw = np.asarray(dw, dtype=np.float64)
    sigma, u, v = rank1(dw)
    nz = v[np.argsort(-np.abs(v))[:support_size]]
    cols = rng.choice(dw.shape[1], size=support_size, replace=False)
    vr = np.zeros(dw.shape[1])
    vr[cols] = rng.permutation(nz)
    vr /= np.linalg.norm(vr)
    ur = rng.standard_normal(dw.shape[0])
    ur /= np.linalg.norm(ur)
    return np.linalg.norm(dw) * np.outer(ur, vr), cols


# ------------------------------------------------------------------ E1a: weights-only look at an edit (docs/weight-space-w4.md 6.4)

def rank_of(columns, scores):
    """The 1-based ranks (1 = largest) of `columns` in `scores`."""
    order = np.argsort(-np.asarray(scores))
    rank = np.empty(len(scores), dtype=np.int64)
    rank[order] = np.arange(1, len(scores) + 1)
    return rank[np.asarray(columns)]


def spearman(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)

    def rk(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[o] = np.arange(len(x))
        # average ranks over ties
        xs = x[o]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and xs[j + 1] == xs[i]:
                j += 1
            r[o[i:j + 1]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rk(a), rk(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def e1a(w_base, dw, top=96):
    """The weights-only facts of docs/weight-space-w4.md 6.4 for one tensor: `w_base`, `dw` are [out, in]."""
    w_base, dw = np.asarray(w_base, dtype=np.float64), np.asarray(dw, dtype=np.float64)
    sigma, u, v = rank1(dw)
    cols = np.nonzero(np.abs(dw).sum(0) > 0)[0]
    proj = u @ w_base                      # u . W_j for every column j
    colnorm = np.linalg.norm(w_base, axis=0)
    x = proj[cols]
    y = v[cols] * sigma
    # v_j sigma = c (u . W_j) over the support
    c = float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else float("nan")
    rho = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    r1 = rank_of(cols, np.abs(proj))
    r2 = rank_of(cols, np.abs(proj) / np.maximum(colnorm, 1e-30))
    rn = rank_of(cols, colnorm)
    idx = np.arange(w_base.shape[1])
    member = np.zeros(w_base.shape[1])
    member[cols] = 1.0
    return {
        "sigma": sigma, "support": int(cols.size), "slope_c": c, "pearson_rho": rho,
        "rank_by_abs_uW": {"median": float(np.median(r1)), "in_top64": int((r1 <= 64).sum()), "in_top96": int((r1 <= top).sum())},
        "rank_by_abs_uW_normalised": {"median": float(np.median(r2)), "in_top64": int((r2 <= 64).sum()), "in_top96": int((r2 <= top).sum())},
        "rank_by_colnorm": {"median": float(np.median(rn)), "in_top64": int((rn <= 64).sum()), "in_top96": int((rn <= top).sum())},
        "spearman_index_vs": {"abs_uW": spearman(idx, np.abs(proj)), "abs_uW_normalised": spearman(idx, np.abs(proj) / np.maximum(colnorm, 1e-30)),
                              "colnorm": spearman(idx, colnorm), "membership": spearman(idx, member)},
        "median_column": float(np.median(cols)), "columns_below_500": int((cols < 500).sum()),
        "h1": bool(-1.2 <= c <= -0.8 and abs(rho) >= 0.9), "h2": bool((r1 <= top).sum() >= 48),
    }


# ------------------------------------------------------------------ repetition (docs/weight-space-w4.md 4.4)

def rep4(tokens):
    """1 - distinct 4-grams / 4-grams of a token list; None when it has fewer than 36 tokens."""
    n = len(tokens)
    if n < 36:
        return None
    grams = {tuple(tokens[i:i + 4]) for i in range(n - 3)}
    return 1.0 - len(grams) / (n - 3)


def tail_loop(tokens, max_period=64):
    """The smallest period p (1..max_period) such that the last max(32, 4p) tokens are p-periodic, or 0."""
    n = len(tokens)
    for p in range(1, max_period + 1):
        m = max(32, 4 * p)
        if m > n:
            break
        tail = tokens[n - m:]
        if all(tail[i] == tail[i + p] for i in range(m - p)):
            return p
    return 0


# ------------------------------------------------------------------ corpus rules (docs/weight-space-w4.md 3.1)

def split_of(unit_id):
    """'fit' or 'eval': a function of the unit's id and the salt, not of any order."""
    h = hashlib.sha256((unit_id + SALT).encode("utf-8")).digest()
    return "fit" if h[0] % 2 == 0 else "eval"


def cut_windows(ids, max_len=2048, min_len=32):
    """Consecutive windows of at most max_len tokens; a window shorter than min_len is dropped."""
    out = [ids[i:i + max_len] for i in range(0, len(ids), max_len)]
    return [w for w in out if len(w) >= min_len]


def is_spanish(text):
    t = text.lower()
    import re
    es = len(re.findall(r"\b(el|la|los|las|de|que|para|con|una|por|como|es|del)\b", t))
    en = len(re.findall(r"\b(the|and|of|to|is|that|for|with|in|are)\b", t))
    return es > 30 and es > en * 0.8


def has_cjk(text, share=0.1):
    n = sum(1 for c in text if "　" <= c <= "鿿" or "가" <= c <= "힯")
    return n > share * max(len(text), 1)
