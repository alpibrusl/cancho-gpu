#!/usr/bin/env python3
"""The specification of the Qwen2 forward pass (docs/resident.md section 20; roadmap M7.1): binary64 numpy written from the mathematics of the architecture, with no framework.

Qwen2: tokens -> embedding rows -> L blocks of (RMSNorm, q/k/v projections with biases, rotary embedding on q and k, grouped-query causal attention, o projection, residual;
RMSNorm, SwiGLU MLP, residual) -> RMSNorm -> the tied embedding as the output layer. The rotary form is HF's "rotate half": the two halves of a head are the pairs, so with
c, s = cos, sin of position * theta^(-j / (dh / 2)), a head (x1, x2) becomes (x1 c - x2 s, x2 c + x1 s).

Parameters are a dict of arrays (names in `names`); `init_params` makes a random Qwen-shaped model for tests, `load_params` reads the files of convert.py.
`forward` gives the residual stream after every block (to localise an error layer by layer) and `head` the loss of each position and the arg max of its logits, chunked over the vocabulary."""
import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    layers: int = 24
    d: int = 896
    h: int = 14      # query heads
    hkv: int = 2     # key/value heads
    dh: int = 64
    ff: int = 4864
    v: int = 151936
    t: int = 512     # tokens of a sequence
    nb: int = 8      # sequences of a batch
    eps: float = 1e-6
    theta: float = 1e6

    @property
    def n(self):
        return self.nb * self.t

    @property
    def dkv(self):
        return self.hkv * self.dh

    @property
    def group(self):
        return self.h // self.hkv


def from_hf(cfg, t=512, nb=8):
    return Config(layers=cfg["num_hidden_layers"], d=cfg["hidden_size"], h=cfg["num_attention_heads"], hkv=cfg["num_key_value_heads"],
                  dh=cfg["hidden_size"] // cfg["num_attention_heads"], ff=cfg["intermediate_size"], v=cfg["vocab_size"], t=t, nb=nb,
                  eps=cfg["rms_norm_eps"], theta=cfg["rope_theta"])


def layer_names(l):
    return [f"l{l:02d}.{k}" for k in ("ln1", "wqkv", "bqkv", "wo", "ln2", "wgu", "wd")]


def names(c):
    return ["emb", "norm"] + [n for l in range(c.layers) for n in layer_names(l)]


def shapes(c):
    out = {"emb": (c.v, c.d), "norm": (c.d,)}
    for l in range(c.layers):
        p = f"l{l:02d}."
        out.update({p + "ln1": (c.d,), p + "wqkv": (c.d + 2 * c.dkv, c.d), p + "bqkv": (c.d + 2 * c.dkv,), p + "wo": (c.d, c.d), p + "ln2": (c.d,),
                    p + "wgu": (2 * c.ff, c.d), p + "wd": (c.d, c.ff)})
    return out


def init_params(c, seed=1):
    """A random Qwen-shaped model. Weights are rounded to f16 (the device holds them so) and the biases are large enough to matter."""
    rng = np.random.default_rng(seed)
    P = {}
    for name, sh in shapes(c).items():
        kind = name.split(".")[-1]
        if kind in ("ln1", "ln2", "norm"):
            w = 1.0 + 0.2 * rng.standard_normal(sh)
        elif kind == "bqkv":
            w = 0.3 * rng.standard_normal(sh)
        elif kind == "emb":
            w = 0.5 * rng.standard_normal(sh)
        else:
            w = rng.standard_normal(sh) / np.sqrt(sh[1])
        P[name] = w.astype(np.float16).astype(np.float64) if kind not in ("ln1", "ln2", "norm", "bqkv") else w.astype(np.float32).astype(np.float64)
    return P


def load_one(directory, name, sh, G=None):
    """One parameter (binary64) from the files convert.py wrote. A quantised matrix (`NAME.q`, `.s`, `.g`: quant.py pack, convert.py --quant; docs/resident.md section 26) is the decoded bytes rounded to f16,
    which is what the session's dequantise kernel produces; `G`, its group size, is read from the manifest when not given."""
    kind = name.split(".")[-1]
    dt = "<f4" if kind in ("ln1", "ln2", "norm", "bqkv") else "<f2"
    ext = "f32" if dt == "<f4" else "f16"
    if os.path.exists(os.path.join(directory, name + ".q")):
        import nvfp4
        G = G or json.load(open(os.path.join(directory, "manifest.json")))["quant"]["group"]
        n_, k_ = sh
        q = np.fromfile(os.path.join(directory, name + ".q"), dtype=np.uint8).reshape(n_, k_ // 2)
        sc = np.fromfile(os.path.join(directory, name + ".s"), dtype=np.uint8).reshape(n_, k_ // G)
        rw = np.fromfile(os.path.join(directory, name + ".g"), dtype="<f4")
        return nvfp4.unpack_rows(q, sc, rw, G).astype(np.float16).astype(np.float64)
    a = np.fromfile(os.path.join(directory, f"{name}.{ext}"), dtype=dt)
    if a.size != int(np.prod(sh)):
        raise SystemExit(f"{name}: {a.size} words, the model has {int(np.prod(sh))}")
    return a.astype(np.float64).reshape(sh)


def load_params(directory, c):
    """The parameters (binary64) from the files convert.py wrote."""
    return {name: load_one(directory, name, sh) for name, sh in shapes(c).items()}


def config_of(directory, t=512, nb=8):
    return from_hf(json.load(open(os.path.join(directory, "manifest.json")))["config"], t, nb)


def rms(x, g, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g


def rope_tables(c):
    """cos, sin of [t, dh / 2]: position * theta^(-j / (dh / 2))."""
    half = c.dh // 2
    ang = np.arange(c.t)[:, None] * c.theta ** (-np.arange(half) / half)[None, :]
    return np.cos(ang), np.sin(ang)


def rope(x, cs, sn, heads, c):
    """x [nb t, heads dh]: every head rotated by the position of its row (row mod t)."""
    half = c.dh // 2
    y = x.reshape(c.nb, c.t, heads, c.dh)
    x1, x2 = y[..., :half], y[..., half:]
    co, si = cs[None, :, None, :], sn[None, :, None, :]
    return np.concatenate([x1 * co - x2 * si, x2 * co + x1 * si], -1).reshape(c.n, heads * c.dh)


def silu(x):
    return x / (1.0 + np.exp(-x))


def attention(q, k, v, c):
    """Causal grouped-query attention: q [n, h dh], k and v [n, hkv dh]; query head h reads key/value head h // group."""
    qh = q.reshape(c.nb, c.t, c.h, c.dh)
    kh = k.reshape(c.nb, c.t, c.hkv, c.dh)
    vh = v.reshape(c.nb, c.t, c.hkv, c.dh)
    out = np.zeros_like(qh)
    mask = np.triu(np.full((c.t, c.t), -np.inf), 1)
    for h in range(c.h):
        g = h // c.group
        s = np.einsum("bid,bjd->bij", qh[:, :, h], kh[:, :, g]) / np.sqrt(c.dh) + mask
        s -= s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        out[:, :, h] = np.einsum("bij,bjd->bid", p, vh[:, :, g])
    return out.reshape(c.n, c.h * c.dh)


def layer(c, P, l, x, cs, sn):
    p = f"l{l:02d}."
    h = rms(x, P[p + "ln1"], c.eps)
    qkv = h @ P[p + "wqkv"].T + P[p + "bqkv"]
    q, k, v = qkv[:, :c.d], qkv[:, c.d:c.d + c.dkv], qkv[:, c.d + c.dkv:]
    q, k = rope(q, cs, sn, c.h, c), rope(k, cs, sn, c.hkv, c)
    x = x + attention(q, k, v, c) @ P[p + "wo"].T
    h = rms(x, P[p + "ln2"], c.eps)
    gu = h @ P[p + "wgu"].T
    return x + (silu(gu[:, :c.ff]) * gu[:, c.ff:]) @ P[p + "wd"].T


def forward(c, P, tokens, layers=None):
    """tokens [nb, t] (ints). Returns the residual stream after the embedding and after each block: a list of [n, d] arrays (the last is before the final norm)."""
    cs, sn = rope_tables(c)
    x = P["emb"][np.asarray(tokens).reshape(-1)]
    xs = [x]
    for l in range(c.layers if layers is None else layers):
        x = layer(c, P, l, x, cs, sn)
        xs.append(x)
    return xs


def head(c, P, x, targets, chunk=16384):
    """The final norm, the tied output layer and the cross-entropy: per row, lse(logits) - logits[target], and the arg max of the logits. Chunked over the vocabulary."""
    h = rms(x, P["norm"], c.eps)
    n = h.shape[0]
    m = np.full(n, -np.inf)
    z = np.zeros(n)
    tl = np.zeros(n)
    best = np.zeros(n, dtype=np.int64)
    tg = np.asarray(targets).reshape(-1)
    for j in range(0, c.v, chunk):
        lg = h @ P["emb"][j:j + chunk].T
        cm = lg.max(-1)
        am = lg.argmax(-1) + j
        nm = np.maximum(m, cm)
        z = z * np.exp(m - nm) + np.exp(lg - nm[:, None]).sum(-1)
        best = np.where(cm > m, am, best)
        m = nm
        inside = (tg >= j) & (tg < j + chunk)
        tl[inside] = lg[np.nonzero(inside)[0], tg[inside] - j]
    return np.log(z) + m - tl, best


def mean_loss(rows, c):
    """The mean over the positions that have a next token (the last of each sequence has none), as transformers' labels=input_ids does."""
    return float(rows.reshape(c.nb, c.t)[:, :-1].mean())


def next_targets(tokens):
    """The target of each position: the next token of its sequence (the last position's target is itself, and is not counted)."""
    t = np.asarray(tokens)
    return np.concatenate([t[:, 1:], t[:, -1:]], 1)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LoRA and the backward pass (docs/resident.md section 21; roadmap M7.2). The base weights, the embeddings (tied with the output layer), the norm gains and the biases are frozen: only the
# adapters of the seven linear layers of every block are trained, y = x W^T + (alpha / r) (x A^T) B^T as peft writes it, A [r, in] and B [out, r] per projection. The backward is written by
# hand (as gpt_ref.py's is) and is checked against finite differences (`fd_check`) and, on the real weights, against peft's autograd (examples/qwen/compare_lora.py).

PROJ = ("q", "k", "v", "o", "gate", "up", "down")


def proj_dims(c, p):
    """(in, out) of the projection p of a block."""
    return {"q": (c.d, c.d), "k": (c.d, c.dkv), "v": (c.d, c.dkv), "o": (c.d, c.d), "gate": (c.d, c.ff), "up": (c.d, c.ff), "down": (c.ff, c.d)}[p]


def lora_names(c):
    return [f"l{l:02d}.{p}.{ab}" for l in range(c.layers) for p in PROJ for ab in ("A", "B")]


def init_lora(c, r=16, seed=7, b_std=0.0):
    """Adapters as peft initialises them: A ~ U(-1/sqrt(in), 1/sqrt(in)) (kaiming_uniform with a = sqrt(5)), B = 0; with `b_std` > 0 the B matrices are random instead (gradients of A are zero while B is
    zero, so a check of the whole backward needs a B that is not)."""
    rng = np.random.default_rng(seed)
    AB = {}
    for l in range(c.layers):
        for p in PROJ:
            i, o = proj_dims(c, p)
            AB[f"l{l:02d}.{p}.A"] = rng.uniform(-1.0, 1.0, (r, i)) / np.sqrt(i)
            AB[f"l{l:02d}.{p}.B"] = b_std * rng.standard_normal((o, r))
    return AB


def rms_back(dy, x, g, eps):
    """dx of y = x rsqrt(mean(x^2) + eps) g (the gain is frozen: no gradient for it)."""
    rs = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    u = dy * g
    return u * rs - x * (rs ** 3) * (u * x).mean(-1, keepdims=True)


def rope_back(dy, cs, sn, heads, c):
    half = c.dh // 2
    y = dy.reshape(c.nb, c.t, heads, c.dh)
    d1, d2 = y[..., :half], y[..., half:]
    co, si = cs[None, :, None, :], sn[None, :, None, :]
    return np.concatenate([d1 * co + d2 * si, d2 * co - d1 * si], -1).reshape(c.n, heads * c.dh)


def lin(x, W, A, B, s):
    """x W^T + s (x A^T) B^T and the middle activation x A^T."""
    mid = x @ A.T
    return x @ W.T + s * (mid @ B.T), mid


def attention_cached(q, k, v, c):
    qh = q.reshape(c.nb, c.t, c.h, c.dh)
    kh = k.reshape(c.nb, c.t, c.hkv, c.dh)
    vh = v.reshape(c.nb, c.t, c.hkv, c.dh)
    out = np.zeros_like(qh)
    pr = np.zeros((c.nb, c.h, c.t, c.t))
    mask = np.triu(np.full((c.t, c.t), -np.inf), 1)
    for h in range(c.h):
        g = h // c.group
        s = np.einsum("bid,bjd->bij", qh[:, :, h], kh[:, :, g]) / np.sqrt(c.dh) + mask
        s -= s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        pr[:, h] = p
        out[:, :, h] = np.einsum("bij,bjd->bid", p, vh[:, :, g])
    return out.reshape(c.n, c.h * c.dh), pr


def attention_back(do, q, k, v, pr, c):
    """dq, dk, dv of the causal grouped-query attention; the key/value gradients are summed over the `group` query heads that read each key/value head."""
    qh = q.reshape(c.nb, c.t, c.h, c.dh)
    kh = k.reshape(c.nb, c.t, c.hkv, c.dh)
    vh = v.reshape(c.nb, c.t, c.hkv, c.dh)
    doh = do.reshape(c.nb, c.t, c.h, c.dh)
    dq, dk, dv = np.zeros_like(qh), np.zeros_like(kh), np.zeros_like(vh)
    for h in range(c.h):
        g = h // c.group
        p = pr[:, h]
        dp = np.einsum("bid,bjd->bij", doh[:, :, h], vh[:, :, g])
        ds = p * (dp - (dp * p).sum(-1, keepdims=True)) / np.sqrt(c.dh)
        dq[:, :, h] = np.einsum("bij,bjd->bid", ds, kh[:, :, g])
        dk[:, :, g] += np.einsum("bij,bid->bjd", ds, qh[:, :, h])
        dv[:, :, g] += np.einsum("bij,bid->bjd", p, doh[:, :, h])
    return dq.reshape(c.n, c.h * c.dh), dk.reshape(c.n, c.dkv), dv.reshape(c.n, c.dkv)


def lora_forward(c, P, AB, tokens, s):
    """The forward pass with adapters, keeping what the backward needs. Returns (xs, cache): xs the residual stream after the embedding and every block."""
    cs, sn = rope_tables(c)
    x = P["emb"][np.asarray(tokens).reshape(-1)]
    xs, cache = [x], []
    for l in range(c.layers):
        p, a = f"l{l:02d}.", lambda n: (AB[f"l{l:02d}.{n}.A"], AB[f"l{l:02d}.{n}.B"])
        h1 = rms(x, P[p + "ln1"], c.eps)
        W, bq = P[p + "wqkv"], P[p + "bqkv"]
        q, mq = lin(h1, W[:c.d], *a("q"), s)
        k, mk = lin(h1, W[c.d:c.d + c.dkv], *a("k"), s)
        v, mv = lin(h1, W[c.d + c.dkv:], *a("v"), s)
        q, k, v = q + bq[:c.d], k + bq[c.d:c.d + c.dkv], v + bq[c.d + c.dkv:]
        q, k = rope(q, cs, sn, c.h, c), rope(k, cs, sn, c.hkv, c)
        att, pr = attention_cached(q, k, v, c)
        ao, mo = lin(att, P[p + "wo"], *a("o"), s)
        x1 = x + ao
        h2 = rms(x1, P[p + "ln2"], c.eps)
        gt, mg = lin(h2, P[p + "wgu"][:c.ff], *a("gate"), s)
        up, mu = lin(h2, P[p + "wgu"][c.ff:], *a("up"), s)
        act = silu(gt) * up
        dn, md = lin(act, P[p + "wd"], *a("down"), s)
        cache.append(dict(x=x, h1=h1, mid=dict(q=mq, k=mk, v=mv, o=mo, gate=mg, up=mu, down=md), q=q, k=k, v=v, pr=pr, att=att, x1=x1, h2=h2, gt=gt, up=up, act=act))
        x = x1 + dn
        xs.append(x)
    return xs, cache


def head_back(c, P, x, tokens, chunk=16384):
    """Loss (mean over the positions with a next token), and the gradient of it with respect to the residual stream x entering the final norm. Two passes over the vocabulary in chunks."""
    h = rms(x, P["norm"], c.eps)
    tg = next_targets(tokens).reshape(-1)
    w = np.ones((c.nb, c.t))
    w[:, -1] = 0.0
    w = w.reshape(-1)
    nc = w.sum()
    rows, _ = head(c, P, x, tg, chunk)
    # the log-sum-exp of every row, from the chunks
    m = np.full(c.n, -np.inf)
    z = np.zeros(c.n)
    for j in range(0, c.v, chunk):
        lg = h @ P["emb"][j:j + chunk].T
        nm = np.maximum(m, lg.max(-1))
        z = z * np.exp(m - nm) + np.exp(lg - nm[:, None]).sum(-1)
        m = nm
    lse = np.log(z) + m
    dh = np.zeros_like(h)
    for j in range(0, c.v, chunk):
        lg = h @ P["emb"][j:j + chunk].T
        dl = np.exp(lg - lse[:, None])
        inside = (tg >= j) & (tg < j + chunk)
        dl[np.nonzero(inside)[0], tg[inside] - j] -= 1.0
        dh += (dl * (w / nc)[:, None]) @ P["emb"][j:j + chunk]
    return float((rows * w).sum() / nc), rms_back(dh, x, P["norm"], c.eps)


def lora_loss_grads(c, P, AB, tokens, s=2.0, grads=True):
    """The mean loss, and with `grads` the gradient of it with respect to every adapter matrix (a dict like AB)."""
    xs, cache = lora_forward(c, P, AB, tokens, s)
    if not grads:
        tg = next_targets(tokens).reshape(-1)
        rows, _ = head(c, P, xs[-1], tg)
        return mean_loss(rows, c), None
    loss, g = head_back(c, P, xs[-1], tokens)
    cs, sn = rope_tables(c)
    G = {}
    for l in reversed(range(c.layers)):
        p, k_ = f"l{l:02d}.", cache[l]
        a = lambda n: (AB[f"l{l:02d}.{n}.A"], AB[f"l{l:02d}.{n}.B"])

        def back(name, dy, x, W):
            """dx of y = x W^T + s (x A^T) B^T given dy, and the gradients of A and B (stored into G)."""
            A, B = a(name)
            dmid = s * (dy @ B)
            G[f"l{l:02d}.{name}.B"] = s * dy.T @ k_["mid"][name]
            G[f"l{l:02d}.{name}.A"] = dmid.T @ x
            return dy @ W + dmid @ A

        dact = back("down", g, k_["act"], P[p + "wd"])
        sg = 1.0 / (1.0 + np.exp(-k_["gt"]))
        dgt = dact * k_["up"] * (sg * (1.0 + k_["gt"] * (1.0 - sg)))
        dup = dact * (k_["gt"] * sg)
        dh2 = back("gate", dgt, k_["h2"], P[p + "wgu"][:c.ff]) + back("up", dup, k_["h2"], P[p + "wgu"][c.ff:])
        g1 = g + rms_back(dh2, k_["x1"], P[p + "ln2"], c.eps)
        datt = back("o", g1, k_["att"], P[p + "wo"])
        dq, dk, dv = attention_back(datt, k_["q"], k_["k"], k_["v"], k_["pr"], c)
        dq, dk = rope_back(dq, cs, sn, c.h, c), rope_back(dk, cs, sn, c.hkv, c)
        W = P[p + "wqkv"]
        dh1 = back("q", dq, k_["h1"], W[:c.d]) + back("k", dk, k_["h1"], W[c.d:c.d + c.dkv]) + back("v", dv, k_["h1"], W[c.d + c.dkv:])
        g = g1 + rms_back(dh1, k_["x"], P[p + "ln1"], c.eps)
    return loss, G


def fd_check(c, P, AB, tokens, s=2.0, eps=1e-6, seed=0):
    """Finite differences: for every adapter matrix a random direction d, (L(AB + eps d) - L(AB - eps d)) / 2 eps against <grad, d>. Returns {name: relative difference} and the loss."""
    loss, G = lora_loss_grads(c, P, AB, tokens, s)
    rng = np.random.default_rng(seed)
    out = {}
    for name in AB:
        d = rng.standard_normal(AB[name].shape)
        d /= np.linalg.norm(d)
        up, dn = dict(AB), dict(AB)
        up[name], dn[name] = AB[name] + eps * d, AB[name] - eps * d
        fd = (lora_loss_grads(c, P, up, tokens, s, grads=False)[0] - lora_loss_grads(c, P, dn, tokens, s, grads=False)[0]) / (2 * eps)
        an = float((G[name] * d).sum())
        out[name] = abs(fd - an) / max(abs(fd), abs(an), 1e-12)
    return out, loss


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Full fine-tuning (docs/resident.md section 24; roadmap M7.4, part B): every parameter is trained, the embedding table (tied with the output layer) included. The gradient of the table is the sum of two
# scatter-adds: the output layer's (dlogits^T h, one row per vocabulary entry) and the input side's (the gradient of the embedding of each token added into the row of its id).

def rms_back_full(dy, x, g, eps):
    """dx and the gain's gradient of y = x rsqrt(mean(x^2) + eps) g."""
    rs = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    u = dy * g
    return u * rs - x * (rs ** 3) * (u * x).mean(-1, keepdims=True), (dy * x * rs).sum(0)


def head_back_full(c, P, x, tokens, chunk=16384):
    """Like head_back, and also the gradients of the final norm's gain and of the table's rows through the output layer. Returns (loss, dx, dnorm, demb [v, d])."""
    h = rms(x, P["norm"], c.eps)
    tg = next_targets(tokens).reshape(-1)
    w = np.ones((c.nb, c.t))
    w[:, -1] = 0.0
    w = w.reshape(-1)
    nc = w.sum()
    rows, _ = head(c, P, x, tg, chunk)
    m = np.full(c.n, -np.inf)
    z = np.zeros(c.n)
    for j in range(0, c.v, chunk):
        lg = h @ P["emb"][j:j + chunk].T
        nm = np.maximum(m, lg.max(-1))
        z = z * np.exp(m - nm) + np.exp(lg - nm[:, None]).sum(-1)
        m = nm
    lse = np.log(z) + m
    dh = np.zeros_like(h)
    demb = np.zeros((c.v, c.d))
    for j in range(0, c.v, chunk):
        lg = h @ P["emb"][j:j + chunk].T
        dl = np.exp(lg - lse[:, None])
        inside = (tg >= j) & (tg < j + chunk)
        dl[np.nonzero(inside)[0], tg[inside] - j] -= 1.0
        dl = dl * (w / nc)[:, None]
        dh += dl @ P["emb"][j:j + chunk]
        demb[j:j + chunk] = dl.T @ h
    dx, dg = rms_back_full(dh, x, P["norm"], c.eps)
    return float((rows * w).sum() / nc), dx, dg, demb


def full_loss_grads(c, P, tokens, grads=True):
    """The mean loss and the gradient of it with respect to every parameter ({name: array} like P). Written by hand like lora_loss_grads (the adapters of the forward are absent)."""
    zero = {}
    for l in range(c.layers):
        for p in PROJ:
            i, o = proj_dims(c, p)
            zero[f"l{l:02d}.{p}.A"] = np.zeros((1, i))
            zero[f"l{l:02d}.{p}.B"] = np.zeros((o, 1))
    xs, cache = lora_forward(c, P, zero, tokens, 0.0)
    if not grads:
        rows, _ = head(c, P, xs[-1], next_targets(tokens).reshape(-1))
        return mean_loss(rows, c), None
    loss, g, dnorm, demb = head_back_full(c, P, xs[-1], tokens)
    G = {"norm": dnorm}
    cs, sn = rope_tables(c)
    for l in reversed(range(c.layers)):
        p, k_ = f"l{l:02d}.", cache[l]
        dact_in = g
        G[p + "wd"] = dact_in.T @ k_["act"]
        dact = dact_in @ P[p + "wd"]
        sg = 1.0 / (1.0 + np.exp(-k_["gt"]))
        dgt = dact * k_["up"] * (sg * (1.0 + k_["gt"] * (1.0 - sg)))
        dup = dact * (k_["gt"] * sg)
        G[p + "wgu"] = np.concatenate([dgt.T @ k_["h2"], dup.T @ k_["h2"]])
        dh2 = dgt @ P[p + "wgu"][:c.ff] + dup @ P[p + "wgu"][c.ff:]
        dx2, G[p + "ln2"] = rms_back_full(dh2, k_["x1"], P[p + "ln2"], c.eps)
        g1 = g + dx2
        G[p + "wo"] = g1.T @ k_["att"]
        datt = g1 @ P[p + "wo"]
        dq, dk, dv = attention_back(datt, k_["q"], k_["k"], k_["v"], k_["pr"], c)
        dq, dk = rope_back(dq, cs, sn, c.h, c), rope_back(dk, cs, sn, c.hkv, c)
        dqkv = np.concatenate([dq, dk, dv], 1)
        G[p + "bqkv"] = dqkv.sum(0)
        G[p + "wqkv"] = dqkv.T @ k_["h1"]
        dh1 = dqkv @ P[p + "wqkv"]
        dx1, G[p + "ln1"] = rms_back_full(dh1, k_["x"], P[p + "ln1"], c.eps)
        g = g1 + dx1
    tok = np.asarray(tokens).reshape(-1)
    np.add.at(demb, tok, g)  # the input side of the tied table: the gradient of each token's embedding added into its row
    G["emb"] = demb
    return loss, G


def fd_check_full(c, P, tokens, names=None, eps=1e-6, seed=0):
    """Finite differences along a random direction for each parameter in `names` (default all): {name: relative difference}, and the loss."""
    loss, G = full_loss_grads(c, P, tokens)
    rng = np.random.default_rng(seed)
    out = {}
    for name in names or list(G):
        d = rng.standard_normal(P[name].shape)
        d /= np.linalg.norm(d)
        up, dn = dict(P), dict(P)
        up[name], dn[name] = P[name] + eps * d, P[name] - eps * d
        fd = (full_loss_grads(c, up, tokens, grads=False)[0] - full_loss_grads(c, dn, tokens, grads=False)[0]) / (2 * eps)
        an = float((G[name] * d).sum())
        out[name] = abs(fd - an) / max(abs(fd), abs(an), 1e-12)
    return out, loss
