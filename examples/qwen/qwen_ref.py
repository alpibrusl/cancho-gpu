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


def load_params(directory, c):
    """The parameters (binary64) from the files convert.py wrote."""
    P = {}
    for name, sh in shapes(c).items():
        kind = name.split(".")[-1]
        dt = "<f4" if kind in ("ln1", "ln2", "norm", "bqkv") else "<f2"
        ext = "f32" if dt == "<f4" else "f16"
        a = np.fromfile(os.path.join(directory, f"{name}.{ext}"), dtype=dt)
        if a.size != int(np.prod(sh)):
            raise SystemExit(f"{name}: {a.size} words, the model has {int(np.prod(sh))}")
        P[name] = a.astype(np.float64).reshape(sh)
    return P


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
