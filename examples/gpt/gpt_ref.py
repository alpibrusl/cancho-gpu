#!/usr/bin/env python3
"""The specification of the GPT of docs/roadmap-models.md M1: a pre-LayerNorm transformer decoder (GELU tanh form, causal multi-head attention, untied output layer) written
in binary64 numpy with its backward pass by hand, and checked against finite differences. The kernels of gpt_session.py are held to it.

Tokens enter as one-hot rows of a [V + T] embedding matrix (the characters' V ids, then the T positions), so the embedding needs no gather and its gradient no scatter.
The vocabulary is padded to VP (a multiple of 64) with logits masked to -1e9."""
import math

import numpy as np

EPS = 1e-5
DEBUG = {}  # the last forward's logits, their gradient and the final-LayerNorm output, for the tests


class Config:
    def __init__(self, v=65, vp=128, t=64, d=128, h=2, layers=2, b=2):
        self.v, self.vp, self.t, self.d, self.h, self.layers, self.b = v, vp, t, d, h, layers, b
        self.dh = d // h
        self.n = b * t


def init_params(c, seed=1):
    rng = np.random.RandomState(seed)
    std = 0.02
    p = {"emb": rng.normal(0, std, size=(c.vp + c.t, c.d))}
    p["emb"][c.v:c.vp] = 0.0  # padding ids are never used
    for l in range(c.layers):
        p[f"ln1g{l}"], p[f"ln1b{l}"] = np.ones(c.d), np.zeros(c.d)
        p[f"wqkv{l}"], p[f"bqkv{l}"] = rng.normal(0, std, size=(3 * c.d, c.d)), np.zeros(3 * c.d)
        p[f"wo{l}"], p[f"bo{l}"] = rng.normal(0, std / math.sqrt(2 * c.layers), size=(c.d, c.d)), np.zeros(c.d)
        p[f"ln2g{l}"], p[f"ln2b{l}"] = np.ones(c.d), np.zeros(c.d)
        p[f"w1{l}"], p[f"b1{l}"] = rng.normal(0, std, size=(4 * c.d, c.d)), np.zeros(4 * c.d)
        p[f"w2{l}"], p[f"b2{l}"] = rng.normal(0, std / math.sqrt(2 * c.layers), size=(c.d, 4 * c.d)), np.zeros(c.d)
    p["lnfg"], p["lnfb"] = np.ones(c.d), np.zeros(c.d)
    p["wlm"] = rng.normal(0, std, size=(c.vp, c.d))
    return p


def ln_fwd(x, g, b):
    mu = x.mean(1, keepdims=True)
    xc = x - mu
    rstd = 1.0 / np.sqrt((xc * xc).mean(1, keepdims=True) + EPS)
    xh = xc * rstd
    return xh * g + b, (xh, rstd)


def ln_bwd(dy, g, cache):
    xh, rstd = cache
    dxh = dy * g
    dx = rstd * (dxh - dxh.mean(1, keepdims=True) - xh * (dxh * xh).mean(1, keepdims=True))
    return dx, (dy * xh).sum(0), dy.sum(0)


def gelu(u):
    k = math.sqrt(2.0 / math.pi)
    return 0.5 * u * (1.0 + np.tanh(k * (u + 0.044715 * u ** 3)))


def gelu_grad(u):
    k = math.sqrt(2.0 / math.pi)
    inner = k * (u + 0.044715 * u ** 3)
    th = np.tanh(inner)
    return 0.5 * (1.0 + th) + 0.5 * u * (1.0 - th * th) * k * (1.0 + 3 * 0.044715 * u * u)


def one_hot(c, tokens):
    """[N, VP + T] rows: the token's id and its position in the sequence."""
    n = len(tokens)
    oh = np.zeros((n, c.vp + c.t))
    oh[np.arange(n), tokens] = 1.0
    oh[np.arange(n), c.vp + np.arange(n) % c.t] = 1.0
    return oh


def forward(c, p, tokens, targets, grads=False):
    n, d, h, dh, t = c.n, c.d, c.h, c.dh, c.t
    oh = one_hot(c, tokens)
    x = oh @ p["emb"]
    cache = []
    mask = np.triu(np.full((t, t), -1e9), 1)
    for l in range(c.layers):
        a1, ln1c = ln_fwd(x, p[f"ln1g{l}"], p[f"ln1b{l}"])
        qkv = a1 @ p[f"wqkv{l}"].T + p[f"bqkv{l}"]
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
        o = np.zeros((n, d))
        probs = np.zeros((c.b, h, t, t))
        for b in range(c.b):
            for hh in range(h):
                sl = slice(b * t, (b + 1) * t)
                cs = slice(hh * dh, (hh + 1) * dh)
                s = q[sl, cs] @ k[sl, cs].T / math.sqrt(dh) + mask
                s = s - s.max(1, keepdims=True)
                e = np.exp(s)
                pr = e / e.sum(1, keepdims=True)
                probs[b, hh] = pr
                o[sl, cs] = pr @ v[sl, cs]
        x1 = x + o @ p[f"wo{l}"].T + p[f"bo{l}"]
        a2, ln2c = ln_fwd(x1, p[f"ln2g{l}"], p[f"ln2b{l}"])
        u = a2 @ p[f"w1{l}"].T + p[f"b1{l}"]
        hm = gelu(u)
        x2 = x1 + hm @ p[f"w2{l}"].T + p[f"b2{l}"]
        cache.append((x, a1, ln1c, qkv, probs, o, x1, a2, ln2c, u, hm))
        DEBUG[f"x{l}"], DEBUG[f"x1_{l}"], DEBUG[f"x2_{l}"], DEBUG[f"qkv{l}"], DEBUG[f"o{l}"], DEBUG[f"a1_{l}"], DEBUG[f"a2_{l}"] = x, x1, x2, qkv, o, a1, a2
        x = x2
    af, lnfc = ln_fwd(x, p["lnfg"], p["lnfb"])
    logits = af @ p["wlm"].T
    logits[:, c.v:] = -1e9
    mx = logits.max(1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(logits - mx).sum(1))
    loss = float(np.mean(lse - logits[np.arange(n), targets]))
    if not grads:
        return loss
    pr = np.exp(logits - lse[:, None])
    dl = pr.copy()
    dl[np.arange(n), targets] -= 1.0
    dl /= n
    DEBUG.update(dl=dl, af=af, logits=logits)
    g = {"wlm": dl.T @ af}
    daf = dl @ p["wlm"]
    dx, g["lnfg"], g["lnfb"] = ln_bwd(daf, p["lnfg"], lnfc)
    for l in reversed(range(c.layers)):
        xin, a1, ln1c, qkv, probs, o, x1, a2, ln2c, u, hm = cache[l]
        g[f"w2{l}"], g[f"b2{l}"] = dx.T @ hm, dx.sum(0)
        du = (dx @ p[f"w2{l}"]) * gelu_grad(u)
        g[f"w1{l}"], g[f"b1{l}"] = du.T @ a2, du.sum(0)
        da2 = du @ p[f"w1{l}"]
        dln2, g[f"ln2g{l}"], g[f"ln2b{l}"] = ln_bwd(da2, p[f"ln2g{l}"], ln2c)
        dx1 = dx + dln2
        g[f"wo{l}"], g[f"bo{l}"] = dx1.T @ o, dx1.sum(0)
        do = dx1 @ p[f"wo{l}"]
        dqkv = np.zeros((n, 3 * d))
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
        for b in range(c.b):
            for hh in range(h):
                sl = slice(b * t, (b + 1) * t)
                cs = slice(hh * dh, (hh + 1) * dh)
                pr_ = probs[b, hh]
                dpr = do[sl, cs] @ v[sl, cs].T
                ds = pr_ * (dpr - (dpr * pr_).sum(1, keepdims=True)) / math.sqrt(dh)
                dqkv[sl, 2 * d + hh * dh:2 * d + (hh + 1) * dh] = pr_.T @ do[sl, cs]
                dqkv[sl, cs] = ds @ k[sl, cs]
                dqkv[sl, d + hh * dh:d + (hh + 1) * dh] = ds.T @ q[sl, cs]
        g[f"wqkv{l}"], g[f"bqkv{l}"] = dqkv.T @ a1, dqkv.sum(0)
        da1 = dqkv @ p[f"wqkv{l}"]
        dln1, g[f"ln1g{l}"], g[f"ln1b{l}"] = ln_bwd(da1, p[f"ln1g{l}"], ln1c)
        dx = dx1 + dln1
    g["emb"] = oh.T @ dx
    return loss, g


if __name__ == "__main__":
    c = Config(v=65, vp=128, t=16, d=32, h=2, layers=2, b=2)
    p = init_params(c, 3)
    rng = np.random.RandomState(4)
    for key in p:  # larger random weights so every path matters
        p[key] = p[key] + 0.2 * rng.normal(size=p[key].shape) if p[key].ndim == 2 else p[key] + 0.1 * rng.normal(size=p[key].shape)
    p["emb"][c.v:c.vp] = 0.0
    tokens = rng.randint(0, c.v, size=c.n)
    targets = rng.randint(0, c.v, size=c.n)
    loss, g = forward(c, p, tokens, targets, grads=True)
    worst = 0.0
    for key in p:
        flat = p[key].reshape(-1)
        for idx in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + 1e-6
            lp = forward(c, p, tokens, targets)
            flat[idx] = old - 1e-6
            lm = forward(c, p, tokens, targets)
            flat[idx] = old
            fd = (lp - lm) / 2e-6
            an = g[key].reshape(-1)[idx]
            worst = max(worst, abs(fd - an) / max(1e-7, abs(fd), abs(an)) if max(abs(fd), abs(an)) > 1e-7 else 0.0)
    print(f"loss {loss:.5f}; finite differences against the hand-written backward over {len(p)} tensors: worst relative difference {worst:.2e}")
