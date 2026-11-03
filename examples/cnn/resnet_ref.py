#!/usr/bin/env python3
"""The specification of the ResNet-18-class backbone of docs/roadmap-models.md M9 (V1): binary64 numpy with its backward pass by hand, checked against finite differences (run this file),
and against torch (`tests/resnet.py`, float64 autograd on the same weights).

NHWC. The network is the ResNet-18 of He et al. as torchvision builds it, with two differences that are on purpose: a classifier without bias (the session's head is `wfc`, as the M4
CNN's) and the widths and blocks are a configuration. A 7x7 stride-2 convolution (padding 3), BatchNorm, ReLU, 3x3 stride-2 max-pooling (padding 1); then four stages of residual blocks
(BasicBlock: relu(bn(conv3(relu(bn(conv3(a))))) + shortcut)); the first block of stages 1 to 3 has a stride-2 3x3 convolution first and a 1x1 stride-2 convolution with its own BatchNorm on the
shortcut; a global average; a linear classifier. BatchNorm over the batch and the pixels (biased variance in the forward, eps 1e-5).
The max-pool's tie rule is torch's: the gradient goes to the first maximum of the window in row-major order, padding counting as -inf.
With `QUANT` the arithmetic rounds to binary16 where the device does (the weights, every activation, the gradients of the pre-activations): the reference the kernels are held to."""
import math

import numpy as np

EPS = 1e-5
QUANT = False
LOSS_SCALE = 16.0


def q(x):
    return x.astype(np.float16).astype(np.float64) if QUANT else x


def qg(x):
    return q(x * LOSS_SCALE) / LOSS_SCALE if QUANT else x


class Config:
    def __init__(self, h=96, w=128, cin=3, widths=(64, 128, 256, 512), blocks=(2, 2, 2, 2), classes=10, n=8):
        self.h, self.w, self.cin, self.widths, self.blocks, self.classes, self.n = h, w, cin, tuple(widths), tuple(blocks), classes, n
        assert h % 32 == 0 and w % 32 == 0 and len(widths) == 4

    def layers(self):
        """[(name, kind, cin, cout)] in forward order; kind: stem (7x7 s2), c3 (3x3 s1), c3s2 (3x3 s2), c1s2 (1x1 s2)."""
        out = [("stem", "stem", self.cin, self.widths[0])]
        for s, (wd, nb) in enumerate(zip(self.widths, self.blocks)):
            for b in range(nb):
                first = b == 0 and s > 0
                out.append((f"s{s}b{b}a", "c3s2" if first else "c3", self.widths[s - 1] if first else wd, wd))
                out.append((f"s{s}b{b}b", "c3", wd, wd))
                if first:
                    out.append((f"s{s}b{b}d", "c1s2", self.widths[s - 1], wd))
        return out

    def size_at(self, s):
        """The side of the feature map in stage s (stage 0 follows the max-pool)."""
        return self.h // 4 // (2 ** s), self.w // 4 // (2 ** s)


KIND = {"stem": (7, 2, 3), "c3": (3, 1, 1), "c3s2": (3, 2, 1), "c1s2": (1, 2, 0)}  # (kernel, stride, pad)


def init_params(c, seed=1):
    rng = np.random.RandomState(seed)
    p = {}
    for name, kind, ci, co in c.layers():
        k = KIND[kind][0]
        p["w_" + name] = rng.normal(0, math.sqrt(2.0 / (k * k * ci)), size=(co, k, k, ci))
        p["g_" + name] = np.ones(co)
        p["b_" + name] = np.zeros(co)
    p["wfc"] = rng.normal(0, 1.0 / math.sqrt(c.widths[-1]), size=(c.classes, c.widths[-1]))
    return p


def param_names(c):
    mats = ["w_" + n for n, _, _, _ in c.layers()] + ["wfc"]
    vecs = [k + n for n, _, _, _ in c.layers() for k in ("g_", "b_")]
    return mats, vecs


def conv_f(x, w, stride, pad):
    n, h, wd, ci = x.shape
    co, kh, kw, _ = w.shape
    ho, wo = (h + 2 * pad - kh) // stride + 1, (wd + 2 * pad - kw) // stride + 1
    xp = np.zeros((n, h + 2 * pad, wd + 2 * pad, ci))
    xp[:, pad:pad + h, pad:pad + wd] = x
    y = np.zeros((n, ho, wo, co))
    for dy in range(kh):
        for dx in range(kw):
            y += np.einsum("nhwc,oc->nhwo", xp[:, dy:dy + stride * (ho - 1) + 1:stride, dx:dx + stride * (wo - 1) + 1:stride], w[:, dy, dx])
    return y


def conv_b(dy, x, w, stride, pad):
    n, h, wd, ci = x.shape
    co, kh, kw, _ = w.shape
    ho, wo = dy.shape[1], dy.shape[2]
    xp = np.zeros((n, h + 2 * pad, wd + 2 * pad, ci))
    xp[:, pad:pad + h, pad:pad + wd] = x
    dxp = np.zeros_like(xp)
    dw = np.zeros_like(w)
    for a in range(kh):
        for b in range(kw):
            sl = (slice(None), slice(a, a + stride * (ho - 1) + 1, stride), slice(b, b + stride * (wo - 1) + 1, stride))
            dw[:, a, b] = np.einsum("nhwo,nhwc->oc", dy, xp[sl])
            dxp[sl] += np.einsum("nhwo,oc->nhwc", dy, w[:, a, b])
    return dw, dxp[:, pad:pad + h, pad:pad + wd]


def maxpool_f(x):
    """3x3, stride 2, padding 1 (-inf): the pooled map and the index (0..8, row-major) of the first maximum of each window."""
    n, h, w, c = x.shape
    ho, wo = (h + 2 - 3) // 2 + 1, (w + 2 - 3) // 2 + 1
    xp = np.full((n, h + 2, w + 2, c), -np.inf)
    xp[:, 1:-1, 1:-1] = x
    cand = np.stack([xp[:, dy:dy + 2 * (ho - 1) + 1:2, dx:dx + 2 * (wo - 1) + 1:2] for dy in range(3) for dx in range(3)])
    idx = cand.argmax(0)  # the first maximum
    return cand.max(0), idx


def maxpool_b(dy, idx, shape):
    n, h, w, c = shape
    ho, wo = dy.shape[1], dy.shape[2]
    dxp = np.zeros((n, h + 2, w + 2, c))
    for k in range(9):
        dyk, dxk = divmod(k, 3)
        dxp[:, dyk:dyk + 2 * (ho - 1) + 1:2, dxk:dxk + 2 * (wo - 1) + 1:2] += dy * (idx == k)
    return dxp[:, 1:-1, 1:-1]


DEBUG = {}


def forward(c, p, x, labels, grads=False, running=None, stats=None):
    """x: [N, H, W, cin]; labels: [N]. With `running` ({layer: (mean, var)}) BatchNorm uses those (evaluation); with `stats` the batch's mean and unbiased variance of each layer are written there.
    Returns the loss, and with `grads` the gradients too; the logits are left in DEBUG["logits"]."""
    g = {}
    kinds = {n: k for n, k, _, _ in c.layers()}

    def layer(name, a, res=None, relu=True):
        _, stride, pad = KIND[kinds[name]]
        wq = q(p["w_" + name])
        z = conv_f(a, wq, stride, pad)
        if running is None:
            mu, var = z.mean((0, 1, 2)), z.var((0, 1, 2))
            m = z.shape[0] * z.shape[1] * z.shape[2]
            if stats is not None:
                stats[name] = (mu, var * m / (m - 1))
        else:
            mu, var = running[name]
        istd = 1.0 / np.sqrt(var + EPS)
        xh = (z - mu) * istd
        pre = xh * p["g_" + name] + p["b_" + name] + (0 if res is None else res)
        out = q(np.maximum(pre, 0) if relu else pre)
        DEBUG["z_" + name], DEBUG["out_" + name] = z, out

        def bwd(dout):
            dpre = dout * (pre > 0) if relu else dout
            g["g_" + name] = (dpre * xh).sum((0, 1, 2))
            g["b_" + name] = dpre.sum((0, 1, 2))
            dxh = dpre * p["g_" + name]
            dz = qg(istd * (dxh - dxh.mean((0, 1, 2)) - xh * (dxh * xh).mean((0, 1, 2))))
            g["w_" + name], da = conv_b(dz, a, wq, stride, pad)
            return da, dpre
        return out, bwd

    x = q(x)
    a, b_stem = layer("stem", x)
    pooled, idx = maxpool_f(a)
    a_pre = a
    a = q(pooled)
    blocks = []
    for s, nb in enumerate(c.blocks):
        for k in range(nb):
            a0 = a
            first = k == 0 and s > 0
            u, b1 = layer(f"s{s}b{k}a", a0)
            if first:
                sc, bd = layer(f"s{s}b{k}d", a0, relu=False)
                a, b2 = layer(f"s{s}b{k}b", u, res=sc)
            else:
                sc, bd = a0, None
                a, b2 = layer(f"s{s}b{k}b", u, res=a0)
            blocks.append((b1, b2, bd))
    feat = q(a.mean((1, 2)))
    logits = feat @ q(p["wfc"]).T
    mx = logits.max(1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(logits - mx).sum(1))
    n = len(labels)
    loss = float(np.mean(lse - logits[np.arange(n), labels]))
    DEBUG["logits"], DEBUG["feat"] = logits, feat
    if not grads:
        return loss
    pr = np.exp(logits - lse[:, None])
    dl = pr.copy()
    dl[np.arange(n), labels] -= 1.0
    dl = qg(dl / n)
    g["wfc"] = dl.T @ feat
    da = np.broadcast_to((dl @ q(p["wfc"]))[:, None, None, :] / (a.shape[1] * a.shape[2]), a.shape).copy()
    for b1, b2, bd in reversed(blocks):
        da2, skip = b2(da)
        da1, _ = b1(da2)
        if bd is None:
            da = da1 + skip
        else:
            dad, _ = bd(skip)
            da = da1 + dad
    da = maxpool_b(da, idx, a_pre.shape)
    b_stem(da)
    return loss, g


def evaluate(c, p, running, images, labels, chunk=200):
    right, tot = 0, 0.0
    for i in range(0, len(images), chunk):
        x, y = images[i:i + chunk], labels[i:i + chunk]
        tot += forward(c, p, x, y, running=running) * len(y)
        right += int((DEBUG["logits"].argmax(1) == y).sum())
    return right / len(images), tot / len(images)


if __name__ == "__main__":
    c = Config(h=32, w=64, cin=3, widths=(4, 4, 6, 8), blocks=(1, 1, 1, 1), classes=5, n=4)
    p = init_params(c, 3)
    rng = np.random.RandomState(4)
    for k in p:
        p[k] = p[k] + (0.3 * rng.normal(size=p[k].shape) if k.startswith(("g_", "b_")) else 0.1 * rng.normal(size=p[k].shape))
    x = rng.normal(size=(c.n, c.h, c.w, c.cin))
    y = rng.randint(0, c.classes, size=c.n)
    loss, g = forward(c, p, x, y, grads=True)
    worst = 0.0
    for key in p:
        flat = p[key].reshape(-1)
        for i in rng.choice(flat.size, size=min(8, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + 1e-6
            lp = forward(c, p, x, y)
            flat[i] = old - 1e-6
            lm = forward(c, p, x, y)
            flat[i] = old
            fd = (lp - lm) / 2e-6
            an = g[key].reshape(-1)[i]
            if max(abs(fd), abs(an)) > 1e-7:
                worst = max(worst, abs(fd - an) / max(abs(fd), abs(an)))
    print(f"loss {loss:.5f}; finite differences against the hand-written backward over {len(p)} tensors: worst relative difference {worst:.2e}")
