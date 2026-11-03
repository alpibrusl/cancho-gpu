#!/usr/bin/env python3
"""The specification of the CIFAR-style convolutional network of docs/roadmap-models.md M4: a ResNet-style network written in binary64 numpy with its backward pass by hand,
checked against finite differences (run this file). The kernels of cnn_session.py are held to it.

NHWC, 3x3 convolutions with a zero border and no bias (BatchNorm follows each), BatchNorm over the batch and the pixels (biased variance, eps 1e-5), ReLU, 2x2 average pooling,
a global average, and a linear classifier with no bias. Stage 0 is a stem convolution and `blocks[0]` residual blocks; each later stage is a 2x2 average pool, a convolution to the stage's
width, and its residual blocks. A residual block is relu(bn(conv(relu(bn(conv(a))))) + a). With widths (32, 64, 128) and blocks (1, 2, 2) it has 0.87 M parameters."""
import math

import numpy as np

EPS = 1e-5
QUANT = False  # round to binary16 where the device does (activations, weights, the gradients of pre-activations): the reference of the mixed-precision arithmetic
LOSS_SCALE = 16.0


def q(x):
    return x.astype(np.float16).astype(np.float64) if QUANT else x


def qg(x):
    """A gradient the device keeps in f16, scaled by the loss scale."""
    return q(x * LOSS_SCALE) / LOSS_SCALE if QUANT else x


class Config:
    def __init__(self, h=32, cin=3, widths=(32, 64, 128), blocks=(1, 2, 2), classes=10, n=128):
        self.h, self.cin, self.widths, self.blocks, self.classes, self.n = h, cin, tuple(widths), tuple(blocks), classes, n
        assert h % (2 ** (len(widths) - 1)) == 0

    def layers(self):
        """[(name, cin, cout)] in forward order."""
        out = [("stem", self.cin, self.widths[0])]
        for s, (w, nb) in enumerate(zip(self.widths, self.blocks)):
            if s > 0:
                out.append((f"up{s}", self.widths[s - 1], w))
            for b in range(nb):
                out += [(f"s{s}b{b}a", w, w), (f"s{s}b{b}b", w, w)]
        return out

    def size_at(self, s):
        return self.h // (2 ** s)


def init_params(c, seed=1):
    rng = np.random.RandomState(seed)
    p = {}
    for name, ci, co in c.layers():
        p["w_" + name] = rng.normal(0, math.sqrt(2.0 / (9 * ci)), size=(co, 3, 3, ci))
        p["g_" + name] = np.ones(co)
        p["b_" + name] = np.zeros(co)
    p["wfc"] = rng.normal(0, 1.0 / math.sqrt(c.widths[-1]), size=(c.classes, c.widths[-1]))
    return p


def param_names(c):
    """(matrices, vectors): the tensors that get weight decay, and the BatchNorm scales and shifts that do not."""
    mats = ["w_" + n for n, _, _ in c.layers()] + ["wfc"]
    vecs = [k + n for n, _, _ in c.layers() for k in ("g_", "b_")]
    return mats, vecs


def conv_f(x, w):
    n, h, wd, ci = x.shape
    xp = np.zeros((n, h + 2, wd + 2, ci))
    xp[:, 1:-1, 1:-1] = x
    y = np.zeros((n, h, wd, w.shape[0]))
    for dy in range(3):
        for dx in range(3):
            y += np.einsum("nhwc,oc->nhwo", xp[:, dy:dy + h, dx:dx + wd], w[:, dy, dx])
    return y


def conv_b(dy, x, w):
    n, h, wd, ci = x.shape
    xp = np.zeros((n, h + 2, wd + 2, ci))
    xp[:, 1:-1, 1:-1] = x
    dxp = np.zeros_like(xp)
    dw = np.zeros_like(w)
    for a in range(3):
        for b in range(3):
            dw[:, a, b] = np.einsum("nhwo,nhwc->oc", dy, xp[:, a:a + h, b:b + wd])
            dxp[:, a:a + h, b:b + wd] += np.einsum("nhwo,oc->nhwc", dy, w[:, a, b])
    return dw, dxp[:, 1:-1, 1:-1]


def pool_f(x):
    n, h, w, c = x.shape
    return q(x.reshape(n, h // 2, 2, w // 2, 2, c).mean((2, 4)))


def pool_b(dy):
    return np.repeat(np.repeat(dy, 2, 1), 2, 2) * 0.25


def forward(c, p, x, labels, grads=False, running=None, stats=None):
    """x: [N, H, W, cin]; labels: [N] ints. With `running` ({name: (mean, var)}) the BatchNorm uses those instead of the batch's (evaluation); with `stats` (a dict) the batch's
    mean and unbiased variance of each layer are written there. Returns the loss (and, with grads, the gradients), and the logits are left in DEBUG["logits"]."""
    g = {}
    tape = []

    def layer(name, a, res=None):
        w = p["w_" + name]
        wq = q(w)
        z = conv_f(a, wq)
        if running is None:
            mu = z.mean((0, 1, 2))
            var = z.var((0, 1, 2))
            m = z.shape[0] * z.shape[1] * z.shape[2]
            if stats is not None:
                stats[name] = (mu, var * m / (m - 1))
        else:
            mu, var = running[name]
        istd = 1.0 / np.sqrt(var + EPS)
        xh = (z - mu) * istd
        pre = xh * p["g_" + name] + p["b_" + name] + (0 if res is None else res)
        out = q(np.maximum(pre, 0))
        DEBUG["z_" + name], DEBUG["pre_" + name], DEBUG["out_" + name] = z, pre, out

        def bwd(dout):
            dpre = dout * (pre > 0)
            g["g_" + name] = (dpre * xh).sum((0, 1, 2))
            g["b_" + name] = dpre.sum((0, 1, 2))
            m = z.shape[0] * z.shape[1] * z.shape[2]
            dxh = dpre * p["g_" + name]
            dz = istd * (dxh - dxh.mean((0, 1, 2)) - xh * (dxh * xh).mean((0, 1, 2)))
            dz = qg(dz)
            DEBUG["dz_" + name] = dz
            g["w_" + name], da = conv_b(dz, a, wq)
            DEBUG["da_" + name], DEBUG["dpre_" + name] = da, dpre
            return da, dpre
        return out, bwd

    a = x
    a, b = layer("stem", a)
    tape.append(("plain", b))
    for s, nb in enumerate(c.blocks):
        if s > 0:
            a = pool_f(a)
            tape.append(("pool", None))
            a, b = layer(f"up{s}", a)
            tape.append(("plain", b))
        for k in range(nb):
            a0 = a
            u, b1 = layer(f"s{s}b{k}a", a0)
            a, b2 = layer(f"s{s}b{k}b", u, res=a0)
            tape.append(("block", b1, b2))
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
    dl /= n
    dl = qg(dl)
    g["wfc"] = dl.T @ feat
    dfeat = dl @ q(p["wfc"])
    da = np.broadcast_to(dfeat[:, None, None, :] / (a.shape[1] * a.shape[2]), a.shape).copy()
    for op in reversed(tape):
        if op[0] == "plain":
            da, _ = op[1](da)
        elif op[0] == "pool":
            da = pool_b(da)
        else:
            da2, skip = op[2](da)
            da1, _ = op[1](da2)
            da = da1 + skip
    return loss, g


DEBUG = {}


def evaluate(c, p, running, images, labels, chunk=500):
    """Accuracy and loss of the network in evaluation mode (the running statistics) on [N, H, W, cin] images."""
    right, tot = 0, 0.0
    for i in range(0, len(images), chunk):
        x, y = images[i:i + chunk], labels[i:i + chunk]
        tot += forward(c, p, x, y, running=running) * len(y)
        right += int((DEBUG["logits"].argmax(1) == y).sum())
    return right / len(images), tot / len(images)


if __name__ == "__main__":
    c = Config(h=8, cin=3, widths=(8, 8, 16), blocks=(1, 1, 1), classes=10, n=6)
    p = init_params(c, 3)
    rng = np.random.RandomState(4)
    for k in p:  # larger random weights and non-trivial BatchNorm scales so every path matters
        p[k] = p[k] + (0.3 * rng.normal(size=p[k].shape) if k.startswith(("g_", "b_")) else 0.1 * rng.normal(size=p[k].shape))
    x = rng.normal(size=(c.n, c.h, c.h, c.cin))
    y = rng.randint(0, c.classes, size=c.n)
    loss, g = forward(c, p, x, y, grads=True)
    worst = 0.0
    for key in p:
        flat = p[key].reshape(-1)
        for idx in rng.choice(flat.size, size=min(8, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + 1e-6
            lp = forward(c, p, x, y)
            flat[idx] = old - 1e-6
            lm = forward(c, p, x, y)
            flat[idx] = old
            fd = (lp - lm) / 2e-6
            an = g[key].reshape(-1)[idx]
            if max(abs(fd), abs(an)) > 1e-7:
                worst = max(worst, abs(fd - an) / max(abs(fd), abs(an)))
    print(f"loss {loss:.5f}; finite differences against the hand-written backward over {len(p)} tensors: worst relative difference {worst:.2e}")
