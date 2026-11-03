#!/usr/bin/env python3
"""The torch ResNet of resnet_ref.py (torchvision's resnet18 structure written out, so no torchvision and no download is needed): the module, a loader for resnet_ref's parameters, and
the loss and gradients of one batch in training mode (BatchNorm over the batch), for float64 (the specification's check) or float32 (what torch would run)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import resnet_ref as R


class Block(nn.Module):
    def __init__(self, ci, co, stride):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(co)
        self.c2 = nn.Conv2d(co, co, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(co)
        self.down = nn.Sequential(nn.Conv2d(ci, co, 1, stride, bias=False), nn.BatchNorm2d(co)) if stride != 1 or ci != co else None

    def forward(self, x):
        y = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        return F.relu(y + (x if self.down is None else self.down(x)))


class ResNet(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.stem = nn.Conv2d(c.cin, c.widths[0], 7, 2, 3, bias=False)
        self.bn = nn.BatchNorm2d(c.widths[0])
        blocks = []
        for s, (w, nb) in enumerate(zip(c.widths, c.blocks)):
            for b in range(nb):
                blocks.append(Block(c.widths[s - 1] if b == 0 and s > 0 else w, w, 2 if b == 0 and s > 0 else 1))
        self.blocks = nn.ModuleList(blocks)
        self.fc = nn.Linear(c.widths[-1], c.classes, bias=False)

    def forward(self, x):  # NCHW
        x = F.max_pool2d(F.relu(self.bn(self.stem(x))), 3, 2, 1)
        for b in self.blocks:
            x = b(x)
        return self.fc(x.mean((2, 3)))


def load(model, c, P):
    """resnet_ref's parameters (NHWC weights [co, kh, kw, ci]) into the module."""
    def conv(m, name):
        m.weight.data = torch.tensor(P["w_" + name]).permute(0, 3, 1, 2).contiguous().to(m.weight.dtype)

    def bn(m, name):
        m.weight.data, m.bias.data = torch.tensor(P["g_" + name]).to(m.weight.dtype), torch.tensor(P["b_" + name]).to(m.bias.dtype)
    conv(model.stem, "stem")
    bn(model.bn, "stem")
    i = 0
    for s, nb in enumerate(c.blocks):
        for b in range(nb):
            m = model.blocks[i]
            i += 1
            conv(m.c1, f"s{s}b{b}a")
            bn(m.b1, f"s{s}b{b}a")
            conv(m.c2, f"s{s}b{b}b")
            bn(m.b2, f"s{s}b{b}b")
            if m.down is not None:
                conv(m.down[0], f"s{s}b{b}d")
                bn(m.down[1], f"s{s}b{b}d")
    model.fc.weight.data = torch.tensor(P["wfc"]).to(model.fc.weight.dtype)


def grads(model, c):
    """The module's gradients under resnet_ref's names and layouts."""
    g = {}

    def conv(m, name):
        g["w_" + name] = m.weight.grad.permute(0, 2, 3, 1).double().numpy()

    def bn(m, name):
        g["g_" + name], g["b_" + name] = m.weight.grad.double().numpy(), m.bias.grad.double().numpy()
    conv(model.stem, "stem")
    bn(model.bn, "stem")
    i = 0
    for s, nb in enumerate(c.blocks):
        for b in range(nb):
            m = model.blocks[i]
            i += 1
            conv(m.c1, f"s{s}b{b}a")
            bn(m.b1, f"s{s}b{b}a")
            conv(m.c2, f"s{s}b{b}b")
            bn(m.b2, f"s{s}b{b}b")
            if m.down is not None:
                conv(m.down[0], f"s{s}b{b}d")
                bn(m.down[1], f"s{s}b{b}d")
    g["wfc"] = model.fc.weight.grad.double().numpy()
    return g


def step(c, P, x, y, dtype=torch.float64):
    """Loss and gradients of one training-mode batch (x: [N, H, W, cin] NHWC)."""
    model = ResNet(c).to(dtype)
    load(model, c, P)
    model.train()
    out = model(torch.tensor(x).permute(0, 3, 1, 2).to(dtype).contiguous())
    loss = F.cross_entropy(out, torch.tensor(y))
    loss.backward()
    return float(loss), grads(model, c), out.detach().double().numpy()


if __name__ == "__main__":
    import time
    for c in (R.Config(h=32, w=64, widths=(4, 4, 6, 8), blocks=(1, 1, 1, 1), classes=5, n=4), R.Config(h=96, w=128, n=2)):
        P = R.init_params(c, 3)
        rng = np.random.RandomState(4)
        for k in P:
            P[k] = P[k] + (0.3 * rng.normal(size=P[k].shape) if k.startswith(("g_", "b_")) else 0.05 * rng.normal(size=P[k].shape))
        x = rng.normal(size=(c.n, c.h, c.w, c.cin))
        y = rng.randint(0, c.classes, size=c.n)
        t = time.time()
        loss, g = R.forward(c, P, x, y, grads=True)
        lt, gt, _ = step(c, P, x, y)
        worst = max(float(np.abs(g[k] - gt[k]).max() / (np.abs(gt[k]).max() + 1e-30)) for k in g)
        print(f"{c.widths} {c.h}x{c.w}: loss spec {loss:.10f} torch {lt:.10f}; worst gradient difference relative to the tensor's peak {worst:.2e}; {sum(v.size for k, v in P.items())} parameters ({time.time() - t:.1f} s)")
