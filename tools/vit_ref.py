#!/usr/bin/env python3
"""D9a of docs/diffusion.md: the Qwen3-VL vision tower of Qwen-Image-2.1-Turbo in float64 (numpy), written from the maths in transformers' modeling_qwen3_vl.py, and held to torch.

    vit_ref.py [--size N] [--torch32] [--save DIR]

The tower reads an image the pipeline has already resized (each side a multiple of 32). It returns what the text encoder takes from it: the merged embeddings [(H/32)(W/32), 4096] and the three deepstack
feature maps [3, (H/32)(W/32), 4096]. `--size N` (or `HxW`) runs on a synthetic N x N image (a smooth gradient plus fixed noise: the tower's arithmetic does not depend on what the picture is; the test
photograph is for the end-to-end run); `--torch32` also runs transformers' own vision model in float32 on the same pixels (its weights are the checkpoint's BF16, widened) and prints the difference;
`--save DIR` writes the patches, the float64 outputs and the tables for the kernels' tests (tests/vit_*.py).

Layout, from the source: pixels are cut into 16 x 16 patches in merge-block order (each 2 x 2 group of patches adjacent), the temporal axis is the image repeated twice (so the 3D convolution
of kernel 2 x 16 x 16 is one matrix product on the sum of its two time slices), a learned 48 x 48 table is resampled to the image's grid by bilinear interpolation with align_corners, rotary
positions are the patch's (row, column) with 18 frequencies each (theta 10000), rotating the 72 channels of a head as rotate_half over [h, w, h, w].
"""
import json
import glob
import os
import sys

import numpy as np

CKPT = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/*/"))[0]
CFG = json.load(open(CKPT + "text_encoder/config.json"))["vision_config"]
D, NH, HD, FF, DEPTH, P, M = CFG["hidden_size"], CFG["num_heads"], CFG["hidden_size"] // CFG["num_heads"], CFG["intermediate_size"], CFG["depth"], CFG["patch_size"], CFG["spatial_merge_size"]
DEEP = CFG["deepstack_visual_indexes"]
SIDE = int(CFG["num_position_embeddings"] ** 0.5)
THETA = 10000.0  # the rope_theta default of Qwen3VLVisionConfig (the checkpoint's config.json does not set one)
_f = None


def W(name):
    global _f
    if _f is None:
        from safetensors import safe_open
        _f = safe_open(CKPT + "text_encoder/model.safetensors", "pt")
    return _f.get_tensor("model.visual." + name).float().numpy().astype(np.float64)


def synthetic(n, m=None):
    m = m or n
    yy, xx = np.mgrid[0:n, 0:m] / np.array([n - 1, m - 1])[:, None, None]
    rng = np.random.default_rng(3)
    img = np.stack([xx, yy, 0.5 * (xx + yy)], -1) * 255.0 + rng.normal(0, 12, (n, m, 3))
    return np.clip(np.round(img), 0, 255)


def patches(img):
    """[H, W, 3] pixels in 0..255 to [(H/16)(W/16), 3 * 2 * 16 * 16]: normalised (mean 0.5, std 0.5), merge-block order, channels, time, rows, columns."""
    x = (img / 255.0 - 0.5) / 0.5
    h, w = x.shape[0] // P, x.shape[1] // P
    x = np.broadcast_to(x.transpose(2, 0, 1)[None], (2, 3, x.shape[0], x.shape[1]))  # [T=2, C, H, W]
    x = x.reshape(2, 3, h // M, M, P, w // M, M, P)  # t c  bh m1 p1  bw m2 p2
    x = x.transpose(2, 5, 3, 6, 1, 0, 4, 7)  # bh bw m1 m2 c t p1 p2
    return x.reshape(h * w, 3 * 2 * P * P), h, w


def pos_table(h, w):
    """[(h w), 1152]: the learned table resampled bilinearly (align_corners) to the grid, in merge-block order."""
    tab = W("pos_embed.weight").reshape(SIDE, SIDE, D)

    def axis(n):
        src = np.arange(n) * (SIDE - 1) / (n - 1) if n > 1 else np.zeros(n)
        lo = np.floor(src).astype(int)
        hi = np.minimum(lo + 1, SIDE - 1)
        return lo, hi, src - lo
    rl, rh, rf = axis(h)
    cl, ch, cf = axis(w)
    g = (tab[rl][:, cl] * ((1 - rf)[:, None, None] * (1 - cf)[None, :, None]) + tab[rl][:, ch] * ((1 - rf)[:, None, None] * cf[None, :, None])
         + tab[rh][:, cl] * (rf[:, None, None] * (1 - cf)[None, :, None]) + tab[rh][:, ch] * (rf[:, None, None] * cf[None, :, None]))  # [h, w, D]
    return block_order(g, h, w).reshape(h * w, D)


def block_order(a, h, w):
    """[h, w, ...] raster to merge-block order [(h/2) (w/2) 2 2 ...] flattened over the first four axes."""
    a = a.reshape(h // M, M, w // M, M, *a.shape[2:]).transpose(0, 2, 1, 3, *range(4, a.ndim + 2))
    return a.reshape(h * w, *a.shape[4:])


def rope_tables(h, w):
    """cos, sin [(h w), 72]: row and column positions in merge-block order, 18 frequencies each, laid out [h, w, h, w]."""
    inv = 1.0 / (THETA ** (np.arange(0, HD // 2, 2) / (HD // 2)))
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rr = block_order(rr[..., None], h, w)[:, 0]
    cc = block_order(cc[..., None], h, w)[:, 0]
    fh, fw = rr[:, None] * inv, cc[:, None] * inv
    hw = np.concatenate([fh, fw], -1)
    ang = np.concatenate([hw, hw], -1)
    return np.cos(ang), np.sin(ang)


def layer_norm(x, w, b, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(v + eps) * w + b


def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_erf(x):
    from math import erf
    return 0.5 * x * (1 + np.vectorize(erf)(x / np.sqrt(2)))


def rotate_half(x):
    return np.concatenate([-x[..., HD // 2:], x[..., :HD // 2]], -1)


def merger(x, name, postshuffle):
    n = W(name + ".norm.weight"), W(name + ".norm.bias")
    y = x.reshape(-1, D * M * M)
    y = layer_norm(y, *n) if postshuffle else layer_norm(x, *n).reshape(-1, D * M * M)
    y = gelu_erf(y @ W(name + ".linear_fc1.weight").T + W(name + ".linear_fc1.bias"))
    return y @ W(name + ".linear_fc2.weight").T + W(name + ".linear_fc2.bias")


def tower(img, upto=DEPTH):
    pat, h, w = patches(img)
    wk = W("patch_embed.proj.weight").reshape(D, -1)
    x = pat @ wk.T + W("patch_embed.proj.bias") + pos_table(h, w)
    cos, sin = rope_tables(h, w)
    deep = []
    for i in range(upto):
        b = f"blocks.{i}."
        y = layer_norm(x, W(b + "norm1.weight"), W(b + "norm1.bias"))
        qkv = (y @ W(b + "attn.qkv.weight").T + W(b + "attn.qkv.bias")).reshape(-1, 3, NH, HD)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        o = np.empty_like(q)
        for hd in range(NH):
            s = q[:, hd] @ k[:, hd].T / np.sqrt(HD)
            s -= s.max(-1, keepdims=True)
            e = np.exp(s)
            o[:, hd] = (e / e.sum(-1, keepdims=True)) @ v[:, hd]
        x = x + o.reshape(-1, D) @ W(b + "attn.proj.weight").T + W(b + "attn.proj.bias")
        y = layer_norm(x, W(b + "norm2.weight"), W(b + "norm2.bias"))
        x = x + gelu_tanh(y @ W(b + "mlp.linear_fc1.weight").T + W(b + "mlp.linear_fc1.bias")) @ W(b + "mlp.linear_fc2.weight").T + W(b + "mlp.linear_fc2.bias")
        if i in DEEP:
            deep.append(merger(x, f"deepstack_merger_list.{DEEP.index(i)}", True))
    return (merger(x, "merger", False) if upto == DEPTH else None), np.stack(deep) if deep else None, pat, (h, w)


def torch32(pat, h, w):
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    cfg = Qwen3VLVisionConfig(**CFG)
    cfg._attn_implementation = "eager"
    m = Qwen3VLVisionModel(cfg).float().eval()
    f = safe_open(CKPT + "text_encoder/model.safetensors", "pt")
    sd = {k[len("model.visual."):]: f.get_tensor(k).float() for k in f.keys() if k.startswith("model.visual.")}
    m.load_state_dict(sd)
    with torch.no_grad():
        o = m(torch.tensor(pat, dtype=torch.float32), torch.tensor([[1, h, w]]))
    return o.pooler_output.numpy().astype(np.float64), np.stack([d.numpy().astype(np.float64) for d in o.deepstack_features]), float(m.rotary_pos_emb.inv_freq[1] / m.rotary_pos_emb.inv_freq[0])


if __name__ == "__main__":
    sz = sys.argv[sys.argv.index("--size") + 1] if "--size" in sys.argv else "64"
    n, m = (int(a) for a in (sz.split("x") if "x" in sz else (sz, sz)))
    out, deep, pat, (h, w) = tower(synthetic(n, m))
    print("image %d x %d: %d patches, %d merged tokens; merged rms %.4g, deepstack rms %s" % (n, m, h * w, h * w // 4, np.sqrt((out ** 2).mean()), np.sqrt((deep ** 2).mean((1, 2))).round(4)))
    if "--torch32" in sys.argv:
        to, td, ratio = torch32(pat, h, w)
        rms = np.sqrt((out ** 2).mean())
        e = np.abs(to - out)
        print("float64 vs torch float32: merged mean abs %.3g (%.3g of rms), max %.3g; deepstack mean abs %s, max %.3g  (rope inv_freq ratio %.6g, expect %.6g)" % (e.mean(), e.mean() / rms, e.max(), np.abs(td - deep).mean((1, 2)), np.abs(td - deep).max(), ratio, THETA ** (-2 / (HD // 2))))
    if "--save" in sys.argv:
        d = sys.argv[sys.argv.index("--save") + 1]
        os.makedirs(d, exist_ok=True)
        np.save(d + "/patches.npy", pat)
        np.save(d + "/merged_f64.npy", out)
        np.save(d + "/deep_f64.npy", deep)
        print("saved to", d)
