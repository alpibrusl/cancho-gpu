#!/usr/bin/env python3
"""V2 of docs/diffusion.md section 6: the VAE decoder of Wan2.2-TI2V-5B (AutoencoderKLWan, z_dim 48, 4 x 16 x 16) in float64, from the checkpoint's weights, held to torch float32.

    python tools/wan_vae_ref.py [--frames F] [--size HxW] [--seed S] [--torch32]

Written from the mathematics of diffusers' autoencoder_kl_wan.py, `_decode`'s frame-by-frame loop with its feature cache, but over the whole clip at once (the cache only makes the causal convolutions
streaming; this file checks that this reading is the same function):
  z [48, F, h, w] -> post_quant_conv (1x1x1) -> conv_in (3x3x3 causal) 48 -> 1024 -> mid block (residual, attention per frame over h w positions, residual) -> four up blocks of
  (1024 -> 1024), (1024 -> 1024), (1024 -> 512), (512 -> 256), each three residual blocks, then for the first three a nearest 2x spatial upsample and a 3x3 convolution, and for the first two **also**
  a temporal doubling, plus an average-free *shortcut* (DupUp3D: the block's input, channels repeated, rearranged into (time x 2, 2 x 2) positions) added to the output;
  RMS norm, SiLU, conv_out (3x3x3 causal) 256 -> 12, unpatchify 2 (12 = 3 x 2 x 2) to 3 channels, clamp to [-1, 1]. T = 1 + 4 (F - 1) frames, 16 h x 16 w pixels.
A causal 3x3x3 convolution pads two zero frames before the clip (none after). The temporal doubling (`upsample3d`): the first latent frame passes through as one frame; the others go through
`time_conv` (3 x 1 x 1, causal, 1024 -> 2048) **with the first frame not seen by it** (the cache holds "Rep", not that frame's features: its place in the causal window is zeros), the 2048 channels are
(2, 1024) and the two halves interleave in time (frame 2i is the first half, 2i + 1 the second); the DupUp shortcut's `first_chunk` drops the first of the two frames it makes for the first latent frame.
"""
import glob
import json
import math
import os
import sys

import numpy as np
from safetensors import safe_open

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/*/vae"))[0]
_f = safe_open(os.path.join(snap, "diffusion_pytorch_model.safetensors"), framework="pt")
CFG = json.load(open(os.path.join(snap, "config.json")))
W = lambda name: _f.get_tensor(name).double().numpy()
HAS = lambda name: name in _f.keys()


def conv3(x, name, k=3):
    """Causal 3D convolution. x [C, T, H, W]; weight [O, C, kt, k, k] (or [O, C, 1, 1, 1]); zero padding of kt - 1 frames in front, (k - 1) / 2 around."""
    w, b = W(name + ".weight"), W(name + ".bias")
    o, c, kt = w.shape[:3]
    t, h, wd = x.shape[1:]
    p = (w.shape[3] - 1) // 2
    xp = np.pad(x, ((0, 0), (kt - 1, 0), (p, p), (p, p)))
    y = np.zeros((o, t * h * wd))
    for dt in range(kt):
        for dy in range(w.shape[3]):
            for dx in range(w.shape[4]):
                y += w[:, :, dt, dy, dx] @ xp[:, dt:dt + t, dy:dy + h, dx:dx + wd].reshape(c, -1)
    return (y + b[:, None]).reshape(o, t, h, wd)


def conv2(x, name):
    """The 2D 3x3 convolution of the upsampler, per frame. x [C, T, H, W]."""
    w, b = W(name + ".weight"), W(name + ".bias")
    o, c = w.shape[:2]
    t, h, wd = x.shape[1:]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    y = np.zeros((o, t * h * wd))
    for dy in range(3):
        for dx in range(3):
            y += w[:, :, dy, dx] @ xp[:, :, dy:dy + h, dx:dx + wd].reshape(c, -1)
    return (y + b[:, None]).reshape(o, t, h, wd)


def rmsnorm(x, gamma):
    c = x.shape[0]
    n = np.sqrt((x ** 2).sum(0, keepdims=True))
    return x / np.maximum(n, 1e-12) * math.sqrt(c) * gamma.reshape(c, *([1] * (x.ndim - 1)))


def silu(x):
    return x / (1 + np.exp(-x))


def resblock(x, p):
    h = conv3(x, p + ".conv_shortcut") if HAS(p + ".conv_shortcut.weight") else x
    y = conv3(silu(rmsnorm(x, W(p + ".norm1.gamma"))), p + ".conv1")
    y = conv3(silu(rmsnorm(y, W(p + ".norm2.gamma"))), p + ".conv2")
    return y + h


def attention(x, p):
    """One head of C channels over the h w positions of each frame (no attention across frames)."""
    c, t, h, w = x.shape
    out = np.empty_like(x)
    wq, bq = W(p + ".to_qkv.weight")[:, :, 0, 0], W(p + ".to_qkv.bias")
    wp, bp = W(p + ".proj.weight")[:, :, 0, 0], W(p + ".proj.bias")
    for i in range(t):
        n = rmsnorm(x[:, i], W(p + ".norm.gamma"))
        qkv = wq @ n.reshape(c, h * w) + bq[:, None]
        q, k, v = (qkv[j * c:(j + 1) * c].T for j in range(3))
        s = q @ k.T / math.sqrt(c)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        out[:, i] = (wp @ (s @ v).T + bp[:, None]).reshape(c, h, w) + x[:, i]
    return out


def dupup(x, out, ft, first_chunk, fs=2):
    """DupUp3D: x [C, T, H, W] -> [out, T ft, H fs, W fs] (channels repeated, depth to space), the first ft - 1 frames dropped when first_chunk."""
    c, t, h, w = x.shape
    reps = out * ft * fs * fs // c
    x = np.repeat(x, reps, axis=0).reshape(out, ft, fs, fs, t, h, w)  # torch: view (out, ft, fs, fs, T, h, w)
    x = x.transpose(0, 4, 1, 5, 2, 6, 3).reshape(out, t * ft, h * fs, w * fs)  # (out, T, ft, h, fs, w, fs) -> torch permute (0, 1, 5, 2, 6, 3, 7, 4) of the 8-d view
    return x[:, ft - 1:] if first_chunk else x


def upsample3d(x, p):
    """nearest 2x spatial + 3x3 conv per frame; for the temporal one first the doubling by `time_conv` (the first frame passes alone, not seen by the causal window of the others)."""
    c, t, h, w = x.shape
    if HAS(p + ".time_conv.weight"):
        if t > 1:
            y = conv3(x[:, 1:], p + ".time_conv")  # [2c, t - 1, h, w]: frame 0 is not in the window (zeros)
            y = y.reshape(2, c, t - 1, h, w)
            d = np.stack([y[0], y[1]], 2).reshape(c, 2 * (t - 1), h, w)  # interleave in time
            x = np.concatenate([x[:, :1], d], 1)
    x = x.repeat(2, axis=2).repeat(2, axis=3)
    return conv2(x, p + ".resample.1")


def decode(z):
    """z [48, F, h, w] (as the pipeline hands it: already z * std + mean) -> video [3, 1 + 4 (F - 1), 16 h, 16 w] in [-1, 1]."""
    x = conv3(z, "post_quant_conv", 1)
    x = conv3(x, "decoder.conv_in")
    p = "decoder.mid_block."
    x = resblock(x, p + "resnets.0")
    x = attention(x, p + "attentions.0")
    x = resblock(x, p + "resnets.1")
    outs = [1024, 1024, 512, 256]
    temporal = [True, True, False, False]
    for i in range(4):
        p = "decoder.up_blocks.%d." % i
        x0 = x
        for r in range(3):
            x = resblock(x, p + "resnets.%d" % r)
        if i < 3:
            x = upsample3d(x, p + "upsampler")
            x = x + dupup(x0, outs[i], 2 if temporal[i] else 1, True)
    x = silu(rmsnorm(x, W("decoder.norm_out.gamma")))
    x = conv3(x, "decoder.conv_out")  # [12, T, 8h... ] patches of 2
    c, t, h, w = x.shape
    x = x.reshape(3, 2, 2, t, h, w).transpose(0, 3, 4, 2, 5, 1).reshape(3, t, h * 2, w * 2)  # torch unpatchify: view (c, ps, ps, f, h, w), permute (0, 1, 4, 5, 3, 6, 2)... see the check
    return np.clip(x, -1.0, 1.0)


def latent(f, h, w, seed):
    """A latent as the denoiser would hand it: z * std + mean for z of unit variance."""
    rng = np.random.default_rng(seed)
    mu, sd = np.array(CFG["latents_mean"])[:, None, None, None], np.array(CFG["latents_std"])[:, None, None, None]
    return rng.standard_normal((48, f, h, w)) * sd * 0.8 + mu


if __name__ == "__main__":
    arg = lambda name, default=None: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
    F = int(arg("--frames", 3))
    h, w = (int(v) for v in arg("--size", "4x4").split("x"))
    z = latent(F, h, w, int(arg("--seed", 0)))
    got = decode(z)
    print("latent [48, %d, %d, %d] -> video %s, rms %.4g" % (F, h, w, got.shape, np.sqrt((got ** 2).mean())))
    if "--torch32" in sys.argv:
        import torch
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32)
        with torch.no_grad():
            t32 = vae.decode(torch.tensor(z, dtype=torch.float32)[None], return_dict=False)[0][0].double().numpy()
        d = np.abs(got - t32)
        print("float64 vs torch float32: shape %s vs %s, mean abs error %.3g (%.3g of the rms), max %.3g" % (got.shape, t32.shape, d.mean(), d.mean() / np.sqrt((t32 ** 2).mean()), d.max()))
