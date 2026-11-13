#!/usr/bin/env python3
"""D9d of docs/diffusion.md: the VAE encoder of Qwen-Image-2.1-Turbo in float64, from the checkpoint's weights, written from the mathematics of autoencoder_kl_qwenimage21.py (`_encode` for one frame, then the mean
of the posterior, the `argmax` mode the pipeline uses), held to torch float32.

    python tools/vae_enc_ref.py [--size N | HxW] [--torch32] [--save DIR]

The encoder, for an image x [4, H, W] in [-1, 1] (RGB scaled from 0..255, and an alpha channel of 1.0: `VaeImageProcessor.preprocess` of an RGBA image gives a constant 1 there):
  conv3x3 (4 -> 96); five down blocks of (in -> out) = (96 -> 96), (96 -> 192), (192 -> 384), (384 -> 768), (768 -> 768): two residual blocks, then for the first four a downsample, plus an **average-pool shortcut**
  of the block's input added to the output; mid block (residual 768, attention of one head of 768 over all positions, residual 768); RMS norm, SiLU, conv3x3 (768 -> 128), the 1x1 `quant_conv`; the mean is the first 64
  channels; the pipeline then subtracts the latents' mean and divides by their std (config `latents_mean`, `latents_std`).
The downsample is a zero pad of one row at the bottom and one column on the right, then a 3x3 convolution of stride 2 (no padding). The shortcut is AvgDown3D: channels grouped into `in * factor / out` consecutive values
after a space-to-depth over (time, rows, columns) with the **first frame padded by a zero frame when the block is temporal** (factor_t 2, blocks 1 to 3: the encoder's `temperal_downsample` is [False, True, True, True]);
for one frame this makes output channel 2c a zero and channel 2c + 1 the 2 x 2 mean of input channel c. The blocks' `time_conv` is not applied to a first chunk.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vae_ref as v  # noqa: E402

W, conv, rmsnorm, silu, resblock, attention = v.W, v.conv, v.rmsnorm, v.silu, v.resblock, v.attention
CFG = json.load(open(os.path.join(v.snap, "vae/config.json")))


def conv_s2(x, name):
    """The downsampler: zero pad (right 1, bottom 1), 3 x 3, stride 2, no padding. x [C, H, W] -> [C, H/2, W/2]."""
    w, b = W(name + ".weight"), W(name + ".bias")
    o, c = w.shape[:2]
    h, wd = x.shape[1:]
    xp = np.pad(x, ((0, 0), (0, 1), (0, 1)))
    y = np.zeros((o, (h // 2) * (wd // 2)))
    for dy in range(3):
        for dx in range(3):
            y += w[:, :, dy, dx] @ xp[:, dy:dy + h:2, dx:dx + wd:2].reshape(c, -1)
    v.FLOPS[name] = 2.0 * o * c * 9 * (h // 2) * (wd // 2)
    return (y + b[:, None]).reshape(o, h // 2, wd // 2)


def avgdown(x, out, ft, fs):
    """QwenImage21AvgDown3D for one frame [C, H, W]: pad_t = ft - 1 zero frames at the front, space to depth over (ft, fs, fs), mean over groups of in * ft * fs * fs / out channels."""
    c, h, w = x.shape
    t = ft  # one frame plus ft - 1 padding frames in front
    xt = np.zeros((c, t, h, w))
    xt[:, t - 1] = x
    xt = xt.reshape(c, t // ft, ft, h // fs, fs, w // fs, fs)  # c, T', ft, H', fs_h, W', fs_w
    xt = xt.transpose(0, 2, 4, 6, 1, 3, 5)  # c, ft, fs_h, fs_w, T', H', W'
    factor = ft * fs * fs
    xt = xt.reshape(c * factor, h // fs, w // fs)  # T' = 1
    g = c * factor // out
    return xt.reshape(out, g, h // fs, w // fs).mean(1)


def encode(x):
    """x [4, H, W] in [-1, 1] -> the posterior's mean [64, H/16, W/16] (not yet normalised by the latents' mean and std)."""
    x = conv(x, "encoder.conv_in", 3)
    dims = [96, 96, 192, 384, 768, 768]
    temporal = [False, True, True, True, False]
    for i in range(5):
        p = "encoder.down_blocks.%d." % i
        x0 = x
        for r in range(2):
            x = resblock(x, p + "resnets.%d" % r)
        if i < 4:
            x = conv_s2(x, p + "downsampler.resample.1")
            x = x + avgdown(x0, dims[i + 1], 2 if temporal[i] else 1, 2)
        else:
            x = x + avgdown(x0, dims[i + 1], 1, 1)
    p = "encoder.mid_block."
    x = resblock(x, p + "resnets.0")
    x = attention(x, p + "attentions.0")
    x = resblock(x, p + "resnets.1")
    x = silu(rmsnorm(x, W("encoder.norm_out.gamma")))
    x = conv(x, "encoder.conv_out", 3)
    x = conv(x, "quant_conv", 1)
    return x[:64]


def normalise(m):
    mu, sd = np.array(CFG["latents_mean"])[:, None, None], np.array(CFG["latents_std"])[:, None, None]
    return (m - mu) / sd


def synthetic(h, w):
    yy, xx = np.mgrid[0:h, 0:w] / np.array([h - 1, w - 1])[:, None, None]
    rng = np.random.default_rng(5)
    rgb = np.clip(np.round(np.stack([xx, yy, 0.5 * (xx + yy)], 0) * 255 + rng.normal(0, 12, (3, h, w))), 0, 255)
    return np.concatenate([rgb / 127.5 - 1.0, np.ones((1, h, w))], 0)  # RGBA scaled to [-1, 1], alpha 1


if __name__ == "__main__":
    sz = sys.argv[sys.argv.index("--size") + 1] if "--size" in sys.argv else "64"
    h, w = (int(a) for a in (sz.split("x") if "x" in sz else (sz, sz)))
    img = synthetic(h, w)
    m = encode(img)
    print("image %d x %d -> mean %s, rms %.4g; normalised rms %.4g" % (h, w, m.shape, np.sqrt((m ** 2).mean()), np.sqrt((normalise(m) ** 2).mean())))
    if "--torch32" in sys.argv:
        import torch
        from diffusers import AutoencoderKLQwenImage21
        vae = AutoencoderKLQwenImage21.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", subfolder="vae", torch_dtype=torch.float32)
        with torch.no_grad():
            t32 = vae.encode(torch.tensor(img, dtype=torch.float32)[None, :, None], return_dict=False)[0].mode()[0, :, 0].double().numpy()
        d = np.abs(m - t32)
        print("float64 vs torch float32 (the posterior's mean): mean abs error %.3g (%.3g of the rms), max %.3g" % (d.mean(), d.mean() / np.sqrt((m ** 2).mean()), d.max()))
    if "--save" in sys.argv:
        d = sys.argv[sys.argv.index("--save") + 1]
        os.makedirs(d, exist_ok=True)
        np.save(d + "/image.npy", img)
        np.save(d + "/mean_f64.npy", m)
        print("saved to", d)
