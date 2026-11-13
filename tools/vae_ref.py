"""D4a of docs/diffusion.md: the VAE decoder of Qwen-Image-2.1-Turbo in float64, from the checkpoint's weights, written from the mathematics of autoencoder_kl_qwenimage21.py (AutoencoderKLQwenImage21._decode for one
frame: post_quant_conv, then Decoder3d with the feature cache's first-frame branches), held against D0's BF16 decode on MPS; and the count of what it costs.

    python tools/vae_ref.py GOLDEN_DIR

The decoder, for a latent z [64, h, w] (already multiplied by the latents' std and shifted by their mean, as the pipeline does; this is D0's `vae_in`):
  x = conv1x1(z) (post_quant_conv); x = conv3x3(x) to 1152 channels (conv_in);
  mid block: residual(1152), attention (one head of 1152 over all h w positions, an RMS norm before it, 1x1 projections), residual(1152);
  five up blocks of (in -> out) = (1152 -> 1152), (1152 -> 1152), (1152 -> 576), (576 -> 288), (288 -> 144): three residual blocks, then, for the first four, a nearest-neighbour upsample by 2 and a 3x3 convolution,
  plus a *shortcut*: the block's input with its channels repeated and rearranged into 2 x 2 spatial positions (DupUp3D: depth-to-space of repeated channels) added to the output;
  RMS norm, SiLU, conv3x3 to 4 channels (conv_out), clamp to [-1, 1].
A residual block is: shortcut = conv1x1(x) when the channels change, else x; y = conv3x3(silu(rmsnorm(conv3x3(silu(rmsnorm(x)))))) (each rmsnorm: x / max(|x|, 1e-12) over the channels times sqrt(C) times gamma); y + shortcut.
For a single frame the temporal parts do nothing but one thing: the first three up blocks are 'temporal' (factor_t 2) and DupUp's first-chunk rule keeps the second of the two frames its rearrangement makes.
"""
import json
import math
import os
import sys

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

snap = snapshot_download("Qwen/Qwen-Image-2.1-Turbo", allow_patterns=["vae/*"])
_f = safe_open(os.path.join(snap, "vae/diffusion_pytorch_model.safetensors"), framework="pt")
W = lambda name: _f.get_tensor(name).double().numpy()
FLOPS = {}


def conv(x, name, k, pad=None, tag=None):
    """x [C, H, W]; weight [O, C, k, k]; bias [O]."""
    w, b = W(name + ".weight"), W(name + ".bias")
    o, c = w.shape[0], w.shape[1]
    h, wd = x.shape[1:]
    if k == 3:
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
        y = np.zeros((o, h * wd))
        for dy in range(3):
            for dx in range(3):
                y += w[:, :, dy, dx] @ xp[:, dy:dy + h, dx:dx + wd].reshape(c, h * wd)
    else:
        y = w[:, :, 0, 0] @ x.reshape(c, h * wd)
    FLOPS[tag or name] = 2.0 * o * c * k * k * h * wd
    return (y + b[:, None]).reshape(o, h, wd)


def rmsnorm(x, gamma):
    c = x.shape[0]
    n = np.sqrt((x ** 2).sum(0, keepdims=True))
    return x / np.maximum(n, 1e-12) * math.sqrt(c) * gamma.reshape(c, *([1] * (x.ndim - 1)))


def silu(x):
    return x / (1 + np.exp(-x))


def resblock(x, p):
    h = conv(x, p + ".conv_shortcut", 1) if (p + ".conv_shortcut.weight") in _f.keys() else x
    y = conv(silu(rmsnorm(x, W(p + ".norm1.gamma"))), p + ".conv1", 3)
    y = conv(silu(rmsnorm(y, W(p + ".norm2.gamma"))), p + ".conv2", 3)
    return y + h


def attention(x, p):
    c, h, w = x.shape
    n = rmsnorm(x, W(p + ".norm.gamma"))
    qkv = conv(n, p + ".to_qkv", 1)  # [3c, h, w]
    q, k, v = (qkv[i * c:(i + 1) * c].reshape(c, h * w).T for i in range(3))  # [hw, c]
    s = q @ k.T / math.sqrt(c)
    s = np.exp(s - s.max(-1, keepdims=True))
    s /= s.sum(-1, keepdims=True)
    FLOPS[p + ".scores"] = 2.0 * h * w * h * w * c * 2
    return conv((s @ v).T.reshape(c, h, w), p + ".proj", 1) + x


def dupup(x, out, ft, fs=2):
    """DupUp3D for one frame, first_chunk=True: channels repeated, then 'depth to space' over (ft, fs, fs), the first ft - 1 frames dropped."""
    c, h, w = x.shape
    reps = out * ft * fs * fs // c
    x = np.repeat(x, reps, axis=0)  # [out * ft * fs * fs, h, w]
    x = x.reshape(out, ft, fs, fs, h, w).transpose(0, 4, 1, 5, 2, 3)  # out, h, ft, w, fs(h), fs(w) -> see below
    # torch: view (out, ft, fs, fs, T=1, h, w) then permute (0, 1, 5, 2, 6, 3, 7, 4): out, T, ft, h, fs, w, fs -> frames (T ft), rows (h fs), cols (w fs)
    x = np.asarray(x)  # [out, h, ft, w, fs, fs]
    x = np.transpose(x, (0, 2, 1, 4, 3, 5)).reshape(out, ft, h * fs, w * fs)  # (out, frame, h fs, w fs)
    return x[:, ft - 1]


def upsample_conv(x, p):
    x = x.repeat(2, axis=1).repeat(2, axis=2)  # nearest-exact for an integer scale
    return conv(x, p + ".resample.1", 3)


def decode(z):
    x = conv(z, "post_quant_conv", 1)
    x = conv(x, "decoder.conv_in", 3)
    p = "decoder.mid_block."
    x = resblock(x, p + "resnets.0")
    x = attention(x, p + "attentions.0")
    x = resblock(x, p + "resnets.1")
    outs = [1152, 1152, 576, 288, 144]
    temporal = [True, True, True, False, False]
    for i in range(5):
        p = "decoder.up_blocks.%d." % i
        x0 = x
        for r in range(3):
            x = resblock(x, p + "resnets.%d" % r)
        if i < 4:
            x = upsample_conv(x, p + "upsampler")
            x = x + dupup(x0, outs[i], 2 if temporal[i] else 1)
    x = silu(rmsnorm(x, W("decoder.norm_out.gamma")))
    return np.clip(conv(x, "decoder.conv_out", 3), -1.0, 1.0)


def main():
    gold = sys.argv[1]
    z = np.load(os.path.join(gold, "vae_in.npy")).astype(np.float64)[0, :, 0]
    want = np.load(os.path.join(gold, "vae_out.npy")).astype(np.float64)[0, :, 0]
    got = decode(z)
    d = np.abs(got - want)
    print("decoder in float64 vs torch BF16 on MPS: output %s, mean abs error %.4g, max abs error %.4g (output rms %.4g; the output is clamped to [-1, 1])" % (got.shape, d.mean(), d.max(), np.sqrt((want ** 2).mean())))
    per_channel = ["%.4g" % d[c].mean() for c in range(d.shape[0])]
    print("   mean abs error per channel:", ", ".join(per_channel))
    try:  # the same decoder in torch float32 on the CPU: if it agrees with the float64 one, the one above is the decoder and the BF16 error is BF16's
        import torch
        from diffusers import AutoencoderKLQwenImage21
        vae = AutoencoderKLQwenImage21.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", subfolder="vae", torch_dtype=torch.float32)
        with torch.no_grad():
            t32 = vae.decode(torch.tensor(z, dtype=torch.float32)[None, :, None], return_dict=False)[0][0, :, 0].double().numpy()
        d32 = np.abs(got - t32)
        print("float64 vs torch float32 on the CPU: mean abs error %.3g, max abs error %.3g;  torch float32 vs torch BF16 (D0): mean abs error %.3g, max %.3g" % (d32.mean(), d32.max(), np.abs(t32 - want).mean(), np.abs(t32 - want).max()))
    except Exception as e:  # noqa: BLE001
        print("torch float32 check skipped:", repr(e)[:120])
    tot = sum(FLOPS.values())
    print("FLOPs for a %dx%d latent (%dx%d image): %.3g in all" % (z.shape[1], z.shape[2], z.shape[1] * 16, z.shape[2] * 16, tot))
    for k, v in sorted(FLOPS.items(), key=lambda kv: -kv[1])[:8]:
        print("   %-48s %.3g (%.0f%%)" % (k, v, 100 * v / tot))
    scale = (64 * 64) / (z.shape[1] * z.shape[2])
    print("   scaled to a 64x64 latent (1024x1024): %.3g FLOPs for the convolutions, the attention's scores grow by %.0f times more" % (tot * scale, scale * scale / scale))


if __name__ == "__main__":
    main()
