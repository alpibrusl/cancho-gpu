#!/usr/bin/env python3
"""V0 of docs/diffusion.md section 6: the front end and the blocks of Wan2.2-TI2V-5B's transformer in float64, written from the mathematics of diffusers' `transformer_wan.py` (`WanTransformer3DModel`), held to torch.

    python tools/wan_ref.py [--frames F] [--size HxW] [--text N] [--blocks B] [--seed S] [--torch32] [--bf16] [--mutate NAME]

The weights are the first shard of `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (the patch embedding, the condition embedder, the head's table and blocks 0 to 7: `hf download` it first; the cache is read). A latent [48, F, H, W] is patched by a 1 x 2 x 2 convolution into F x H/2 x W/2 tokens of width 3,072;
the timestep is **one per token** (TI2V: the conditioning frame's tokens are at t = 0, the others at the sampled t), embedded by a sinusoidal table and two linear layers, and projected to six modulation vectors a token; the text (umT5's [N, 4,096]) goes through
`text_embedder` (linear, tanh GELU, linear); the rotary table is the three axes of one head's 128 channels split 44 (time), 42 (height), 42 (width), each axis the half-frequencies of its own length repeated for the pair. A block: the layer norm without affine modulated by
(shift, scale) from `scale_shift_table + temb`, self-attention with RMS norms of q and k across the 24 heads (3,072) and the rotary rotation of adjacent pairs, a gated residual; a layer norm with affine, cross-attention to the text (no rotary), a plain residual;
a modulated layer norm, a feed-forward with a tanh GELU (14,336), a gated residual.

`--torch32` builds torch's `WanTransformer3DModel` with the same weights in float32 (the blocks and embedders run one by one so that every stage is compared) and prints each difference; `--bf16` does the same in BF16 (torch's floor, as D2 to D9 measured it for the other models);
`--mutate` makes one deliberate host-side error (`rope-pairs`: the rotary rotating the halves instead of adjacent pairs; `no-gate`; `qk-norm-per-head`; `time-flip`: the sinusoidal table with sine first) that the comparison must refuse.
"""
import glob
import json
import math
import os
import sys

import numpy as np
from safetensors import safe_open

arg = lambda name, default=None: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/*"))[0]
CFG = json.load(open(SNAP + "/transformer/config.json"))
MUT = arg("--mutate")
D, NH, HD, FFN, EPS = CFG["num_attention_heads"] * CFG["attention_head_dim"], CFG["num_attention_heads"], CFG["attention_head_dim"], CFG["ffn_dim"], CFG["eps"]
_f = safe_open(SNAP + "/transformer/diffusion_pytorch_model-00001-of-00005.safetensors", framework="np")
W = lambda name: _f.get_tensor(name).astype(np.float64)


def layer_norm(x, w=None, b=None):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    y = (x - m) / np.sqrt(v + EPS)
    return y if w is None else y * w + b


def rms_norm(x, w):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + EPS) * w


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def silu(x):
    return x / (1.0 + np.exp(-x))


def linear(x, name):
    return x @ W(name + ".weight").T + W(name + ".bias")


def patchify(lat):
    """[48, F, H, W] -> [F * H/2 * W/2, 3072]: the 1 x 2 x 2 stride-2 convolution (kernel [3072, 48, 1, 2, 2])."""
    c, f, h, w = lat.shape
    t = lat.reshape(c, f, h // 2, 2, w // 2, 2).transpose(1, 2, 4, 0, 3, 5).reshape(f * (h // 2) * (w // 2), c * 4)
    return t @ W("patch_embedding.weight").reshape(D, -1).T + W("patch_embedding.bias")


def rope_tables(f, h, w, theta=10000.0):
    """cos, sin [tokens, 64]: the table per pair of channels (the frequencies of the axis, one per pair), t axis 44 channels, h and w 42 each."""
    hd = wd = 2 * (HD // 6)
    td = HD - hd - wd
    cos, sin = [], []
    for dim, n, rep in ((td, f, (h * w, 1)), (hd, h, (None, w)), (wd, w, (None, None))):
        fr = 1.0 / theta ** (np.arange(0, dim, 2)[: dim // 2] / dim)
        ang = np.outer(np.arange(n), fr)
        cos.append(np.cos(ang))
        sin.append(np.sin(ang))
    ct = np.repeat(cos[0], h * w, axis=0)                                                       # time varies slowest
    st = np.repeat(sin[0], h * w, axis=0)
    ch = np.tile(np.repeat(cos[1], w, axis=0), (f, 1))
    sh = np.tile(np.repeat(sin[1], w, axis=0), (f, 1))
    cw = np.tile(cos[2], (f * h, 1))
    sw = np.tile(sin[2], (f * h, 1))
    return np.concatenate([ct, ch, cw], 1), np.concatenate([st, sh, sw], 1)


def rotate(x, cos, sin):
    """x [tokens, NH, 128]: adjacent pairs (2i, 2i + 1) rotated by the angle of pair i."""
    if MUT == "rope-pairs":
        x1, x2 = x[..., : HD // 2], x[..., HD // 2:]
        c, s = cos[:, None, :], sin[:, None, :]
        return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[:, None, :], sin[:, None, :]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def time_embed(t):
    """[tokens] timesteps -> (temb [tokens, 3072], proj [tokens, 6, 3072])."""
    half = 128
    fr = np.exp(-math.log(10000.0) * np.arange(half) / half)
    a = t[:, None] * fr[None]
    emb = np.concatenate([np.sin(a), np.cos(a)], 1) if MUT == "time-flip" else np.concatenate([np.cos(a), np.sin(a)], 1)
    temb = linear(silu(linear(emb, "condition_embedder.time_embedder.linear_1")), "condition_embedder.time_embedder.linear_2")
    proj = linear(silu(temb), "condition_embedder.time_proj").reshape(len(t), 6, D)
    return temb, proj


def text_embed(txt):
    return linear(gelu_tanh(linear(txt, "condition_embedder.text_embedder.linear_1")), "condition_embedder.text_embedder.linear_2")


def attention(q, k, v):
    """[nq, NH, 128], [nk, NH, 128] -> [nq, 3072]"""
    s = np.einsum("qhd,khd->hqk", q, k) / math.sqrt(HD)
    s -= s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", p, v).reshape(len(q), D)


def block(x, ctx, proj, cos, sin, b):
    """One block: x [tokens, 3072], ctx [N, 3072] (the text after its embedder), proj [tokens, 6, 3072] (per token)."""
    p = "blocks.%d." % b
    sh1, sc1, g1, sh2, sc2, g2 = np.moveaxis(W(p + "scale_shift_table") + proj, 1, 0)
    if MUT == "no-gate":
        g1 = np.ones_like(g1)
    h = layer_norm(x) * (1 + sc1) + sh1
    q, k, v = (linear(h, p + "attn1.to_" + n) for n in "qkv")
    if MUT == "qk-norm-per-head":
        q = (q.reshape(-1, NH, HD) / np.sqrt((q.reshape(-1, NH, HD) ** 2).mean(-1, keepdims=True) + EPS)).reshape(-1, D) * W(p + "attn1.norm_q.weight")
        k = (k.reshape(-1, NH, HD) / np.sqrt((k.reshape(-1, NH, HD) ** 2).mean(-1, keepdims=True) + EPS)).reshape(-1, D) * W(p + "attn1.norm_k.weight")
    else:
        q, k = rms_norm(q, W(p + "attn1.norm_q.weight")), rms_norm(k, W(p + "attn1.norm_k.weight"))
    q, k, v = (a.reshape(-1, NH, HD) for a in (q, k, v))
    a = attention(rotate(q, cos, sin), rotate(k, cos, sin), v)
    x = x + linear(a, p + "attn1.to_out.0") * g1
    h = layer_norm(x, W(p + "norm2.weight"), W(p + "norm2.bias"))
    q = rms_norm(linear(h, p + "attn2.to_q"), W(p + "attn2.norm_q.weight")).reshape(-1, NH, HD)
    k = rms_norm(linear(ctx, p + "attn2.to_k"), W(p + "attn2.norm_k.weight")).reshape(-1, NH, HD)
    v = linear(ctx, p + "attn2.to_v").reshape(-1, NH, HD)
    x = x + linear(attention(q, k, v), p + "attn2.to_out.0")
    h = layer_norm(x) * (1 + sc2) + sh2
    ff = linear(gelu_tanh(linear(h, p + "ffn.net.0.proj")), p + "ffn.net.2")
    return x + ff * g2


def inputs(f, h, w, ntext, seed):
    rng = np.random.default_rng(seed)
    lat = rng.standard_normal((48, f, h, w))
    txt = rng.standard_normal((ntext, CFG["text_dim"]))
    t = np.full(f * (h // 2) * (w // 2), 500.0)
    t[: (h // 2) * (w // 2)] = 0.0  # the conditioning frame's tokens are at t = 0
    return lat, txt, t


def forward(lat, txt, t, nblocks, stages=None):
    f, h, w = lat.shape[1:]
    x = patchify(lat)
    cos, sin = rope_tables(f, h // 2, w // 2)
    temb, proj = time_embed(t)
    ctx = text_embed(txt)
    if stages is not None:
        stages.update({"patch": x, "temb": temb, "proj": proj, "text": ctx, "cos": cos, "sin": sin})
    for b in range(nblocks):
        x = block(x, ctx, proj, cos, sin, b)
        if stages is not None:
            stages["block%d" % b] = x
    return x


if __name__ == "__main__":
    f, (h, w) = int(arg("--frames", 3)), (int(v) for v in arg("--size", "8x12").split("x"))
    ntext, nblocks, seed = int(arg("--text", 16)), int(arg("--blocks", 2)), int(arg("--seed", 0))
    lat, txt, t = inputs(f, h, w, ntext, seed)
    st = {}
    out = forward(lat, txt, t, nblocks, st)
    print("latent [48, %d, %d, %d] -> %d tokens, text %d tokens, %d blocks: output rms %.4g" % (f, h, w, len(out), ntext, nblocks, np.sqrt((out ** 2).mean())))
    if "--torch32" in sys.argv or "--bf16" in sys.argv:
        import torch
        from diffusers import WanTransformer3DModel
        cfg = dict(CFG)
        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        cfg["num_layers"] = nblocks
        for dt, name in ((torch.float32, "float32"), (torch.bfloat16, "bfloat16")):
            if (dt == torch.float32 and "--torch32" not in sys.argv) or (dt == torch.bfloat16 and "--bf16" not in sys.argv):
                continue
            m = WanTransformer3DModel(**cfg)
            sd = {k: torch.tensor(_f.get_tensor(k)) for k in _f.keys() if k.startswith(("patch_embedding", "condition_embedder")) or any(k.startswith("blocks.%d." % b) for b in range(nblocks))}
            miss = m.load_state_dict(sd, strict=False)
            assert not [k for k in miss.missing_keys if not k.startswith(("proj_out", "scale_shift_table"))], miss.missing_keys
            m = m.to(dt).eval()
            with torch.no_grad():
                l = torch.tensor(lat, dtype=dt)[None]
                rope = m.rope(l)
                x = m.patch_embedding(l).flatten(2).transpose(1, 2).contiguous()
                tt = torch.tensor(t, dtype=torch.float32)[None]
                temb, tproj, ctx, _ = m.condition_embedder(tt.flatten(), torch.tensor(txt, dtype=dt)[None], None, timestep_seq_len=tt.shape[1])
                tproj = tproj.unflatten(2, (6, -1))
                got = {"patch": x[0], "temb": temb[0], "proj": tproj[0], "text": ctx[0], "cos": rope[0][0, :, 0, 0::2], "sin": rope[1][0, :, 0, 1::2]}
                for b in range(nblocks):
                    x = m.blocks[b](x, ctx, tproj, rope)
                    got["block%d" % b] = x[0]
            print("\ntorch %s against float64 (mean abs error, of the rms; max abs error):" % name)
            for k in st:
                a, e = got[k].float().double().numpy(), st[k]
                r = float(np.sqrt((e ** 2).mean()))
                d = np.abs(a - e)
                print("  %-8s mean %.3g (%.3g of the rms %.3g), max %.3g" % (k, d.mean(), d.mean() / r, r, d.max()))
