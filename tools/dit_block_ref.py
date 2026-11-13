"""D2a of docs/diffusion.md: transformer block 0 of Qwen-Image-2.1-Turbo in float64, from the checkpoint's weights and D0's goldens, written from the
mathematics of the block (transformer_qwenimage21.py: QwenImage21TransformerBlock), and held against the BF16 block output torch produced on the MPS GPU.

    python tools/dit_block_ref.py GOLDEN_DIR

It says how far BF16 torch is from the exact block, which is the noise floor any kernel of D2 can be asked to meet (a kernel closer to the float64 block than
torch's own BF16 is closer to the truth, not "wrong"); how many of the block's weights change if they are narrowed from BF16 to f16 (the language has no BF16);
and the largest activation seen in the block against f16's limit of 65504.

The block, with x the [S, 4096] input, S = 21 text + 256 image tokens, and the shared modulation rows (row 0 from the sampled timestep, row 1 from t = 0; the text
tokens take row 1, the target image's tokens row 0):

    n1 = layernorm(x) (no affine, eps 1e-6);   a = n1 * (1 + scale1)
    q, k, v = a Wq^T, a Wk^T, a Wv^T  (32 heads of 128);   q, k = rmsnorm_head(q), rmsnorm_head(k) * weight, eps 1e-6;   q, k = rope(q, k)
    o = softmax(q k^T / sqrt(128) + mask) v (block-causal: q >= kv, or both in the target image);   x = x + tanh(gate1) * (o Wo^T)
    n2 = layernorm(x);   b = n2 * (1 + scale2);   x = x + tanh(gate2) * (silu(b Wg^T) * (b Wp^T)) Wout^T
"""
import json
import os
import sys

import numpy as np
from safetensors import safe_open
from huggingface_hub import snapshot_download


SNAP = None
INDEX = None


def _snap():
    global SNAP, INDEX
    if SNAP is None:
        SNAP = snapshot_download("Qwen/Qwen-Image-2.1-Turbo", allow_patterns=["transformer/*"])
        INDEX = json.load(open(os.path.join(SNAP, "transformer/diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    return SNAP, INDEX


def weight(name):
    snap, index = _snap()
    with safe_open(os.path.join(snap, "transformer", index[name]), framework="pt") as f:
        return f.get_tensor(name)


def load64(name):
    return weight(name).double().numpy()


BLOCK0 = {"q": "attn.to_q.weight", "k": "attn.to_k.weight", "v": "attn.to_v.weight", "o": "attn.to_out.0.weight", "nq": "attn.norm_q.weight", "nk": "attn.norm_k.weight",
          "g": "img_mlp.gate_layer.weight", "p": "img_mlp.proj.weight", "out": "img_mlp.out.weight"}


def layernorm(v):
    mu = v.mean(-1, keepdims=True)
    var = ((v - mu) ** 2).mean(-1, keepdims=True)
    return (v - mu) / np.sqrt(var + 1e-6)


def silu(v):
    return v / (1 + np.exp(-v))


def block(x, w, mod, rope, tmask, peak=None):
    """The block on x [S, 4096] in float64: w the nine weights (BLOCK0's keys), mod [2, 16384], rope [S, 64, 2], tmask [S] bool (the target image's tokens)."""
    S, D = x.shape
    H, HD = 32, 128
    peak = {} if peak is None else peak

    def pick(rows):  # rows [2, 4096]: target tokens take row 0, the others row 1
        return np.where(tmask[:, None], rows[0][None], rows[1][None])

    s1, g1, s2, g2 = (mod[:, i * D:(i + 1) * D] for i in range(4))
    a = layernorm(x) * (1 + pick(s1))
    q, k, v = (a @ w[n].T for n in ("q", "k", "v"))
    q, k, v = (t.reshape(S, H, HD) for t in (q, k, v))
    for name, t in (("q", q), ("k", k)):
        r = 1 / np.sqrt((t ** 2).mean(-1, keepdims=True) + 1e-6)
        t[...] = t * r * w["n" + name]
    fr = rope[..., 0] + 1j * rope[..., 1]  # [S, 64]

    def rot(t):
        z = t[..., 0::2] + 1j * t[..., 1::2]
        z = z * fr[:, None, :]
        out = np.empty_like(t)
        out[..., 0::2], out[..., 1::2] = z.real, z.imag
        return out

    q, k = rot(q), rot(k)
    idx = np.arange(S)
    allowed = (idx[:, None] >= idx[None, :]) | (tmask[:, None] & tmask[None, :])
    o = np.empty((S, H, HD))
    for h in range(H):
        sc = q[:, h] @ k[:, h].T / np.sqrt(HD)
        sc = np.where(allowed, sc, -np.inf)
        p = np.exp(sc - sc.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        o[:, h] = p @ v[:, h]
    peak["scores"] = max(peak.get("scores", 0), float(np.abs(q).max()), float(np.abs(k).max()))
    x = x + np.tanh(pick(g1)) * (o.reshape(S, D) @ w["o"].T)
    peak["after attention"] = float(np.abs(x).max())
    b = layernorm(x) * (1 + pick(s2))
    m = (silu(b @ w["g"].T) * (b @ w["p"].T)) @ w["out"].T
    peak["mlp output"] = float(np.abs(m).max())
    x = x + np.tanh(pick(g2)) * m
    peak["output"] = float(np.abs(x).max())
    return x


def goldens(gold):
    ld = lambda name: np.load(os.path.join(gold, name + ".npy")).astype(np.float64)
    return ld("block0_in")[0], ld("block0_out")[0], ld("block0_modulation"), ld("block0_rope"), ld("block0_target_mask") > 0.5


def main():
    gold = sys.argv[1]
    x, want, mod, rope, tmask = goldens(gold)
    peak = {"input": float(np.abs(x).max())}
    w = {k: load64("transformer_blocks.0." + v) for k, v in BLOCK0.items()}
    got = block(x.copy(), w, mod, rope, tmask, peak)

    def report(label, a, b):
        d = np.abs(a - b)
        rms = float(np.sqrt((b ** 2).mean()))
        print("%-46s max abs %.4g  mean abs %.4g  relative to the output's rms (%.3g): max %.3g  mean %.3g" % (label, d.max(), d.mean(), rms, d.max() / rms, d.mean() / rms))


    report("float64 block vs torch BF16 on MPS", got, want)
    d = np.abs(got - want)
    i = np.unravel_index(d.argmax(), d.shape)
    print("   the largest error is at token %d, channel %d: exact %.2f, torch %.2f (BF16 spacing there: %.0f)" % (i[0], i[1], got[i], want[i], 2.0 ** (np.floor(np.log2(abs(want[i]))) - 7)))
    rel = d / (np.abs(want) * 2.0 ** -8 + 1e-30)
    print("   error in units of half a BF16 step of the value (1 = one rounding): median %.3g, 99th percentile %.3g, 99.9th %.3g, max %.3g" % tuple(np.percentile(rel, [50, 99, 99.9, 100])))
    print("   largest values seen:", {k: round(v, 2) for k, v in peak.items()}, "(f16 holds 65504)")

    # the weights narrowed to f16: how many change, and what the block gives with them
    changed = total = 0
    w16 = {}
    for k, v in w.items():
        n = v.astype(np.float16).astype(np.float64)
        w16[k] = n
        changed += int((n != v).sum())
        total += v.size
    print("weights of block 0 that change when narrowed BF16 -> f16: %d of %d (%.4f%%)" % (changed, total, 100.0 * changed / total))
    peak.clear()
    report("float64 block, weights narrowed to f16, vs exact", block(x.copy(), w16, mod, rope, tmask), got)


if __name__ == "__main__":
    main()
