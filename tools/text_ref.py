"""D5a of docs/diffusion.md: the text encoder of Qwen-Image-2.1-Turbo (the language model of Qwen3-VL, 36 layers) in float64, from the checkpoint's weights and D0's token ids, written from the mathematics of Qwen3-VL's
text model, held against the BF16 hidden states torch produced on MPS (the transformer's `encoder_hidden_states` at step 0, D0's `text_embeds`).

    python tools/text_ref.py GOLDEN_DIR [OUTDIR]

What the pipeline takes is the residual stream after the 36 layers **before the final RMSNorm** (it hooks `norm` to return its input: `_get_qwen_prompt_embeds`), from row `drop_idx` on (the chat template's system part).
A layer, for N tokens: h = rmsnorm(x) (weight, eps 1e-6); q, k, v = h Wq^T, h Wk^T, h Wv^T (32 query heads and 8 key/value heads of 128: grouped attention, head i reads key/value head i // 4); q, k = per-head rmsnorm
(weight [128], eps 1e-6) *then* rotary embedding (rotate-half over the 128 channels: pairs (j, j + 64), theta 5e6, position = the token's index, the same on the three axes of mRoPE, so it is the ordinary rotary
embedding for text); causal softmax(q k^T / sqrt(128)) v; x = x + o Wo^T; x = x + (silu(rmsnorm(x) Wg^T) * (rmsnorm(x) Wu^T)) Wd^T.
Writes OUTDIR/layers_f64.npy ([36, N, 4096] float32, the residual stream after each layer) when OUTDIR is given, and prints the per-layer peak of every tensor the GPU path would hold in f16.
"""
import math
import os
import sys

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

snap = snapshot_download("Qwen/Qwen-Image-2.1-Turbo", allow_patterns=["text_encoder/*"])
_f = safe_open(os.path.join(snap, "text_encoder/model.safetensors"), framework="pt")
W = lambda name: _f.get_tensor("model.language_model." + name).double().numpy()
NL, H, KV, HD, THETA = 36, 32, 8, 128, 5e6


def rmsnorm(x, w):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6) * w


def silu(v):
    return v / (1 + np.exp(-v))


def rope_tables(n):
    inv = 1.0 / THETA ** (np.arange(0, HD, 2) / HD)  # [64]
    ang = np.arange(n)[:, None] * inv[None]
    return np.cos(ang), np.sin(ang)


def rot(t, cos, sin):  # t [N, heads, 128]; rotate-half: (t1, t2) -> (t1 c - t2 s, t2 c + t1 s)
    t1, t2 = t[..., :HD // 2], t[..., HD // 2:]
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([t1 * c - t2 * s, t2 * c + t1 * s], axis=-1)


def layer(x, i, cos, sin, peak):
    p = "layers.%d." % i
    n = x.shape[0]
    h = rmsnorm(x, W(p + "input_layernorm.weight"))
    peak["rmsnorm out (f16 input of q, k, v)"] = float(np.abs(h).max())
    q = (h @ W(p + "self_attn.q_proj.weight").T).reshape(n, H, HD)
    k = (h @ W(p + "self_attn.k_proj.weight").T).reshape(n, KV, HD)
    v = (h @ W(p + "self_attn.v_proj.weight").T).reshape(n, KV, HD)
    q = rot(rmsnorm(q, W(p + "self_attn.q_norm.weight")), cos, sin)
    k = rot(rmsnorm(k, W(p + "self_attn.k_norm.weight")), cos, sin)
    causal = np.tril(np.ones((n, n), dtype=bool))
    o = np.empty((n, H, HD))
    for hd in range(H):
        s = q[:, hd] @ k[:, hd // (H // KV)].T / math.sqrt(HD)
        s = np.where(causal, s, -np.inf)
        e = np.exp(s - s.max(-1, keepdims=True))
        e /= e.sum(-1, keepdims=True)
        o[:, hd] = e @ v[:, hd // (H // KV)]
    peak["attention o (f16 input of o_proj)"] = float(np.abs(o).max())
    x = x + o.reshape(n, H * HD) @ W(p + "self_attn.o_proj.weight").T
    h2 = rmsnorm(x, W(p + "post_attention_layernorm.weight"))
    peak["rmsnorm 2 out (f16 input of gate, up)"] = float(np.abs(h2).max())
    a = silu(h2 @ W(p + "mlp.gate_proj.weight").T) * (h2 @ W(p + "mlp.up_proj.weight").T)
    peak["swiglu (f16 input of down)"] = float(np.abs(a).max())
    return x + a @ W(p + "mlp.down_proj.weight").T


def encode(ids):
    emb = _f.get_slice("model.language_model.embed_tokens.weight")
    x = np.stack([emb[int(t):int(t) + 1][0].double().numpy() for t in ids])
    cos, sin = rope_tables(len(ids))
    after, peaks = [], []
    for i in range(NL):
        peak = {}
        x = layer(x, i, cos, sin, peak)
        peak["residual"] = float(np.abs(x).max())
        after.append(x.astype(np.float32))
        peaks.append(peak)
    return x, np.stack(after), peaks


def main():
    gold = sys.argv[1]
    ids = np.load(os.path.join(gold, "text_ids.npy"))[0].astype(int)
    drop = int(np.load(os.path.join(gold, "drop_idx.npy"))[0])
    want = np.load(os.path.join(gold, "text_embeds.npy")).astype(np.float64)[0]
    x, after, peaks = encode(ids)
    if len(sys.argv) > 2:
        os.makedirs(sys.argv[2], exist_ok=True)
        np.save(os.path.join(sys.argv[2], "layers_f64.npy"), after)
    got = x[drop:]
    d = np.abs(got - want)
    rms = float(np.sqrt((want ** 2).mean()))
    print("%d tokens (%d dropped); text encoder in float64 vs torch BF16 on MPS: %s, mean abs error %.4g (%.3g of the rms %.3g), max abs error %.4g" % (len(ids), drop, got.shape, d.mean(), d.mean() / rms, rms, d.max()))
    i = np.unravel_index(d.argmax(), d.shape)
    print("   the largest error is at row %d, channel %d: exact %.2f, torch %.2f" % (i[0], i[1], got[i], want[i]))
    if "--torch32" in sys.argv:  # the same model in torch float32 on the CPU (35 GB of memory): if it agrees with the float64 reference, the reference is the model and the BF16 error is BF16's
        import torch
        from transformers import Qwen3VLForConditionalGeneration
        m = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", subfolder="text_encoder", torch_dtype=torch.float32, low_cpu_mem_usage=True)
        lm = getattr(m.model, "language_model", m.model)
        h = lm.norm.register_forward_hook(lambda mod, a, out: a[0])
        with torch.no_grad():
            hs = m(input_ids=torch.tensor(ids)[None], attention_mask=torch.ones(1, len(ids), dtype=torch.long), output_hidden_states=True).hidden_states[-1][0].double().numpy()
        h.remove()
        d32 = np.abs(got - hs[drop:])
        d33 = np.abs(want - hs[drop:])
        print("float64 vs torch float32 on the CPU: mean abs error %.3g, max %.3g;  torch float32 vs torch BF16 (D0): mean abs error %.3g, max %.3g" % (d32.mean(), d32.max(), d33.mean(), d33.max()))
    mx = {}
    for pk in peaks:
        for k, v in pk.items():
            mx[k] = max(mx.get(k, 0.0), v)
    print("   largest values over the 36 layers (f16 holds 65504):", {k: round(v, 1) for k, v in mx.items()})


if __name__ == "__main__":
    main()
