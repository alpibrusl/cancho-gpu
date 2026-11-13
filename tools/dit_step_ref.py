"""D2d of docs/diffusion.md: one full denoising step of Qwen-Image-2.1-Turbo's transformer (all 32 blocks) in float64, from the checkpoint's weights and D0's goldens, written from
the mathematics of transformer_qwenimage21.py (QwenImage21Transformer2DModel.forward), held against the BF16 output torch produced.

    python tools/dit_step_ref.py GOLDEN_DIR OUTDIR

Writes OUTDIR/blocks_f64.npy ([32, 277, 4096] float32: the residual stream after each block, for the GPU's per-block comparison) and prints, per block, the largest value of every tensor that
the GPU path holds in f16 (the f16 limit is 65504): the inputs of the projections and the output of SwiGLU. The front end is img_in (64 -> 4096), txt_in (RMSNorm with a zero-centred weight, a
linear layer, GELU (tanh), a linear layer), the timestep embedding (sinusoid of 256, a linear layer, SiLU, a linear layer) and the shared modulation (SiLU, a linear layer to 4 x 4096); the
back end is norm_out (a layer norm times 1 + scale) and proj_out (4096 -> 64). Row 0 of the timestep batch is the sampled one, row 1 is t = 0 (text tokens read it: `causal_condition`).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dit_block_ref as ref  # noqa: E402


def silu(v):
    return v / (1 + np.exp(-v))


def gelu_tanh(v):
    return 0.5 * v * (1 + np.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v ** 3)))


def front(gold, latents=None, t=None, text=None):
    """The front end in float64. `latents` [256, 64] and `t` (the timestep / 1000) default to D0's step 0."""
    ld = lambda n: np.load(os.path.join(gold, n + ".npy")).astype(np.float64)
    text = ld("text_embeds")[0] if text is None else text
    latents = ld("step0_in")[0] if latents is None else latents
    t = float(ld("step0_timestep")[0]) if t is None else t
    w = ref.load64
    tw = w("txt_in.text_norm.weight")
    tn = text / np.sqrt((text ** 2).mean(-1, keepdims=True) + 1e-6) * (tw + 1)
    txt = gelu_tanh(tn @ w("txt_in.in_layer.weight").T) @ w("txt_in.out_layer.weight").T
    img = latents @ w("img_in.weight").T
    x = np.concatenate([txt, img], axis=0)
    freqs = np.exp(-math.log(10000) * np.arange(128) / 128)
    args = np.array([t, 0.0])[:, None] * 1000.0 * freqs[None]
    emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    temb = silu(emb @ w("time_text_embed.timestep_embedder.linear_1.weight").T) @ w("time_text_embed.timestep_embedder.linear_2.weight").T
    mod = silu(temb) @ w("modulation.1.weight").T
    return x, temb, mod


def back(x, temb, tmask):
    scale = silu(temb) @ ref.load64("norm_out.linear.weight").T
    rows = np.where(tmask[:, None], scale[0][None], scale[1][None])
    return (ref.layernorm(x) * (1 + rows)) @ ref.load64("proj_out.weight").T


def main():
    gold, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    _, want_block0, mod0, rope, tmask = ref.goldens(gold)
    x, temb, mod = front(gold)
    print("front end: sequence %s, modulation differs from D0's block-0 golden by %.3g (max abs)" % (x.shape, np.abs(mod - mod0).max()))
    after = []
    for i in range(32):
        w = {k: ref.load64("transformer_blocks.%d.%s" % (i, v)) for k, v in ref.BLOCK0.items()}
        peak = {}
        x = ref.block(x, w, mod, rope, tmask, peak)
        after.append(x.astype(np.float32))
        print("block %2d  residual max %8.1f  %s" % (i, np.abs(x).max(), "  ".join("%s %.1f" % (k.split(" (")[0] + (" [" + k.split("(")[1] if "(" in k else ""), v) for k, v in peak.items() if k not in ("scores", "after attention", "mlp output", "output"))), flush=True)
    np.save(os.path.join(out, "blocks_f64.npy"), np.stack(after))
    y = back(x, temb, tmask)
    np.save(os.path.join(out, "step_out_f64.npy"), y)
    want = np.load(os.path.join(gold, "step0_out.npy")).astype(np.float64)[0]
    d = np.abs(y - want)
    print("step output [277, 64] against torch BF16 (step0_out): mean abs error %.4g, max abs error %.4g, output rms %.4g (relative mean %.3g)" % (d.mean(), d.max(), np.sqrt((want ** 2).mean()), d.mean() / np.sqrt((want ** 2).mean())))
    print("the block-0 golden check of the front end: block 0 from this front end vs D0's golden is in tools/dit_block_ref.py")


if __name__ == "__main__":
    main()
