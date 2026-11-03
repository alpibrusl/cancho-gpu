#!/usr/bin/env python3
"""The f16 range of a Qwen2 forward pass (docs/resident.md sections 25.8 gap 6 and 28; roadmap G8): which activation of the session is rounded to f16, and how close its largest value comes to f16's 65,504.

    python3 examples/qwen/range_check.py [--base WEIGHTS_DIR] [--nb 4] [--t 512] [--dtype float32]      # QWEN_MODEL names the checkpoint's cache directory

transformers' forward pass on the CPU over the fixed batch (examples/qwen/data.py), with hooks on exactly the tensors the session holds in f16 (or hands to a product as an f16 operand):
the RMSNorm outputs (the operands of the q/k/v and gate/up products), the q, k and v projections with their biases (rounded to f16 after the rotary embedding), the input of the o projection, the gate and up
products (the lean step keeps them as f16), the input of the down projection (the SwiGLU output, an f16 operand), and the largest logit. The residual stream is f32 in the session and is only
reported. `--base` replaces the block matrices by those of a weights directory (a dequantised 4-bit base). Prints, per layer, the largest absolute value of each tensor, then the maxima over the layers, the
number of values above 65,504 and of non-finite ones, and where the largest value of the stream sits (position and channel)."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402
import qwen_ref as R  # noqa: E402

arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
F16_MAX = 65504.0


def main():
    from transformers import AutoModelForCausalLM
    nb, t = int(arg("--nb", 4)), int(arg("--t", 512))
    base = arg("--base", None)
    dt = getattr(torch, arg("--dtype", "float32"))
    model = AutoModelForCausalLM.from_pretrained(data.snapshot(), dtype=dt).eval()
    if base:
        import peft_reference
        peft_reference.override_base(model, R.config_of(base), base, "cpu", dt)
    tokens = torch.tensor(data.fixed_batch(nb, t))
    stats = {}
    where = {}

    def note(key, x, layer):
        x = x.detach().float()
        a = x.abs()
        m = float(a.max())
        row = stats.setdefault(key, {})
        row[layer] = (m, int((a > F16_MAX).sum()), int((~torch.isfinite(x)).sum()))
        if key == "resid" and m >= max((v[0] for v in row.values()), default=0):
            idx = int(a.argmax())
            tok = idx // x.shape[-1]
            where[layer] = (tok // x.shape[-2], tok % x.shape[-2], idx % x.shape[-1])

    hooks = []
    for l, ly in enumerate(model.model.layers):
        hooks.append(ly.input_layernorm.register_forward_hook(lambda m, i, o, l=l: note("norm1 out", o, l)))
        hooks.append(ly.post_attention_layernorm.register_forward_hook(lambda m, i, o, l=l: note("norm2 out", o, l)))
        for nm in ("q_proj", "k_proj", "v_proj"):
            hooks.append(getattr(ly.self_attn, nm).register_forward_hook(lambda m, i, o, l=l, nm=nm: note(nm[0], o, l)))
        hooks.append(ly.self_attn.o_proj.register_forward_pre_hook(lambda m, i, l=l: note("o in", i[0], l)))
        hooks.append(ly.mlp.gate_proj.register_forward_hook(lambda m, i, o, l=l: note("gate", o, l)))
        hooks.append(ly.mlp.up_proj.register_forward_hook(lambda m, i, o, l=l: note("up", o, l)))
        hooks.append(ly.mlp.down_proj.register_forward_pre_hook(lambda m, i, l=l: note("down in", i[0], l)))
        hooks.append(ly.register_forward_hook(lambda m, i, o, l=l: note("resid", o[0] if isinstance(o, tuple) else o, l)))
    hooks.append(model.model.norm.register_forward_hook(lambda m, i, o: note("final norm out", o, 0)))
    with torch.no_grad():
        out = model(input_ids=tokens, labels=tokens)
    note("logits", out.logits, 0)
    print(f"loss {out.loss.item():.6f} ({nb} x {t} tokens, {arg('--dtype', 'float32')}{', base ' + base if base else ''})")
    keys = ["norm1 out", "q", "k", "v", "o in", "norm2 out", "gate", "up", "down in", "resid"]
    print(f"{'layer':>5s} " + " ".join(f"{k:>10s}" for k in keys))
    nl = len(model.model.layers)
    for l in range(nl):
        print(f"{l:5d} " + " ".join(f"{stats[k][l][0]:10.1f}" for k in keys))
    print("\nmaximum over the layers (layer), the number of values above 65,504 and of non-finite values over all layers:")
    for k in keys + ["final norm out", "logits"]:
        mx = max(stats[k].items(), key=lambda kv: kv[1][0])
        over = sum(v[1] for v in stats[k].values())
        nonfin = sum(v[2] for v in stats[k].values())
        print(f"  {k:15s} {mx[1][0]:12.1f} (layer {mx[0]:2d}) = {mx[1][0] / F16_MAX * 100:6.2f}% of 65504;  above: {over}, non-finite: {nonfin}")
    big = sorted(((v[0], l) for l, v in stats["resid"].items()), reverse=True)[:3]
    print("largest residual-stream values:", [(round(m, 1), f"layer {l}", f"sequence {where[l][0]} position {where[l][1]}, channel {where[l][2]}") for m, l in big if l in where])
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
