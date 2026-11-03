#!/usr/bin/env python3
"""The sanity baseline of docs/roadmap-models.md M10: a tiny RANDOM-INIT Qwen3_5 model (real gated-delta geometry: 48 value heads, 16 key heads, 128 x 128 state; a narrow
residual stream) through mlx-lm's own LoRA path (mlx_lm.tuner.utils.linear_to_lora_layers, QLoRA with nn.quantize), one AdamW step at a time: seconds a step, tokens a second and
MLX's peak memory. Nothing is downloaded; the weights are random.

    python3 examples/qwen35/mlx_tiny.py [--layers 4] [--seq 128 256 512] [--batch 1] [--ckpt] [--quant] [--chunked]

mlx-lm's training path runs the delta rule as a Python loop of `_gated_delta_step_ops` over the tokens (`use_kernel=not self.training`, models/qwen3_5.py), differentiated by
MLX's autodiff; its Metal kernel (`gated_delta_kernel`) is the inference path and has no gradient. One GPU job: check `ps` first."""
import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers


def build(layers, hidden=512, ff=1024, vocab=2048, quant=False):
    text = dict(model_type="qwen3_5_text", hidden_size=hidden, intermediate_size=ff, num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                vocab_size=vocab, linear_num_value_heads=48, linear_num_key_heads=16, linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
                full_attention_interval=4, max_position_embeddings=4096, rms_norm_eps=1e-6,
                rope_parameters={"type": "default", "mrope_section": [1, 1, 0], "rope_theta": 10000000, "partial_rotary_factor": 0.25})
    model = qwen3_5.Model(qwen3_5.ModelArgs(model_type="qwen3_5", text_config=text))
    mx.eval(model.parameters())
    if quant:
        nn.quantize(model, group_size=64, bits=4)
    return model


def run(model, seq, batch, ckpt, steps=3):
    model.train()
    if ckpt:
        grad_checkpoint(model.language_model.model.layers[0])
    opt = optim.AdamW(1e-4)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y).mean()
    vg = nn.value_and_grad(model, loss_fn)
    x = mx.random.randint(0, 2000, (batch, seq))
    y = mx.random.randint(0, 2000, (batch, seq))
    ts = []
    mx.reset_peak_memory()
    for i in range(steps):
        t = time.time()
        loss, g = vg(model, x, y)
        opt.update(model, g)
        mx.eval(loss, model.parameters(), opt.state)
        ts.append(time.time() - t)
    return min(ts[1:]) if len(ts) > 1 else ts[0], mx.get_peak_memory() / 1e9, float(loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--quant", action="store_true")
    ap.add_argument("--chunked", action="store_true", help="replace mlx-lm's token loop by mlx_chunked.py (the chunked form in MLX ops)")
    a = ap.parse_args()
    if a.chunked:
        import mlx_chunked
        qwen3_5.gated_delta_update = mlx_chunked.gated_delta_update_chunked
    for seq in a.seq:
        model = build(a.layers, quant=a.quant)
        linear_to_lora_layers(model.language_model.model, a.layers, {"rank": 16, "scale": 2.0, "dropout": 0.0})
        model.freeze()
        # unfreeze the LoRA parameters only (mlx-lm's `train_model` does the same through lora layers' trainable_parameters)
        for _, m in model.named_modules():
            if hasattr(m, "lora_a"):
                m.unfreeze(keys=["lora_a", "lora_b"])
        n = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
        dt, peak, loss = run(model, seq, a.batch, a.ckpt)
        print(f"layers {a.layers} seq {seq} batch {a.batch} ckpt {a.ckpt} quant {a.quant} chunked {a.chunked}: {dt*1000:.0f} ms a step, {a.batch*seq/dt:.0f} tokens/s, peak {peak:.2f} GB, trainable {n/1e6:.2f} M, loss {loss:.3f}")


if __name__ == "__main__":
    main()
