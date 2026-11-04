"""One LoRA step of the real Qwen3.8-27B-4bit in MLX (docs/roadmap-models.md 9.4 measurement note, 2026-11-04). Offline: HF_HUB_OFFLINE=1. One GPU job; run under a swap/RSS watcher.
    python3 examples/qwen35/mlx_27b_step.py [--ckpt] [--chunked] [--steps 3]"""
import argparse, os, time, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models import qwen3_5
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", action="store_true"); ap.add_argument("--chunked", action="store_true")
ap.add_argument("--steps", type=int, default=3)
a = ap.parse_args()
if a.chunked:
    import mlx_chunked
    qwen3_5.gated_delta_update = mlx_chunked.gated_delta_update_chunked
model, tok = load("mlx-community/Qwen3.8-27B-4bit")
mx.eval(model.parameters())
print("loaded, active GB", mx.get_active_memory()/1e9, flush=True)
model.freeze()
lm = model.language_model.model
linear_to_lora_layers(lm, -1, {"rank": 16, "scale": 2.0, "dropout": 0.0})
for _, m in model.named_modules():
    if hasattr(m, "lora_a"): m.unfreeze(keys=["lora_a", "lora_b"])
n = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
model.train()
if a.ckpt: grad_checkpoint(lm.layers[0])
opt = optim.AdamW(1e-4)
def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x).astype(mx.float32), y).mean()
vg = nn.value_and_grad(model, loss_fn)
mx.random.seed(0)
x = mx.random.randint(0, 200000, (1, 512)); y = mx.random.randint(0, 200000, (1, 512))
mx.reset_peak_memory(); ts = []
for i in range(a.steps):
    t = time.time()
    loss, g = vg(model, x, y); opt.update(model, g); mx.eval(loss, model.parameters(), opt.state)
    ts.append(time.time()-t)
    print(f"step {i}: {ts[-1]:.2f}s loss {float(loss):.3f} peak {mx.get_peak_memory()/1e9:.2f} GB", flush=True)
w = ts[1:]; dt = sum(w)/len(w)
print("RESULT", json.dumps(dict(ckpt=a.ckpt, chunked=a.chunked, trainable_M=n/1e6, s_per_step=dt, tok_s=512/dt, peak_GB=mx.get_peak_memory()/1e9)))
