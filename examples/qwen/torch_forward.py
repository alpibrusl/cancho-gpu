#!/usr/bin/env python3
"""Tokens a second of the Qwen2.5-0.5B forward pass (with the loss) in torch on MPS, on the fixed batch of data.py, and the memory it holds (docs/resident.md section 20).

    python3 examples/qwen/torch_forward.py [--dtype float32|float16] [--b 8] [--t 512] [--reps 8]   # float16 is autocast (the weights stay float32)"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402

arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
dtype = getattr(torch, arg("--dtype", "float32"))
B, T, reps = int(arg("--b", 8)), int(arg("--t", 512)), int(arg("--reps", 8))
from transformers import AutoModelForCausalLM  # noqa: E402

model = AutoModelForCausalLM.from_pretrained(data.snapshot(), dtype=torch.float32).to("mps").eval()
ids = torch.tensor(data.fixed_batch(B, T), device="mps")
times = []
for r in range(reps + 2):
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast("mps", dtype=dtype, enabled=dtype != torch.float32):
        loss = model(input_ids=ids, labels=ids).loss
    torch.mps.synchronize()
    times.append(time.perf_counter() - t0)
ms = sorted(times[2:])[len(times[2:]) // 2] * 1000
print(f"torch MPS {arg('--dtype', 'float32')}{' autocast' if dtype != torch.float32 else ''} B={B} T={T}: loss {loss.item():.5f}, {ms:.0f} ms a forward (median of {reps}), {B * T / ms * 1000:.0f} tokens/s, "
      f"peak driver memory {torch.mps.driver_allocated_memory() / 1e9:.1f} GB, peak current allocated {torch.mps.current_allocated_memory() / 1e9:.1f} GB")
