#!/usr/bin/env python3
"""The target for roadmap M7 (docs/roadmap-models.md): how fast torch on MPS fine-tunes Qwen2.5 with LoRA on this machine. Rank-16 adapters on every linear layer of the blocks, batch of
B sequences of T tokens of random ids (speed only), AdamW; prints ms a step, tokens a second, peak memory.

    python3 examples/qwen/torch_baseline.py [--model Qwen/Qwen2.5-0.5B] [--b 8] [--t 512] [--dtype float32|float16|bfloat16] [--steps 8]"""
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
name, B, T, steps = arg("--model", "Qwen/Qwen2.5-0.5B"), int(arg("--b", 8)), int(arg("--t", 512)), int(arg("--steps", 8))
dtype = getattr(torch, arg("--dtype", "float32"))
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to("mps")
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.0))
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
ids = torch.randint(0, 151000, (B, T), device="mps")
times = []
for s in range(steps):
    t0 = time.perf_counter()
    with torch.autocast("mps", dtype=dtype, enabled=dtype != torch.float32):
        loss = model(input_ids=ids, labels=ids).loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    torch.mps.synchronize()
    times.append(time.perf_counter() - t0)
ms = sum(times[2:]) / len(times[2:]) * 1000
print(f"{name} {arg('--dtype', 'float32')} B={B} T={T}: {trainable / 1e6:.1f} M trainable, {ms:.0f} ms a step, {B * T / ms * 1000:.0f} tokens/s, peak driver memory {torch.mps.driver_allocated_memory() / 1e9:.1f} GB")
