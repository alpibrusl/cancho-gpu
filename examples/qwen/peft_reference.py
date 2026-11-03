#!/usr/bin/env python3
"""peft's answers for the LoRA checks (docs/resident.md section 21; roadmap M7.2): the loss and the gradient of every adapter matrix of Qwen2.5-0.5B with rank-16 LoRA on the seven linear layers of the
blocks (alpha 32, no dropout), float32 on the CPU, for adapters given as {'l00.q.A': [r, in], 'l00.q.B': [out, r], ...} (qwen_ref.init_lora) and a batch of token ids; and, for a short training run,
the loss after each AdamW step on a list of batches.

    python3 examples/qwen/peft_reference.py OUTDIR [--b-std 0.02] [--seed 7]       # writes OUTDIR/grads_<tag>.npz and loss_<tag>.npy"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402
import qwen_ref as R  # noqa: E402
import qwen_session as S  # noqa: E402

MODULE = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}


def override_base(model, c, base, device, dtype):
    """Replace the block matrices of the loaded model by those of the f16 weights directory `base` (a dequantised 4-bit base, quant.py dequant): the model the session's QLoRA run is a LoRA run of (docs/resident.md
    section 27). Everything else (embeddings, norms, biases) stays the checkpoint's."""
    t = lambda a: torch.tensor(a, dtype=dtype, device=device)
    sh = R.shapes(c)
    get = lambda name: R.load_one(base, name, sh[name])  # one matrix at a time: a 7B model's binary64 copy would not fit
    for l in range(c.layers):
        ly = model.model.layers[l]
        w = get(f"l{l:02d}.wqkv")
        ly.self_attn.q_proj.weight.data = t(w[:c.d])
        ly.self_attn.k_proj.weight.data = t(w[c.d:c.d + c.dkv])
        ly.self_attn.v_proj.weight.data = t(w[c.d + c.dkv:])
        ly.self_attn.o_proj.weight.data = t(get(f"l{l:02d}.wo"))
        w = get(f"l{l:02d}.wgu")
        ly.mlp.gate_proj.weight.data = t(w[:c.ff])
        ly.mlp.up_proj.weight.data = t(w[c.ff:])
        ly.mlp.down_proj.weight.data = t(get(f"l{l:02d}.wd"))


def build(c, AB, r=16, alpha=32, device="cpu", dtype=torch.float32, base=None):
    """The peft model with the adapters set to AB (float64 arrays, rounded to float32); with `base` (a weights directory) its block matrices are that directory's."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(data.snapshot(), torch_dtype=dtype).to(device)
    if base:
        override_base(model, c, base, device, dtype)
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.0))
    mods = dict(model.named_modules())
    handles = {}
    for l in range(c.layers):
        for p, m in MODULE.items():
            mod = mods[f"base_model.model.model.layers.{l}.{m}"]
            la, lb = mod.lora_A["default"], mod.lora_B["default"]
            la.weight.data = torch.tensor(AB[f"l{l:02d}.{p}.A"], dtype=torch.float32, device=device)
            lb.weight.data = torch.tensor(AB[f"l{l:02d}.{p}.B"], dtype=torch.float32, device=device)
            handles[f"l{l:02d}.{p}.A"], handles[f"l{l:02d}.{p}.B"] = la.weight, lb.weight
    return model, handles


def loss_and_grads(model, handles, tokens):
    ids = torch.tensor(np.asarray(tokens), device=next(model.parameters()).device)
    model.zero_grad()
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    return loss.item(), {k: p.grad.detach().cpu().numpy().astype(np.float64) for k, p in handles.items()}


def train(model, handles, batches, steps, lr, betas=(0.9, 0.999), wd=0.0, lr_fn=None):
    """AdamW on the adapters, one batch per step (taken round and round); the loss before each step, and the adapters at the end."""
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(list(handles.values()), lr=lr, betas=betas, eps=1e-8, weight_decay=wd)
    losses = []
    for s in range(steps):
        if lr_fn:
            for g in opt.param_groups:
                g["lr"] = lr_fn(s + 1)
        ids = torch.tensor(np.asarray(batches[s % len(batches)]), device=dev)
        opt.zero_grad()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, {k: p.detach().cpu().numpy().astype(np.float64) for k, p in handles.items()}


if __name__ == "__main__":
    out = sys.argv[1]
    arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    wdir = arg("--weights", data.WEIGHTS)
    c = R.config_of(wdir)
    tok = data.fixed_batch(c.nb, c.t)
    for tag, bstd in (("init", 0.0), ("randB", float(arg("--b-std", 0.02)))):
        AB = R.init_lora(c, 16, int(arg("--seed", 7)), b_std=bstd)
        model, handles = build(c, AB, base=wdir if S.quant_group(wdir) else None)
        t0 = time.time()
        loss, grads = loss_and_grads(model, handles, tok)
        print(f"{tag}: loss {loss:.6f}, {time.time() - t0:.1f} s")
        np.savez(os.path.join(out, f"grads_{tag}.npz"), loss=np.array([loss]), **grads)
        del model, handles
