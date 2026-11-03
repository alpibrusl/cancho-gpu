#!/usr/bin/env python3
"""A smoke LoRA run on the M8 data: does it train? (docs/roadmap-models.md section 7; NOT the experiment's training run, which is Phase 2.)

    python3 examples/qwen/m8_smoke.py DATA_DIR [--steps 30] [--batch 4] [--lr 2e-4] [--model DIR] [--device mps]

Qwen2.5-1.5B-Instruct (float32 on the device), peft LoRA r=16 alpha=32 on q, k, v, o, gate, up, down, AdamW (0.9, 0.999), a linear warm-up over a tenth of the steps then cosine to 0, `--batch` sequences of 512 a step taken in order
from packed.npz's train split (m8_pack.py; the loss counts only the tokens its mask marks: whole documents and the replies of pairs). Before the first step and after the last it measures the loss on a fixed
sample of the two held-out splits (the first `--heldout` sequences of each). Writes DATA_DIR/smoke.json: the loss of every step, the held-out losses before and after, the seconds a step. Exit 0 when the mean of the
last quarter of the steps' training losses is below the mean of the first quarter and the held-out loss fell on both splits."""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


def masked_loss(model, x, m):
    logits = model(input_ids=x).logits[:, :-1].float()
    tgt = x[:, 1:]
    w = m[:, 1:].float()
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
    return (ce * w).sum() / w.sum().clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--heldout", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model", default="~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct")
    a = ap.parse_args()
    z = np.load(os.path.join(a.data, "packed.npz"))
    dev = a.device
    d = glob.glob(os.path.join(os.path.expanduser(a.model), "snapshots", "*"))
    model = AutoModelForCausalLM.from_pretrained(d[0] if d else os.path.expanduser(a.model), dtype=torch.float32)
    torch.manual_seed(7)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    warm = max(1, a.steps // 10)
    lr_at = lambda t: a.lr * (t / warm if t <= warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, a.steps - warm))))

    def heldout(split):
        x, m = z[split][:a.heldout], z[split + "_mask"][:a.heldout]
        model.eval()
        tot = []
        with torch.no_grad():
            for i in range(0, len(x), 4):
                tot.append(masked_loss(model, torch.tensor(x[i:i + 4], dtype=torch.long, device=dev), torch.tensor(m[i:i + 4], device=dev)).item())
        model.train()
        return float(np.mean(tot))

    res = {"trainable": sum(p.numel() for p in params), "steps": a.steps, "batch": a.batch, "lr": a.lr, "train_sequences_available": int(z["train"].shape[0])}
    res["heldout_before"] = {s: heldout(s) for s in ("heldout_repos", "heldout_files")}
    print("held-out before:", res["heldout_before"], flush=True)
    losses, secs = [], []
    X, M = z["train"], z["train_mask"]
    model.train()
    for s in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s + 1)
        i = (s * a.batch) % (len(X) - a.batch)
        x, m = torch.tensor(X[i:i + a.batch], dtype=torch.long, device=dev), torch.tensor(M[i:i + a.batch], device=dev)
        t0 = time.perf_counter()
        opt.zero_grad()
        loss = masked_loss(model, x, m)
        loss.backward()
        opt.step()
        if dev == "mps":
            torch.mps.synchronize()
        secs.append(time.perf_counter() - t0)
        losses.append(loss.item())
        print(f"step {s + 1}: loss {losses[-1]:.4f}  {secs[-1]:.1f}s", flush=True)
    res["losses"], res["seconds_per_step"] = losses, float(np.mean(secs[1:] or secs))
    res["heldout_after"] = {s: heldout(s) for s in ("heldout_repos", "heldout_files")}
    q = max(1, a.steps // 4)
    res["first_quarter"], res["last_quarter"] = float(np.mean(losses[:q])), float(np.mean(losses[-q:]))
    print("held-out after:", res["heldout_after"], "first/last quarter:", res["first_quarter"], res["last_quarter"])
    json.dump(res, open(os.path.join(a.data, "smoke.json"), "w"), indent=1)
    ok = res["last_quarter"] < res["first_quarter"] and all(res["heldout_after"][s] < res["heldout_before"][s] for s in res["heldout_after"])
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
