#!/usr/bin/env python3
"""torch's side of the full fine-tuning run of roadmap M7.4 part B (docs/resident.md section 24): the same batches (finetune_data.py), the same starting weights (the f16 values of the converted weights,
as float32 in torch), the same AdamW and schedule as finetune_full.py, on MPS in float32.

    python3 examples/qwen/torch_full.py WORK [--steps 20] [--lr 1e-5] [--every 10] [--grads]      # WORK/torch_full.json

With --grads the gradient of every parameter at step 1 is saved (WORK/torch_grads.npz) for the comparison of finetune_full.py. Records the loss and the time of every step (forward, backward and
optimiser, synchronised), the held-out loss every `--every` steps and before the first, and the driver's allocation; run it under `/usr/bin/time -l` for the process's peak footprint."""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402
import finetune as F  # noqa: E402
import qwen_ref as R  # noqa: E402

arg = F.arg


def load_weights(model, c, wdir):
    """Sets the model's parameters to the converted weights (the f16 values, which the cancho-gpu session starts from too)."""
    P = R.load_params(wdir, c)
    sd = dict(model.named_parameters())
    put = lambda name, a: sd[name].data.copy_(torch.tensor(a, dtype=torch.float32))
    put("model.embed_tokens.weight", P["emb"])
    put("model.norm.weight", P["norm"])
    for l in range(c.layers):
        q, a, n = f"model.layers.{l}.", f"model.layers.{l}.self_attn.", f"l{l:02d}."
        put(q + "input_layernorm.weight", P[n + "ln1"])
        put(q + "post_attention_layernorm.weight", P[n + "ln2"])
        w, b = P[n + "wqkv"], P[n + "bqkv"]
        put(a + "q_proj.weight", w[:c.d])
        put(a + "k_proj.weight", w[c.d:c.d + c.dkv])
        put(a + "v_proj.weight", w[c.d + c.dkv:])
        put(a + "q_proj.bias", b[:c.d])
        put(a + "k_proj.bias", b[c.d:c.d + c.dkv])
        put(a + "v_proj.bias", b[c.d + c.dkv:])
        put(a + "o_proj.weight", P[n + "wo"])
        put(q + "mlp.gate_proj.weight", P[n + "wgu"][:c.ff])
        put(q + "mlp.up_proj.weight", P[n + "wgu"][c.ff:])
        put(q + "mlp.down_proj.weight", P[n + "wd"])
    return sd


def our_names(c, sd):
    """{our parameter name: array of the torch parameters} for the gradient comparison."""
    out = {"emb": sd["model.embed_tokens.weight"].grad, "norm": sd["model.norm.weight"].grad}
    for l in range(c.layers):
        q, a, n = f"model.layers.{l}.", f"model.layers.{l}.self_attn.", f"l{l:02d}."
        g = lambda k: sd[k].grad
        out[n + "ln1"], out[n + "ln2"] = g(q + "input_layernorm.weight"), g(q + "post_attention_layernorm.weight")
        out[n + "wqkv"] = torch.cat([g(a + "q_proj.weight"), g(a + "k_proj.weight"), g(a + "v_proj.weight")])
        out[n + "bqkv"] = torch.cat([g(a + "q_proj.bias"), g(a + "k_proj.bias"), g(a + "v_proj.bias")])
        out[n + "wo"] = g(a + "o_proj.weight")
        out[n + "wgu"] = torch.cat([g(q + "mlp.gate_proj.weight"), g(q + "mlp.up_proj.weight")])
        out[n + "wd"] = g(q + "mlp.down_proj.weight")
    return out


def main():
    from transformers import AutoModelForCausalLM
    work = sys.argv[1]
    steps, lr, every = int(arg("--steps", 20)), float(arg("--lr", 1e-5)), int(arg("--every", 10))
    c = R.config_of(F.WDIR)
    batches, held = F.load(work)
    lr_fn = F.lr_schedule(steps, lr)
    model = AutoModelForCausalLM.from_pretrained(data.snapshot(), dtype=torch.float32).to("mps")
    sd = load_weights(model, c, F.WDIR)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    def evaluate():
        tot = []
        with torch.no_grad():
            for b in held:
                ids = torch.tensor(b, device="mps")
                tot.append(model(input_ids=ids, labels=ids).loss.item())
        return float(np.mean(tot))

    heldout = {0: evaluate()}
    print(f"step 0: held-out {heldout[0]:.5f}", flush=True)
    losses, ms, peak = [], [], 0
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_fn(s + 1)
        ids = torch.tensor(batches[s], device="mps")
        torch.mps.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        if s == 0 and "--grads" in sys.argv:
            np.savez(os.path.join(work, "torch_grads.npz"), **{k: v.detach().cpu().numpy().astype(np.float64) for k, v in our_names(c, sd).items()})
        opt.step()
        torch.mps.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
        losses.append(loss.item())
        if s < 3:
            peak = max(peak, torch.mps.driver_allocated_memory())
        if (s + 1) % every == 0 or s + 1 == steps:
            heldout[s + 1] = evaluate()
            print(f"step {s + 1}: loss {losses[-1]:.5f}, {np.median(ms[-every:]):.0f} ms, held-out {heldout[s + 1]:.5f}", flush=True)
    tms = ms[2:]
    res = dict(losses=losses, ms=ms, median_ms=float(np.median(tms)), mean_ms=float(np.mean(tms)), peak_bytes=int(peak), driver_final_bytes=int(torch.mps.driver_allocated_memory()), heldout=heldout, steps=steps, lr=lr)
    json.dump(res, open(os.path.join(work, "torch_full.json"), "w"))
    print(f"torch float32 full fine-tune: {steps} steps, median {res['median_ms']:.1f} ms a step ({c.n / res['median_ms'] * 1000:.0f} tokens a second), driver memory {peak / 1e9:.2f} GB, loss {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
