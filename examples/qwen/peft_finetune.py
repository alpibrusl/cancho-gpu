#!/usr/bin/env python3
"""peft's side of the 300-step fine-tuning run of roadmap M7.3 (docs/resident.md section 22): the same batches (finetune_data.py), the same initial adapters and the same AdamW and schedule as finetune.py,
on MPS in float16 autocast or float32.

    python3 examples/qwen/peft_finetune.py WORK --dtype float16|float32 [--steps 300] [--lr 2e-4] [--every 30] [--loss-scale 1] [--tag NAME]      # WORK/peft_<dtype>.json, WORK/peft_<dtype>_final.npz

With `--loss-scale S` the loss is multiplied by S before the backward pass and the gradients divided by S before the optimiser (a static scale, as the session has), and the files are named `peft_<dtype>_<tag>`.

Records the loss of every step, the time of every step (forward, backward and optimiser, synchronised; the held-out evaluations are not timed), the held-out loss every `--every` steps (and before the first)
and the driver's allocation; run it under `/usr/bin/time -l` for the process's peak footprint, the number finetune.py reports for the session."""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import finetune as F  # noqa: E402
import peft_reference as PR  # noqa: E402
import qwen_ref as R  # noqa: E402

arg = F.arg


def main():
    work = sys.argv[1]
    dtn = arg("--dtype", "float16")
    dt = getattr(torch, dtn)
    steps, lr, every = int(arg("--steps", 300)), float(arg("--lr", 2e-4)), int(arg("--every", 30))
    scale = float(arg("--loss-scale", 1.0))
    tag = dtn + (("_" + arg("--tag", "scaled")) if scale != 1.0 else "")
    c = R.config_of(F.WDIR)
    batches, held = F.load(work)
    lr_fn = F.lr_schedule(steps, lr)
    model, handles = PR.build(c, F.initial_adapters(c), device="mps", dtype=dt)
    opt = torch.optim.AdamW(list(handles.values()), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    ac = lambda: torch.autocast("mps", dtype=dt, enabled=dt != torch.float32)

    def evaluate():
        tot = []
        with torch.no_grad(), ac():
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
        with ac():
            loss = model(input_ids=ids, labels=ids).loss
        (loss * scale).backward()
        if scale != 1.0:
            for p in handles.values():
                p.grad /= scale
        opt.step()
        torch.mps.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
        losses.append(loss.item())
        if s < every:
            peak = max(peak, torch.mps.driver_allocated_memory())
        if (s + 1) % every == 0:
            heldout[s + 1] = evaluate()
            print(f"step {s + 1}: loss {losses[-1]:.5f}, {np.median(ms[-every:]):.0f} ms, held-out {heldout[s + 1]:.5f}", flush=True)
    tms = ms[2:]
    res = dict(losses=losses, ms=ms, median_ms=float(np.median(tms)), mean_ms=float(np.mean(tms)), peak_bytes=int(peak), driver_final_bytes=int(torch.mps.driver_allocated_memory()), heldout=heldout, steps=steps,
               lr=lr)
    json.dump(res, open(os.path.join(work, f"peft_{tag}.json"), "w"))
    np.savez(os.path.join(work, f"peft_{tag}_final.npz"), **{k: p.detach().float().cpu().numpy().astype(np.float64) for k, p in handles.items()})
    print(f"peft {tag}: {steps} steps, median {res['median_ms']:.1f} ms a step ({c.n / res['median_ms'] * 1000:.0f} tokens a second), driver memory during the first {every} steps {peak / 1e9:.2f} GB, loss {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
