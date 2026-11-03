#!/usr/bin/env python3
"""Where the layer-0 q B of the 40-step run of docs/resident.md section 21.2 differs from peft's (roadmap M7.3 (a); docs/resident.md section 22.4).

    python3 examples/qwen/adam_noise.py WORK [--steps 40] [--lr 2e-4] [--batches 8]

The same short AdamW run (batches taken round and round from the first `--batches` of the corpus, linear warm-up over steps / 8, cosine, lr) in peft (float32, CPU) and in the session, with the gradient of
a few adapter matrices recorded at every step in both: peft's by a hook before `opt.step()`, the session's by `save` and `checkpoint` after each step. Then, in float64:
  * AdamW fed peft's gradients from the same start against peft's own result (does torch's float32 Adam have the noise?);
  * AdamW fed the session's gradients against the session's result (does `adamw_l` have it?);
  * AdamW fed one run's gradients against AdamW fed the other's (what the gradients' difference alone does);
  * the row-wise error of the session's matrix, and the size of the gradients there against Adam's epsilon;
and the held-out loss of peft's adapters with the session's matrix put in place of peft's."""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import finetune as F  # noqa: E402
import peft_reference as PR  # noqa: E402
import qwen_ref as R  # noqa: E402
import qwen_train as T  # noqa: E402

arg = F.arg
KEYS = ["l00.q.B", "l00.q.A", "l08.q.B", "l12.q.B"]


def adam64(gs, p0, lr_fn, b1=0.9, b2=0.999, eps=1e-8):
    p, m, v = p0.copy(), np.zeros_like(p0), np.zeros_like(p0)
    for t, g in enumerate(gs, 1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        p = p - lr_fn(t) * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return p


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def cos(a, b):
    return float(a.ravel() @ b.ravel() / np.linalg.norm(a) / np.linalg.norm(b))


def peft_run(c, AB, batches, steps, lr, lr_fn):
    model, handles = PR.build(c, AB)
    opt = torch.optim.AdamW(list(handles.values()), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    gs = {k: [] for k in KEYS}
    losses = []
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_fn(s + 1)
        ids = torch.tensor(batches[s % len(batches)])
        opt.zero_grad()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        for k in KEYS:
            gs[k].append(handles[k].grad.detach().numpy().astype(np.float64).copy())
        opt.step()
        losses.append(loss.item())
        print(f"peft step {s + 1}: {losses[-1]:.5f}", flush=True)
    return gs, {k: p.detach().numpy().astype(np.float64) for k, p in handles.items()}, losses


def session_run(c, AB, batches, steps, lr, lr_fn, d):
    sess = T.write(d, c, F.WDIR, np.stack([batches[s % len(batches)] for s in range(steps)]), AB, r=16, loss_scale=16.0, train=True, steps=steps, lr=lr, lr_fn=lr_fn, save=os.path.join(d, "ad.bin"),
                   grads=os.path.join(d, "grads.bin"))
    p = F.spawn(sess)
    ncount = float(c.nb * (c.t - 1))
    gs = {k: [] for k in KEYS}
    losses = []
    for s in range(steps):
        losses.append(float(F.ask(p, "1")[0].split()[0]))
        F.ask(p, "checkpoint")
        g = T.split_grads(c, 16, T.read_fused(c, 16, os.path.join(d, "grads.bin")), ncount, 16.0)
        for k in KEYS:
            gs[k].append(g[k])
    F.close(p)
    return gs, F.split_fused(c, T.read_fused(c, 16, os.path.join(d, "ad.bin"))), losses


def heldout_peft(c, held, final, swaps):
    """Held-out loss (float32, MPS) of peft's final adapters, with the matrices of `swaps` ({name: array}) put in place of its own."""
    AB = dict(final)
    AB.update(swaps)
    model, _ = PR.build(c, AB, device="mps")
    tot = []
    with torch.no_grad():
        for b in held:
            ids = torch.tensor(b, device="mps")
            tot.append(model(input_ids=ids, labels=ids).loss.item())
    del model
    return float(np.mean(tot))


def main():
    work = sys.argv[1]
    steps, lr, k = int(arg("--steps", 40)), float(arg("--lr", 2e-4)), int(arg("--batches", 8))
    c = R.config_of(F.WDIR)
    train, held = F.load(work)
    batches = train[:k]
    warm = max(1, steps // 8)
    lr_fn = lambda t: lr * (t / warm if t <= warm else 0.5 * (1 + np.cos(np.pi * (t - warm) / max(1, steps - warm))))
    AB = F.initial_adapters(c)
    cache = os.path.join(work, f"adam_noise_{steps}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        gp, gs, fp, fs = [z[x].item() for x in ("gp", "gs", "fp", "fs")]
    else:
        torch.set_num_threads(os.cpu_count())
        gs, fs, ls = session_run(c, AB, batches, steps, lr, lr_fn, os.path.join(work, "s_noise"))
        gp, fp, lp = peft_run(c, AB, batches, steps, lr, lr_fn)
        print(f"largest |loss difference| over the {steps} steps: {np.abs(np.array(ls) - np.array(lp)).max():.2e}")
        np.savez(cache, gp=gp, gs=gs, fp=fp, fs=fs)
    # the matrices' final distance from peft's
    cs = {n: cos(fs[n], fp[n]) for n in fp}
    rs = {n: rel(fs[n], fp[n]) for n in fp}
    worst = sorted(rs, key=rs.get)[-5:][::-1]
    print("largest relative norm errors of the session's adapters against peft's after the run: " + ", ".join(f"{n} {rs[n]:.3f} (cos {cs[n]:.4f})" for n in worst))
    for n in KEYS:
        B0 = AB[n]
        a_peft = adam64(gp[n], B0, lr_fn)
        a_ses = adam64(gs[n], B0, lr_fn)
        print(f"\n== {n}: {B0.shape}")
        print(f"  peft's final against float64 AdamW on peft's gradients      : rel {rel(fp[n], a_peft):.2e}")
        print(f"  session's final against float64 AdamW on its own gradients  : rel {rel(fs[n], a_ses):.2e}")
        print(f"  float64 AdamW on session's gradients against on peft's      : rel {rel(a_ses, a_peft):.2e}")
        print(f"  session's final against peft's                              : rel {rel(fs[n], fp[n]):.2e}, cos {cos(fs[n], fp[n]):.5f}")
        g1p, g1s = gp[n][0], gs[n][0]
        print(f"  step-1 gradient (same weights): session against peft rel {rel(g1s, g1p):.2e}, cos {cos(g1s, g1p):.5f}; |g| rms {np.sqrt((g1p ** 2).mean()):.2e}, epsilon 1e-8")
        if n.endswith(".B"):
            rowg = np.sqrt((g1p ** 2).mean(1))
            rowe = np.linalg.norm(fs[n] - fp[n], axis=1) / (np.linalg.norm(fp[n], axis=1) + 1e-30)
            small = rowg < 1e-3 * rowg.max()
            print(f"  rows: {small.sum()} of {len(rowg)} have a step-1 gradient under 1e-3 of the largest row's (rms {rowg[small].max() if small.any() else 0:.1e} at most; those rows' mean relative error "
                  f"{rowe[small].mean() if small.any() else 0:.2f}, the others' {rowe[~small].mean():.2f}); rows with the 8 largest errors: {sorted(np.argsort(rowe)[-8:].tolist())}")
            gp_all = np.stack(gp[n])
            gs_all = np.stack(gs[n])
            row_noise = np.sqrt(((gs_all - gp_all) ** 2).mean((0, 2)))
            row_sig = np.sqrt((gp_all ** 2).mean((0, 2)))
            print(f"  over the run, per row: gradient rms {np.median(row_sig):.1e} (median), difference session - peft rms {np.median(row_noise):.1e}; the rows with the 8 largest errors have signal {np.sort(row_sig[np.argsort(rowe)[-8:]])[::-1][:3]} "
                  f"and difference {np.sort(row_noise[np.argsort(rowe)[-8:]])[::-1][:3]}")
    base = heldout_peft(c, held, fp, {})
    print(f"\nheld-out loss (float32 MPS) of peft's adapters after {steps} steps: {base:.5f}")
    for what, sw in (("layer 0 q B from the session", {"l00.q.B": fs["l00.q.B"]}), ("layers 0, 8 and 12 q B from the session", {n: fs[n] for n in ("l00.q.B", "l08.q.B", "l12.q.B")}), ("every matrix from the session", fs)):
        v = heldout_peft(c, held, fp, sw)
        print(f"  with {what}: {v:.5f} ({v - base:+.5f})")
    # the initial loss, for scale
    v0 = heldout_peft(c, held, AB, {})
    print(f"  (before training: {v0:.5f})")


if __name__ == "__main__":
    main()
