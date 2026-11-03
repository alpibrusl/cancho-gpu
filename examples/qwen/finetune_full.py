#!/usr/bin/env python3
"""The full fine-tuning run of roadmap M7.4 part B (docs/resident.md section 24): every parameter of Qwen2.5-0.5B trained by AdamW in one Metal session (qwen_full.py), against torch (torch_full.py).

    python3 examples/qwen/finetune_full.py WORK train [--steps 20] [--lr 1e-5] [--every 10]     # WORK/cancho_full.json, checkpoints WORK/ckf/stepNN/ (weights directories)
    python3 examples/qwen/finetune_full.py WORK eval                                          # held-out loss of the starting weights and of each checkpoint
    python3 examples/qwen/finetune_full.py WORK grads                                         # the gradient of every parameter at step 1 against torch's (torch_full.py --grads)
    python3 examples/qwen/finetune_full.py WORK compare                                       # the curves side by side

The corpus is WORK/corpus.npz (finetune_data.py). The whole run is one session: a step's ids are an f32 table of 4,096 numbers, there is no 512 MiB table of embeddings to hand over."""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import finetune as F  # noqa: E402
import qwen_full as Q  # noqa: E402
import qwen_ref as R  # noqa: E402
import qwen_session as S  # noqa: E402
import qwen_train as T  # noqa: E402

arg = F.arg


def train(work, c):
    steps, lr, every = int(arg("--steps", 20)), float(arg("--lr", 1e-5)), int(arg("--every", 10))
    batches, _ = F.load(work)
    d = os.path.join(work, "s_full")
    ck = os.path.join(d, "ck")
    os.makedirs(ck, exist_ok=True)
    sess = Q.write(d, c, F.WDIR, batches[:steps], steps=steps, lr=lr, lr_fn=F.lr_schedule(steps, lr), loss_scale=float(arg("--loss-scale", 16.0)), save=ck)
    p = F.spawn(sess)
    losses, times = [], []
    for s in range(steps):
        out, secs = F.ask(p, "1")
        losses.append(float(out.split()[0]))
        times.append(secs * 1000 if s > 0 else None)
        if (s + 1) % every == 0 or s + 1 == steps:
            F.ask(p, "checkpoint")
            dst = os.path.join(work, "ckf", f"step{s + 1:03d}")
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(ck, dst)
            shutil.copy(os.path.join(F.WDIR, "manifest.json"), dst)
        print(f"step {s + 1}: loss {losses[-1]:.5f}, {times[-1] or 0:.0f} ms", flush=True)
    peak = F.close(p)
    ts = [t for t in times if t is not None]
    res = dict(losses=losses, ms=times, median_ms=float(np.median(ts)), mean_ms=float(np.mean(ts)), peak_bytes=peak, steps=steps, lr=lr)
    json.dump(res, open(os.path.join(work, "cancho_full.json"), "w"))
    print(f"{steps} steps: median {res['median_ms']:.1f} ms a step ({c.n / res['median_ms'] * 1000:.0f} tokens a second), peak footprint {peak / 1e9:.2f} GB, loss {losses[0]:.4f} -> {losses[-1]:.4f}")


def heldout(work, c, wdir, tag):
    """Mean held-out loss of the weights directory `wdir`: the forward and backward programs of the LoRA session with adapters that are zero (B = 0), four requests of one batch each."""
    _, held = F.load(work)
    AB = R.init_lora(c, 16, 7, b_std=0.0)
    d = os.path.join(work, "s_eval_full")
    sess = T.write(d, c, wdir, held, AB, r=16, loss_scale=16.0, train=False, steps=1)
    p = F.spawn(sess)
    ls = [float(F.ask(p, "1")[0].split()[0]) for _ in range(len(held))]
    F.close(p)
    return float(np.mean(ls))


def evaluate(work, c):
    res = json.load(open(os.path.join(work, "cancho_full.json")))
    ev = {0: heldout(work, c, F.WDIR, "start")}
    print(f"step 0: held-out {ev[0]:.5f}", flush=True)
    for f in sorted(os.listdir(os.path.join(work, "ckf"))):
        step = int(f[4:])
        ev[step] = heldout(work, c, os.path.join(work, "ckf", f), f)
        print(f"step {step}: held-out {ev[step]:.5f}", flush=True)
    res["heldout"] = ev
    json.dump(res, open(os.path.join(work, "cancho_full.json"), "w"))


def grads(work, c):
    """One step with the raw gradients saved, against torch's at step 1."""
    batches, _ = F.load(work)
    d = os.path.join(work, "s_gfull")
    gp = os.path.join(d, "grads.bin")
    ls = 16.0
    sess = Q.write(d, c, F.WDIR, batches[:1], steps=1, lr=1e-5, loss_scale=ls, grads=gp, reuse=False)
    p = F.spawn(sess)
    F.ask(p, "1")
    F.ask(p, "checkpoint")
    F.close(p)
    bc = min(2304, -(-c.v // 128) * 128)
    vp = -(-c.v // bc) * bc
    G = Q.read_grads(c, gp, ls, float(c.nb * (c.t - 1)), vp)
    ref = np.load(os.path.join(work, "torch_grads.npz"))
    worst = {}
    for k in ref.files:
        a, b = G[k].ravel(), ref[k].ravel()
        na, nb_ = np.linalg.norm(a), np.linalg.norm(b)
        cos = float(a @ b / (na * nb_)) if na > 0 and nb_ > 0 else (1.0 if na == nb_ else 0.0)
        rel = float(np.linalg.norm(a - b) / nb_) if nb_ > 0 else float(na)
        worst[k] = (cos, rel, nb_)
    kinds = {}
    for k, v in worst.items():
        kinds.setdefault(k.split(".")[-1], []).append((k, v))
    print(f"{'tensor kind':10s} {'smallest cosine':>16s} {'largest rel. norm error':>24s}")
    for kind, lst in kinds.items():
        mc = min(lst, key=lambda x: x[1][0])
        mr = max(lst, key=lambda x: x[1][1])
        print(f"{kind:10s} {mc[1][0]:16.6f} ({mc[0]}) {mr[1][1]:12.2e} ({mr[0]})")
    print(f"{len(worst)} tensors: smallest cosine {min(v[0] for v in worst.values()):.6f}, largest relative norm error {max(v[1] for v in worst.values()):.2e}")


def compare(work):
    a, b = json.load(open(os.path.join(work, "cancho_full.json"))), json.load(open(os.path.join(work, "torch_full.json")))
    la, lb = np.array(a["losses"]), np.array(b["losses"])
    print(f"{'step':>4s} {'cancho-gpu':>11s} {'torch f32':>11s} {'diff':>10s}")
    for s in range(len(la)):
        print(f"{s + 1:4d} {la[s]:11.5f} {lb[s]:11.5f} {la[s] - lb[s]:+10.5f}")
    print(f"largest |difference| of the training loss {np.abs(la - lb).max():.5f} (mean {np.abs(la - lb).mean():.5f}, step {int(np.abs(la - lb).argmax()) + 1})")
    for s in sorted(int(k) for k in a.get("heldout", {})):
        ha, hb = a["heldout"][str(s)] if str(s) in a["heldout"] else a["heldout"][s], b["heldout"][str(s)] if str(s) in b["heldout"] else b["heldout"][s]
        print(f"held-out step {s}: cancho-gpu {ha:.5f}, torch {hb:.5f}, difference {ha - hb:+.5f}")
    for n, r in (("cancho-gpu", a), ("torch float32", b)):
        print(f"{n}: median {r['median_ms']:.1f} ms a step, {4096 / r['median_ms'] * 1000:.0f} tokens a second, peak {r.get('peak_bytes', 0) / 1e9:.2f} GB")


if __name__ == "__main__":
    work, cmd = sys.argv[1], sys.argv[2]
    cfg = R.config_of(F.WDIR)
    {"train": lambda: train(work, cfg), "eval": lambda: evaluate(work, cfg), "grads": lambda: grads(work, cfg), "compare": lambda: compare(work)}[cmd]()
