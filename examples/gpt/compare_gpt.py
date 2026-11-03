#!/usr/bin/env python3
"""Train the same GPT on the same text with cancho-gpu (a Metal session) and torch (MPS, float32 and float16 autocast; CPU with --cpu), and compare the loss curves and the time
(docs/resident.md sections 15, 17, 18). The text is the Markdown documentation of two directories (no download), characters as tokens; the model, its start (gpt_ref.init_params), the
batches and AdamW are the same for both.

    python3 examples/gpt/compare_gpt.py [--steps 1500] [--layers 4] [--d 256] [--heads 4] [--t 128] [--b 32] [--lr 1e-3] [--only cancho,torch-mps,torch-mps-amp]"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "examples", "train"))
import bpe  # noqa: E402
import corpus  # noqa: E402
import gpt_ref as R  # noqa: E402
import gpt_session as G  # noqa: E402

arg = lambda name, default: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
STEPS = int(arg("--steps", 1500))
LR = float(arg("--lr", 1e-3))
WD = 0.1
ONLY = arg("--only", "cancho,torch-mps,torch-mps-amp").split(",")
SESSION = os.environ.get("SESSION", os.path.join(ROOT, "build", "cancho-gpu-metal-session"))
DIR = arg("--dir", os.path.join(ROOT, "build", "gpt"))
EVAL_EVERY = int(arg("--eval", 250))


def lr_at(step):
    """Linear warm-up for 5% of the steps, then cosine to 10% of the peak."""
    warm = max(1, STEPS // 20)
    if step <= warm:
        return LR * step / warm
    prog = (step - warm) / max(1, STEPS - warm)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


def val_loss(c, params, val_x, val_y):
    return float(np.mean([R.forward(c, params, x, y) for x, y in zip(val_x, val_y)]))


def run_cancho(c, params, tok, tgt, val_x, val_y):
    import subprocess
    # One session that answers only the loss. The time is the requests that run steps; at each stretch's end a `checkpoint` request (not timed) has the device program write the
    # parameters to a file (`save`, docs/resident.md section 18) and the validation loss is the float64 forward of what that file holds.
    ckpt = os.path.join(DIR, "cancho-ckpt.bin")
    if os.path.exists(ckpt):
        os.remove(ckpt)
    sess = G.write(os.path.join(DIR, "cancho"), c, tok, tgt, STEPS, lr=LR, wd=WD, params=params, lr_fn=lr_at, save=ckpt)
    p = subprocess.Popen([SESSION, sess], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    p.stdin.write("1\n")
    p.stdin.flush()
    first = p.stdout.readline()
    if not first:
        raise SystemExit("cancho session: " + p.stderr.read()[:300])
    curve, secs, done = [(1, float(first.split()[0]), None)], 0.0, 1
    while done < STEPS:
        k = min(EVAL_EVERY, STEPS - done)
        t0 = time.perf_counter()
        p.stdin.write(f"{k}\n")
        p.stdin.flush()
        loss = float(p.stdout.readline().split()[0])
        secs += time.perf_counter() - t0
        done += k
        G.checkpoint(p, c)
        curve.append((done, loss, val_loss(c, G.read_checkpoint(c, ckpt), val_x, val_y)))
    p.stdin.close()
    p.wait()
    return secs / (STEPS - 1) * 1000, curve


def run_torch(c, params, tok, tgt, val_x, val_y, device, amp):
    import torch
    mats, vecs = G.names(c)
    P = {k: torch.tensor(np.asarray(v, dtype=np.float32), device=device, requires_grad=True) for k, v in params.items()}
    decay = [P[k] for k in mats]
    nodecay = [P[k] for k in vecs]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": nodecay, "weight_decay": 0.0}], lr=LR, betas=(G.B1, G.B2), eps=G.EPS)
    n, d, h, dh, t, nb = c.n, c.d, c.h, c.dh, c.t, c.b
    mask = torch.triu(torch.full((t, t), -1e9, device=device), 1)
    tokt = torch.tensor(tok, device=device)
    tgtt = torch.tensor(tgt, device=device)
    pos = torch.arange(n, device=device) % t + c.vp

    def ln(x, g, b):
        return torch.nn.functional.layer_norm(x, (d,), g, b, 1e-5)

    def forward(tk, tg):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            x = P["emb"][tk] + P["emb"][pos]
            for l in range(c.layers):
                a = ln(x, P[f"ln1g{l}"], P[f"ln1b{l}"])
                qkv = a @ P[f"wqkv{l}"].T + P[f"bqkv{l}"]
                q, k, v = [z.view(nb, t, h, dh).transpose(1, 2) for z in qkv.split(d, 1)]
                s = (q @ k.transpose(-1, -2)) / math.sqrt(dh) + mask
                o = (torch.softmax(s.float(), -1).to(v.dtype) @ v).transpose(1, 2).reshape(n, d)
                x = x + o @ P[f"wo{l}"].T + P[f"bo{l}"]
                a = ln(x, P[f"ln2g{l}"], P[f"ln2b{l}"])
                u = a @ P[f"w1{l}"].T + P[f"b1{l}"]
                x = x + torch.nn.functional.gelu(u, approximate="tanh") @ P[f"w2{l}"].T + P[f"b2{l}"]
            logits = ln(x, P["lnfg"], P["lnfb"]) @ P["wlm"].T
        logits = logits.float()
        logits[:, c.v:] = -1e9
        return torch.nn.functional.cross_entropy(logits, tg)

    sync = torch.mps.synchronize if device == "mps" else (lambda: None)
    curve, secs = [], 0.0
    for step in range(1, STEPS + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        t0 = time.perf_counter()
        loss = forward(tokt[step - 1], tgtt[step - 1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            sync()
        if step > 1:
            secs += time.perf_counter() - t0
        if step == 1 or step % EVAL_EVERY == 0:
            curve.append((step, float(loss.detach()), val_loss(c, {k: v.detach().cpu().numpy().astype(np.float64) for k, v in P.items()}, val_x, val_y)))
    sync()
    return secs / (STEPS - 1) * 1000, curve


def main():
    layers, d, heads, t, b = int(arg("--layers", 4)), int(arg("--d", 256)), int(arg("--heads", 4)), int(arg("--t", 128)), int(arg("--b", 32))
    nbpe = int(arg("--bpe", 0))  # 0: characters of the Markdown docs; N: a BPE vocabulary of N tokens over the Markdown, Rust and cancho files on this machine
    if nbpe:
        ids, vocab = bpe.corpus_ids(nbpe)
    else:
        ids, vocab = corpus.load([os.path.join(ROOT, "docs"), os.path.join(ROOT, "..", "cancho", "docs")])
    cut = int(len(ids) * 0.9)
    c = R.Config(v=len(vocab), vp=nbpe or 128, t=t, d=d, h=heads, layers=layers, b=b)
    params = R.init_params(c, 1)
    tok, tgt = corpus.batches(ids[:cut], t, b, STEPS, 1)
    vx, vy = corpus.batches(ids[cut:], t, b, 4, 99)
    val_x, val_y = list(vx), list(vy)
    nparams = sum(int(np.prod(v.shape)) for v in params.values())
    print(f"GPT: {layers} layers, d {d}, {heads} heads, context {t}, batch {b} ({b * t} tokens a step), {nparams / 1e6:.2f} M parameters; {len(ids)} {"tokens" if nbpe else "characters"} ({len(vocab)} distinct), {STEPS} steps, AdamW {LR}", flush=True)
    rows = []
    for name in ONLY:
        os.makedirs(DIR, exist_ok=True)
        if name == "cancho":
            ms, curve = run_cancho(c, params, tok, tgt, val_x, val_y)
        elif name.startswith("torch"):
            ms, curve = run_torch(c, params, tok, tgt, val_x, val_y, "cpu" if "cpu" in name else "mps", name.endswith("amp"))
        else:
            continue
        rows.append((name, ms, curve))
        print(f"{name}: {ms:.1f} ms a step ({b * t / ms * 1000:.0f} tokens/s); (step, train loss of that step, validation loss): " + " ".join(f"({s}, {tl:.3f}, {'-' if vl is None else f'{vl:.3f}'})" for s, tl, vl in curve), flush=True)
    print("\n| runtime | ms a step | tokens a second | training time | validation loss at the end |\n|---|---|---|---|---|")
    for name, ms, curve in rows:
        print(f"| {name} | {ms:.1f} | {b * t / ms * 1000:.0f} | {ms * STEPS / 1000:.0f} s | {curve[-1][2]:.3f} |")


main()
