#!/usr/bin/env python3
"""Serve the GPT that gpt_session.py trains (docs/resident.md section 16; roadmap M2). `train` trains it in a Metal session whose `save` directives write the checkpoint (the f32 weights, `ckpt.bin`; docs/resident.md section 18) and keeps it as an .npz with the
vocabulary; `generate` loads the checkpoint, starts a forward-only Metal session (gpt_session.write(serve=True)) and decodes greedily, checking every token against torch on the same
weights (float32 on MPS, recomputing the window as the session does, and with a key/value cache), and reports tokens a second.

    python3 examples/gpt/serve_gpt.py train    [--steps 1500] [--layers 4] [--d 256] [--heads 4] [--t 128] [--ckpt build/gpt/ckpt.npz]
    python3 examples/gpt/serve_gpt.py generate [--ckpt ...] [--new 96] [--prompt 32]"""
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "examples", "train"))
import corpus  # noqa: E402
import gpt_ref as R  # noqa: E402
import gpt_session as G  # noqa: E402

arg = lambda name, default: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
SESSION = os.environ.get("SESSION", os.path.join(ROOT, "build", "cancho-gpu-metal-session"))
CKPT = arg("--ckpt", os.path.join(ROOT, "build", "gpt", "ckpt.npz"))
DIR = os.path.dirname(CKPT)


def spawn(path):
    return subprocess.Popen([SESSION, path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def train():
    steps, lr = int(arg("--steps", 1500)), 1e-3
    layers, d, heads, t, b = int(arg("--layers", 4)), int(arg("--d", 256)), int(arg("--heads", 4)), int(arg("--t", 128)), int(arg("--b", 32))
    ids, vocab = corpus.load([os.path.join(ROOT, "docs"), os.path.join(ROOT, "..", "cancho", "docs")])
    c = R.Config(v=len(vocab), vp=128, t=t, d=d, h=heads, layers=layers, b=b)
    params = R.init_params(c, 1)
    tok, tgt = corpus.batches(ids[: int(len(ids) * 0.9)], t, b, steps, 1)

    def lr_at(s):
        warm = max(1, steps // 20)
        return lr * s / warm if s <= warm else lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    # the parameters leave the session through `save` (docs/resident.md section 18): the raw f32 weights in CKPT.bin, written by the device program on `checkpoint`
    raw = os.path.splitext(CKPT)[0] + ".bin"
    sess = G.write(os.path.join(DIR, "train"), c, tok, tgt, steps, lr=lr, params=params, lr_fn=lr_at, save=raw)
    p = spawn(sess)
    t0 = time.perf_counter()
    p.stdin.write(f"{steps}\n")
    p.stdin.flush()
    nums = [float(v) for v in p.stdout.readline().split()]
    if not nums:
        raise SystemExit("session: " + p.stderr.read()[:300])
    written = G.checkpoint(p, c)
    p.stdin.close()
    p.wait()
    trained = G.read_checkpoint(c, raw)
    np.savez(CKPT, vocab=np.array(vocab), cfg=np.array([len(vocab), 128, t, d, heads, layers]), **trained)
    print(f"trained {steps} steps in {time.perf_counter() - t0:.0f} s, last batch loss {nums[0]:.3f}; checkpoint {raw} ({written[0]} bytes) and {CKPT}")


def load():
    z = np.load(CKPT)
    v, vp, t, d, h, layers = [int(x) for x in z["cfg"]]
    c = R.Config(v=v, vp=vp, t=t, d=d, h=h, layers=layers, b=1)
    return c, {k: z[k] for k in z.files if k not in ("vocab", "cfg")}, [str(x) for x in z["vocab"]]


def cancho_decoder(c, params):
    sess = G.write(os.path.join(DIR, "serve"), c, np.zeros((1, c.n)), np.zeros((1, c.n)), 1, params=params, serve=True)
    p = spawn(sess)

    def logits(window):
        sel = np.zeros(c.n)
        sel[len(window) - 1] = 1
        full = np.zeros(c.n)
        full[: len(window)] = window
        p.stdin.write(" ".join(map(str, full.astype(int))) + " " + " ".join(map(str, sel.astype(int))) + "\n")
        p.stdin.flush()
        line = p.stdout.readline()
        if not line:
            raise SystemExit("session: " + p.stderr.read()[:300])
        return np.array([float(x) for x in line.split()])

    return logits, p


def torch_decoders(c, params):
    import torch
    dev = "mps"
    P = {k: torch.tensor(np.asarray(v, dtype=np.float32), device=dev) for k, v in params.items()}
    d, h, dh, t = c.d, c.h, c.dh, c.t
    ln = lambda x, g, b: torch.nn.functional.layer_norm(x, (d,), g, b, 1e-5)
    gelu = lambda u: torch.nn.functional.gelu(u, approximate="tanh")

    def window(tokens):
        n = len(tokens)
        tk = torch.tensor(tokens, device=dev)
        x = P["emb"][tk] + P["emb"][torch.arange(n, device=dev) + c.vp]
        mask = torch.triu(torch.full((n, n), -1e9, device=dev), 1)
        for l in range(c.layers):
            a = ln(x, P[f"ln1g{l}"], P[f"ln1b{l}"])
            q, k, v = [z.view(n, h, dh).transpose(0, 1) for z in (a @ P[f"wqkv{l}"].T + P[f"bqkv{l}"]).split(d, 1)]
            o = (torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh) + mask, -1) @ v).transpose(0, 1).reshape(n, d)
            x = x + o @ P[f"wo{l}"].T + P[f"bo{l}"]
            x = x + gelu(ln(x, P[f"ln2g{l}"], P[f"ln2b{l}"]) @ P[f"w1{l}"].T + P[f"b1{l}"]) @ P[f"w2{l}"].T + P[f"b2{l}"]
        return (ln(x, P["lnfg"], P["lnfb"]) @ P["wlm"].T)[-1, : c.v]

    def cached():
        kc = [None] * c.layers
        vc = [None] * c.layers

        def step(tok, pos):
            x = P["emb"][tok] + P["emb"][c.vp + pos]
            x = x.view(1, d)
            for l in range(c.layers):
                a = ln(x, P[f"ln1g{l}"], P[f"ln1b{l}"])
                q, k, v = [z.view(1, h, dh).transpose(0, 1) for z in (a @ P[f"wqkv{l}"].T + P[f"bqkv{l}"]).split(d, 1)]
                kc[l] = k if kc[l] is None else torch.cat([kc[l], k], 1)
                vc[l] = v if vc[l] is None else torch.cat([vc[l], v], 1)
                o = (torch.softmax(q @ kc[l].transpose(-1, -2) / math.sqrt(dh), -1) @ vc[l]).transpose(0, 1).reshape(1, d)
                x = x + o @ P[f"wo{l}"].T + P[f"bo{l}"]
                x = x + gelu(ln(x, P[f"ln2g{l}"], P[f"ln2b{l}"]) @ P[f"w1{l}"].T + P[f"b1{l}"]) @ P[f"w2{l}"].T + P[f"b2{l}"]
            return (ln(x, P["lnfg"], P["lnfb"]) @ P["wlm"].T)[0, : c.v]

        return step

    return window, cached, torch


def generate():
    c, params, vocab = load()
    new, plen = int(arg("--new", 96)), int(arg("--prompt", 32))
    ids, _ = corpus.load([os.path.join(ROOT, "docs"), os.path.join(ROOT, "..", "cancho", "docs")])
    ids = ids[int(len(ids) * 0.9):]  # held-out text for the prompts
    logits, p = cancho_decoder(c, params)
    window, cached, torch = torch_decoders(c, params)
    sync = torch.mps.synchronize
    rows = []
    for k in range(6):
        prompt = [int(x) for x in ids[k * 997: k * 997 + plen]]
        # cancho: the window is recomputed for every token
        seq, t0 = list(prompt), time.perf_counter()
        for _ in range(new):
            seq.append(int(np.argmax(logits(seq)[: c.v])))
        tc = time.perf_counter() - t0
        # torch, the same algorithm (the window recomputed)
        ref, t0 = list(prompt), time.perf_counter()
        for _ in range(new):
            ref.append(int(torch.argmax(window(ref)).item()))
        tw = time.perf_counter() - t0
        # torch with a key/value cache
        step, kv, t0 = cached(), list(prompt), time.perf_counter()
        for i, tk in enumerate(prompt[:-1]):
            step(tk, i)
        last = prompt[-1]
        for i in range(new):
            nxt = int(torch.argmax(step(last, plen - 1 + i)).item())
            kv.append(nxt)
            last = nxt
        tk_ = time.perf_counter() - t0
        # teacher-forced: cancho's choice at each position of torch's own continuation (no divergence to cascade)
        agree = sum(int(np.argmax(logits(ref[: plen + i])[: c.v])) == ref[plen + i] for i in range(new))
        first = next((i for i in range(new) if seq[plen + i] != ref[plen + i]), new)
        rows.append((tc, tw, tk_, first, agree, seq == ref, ref == kv))
        print(f"prompt {k}: cancho {new / tc:.0f} tok/s, torch recompute {new / tw:.0f}, torch KV cache {new / tk_:.0f}; identical to torch for the first {first} of {new} tokens, "
              f"teacher-forced agreement {agree}/{new}; torch recompute == torch cache: {ref == kv}", flush=True)
        if k == 0:
            print("  cancho:", repr("".join(vocab[i] for i in seq[plen:])))
            print("  torch :", repr("".join(vocab[i] for i in ref[plen:])))
    p.stdin.close()
    p.wait()
    n = len(rows)
    print(f"\nmean tokens/s: cancho {new * n / sum(r[0] for r in rows):.0f}, torch recompute {new * n / sum(r[1] for r in rows):.0f}, torch KV cache {new * n / sum(r[2] for r in rows):.0f}; "
          f"teacher-forced agreement {sum(r[4] for r in rows)}/{new * n}; fully identical continuations {sum(r[5] for r in rows)}/{n}")


{"train": train, "generate": generate}[sys.argv[1]]()
