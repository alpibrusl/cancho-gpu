#!/usr/bin/env python3
"""Text from the base and the fine-tuned Qwen2.5-0.5B on held-out prompts (docs/resident.md section 22; roadmap M7.3 (c)).

    python3 examples/qwen/generate.py WORK [--new 32] [--prompt 64]

The adapters WORK/cancho_final.npz are merged into the f16 weights (W + alpha/r B A per projection, rounded to f16, in WORK/merged/), and 8 held-out prompts (the first `--prompt` tokens of the 8 sequences of
the first held-out batch) are continued greedily by the forward session of M7.1: the 8 windows are the 8 sequences of the batch, a token is the arg max the session answers at the last prompt position
(the mask is causal, so the zero padding after it does not matter), and every new token recomputes the window in a new session (the request line cannot carry an embedded batch; section 22). The same is done
by transformers (float32, CPU) on the same merged weights, and the tokens are compared. The held-out loss of the merged model through the forward session is printed against the unmerged adapters'."""
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402
import finetune as F  # noqa: E402
import qwen_ref as R  # noqa: E402
import qwen_session as S  # noqa: E402
import qwen_train as T  # noqa: E402

arg = F.arg


def merge(c, AB, wdir, out, alpha=32.0, r=16):
    """The weight directory `out`: every file of `wdir` (linked) except the q/k/v, o, gate/up and down matrices, which are f16(W + alpha/r B A)."""
    os.makedirs(out, exist_ok=True)
    s = alpha / r
    for f in os.listdir(wdir):
        dst = os.path.join(out, f)
        if os.path.lexists(dst):
            os.remove(dst)
        if not f.endswith((".wqkv.f16", ".wo.f16", ".wgu.f16", ".wd.f16")):
            os.symlink(os.path.abspath(os.path.join(wdir, f)), dst)
    for l in range(c.layers):
        for g, (A, B, _) in T.fuse(c, r, AB, l).items():
            name = {"qkv": "wqkv", "o": "wo", "gu": "wgu", "dn": "wd"}[g]
            w = np.fromfile(os.path.join(wdir, f"l{l:02d}.{name}.f16"), dtype="<f2")
            w = w.reshape(B.shape[0], A.shape[1]).astype(np.float64)
            (w + s * (B @ A)).astype("<f2").tofile(os.path.join(out, f"l{l:02d}.{name}.f16"))


def forward_window(c, wdir, windows, d, binary):
    """(loss of each position, 0-based arg max of each position) of one forward pass over `windows` [8, 512] in a new session."""
    sess = S.write(d, c, wdir, windows)
    ss = S.Session(binary, sess)
    loss, am, _ = ss.forward(c.n)
    ss.close()
    return loss, am


def greedy(c, wdir, prompts, new, d, binary):
    """The `new` tokens that follow each prompt ([8, p]) by arg max, recomputing the window for each."""
    nb, p = prompts.shape
    win = np.zeros((nb, c.t), dtype=np.int64)
    win[:, :p] = prompts
    out = []
    for k in range(new):
        _, am = forward_window(c, wdir, win, d, binary)
        nxt = am.reshape(nb, c.t)[:, p + k - 1]
        win[:, p + k] = nxt
        out.append(nxt)
    return np.stack(out, 1)


def hf_greedy(c, wdir, prompts, new):
    """transformers (float32, CPU) with the q/k/v, o, gate/up, down weights of `wdir` (f16 values) in place of its own: the same continuation by arg max."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(data.snapshot(), torch_dtype=torch.float32)
    f16 = lambda n, shape: torch.tensor(np.fromfile(os.path.join(wdir, n), dtype="<f2").astype(np.float32).reshape(shape))
    d, dkv, ff = c.d, c.dkv, c.ff
    with torch.no_grad():
        for l, blk in enumerate(model.model.layers):
            qkv = f16(f"l{l:02d}.wqkv.f16", (d + 2 * dkv, d))
            blk.self_attn.q_proj.weight.copy_(qkv[:d])
            blk.self_attn.k_proj.weight.copy_(qkv[d:d + dkv])
            blk.self_attn.v_proj.weight.copy_(qkv[d + dkv:])
            blk.self_attn.o_proj.weight.copy_(f16(f"l{l:02d}.wo.f16", (d, d)))
            gu = f16(f"l{l:02d}.wgu.f16", (2 * ff, d))
            blk.mlp.gate_proj.weight.copy_(gu[:ff])
            blk.mlp.up_proj.weight.copy_(gu[ff:])
            blk.mlp.down_proj.weight.copy_(f16(f"l{l:02d}.wd.f16", (d, ff)))
        ids = torch.tensor(prompts)
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=new, do_sample=False)
    return out[:, prompts.shape[1]:].numpy()


def main():
    work = sys.argv[1]
    new, p = int(arg("--new", 32)), int(arg("--prompt", 64))
    c = R.config_of(F.WDIR)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(data.snapshot(), "tokenizer.json"))
    _, held = F.load(work)
    AB = {k: v for k, v in np.load(os.path.join(work, "cancho_final.npz")).items()}
    merged = os.path.join(work, "merged")
    merge(c, AB, F.WDIR, merged)
    d = os.path.join(work, "s_gen")
    res = {}
    for name, wdir in (("base", F.WDIR), ("fine-tuned", merged)):
        t0 = time.time()
        prompts = held[0, :, :p]
        ours = greedy(c, wdir, prompts, new, d, F.METAL)
        t1 = time.time()
        ref = hf_greedy(c, wdir, prompts, new)
        same = (ours == ref)
        first_div = [int(np.argmin(r)) if not r.all() else new for r in same]
        truth = held[0, :, p:p + new]
        res[name] = dict(ours=ours.tolist(), hf=ref.tolist(), first_divergence=first_div, agree=float(same.mean()), vs_truth=float((ours == truth).mean()))
        print(f"{name}: session vs transformers on the same weights: {int(same.all(1).sum())} of 8 continuations identical for {new} tokens, first difference at token {first_div}; "
              f"tokens equal to the held-out text's: {100 * (ours == truth).mean():.1f}% ({t1 - t0:.0f} s for {new} session passes)")
    # the loss of the merged weights through the forward session, against the adapters' (unmerged) session
    for name, wdir in (("base", F.WDIR), ("merged", merged)):
        ls = []
        for b in range(held.shape[0]):
            loss, _ = forward_window(c, wdir, held[b], d, F.METAL)
            ls.append(float(loss.reshape(c.nb, c.t)[:, :-1].mean()))
        res["loss_" + name] = float(np.mean(ls))
        print(f"held-out loss, {name} weights, forward session: {np.mean(ls):.5f}")
    for b in (0, 3):
        print(f"--- prompt {b}: {tok.decode(held[0, b, :p].tolist())[-160:]!r}")
        print(f"    held-out text : {tok.decode(held[0, b, p:p + new].tolist())!r}")
        print(f"    base          : {tok.decode(res['base']['ours'][b])!r}")
        print(f"    fine-tuned    : {tok.decode(res['fine-tuned']['ours'][b])!r}")
    json.dump(res, open(os.path.join(work, "generate.json"), "w"))


if __name__ == "__main__":
    main()
