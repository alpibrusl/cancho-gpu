#!/usr/bin/env python3
"""transformers' answers for the fixed batch (docs/resident.md section 20): the loss (labels = input ids, float32 on the CPU), the arg max of every position's logits and the hidden
state after every block, written to a directory so that the session's answers can be held to them. Also checks the binary64 specification (qwen_ref.py) against them.

    python3 examples/qwen/hf_reference.py OUTDIR [--nb 8] [--t 512]"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402

arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


def run(out, nb, t, device="cpu"):
    from transformers import AutoModelForCausalLM
    os.makedirs(out, exist_ok=True)
    tokens = data.fixed_batch(nb, t)
    np.save(os.path.join(out, "tokens.npy"), tokens)
    model = AutoModelForCausalLM.from_pretrained(data.snapshot(), torch_dtype=torch.float32).to(device).eval()
    ids = torch.tensor(tokens, device=device)
    t0 = time.time()
    last = []
    hook = model.model.norm.register_forward_hook(lambda m, i, o_: last.append(i[0].detach().reshape(-1, i[0].shape[-1]).cpu().numpy()))  # the last block's output, before the final norm
    with torch.no_grad():
        o = model(input_ids=ids, labels=ids, output_hidden_states=True)
    hook.remove()
    print(f"transformers float32 on {device}: {time.time() - t0:.1f} s, loss {o.loss.item():.6f}")
    # hidden[k] is the residual stream after k blocks, k = 0 (the embedding) to the last block; transformers' last hidden state is after the final norm, so the hook's is used
    hs = np.stack([h.reshape(-1, h.shape[-1]).cpu().numpy() for h in o.hidden_states[:-1]] + last)
    np.save(os.path.join(out, "hidden.npy"), hs.astype(np.float32))
    np.save(os.path.join(out, "argmax.npy"), o.logits.argmax(-1).reshape(-1).cpu().numpy())
    np.save(os.path.join(out, "loss.npy"), np.array([o.loss.item()]))
    top2 = o.logits.topk(2, dim=-1).values.reshape(-1, 2).cpu().numpy()
    np.save(os.path.join(out, "margin.npy"), top2[:, 0] - top2[:, 1])  # how far the largest logit is from the second largest, to judge a disagreement of the arg max
    # per-position losses of the same logits, for comparing rows
    lg = o.logits.float()
    rows = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="none").reshape(nb, t - 1)
    np.save(os.path.join(out, "rows.npy"), rows.cpu().numpy())
    return o.loss.item()


if __name__ == "__main__":
    run(sys.argv[1], int(arg("--nb", 8)), int(arg("--t", 512)))
