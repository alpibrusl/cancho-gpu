#!/usr/bin/env python3
"""The reference for `cancho-gpu-run`: the same checkpoint run by Hugging Face
transformers in float32 on the CPU, greedy.

    reference.py <model dir> <new tokens> <prompt>...

Prints, for each prompt, `tokens: <ids>` (the greedy continuation) and
`logits: <id>=<value> ...` (the five largest logits at the last prompt
position), the forms scripts/model-check.sh compares.
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

args = sys.argv[1:]
follow = None
if args[0] == "--follow":
    follow = [int(x) for x in args[1].split()]
    args = args[2:]
model_dir, new = args[0], int(args[1])
sys.argv = [sys.argv[0]] + args
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
model.eval()
for prompt in sys.argv[3:]:
    ids = tok(prompt, return_tensors="pt").input_ids
    if follow is not None:
        # Teacher-forced: the top logits at each position of a given continuation.
        full = torch.cat([ids, torch.tensor([follow])], dim=1)
        with torch.no_grad():
            rows = model(full).logits[0]
        for k in range(len(follow) + 1):
            row = rows[ids.shape[1] - 1 + k]
            top = torch.topk(row, 5)
            print(f"step {ids.shape[1] - 1 + k} top:", " ".join(f"{i}={v:.4f}" for i, v in zip(top.indices.tolist(), top.values.tolist())))
        continue
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    top = torch.topk(logits, 5)
    out = model.generate(ids, max_new_tokens=new, do_sample=False, repetition_penalty=1.0, top_k=None, top_p=None, temperature=None)
    print("prompt:", repr(prompt))
    print("tokens:", " ".join(str(i) for i in out[0, ids.shape[1]:].tolist()))
    print("logits:", " ".join(f"{i}={v:.4f}" for i, v in zip(top.indices.tolist(), top.values.tolist())))
