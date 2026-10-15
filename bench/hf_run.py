#!/usr/bin/env python3
"""Hugging Face transformers in float32 on the CPU, for bench/bench.py (the
reference the correctness checks use, and a baseline). Prints one JSON object:
load seconds and, per run, prefill tokens a second, decode tokens a second
(from the difference of a `new`-token and a 1-token generation) and the ids.

    hf_run.py <model dir> <prompt> <new tokens> <runs>
"""
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, prompt, new, runs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
t = time.time()
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
model.eval()
load_s = time.time() - t
ids = tok(prompt, return_tensors="pt").input_ids
kw = dict(do_sample=False, repetition_penalty=1.0, top_k=None, top_p=None, temperature=None)
out = []
for _ in range(runs + 1):
    with torch.no_grad():
        t0 = time.time(); model(ids); pre = time.time() - t0
        t0 = time.time(); model.generate(ids, max_new_tokens=1, **kw); one = time.time() - t0
        t0 = time.time(); gen = model.generate(ids, max_new_tokens=new, **kw); full = time.time() - t0
    got = gen[0, ids.shape[1]:].tolist()
    out.append({"prompt_tps": ids.shape[1] / pre, "decode_tps": (len(got) - 1) / (full - one),
                "prompt_tokens": ids.shape[1], "decode_tokens": len(got), "ids": got, "text": tok.decode(got)})
print(json.dumps({"load_s": load_s, "runs": out[1:]}))
