#!/usr/bin/env python3
"""MLX-LM on one checkpoint, for bench/bench.py: loads it, then generates `new`
tokens greedily `runs` times from a raw prompt (no chat template), printing one
JSON object: load seconds, and per run the prompt and generation tokens a second
as MLX reports them, its peak memory, and the token ids and text.

    mlx_run.py <model dir> <prompt> <new tokens> <runs>
"""
import json
import sys
import time

from mlx_lm import load, stream_generate

model_dir, prompt, new, runs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
t = time.time()
model, tok = load(model_dir)
load_s = time.time() - t
out = []
for _ in range(runs + 1):  # the first is a warm-up and is dropped
    ids, text, last = [], "", None
    for r in stream_generate(model, tok, prompt, max_tokens=new):
        ids.append(r.token)
        text += r.text
        last = r
    out.append({"prompt_tps": last.prompt_tps, "decode_tps": last.generation_tps,
                "peak_gb": last.peak_memory, "prompt_tokens": last.prompt_tokens,
                "decode_tokens": last.generation_tokens, "ids": ids, "text": text})
print(json.dumps({"load_s": load_s, "runs": out[1:]}))
