#!/usr/bin/env python3
"""The shim's greedy text equals transformers' on the same weights (docs/roadmap-models.md M8).

    python3 examples/qwen/check_serve.py [--new 64] [--primer FILE] [--device mps]

Four prompts (short ones, and the primer in front of a task when --primer is given) are continued greedily two ways on ONE model object (f16, the same device): by `Served.generate`
(chunked prefill, the key/value cache, prefix reuse: the second prompt of the same primer reuses the first's) and by `model.generate(do_sample=False)` with no repetition penalty. Prints the number of tokens
that agree and the first divergence; exit 0 when every continuation is identical."""
import argparse
import sys
import types

import torch

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import serve_ollama as S  # noqa: E402

PROMPTS = ["Write a cancho function that returns the larger of two ints.",
           "Reply with the complete file, with the task done: fn clamp(x: int, lo: int, hi: int) -> [] int",
           "The cancho program below is refused by its compiler: error[E0308] mismatched types. Reply with the complete corrected file.",
           "Explain in two sentences what a capability is in a systems language."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--primer")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model", default="~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tokenizer", default="~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B")
    a = ap.parse_args()
    S.State.args = types.SimpleNamespace(tokenizer=a.tokenizer, device=a.device)
    m = S.Served("check", a.model, a.device, torch.float16, True, 2048)
    primer = open(a.primer).read() if a.primer else ""
    bad = 0
    for i, p in enumerate(PROMPTS):
        text = m.render(primer + p)
        ids = m.tok(text, add_special_tokens=False)["input_ids"]
        resp, reason, n_prompt, n_new, tp, td, reused = m.generate(text, a.new, 0.0, 0, 1.0, 0, 1 << 20)
        ours = m.tok(resp, add_special_tokens=False)["input_ids"]
        S.GROUPED_DECODE[0] = False      # the reference is the library's own attention
        with torch.no_grad():
            x = torch.tensor([ids], device=a.device)
            out = m.model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=a.new, do_sample=False, repetition_penalty=1.0,
                                   temperature=None, top_p=None, top_k=None, eos_token_id=list(m.eos), pad_token_id=m.tok.eos_token_id)
        S.GROUPED_DECODE[0] = True
        hf = out[0, len(ids):].tolist()
        hf = hf[:hf.index(min(m.eos & set(hf)))] if m.eos & set(hf) else hf
        n = min(len(ours), len(hf))
        div = next((k for k in range(n) if ours[k] != hf[k]), n if len(ours) == len(hf) else n)
        same = ours == hf
        bad += not same
        print(f"prompt {i}: {n_prompt} prompt tokens ({reused} reused), {len(ours)} vs {len(hf)} new tokens, identical={same}, first divergence at {div}, shim {n_new / td:.1f} tok/s, prefill {tp:.2f}s")
        if not same:
            print("   shim:", repr(m.tok.decode(ours[div:div + 12])), " hf:", repr(m.tok.decode(hf[div:div + 12])))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
