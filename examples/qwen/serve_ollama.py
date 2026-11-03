#!/usr/bin/env python3
"""An Ollama-API-compatible endpoint over a local Qwen2.5 (docs/roadmap-models.md M8).

    python3 examples/qwen/serve_ollama.py --port 11435 \
        --model base=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct \
        --model tuned=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct@/path/to/peft_adapter_dir

cancho-code's `--provider ollama` agent (scripts/cancho-agent.sh) POSTs `/api/generate` with `stream:false`, a `prompt` (the primer, when there is one, in front of the
task) and `options` {num_ctx, temperature, num_predict}. This answers it: the model's chat template wraps the prompt as one user turn (what Ollama does for a `/api/generate`
without `raw`), the tokens are sampled with Ollama's defaults for what the request leaves out (top_k 20, top_p 0.8: the Instruct model's generation_config; no repetition penalty), and the answer carries `response`, `done`, `done_reason`,
`prompt_eval_count`, `eval_count` and the durations in nanoseconds. `/api/tags`, `/api/chat` and `/api/show` are there as far as a client needs to see the model.

`--model NAME=DIR[@ADAPTER]`: DIR is a Hugging Face cache directory (or a snapshot) of a Qwen2 model; ADAPTER is a peft LoRA directory, merged into the weights at load
(`merge_and_unload` on float32, rounded once to float16: the weights served are f16 for every arm, so arms differ only in weights and prompt). One model at a time is served
(requests are serialised: one GPU job), a request for another name swaps it.

Prefix reuse: the key/value cache of the last prompt is kept, and a request that starts with the same tokens (the primer, in the arms that have one) recomputes only the rest.
`--no-prefix-cache` turns it off. Prefill is done in chunks of `--chunk` tokens.

Why Python and not cancho: cancho-gpu's own server (`cancho-gpu-serve`, docs/serving.md) serves the 27B only, has a 1,024-position cache, no prefix cache, and a prompt that does not fit is
refused with a 400; the primer alone is about 10,700 Qwen tokens. The 1.5B path in this directory (qwen_session.py) runs 512-token windows and recomputes the window for every new token."""
import argparse
import glob
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

LOCK = threading.Lock()
GROUPED_DECODE = [True]     # check_serve.py turns it off to compare with the library's own attention


def _decode_attention(module, q, k, v, mask, dropout=0.0, scaling=None, **kw):
    """Attention for one query position: the library's sdpa path repeats the key/value heads to the query heads and builds a mask, which at a context of 10,000 tokens made a decode step 3 to 5 times slower
    (measured, docs/roadmap-models.md M8). Here the 6 query heads that share a key/value head are given to sdpa as 6 query positions of that one head (no repeat, no mask: all of them see the whole cache)."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    if q.shape[2] != 1 or not GROUPED_DECODE[0]:
        return sdpa_attention_forward(module, q, k, v, mask, dropout=dropout, scaling=scaling, **kw)
    b, h, _, d = q.shape
    g = k.shape[1]
    o = torch.nn.functional.scaled_dot_product_attention(q.view(b, g, h // g, d), k, v, scale=scaling)
    return o.reshape(b, 1, h, d), None


AttentionInterface.register("sdpa", _decode_attention)


def snapshot(path):
    path = os.path.expanduser(path)
    s = glob.glob(os.path.join(path, "snapshots", "*"))
    return s[0] if s else path


class Served:
    """One model on the device: weights (f16), tokenizer, the cache of the last prompt."""

    def __init__(self, name, spec, device, dtype, prefix_cache, chunk):
        base, _, adapter = spec.partition("@")
        self.name, self.device, self.prefix_cache, self.chunk = name, device, prefix_cache, chunk
        # the cached Qwen2.5-1.5B-Instruct snapshot holds the weights and config only (no tokenizer files, and no download is allowed); the tokenizer and the chat template are the
        # Qwen2.5-1.5B repository's, which the family shares (the same vocabulary, merges and <|im_start|>/<|im_end|> tokens)
        self.tok = AutoTokenizer.from_pretrained(snapshot(State.args.tokenizer))
        t0 = time.time()
        m = AutoModelForCausalLM.from_pretrained(snapshot(base), torch_dtype=torch.float32)
        if adapter:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, os.path.expanduser(adapter)).merge_and_unload()
        # f32 -> f16 once, for base and merged alike (bf16 values narrow to f16 exactly in range: docs/inference.md section 1)
        self.model = m.to(dtype).to(device).eval()
        self.spec, self.load_s = spec, time.time() - t0
        self.past, self.past_ids = None, []
        self.eos = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id}

    def render(self, prompt, system=None, raw=False):
        if raw:
            return prompt
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, text, num_predict, temperature, top_k, top_p, seed, num_ctx, stop=()):
        """(response text, done_reason, prompt tokens, new tokens, prefill seconds, decode seconds, tokens reused)."""
        from transformers import DynamicCache
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > num_ctx - 1:       # Ollama truncates a prompt that does not fit; the driver sizes num_ctx so this does not happen
            ids = ids[-(num_ctx - 1):]
        n_prompt = len(ids)
        t0 = time.time()
        keep = 0
        if self.prefix_cache and self.past is not None:
            while keep < min(len(ids), len(self.past_ids)) and ids[keep] == self.past_ids[keep]:
                keep += 1
            keep = min(keep, len(ids) - 1)      # at least one token is computed: its logits start the answer
        if keep > 0:
            self.past.crop(keep)
            past = self.past
        else:
            past = DynamicCache()
        pos = keep
        logits = None
        while pos < len(ids):
            chunk = torch.tensor([ids[pos:pos + self.chunk]], device=self.device)
            out = self.model(input_ids=chunk, past_key_values=past, use_cache=True, logits_to_keep=1)
            logits, past = out.logits[:, -1, :], out.past_key_values
            pos += chunk.shape[1]
        self.sync()
        t1 = time.time()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        new, reason = [], "length"
        text_out = ""
        while len(new) < num_predict and n_prompt + len(new) < num_ctx:
            nxt = self.sample(logits[0].float().cpu(), temperature, top_k, top_p, gen)
            if nxt in self.eos:
                reason = "stop"
                break
            new.append(nxt)
            if stop:
                text_out = self.tok.decode(new)
                hit = [s for s in stop if s in text_out]
                if hit:
                    reason = "stop"
                    break
            out = self.model(input_ids=torch.tensor([[nxt]], device=self.device), past_key_values=past, use_cache=True)
            logits, past = out.logits[:, -1, :], out.past_key_values
        self.sync()
        t2 = time.time()
        text_out = self.tok.decode(new)
        for s in stop:
            if s in text_out:
                text_out = text_out[:text_out.index(s)]
        self.past = past
        self.past_ids = (ids + new)[:past.get_seq_length()]     # the last sampled token's key/value were never computed when the loop ended
        return text_out, reason, n_prompt, len(new), t1 - t0, t2 - t1, keep

    def sync(self):
        if self.device == "mps":
            torch.mps.synchronize()

    @staticmethod
    def sample(logits, temperature, top_k, top_p, gen):
        if temperature <= 0:
            return int(torch.argmax(logits))
        logits = logits / temperature
        if top_k and top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
        probs = torch.softmax(logits, -1)
        if 0 < top_p < 1:
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, 0)
            sp[(cum - sp) > top_p] = 0.0
            probs = torch.zeros_like(probs).scatter_(0, si, sp)
            probs = probs / probs.sum()
        return int(torch.multinomial(probs, 1, generator=gen))


class State:
    models = {}      # name -> spec
    current = None
    args = None
    counter = 0
    log = None


def served(name):
    S = State
    if name not in S.models:
        hit = [k for k in S.models if name.split(":")[0] == k.split(":")[0]]
        if not hit:
            return None
        name = hit[0]
    if S.current is None or S.current.name != name:
        S.current = None
        if S.args.device == "mps":
            torch.mps.empty_cache()
        S.current = Served(name, S.models[name], S.args.device, torch.float16, not S.args.no_prefix_cache, S.args.chunk)
        print(f"[serve] loaded {name} ({S.models[name]}) in {S.current.load_s:.1f}s", file=sys.stderr, flush=True)
    return S.current


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", ""):
            body = b"Ollama is running"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/tags":
            self.send_json(200, {"models": [{"name": k, "model": k, "details": {"family": "qwen2"}} for k in State.models]})
        elif self.path == "/api/version":
            self.send_json(200, {"version": "0.0.0-cancho-gpu-m8"})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self.send_json(400, {"error": "invalid JSON"})
        if self.path == "/api/show":
            return self.send_json(200, {"details": {"family": "qwen2"}})
        if self.path not in ("/api/generate", "/api/chat"):
            return self.send_json(404, {"error": "not found"})
        o = req.get("options") or {}
        with LOCK:
            m = served(req.get("model", ""))
            if m is None:
                return self.send_json(404, {"error": f"model '{req.get('model')}' not found"})
            State.counter += 1
            seed = int(o["seed"]) if "seed" in o else State.args.seed * 1000003 + State.counter
            if self.path == "/api/chat":
                text = m.tok.apply_chat_template(req.get("messages", []), tokenize=False, add_generation_prompt=True)
            else:
                text = m.render(req.get("prompt", ""), req.get("system"), bool(req.get("raw")))
            t0 = time.time()
            npred = int(o.get("num_predict", 128))
            resp, reason, np_, ne, tp, td, reused = m.generate(
                text, npred if npred > 0 else 4096, float(o.get("temperature", 0.8)),
                int(o.get("top_k", 20)), float(o.get("top_p", 0.8)), seed, int(o.get("num_ctx", 2048)), tuple(o.get("stop") or ()))
            total = time.time() - t0
        rec = {"t": round(time.time(), 1), "model": m.name, "prompt_tokens": np_, "reused": reused, "new_tokens": ne, "prefill_s": round(tp, 3),
               "decode_s": round(td, 3), "tok_s": round(ne / td, 2) if td > 0 and ne else None, "total_s": round(total, 3), "reason": reason, "seed": seed}
        if State.log:
            State.log.write(json.dumps(rec) + "\n")
            State.log.flush()
        out = {"model": req.get("model"), "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "done": True, "done_reason": reason,
               "total_duration": int(total * 1e9), "load_duration": 0, "prompt_eval_count": np_, "prompt_eval_duration": int(tp * 1e9),
               "eval_count": ne, "eval_duration": int(td * 1e9)}
        if self.path == "/api/chat":
            out["message"] = {"role": "assistant", "content": resp}
        else:
            out["response"] = resp
        self.send_json(200, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", action="append", required=True, help="NAME=DIR[@ADAPTER]")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tokenizer", default="~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B", help="a directory with tokenizer.json and tokenizer_config.json (chat template)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--request-log", help="JSON lines: one per request (tokens, seconds, tokens/s)")
    a = ap.parse_args()
    State.args = a
    for spec in a.model:
        k, _, v = spec.partition("=")
        State.models[k] = v
    if a.request_log:
        State.log = open(a.request_log, "a")
    served(next(iter(State.models)))
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[serve] listening on {a.host}:{a.port}, models {list(State.models)}", file=sys.stderr, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
