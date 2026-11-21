#!/usr/bin/env python3
"""The same requests to Ollama and to `cancho-gpu-serve`, one after the other, on this machine (docs/serving.md
section 8): needs a Mac, a running Ollama with the model, and a built `cancho-gpu-serve`; not in the gate.

    tests/ollama_compare.py [ollama|ours|both] [ollama store]

For each case the time to the first token (wall clock, streamed) and, where tokens are generated, the decode rate
(tokens a second from the first chunk to the last). Ollama's own counters (`prompt_eval_duration`, `eval_duration`)
are printed beside the wall clock. Thinking is off and the temperature is 0 in both. Ollama is unloaded before our
server is started, so the two never share the GPU.

Cases: a cold prompt of three lengths (each a text never seen), decode on prose, a conversation of four turns, two
conversations taking turns, and a long shared system message followed by different questions.
"""
import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHO = sys.argv[1] if len(sys.argv) > 1 else "both"
STORE = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/.ollama/models")
MODEL = "qwen3.8:27b-mlx"
NONCE = int(time.time())
WORDS = "The harbour master counted nine boats before the fog came in and the lamp was lit. "


def stream(port, path, body, pick):
    """Returns (seconds to the first token, seconds from first to last, chunks with text, the last object)."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    t0 = time.time()
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        sys.exit(f"{port}{path}: {r.status} {r.read()[:300]}")
    first = last = None
    n = 0
    final = None
    text = ""
    buf = b""
    while True:
        chunk = r.read1(65536)
        if not chunk:
            break
        now = time.time()
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if line.startswith(b"data: "):
                line = line[6:]
            if not line.startswith(b"{"):
                continue
            d = json.loads(line)
            piece = pick(d)
            if piece:
                text += piece
                n += 1
                first = first or now
                last = now
            final = d if (d.get("usage") or d.get("done")) else final
    return (first or time.time()) - t0, (last - first) if first else 0.0, n, final, text


class Ollama:
    name = "ollama"

    def __init__(self):
        self.chat([{"role": "user", "content": "hi"}], 1)  # load

    def chat(self, messages, max_tokens):
        body = dict(model=MODEL, messages=messages, stream=True, think=False, keep_alive="10m", options=dict(temperature=0, num_predict=max_tokens, num_ctx=4096))
        ttft, span, n, fin, text = stream(11434, "/api/chat", body, lambda d: d.get("message", {}).get("content"))
        fin = fin or {}
        out = fin.get("eval_count", n)
        own = f"own: prompt {fin.get('prompt_eval_count', 0)} tok in {fin.get('prompt_eval_duration', 0) / 1e9:.2f}s, eval {out} tok in {fin.get('eval_duration', 0) / 1e9:.2f}s"
        return ttft, span, out, fin.get("prompt_eval_count", 0), text, own

    def stop(self):
        stream(11434, "/api/chat", dict(model=MODEL, messages=[], keep_alive=0), lambda d: None)
        time.sleep(3)


class Ours:
    name = "ours"

    def __init__(self, flags=()):
        self.log = tempfile.mktemp(suffix=".log")
        self.p = subprocess.Popen([f"{ROOT}/build/cancho-gpu-serve", STORE, f"{ROOT}/tests/chx", "--port", "18072", *flags], stderr=open(self.log, "w"))
        for _ in range(180):
            try:
                socket.create_connection(("127.0.0.1", 18072), timeout=1).close()
                break
            except OSError:
                time.sleep(1)
        self.chat([{"role": "user", "content": "hi"}], 1)

    def chat(self, messages, max_tokens):
        body = dict(messages=messages, temperature=0, max_tokens=max_tokens, stream=True, stream_options={"include_usage": True})
        ttft, span, n, fin, text = stream(18072, "/v1/chat/completions", body, lambda d: d["choices"][0]["delta"].get("content") if d.get("choices") else None)
        u = (fin or {}).get("usage", {})
        own = f"cached {u.get('prompt_tokens_details', {}).get('cached_tokens', 0)}"
        return ttft, span, u.get("completion_tokens", n), u.get("prompt_tokens", 0), text, own

    def stop(self):
        self.p.terminate()
        self.p.wait()


def row(srv, case, messages, max_tokens=1):
    ttft, span, out, prompt, text, own = srv.chat(messages, max_tokens)
    rate = f"{(out - 1) / span:6.1f} tok/s" if out > 8 and span > 0 else " " * 12
    print(f"{srv.name:6} {case:46} prompt {prompt:4}  first token {ttft:6.2f}s  {rate}  out {out:3}  [{own}]", flush=True)
    return text


def run(srv, tag):
    global NONCE
    for reps in (6, 22, 42):
        NONCE += 1
        row(srv, f"cold, {reps} sentences", [{"role": "user", "content": f"Text {tag} {NONCE}. " + WORDS * reps + "Say ok."}])
    for i in range(2):
        NONCE += 1
        row(srv, f"decode on prose, run {i + 1}", [{"role": "user", "content": f"({NONCE}) Write a story of about 300 words about a lighthouse keeper who finds a letter."}], 256)

    # One conversation, four turns: each turn's history is the last one plus an answer and a question.
    NONCE += 1
    hist = [{"role": "user", "content": f"Conversation {tag} {NONCE}. " + WORDS * 10 + "Name one boat."}]
    for turn in range(4):
        a = row(srv, f"conversation, turn {turn + 1}", hist, 24)
        hist += [{"role": "assistant", "content": a}, {"role": "user", "content": WORDS * 4 + f"Name boat number {turn + 2}."}]

    # Two conversations taking turns: the case one slot cannot serve.
    convs = []
    for k in "AB":
        NONCE += 1
        convs.append([{"role": "user", "content": f"Chat {k} {tag} {NONCE}. " + WORDS * 12 + "Name one boat."}])
    for turn in range(3):
        for k, h in zip("AB", convs):
            a = row(srv, f"two chats, {k} turn {turn + 1}", h, 16)
            h += [{"role": "assistant", "content": a}, {"role": "user", "content": WORDS * 2 + "Name another."}]

    # One long system message, different questions: the shared part ends inside the prompt, before the last opener.
    NONCE += 1
    system = {"role": "system", "content": f"Rules {tag} {NONCE}. " + WORDS * 30}
    for q in ("How many boats?", "What was lit?", "Who counted?"):
        row(srv, f"shared system message: {q}", [system, {"role": "user", "content": q}], 8)


if WHO in ("ollama", "both"):
    o = Ollama()
    try:
        run(o, "o")
    finally:
        o.stop()
if WHO in ("ours", "both"):
    s = Ours()
    try:
        run(s, "c")
    finally:
        s.stop()
