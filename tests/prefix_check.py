#!/usr/bin/env python3
"""Hold the prefix cache of `cancho-gpu-serve` (docs/serving.md section 7) to the server without it: needs a Mac,
the model in Ollama's store and a built `cancho-gpu-serve`; not in the gate.

    tests/prefix_check.py [ollama store]
    tests/prefix_check.py miss [ollama store]      only the cost of a miss (a request that reuses nothing)

Two servers are started one after the other, one without `--prefix-cache` and one with it. Every conversation is
sent to both with the same history (the answers of the server without the cache), and for each turn:

1. the answer is the same, greedy and with a fixed seed at temperature 1 (the cache must not change a token);
2. the tokens reused (`usage.prompt_tokens_details.cached_tokens`) are what the case expects: a later turn of the
   conversation that took the slot reuses it, anything else reuses nothing;
3. the time to the first token (`max_tokens: 1`) and the decode rate (128 tokens against 1) are recorded for both.

Cases: two turns, several turns, thinking on, a system message, an interleaved second conversation, a repeated
request, an edited earlier message, and a request refused before it runs.
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
ARGS = [a for a in sys.argv[1:] if a not in ("miss", "control")]
MISS_ONLY = "miss" in sys.argv[1:]
CONTROL = "control" in sys.argv[1:]  # with `miss`: both servers without the cache (a baseline for the order and the process)
STORE = ARGS[0] if ARGS else os.path.expanduser("~/.ollama/models")
failed = 0
rows = []


def check(name, ok, detail=""):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))
    if not ok:
        failed += 1


class Server:
    def __init__(self, port, flags):
        self.port = port
        self.log = tempfile.mktemp(suffix=".log")
        self.p = subprocess.Popen([f"{ROOT}/build/cancho-gpu-serve", STORE, f"{ROOT}/tests/chx", "--port", str(port)] + flags, stderr=open(self.log, "w"))
        for _ in range(180):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                return
            except OSError:
                time.sleep(1)
        sys.exit("the server did not come up")

    def stop(self):
        self.p.terminate()
        self.p.wait()

    def decode_rate(self, messages, **kw):
        """Tokens a second from the first token's chunk to the last, streamed: the prefill is not in it."""
        body = dict(messages=messages, temperature=0, max_tokens=128, stream=True, stream_options={"include_usage": True}, **kw)
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        first = last = None
        n = 0
        buf = b""
        while True:
            chunk = r.read1(4096)
            if not chunk:
                break
            now = time.time()
            buf += chunk
            while b"\n\n" in buf:
                line, buf = buf.split(b"\n\n", 1)
                if line.startswith(b"data: {"):
                    d = json.loads(line[6:])
                    if d["choices"] and (d["choices"][0]["delta"].get("content") or d["choices"][0]["delta"].get("reasoning_content")):
                        first = first or now
                        last = now
                    if d.get("usage"):
                        n = d["usage"]["completion_tokens"]
        c.close()
        return (n - 1) / (last - first) if first and last and last > first and n > 1 else 0, n

    def chat(self, messages, **kw):
        body = dict(messages=messages, temperature=0, **kw)
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        t0 = time.time()
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        d = r.read().decode()
        dt = time.time() - t0
        c.close()
        return r.status, json.loads(d), dt


filler = "The quick brown fox jumps over the lazy dog near the river bank. "


def user(i, words):
    return {"role": "user", "content": f"Part {i}. Remember this: " + filler * words + "Reply with one short sentence."}


def run_pair(off, on, name, messages, expect, **kw):
    """One request to both servers; compare the answers; return (answer message, cached tokens, times)."""
    s1, d1, t_off = off.chat(messages, max_tokens=40, **kw)
    s2, d2, t_on = on.chat(messages, max_tokens=40, **kw)
    ok = s1 == 200 and s2 == 200
    if not ok:
        print(f"     status {s1} / {s2}: {str(d1)[:160]} / {str(d2)[:160]}")
    a, b = (d1["choices"][0]["message"], d2["choices"][0]["message"]) if ok else ({}, {})
    check(f"{name}: the answer is the same", ok and a == b, f"{a!r} / {b!r}")
    cached = d2["usage"]["prompt_tokens_details"]["cached_tokens"] if ok else -1
    prompt = d2["usage"]["prompt_tokens"] if ok else -1
    check(f"{name}: tokens reused ({cached} of {prompt}; expected {expect.__name__})", expect(cached, prompt), f"{cached} of {prompt}")
    check(f"{name}: the server without the cache reuses none", ok and d1["usage"]["prompt_tokens_details"]["cached_tokens"] == 0)
    return a, cached, prompt


def none(c, n):
    return c == 0


def some(c, n):
    return 0 < c < n and c % 8 == 0


def timing(off, on, name, messages, **kw):
    """First-token time (1 token), and the decode rate from the stream (median of three), with and without the
    cache. The request ends in a user message that asks for a long answer, so there are 128 tokens."""
    messages = messages + [{"role": "user", "content": "Now write the numbers from one to sixty in words, separated by commas."}]
    out = {}
    for label, srv in (("off", off), ("on", on)):
        _, d1, t1 = srv.chat(messages, max_tokens=1, **kw)
        rates = sorted(srv.decode_rate(messages, **kw) for _ in range(3))
        out[label] = (t1, rates[1][0], d1["usage"]["prompt_tokens"], d1["usage"]["prompt_tokens_details"]["cached_tokens"], rates[1][1])
    rows.append((name, out))


def conversation(off, on, name, system, turns, words, think=False, sampled=False):
    """A conversation of `turns` user messages: from turn 2 on, every turn must reuse a prefix."""
    kw = {"enable_thinking": True, "reasoning_effort": "low"} if think else {}
    if sampled:
        kw.update(temperature=1, seed=7)
        kw.pop("temperature")
    history = [{"role": "system", "content": system}] if system else []
    for i in range(turns):
        history.append(user(i + 1, words))
        a, cached, prompt = run_pair(off, on, f"{name} turn {i + 1}", list(history), none if i == 0 else some, **kw)
        reply = {"role": "assistant", "content": a.get("content", "")}
        if think and a.get("reasoning_content"):
            reply["reasoning_content"] = a["reasoning_content"]
        history.append(reply)
        if i == turns - 1:
            timing(off, on, f"{name}, turn {i + 2}, {prompt} tokens of history", list(history), **kw)
    return history


def miss_cost(off, on, n=8):
    """The snapshot's price: requests that cannot hit (each one a new text), one token each, the servers taking turns to go first."""
    times = {"off": [], "on": []}
    for i in range(n):
        msgs = [{"role": "user", "content": f"Unrelated text number {i * 7919}. " + (f"word{i} " + filler) * 20 + "Say ok."}]
        for label, srv in ((("off", off), ("on", on)) if i % 2 == 0 else (("on", on), ("off", off))):
            s, d, dt = srv.chat(msgs, max_tokens=1)
            assert s == 200 and d["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
            times[label].append(dt)
    mo, mn = sum(times["off"]) / n, sum(times["on"]) / n
    print(f"a miss, {n} requests of about 330 tokens: {mo * 1000:.0f} ms without the cache, {mn * 1000:.0f} ms with it ({(mn - mo) * 1000:+.0f} ms)")
    check("a miss costs at most 100 ms more than without the cache", mn - mo <= 0.1, f"{(mn - mo) * 1000:.0f} ms")


off = Server(18070, [])
on = None
try:
    # The servers are started one after the other only to keep memory and timing apart: both are up for the checks.
    on = Server(18071, [] if CONTROL else ["--prefix-cache"])
    if MISS_ONLY:
        miss_cost(off, on)
        sys.exit(1 if failed else 0)
    miss_cost(off, on)
    conversation(off, on, "plain", None, 4, 12)
    conversation(off, on, "system message", "You answer in at most ten words.", 3, 14)
    conversation(off, on, "thinking on", None, 3, 10, think=True)
    conversation(off, on, "sampled (temperature 1, seed 7)", None, 3, 12, sampled=True)

    # An interleaved second conversation takes the slot: the first one's next turn reuses nothing, the one after it does.
    a1 = [user(1, 14)]
    ra, _, _ = run_pair(off, on, "interleave A1", a1, none)
    b1 = [{"role": "user", "content": "A different chat. " + filler * 10}]
    run_pair(off, on, "interleave B1", b1, none)
    a2 = a1 + [{"role": "assistant", "content": ra["content"]}, user(2, 14)]
    ra2, _, _ = run_pair(off, on, "interleave A2 (the slot is B's)", a2, none)
    a3 = a2 + [{"role": "assistant", "content": ra2["content"]}, user(3, 14)]
    run_pair(off, on, "interleave A3", a3, some)

    # The same request again reuses the slot it left; an edited earlier message reuses nothing.
    run_pair(off, on, "repeat 1", a3, some)
    run_pair(off, on, "repeat 2", a3, some)
    edited = [dict(a3[0], content=a3[0]["content"].replace("Part 1", "Part one"))] + a3[1:]
    run_pair(off, on, "edited first message", edited, none)

    # A request refused before it runs (past the context) leaves the slot alone.
    s, d, _ = on.chat([{"role": "user", "content": "word " * 1100}], max_tokens=5)
    check("a prompt past the context is refused", s == 400)
    run_pair(off, on, "after a refusal", edited, some)

    # A client that leaves mid-stream, then the same conversation again.
    sk = socket.create_connection(("127.0.0.1", on.port))
    body = json.dumps({"messages": a3, "stream": True, "max_tokens": 200, "temperature": 0})
    sk.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n{body}").encode())
    sk.recv(256)
    sk.close()
    run_pair(off, on, "after a client left mid-stream", a3, lambda c, n: 0 <= c < n)
finally:
    off.stop()
    if on:
        on.stop()

print()
print(f"{'case':58s} {'first token off':>16s} {'on':>8s} {'reused':>7s}   {'decode tok/s off':>16s} {'on':>6s}  tokens")
for name, o in rows:
    print(f"{name:58s} {o['off'][0]:15.2f}s {o['on'][0]:7.2f}s {o['on'][3]:7d}   {o['off'][1]:16.1f} {o['on'][1]:6.1f}  {o['off'][4]}/{o['on'][4]}")
print(f"{failed} failed" if failed else "all checks passed")
sys.exit(1 if failed else 0)
