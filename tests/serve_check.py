#!/usr/bin/env python3
"""Hold `cancho-gpu-serve` to the engine it serves: needs a Mac, the model in Ollama's store, and a built
`cancho-gpu-run35` and `cancho-gpu-serve`; not in the gate.

    tests/serve_check.py [ollama store]

1. For three requests (a plain question, a system message and a second turn, thinking on) the greedy
   completion of the server is the greedy completion of `cancho-gpu-run35` fed the prompt that
   `cancho-gpu chat` renders (the engine's own path): the same text.
2. A streamed answer is the non-streamed one, delta by delta.
3. The same request three times gives the same answer (the engine resets cleanly), with another in between.
4. A client that goes away in the middle of a stream does not stop the server.
5. A prompt past the context, a bad body, tools, an image, n = 2, no key: each refused with its status.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import http.client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.ollama/models")
PORT = 18000 + os.getpid() % 1000
KEY = "check-key"
failed = 0


def check(name, ok, detail=""):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))
    if not ok:
        failed += 1


REQUESTS = {
    "plain": {"messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}], "max_tokens": 40, "temperature": 0},
    "system_two_turns": {"messages": [{"role": "system", "content": "You answer in at most five words."}, {"role": "user", "content": "Name a fruit."},
                                      {"role": "assistant", "content": "Banana."}, {"role": "user", "content": "And a vegetable?"}], "max_tokens": 24, "temperature": 0},
    "thinking": {"messages": [{"role": "user", "content": "What is 17 times 3?"}], "enable_thinking": True, "reasoning_effort": "low", "max_tokens": 200, "temperature": 0},
}


def engine_answer(req):
    """What the engine says to the rendered prompt, greedy: the CLI, no server."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(req, f)
    prompt = subprocess.run([f"{ROOT}/build/cancho-gpu", "chat", f.name], capture_output=True, check=True).stdout.decode()
    os.unlink(f.name)
    out = subprocess.run([f"{ROOT}/build/cancho-gpu-run35", STORE, f"{ROOT}/tests/chx", prompt, str(req["max_tokens"])], capture_output=True, check=True).stdout.decode()
    text = out.split("\ntokens:")[0]
    return text.replace("<|im_end|>", "")


expected = {k: engine_answer(v) for k, v in REQUESTS.items()}

log = open(tempfile.mktemp(suffix=".log"), "w")
server = subprocess.Popen([f"{ROOT}/build/cancho-gpu-serve", STORE, f"{ROOT}/tests/chx", "--port", str(PORT), "--api-key", KEY], stderr=log)
try:
    for _ in range(120):
        try:
            socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
            break
        except OSError:
            time.sleep(1)
    else:
        sys.exit("the server did not come up")

    def call(method, path, body=None, key=KEY, raw=None):
        c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        c.request(method, path, raw if raw is not None else (json.dumps(body) if body is not None else None), headers)
        r = c.getresponse()
        data = r.read().decode()
        c.close()
        return r.status, data

    def chat(req):
        s, d = call("POST", "/v1/chat/completions", req)
        return s, json.loads(d)

    def streamed(req):
        c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
        c.request("POST", "/v1/chat/completions", json.dumps(dict(req, stream=True)), {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
        r = c.getresponse()
        reasoning, content, finish, done = "", "", None, False
        for line in r.read().decode().split("\n\n"):
            if line == "data: [DONE]":
                done = True
            elif line.startswith("data: {"):
                d = json.loads(line[6:])
                for ch in d["choices"]:
                    reasoning += ch["delta"].get("reasoning_content", "")
                    content += ch["delta"].get("content", "")
                    finish = ch["finish_reason"] or finish
        return r.status, reasoning, content, finish, done

    s, d = call("GET", "/health", key=None)
    check("health needs no key", s == 200 and json.loads(d)["status"] == "ok", d)
    s, d = call("GET", "/v1/models", key=None)
    check("models needs the key", s == 401, d)
    s, d = call("GET", "/v1/models")
    check("models", s == 200 and json.loads(d)["data"][0]["id"] == "qwen3.8:27b-mlx", d)

    # 1, 2: the engine's answers, whole and streamed
    for name, req in REQUESTS.items():
        s, d = chat(req)
        msg = d["choices"][0]["message"]
        ours = (("<think>\n" + msg.get("reasoning_content", "") + "\n</think>\n\n") if "reasoning_content" in msg else "") + msg["content"]
        want = expected[name]
        norm = lambda x: "".join(x.split())
        check(f"{name}: the server's greedy answer is the engine's", s == 200 and norm(ours.replace("<think>", "").replace("</think>", "")) == norm(want.replace("<think>", "").replace("</think>", "")),
              f"server {ours!r} engine {want!r}")
        s2, r, c, fin, done = streamed(req)
        check(f"{name}: streamed is the same answer", s2 == 200 and done and c == msg["content"] and r == msg.get("reasoning_content", "") and fin == d["choices"][0]["finish_reason"],
              f"stream {r!r} / {c!r} {fin} against {msg!r}")

    # 3: determinism with another request in between
    a = chat(REQUESTS["plain"])[1]["choices"][0]["message"]["content"]
    chat(REQUESTS["system_two_turns"])
    b = chat(REQUESTS["plain"])[1]["choices"][0]["message"]["content"]
    check("the same request after another gives the same answer", a == b, f"{a!r} {b!r}")

    # 4: a client that leaves in the middle of a stream
    sk = socket.create_connection(("127.0.0.1", PORT))
    body = json.dumps({"messages": [{"role": "user", "content": "Count from 1 to 100."}], "stream": True, "max_tokens": 200, "temperature": 0})
    sk.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {KEY}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n{body}").encode())
    sk.recv(512)
    sk.close()
    t0 = time.time()
    s, d = chat(REQUESTS["plain"])
    check("a request after a client left mid-stream is served", s == 200, d)

    # 5: refusals
    long_prompt = {"messages": [{"role": "user", "content": "word " * 1100}], "max_tokens": 5}
    s, d = call("POST", "/v1/chat/completions", long_prompt)
    check("a prompt past the context is a 400 that says so", s == 400 and "context" in d, f"{s} {d[:200]}")
    for label, kw, want in (
        ("not JSON", {"raw": "not json"}, 400),
        ("empty messages", {"body": {"messages": []}}, 400),
        ("tools", {"body": {"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function"}]}}, 400),
        ("an image", {"body": {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}}, 400),
        ("n = 2", {"body": {"messages": [{"role": "user", "content": "x"}], "n": 2}}, 400),
        ("max_tokens 0", {"body": {"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}}, 400),
        ("no key", {"body": REQUESTS["plain"], "key": None}, 401),
        ("a wrong key", {"body": REQUESTS["plain"], "key": "nope"}, 401),
    ):
        s, d = call("POST", "/v1/chat/completions", kw.pop("body", None), **kw)
        check(f"refused: {label}", s == want and "error" in json.loads(d), f"{s} {d[:200]}")
    s, d = call("GET", "/v1/chat/completions")
    check("GET on the chat route is a 405", s == 405, f"{s} {d[:100]}")
    s, d = call("GET", "/nope")
    check("an unknown route is a 404", s == 404, f"{s} {d[:100]}")
finally:
    server.terminate()
    server.wait()
print(f"{failed} failed" if failed else "all checks passed")
sys.exit(1 if failed else 0)
