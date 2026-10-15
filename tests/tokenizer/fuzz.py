#!/usr/bin/env python3
"""Hostile tokenizer files and hostile text: nothing may trap or hang.

    fuzz.py <cancho-gpu> <tokenizer.json> <model.gguf> <corpus> [mutations]

`mutations` (default 300) copies of each of the tokenizer.json, the GGUF and
the corpus, with one to four bytes replaced from a fixed-seed generator, go
through `cancho-gpu tokenize`. The answer must be exit 0 (the dump) or exit 1
with `error[tok-...]`, `error[gguf-...]` or `error[io]` on standard error,
within ten seconds.
"""
import os
import subprocess
import sys
import tempfile

exe, js, gg, corpus = sys.argv[1:5]
mutations = int(sys.argv[5]) if len(sys.argv) > 5 else 300
state = 777


def rnd(n):
    global state
    state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (state >> 33) % n


def mutate(data, region):
    b = bytearray(data)
    for _ in range(1 + rnd(4)):
        b[rnd(min(region, len(b)))] = [0, 1, 34, 92, 123, 125, 127, 128, 192, 237, 244, 255, rnd(256)][rnd(13)]
    return bytes(b)


small = open(corpus, "rb").read()
cases = []
for _ in range(mutations):
    cases.append(("json", mutate(open(js, "rb").read(), 4000), small))
    cases.append(("gguf", mutate(open(gg, "rb").read(), 3000), small))
    cases.append(("corpus", None, mutate(small, len(small))))
bad = 0
with tempfile.TemporaryDirectory() as d:
    tp, cp = os.path.join(d, "t"), os.path.join(d, "c")
    for i, (kind, tok, text) in enumerate(cases):
        src = {"json": js, "gguf": gg, "corpus": js}[kind]
        if tok is not None:
            with open(tp, "wb") as f:
                f.write(tok)
            src = tp
        with open(cp, "wb") as f:
            f.write(text)
        try:
            r = subprocess.run([exe, "tokenize", src, cp], capture_output=True, timeout=10)
        except subprocess.TimeoutExpired:
            print(f"case {i} ({kind}): timed out")
            bad += 1
            continue
        err = r.stderr.decode(errors="replace")
        if r.returncode == 0:
            continue
        if r.returncode != 1 or not (err.startswith("error[tok-") or err.startswith("error[gguf-") or err.startswith("error[io]")):
            print(f"case {i} ({kind}): exit {r.returncode}: {err[:100]!r}")
            bad += 1
print(f"{len(cases)} hostile inputs, {bad} bad")
sys.exit(1 if bad else 0)
