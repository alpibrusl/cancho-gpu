#!/usr/bin/env python3
"""Hostile safetensors bytes: no input may trap or hang the reader.

    fuzz.py <cancho-gpu> <file.safetensors> [mutations]

Every proper prefix of the file, and `mutations` (default 1500) copies with
one to four bytes of the length word or the header replaced from a
fixed-seed generator, go through `cancho-gpu safetensors`. The answer must
be exit 0 (the dump) or exit 1 with `error[st-...]` on standard error,
within five seconds.
"""
import os
import subprocess
import sys
import tempfile

exe, src = sys.argv[1], sys.argv[2]
mutations = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
data = open(src, "rb").read()
state = 4242


def rnd(n):
    global state
    state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (state >> 33) % n


header = 8 + int.from_bytes(data[:8], "little")
cases = [data[:n] for n in range(len(data))]
for _ in range(mutations):
    b = bytearray(data)
    for _ in range(1 + rnd(4)):
        b[rnd(header)] = [0, 1, 34, 45, 91, 93, 123, 125, 127, 128, 255, rnd(256)][rnd(12)]
    cases.append(bytes(b))

bad = 0
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "f.safetensors")
    for i, c in enumerate(cases):
        with open(path, "wb") as f:
            f.write(c)
        try:
            r = subprocess.run([exe, "safetensors", path], capture_output=True, timeout=5)
        except subprocess.TimeoutExpired:
            print(f"case {i}: timed out")
            bad += 1
            continue
        err = r.stderr.decode(errors="replace")
        if r.returncode == 0:
            continue
        if r.returncode != 1 or not err.startswith("error[st-"):
            print(f"case {i}: exit {r.returncode}: {err[:100]!r}")
            bad += 1
print(f"{len(cases)} hostile files, {bad} bad")
sys.exit(1 if bad else 0)
