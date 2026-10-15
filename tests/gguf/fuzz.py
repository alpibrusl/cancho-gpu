#!/usr/bin/env python3
"""Hostile GGUF bytes: no input may trap or hang the reader.

    fuzz.py <cancho-gpu> <file.gguf> [mutations]

Every proper prefix of the fixture's header, and `mutations` (default
1500) copies with one to four bytes replaced from a fixed-seed generator,
go through `cancho-gpu gguf`. The answer must be exit 0 (the dump) or
exit 1 with `error[gguf-...]` on standard error, within five seconds.
"""
import os
import subprocess
import sys
import tempfile

exe, src = sys.argv[1], sys.argv[2]
mutations = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
data = open(src, "rb").read()
state = 12345


def rnd(n):
    global state
    state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (state >> 33) % n


cases = [data[:n] for n in range(len(data))]
header = 1100  # past the last directory entry of tiny.gguf, before its data
for _ in range(mutations):
    b = bytearray(data)
    for _ in range(1 + rnd(4)):
        # Most bytes of interest are lengths and counts: aim at the header.
        b[rnd(header)] = [0, 1, 7, 8, 9, 127, 128, 255, rnd(256)][rnd(9)]
    cases.append(bytes(b))

bad = 0
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "f.gguf")
    for i, c in enumerate(cases):
        with open(path, "wb") as f:
            f.write(c)
        try:
            r = subprocess.run([exe, "gguf", path, "--sum"], capture_output=True, timeout=5)
        except subprocess.TimeoutExpired:
            print(f"case {i}: timed out")
            bad += 1
            continue
        err = r.stderr.decode(errors="replace")
        if r.returncode == 0:
            continue
        if r.returncode != 1 or not err.startswith("error[gguf-"):
            print(f"case {i}: exit {r.returncode}: {err[:100]!r}")
            bad += 1
print(f"{len(cases)} hostile files, {bad} bad")
sys.exit(1 if bad else 0)
