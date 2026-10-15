#!/usr/bin/env python3
"""The mutated files of fuzz.py, held to the `safetensors` library.

    differential.py <cancho-gpu> <file.safetensors> [mutations]

For each file the library and `cancho-gpu safetensors` both answer accept
or refuse; where both accept, the dumps must be identical. A file the
library accepts and cancho-gpu refuses, or the reverse, is listed with
cancho-gpu's rule: those are the differences `docs/runtime.md` §7 states
(duplicate keys, escapes in names, the 1 MiB header limit) and anything
else is a bug. Not in the gate: it needs the library and torch.
"""
import os
import subprocess
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
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

tally = Counter()
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "f.safetensors")
    for i, c in enumerate(cases):
        with open(path, "wb") as f:
            f.write(c)
        py = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "safetensors_dump.py"), path], capture_output=True)
        us = subprocess.run([exe, "safetensors", path], capture_output=True)
        if py.returncode == 0 and us.returncode == 0:
            tally["both accept, same" if py.stdout == us.stdout else "both accept, DIFFERENT"] += 1
        elif py.returncode != 0 and us.returncode != 0:
            tally["both refuse"] += 1
        elif py.returncode == 0:
            rule = us.stderr.decode().split("]")[0].split("[")[-1]
            tally["library accepts, we refuse: " + rule] += 1
        else:
            tally["library refuses, we accept"] += 1
            print(f"case {i}: library refuses ({py.stdout.decode()[:80]!r}), we accept")
for k, v in sorted(tally.items()):
    print(f"{v:6d}  {k}")
sys.exit(1 if tally["both accept, DIFFERENT"] or tally["library refuses, we accept"] else 0)
