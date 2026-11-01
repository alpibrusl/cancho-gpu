#!/usr/bin/env python3
"""The Python oracles of the model readers still agree with the goldens made by the Rust ones they replaced (docs/native.md N2f).

    oracles.py

* `tools/gguf_dump.py` against golden/gguf/ (3 fixtures): identical text.
* `tools/sampler_dump.py` against golden/sampler/ (75 cases, `tests/sampler/cases.txt`): the picks and the nucleus ids identical,
  the probabilities within 8 ulps (`tests/sampler/compare.py`'s rule); the count that are bit-identical is printed. The sampler's
  `exp` is numpy's here and was the platform's `expf` in the Rust, so another platform may differ in a few last places.

These are the independent checks of `cancho-gpu gguf` and `sample`; `tests/gguf.sh` and `tests/sampler.sh` hold `cancho-gpu` to the
same goldens. Needs numpy for the sampler; skips it (exit 0) without.
"""
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
bad = 0


def say(good, text):
    global bad
    print(f"{'ok  ' if good else 'FAIL'} {text}")
    bad += 0 if good else 1


for f in sorted(glob.glob("tests/gguf/*.gguf")):
    name = os.path.basename(f)[:-5]
    r = subprocess.run([sys.executable, "tools/gguf_dump.py", f, "--sum"], capture_output=True, text=True)
    want = open(f"golden/gguf/{name}.txt").read()
    say(r.returncode == 0 and r.stdout == want, f"gguf_dump.py reproduces golden/gguf/{name}.txt")

try:
    import numpy  # noqa: F401
except ImportError:
    print("skip sampler_dump.py: numpy is not installed")
    sys.exit(1 if bad else 0)

cases = n = fail = rows = probs = same = worst = 0
for line in open("tests/sampler/cases.txt"):
    fields = line.split()
    if not fields:
        continue
    n += 1
    r = subprocess.run([sys.executable, "tools/sampler_dump.py", "tests/sampler/" + fields[0], *fields[1:]], capture_output=True, text=True)
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"oracles-sampler-{os.getpid()}.txt")
    open(out, "w").write(r.stdout)
    c = subprocess.run([sys.executable, "tests/sampler/compare.py", out, "golden/sampler/%03d.txt" % n, "8"], capture_output=True, text=True)
    os.remove(out)
    if r.returncode or c.returncode:
        fail += 1
        print(f"     case {n} {fields}: {(r.stderr or c.stdout).strip()[:160]}")
        continue
    m = [int(x) for x in re.findall(r"\d+", c.stdout)]
    rows, probs, same, worst = rows + m[0], probs + m[1], same + m[2], max(worst, m[3])
say(fail == 0, f"sampler_dump.py reproduces golden/sampler/ ({n} cases, {rows} rows, picks and ids identical, {same} of {probs} probabilities bit-identical, worst {worst} ulps)")
sys.exit(1 if bad else 0)
