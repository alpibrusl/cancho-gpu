#!/usr/bin/env python3
"""`cancho-gpu tokenize` against Hugging Face `tokenizers`, on real tokenizer.json files (docs/native.md N2f).

    tokenizer_hf.py [build/cancho-gpu]

`tokenizers` reads each file's own `normalizer`, `pre_tokenizer`, `decoder` and added tokens and knows nothing of this project, so
it is the independent oracle the lex-rt Rust tokenizer was. **They do not agree, and this test says exactly how.** The hand-written
corpus (tests/tokenizer/corpus.txt, 37 records) differs in 3 records, and the random one (tests/tokenizer/corpus-random.txt, 500
records over the scripts and marks the split and NFC care about) in a few hundred. The causes seen (docs/runtime.md section 9 has
the account): the pre-tokenizer is one hard-wired pattern whatever the file says; `\\s*[\\r\\n]+` does not take the longest run
(record 9, `"  \\n  \\n x"`, the same on both tokenizers); the character classes `\\p{L}`, `\\p{M}`, `\\p{N}` are
approximations (`Ⅷ`, `⁴`, Thai vowel marks); and NFC lacks the singleton mappings (`Ω` U+2126, `Å` U+212B).

What this test holds: for each real tokenizer found, the set of records of the hand-written corpus that differ, and the number
that differ in the random one, are exactly the pinned ones. A divergence that appears is a regression; one that goes is a fix,
and the pin below is lowered with it. It is skipped (exit 0) when `tokenizers` or the tokenizer files are missing.
"""
import glob
import os
import subprocess
import sys

BIN = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "build/cancho-gpu")
HOME = os.path.expanduser("~")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# name: (candidate tokenizer.json paths, pinned differing records of corpus.txt, pinned count of differing records of corpus-random.txt)
PINS = {
    "qwen2.5-0.5b": (sorted(glob.glob(HOME + "/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*/tokenizer.json")), [9, 17, 19], 234),
    "qwen3.8-27b": ([os.path.realpath(os.path.join(ROOT, "..", "models", "qwen3.8-27b", "tokenizer.json"))], [9, 17, 19], 211),
}

try:
    from tokenizers import Tokenizer
except ImportError:
    print("skip tokenizer_hf: the `tokenizers` library is not installed")
    sys.exit(0)

bad = 0
ran = 0
for name, (paths, pinned_small, pinned_random) in PINS.items():
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print(f"skip {name}: no tokenizer.json found")
        continue
    ran += 1
    path = paths[0]
    tok = Tokenizer.from_file(path)
    for corpus_name, pinned in (("corpus.txt", pinned_small), ("corpus-random.txt", pinned_random)):
        cp = os.path.join(ROOT, "tests", "tokenizer", corpus_name)
        records = open(cp, encoding="utf-8", newline="").read().split("\n%%\n")  # newline="": the corpus has \r\n
        r = subprocess.run([BIN, "tokenize", path, cp], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL {name} {corpus_name}: cancho-gpu refused: {r.stderr[:120]}")
            bad += 1
            continue
        lines = [l for l in r.stdout.split("\n") if l.startswith("ids ")]
        differing = []
        for i, rec in enumerate(records):
            body = lines[i].split(": ", 1)[1].strip()
            mine = [int(x) for x in body.split()] if body else []
            if mine != tok.encode(rec, add_special_tokens=False).ids:
                differing.append(i)
        if isinstance(pinned, list):
            good = differing == pinned
            what = f"records {differing}"
        else:
            good = len(differing) == pinned
            what = f"{len(differing)} of {len(records)} records"
        print(f"{'ok  ' if good else 'FAIL'} {name} {corpus_name}: differs from Hugging Face in {what}" + ("" if good else f" (pinned: {pinned})"))
        bad += 0 if good else 1
if ran == 0:
    print("skip tokenizer_hf: no real tokenizer found")
sys.exit(1 if bad else 0)
