#!/usr/bin/env python3
r"""`cancho-gpu tokenize` against Hugging Face `tokenizers`, on real tokenizer.json files (docs/native.md N2f, issue #51).

    tokenizer_hf.py [build/cancho-gpu]

`tokenizers` reads each file's own `normalizer`, `pre_tokenizer`, `decoder` and added tokens and knows nothing of this project, so
it is the independent oracle the lex-rt Rust tokenizer was. **They agree.** This test used to pin a long list of differences
(3 of the 37 hand-written records and 211 to 234 of the 500 random ones differed), whose causes were: NFC without decomposition
(`Ω` U+2126, `Å` U+212B); `\s*[\r\n]+` not taking the longest run; `\p{L}`, `\p{M}`, `\p{N}` as approximations; the pattern not read from
the file; added tokens looked for after NFC instead of before it. All fixed. What it holds now: on each real tokenizer found, no
record of the hand-written corpus (37), the random one (500) or a second random one of 1,000 records from a seed the fixing was not
tuned against differs. It is skipped (exit 0) when `tokenizers` or the tokenizer files are missing.
"""
import glob
import os
import subprocess
import sys

BIN = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "build/cancho-gpu")
HOME = os.path.expanduser("~")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# name: candidate tokenizer.json paths
TOKENIZERS = {
    "qwen2.5-0.5b": sorted(glob.glob(HOME + "/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*/tokenizer.json")),
    "qwen3.8-27b": [os.path.realpath(os.path.join(ROOT, "..", "models", "qwen3.8-27b", "tokenizer.json"))],
}

try:
    from tokenizers import Tokenizer
except ImportError:
    print("skip tokenizer_hf: the `tokenizers` library is not installed")
    sys.exit(0)

def second_random_corpus():
    """1,000 random records from a seed (424242) the fixing of issue #51 was not tuned against, written to a temporary file."""
    import importlib.util
    import tempfile
    spec = importlib.util.spec_from_file_location("make_fixtures", os.path.join(ROOT, "tests", "tokenizer", "make_fixtures.py"))
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    f = tempfile.NamedTemporaryFile("wb", delete=False, suffix=".txt")
    f.write(mf.random_corpus(1000, seed=424242))
    f.close()
    return f.name


bad = 0
ran = 0
extra = second_random_corpus()
for name, paths in TOKENIZERS.items():
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print(f"skip {name}: no tokenizer.json found")
        continue
    ran += 1
    path = paths[0]
    tok = Tokenizer.from_file(path)
    for cp in (os.path.join(ROOT, "tests", "tokenizer", "corpus.txt"), os.path.join(ROOT, "tests", "tokenizer", "corpus-random.txt"), extra):
        corpus_name = os.path.basename(cp) if cp != extra else "random seed 424242"
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
        good = not differing
        print(f"{'ok  ' if good else 'FAIL'} {name} {corpus_name}: {len(records)} records, " + ("none differ from Hugging Face" if good else f"{len(differing)} differ from Hugging Face: records {differing[:10]}"))
        bad += 0 if good else 1
os.remove(extra)
if ran == 0:
    print("skip tokenizer_hf: no real tokenizer found")
sys.exit(1 if bad else 0)
