#!/usr/bin/env python3
"""The corpus of the 300-step fine-tuning run (docs/resident.md section 22; roadmap M7.3): the Markdown, Rust and cancho files of this machine's `cancho` and `cancho-gpu` checkouts, tokenized with
Qwen's own tokenizer, packed into sequences of 512 tokens, with a held-out split. No download.

    python3 examples/qwen/finetune_data.py WORKDIR [--train 2400] [--heldout 32] [--roots DIR,DIR]      # writes WORKDIR/corpus.npz and WORKDIR/corpus.txt

* Files with the same content are kept once (the checkouts and their worktrees repeat files), files over 100 KB are left out (generated tables would dominate), and the order is a hash of the content.
* The split is by file *name*, not by file: a file whose base name hashes into 1 of 20 goes to the held-out side with every other file of that name (the copies of one document in two checkouts, edited, are
  near-duplicates, and a held-out loss on a near-duplicate of a training file measures memorisation).
* Each file is its text and then Qwen's end-of-text token; the files of a side are laid end to end and cut into sequences of 512 (the last, short one dropped).
The result depends on the files the machine has when it runs, so it is saved with `corpus.txt` (the files, their sizes and their hashes) and the numbers of the document are for that file."""
import glob
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402

EOS = 151643  # <|endoftext|>
SKIP = ("/target/", "/build/", "/.git/", "/node_modules/", "/.claude/")


def files(roots, max_bytes=100_000):
    """[(path, bytes)] for the distinct contents, in hash order."""
    seen = {}
    for r in roots:
        for ext in ("md", "rs", "cho"):
            for f in sorted(glob.glob(os.path.join(r, "**", f"*.{ext}"), recursive=True)):
                if any(x in f for x in SKIP):
                    continue
                try:
                    b = open(f, "rb").read()
                    b.decode("utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if len(b) > max_bytes or not b.strip():
                    continue
                seen.setdefault(hashlib.sha1(b).hexdigest(), (f, b))
    return [seen[h] for h in sorted(seen)]


def is_heldout(path):
    return int(hashlib.sha1(os.path.basename(path).encode()).hexdigest()[:8], 16) % 20 == 0


def pack(tok, items, nseq, t=512):
    ids = []
    for _, b in items:
        ids += tok.encode(b.decode("utf-8")).ids + [EOS]
        if len(ids) >= nseq * t:
            break
    if len(ids) < nseq * t:
        raise SystemExit(f"only {len(ids)} tokens, {nseq * t} needed")
    return np.array(ids[:nseq * t], dtype=np.int32).reshape(nseq, t)


def build(roots, ntrain=2400, nheld=32, t=512):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(data.snapshot(), "tokenizer.json"))
    fs = files(roots)
    held = [x for x in fs if is_heldout(x[0])]
    train = [x for x in fs if not is_heldout(x[0])]
    return pack(tok, train, ntrain, t), pack(tok, held, nheld, t), fs, held


if __name__ == "__main__":
    arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
    work = sys.argv[1]
    os.makedirs(work, exist_ok=True)
    base = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    roots = arg("--roots", ",".join([os.path.join(base, "cancho"), os.path.join(base, "cancho-gpu")])).split(",")
    tr, he, fs, held = build(roots, int(arg("--train", 2400)), int(arg("--heldout", 32)))
    np.savez(os.path.join(work, "corpus.npz"), train=tr, heldout=he)
    with open(os.path.join(work, "corpus.txt"), "w") as f:
        for path, b in fs:
            f.write(f"{'heldout' if is_heldout(path) else 'train':8s} {hashlib.sha1(b).hexdigest()[:12]} {len(b):7d} {path}\n")
    print(f"{len(fs)} distinct files ({sum(len(b) for _, b in fs) / 1e6:.1f} MB), {len(held)} held out; train {tr.shape} = {tr.size} tokens, held-out {he.shape} = {he.size} tokens; "
          f"roots {roots}")
