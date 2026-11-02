#!/usr/bin/env python3
"""A small text corpus with no download: the Markdown documents of a directory tree, ASCII only, as character ids (docs/resident.md section 15)."""
import glob
import os

import numpy as np


def load(roots, limit=4_000_000):
    text = []
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "*.md"), recursive=True)):
            with open(path, encoding="utf-8", errors="ignore") as f:
                text.append(f.read())
    s = "\n".join(text)
    s = "".join(ch for ch in s if ch == "\n" or 32 <= ord(ch) < 127)[:limit]
    vocab = sorted(set(s))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    ids = np.array([stoi[ch] for ch in s], dtype=np.int64)
    return ids, vocab


def batches(ids, t, b, steps, seed):
    """`steps` batches of b windows of t+1 characters at random places: tokens and the characters that follow them."""
    rng = np.random.RandomState(seed)
    starts = rng.randint(0, len(ids) - t - 1, size=(steps, b))
    x = np.stack([[ids[s:s + t] for s in row] for row in starts])
    y = np.stack([[ids[s + 1:s + t + 1] for s in row] for row in starts])
    return x.reshape(steps, b * t), y.reshape(steps, b * t)
