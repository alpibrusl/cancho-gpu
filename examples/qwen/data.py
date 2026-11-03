#!/usr/bin/env python3
"""A fixed batch of real text for the Qwen checks (docs/resident.md section 20): the Markdown, Rust and cancho files of this repository and of its sibling `cancho` checkout
(when there is one), in sorted path order, tokenized with Qwen's own tokenizer, cut into `nb` sequences of `t` tokens taken at evenly spaced places of the stream. No download."""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODEL = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B")


def snapshot(model=MODEL):
    return glob.glob(os.path.join(model, "snapshots", "*"))[0]


def corpus_files():
    roots = [ROOT]
    sib = os.path.join(os.path.dirname(ROOT), "cancho")
    if os.path.isdir(sib):
        roots.append(sib)
    files = []
    for r in roots:
        for ext in ("md", "rs", "cho"):
            files += glob.glob(os.path.join(r, "**", f"*.{ext}"), recursive=True)
    return sorted(f for f in files if "/target/" not in f and "/build/" not in f and "/.git/" not in f and "/node_modules/" not in f)


BATCH = os.path.join(HERE, "batch_8x512.npy")  # the batch the numbers of docs/resident.md section 20 were measured on (the corpus changes as the repositories do)


def fixed_batch(nb=8, t=512, model=MODEL, limit_bytes=6_000_000):
    """The committed batch (examples/qwen/batch_8x512.npy) when it has the shape asked for, else a new one from the files below."""
    if os.path.exists(BATCH):
        b = np.load(BATCH).astype(np.int64)
        if b.shape == (nb, t):
            return b
        if b.shape[0] * b.shape[1] >= nb * t and nb * t > 0 and (nb, t) != (b.shape[0], b.shape[1]):
            return b.reshape(-1)[:nb * t].reshape(nb, t)
    return build_batch(nb, t, model, limit_bytes)


def build_batch(nb=8, t=512, model=MODEL, limit_bytes=6_000_000):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(snapshot(model), "tokenizer.json"))
    text, used = [], 0
    for f in corpus_files():
        try:
            s = open(f, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        text.append(s)
        used += len(s)
        if used > limit_bytes:
            break
    ids = np.array(tok.encode("\n\n".join(text)).ids, dtype=np.int64)
    if len(ids) < nb * t:
        raise SystemExit(f"only {len(ids)} tokens of text, {nb * t} needed")
    starts = (np.arange(nb) * ((len(ids) - t) // max(nb - 1, 1))).astype(np.int64)
    return np.stack([ids[s:s + t] for s in starts])


def train_batches(k, nb=8, t=512, model=MODEL, limit_bytes=6_000_000):
    """`k` different batches of `nb` sequences of `t` tokens for a short training run (docs/resident.md section 21): the same corpus as `build_batch`, window j of batch i starting at
    (i + j k) / (k nb) of the stream, so no two windows overlap more than by their position; batch 0 is not the fixed batch. Needs the tokenizer files of the model; no download."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(snapshot(model), "tokenizer.json"))
    text, used = [], 0
    for f in corpus_files():
        try:
            s = open(f, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        text.append(s)
        used += len(s)
        if used > limit_bytes:
            break
    ids = np.array(tok.encode("\n\n".join(text)).ids, dtype=np.int64)
    span = len(ids) - t
    return np.stack([np.stack([ids[int((i + j * k) / (k * nb) * span):][:t] for j in range(nb)]) for i in range(k)])


if __name__ == "__main__":
    if "--rebuild" in sys.argv:
        b = build_batch()
        np.save(BATCH, b.astype(np.int32))
    b = fixed_batch()
    print(b.shape, b[0, :12], len(corpus_files()), "files")
