#!/usr/bin/env python3
"""A byte-pair-encoding tokenizer trained here, on text already on this machine (docs/resident.md section 17): ASCII only; the text is split into words (a run of letters, digits or one
other character, with the space before it), merges are learned over the table of distinct words with their counts, and a token is a string. No download.

    ids, vocab = bpe.corpus_ids(2048)   # the ids of the corpus, cached in build/gpt, and the token strings"""
import collections
import glob
import hashlib
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORD = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9 ]|\s")
SKIP = ("/target/", "/node_modules/", "/.git/", "/golden/", "/build/", "/.venv/", "/venv/", "/site-packages/", "/dist/", "/vendor/")


def gather(limit=40_000_000):
    """The ASCII text of the Markdown, Rust and cancho files under the parent of this repository, each distinct file once."""
    text, seen, total = [], set(), 0
    base = os.path.dirname(ROOT)
    for ext in ("md", "cho", "rs"):
        for path in sorted(glob.glob(os.path.join(base, "**", f"*.{ext}"), recursive=True)):
            if any(s in path for s in SKIP):
                continue
            try:
                raw = open(path, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            key = hashlib.md5(raw.encode("utf-8", "ignore")).digest()
            if key in seen or len(raw) < 200:
                continue
            seen.add(key)
            s = "".join(ch for ch in raw if ch == "\n" or 32 <= ord(ch) < 127)
            text.append(s)
            total += len(s)
            if total >= limit:
                return "\n".join(text)
    return "\n".join(text)


def train(text, size):
    words = collections.Counter(WORD.findall(text))
    seqs = {w: tuple(w) for w in words}
    chars = sorted({ch for w in words for ch in w})
    vocab = list(chars)
    pairs = collections.Counter()
    where = collections.defaultdict(set)
    for w, s in seqs.items():
        for a, b in zip(s, s[1:]):
            pairs[a, b] += words[w]
            where[a, b].add(w)
    merges = []
    while len(vocab) < size and pairs:
        (a, b), cnt = pairs.most_common(1)[0]
        if cnt < 2:
            break
        new = a + b
        merges.append((a, b))
        vocab.append(new)
        for w in list(where[a, b]):
            s = seqs[w]
            f = words[w]
            for x, y in zip(s, s[1:]):
                pairs[x, y] -= f
                where[x, y].discard(w) if pairs[x, y] <= 0 else None
            out, i = [], 0
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    out.append(new)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            seqs[w] = tuple(out)
            for x, y in zip(out, out[1:]):
                pairs[x, y] += f
                where[x, y].add(w)
        pairs = collections.Counter({k: v for k, v in pairs.items() if v > 0})
    return vocab, merges


def encode(text, vocab, merges):
    rank = {m: i for i, m in enumerate(merges)}
    index = {t: i for i, t in enumerate(vocab)}
    cache, ids = {}, []
    for w in WORD.findall(text):
        if w not in cache:
            s = list(w)
            while len(s) > 1:
                best = min(((rank.get((x, y), 1 << 30), i) for i, (x, y) in enumerate(zip(s, s[1:]))))
                if best[0] == 1 << 30:
                    break
                i = best[1]
                s[i:i + 2] = [s[i] + s[i + 1]]
            cache[w] = [index[t] for t in s]
        ids.extend(cache[w])
    return np.array(ids, dtype=np.int64)


def corpus_ids(size, limit=40_000_000):
    path = os.path.join(ROOT, "build", "gpt", f"bpe{size}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["ids"], [str(t) for t in z["vocab"]]
    text = gather(limit)
    vocab, merges = train(text[: min(len(text), 8_000_000)], size)
    ids = encode(text, vocab, merges)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, ids=ids, vocab=np.array(vocab), nchars=len(text))
    return ids, vocab


if __name__ == "__main__":
    import sys, time
    t0 = time.time()
    ids, vocab = corpus_ids(int(sys.argv[1]) if len(sys.argv) > 1 else 2048)
    print(f"{len(ids)} tokens, vocabulary {len(vocab)}, {time.time() - t0:.0f} s; longest tokens: {sorted(vocab, key=len)[-5:]}")
