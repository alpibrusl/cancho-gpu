#!/usr/bin/env python3
"""Writes the tokenizer fixtures of tests/tokenizer/ (committed; this is how
they were made) and the corpus every one of them is run over.

    python3 tests/tokenizer/make_fixtures.py

A toy byte-level BPE is trained on the corpus itself, so its merges
exercise the splits, the NFC cases and the multi-byte characters; it is then
written as three `tokenizer.json` files (merges as pairs, as strings with
`ignore_merges`, and with every non-ASCII key escaped as \\uXXXX) and as a
GGUF the way llama.cpp stores a Qwen tokenizer.
"""
import importlib.util
import json
import os
import struct
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(HERE, "..", "gguf", "make_fixtures.py"))
gf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gf)


def byte_chars():
    chars, spare = [], 0
    for b in range(256):
        if 0x21 <= b <= 0x7E or 0xA1 <= b <= 0xAC or 0xAE <= b <= 0xFF:
            chars.append(chr(b))
        else:
            chars.append(chr(0x100 + spare))
            spare += 1
    return chars


CHARS = byte_chars()

RECORDS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "I'm sure it's fine; they've said we'll go, you'd agree. DON'T SHOUT, I'LL NOT.",
    "12345 67890 3.14159 1,000,000 0042",
    "a  b   c    d",
    "trailing spaces   ",
    "   leading spaces",
    "tabs\tand\t\ttabs\t \tmixed",
    "line one\nline two\r\nline three\n\n\nafter blank lines\r\n\r\n",
    "  \n  \n x",
    "...!?!?--==++**&&  ((nested))  [brackets] {braces} <angle>",
    "snake_case camelCase kebab-case SCREAMING_CASE path/to/file.txt",
    "caf\u00e9 cafe\u0301 \u00c5ngstr\u00f6m A\u030angstro\u0308m na\u00efve nai\u0308ve",
    "\u0041\u030a \u1e9b\u0323 q\u0307\u0323 o\u0323\u0302 e\u0302\u0323",
    "\u1100\u1161\u11a8 \u1112\u1161\u11ab \uac00\u11a8 \ud55c\uae00",
    "\u4f60\u597d\uff0c\u4e16\u754c\uff01 \u3053\u3093\u306b\u3061\u306f \uc548\ub155\ud558\uc138\uc694",
    "\u043f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440 \u0391\u03b8\u03ae\u03bd\u03b1 \u05e9\u05dc\u05d5\u05dd \u0645\u0631\u062d\u0628\u0627",
    "\u0939\u093f\u0928\u094d\u0926\u0940 \u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35 \u0b95\u0bc6\u0ba9\u0bcd",
    "emoji \U0001f600 \U0001f468\u200d\U0001f469\u200d\U0001f467 \U0001f3f3\ufe0f\u200d\U0001f308 \u2764\ufe0f",
    "\u2460\u2461 \u00bd \u00b2 \u0663\u0664 \u2167 \u0967\u0968\u0969",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>",
    "<|endoftext|><|endoftext|> text <|im_start|><|im_start|>",
    "<|a|b|> then <|a|> and <|a|b <|im|> <|im_end",
    "a\u00a0b\u2003c\u3000d \u200bzero\u200bwidth\u200b",
    "\x01\x02 control \x7f chars \x00 nul \x1b[31mred\x1b[0m",
    "\ufffd replacement \U0010ffff max \u0080 \u009f c1",
    "x" * 300,
    "word " * 120,
    "abc" + "\u0301" * 40 + "def",
    "",
    "\n",
    " ",
    "'",
    "'s'S'T'tre'VE'm'LL'd'x'",
    "1\u00b22\u00b3 x\u2082 H\u2082O 10\u2074",
    "def f(x):\n    return x * 2  # twice\n\nclass A:\n\tpass\n",
    "https://example.com/path?query=1&b=2#frag mailto:a.b@c.org",
]


def corpus_bytes():
    return "\n%%\n".join(RECORDS).encode("utf-8")


def random_corpus(records=500, seed=20261015):
    """Random well-formed text from the places the split and NFC care about:
    letters of several scripts, digits of several kinds, apostrophes, every
    kind of space and newline, combining marks, Hangul jamo and syllables,
    emoji, and the added tokens."""
    state = [seed]

    def rnd(n):
        state[0] = (state[0] * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (state[0] >> 33) % n

    pools = [
        [chr(c) for c in range(0x61, 0x7B)], [chr(c) for c in range(0x41, 0x5B)],
        [chr(c) for c in range(0x30, 0x3A)], list(" " * 6 + "\n\t\r"),
        list("'.,;:!?-_/\\()[]{}<>@#$%^&*+=~`\"|"),
        [chr(c) for c in range(0xC0, 0x180)], [chr(c) for c in range(0x300, 0x370)],
        [chr(c) for c in range(0x1100, 0x1113)] + [chr(c) for c in range(0x1161, 0x1176)] + [chr(c) for c in range(0x11A8, 0x11C3)],
        [chr(c) for c in range(0xAC00, 0xAC00 + 600, 7)], [chr(c) for c in range(0x4E00, 0x4E40)],
        [chr(c) for c in range(0x3041, 0x3060)], [chr(c) for c in range(0x430, 0x450)],
        [chr(c) for c in range(0x5D0, 0x5E0)] + [chr(c) for c in range(0x591, 0x5A0)],
        [chr(c) for c in range(0x627, 0x640)] + [chr(c) for c in range(0x64B, 0x655)],
        [chr(c) for c in range(0x905, 0x915)] + [chr(c) for c in range(0x93A, 0x950)],
        [chr(c) for c in range(0xE01, 0xE20)] + [chr(c) for c in (0xE31, 0xE34, 0xE47)],
        [chr(c) for c in range(0x1F600, 0x1F620)], list("\u00a0\u2000\u2003\u200a\u2028\u2029\u3000\u0085\u200b\u00ad\ufeff"),
        list("\u00b2\u00b3\u00bd\u2160\u2167\u0663\u0664\u0967\u2460\u2082\u2074"),
        [chr(c) for c in (0x212B, 0x2126, 0x1E9B, 0x323, 0x307, 0x302, 0x301, 0x308, 0x30A, 0x1D165, 0x1D16E)],
        list("\ufe0f\u200d\ufffd\u0000\u001b\u007f\u0080\u009f"),
    ]
    specials = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|a|b|>", "<|a|>", "<|im|>", "'s", "'LL", "'re", "'d"]
    out = []
    for _ in range(records):
        parts = []
        for _ in range(1 + rnd(30)):
            if rnd(12) == 0:
                parts.append(specials[rnd(len(specials))])
            else:
                pool = pools[rnd(len(pools))]
                parts.append("".join(pool[rnd(len(pool))] for _ in range(1 + rnd(6))))
        out.append("".join(parts))
    return "\n%%\n".join(r for r in out if "\n%%\n" not in r and not r.endswith("\n%%") and not r.startswith("%%\n")).encode("utf-8")


def train(text, merges):
    """A toy BPE over the byte-level characters of `text`, on pieces that
    are a space plus a word (close enough to the real split to be useful)."""
    words = Counter()
    for piece in text.replace("\n", " \n").split(" "):
        if piece:
            words[tuple(CHARS[b] for b in (" " + piece).encode("utf-8"))] += 1
    out = []
    for _ in range(merges):
        pairs = Counter()
        for w, n in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += n
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        out.append(best)
        merged = {}
        for w, n in words.items():
            ws, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                    ws.append(w[i] + w[i + 1])
                    i += 2
                else:
                    ws.append(w[i])
                    i += 1
            merged[tuple(ws)] = merged.get(tuple(ws), 0) + n
        words = merged
    return out


def tokenizer(merges_as, ignore_merges=None, ensure_ascii=False, specials=True):
    text = "\n%%\n".join(RECORDS)
    merges = train(text, 600)
    vocab = {c: i for i, c in enumerate(CHARS)}
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    added = []
    if specials:
        for c in ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|a|>", "<|a|b|>", "<|im|>"]:
            added.append({"id": len(vocab) + len(added), "content": c, "special": True})
    model = {"type": "BPE", "vocab": vocab,
             "merges": [[a, b] for a, b in merges] if merges_as == "pairs" else [a + " " + b for a, b in merges]}
    if ignore_merges is not None:
        model["ignore_merges"] = ignore_merges
    doc = {"version": "1.0", "added_tokens": added, "normalizer": {"type": "NFC"}, "model": model}
    return json.dumps(doc, ensure_ascii=ensure_ascii, separators=(",", ":")).encode("utf-8"), vocab, merges, added


def gguf_tokenizer(model="gpt2", pre="qwen35", types=True, drop_merges=False):
    _, vocab, merges, added = tokenizer("strings")
    tokens = list(vocab.keys())
    kinds = [1] * len(tokens)
    for a in added:
        tokens.append(a["content"])
        kinds.append(3 if a["content"] != "<|im|>" else 4)
    tokens += ["<unused0>", "<unused1>"]
    kinds += [5, 5]
    kvs = [
        gf.kv("general.architecture", 8, gf.val(8, "qwen35")),
        gf.kv("tokenizer.ggml.model", 8, gf.val(8, model)),
        gf.kv("tokenizer.ggml.tokens", 9, gf.arr(8, [gf.val(8, t) for t in tokens])),
    ]
    if pre is not None:
        kvs.append(gf.kv("tokenizer.ggml.pre", 8, gf.val(8, pre)))
    if types:
        kvs.append(gf.kv("tokenizer.ggml.token_type", 9, gf.arr(5, [gf.val(5, k) for k in kinds])))
    if not drop_merges:
        kvs.append(gf.kv("tokenizer.ggml.merges", 9, gf.arr(8, [gf.val(8, a + " " + b) for a, b in merges])))
    return gf.file(3, kvs, [], b"")


def reject():
    out = {}
    good, _, _, _ = tokenizer("pairs")
    doc = json.loads(good)
    bad = lambda d: json.dumps(d, ensure_ascii=False).encode()
    out["tok-json.json"] = good[:200]
    out["tok-model.json"] = bad({"model": {"type": "WordPiece", "vocab": {}}})
    out["tok-model-missing.json"] = bad({"added_tokens": []})
    out["tok-vocab.json"] = bad({"model": {"type": "BPE", "merges": []}})
    out["tok-vocab-id.json"] = bad({"model": {"type": "BPE", "vocab": {"a": -1}, "merges": []}})
    d = json.loads(good); d["model"]["merges"][3] = 7
    out["tok-merge.json"] = bad(d)
    d = json.loads(good); d["model"]["merges"][3] = "nospacehere"
    out["tok-merge-split.json"] = bad(d)
    d = json.loads(good); d["added_tokens"][0]["content"] = ""
    out["tok-added.json"] = bad(d)
    out["tok-gguf-model.gguf"] = gguf_tokenizer(model="llama")
    out["tok-gguf-pre.gguf"] = gguf_tokenizer(pre="llama-bpe")
    out["tok-gguf-pre-missing.gguf"] = gguf_tokenizer(pre=None)
    out["tok-gguf-missing.gguf"] = gguf_tokenizer(drop_merges=True)
    return out


if __name__ == "__main__":
    with open(os.path.join(HERE, "corpus.txt"), "wb") as f:
        f.write(corpus_bytes())
    with open(os.path.join(HERE, "corpus-random.txt"), "wb") as f:
        f.write(random_corpus())
    for name, kw in (("toy-pairs.json", dict(merges_as="pairs")),
                     ("toy-strings.json", dict(merges_as="strings", ignore_merges=True)),
                     ("toy-ascii.json", dict(merges_as="pairs", ensure_ascii=True, ignore_merges=False))):
        with open(os.path.join(HERE, name), "wb") as f:
            f.write(tokenizer(**kw)[0])
    with open(os.path.join(HERE, "toy.gguf"), "wb") as f:
        f.write(gguf_tokenizer())
    os.makedirs(os.path.join(HERE, "reject"), exist_ok=True)
    for name, data in reject().items():
        with open(os.path.join(HERE, "reject", name), "wb") as f:
            f.write(data)
