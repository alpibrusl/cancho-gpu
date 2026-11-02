#!/usr/bin/env python3
"""The oracle of `cancho-gpu sample`: a second implementation of the sampler's algorithm, in Python, printing the form of
`src/sample.cho` (docs/runtime.md section 10).

    sampler_dump.py <logits.f32> <vocab> <temperature> <top_p> <top_k> <seed>

The sampler is an *algorithm*, not a function of mathematics: temperature, then top-k, then top-p, renormalised; a draw from
a `xorshift64*` stream seeded through `splitmix64`; temperature 0 or top-k 1 is the argmax (the last of equal maxima); logits
are ordered by `total_cmp`, largest first, ties by index. This implements that description with exact 64-bit integer
arithmetic for the stream and `numpy.float32` for every probability, in the order the algorithm gives, and shares no code with
`cancho-gpu` or with the Rust it began as. It replaced `tools/sampler_dump.rs` (lex-rt's `Sampler`) and agrees with every
golden that dumper made (picks and nucleus ids identical, probabilities within the ulps `tests/sampler/compare.py` allows):
`docs/native.md` N2f. The one place it cannot match to the bit is `exp`, which here is numpy's and there was the platform's
`expf`.
"""
import struct
import sys

import numpy as np

M64 = (1 << 64) - 1
f32 = np.float32


def mix(seed):
    """splitmix64 finalised, with the low bit forced so the state is never zero."""
    z = (seed + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return (z ^ (z >> 31)) | 1


class Sampler:
    def __init__(self, temperature, top_p, top_k, seed):
        self.temperature, self.top_p, self.top_k, self.state = f32(temperature), f32(top_p), max(top_k, 1), mix(seed)

    def next_unit(self):
        x = self.state
        x ^= x >> 12
        x ^= (x << 25) & M64
        x ^= x >> 27
        self.state = x
        v = ((x * 0x2545F4914F6CDD1D) & M64) >> 11
        return f32(v) / f32(1 << 53)  # v < 2^53 is exact in binary64, so one rounding to f32

    def nucleus(self, logits):
        keys = [total_key(x) for x in logits]
        if self.temperature <= 0 or self.top_k == 1:
            best = 0
            for i in range(len(logits)):  # max_by keeps the last of equal maxima
                if keys[i] >= keys[best]:
                    best = i
            return [best], [f32(1.0)]
        k = min(self.top_k, len(logits))
        idx = sorted(range(len(logits)), key=lambda i: (-keys[i], i))[:k]
        top = logits[idx[0]]
        p = [np.exp((logits[i] - top) / self.temperature) for i in idx]
        total = f32(0.0)
        for x in p:
            total = f32(total + x)
        p = [f32(x / total) for x in p]
        cut, run = len(p), f32(0.0)
        for j, x in enumerate(p):
            run = f32(run + x)
            if run >= self.top_p:
                cut = j + 1
                break
        idx, p = idx[:cut], p[:cut]
        kept = f32(0.0)
        for x in p:
            kept = f32(kept + x)
        return idx, [f32(x / kept) for x in p]

    def draw(self, idx, p):
        r = self.next_unit()
        acc = f32(0.0)
        for j, x in enumerate(p):
            acc = f32(acc + x)
            if acc > r:
                return idx[j]
        return idx[-1]

    def pick(self, logits):
        idx, p = self.nucleus(logits)
        return self.draw(idx, p)


def total_key(x):
    """f32::total_cmp as an integer: NaN, infinities and signed zeros ordered by bit pattern."""
    b = struct.unpack("<i", struct.pack("<f", x))[0]
    return b ^ (((b >> 31) & 0xFFFFFFFF) >> 1)


def main():
    path, vocab = sys.argv[1], int(sys.argv[2])
    s = Sampler(float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    raw = open(path, "rb").read()
    words = np.frombuffer(raw[: len(raw) // 4 * 4], dtype="<f4")
    with np.errstate(all="ignore"):
        for row in range(len(words) // vocab):
            logits = [f32(x) for x in words[row * vocab:(row + 1) * vocab]]
            ids, probs = s.nucleus(logits)
            pick = s.pick(logits)
            parts = [("%d=nan" % i) if p != p else "%d=%08x" % (i, struct.unpack("<I", struct.pack("<f", p))[0]) for i, p in zip(ids, probs)]
            print("row %d pick %d nucleus %d: %s" % (row, pick, len(ids), " ".join(parts)))


main()
