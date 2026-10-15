#!/usr/bin/env python3
"""Compare two sampler dumps (`cancho-gpu sample` and the oracle's).

    compare.py <ours> <theirs> [max ulps]

The picks and the ids of every nucleus must be equal. A probability is an f32;
lex-rt's `exp` is the platform's `expf` and this program's is the correctly
rounded f32 (docs/runtime.md §10), so probabilities may differ by a few units
in the last place: the largest difference allowed is `max ulps` (default 8).
Prints how many probabilities were bit-identical.
"""
import sys

a, b = open(sys.argv[1]).read().splitlines(), open(sys.argv[2]).read().splitlines()
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 8
if len(a) != len(b):
    print(f"{len(a)} rows against {len(b)}")
    sys.exit(1)
total = same = worst = 0
for i, (x, y) in enumerate(zip(a, b)):
    hx, _, rx = x.partition(": ")
    hy, _, ry = y.partition(": ")
    if hx != hy:
        print(f"row {i}: {hx!r} against {hy!r}")
        sys.exit(1)
    px, py = rx.split(), ry.split()
    for u, v in zip(px, py):
        iu, _, pu = u.partition("=")
        iv, _, pv = v.partition("=")
        if iu != iv or (pu == "nan") != (pv == "nan"):
            print(f"row {i}: {u} against {v}")
            sys.exit(1)
        total += 1
        if pu == "nan":
            same += 1
            continue
        d = abs(int(pu, 16) - int(pv, 16))
        same += d == 0
        worst = max(worst, d)
        if d > limit:
            print(f"row {i}: {u} against {v} ({d} ulps)")
            sys.exit(1)
print(f"{len(a)} rows, {total} probabilities, {same} bit-identical, worst {worst} ulps")
