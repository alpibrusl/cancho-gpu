#!/bin/sh
# The loops of docs/native.md section 1: the least-squares loss and gradient at 4096 x 128 repeated 2000 times, in cancho and in C,
# so the time per multiply-add is the language's and the compiler's and not an interpreter's. Each prints its result as an exit status
# (the work cannot be removed), and the two languages must agree on it.
#
#     CANCHO=<compiler> bench/native/run.sh
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
CANCHO=${CANCHO:-cancho}
CC=${CC:-clang}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
for v in f64 f32 f32_free; do
  CANCHO_COMPILER=auto "$CANCHO" build "$HERE/lsq_$v.cho" --std -o "$W/cho_$v" >/dev/null 2>&1
  "$CC" -O2 "$HERE/lsq_$v.c" -o "$W/c_$v"
done
python3 - "$W" <<'P'
import subprocess, sys, time, statistics
w = sys.argv[1]
print(f"{'program':28s} {'exit':>4} {'ns per multiply-add':>20}")
for v, what in (("f64", "plain f64 loops"), ("f32", "plain f32 loops"), ("f32_free", "f32, free summation order")):
    for lang in ("cho", "c"):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = subprocess.run([f"{w}/{lang}_{v}"])
            ts.append(time.perf_counter() - t0)
        print(f"{lang + ' ' + what:28s} {r.returncode:4d} {statistics.median(ts) * 1e9 / (2000 * 1048576):20.3f}")
P
