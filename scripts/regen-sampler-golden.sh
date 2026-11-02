#!/bin/sh
# Record golden/sampler/ with the Python implementation of the sampler: for each line of tests/sampler/cases.txt, the dump
# tools/sampler_dump.py prints. Needs python3 and numpy.
#
#     scripts/regen-sampler-golden.sh
#
# The goldens were first made by lex-rt's Rust sampler (golden/sampler/PROVENANCE); tools/sampler_dump.py reproduces them
# (picks and ids identical, every probability bit-identical on the platform in golden/sampler/PLATFORM: tests/oracles.py).
# They depend on the platform's `exp` (docs/runtime.md section 10). docs/native.md N2f.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
echo "$(uname -sm), numpy $(python3 -c 'import numpy; print(numpy.__version__)')" > "$ROOT/golden/sampler/PLATFORM"
n=0
while read -r file vocab temp top_p top_k seed; do
  [ -n "$file" ] || continue
  n=$((n + 1))
  python3 "$ROOT/tools/sampler_dump.py" "$ROOT/tests/sampler/$file" "$vocab" "$temp" "$top_p" "$top_k" "$seed" > "$ROOT/golden/sampler/$(printf '%03d' "$n").txt"
done < "$ROOT/tests/sampler/cases.txt"
echo "golden/sampler/: $n cases from tools/sampler_dump.py"
