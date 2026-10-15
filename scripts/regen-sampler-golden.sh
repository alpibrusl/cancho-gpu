#!/bin/sh
# Record golden/sampler/ from lex-rt's sampler: for each line of
# tests/sampler/cases.txt, the dump tools/sampler_dump.rs prints.
#
#     scripts/regen-sampler-golden.sh <lex-gpu checkout>
#
# The goldens are lex-rt's output, never this program's. They depend on the
# platform's `expf` (docs/runtime.md §10); golden/sampler/PLATFORM says which.
set -eu
LEX_GPU=${1:?usage: regen-sampler-golden.sh <lex-gpu checkout>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EX=$LEX_GPU/crates/lex-rt/examples/sampler_dump.rs
mkdir -p "$LEX_GPU/crates/lex-rt/examples"
cp "$ROOT/tools/sampler_dump.rs" "$EX"
trap 'rm -f "$EX"' EXIT
(cd "$LEX_GPU" && cargo build -q -p lex-rt --example sampler_dump)
rm -rf "$ROOT/golden/sampler"
mkdir -p "$ROOT/golden/sampler"
git -C "$LEX_GPU" rev-parse HEAD > "$ROOT/golden/sampler/LEX_GPU"
echo "$(uname -sm), rustc $(rustc --version | cut -d' ' -f2)" > "$ROOT/golden/sampler/PLATFORM"
n=0
while read -r file vocab temp top_p top_k seed; do
  [ -n "$file" ] || continue
  n=$((n + 1))
  "$LEX_GPU/target/debug/examples/sampler_dump" "$ROOT/tests/sampler/$file" "$vocab" "$temp" "$top_p" "$top_k" "$seed" > "$ROOT/golden/sampler/$(printf '%03d' "$n").txt"
done < "$ROOT/tests/sampler/cases.txt"
echo "golden/sampler/: $n cases from lex-gpu $(cat "$ROOT/golden/sampler/LEX_GPU")"
