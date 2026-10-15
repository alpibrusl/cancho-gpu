#!/bin/sh
# Build the oracle of `cancho-gpu tokenize` (lex-rt's tokenizer, through
# tools/tokenizer_dump.rs) and record golden/tokenizer/ from it: for every
# tokenizer fixture of tests/tokenizer/*.json and *.gguf, the dump over
# tests/tokenizer/corpus.txt and corpus-random.txt.
#
#     scripts/regen-tokenizer-golden.sh <lex-gpu checkout>
#
# The goldens are lex-rt's output, never this program's;
# golden/tokenizer/LEX_GPU records the commit. The example is copied into
# the checkout for the build and removed after.
set -eu
LEX_GPU=${1:?usage: regen-tokenizer-golden.sh <lex-gpu checkout>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EX=$LEX_GPU/crates/lex-rt/examples/tokenizer_dump.rs
mkdir -p "$LEX_GPU/crates/lex-rt/examples"
cp "$ROOT/tools/tokenizer_dump.rs" "$EX"
trap 'rm -f "$EX"' EXIT
(cd "$LEX_GPU" && cargo build -q -p lex-rt --example tokenizer_dump)
rm -rf "$ROOT/golden/tokenizer"
mkdir -p "$ROOT/golden/tokenizer"
git -C "$LEX_GPU" rev-parse HEAD > "$ROOT/golden/tokenizer/LEX_GPU"
for f in "$ROOT"/tests/tokenizer/*.json "$ROOT"/tests/tokenizer/*.gguf; do
  [ -f "$f" ] || continue
  "$LEX_GPU/target/debug/examples/tokenizer_dump" "$f" "$ROOT/tests/tokenizer/corpus.txt" > "$ROOT/golden/tokenizer/$(basename "$f").txt"
  "$LEX_GPU/target/debug/examples/tokenizer_dump" "$f" "$ROOT/tests/tokenizer/corpus-random.txt" > "$ROOT/golden/tokenizer/$(basename "$f").random.txt"
done
echo "golden/tokenizer/ regenerated from lex-gpu $(cat "$ROOT/golden/tokenizer/LEX_GPU")"
