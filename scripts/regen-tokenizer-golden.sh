#!/bin/sh
# Regenerate golden/tokenizer/: for every tokenizer fixture of tests/tokenizer/*.json, the dump of Hugging Face `tokenizers`
# (tools/tokenizer_dump.py) over tests/tokenizer/corpus.txt and corpus-random.txt; the one GGUF fixture, which that library cannot
# read, is recorded from `cancho-gpu tokenize` (a regression record). Needs python3 and the `tokenizers` library.
#
#     scripts/regen-tokenizer-golden.sh
#
# After docs/native.md N2f and issue #51 the JSON goldens are independent: they are what Hugging Face prints, and
# `cancho-gpu tokenize` reproduces them. tests/tokenizer_hf.py does the same on real tokenizers.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${CANCHO_GPU:-$ROOT/build/cancho-gpu}
for f in "$ROOT"/tests/tokenizer/*.json; do
  python3 "$ROOT/tools/tokenizer_dump.py" "$f" "$ROOT/tests/tokenizer/corpus.txt" > "$ROOT/golden/tokenizer/$(basename "$f").txt"
  python3 "$ROOT/tools/tokenizer_dump.py" "$f" "$ROOT/tests/tokenizer/corpus-random.txt" > "$ROOT/golden/tokenizer/$(basename "$f").random.txt"
done
for f in "$ROOT"/tests/tokenizer/*.gguf; do
  [ -f "$f" ] || continue
  "$BIN" tokenize "$f" "$ROOT/tests/tokenizer/corpus.txt" > "$ROOT/golden/tokenizer/$(basename "$f").txt"
  "$BIN" tokenize "$f" "$ROOT/tests/tokenizer/corpus-random.txt" > "$ROOT/golden/tokenizer/$(basename "$f").random.txt"
done
echo "golden/tokenizer/ regenerated: the JSON fixtures by Hugging Face, the GGUF fixture by cancho-gpu (a regression record)"
