#!/bin/sh
# Record golden/tokenizer/ from this program: for every tokenizer fixture of tests/tokenizer/*.json and *.gguf, the dump of
# `cancho-gpu tokenize` over tests/tokenizer/corpus.txt and corpus-random.txt.
#
#     scripts/record-tokenizer-golden.sh
#
# These are regression records, not an oracle: they were first made by lex-rt's Rust tokenizer (golden/tokenizer/PROVENANCE) and
# `cancho-gpu tokenize` reproduced them. The independent comparison is Hugging Face `tokenizers` on real tokenizers
# (tests/tokenizer_hf.py), and it finds differences (issue #51); the fixtures here are minimal files that library does not read.
# A diff in golden/tokenizer/ after a change is read as text.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${CANCHO_GPU:-$ROOT/build/cancho-gpu}
for f in "$ROOT"/tests/tokenizer/*.json "$ROOT"/tests/tokenizer/*.gguf; do
  [ -f "$f" ] || continue
  "$BIN" tokenize "$f" "$ROOT/tests/tokenizer/corpus.txt" > "$ROOT/golden/tokenizer/$(basename "$f").txt"
  "$BIN" tokenize "$f" "$ROOT/tests/tokenizer/corpus-random.txt" > "$ROOT/golden/tokenizer/$(basename "$f").random.txt"
done
echo "golden/tokenizer/ recorded from this program: review with git diff golden/tokenizer/"
