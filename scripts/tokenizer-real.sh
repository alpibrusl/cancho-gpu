#!/bin/sh
# Compare `cancho-gpu tokenize` with Hugging Face `tokenizers` on real tokenizer.json files and a corpus (not in the gate: see
# tests/tokenizer_hf.py, which pins the known differences for the two it finds).
#
#     scripts/tokenizer-real.sh <tokenizer.json> <corpus>...
#
# The corpus is records separated by a line holding only `%%`. Prints, for each pair, how many records differ. Needs the
# `tokenizers` library. The differences found so far are issue #51 (NFC singletons, `\s*[\r\n]+`, character classes).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
tok=${1:?usage: tokenizer-real.sh <tokenizer.json> <corpus>...}
shift
for c in "$@"; do
  "$ROOT/build/cancho-gpu" tokenize "$tok" "$c" > "${TMPDIR:-/tmp}/tok-real.a" 2>&1 || { echo "REFUSED  $c"; continue; }
  python3 "$ROOT/tools/tokenizer_dump.py" "$tok" "$c" > "${TMPDIR:-/tmp}/tok-real.b" 2>&1 || { echo "ORACLE REFUSED  $c"; continue; }
  total=$(grep -c '^ids ' "${TMPDIR:-/tmp}/tok-real.a")
  differ=$(diff "${TMPDIR:-/tmp}/tok-real.a" "${TMPDIR:-/tmp}/tok-real.b" | grep -c '^< ids ' || true)
  echo "$c: $differ of $total records differ from Hugging Face"
done
