#!/bin/sh
# The BPE tokenizer (docs/runtime.md §9):
#
#   * each tokenizer fixture of tests/tokenizer/ (three tokenizer.json forms
#     and a GGUF) gives, over the hand-written corpus and over 500 random
#     records, exactly the ids and decoded text recorded in golden/tokenizer/
#     (a regression record: made by lex-rt's Rust tokenizer first, not confirmed by an
#     independent one; tests/tokenizer_hf.py compares with Hugging Face, issue #51);
#   * each fixture of tests/tokenizer/reject/ is refused with the rule its
#     name starts with, and so is text that is not UTF-8;
#   * every `tok-` rule in the sources has such a fixture;
#   * hostile tokenizers and text (tests/tokenizer/fuzz.py) never trap.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${CANCHO_GPU:-$ROOT/build/cancho-gpu}
D=$ROOT/tests/tokenizer
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
fail=0
n=0
for f in "$D"/*.json "$D"/*.gguf; do
  name=$(basename "$f")
  for kind in "" ".random"; do
    corpus=$D/corpus.txt
    [ "$kind" = ".random" ] && corpus=$D/corpus-random.txt
    n=$((n + 1))
    if ! "$BIN" tokenize "$f" "$corpus" > "$WORK/out" 2>"$WORK/err"; then
      echo "$name$kind: refused: $(head -1 "$WORK/err")"; fail=1
    elif ! cmp -s "$WORK/out" "$ROOT/golden/tokenizer/$name$kind.txt"; then
      echo "$name$kind: differs from golden/tokenizer/$name$kind.txt"; diff "$WORK/out" "$ROOT/golden/tokenizer/$name$kind.txt" | head -4; fail=1
    fi
  done
done
rules=$(grep -ho '"tok-[a-z-]*"' "$ROOT"/src/bpe.cho "$ROOT"/src/tokenize.cho | tr -d '"' | sort -u)
refused=0
for f in "$D"/reject/*; do
  name=$(basename "$f")
  name=${name%.*}
  rule=""
  for r in $rules; do
    case "$name" in "$r"|"$r"-*) rule=$r ;; esac
  done
  refused=$((refused + 1))
  status=0
  "$BIN" tokenize "$f" "$D/corpus.txt" >"$WORK/out" 2>"$WORK/err" || status=$?
  if [ "$status" -ne 1 ]; then
    echo "$name: exit $status, expected 1"; fail=1
  elif ! grep -q "^error\[$rule\]" "$WORK/err"; then
    echo "$name: expected error[$rule], got: $(head -1 "$WORK/err")"; fail=1
  elif [ -s "$WORK/out" ]; then
    echo "$name: refused but wrote output"; fail=1
  fi
done
# Text that is not UTF-8.
printf 'good\n%%%%\nbad \377\376 bytes' > "$WORK/bad.txt"
status=0; "$BIN" tokenize "$D/toy-pairs.json" "$WORK/bad.txt" >"$WORK/out" 2>"$WORK/err" || status=$?
{ [ "$status" -eq 1 ] && grep -q '^error\[tok-utf8\]' "$WORK/err" && [ ! -s "$WORK/out" ]; } || { echo "bad UTF-8 text: expected error[tok-utf8]"; fail=1; }
for r in $rules; do
  ok=0
  for f in "$D/reject/$r".* "$D/reject/$r"-*; do
    [ -f "$f" ] && ok=1
  done
  [ "$r" = tok-utf8 ] && ok=1
  [ "$ok" = 1 ] || { echo "rule \`$r\` has no fixture in tests/tokenizer/reject/"; fail=1; }
done
status=0; "$BIN" tokenize "$D/nothing.json" "$D/corpus.txt" >/dev/null 2>"$WORK/err" || status=$?
grep -q '^error\[io\]' "$WORK/err" || { echo "a missing tokenizer: expected error[io]"; fail=1; }
python3 "$D/fuzz.py" "$BIN" "$D/toy-pairs.json" "$D/toy.gguf" "$D/corpus.txt" 250 || fail=1
echo "$n tokenizer runs match the recorded goldens, $refused fixtures refused"
exit $fail
