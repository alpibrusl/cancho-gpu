#!/bin/sh
# The sampler (docs/runtime.md §10): each case of tests/sampler/cases.txt
# (the logits file, vocab, temperature, top_p, top_k and seed `cancho-gpu
# sample` is given) must give the picks and nucleus ids lex-rt's sampler
# gives, and probabilities within a few units in the last place
# (tests/sampler/compare.py says why they are not all identical), against
# golden/sampler/. And bad arguments and files are refused, not trapped.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${CANCHO_GPU:-$ROOT/build/cancho-gpu}
D=$ROOT/tests/sampler
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
fail=0
n=0
rows=0; probs=0; same=0; worst=0
while read -r file vocab temp top_p top_k seed; do
  [ -n "$file" ] || continue
  n=$((n + 1))
  id=$(printf '%03d' "$n")
  if ! "$BIN" sample "$D/$file" "$vocab" "$temp" "$top_p" "$top_k" "$seed" > "$WORK/out" 2>"$WORK/err"; then
    echo "case $n ($file $vocab $temp $top_p $top_k $seed): refused: $(head -1 "$WORK/err")"; fail=1; continue
  fi
  if ! out=$(python3 "$D/compare.py" "$WORK/out" "$ROOT/golden/sampler/$id.txt" 8); then
    echo "case $n ($file $vocab $temp $top_p $top_k $seed): $out"; fail=1; continue
  fi
  set -- $out
  rows=$((rows + $1)); probs=$((probs + $3)); same=$((same + $5))
  [ "$8" -gt "$worst" ] && worst=$8
done < "$D/cases.txt"
# Refusals.
status=0; "$BIN" sample "$D/vocab-2.f32" 3 1 1 1 1 >/dev/null 2>"$WORK/err" || status=$?
{ [ "$status" -eq 1 ] && grep -q '^error\[sample-rows\]' "$WORK/err"; } || { echo "a file that is not whole rows: expected error[sample-rows]"; fail=1; }
status=0; "$BIN" sample "$D/vocab-2.f32" 2 x 1 1 1 >/dev/null 2>"$WORK/err" || status=$?
[ "$status" -eq 2 ] || { echo "a bad temperature: exit $status, expected 2"; fail=1; }
status=0; "$BIN" sample "$D/vocab-2.f32" 0 1 1 1 1 >/dev/null 2>"$WORK/err" || status=$?
[ "$status" -eq 2 ] || { echo "vocab 0: exit $status, expected 2"; fail=1; }
status=0; "$BIN" sample "$D/vocab-2.f32" 2 1 1 -1 1 >/dev/null 2>"$WORK/err" || status=$?
[ "$status" -eq 2 ] || { echo "a negative top_k: exit $status, expected 2"; fail=1; }
status=0; "$BIN" sample "$D/nothing.f32" 2 1 1 1 1 >/dev/null 2>"$WORK/err" || status=$?
grep -q '^error\[io\]' "$WORK/err" || { echo "a missing file: expected error[io]"; fail=1; }
: > "$WORK/empty.f32"
status=0; "$BIN" sample "$WORK/empty.f32" 2 1 1 1 1 >/dev/null 2>"$WORK/err" || status=$?
[ "$status" -eq 1 ] || { echo "an empty file: exit $status, expected 1"; fail=1; }
echo "$n sampler cases match lex-rt's: $rows rows, $probs probabilities, $same bit-identical, worst $worst ulps"
exit $fail
