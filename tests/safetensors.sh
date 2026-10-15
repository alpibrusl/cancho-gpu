#!/bin/sh
# The safetensors reader (docs/runtime.md §7-8):
#
#   * each fixture of tests/safetensors/*.safetensors dumps byte-identically
#     with the dump the `safetensors` library's reader gives
#     (golden/safetensors/, from tools/safetensors_dump.py);
#   * each fixture of tests/safetensors/reject/ is refused, with the rule
#     its name starts with;
#   * every `st-` rule in the sources has such a fixture;
#   * hostile bytes (tests/safetensors/fuzz.py) never trap.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${CANCHO_GPU:-$ROOT/build/cancho-gpu}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
fail=0
n=0
for f in "$ROOT"/tests/safetensors/*.safetensors; do
  name=$(basename "$f" .safetensors)
  n=$((n + 1))
  if ! "$BIN" safetensors "$f" > "$WORK/out" 2>"$WORK/err"; then
    echo "$name: refused: $(head -1 "$WORK/err")"; fail=1
  elif ! cmp -s "$WORK/out" "$ROOT/golden/safetensors/$name.txt"; then
    echo "$name: differs from golden/safetensors/$name.txt"; diff "$WORK/out" "$ROOT/golden/safetensors/$name.txt" | head -5; fail=1
  fi
done
rules=$(grep -ho '"st-[a-z-]*"' "$ROOT"/src/safetensors*.cho | tr -d '"' | sort -u)
refused=0
for f in "$ROOT"/tests/safetensors/reject/*.safetensors; do
  name=$(basename "$f" .safetensors)
  rule=""
  for r in $rules; do
    case "$name" in "$r"|"$r"-*) rule=$r ;; esac
  done
  refused=$((refused + 1))
  status=0
  "$BIN" safetensors "$f" >"$WORK/out" 2>"$WORK/err" || status=$?
  if [ "$status" -ne 1 ]; then
    echo "$name: exit $status, expected 1"; fail=1
  elif ! grep -q "^error\[$rule\]" "$WORK/err"; then
    echo "$name: expected error[$rule], got: $(head -1 "$WORK/err")"; fail=1
  elif [ -s "$WORK/out" ]; then
    echo "$name: refused but wrote output"; fail=1
  fi
done
for r in $rules; do
  ok=0
  for f in "$ROOT/tests/safetensors/reject/$r".safetensors "$ROOT/tests/safetensors/reject/$r"-*.safetensors; do
    [ -f "$f" ] && ok=1
  done
  [ "$ok" = 1 ] || { echo "rule \`$r\` has no fixture in tests/safetensors/reject/"; fail=1; }
done
# The size option: a file read whole needs none; a bad one is a usage error.
status=0; "$BIN" safetensors "$ROOT/tests/safetensors/tiny.safetensors" size=abc >/dev/null 2>"$WORK/err" || status=$?
[ "$status" -eq 2 ] || { echo "size=abc: exit $status, expected 2"; fail=1; }
status=0; "$BIN" safetensors "$ROOT/tests/safetensors/nothing.safetensors" >/dev/null 2>"$WORK/err" || status=$?
grep -q '^error\[io\]' "$WORK/err" || { echo "a missing file: expected error[io]"; fail=1; }
python3 "$ROOT/tests/safetensors/fuzz.py" "$BIN" "$ROOT/tests/safetensors/tiny.safetensors" 1000 || fail=1
echo "$n safetensors fixtures match the library's dump, $refused refused"
exit $fail
