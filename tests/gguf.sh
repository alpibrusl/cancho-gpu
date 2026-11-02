#!/bin/sh
# The GGUF reader (docs/runtime.md):
#
#   * each fixture of tests/gguf/*.gguf dumps byte-identically with the
#     dump the independent Python reader prints (golden/gguf/, from tools/gguf_dump.py,
#     which tests/oracles.py holds to the goldens);
#   * each fixture of tests/gguf/reject/ is refused, with the rule its
#     name starts with, and so is the one fixture too large to commit;
#   * every `gguf-` rule in the sources has such a fixture;
#   * hostile bytes (tests/gguf/fuzz.py) never trap.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${CANCHO_GPU:-$ROOT/build/cancho-gpu}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
fail=0
n=0
for f in "$ROOT"/tests/gguf/*.gguf; do
  name=$(basename "$f" .gguf)
  n=$((n + 1))
  if ! "$BIN" gguf "$f" --sum > "$WORK/out" 2>"$WORK/err"; then
    echo "$name: refused: $(head -1 "$WORK/err")"; fail=1
  elif ! cmp -s "$WORK/out" "$ROOT/golden/gguf/$name.txt"; then
    echo "$name: differs from golden/gguf/$name.txt"; diff "$WORK/out" "$ROOT/golden/gguf/$name.txt" | head -5; fail=1
  fi
done
rules=$(grep -ho '"gguf-[a-z-]*"' "$ROOT"/src/gguf*.cho | tr -d '"' | sort -u)
refused=0
check_refusal() { # file
  status=0
  "$BIN" gguf "$1" >"$WORK/out" 2>"$WORK/err" || status=$?
  name=$(basename "$1" .gguf)
  rule=""
  for r in $rules; do
    case "$name" in "$r"|"$r"-*) rule=$r ;; esac
  done
  refused=$((refused + 1))
  if [ "$status" -ne 1 ]; then
    echo "$name: exit $status, expected 1"; fail=1
  elif ! grep -q "^error\[$rule\]" "$WORK/err"; then
    echo "$name: expected error[$rule], got: $(head -1 "$WORK/err")"; fail=1
  elif [ -s "$WORK/out" ]; then
    echo "$name: refused but wrote output"; fail=1
  fi
}
python3 "$ROOT/tests/gguf/make_fixtures.py" --big "$WORK"
for f in "$ROOT"/tests/gguf/reject/*.gguf "$WORK/gguf-string.gguf"; do
  check_refusal "$f"
done
for r in $rules; do
  ok=0
  for f in "$ROOT/tests/gguf/reject/$r".gguf "$ROOT/tests/gguf/reject/$r"-*.gguf; do
    [ -f "$f" ] && ok=1
  done
  [ "$r" = gguf-string ] && ok=1
  # Needs the bytes after the header, so only `--sum` finds it.
  [ "$r" = gguf-tensor-range ] && ok=1
  [ "$ok" = 1 ] || { echo "rule \`$r\` has no fixture in tests/gguf/reject/"; fail=1; }
done
# gguf-tensor-range: a file cut inside its tensor data.
head -c 1500 "$ROOT/tests/gguf/tiny.gguf" > "$WORK/cut.gguf"
status=0; "$BIN" gguf "$WORK/cut.gguf" --sum >/dev/null 2>"$WORK/err" || status=$?
[ "$status" -eq 1 ] && grep -q '^error\[gguf-tensor-range\]' "$WORK/err" || { echo "a cut tensor: expected error[gguf-tensor-range]"; fail=1; }
status=0; "$BIN" gguf "$ROOT/tests/gguf/nothing.gguf" >/dev/null 2>"$WORK/err" || status=$?
grep -q '^error\[io\]' "$WORK/err" || { echo "a missing file: expected error[io]"; fail=1; }
python3 "$ROOT/tests/gguf/fuzz.py" "$BIN" "$ROOT/tests/gguf/tiny.gguf" 1000 || fail=1
echo "$n GGUF fixtures match the independent dump, $refused refused"
exit $fail
