#!/bin/sh
# Every fixture under tests/reject/ must be refused, with the rule its
# first line names (`// RULE <tag>`), and with nothing written. A second
# line `// ARGS name=value ...` gives the constants.
#
# And every rule the compiler can refuse with has a fixture here: the
# list of tags is read out of the sources (`err.fail` and the checker's
# `diag`), so a new rule without a fixture fails this script.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${CANCHO_GPU:-$ROOT/build/cancho-gpu}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
fail=0
count=0
for f in "$ROOT"/tests/reject/*.lx; do
  count=$((count + 1))
  rule=$(sed -n '1s|^// RULE ||p' "$f")
  args=$(sed -n '2s|^// ARGS ||p' "$f")
  rm -rf "$WORK/out"; mkdir -p "$WORK/out"
  status=0
  # shellcheck disable=SC2086
  "$BIN" emit "$f" "$WORK/out" $args >/dev/null 2>"$WORK/err" || status=$?
  if [ "$status" -ne 1 ]; then
    echo "$(basename "$f"): exit $status, expected 1"; fail=1
  elif ! grep -q "^error\[$rule\]" "$WORK/err"; then
    echo "$(basename "$f"): expected error[$rule], got: $(head -1 "$WORK/err")"; fail=1
  elif [ -n "$(ls "$WORK/out")" ]; then
    echo "$(basename "$f"): refused but wrote $(ls "$WORK/out")"; fail=1
  fi
done
# Rules that need no fixture: `io` and `usage` are about the command
# line, not a program, and are checked below; `no-schedule` is a file
# with no schedule for either target, also below.
for rule in $(grep -ho 'fail(m, t, "[a-z-]*"\|diag(m, t, c, "[a-z-]*"\|bad(m, t, "[a-z-]*"' "$ROOT"/src/*.cho | grep -o '"[a-z-]*"' | tr -d '"' | sort -u); do
  # `scope` is the checker's defence against an elaborator that bound a
  # name outside its block; the surface has no way to write one (a
  # binding in a loop body is not in the outer environment at all, so it
  # is `unbound` first). Kept, as the Rust keeps it, and exempt here.
  case "$rule" in usage|no-schedule|scope) continue ;; esac
  if [ ! -f "$ROOT/tests/reject/$rule.lx" ]; then
    echo "rule \`$rule\` has no fixture in tests/reject/"; fail=1
  fi
done
# The command line's own refusals.
status=0; "$BIN" >/dev/null 2>&1 || status=$?
[ "$status" -eq 2 ] || { echo "no arguments: exit $status, expected 2"; fail=1; }
status=0; "$BIN" emit "$ROOT/tests/lx/matvec.lx" "$WORK" n=x k=8 >/dev/null 2>"$WORK/err" || status=$?
grep -q '^error\[usage\]' "$WORK/err" || { echo "a bad constant: expected error[usage]"; fail=1; }
status=0; "$BIN" emit "$ROOT/tests/reject/missing.lx" "$WORK" >/dev/null 2>"$WORK/err" || status=$?
grep -q '^error\[io\]' "$WORK/err" || { echo "a missing file: expected error[io]"; fail=1; }
status=0; "$BIN" check "$ROOT/tests/reject/no-schedule.lx.txt" >/dev/null 2>"$WORK/err" || status=$?
grep -q '^error\[no-schedule\]' "$WORK/err" || { echo "no schedule: expected error[no-schedule]"; fail=1; }
echo "$count refusal fixtures checked"
exit $fail
