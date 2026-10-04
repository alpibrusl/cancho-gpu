#!/bin/sh
# Hold lexsys-gpu to the committed goldens (golden/, the Rust emitter's
# output, scripts/regen-golden.sh): every case of tests/cases.txt either
# writes exactly the golden files, byte for byte, or is refused where the
# Rust refused. Needs nothing but the built compiler.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${LEXSYS_GPU:-$ROOT/build/lexsys-gpu}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
pass=0
fail=0
while read -r file rest; do
  case "$file" in ''|'#'*) continue ;; esac
  # shellcheck disable=SC2086
  case_name=$("$ROOT/scripts/case-name.sh" "$file" $rest)
  want=$ROOT/golden/$case_name
  if [ ! -d "$want" ]; then
    echo "$case_name: no golden (run scripts/regen-golden.sh)"; fail=$((fail + 1)); continue
  fi
  out=$WORK/$case_name
  mkdir -p "$out"
  status=0
  # shellcheck disable=SC2086
  "$BIN" emit "$ROOT/$file" "$out" $rest >"$WORK/stdout" 2>"$WORK/err" || status=$?
  case " $rest " in *" --run "*) grep -v "^$out/" "$WORK/stdout" > "$out/RUN.txt" || true ;; esac
  if [ -f "$want/REFUSED" ]; then
    if [ "$status" -eq 1 ] && [ -z "$(ls "$out")" ]; then
      pass=$((pass + 1))
    else
      echo "$case_name: the Rust refuses this, lexsys-gpu exited $status"; fail=$((fail + 1))
    fi
  elif [ "$status" -ne 0 ]; then
    echo "$case_name: refused: $(head -1 "$WORK/err")"; fail=$((fail + 1))
  elif diff -r "$want" "$out" >"$WORK/diff"; then
    pass=$((pass + 1))
  else
    echo "$case_name: differs from the golden"; head -20 "$WORK/diff"; fail=$((fail + 1))
  fi
done < "$ROOT/tests/cases.txt"
echo "$pass of $((pass + fail)) golden cases agree"
[ "$fail" -eq 0 ]
