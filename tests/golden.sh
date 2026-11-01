#!/bin/sh
# Hold cancho-gpu to its recorded goldens (golden/, scripts/record-golden.sh): every case of tests/cases.txt either writes exactly
# the files recorded, byte for byte, or is refused (exit 1) writing exactly the files recorded beside golden/<case>/REFUSED.
# Needs nothing but the built compiler.
#
# These files are a regression net and not an oracle (docs/native.md N2): they were first agreed with lex-gpu's Rust emitter
# (golden/PROVENANCE) and have been recorded from this compiler since. What a kernel computes is held to float64 references by
# tests/kernels.py; a golden that must change is re-recorded and its diff read.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${CANCHO_GPU:-$ROOT/build/cancho-gpu}
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
    echo "$case_name: no golden (run scripts/record-golden.sh)"; fail=$((fail + 1)); continue
  fi
  out=$WORK/$case_name
  mkdir -p "$out"
  status=0
  # shellcheck disable=SC2086
  "$BIN" emit "$ROOT/$file" "$out" $rest >"$WORK/stdout" 2>"$WORK/err" || status=$?
  case " $rest " in *" --run "*) grep -v "^$out/" "$WORK/stdout" > "$out/RUN.txt" || true ;; esac
  if [ -f "$want/REFUSED" ]; then
    # Refused, with exactly the files written before refusing (none, for most; for an over-budget Metal kernel, the CUDA one).
    if [ "$status" -eq 1 ] && diff -r -x REFUSED "$want" "$out" >"$WORK/diff"; then
      pass=$((pass + 1))
    else
      echo "$case_name: recorded as refused (exit 1, writing what the golden holds); cancho-gpu exited $status"; head -5 "$WORK/diff"; fail=$((fail + 1))
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
