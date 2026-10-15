#!/bin/sh
# Hold cancho-gpu to the committed goldens (golden/, the Rust emitter's
# output, scripts/regen-golden.sh): every case of tests/cases.txt either
# writes exactly the golden files, byte for byte, or is refused where the
# Rust refused. Needs nothing but the built compiler.
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
    echo "$case_name: no golden (run scripts/regen-golden.sh)"; fail=$((fail + 1)); continue
  fi
  out=$WORK/$case_name
  mkdir -p "$out"
  status=0
  # shellcheck disable=SC2086
  "$BIN" emit "$ROOT/$file" "$out" $rest >"$WORK/stdout" 2>"$WORK/err" || status=$?
  case " $rest " in *" --run "*) grep -v "^$out/" "$WORK/stdout" > "$out/RUN.txt" || true ;; esac
  if [ -f "$want/REFUSED" ]; then
    # Refused, with exactly the files the Rust wrote before refusing (none,
    # for most; for an over-budget Metal kernel, the CUDA one).
    if [ "$status" -eq 1 ] && diff -r -x REFUSED "$want" "$out" >"$WORK/diff"; then
      pass=$((pass + 1))
    else
      echo "$case_name: the Rust refuses this (exit 1, writing what the golden holds); cancho-gpu exited $status"; head -5 "$WORK/diff"; fail=$((fail + 1))
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
# The lowering paths the Rust does not have: held to what this compiler recorded
# (golden-self/, scripts/record-self-golden.sh), not to an oracle.
spass=0
sfail=0
while read -r file rest; do
  case "$file" in ''|'#'*) continue ;; esac
  # shellcheck disable=SC2086
  case_name=$("$ROOT/scripts/case-name.sh" "$file" $rest)
  want=$ROOT/golden-self/$case_name
  if [ ! -d "$want" ]; then
    echo "$case_name: no recorded output (run scripts/record-self-golden.sh)"; sfail=$((sfail + 1)); continue
  fi
  out=$WORK/self-$case_name
  mkdir -p "$out"
  status=0
  # shellcheck disable=SC2086
  "$BIN" emit "$ROOT/$file" "$out" $rest >"$WORK/stdout" 2>"$WORK/err" || status=$?
  case " $rest " in *" --run "*) grep -v "^$out/" "$WORK/stdout" > "$out/RUN.txt" || true ;; esac
  if [ "$status" -ne 0 ]; then
    echo "$case_name: refused: $(head -1 "$WORK/err")"; sfail=$((sfail + 1))
  elif diff -r "$want" "$out" >"$WORK/diff"; then
    spass=$((spass + 1))
  else
    echo "$case_name: differs from golden-self/"; head -20 "$WORK/diff"; sfail=$((sfail + 1))
  fi
done < "$ROOT/tests/self-cases.txt"
echo "$spass of $((spass + sfail)) self-recorded cases agree"
[ "$fail" -eq 0 ] && [ "$sfail" -eq 0 ]
