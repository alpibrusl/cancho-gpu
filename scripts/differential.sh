#!/bin/sh
# Run lex-gpu's Rust emitter and lexsys-gpu over the same `.lx` files and
# constants, and require the same outcome: both refuse, or both write the
# same files with the same bytes.
#
#     scripts/differential.sh <lex-gpu checkout> [cases-file]
#
# A case is one line of `cases-file` (default tests/cases.txt):
# `<file.lx> [name=value ...]`, the path relative to this repository. Needs `cargo` and the lex-gpu checkout; the
# CI gate instead diffs the committed goldens (tests/golden.sh).
set -eu
LEX_GPU=${1:?usage: differential.sh <lex-gpu checkout> [cases-file]}
CASES=${2:-tests/cases.txt}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OURS=${LEXSYS_GPU:-$ROOT/build/lexsys-gpu}
(cd "$LEX_GPU" && cargo build -q -p lex-msl --example emit_lx)
THEIRS=$LEX_GPU/target/debug/examples/emit_lx
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
pass=0
fail=0
n=0
while read -r file rest; do
  case "$file" in ''|'#'*) continue ;; esac
  n=$((n + 1))
  path=$ROOT/$file
  mkdir -p "$WORK/$n/rust" "$WORK/$n/ours"
  # shellcheck disable=SC2086
  if "$THEIRS" "$path" "$WORK/$n/rust" $rest >"$WORK/$n/rust.out" 2>/dev/null; then r=ok; else r=refused; fi
  # shellcheck disable=SC2086
  if "$OURS" emit "$path" "$WORK/$n/ours" $rest >"$WORK/$n/ours.out" 2>"$WORK/$n/err"; then o=ok; else o=refused; fi
  # What `--run` printed, without the lines naming the files written.
  case " $rest " in *" --run "*)
    grep -v "^$WORK/$n/rust/" "$WORK/$n/rust.out" > "$WORK/$n/rust/RUN.txt" || true
    grep -v "^$WORK/$n/ours/" "$WORK/$n/ours.out" > "$WORK/$n/ours/RUN.txt" || true ;;
  esac
  if [ "$r" = "$o" ] && diff -r "$WORK/$n/rust" "$WORK/$n/ours" >/dev/null; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
    echo "DIFFERS: $file $rest (rust: $r, lexsys-gpu: $o)"
    diff -r "$WORK/$n/rust" "$WORK/$n/ours" | head -20 || true
    head -3 "$WORK/$n/err"
  fi
done < "$CASES"
echo "$pass of $n cases agree"
[ "$fail" -eq 0 ]
