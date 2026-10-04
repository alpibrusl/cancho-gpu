#!/bin/sh
# Regenerate golden/ from lex-gpu's Rust emitter: for every case of
# tests/cases.txt that the Rust accepts, the files it writes, in
# golden/<case>/; for one it refuses, golden/<case>/REFUSED.
#
#     scripts/regen-golden.sh <lex-gpu checkout>
#
# The goldens are the Rust's output, never this compiler's: regenerating
# them from lexsys-gpu would make tests/golden.sh check the compiler
# against itself. golden/LEX_GPU records the commit they came from.
set -eu
LEX_GPU=${1:?usage: regen-golden.sh <lex-gpu checkout>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
(cd "$LEX_GPU" && cargo build -q -p lex-msl --example emit_lx)
THEIRS=$LEX_GPU/target/debug/examples/emit_lx
rm -rf "$ROOT/golden"
mkdir -p "$ROOT/golden"
git -C "$LEX_GPU" rev-parse HEAD > "$ROOT/golden/LEX_GPU"
while read -r file rest; do
  case "$file" in ''|'#'*) continue ;; esac
  dir=$ROOT/golden/$("$ROOT/scripts/case-name.sh" "$file" $rest)
  mkdir -p "$dir"
  # shellcheck disable=SC2086
  if "$THEIRS" "$ROOT/$file" "$dir" $rest >"$dir/.stdout" 2>/dev/null; then
    # A `--run` case also records what the interpreter printed: every
    # line but the ones naming the files written.
    case " $rest " in *" --run "*) grep -v "^$dir/" "$dir/.stdout" > "$dir/RUN.txt" || true ;; esac
    rm -f "$dir/.stdout"
  else
    rm -f "$dir"/* "$dir/.stdout"
    touch "$dir/REFUSED"
  fi
done < "$ROOT/tests/cases.txt"
echo "golden/ regenerated from lex-gpu $(cat "$ROOT/golden/LEX_GPU")"
