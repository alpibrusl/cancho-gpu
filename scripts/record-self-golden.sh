#!/bin/sh
# Record golden-self/ from this compiler for the cases of tests/self-cases.txt:
# lowering paths the Rust emitter does not have (no oracle can hold them).
# Review the diff as text before committing it: it is what the next change is
# held to, not an independent answer. tests/golden.sh checks it.
#
#     scripts/record-self-golden.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${CANCHO_GPU:-$ROOT/build/cancho-gpu}
mkdir -p "$ROOT/golden-self"
while read -r file rest; do
  case "$file" in ''|'#'*) continue ;; esac
  # shellcheck disable=SC2086
  dir=$ROOT/golden-self/$("$ROOT/scripts/case-name.sh" "$file" $rest)
  rm -rf "$dir"
  mkdir -p "$dir"
  # shellcheck disable=SC2086
  "$BIN" emit "$ROOT/$file" "$dir" $rest >"$dir/.stdout"
  case " $rest " in *" --run "*) grep -v "^$dir/" "$dir/.stdout" > "$dir/RUN.txt" || true ;; esac
  rm -f "$dir/.stdout"
done < "$ROOT/tests/self-cases.txt"
echo "golden-self/ recorded from this compiler"
