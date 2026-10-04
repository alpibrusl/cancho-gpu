#!/bin/sh
# No source file over 2,000 lines -- lex-sys's own rule
# (`crates/lex-sys/tests/files.rs` there). Split by concern; never raise
# the line.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
fail=0
for f in "$ROOT"/src/*.ls "$ROOT"/device/*.ls "$ROOT"/shim/*.c "$ROOT"/tests/mock/*.c "$ROOT"/tests/*.ls "$ROOT"/tests/*.sh "$ROOT"/scripts/*.sh; do
  n=$(wc -l < "$f")
  if [ "$n" -gt 2000 ]; then
    echo "$f: $n lines, over the 2,000-line budget"; fail=1
  fi
done
exit $fail
