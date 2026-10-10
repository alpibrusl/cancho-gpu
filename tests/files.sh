#!/bin/sh
# No source file over 2,000 lines -- cancho's own rule
# (`crates/cancho/tests/files.rs` there). Split by concern; never raise
# the line.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
fail=0
for f in "$ROOT"/src/*.cho "$ROOT"/device/*.cho "$ROOT"/shim/*.c "$ROOT"/tests/mock/*.c "$ROOT"/tests/*.cho "$ROOT"/tests/*.sh "$ROOT"/scripts/*.sh; do
  n=$(wc -l < "$f")
  if [ "$n" -gt 2000 ]; then
    echo "$f: $n lines, over the 2,000-line budget"; fail=1
  fi
done
exit $fail
