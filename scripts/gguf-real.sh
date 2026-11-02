#!/bin/sh
# Compare the GGUF dump of cancho-gpu with the independent Python reader's on real model files (not in the gate: the files are
# gigabytes; the Python reader maps them, it does not read them whole).
#
#     scripts/gguf-real.sh <file.gguf>...
#
# Headers only (no --sum: cancho-gpu reads the first 32 MiB of a file).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
fail=0
for f in "$@"; do
  if ! "$ROOT/build/cancho-gpu" gguf "$f" > "${TMPDIR:-/tmp}/gguf-real.a" 2>&1 \
    || ! python3 "$ROOT/tools/gguf_dump.py" "$f" > "${TMPDIR:-/tmp}/gguf-real.b" 2>&1; then
    echo "REFUSED  $f"; fail=1
  elif cmp -s "${TMPDIR:-/tmp}/gguf-real.a" "${TMPDIR:-/tmp}/gguf-real.b"; then
    echo "same  $f ($(head -1 "${TMPDIR:-/tmp}/gguf-real.a"))"
  else
    echo "DIFF  $f"; fail=1
  fi
done
exit $fail
