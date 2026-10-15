#!/bin/sh
# Compare the GGUF dump of cancho-gpu with lex-rt's on real model files
# (not in the gate: the files are gigabytes). The Rust dumper is the one
# scripts/regen-gguf-golden.sh builds.
#
#     scripts/gguf-real.sh <lex-gpu checkout> <file.gguf>...
#
# Headers only (no --sum: cancho-gpu reads the first 32 MiB of a file).
set -eu
LEX_GPU=${1:?usage: gguf-real.sh <lex-gpu checkout> <file.gguf>...}
shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
THEIRS=$LEX_GPU/target/debug/examples/gguf_dump
[ -x "$THEIRS" ] || { echo "build it first: scripts/regen-gguf-golden.sh $LEX_GPU"; exit 2; }
fail=0
for f in "$@"; do
  if ! "$ROOT/build/cancho-gpu" gguf "$f" > "${TMPDIR:-/tmp}/gguf-real.a" 2>&1 \
    || ! "$THEIRS" "$f" > "${TMPDIR:-/tmp}/gguf-real.b" 2>&1; then
    echo "REFUSED  $f"; fail=1
  elif cmp -s "${TMPDIR:-/tmp}/gguf-real.a" "${TMPDIR:-/tmp}/gguf-real.b"; then
    echo "same  $f ($(head -1 "${TMPDIR:-/tmp}/gguf-real.a"))"
  else
    echo "DIFF  $f"; fail=1
  fi
done
exit $fail
