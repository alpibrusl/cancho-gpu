#!/bin/sh
# Regenerate golden/gguf/ from lex-rt's GGUF reader: for every fixture of
# tests/gguf/*.gguf, the dump `tools/gguf_dump.rs` prints (with --sum).
#
#     scripts/regen-gguf-golden.sh <lex-gpu checkout>
#
# As with golden/, the files are the Rust's output, never this compiler's.
# The example is copied into the checkout for the build and removed after;
# golden/gguf/LEX_GPU records the commit.
set -eu
LEX_GPU=${1:?usage: regen-gguf-golden.sh <lex-gpu checkout>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EX=$LEX_GPU/crates/lex-rt/examples/gguf_dump.rs
mkdir -p "$LEX_GPU/crates/lex-rt/examples"
cp "$ROOT/tools/gguf_dump.rs" "$EX"
trap 'rm -f "$EX"' EXIT
(cd "$LEX_GPU" && cargo build -q -p lex-rt --example gguf_dump)
rm -rf "$ROOT/golden/gguf"
mkdir -p "$ROOT/golden/gguf"
git -C "$LEX_GPU" rev-parse HEAD > "$ROOT/golden/gguf/LEX_GPU"
for f in "$ROOT"/tests/gguf/*.gguf; do
  "$LEX_GPU/target/debug/examples/gguf_dump" "$f" --sum > "$ROOT/golden/gguf/$(basename "$f" .gguf).txt"
done
echo "golden/gguf/ regenerated from lex-gpu $(cat "$ROOT/golden/gguf/LEX_GPU")"
