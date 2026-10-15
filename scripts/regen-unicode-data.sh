#!/bin/sh
# Regenerate src/unicode_data.cho from the oracle's own Unicode predicates
# and NFC tables (lex-rt, built by the toolchain on this machine).
#
#     scripts/regen-unicode-data.sh <lex-gpu checkout>
#
# The example is copied into the checkout for the build and removed after.
set -eu
LEX_GPU=${1:?usage: regen-unicode-data.sh <lex-gpu checkout>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EX=$LEX_GPU/crates/lex-rt/examples/unicode_tables.rs
mkdir -p "$LEX_GPU/crates/lex-rt/examples"
cp "$ROOT/tools/unicode_tables.rs" "$EX"
trap 'rm -f "$EX"' EXIT
(cd "$LEX_GPU" && RUSTC_VERSION=$(rustc --version | cut -d' ' -f2) cargo build -q -p lex-rt --example unicode_tables)
"$LEX_GPU/target/debug/examples/unicode_tables" > "$ROOT/src/unicode_data.cho"
echo "src/unicode_data.cho regenerated with rustc $(rustc --version | cut -d' ' -f2) from lex-gpu $(git -C "$LEX_GPU" rev-parse --short HEAD)"
