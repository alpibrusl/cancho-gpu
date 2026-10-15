#!/bin/sh
# Regenerate golden/safetensors/ from the `safetensors` library: for every
# fixture of tests/safetensors/*.safetensors, the dump
# `tools/safetensors_dump.py` prints.
#
#     scripts/regen-safetensors-golden.sh
#
# Needs python3 with `safetensors` and `torch` (the library's own reader is
# the oracle). As with golden/, the files are the oracle's output, never
# this compiler's; golden/safetensors/ORACLE records the library version.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
rm -rf "$ROOT/golden/safetensors"
mkdir -p "$ROOT/golden/safetensors"
python3 -c 'import safetensors; print("safetensors " + safetensors.__version__)' > "$ROOT/golden/safetensors/ORACLE"
for f in "$ROOT"/tests/safetensors/*.safetensors; do
  python3 "$ROOT/tools/safetensors_dump.py" "$f" > "$ROOT/golden/safetensors/$(basename "$f" .safetensors).txt"
done
echo "golden/safetensors/ regenerated from $(cat "$ROOT/golden/safetensors/ORACLE")"
