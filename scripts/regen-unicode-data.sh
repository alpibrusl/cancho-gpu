#!/bin/sh
# Regenerate src/unicode_data.cho from Python's `unicodedata` (tools/unicode_tables.py): the general categories L, M, N, White_Space,
# the combining classes, the canonical decompositions and compositions. Needs python3 and nothing else.
#
#     scripts/regen-unicode-data.sh
#
# The Unicode version is that of the python3 that runs it, and is written in the generated file's header; a different Python may
# give a different file, and then tests/tokenizer_hf.py (the same Unicode data as Hugging Face's, or not) is the check.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
python3 "$ROOT/tools/unicode_tables.py" > "$ROOT/src/unicode_data.cho"
echo "src/unicode_data.cho regenerated with Unicode $(python3 -c 'import unicodedata; print(unicodedata.unidata_version)')"
