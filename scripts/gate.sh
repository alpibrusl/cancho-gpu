#!/bin/sh
# The gate: everything that must pass before a change is called done.
#
#     CANCHO=<path to the pinned cancho> scripts/gate.sh
#
# Canonical formatting, the file budget, the build, the unit tests, the
# goldens (byte-identical with lex-gpu's Rust emitter), the refusal
# fixtures, the device path against a mock driver (docs/device.md) and the
# GGUF reader against the Rust dump (docs/runtime.md). The live
# differential against a lex-gpu checkout (scripts/differential.sh) is not
# in it: it needs cargo and lex-gpu, and the goldens are its recorded
# answer.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CANCHO=${CANCHO:-cancho}
cd "$ROOT"
"$CANCHO" fmt --check src tests device
tests/files.sh
"$CANCHO" build
"$CANCHO" test
tests/golden.sh
tests/reject.sh
CANCHO="$CANCHO" tests/device.sh
tests/gguf.sh
tests/safetensors.sh
tests/tokenizer.sh
tests/sampler.sh
tests/chat.sh
python3 tests/kernels.py "${CANCHO_GPU:-build/cancho-gpu}"
CANCHO="$CANCHO" tests/authority.sh
echo "gate: ok"
