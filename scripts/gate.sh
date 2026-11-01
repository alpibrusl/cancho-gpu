#!/bin/sh
# The gate: everything that must pass before a change is called done.
#
#     CANCHO=<path to the pinned cancho> scripts/gate.sh
#
# Canonical formatting, the file budget, the build, the unit tests, the
# goldens (regression records of this compiler: scripts/record-golden.sh), the
# refusal fixtures, the kernels against float64 references (tests/kernels.py),
# the device path against a mock driver (docs/device.md), the sessions
# (tests/session.py) and the readers against their recorded oracles
# (docs/runtime.md; external tools, docs/native.md N2f).
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
tests/run.sh
CANCHO="$CANCHO" tests/device.sh
CANCHO="$CANCHO" tests/device-io.sh
tests/gguf.sh
tests/safetensors.sh
tests/tokenizer.sh
python3 tests/tokenizer_hf.py "${CANCHO_GPU:-build/cancho-gpu}"
python3 tests/oracles.py
tests/sampler.sh
tests/chat.sh
python3 tests/kernels.py "${CANCHO_GPU:-build/cancho-gpu}"
CANCHO="$CANCHO" scripts/build-session.sh
python3 tests/session.py build/cancho-gpu-session
CANCHO="$CANCHO" tests/authority.sh
echo "gate: ok"
