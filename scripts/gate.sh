#!/bin/sh
# The gate: everything that must pass before a change is called done.
#
#     CANCHO=<path to the pinned cancho> scripts/gate.sh
#
# Canonical formatting, the file budget, the build, the unit tests, the
# goldens (regression records of this compiler: scripts/record-golden.sh), the
# refusal fixtures, the kernels against float64 references (tests/kernels.py),
# the device path against a mock driver (docs/device.md), the cloud scripts' cost guard rails against a stub gcloud (tests/gcp.sh, docs/cloud.md), the sessions
# (tests/session.py), the weight-space harness and its gradient arm (tests/weightspace.py, docs/weight-space.md), W1's weight diff and verdict rule (tests/w1.py) and the readers against their recorded oracles
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
tests/gcp.sh
tests/gguf.sh
tests/safetensors.sh
tests/tokenizer.sh
python3 tests/tokenizer_hf.py "${CANCHO_GPU:-build/cancho-gpu}"
python3 tests/oracles.py
python3 tests/m8.py
python3 tests/m8_loops.py
tests/sampler.sh
tests/chat.sh
python3 tests/kernels.py "${CANCHO_GPU:-build/cancho-gpu}"
CANCHO="$CANCHO" scripts/build-session.sh
python3 tests/session.py build/cancho-gpu-session
python3 tests/train.py build/cancho-gpu-session
python3 tests/cnn.py build/cancho-gpu-session
python3 tests/probes.py build/cancho-gpu build/cancho-gpu-session
python3 tests/weightspace.py build/cancho-gpu-session
python3 tests/w1.py
python3 tests/resnet.py build/cancho-gpu-session
if python3 -c 'import numpy' 2>/dev/null; then
  python3 examples/qwen/quant.py validate
else
  echo "skip quant.py validate: numpy is not installed"
fi
python3 tests/gdn.py
python3 tests/qwen.py build/cancho-gpu-session
python3 tests/qwen_quant.py build/cancho-gpu-session
CANCHO="$CANCHO" tests/authority.sh
echo "gate: ok"
