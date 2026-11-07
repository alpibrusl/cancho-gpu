#!/bin/sh
# The gate: everything that must pass before a change is called done.
#
#     CANCHO=<path to the pinned cancho> scripts/gate.sh
#
# Canonical formatting, the file budget, the build, the unit tests, the
# goldens (regression records of this compiler: scripts/record-golden.sh), the
# refusal fixtures, the kernels against float64 references (tests/kernels.py),
# the cloud scripts' guard rails against a stub gcloud (tests/gcp.sh, docs/cloud.md), the device path against a mock driver and against an emulating one that runs the emitted CUDA on the CPU (docs/device.md 5, 5b), the sessions
# (tests/session.py), the weight-space harness and its gradient arm (tests/weightspace.py, docs/weight-space.md), W1's weight diff and verdict rule (tests/w1.py), the Hugging Face reference goldens for Qwen3.5-0.8B (tests/hf08.py, golden/hf08/) and the readers against their recorded oracles
# (docs/runtime.md; external tools, docs/native.md N2f) the Qwen3.5-0.8B weight loader on a toy checkpoint (tests/qwen08_load.sh) and its engine's build and refusals against the mock driver (tests/qwen08_engine.sh, docs/weight-space.md 14), the W1 generator's front door (tests/w1_driver.sh) and the cancho behaviour backend against a fake engine (tests/w1_cancho.py, T9), and W4's pure parts: KL/perplexity, the null-space projection, the repetition measures, the corpus rules and the gates, on toy data (tests/w4.py, docs/weight-space-w4.md; skipped without numpy) and its checkpoint writer on a toy checkpoint (tests/w4_ckpt.py, docs/weight-space-w4.md 13; skipped without numpy), E2c's pure parts (tests/w4_e2c.py, docs/weight-space-w4.md 14; skipped without numpy) and T12's (tests/w1_t12.py, docs/weight-space.md 14.15).
# The edit patch (tests/w1_patch.py, docs/weight-space.md 14.13) is held beside them.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CANCHO=${CANCHO:-cancho}
cd "$ROOT"
"$CANCHO" fmt --check src tests device tools
tests/files.sh
"$CANCHO" build
"$CANCHO" test
tests/golden.sh
tests/reject.sh
tests/run.sh
CANCHO="$CANCHO" tests/device.sh
CANCHO="$CANCHO" tests/device-io.sh
tests/gcp.sh
python3 tests/cuda_static.py
scripts/build-emu.sh
python3 tests/cuda_emu_selftest.py
CANCHO="$CANCHO" tests/cuda-emu.sh
tests/gguf.sh
tests/safetensors.sh
tests/tokenizer.sh
python3 tests/tokenizer_hf.py "${CANCHO_GPU:-build/cancho-gpu}"
python3 tests/detokenize_hf.py "${CANCHO_GPU:-build/cancho-gpu}"
python3 tests/chat_w1.py "${CANCHO_GPU:-build/cancho-gpu}"
python3 tests/oracles.py
python3 tests/m8.py
python3 tests/m8_loops.py
tests/sampler.sh
python3 tests/sampler_penalty.py "${CANCHO_GPU:-build/cancho-gpu}"
CANCHO="$CANCHO" tests/qwen08_load.sh
CANCHO="$CANCHO" tests/qwen08_engine.sh
CANCHO="$CANCHO" tests/w1_driver.sh
tests/chat.sh
python3 tests/kernels.py "${CANCHO_GPU:-build/cancho-gpu}"
CANCHO="$CANCHO" scripts/build-session.sh
python3 tests/session.py build/cancho-gpu-session
python3 tests/samp_dev.py build/cancho-gpu-session
python3 tests/carry_dev.py build/cancho-gpu-session
python3 tests/train.py build/cancho-gpu-session
python3 tests/cnn.py build/cancho-gpu-session
python3 tests/probes.py build/cancho-gpu build/cancho-gpu-session
python3 tests/weightspace.py build/cancho-gpu-session
python3 tests/w1.py
python3 tests/w1_patch.py
python3 tests/w1_cancho.py
python3 tests/w4.py
python3 tests/w4_ckpt.py
python3 tests/w4_e2c.py
python3 tests/w1_t12.py
python3 tests/hf08.py
python3 tests/resnet.py build/cancho-gpu-session
if python3 -c 'import numpy' 2>/dev/null; then
  python3 examples/qwen/quant.py validate
else
  echo "skip quant.py validate: numpy is not installed"
fi
python3 tests/gdn.py
python3 tests/qwen.py build/cancho-gpu-session
python3 tests/qwen_quant.py build/cancho-gpu-session
CANCHO="$CANCHO" scripts/build-qsw.sh
python3 tests/qsw.py build/qsw
CANCHO="$CANCHO" tests/authority.sh
echo "gate: ok"
