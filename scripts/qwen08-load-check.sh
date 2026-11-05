#!/bin/sh
# Check C2 (docs/weight-space.md section 14.6) on the real Qwen3.5-0.8B
# checkpoints: every text tensor put on the device by `cancho-gpu-qwen08-load`
# and read back equals the file (f32: bit-exact bf16 widening, F32 tensors as F32;
# f16: numpy's astype(float16), with the count of values that differ from the
# bf16 originals). Needs numpy and the models; says so and exits 0 when they are
# absent. Not in the gate.
#
# By default the "device" is tests/mock/libmetalgpu_mock.c, whose buffers are host
# memory (no GPU; 3 GB for f32). `REAL=1` links the Metal shim and uses the GPU.
#
#     CANCHO=<pinned cancho> [REAL=1] scripts/qwen08-load-check.sh [model dir ...]
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ "$#" -eq 0 ]; then
  set -- "$HOME/models/qwen35-08b-base" "$HOME/models/qwen35-08b-edited"
fi
if ! python3 -c 'import numpy' 2>/dev/null; then
  echo "skip qwen08-load-check: numpy is not installed"; exit 0
fi
found=0
for d in "$@"; do
  [ -f "$d/config.json" ] && found=1
done
if [ "$found" = 0 ]; then
  echo "skip qwen08-load-check: no checkpoint in $*"; exit 0
fi
if [ "${REAL:-}" = 1 ]; then
  "$ROOT/scripts/build-qwen08-load.sh" >/dev/null
  BIN=$ROOT/build/cancho-gpu-qwen08-load
else
  MOCK=1 "$ROOT/scripts/build-qwen08-load.sh" >/dev/null
  BIN=$ROOT/build/cancho-gpu-qwen08-load-mock
fi
fail=0
for d in "$@"; do
  for mode in f32 f16; do
    python3 "$ROOT/tests/qwen08_load.py" model "$BIN" "$d" "$mode" || fail=1
  done
done
exit $fail
