#!/bin/sh
# Build `cancho-gpu-qwen08-load` (docs/weight-space.md section 14, T5): the
# Qwen3.5-0.8B weight loader and its C2 check.
#
#     CANCHO=<pinned cancho> scripts/build-qwen08-load.sh          # Metal shim: build/cancho-gpu-qwen08-load
#     CANCHO=<pinned cancho> MOCK=1 scripts/build-qwen08-load.sh   # tests/mock/libmetalgpu_mock.c, whose
#                                                                  # buffers are host memory (no GPU):
#                                                                  # build/cancho-gpu-qwen08-load-mock
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CANCHO=${CANCHO:-cancho}
CC=${CC:-clang}
mkdir -p "$ROOT/build/mock"
# shellcheck disable=SC2046
set -- $(ls "$ROOT"/src/*.cho | grep -v '/main\.cho$') "$ROOT"/device/metal.cho "$ROOT"/device/qwen08_dev.cho "$ROOT"/device/qwen08-load-main.cho
if [ "${MOCK:-}" = "1" ]; then
  "$CANCHO" build "$@" --std --emit obj -l metalgpu -l dl -L "$ROOT/build" -o "$ROOT/build/cancho-gpu-qwen08-load-mock.o"
  "$CC" -std=c11 -c "$ROOT/tests/mock/libmetalgpu_mock.c" -o "$ROOT/build/mock/metal_mock_q08.o"
  "$CC" "$ROOT/build/cancho-gpu-qwen08-load-mock.o" "$ROOT/build/mock/metal_mock_q08.o" -o "$ROOT/build/cancho-gpu-qwen08-load-mock" -ldl -lm
  echo "built $ROOT/build/cancho-gpu-qwen08-load-mock"
else
  "$CC" -fobjc-arc -fno-exceptions -fno-objc-exceptions -Wall -Wextra -Werror -c "$ROOT/shim/canchogpu-metal.mm" -o "$ROOT/build/canchogpu-metal.o"
  "$CANCHO" build "$@" --std --emit obj -l metalgpu -l dl -L "$ROOT/build" -o "$ROOT/build/cancho-gpu-qwen08-load.o"
  "$CC" "$ROOT/build/cancho-gpu-qwen08-load.o" "$ROOT/build/canchogpu-metal.o" -o "$ROOT/build/cancho-gpu-qwen08-load" -framework Metal -framework Foundation -ldl
  echo "built $ROOT/build/cancho-gpu-qwen08-load"
fi
