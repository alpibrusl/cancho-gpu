#!/bin/sh
# Build `cancho-gpu-run08` (docs/weight-space.md section 14, T6): the Qwen3.5-0.8B engine and its tracer, for Metal.
#
#     CANCHO=<pinned cancho> scripts/build-run08.sh          # Metal shim: build/cancho-gpu-run08
#     CANCHO=<pinned cancho> MOCK=1 scripts/build-run08.sh   # tests/mock/libmetalgpu_mock.c (no GPU):
#                                                            # build/cancho-gpu-run08-mock
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CANCHO=${CANCHO:-cancho}
CC=${CC:-clang}
mkdir -p "$ROOT/build/mock"
# shellcheck disable=SC2046
set -- $(ls "$ROOT"/src/*.cho | grep -v '/main\.cho$') "$ROOT"/device/metal.cho "$ROOT"/device/qwen35.cho "$ROOT"/device/qwen08_dev.cho "$ROOT"/device/qwen08.cho "$ROOT"/device/run08-main.cho
if [ "${MOCK:-}" = "1" ]; then
  "$CANCHO" build "$@" --std --emit obj -l metalgpu -l dl -L "$ROOT/build" -o "$ROOT/build/cancho-gpu-run08-mock.o"
  "$CC" -std=c11 -c "$ROOT/tests/mock/libmetalgpu_mock.c" -o "$ROOT/build/mock/metal_mock_run08.o"
  "$CC" "$ROOT/build/cancho-gpu-run08-mock.o" "$ROOT/build/mock/metal_mock_run08.o" -o "$ROOT/build/cancho-gpu-run08-mock" -ldl -lm
  echo "built $ROOT/build/cancho-gpu-run08-mock"
else
  "$CC" -fobjc-arc -fno-exceptions -fno-objc-exceptions -Wall -Wextra -Werror -c "$ROOT/shim/canchogpu-metal.mm" -o "$ROOT/build/canchogpu-metal.o"
  "$CANCHO" build "$@" --std --emit obj -l metalgpu -l dl -L "$ROOT/build" -o "$ROOT/build/cancho-gpu-run08.o"
  "$CC" "$ROOT/build/cancho-gpu-run08.o" "$ROOT/build/canchogpu-metal.o" -o "$ROOT/build/cancho-gpu-run08" -framework Metal -framework Foundation -ldl
  echo "built $ROOT/build/cancho-gpu-run08"
fi
