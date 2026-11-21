#!/bin/sh
# Build `cancho-gpu-probe-copy` (docs/serving.md section 7.8, step 1): the prefix cache probe.
#
#     CANCHO=<pinned cancho> scripts/build-probe-copy.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CANCHO=${CANCHO:-cancho}
CC=${CC:-clang}
mkdir -p "$ROOT/build"
"$CC" -fobjc-arc -fno-exceptions -fno-objc-exceptions -Wall -Wextra -Werror -c "$ROOT/shim/canchogpu-metal.mm" -o "$ROOT/build/canchogpu-metal.o"
# shellcheck disable=SC2046
"$CANCHO" build $(ls "$ROOT"/src/*.cho | grep -v '/main\.cho$') "$ROOT"/device/metal.cho "$ROOT"/device/qwen35.cho "$ROOT"/device/qwen35spec.cho "$ROOT"/device/probe-copy-main.cho --std --emit obj -l metalgpu -l dl -L "$ROOT/build" -o "$ROOT/build/cancho-gpu-probe-copy.o"
"$CC" "$ROOT/build/cancho-gpu-probe-copy.o" "$ROOT/build/canchogpu-metal.o" -o "$ROOT/build/cancho-gpu-probe-copy" -framework Metal -framework Foundation -ldl
echo "built $ROOT/build/cancho-gpu-probe-copy"
