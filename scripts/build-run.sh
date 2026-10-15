#!/bin/sh
# Build `cancho-gpu-run` (docs/inference.md): the model runner for Metal.
#
#     CANCHO=<pinned cancho> scripts/build-run.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CANCHO=${CANCHO:-cancho}
CC=${CC:-clang}
mkdir -p "$ROOT/build"
"$CC" -Wall -Wextra -Werror -c "$ROOT/shim/canchogpu-metal.mm" -o "$ROOT/build/canchogpu-metal.o"
# shellcheck disable=SC2046
"$CANCHO" build $(ls "$ROOT"/src/*.cho | grep -v '/main\.cho$') "$ROOT"/device/metal.cho "$ROOT"/device/run-main.cho --std --emit obj -l metalgpu -l dl -L "$ROOT/build" -o "$ROOT/build/cancho-gpu-run.o"
"$CC" "$ROOT/build/cancho-gpu-run.o" "$ROOT/build/canchogpu-metal.o" -o "$ROOT/build/cancho-gpu-run" -framework Metal -framework Foundation -ldl
echo "built $ROOT/build/cancho-gpu-run"
