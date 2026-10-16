#!/bin/sh
# Build `cancho-gpu-serve` (docs/serving.md): the OpenAI-compatible endpoint over the 27B on Metal.
#
#     CANCHO=<pinned cancho> scripts/build-serve.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CANCHO=${CANCHO:-cancho}
CC=${CC:-clang}
mkdir -p "$ROOT/build"
"$CC" -Wall -Wextra -Werror -c "$ROOT/shim/canchogpu-metal.mm" -o "$ROOT/build/canchogpu-metal.o"
# shellcheck disable=SC2046
"$CANCHO" build $(ls "$ROOT"/src/*.cho | grep -v '/main\.cho$') "$ROOT"/device/metal.cho "$ROOT"/device/qwen35.cho "$ROOT"/device/serve-main.cho --std --emit obj -l metalgpu -l dl -L "$ROOT/build" -o "$ROOT/build/cancho-gpu-serve.o"
"$CC" "$ROOT/build/cancho-gpu-serve.o" "$ROOT/build/canchogpu-metal.o" -o "$ROOT/build/cancho-gpu-serve" -framework Metal -framework Foundation -ldl
echo "built $ROOT/build/cancho-gpu-serve"
