#!/bin/sh
# Build `cancho-gpu-metal-device` for Metal GPU support (docs/device.md).
#
#     CANCHO=<pinned cancho> scripts/build-metal-device.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CANCHO=${CANCHO:-cancho}
CC=${CC:-clang}
mkdir -p "$ROOT/build"
# Compile the Metal shim (Objective-C++)
"$CC" -fobjc-arc -fno-exceptions -fno-objc-exceptions -Wall -Wextra -Werror -c "$ROOT/shim/canchogpu-metal.mm" -o "$ROOT/build/canchogpu-metal.o"
# Generate object file with cancho - exclude experimental modules
# shellcheck disable=SC2046
"$CANCHO" build $(ls "$ROOT"/src/*.cho | grep -v '/main\.cho$' | grep -v '/client\.cho$' | grep -v '/server\.cho$' | grep -v '/sample\.cho$' | grep -v '/chat\.cho$') "$ROOT"/device/metal.cho "$ROOT"/device/metal-main.cho --std --emit obj -l metalgpu -l dl -L "$ROOT/build" -o "$ROOT/build/cancho-gpu-metal-device.o"
# Link with Metal framework
"$CC" "$ROOT/build/cancho-gpu-metal-device.o" "$ROOT/build/canchogpu-metal.o" -o "$ROOT/build/cancho-gpu-metal-device" -framework Metal -framework Foundation -ldl
echo "built $ROOT/build/cancho-gpu-metal-device"
