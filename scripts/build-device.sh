#!/bin/sh
# Build `cancho-gpu-device` (docs/device.md): the shim as a static
# library, then the program linked against it. `cancho-gpu` itself is
# built by `cancho build` from cancho.toml and links nothing but libc;
# the manifest has no way to name a library, hence this script.
#
#     CANCHO=<pinned cancho> scripts/build-device.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CANCHO=${CANCHO:-cancho}
CC=${CC:-cc}
mkdir -p "$ROOT/build"
"$CC" -std=c11 -Wall -Wextra -Werror -O2 -c "$ROOT/shim/lexgpu.c" -o "$ROOT/build/lexgpu.o"
ar rcs "$ROOT/build/liblexgpu.a" "$ROOT/build/lexgpu.o"
# shellcheck disable=SC2046
"$CANCHO" build $(ls "$ROOT"/src/*.cho | grep -v '/main\.cho$') "$ROOT"/device/*.cho --std \
  -l lexgpu -l dl -L "$ROOT/build" -o "$ROOT/build/cancho-gpu-device"
echo "built $ROOT/build/cancho-gpu-device"
