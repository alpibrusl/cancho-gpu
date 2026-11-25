#!/bin/sh
# Build the emulating driver and NVRTC (docs/device.md section 5b):
# build/emu/libcuda_emu.so and build/emu/libnvrtc_emu.so, from tests/emu/, and
# the same two under HIP's names (docs/amd.md section 7):
# build/emu/libamdhip64_emu.so, build/emu/libhiprtc_emu.so, with
# build/emu/wave_selftest, the test of the emulator's own lane width.
#
#     scripts/build-emu.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CC=${CC:-cc}
mkdir -p "$ROOT/build/emu"
"$CC" -std=c11 -Wall -Wextra -Werror -shared -fPIC "$ROOT/tests/emu/libcuda_emu.c" -o "$ROOT/build/emu/libcuda_emu.so"
"$CC" -std=c11 -Wall -Wextra -Werror -shared -fPIC -DEMU_INCLUDE_DIR="\"$ROOT/tests/emu/include\"" \
  "$ROOT/tests/emu/libnvrtc_emu.c" -o "$ROOT/build/emu/libnvrtc_emu.so"
"$CC" -std=c11 -Wall -Wextra -Werror -shared -fPIC "$ROOT/tests/emu/libamdhip64_emu.c" -o "$ROOT/build/emu/libamdhip64_emu.so"
"$CC" -std=c11 -Wall -Wextra -Werror -shared -fPIC -DEMU_INCLUDE_DIR="\"$ROOT/tests/emu/include\"" \
  "$ROOT/tests/emu/libhiprtc_emu.c" -o "$ROOT/build/emu/libhiprtc_emu.so"
"$CC" -std=c11 -Wall -Wextra -Werror "$ROOT/tests/emu/wave_selftest.c" -o "$ROOT/build/emu/wave_selftest"
