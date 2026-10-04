#!/bin/sh
# Build `lexsys-gpu-device` (docs/device.md): the shim as a static
# library, then the program linked against it. `lexsys-gpu` itself is
# built by `lex-sys build` from lex-sys.toml and links nothing but libc;
# the manifest has no way to name a library, hence this script.
#
#     LEX_SYS=<pinned lex-sys> scripts/build-device.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LEX_SYS=${LEX_SYS:-lex-sys}
CC=${CC:-cc}
mkdir -p "$ROOT/build"
"$CC" -std=c11 -Wall -Wextra -Werror -O2 -c "$ROOT/shim/lexgpu.c" -o "$ROOT/build/lexgpu.o"
ar rcs "$ROOT/build/liblexgpu.a" "$ROOT/build/lexgpu.o"
# shellcheck disable=SC2046
"$LEX_SYS" build $(ls "$ROOT"/src/*.ls | grep -v '/main\.ls$') "$ROOT"/device/*.ls --std \
  -l lexgpu -l dl -L "$ROOT/build" -o "$ROOT/build/lexsys-gpu-device"
echo "built $ROOT/build/lexsys-gpu-device"
