#!/bin/sh
# Build the server of Kolibri 1.1 (docs/kolibri.md section 18): `cancho-gpu-kolibri-cpu` on the CPU interpreter's session, which the gate tests, and on a Mac `cancho-gpu-kolibri` on the Metal one.
#
#     CANCHO=<pinned cancho> scripts/build-kolibri-serve.sh [cpu]
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CANCHO=${CANCHO:-cancho}
CC=${CC:-clang}
mkdir -p "$ROOT/build"
K="$ROOT/kolibri/kchat.cho $ROOT/kolibri/kdraw.cho $ROOT/kolibri/kserve.cho"
# shellcheck disable=SC2046,SC2086
"$CANCHO" build $(ls "$ROOT"/src/*.cho | grep -v '/main\.cho$') $K "$ROOT"/kolibri/serve-cpu-main.cho --std -o "$ROOT/build/cancho-gpu-kolibri-cpu" 2>&1 | grep -v "^ld: warning" || true
test -x "$ROOT/build/cancho-gpu-kolibri-cpu"
echo "built $ROOT/build/cancho-gpu-kolibri-cpu"
if [ "${1:-}" = cpu ] || [ "$(uname)" != Darwin ]; then exit 0; fi
"$CC" -fobjc-arc -fno-exceptions -fno-objc-exceptions -Wall -Wextra -Werror -c "$ROOT/shim/canchogpu-metal.mm" -o "$ROOT/build/canchogpu-metal.o"
# shellcheck disable=SC2046,SC2086
"$CANCHO" build $(ls "$ROOT"/src/*.cho | grep -v '/main\.cho$') $K "$ROOT"/device/metal.cho "$ROOT"/device/msession.cho "$ROOT"/kolibri/serve-metal-main.cho --std --emit obj -l metalgpu -l dl -L "$ROOT/build" -o "$ROOT/build/cancho-gpu-kolibri.o" 2>&1 | grep -v "^ld: warning" || true
"$CC" "$ROOT/build/cancho-gpu-kolibri.o" "$ROOT/build/canchogpu-metal.o" -o "$ROOT/build/cancho-gpu-kolibri" -framework Metal -framework Foundation -ldl 2>&1 | grep -v "^ld: warning" || true
test -x "$ROOT/build/cancho-gpu-kolibri"
echo "built $ROOT/build/cancho-gpu-kolibri"
