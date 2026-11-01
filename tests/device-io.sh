#!/bin/sh
# Gates 1 and 2 of docs/run.md section 7: `--in` / `--out` on both device
# programs against mock drivers (tests/device_io.py). CUDA is built as
# `scripts/build-device.sh` does and run against the mock driver and NVRTC;
# Metal is linked against `tests/mock/libmetalgpu_mock.c` in place of the
# Objective-C++ shim, so it runs wherever cancho and a C compiler do. The shim
# itself, and any real hardware, are not exercised (docs/run.md section 7.2 3-4).
#
#     CANCHO=<pinned cancho> tests/device-io.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CANCHO=${CANCHO:-cancho}
CC=${CC:-cc}
mkdir -p "$ROOT/build/mock"
"$ROOT/scripts/build-device.sh" >/dev/null
for lib in libcuda_mock libnvrtc_mock; do
  "$CC" -std=c11 -Wall -Wextra -Werror -shared -fPIC "$ROOT/tests/mock/$lib.c" -o "$ROOT/build/mock/$lib.so"
done
# shellcheck disable=SC2046
"$CANCHO" build $(ls "$ROOT"/src/*.cho | grep -v '/main\.cho$') "$ROOT"/device/metal.cho "$ROOT"/device/metal-main.cho --std --emit obj \
  -l metalgpu -l dl -L "$ROOT/build" -o "$ROOT/build/cancho-gpu-metal-device-mock.o" >/dev/null
"$CC" -std=c11 -c "$ROOT/tests/mock/libmetalgpu_mock.c" -o "$ROOT/build/mock/metal_mock.o"
"$CC" "$ROOT/build/cancho-gpu-metal-device-mock.o" "$ROOT/build/mock/metal_mock.o" -o "$ROOT/build/cancho-gpu-metal-device-mock" -ldl -lm
status=0
python3 "$ROOT/tests/device_io.py" cuda "$ROOT/build/cancho-gpu-device" CANCHOGPU_MOCK_LOG \
  "CANCHOGPU_LIBCUDA=$ROOT/build/mock/libcuda_mock.so" "CANCHOGPU_LIBNVRTC=$ROOT/build/mock/libnvrtc_mock.so" || status=1
python3 "$ROOT/tests/device_io.py" metal "$ROOT/build/cancho-gpu-metal-device-mock" CANCHOGPU_METAL_MOCK_LOG || status=1
exit $status
