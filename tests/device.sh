#!/bin/sh
# The device path without a device (docs/device.md section 5): build the
# mock driver and NVRTC, run `cancho-gpu-device` on each of lex-gpu's
# kernels against them, and check what reached the "driver" against what
# the program is -- the entry the golden .cu defines, the grid and threads
# `emit` reports, one argument per parameter. The mock runs no kernel, so
# every output reads back as zeros and the comparison must refuse with
# `device-mismatch`: that is the check that the comparison can fail.
# Then once with no driver at all, which must be `device`, not a trap.
#
#     CANCHO=<pinned cancho> tests/device.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CC=${CC:-cc}
"$ROOT/scripts/build-device.sh" >/dev/null
BIN=$ROOT/build/cancho-gpu-device
MOCK=$ROOT/build/mock
mkdir -p "$MOCK"
for lib in libcuda_mock libnvrtc_mock; do
  "$CC" -std=c11 -Wall -Wextra -Werror -shared -fPIC "$ROOT/tests/mock/$lib.c" -o "$MOCK/$lib.so"
done
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
fail=0
count=0
while read -r file rest; do
  count=$((count + 1))
  # shellcheck disable=SC2086
  name=$("$ROOT/scripts/case-name.sh" "$file" $rest)
  golden=$(ls "$ROOT/golden/$name/"*.cu)
  entry=$(basename "$golden" .cu)
  params=$(grep -c '__restrict__' "$golden")
  # What `emit` says about the launch: `<path>: <threads> threads, grid <gx>x<gy>, ...`
  mkdir -p "$WORK/emit"
  # shellcheck disable=SC2086
  launch=$("$ROOT/build/cancho-gpu" emit "$ROOT/$file" "$WORK/emit" $rest | grep '\.cu:')
  threads=$(echo "$launch" | sed 's/.*: \([0-9]*\) threads.*/\1/')
  grid=$(echo "$launch" | sed 's/.*grid \([0-9]*x[0-9]*\),.*/\1/')
  rm -f "$WORK/log"
  status=0
  # shellcheck disable=SC2086
  LEXGPU_LIBCUDA=$MOCK/libcuda_mock.so LEXGPU_LIBNVRTC=$MOCK/libnvrtc_mock.so LEXGPU_MOCK_LOG=$WORK/log \
    "$BIN" run "$ROOT/$file" $rest >"$WORK/out" 2>"$WORK/err" || status=$?
  want="launch $entry grid $grid block $threads args $params bytes"
  if ! grep -q "^compile --gpu-architecture=compute_89" "$WORK/log"; then
    echo "$name: NVRTC was not asked for compute_89"; fail=1
  elif ! grep -q "^$want " "$WORK/log"; then
    echo "$name: expected \`$want ...\`, the driver saw: $(grep '^launch' "$WORK/log" || echo nothing)"; fail=1
  elif [ "$(grep '^launch' "$WORK/log" | sed 's/.* bytes //')" != "$(sed -n 's/^upload: [^ ]* \([0-9]*\) bytes$/\1/p' "$WORK/out" | paste -sd, -)" ]; then
    echo "$name: the driver's argument sizes are not the uploads, in order:"; grep '^launch' "$WORK/log"; grep '^upload' "$WORK/out"; fail=1
  elif [ "$status" -ne 1 ] || ! grep -q '^error\[device-mismatch\]' "$WORK/err"; then
    echo "$name: zeros from the mock were not refused as device-mismatch (exit $status)"; fail=1
  elif ! grep -q '^reference: ' "$WORK/out" || ! grep -q '^device: ' "$WORK/out"; then
    echo "$name: missing a summary"; fail=1
  fi
done <<CASES
tests/lx/lex-gpu/silu_mul.lx
tests/lx/lex-gpu/rmsnorm.lx n=4096 eps=1e-5
tests/lx/lex-gpu/gemm.lx m=64 n=32 k=256
tests/lx/lex-gpu/gemm_mma.lx m=128 n=128 k=64
tests/lx/lex-gpu/gemm_fp4.lx m=128 n=128 k=64
tests/lx/lex-gpu/gemm_fp4_res.lx m=128 n=128 k=64
CASES
# The byte sizes, for one kernel, by hand: x f16[128, 64], wq i8[128, 32],
# ws i8[128, 4], wg f32[128], y f32[128, 128].
rm -f "$WORK/log"
LEXGPU_LIBCUDA=$MOCK/libcuda_mock.so LEXGPU_LIBNVRTC=$MOCK/libnvrtc_mock.so LEXGPU_MOCK_LOG=$WORK/log \
  "$BIN" run "$ROOT/tests/lx/lex-gpu/gemm_fp4.lx" m=128 n=128 k=64 >/dev/null 2>&1 || true
grep -q 'bytes 16384,4096,512,512,65536$' "$WORK/log" || { echo "gemm_fp4: wrong buffer sizes: $(cat "$WORK/log")"; fail=1; }
# No driver: a refusal, not a trap.
status=0
LEXGPU_LIBCUDA=$WORK/nothing.so "$BIN" run "$ROOT/tests/lx/lex-gpu/silu_mul.lx" >/dev/null 2>"$WORK/err" || status=$?
if [ "$status" -ne 1 ] || ! grep -q '^error\[device\]: opening the device: no CUDA driver' "$WORK/err"; then
  echo "no driver: expected error[device], exit 1; got exit $status: $(cat "$WORK/err")"; fail=1
fi
echo "$count kernels through the mock driver, and the no-driver refusal"
exit $fail
