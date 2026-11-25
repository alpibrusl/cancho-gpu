#!/bin/sh
# The device path without a device (docs/device.md section 5): build the
# mock driver and NVRTC, run `cancho-gpu-device` on each of the ported
# kernels (tests/chx/ported) against them, and check what reached the "driver" against what
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
for lib in libcuda_mock libnvrtc_mock libamdhip64_mock libhiprtc_mock; do
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
  CANCHOGPU_LIBCUDA=$MOCK/libcuda_mock.so CANCHOGPU_LIBNVRTC=$MOCK/libnvrtc_mock.so CANCHOGPU_MOCK_LOG=$WORK/log \
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
tests/chx/ported/silu_mul.chx
tests/chx/ported/rmsnorm.chx n=4096 eps=1e-5
tests/chx/ported/gemm.chx m=64 n=32 k=256
tests/chx/ported/gemm_mma.chx m=128 n=128 k=64
tests/chx/ported/gemm_fp4.chx m=128 n=128 k=64
tests/chx/ported/gemm_fp4_res.chx m=128 n=128 k=64
CASES
# The byte sizes, for one kernel, by hand: x f16[128, 64], wq i8[128, 32],
# ws i8[128, 4], wg f32[128], y f32[128, 128].
rm -f "$WORK/log"
CANCHOGPU_LIBCUDA=$MOCK/libcuda_mock.so CANCHOGPU_LIBNVRTC=$MOCK/libnvrtc_mock.so CANCHOGPU_MOCK_LOG=$WORK/log \
  "$BIN" run "$ROOT/tests/chx/ported/gemm_fp4.chx" m=128 n=128 k=64 >/dev/null 2>&1 || true
grep -q 'bytes 16384,4096,512,512,65536$' "$WORK/log" || { echo "gemm_fp4: wrong buffer sizes: $(cat "$WORK/log")"; fail=1; }
# No driver: a refusal, not a trap.
status=0
CANCHOGPU_LIBCUDA=$WORK/nothing.so "$BIN" run "$ROOT/tests/chx/ported/silu_mul.chx" >/dev/null 2>"$WORK/err" || status=$?
if [ "$status" -ne 1 ] || ! grep -q '^error\[device\]: opening the device: no CUDA driver' "$WORK/err"; then
  echo "no driver: expected error[device], exit 1; got exit $status: $(cat "$WORK/err")"; fail=1
fi
# The same through the mock HIP runtime and hipRTC (docs/amd.md sections 3.4 and 3.5): the program compiles for amd-cdna3,
# the entry is the golden .hip's, hipRTC is given a program named kernel.hip and no architecture, and the arguments arrive
# both ways hipModuleLaunchKernel takes them.
hip_count=0
hip_run() {
  CANCHOGPU_LIBHIP=$MOCK/libamdhip64_mock.so CANCHOGPU_LIBHIPRTC=$MOCK/libhiprtc_mock.so CANCHOGPU_MOCK_LOG=$WORK/log "$@"
}
while read -r file rest; do
  hip_count=$((hip_count + 1))
  # shellcheck disable=SC2086
  name=$("$ROOT/scripts/case-name.sh" "$file" $rest)
  golden=$(ls "$ROOT/golden/$name/"*.hip)
  entry=$(basename "$golden" .hip)
  params=$(grep -c '__restrict__' "$golden")
  # shellcheck disable=SC2086
  launch=$("$ROOT/build/cancho-gpu" emit "$ROOT/$file" "$WORK/emit" $rest | grep '\.hip:')
  threads=$(echo "$launch" | sed 's/.*: \([0-9]*\) threads.*/\1/')
  grid=$(echo "$launch" | sed 's/.*grid \([0-9]*x[0-9]*\),.*/\1/')
  want="launch $entry grid $grid block $threads args $params bytes"
  for by in params extra; do
    rm -f "$WORK/log"
    status=0
    # shellcheck disable=SC2086
    CANCHOGPU_HIP_ARGS=$by hip_run "$BIN" run "$ROOT/$file" $rest >"$WORK/out" 2>"$WORK/err" || status=$?
    if ! grep -q "^compile-hip kernel.hip\$" "$WORK/log"; then
      echo "$name [hip, $by]: hipRTC was not given kernel.hip with no option: $(grep '^compile' "$WORK/log" || echo nothing)"; fail=1
    elif ! grep -q "^$want " "$WORK/log"; then
      echo "$name [hip, $by]: expected \`$want ...\`, the runtime saw: $(grep '^launch' "$WORK/log" || echo nothing)"; fail=1
    elif ! grep -q "^args-by $by\$" "$WORK/log"; then
      echo "$name [hip, $by]: the arguments did not arrive by $by"; fail=1
    elif [ "$(grep '^launch' "$WORK/log" | sed 's/.* bytes //')" != "$(sed -n 's/^upload: [^ ]* \([0-9]*\) bytes$/\1/p' "$WORK/out" | paste -sd, -)" ]; then
      echo "$name [hip, $by]: the runtime's argument sizes are not the uploads, in order:"; grep '^launch' "$WORK/log"; grep '^upload' "$WORK/out"; fail=1
    elif [ "$status" -ne 1 ] || ! grep -q '^error\[device-mismatch\]' "$WORK/err"; then
      echo "$name [hip, $by]: zeros from the mock were not refused as device-mismatch (exit $status)"; fail=1
    fi
  done
done <<CASES
tests/chx/ported/silu_mul.chx
tests/chx/activations.chx r=4 n=8
tests/chx/attn_gate.chx r=24 n=256
tests/chx/amax_id.chx r=1 nq=2374 v1=151937
CASES
# The architecture reaches hipRTC only when asked for.
rm -f "$WORK/log"
CANCHOGPU_HIP_ARCH=gfx942 hip_run "$BIN" run "$ROOT/tests/chx/ported/silu_mul.chx" >/dev/null 2>&1 || true
grep -q '^compile-hip kernel.hip --gpu-architecture=gfx942$' "$WORK/log" || { echo "hip: the architecture asked for did not reach hipRTC: $(cat "$WORK/log")"; fail=1; }
# A schedule the target cannot run (96 threads, not whole waves of 64) is refused before the runtime is touched.
rm -f "$WORK/log"
status=0
hip_run "$BIN" run "$ROOT/tests/reject/lowering-wave-hip.chx" n=4096 eps=1e-5 >/dev/null 2>"$WORK/err" || status=$?
if [ "$status" -ne 1 ] || ! grep -q '^error\[lowering\]' "$WORK/err" || [ -e "$WORK/log" ]; then
  echo "hip: 96 threads: expected error[lowering] and no compile, exit 1; got exit $status: $(cat "$WORK/err")"; fail=1
fi
# No runtime, and a runtime without hipRTC: refusals, not traps.
status=0
CANCHOGPU_LIBHIP=$WORK/nothing.so "$BIN" run "$ROOT/tests/chx/ported/silu_mul.chx" >/dev/null 2>"$WORK/err" || status=$?
if [ "$status" -ne 1 ] || ! grep -q '^error\[device\]: opening the device: no HIP runtime' "$WORK/err"; then
  echo "no HIP runtime: expected error[device], exit 1; got exit $status: $(cat "$WORK/err")"; fail=1
fi
status=0
CANCHOGPU_LIBHIP=$MOCK/libamdhip64_mock.so CANCHOGPU_LIBHIPRTC=$WORK/nothing.so "$BIN" run "$ROOT/tests/chx/ported/silu_mul.chx" >/dev/null 2>"$WORK/err" || status=$?
if [ "$status" -ne 1 ] || ! grep -q '^error\[device\]: opening the device: no hipRTC' "$WORK/err"; then
  echo "no hipRTC: expected error[device], exit 1; got exit $status: $(cat "$WORK/err")"; fail=1
fi
echo "$count kernels through the mock driver, and the no-driver refusal"
echo "$hip_count kernels through the mock HIP runtime, each with its arguments by params and by extra, and its refusals"
exit $fail
