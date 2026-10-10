#!/bin/sh
# The Metal device path test (mirroring tests/device.sh for CUDA).
# Build the mock Metal framework, run `cancho-gpu-metal-device` on each
# kernel against it, and check what reached the "driver" against what
# the program is. The mock runs no kernel, so every output reads back
# as zeros and the comparison must refuse with `device-mismatch`.
#
#     CANCHO=<pinned cancho> tests/metal-device.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CC=${CC:-clang}
"$ROOT/scripts/build-metal-device.sh" >/dev/null
BIN=$ROOT/build/cancho-gpu-metal-device
MOCK=$ROOT/build/mock
mkdir -p "$MOCK"
# Build the mock Metal shim
"$CC" -Wall -Wextra -Werror -shared -fPIC "$ROOT/tests/mock/libmetalgpu_mock.c" -o "$MOCK/libmetalgpu.so"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
fail=0
count=0

# Test with the mock Metal shim
while read -r file rest; do
  count=$((count + 1))
  # shellcheck disable=SC2086
  name=$("$ROOT/scripts/case-name.sh" "$file" $rest)
  
  rm -f "$WORK/log"
  status=0
  # shellcheck disable=SC2086
  LEXGPU_METAL_FRAMEWORK=$MOCK/libmetalgpu.so LEXGPU_METAL_MOCK_LOG=$WORK/log \
    "$BIN" run "$ROOT/$file" $rest >"$WORK/out" 2>"$WORK/err" || status=$?
  
  # Check that the mock was called
  if [ ! -f "$WORK/log" ] || [ ! -s "$WORK/log" ]; then
    echo "$name: mock Metal shim was not called"
    fail=1
  elif ! grep -q "^launch" "$WORK/log"; then
    echo "$name: expected launch log from mock, got: $(cat "$WORK/log")"
    fail=1
  elif [ "$status" -ne 1 ] || ! grep -q '^error\[device-mismatch\]' "$WORK/err"; then
    echo "$name: zeros from the mock were not refused as device-mismatch (exit $status)"
    fail=1
  elif ! grep -q '^reference: ' "$WORK/out" || ! grep -q '^device: ' "$WORK/out"; then
    echo "$name: missing a summary"
    fail=1
  fi
done <<CASES
tests/lx/lex-gpu/silu_mul.lx
tests/lx/lex-gpu/gemm_fp4.lx m=128 n=128 k=64
tests/lx/softmax.lx r=4 n=64
CASES

# No driver: a refusal, not a trap.
status=0
LEXGPU_METAL_FRAMEWORK=$WORK/nothing.so "$BIN" run "$ROOT/tests/lx/softmax.lx" r=4 n=64 >/dev/null 2>"$WORK/err" || status=$?
if [ "$status" -ne 1 ] || ! grep -q '^error\[device\]' "$WORK/err"; then
  echo "no driver: expected error[device], exit 1; got exit $status: $(cat "$WORK/err")"; fail=1
fi
echo "$count kernels through the mock Metal driver, and the no-driver refusal"
exit $fail
