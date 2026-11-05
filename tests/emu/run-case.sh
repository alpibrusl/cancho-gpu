#!/bin/sh
# One case of tests/cases.txt through the emulating driver (tests/cuda-emu.sh):
#
#     run-case.sh <outdir> '<file.chx> [name=value ...]'
#
# Writes <outdir>/<case>.out (everything the device program and the emulator
# said) and one line on standard output: `PASS|FAIL <case> <exit> <summary>`,
# where the summary is the worst difference against the interpreter, or the
# first error / emulator finding.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$1
set -- $2
file=$1
shift
name=$("$ROOT/scripts/case-name.sh" "$file" "$@")
EMU=$ROOT/build/emu
status=0
CANCHOGPU_LIBCUDA=$EMU/libcuda_emu.so CANCHOGPU_LIBNVRTC=$EMU/libnvrtc_emu.so \
  CANCHOGPU_EMU_CACHE=${CANCHOGPU_EMU_CACHE:-$EMU/cache} \
  "$ROOT/build/cancho-gpu-device" run "$ROOT/$file" "$@" >"$OUT/$name.out" 2>&1 || status=$?
worst=$(sed -n 's/^compare: \(.*\)/\1/p' "$OUT/$name.out" | head -3 | tr '\n' ';')
find=$(grep -m1 -E '^(emu-race|emu-warn|error)' "$OUT/$name.out" | cut -c1-300)
# The device program's threshold is 2e-2 of the peak. Measured over every case
# the worst is 4.4e-4 (one half-precision ulp), so a tighter 2e-3 catches an
# error ten times smaller than the program's own check would.
tight=$(sed -n 's/^compare: .* = \(.*\) of peak \(.*\)$/\1 \2/p' "$OUT/$name.out" | awk '$2 > 0 && $1 / $2 > 0.002 { print "worst " $1 " of peak " $2 " is past 2e-3"; exit }')
if [ "$status" -eq 0 ] && [ -n "$tight" ]; then
  echo "FAIL $name 0 ${worst} [tight: $tight]"
elif [ "$status" -eq 0 ] && ! grep -q '^emu-race' "$OUT/$name.out"; then
  echo "PASS $name 0 ${worst}${find:+ [$find]}"
else
  echo "FAIL $name $status ${worst}${find:+ [$find]}"
fi
