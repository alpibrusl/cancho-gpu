#!/bin/sh
# The emitted CUDA, run: every case of tests/cases.txt that emits a kernel goes
# through `cancho-gpu-device` with the EMULATING driver and NVRTC (tests/emu/,
# docs/device.md section 5b), which compile the CUDA with the system C++
# compiler and run the grid on the CPU, and the result is held to the
# interpreter at the device program's threshold (2e-2 of the peak). A case
# fails on a mismatch, a deadlock, an illegal access, a misaligned access, a
# shared-memory overrun, or a result that depends on the schedule
# (`emu-race`). Cases whose failure is a real bug in the emitted CUDA are
# listed in tests/cuda-emu-known.txt with the diagnosis; they must keep
# failing (a fixed bug is a stale line) and nothing else may fail.
#
#     CANCHO=<pinned cancho> tests/cuda-emu.sh
#
#     JOBS=4                 cases in parallel (CPU only; no GPU is touched)
#     CUDA_EMU_OUT=<dir>     keep every case's output and results.txt there
#     CUDA_EMU_ONLY=<regex>  only the cases whose name matches
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
"$ROOT/scripts/build-emu.sh"
"$ROOT/scripts/build-device.sh" >/dev/null
# The emulator must be able to fail on the real emitted CUDA: each mutant
# below is a golden kernel with one thing broken (CANCHOGPU_EMU_MUTATE), and
# must be refused for the reason given.
mutant() { # <expected text in the output> <file> <consts> <from=>to>
  want=$1; file=$2; consts=$3; mut=$4
  status=0
  # shellcheck disable=SC2086
  CANCHOGPU_LIBCUDA=$ROOT/build/emu/libcuda_emu.so CANCHOGPU_LIBNVRTC=$ROOT/build/emu/libnvrtc_emu.so \
    CANCHOGPU_EMU_CACHE=$ROOT/build/emu/cache CANCHOGPU_EMU_MUTATE="$mut" \
    "$ROOT/build/cancho-gpu-device" run "$ROOT/$file" $consts >"$WORKM" 2>&1 || status=$?
  if [ "$status" -ne 1 ] || ! grep -q "$want" "$WORKM"; then
    echo "mutant $file $consts ($mut) was not refused with \`$want\` (exit $status): $(head -c 300 "$WORKM")"; return 1
  fi
}
WORKM=$(mktemp)
mfail=0
mutant 'device-mismatch' tests/chx/ported/rmsnorm.chx "n=4096 eps=1e-5" '__syncthreads();=>' || mfail=1
mutant 'device-mismatch' tests/chx/ported/gemm_mma.chx "m=128 n=128 k=64" 'mem_row_major=>mem_col_major' || mfail=1
mutant 'past the end of buffer' tests/chx/ported/rmsnorm.chx "n=4096 eps=1e-5" 'j < 4096u=>j < 4352u' || mfail=1
mutant '256-bit' tests/chx/ported/gemm_mma.chx "m=128 n=128 k=64" '(arena + 18432)=>(arena + 18448)' || mfail=1
mutant 'device-mismatch' tests/chx/softmax.chx "r=4 n=64" 'expf(=>exp2f(' || mfail=1
[ "$mfail" -eq 0 ] || exit 1
echo "5 mutants of emitted kernels refused as they must be"
OUT=${CUDA_EMU_OUT:-$(mktemp -d)}
mkdir -p "$OUT"
if [ -z "${CUDA_EMU_OUT:-}" ]; then trap 'rm -rf "$OUT" "$WORKM"' EXIT; else trap 'rm -f "$WORKM"' EXIT; fi
: >"$OUT/cases.txt"
while read -r file rest; do
  case "$file" in ''|'#'*) continue ;; esac
  # shellcheck disable=SC2086
  name=$("$ROOT/scripts/case-name.sh" "$file" $rest)
  ls "$ROOT/golden/$name/"*.cu >/dev/null 2>&1 || continue
  if [ -n "${CUDA_EMU_ONLY:-}" ] && ! echo "$name" | grep -Eq "$CUDA_EMU_ONLY"; then continue; fi
  echo "$file${rest:+ $rest}" >>"$OUT/cases.txt"
done <"$ROOT/tests/cases.txt"
total=$(wc -l <"$OUT/cases.txt" | tr -d ' ')
# One case a line; NUL-separated so a line is one argument.
tr '\n' '\0' <"$OUT/cases.txt" | xargs -0 -P "${JOBS:-4}" -n 1 sh -c '"$0/tests/emu/run-case.sh" "$1" "$2"' "$ROOT" "$OUT" >"$OUT/results.txt" 2>"$OUT/errors.txt" || true
sort -o "$OUT/results.txt" "$OUT/results.txt"
passed=$(grep -c '^PASS ' "$OUT/results.txt" || true)
failed=$(grep -c '^FAIL ' "$OUT/results.txt" || true)
fail=0
if [ $((passed + failed)) -ne "$total" ]; then
  echo "only $((passed + failed)) of $total cases reported:"; head -5 "$OUT/errors.txt"; fail=1
fi
known_bug=0
known_noref=0
while IFS='|' read -r name kind _; do
  case "$name" in ''|'#'*) continue ;; esac
  line=$(grep "^FAIL $name " "$OUT/results.txt" || true)
  if [ -z "$line" ]; then
    echo "$name: listed in tests/cuda-emu-known.txt as $kind but it does not fail -- remove the line"; fail=1
  elif [ "$kind" = noref ]; then
    case "$line" in *'error[interp]'*) known_noref=$((known_noref + 1)) ;; *) echo "$name: listed as noref but failed otherwise: $line"; fail=1 ;; esac
  else
    known_bug=$((known_bug + 1))
  fi
done <"$ROOT/tests/cuda-emu-known.txt"
grep '^FAIL ' "$OUT/results.txt" | while read -r _ name rest; do
  if ! grep -q "^$name|" "$ROOT/tests/cuda-emu-known.txt"; then
    echo "FAIL $name $rest"
  fi
done >"$OUT/unexpected.txt"
if [ -s "$OUT/unexpected.txt" ]; then cat "$OUT/unexpected.txt"; fail=1; fi
echo "$total emitting cases through the emulator: $passed passed, $failed failed ($known_bug real bugs in the emitted CUDA, $known_noref without a reference: the interpreter cannot hold them)"
exit $fail
