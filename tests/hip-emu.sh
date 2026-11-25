#!/bin/sh
# The emitted HIP, run: every case of tests/cases.txt that emits a `.hip`
# kernel goes through `cancho-gpu-device` with the EMULATING HIP runtime and
# hipRTC (tests/emu/libamdhip64_emu.c, libhiprtc_emu.c; docs/amd.md section
# 7), which compile the text with the system C++ compiler and run the grid on
# the CPU in waves of 64 lanes, and the result is held to the interpreter as
# tests/cuda-emu.sh holds the CUDA. Before the cases:
#
#   * the emulator's own width (build/emu/wave_selftest): one hand-written
#     kernel sums as one wave at 64 lanes and as two at 32;
#   * mutants of emitted HIP kernels, which must be refused;
#   * the CUDA text at 64 lanes: as emitted it must be refused, because its
#     32-bit shuffle mask does not name the upper half of a wave; with the
#     mask widened to 64 bits and nothing else changed it must AGREE with
#     the interpreter (two kernels here; HIP_EMU_CUDA64=1 runs every
#     shuffling case, which is the measurement of docs/amd.md section 7).
#
# It says nothing of AMD's compiler or of an AMD GPU: no ROCm is involved.
#
#     CANCHO=<pinned cancho> tests/hip-emu.sh
#
#     JOBS=4                cases in parallel (CPU only; no GPU is touched)
#     HIP_EMU_OUT=<dir>     keep every case's output and results.txt there
#     HIP_EMU_ONLY=<regex>  only the cases whose name matches
#     HIP_EMU_CUDA64=1      instead of the HIP cases: every case whose CUDA
#                           text shuffles and names no wmma, at 64 lanes
#                           with the mask widened
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
"$ROOT/scripts/build-emu.sh"
"$ROOT/scripts/build-device.sh" >/dev/null
EMU=$ROOT/build/emu
CANCHOGPU_EMU_CACHE=$EMU/cache "$EMU/wave_selftest" "$EMU/libamdhip64_emu.so" "$EMU/libhiprtc_emu.so"
WORKM=$(mktemp)
refused() { # <what> <expected text in the output> <file> <consts>, the environment from the caller
  what=$1; want=$2; file=$3; consts=$4
  status=0
  # shellcheck disable=SC2086
  CANCHOGPU_EMU_CACHE=$EMU/cache "$ROOT/build/cancho-gpu-device" run "$ROOT/$file" $consts >"$WORKM" 2>&1 || status=$?
  if [ "$status" -ne 1 ] || ! grep -q "$want" "$WORKM"; then
    echo "$what: $file $consts was not refused with \`$want\` (exit $status): $(tail -c 300 "$WORKM")"; return 1
  fi
}
hip_mutant() { # <expected> <file> <consts> <from=>to>
  CANCHOGPU_LIBHIP=$EMU/libamdhip64_emu.so CANCHOGPU_LIBHIPRTC=$EMU/libhiprtc_emu.so CANCHOGPU_EMU_MUTATE="$4" \
    refused "hip mutant ($4)" "$1" "$2" "$3"
}
cuda_at_64() { # <expected> <file> <consts> [<from=>to>]
  CANCHOGPU_LIBCUDA=$EMU/libcuda_emu.so CANCHOGPU_LIBNVRTC=$EMU/libnvrtc_emu.so CANCHOGPU_EMU_WARP=64 CANCHOGPU_EMU_MUTATE="${4:-}" \
    refused "the CUDA text at 64 lanes${4:+ ($4)}" "$1" "$2" "$3"
}
mfail=0
hip_mutant 'device-mismatch' tests/chx/ported/silu_mul.chx "" 'expf(=>exp2f(' || mfail=1
hip_mutant 'illegal memory access' tests/chx/attn_gate.chx "r=24 n=256" '+ (e)] = =>+ (e) + 256u] = ' || mfail=1
hip_mutant 'device-mismatch' tests/chx/ported/rmsnorm.chx "n=4096 eps=1e-5" '__syncthreads();=>' || mfail=1
hip_mutant 'does not name' tests/chx/ported/rmsnorm.chx "n=4096 eps=1e-5" '0xffffffffffffffffull=>0xffffffffu' || mfail=1
hip_mutant 'kernel.hip(' tests/chx/ported/silu_mul.chx "" 'float v4[1];=>nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> v4;' || mfail=1
WIDE='0xffffffffu=>0xffffffffffffffffull'
cuda_at_64 'does not name' tests/chx/ported/rmsnorm.chx "n=4096 eps=1e-5" || mfail=1
agrees() { # <file> <consts>: the CUDA text, 64 lanes, the mask widened
  status=0
  # shellcheck disable=SC2086
  CANCHOGPU_LIBCUDA=$EMU/libcuda_emu.so CANCHOGPU_LIBNVRTC=$EMU/libnvrtc_emu.so CANCHOGPU_EMU_WARP=64 CANCHOGPU_EMU_MUTATE="$WIDE" \
    CANCHOGPU_EMU_CACHE=$EMU/cache "$ROOT/build/cancho-gpu-device" run "$ROOT/$1" $2 >"$WORKM" 2>&1 || status=$?
  if [ "$status" -ne 0 ] || grep -q '^emu-' "$WORKM"; then
    echo "the CUDA text at 64 lanes with a 64-bit mask: $1 $2 did not agree (exit $status): $(tail -c 300 "$WORKM")"; return 1
  fi
}
agrees tests/chx/ported/rmsnorm.chx "n=4096 eps=1e-5" || mfail=1
agrees tests/chx/softmax.chx "r=4 n=64" || mfail=1
[ "$mfail" -eq 0 ] || exit 1
echo "5 mutants of emitted HIP kernels refused; the CUDA text at 64 lanes refused for its 32-bit mask, and right with a 64-bit one (2 kernels)"
OUT=${HIP_EMU_OUT:-$(mktemp -d)}
mkdir -p "$OUT"
if [ -z "${HIP_EMU_OUT:-}" ]; then trap 'rm -rf "$OUT" "$WORKM"' EXIT; else trap 'rm -f "$WORKM"' EXIT; fi
: >"$OUT/cases.txt"
while read -r file rest; do
  case "$file" in ''|'#'*) continue ;; esac
  # shellcheck disable=SC2086
  name=$("$ROOT/scripts/case-name.sh" "$file" $rest)
  if [ -n "${HIP_EMU_CUDA64:-}" ]; then
    ls "$ROOT/golden/$name/"*.cu >/dev/null 2>&1 || continue
    grep -q '__shfl' "$ROOT/golden/$name/"*.cu || continue
    if grep -q 'wmma' "$ROOT/golden/$name/"*.cu; then continue; fi
  else
    ls "$ROOT/golden/$name/"*.hip >/dev/null 2>&1 || continue
  fi
  if [ -n "${HIP_EMU_ONLY:-}" ] && ! echo "$name" | grep -Eq "$HIP_EMU_ONLY"; then continue; fi
  echo "$file${rest:+ $rest}" >>"$OUT/cases.txt"
done <"$ROOT/tests/cases.txt"
total=$(wc -l <"$OUT/cases.txt" | tr -d ' ')
if [ -n "${HIP_EMU_CUDA64:-}" ]; then
  export EMU_BACKEND=cuda CANCHOGPU_EMU_WARP=64 CANCHOGPU_EMU_MUTATE="$WIDE"
  what="cases whose CUDA text shuffles, at 64 lanes with a 64-bit mask"
else
  export EMU_BACKEND=hip
  what="cases that emit HIP through the emulator at 64 lanes"
fi
tr '\n' '\0' <"$OUT/cases.txt" | xargs -0 -P "${JOBS:-4}" -n 1 sh -c '"$0/tests/emu/run-case.sh" "$1" "$2"' "$ROOT" "$OUT" >"$OUT/results.txt" 2>"$OUT/errors.txt" || true
sort -o "$OUT/results.txt" "$OUT/results.txt"
passed=$(grep -c '^PASS ' "$OUT/results.txt" || true)
failed=$(grep -c '^FAIL ' "$OUT/results.txt" || true)
fail=0
if [ $((passed + failed)) -ne "$total" ]; then
  echo "only $((passed + failed)) of $total cases reported:"; head -5 "$OUT/errors.txt"; fail=1
fi
# A case the interpreter cannot hold has no reference under either emulator:
# the `noref` lines of tests/cuda-emu-known.txt are the list, and such a case
# must fail that way here too. Nothing else may fail.
grep '^FAIL ' "$OUT/results.txt" | while read -r _ name rest; do
  if grep -q "^$name|noref|" "$ROOT/tests/cuda-emu-known.txt"; then
    case "$rest" in *'error[interp]'*) ;; *) echo "FAIL $name $rest (listed noref, failed otherwise)" ;; esac
  else
    echo "FAIL $name $rest"
  fi
done >"$OUT/unexpected.txt"
if [ -s "$OUT/unexpected.txt" ]; then cat "$OUT/unexpected.txt"; fail=1; fi
noref=$((failed - $(wc -l <"$OUT/unexpected.txt" | tr -d ' ')))
echo "$total $what: $passed passed, $failed failed ($noref without a reference: the interpreter cannot hold them)"
exit $fail
