#!/usr/bin/env bash
# Runs ON a machine with an AMD GPU and ROCm (started by hip_test.sh), from the
# home directory with the source unpacked in ./cancho-gpu. Writes everything to
# ~/results. docs/amd.md slice H6: the first time AMD's own compiler and an AMD
# card see this project's HIP text.
#
#   bash cancho-gpu/scripts/amd/remote_hip.sh
#
# Steps (STEPS, space separated). Each is logged to ~/results/<step>.log and is
# independent: one failing does not stop the others.
#
#   machine       ROCm's version, the card, the two libraries (libamdhip64, libhiprtc)
#   toolchain     clang, the compiler at the cancho.toml pin, cancho-gpu
#                 (scripts/gcp/remote_cuda.sh's own functions: one install path)
#   hiprtc        every golden .hip under the REAL hipRTC (scripts/amd/hiprtc_check.c):
#                 how many compile, and how many of those that name `__half`
#   wave          one hand-written kernel on the card (tests/emu/wave_selftest.c):
#                 the card's warpSize, and whether a shuffle is what the emulator models
#   device_cases  every case of tests/cases.txt that emits HIP, through the card,
#                 against the interpreter; first with the launch's arguments by
#                 `kernelParams`, and by `extra` if that way fails
#   job           $JOB alone (the command is ~/job.sh), with the toolchain built
#
# ~/results/answers.txt collects what docs/amd.md section 1 marks unconfirmed.
#
# Environment: ROCM_PATH (/opt/rocm), CASE_TIMEOUT (300 s a case), JOBS (4),
# HIP_ONLY (a regex on case names), CANCHO_RELEASE / CANCHO_FLAVOUR / CANCHO_REPO
# (as scripts/gcp/remote_cuda.sh).
# HIP_LOCAL=1: no install and no home directory: the checkout this file is in, its
# build/ as it stands, results in $HIP_RESULTS, and the libraries the caller names
# (CANCHOGPU_LIBHIP, CANCHOGPU_LIBHIPRTC). That is how tests/amd-cloud.sh runs this
# script against the emulator.
#
# Exit status: 0 when no step FAILED; 1 otherwise.
set -uo pipefail

STEPS="${STEPS:-machine toolchain hiprtc wave device_cases}"
[ -n "${JOB:-}" ] && STEPS="machine toolchain job"
CASE_TIMEOUT="${CASE_TIMEOUT:-300}"
JOBS="${JOBS:-4}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIP_LOCAL="${HIP_LOCAL:-0}"
SKIP=77

if [ "$HIP_LOCAL" = 1 ]; then
  SRC="$(cd "$(dirname "$0")/../.." && pwd)"
  R="${HIP_RESULTS:?HIP_LOCAL=1 needs HIP_RESULTS}"
  mkdir -p "$R"
  cd "$SRC" || exit 1
  want() { case " $STEPS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
  cancho_cmd() { echo "${CANCHO:-cancho}"; }
else
  # The install path is the NVIDIA script's, read as a library: it sets R=~/results,
  # SRC=~/cancho-gpu, changes into it, and defines want, s_toolchain and cancho_cmd.
  REMOTE_CUDA_LIB=1 MODE=run . "$HOME/cancho-gpu/scripts/gcp/remote_cuda.sh"
  export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}" PATH="$ROCM_PATH/bin:$PATH"
fi
ANS="$R/answers.txt"
: > "$ANS"

# The two libraries: the caller's, else ROCm's. hipRTC is its own library since
# ROCm 7.0 and was inside libamdhip64 before.
find_libs() {
  LIBHIP="${CANCHOGPU_LIBHIP:-}"
  LIBHIPRTC="${CANCHOGPU_LIBHIPRTC:-}"
  [ -n "$LIBHIP" ] || LIBHIP=$(ls "$ROCM_PATH"/lib/libamdhip64.so* 2>/dev/null | head -1)
  [ -n "$LIBHIPRTC" ] || LIBHIPRTC=$(ls "$ROCM_PATH"/lib/libhiprtc.so* 2>/dev/null | head -1)
  [ -n "$LIBHIPRTC" ] || LIBHIPRTC="$LIBHIP"
  [ -n "$LIBHIP" ] || { echo "no libamdhip64.so under $ROCM_PATH/lib (set ROCM_PATH or CANCHOGPU_LIBHIP)"; return 1; }
  export CANCHOGPU_LIBHIP="$LIBHIP" CANCHOGPU_LIBHIPRTC="$LIBHIPRTC" ROCM_PATH
}
# The include path hipRTC is given, when ROCm's headers are where it says.
inc_opt() { [ -f "$ROCM_PATH/include/hip/hip_fp16.h" ] && echo "-I$ROCM_PATH/include"; }

s_machine() {
  find_libs || return 1
  echo "libamdhip64: $LIBHIP"; echo "libhiprtc:   $LIBHIPRTC"
  cat "$ROCM_PATH/.info/version" 2>/dev/null || echo "no $ROCM_PATH/.info/version"
  { amd-smi static --gpu 0 2>/dev/null || rocm-smi --showproductname --showmeminfo vram 2>/dev/null || echo "neither amd-smi nor rocm-smi answered"; } | head -60
  { lscpu | head -12; uname -r; } 2>&1
  echo "rocm: $(cat "$ROCM_PATH/.info/version" 2>/dev/null || echo unknown)" >> "$ANS"
  return 0
}
s_hiprtc() {
  find_libs || return 1
  cc -std=c11 -O1 scripts/amd/hiprtc_check.c -o build/hiprtc_check -ldl || return 1
  local files inc opts=()
  files=$(ls golden/*/*.hip 2>/dev/null)
  if [ -n "${HIP_ONLY:-}" ]; then files=$(for f in $files; do basename "$(dirname "$f")" | grep -Eq "$HIP_ONLY" && echo "$f"; done); fi
  [ -n "$files" ] || { echo "no golden .hip files"; return 1; }
  inc=$(inc_opt); [ -n "$inc" ] && opts=(-o "$inc")
  [ -n "${CANCHOGPU_HIP_ARCH:-}" ] && opts+=(-o "--gpu-architecture=$CANCHOGPU_HIP_ARCH")
  # shellcheck disable=SC2086
  build/hiprtc_check "$LIBHIPRTC" ${opts[@]+"${opts[@]}"} $files > "$R/hiprtc.txt" 2> "$R/hiprtc-logs.txt"
  local rc=$? ok bad half_bad
  ok=$(grep -c '^ok ' "$R/hiprtc.txt"); bad=$(grep -c '^FAIL ' "$R/hiprtc.txt")
  half_bad=0
  for f in $(sed -n 's/^FAIL //p' "$R/hiprtc.txt"); do grep -q '__half' "$f" && half_bad=$((half_bad + 1)); done
  tail -1 "$R/hiprtc.txt"
  echo "of the $bad that failed, $half_bad name __half"
  head -40 "$R/hiprtc-logs.txt"
  { echo "hipRTC compiled $ok of $((ok + bad)) golden .hip files"
    echo "halves (hip/hip_fp16.h under hipRTC): $(grep -l '__half' $files | wc -l | tr -d ' ') files name __half, $half_bad of them failed to compile"
  } >> "$ANS"
  return $rc
}
s_wave() {
  find_libs || return 1
  cc -std=c11 -O1 tests/emu/wave_selftest.c -o build/wave_selftest -ldl || return 1
  local rc
  WAVE_HIPRTC_OPT="$(inc_opt)" build/wave_selftest "$LIBHIP" "$LIBHIPRTC" real | tee "$R/wave.txt"
  rc=${PIPESTATUS[0]}
  echo "the wave: $(tr '\n' ';' < "$R/wave.txt")" >> "$ANS"
  return "$rc"
}
s_device_cases() {
  find_libs || return 1
  [ -x build/cancho-gpu ] || { echo "no build/cancho-gpu: the toolchain step failed"; return $SKIP; }
  [ "$HIP_LOCAL" = 1 ] || { CANCHO="$(cancho_cmd)" scripts/build-device.sh || return 1; }
  [ -x build/cancho-gpu-device ] || { echo "no build/cancho-gpu-device"; return 1; }
  local out="$R/cases" first way=params
  mkdir -p "$out"
  : > "$out/cases.txt"
  while read -r file rest; do
    case "$file" in ''|'#'*) continue ;; esac
    # shellcheck disable=SC2086
    name=$(scripts/case-name.sh "$file" $rest)
    ls "golden/$name/"*.hip >/dev/null 2>&1 || continue
    if [ -n "${HIP_ONLY:-}" ] && ! echo "$name" | grep -Eq "$HIP_ONLY"; then continue; fi
    echo "$file${rest:+ $rest}" >> "$out/cases.txt"
  done < tests/cases.txt
  [ -s "$out/cases.txt" ] || { echo "no case emits HIP"; return 1; }
  # Which way a launch takes its arguments (docs/amd.md section 1, unconfirmed): the first case decides.
  first=$(head -1 "$out/cases.txt")
  if ! EMU_BACKEND=real CASE_TIMEOUT="$CASE_TIMEOUT" tests/emu/run-case.sh "$out" "$first" | grep -q '^PASS '; then
    if EMU_BACKEND=real CASE_TIMEOUT="$CASE_TIMEOUT" CANCHOGPU_HIP_ARGS=extra tests/emu/run-case.sh "$out" "$first" | grep -q '^PASS '; then
      way=extra
    else
      echo "the first case ($first) passes with its arguments neither way; running the rest by kernelParams"
      way="params (the first case failed both ways)"
    fi
  fi
  echo "a launch's arguments: $way" | tee -a "$ANS"
  [ "$way" = extra ] && export CANCHOGPU_HIP_ARGS=extra
  tr '\n' '\0' < "$out/cases.txt" | EMU_BACKEND=real CASE_TIMEOUT="$CASE_TIMEOUT" xargs -0 -P "$JOBS" -n 1 sh -c '"$0/tests/emu/run-case.sh" "$1" "$2"' "$SRC" "$out" > "$R/device_cases.txt" 2> "$R/device_cases.err"
  sort -o "$R/device_cases.txt" "$R/device_cases.txt"
  local total passed noref mismatch other
  total=$(wc -l < "$out/cases.txt" | tr -d ' ')
  passed=$(grep -c '^PASS ' "$R/device_cases.txt")
  noref=$(grep -c 'error\[interp\]' "$R/device_cases.txt")
  mismatch=$(grep -c 'device-mismatch\|\[tight:' "$R/device_cases.txt")
  other=$((total - passed - noref - mismatch))
  echo "$total cases that emit HIP on the card: $passed agree with the interpreter, $mismatch disagree, $noref without a reference, $other failed otherwise" | tee -a "$ANS"
  grep '^FAIL ' "$R/device_cases.txt" | grep -v 'error\[interp\]' | head -40
  [ "$mismatch" -eq 0 ] && [ "$other" -eq 0 ]
}
s_job() {
  [ -f "$HOME/job.sh" ] || { echo "no ~/job.sh"; return 1; }
  find_libs || return 1
  CANCHO="$(cancho_cmd)" bash "$HOME/job.sh"
}

: > "$R/summary.tsv"
run_step() { # name function
  local name=$1 fn=$2 t0=$SECONDS rc
  echo; echo "=== $name"
  "$fn" 2>&1 | tee "$R/$name.log"
  rc=${PIPESTATUS[0]}
  local status=PASS; [ "$rc" = $SKIP ] && status=SKIP; { [ "$rc" != 0 ] && [ "$rc" != $SKIP ]; } && status=FAIL
  printf '%s\t%s\t%ss\n' "$name" "$status" "$((SECONDS - t0))" >> "$R/summary.tsv"
  echo "=== $name: $status (exit $rc, $((SECONDS - t0))s)"
}
want machine && run_step machine s_machine
want toolchain && [ "$HIP_LOCAL" != 1 ] && run_step toolchain s_toolchain
want hiprtc && run_step hiprtc s_hiprtc
want wave && run_step wave s_wave
want device_cases && run_step device_cases s_device_cases
want job && run_step job s_job
echo; echo "=== answers (docs/amd.md section 1)"; cat "$ANS"
echo; echo "=== summary"; cat "$R/summary.tsv"
if grep -q "FAIL" "$R/summary.tsv"; then exit 1; fi
exit 0
