#!/usr/bin/env bash
# Runs ON the NVIDIA VM (started by cuda_test.sh), from the home directory with
# the source unpacked in ./cancho-gpu. Writes everything to ~/results.
#
#   bash cancho-gpu/scripts/gcp/remote_cuda.sh            # the steps in $STEPS
#   bash cancho-gpu/scripts/gcp/remote_cuda.sh prepare    # build_image.sh: no GPU needed
#
# Steps (STEPS, space separated; default the first six). Each is logged to
# ~/results/<step>.log and is INDEPENDENT: one failing does not stop the others
# (a step whose own prerequisite failed says SKIP, not FAIL).
#
#   machine       1. nvidia-smi, driver, CUDA, CPU
#   toolchain        clang, Rust, the compiler at the cancho.toml pin, cancho-gpu
#   gate          2. the gate's GPU-free steps, one by one (build/test/goldens/
#                    refusals/kernels vs float64/mock-driver device path/...)
#   nvrtc         3. the CUDA in golden/ under the REAL NVRTC, compute_89 (+ ptxas)
#   device_cases  4. tests/cases.txt through the GPU against the interpreter
#   gemm_bench    5. our emitted GEMM vs cuBLAS vs torch, TFLOPS
#   job           6. $JOB, alone (JOB=1 from the driver; the command is ~/job.sh)
#   baselines     7. torch + peft LoRA, Qwen2.5-0.5B/1.5B at 4x512 (opt-in)
#
# Environment: CASE_TIMEOUT (300, s per case in step 4), JOBS (4), SIZES
# (1024,2048,4096,8192), DTYPES ("float32 float16"), BASE_MODELS ("Qwen/Qwen2.5-0.5B
# Qwen/Qwen2.5-1.5B"), KNOWN_OPEN (gate steps expected to fail: "tests/sampler.sh"),
# TORCH_INDEX (https://download.pytorch.org/whl/cu126), CANCHO_REPO.
#
# THE EMULATOR HOOK: another branch (cuda-emu) is building a CUDA-on-CPU
# emulating mock driver. When it lands, its entry point is run by the `gate`
# step through gate.sh itself (a gate line), and `emulator_hook` below also runs
# $EMU_ENTRY (default tests/cuda-emu.sh) when that file exists, so the
# emulator's verdict sits in the same results as the real GPU's. Nothing here
# depends on it.
#
# Exit status: 0 when no step FAILED; 1 otherwise.
set -uo pipefail

MODE="${1:-run}"
STEPS="${STEPS:-machine toolchain gate nvrtc device_cases gemm_bench}"
CASE_TIMEOUT="${CASE_TIMEOUT:-300}"
JOBS="${JOBS:-4}"
SIZES="${SIZES:-1024,2048,4096,8192}"
DTYPES="${DTYPES:-float32 float16}"
BASE_MODELS="${BASE_MODELS:-Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B}"
KNOWN_OPEN="${KNOWN_OPEN:-tests/sampler.sh}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
CANCHO_REPO="${CANCHO_REPO:-https://github.com/alpibrusl/cancho}"
EMU_ENTRY="${EMU_ENTRY:-tests/cuda-emu.sh}"
R="$HOME/results"
SRC="$HOME/cancho-gpu"
VENV="$HOME/venv"
CANCHO_BIN="$HOME/cancho-bin/cancho"
export HF_HOME="$HOME/hf"
mkdir -p "$R"
cd "$SRC" || exit 1
for d in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-13; do
  if [ -d "$d" ]; then export CUDA_HOME="$d" CUDA_PATH="$d" PATH="$d/bin:$PATH" LD_LIBRARY_PATH="$d/lib64:${LD_LIBRARY_PATH:-}"; break; fi
done
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
REV="$(sed -n 's/^cancho = "\([0-9a-f]*\)".*/\1/p' cancho.toml)"

SKIP=77
want() { case " $STEPS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# ---- the toolchain: the same on a pre-built image and on a stock one --------
apt_tools() {
  command -v clang >/dev/null && command -v cc >/dev/null && python3 -c 'import numpy, jinja2' 2>/dev/null \
    && python3 -c 'import venv, ensurepip' 2>/dev/null && return 0
  sudo apt-get update -qq
  # clang: the cancho compiler's default backend shells out to it.
  # jinja2, numpy: tests/chat.sh and the kernel references (CI installs jinja2 too).
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential clang python3-venv \
    python3-jinja2 python3-numpy git curl
}
build_compiler() {
  # The compiler is the commit in cancho.toml (CI reads it the same way). An
  # image carries a build of it, marked with the rev.
  if [ -x "$CANCHO_BIN" ] && [ "$(cat "$HOME/cancho-bin/rev" 2>/dev/null)" = "$REV" ]; then
    echo "compiler $REV already built ($CANCHO_BIN)"; return 0
  fi
  if ! command -v cargo >/dev/null; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null || return 1
    . "$HOME/.cargo/env"
  fi
  rm -rf "$HOME/cancho"
  if [ -f "$HOME/cancho-src.tar.gz" ]; then
    mkdir -p "$HOME/cancho" && tar -xzf "$HOME/cancho-src.tar.gz" -C "$HOME/cancho" || return 1
  else
    git clone -q "$CANCHO_REPO" "$HOME/cancho" && git -C "$HOME/cancho" checkout -q "$REV" || return 1
  fi
  (cd "$HOME/cancho" && cargo build --release -p cancho) || return 1
  mkdir -p "$HOME/cancho-bin" && cp "$HOME/cancho/target/release/cancho" "$CANCHO_BIN" \
    && echo "$REV" > "$HOME/cancho-bin/rev"
}
pyenv() {
  # torch and cupy for the benchmark; transformers and peft only for baselines.
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV" || return 1
  local pkgs=""
  "$VENV/bin/python" -c 'import torch, cupy' 2>/dev/null || pkgs=1
  if [ -n "$pkgs" ]; then
    "$VENV/bin/pip" install -q numpy cupy-cuda12x || return 1
    "$VENV/bin/pip" install -q torch --index-url "$TORCH_INDEX" || return 1
  fi
  if want baselines || [ "${WITH_BASELINE_LIBS:-}" = 1 ]; then
    "$VENV/bin/python" -c 'import transformers, peft' 2>/dev/null \
      || "$VENV/bin/pip" install -q transformers peft accelerate || return 1
  fi
  "$VENV/bin/pip" freeze > "$R/pip-freeze.txt"
}

if [ "$MODE" = prepare ]; then
  # build_image.sh: everything but the GPU. Sets the image's contents.
  WITH_BASELINE_LIBS="${WITH_BASELINE_LIBS:-0}"
  apt_tools && build_compiler && pyenv || { echo "preparation failed"; exit 1; }
  if [ "$WITH_BASELINE_LIBS" = 1 ]; then
    for m in $BASE_MODELS; do
      "$VENV/bin/python" -c "from huggingface_hub import snapshot_download as s; s('$m')" || exit 1
    done
  fi
  echo "image contents ready"
  exit 0
fi

# ---- steps ------------------------------------------------------------------
s_machine() {
  # The pre-built image carries the driver already; with no module for the
  # kernel it booted there is nothing to wait for, only GPU time to lose.
  if [ -f "$HOME/cancho-bin/rev" ] && ! find "/lib/modules/$(uname -r)" -name 'nvidia.ko*' | grep -q .; then
    echo "no NVIDIA module for kernel $(uname -r) in the pre-built image; rebuild it (build_image.sh)"
    return 1
  fi
  for _ in $(seq 1 90); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
  nvidia-smi || { echo "no NVIDIA driver after 15 min"; return 1; }
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,clocks.max.sm,power.limit --format=csv
  { lscpu | head -20; nvcc --version 2>/dev/null || ls /usr/local | grep -i cuda; uname -r; } 2>&1
  if command -v curl >/dev/null; then curl -s -H 'Metadata-Flavor: Google' \
      http://metadata.google.internal/computeMetadata/v1/instance/machine-type; echo; fi
  # A shipped CUDA arch other than 8.9 changes what the other steps mean.
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
  [ "$cap" = "8.9" ] || echo "NOTE: compute capability $cap is not 8.9; the lowering is Ada's (docs/cloud.md)"
  return 0
}
s_toolchain() {
  apt_tools || return 1
  build_compiler || return 1
  "$CANCHO_BIN" --version 2>&1 | head -1
  CANCHO="$CANCHO_BIN" "$CANCHO_BIN" build || return 1
  ls -l build/cancho-gpu
}
gate_units() {
  # gate.sh's own lines, so this never drifts from it: one output line per
  # top-level command; an `if ... fi` block is one unit, its newlines written
  # as \001 and put back by the reader.
  awk '/^cd "\$ROOT"/{on=1; next} /^echo "gate: ok"/{on=0}
       on { if ($0 ~ /^if /) d++;
            if (d > 0) { buf = buf (buf == "" ? "" : "\001") $0 } else { print }
            if ($0 ~ /^fi/) { d--; if (d == 0) { print buf; buf = "" } } }' scripts/gate.sh
}
is_known_open() { case " $KNOWN_OPEN " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
s_gate() {
  [ -x build/cancho-gpu ] || { echo "no build/cancho-gpu: the toolchain step failed"; return $SKIP; }
  export CANCHO="$CANCHO_BIN" ROOT="$SRC"
  local fails=0 known=0 n=0 line unit head rc
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    unit="$(printf '%s' "$line" | tr '\001' '\n')"
    head="${unit%%$'\n'*}"
    n=$((n + 1))
    echo "--- gate unit $n: $head"
    sh -c "set -eu; $unit"; rc=$?
    if [ "$rc" = 0 ]; then
      is_known_open "$head" && echo "UNEXPECTED PASS of a known-open step: drop it from KNOWN_OPEN and fix the docs"
      echo "ok"
    elif is_known_open "$head"; then
      known=$((known + 1)); echo "KNOWN-OPEN, expected (the Linux sampler golden has an open mismatch): exit $rc"
    else
      fails=$((fails + 1)); echo "FAIL: exit $rc"
    fi
  done < <(gate_units)
  echo "gate units: $n, failed $fails, known-open $known"
  [ "$fails" = 0 ]
}
emulator_hook() {
  [ -f "$EMU_ENTRY" ] || { echo "no $EMU_ENTRY yet (the cuda-emu branch): nothing to run"; return $SKIP; }
  CANCHO="$CANCHO_BIN" bash "$EMU_ENTRY"
}
s_nvrtc() {
  local files
  files=$(ls golden/*/*.cu 2>/dev/null)
  [ -n "$files" ] || { echo "no golden .cu files"; return 1; }
  # shellcheck disable=SC2086
  python3 scripts/gcp/nvrtc_check.py --arch compute_89 --ptxas --json "$R/nvrtc.json" $files
}
s_device_cases() {
  [ -x build/cancho-gpu ] || { echo "no build/cancho-gpu: the toolchain step failed"; return $SKIP; }
  CANCHO="$CANCHO_BIN" scripts/build-device.sh || return 1
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  python3 scripts/gcp/device_cases.py --bin build/cancho-gpu-device --timeout "$CASE_TIMEOUT" \
    --jobs "$JOBS" --json "$R/device_cases.json"
}
s_gemm_bench() {
  [ -x build/cancho-gpu ] || { echo "no build/cancho-gpu: the toolchain step failed"; return $SKIP; }
  pyenv || return 1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv -lms 500 > "$R/gemm-clocks.csv" 2>&1 &
  local smi=$!
  "$VENV/bin/python" scripts/gcp/gemm_bench.py --cancho-gpu build/cancho-gpu --sizes "$SIZES" --json "$R/gemm.json"
  local rc=$?
  kill $smi 2>/dev/null
  return $rc
}
s_job() {
  [ -x build/cancho-gpu ] || { echo "no build/cancho-gpu: the toolchain step failed"; return $SKIP; }
  export CANCHO="$CANCHO_BIN"
  cat "$HOME/job.sh"
  bash -e -o pipefail "$HOME/job.sh"
}
s_baselines() {
  # Public Hugging Face downloads: these runs carry no user data. The repo's
  # script targets MPS; the copy below is the same code with the device named.
  WITH_BASELINE_LIBS=1 pyenv || return 1
  sed -e 's/"mps"/"cuda"/g' -e 's/torch\.mps\.synchronize()/torch.cuda.synchronize()/' \
      -e 's/torch\.mps\.driver_allocated_memory()/torch.cuda.max_memory_allocated()/' \
      examples/qwen/torch_baseline.py > "$R/torch_baseline_cuda.py"
  if grep -n 'mps' "$R/torch_baseline_cuda.py"; then
    echo "torch_baseline.py has MPS references this port did not translate; update the sed in remote_cuda.sh"
    return 1
  fi
  local rc=0 m d
  for m in $BASE_MODELS; do
    for d in $DTYPES; do
      "$VENV/bin/python" "$R/torch_baseline_cuda.py" --model "$m" --b 4 --t 512 --dtype "$d" --steps 8 || rc=1
    done
  done
  return $rc
}

# ---- run --------------------------------------------------------------------
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

if [ -n "${JOB:-}" ]; then
  # A job needs the driver and the toolchain, and nothing else.
  run_step machine s_machine
  run_step toolchain s_toolchain
  run_step job s_job
else
  want machine && run_step machine s_machine
  { want toolchain || want gate || want device_cases || want gemm_bench || want baselines; } && run_step toolchain s_toolchain
  want gate && { run_step gate s_gate; run_step emulator emulator_hook; }
  want nvrtc && run_step nvrtc s_nvrtc
  want device_cases && run_step device_cases s_device_cases
  want gemm_bench && run_step gemm_bench s_gemm_bench
  want baselines && run_step baselines s_baselines
fi

echo; echo "=== summary"; column -t -s "$(printf '\t')" "$R/summary.tsv" 2>/dev/null || cat "$R/summary.tsv"
if grep -q "FAIL" "$R/summary.tsv"; then exit 1; fi
exit 0
