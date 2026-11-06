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
#   gemm_bench    5. our emitted GEMM vs cuBLAS vs torch, TFLOPS, with clocks/power/occupancy per shape
#   shuffle_repro    the emulator's two NVFP4 shuffle bugs on the real GPU (shuffle_repro.txt)
#   fp4_perf         the NVFP4 matvec kernels, FP4_V[nibble] vs the old __shfl_sync lowering: GB/s and ratio (fp4_perf.py)
#   sandbox_linux    why tests/weightspace.py's network-call check fails on Linux (audit_probe.py)
#   job           6. $JOB, alone (JOB=1 from the driver; the command is ~/job.sh)
#   baselines     7. torch + peft LoRA, Qwen2.5-0.5B/1.5B at 4x512 (opt-in)
#   w1               W1's behavioural run (docs/weight-space.md 13), in chunks, resumable (opt-in; w1_cloud.sh drives it
#                    with STEPS='machine w1': no compiler is needed). See s_w1 for its W1_* settings.
#
# Environment: CASE_TIMEOUT (300, s per case in step 4), JOBS (4), SIZES
# (1024,2048,4096,8192), DTYPES ("float32 float16"), BASE_MODELS ("Qwen/Qwen2.5-0.5B
# Qwen/Qwen2.5-1.5B"), KNOWN_OPEN (gate steps expected to fail: "tests/sampler.sh"),
# TORCH_INDEX (https://download.pytorch.org/whl/cu126), CANCHO_REPO.
# The compiler: CANCHO_RELEASE (default latest; a tag, a sha, or `source`) installs
# the cancho-rust-<sha>-linux-x86_64 release asset after checking its published
# sha256, and builds the pinned commit from source if that fails. CANCHO_FLAVOUR
# (rust | standalone). A release whose rev is not the cancho.toml pin runs with
# --ignore-compiler-rev, and summary.tsv says NOT-PINNED.
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
STEPS="${STEPS:-machine toolchain gate nvrtc device_cases gemm_bench shuffle_repro fp4_perf sandbox_linux}"
CASE_TIMEOUT="${CASE_TIMEOUT:-300}"
JOBS="${JOBS:-4}"
SIZES="${SIZES:-}"                 # empty: gemm_bench.py's own sizes and shapes
DTYPES="${DTYPES:-float32 float16}"
BASE_MODELS="${BASE_MODELS:-Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B}"
KNOWN_OPEN="${KNOWN_OPEN:-tests/sampler.sh}"
W1_PIP="${W1_PIP:-transformers>=5.2.0 pyarrow huggingface_hub numpy}"
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
need_python_ml() { want gemm_bench || want fp4_perf || want baselines || want w1 || [ "${WITH_BASELINE_LIBS:-0}" = 1 ]; }
apt_tools() {
  # Only what the requested steps use. build-essential: `cc` for the shim and
  # the mock drivers. numpy/jinja2: the gate's references (CI installs jinja2
  # too). venv: the benchmark's torch/cupy. git: only to build from source.
  local pkgs="build-essential curl ca-certificates" p missing=""
  if [ "$MODE" = prepare ] || want gate || want sandbox_linux; then pkgs="$pkgs python3-jinja2 python3-numpy"; fi
  if [ "$MODE" = prepare ] || need_python_ml; then pkgs="$pkgs python3-venv"; fi
  if [ "${CANCHO_RELEASE:-latest}" = source ]; then pkgs="$pkgs git"; fi
  for p in $pkgs; do dpkg -s "$p" >/dev/null 2>&1 || missing="$missing $p"; done
  if [ -n "$missing" ] || ! pick_clang; then
    sudo apt-get update -qq
    # shellcheck disable=SC2086
    [ -z "$missing" ] || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $missing
    pick_clang || for c in clang-18 clang-17 clang-16 clang-15 clang; do
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$c" && break
    done
  fi
  pick_clang || { echo "no clang >= 15 (the compiler's default backend shells out to it)"; return 1; }
  plain_clang || return 1
  echo "CLANG=$CLANG ($("$CLANG" --version | head -1))"
  { echo "CLANG=$CLANG"; clang --version; } >> "$R/machine.txt" 2>&1
}
plain_clang() {
  # cancho runs a plain `clang` found by NAME on PATH in places, not only $CLANG
  # ("cannot run `clang`: No such file or directory" on an image with only
  # clang-18). Make `clang` resolve to a version it can read (>= 15): a symlink
  # first on PATH when it is missing or too old.
  local v
  v=$(clang --version 2>/dev/null | sed -n 's/.*version \([0-9]*\)\..*/\1/p' | head -1)
  if [ "${v:-0}" -lt 15 ]; then
    sudo ln -sf "$(command -v "$CLANG")" "${PLAIN_CLANG_DIR:-/usr/local/bin}/clang" || return 1
    hash -r
  fi
  [ -e "${PLAIN_CLANG_ALT:-/usr/bin}/clang" ] || sudo ln -sf "$(command -v "$CLANG")" "${PLAIN_CLANG_ALT:-/usr/bin}/clang" || true
  v=$(clang --version 2>/dev/null | sed -n 's/.*version \([0-9]*\)\..*/\1/p' | head -1)
  [ "${v:-0}" -ge 15 ] || { echo "plain \`clang\` is still not >= 15"; return 1; }
}
pick_clang() {
  # The LLVM IR the compiler writes is too new for clang 14 (Ubuntu 22.04's
  # default `clang`; docs/standalone-cancho.md), and the Rust build reads $CLANG.
  local c v
  for c in clang-18 clang-17 clang-16 clang-15 clang; do
    command -v "$c" >/dev/null || continue
    v=$("$c" --version | sed -n 's/.*version \([0-9]*\)\..*/\1/p' | head -1)
    if [ "${v:-0}" -ge 15 ]; then export CLANG="$c"; return 0; fi
  done
  return 1
}
RELEASES="${CANCHO_RELEASES_BASE:-https://github.com/alpibrusl/cancho/releases}"
CANCHO_RELEASE="${CANCHO_RELEASE:-latest}"   # latest | cancho-<sha> | <sha> | source
CANCHO_FLAVOUR="${CANCHO_FLAVOUR:-rust}"     # rust (CI's compiler; reads $CLANG) | standalone
ID_FILE="$HOME/cancho-bin/id"
resolve_release() {
  # The repository is public, so the VM downloads. (A private one: ship a
  # local checkout with CANCHO_REPO=<dir> and CANCHO_RELEASE=source.)
  local final
  case "$CANCHO_RELEASE" in
    latest)
      final=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "$RELEASES/latest") || return 1
      REL_TAG="${final##*/tag/}" ;;
    cancho-*) REL_TAG="$CANCHO_RELEASE" ;;
    *) REL_TAG="cancho-$CANCHO_RELEASE" ;;
  esac
  case "$REL_TAG" in cancho-[0-9a-f]*) ;; *) echo "not a release tag: $REL_TAG"; return 1 ;; esac
  REL_REV="${REL_TAG#cancho-}"
}
fetch_release() {
  local base name d want got bin
  case "$CANCHO_FLAVOUR" in
    rust) name="cancho-rust-$REL_REV-linux-x86_64" ;;
    standalone) name="cancho-$REL_REV-linux-x86_64" ;;
    *) echo "CANCHO_FLAVOUR must be rust or standalone"; return 1 ;;
  esac
  d=$(mktemp -d)
  base="$RELEASES/download/$REL_TAG"
  curl -fsSL -o "$d/$name.tar.gz" "$base/$name.tar.gz" && curl -fsSL -o "$d/$name.tar.gz.sha256" "$base/$name.tar.gz.sha256" \
    || { echo "download of $name failed"; rm -rf "$d"; return 1; }
  want=$(awk '{print $1; exit}' "$d/$name.tar.gz.sha256")
  got=$(sha256sum "$d/$name.tar.gz" | awk '{print $1}')
  if [ -z "$want" ] || [ "$want" != "$got" ]; then
    echo "sha256 MISMATCH for $name.tar.gz (published $want, got $got): not used"
    rm -rf "$d"; return 1
  fi
  echo "sha256 ok: $got"
  tar -xzf "$d/$name.tar.gz" -C "$d" || { rm -rf "$d"; return 1; }
  bin=$(find "$d" -path '*/bin/cancho' -type f | head -1)
  [ -n "$bin" ] || { echo "no bin/cancho in $name.tar.gz"; rm -rf "$d"; return 1; }
  mkdir -p "$HOME/cancho-bin" && cp "$bin" "$CANCHO_BIN" && chmod +x "$CANCHO_BIN" || { rm -rf "$d"; return 1; }
  # A release must know which commit it is, or the project refuses it
  # ("does not know which commit it was built from"). The rust asset prints
  # `cancho 0.0.0 (rev <sha8>...`; the standalone one has no --version yet.
  if [ "$CANCHO_FLAVOUR" = rust ] && ! "$CANCHO_BIN" --version 2>&1 | grep -q "${REL_REV:0:8}"; then
    echo "the release binary does not report rev ${REL_REV:0:8} in --version: not used"
    rm -f "$CANCHO_BIN"; return 1
  fi
  echo "$REL_REV" > "$HOME/cancho-bin/rev"
  echo "release:$REL_TAG:$CANCHO_FLAVOUR" > "$ID_FILE"
  rm -rf "$d"
}
build_compiler_source() {
  # The pinned commit, built here: needs Rust (installed only on this path).
  if [ -x "$CANCHO_BIN" ] && [ "$(cat "$ID_FILE" 2>/dev/null)" = "source:$REV" ]; then
    echo "compiler $REV already built ($CANCHO_BIN)"; return 0
  fi
  if ! command -v git >/dev/null; then sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git || return 1; fi
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
  # The source arrives without .git (git archive), and the compiler stamps the
  # commit it was built from, then refuses a project that pins another: stamp
  # the pin explicitly (CANCHO_REV) or every project build is refused.
  (cd "$HOME/cancho" && CANCHO_REV="$REV" cargo build --release -p cancho) || return 1
  mkdir -p "$HOME/cancho-bin" && cp "$HOME/cancho/target/release/cancho" "$CANCHO_BIN" \
    && echo "$REV" > "$HOME/cancho-bin/rev" && echo "source:$REV" > "$ID_FILE"
}
build_compiler() {
  # A release binary by default, checked against its published sha256; on any
  # failure (no network, no release, mismatch) the pinned commit is built.
  if [ "$CANCHO_RELEASE" != source ]; then
    if resolve_release; then
      if [ -x "$CANCHO_BIN" ] && [ "$(cat "$ID_FILE" 2>/dev/null)" = "release:$REL_TAG:$CANCHO_FLAVOUR" ]; then
        echo "compiler $REL_TAG ($CANCHO_FLAVOUR) already installed"; return 0
      fi
      fetch_release && return 0
    fi
    echo "release install failed: falling back to building the pinned commit $REV from source"
  fi
  build_compiler_source
}
# The compiler command the steps use. cancho-gpu refuses any compiler but the
# one in cancho.toml for project builds; when the installed one is a different
# rev, a wrapper adds --ignore-compiler-rev to project-mode build/test (file
# mode takes no such flag). Results are then NOT pinned-compiler results, and
# the summary says so.
cancho_cmd() {
  if [ "$(cat "$HOME/cancho-bin/rev" 2>/dev/null)" = "$REV" ]; then echo "$CANCHO_BIN"; return; fi
  local w="$HOME/cancho-bin/cancho-ignore-rev"
  mkdir -p "$HOME/cancho-bin"
  cat > "$w" <<WRAP
#!/bin/sh
REAL=$CANCHO_BIN
case "\$1" in
  build|test)
    for a in "\$@"; do case "\$a" in *.cho) exec "\$REAL" "\$@" ;; esac; done
    exec "\$REAL" "\$@" --ignore-compiler-rev ;;
  *) exec "\$REAL" "\$@" ;;
esac
WRAP
  chmod +x "$w"; echo "$w"
}
pyenv() {
  # torch and cupy for the benchmark; transformers and peft only for baselines; the w1 step needs torch, transformers, pyarrow
  # and huggingface_hub, and cupy only if a benchmark step runs too.
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV" || return 1
  local pkgs="" probe='import torch, cupy'
  if want w1 && ! want gemm_bench && ! want fp4_perf && ! want baselines; then probe='import torch'; fi
  "$VENV/bin/python" -c "$probe" 2>/dev/null || pkgs=1
  if [ -n "$pkgs" ]; then
    "$VENV/bin/pip" install -q numpy || return 1
    [ "$probe" = 'import torch' ] || "$VENV/bin/pip" install -q cupy-cuda12x || return 1
    "$VENV/bin/pip" install -q torch --index-url "$TORCH_INDEX" || return 1
  fi
  if want baselines || [ "${WITH_BASELINE_LIBS:-}" = 1 ]; then
    "$VENV/bin/python" -c 'import transformers, peft' 2>/dev/null \
      || "$VENV/bin/pip" install -q transformers peft accelerate || return 1
  fi
  if want w1 && ! "$VENV/bin/python" -c 'import transformers, pyarrow, huggingface_hub, numpy' 2>/dev/null; then
    # shellcheck disable=SC2086
    "$VENV/bin/pip" install -q $W1_PIP || return 1
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
  { lscpu | head -20; nvcc --version 2>/dev/null || ls -d /usr/local/*cuda* 2>/dev/null; uname -r; } 2>&1
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
  echo "compiler: $(cat "$ID_FILE"); cancho.toml pins $REV; installed rev $(cat "$HOME/cancho-bin/rev")"
  if [ "$(cat "$HOME/cancho-bin/rev")" != "$REV" ]; then
    echo "NOT THE PINNED COMPILER: building and testing with --ignore-compiler-rev"
  fi
  local cc_; cc_="$(cancho_cmd)"
  CANCHO="$cc_" "$cc_" build || return 1
  ls -l build/cancho-gpu
}
gate_units() {
  # gate.sh's own lines, so this never drifts from it: one output line per
  # top-level command; an `if ... fi` block is one unit, its newlines written
  # as \001 and put back by the reader.
  awk '/^cd "\$ROOT"/{on=1; next} /^echo "gate: ok"/{on=0}
       on { if ($0 ~ /^if /) d++;
            if (d > 0) { buf = buf (buf == "" ? "" : "\001") $0 } else { print }
            if ($0 ~ /^fi/) { d--; if (d == 0) { print buf; buf = "" } } }' "${GATE_FILE:-scripts/gate.sh}"
}
is_known_open() { case " $KNOWN_OPEN " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
s_gate() {
  [ -x build/cancho-gpu ] || { echo "no build/cancho-gpu: the toolchain step failed"; return $SKIP; }
  CANCHO="$(cancho_cmd)"; export CANCHO ROOT="$SRC"
  local fails=0 known=0 n=0 line unit head rc
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    unit="$(printf '%s' "$line" | tr '\001' '\n')"
    head="${unit%%$'\n'*}"
    n=$((n + 1))
    echo "--- gate unit $n: $head"
    if [ "$head" = tests/gcp.sh ]; then
      # The shipped tree is a `git archive`: no .git, and that test archives HEAD (exit 128).
      echo "SKIPPED on VM: tests/gcp.sh needs a git checkout (this tree has no .git); it runs in the repository's own gate"
      printf 'gate:tests/gcp.sh\tSKIPPED-ON-VM\tno .git in the shipped tree\n' >> "$R/summary.tsv"
      continue
    fi
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
  CANCHO="$(cancho_cmd)" bash "$EMU_ENTRY"
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
  CANCHO="$(cancho_cmd)" scripts/build-device.sh || return 1
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  python3 scripts/gcp/device_cases.py --bin build/cancho-gpu-device --timeout "$CASE_TIMEOUT" \
    --jobs "$JOBS" --json "$R/device_cases.json"
}
s_gemm_bench() {
  [ -x build/cancho-gpu ] || { echo "no build/cancho-gpu: the toolchain step failed"; return $SKIP; }
  pyenv || return 1
  # Clocks/power/throttle bits are sampled by gemm_bench.py itself, per shape and
  # implementation, at 100 ms; this whole-step trace is the coarse companion.
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv -lms 500 > "$R/gemm-clocks.csv" 2>&1 &
  local smi=$! sz=()
  [ -z "$SIZES" ] || sz=(--sizes "$SIZES")
  "$VENV/bin/python" scripts/gcp/gemm_bench.py --cancho-gpu build/cancho-gpu ${sz[@]+"${sz[@]}"} --json "$R/gemm.json"
  local rc=$?
  kill $smi 2>/dev/null
  return $rc
}
ensure_device() {
  [ -x build/cancho-gpu ] || { echo "no build/cancho-gpu: the toolchain step failed"; return $SKIP; }
  [ -x build/cancho-gpu-device ] || CANCHO="$(cancho_cmd)" scripts/build-device.sh
}
s_shuffle_repro() {
  # FAIL here means a bug was SEEN on the hardware (shuffle_repro.py says so).
  ensure_device || return $?
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  python3 scripts/gcp/shuffle_repro.py --bin build/cancho-gpu-device --timeout "${REPRO_TIMEOUT:-60}" --json "$R/shuffle_repro.json"
}
s_fp4_perf() {
  # FAIL: a variant timed out, errored, or the old and new outputs disagreed.
  [ -x build/cancho-gpu ] || { echo "no build/cancho-gpu: the toolchain step failed"; return $SKIP; }
  pyenv || return 1
  "$VENV/bin/python" scripts/gcp/fp4_perf.py --cancho-gpu build/cancho-gpu --timeout "${REPRO_TIMEOUT:-60}" --json "$R/fp4_perf.json"
}
s_sandbox_linux() {
  [ -x build/cancho-gpu ] || { echo "no build/cancho-gpu: the toolchain step failed"; return $SKIP; }
  [ -x build/cancho-gpu-session ] || CANCHO="$(cancho_cmd)" scripts/build-session.sh || return 1
  echo "--- the failing check, verbose (PYTHONUNBUFFERED, every line kept)"
  PYTHONUNBUFFERED=1 python3 tests/weightspace.py build/cancho-gpu-session
  local rc=$?
  echo "--- tests/weightspace.py exit $rc"
  echo "--- the same network calls, bare and sandboxed, with the audit events seen"
  python3 scripts/gcp/audit_probe.py
  return $rc
}
s_job() {
  [ -x build/cancho-gpu ] || { echo "no build/cancho-gpu: the toolchain step failed"; return $SKIP; }
  CANCHO="$(cancho_cmd)"; export CANCHO
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

s_w1() {
  # docs/weight-space.md 13.2, 13.3, 13.5 on this card, in chunks: `behave --n K` for K = CHUNK, 2*CHUNK, ... N. Each chunk
  # leaves base and edited complete for problems 0..K-1 in ~/results/w1_out/*.jsonl, so a VM that dies between two chunks (or in
  # one) loses at most that chunk, and the driver (w1_cloud.sh) pulls the files while this runs. Finished problems are skipped
  # (run_behaviour), so a rerun over a seeded directory goes straight to the first unfinished batch. CHUNK is a multiple of BATCH:
  # the per-batch seed is seed + the first problem of the batch, so aligned chunks give the seeds of one uninterrupted run.
  # Settings: W1_N (200), W1_MAX_NEW (8192), W1_BATCH (8), W1_CHUNK (24), W1_SEED (0), W1_BASE_REV / W1_EDITED_REV (13.1's pins),
  # W1_STOP_FILE (a file whose existence ends the loop after the running chunk). W1_EDITED_SUB (a directory name under ~/models, uploaded by
  # w1_cloud.sh's EDITED_DIR): a variant checkpoint (docs/weight-space-w4.md 13) is the edited arm, not the published edit: it is neither
  # downloaded nor held to `diff --w1-expect` (it is not the published edit), and `behave` runs with --no-verify as always. The outcome is one word in ~/results/w1_status.
  local n="${W1_N:-200}" mx="${W1_MAX_NEW:-8192}" batch="${W1_BATCH:-8}" chunk="${W1_CHUNK:-24}" seed="${W1_SEED:-0}"
  local m="$HOME/models" out="$R/w1_out" w1py="$SRC/examples/weightspace/w1.py" py="$VENV/bin/python" hf="${W1_HF:-$VENV/bin/hf}"
  local base_rev="${W1_BASE_REV:-2fc06364715b967f1860aea9cf38778875588b17}" edited_rev="${W1_EDITED_REV:-d1671a178705bcab7644806d23c535517f35723b}"
  local k rc edited="$m/qwen35-08b-edited"
  [ -n "${W1_EDITED_SUB:-}" ] && edited="$m/$W1_EDITED_SUB"
  mkdir -p "$out"
  if [ "$((chunk % batch))" != 0 ]; then
    echo "W1_CHUNK=$chunk is not a multiple of W1_BATCH=$batch"; echo SETUP-FAILED > "$R/w1_status"; return 1
  fi
  if ! python3 -c 'import ensurepip' 2>/dev/null; then
    sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv
  fi
  pyenv || { echo "python environment failed"; echo SETUP-FAILED > "$R/w1_status"; return 1; }
  [ -x "$hf" ] || { echo "no hf command at $hf (huggingface_hub too old?)"; echo SETUP-FAILED > "$R/w1_status"; return 1; }
  # 13.2's commands: each --exclude pattern needs its own flag (the first run passed them after one and got LICENSE and README).
  { "$hf" download Qwen/Qwen3.5-0.8B --revision "$base_rev" --local-dir "$m/qwen35-08b-base" --exclude README.md --exclude LICENSE --exclude .gitattributes \
    && { [ -n "${W1_EDITED_SUB:-}" ] || "$hf" download RikiyaT/Qwen3.5-0.8B-weight-space-programming --revision "$edited_rev" --local-dir "$m/qwen35-08b-edited" --exclude "eval/*" --exclude README.md --exclude LICENSE --exclude .gitattributes; } \
    && "$hf" download openai/gsm8k main/test-00000-of-00001.parquet --repo-type dataset --local-dir "$m/gsm8k"; } \
    || { echo "download failed"; echo SETUP-FAILED > "$R/w1_status"; return 1; }
  # (a) the edit is the published one, or nothing runs; once per VM, so the chunks pass --no-verify.
  if [ -n "${W1_EDITED_SUB:-}" ]; then
    if [ -f "$edited/config.json" ] && ls "$edited"/*.safetensors >/dev/null 2>&1; then echo "variant checkpoint $edited (no diff --w1-expect: it is not the published edit)"
    else echo "variant checkpoint $edited is missing or incomplete (upload failed?)"; echo SETUP-FAILED > "$R/w1_status"; return 1; fi
  elif "$py" "$w1py" diff "$m/qwen35-08b-base" "$m/qwen35-08b-edited" --w1-expect --json "$R/w1_diff.json" > "$R/w1_diff.txt" 2>&1; then
    tail -3 "$R/w1_diff.txt"; echo "diff --w1-expect: all checks pass"
  else
    tail -20 "$R/w1_diff.txt"; echo "diff --w1-expect FAILED: the edited checkpoint is not the published edit; nothing is run"
    echo DIFF-FAILED > "$R/w1_status"; return 1
  fi
  echo RUNNING > "$R/w1_status"
  k=$chunk
  while :; do
    [ "$k" -le "$n" ] || k=$n
    if "$py" "$w1py" status --dir "$out" --n "$k" --max-new "$mx"; then
      echo "chunk n=$k: already complete, skipped"
    else
      echo "chunk n=$k: starting ($(date -u +%H:%M:%S))"
      "$py" "$w1py" behave --base "$m/qwen35-08b-base" --edited "$edited" --data "$m/gsm8k/main/test-00000-of-00001.parquet" \
        --out "$out" --n "$k" --max-new "$mx" --batch "$batch" --seed "$seed" --device cuda --gpu-free --no-verify
      rc=$?
      [ "$rc" = 0 ] || { echo "chunk n=$k: behave exited $rc"; echo "FAILED chunk n=$k" > "$R/w1_status"; return 1; }
      echo "chunk n=$k: done ($(date -u +%H:%M:%S))"
    fi
    [ "$k" -lt "$n" ] || break
    if [ -e "${W1_STOP_FILE:-$HOME/w1_stop}" ]; then
      echo "stop file present: ending after chunk n=$k"; echo "STOPPED n=$k" > "$R/w1_status"; return 0
    fi
    k=$((k + chunk))
  done
  echo DONE > "$R/w1_status"
}

# ---- run --------------------------------------------------------------------
[ "${REMOTE_CUDA_LIB:-0}" = 1 ] && return 0   # tests/gcp.sh sources the functions above

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
  { want toolchain || want gate || want device_cases || want gemm_bench || want shuffle_repro || want fp4_perf || want sandbox_linux || want baselines; } && run_step toolchain s_toolchain
  want gate && { run_step gate s_gate; run_step emulator emulator_hook; }
  want nvrtc && run_step nvrtc s_nvrtc
  want device_cases && run_step device_cases s_device_cases
  want gemm_bench && run_step gemm_bench s_gemm_bench
  want shuffle_repro && run_step shuffle_repro s_shuffle_repro
  want fp4_perf && run_step fp4_perf s_fp4_perf
  want baselines && run_step baselines s_baselines
  want w1 && run_step w1 s_w1
  # Last, and independent of the GPU: it must run even if shuffle_repro wedged it.
  want sandbox_linux && run_step sandbox_linux s_sandbox_linux
fi

if [ -f "$HOME/cancho-bin/rev" ]; then
  if [ "$(cat "$HOME/cancho-bin/rev")" != "$REV" ]; then st=NOT-PINNED-ignore-compiler-rev
  elif [ "$(cut -d: -f1 "$ID_FILE" 2>/dev/null)" = source ]; then st=PINNED-by-CANCHO_REV
  else st=PINNED; fi
  printf 'compiler\t%s\t%s (cancho.toml pin %s)\n' "$st" "$(cat "$ID_FILE" 2>/dev/null)" "$REV" | tee -a "$R/summary.tsv" > "$R/compiler.txt"
fi
echo; echo "=== summary"; column -t -s "$(printf '\t')" "$R/summary.tsv" 2>/dev/null || cat "$R/summary.tsv"
if grep -q "FAIL" "$R/summary.tsv"; then exit 1; fi
exit 0
