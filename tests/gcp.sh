#!/usr/bin/env bash
# The cost guard rails of scripts/gcp/ (docs/cloud.md), checked WITHOUT a cloud:
# a stub `gcloud` on PATH records its arguments, and nothing here can reach
# Google. What is asserted:
#
#   DRY_RUN=1   prints the VM create and runs no gcloud at all
#   the create  is Spot (--provisioning-model=SPOT), has --max-run-duration and
#               --instance-termination-action DELETE, the purpose label, a
#               location policy per region, EU first; SPOT=0 drops only Spot
#   cleanup     the VM is deleted on a normal exit, on a failing run, on
#               SIGTERM and on SIGHUP during the run (SIGINT is the same
#               handler; a non-interactive shell cannot be sent one in the
#               background, so it is not driven here)
#   preemption  a failed run on a VM Google preempted exits 75
#   results     are copied home when the run fails
#   project     none given and none configured is refused (exit 2); one taken
#               from gcloud's default is printed and waits unless YES=1
#   sweep       lists, exits 1 with strays, deletes only with DELETE=1
#   w1_cloud    (W1 in restartable steps, docs/cloud.md) the plan in a dry run; the loop restarts after a preemption, seeds
#               the next VM with what was pulled, pulls while a VM runs, stops at TOTAL_MAX / MAX_VMS / no progress /
#               a failed diff / no capacity, leaves no VM, computes the verdict locally; the remote w1 step's commands
##   scripts     parse (bash -n; shellcheck when installed)
#
# The remote steps, the NVRTC check, the device cases and the benchmark need a
# GPU and are NOT run here; what this cannot show is listed in docs/cloud.md.
#
#     tests/gcp.sh
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
G=$ROOT/scripts/gcp
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
fail=0
ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; fail=1; }
has() { # file pattern what
  if grep -q -e "$2" "$1"; then ok "$3"; else bad "$3 (no \`$2\` in $1)"; fi
}
hasnt() {
  if grep -q -e "$2" "$1"; then bad "$3 (found \`$2\` in $1)"; else ok "$3"; fi
}

mkdir -p "$W/bin"
cat > "$W/bin/gcloud" <<'STUB'
#!/bin/sh
# A stub gcloud: records every call, pretends a Spot VM can be made.
echo "$*" >> "$STUB_LOG"
case "$*" in
  *"config get-value project"*) echo "${STUB_CONFIG_PROJECT:-}"; exit 0 ;;
  *"images list"*) echo "common-cu12-ubuntu-2204"; exit 0 ;;
  *"describe-from-family"*) exit 1 ;;
  *"instances bulk create"*) touch "$STUB_STATE/vm"; exit 0 ;;
  *"instances list"*"labels.purpose"*) [ -f "$STUB_STATE/stray" ] && echo "cancho-gpu-l4-old	europe-west4-b	RUNNING"; exit 0 ;;
  *"instances list"*) [ -f "$STUB_STATE/vm" ] && echo "europe-west4-b"; exit 0 ;;
  *"instances delete"*) rm -f "$STUB_STATE/vm" "$STUB_STATE/stray"; exit 0 ;;
  *"compute ssh"*"--command true"*) exit 0 ;;
  *"compute ssh"*) echo "remote output"; touch "$STUB_STATE/ssh_running"
                  [ -n "${STUB_SSH_SLEEP:-}" ] && exec sleep "$STUB_SSH_SLEEP"
                  exit "${STUB_SSH_STATUS:-0}" ;;
  *"compute scp"*"~/results"*) mkdir -p "$STUB_STATE/home"; touch "$STUB_STATE/results_copied"; exit 0 ;;
  *"compute scp"*) exit 0 ;;
  *"operations list"*) [ "${STUB_PREEMPT:-0}" = 1 ] && echo "operation-1"; exit 0 ;;
esac
exit 0
STUB
chmod +x "$W/bin/gcloud"
# A gcloud that must never be called: DRY_RUN's promise.
mkdir -p "$W/never"
printf '#!/bin/sh\necho "CALLED: $*" >> "%s"\nexit 99\n' "$W/never.log" > "$W/never/gcloud"
chmod +x "$W/never/gcloud"

new_case() { # name -> sets S (state dir), L (log)
  S=$W/$1.state; mkdir -p "$S"; L=$W/$1.gcloud.log; : > "$L"
  export STUB_LOG=$L STUB_STATE=$S
  unset STUB_SSH_SLEEP STUB_SSH_STATUS STUB_PREEMPT STUB_CONFIG_PROJECT
}

# --- scripts parse ----------------------------------------------------------
for f in "$G"/*.sh "$ROOT/tests/gcp.sh"; do
  if bash -n "$f"; then ok "bash -n $(basename "$f")"; else bad "bash -n $f"; fi
done
for f in "$G"/*.py; do
  if python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f"; then ok "python parses $(basename "$f")"; else bad "python parse $f"; fi
done
if command -v shellcheck >/dev/null; then
  if shellcheck -x -S warning "$G"/*.sh; then ok "shellcheck"; else bad "shellcheck"; fi
else
  echo "skip shellcheck: not installed"
fi

# --- DRY_RUN ----------------------------------------------------------------
new_case dry
PATH=$W/never:$PATH DRY_RUN=1 GCP_PROJECT=fake-project OUT_DIR=$W/dry.out \
  bash "$G/cuda_test.sh" > "$W/dry.txt" 2>&1; rc=$?
[ "$rc" = 0 ] && ok "dry run exits 0" || bad "dry run exits $rc"
[ -e "$W/never.log" ] && bad "dry run called gcloud: $(cat "$W/never.log")" || ok "dry run called no gcloud"
has "$W/dry.txt" "DRY_RUN> gcloud --project fake-project --quiet compute instances bulk create" "dry run prints the create"
has "$W/dry.txt" "--provisioning-model=SPOT" "dry run: Spot by default"
has "$W/dry.txt" "--max-run-duration 90m" "dry run: --max-run-duration 90m (MAX_RUN default)"
has "$W/dry.txt" "--instance-termination-action DELETE" "dry run: termination action DELETE"
has "$W/dry.txt" "--labels purpose=cancho-gpu-test" "dry run: purpose label"
has "$W/dry.txt" "--region europe-west4 " "dry run: EU region first"
has "$W/dry.txt" "--machine-type g2-standard-8" "dry run: L4 machine"
has "$W/dry.txt" "instances delete cancho-gpu-l4-" "dry run: prints the cleanup delete"
for gpu in a100 h100; do
  PATH=$W/never:$PATH DRY_RUN=1 GPU=$gpu GCP_PROJECT=fake-project OUT_DIR=$W/dry.out \
    bash "$G/cuda_test.sh" > "$W/dry-$gpu.txt" 2>&1
  has "$W/dry-$gpu.txt" "Ada (sm_89, the L4) ONLY" "GPU=$gpu: the loud Ada-only note"
done
PATH=$W/never:$PATH DRY_RUN=1 SPOT=0 MAX_RUN=1h GCP_PROJECT=fake-project OUT_DIR=$W/dry.out \
  bash "$G/cuda_test.sh" > "$W/dry-od.txt" 2>&1
hasnt "$W/dry-od.txt" "provisioning-model" "SPOT=0: on demand"
has "$W/dry-od.txt" "--max-run-duration 1h" "SPOT=0 still caps the run (MAX_RUN=1h)"
has "$W/dry-od.txt" "--instance-termination-action DELETE" "SPOT=0 still deletes on termination"
PATH=$W/never:$PATH DRY_RUN=1 JOB='echo hi' BASELINES=1 GCP_PROJECT=fake-project OUT_DIR=$W/dry.out \
  bash "$G/cuda_test.sh" > "$W/dry-job.txt" 2>&1
has "$W/dry-job.txt" "JOB='1'" "JOB mode reaches the remote command"
has "$W/dry-job.txt" "baselines" "BASELINES=1 adds the step"
PATH=$W/never:$PATH DRY_RUN=1 bash "$G/sweep.sh" > "$W/dry-sweep.txt" 2>&1
has "$W/dry-sweep.txt" "DRY_RUN> gcloud --project DRY-RUN-PROJECT --quiet compute instances list" "sweep dry run"
PATH=$W/never:$PATH DRY_RUN=1 GCP_PROJECT=fake-project OUT_DIR=$W/dry.out \
  bash "$G/build_image.sh" > "$W/dry-image.txt" 2>&1
has "$W/dry-image.txt" "--provisioning-model=SPOT --instance-termination-action DELETE --max-run-duration 2h" "image builder is Spot, capped, deleted"
[ -e "$W/never.log" ] && bad "a dry run called gcloud: $(cat "$W/never.log")" || ok "no dry run called gcloud"

# --- the project ------------------------------------------------------------
new_case noproj
PATH=$W/bin:$PATH OUT_DIR=$W/np.out bash "$G/cuda_test.sh" > "$W/np.txt" 2>&1; rc=$?
[ "$rc" = 2 ] && ok "no project: refused, exit 2" || bad "no project: exit $rc"
[ -s "$L" ] && ! grep -q "bulk create" "$L" && ok "no project: nothing created" || bad "no project: $(cat "$L")"
new_case cfgproj
STUB_CONFIG_PROJECT=cfg-project CONFIRM_SECS=1 PATH=$W/bin:$PATH OUT_DIR=$W/cp.out \
  bash "$G/cuda_test.sh" > "$W/cp.txt" 2>&1
has "$W/cp.txt" "billing project: cfg-project (from gcloud config get-value project)" "the configured project is printed"
has "$W/cp.txt" "continuing in 1s -- Ctrl-C to abort" "the configured project waits"
new_case yes
STUB_CONFIG_PROJECT=cfg-project YES=1 PATH=$W/bin:$PATH OUT_DIR=$W/y.out bash "$G/cuda_test.sh" > "$W/y.txt" 2>&1
hasnt "$W/y.txt" "continuing in" "YES=1 skips the wait"

# --- a real (stubbed) run ---------------------------------------------------
run_driver() { # name, extra env words...; sets rc
  local name=$1; shift
  new_case "$name"
  env "$@" PATH="$W/bin:$PATH" GCP_PROJECT=fake-project OUT_DIR="$W/$name.out" \
    bash "$G/cuda_test.sh" > "$W/$name.txt" 2>&1
  rc=$?
}
run_driver ok STUB_SSH_STATUS=0
[ "$rc" = 0 ] && ok "a passing run exits 0" || bad "a passing run exits $rc"
has "$L" "instances bulk create --region europe-west4 --count 1 --predefined-names cancho-gpu-l4-" "create asks the first EU region for one VM"
has "$L" "--provisioning-model=SPOT" "create is Spot"
has "$L" "--max-run-duration 90m --instance-termination-action DELETE" "create has --max-run-duration and DELETE"
has "$L" "--labels purpose=cancho-gpu-test" "create is labelled"
has "$L" "--location-policy europe-west4-a=allow,europe-west4-b=allow,europe-west4-c=allow" "create gives the region's zones"
has "$L" "instances delete cancho-gpu-l4-" "the VM is deleted on a normal exit"
[ ! -f "$S/vm" ] && ok "no VM left in the stub project" || bad "VM left behind"
run_driver spot0 SPOT=0 STUB_SSH_STATUS=0
hasnt "$L" "provisioning-model" "SPOT=0 asks for on-demand"
has "$L" "--instance-termination-action DELETE" "on-demand still DELETEs"

run_driver fail STUB_SSH_STATUS=3
[ "$rc" = 3 ] && ok "a failing run exits with the remote status" || bad "failing run exits $rc"
has "$L" "instances delete" "the VM is deleted after a failing run"
[ -f "$S/results_copied" ] && ok "results are copied home even when the run fails" || bad "results not copied after a failure"

run_driver pre STUB_SSH_STATUS=255 STUB_PREEMPT=1
[ "$rc" = 75 ] && ok "preemption exits 75" || bad "preemption exits $rc"
has "$L" "instances delete" "the VM is deleted after a preemption"

run_driver keep KEEP=1 STUB_SSH_STATUS=0
hasnt "$L" "instances delete" "KEEP=1 does not delete"
has "$W/keep.txt" "KEEP=1: cancho-gpu-l4-" "KEEP=1 prints how to delete it"

# Signals: the remote command hangs (the stub sleeps); signal the script.
for sig in TERM HUP; do
  new_case "sig$sig"
  STUB_SSH_SLEEP=60 PATH=$W/bin:$PATH GCP_PROJECT=fake-project OUT_DIR=$W/sig.out \
    bash "$G/cuda_test.sh" > "$W/sig$sig.txt" 2>&1 &
  pid=$!
  for _ in $(seq 1 100); do [ -f "$S/ssh_running" ] && break; sleep 0.1; done
  if [ ! -f "$S/ssh_running" ]; then bad "$sig: the run never reached the remote command"; kill "$pid" 2>/dev/null; continue; fi
  kill -s "$sig" "$pid"
  wait "$pid"; rc=$?
  want=143; [ "$sig" = HUP ] && want=129
  [ "$rc" = "$want" ] && ok "$sig: exits $want" || bad "$sig: exit $rc, wanted $want"
  has "$L" "instances delete cancho-gpu-l4-" "$sig: the VM is deleted"
  [ ! -f "$S/vm" ] && ok "$sig: no VM left" || bad "$sig: VM left behind"
  pkill -f "sleep 60" 2>/dev/null || true
done

# --- the compiler install (remote_cuda.sh sourced as a library) -------------
# Stub curl serves a fake release from $W/rel; a stub cargo and git stand in for
# the source build. Nothing leaves the machine.
PIN=$(sed -n 's/^cancho = "\([0-9a-f]*\)".*/\1/p' "$ROOT/cancho.toml")
REL=5a5dd1f21250b96b3f735a45b781489374219466
mkdir -p "$W/cb" "$W/rel"
cat > "$W/cb/curl" <<'STUB'
#!/bin/sh
out=""; url=""; wfmt=""
while [ $# -gt 0 ]; do
  case "$1" in -o|--output) out=$2; shift ;; -w) wfmt=$2; shift ;; -*) ;; *) url=$1 ;; esac
  shift
done
echo "curl $url" >> "$STUB_CURL_LOG"
if [ -n "$wfmt" ]; then echo "https://github.com/x/cancho/releases/tag/cancho-$STUB_REL"; exit 0; fi
f="$STUB_REL_DIR/$(basename "$url")"
[ -f "$f" ] || exit 22
cp "$f" "$out"
STUB
cat > "$W/cb/cargo" <<'STUB'
#!/bin/sh
echo "cargo $* CANCHO_REV=${CANCHO_REV:-unset}" >> "$STUB_CURL_LOG"
mkdir -p target/release && printf '#!/bin/sh\necho built-from-source\n' > target/release/cancho && chmod +x target/release/cancho
STUB
cat > "$W/cb/git" <<'STUB'
#!/bin/sh
echo "git $*" >> "$STUB_CURL_LOG"
[ "$1" = clone ] && eval "mkdir -p \"\${$#}\""
exit 0
STUB
chmod +x "$W/cb/curl" "$W/cb/cargo" "$W/cb/git"
name=cancho-rust-$REL-linux-x86_64
mkdir -p "$W/pk/$name/bin"
printf '#!/bin/sh\necho "cancho 0.0.0 (rev 5a5dd1f2…, host x86_64)"\n' > "$W/pk/$name/bin/cancho"; chmod +x "$W/pk/$name/bin/cancho"
tar -czf "$W/rel/$name.tar.gz" -C "$W/pk" "$name"
sum=$( (sha256sum "$W/rel/$name.tar.gz" 2>/dev/null || shasum -a 256 "$W/rel/$name.tar.gz") | cut -d' ' -f1)
printf '%s  %s.tar.gz\n' "$sum" "$name" > "$W/rel/$name.tar.gz.sha256"
install_case() { # name, extra env words; runs install and the summary logic in a fresh HOME
  local c=$1; shift
  export STUB_CURL_LOG=$W/$c.calls STUB_REL=$REL STUB_REL_DIR=$W/rel; : > "$STUB_CURL_LOG"
  local h=$W/$c.home; mkdir -p "$h/cancho-gpu"; cp "$ROOT/cancho.toml" "$h/cancho-gpu/"
  ( cd "$h/cancho-gpu" && env HOME="$h" PATH="$W/cb:$PATH" REMOTE_CUDA_LIB=1 STEPS=toolchain "$@" bash -c '
      . '"$G"'/remote_cuda.sh; build_compiler; echo "rc=$? id=$(cat $HOME/cancho-bin/id) rev=$(cat $HOME/cancho-bin/rev)"; echo "cmd=$(cancho_cmd)"' ) > "$W/$c.txt" 2>&1
}
install_case rel
has "$W/rel.txt" "rc=0 id=release:cancho-$REL:rust rev=$REL" "release path chosen: latest resolved, rust asset installed"
has "$W/rel.txt" "sha256 ok" "the sha256 was checked"
hasnt "$W/rel.calls" "cargo\|git clone\|rustup" "no Rust and no source build when a release is used"
has "$W/rel.calls" "cancho-rust-$REL-linux-x86_64.tar.gz.sha256" "the .sha256 asset is fetched"
if [ "$PIN" != "$REL" ]; then
  has "$W/rel.txt" "cmd=.*cancho-ignore-rev" "rev differs from the pin: the --ignore-compiler-rev wrapper is used"
  has "$W/rel.home/cancho-bin/cancho-ignore-rev" '--ignore-compiler-rev' "the wrapper adds the flag to project-mode build/test"
fi
# The same release when it IS the pin: no flag.
mkdir -p "$W/same.home/cancho-gpu"
sed "s/^cancho = \"[0-9a-f]*\"/cancho = \"$REL\"/" "$ROOT/cancho.toml" > "$W/same.home/cancho-gpu/cancho.toml"
( cd "$W/same.home/cancho-gpu" && STUB_CURL_LOG=$W/same.calls STUB_REL=$REL STUB_REL_DIR=$W/rel HOME=$W/same.home PATH=$W/cb:$PATH \
  REMOTE_CUDA_LIB=1 bash -c '. '"$G"'/remote_cuda.sh; build_compiler >/dev/null; echo "cmd=$(cancho_cmd)"' ) > "$W/same.txt" 2>&1
hasnt "$W/same.txt" "ignore-rev" "rev equal to the pin: no --ignore-compiler-rev"
# A published hash that does not match: refused, the pinned commit is built.
cp "$W/rel/$name.tar.gz.sha256" "$W/rel/good.sha"
printf '%064d  %s.tar.gz\n' 0 "$name" > "$W/rel/$name.tar.gz.sha256"
install_case bad
has "$W/bad.txt" "sha256 MISMATCH" "a sha256 mismatch is reported"
has "$W/bad.txt" "falling back to building the pinned commit $PIN" "a mismatch falls back to the source build"
has "$W/bad.calls" "cargo build --release -p cancho CANCHO_REV=$PIN" "the source build stamps the pinned rev (CANCHO_REV)"
has "$W/bad.txt" "id=source:$PIN rev=$PIN" "the fallback binary is the pinned one"
hasnt "$W/bad.txt" "ignore-rev" "a pinned source build needs no --ignore-compiler-rev"
cp "$W/rel/good.sha" "$W/rel/$name.tar.gz.sha256"
# A release binary that does not say which commit it is would be refused by the project: not used.
cp "$W/pk/$name/bin/cancho" "$W/anon.bak"
printf '#!/bin/sh\necho "cancho 0.0.0"\n' > "$W/pk/$name/bin/cancho"
tar -czf "$W/rel/$name.tar.gz" -C "$W/pk" "$name"
printf '%s  %s.tar.gz\n' "$( (sha256sum "$W/rel/$name.tar.gz" 2>/dev/null || shasum -a 256 "$W/rel/$name.tar.gz") | cut -d' ' -f1)" "$name" > "$W/rel/$name.tar.gz.sha256"
install_case anon
has "$W/anon.txt" "does not report rev" "a release binary that does not report its rev is not used"
has "$W/anon.txt" "id=source:$PIN" "and the pinned source build replaces it"
cp "$W/anon.bak" "$W/pk/$name/bin/cancho"
# No release at all (download fails): the same fallback.
rm -f "$W/rel/$name.tar.gz"
install_case gone
has "$W/gone.txt" "falling back to building the pinned commit" "a failed download falls back to the source build"
# CANCHO_RELEASE=source never touches the release.
install_case src CANCHO_RELEASE=source
hasnt "$W/src.calls" "releases" "CANCHO_RELEASE=source asks for no release"
has "$W/src.txt" "id=source:$PIN" "CANCHO_RELEASE=source builds the pin"

# --- a plain `clang` on PATH (cancho runs it by name) ------------------------
mkdir -p "$W/cl/bin" "$W/cl/local" "$W/cl/usr" "$W/cl/home/cancho-gpu"
printf '#!/bin/sh\necho "Ubuntu clang version 18.1.3"\n' > "$W/cl/bin/clang-18"
printf '#!/bin/sh\nexec "$@"\n' > "$W/cl/bin/sudo"
chmod +x "$W/cl/bin/clang-18" "$W/cl/bin/sudo"
cp "$ROOT/cancho.toml" "$W/cl/home/cancho-gpu/"
# A PATH with the tools the script needs and no clang (this Mac has an Xcode shim in /usr/bin).
mkdir -p "$W/cl/tools"
for t in sed ln head mkdir cat bash sh dirname env cut awk tr grep find seq sleep uname dpkg; do
  p=$(command -v "$t" 2>/dev/null) && ln -sf "$p" "$W/cl/tools/$t"
done
( cd "$W/cl/home/cancho-gpu" && HOME=$W/cl/home PATH="$W/cl/local:$W/cl/bin:$W/cl/tools" REMOTE_CUDA_LIB=1 \
  PLAIN_CLANG_DIR=$W/cl/local PLAIN_CLANG_ALT=$W/cl/usr bash -c '
    . '"$G"'/remote_cuda.sh; mkdir -p $HOME/results
    command -v clang >/dev/null && echo "BEFORE: clang exists"
    pick_clang && plain_clang; echo "rc=$? CLANG=$CLANG"; command -v clang; clang --version' ) > "$W/cl.txt" 2>&1
hasnt "$W/cl.txt" "BEFORE: clang exists" "setup: no plain clang to begin with"
has "$W/cl.txt" "rc=0 CLANG=clang-18" "clang-18 is chosen"
has "$W/cl.txt" "$W/cl/local/clang" "a plain clang now resolves, first on PATH"
has "$W/cl.txt" "clang version 18" "and it is the chosen version"
has "$G/remote_cuda.sh" "^  plain_clang || return 1" "apt_tools calls plain_clang"

# --- the driver survives its own file changing mid-run --------------------
new_case edit
git clone -q "$ROOT" "$W/tree" 2>/dev/null || mkdir -p "$W/tree"
mkdir -p "$W/tree/scripts/gcp"; cp "$G/cuda_test.sh" "$W/tree/scripts/gcp/cuda_test.sh"
DC=$W/tree/scripts/gcp/cuda_test.sh
STUB_SSH_SLEEP=3 PATH=$W/bin:$PATH GCP_PROJECT=fake-project OUT_DIR=$W/edit.out \
  bash "$DC" > "$W/edit.txt" 2>&1 &
pid=$!
for _ in $(seq 1 100); do [ -f "$S/ssh_running" ] && break; sleep 0.1; done
# Shift every byte offset, and append garbage that is not valid shell, in place.
{ echo '# edited while running'; cat "$G/cuda_test.sh"; printf 'echo (garbage\n)(\n'; } > "$W/driver_new.sh"
cat "$W/driver_new.sh" > "$DC"
wait "$pid"; rc=$?
[ "$rc" = 0 ] && ok "editing the driver mid-run does not break the run (exit 0)" || bad "mid-run edit: exit $rc: $(tail -3 "$W/edit.txt")"
hasnt "$W/edit.txt" "syntax error" "no syntax error from the edited file"
[ -f "$S/results_copied" ] && ok "results copied home after a mid-run edit" || bad "results not copied after a mid-run edit"
has "$L" "instances delete" "VM deleted after a mid-run edit"

# --- the new steps -----------------------------------------------------------
has "$W/dry.txt" "STEPS='machine toolchain gate nvrtc device_cases gemm_bench shuffle_repro fp4_perf sandbox_linux'" "default STEPS include shuffle_repro, fp4_perf and sandbox_linux"
# gate on the VM: tests/gcp.sh is skipped VISIBLY; a known-open unit does not fail; a real failure does.
mkdir -p "$W/gt/home/cancho-gpu/build" "$W/gt/home/results" "$W/gt/home/cancho-bin"
cp "$ROOT/cancho.toml" "$W/gt/home/cancho-gpu/"
printf '#!/bin/sh\n' > "$W/gt/home/cancho-gpu/build/cancho-gpu"; chmod +x "$W/gt/home/cancho-gpu/build/cancho-gpu"
printf '#!/bin/sh\nexit 0\n' > "$W/gt/home/cancho-bin/cancho"; chmod +x "$W/gt/home/cancho-bin/cancho"
echo "$PIN" > "$W/gt/home/cancho-bin/rev"
cat > "$W/gt/gate.sh" <<'GATE'
set -eu
ROOT=x
cd "$ROOT"
true
tests/sampler.sh
tests/gcp.sh
false
if true; then
  true
fi
echo "gate: ok"
GATE
for mode in known-only real-fail; do
  gf=$W/gt/gate-$mode.sh
  if [ $mode = known-only ]; then grep -v '^false$' "$W/gt/gate.sh" > "$gf"; else cp "$W/gt/gate.sh" "$gf"; fi
  : > "$W/gt/home/results/summary.tsv"
  ( cd "$W/gt/home/cancho-gpu" && HOME=$W/gt/home PATH="$W/cb:$PATH" REMOTE_CUDA_LIB=1 GATE_FILE=$gf bash -c '
      . '"$G"'/remote_cuda.sh
      sh() { if [ "$2" = "set -eu; tests/sampler.sh" ]; then return 3; fi; if [ "$2" = "set -eu; false" ]; then return 1; fi; return 0; }
      s_gate; echo "gate rc=$?"' ) > "$W/gate-$mode.txt" 2>&1
done
has "$W/gate-known-only.txt" "SKIPPED on VM: tests/gcp.sh" "gate: tests/gcp.sh is skipped visibly on the VM"
has "$W/gt/home/results/summary.tsv" "gate:tests/gcp.sh.SKIPPED-ON-VM" "gate: the skip is a row in summary.tsv"
has "$W/gate-known-only.txt" "KNOWN-OPEN" "gate: the known-open sampler unit is reported as such"
has "$W/gate-known-only.txt" "gate rc=0" "gate: known-open and skipped units do not fail the step"
has "$W/gate-real-fail.txt" "FAIL: exit 1" "gate: a real failure is reported"
has "$W/gate-real-fail.txt" "gate rc=1" "gate: and fails the step"
# shuffle_repro.py: PASS, MISMATCH, TIMEOUT (killed, reported), ERROR, REFUSED, then a wedged GPU stops the step.
mkdir -p "$W/sr/bin"
cat > "$W/sr/dev" <<'DEV'
#!/bin/sh
case "$*" in
  *pass*) echo "worst difference 1.0e-7 of peak 9.0"; exit 0 ;;
  *mismatch*) echo "error[device-mismatch]: output y differs from the reference by 0.78 of peak 9.09" >&2; exit 1 ;;
  *hang*) exec sleep 30 ;;
  *error*) echo "error[device]: launching the kernel: an illegal instruction was encountered" >&2; exit 1 ;;
  *refused*) echo "error[offset]: not a multiple" >&2; exit 1 ;;
esac
DEV
chmod +x "$W/sr/dev"
printf 'tests/chx/pass.chx n=1\ntests/chx/mismatch.chx n=1   # note\ntests/chx/hang.chx n=1\ntests/chx/error.chx n=1\ntests/chx/refused.chx n=1\n' > "$W/sr/cases.txt"
printf '#!/bin/sh\necho "0 MiB, 0 %%"\n' > "$W/sr/bin/nvidia-smi"; chmod +x "$W/sr/bin/nvidia-smi"
PATH=$W/sr/bin:$PATH python3 "$G/shuffle_repro.py" --bin "$W/sr/dev" --cases "$W/sr/cases.txt" --timeout 6 --json "$W/sr/o.json" > "$W/sr.txt" 2>&1; rc=$?
[ "$rc" = 1 ] && ok "shuffle_repro exits 1 when a bug is seen" || bad "shuffle_repro exit $rc"
for st in "PASS      tests/chx/pass" "MISMATCH  tests/chx/mismatch" "TIMEOUT   tests/chx/hang" "ERROR     tests/chx/error" "REFUSED   tests/chx/refused"; do
  has "$W/sr.txt" "^$st" "shuffle_repro: $st"
done
has "$W/sr.txt" "BUG SEEN ON HARDWARE in 3 case" "shuffle_repro: counts MISMATCH, TIMEOUT and ERROR as the bug"
# a hang is reported in about the timeout, not left stuck
printf 'tests/chx/hang.chx n=1\ntests/chx/pass.chx n=1\n' > "$W/sr/cases2.txt"
printf '#!/bin/sh\necho "NVRM nothing"; exit 9\n' > "$W/sr/bin/nvidia-smi"
PATH=$W/sr/bin:$PATH python3 "$G/shuffle_repro.py" --bin "$W/sr/dev" --cases "$W/sr/cases2.txt" --timeout 6 > "$W/sr2.txt" 2>&1
has "$W/sr2.txt" "^WEDGED" "shuffle_repro: a GPU that stops answering after a hang is WEDGED"
has "$W/sr2.txt" "stopping the step here" "shuffle_repro: and the step stops cleanly"
hasnt "$W/sr2.txt" "tests/chx/pass.chx n=1 .*PASS" "shuffle_repro: nothing runs after a wedge"
# the case list parses and names files that exist
bad_files=$(python3 - "$G/shuffle_repro.txt" "$ROOT" <<'PY'
import os, sys
for line in open(sys.argv[1]):
    line = line.split("#")[0].strip()
    if line and not os.path.exists(os.path.join(sys.argv[2], line.split()[0])):
        print(line)
PY
)
[ -z "$bad_files" ] && ok "shuffle_repro.txt names only existing .chx files" || bad "shuffle_repro.txt: missing files: $bad_files"
# the audit probe runs and prints both parts (it is Python only, but it
# imports numpy, as the sandboxed programs it mirrors do)
if python3 -c 'import numpy' 2>/dev/null; then
  python3 "$G/audit_probe.py" > "$W/probe.txt" 2>&1; rc=$?
  [ "$rc" = 0 ] && ok "audit_probe exits 0" || bad "audit_probe exit $rc"
  has "$W/probe.txt" "=== 1. BARE" "audit_probe: bare part"
  has "$W/probe.txt" "=== 2. SANDBOXED" "audit_probe: sandboxed part"
  has "$W/probe.txt" "event socket.connect" "audit_probe: lists the socket.connect audit event"
else
  echo "skip audit_probe: numpy is not installed"
fi

# --- every step in the default STEPS, and every s_* function, is dispatched --
# (run 4 shipped two steps that were defined and never called)
default_steps=$(sed -n 's/^STEPS="\${STEPS:-\(.*\)}"$/\1/p' "$G/cuda_test.sh" | head -1)
remote_steps=$(sed -n 's/^STEPS="\${STEPS:-\(.*\)}"$/\1/p' "$G/remote_cuda.sh" | head -1)
[ "$default_steps" = "$remote_steps" ] && ok "the driver and the remote script have the same default STEPS" || bad "default STEPS differ: [$default_steps] vs [$remote_steps]"
for st in $default_steps baselines; do
  if grep -q "^  want $st && .*run_step $st s_$st" "$G/remote_cuda.sh" || grep -q "^  want $st && { run_step $st s_$st" "$G/remote_cuda.sh" || grep -q "run_step $st s_$st" "$G/remote_cuda.sh"; then
    ok "step $st is dispatched by main"
  else bad "step $st is in STEPS but main never runs it"; fi
done
for fn in $(sed -n 's/^\(s_[a-z_]*\)() {.*/\1/p' "$G/remote_cuda.sh"); do
  grep -q "run_step [a-z_]* $fn" "$G/remote_cuda.sh" && ok "$fn is called by a run_step" || bad "$fn is defined and never called"
done
a=$(grep -n "run_step shuffle_repro" "$G/remote_cuda.sh" | tail -1 | cut -d: -f1)
b=$(grep -n "run_step sandbox_linux" "$G/remote_cuda.sh" | tail -1 | cut -d: -f1)
[ -n "$a" ] && [ -n "$b" ] && [ "$b" -gt "$a" ] && ok "sandbox_linux is dispatched after shuffle_repro (it needs no GPU)" || bad "sandbox_linux must be dispatched after shuffle_repro"
# gemm_bench's occupancy: with no API it must say so, or give a labelled estimate, never a bare None.
python3 - "$G" > "$W/occ.txt" 2>&1 <<'PY'
import sys, types
cp = types.ModuleType("cupy"); cp.cuda = types.SimpleNamespace(driver=types.SimpleNamespace())
sys.modules["cupy"] = cp; sys.modules["torch"] = types.ModuleType("torch")
sys.path.insert(0, sys.argv[1])
import gemm_bench as g
k = types.SimpleNamespace(ptr=0, num_regs=128, shared_size_bytes=36864, local_size_bytes=0)
n, how, errs = g.occupancy(k, 256, 58)
print("ESTIMATE", n, how, errs)
info = g.kernel_info(k, "", 256, (64, 64), 58)
print("INFO", info.get("blocks_per_sm"), info.get("occupancy_source"), info.get("waves"), info.get("occupancy_warps_pct"))
PY
has "$W/occ.txt" "ESTIMATE 2 ESTIMATE from regs/smem" "occupancy falls back to a labelled estimate (128 regs, 36,864 B: 2 blocks/SM)"
has "$W/occ.txt" "cupy binding" "and keeps the error of the API that failed"
has "$W/occ.txt" "INFO 2 ESTIMATE" "kernel_info carries the source"

# --- fp4_perf: the old lowering is a text edit of the new, and it must be EXACTLY the old golden -
# (stdlib only; the GPU part needs cupy and is not run here)
python3 - "$G" "$ROOT" > "$W/fp4.txt" 2>&1 <<'PY'
import subprocess, sys
sys.path.insert(0, sys.argv[1])
import fp4_perf as f
root = sys.argv[2]
g = "golden/batch_fp4_b=3_n=40_k=96/batch_fp4_3_40_96.cu"
# a kernel that still has table lookups: the golden as it was before the split-K step took a partial last round
# (K = 96 is not a multiple of 512, so the emitter now writes fp4_pair here), kept as a fixture
new = open(root + "/tests/fp4_lookup_fixture.cu").read()
old, n = f.old_variant(new)
print("LOOKUPS", n, "SHFL", old.count("__shfl_sync(0xffffffffu, fp4_lane, "), "LANEDEF", old.count(f.LANE_DEF), "LEFT", old.count("FP4_V[(") )
# the declaration of the table is not a lookup
print("TABLE", "__constant__ float FP4_V[16]" in old)
pre = subprocess.run(["git", "-C", root, "show", "46670b5^:" + g], capture_output=True, text=True)
print("EXACT" if pre.returncode == 0 and pre.stdout == old else ("NOGIT" if pre.returncode else "DIFFERENT"))
same, z = f.old_variant("no lookup here\n")
print("NOOP", z, same == "no lookup here\n")
PY
has "$W/fp4.txt" "^LOOKUPS 5 SHFL 5 LANEDEF 1 LEFT 0" "fp4_perf: old_variant puts back the shuffles and the fp4_lane line"
has "$W/fp4.txt" "^TABLE True" "fp4_perf: the table's declaration is not rewritten"
has "$W/fp4.txt" "^NOOP 0 True" "fp4_perf: a kernel without a lookup is unchanged"
if grep -q "^NOGIT" "$W/fp4.txt"; then echo "skip fp4_perf exactness: commit 46670b5 is not in this checkout"
else has "$W/fp4.txt" "^EXACT" "fp4_perf: the old variant is byte-for-byte the golden before PR #104"; fi
cat > "$W/bin/fp4-stub" <<'STUB'
#!/bin/sh
# emit <file.chx> <dir> [consts]: copy the golden .cu of the same algo, as the compiler would write it
b=$(basename "$2" .chx)
d=$(ls -d "$ROOT"/golden/"$b"_[nm]=* | head -1)
cp "$d"/*.cu "$3/"
echo "x.cu: 256 threads, grid 4x1, 0 B of tiles"
STUB
chmod +x "$W/bin/fp4-stub"
ROOT=$ROOT python3 "$G/fp4_perf.py" --dry-run --cancho-gpu "$W/bin/fp4-stub" > "$W/fp4dry.txt" 2>&1; rc=$?
[ "$rc" = 0 ] && ok "fp4_perf --dry-run exits 0 against a stub compiler" || bad "fp4_perf --dry-run exit $rc: $(tail -3 "$W/fp4dry.txt")"
has "$W/fp4dry.txt" "matvec_fp4_swiglu n=17408 k=5152" "fp4_perf: the swiglu case is listed"
hasnt "$W/fp4dry.txt" "ERROR" "fp4_perf: every case's parameters match its declarations"
# the real run needs cupy: without it, it says so and fails rather than pretending
if ! python3 -c 'import cupy' 2>/dev/null; then
  python3 "$G/fp4_perf.py" --cancho-gpu "$W/bin/fp4-stub" > "$W/fp4nocupy.txt" 2>&1; rc=$?
  { [ "$rc" = 1 ] && grep -q "needs numpy and cupy" "$W/fp4nocupy.txt"; } && ok "fp4_perf without cupy says so and exits 1" || bad "fp4_perf without cupy: exit $rc"
fi

# --- sweep ------------------------------------------------------------------
new_case sweep0
PATH=$W/bin:$PATH GCP_PROJECT=fake-project bash "$G/sweep.sh" > "$W/sweep0.txt" 2>&1; rc=$?
[ "$rc" = 0 ] && ok "sweep: nothing left, exit 0" || bad "sweep: exit $rc"
new_case sweep1
touch "$S/stray"
PATH=$W/bin:$PATH GCP_PROJECT=fake-project bash "$G/sweep.sh" > "$W/sweep1.txt" 2>&1; rc=$?
[ "$rc" = 1 ] && ok "sweep: a stray is listed and exits 1" || bad "sweep: exit $rc"
has "$W/sweep1.txt" "cancho-gpu-l4-old" "sweep names the stray"
hasnt "$L" "instances delete" "sweep without DELETE=1 deletes nothing"
has "$L" "labels.purpose=cancho-gpu-test" "sweep filters on the label"
PATH=$W/bin:$PATH GCP_PROJECT=fake-project DELETE=1 bash "$G/sweep.sh" > "$W/sweep2.txt" 2>&1; rc=$?
has "$L" "instances delete cancho-gpu-l4-old --zone europe-west4-b" "DELETE=1 deletes it, in its zone"

# --- w1_cloud.sh: W1 in small restartable steps (docs/cloud.md) ---------------
# Its own stateful stub: the "VM" has a home with results/w1_out, the driver seeds it with scp, the remote command extends it as
# the plan file says (line k = the k-th VM: <records per model after this VM> <DONE|PREEMPT|CAP|DROP|DIFF|STALL>), and an scp from
# ~/results/* copies it back. Nothing leaves the machine.
if python3 -c 'import numpy' 2>/dev/null; then
mkdir -p "$W/w1bin"
cat > "$W/w1bin/gcloud" <<'STUB'
#!/bin/sh
echo "$*" >> "$STUB_LOG"
S=$STUB_STATE; cur=$S/current
for last; do :; done
case "$*" in
  *"config get-value project"*) echo "${STUB_CONFIG_PROJECT:-}"; exit 0 ;;
  *"images list"*) echo "common-cu12-ubuntu-2204"; exit 0 ;;
  *"describe-from-family"*) exit 1 ;;
  *"instances bulk create"*)
    if [ "${STUB_CAPACITY_FAIL:-0}" = 1 ]; then echo "ERROR: code: ZONE_RESOURCE_POOL_EXHAUSTED" >&2; exit 1; fi
    n=$(cat "$S/vmcount" 2>/dev/null || echo 0); echo $((n + 1)) > "$S/vmcount"
    rm -rf "$cur" "$S/preempted"; mkdir -p "$cur/w1_out"; touch "$S/vm"; exit 0 ;;
  *"instances list"*"labels.purpose"*) exit 0 ;;
  *"instances list"*) [ -f "$S/vm" ] && echo "europe-west4-b"; exit 0 ;;
  *"instances delete"*) rm -f "$S/vm"; exit 0 ;;
  *"operations list"*) [ -f "$S/preempted" ] && echo "operation-1"; exit 0 ;;
  *"compute ssh"*"remote_cuda.sh"*)
    touch "$S/ssh_running"
    k=$(cat "$S/vmcount"); line=$(sed -n "${k}p" "$STUB_PLAN"); target=${line% *}; word=${line#* }
    mx=$(echo "$*" | sed -n "s/.*W1_MAX_NEW='\([0-9]*\)'.*/\1/p")
    if [ "$word" != STALL ]; then
      for tag in base edited; do
        f=$cur/w1_out/${tag}_$mx.jsonl; have=0; [ -f "$f" ] && have=$(wc -l < "$f" | tr -d ' ')
        i=$have
        while [ "$i" -lt "$target" ]; do
          if [ $tag = edited ]; then c=true; h=false; else c=false; h=true; fi
          echo "{\"model\": \"$tag\", \"id\": $i, \"max_new\": $mx, \"correct\": $c, \"hit_limit\": $h, \"n_tokens\": 5}" >> "$f"
          i=$((i + 1))
        done
      done
    fi
    [ -n "${STUB_SSH_SLEEP:-}" ] && sleep "$STUB_SSH_SLEEP"
    case "$word" in
      DONE) echo DONE > "$cur/w1_status"; echo "remote: done"; exit 0 ;;
      PREEMPT) rm -f "$S/vm"; touch "$S/preempted"; exit 255 ;;
      CAP) rm -f "$S/vm"; exit 255 ;;
      DROP) exit 255 ;;        # the ssh connection dies, the VM lives
      DIFF) echo DIFF-FAILED > "$cur/w1_status"; exit 1 ;;
      *) echo "FAILED chunk n=8" > "$cur/w1_status"; exit 1 ;;
    esac ;;
  *"compute ssh"*) exit 0 ;;
  *"compute scp"*":~/results/* "*)
    [ -f "$S/vm" ] || exit 1
    mkdir -p "$last"; cp -R "$cur"/. "$last"/
    if [ "${STUB_TRUNCATE:-0}" = 1 ]; then for f in "$last"/w1_out/*.jsonl; do head -1 "$f" > "$f.t" && mv "$f.t" "$f"; done; fi
    touch "$S/pulled"; exit 0 ;;
  *"compute scp"*":~/results/w1_out/"*)
    for a in "$@"; do [ -f "$a" ] && cp "$a" "$cur/w1_out/"; done; touch "$S/seeded"; exit 0 ;;
  *"compute scp"*) exit 0 ;;
esac
exit 0
STUB
chmod +x "$W/w1bin/gcloud"
W1=$G/w1_cloud.sh

# the plan, read without a cloud
PATH=$W/never:$PATH DRY_RUN=1 GCP_PROJECT=fake-project W1_DIR=$W/w1dry bash "$W1" > "$W/w1dry.txt" 2>&1; rc=$?
[ "$rc" = 0 ] && ok "w1 dry run exits 0" || bad "w1 dry run exits $rc: $(tail -3 "$W/w1dry.txt")"
[ -e "$W/never.log" ] && bad "w1 dry run called gcloud: $(cat "$W/never.log")" || ok "w1 dry run called no gcloud"
has "$W/w1dry.txt" "plan: W1 behaviour, problems 0..199 of GSM8K, --max-new 8192, batch 8" "w1 dry run: the fixed protocol (13.3)"
has "$W/w1dry.txt" "chunks of 24 problems; results copied home every 10 min" "w1 dry run: CHUNK 24, SYNC 10"
has "$W/w1dry.txt" "at most 6 VMs and 6h of VM time in all" "w1 dry run: MAX_VMS 6, TOTAL_MAX 6h"
has "$W/w1dry.txt" "from memory, not looked up" "w1 dry run: the cost is said to be from memory"
has "$W/w1dry.txt" "--provisioning-model=SPOT" "w1 dry run: Spot"
has "$W/w1dry.txt" "--max-run-duration 2h --instance-termination-action DELETE" "w1 dry run: the per-VM cap (MAX_RUN 2h) and DELETE"
has "$W/w1dry.txt" "--labels purpose=cancho-gpu-w1" "w1 dry run: the purpose label"
has "$W/w1dry.txt" "--region europe-west4 " "w1 dry run: EU region first"
has "$W/w1dry.txt" "--machine-type g2-standard-8" "w1 dry run: the L4 machine"
has "$W/w1dry.txt" "instances delete cancho-w1-" "w1 dry run: prints the delete"
has "$W/w1dry.txt" "STEPS='machine w1'.*W1_N='200' W1_MAX_NEW='8192' W1_BATCH='8' W1_CHUNK='24'" "w1 dry run: the remote steps and the protocol reach the VM"
has "$W/w1dry.txt" "scp .*data/\*.jsonl .*:~/results/w1_out/" "w1 dry run: the VM is seeded with the pulled jsonl"
has "$W/w1dry.txt" "scp .*--recurse cancho-w1-.*:~/results/\*" "w1 dry run: the results are copied back"
PATH=$W/never:$PATH DRY_RUN=1 SMOKE=1 SPOT=0 MAX_RUN=45m GCP_PROJECT=fake-project W1_DIR=$W/w1dry bash "$W1" > "$W/w1dry2.txt" 2>&1
has "$W/w1dry2.txt" "problems 0..7 of GSM8K, --max-new 512" "SMOKE=1: n=8 --max-new 512"
has "$W/w1dry2.txt" "SMOKE=1: does it load" "SMOKE=1 says it is not a result"
has "$W/w1dry2.txt" "W1_N='8' W1_MAX_NEW='512'.*W1_CHUNK='8'" "SMOKE=1 reaches the VM"
hasnt "$W/w1dry2.txt" "provisioning-model" "w1 SPOT=0: on demand"
has "$W/w1dry2.txt" "--max-run-duration 45m --instance-termination-action DELETE" "w1 SPOT=0 still caps and deletes (MAX_RUN=45m)"
# the VM is made as cuda_test.sh makes it: the same zones, machine and flags
zl() { sed -n 's/^ *l4)   MACHINE=g2-standard-8; DEFAULT_ZONES="\(.*\)" ;;$/\1/p;s/^DEFAULT_ZONES="\(.*\)"$/\1/p' "$1"; }
[ -n "$(zl "$G/cuda_test.sh")" ] && [ "$(zl "$G/cuda_test.sh")" = "$(zl "$W1")" ] && ok "w1 uses cuda_test.sh's L4 zone list" || bad "w1 zone list differs from cuda_test.sh"
for flag in "--maintenance-policy TERMINATE" "--boot-disk-size 150GB --boot-disk-type pd-ssd" "--metadata install-nvidia-driver=True" "--instance-termination-action DELETE"; do
  has "$G/cuda_test.sh" "$flag" "cuda_test.sh has $flag" ; has "$W1" "$flag" "w1_cloud.sh has $flag"
done
# refused inputs create nothing
for env in "CHUNK=10" "MAX_RUN=soon" "TOTAL_MAX=soon" "N=0" "SYNC=0"; do
  new_case w1bad
  env "$env" STUB_PLAN=$W/plan PATH="$W/w1bin:$PATH" GCP_PROJECT=fake-project W1_DIR=$W/w1bad bash "$W1" > "$W/w1bad.txt" 2>&1; rc=$?
  { [ "$rc" = 2 ] && ! grep -q "bulk create" "$L"; } && ok "w1 $env: refused, exit 2, nothing created" || bad "w1 $env: exit $rc"
done

w1_run() { # name, plan lines (one per VM, "\n"-separated), env words...; sets rc, S, L
  local name=$1 plan=$2; shift 2
  new_case "$name"
  printf '%b' "$plan" > "$W/$name.plan"
  env W1_DIR="$W/$name.dir" "$@" STUB_PLAN="$W/$name.plan" PATH="$W/w1bin:$PATH" GCP_PROJECT=fake-project CAP_WAIT=0 \
    bash "$W1" > "$W/$name.txt" 2>&1
  rc=$?
}
nvm() { grep -c "instances bulk create" "$L"; }
lines() { wc -l < "$1" | tr -d ' '; }

# a preemption: the first VM is taken back after 3 problems, the second finishes; results are pulled while it runs and between VMs
w1_run pre "3 PREEMPT\n8 DONE\n" N=8 MAX_NEW=512 CHUNK=8 SYNC_SECS=1 STUB_SSH_SLEEP=3
[ "$rc" = 0 ] && ok "w1: preempted once, then done: exit 0" || bad "w1 restart: exit $rc: $(tail -5 "$W/pre.txt")"
[ "$(nvm)" = 2 ] && ok "w1: the loop restarted after the preemption (2 VMs)" || bad "w1 restart: $(nvm) VMs"
has "$W/pre.txt" "VM 1 ended: PREEMPTED" "w1: the preemption is recognised"
has "$W/pre.txt" "synced: base 3/8 edited 3/8" "w1: results were pulled while the VM was running"
[ -f "$S/seeded" ] && [ "$(lines "$W/pre.dir/data/base_512.jsonl")" = 8 ] && [ "$(lines "$W/pre.dir/data/edited_512.jsonl")" = 8 ] && ok "w1: the pulled results are home, and complete" || bad "w1 restart: data not complete"
has "$L" "scp .*base_512.jsonl.*:~/results/w1_out/" "w1: the second VM was seeded with the first one's results"
awk '/instances bulk create/{c++} /w1_out\/$/ && c==2 {s=1} END{exit !s}' "$L" && ok "w1: the seed went to the second VM, after its creation" || bad "w1: no seed after the second create"
[ "$(grep -c 'instances delete' "$L")" = 1 ] && ok "w1: the VM Google took back is not deleted again (it is gone); the one that finished is" || bad "w1: deletes: $(grep -c 'instances delete' "$L")"
has "$W/pre.txt" "W1 verdict (cap 512): reproduces" "w1: the verdict is computed locally from the pulled jsonl"
has "$W/pre.txt" "cap check (13.3(d)): ASKED" "w1: and says the rule asks for the cap check"
hasnt "$L" "W1_N='60'" "w1: the cap check is not started by itself"
[ ! -f "$S/vm" ] && ok "w1: no VM left" || bad "w1: VM left behind"
[ -f "$W/pre.dir/verdict_8_512.json" ] && ok "w1: the verdict json is written" || bad "w1: no verdict json"
has "$L" "--labels purpose=cancho-gpu-w1" "w1: the stubbed create is labelled"
has "$L" "--max-run-duration 2h --instance-termination-action DELETE" "w1: the stubbed create has the cap and DELETE"

# TOTAL_MAX: VMs of 2 h each against a 3 h total: the second gets the hour that is left, then it stops
w1_run tot "1 DROP\n2 DROP\n3 DROP\n4 DROP\n" N=8 MAX_NEW=512 CHUNK=8 TOTAL_MAX=3h MAX_RUN=2h W1_FAKE_VM_SECS=7200
[ "$rc" = 3 ] && ok "w1 TOTAL_MAX: stops with work remaining, exit 3" || bad "w1 TOTAL_MAX: exit $rc"
[ "$(nvm)" = 2 ] && ok "w1 TOTAL_MAX: two VMs, not more" || bad "w1 TOTAL_MAX: $(nvm) VMs"
has "$L" "--max-run-duration 2h " "w1 TOTAL_MAX: the first VM's cap is MAX_RUN"
has "$L" "--max-run-duration 60m " "w1 TOTAL_MAX: the second VM's cap is the time left"
has "$W/tot.txt" "TOTAL_MAX: 240 of 180 minutes" "w1 TOTAL_MAX: says why it stopped"
has "$W/tot.txt" "STOPPED before DONE: base 2/8 edited 2/8" "w1 TOTAL_MAX: says how far it got"
has "$W/tot.txt" "run the same command again" "w1 TOTAL_MAX: says how to continue"
hasnt "$W/tot.txt" "W1 verdict" "w1 TOTAL_MAX: no verdict from an incomplete run"
[ ! -f "$S/vm" ] && ok "w1 TOTAL_MAX: no VM left" || bad "w1 TOTAL_MAX: VM left behind"
w1_run tot2 "1 DROP\n2 DROP\n" N=8 MAX_NEW=512 CHUNK=8 TOTAL_MAX=100m MAX_RUN=2h W1_FAKE_VM_SECS=5800
has "$L" "--max-run-duration 100m " "w1 TOTAL_MAX below MAX_RUN: the first VM's cap is the total"
[ "$(nvm)" = 1 ] && ok "w1 TOTAL_MAX: under 5 minutes left starts no VM" || bad "w1 TOTAL_MAX: $(nvm) VMs"

# MAX_VMS
w1_run mv "1 DROP\n2 DROP\n3 DROP\n" N=8 MAX_NEW=512 CHUNK=8 MAX_VMS=2 W1_FAKE_VM_SECS=600
[ "$rc" = 3 ] && [ "$(nvm)" = 2 ] && ok "w1 MAX_VMS=2: two VMs, exit 3" || bad "w1 MAX_VMS: exit $rc, $(nvm) VMs"
has "$W/mv.txt" "MAX_VMS: 2 VMs were started" "w1 MAX_VMS: says why it stopped"
[ ! -f "$S/vm" ] && ok "w1 MAX_VMS: no VM left" || bad "w1 MAX_VMS: VM left behind"

# a VM that finishes its part but not the job is deleted before the next one is created
w1_run seq "4 DONE\n8 DONE\n" N=8 MAX_NEW=512 CHUNK=8
[ "$rc" = 0 ] && [ "$(nvm)" = 2 ] && ok "w1: a VM that ends with work remaining is followed by another (2 VMs, exit 0)" || bad "w1 seq: exit $rc, $(nvm) VMs"
awk '/instances bulk create/{c++; if (c==2 && d<1) bad=1} /instances delete/{d++} END{exit bad}' "$L" && ok "w1: the first VM was deleted before the second was created" || bad "w1: second VM created before the first was deleted"
[ "$(grep -c 'instances delete' "$L")" = 2 ] && ok "w1: every VM was deleted" || bad "w1: deletes: $(grep -c 'instances delete' "$L")"

# a VM that does nothing, twice: stop, do not burn the budget
w1_run stall "0 STALL\n0 STALL\n0 STALL\n" N=8 MAX_NEW=512 CHUNK=8 W1_FAKE_VM_SECS=600
[ "$rc" = 1 ] && [ "$(nvm)" = 2 ] && ok "w1: two VMs without progress stop the loop (exit 1, 2 VMs)" || bad "w1 stall: exit $rc, $(nvm) VMs"
has "$W/stall.txt" "two VMs in a row made no progress" "w1: and says so"
# diff --w1-expect failing is not retried
w1_run diff "0 DIFF\n8 DONE\n" N=8 MAX_NEW=512 CHUNK=8 W1_FAKE_VM_SECS=600
[ "$rc" = 1 ] && [ "$(nvm)" = 1 ] && ok "w1: a failed diff --w1-expect stops after one VM (exit 1)" || bad "w1 diff: exit $rc, $(nvm) VMs"
has "$W/diff.txt" "not the published edit" "w1: and says why"
[ ! -f "$S/vm" ] && ok "w1 diff failure: no VM left" || bad "w1 diff failure: VM left behind"
# no capacity anywhere: three rounds, then stop
w1_run cap "" N=8 MAX_NEW=512 CHUNK=8 STUB_CAPACITY_FAIL=1
[ "$rc" = 1 ] && [ "$(grep -c 'region europe-west4 ' "$L")" = 3 ] && ok "w1: no capacity three times running stops (exit 1)" || bad "w1 capacity: exit $rc"
has "$W/cap.txt" "no region had capacity" "w1: and says so"
# a truncated copy never replaces what is home
mkdir -p "$W/trunc.dir/data"
for t in base edited; do for i in 0 1 2 3 4; do echo "{\"model\": \"$t\", \"id\": $i, \"max_new\": 512, \"correct\": true, \"hit_limit\": false, \"n_tokens\": 5}"; done > "$W/trunc.dir/data/${t}_512.jsonl"; done
w1_run trunc "5 CAP\n" N=8 MAX_NEW=512 CHUNK=8 MAX_VMS=1 STUB_TRUNCATE=1 W1_FAKE_VM_SECS=600
[ "$(lines "$W/trunc.dir/data/base_512.jsonl")" = 5 ] && ok "w1: a shorter copy does not replace the jsonl that is home" || bad "w1: the jsonl at home was replaced by a shorter copy"
# a rerun continues: the seed is what is home
w1_run again "8 DONE\n" N=8 MAX_NEW=512 CHUNK=8
[ "$rc" = 0 ] && ok "w1: a run over a partial directory continues and finishes" || bad "w1 continue: exit $rc"
rm -rf "$W/again2.dir"; cp -R "$W/again.dir" "$W/again2.dir"
w1_run done2 "" N=8 MAX_NEW=512 CHUNK=8 W1_DIR=$W/again2.dir
[ "$rc" = 0 ] && [ "$(nvm)" = 0 ] && ok "w1: a finished directory starts no VM and prints the verdict" || bad "w1 finished: exit $rc, $(nvm) VMs"
has "$W/done2.txt" "=== DONE" "w1: DONE"
# SMOKE on one VM
w1_run smoke "8 DONE\n" SMOKE=1 W1_DIR=$W/smoke.dir
[ "$rc" = 0 ] && [ "$(nvm)" = 1 ] && ok "w1 SMOKE=1: n=8, one VM, done" || bad "w1 smoke: exit $rc, $(nvm) VMs"
has "$W/smoke.txt" "SMOKE: that is a check that the weights load" "w1 SMOKE=1: not a result, and says so"
has "$L" "W1_N='8' W1_MAX_NEW='512'" "w1 SMOKE=1: n=8, --max-new 512 on the VM"

# a signal: the results are copied home and the VM is deleted
for sig in TERM HUP; do
  new_case "w1sig$sig"
  printf '3 DONE\n' > "$W/w1sig.plan"
  STUB_PLAN=$W/w1sig.plan STUB_SSH_SLEEP=60 N=8 MAX_NEW=512 CHUNK=8 PATH=$W/w1bin:$PATH GCP_PROJECT=fake-project W1_DIR=$W/w1sig$sig.dir \
    bash "$W1" > "$W/w1sig$sig.txt" 2>&1 &
  pid=$!
  for _ in $(seq 1 100); do [ -f "$S/ssh_running" ] && break; sleep 0.1; done
  if [ ! -f "$S/ssh_running" ]; then bad "w1 $sig: the run never reached the remote command"; kill "$pid" 2>/dev/null; continue; fi
  kill -s "$sig" "$pid"
  wait "$pid"; rc=$?
  want=143; [ "$sig" = HUP ] && want=129
  [ "$rc" = "$want" ] && ok "w1 $sig: exits $want" || bad "w1 $sig: exit $rc, wanted $want"
  has "$L" "instances delete cancho-w1-" "w1 $sig: the VM is deleted"
  [ ! -f "$S/vm" ] && ok "w1 $sig: no VM left" || bad "w1 $sig: VM left behind"
  [ "$(lines "$W/w1sig$sig.dir/data/base_512.jsonl" 2>/dev/null)" = 3 ] && ok "w1 $sig: the results were copied home first" || bad "w1 $sig: results not copied home"
  pkill -f "sleep 60" 2>/dev/null || true
done

# the remote step, with a fake python and hf: 13.2's commands, the diff once, the chunks in order, the resume, the stop file
mkdir -p "$W/rs/venv/bin" "$W/rs/home/cancho-gpu" "$W/rs/home/results"
cp "$ROOT/cancho.toml" "$W/rs/home/cancho-gpu/"
cat > "$W/rs/venv/bin/python" <<'FAKE'
#!/bin/sh
# python <w1.py> status|diff|behave ... : records the call; complete up to the number in $RS_DONE
echo "$*" >> "$RS_LOG"
sub=$2; n=""
while [ $# -gt 0 ]; do [ "$1" = --n ] && n=$2; shift; done
case "$sub" in
  diff) exit "${RS_DIFF_RC:-0}" ;;
  status) [ "$(cat "$RS_DONE" 2>/dev/null || echo 0)" -ge "$n" ] && exit 0; exit 1 ;;
  behave) echo "$n" > "$RS_DONE"; exit "${RS_BEHAVE_RC:-0}" ;;
esac
FAKE
cat > "$W/rs/venv/bin/hf" <<'FAKE'
#!/bin/sh
echo "hf $*" >> "$RS_LOG"
FAKE
chmod +x "$W/rs/venv/bin/python" "$W/rs/venv/bin/hf"
rs_case() { # name, env words...
  local c=$1; shift
  export RS_LOG=$W/rs.$c.log RS_DONE=$W/rs.$c.done; : > "$RS_LOG"
  rm -f "$W/rs/home/w1_stop"
  ( cd "$W/rs/home/cancho-gpu" && env HOME="$W/rs/home" REMOTE_CUDA_LIB=1 STEPS="machine w1" W1_N=50 W1_CHUNK=24 W1_BATCH=8 "$@" bash -c '
      . '"$G"'/remote_cuda.sh; VENV='"$W"'/rs/venv; pyenv() { return 0; }
      s_w1; echo "rc=$? status=$(cat $R/w1_status)"' ) > "$W/rs.$c.txt" 2>&1
}
rs_case fresh
has "$W/rs.fresh.txt" "rc=0 status=DONE" "w1 step: runs to DONE"
c=$(grep -c "^hf download" "$W/rs.fresh.log"); [ "$c" = 3 ] && ok "w1 step: three downloads (base, edited, GSM8K)" || bad "w1 step: $c downloads"
has "$W/rs.fresh.log" "^hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 .*--exclude README.md --exclude LICENSE --exclude .gitattributes" "w1 step: the base, pinned, with one --exclude per pattern (13.2)"
has "$W/rs.fresh.log" "^hf download RikiyaT/Qwen3.5-0.8B-weight-space-programming --revision d1671a178705bcab7644806d23c535517f35723b .*--exclude eval/\* --exclude README.md" "w1 step: the edited, pinned, with the repeated --exclude (13.2)"
has "$W/rs.fresh.log" "^hf download openai/gsm8k main/test-00000-of-00001.parquet --repo-type dataset" "w1 step: the GSM8K parquet"
[ "$(grep -c " diff .*--w1-expect" "$W/rs.fresh.log")" = 1 ] && ok "w1 step: diff --w1-expect once" || bad "w1 step: diff count"
[ "$(grep ' behave ' "$W/rs.fresh.log" | sed 's/.*--n \([0-9]*\) .*/\1/' | tr '\n' ' ')" = "24 48 50 " ] && ok "w1 step: behave in chunks 24, 48, 50" || bad "w1 step: chunks $(grep ' behave ' "$W/rs.fresh.log" | tr '\n' '|')"
has "$W/rs.fresh.log" "behave .*--max-new 8192 --batch 8 --seed 0 --device cuda --gpu-free --no-verify" "w1 step: the protocol, on cuda, verified once so not per chunk"
{ export RS_LOG=$W/rs.resume3.log RS_DONE=$W/rs.resume3.done; : > "$RS_LOG"; echo 30 > "$RS_DONE"; }
( cd "$W/rs/home/cancho-gpu" && env HOME="$W/rs/home" REMOTE_CUDA_LIB=1 STEPS="machine w1" W1_N=50 W1_CHUNK=24 W1_BATCH=8 bash -c '
    . '"$G"'/remote_cuda.sh; VENV='"$W"'/rs/venv; pyenv() { return 0; }; s_w1; echo "rc=$? status=$(cat $R/w1_status)"' ) > "$W/rs.resume3.txt" 2>&1
[ "$(grep ' behave ' "$W/rs.resume3.log" | sed 's/.*--n \([0-9]*\) .*/\1/' | tr '\n' ' ')" = "48 50 " ] && has "$W/rs.resume3.txt" "chunk n=24: already complete, skipped" "w1 step: a seeded directory skips the finished chunk and resumes at the next" || bad "w1 step: resume ran $(grep ' behave ' "$W/rs.resume3.log" | tr '\n' '|')"
rs_case diff RS_DIFF_RC=1
{ has "$W/rs.diff.txt" "rc=1 status=DIFF-FAILED" "w1 step: a failing diff --w1-expect ends it as DIFF-FAILED"; hasnt "$W/rs.diff.log" " behave " "w1 step: and nothing is generated"; }
rs_case badchunk W1_CHUNK=10
has "$W/rs.badchunk.txt" "status=SETUP-FAILED" "w1 step: a CHUNK that is not a multiple of BATCH is refused"
rs_case behavefail RS_BEHAVE_RC=5
has "$W/rs.behavefail.txt" "rc=1 status=FAILED chunk n=24" "w1 step: a failing chunk is reported with its n"
touch "$W/rs/home/w1_stop"
( cd "$W/rs/home/cancho-gpu" && env HOME="$W/rs/home" REMOTE_CUDA_LIB=1 STEPS="machine w1" W1_N=50 W1_CHUNK=24 W1_BATCH=8 RS_LOG="$W/rs.stop.log" RS_DONE="$W/rs.stop.done" bash -c '
    . '"$G"'/remote_cuda.sh; VENV='"$W"'/rs/venv; pyenv() { return 0; }; s_w1; echo "rc=$? status=$(cat $R/w1_status)"' ) > "$W/rs.stop.txt" 2>&1
has "$W/rs.stop.txt" "status=STOPPED n=24" "w1 step: the stop file ends it cleanly after a chunk"
[ "$(grep -c ' behave ' "$W/rs.stop.log")" = 1 ] && ok "w1 step: and starts no further chunk" || bad "w1 step: chunks after the stop file"
else
  echo "skip w1_cloud.sh loop tests: numpy is not installed"
fi
sweepw1=$(grep -c "labels.purpose=cancho-gpu-w1" "$G/sweep.sh")
[ "$sweepw1" -ge 1 ] && ok "sweep.sh finds w1_cloud.sh's VMs by label" || bad "sweep.sh does not know purpose=cancho-gpu-w1"

echo "scripts/gcp: guard rails checked against a stub gcloud"
exit $fail
