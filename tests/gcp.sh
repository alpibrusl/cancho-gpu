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
#   scripts     parse (bash -n; shellcheck when installed)
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
new = open(root + "/" + g).read()
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

echo "scripts/gcp: guard rails checked against a stub gcloud"
exit $fail
