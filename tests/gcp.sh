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
has "$W/dry.txt" "--max-run-duration 2h" "dry run: --max-run-duration 2h (MAX_RUN default)"
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
has "$L" "--max-run-duration 2h --instance-termination-action DELETE" "create has --max-run-duration and DELETE"
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
