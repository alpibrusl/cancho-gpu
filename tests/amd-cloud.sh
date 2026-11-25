#!/bin/sh
# scripts/amd/ without an AMD GPU and without DigitalOcean (docs/amd.md slice H6):
#
#   * the launcher, hip_test.sh, against a stub doctl, ssh and scp that log what
#     they are asked: a dry run runs nothing; a run creates, runs and deletes in
#     that order; the Droplet is deleted when the remote run fails, when the
#     script is signalled and when its time cap is reached; no key, a bad cap
#     and a failed create stop before anything is billed; `sweep` lists and
#     deletes;
#   * the on-card script, remote_hip.sh, with the emulating HIP runtime and
#     hipRTC of tests/emu/ in place of AMD's, on a few cases;
#   * hiprtc_check.c on a file that does not compile.
#
# It shows the scripts do what they say with the commands they issue. It does
# not show that DigitalOcean accepts those commands: no real doctl is run.
#
#     CANCHO=<pinned cancho> tests/amd-cloud.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
mkdir -p "$W/bin"
cat > "$W/bin/doctl" <<'STUB'
#!/bin/sh
echo "doctl $*" >> "$STUB_LOG"
case "$3" in
  create)
    [ -z "${STUB_CREATE_FAIL:-}" ] || { echo "Error: POST droplets: 422 size is unavailable" >&2; exit 1; }
    echo "4242 $4 10.9.8.7 active" > "$STUB_STATE"; echo "4242 10.9.8.7" ;;
  list) [ -f "$STUB_STATE" ] && cat "$STUB_STATE" || true ;;
  delete) rm -f "$STUB_STATE" ;;
esac
STUB
cat > "$W/bin/ssh" <<'STUB'
#!/bin/sh
for last; do :; done
echo "ssh $last" | cut -c1-120 >> "$STUB_LOG"
[ "$last" = true ] && exit 0
[ -z "${STUB_SSH_SLEEP:-}" ] || sleep "$STUB_SSH_SLEEP"
exit "${STUB_SSH_RC:-0}"
STUB
cat > "$W/bin/scp" <<'STUB'
#!/bin/sh
echo "scp" >> "$STUB_LOG"
STUB
chmod +x "$W/bin/"*
export STUB_LOG="$W/log" STUB_STATE="$W/state"
fail=0
ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; fail=1; }
T="$ROOT/scripts/amd/hip_test.sh"
launch() { # [VAR=value ...]: one run of the launcher with the stubs first on PATH
  : > "$STUB_LOG"; rm -f "$STUB_STATE"
  status=0
  env PATH="$W/bin:$PATH" YES=1 SSH_RETRY_SECS=0 OUT_DIR="$W/out" "$@" "$T" > "$W/stdout" 2> "$W/stderr" || status=$?
}
line() { grep -n "$1" "$STUB_LOG" | head -1 | cut -d: -f1; }

launch DRY_RUN=1
if [ "$status" -eq 0 ] && [ ! -s "$STUB_LOG" ] && grep -q "DRY_RUN> doctl compute droplet create .* --size gpu-mi300x1-192gb .* --tag-name cancho-gpu-test" "$W/stdout" && grep -q "DRY_RUN> doctl compute droplet delete" "$W/stdout"; then
  ok "a dry run prints the create and the delete and runs nothing"
else bad "dry run: exit $status, $(wc -l < "$STUB_LOG") commands run"; fi

launch DO_SSH_KEY=aa:bb
c=$(line 'droplet create'); s=$(line 'remote_hip.sh'); d=$(line 'droplet delete')
if [ "$status" -eq 0 ] && [ -n "$c" ] && [ -n "$s" ] && [ -n "$d" ] && [ "$c" -lt "$s" ] && [ "$s" -lt "$d" ] && [ ! -f "$STUB_STATE" ] && grep -q -- '--ssh-keys aa:bb' "$STUB_LOG"; then
  ok "a run creates, runs the remote script and deletes, in that order"
else bad "normal run: exit $status; $(cat "$STUB_LOG")"; fi

launch DO_SSH_KEY=aa:bb STUB_SSH_RC=3
if [ "$status" -eq 3 ] && [ ! -f "$STUB_STATE" ] && grep -q 'droplet delete 4242 --force' "$STUB_LOG"; then
  ok "a failing remote run is the exit status, and the Droplet is deleted"
else bad "failing remote: exit $status"; fi

launch
if [ "$status" -eq 2 ] && [ ! -s "$STUB_LOG" ]; then ok "no SSH key: refused before anything is created"; else bad "no key: exit $status"; fi

launch DO_SSH_KEY=aa:bb MAX_RUN=soon
if [ "$status" -eq 2 ] && [ ! -s "$STUB_LOG" ]; then ok "a cap that is not minutes: refused before anything is created"; else bad "bad cap: exit $status"; fi

launch DO_SSH_KEY=aa:bb STUB_CREATE_FAIL=1
if [ "$status" -eq 1 ] && ! grep -q '^ssh' "$STUB_LOG" && grep -q 'size is unavailable' "$W/stderr"; then
  ok "a failed create ends the run with DigitalOcean's reason"
else bad "failed create: exit $status"; fi

launch DO_SSH_KEY=aa:bb STUB_SSH_SLEEP=3 WATCHDOG_SECS=1
if [ ! -f "$STUB_STATE" ] && grep -q 'MAX_RUN' "$W/stderr" && [ "$(grep -c 'droplet delete' "$STUB_LOG")" -ge 1 ]; then
  ok "the time cap deletes the Droplet while the remote run is still going"
else bad "watchdog: $(cat "$STUB_LOG"; cat "$W/stderr")"; fi

: > "$STUB_LOG"; rm -f "$STUB_STATE"
env PATH="$W/bin:$PATH" YES=1 SSH_RETRY_SECS=0 OUT_DIR="$W/out" DO_SSH_KEY=aa:bb STUB_SSH_SLEEP=20 "$T" > "$W/stdout" 2> "$W/stderr" &
pid=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do grep -q 'remote_hip.sh' "$STUB_LOG" 2>/dev/null && break; sleep 1; done
kill -TERM "$pid"
status=0; wait "$pid" || status=$?
if [ "$status" -eq 143 ] && [ ! -f "$STUB_STATE" ] && grep -q 'droplet delete 4242 --force' "$STUB_LOG"; then
  ok "SIGTERM mid-run deletes the Droplet and exits 143"
else bad "signal: exit $status; $(cat "$STUB_LOG")"; fi

echo "4242 cancho-gpu-mi300x-old 10.9.8.7 active" > "$STUB_STATE"; : > "$STUB_LOG"
env PATH="$W/bin:$PATH" "$T" sweep > "$W/stdout" 2>&1
if grep -q 'cancho-gpu-mi300x-old' "$W/stdout" && [ -f "$STUB_STATE" ] && grep -q -- '--gpus --tag-name cancho-gpu-test' "$STUB_LOG"; then ok "sweep lists a stray and leaves it"; else bad "sweep list"; fi
env PATH="$W/bin:$PATH" DELETE=1 "$T" sweep > "$W/stdout" 2>&1
if [ ! -f "$STUB_STATE" ] && grep -q 'deleted cancho-gpu-mi300x-old' "$W/stdout"; then ok "DELETE=1 sweep deletes it"; else bad "sweep delete"; fi

# The on-card script, with the emulator standing in for the card.
"$ROOT/scripts/build-emu.sh"
"$ROOT/scripts/build-device.sh" >/dev/null
E=$ROOT/build/emu
status=0
HIP_LOCAL=1 HIP_RESULTS="$W/results" STEPS="machine hiprtc wave device_cases" JOBS=2 \
  HIP_ONLY='^silu_mul$|^rmsnorm_n=4096_eps=1e-5$|^gemm_mma_m=128_n=128_k=64$' ROCM_PATH="$W/no-rocm" \
  CANCHOGPU_LIBHIP=$E/libamdhip64_emu.so CANCHOGPU_LIBHIPRTC=$E/libhiprtc_emu.so CANCHOGPU_EMU_CACHE=$E/cache \
  "$ROOT/scripts/amd/remote_hip.sh" > "$W/remote.out" 2>&1 || status=$?
if [ "$status" -eq 0 ] && grep -q '^hipRTC compiled 3 of 3 golden .hip files' "$W/results/answers.txt" \
    && grep -q 'a 64-lane wave' "$W/results/answers.txt" && grep -q "^a launch's arguments: params" "$W/results/answers.txt" \
    && grep -q '^3 cases that emit HIP on the card: 3 agree' "$W/results/answers.txt" && ! grep -q FAIL "$W/results/summary.tsv"; then
  ok "remote_hip.sh against the emulator: 3 kernels compile, the wave is 64 lanes, 3 cases agree"
else bad "remote_hip.sh: exit $status: $(tail -15 "$W/remote.out")"; fi

printf '#include <hip/hip_fp16.h>\nextern "C" __global__ void broken(float* y) { y[0] = nothing; }\n' > "$W/broken.hip"
status=0
CANCHOGPU_EMU_CACHE=$E/cache "$ROOT/build/hiprtc_check" "$E/libhiprtc_emu.so" "$W/broken.hip" > "$W/check.out" 2> "$W/check.err" || status=$?
if [ "$status" -eq 1 ] && grep -q "^FAIL $W/broken.hip" "$W/check.out" && grep -q 'nothing' "$W/check.err"; then
  ok "hiprtc_check reports a kernel that does not compile, with the compiler's log"
else bad "hiprtc_check: exit $status"; fi

[ "$fail" -eq 0 ] && echo "scripts/amd: the launcher's guard rails against stub commands, and the on-card script against the emulator"
exit $fail
