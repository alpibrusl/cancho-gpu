#!/usr/bin/env bash
# Run cancho-gpu's HIP checks on a real AMD GPU (one Instinct MI300X Droplet of
# the AMD Developer Cloud, which is DigitalOcean), bring the results home, and
# delete the Droplet. docs/amd.md slice H6; modelled on scripts/gcp/cuda_test.sh.
#
#   doctl auth init                                  # once; this script never sees the token
#   DRY_RUN=1 scripts/amd/hip_test.sh                # print every command; run none
#   DO_SSH_KEY=<fingerprint or id> scripts/amd/hip_test.sh
#   scripts/amd/hip_test.sh sweep                    # list this script's Droplets; DELETE=1 deletes them
#
# What runs on the Droplet is scripts/amd/remote_hip.sh, against the committed
# HEAD (`git archive`): uncommitted changes are not tested.
#
# Knobs (environment):
#   DO_SSH_KEY   the SSH key (fingerprint or ID, as `doctl compute ssh-key list`
#                shows it) put in the Droplet's root account. Required: without
#                one DigitalOcean mails a password, which this script cannot use.
#   SSH_KEY_FILE the private key file for it, if not ssh's default.
#   YES=1        skip the wait before the Droplet is created.
#   DRY_RUN=1    print each doctl, ssh and scp command instead of running it.
#                Nothing is created, read or deleted; no login is needed.
#   SIZE         gpu-mi300x1-192gb (one MI300X, 192 GB).
#   IMAGE        gpu-amd-base (DigitalOcean's AI/ML-ready AMD image).
#   REGION       atl1 (the one region that lists the MI300X on demand).
#   MAX_RUN      minutes the Droplet may live (default 90): a watchdog in THIS
#                process deletes it then. See "Cost guard rails".
#   KEEP=1       leave the Droplet running afterwards; you delete it.
#   STEPS, JOB, CASE_TIMEOUT, JOBS, HIP_ONLY, CANCHOGPU_HIP_ARCH, CANCHO_RELEASE,
#   CANCHO_FLAVOUR are passed through to remote_hip.sh; its header lists them.
#   OUT_DIR      where results go (default results/amd/<time>).
#
# The size, image and region names and the price printed below were READ in
# DigitalOcean's documentation on 2026-11-25 and never used by this project:
# the pages disagree about the image, and `atl1` is inferred from "Atlanta
# (ATL1)". The first real run confirms them; `doctl compute size list`,
# `image list-distribution` and `region list` show what the account can use.
#
# Exit status: the remote run's; 2 for a usage error; 1 when no Droplet came up.
#
# Cost guard rails, and their limit. A GPU Droplet is billed by the second
# from creation until it is DESTROYED: powering it off does not stop billing,
# and DigitalOcean has no "delete after N minutes" of its own. So everything
# here is this process's doing: the Droplet is deleted on exit, on error and on
# SIGINT, SIGTERM and SIGHUP, found by its NAME among the Droplets tagged
# cancho-gpu-test (not by a remembered id), and a watchdog deletes it after
# MAX_RUN minutes. IF THIS PROCESS IS KILLED WITH SIGKILL, OR THE MACHINE IT
# RUNS ON SLEEPS OR LOSES POWER, NOTHING DELETES THE DROPLET: run
# `scripts/amd/hip_test.sh sweep` afterwards, and after any run you doubt.
set -euo pipefail
main() {

TAG=cancho-gpu-test
SIZE="${SIZE:-gpu-mi300x1-192gb}"
IMAGE="${IMAGE:-gpu-amd-base}"
REGION="${REGION:-atl1}"
MAX_RUN="${MAX_RUN:-90}"
KEEP="${KEEP:-0}"
DRY_RUN="${DRY_RUN:-0}"
YES="${YES:-0}"
CONFIRM_SECS="${CONFIRM_SECS:-10}"
PRICE="${PRICE:-2.59}"   # USD an hour for gpu-mi300x1-192gb, as read; only printed
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

run() { # every doctl, ssh and scp call goes through here
  if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN> $*"; return 0; fi
  "$@"
}
# The Droplets this script made: `<id> <name> <ip> <status>` a line. GPU Droplets are listed only with --gpus.
mine() { doctl compute droplet list --gpus --tag-name "$TAG" --format ID,Name,PublicIPv4,Status --no-header; }

if [ "${1:-}" = sweep ]; then
  list="$(mine)"
  if [ -z "$list" ]; then echo "no Droplet tagged $TAG"; return 0; fi
  echo "$list"
  if [ "${DELETE:-0}" = 1 ]; then
    echo "$list" | while read -r id name _; do doctl compute droplet delete "$id" --force && echo "deleted $name ($id)"; done
  else
    echo "these are billed until destroyed; DELETE=1 scripts/amd/hip_test.sh sweep deletes them"
  fi
  return 0
fi
case "$MAX_RUN" in ''|*[!0-9]*) echo "MAX_RUN is minutes, a whole number" >&2; return 2 ;; esac
if [ -z "${DO_SSH_KEY:-}" ]; then
  if [ "$DRY_RUN" = 1 ]; then DO_SSH_KEY="DRY-RUN-KEY"; else
    echo "set DO_SSH_KEY to an SSH key of the account (doctl compute ssh-key list): the Droplet is reached with it" >&2
    return 2
  fi
fi

STAMP="$(date -u +%Y%m%d-%H%M%S)"
NAME="cancho-gpu-mi300x-${STAMP}"
if [ -n "${OUT_DIR:-}" ]; then OUT="$OUT_DIR"; else OUT="$ROOT/results/amd/${STAMP}"; fi
mkdir -p "$OUT"
echo "one $SIZE in $REGION from $IMAGE, for at most $MAX_RUN minutes: about \$$PRICE an hour as read on 2026-11-25, billed by the second until the Droplet is destroyed"
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN=1: commands are printed, none is run (results dir $OUT)"
elif [ "$YES" != 1 ]; then
  echo "creating it in ${CONFIRM_SECS}s -- Ctrl-C to abort (YES=1 skips this)"
  sleep "$CONFIRM_SECS"
fi

ID=""; IP=""
RESULTS_DONE=0
SIGNALLED=0
WATCHDOG=""
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$OUT/known_hosts" -o ConnectTimeout=15 -o BatchMode=yes)
[ -n "${SSH_KEY_FILE:-}" ] && SSH_OPTS+=(-i "$SSH_KEY_FILE")
# Where the Droplet named $NAME is, if it exists: asked by name, not remembered,
# so an interrupt between "created" and "we know its id" leaves nothing behind.
locate() {
  if [ "$DRY_RUN" = 1 ]; then ID="DRY-RUN-ID"; IP="DRY-RUN-IP"; return 0; fi
  local line
  line="$(mine 2>/dev/null | awk -v n="$NAME" '$2 == n { print $1, $3; exit }')" || true
  ID="${line%% *}"; IP="${line#* }"
  [ -n "$ID" ]
}
collect_results() {
  [ "$RESULTS_DONE" = 1 ] && return 0
  [ -n "$IP" ] || return 0
  RESULTS_DONE=1
  run scp "${SSH_OPTS[@]}" -r "root@$IP:results/." "$OUT/" || true
}
cleanup() {
  [ -n "$WATCHDOG" ] && kill "$WATCHDOG" 2>/dev/null || true
  locate || return 0
  [ "$SIGNALLED" = 1 ] || collect_results
  if [ "$KEEP" = 1 ]; then
    echo "KEEP=1: $NAME ($ID) is still running AND BILLED. Delete it with:"
    echo "  doctl compute droplet delete $ID --force"
    return 0
  fi
  if run doctl compute droplet delete "$ID" --force; then
    [ "$DRY_RUN" = 1 ] || echo "deleted $NAME ($ID)"
  else
    echo "COULD NOT DELETE $NAME ($ID): it is still billed. Run: scripts/amd/hip_test.sh sweep" >&2
  fi
  return 0
}
on_signal() {
  trap - EXIT
  SIGNALLED=1
  echo "signal $1: stopping and deleting $NAME" >&2
  # shellcheck disable=SC2046
  kill $(jobs -p) 2>/dev/null || true
  cleanup
  exit "$2"
}
trap cleanup EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

if ! run doctl compute droplet create "$NAME" --size "$SIZE" --image "$IMAGE" --region "$REGION" \
    --ssh-keys "$DO_SSH_KEY" --tag-name "$TAG" --wait --format ID,PublicIPv4 --no-header >"$OUT/create.out" 2>"$OUT/create.log"; then
  echo "no Droplet: $(tail -3 "$OUT/create.log")" >&2
  return 1
fi
[ "$DRY_RUN" = 1 ] && cat "$OUT/create.out"
locate || { echo "created, but $NAME is not among the Droplets tagged $TAG" >&2; return 1; }
echo "$NAME up: id $ID, $IP"
if [ "$DRY_RUN" != 1 ]; then
  # The watchdog: this process's own cap on the Droplet's life (the header says what it cannot do).
  ( sleep "${WATCHDOG_SECS:-$((MAX_RUN * 60))}"; echo "MAX_RUN ($MAX_RUN min) reached: deleting $NAME" >&2; doctl compute droplet delete "$ID" --force >/dev/null 2>&1 ) &
  WATCHDOG=$!
fi

if [ "$DRY_RUN" = 1 ]; then
  run ssh "${SSH_OPTS[@]}" "root@$IP" true
else
  up=0
  for _ in $(seq 1 60); do
    if ssh "${SSH_OPTS[@]}" "root@$IP" true >/dev/null 2>&1; then up=1; break; fi
    sleep "${SSH_RETRY_SECS:-10}"
  done
  [ "$up" = 1 ] || { echo "$NAME never answered ssh" >&2; return 1; }
fi

SHIP="$OUT/src.tar.gz"
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN> git -C $ROOT archive --format=tar.gz -o $SHIP HEAD"
else
  git -C "$ROOT" archive --format=tar.gz -o "$SHIP" HEAD
fi
run scp "${SSH_OPTS[@]}" "$SHIP" "root@$IP:src.tar.gz"
if [ -n "${JOB:-}" ]; then
  printf '%s\n' "$JOB" > "$OUT/job.sh"
  run scp "${SSH_OPTS[@]}" "$OUT/job.sh" "root@$IP:job.sh"
fi
fwd=""
for v in STEPS CASE_TIMEOUT JOBS HIP_ONLY CANCHOGPU_HIP_ARCH CANCHO_RELEASE CANCHO_FLAVOUR; do
  if [ -n "${!v:-}" ]; then fwd="$fwd $v='${!v}'"; fi
done
remote_cmd="mkdir -p cancho-gpu results && tar -xzf src.tar.gz -C cancho-gpu && JOB='${JOB:+1}'$fwd bash cancho-gpu/scripts/amd/remote_hip.sh"

# A failing run must still bring its logs home: no errexit from here on.
set +e
if [ "$DRY_RUN" = 1 ]; then
  run ssh "${SSH_OPTS[@]}" "root@$IP" "$remote_cmd"
  status=0
else
  rm -f "$OUT/remote.status"
  { ssh "${SSH_OPTS[@]}" "root@$IP" "$remote_cmd"; echo $? > "$OUT/remote.status"; } 2>&1 | tee "$OUT/remote.log" &
  wait $!
  status=$(cat "$OUT/remote.status" 2>/dev/null || echo 255)
fi
collect_results
echo "results in $OUT (remote exit $status)"
return "$status"
}
main "$@"; exit $?
