#!/usr/bin/env bash
# W1's behavioural run (docs/weight-space.md 13.3, 13.5; examples/weightspace/w1.py behave) on Spot L4 VMs, in small
# restartable steps: a VM that is preempted or reaches its cap loses at most one sync interval of work, and the
# next VM carries on from what was pulled. Modelled on cuda_test.sh (same VM, same cost guard rails; docs/cloud.md says so).
#
#   gcloud auth login                              # once; the script cannot prompt
#   DRY_RUN=1 scripts/gcp/w1_cloud.sh              # the plan and every gcloud command; none is run
#   SMOKE=1 GCP_PROJECT=<p> scripts/gcp/w1_cloud.sh   # 13.3(b): n=8, --max-new 512, one VM; not a result
#   GCP_PROJECT=<p> scripts/gcp/w1_cloud.sh        # 13.3(c): problems 0..199, --max-new 8192, base and edited
#   N=60 MAX_NEW=16384 GCP_PROJECT=<p> scripts/gcp/w1_cloud.sh   # 13.3(d), only if the verdict asks for it
#   scripts/gcp/sweep.sh                           # anything left running? (label purpose=cancho-gpu-w1)
#
# THE LOOP. 1. create a Spot L4 VM exactly as cuda_test.sh does (region fallback, --max-run-duration, termination DELETE,
# label, trap cleanup); 2. seed it with the *.jsonl already pulled into $W1_DIR/data; 3. on the VM, remote_cuda.sh's `machine`
# and `w1` steps (no compiler: the `w1` step is Python only) fetch the pinned checkpoints and the parquet (13.2's commands),
# run `diff --w1-expect` once, then `behave --n CHUNK`, `--n 2*CHUNK`, ... `--n N`; each chunk ends with base and edited
# complete for its problems, and `behave` skips what the jsonl already holds; 4. every SYNC minutes this script copies the VM's
# results home, so what a preemption loses is at most the work since the last copy; 5. when the VM ends (done, preempted, at
# its cap) and work remains it creates the next one, until DONE, TOTAL_MAX of VM time summed over VMs, MAX_VMS, a VM that
# makes no progress twice running (a setup that keeps failing is that), or diff --w1-expect failing, which another VM would repeat;
# 6. it prints the verdict computed here from the pulled jsonl (`w1.py verdict`, numpy only) and whether 13.3(d) asks for the
# cap check, which it never starts by itself.
# Rerun the same command to continue: the pulled results seed the next VM, and the limits count from zero again.
#
# Knobs (environment):
#   GCP_PROJECT   billing goes here; as cuda_test.sh: printed, and a taken one waits CONFIRM_SECS (10) unless YES=1.
#   DRY_RUN=1     print the plan and each gcloud command; run no gcloud (one VM is shown, not the loop).
#   SMOKE=1       N=8, MAX_NEW=512, CHUNK=8, MAX_VMS=1, MAX_RUN=1h (each unless given), results in $ROOT/results/w1_cloud_smoke (13.3(b): do the weights load,
#                 how many tokens per second; it is not a result).
#   N (200)  MAX_NEW (8192)  BATCH (8)  SEED (0)   the protocol, fixed by 13.3; change them only for 13.3(d) or a smoke run.
#   CHUNK         problems per chunk (default 24); a multiple of BATCH, so the chunks' batches (and seeds) are those of one
#                 uninterrupted run.
#   SYNC          minutes between copies of the VM's results (default 10).
#   MAX_RUN       cap on one VM's life (default 2h), enforced by GCE (--max-run-duration), whatever happens to this script.
#   TOTAL_MAX     cap on the VM time summed over this invocation's VMs (default 6h). The last VM's cap is the time left, and
#                 no VM is started with under 5 minutes left. Each VM is billed from its creation to its deletion.
#   MAX_VMS       VMs started by this invocation (default 6).
#   SPOT=0        on-demand (about 4x the price, never preempted); the caps still apply.
#   ZONES         as cuda_test.sh (EU first; this run carries no user data: public weights and GSM8K).
#   IMAGE_FAMILY  as cuda_test.sh (cancho-gpu-l4 if it exists in the project, else the stock Deep Learning image).
#   W1_DIR        where results go (default $ROOT/results/w1_cloud): data/ is the behave directory (base_<cap>.jsonl and
#                 edited_<cap>.jsonl, what the next VM is seeded with), vm-<stamp>-<k>/ each VM's logs and copies.
#   W1_PIP        pip requirements for the VM (default "transformers>=5.2.0 pyarrow huggingface_hub numpy").
#   W1_FAKE_VM_SECS  tests only (tests/gcp.sh): account this many seconds for every VM instead of the clock.
#
# What runs on the VM is the committed HEAD (`git archive`); uncommitted changes are not shipped.
# Exit status: 0 done; 3 stopped with work remaining (limits, no progress; rerun to continue); 1 a failure; 2 usage.
set -uo pipefail
# The whole body is one function, called on the LAST line (see cuda_test.sh: bash reads a script incrementally).
main() {

SPOT="${SPOT:-1}"
DRY_RUN="${DRY_RUN:-0}"
YES="${YES:-0}"
CONFIRM_SECS="${CONFIRM_SECS:-10}"
SMOKE="${SMOKE:-0}"
TOTAL_MAX="${TOTAL_MAX:-6h}"
SYNC="${SYNC:-10}"
BATCH="${BATCH:-8}"
SEED="${SEED:-0}"
CAP_WAIT="${CAP_WAIT:-60}"            # seconds between two rounds that found no capacity
MIN_VM_SECS=300
IMAGE_FAMILY="${IMAGE_FAMILY:-cancho-gpu-l4}"
W1_PIP="${W1_PIP:-transformers>=5.2.0 pyarrow huggingface_hub numpy}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ "$SMOKE" = 1 ]; then
  N="${N:-8}"; MAX_NEW="${MAX_NEW:-512}"; CHUNK="${CHUNK:-8}"; MAX_VMS="${MAX_VMS:-1}"; MAX_RUN="${MAX_RUN:-1h}"
  W1_DIR="${W1_DIR:-$ROOT/results/w1_cloud_smoke}"
else
  N="${N:-200}"; MAX_NEW="${MAX_NEW:-8192}"; CHUNK="${CHUNK:-24}"; MAX_VMS="${MAX_VMS:-6}"; MAX_RUN="${MAX_RUN:-2h}"
  W1_DIR="${W1_DIR:-$ROOT/results/w1_cloud}"
fi
SYNC_SECS="${SYNC_SECS:-$((SYNC * 60))}"

# The same machine, zones and flags as cuda_test.sh's GPU=l4 (tests/gcp.sh holds the two lists equal).
MACHINE=g2-standard-8
DEFAULT_ZONES="europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c europe-west3-a europe-west3-b europe-west2-a europe-west2-b us-central1-a us-central1-b us-central1-c us-east1-c us-east1-d us-east4-a us-east4-c us-west1-a us-west1-b us-west4-a asia-northeast1-a asia-northeast1-c asia-east1-a asia-east1-b asia-east1-c asia-northeast3-a asia-northeast3-b asia-southeast1-a asia-southeast1-b asia-southeast1-c asia-south1-a asia-south1-b asia-south1-c"
ZONES="${ZONES:-$DEFAULT_ZONES}"
LABEL="purpose=cancho-gpu-w1"

# "2h", "90m", "45s" or a plain number of minutes -> seconds.
dur_secs() {
  local v=$1 num
  num="${v%[hms]}"
  case "$num" in ''|*[!0-9]*) return 1 ;; esac
  case "$v" in *h) echo $((num * 3600)) ;; *m) echo $((num * 60)) ;; *s) echo "$num" ;; *) echo $((num * 60)) ;; esac
}
usage() { echo "$*" >&2; exit 2; }
MAX_RUN_SECS=$(dur_secs "$MAX_RUN") || usage "MAX_RUN=$MAX_RUN: use 2h, 90m or 45s"
TOTAL_SECS=$(dur_secs "$TOTAL_MAX") || usage "TOTAL_MAX=$TOTAL_MAX: use 6h, 90m or 45s"
for v in N MAX_NEW CHUNK BATCH SEED MAX_VMS SYNC_SECS CAP_WAIT; do
  case "${!v}" in ''|*[!0-9]*) usage "$v=${!v}: a non-negative whole number" ;; esac
done
{ [ "$N" -ge 1 ] && [ "$MAX_NEW" -ge 1 ] && [ "$BATCH" -ge 1 ] && [ "$CHUNK" -ge 1 ] && [ "$MAX_VMS" -ge 1 ]; } || usage "N, MAX_NEW, BATCH, CHUNK and MAX_VMS must be positive"
[ $((CHUNK % BATCH)) = 0 ] || usage "CHUNK=$CHUNK is not a multiple of BATCH=$BATCH (aligned chunks keep the seeds of an uninterrupted run)"
[ "$SYNC_SECS" -ge 1 ] || usage "SYNC must be at least 1"
python3 -c 'import numpy' 2>/dev/null || usage "the status and the verdict are computed here and need python3 with numpy"

if [ -n "${GCP_PROJECT:-}" ]; then
  PROJECT_FROM="GCP_PROJECT"
elif [ "$DRY_RUN" = 1 ]; then
  GCP_PROJECT="DRY-RUN-PROJECT"; PROJECT_FROM="dry run placeholder"
else
  GCP_PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
  PROJECT_FROM="gcloud config get-value project"
fi
if [ -z "$GCP_PROJECT" ] || [ "$GCP_PROJECT" = "(unset)" ]; then
  echo "set GCP_PROJECT (or gcloud config set project): billing goes to it" >&2
  exit 2
fi
echo "billing project: $GCP_PROJECT (from $PROJECT_FROM)"

W1PY="$ROOT/examples/weightspace/w1.py"
DATA="$W1_DIR/data"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$DATA"

echo "plan: W1 behaviour, problems 0..$((N - 1)) of GSM8K, --max-new $MAX_NEW, batch $BATCH, seed $SEED, base and edited"
echo "      chunks of $CHUNK problems; results copied home every $((SYNC_SECS / 60)) min ($SYNC_SECS s); in $DATA"
echo "      each VM: Spot=$SPOT $MACHINE, at most $MAX_RUN; at most $MAX_VMS VMs and $TOTAL_MAX of VM time in all"
echo "      cost: Spot L4 is about \$0.3/h (from memory, not looked up) and the 150 GB pd-ssd a few cents an hour more: a full"
echo "      $TOTAL_MAX is about \$2-2.5; on-demand (SPOT=0) about \$5.5. Every VM is deleted when its work ends or at its cap."
[ "$SMOKE" = 1 ] && echo "      SMOKE=1: does it load and how fast; this is not a result"
if [ "$PROJECT_FROM" != GCP_PROJECT ] && [ "$YES" != 1 ] && [ "$DRY_RUN" != 1 ]; then
  echo "this project was not given explicitly; it will be billed for L4 VMs."
  echo "continuing in ${CONFIRM_SECS}s -- Ctrl-C to abort (YES=1 skips this)"
  sleep "$CONFIRM_SECS"
fi
[ "$DRY_RUN" = 1 ] && echo "DRY_RUN=1: gcloud commands are printed, none is run (results dir $W1_DIR)"

gc() {
  if [ "$DRY_RUN" = 1 ]; then
    echo "DRY_RUN> gcloud --project $GCP_PROJECT --quiet $*"
    return 0
  fi
  gcloud --project "$GCP_PROJECT" --quiet "$@"
}

# ---- what is done: read from the pulled jsonl, here -------------------------
complete() { python3 "$W1PY" status --dir "$DATA" --n "$N" --max-new "$MAX_NEW" >/dev/null 2>&1; }
status_line() { python3 "$W1PY" status --dir "$DATA" --n "$N" --max-new "$MAX_NEW" 2>/dev/null || true; }
have() { # problems finished, base plus edited
  status_line | sed -n 's/^base \([0-9]*\)\/[0-9]* edited \([0-9]*\)\/.*/\1 \2/p' | { read -r a b; echo $((${a:-0} + ${b:-0})); }
}

# ---- the VM: created and deleted as cuda_test.sh does ------------------------
NAME=""; ZONE=""; VMDIR=""; PULL=""
where() {
  [ -n "$NAME" ] || return 0
  if [ "$DRY_RUN" = 1 ]; then echo "${ZONES%% *}"; return 0; fi
  gc compute instances list --filter="name=$NAME" --format="value(zone.basename())" 2>/dev/null | head -1 || true
}
# Copy the VM's results home; a jsonl replaces the local one only if it has MORE lines (a copy cut short never costs what is
# already here). Quiet on failure: a preempted VM is gone and has nothing to give.
merge_pull() {
  local f b new old
  for f in "$PULL"/w1_out/*.jsonl; do
    [ -f "$f" ] || continue
    b="$(basename "$f")"
    new="$(wc -l < "$f" | tr -d ' ')"; old=0
    [ -f "$DATA/$b" ] && old="$(wc -l < "$DATA/$b" | tr -d ' ')"
    if [ "$new" -gt "$old" ]; then cp "$f" "$DATA/$b.part" && mv "$DATA/$b.part" "$DATA/$b"; fi
  done
}
pull() {
  [ -n "$ZONE" ] || return 0
  mkdir -p "$PULL"
  if gc compute scp --zone "$ZONE" --recurse "$NAME:~/results/*" "$PULL/" 2>>"$VMDIR/pull.err" >/dev/null; then
    merge_pull
    echo "synced: $(status_line)"
  else
    echo "sync: could not copy from $NAME (see $VMDIR/pull.err)" >&2
  fi
}
delete_vm() {
  local z
  z="$(where)"
  if [ -n "$z" ]; then
    if [ "$DRY_RUN" = 1 ]; then gc compute instances delete "$NAME" --zone "$z"
    else gc compute instances delete "$NAME" --zone "$z" >/dev/null 2>&1 && echo "deleted $NAME in $z"; fi
  fi
  NAME=""; ZONE=""
  return 0
}
cleanup() {
  [ -n "$NAME" ] || return 0
  [ -n "$(where)" ] || return 0
  pull
  delete_vm
}
# A signal ends the script after cleaning up; the VM's results are copied first (they are paid for), and a second signal
# during the copy ends it at once (the VM's cap and `sweep.sh` are then the guards).
on_signal() {
  trap - EXIT INT TERM HUP
  echo "signal $1: stopping, copying the results home and deleting ${NAME:-no VM}" >&2
  # shellcheck disable=SC2046
  kill $(jobs -p) 2>/dev/null || true
  pkill -P $$ 2>/dev/null || true
  cleanup
  exit "$2"
}
trap cleanup EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

# The image, once (the same choice as cuda_test.sh).
if [ "$DRY_RUN" = 1 ]; then
  gc compute images list --project deeplearning-platform-release --filter='family~^common-cu12.*ubuntu'
  gc compute images describe-from-family "$IMAGE_FAMILY"
  IMG=(--image-project deeplearning-platform-release --image-family "common-cu12-ubuntu-<newest family>")
else
  FAMILY="$(gcloud compute images list --project deeplearning-platform-release \
    --filter='family~^common-cu12.*ubuntu' --format='value(family)' | sort | tail -1)"
  [ -n "$FAMILY" ] || { echo "no common-cu12 ubuntu image family found" >&2; exit 1; }
  if gc compute images describe-from-family "$IMAGE_FAMILY" >/dev/null 2>&1; then
    IMG=(--image-project "$GCP_PROJECT" --image-family "$IMAGE_FAMILY")
    echo "image: $GCP_PROJECT/$IMAGE_FAMILY (pre-built)"
  else
    IMG=(--image-project deeplearning-platform-release --image-family "$FAMILY")
    echo "image family: deeplearning-platform-release/$FAMILY (no pre-built $IMAGE_FAMILY)"
  fi
fi

# The source goes up with every VM; make it before any VM exists.
SHIP="$W1_DIR/src.tar.gz"
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN> git -C $ROOT archive --format=tar.gz -o $SHIP HEAD"
else
  git -C "$ROOT" archive --format=tar.gz -o "$SHIP" HEAD || { echo "git archive failed" >&2; exit 1; }
fi

spot_flags=()
[ "$SPOT" = 1 ] && spot_flags=(--provisioning-model=SPOT)
REGIONS=$(for z in $ZONES; do echo "${z%-*}"; done | awk '!seen[$0]++')
# One request per region, as cuda_test.sh. $1: the VM's cap. Sets ZONE; returns 1 when no region had capacity.
create_vm() {
  local cap=$1 r allow why
  ZONE=""
  for r in $REGIONS; do
    echo "trying $MACHINE in $r"
    allow=$(for z in $ZONES; do if [ "${z%-*}" = "$r" ]; then printf '%s=allow,' "$z"; fi; done)
    if gc compute instances bulk create --region "$r" --count 1 \
        --predefined-names "$NAME" \
        --location-policy "${allow%,}" \
        --machine-type "$MACHINE" \
        --maintenance-policy TERMINATE ${spot_flags[@]+"${spot_flags[@]}"} \
        --max-run-duration "$cap" --instance-termination-action DELETE \
        "${IMG[@]}" \
        --boot-disk-size 150GB --boot-disk-type pd-ssd \
        --metadata install-nvidia-driver=True \
        --labels "$LABEL" 2>"$VMDIR/create-$r.log" >"$VMDIR/create-$r.out"; then
      [ "$DRY_RUN" = 1 ] && cat "$VMDIR/create-$r.out"
      ZONE="$(where)"
      [ -n "$ZONE" ] && return 0
      echo "  $r: created, but $NAME is not listed anywhere" >&2
      continue
    fi
    why=$(grep -oE "code: [A-Z_]+|currently unavailable|[Qq]uota [^.]*" "$VMDIR/create-$r.log" | head -1 || true)
    echo "  $r: ${why:-failed, see $VMDIR/create-$r.log}" >&2
  done
  return 1
}

# One VM, start to end. $1 its cap. Sets VM_SECS and VM_STATUS (the VM's own word: DONE, DIFF-FAILED, ...).
run_vm() {
  local cap=$1 t0 rpid last i remote_cmd files=()
  VM_SECS=0; VM_STATUS=""
  if ! create_vm "$cap"; then NAME=""; return 1; fi
  t0=$SECONDS
  echo "$NAME up in $ZONE (cap $cap)"
  if [ "$DRY_RUN" = 1 ]; then
    gc compute ssh "$NAME" --zone "$ZONE" --command true
  else
    for i in $(seq 1 60); do
      gc compute ssh "$NAME" --zone "$ZONE" --command true >/dev/null 2>&1 && break
      [ "$i" = 60 ] || sleep 10
    done
  fi
  gc compute scp --zone "$ZONE" "$SHIP" "$NAME:~/src.tar.gz"
  gc compute ssh "$NAME" --zone "$ZONE" --command 'mkdir -p ~/results/w1_out'
  # seed: what is already home
  for f in "$DATA"/*.jsonl; do [ -f "$f" ] && files+=("$f"); done
  if [ "${#files[@]}" -gt 0 ]; then
    gc compute scp --zone "$ZONE" "${files[@]}" "$NAME:~/results/w1_out/"
  elif [ "$DRY_RUN" = 1 ]; then
    gc compute scp --zone "$ZONE" "$DATA/*.jsonl" "$NAME:~/results/w1_out/"
  fi
  remote_cmd="mkdir -p cancho-gpu && tar -xzf src.tar.gz -C cancho-gpu && STEPS='machine w1' GPU='l4' W1_N='$N' W1_MAX_NEW='$MAX_NEW' W1_BATCH='$BATCH' W1_CHUNK='$CHUNK' W1_SEED='$SEED' W1_PIP='$W1_PIP' bash cancho-gpu/scripts/gcp/remote_cuda.sh"
  if [ "$DRY_RUN" = 1 ]; then
    gc compute ssh "$NAME" --zone "$ZONE" --command "$remote_cmd"
    echo "DRY_RUN: while that runs, every $((SYNC_SECS / 60)) min: gcloud compute scp --recurse $NAME:~/results/* $PULL/ (jsonl merged into $DATA)"
    gc compute scp --zone "$ZONE" --recurse "$NAME:~/results/*" "$PULL/"
  else
    rm -f "$VMDIR/remote.status"
    { gc compute ssh "$NAME" --zone "$ZONE" --command "$remote_cmd" </dev/null; echo $? > "$VMDIR/remote.status"; } \
      2>&1 | tee "$VMDIR/remote.log" &
    rpid=$!
    last=$SECONDS
    while kill -0 "$rpid" 2>/dev/null; do
      sleep 1
      if [ $((SECONDS - last)) -ge "$SYNC_SECS" ]; then pull; last=$SECONDS; fi
    done
    wait "$rpid" 2>/dev/null
    echo "remote command ended (exit $(cat "$VMDIR/remote.status" 2>/dev/null || echo unknown))"
    pull
    VM_STATUS=""
    [ -f "$PULL/w1_status" ] && VM_STATUS="$(tr -d '\n' < "$PULL/w1_status")"
    if [ -z "$VM_STATUS" ] || [ "$VM_STATUS" = RUNNING ]; then
      if [ -n "$(gc compute operations list --filter="operationType=compute.instances.preempted AND targetLink~/$NAME\$" \
          --format='value(name)' 2>/dev/null)" ]; then VM_STATUS=PREEMPTED; else VM_STATUS="ENDED-EARLY(cap or error)"; fi
    fi
  fi
  delete_vm
  VM_SECS=$((SECONDS - t0))
  [ -n "${W1_FAKE_VM_SECS:-}" ] && VM_SECS=$W1_FAKE_VM_SECS
  return 0
}

# ---- the loop ----------------------------------------------------------------
USED=0; VMS=0; NOPROG=0; CAPFAIL=0; REASON=""; FATAL=0
while :; do
  if complete; then REASON=DONE; break; fi
  if [ "$VMS" -ge "$MAX_VMS" ]; then REASON="MAX_VMS: $MAX_VMS VMs were started"; break; fi
  left=$((TOTAL_SECS - USED))
  if [ "$left" -lt "$MIN_VM_SECS" ]; then REASON="TOTAL_MAX: $((USED / 60)) of $((TOTAL_SECS / 60)) minutes of VM time used"; break; fi
  cap="$MAX_RUN"
  [ "$left" -ge "$MAX_RUN_SECS" ] || cap="$((left / 60))m"
  VMS=$((VMS + 1))
  NAME="cancho-w1-${STAMP}-${VMS}"
  VMDIR="$W1_DIR/vm-${STAMP}-${VMS}"; PULL="$VMDIR/pull"
  mkdir -p "$VMDIR"
  before="$(have)"
  echo
  echo "=== VM $VMS of at most $MAX_VMS: $NAME, cap $cap; $(status_line); $((USED / 60)) min of $((TOTAL_SECS / 60)) used"
  if ! run_vm "$cap"; then
    VMS=$((VMS - 1)); CAPFAIL=$((CAPFAIL + 1))
    if [ "$CAPFAIL" -ge 3 ]; then REASON="no region had capacity (or quota) for $MACHINE three times running; see $VMDIR/create-*.log"; FATAL=1; break; fi
    echo "no region had capacity (or quota): waiting ${CAP_WAIT}s and trying again ($CAPFAIL of 3)" >&2
    sleep "$CAP_WAIT"
    continue
  fi
  CAPFAIL=0
  USED=$((USED + VM_SECS))
  after="$(have)"
  echo "=== VM $VMS ended: ${VM_STATUS:-dry run}; ${VM_SECS}s; problems finished $before -> $after (base plus edited)"
  if [ "$DRY_RUN" = 1 ]; then REASON="dry run: one VM is shown; a real run repeats this until DONE, TOTAL_MAX or MAX_VMS"; break; fi
  case "$VM_STATUS" in
    DIFF-FAILED) REASON="diff --w1-expect failed on the VM: the edited checkpoint is not the published edit (see $PULL/w1_diff.txt); another VM would say the same"; FATAL=1; break ;;
  esac
  if [ "$after" -gt "$before" ]; then NOPROG=0; else NOPROG=$((NOPROG + 1)); fi
  if [ "$NOPROG" -ge 2 ]; then REASON="two VMs in a row made no progress (last: ${VM_STATUS}; see $VMDIR/remote.log)"; FATAL=1; break; fi
done

# ---- the end -----------------------------------------------------------------
echo
echo "=== $REASON"
echo "VM time this invocation: $((USED / 60)) min over $VMS VM(s) (Spot L4 about \$0.3/h from memory: about \$$(awk -v s="$USED" 'BEGIN{printf "%.2f", s/3600*0.34}') with the disk)"
if [ "$DRY_RUN" = 1 ]; then echo "dry run: no VM was created, nothing was read or deleted"; exit 0; fi
tokens="$(python3 - "$DATA" "$MAX_NEW" <<'PY'
import glob, json, sys
n = 0
for f in glob.glob(sys.argv[1] + "/*_%s.jsonl" % sys.argv[2]):
    for line in open(f, encoding="utf-8"):
        try:
            n += int(json.loads(line).get("n_tokens", 0))
        except (ValueError, AttributeError, TypeError):
            pass
print(n)
PY
)"
echo "tokens in the pulled records: $tokens (setup and model loading are in the VM time)"
if complete; then
  python3 "$W1PY" verdict --dir "$DATA" --n "$N" --max-new "$MAX_NEW" --json "$W1_DIR/verdict_${N}_${MAX_NEW}.json"
  [ "$SMOKE" = 1 ] && echo "SMOKE: that is a check that the weights load and run, not a result"
  echo "verdict written to $W1_DIR/verdict_${N}_${MAX_NEW}.json"
  exit 0
fi
echo "STOPPED before DONE: $(status_line)"
echo "no verdict: it needs problems 0..$((N - 1)) for both models. What is home stays in $DATA."
if [ "$FATAL" = 1 ]; then exit 1; fi
echo "to continue, run the same command again: the pulled results seed the next VM (the limits count from zero)."
exit 3
}
main "$@"; exit $?
