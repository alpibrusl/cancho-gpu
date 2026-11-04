#!/usr/bin/env bash
# Run cancho-gpu's CUDA checks on a real NVIDIA GPU in Google Cloud, bring the
# results home, and delete the VM. Modelled on lex-gpu's scripts/gcp/nvidia_test.sh
# (docs/cloud.md says what differs and why).
#
#   gcloud auth login                       # once; the script cannot prompt
#   DRY_RUN=1 scripts/gcp/cuda_test.sh      # print every gcloud command; run none
#   GCP_PROJECT=my-project scripts/gcp/cuda_test.sh
#
# What runs on the VM is scripts/gcp/remote_cuda.sh, against the committed HEAD
# (`git archive`): uncommitted changes are not tested.
#
# Knobs (environment):
#   GCP_PROJECT  billing goes here. Given explicitly, or taken from
#                `gcloud config get-value project`; either way it is PRINTED,
#                and a taken one waits CONFIRM_SECS (10) first -- Ctrl-C aborts.
#   YES=1        skip that wait.
#   DRY_RUN=1    print each gcloud command instead of running it. Nothing is
#                created, read or deleted; no login is needed.
#   GPU          l4 (default, g2-standard-8, 24 GB, Ada sm_89), a100
#                (a2-highgpu-1g) or h100 (a3-highgpu-1g). The CUDA lowering is
#                Ada's ONLY (README, Known gaps): on a100/h100 the code is
#                compiled for the device's own arch by NVRTC but scheduled for
#                Ada, so a result there is not an Ada result.
#   ZONES        zones to try, EU first. These runs carry no user data (this
#                repository, public weights), which is why leaving the EU is
#                allowed here; set it to pin a region.
#   SPOT=0       on-demand instead of the default Spot (60-70% cheaper, can be
#                preempted: exit 75).
#   MAX_RUN      hard cap on the VM's life (default 2h), enforced by GCE.
#   KEEP=1       leave the VM running afterwards; you delete it.
#   IMAGE_FAMILY boot from this family in GCP_PROJECT (default cancho-gpu-l4,
#                made by build_image.sh), else the stock Deep Learning image.
#   JOB          run this one shell command on the GPU and nothing else (in the
#                source tree, compiler built; CANCHO is exported):
#                  JOB='scripts/build-device.sh && build/cancho-gpu-device run tests/chx/ported/gemm_mma.chx m=256 n=256 k=128'
#   STEPS        which steps to run, default "machine toolchain gate nvrtc
#                device_cases gemm_bench". Add "baselines" (or BASELINES=1).
#   BASELINES=1  also the torch/peft LoRA baselines, Qwen2.5-0.5B and 1.5B at
#                4x512 (public Hugging Face downloads; no user data).
#   CANCHO_REPO  where the compiler at the cancho.toml pin comes from: a git
#                URL (default https://github.com/alpibrusl/cancho) or a LOCAL
#                checkout, from which the pinned commit is shipped by `git
#                archive` (use this if the repository is private).
#   CANCHO_RELEASE the compiler the VM INSTALLS instead of building: `latest`
#                (default), a release tag/sha, or `source` (build the pinned
#                commit from CANCHO_REPO, which needs Rust and minutes). The
#                VM downloads cancho-rust-<sha>-linux-x86_64.tar.gz from
#                github.com/alpibrusl/cancho/releases (public), checks its
#                published sha256 and, if the download or the check fails,
#                falls back to building from source. A release whose rev is not
#                the cancho.toml pin is run with --ignore-compiler-rev and
#                summary.tsv says NOT-PINNED: such a run is not a pinned-compiler
#                result. CANCHO_FLAVOUR=rust (default; CI's compiler, reads
#                $CLANG) or standalone (built by cancho; untested here).
#   OUT_DIR      where results go (default results/gcp/<time>-<gpu>).
#   More (CASE_TIMEOUT, JOBS, SIZES, DTYPES, KNOWN_OPEN, TORCH_INDEX) are
#   passed through to remote_cuda.sh; its header lists them.
#
# Exit status: the remote run's, or 75 when Google preempted the Spot VM
# (retry: `for i in 1 2 3; do ...cuda_test.sh; [ $? = 75 ] || break; done`).
#
# Cost guard rails: Spot; --max-run-duration + --instance-termination-action
# DELETE (GCE deletes the VM even if this script is killed); the VM is deleted
# on exit, error, SIGINT, SIGTERM and SIGHUP, found by NAME in the project
# (not by remembered zone); every VM carries label purpose=cancho-gpu-test so
# scripts/gcp/sweep.sh can find strays.
set -euo pipefail

GPU="${GPU:-l4}"
SPOT="${SPOT:-1}"
MAX_RUN="${MAX_RUN:-2h}"
KEEP="${KEEP:-0}"
DRY_RUN="${DRY_RUN:-0}"
YES="${YES:-0}"
CONFIRM_SECS="${CONFIRM_SECS:-10}"
BASELINES="${BASELINES:-0}"
STEPS="${STEPS:-machine toolchain gate nvrtc device_cases gemm_bench}"
[ "$BASELINES" = 1 ] && STEPS="$STEPS baselines"
IMAGE_FAMILY="${IMAGE_FAMILY:-cancho-gpu-l4}"
CANCHO_REPO="${CANCHO_REPO:-https://github.com/alpibrusl/cancho}"

case "$GPU" in
  l4)   MACHINE=g2-standard-8; DEFAULT_ZONES="europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c europe-west3-a europe-west3-b europe-west2-a europe-west2-b us-central1-a us-central1-b us-central1-c us-east1-c us-east1-d us-east4-a us-east4-c us-west1-a us-west1-b us-west4-a asia-northeast1-a asia-northeast1-c asia-east1-a asia-east1-b asia-east1-c asia-northeast3-a asia-northeast3-b asia-southeast1-a asia-southeast1-b asia-southeast1-c asia-south1-a asia-south1-b asia-south1-c" ;;
  a100) MACHINE=a2-highgpu-1g; DEFAULT_ZONES="europe-west4-a europe-west4-b us-central1-a us-central1-b us-central1-c us-east1-b" ;;
  h100) MACHINE=a3-highgpu-1g; DEFAULT_ZONES="europe-west4-b europe-west4-c europe-west1-b us-central1-a us-east4-a us-east5-a" ;;
  *) echo "GPU must be l4, a100 or h100" >&2; exit 2 ;;
esac
ZONES="${ZONES:-$DEFAULT_ZONES}"
if [ "$GPU" != l4 ]; then
  cat >&2 <<NOTE
************************************************************************
* GPU=$GPU: cancho-gpu's CUDA lowering targets Ada (sm_89, the L4) ONLY.  *
* NVRTC will compile for this card's own architecture, but the tiles,   *
* the mma atoms and the schedules are Ada's. A pass here is not a Hopper *
* or Ampere result, and its speed is not what a Hopper lowering would    *
* reach. See README (Known gaps) and docs/device.md section 5.          *
************************************************************************
NOTE
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# The project, and the pause that keeps a stray gcloud default from billing
# the wrong one.
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
if [ "$PROJECT_FROM" != GCP_PROJECT ] && [ "$YES" != 1 ] && [ "$DRY_RUN" != 1 ]; then
  echo "this was not given explicitly; it will be billed for a $GPU GPU VM."
  echo "continuing in ${CONFIRM_SECS}s -- Ctrl-C to abort (YES=1 skips this)"
  sleep "$CONFIRM_SECS"
fi

STAMP="$(date -u +%Y%m%d-%H%M%S)"
NAME="cancho-gpu-${GPU}-${STAMP}"
if [ -n "${OUT_DIR:-}" ]; then OUT="$OUT_DIR"; else OUT="$ROOT/results/gcp/${STAMP}-${GPU}"; fi
mkdir -p "$OUT"
[ "$DRY_RUN" = 1 ] && echo "DRY_RUN=1: gcloud commands are printed, none is run (results dir $OUT)"

# Every gcloud call goes through here.
gc() {
  if [ "$DRY_RUN" = 1 ]; then
    echo "DRY_RUN> gcloud --project $GCP_PROJECT --quiet $*"
    return 0
  fi
  gcloud --project "$GCP_PROJECT" --quiet "$@"
}

if [ "$DRY_RUN" = 1 ]; then
  FAMILY="common-cu12-ubuntu-<newest family>"
  gc compute images list --project deeplearning-platform-release --filter='family~^common-cu12.*ubuntu'
  gc compute images describe-from-family "$IMAGE_FAMILY"
  IMG=(--image-project deeplearning-platform-release --image-family "$FAMILY")
else
  # Newest Deep Learning VM image with CUDA 12 and the NVIDIA driver installer.
  FAMILY="$(gcloud compute images list --project deeplearning-platform-release \
    --filter='family~^common-cu12.*ubuntu' --format='value(family)' | sort | tail -1)"
  [ -n "$FAMILY" ] || { echo "no common-cu12 ubuntu image family found" >&2; exit 1; }
  # The image build_image.sh makes (compiler, Rust, torch, cupy already on it).
  if gc compute images describe-from-family "$IMAGE_FAMILY" >/dev/null 2>&1; then
    IMG=(--image-project "$GCP_PROJECT" --image-family "$IMAGE_FAMILY")
    echo "image: $GCP_PROJECT/$(gc compute images describe-from-family "$IMAGE_FAMILY" --format='value(name)') (pre-built)"
  else
    IMG=(--image-project deeplearning-platform-release --image-family "$FAMILY")
    echo "image family: deeplearning-platform-release/$FAMILY (no pre-built $IMAGE_FAMILY; build_image.sh is optional)"
  fi
fi

ZONE=""
# Where the VM named $NAME is, if it exists anywhere in the project. Asked by
# name, not remembered: a request is made per *region* and Google picks the
# zone, so an interrupt between "created" and "we know the zone" would leave
# a GPU running that nothing knows to delete.
where() {
  if [ "$DRY_RUN" = 1 ]; then echo "${ZONES%% *}"; return 0; fi
  gc compute instances list --filter="name=$NAME" --format="value(zone.basename())" 2>/dev/null \
    | head -1 || true
}
cleanup() {
  local z
  z="$(where)"
  [ -n "$z" ] || return 0
  if [ "$KEEP" = 1 ]; then
    echo "KEEP=1: $NAME is still running in $z. Delete it with:"
    echo "  gcloud --project $GCP_PROJECT compute instances delete $NAME --zone $z"
    return 0
  fi
  if [ "$DRY_RUN" = 1 ]; then gc compute instances delete "$NAME" --zone "$z"; return 0; fi
  gc compute instances delete "$NAME" --zone "$z" >/dev/null 2>&1 && echo "deleted $NAME in $z"
  return 0
}
# A signal must END the script after cleaning up: a trap that merely returned
# would let the run carry on, and bash defers a trap until the foreground
# child exits, which is why the remote command below runs in the background
# and is `wait`ed for (wait is interrupted by a trapped signal).
on_signal() {
  trap - EXIT
  echo "signal $1: stopping and deleting $NAME" >&2
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

spot_flags=()
[ "$SPOT" = 1 ] && spot_flags=(--provisioning-model=SPOT)
# One request per region, not per zone: Google places the VM in whichever zone
# of the region has capacity, and never more than one VM is asked for.
REGIONS=$(for z in $ZONES; do echo "${z%-*}"; done | awk '!seen[$0]++')
for r in $REGIONS; do
  echo "trying $MACHINE in $r"
  allow=$(for z in $ZONES; do if [ "${z%-*}" = "$r" ]; then printf '%s=allow,' "$z"; fi; done)
  if gc compute instances bulk create --region "$r" --count 1 \
      --predefined-names "$NAME" \
      --location-policy "${allow%,}" \
      --machine-type "$MACHINE" \
      --maintenance-policy TERMINATE ${spot_flags[@]+"${spot_flags[@]}"} \
      --max-run-duration "$MAX_RUN" --instance-termination-action DELETE \
      "${IMG[@]}" \
      --boot-disk-size 150GB --boot-disk-type pd-ssd \
      --metadata install-nvidia-driver=True \
      --labels purpose=cancho-gpu-test 2>"$OUT/create-$r.log" >"$OUT/create-$r.out"; then
    [ "$DRY_RUN" = 1 ] && cat "$OUT/create-$r.out"
    ZONE="$(where)"
    [ -n "$ZONE" ] && break
    echo "  $r: created, but $NAME is not listed anywhere" >&2
    continue
  fi
  why=$(grep -oE "code: [A-Z_]+|currently unavailable|[Qq]uota [^.]*" "$OUT/create-$r.log" | head -1 || true)
  echo "  $r: ${why:-failed, see $OUT/create-$r.log}" >&2
done
[ -n "$ZONE" ] || { echo "no region had capacity (or quota) for $MACHINE; see $OUT/create-*.log" >&2; exit 1; }
echo "$NAME up in $ZONE"

# SSH comes up before the driver finishes installing; remote_cuda.sh waits for it.
if [ "$DRY_RUN" = 1 ]; then
  gc compute ssh "$NAME" --zone "$ZONE" --command true
else
  for _ in $(seq 1 60); do
    gc compute ssh "$NAME" --zone "$ZONE" --command true >/dev/null 2>&1 && break
    sleep 10
  done
fi

SHIP="$OUT/src.tar.gz"
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN> git -C $ROOT archive --format=tar.gz -o $SHIP HEAD"
else
  git -C "$ROOT" archive --format=tar.gz -o "$SHIP" HEAD
fi
gc compute scp --zone "$ZONE" "$SHIP" "$NAME:~/src.tar.gz"
# The compiler at the pin: from CANCHO_REPO when that is a local checkout.
if [ -d "$CANCHO_REPO" ]; then
  REV="$(sed -n 's/^cancho = "\([0-9a-f]*\)".*/\1/p' "$ROOT/cancho.toml")"
  if [ "$DRY_RUN" = 1 ]; then
    echo "DRY_RUN> git -C $CANCHO_REPO archive --format=tar.gz -o $OUT/cancho-src.tar.gz $REV"
  else
    git -C "$CANCHO_REPO" archive --format=tar.gz -o "$OUT/cancho-src.tar.gz" "$REV"
  fi
  gc compute scp --zone "$ZONE" "$OUT/cancho-src.tar.gz" "$NAME:~/cancho-src.tar.gz"
  REMOTE_REPO=local
else
  REMOTE_REPO="$CANCHO_REPO"
fi
# A job travels as a file, so its own quotes need no escaping.
if [ -n "${JOB:-}" ]; then
  printf '%s\n' "$JOB" > "$OUT/job.sh"
  gc compute scp --zone "$ZONE" "$OUT/job.sh" "$NAME:~/job.sh"
fi

# Settings the remote script reads, passed as one quoted env prefix.
fwd=""
for v in CASE_TIMEOUT JOBS SIZES DTYPES KNOWN_OPEN TORCH_INDEX BASE_MODELS CANCHO_RELEASE CANCHO_FLAVOUR; do
  if [ -n "${!v:-}" ]; then fwd="$fwd $v='${!v}'"; fi
done
remote_cmd="mkdir -p cancho-gpu && tar -xzf src.tar.gz -C cancho-gpu && STEPS='$STEPS' CANCHO_REPO='$REMOTE_REPO' JOB='${JOB:+1}' GPU='$GPU'$fwd bash cancho-gpu/scripts/gcp/remote_cuda.sh"

# A failing run must still bring its logs home: no errexit from here on.
set +e
if [ "$DRY_RUN" = 1 ]; then
  gc compute ssh "$NAME" --zone "$ZONE" --command "$remote_cmd"
  status=0
else
  rm -f "$OUT/remote.status"
  { gc compute ssh "$NAME" --zone "$ZONE" --command "$remote_cmd"; echo $? > "$OUT/remote.status"; } \
    2>&1 | tee "$OUT/remote.log" &
  wait $!
  status=$(cat "$OUT/remote.status" 2>/dev/null || echo 255)
fi
gc compute scp --zone "$ZONE" --recurse "$NAME:~/results/*" "$OUT/" || true
echo "results in $OUT (remote exit $status)"
# A Spot VM Google took back is not a failed run: it is no run, and says
# nothing about the code. Exit 75 (EX_TEMPFAIL) so a caller can retry.
if [ "$status" != 0 ] && [ -n "$(gc compute operations list \
    --filter="operationType=compute.instances.preempted AND targetLink~/$NAME\$" \
    --format='value(name)' 2>/dev/null)" ]; then
  echo "$NAME was preempted: no result, exit 75 to retry"
  exit 75
fi
exit "$status"
