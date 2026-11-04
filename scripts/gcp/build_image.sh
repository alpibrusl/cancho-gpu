#!/usr/bin/env bash
# OPTIONAL. Build the disk image GPU runs can boot from, on a cheap Spot VM
# with no GPU.
#
#   GCP_PROJECT=my-project scripts/gcp/build_image.sh
#   DRY_RUN=1 scripts/gcp/build_image.sh        # print the gcloud commands
#
# Why this is optional, and when it earns its keep (docs/cloud.md): the cancho
# compiler is installed from a ~1 MB release binary (CANCHO_RELEASE; building
# it from source is the fallback), so what a run installs is clang and a few apt
# packages, about a minute (not measured). At Spot L4 prices that is cents per
# run, so a single run does not need an image. What does: torch + cupy (~4 GB of
# wheels) and, with WITH_BASELINE_LIBS=1, transformers/peft and the Qwen weights
# (~4 GB) fetched at the GPU's hourly rate on every run, and a loop of runs
# while iterating on the scripts. The image holds the toolchain, the compiler
# at the cancho.toml pin (remote_cuda.sh rebuilds it when the pin moves), the
# venv and, optionally, the weights. It costs storage every month it exists.
#
# Knobs: GCP_PROJECT (or gcloud's default, printed, 10 s pause unless YES=1),
# MACHINE (e2-standard-16), ZONES, IMAGE_FAMILY (cancho-gpu-l4), KEEP_IMAGES
# (1), WITH_BASELINE_LIBS=1, CANCHO_REPO (as in cuda_test.sh), DRY_RUN, KEEP.
#
# The builder is Spot, labelled purpose=cancho-gpu-image-build, deleted on exit
# (and on SIGINT/SIGTERM/SIGHUP) and by GCE after 2 hours. The GPU driver is
# NOT installed here (no GPU): the stock Deep Learning image the GPU VM would
# boot already carries a driver for its kernel, which this keeps by holding the
# kernel packages and checks at the end, as lex-gpu's prepare_image.sh does.
set -euo pipefail
MACHINE="${MACHINE:-e2-standard-16}"
ZONES="${ZONES:-europe-west4-a europe-west4-b europe-west1-b europe-west1-c us-central1-a us-central1-b us-east1-b}"
IMAGE_FAMILY="${IMAGE_FAMILY:-cancho-gpu-l4}"
KEEP_IMAGES="${KEEP_IMAGES:-1}"
DRY_RUN="${DRY_RUN:-0}"
YES="${YES:-0}"
KEEP="${KEEP:-0}"
CONFIRM_SECS="${CONFIRM_SECS:-10}"
WITH_BASELINE_LIBS="${WITH_BASELINE_LIBS:-0}"
CANCHO_REPO="${CANCHO_REPO:-https://github.com/alpibrusl/cancho}"

if [ -n "${GCP_PROJECT:-}" ]; then FROM_ENV=1
elif [ "$DRY_RUN" = 1 ]; then GCP_PROJECT=DRY-RUN-PROJECT; FROM_ENV=1
else GCP_PROJECT="$(gcloud config get-value project 2>/dev/null || true)"; FROM_ENV=0; fi
if [ -z "$GCP_PROJECT" ] || [ "$GCP_PROJECT" = "(unset)" ]; then
  echo "set GCP_PROJECT (or gcloud config set project): billing goes to it" >&2; exit 2
fi
echo "billing project: $GCP_PROJECT"
if [ "$FROM_ENV" = 0 ] && [ "$YES" != 1 ]; then
  echo "taken from gcloud's default, not given explicitly; continuing in ${CONFIRM_SECS}s -- Ctrl-C to abort (YES=1 skips)"
  sleep "$CONFIRM_SECS"
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
NAME="cancho-image-builder-${STAMP}"
IMAGE="${IMAGE_FAMILY}-${STAMP}"
OUT="${OUT_DIR:-$ROOT/results/gcp/${STAMP}-image}"
mkdir -p "$OUT"
gc() {
  if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN> gcloud --project $GCP_PROJECT --quiet $*"; return 0; fi
  gcloud --project "$GCP_PROJECT" --quiet "$@"
}

if [ "$DRY_RUN" = 1 ]; then
  BASE="common-cu12-ubuntu-<newest family>"
  gc compute images list --project deeplearning-platform-release --filter='family~^common-cu12.*ubuntu'
else
  BASE="$(gcloud compute images list --project deeplearning-platform-release \
    --filter='family~^common-cu12.*ubuntu' --format='value(family)' | sort | tail -1)"
  [ -n "$BASE" ] || { echo "no common-cu12 ubuntu image family found" >&2; exit 1; }
fi
echo "base: deeplearning-platform-release/$BASE"

ZONE=""
where() {
  if [ "$DRY_RUN" = 1 ]; then echo "${ZONES%% *}"; return 0; fi
  gc compute instances list --filter="name=$NAME" --format="value(zone.basename())" 2>/dev/null | head -1 || true
}
cleanup() {
  local z
  z="$(where)"
  [ -n "$z" ] || return 0
  if [ "$KEEP" = 1 ]; then echo "KEEP=1: $NAME still runs in $z"; return 0; fi
  if [ "$DRY_RUN" = 1 ]; then gc compute instances delete "$NAME" --zone "$z"; return 0; fi
  gc compute instances delete "$NAME" --zone "$z" >/dev/null 2>&1 && echo "deleted $NAME in $z"
  return 0
}
on_signal() { trap - EXIT; echo "signal $1: deleting $NAME" >&2; pkill -P $$ 2>/dev/null || true; cleanup; exit "$2"; }
trap cleanup EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

for z in $ZONES; do
  echo "trying $MACHINE in $z"
  if gc compute instances create "$NAME" --zone "$z" \
      --machine-type "$MACHINE" \
      --provisioning-model=SPOT --instance-termination-action DELETE --max-run-duration 2h \
      --image-project deeplearning-platform-release --image-family "$BASE" \
      --boot-disk-size 150GB --boot-disk-type pd-balanced \
      --labels purpose=cancho-gpu-image-build >"$OUT/create-$z.out" 2>"$OUT/create-$z.log"; then
    [ "$DRY_RUN" = 1 ] && cat "$OUT/create-$z.out"
    ZONE="$z"
    break
  fi
  why=$(grep -oE "code: [A-Z_]+|[A-Z_]+EXHAUSTED[A-Z_]*|[Qq]uota [^.]*" "$OUT/create-$z.log" | head -1 || true)
  echo "  $z: ${why:-failed, see $OUT/create-$z.log}"
done
[ -n "$ZONE" ] || { echo "no zone had a $MACHINE" >&2; exit 1; }
echo "$NAME up in $ZONE"

if [ "$DRY_RUN" != 1 ]; then
  for _ in $(seq 1 60); do gc compute ssh "$NAME" --zone "$ZONE" --command true >/dev/null 2>&1 && break; sleep 10; done
fi
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN> git -C $ROOT archive --format=tar.gz -o $OUT/src.tar.gz HEAD"
else
  git -C "$ROOT" archive --format=tar.gz -o "$OUT/src.tar.gz" HEAD
fi
gc compute scp --zone "$ZONE" "$OUT/src.tar.gz" "$NAME:~/src.tar.gz"
if [ -d "$CANCHO_REPO" ]; then
  REV="$(sed -n 's/^cancho = "\([0-9a-f]*\)".*/\1/p' "$ROOT/cancho.toml")"
  if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN> git -C $CANCHO_REPO archive ... $REV"
  else git -C "$CANCHO_REPO" archive --format=tar.gz -o "$OUT/cancho-src.tar.gz" "$REV"; fi
  gc compute scp --zone "$ZONE" "$OUT/cancho-src.tar.gz" "$NAME:~/cancho-src.tar.gz"
  REMOTE_REPO=local
else
  REMOTE_REPO="$CANCHO_REPO"
fi
# `prepare` leaves ~/cancho-src.tar.gz and the source behind; a run brings its own source.
gc compute ssh "$NAME" --zone "$ZONE" --command \
  "mkdir -p cancho-gpu && tar -xzf src.tar.gz -C cancho-gpu && CANCHO_REPO='$REMOTE_REPO' CANCHO_RELEASE='${CANCHO_RELEASE:-latest}' WITH_BASELINE_LIBS='$WITH_BASELINE_LIBS' bash cancho-gpu/scripts/gcp/remote_cuda.sh prepare" \
  2>&1 | tee "$OUT/prepare.log"
if [ "$DRY_RUN" != 1 ]; then
  grep -q "image contents ready" "$OUT/prepare.log" || { echo "preparation did not finish; no image made" >&2; exit 1; }
fi
# Clean the disk, keep the boot kernel the driver was built for (as lex-gpu's prepare_image.sh).
gc compute ssh "$NAME" --zone "$ZONE" --command \
  'set -e; sudo systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer >/dev/null 2>&1 || true; sudo apt-mark hold linux-image-gcp linux-gcp linux-headers-gcp >/dev/null 2>&1 || true; boot=$(ls /boot/vmlinuz-* | sed "s|^/boot/vmlinuz-||" | sort -V | tail -1); find "/lib/modules/$boot" -name "nvidia.ko*" | grep -q . || { echo "kernel $boot has no NVIDIA module; not an image a GPU can use"; exit 1; }; rm -rf ~/cancho-gpu ~/src.tar.gz ~/results; sudo apt-get clean; sync; echo "boot kernel $boot has the NVIDIA module"'

# An image of a running disk can be inconsistent; stop the builder first.
gc compute instances stop "$NAME" --zone "$ZONE" >/dev/null
gc compute images create "$IMAGE" --source-disk "$NAME" --source-disk-zone "$ZONE" \
  --family "$IMAGE_FAMILY" --labels purpose=cancho-gpu-test >/dev/null
echo "image $IMAGE (family $IMAGE_FAMILY)"

# Keep the newest few; images cost storage every month they exist.
if [ "$DRY_RUN" = 1 ]; then
  gc compute images list --filter="family=$IMAGE_FAMILY" --sort-by=~creationTimestamp --format="value(name)"
else
  gc compute images list --filter="family=$IMAGE_FAMILY" \
    --sort-by=~creationTimestamp --format="value(name)" | tail -n +"$((KEEP_IMAGES + 1))" |
    while read -r old; do gc compute images delete "$old" >/dev/null && echo "deleted old image $old"; done
fi
