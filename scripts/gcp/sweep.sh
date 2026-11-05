#!/usr/bin/env bash
# Find, and optionally delete, anything this project's GPU tests left running.
#
#   scripts/gcp/sweep.sh                  # list only; exit 1 if anything is left
#   DELETE=1 scripts/gcp/sweep.sh         # delete what it lists
#   DRY_RUN=1 scripts/gcp/sweep.sh        # print the gcloud commands, run none
#
#   GCP_PROJECT  the project (default: `gcloud config get-value project`;
#                printed either way)
#
# cuda_test.sh deletes its VM on exit and asks GCE to delete it again after
# MAX_RUN, so this should find nothing. It exists because "should" is not a
# billing guarantee: a SIGKILL, a laptop that sleeps mid-run or a crashed
# gcloud skip the first guard, and the second is up to MAX_RUN of a GPU you
# are not using.
#
# Matches the labels purpose=cancho-gpu-test (GPU runs), purpose=cancho-gpu-w1
# (w1_cloud.sh's VMs) and purpose=cancho-gpu-image-build (build_image.sh's builder), so it never
# touches anything else in the project. Images are listed, never deleted:
# they cost storage, not compute, and build_image.sh prunes its own.
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
if [ -n "${GCP_PROJECT:-}" ]; then
  PROJECT="$GCP_PROJECT"
elif [ "$DRY_RUN" = 1 ]; then
  PROJECT="DRY-RUN-PROJECT"
else
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
  echo "set GCP_PROJECT (or gcloud config set project)" >&2; exit 2
fi
echo "project: $PROJECT"

gc() {
  if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN> gcloud --project $PROJECT --quiet $*"; return 0; fi
  gcloud --project "$PROJECT" --quiet "$@"
}

FOUND="$(gc compute instances list \
  --filter="labels.purpose=cancho-gpu-test OR labels.purpose=cancho-gpu-w1 OR labels.purpose=cancho-gpu-image-build" \
  --format="value(name,zone.basename(),status)")"
if [ "$DRY_RUN" = 1 ]; then echo "$FOUND"; FOUND=""; fi

IMAGES="$(gc compute images list --no-standard-images \
  --filter="labels.purpose=cancho-gpu-test" --format="value(name,family,diskSizeGb)" 2>/dev/null || true)"
if [ "$DRY_RUN" = 1 ]; then echo "$IMAGES"; IMAGES=""; fi
if [ -n "$IMAGES" ]; then
  echo "images (storage cost only; kept):"
  echo "$IMAGES" | sed 's/^/  /'
fi

if [ -z "$FOUND" ]; then
  echo "$PROJECT: nothing running with purpose=cancho-gpu-test or cancho-gpu-w1"
  exit 0
fi

echo "$PROJECT: left over:"
echo "$FOUND" | sed 's/^/  /'

if [ "${DELETE:-0}" != 1 ]; then
  echo
  echo "these are still billing. delete them with:"
  echo "  DELETE=1 GCP_PROJECT=$PROJECT $0"
  exit 1
fi

echo "$FOUND" | while read -r name zone _; do
  [ -n "$name" ] || continue
  echo "deleting $name in $zone"
  gc compute instances delete "$name" --zone "$zone"
done
