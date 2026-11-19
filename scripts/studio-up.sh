#!/bin/sh
# Start the studio: its server (the page and the API) and its worker (the GPU jobs), and print the address with the token.
#
#     scripts/studio-up.sh [PORT] [DATA_DIR]          (default: 8090 and build/studio-data; Ctrl-C stops both)
#
# Needs the pinned cancho (CANCHO), cancho-ui (UI) and the cancho packages (PK), as studio/check.sh does; builds what is missing into build/.
# `--rebuild` as the last argument builds the server, the worker and the page again. A job needs the Metal session (`scripts/build-metal-session.sh`)
# and the weights `scripts/media doctor` lists; the page and the API work without them (a job then fails with its log).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CANCHO=${CANCHO:-cancho}
UI=${UI:-cancho-ui}
PK=${PK:-$ROOT/../cancho/packages}
REBUILD=0
for a in "$@"; do [ "$a" = "--rebuild" ] && REBUILD=1; done
PORT=${1:-8090}
case "$PORT" in --rebuild) PORT=8090;; esac
DATA=${2:-$ROOT/build/studio-data}
case "$DATA" in --rebuild) DATA=$ROOT/build/studio-data;; esac
case "$DATA" in /*) ;; *) echo "studio-up: the data directory must be an absolute path: $DATA" >&2; exit 2;; esac
SITE=$ROOT/build/studio-site
cd "$ROOT"
mkdir -p build "$DATA" "$SITE"
if [ "$REBUILD" = 1 ] || [ ! -x build/studio ]; then
  "$CANCHO" build studio/studio.cho studio/files.cho studio/jobs.cho studio/results.cho studio/auth.cho studio/range.cho studio/upload.cho studio/compare.cho studio/sets.cho studio/cases.cho studio/gallery.cho studio/sessions.cho studio/library.cho studio/picks.cho "$PK"/http-server/server.cho --std -o build/studio 2>&1 | grep -v "^ld: warning" || true
fi
if [ "$REBUILD" = 1 ] || [ ! -x build/studio_worker ]; then
  "$CANCHO" build studio/worker.cho studio/files.cho studio/rewrite_step.cho studio/rewrite.cho --std -o build/studio_worker 2>&1 | grep -v "^ld: warning" || true
fi
if [ "$REBUILD" = 1 ] || [ ! -f "$SITE/index.html" ]; then
  "$UI" build studio/ui/studio.cui --page -o "$SITE" --name cancho-studio >/dev/null
fi
[ -x build/studio ] && [ -x build/studio_worker ] || { echo "studio-up: the server or the worker did not build (is CANCHO the pinned cancho and PK the packages directory?)" >&2; exit 1; }
LOG=$DATA/server.log
build/studio "$PORT" "$DATA" "$SITE" > "$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT INT TERM
n=0
until grep -q "open http" "$LOG" 2>/dev/null; do
  n=$((n + 1)); [ $n -gt 100 ] && { echo "studio-up: the server did not start:" >&2; cat "$LOG" >&2; exit 1; }
  kill -0 $SERVER 2>/dev/null || { echo "studio-up: the server stopped:" >&2; cat "$LOG" >&2; exit 1; }
  sleep 0.1
done
echo "studio: $(grep 'open http' "$LOG" | sed 's/^open //')"
echo "data:   $DATA   (jobs, results and the log of the server)"
echo "worker: running; jobs start when you make them in the page. Ctrl-C stops the studio."
build/studio_worker "$ROOT/scripts/media" "$HOME" "$DATA" --every 5
