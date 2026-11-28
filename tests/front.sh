#!/bin/sh
# The gate's stage for cancho-gpu-front (docs/one-server.md 7): front/check.sh, 24 checks against fake runners, no model, no GPU.
#
#     CANCHO=<pinned cancho> tests/front.sh
#
# It needs checkouts of cancho (packages/http-server), cancho-web and cancho-schema at the revisions of front/deps/revs (CI checks
# them out; locally the defaults are this checkout's siblings, as for scripts/build-front.sh). Without them it says so and passes,
# unless FRONT_REQUIRED=1 (CI sets it), when a missing checkout fails the gate.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WEB_DIR=${WEB_DIR:-$ROOT/../cancho-web}
SCHEMA_DIR=${SCHEMA_DIR:-$ROOT/../cancho-schema}
CANCHO_DIR=${CANCHO_DIR:-$ROOT/../cancho}
for d in "$CANCHO_DIR/packages/http-server/.cancho-vcs" "$WEB_DIR/.cancho-vcs" "$SCHEMA_DIR/.cancho-vcs"; do
  if [ ! -d "$d" ]; then
    if [ "${FRONT_REQUIRED:-0}" = 1 ]; then echo "front: no store at $d (FRONT_REQUIRED=1)"; exit 1; fi
    echo "front: skipped, no store at $d"
    exit 0
  fi
done
export CANCHO CANCHO_DIR WEB_DIR SCHEMA_DIR
out=$("$ROOT/front/check.sh" 2>&1)
status=$?
printf '%s\n' "$out" | grep -v '^ok ' || true
[ $status -eq 0 ] && echo "front: $(printf '%s\n' "$out" | grep -c '^ok ') checks of cancho-gpu-front pass"
exit $status
