#!/bin/sh
# Build `cancho-gpu-front` (docs/one-server.md) and the fake runner of its checks, against the packages its locks pin:
#
#     scripts/build-front.sh
#
#   CANCHO        the compiler (default: cancho on PATH)
#   CANCHO_DIR    a checkout of cancho, for packages/http-server's store     (default: ../cancho)
#   WEB_DIR       a checkout of cancho-web, for its store                      (default: ../cancho-web)
#   SCHEMA_DIR    a checkout of cancho-schema, for its store                   (default: ../cancho-schema)
#
# `vcs fetch` refuses a store whose sources are not the ones a lock pins (front/deps/*.lock), so a checkout at another revision is a
# refusal, not another program.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CANCHO=${CANCHO:-cancho}
CANCHO_DIR=${CANCHO_DIR:-$ROOT/../cancho}
WEB_DIR=${WEB_DIR:-$ROOT/../cancho-web}
SCHEMA_DIR=${SCHEMA_DIR:-$ROOT/../cancho-schema}
deps=$ROOT/build/front-deps
rm -rf "$deps"
mkdir -p "$deps"
"$CANCHO" vcs fetch --lock "$ROOT/front/deps/web.lock" --store "$WEB_DIR/.cancho-vcs" -o "$deps" >/dev/null
"$CANCHO" vcs fetch --lock "$ROOT/front/deps/http-server.lock" --store "$CANCHO_DIR/packages/http-server/.cancho-vcs" -o "$deps" >/dev/null
"$CANCHO" vcs fetch --lock "$ROOT/front/deps/schema.lock" --store "$SCHEMA_DIR/.cancho-vcs" -o "$deps" >/dev/null
"$CANCHO" build "$ROOT/front/front.cho" "$deps"/*.cho --std -o "$ROOT/build/cancho-gpu-front" 2>&1 | grep -v '^ld: warning' || true
"$CANCHO" build "$ROOT/front/fake_runner.cho" --std -o "$ROOT/build/fake-runner" 2>&1 | grep -v '^ld: warning' || true
[ -x "$ROOT/build/cancho-gpu-front" ] && [ -x "$ROOT/build/fake-runner" ] || { echo "build-front: a build failed" >&2; exit 1; }
echo "built $ROOT/build/cancho-gpu-front and $ROOT/build/fake-runner"
