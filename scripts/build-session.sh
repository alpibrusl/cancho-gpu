#!/bin/sh
# Build `cancho-gpu-session` (docs/resident.md): a session on the CPU interpreter.
#
#     CANCHO=<pinned cancho> scripts/build-session.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CANCHO=${CANCHO:-cancho}
mkdir -p "$ROOT/build"
# shellcheck disable=SC2046
"$CANCHO" build $(ls "$ROOT"/src/*.cho | grep -v '/main\.cho$') "$ROOT"/session/main.cho --std -o "$ROOT/build/cancho-gpu-session"
echo "built $ROOT/build/cancho-gpu-session"
