#!/bin/sh
# Build `build/kolibri_gpu`, the host side of Kolibri 1.1's cached step (tests/kolibri_gpu.cho, docs/kolibri.md section 12), with this repository's GGUF reader: CANCHO=<pinned cancho> scripts/build-kolibri.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p build
C="${CANCHO:-cancho}"
"$C" build tests/kolibri_gpu.cho src/gguf.cho src/fileio.cho src/f32.cho src/mem.cho src/text.cho src/err.cho --std -o build/kolibri_gpu 2>&1 | grep -v "^ld: warning" || true
test -x build/kolibri_gpu
