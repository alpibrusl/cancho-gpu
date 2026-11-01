#!/bin/sh
# Build forward_opennn against a built OpenNN (CPU only is enough; see
# docs/comparison.md for the build this was measured with).
#
#     OPENNN_SRC=<checkout> OPENNN_BUILD=<its cmake build dir> bench/opennn/build.sh
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
: "${OPENNN_SRC:?the OpenNN checkout}" "${OPENNN_BUILD:?the OpenNN cmake build directory}"
mkdir -p "$ROOT/build"
g++ -std=gnu++20 -O3 -march=native -fopenmp -DEIGEN_MAX_ALIGN_BYTES=64 -DEIGEN_MPL2_ONLY -DEIGEN_NO_DEBUG \
  -DEIGEN_USE_THREADS -DOPENNN_NO_VISION -Wno-interference-size \
  -I"$OPENNN_SRC" -I"$OPENNN_BUILD/_deps/eigen-src" -I"$OPENNN_BUILD/_deps/opennn_zlib-src" -I"$OPENNN_BUILD/_deps/opennn_zlib-build" \
  "$ROOT/bench/opennn/forward_opennn.cpp" "$OPENNN_BUILD/opennn/libopennn_core.a" "$OPENNN_BUILD/_deps/opennn_zlib-build/libz.a" \
  -o "$ROOT/build/forward_opennn"
echo "built $ROOT/build/forward_opennn"
