#!/bin/sh
# Build OpenNN's own `dense` benchmark driver (benchmarks/families/dense.cpp) on macOS, CPU only, with Apple clang and Homebrew's libomp, by compiling its sources directly:
# no cmake, and nothing in the checkout is modified (a cmake build of the checkout this was measured with, 2026-11, did not configure on macOS).
#
#     bench/opennn/build_dense.sh WORKDIR        -> WORKDIR/dense_opennn
#
# Needs git, clang++ and `brew install libomp`; fetches OpenNN and Eigen 5.0.1.
set -eu
W=$(mkdir -p "${1:?a work directory}" && cd "$1" && pwd)
mkdir -p "$W/obj"
[ -d "$W/opennn-src" ] || git clone --depth 1 https://github.com/Artelnics/opennn.git "$W/opennn-src"
[ -d "$W/eigen" ] || git clone --depth 1 --branch 5.0.1 https://gitlab.com/libeigen/eigen.git "$W/eigen"
cd "$W/opennn-src"
OMP=$(brew --prefix libomp)
FLAGS="-std=c++20 -O3 -Xpreprocessor -fopenmp -I$OMP/include -DEIGEN_MAX_ALIGN_BYTES=64 -DEIGEN_MPL2_ONLY -DEIGEN_NO_DEBUG -DEIGEN_USE_THREADS -DOPENNN_NO_VISION -w -I. -I$W/eigen -include opennn/core/pch.h"
find opennn benchmarks/families/dense.cpp -name '*.cpp' | grep -v '/cuda/\|image_processing\|image_dataset\|yolo' > "$W/sources.txt"
xargs -n1 -P 8 sh -c 'o='"$W"'/obj/$(echo "$0" | tr / _).o; [ -f "$o" ] || clang++ '"$FLAGS"' -c "$0" -o "$o"' < "$W/sources.txt"
clang++ -O3 "$W"/obj/*.o -o "$W/dense_opennn" -L"$OMP/lib" -lomp -lz
echo "built $W/dense_opennn"
