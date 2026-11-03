#!/bin/sh
# The probes of docs/resident.md section 25.2: what the .chx surface does with 4-bit data. Each file is checked on both targets; the first five are expected to be refused (the tag is printed).
#     experimental/m75/probes.sh [build/cancho-gpu]
BIN=${1:-build/cancho-gpu}
D=$(dirname "$0")
for f in i8_stage_f32 i8_arithmetic stage_i8 i8_store_f32 f32_store_i8; do
    printf '%-18s ' "$f"; "$BIN" check "$D/$f.chx" n=64 2>&1 | head -1
done
for f in dequant_to_f16:"n=256 k=256 bn=64 bk=64" gemm_mma_nn_fp4:"m=128 n=128 k=128"; do
    printf '%-18s ' "${f%%:*}"; "$BIN" check "$D/${f%%:*}.chx" ${f#*:} 2>&1 | tr '\n' ' '; echo
done
