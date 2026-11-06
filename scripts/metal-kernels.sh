#!/bin/sh
# Run the slice 9b kernels (and, at 0.8B's shape, the long-attention kernels of docs/weight-space.md 14.5) on a Metal GPU and compare every output with the
# interpreter's (docs/kernels.md section 5). Needs an Apple GPU, so it is not
# in the gate.
#
#     CANCHO=<pinned cancho> scripts/metal-kernels.sh
#
# Fails if any output differs from the interpreter by more than 1e-5 of its
# peak (the largest seen is 2.4e-6).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
"$ROOT/scripts/build-metal-device.sh" >/dev/null
BIN=$ROOT/build/cancho-gpu-metal-device
fail=0
while read -r kernel rest; do
  # shellcheck disable=SC2086
  "$BIN" run "$ROOT/tests/chx/$kernel.chx" $rest > "${TMPDIR:-/tmp}/metal-kernels.out" 2>&1 || { echo "$kernel $rest: failed: $(tail -1 "${TMPDIR:-/tmp}/metal-kernels.out")"; fail=1; continue; }
  worst=$(grep '^compare:' "${TMPDIR:-/tmp}/metal-kernels.out" | awk '{d=$(NF-3); p=$NF; r=(p>0)?d/p:d; if (r>m) m=r} END {printf "%.2e", m}')
  ok=$(echo "$worst" | awk '{print ($1 <= 1e-5) ? "ok" : "TOO FAR"}')
  echo "$ok  $kernel $rest: worst relative difference $worst"
  [ "$ok" = ok ] || fail=1
done <<'CASES'
rope r=24 h=32 p=192
rope r=3 h=4 p=2
kv_append nkv=4 tmax=64 d=256 pos=17
attention nkv=2 g=6 t=64 d=256 tmax=96
attention nkv=3 g=2 t=16 d=256 tmax=20
attention nkv=4 g=6 t=1364 d=256 tmax=2048
attention_scores nkv=2 g=4 t=8704 ts=64 d=256 tmax=8704
attention_stats nkv=2 g=4 t=2048 cs=64
attention_stats nkv=2 g=4 t=8704 cs=64
attention_pvl nkv=2 g=4 t=2048 cs=64 dw=32 d=256 tmax=2048
attention_pvl nkv=2 g=4 t=8704 cs=64 dw=32 d=256 tmax=8704
attn_gate r=24 n=256
attn_gate r=3 n=7
matvec_fp4_batch m=1 n=128 k=256
matvec_fp4_batch m=3 n=128 k=256
matvec_fp4_batch m=8 n=128 k=1024
matvec_fp4_res_batch m=4 n=128 k=512
matvec_fp4_swiglu_batch m=4 n=128 k=512
matvec_rows_f16_frag m=1 n=256 k=64
matvec_rows_f16_frag m=5 n=256 k=512
matvec_rows_f16_frag m=13 n=256 k=1024
matvec_rows_f16_frag m=32 n=1536 k=1536
matvec_rows_f16_frag m=32 n=256 k=8960
matvec_rows_f16h m=1 n=256 k=64
matvec_rows_f16h m=5 n=256 k=512
matvec_rows_f16h m=13 n=256 k=1024
matvec_rows_f16h m=32 n=1536 k=1536
matvec_rows_f16h m=32 n=256 k=8960
samp_max r=2 nq=8 G=4
samp_gmax r=2 nq=8
samp_sum r=2 nq=8 G=4 invt=1.25
samp_cum r=2 nq=48 bn=24
samp_pick r=2 nq=8
samp_local r=2 nq=8 G=4 invt=1.25
samp_tok r=2 nq=8
tok_acc r=4 s=16
attention_scores_b nb=2 nkv=2 g=6 t=64 ts=16 d=128 tmax=80
attention_pv_b nb=2 nkv=2 g=6 t=64 dw=16 d=128 tmax=80
attention_scores_b nb=4 nkv=2 g=6 t=1152 ts=64 d=128 tmax=1152
attention_pv_b nb=4 nkv=2 g=6 t=1152 dw=16 d=128 tmax=1152
dec_post r=4
kv_write_b nb=2 nkv=2 tmax=64 ts=16 d=128
ids_mix r=8
CASES
exit $fail
