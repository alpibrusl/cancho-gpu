#!/bin/sh
# Run the slice 9b kernels on a Metal GPU and compare every output with the
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
attn_gate r=24 n=256
attn_gate r=3 n=7
matvec_fp4_batch m=1 n=128 k=256
matvec_fp4_batch m=3 n=128 k=256
matvec_fp4_batch m=8 n=128 k=1024
matvec_fp4_res_batch m=4 n=128 k=512
matvec_fp4_swiglu_batch m=4 n=128 k=512
CASES
exit $fail
