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
  # a case written `f16:name` has an f16 output: a one-bit difference of the f32 result can flip an f16 rounding (half a step is 4.9e-4 of the value), so its bound is 5e-4
  tol=1e-5
  case "$kernel" in f16:*) kernel=${kernel#f16:}; tol=5e-4 ;; esac
  # `f16x:name`: an f16 output of values of every size (not a product's sums): a flipped rounding is a whole step, 9.8e-4 of a value just above a power of two
  case "$kernel" in f16x:*) kernel=${kernel#f16x:}; tol=1e-3 ;; esac
  # shellcheck disable=SC2086
  "$BIN" run "$ROOT/tests/chx/$kernel.chx" $rest > "${TMPDIR:-/tmp}/metal-kernels.out" 2>&1 || { echo "$kernel $rest: failed: $(tail -1 "${TMPDIR:-/tmp}/metal-kernels.out")"; fail=1; continue; }
  worst=$(grep '^compare:' "${TMPDIR:-/tmp}/metal-kernels.out" | awk '{d=$(NF-3); p=$NF; r=(p>0)?d/p:d; if (r>m) m=r} END {printf "%.2e", m}')
  ok=$(echo "$worst" | awk -v t="$tol" '{print ($1 <= t) ? "ok" : "TOO FAR"}')
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
matvec_fp4_batch m=13 n=128 k=1024
matvec_fp4_batch m=16 n=128 k=1024
matvec_fp4_batch m=16 n=64 k=1280
matvec_fp4_batch m=32 n=64 k=384
matvec_fp4_batch m=24 n=96 k=2048
matvec_fp4_batch m=5 n=32 k=160
matvec_fp4_res_batch m=4 n=128 k=512
matvec_fp4_res_batch m=16 n=128 k=2176
matvec_fp4_swiglu_batch m=16 n=64 k=1280
matvec_fp4_swiglu_batch m=4 n=128 k=512
matvec_rows_f16_m m=16 n=48 k=5120
matvec_rows_f16_m m=8 n=48 k=5120
matvec_rows_f16_m m=3 n=48 k=5120
matvec_fp4_batch_h m=16 n=128 k=1024
matvec_fp4_batch_h m=8 n=64 k=1280
matvec_fp4_res_batch_h m=16 n=128 k=2176
f16x:matvec_fp4_swiglu_batch_h m=16 n=64 k=1280
matvec_fp4_pass_h m=48 n=128 k=1024 bm=16
matvec_fp4_res_pass_h m=32 n=128 k=2176 bm=16
f16x:matvec_fp4_swiglu_pass_h m=32 n=64 k=1280 bm=16
rows3 c=10240
rope_rows m=128 r=24 h=32 p=192
rope_rows m=5 r=3 h=4 p=2
kv_rows m=128 nkv=4 tmax=1024 d=256
kv_rows m=5 nkv=2 tmax=8 d=4
matvec_fp4_few m=3 n=64 k=128
matvec_fp4_few m=2 n=64 k=160
matvec_fp4_few m=3 n=256 k=1024
matvec_fp4_res_few m=2 n=64 k=128
matvec_fp4_swiglu_few m=3 n=64 k=128
batch_fp4 b=4 n=64 k=512
matvec_rows_f16_mh m=16 n=48 k=5120
matvec_rows_f16_mh m=8 n=48 k=5120
f16x:rmsnorm_rows_h r=16 n=5120 eps=0.000001
f16x:attn_gate_q_h r=384 n=256
f16x:gated_norm_h h=768 n=128 eps=0.000001
delta_step16 nk=16
delta_step16 nk=2
delta_step8 nk=16
delta_step8 nk=2
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
attention_scores_rows nr=3 nkv=2 g=6 t=64 ts=16 d=256 tmax=96
attention_pv_rows nr=3 nkv=2 g=6 t=64 dw=32 d=256 tmax=96
attention_scores_rows nr=8 nkv=4 g=6 t=256 ts=32 d=256 tmax=1024
attention_pv_rows nr=8 nkv=4 g=6 t=256 dw=64 d=256 tmax=1024
dec_post r=4
kv_write_b nb=2 nkv=2 tmax=64 ts=16 d=128
ids_mix r=8
kv_import nsrc=2 nkv=2 g=6 t=64 ts=4 d=128
dit_scores nh=4 nq=256 t=256 qs=32 ts=64 d=128
dit_stats nh=4 nq=256 t=256 qs=32 cs=64
dit_pv nh=4 nq=256 t=256 qs=32 cs=64 dw=32 d=128
f16:dit_lnmod s=12 n=256
dit_rope s=6 nh=3 h=64
dit_gate_res s=12 n=256
f16:dit_wan_lnmod s=12 n=256
dit_wan_gate_res s=12 n=256
f16:dit_swiglu s=6 n=384 bn=128
f16:dit_geglu s=6 n=384 bn=128
dit_scores_bias nh=3 nq=32 t=32 qs=8 ts=16 d=64
f16:dit_gelu s=4 n=256 bn=128
f16:dit_silu s=4 n=256 bn=128
dit_add s=4 n=256 bn=128
dit_modsplit m=64 n=256 bn=128
dit_rows2 m=64 n=256 bn=128
dit_euler s=12 n=64
f16:vae_norm_silu r=64 c=64 g=9 sc=8.0
f16:dit_cat s=4 n=256 bn=128
f16:vit_ln s=4 n=256
f16:vit_bias_gelu s=4 n=256 bn=128
f16:vit_bias_gelu_erf s=4 n=256 bn=128 m1=-1
conv_bias_gelu_erf s=8 n=64 bn=32 m1=-1
row_mix n=256 bn=128
kv_export s=4 nkv=2 tmax=8
sin_scaled r=64 n=1024
conv1d_f s=64 cin=512 cout=1024 k=3 dl=1
conv1d_f s=256 cin=1024 cout=1536 k=7 dl=1
conv1d_f s=2048 cin=768 cout=1920 k=2 dl=1
conv1d_f s=2432 cin=96 cout=96 k=7 dl=9
dwconv_f s=128 c=1024 k=7 bm=8 bn=256
snake s=64 c=1536 bn=256 g=1
snake s=4 c=16 bn=8 g=3
bias_pad s=512 c=1024 g=6 bn=256
dit_swiglu_f s=128 n=1024 bn=256
wave s=4096 c=32 bm=64 lim=0.25 m1=-1
dec_post_n r=2
amax_cand r=2 nq=4 G=2
amax_id r=3 nq=4 v1=3
samp_pen nq=48 G=2 invt=1.25 rp=1.5 irp=0.6666666666666666
samp_rank nq=48
samp_keep n=3072 k=0.05 bm=256 lo=4
seen_or c=3072 bc=256
dq_q4 n=512 k=2560 bn=8 bk=64
matvec_q4_rows n=512 k=2560
matvec_q4_rows n=2560 k=512
moe_rank nq=6
moe_pick n=384 k=6
axpy_slot n=2560 bn=256 k=6 j=5
group_sum k=3584 G=32
dec_post_g r=1 nq=48 nkv=4
fp4_round r=4 n=128
f16:conv_cols t=5 f=3 c=32 wp=8 r=40
vae_bias r=64 c=64
vae_bias_res r=64 c=64
f16:vae_pad16 r=64 c=64 g=9
f16:vae_up2 h=6 w=6 c=32 r=64 r2=256 g2=15
f16:vae_s2d h=6 w=6 c=32 r=64 r2=64 g2=6
f16:vae_avg2 h=6 w=6 c=32 r=64 r2=64 g2=6
vae_up2_epi h=6 w=6 c=32 r=64 r2=256
conv1 r=64 cin=64 cout=48 g=9
conv3_k16 rp=64 cin=48 cout=48 wp=8 g=9
f16:vae_split16 r=64 c=64 sc=0.125
vae_mask_add n=64
f16:vae_norm r=64 c=64 g=9 sc=8.0
f16:dit_rope16 s=128 nh=2 h=64 sc=0.08838834764831845
dit_scores_mma nh=2 nq=128 t=128 d=128
dit_stats_m nh=2 nq=128 t=128 qs=32 cs=64
dit_stats_c nh=2 nq=128 t=192 qs=16 cs=64
dit_stats_comb nh=2 nq=128 nc=3 qs=32
dit_stats_o nh=2 nq=128 t=192 qs=16 cs=64
spill_relu nh=2 nq=128 t=128 d=128
f16:dit_attn_fused nh=2 nq=128 t=128 d=128
dit_pv_mma nh=2 nq=128 t=128 d=128
gemm_mma_f m=64 n=256 k=96
dit_wan_lnmod_f s=12 n=256
vit_ln_f s=4 n=256
vit_bias_gelu_f s=4 n=256 bn=128
dit_rope16_f s=128 nh=2 h=64 sc=0.08838834764831845
dit_scores_mma_f nh=2 nq=128 t=128 d=128
dit_pv_mma_f nh=2 nq=128 t=128 d=128
CASES
exit $fail
