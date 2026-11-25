#!/bin/sh
# Model files (docs/resident.md section 31, issue #151): `cancho-gpu model-expand` expands the directives of a model file (const, text, block, use, repeat, if, else, end, `$name`, `${expression}`) into the plain session file.
#
#     tests/model.sh [path to cancho-gpu-session]
#
#  1. every tests/model/ok/X.model (with the definitions in X.args, if there) expands to X.out, byte for byte;
#  2. every tests/model/reject/X.model is refused with the rule `session`, exit 1, nothing on standard output, and the message in X.err;
#  3. the five real models of models/ (the text encoder, the transformer, the vision tower's four sessions, the VAE's encoder and decoder) expand to files of the recorded SHA-256 (golden/model/): the expansions that tests/text_img_gpu.py,
#     tests/dit_edit_lib.py, tests/vit_gpu.py, tests/vae_enc_gpu.py and tests/vae_gpu.py write, compared byte for byte when they were recorded (docs/resident.md section 31);
#  4. with a CPU session binary: a model file given to it (with its definitions on the command line) answers what its expansion answers.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${CANCHO_GPU:-$ROOT/build/cancho-gpu}
SESSION=${1:-}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
fail=0
n=0
cd "$ROOT"
for f in tests/model/ok/*.model; do
  b=${f%.model}; args=""
  [ -f "$b.args" ] && args=$(cat "$b.args")
  # shellcheck disable=SC2086
  "$BIN" model-expand "$f" $args > "$WORK/out" 2> "$WORK/err" || { echo "$f: refused: $(head -1 "$WORK/err")"; fail=1; continue; }
  cmp -s "$WORK/out" "$b.out" || { echo "$f: the expansion differs from $b.out"; diff "$WORK/out" "$b.out" | head -5; fail=1; }
  n=$((n + 1))
done
for f in tests/model/reject/*.model; do
  b=${f%.model}
  status=0
  "$BIN" model-expand "$f" > "$WORK/out" 2> "$WORK/err" || status=$?
  [ "$status" -eq 1 ] || { echo "$f: exit $status, expected 1"; fail=1; }
  [ ! -s "$WORK/out" ] || { echo "$f: refused but wrote output"; fail=1; }
  grep -q '^error\[session\]' "$WORK/err" || { echo "$f: expected error[session], got: $(head -1 "$WORK/err")"; fail=1; }
  [ "$(head -1 "$WORK/err")" = "$(head -1 "$b.err")" ] || { echo "$f: the message changed: $(head -1 "$WORK/err")"; fail=1; }
  n=$((n + 1))
done
check() {
  name=$1; shift
  got=$("$BIN" model-expand "$ROOT/models/$name.model" "$@" | shasum -a 256 | cut -d' ' -f1)
  want=$(cat "$ROOT/golden/model/$name.$tag.sha256")
  [ "$got" = "$want" ] || { echo "models/$name.model ($tag): expansion $got, recorded $want"; fail=1; }
  n=$((n + 1))
}
tag=ref;     check qwen3vl_text CHX=/chx WORK=/w SP=1088 NOREF=0
tag=noref;   check qwen3vl_text CHX=/chx WORK=/w SP=1088 NOREF=1
tag=step;    check qwen_image_dit CHX=/chx WORK=/w WDIR=/wd NP=4160
tag=s1;      check qwen3vl_vision CHX=/chx WORK=/w NP=4352 NAME=s1 LO=0 HI=9 FIRST=1 FINAL=0
tag=s2;      check qwen3vl_vision CHX=/chx WORK=/w NP=4352 NAME=s2 LO=9 HI=17 FIRST=0 FINAL=0
tag=s3;      check qwen3vl_vision CHX=/chx WORK=/w NP=4352 NAME=s3 LO=17 HI=25 FIRST=0 FINAL=0
tag=s4;      check qwen3vl_vision CHX=/chx WORK=/w NP=4352 NAME=s4 LO=25 HI=27 FIRST=0 FINAL=1
tag=enc;     check qwen_image_vae_encoder CHX=/chx CONV3=/conv3 WORK=/w IH=1376 IW=768
tag=dec;     check qwen_image_vae_decoder CHX=/chx CONV3=/conv3 WORK=/w H0=86 W0=48
tag=a5s;     check qwen3_asr_audio CHX=/chx CONV3=/conv3 WORK=/w WDIR=/wd BP=10 SP=128
tag=sp128;   check qwen3_asr_text CHX=/chx WORK=/w WDIR=/wd SP=128
tag=t128;    check qwen3_asr_step CHX=/chx QW=/qw TR=/tr WORK=/w WDIR=/wd T=128 NS=109
tag=pre448;  check qwen3_asr_text CHX=/chx WORK=/w WDIR=/wd SP=448 PRE=1 T=512
tag=pre512;  check qwen3_asr_step CHX=/chx QW=/qw TR=/tr WORK=/w WDIR=/wd T=512 NS=101 PRE=1
tag=sp64;    check umt5_enc CHX=/chx WORK=/w WDIR=/wd SP=64
# The codec decoder of Qwen3-TTS (docs/audio.md section 22): 128 frames, and 256 with the stage activations saved.
tag=sp128;   check qwen3_tts_decoder CHX=/chx WORK=/w WDIR=/wd SP=128
tag=probe256; check qwen3_tts_decoder CHX=/chx WORK=/w WDIR=/wd SP=256 PROBE=1
# The talker and the code predictor of Qwen3-TTS (docs/audio.md section 24): S0's 92 rows, and its 67 frames of 16 rows.
tag=talker128; check qwen3_tts_lm CHX=/chx WORK=/w WDIR=/wd SP=128
tag=pred1088; check qwen3_tts_lm CHX=/chx WORK=/w WDIR=/wd SP=1088 D=1024 FF=3072 NL=5 V=30720 PROJ=1
# Qwen3-TTS a frame a step (docs/audio.md section 26): S0's 92 steps free, and with the reference's codes forced and the talker's draw saved.
tag=free128; check qwen3_tts_step CHX=/chx QW=/qw TR=/tr WORK=/w WDIR=/wd T=128 NS=92
tag=forced128; check qwen3_tts_step CHX=/chx QW=/qw TR=/tr WORK=/w WDIR=/wd T=128 NS=92 FORCE=1 PROBE=1
# Kolibri 1.1, a layer's MLP half (docs/kolibri.md section 10): a mixture-of-experts layer, one with its down matrices in Q4_1, and the dense MLP of layers 0 and 1.
tag=moe;     check kolibri_mlp CHX=/chx W=/w
tag=moe41;   check kolibri_mlp CHX=/chx W=/w Q41=1
tag=dense;   check kolibri_mlp CHX=/chx W=/w MOE=0 Q41=1
# The video transformer at 17 frames of 640 x 352: the f16 programs (the expansion models/wan_dit.model had before PREC existed) and the f32-operand ones (docs/diffusion.md section 6, V3c).
tag=f16;     check wan_dit CHX=/chx WORK=/w WDIR=/wd S=1152 T=512 NB=30 HEAD=1
tag=f32;     check wan_dit CHX=/chx WORK=/w WDIR=/wd S=1152 T=512 NB=30 HEAD=1 PREC=1 GEMM=gemm_mma_f LNM=dit_wan_lnmod_f LN=vit_ln_f GELU=vit_bias_gelu_f ROPE=dit_rope16_f SCO=dit_scores_mma_f PVM=dit_pv_mma_f AT=f32
if [ -n "$SESSION" ]; then
  "$BIN" model-expand tests/model/chain.model CHX=tests/chx > "$WORK/chain.session"
  a=$(echo 1 | "$SESSION" "$WORK/chain.session")
  b=$(echo 1 | "$SESSION" tests/model/chain.model CHX=tests/chx)
  [ "$a" = "$b" ] && [ -n "$a" ] || { echo "the session on a model file answers '$b', on its expansion '$a'"; fail=1; }
  # 1 2 3 4 / 5 6 7 8 plus three times 10 20 30 40 / 50 60 70 80
  [ "$a" = "3.1e1 6.2e1 9.3e1 1.24e2 1.55e2 1.86e2 2.17e2 2.48e2" ] || { echo "the chain answers '$a'"; fail=1; }
  n=$((n + 1))
fi
echo "$n model-file checks"
exit $fail
