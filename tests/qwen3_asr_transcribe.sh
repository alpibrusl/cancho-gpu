#!/bin/sh
# tests/qwen3_asr_transcribe.sh -- stage A4a of docs/audio.md, end to end: a wav file to its words through the host's mel, the audio tower's session (A1), the prompt, the prefill pass and the cached steps (A3b),
# and the answer decoded from vocab.json. Needs a Metal GPU, the checkpoint in the Hugging Face cache and the weights the three host programs write (written here on first use), so it is not in the gate.
#
#     tests/qwen3_asr_transcribe.sh <clip.wav> <work dir> [<most tokens of answer, default 192>]
#
# Prints the language and the text on two lines; the stages' times on the lines before them when TIMES=1.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CLIP=$1; K=$2; MOST=${3:-192}
SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/* | tail -1)
WA=$HOME/.cache/cancho-media/qwen3-asr-audio-f16; WT=$HOME/.cache/cancho-media/qwen3-asr-text-f16
F=$ROOT/build/qwen3_asr_front; BA=$ROOT/build/qwen3_asr_audio_gpu; BT=$ROOT/build/qwen3_asr_text_gpu; BS=$ROOT/build/qwen3_asr_step_gpu; SESSION=$ROOT/build/cancho-gpu-metal-session
now() { python3 -c 'import time; print(time.time())'; }
lap() { [ "${TIMES:-0}" = 1 ] && python3 -c "print('  %-34s %.2f s' % ('$1', $(now) - $T0))"; T0=$(now); }
mkdir -p "$K" "$WA" "$WT"
T0=$(now); TS=$T0
[ -f "$WT/embt.f16" ] || { "$BT" weights "$SNAP" "$WT" > /dev/null; "$BS" weights "$SNAP" "$WT" > /dev/null; lap "the language model's weights, once"; }
frames=$("$F" mel "$CLIP" "$K/clip.mel.f32" | sed -n 's/^frames \([0-9]*\) .*/\1/p')
[ -n "$frames" ] || { echo "the mel failed"; exit 1; }
lap "the mel on the host ($frames frames)"
set -- $("$BA" prepare "$SNAP" "$WA" "$K" "$K/clip.mel.f32" "$frames")
BPD=$1; SPD=$2; rows=$4
printf '1\ncheckpoint\n' | "$SESSION" "$ROOT/models/qwen3_asr_audio.model" CHX="$ROOT/tests/chx" CONV3="$ROOT/examples/cnn/conv3.chx" WORK="$K" WDIR="$WA" "$BPD" "$SPD" > /dev/null 2> "$K/audio.err"
lap "the audio tower ($rows rows)"
P=$("$F" prompt "$rows" "$K/prompt.txt" "$K/prefill.txt" | sed -n 's/^prompt //p')
T=$(( (P + MOST + 63) / 64 * 64 ))
SPT=$("$BT" inputs "$WT" "$K" "$K/prefill.txt" "$K/rows.f32" | sed -n 's/^SP=\([0-9]*\) .*/\1/p')
printf '1\ncheckpoint\n' | "$SESSION" "$ROOT/models/qwen3_asr_text.model" CHX="$ROOT/tests/chx" WORK="$K" WDIR="$WT" SP="$SPT" PRE=1 T="$T" > /dev/null 2> "$K/pass.err"
lap "the prompt in one pass ($((P - 1)) rows)"
NS=$((MOST + 1))
"$BS" inputs "$WT" "$K" "$K/prompt.txt" "$K/rows.f32" "$NS" "$T" "$((P - 1))" > /dev/null
printf '%s\ncheckpoint\n' "$NS" | "$SESSION" "$ROOT/models/qwen3_asr_step.model" CHX="$ROOT/tests/chx" QW="$ROOT/examples/qwen" TR="$ROOT/examples/train" WORK="$K" WDIR="$WT" T="$T" NS="$NS" PRE=1 > /dev/null 2> "$K/step.err"
lap "the answer ($NS steps)"
"$F" answer "$SNAP/vocab.json" "$K/ids_out.f32" 0 > "$K/transcript.txt"
lap "the answer as text"
[ "${TIMES:-0}" = 1 ] && python3 -c "print('  %-34s %.2f s' % ('all', $(now) - $TS))"
cat "$K/transcript.txt"
