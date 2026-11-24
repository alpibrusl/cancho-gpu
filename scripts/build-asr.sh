#!/bin/sh
# Build the four host programs of the recognition (docs/audio.md sections 8 to 16), which `scripts/media transcribe` runs: CANCHO=<pinned cancho> scripts/build-asr.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p build
C="${CANCHO:-cancho}"
COMMON="tests/qwen_asr/host.cho tests/wan_enc/st.cho tests/wan_enc/codec.cho studio/files.cho"
"$C" build tests/qwen3_asr_front.cho tests/qwen_asr/wav.cho tests/qwen_asr/mel.cho tests/qwen_asr/answer.cho $COMMON --std -o build/qwen3_asr_front 2>&1 | grep -v "^ld: warning" || true
for p in audio text step; do
  "$C" build tests/qwen3_asr_${p}_gpu.cho $COMMON --std -o build/qwen3_asr_${p}_gpu 2>&1 | grep -v "^ld: warning" || true
done
test -x build/qwen3_asr_front && test -x build/qwen3_asr_audio_gpu && test -x build/qwen3_asr_text_gpu && test -x build/qwen3_asr_step_gpu
