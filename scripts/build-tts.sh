#!/bin/sh
# Build the four host programs of the speech (docs/audio.md sections 22 to 28), which `scripts/media speak` runs: CANCHO=<pinned cancho> scripts/build-tts.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p build
C="${CANCHO:-cancho}"
COMMON="tests/qwen_asr/host.cho tests/wan_enc/st.cho tests/wan_enc/codec.cho studio/files.cho"
"$C" build tests/qwen3_tts_front.cho $COMMON --std -o build/qwen3_tts_front 2>&1 | grep -v "^ld: warning" || true
for p in lm step decoder; do
  "$C" build tests/qwen3_tts_${p}_gpu.cho $COMMON --std -o build/qwen3_tts_${p}_gpu 2>&1 | grep -v "^ld: warning" || true
done
test -x build/qwen3_tts_front && test -x build/qwen3_tts_lm_gpu && test -x build/qwen3_tts_step_gpu && test -x build/qwen3_tts_decoder_gpu
