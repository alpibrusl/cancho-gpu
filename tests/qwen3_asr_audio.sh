#!/bin/sh
# tests/qwen3_asr_audio.sh -- the host side of Qwen3-ASR's audio tower (tests/qwen3_asr_audio_gpu.cho, docs/audio.md A1) without a GPU or the checkpoint: it builds, it refuses what it must with the status
# it documents, and from a mel of one second and an empty snapshot it writes the inputs of the session at their sizes and then says the checkpoint is not there. The tower itself against the reference
# needs the weights and a Metal GPU and is not in the gate (docs/audio.md section 8 has the commands).
set -eu
cd "$(dirname "$0")/.."
C="${CANCHO:-cancho}"
mkdir -p build
$C build tests/qwen3_asr_audio_gpu.cho tests/wan_enc/st.cho tests/wan_enc/codec.cho studio/files.cho --std -o build/qwen3_asr_audio_gpu 2>&1 | grep -v "^ld: warning" || true
B=build/qwen3_asr_audio_gpu
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/snap" "$T/w" "$T/k"
head -c 51200 /dev/zero > "$T/mel.f32"
fail=0
want() { # name, wanted status, the arguments
  name=$1; code=$2; shift 2
  rc=0; "$B" "$@" > "$T/out" 2>&1 || rc=$?
  if [ "$rc" -eq "$code" ]; then echo "ok   $name: exit $code"; else echo "FAIL $name: exit $rc, wanted $code: $(head -1 "$T/out")"; fail=1; fi
}
want "no command" 100
want "prepare, too few arguments" 100 prepare "$T/snap"
want "frames 0" 101 prepare "$T/snap" "$T/w" "$T/k" "$T/mel.f32" 0
want "frames not a number" 101 prepare "$T/snap" "$T/w" "$T/k" "$T/mel.f32" 1x
want "a mutant that does not exist" 101 prepare "$T/snap" "$T/w" "$T/k" "$T/mel.f32" 100 9
want "a relative path" 102 prepare snap "$T/w" "$T/k" "$T/mel.f32" 100
want "the mel of another length" 103 prepare "$T/snap" "$T/w" "$T/k" "$T/mel.f32" 99
want "no mel" 103 prepare "$T/snap" "$T/w" "$T/k" "$T/none.f32" 100
want "no checkpoint" 104 prepare "$T/snap" "$T/w" "$T/k" "$T/mel.f32" 100
# one second is 13 tokens in 64 rows and 5 pieces: the picture 130 x 522 rows of 4, the masks, the positions 64 x 1,024, the attention's mask 16 x 64 x 64
for f in "mel.f32 1086464" "cmask0.f32 69376" "cmask1.f32 18176" "cmask2.f32 4864" "pos.f32 262144" "amask.f32 262144" "ones.f32 256"; do
  set -- $f
  size=$(wc -c < "$T/k/$1" | tr -d ' ')
  if [ "$size" = "$2" ]; then echo "ok   $1: $2 bytes"; else echo "FAIL $1: $size bytes, wanted $2"; fail=1; fi
done
want "compare, too few arguments" 100 compare "$T/k"
want "compare, tokens 0" 101 compare "$T/k" "$T/mel.f32" "$T/mel.f32" 0
want "compare, no rows" 106 compare "$T/k" "$T/mel.f32" "$T/mel.f32" 13
[ "$fail" -eq 0 ] && echo "qwen3_asr_audio: ok"
exit $fail
