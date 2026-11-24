#!/bin/sh
# tests/qwen3_tts_decoder.sh -- the host side of Qwen3-TTS's codec decoder (tests/qwen3_tts_decoder_gpu.cho, docs/audio.md section 22, S1c) without a GPU or the checkpoint: it builds, it refuses what it
# must with the status it documents, from codebooks of zeros it writes the inputs of the session at their sizes, and `compare` holds the samples to 1e-3 and no further. The decoder itself against the
# reference needs the weights and a Metal GPU and is not in the gate (docs/audio.md section 23 has the commands).
set -eu
cd "$(dirname "$0")/.."
C="${CANCHO:-cancho}"
mkdir -p build
$C build tests/qwen3_tts_decoder_gpu.cho tests/qwen_asr/host.cho tests/wan_enc/st.cho tests/wan_enc/codec.cho studio/files.cho --std -o build/qwen3_tts_decoder_gpu 2>&1 | grep -v "^ld: warning" || true
B=build/qwen3_tts_decoder_gpu
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/w" "$T/k" "$T/e" "$T/c"
fail=0
want() { # name, wanted status, the arguments
  name=$1; code=$2; shift 2
  rc=0; "$B" "$@" > "$T/out" 2>&1 || rc=$?
  if [ "$rc" -eq "$code" ]; then echo "ok   $name: exit $code"; else echo "FAIL $name: exit $rc, wanted $code: $(head -1 "$T/out")"; fail=1; fi
}
frame="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 2047"
i=0; while [ $i -lt 65 ]; do echo "$frame"; i=$((i + 1)); done > "$T/codes.txt"
echo "$frame 7" > "$T/ragged.txt"
echo "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 2048" > "$T/past.txt"
echo "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 x" > "$T/letter.txt"
i=0; while [ $i -lt 385 ]; do echo "$frame"; i=$((i + 1)); done > "$T/long.txt"
: > "$T/not.safetensors"
want "no command" 100
want "weights, too few arguments" 100 weights "$T/not.safetensors"
want "weights, a mutant that is prepare's" 101 weights "$T/not.safetensors" "$T/w" 1
want "weights, a relative path" 102 weights not.safetensors "$T/w"
want "weights, not a checkpoint" 104 weights "$T/not.safetensors" "$T/w"
want "weights, no file" 104 weights "$T/none.safetensors" "$T/w"
want "prepare, too few arguments" 100 prepare "$T/w" "$T/k"
want "prepare, a mutant that is weights'" 101 prepare "$T/w" "$T/k" "$T/codes.txt" 3
want "prepare, a relative path" 102 prepare w "$T/k" "$T/codes.txt"
want "prepare, a frame of 17 codes" 103 prepare "$T/w" "$T/k" "$T/ragged.txt"
want "prepare, a code of 2048" 103 prepare "$T/w" "$T/k" "$T/past.txt"
want "prepare, a letter" 103 prepare "$T/w" "$T/k" "$T/letter.txt"
want "prepare, 385 frames" 103 prepare "$T/w" "$T/k" "$T/long.txt"
want "prepare, no codes" 103 prepare "$T/w" "$T/k" "$T/none.txt"
want "prepare, no codebooks" 104 prepare "$T/e" "$T/k" "$T/codes.txt"
# codebooks and products of zeros: 65 frames are 128 rows
i=0; while [ $i -lt 16 ]; do head -c 2097152 /dev/zero > "$T/w/cb$i.f32"; i=$((i + 1)); done
head -c 524288 /dev/zero > "$T/w/cbp0.f32"; head -c 524288 /dev/zero > "$T/w/cbp1.f32"
want "prepare" 0 prepare "$T/w" "$T/k" "$T/codes.txt"
if [ "$(cat "$T/out")" = "SP=128 frames 65 samples 124800" ]; then echo "ok   prepare says SP=128 frames 65 samples 124800"; else echo "FAIL prepare said: $(cat "$T/out")"; fail=1; fi
for f in "x0.f32 266240" "cs.f32 16384" "sn.f32 16384" "amask.f32 1048576" "ones0.f32 1024" "ones1.f32 2048"; do
  set -- $f
  size=$(wc -c < "$T/k/$1" | tr -d ' ')
  if [ "$size" = "$2" ]; then echo "ok   $1: $2 bytes"; else echo "FAIL $1: $size bytes, wanted $2"; fail=1; fi
done
# the mask: row 100 of head 0 sees keys 29 to 100 (72 of them), so its singles 28 and 101 are -1e9 (bytes 28 6b 6e ce) and 29 and 100 are zeros
at() { dd if="$T/k/amask.f32" bs=4 skip=$((100 * 128 + $1)) count=1 2>/dev/null | od -An -tx1 | tr -d ' \n'; }
if [ "$(at 28) $(at 29) $(at 100) $(at 101)" = "286b6ece 00000000 00000000 286b6ece" ]; then echo "ok   the mask's window is 72 keys"; else echo "FAIL the mask at row 100: $(at 28) $(at 29) $(at 100) $(at 101)"; fail=1; fi
# compare: one frame is 1,920 samples of 64 x 1,920 that the session writes; the bound is 1e-3 at every sample
want "compare, too few arguments" 100 compare "$T/c"
want "compare, frames 0" 101 compare "$T/c" "$T/ref.f32" 0
want "compare, no samples" 106 compare "$T/c" "$T/ref.f32" 1
head -c 7680 /dev/zero > "$T/ref.f32"
head -c 491520 /dev/zero > "$T/c/wave.f32"
want "compare, equal" 0 compare "$T/c" "$T/ref.f32" 1
want "compare, the reference of another length" 106 compare "$T/c" "$T/ref.f32" 2
put() { printf "$1" | dd of="$T/c/wave.f32" bs=4 seek=$2 conv=notrunc 2>/dev/null; }
put '\372\355\153\072' 1919
want "compare, the last sample off by 0.0009" 0 compare "$T/c" "$T/ref.f32" 1
put '\340\055\220\072' 5
want "compare, a sample off by 0.0011" 1 compare "$T/c" "$T/ref.f32" 1
put '\000\000\000\000' 5
put '\000\000\300\177' 7
want "compare, a sample that is not a number" 1 compare "$T/c" "$T/ref.f32" 1
put '\000\000\000\000' 7
put '\000\000\000\077' 1920
want "compare, a difference after the clip's last sample" 0 compare "$T/c" "$T/ref.f32" 1
want "stage, too few arguments" 100 stage "$T/c/wave.f32"
want "stage, no reference" 106 stage "$T/c/wave.f32" "$T/none.f32"
want "stage, the reference the longer" 106 stage "$T/ref.f32" "$T/c/wave.f32"
want "stage" 0 stage "$T/c/wave.f32" "$T/ref.f32"
[ "$fail" -eq 0 ] && echo "qwen3_tts_decoder: ok"
exit $fail
