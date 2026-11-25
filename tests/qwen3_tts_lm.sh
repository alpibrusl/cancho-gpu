#!/bin/sh
# tests/qwen3_tts_lm.sh -- the host side of Qwen3-TTS's talker and code predictor (tests/qwen3_tts_lm_gpu.cho, docs/audio.md section 24, S2) without a GPU or the checkpoint: it builds, it refuses what it
# must with the status it documents, from tables of zeros it writes both models' inputs at their sizes with the masks' right bytes, and `check` holds the arg max where the reference's top two are clear and
# only there. The models against the reference need the weights and a Metal GPU and are not in the gate (docs/audio.md section 25 has the commands).
set -eu
cd "$(dirname "$0")/.."
C="${CANCHO:-cancho}"
mkdir -p build
$C build tests/qwen3_tts_lm_gpu.cho tests/qwen_asr/host.cho tests/wan_enc/st.cho tests/wan_enc/codec.cho studio/files.cho --std -o build/qwen3_tts_lm_gpu 2>&1 | grep -v "^ld: warning" || true
B=build/qwen3_tts_lm_gpu
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/w" "$T/p" "$T/k" "$T/e" "$T/c" "$T/q"
fail=0
want() { # name, wanted status, the arguments
  name=$1; code=$2; shift 2
  rc=0; "$B" "$@" > "$T/out" 2>&1 || rc=$?
  if [ "$rc" -eq "$code" ]; then echo "ok   $name: exit $code"; else echo "FAIL $name: exit $rc, wanted $code: $(head -1 "$T/out")"; fail=1; fi
}
zeros() { dd if=/dev/zero of="$1" bs=1 count=0 seek="$2" 2>/dev/null; } # a file of zeros of this many bytes
put() { printf "$2" | dd of="$1" bs=4 seek="$3" conv=notrunc 2>/dev/null; } # four bytes at this single
at() { dd if="$1" bs=4 skip="$2" count=1 2>/dev/null | od -An -tx1 | tr -d ' \n'; }
frame="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 2047"
i=0; while [ $i -lt 4 ]; do echo "$frame"; i=$((i + 1)); done > "$T/codes.txt"
echo "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 2048" > "$T/past.txt"
echo "$frame 7" > "$T/ragged.txt"
echo "10 11 12 20 21 30 31 32 33 34" > "$T/ids.txt"
echo "10 11 12 30 31 32 33 34" > "$T/short.txt"
echo "10 11 12 151676 21 30 31 32 33 34" > "$T/audio.txt"
: > "$T/not.safetensors"
want "no command" 100
want "weights, too few arguments" 100 weights "$T/not.safetensors" "$T/w"
want "weights, a mutant that is the inputs'" 101 weights "$T/not.safetensors" "$T/w" "$T/p" 1
want "weights, a relative path" 102 weights "$T/not.safetensors" w "$T/p"
want "weights, not a checkpoint" 104 weights "$T/not.safetensors" "$T/w" "$T/p"
want "inputs, neither model" 100 inputs other "$T/w" "$T/k"
want "inputs talker, too few arguments" 100 inputs talker "$T/w" "$T/k" "$T/ids.txt" "$T/codes.txt" 2050
want "inputs talker, a language that is a code" 101 inputs talker "$T/w" "$T/k" "$T/ids.txt" "$T/codes.txt" 2047 3061
want "inputs talker, a speaker past the vocabulary" 101 inputs talker "$T/w" "$T/k" "$T/ids.txt" "$T/codes.txt" 2050 3072
want "inputs talker, a mutant that is the predictor's" 101 inputs talker "$T/w" "$T/k" "$T/ids.txt" "$T/codes.txt" 2050 3061 4
want "inputs talker, a relative path" 102 inputs talker "$T/w" k "$T/ids.txt" "$T/codes.txt" 2050 3061
want "inputs talker, 8 ids" 103 inputs talker "$T/w" "$T/k" "$T/short.txt" "$T/codes.txt" 2050 3061
want "inputs talker, the audio token among the ids" 103 inputs talker "$T/w" "$T/k" "$T/audio.txt" "$T/codes.txt" 2050 3061
want "inputs talker, a code of 2048" 103 inputs talker "$T/w" "$T/k" "$T/ids.txt" "$T/past.txt" 2050 3061
want "inputs talker, a frame of 17 codes" 103 inputs talker "$T/w" "$T/k" "$T/ids.txt" "$T/ragged.txt" 2050 3061
want "inputs talker, no tables" 104 inputs talker "$T/e" "$T/k" "$T/ids.txt" "$T/codes.txt" 2050 3061
# tables of zeros, at their sizes
zeros "$T/w/emb.f16" 622329856
zeros "$T/w/tp1.f32" 16777216; zeros "$T/w/tp2.f32" 16777216; zeros "$T/w/tpb1.f32" 8192; zeros "$T/w/tpb2.f32" 8192; zeros "$T/w/cemb.f32" 25165824
i=0; while [ $i -lt 15 ]; do zeros "$T/w/pemb$i.f32" 16777216; i=$((i + 1)); done
# the talker: 10 ids are 2 of text, so a prefix of 13 rows, and 4 frames: 17 rows in 64
want "inputs talker" 0 inputs talker "$T/w" "$T/k" "$T/ids.txt" "$T/codes.txt" 2050 3061
if [ "$(cat "$T/out")" = "SP=64 rows 17 prefix 13 frames 4" ]; then echo "ok   inputs talker says SP=64 rows 17 prefix 13 frames 4"; else echo "FAIL inputs talker said: $(cat "$T/out")"; fail=1; fi
for f in "x0.f32 524288" "cs.f32 16384" "sn.f32 16384" "mask.f32 16384" "pad.f32 8192"; do
  set -- $f
  size=$(wc -c < "$T/k/$1" | tr -d ' ')
  if [ "$size" = "$2" ]; then echo "ok   talker $1: $2 bytes"; else echo "FAIL talker $1: $size bytes, wanted $2"; fail=1; fi
done
# causal over the 17 rows: row 5 sees key 5 and not key 6; row 40, past the sequence, sees key 16 and not key 17 (-1e9 is the bytes 28 6b 6e ce)
m="$T/k/mask.f32"
if [ "$(at $m $((5 * 64 + 5))) $(at $m $((5 * 64 + 6))) $(at $m $((40 * 64 + 16))) $(at $m $((40 * 64 + 17)))" = "00000000 286b6ece 00000000 286b6ece" ]; then echo "ok   the talker's mask is causal over its rows"; else echo "FAIL the talker's mask"; fail=1; fi
# the predictor: 4 frames are 64 rows, from a hidden file of 5 rows, the frames' from row 1
zeros "$T/hidden.f32" 40960
put "$T/hidden.f32" '\000\000\200\077' $((2 * 2048 + 3))
want "inputs predictor, too few arguments" 100 inputs predictor "$T/w" "$T/q" "$T/codes.txt" "$T/hidden.f32"
want "inputs predictor, a mutant that is the talker's" 101 inputs predictor "$T/w" "$T/q" "$T/codes.txt" "$T/hidden.f32" 1 2
want "inputs predictor, a relative path" 102 inputs predictor "$T/w" q "$T/codes.txt" "$T/hidden.f32" 1
want "inputs predictor, rows the hidden file does not have" 103 inputs predictor "$T/w" "$T/q" "$T/codes.txt" "$T/hidden.f32" 2
want "inputs predictor, no hidden file" 103 inputs predictor "$T/w" "$T/q" "$T/codes.txt" "$T/none.f32" 0
want "inputs predictor, no tables" 104 inputs predictor "$T/e" "$T/q" "$T/codes.txt" "$T/hidden.f32" 1
want "inputs predictor" 0 inputs predictor "$T/w" "$T/q" "$T/codes.txt" "$T/hidden.f32" 1
if [ "$(cat "$T/out")" = "SP=64 frames 4" ]; then echo "ok   inputs predictor says SP=64 frames 4"; else echo "FAIL inputs predictor said: $(cat "$T/out")"; fail=1; fi
for f in "x0.f32 524288" "mask.f32 16384" "ones.f32 256"; do
  set -- $f
  size=$(wc -c < "$T/q/$1" | tr -d ' ')
  if [ "$size" = "$2" ]; then echo "ok   predictor $1: $2 bytes"; else echo "FAIL predictor $1: $size bytes, wanted $2"; fail=1; fi
done
# frame 1's first row (row 16) is row 2 of the hidden file: its single 3 is the 1.0 put there
if [ "$(at "$T/q/x0.f32" $((16 * 2048 + 3)))" = "0000803f" ]; then echo "ok   a frame's first row is the talker's row"; else echo "FAIL a frame's first row: $(at "$T/q/x0.f32" $((16 * 2048 + 3)))"; fail=1; fi
# causal within a frame, closed between: row 17 sees keys 16 and 17, not 15 and not 18
m="$T/q/mask.f32"
if [ "$(at $m $((17 * 64 + 15))) $(at $m $((17 * 64 + 16))) $(at $m $((17 * 64 + 17))) $(at $m $((17 * 64 + 18)))" = "286b6ece 00000000 00000000 286b6ece" ]; then echo "ok   the predictor's mask is causal within a frame and closed between frames"; else echo "FAIL the predictor's mask"; fail=1; fi
# its positions restart: single 0 of the cosines is cos(position), 1 at rows 0 and 16 and cos(1) (bytes 40 51 0a 3f) at rows 1 and 17
c="$T/q/cs.f32"
if [ "$(at $c 0) $(at $c 64) $(at $c $((16 * 64))) $(at $c $((17 * 64)))" = "0000803f 40510a3f 0000803f 40510a3f" ]; then echo "ok   the predictor's positions restart in every frame"; else echo "FAIL the predictor's positions: $(at $c 0) $(at $c 64) $(at $c $((16 * 64))) $(at $c $((17 * 64)))"; fail=1; fi
# check talker: a prefix of 1 and 1 frame are 2 rows of 3,072 from the session's 64 rows
want "check, neither model" 100 check other "$T/c" "$T/ref.f32" 1
want "check talker, too few arguments" 100 check talker "$T/c" "$T/ref.f32" 1
want "check talker, frames 0" 101 check talker "$T/c" "$T/ref.f32" 1 0
want "check talker, no logits" 106 check talker "$T/c" "$T/ref.f32" 1 1
zeros "$T/c/logits.f32" 786432; zeros "$T/ref.f32" 24576
put "$T/ref.f32" '\000\000\200\077' 5; put "$T/ref.f32" '\000\000\200\077' $((3072 + 9))
put "$T/c/logits.f32" '\000\000\200\077' 5; put "$T/c/logits.f32" '\000\000\200\077' $((3072 + 9))
want "check talker, the arg maxes equal" 0 check talker "$T/c" "$T/ref.f32" 1 1
put "$T/c/logits.f32" '\000\000\000\000' $((3072 + 9)); put "$T/c/logits.f32" '\000\000\200\077' $((3072 + 10))
want "check talker, a clear row's arg max elsewhere" 1 check talker "$T/c" "$T/ref.f32" 1 1
put "$T/ref.f32" '\000\000\200\076' $((3072 + 9))
want "check talker, the same where the reference's top two are 0.25 apart" 0 check talker "$T/c" "$T/ref.f32" 1 1
put "$T/c/logits.f32" '\000\000\300\177' 7
want "check talker, a value that is not a number" 1 check talker "$T/c" "$T/ref.f32" 1 1
# check predictor: 1 frame is 15 rows of 2,048 from the session's 64 rows of 30,720; row p of the frame reads its columns from 2048 (p - 1)
zeros "$T/q/logits.f32" 7864320; zeros "$T/pref.f32" 122880
want "check predictor, the reference of another size" 106 check predictor "$T/q" "$T/ref.f32" 1
put "$T/pref.f32" '\000\000\200\077' $((2 * 2048 + 4))
put "$T/q/logits.f32" '\000\000\200\077' $((3 * 30720 + 2 * 2048 + 4))
want "check predictor, head 2's arg max in row 3's own columns" 0 check predictor "$T/q" "$T/pref.f32" 1
put "$T/q/logits.f32" '\000\000\000\000' $((3 * 30720 + 2 * 2048 + 4)); put "$T/q/logits.f32" '\000\000\200\077' $((3 * 30720 + 3 * 2048 + 4))
want "check predictor, the same in the next head's columns" 1 check predictor "$T/q" "$T/pref.f32" 1
want "near, too few arguments" 100 near "$T/c/logits.f32" "$T/ref.f32" 0
want "near, the reference the longer" 106 near "$T/ref.f32" "$T/c/logits.f32" 0 3072
want "near, past the end" 106 near "$T/c/logits.f32" "$T/ref.f32" 255 3072
want "near" 0 near "$T/c/logits.f32" "$T/ref.f32" 0 3072
[ "$fail" -eq 0 ] && echo "qwen3_tts_lm: ok"
exit $fail
