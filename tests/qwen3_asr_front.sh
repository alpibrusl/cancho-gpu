#!/bin/sh
# tests/qwen3_asr_front.sh -- the host's front and back of Qwen3-ASR (tests/qwen3_asr_front.cho, docs/audio.md A4a) without a GPU or the checkpoint: the mel against a fixture the reference's feature
# extractor made (tests/qwen_asr/fixture: 0.2 s of two tones and noise, its second half 100 dB down), three mutants of the mel that must be far from it, wav files it reads and ones it refuses, the
# prompt's ids, and an answer decoded from a made-up vocabulary. The resampler's numbers are not held here (only that a clip at 8 kHz gives twice the samples): docs/audio.md section 15 has them.
set -eu
cd "$(dirname "$0")/.."
C="${CANCHO:-cancho}"
mkdir -p build
$C build tests/qwen3_asr_front.cho tests/qwen_asr/host.cho tests/qwen_asr/wav.cho tests/qwen_asr/mel.cho tests/qwen_asr/answer.cho tests/wan_enc/st.cho tests/wan_enc/codec.cho studio/files.cho --std -o build/qwen3_asr_front 2>&1 | grep -v "^ld: warning" || true
B=build/qwen3_asr_front
X=$(pwd -P)/tests/qwen_asr/fixture
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
fail=0
want() { # name, wanted status, the arguments
  name=$1; code=$2; shift 2
  rc=0; "$B" "$@" > "$T/out" 2>&1 || rc=$?
  if [ "$rc" -eq "$code" ]; then echo "ok   $name: exit $code"; else echo "FAIL $name: exit $rc, wanted $code: $(head -1 "$T/out")"; fail=1; fi
}
same() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: $2, wanted $3"; fail=1; fi; }
hex() { od -An -tx1 -j "$2" -N "$3" "$1" | tr -d ' \n'; }
le() { # a number as little-endian bytes: value, count
  v=$1; n=$2; while [ "$n" -gt 0 ]; do printf "\\$(printf %03o $((v & 255)))"; v=$((v >> 8)); n=$((n - 1)); done
}
wav() { # file, format, channels, rate, bits, bytes of silence, [bytes the data chunk claims]
  { printf 'RIFF'; le $((36 + $6)) 4; printf 'WAVEfmt '; le 16 4; le "$2" 2; le "$3" 2; le "$4" 4; le $(($4 * $3 * $5 / 8)) 4; le $(($3 * $5 / 8)) 2; le "$5" 2; printf 'data'; le "${7:-$6}" 4; head -c "$6" /dev/zero; } > "$1"
}
# ---- the mel against the reference's
want "the mel of the fixture's samples" 0 melraw "$X/tone_16k.f32" "$T/a.f32"
same "melraw printed" "$(cat "$T/out")" "frames 20 rate 16000 samples 3200"
want "... within 2e-4 of the reference's" 0 close "$T/a.f32" "$X/tone_mel.f32" 0.0002
want "the mel of the same samples as a wav file" 0 mel "$X/tone_f32.wav" "$T/b.f32"
if cmp -s "$T/a.f32" "$T/b.f32"; then echo "ok   the wav file gives the same mel"; else echo "FAIL the wav file gives another mel"; fail=1; fi
for m in 1 2 3; do
  want "mutant $m of the mel runs" 0 mel "$X/tone_f32.wav" "$T/m.f32" "$m"
  want "mutant $m (1 no window, 2 the last frame for the first, 3 no floor) is far from the reference's" 1 close "$T/m.f32" "$X/tone_mel.f32" 0.0002
done
# ---- wav files: silence gives a mel of -1.5 everywhere ((log10(1e-10) + 4) / 4)
wav "$T/s16.wav" 1 2 16000 16 6400
want "stereo PCM of 16 bits" 0 mel "$T/s16.wav" "$T/s.f32"
same "... 1,600 frames of two channels are 10 frames of mel" "$(cat "$T/out")" "frames 10 rate 16000 samples 1600"
same "... of -1.5" "$(hex "$T/s.f32" 0 8)" "0000c0bf0000c0bf"
wav "$T/s8k.wav" 1 1 8000 16 3200
want "a clip at 8 kHz" 0 mel "$T/s8k.wav" "$T/s.f32"
same "... is resampled to twice the samples" "$(cat "$T/out")" "frames 20 rate 8000 samples 3200"
wav "$T/ext.wav" 65534 1 16000 16 3200
want "an extensible format with no sub-format" 107 mel "$T/ext.wav" "$T/s.f32"
wav "$T/b8.wav" 1 1 16000 8 3200
want "PCM of 8 bits is not read" 107 mel "$T/b8.wav" "$T/s.f32"
wav "$T/r.wav" 1 1 4000 16 3200
want "a rate of 4,000 Hz is not read" 107 mel "$T/r.wav" "$T/s.f32"
wav "$T/cut.wav" 1 1 16000 16 3200 64000
want "a data chunk that says more than the file has" 109 mel "$T/cut.wav" "$T/s.f32"
wav "$T/tiny.wav" 1 1 16000 16 400
want "under 400 samples" 110 mel "$T/tiny.wav" "$T/s.f32"
head -c 100 /dev/zero > "$T/zeros.wav"
want "not a wav file" 105 mel "$T/zeros.wav" "$T/s.f32"
printf 'RIFF' > "$T/short.wav"
want "four bytes" 105 mel "$T/short.wav" "$T/s.f32"
want "no file" 104 mel "$T/none.wav" "$T/s.f32"
want "a relative path" 102 mel s.wav "$T/s.f32"
want "a mutant that does not exist" 101 mel "$T/s16.wav" "$T/s.f32" 4
want "mel, too few arguments" 100 mel "$T/s16.wav"
want "close, files of two sizes" 106 close "$T/a.f32" "$X/tone_16k.f32" 0.0002
want "close, a bound that is not a decimal" 101 close "$T/a.f32" "$T/a.f32" 2e-4
# ---- the prompt
want "the prompt for two rows" 0 prompt 2 "$T/p.txt" "$T/q.txt"
same "... its ids" "$(cat "$T/p.txt")" "151644 8948 198 151645 198 151644 872 198 151669 151676 151676 151670 151645 198 151644 77091 198"
same "... the prefill's" "$(cat "$T/q.txt")" "151644 8948 198 151645 198 151644 872 198 151669 151676 151676 151670 151645 198 151644 77091"
want "the prompt for no rows" 101 prompt 0 "$T/p.txt" "$T/q.txt"
# ---- the answer, from a vocabulary of six tokens (Ġ is the alphabet's space, Ã© the two bytes of an e with an acute accent)
printf '{"language": 7, "\\u0120English": 8, "\\u0120hello": 5, "\\u0120caf": 6, "\\u00c3\\u00a9": 9, "!": 10, "\\u2603": 11}\n' > "$T/vocab.json"
echo "1 2 7 8 151704 5 6 9 10 151645 5" > "$T/ans.txt"
want "an answer in words" 0 words "$T/vocab.json" "$T/ans.txt" 2
same "... the language and the text" "$(tr '\n' '|' < "$T/out")" "English| hello café!|"
printf '\000\000\000\101\000\000\020\101\100\046\024\110\000\000\300\100\200\027\024\110' > "$T/ids.f32"
want "the device's answer (1-based singles: 8 9 151705 6 151646)" 0 answer "$T/vocab.json" "$T/ids.f32" 0
same "... the language and the text" "$(tr '\n' '|' < "$T/out")" "English| hello|"
echo "7 8 5 151645" > "$T/nomark.txt"
want "an answer with no <asr_text>" 107 words "$T/vocab.json" "$T/nomark.txt" 0
echo "7 8 151704 5" > "$T/noend.txt"
want "an answer with no end" 107 words "$T/vocab.json" "$T/noend.txt" 0
echo "7 8 151704 99 151645" > "$T/noid.txt"
want "an id with no token" 108 words "$T/vocab.json" "$T/noid.txt" 0
echo "7 8 151704 11 151645" > "$T/snow.txt"
want "a token outside the alphabet" 108 words "$T/vocab.json" "$T/snow.txt" 0
want "no vocabulary" 108 words "$T/none.json" "$T/ans.txt" 2
want "a first place past the ids" 106 words "$T/vocab.json" "$T/ans.txt" 50
[ "$fail" -eq 0 ] && echo "qwen3_asr_front: ok"
exit $fail
