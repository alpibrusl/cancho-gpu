#!/bin/sh
# tests/qwen3_tts_front.sh -- what stands between a sentence and Qwen3-TTS's sessions and between their samples and a wav file (tests/qwen3_tts_front.cho, docs/audio.md section 28, S4), without a GPU
# or the checkpoint: it builds, it refuses what it must with the status it documents, the tokenizer file it writes from a toy vocabulary and merges is read by `cancho-gpu tokenize` and gives the ids
# the merges say, the prompt's ids and every voice, language and dialect, and the bytes of the samples and of the wav file. The real vocabulary against the reference's ids is by hand (section 29).
set -eu
cd "$(dirname "$0")/.."
C="${CANCHO:-cancho}"
G="${CANCHO_GPU:-build/cancho-gpu}"
mkdir -p build
$C build tests/qwen3_tts_front.cho tests/qwen_asr/host.cho tests/wan_enc/st.cho tests/wan_enc/codec.cho studio/files.cho --std -o build/qwen3_tts_front 2>&1 | grep -v "^ld: warning" || true
B=build/qwen3_tts_front
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/snap" "$T/empty" "$T/novocab"
fail=0
want() { # name, wanted status, the arguments
  name=$1; code=$2; shift 2
  rc=0; "$B" "$@" > "$T/out" 2>&1 || rc=$?
  if [ "$rc" -eq "$code" ]; then echo "ok   $name: exit $code"; else echo "FAIL $name: exit $rc, wanted $code: $(head -1 "$T/out")"; fail=1; fi
}
same() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: $2, wanted $3"; fail=1; fi; }
hex() { od -An -tx1 "$1" | tr -d ' \n'; }
# a toy tokenizer: byte-level tokens, a merge of letters, one with the space, and one whose two halves are a quote and a backslash (what a JSON string must escape)
printf '{"a": 0, "b": 1, "\\u0120": 2, "ab": 3, "\\u0120ab": 4, "\\"": 5, "\\\\": 6, "\\"\\\\": 7}\n' > "$T/snap/vocab.json"
printf '#version: 0.2\na b\n\304\240 ab\n" \\\n' > "$T/snap/merges.txt"
printf '#version: 0.2\na b\n' > "$T/novocab/merges.txt"
want "no command" 100
want "tokenizer, too few arguments" 100 tokenizer "$T/snap"
want "tokenizer, a relative path" 102 tokenizer snap "$T/tok.json"
want "tokenizer, no files" 104 tokenizer "$T/empty" "$T/tok.json"
want "tokenizer, no vocabulary" 104 tokenizer "$T/novocab" "$T/tok.json"
want "tokenizer" 0 tokenizer "$T/snap" "$T/tok.json"
same "tokenizer printed" "$(cat "$T/out")" "tokenizer: merges 3"
printf 'ab ab' > "$T/text1.txt"
printf '"\\' > "$T/text2.txt"
same "the file is read by cancho-gpu tokenize: ab, then space and ab" "$("$G" tokenize "$T/tok.json" "$T/text1.txt" | head -2 | tr '\n' '/')" "tokenizer ids=8/ids 2: 3 4/"
same "a merge of a quote and a backslash" "$("$G" tokenize "$T/tok.json" "$T/text2.txt" | sed -n 2p)" "ids 1: 7"
# the prompt
printf 'tokenizer ids=151643\nids 3: 785 3974 13\ndec "The quick."\n' > "$T/tok.txt"
want "prompt, too few arguments" 100 prompt "$T/tok.txt" ryan english
want "prompt, a voice the checkpoint has not" 101 prompt "$T/tok.txt" bob english "$T/ids.txt"
want "prompt, a language it has not" 101 prompt "$T/tok.txt" ryan latin "$T/ids.txt"
want "prompt, a relative path" 102 prompt tok.txt ryan english "$T/ids.txt"
want "prompt" 0 prompt "$T/tok.txt" ryan english "$T/ids.txt"
same "prompt printed" "$(cat "$T/out")" "text 3 language 2050 speaker 3061"
same "the ids: the role, the text, the end of the turn and the role again" "$(cat "$T/ids.txt")" "151644 77091 198 785 3974 13 151645 198 151644 77091 198"
pair() { "$B" prompt "$T/tok.txt" "$1" "$2" "$T/ids.txt" | cut -d' ' -f4,6; }
same "the nine voices" "$(for v in serena vivian uncle_fu ryan aiden ono_anna sohee eric dylan; do pair $v english | cut -d' ' -f2; done | tr '\n' ' ')" "3066 3065 3010 3061 2861 2873 2864 2875 2878 "
same "the ten languages, and none" "$(for l in chinese english german italian portuguese spanish japanese korean french russian auto; do pair ryan $l | cut -d' ' -f1; done | tr '\n' ' ')" "2055 2050 2053 2070 2071 2054 2058 2064 2061 2069 0 "
same "a dialect's voice with chinese or no language: its dialect" "$(pair eric chinese) $(pair eric auto) $(pair dylan chinese) $(pair dylan auto)" "2062 2875 2062 2875 2074 2878 2074 2878"
same "a dialect's voice with another language: that language" "$(pair eric english) $(pair dylan german)" "2050 2875 2053 2878"
printf 'ids 1: 5\ndec "a"\nids 1: 6\ndec "b"\n' > "$T/two.txt"
want "prompt, two records (a line of %% in the text)" 103 prompt "$T/two.txt" ryan english "$T/ids.txt"
printf 'tokenizer ids=8\nids 0:\ndec ""\n' > "$T/none.txt"
want "prompt, no token" 103 prompt "$T/none.txt" ryan english "$T/ids.txt"
printf 'ids 1: 151643\n' > "$T/past.txt"
want "prompt, an id past the vocabulary" 103 prompt "$T/past.txt" ryan english "$T/ids.txt"
want "prompt, no such file" 103 prompt "$T/nothere.txt" ryan english "$T/ids.txt"
{ printf 'ids 110:'; i=0; while [ $i -lt 110 ]; do printf ' 7'; i=$((i + 1)); done; printf '\n'; } > "$T/long.txt"
want "prompt, 110 tokens" 0 prompt "$T/long.txt" ryan english "$T/ids.txt"
{ printf 'ids 111:'; i=0; while [ $i -lt 111 ]; do printf ' 7'; i=$((i + 1)); done; printf '\n'; } > "$T/longer.txt"
want "prompt, 111 tokens" 105 prompt "$T/longer.txt" ryan english "$T/ids.txt"
# the samples: 0, 0.5, -0.5, 1, -1, 2, -2 and 0.25 as singles
printf '\000\000\000\000\000\000\000\077\000\000\000\277\000\000\200\077\000\000\200\277\000\000\000\100\000\000\000\300\000\000\200\076' > "$T/x.f32"
want "pcm, too few arguments" 100 pcm "$T/s.pcm" "$T/x.f32" 0 8
want "pcm, a last word that is neither" 101 pcm "$T/s.pcm" "$T/x.f32" 0 8 again
want "pcm, nothing to take" 101 pcm "$T/s.pcm" "$T/x.f32" 0 0 new
want "pcm, a relative path" 102 pcm s.pcm "$T/x.f32" 0 8 new
want "pcm, more than the samples" 105 pcm "$T/s.pcm" "$T/x.f32" 1 8 new
want "pcm, no samples" 105 pcm "$T/s.pcm" "$T/nothere.f32" 0 1 new
want "pcm" 0 pcm "$T/s.pcm" "$T/x.f32" 0 7 new
same "pcm printed" "$(cat "$T/out")" "pcm 7"
same "round(32767 x), clamped at 1: 0, 16384, -16384, 32767, -32767, 32767, -32767" "$(hex "$T/s.pcm")" "0000004000c0ff7f0180ff7f0180"
want "pcm, more: one sample after seven skipped" 0 pcm "$T/s.pcm" "$T/x.f32" 7 1 more
same "appended: 0.25 is 8192" "$(hex "$T/s.pcm")" "0000004000c0ff7f0180ff7f01800020"
want "pcm, new again" 0 pcm "$T/s.pcm" "$T/x.f32" 1 2 new
same "new starts the file again" "$(hex "$T/s.pcm")" "004000c0"
# the wav file: 44 bytes of header (RIFF, 36 + 4, WAVE, fmt, 16, PCM, 1 channel, 24000 Hz, 48000 bytes a second, 2 a sample, 16 bits, data, 4), then the samples
want "wav, too few arguments" 100 wav "$T/s.pcm"
want "wav, a relative path" 102 wav "$T/s.pcm" s.wav
want "wav, no samples" 105 wav "$T/nothere.pcm" "$T/s.wav"
printf 'abc' > "$T/odd.pcm"
want "wav, an odd number of bytes" 105 wav "$T/odd.pcm" "$T/s.wav"
want "wav" 0 wav "$T/s.pcm" "$T/s.wav"
same "wav printed" "$(cat "$T/out")" "samples 2"
same "the file's bytes" "$(hex "$T/s.wav")" "524946462800000057415645666d74201000000001000100c05d000080bb00000200100064617461040000000040""00c0"
[ "$fail" -eq 0 ] && echo "qwen3_tts_front: ok"
exit $fail
