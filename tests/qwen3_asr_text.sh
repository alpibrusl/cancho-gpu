#!/bin/sh
# tests/qwen3_asr_text.sh -- the host side of Qwen3-ASR's language model (tests/qwen3_asr_text_gpu.cho, docs/audio.md A2) without a GPU or the checkpoint: it builds, it refuses what it must with the status it
# documents, and from a made-up embedding table of two rows it writes the inputs of a pass at their sizes, with the embedding's halves and the audio row in their places. The pass itself against the reference
# needs the weights and a Metal GPU and is not in the gate (docs/audio.md section 9 has the commands).
set -eu
cd "$(dirname "$0")/.."
C="${CANCHO:-cancho}"
mkdir -p build
$C build tests/qwen3_asr_text_gpu.cho tests/qwen_asr/host.cho tests/wan_enc/st.cho tests/wan_enc/codec.cho studio/files.cho --std -o build/qwen3_asr_text_gpu 2>&1 | grep -v "^ld: warning" || true
B=build/qwen3_asr_text_gpu
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/snap" "$T/w" "$T/k" "$T/empty"
# an embedding table of two rows of 2,048 halves: row 0 zeros; row 1 begins 1.0 (0x3C00), -2.0 (0xC000), 2^-24 (the smallest half, 0x0001)
{ head -c 4096 /dev/zero; printf '\000\074\000\300\001\000'; head -c 4090 /dev/zero; } > "$T/w/emb.f16"
# one audio row of 2,048 singles beginning 3.0
{ printf '\000\000\100\100'; head -c 8188 /dev/zero; } > "$T/rows.f32"
echo "1 151676 0" > "$T/ids.txt"
echo "1 151676 151676" > "$T/two.txt"
echo "1 x 0" > "$T/bad.txt"
echo "151936" > "$T/big.txt"
fail=0
want() { # name, wanted status, the arguments
  name=$1; code=$2; shift 2
  rc=0; "$B" "$@" > "$T/out" 2>&1 || rc=$?
  if [ "$rc" -eq "$code" ]; then echo "ok   $name: exit $code"; else echo "FAIL $name: exit $rc, wanted $code: $(head -1 "$T/out")"; fail=1; fi
}
want "no command" 100
want "weights, too few arguments" 100 weights "$T/snap"
want "weights, a mutant that does not exist" 101 weights "$T/snap" "$T/w" 1
want "weights, a relative path" 102 weights snap "$T/w"
want "weights, no checkpoint" 104 weights "$T/snap" "$T/empty"
want "inputs, too few arguments" 100 inputs "$T/w"
want "inputs, a mutant that does not exist" 101 inputs "$T/w" "$T/k" "$T/ids.txt" "$T/rows.f32" 3
want "inputs, a relative path" 102 inputs "$T/w" k "$T/ids.txt" "$T/rows.f32"
want "inputs, ids that are not numbers" 103 inputs "$T/w" "$T/k" "$T/bad.txt" "$T/rows.f32"
want "inputs, an id past the vocabulary" 103 inputs "$T/w" "$T/k" "$T/big.txt" "$T/rows.f32"
want "inputs, no ids file" 103 inputs "$T/w" "$T/k" "$T/none.txt" "$T/rows.f32"
want "inputs, fewer audio rows than audio tokens" 105 inputs "$T/w" "$T/k" "$T/two.txt" "$T/rows.f32"
want "inputs, no embedding table" 104 inputs "$T/empty" "$T/k" "$T/ids.txt" "$T/rows.f32"
want "inputs" 0 inputs "$T/w" "$T/k" "$T/ids.txt" "$T/rows.f32"
[ "$(cat "$T/out")" = "SP=64 tokens 3" ] && echo "ok   inputs: SP=64 tokens 3" || { echo "FAIL inputs printed: $(cat "$T/out")"; fail=1; }
for f in "x0.f32 524288" "cs.f32 16384" "sn.f32 16384" "mask.f32 16384"; do
  set -- $f
  size=$(wc -c < "$T/k/$1" | tr -d ' ')
  if [ "$size" = "$2" ]; then echo "ok   $1: $2 bytes"; else echo "FAIL $1: $size bytes, wanted $2"; fail=1; fi
done
# row 0 is the embedding of id 1 (1.0, -2.0, 2^-24 as singles), row 1 the audio row (3.0), row 2 the embedding of id 0 (zeros)
hex() { od -An -tx1 -j "$2" -N "$3" "$1" | tr -d ' \n'; }
[ "$(hex "$T/k/x0.f32" 0 12)" = "0000803f000000c000008033" ] && echo "ok   x0 row 0: the halves 1.0, -2.0, 2^-24" || { echo "FAIL x0 row 0: $(hex "$T/k/x0.f32" 0 12)"; fail=1; }
[ "$(hex "$T/k/x0.f32" 8192 4)" = "00004040" ] && echo "ok   x0 row 1: the audio row" || { echo "FAIL x0 row 1: $(hex "$T/k/x0.f32" 8192 4)"; fail=1; }
[ "$(hex "$T/k/x0.f32" 16384 8)" = "0000000000000000" ] && echo "ok   x0 row 2: zeros" || { echo "FAIL x0 row 2"; fail=1; }
# the mask: row 0 sees key 0 only (0.0, then -1e9 = 0xce6e6b28), row 1 keys 0 and 1
[ "$(hex "$T/k/mask.f32" 0 8)" = "00000000286b6ece" ] && [ "$(hex "$T/k/mask.f32" 256 12)" = "0000000000000000286b6ece" ] && echo "ok   mask: causal" || { echo "FAIL mask: $(hex "$T/k/mask.f32" 0 8) $(hex "$T/k/mask.f32" 256 12)"; fail=1; }
want "check, too few arguments" 100 check "$T/k"
want "next, no logits" 106 next "$T/k" "$T/ids.txt"
want "check, no logits" 106 check "$T/k" "$T/ids.txt" 1 "$T/rows.f32"
[ "$fail" -eq 0 ] && echo "qwen3_asr_text: ok"
exit $fail
