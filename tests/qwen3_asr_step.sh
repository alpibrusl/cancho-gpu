#!/bin/sh
# tests/qwen3_asr_step.sh -- the host side of Qwen3-ASR's cached step (tests/qwen3_asr_step_gpu.cho, docs/audio.md A3) without a GPU or the checkpoint: it builds, it refuses what it must with the status it
# documents, from a made-up embedding table it writes the tables of four steps at their sizes with the right bytes, and it reads a made-up answer of the device. The steps themselves against the reference
# need the weights and a Metal GPU and are not in the gate (docs/audio.md section 11 has the commands).
set -eu
cd "$(dirname "$0")/.."
C="${CANCHO:-cancho}"
mkdir -p build
$C build tests/qwen3_asr_step_gpu.cho tests/qwen_asr/host.cho tests/wan_enc/st.cho tests/wan_enc/codec.cho studio/files.cho --std -o build/qwen3_asr_step_gpu 2>&1 | grep -v "^ld: warning" || true
B=build/qwen3_asr_step_gpu
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/snap" "$T/w" "$T/k" "$T/empty" "$T/a"
{ head -c 4096 /dev/zero; printf '\000\074\000\300'; head -c 4092 /dev/zero; } > "$T/w/emb.f16"
{ printf '\000\000\100\100'; head -c 8188 /dev/zero; } > "$T/rows.f32"
echo "1 151676 0" > "$T/ids.txt"
echo "1 x" > "$T/bad.txt"
fail=0
want() { # name, wanted status, the arguments
  name=$1; code=$2; shift 2
  rc=0; "$B" "$@" > "$T/out" 2>&1 || rc=$?
  if [ "$rc" -eq "$code" ]; then echo "ok   $name: exit $code"; else echo "FAIL $name: exit $rc, wanted $code: $(head -1 "$T/out")"; fail=1; fi
}
hex() { od -An -tx1 -j "$2" -N "$3" "$1" | tr -d ' \n'; }
size() { s=$(wc -c < "$1" | tr -d ' '); if [ "$s" = "$2" ]; then echo "ok   $(basename "$1"): $2 bytes"; else echo "FAIL $(basename "$1"): $s bytes, wanted $2"; fail=1; fi; }
same() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: $2, wanted $3"; fail=1; fi; }
want "no command" 100
want "weights, too few arguments" 100 weights "$T/snap"
want "weights, a relative path" 102 weights snap "$T/w"
want "weights, no checkpoint" 104 weights "$T/snap" "$T/empty"
want "inputs, too few arguments" 100 inputs "$T/w" "$T/k"
want "inputs, positions not a multiple of 64" 101 inputs "$T/w" "$T/k" "$T/ids.txt" "$T/rows.f32" 4 100 0
want "inputs, more steps than positions" 101 inputs "$T/w" "$T/k" "$T/ids.txt" "$T/rows.f32" 65 64 0
want "inputs, a mutant that does not exist" 101 inputs "$T/w" "$T/k" "$T/ids.txt" "$T/rows.f32" 4 64 0 4
want "inputs, a relative path" 102 inputs "$T/w" k "$T/ids.txt" "$T/rows.f32" 4 64 0
want "inputs, ids that are not numbers" 103 inputs "$T/w" "$T/k" "$T/bad.txt" "$T/rows.f32" 4 64 0
want "inputs, more rows than steps" 103 inputs "$T/w" "$T/k" "$T/ids.txt" "$T/rows.f32" 2 64 0
want "inputs, no embedding table" 104 inputs "$T/empty" "$T/k" "$T/ids.txt" "$T/rows.f32" 4 64 0
want "inputs" 0 inputs "$T/w" "$T/k" "$T/ids.txt" "$T/rows.f32" 4 64 0
same "inputs printed" "$(cat "$T/out")" "T=64 NS=4 rows 3"
# four steps, three of them rows of the sequence; caches of 64 positions
size "$T/k/xtab.f32" 32768; size "$T/k/mtab.f32" 16; size "$T/k/cos.f32" 1024; size "$T/k/sin.f32" 1024
size "$T/k/sel.f32" 1024; size "$T/k/keep.f32" 1024; size "$T/k/mask.f32" 1024; size "$T/k/accsel.f32" 64; size "$T/k/acckeep.f32" 64
size "$T/k/jrow.f32" 607744; size "$T/k/rank.f32" 607744; size "$T/k/zmk.f32" 607744; size "$T/k/zcache.f32" 262144; size "$T/k/acc0.f32" 16; size "$T/k/ids0.f32" 4
same "xtab row 0: the halves 1.0, -2.0" "$(hex "$T/k/xtab.f32" 0 8)" "0000803f000000c0"
same "xtab row 1: the audio row" "$(hex "$T/k/xtab.f32" 8192 4)" "00004040"
same "mtab: 1 1 1 0" "$(hex "$T/k/mtab.f32" 0 16)" "0000803f0000803f0000803f00000000"
same "sel: step 1 writes position 1" "$(hex "$T/k/sel.f32" 256 12)" "000000000000803f00000000"
same "keep: step 1 keeps all but position 1" "$(hex "$T/k/keep.f32" 256 12)" "0000803f000000000000803f"
same "mask: step 1 sees positions 0 and 1" "$(hex "$T/k/mask.f32" 256 12)" "0000000000000000286b6ece"
same "jrow begins 1, 2" "$(hex "$T/k/jrow.f32" 0 8)" "0000803f00000040"
same "rank begins 151936, 151935" "$(hex "$T/k/rank.f32" 0 8)" "00601448c05f1448"
# the same from position 1 (a prefill holds position 0): two rows left, the audio row first; step 0 writes position 1 and sees positions 0 and 1
want "inputs, a first step past the sequence" 103 inputs "$T/w" "$T/k2" "$T/ids.txt" "$T/rows.f32" 4 64 3
want "inputs, first + steps past the positions" 101 inputs "$T/w" "$T/k2" "$T/ids.txt" "$T/rows.f32" 4 64 61
mkdir -p "$T/k2"
want "inputs from position 1" 0 inputs "$T/w" "$T/k2" "$T/ids.txt" "$T/rows.f32" 4 64 1
same "inputs from position 1 printed" "$(cat "$T/out")" "T=64 NS=4 rows 2"
same "xtab row 0 from position 1: the audio row" "$(hex "$T/k2/xtab.f32" 0 4)" "00004040"
same "mtab from position 1: 1 1 0 0" "$(hex "$T/k2/mtab.f32" 0 16)" "0000803f0000803f0000000000000000"
same "sel from position 1: step 0 writes position 1" "$(hex "$T/k2/sel.f32" 0 12)" "000000000000803f00000000"
same "mask from position 1: step 0 sees positions 0 and 1" "$(hex "$T/k2/mask.f32" 0 12)" "0000000000000000286b6ece"
# a made-up answer of the device: the ids 2, 151677, 1 (1-based) and something that is not an id
printf '\000\000\000\100\100\037\024\110\000\000\200\077\000\000\000\077' > "$T/a/ids_out.f32"
want "ids" 0 ids "$T/a" 0
same "ids printed (0-based; ? for what is not an id)" "$(cat "$T/out")" "1 151676 0 ?"
want "ids, a first step past the answer" 101 ids "$T/a" 4
want "ids, no answer" 106 ids "$T/k" 0
echo "5 1 151676" > "$T/seq.txt"
want "check: the ids chosen are the next ids" 0 check "$T/a" "$T/seq.txt" 1
echo "5 1 7" > "$T/seq2.txt"
want "check: one is not" 1 check "$T/a" "$T/seq2.txt" 1
want "check, the prompt as long as the sequence" 101 check "$T/a" "$T/seq.txt" 3
# the same answer read as steps from position 1: the sequence then has a row before it
echo "9 5 1 151676" > "$T/seq3.txt"
want "check from position 1" 0 check "$T/a" "$T/seq3.txt" 2 1
want "check, a first step after the prompt's last row" 101 check "$T/a" "$T/seq3.txt" 2 2
[ "$fail" -eq 0 ] && echo "qwen3_asr_step: ok"
exit $fail
