#!/bin/sh
# tests/qwen3_tts_step.sh -- the host side of Qwen3-TTS generating a frame a step (tests/qwen3_tts_step_gpu.cho, docs/audio.md section 26, S3) without a GPU or the checkpoint: it builds, it refuses what it
# must with the status it documents, it writes the tables of six steps at their sizes with the right bytes (the prefix's rows, the noise from files and from a seed, the forbidden entries, the forced
# codes, each mutant's one change), and it reads made-up answers of the device: `codes`, `check` and `probe`. The steps themselves against the reference need the weights and a Metal GPU and are not in
# the gate (docs/audio.md section 27 has the commands).
set -eu
cd "$(dirname "$0")/.."
C="${CANCHO:-cancho}"
mkdir -p build
$C build tests/qwen3_tts_step_gpu.cho tests/qwen_asr/host.cho tests/wan_enc/st.cho tests/wan_enc/codec.cho studio/files.cho --std -o build/qwen3_tts_step_gpu 2>&1 | grep -v "^ld: warning" || true
B=build/qwen3_tts_step_gpu
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/w" "$T/k" "$T/k2" "$T/k3" "$T/k4" "$T/k7" "$T/s1" "$T/s2" "$T/s3" "$T/n" "$T/e" "$T/a" "$T/b" "$T/c" "$T/z"
fail=0
want() { # name, wanted status, the arguments
  name=$1; code=$2; shift 2
  rc=0; "$B" "$@" > "$T/out" 2>&1 || rc=$?
  if [ "$rc" -eq "$code" ]; then echo "ok   $name: exit $code"; else echo "FAIL $name: exit $rc, wanted $code: $(head -1 "$T/out")"; fail=1; fi
}
zeros() { dd if=/dev/zero of="$1" bs=1 count=0 seek="$2" 2>/dev/null; } # a file of zeros of this many bytes
put() { printf "$2" | dd of="$1" bs=4 seek="$3" conv=notrunc 2>/dev/null; } # four bytes at this single
at() { dd if="$1" bs=4 skip="$2" count=1 2>/dev/null | od -An -tx1 | tr -d ' \n'; }
size() { s=$(wc -c < "$1" | tr -d ' '); if [ "$s" = "$2" ]; then echo "ok   $(basename "$1"): $2 bytes"; else echo "FAIL $(basename "$1"): $s bytes, wanted $2"; fail=1; fi; }
same() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: $2, wanted $3"; fail=1; fi; }
ONE='\000\000\200\077'; TWO='\000\000\000\100'; THREE='\000\000\100\100'; HALF='\000\000\000\077'
# a prefix of three rows (row 1 begins 2.0), a pad row (begins 3.0), the reference's noise for two frames of the talker (row 1 begins 2.0) and one of the predictor (row 1, code 2's, begins 3.0)
zeros "$T/x0.f32" 24576; put "$T/x0.f32" "$TWO" 2048
zeros "$T/pad.f32" 8192; put "$T/pad.f32" "$THREE" 0
zeros "$T/n/gt.f32" 24576; put "$T/n/gt.f32" "$ONE" 0; put "$T/n/gt.f32" "$TWO" 3072
zeros "$T/n/gp.f32" 122880; put "$T/n/gp.f32" "$ONE" 0; put "$T/n/gp.f32" "$THREE" 2048
zeros "$T/short.f32" 16384
frame="4 2 3 4 5 6 7 8 9 10 11 12 13 14 15 2047"
{ echo "$frame"; echo "$frame"; } > "$T/codes.txt"
echo "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 2048" > "$T/past.txt"
: > "$T/not.safetensors"
want "no command" 100
want "weights, too few arguments" 100 weights "$T/not.safetensors"
want "weights, a mutant that is the inputs'" 101 weights "$T/not.safetensors" "$T/w" 2
want "weights, a relative path" 102 weights "$T/not.safetensors" w
want "weights, not a checkpoint" 104 weights "$T/not.safetensors" "$T/w"
want "inputs, too few arguments" 100 inputs "$T/k" "$T/x0.f32" "$T/pad.f32" 3 6 64 ref "$T/n"
want "inputs, positions not a multiple of 64" 101 inputs "$T/k" "$T/x0.f32" "$T/pad.f32" 3 6 100 ref "$T/n" -
want "inputs, fewer steps than the prefix" 101 inputs "$T/k" "$T/x0.f32" "$T/pad.f32" 3 2 64 ref "$T/n" -
want "inputs, more steps than positions" 101 inputs "$T/k" "$T/x0.f32" "$T/pad.f32" 3 65 64 ref "$T/n" -
want "inputs, a noise that is neither" 101 inputs "$T/k" "$T/x0.f32" "$T/pad.f32" 3 6 64 dice "$T/n" -
want "inputs, a seed of 0" 101 inputs "$T/k" "$T/x0.f32" "$T/pad.f32" 3 6 64 seed 0 -
want "inputs, a mutant that is the weights'" 101 inputs "$T/k" "$T/x0.f32" "$T/pad.f32" 3 6 64 ref "$T/n" - 5
want "inputs, a relative path" 102 inputs k "$T/x0.f32" "$T/pad.f32" 3 6 64 ref "$T/n" -
want "inputs, a code of 2048" 103 inputs "$T/k" "$T/x0.f32" "$T/pad.f32" 3 6 64 ref "$T/n" "$T/past.txt"
want "inputs, fewer rows than the prefix" 105 inputs "$T/k" "$T/short.f32" "$T/pad.f32" 3 6 64 ref "$T/n" -
want "inputs, a pad row that is not one" 105 inputs "$T/k" "$T/x0.f32" "$T/short.f32" 3 6 64 ref "$T/n" -
want "inputs, no noise files" 105 inputs "$T/k" "$T/x0.f32" "$T/pad.f32" 3 6 64 ref "$T/e" -
# six steps: three of the prefix, so frames 0 to 3 are drawn at steps 2 to 5; two frames forced, the end token after them
want "inputs" 0 inputs "$T/k" "$T/x0.f32" "$T/pad.f32" 3 6 64 ref "$T/n" "$T/codes.txt"
same "inputs printed" "$(cat "$T/out")" "T=64 NS=6 prefix 3 frames 4"
K="$T/k"
size "$K/xtab.f32" 49152; size "$K/mtab.f32" 24; size "$K/pad.f32" 8192; size "$K/cos.f32" 1536; size "$K/sin.f32" 1536; size "$K/sel.f32" 1536; size "$K/keep.f32" 1536; size "$K/mask.f32" 1536
size "$K/accsel.f32" 144; size "$K/acckeep.f32" 144; size "$K/valid.f32" 24; size "$K/addt.f32" 73728; size "$K/g0.f32" 73728; size "$K/g1.f32" 49152; size "$K/g15.f32" 49152; size "$K/force.f32" 24
size "$K/tf0.f32" 24; size "$K/tf15.f32" 24; size "$K/pcos0.f32" 256; size "$K/psin15.f32" 256; size "$K/psel15.f32" 256; size "$K/pkeep0.f32" 256; size "$K/pmask7.f32" 256; size "$K/gsel0.f32" 64; size "$K/gkeep15.f32" 64
size "$K/zero16.f32" 64; size "$K/zero2048.f32" 8192; size "$K/zero3072.f32" 12288; size "$K/one1.f32" 4; size "$K/zcache.f32" 262144; size "$K/pzcache.f32" 262144; size "$K/acc0.f32" 384
size "$K/jrow2048.f32" 8192; size "$K/jrow3072.f32" 12288; size "$K/rank2048.f32" 8192; size "$K/rank3072.f32" 12288
same "xtab row 1 is the prefix's row 1; row 3 is zeros" "$(at "$K/xtab.f32" 2048) $(at "$K/xtab.f32" 6144)" "00000040 00000000"
same "mtab: the prefix's three steps" "$(od -An -tx1 "$K/mtab.f32" | tr -d ' \n')" "0000803f0000803f0000803f000000000000000000000000"
same "valid: the steps after a frame's draw" "$(od -An -tx1 "$K/valid.f32" | tr -d ' \n')" "0000000000000000000000000000803f0000803f0000803f"
same "force: the two frames and the end token" "$(od -An -tx1 "$K/force.f32" | tr -d ' \n')" "00000000000000000000803f0000803f0000803f00000000"
same "tf0: code 4 as id 5 at the frames, the end token (2151) after them" "$(od -An -tx1 "$K/tf0.f32" | tr -d ' \n')" "0000803f0000803f0000a0400000a040007006450000803f"
same "tf15: code 2047 as id 2048 at the frames, nothing after" "$(at "$K/tf15.f32" 2) $(at "$K/tf15.f32" 4)" "00000045 0000803f"
same "the pad row" "$(at "$K/pad.f32" 0)" "00004040"
# the forbidden entries (-1e9 is the bytes 28 6b 6e ce): 2047 never, 2048 always, the end token 2150 in the first two frames (steps 2 and 3) and not after
A="$K/addt.f32"
same "addt at step 2: 2047 free, 2048 and the end token forbidden" "$(at $A $((2 * 3072 + 2047))) $(at $A $((2 * 3072 + 2048))) $(at $A $((2 * 3072 + 2150)))" "00000000 286b6ece 286b6ece"
same "addt: the end token forbidden at step 3 and free at step 4; 3071 forbidden" "$(at $A $((3 * 3072 + 2150))) $(at $A $((4 * 3072 + 2150))) $(at $A $((4 * 3072 + 3071)))" "286b6ece 00000000 286b6ece"
# the noise: frame g's row at step 2 + g, zeros before the first frame and past the files' end
same "g0: nothing at step 1, the talker's rows 0 and 1 at steps 2 and 3, zeros at step 4" "$(at "$K/g0.f32" 3072) $(at "$K/g0.f32" 6144) $(at "$K/g0.f32" 9216) $(at "$K/g0.f32" 12288)" "00000000 0000803f 00000040 00000000"
same "g1 and g2 at step 2: the predictor's rows 0 and 1 of frame 0; g1 at step 3: zeros" "$(at "$K/g1.f32" 4096) $(at "$K/g2.f32" 4096) $(at "$K/g1.f32" 6144)" "0000803f 00004040 00000000"
# the predictor's row 3: position 3, slot 3 of its caches, sees rows 0 to 3; slot 5 of the frame's ids
same "psel3 and pkeep3" "$(at "$K/psel3.f32" 2) $(at "$K/psel3.f32" 3) $(at "$K/pkeep3.f32" 3) $(at "$K/pkeep3.f32" 4)" "00000000 0000803f 00000000 0000803f"
same "pmask3" "$(at "$K/pmask3.f32" 3) $(at "$K/pmask3.f32" 4)" "00000000 286b6ece"
same "gsel5 and gkeep5" "$(at "$K/gsel5.f32" 5) $(at "$K/gsel5.f32" 6) $(at "$K/gkeep5.f32" 5)" "0000803f 00000000 00000000"
same "pcos0 is ones, pcos5 is not" "$(at "$K/pcos0.f32" 0) $([ "$(at "$K/pcos5.f32" 0)" != 0000803f ] && echo moved)" "0000803f moved"
same "sel, keep and mask at step 2" "$(at "$K/sel.f32" $((128 + 2))) $(at "$K/keep.f32" $((128 + 2))) $(at "$K/mask.f32" $((128 + 2))) $(at "$K/mask.f32" $((128 + 3)))" "0000803f 00000000 00000000 286b6ece"
same "jrow and rank begin 1, 2 and 3072, 3071" "$(at "$K/jrow3072.f32" 0) $(at "$K/jrow3072.f32" 1) $(at "$K/rank3072.f32" 0) $(at "$K/rank3072.f32" 1)" "0000803f 00000040 00004045 00f03f45"
# the mutants, each its one change
want "inputs, mutant 2" 0 inputs "$T/k2" "$T/x0.f32" "$T/pad.f32" 3 6 64 ref "$T/n" "$T/codes.txt" 2
same "mutant 2: the noise a step late" "$(at "$T/k2/g0.f32" 6144) $(at "$T/k2/g0.f32" 9216)" "00000000 0000803f"
want "inputs, mutant 3" 0 inputs "$T/k3" "$T/x0.f32" "$T/pad.f32" 3 6 64 ref "$T/n" "$T/codes.txt" 3
same "mutant 3: the predictor's row 5 at position 0" "$(at "$T/k3/pcos5.f32" 0) $(at "$T/k3/psin5.f32" 0)" "0000803f 00000000"
want "inputs, mutant 4" 0 inputs "$T/k4" "$T/x0.f32" "$T/pad.f32" 3 6 64 ref "$T/n" "$T/codes.txt" 4
same "mutant 4: no pad row" "$(at "$T/k4/pad.f32" 0)" "00000000"
want "inputs, mutant 7" 0 inputs "$T/k7" "$T/x0.f32" "$T/pad.f32" 3 6 64 ref "$T/n" "$T/codes.txt" 7
same "mutant 7: the end token free in the first frame, 2048 still forbidden" "$(at "$T/k7/addt.f32" $((2 * 3072 + 2150))) $(at "$T/k7/addt.f32" $((2 * 3072 + 2048)))" "00000000 286b6ece"
# the host's own noise: nothing forced, the same seed the same table, another seed another
want "inputs, a seed" 0 inputs "$T/s1" "$T/x0.f32" "$T/pad.f32" 3 6 64 seed 7 -
want "inputs, the seed again" 0 inputs "$T/s2" "$T/x0.f32" "$T/pad.f32" 3 6 64 seed 7 -
want "inputs, another seed" 0 inputs "$T/s3" "$T/x0.f32" "$T/pad.f32" 3 6 64 seed 8 -
same "nothing forced" "$(od -An -tx1 "$T/s1/force.f32" | tr -d ' \n')" "000000000000000000000000000000000000000000000000"
same "the seed's noise: none before the first frame, some at it" "$(at "$T/s1/g0.f32" 3072) $([ "$(at "$T/s1/g0.f32" 6144)" != 00000000 ] && echo some)" "00000000 some"
same "the same seed, the same noise" "$(cmp -s "$T/s1/g0.f32" "$T/s2/g0.f32" && cmp -s "$T/s1/g15.f32" "$T/s2/g15.f32" && echo equal)" "equal"
same "another seed, another noise" "$(cmp -s "$T/s1/g0.f32" "$T/s3/g0.f32" || echo differs)" "differs"
# a made-up answer of the device, [16, 4]: every draw the id 1 (code 0), but code 1 of step 1 the id 3, and the first code of step 3 the end token: with a prefix of 2, two frames
i=0; while [ $i -lt 64 ]; do printf "$ONE"; i=$((i + 1)); done > "$T/a/ids_out.f32"
cp "$T/a/ids_out.f32" "$T/b/ids_out.f32"; cp "$T/a/ids_out.f32" "$T/c/ids_out.f32"
put "$T/a/ids_out.f32" "$THREE" $((1 * 4 + 1)); put "$T/a/ids_out.f32" '\000\160\006\105' 3
want "codes" 0 codes "$T/a" 2 "$T/got.txt"
same "codes printed" "$(cat "$T/out")" "frames 2"
same "the frames written" "$(tr '\n' '/' < "$T/got.txt")" "0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0/0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0/"
want "codes, no end token" 3 codes "$T/b" 2 "$T/got2.txt"
same "codes without an end said so" "$(cat "$T/out")" "frames 3: no step drew the end token"
put "$T/c/ids_out.f32" '\000\020\000\105' $((2 * 4 + 1))
want "codes, a draw that is not a code" 1 codes "$T/c" 2 "$T/got3.txt"
want "codes, no answer" 106 codes "$T/e" 2 "$T/got4.txt"
want "codes, a prefix past the steps" 101 codes "$T/a" 5 "$T/got4.txt"
want "codes, a relative path" 102 codes "$T/a" 2 got.txt
# check: the reference's two frames are the answer's, every margin 1
i=0; while [ $i -lt 33 ]; do printf "$ONE"; i=$((i + 1)); done > "$T/marg.f32"
{ echo "0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0"; echo "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"; } > "$T/ref.txt"
want "check: every draw the reference's" 0 check "$T/a" 2 "$T/ref.txt" "$T/marg.f32"
same "check printed" "$(tr '\n' '/' < "$T/out")" "the talker's draws equal to the reference's: 3 of 3; of the clear ones 3 of 3/the predictor's: 30 of 30; of the clear ones 30 of 30/frames equal before the first that differs: 2 of 2/"
{ echo "0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0"; echo "0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0"; } > "$T/ref2.txt"
want "check: one clear draw is not" 1 check "$T/a" 2 "$T/ref2.txt" "$T/marg.f32"
same "check counted it" "$(sed -n 2,3p "$T/out" | tr '\n' '/')" "the predictor's: 29 of 30; of the clear ones 29 of 30/frames equal before the first that differs: 1 of 2/"
# the same draw with a margin of 0.1 is not clear: frame 1's code 3 is the predictor's draw 15 + 2, after the talker's 3
cp "$T/marg.f32" "$T/marg2.f32"; put "$T/marg2.f32" '\315\314\314\075' $((3 + 15 + 2))
want "check: a draw that differs where the reference is not clear" 0 check "$T/a" 2 "$T/ref2.txt" "$T/marg2.f32"
want "check, margins of another clip" 101 check "$T/a" 2 "$T/ref.txt" "$T/short.f32"
want "check, no answer" 106 check "$T/e" 2 "$T/ref.txt" "$T/marg.f32"
want "check, too few arguments" 100 check "$T/a" 2 "$T/ref.txt"
# probe: a draw's row with entries 0 and 1 kept at 2.0, the reference's probabilities a half each, no noise
i=0; while [ $i -lt 3072 ]; do printf '\050\153\156\316'; i=$((i + 1)); done > "$T/z/z.f32"
put "$T/z/z.f32" "$TWO" 0; put "$T/z/z.f32" "$TWO" 1
zeros "$T/p.f32" 12288; put "$T/p.f32" "$HALF" 0; put "$T/p.f32" "$HALF" 1
zeros "$T/g.f32" 12288
want "probe: the kept are the reference's" 0 probe "$T/z" "$T/p.f32" 0 "$T/g.f32"
cp "$T/p.f32" "$T/p2.f32"; put "$T/p2.f32" '\000\000\000\000' 1; put "$T/p2.f32" "$HALF" 2
want "probe: another entry kept" 1 probe "$T/z" "$T/p2.f32" 0 "$T/g.f32"
cp "$T/p.f32" "$T/p3.f32"; put "$T/p3.f32" '\000\000\200\076' 1
want "probe: a probability that is not ours" 1 probe "$T/z" "$T/p3.f32" 0 "$T/g.f32"
want "probe, a row the reference has not" 101 probe "$T/z" "$T/p.f32" 1 "$T/g.f32"
want "probe, no row saved" 106 probe "$T/e" "$T/p.f32" 0 "$T/g.f32"
[ "$fail" -eq 0 ] && echo "qwen3_tts_step: ok"
exit $fail
