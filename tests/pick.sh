#!/bin/sh
# tests/pick.sh [session binary] -- a table whose block is picked by a value a kernel wrote (docs/kolibri.md K2: `table P.PARAM FILE BLOCKS [type] by Q.OUT SLOT`), on the CPU session
# (the gate) or, given its binary, the Metal one (by hand). Three programs each add the block of a table of three that a slot of an earlier program's output names; the values are small whole
# numbers, so the answers are exact. Then the same with the picks changing from step to step, a table of bytes, and every refusal: a value that is not a block is never a read.
set -eu
cd "$(dirname "$0")/.."
B="${1:-build/cancho-gpu-session}"
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
fail=0
QW=$(pwd)/examples/qwen; CHX=$(pwd)/tests/chx
ask() { printf '%s\n' "$2" | "$B" "$1" QW="$QW" CHX="$CHX" W="$T" 2>&1 | head -c 6000; }
same() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: $2, wanted $3"; fail=1; fi; }
refused() { # name, model, the refusal's text
  out=$(ask "$2" ""); case "$out" in "error[session]: $3"*) echo "ok   $1: refused" ;; *) echo "FAIL $1: $out"; fail=1 ;; esac
}
printf '2 0 1\n' > "$T/idx.txt"; printf '0 0 0\n' > "$T/z3.txt"; printf '0 0 0 0\n' > "$T/z4.txt"
printf '10 11 12 13  20 21 22 23  30 31 32 33\n' > "$T/tab.txt"
model() { # name, the three slots' lines for a, b and c (after `table X.z $W/tab.txt 3`), then extra lines
cat > "$T/$1.model" <<EOM
text QW = examples/qwen
text W = .
program ix \$QW/res_add.chx r=1 c=3 bm=1
resident ix.z \$W/${5:-idx.txt}
resident ix.res \$W/z3.txt
program a \$QW/res_add.chx r=1 c=4 bm=1
table a.z \$W/tab.txt 3 $2
resident a.res \$W/z4.txt
program b \$QW/res_add.chx r=1 c=4 bm=1
table b.z \$W/tab.txt 3 $3
link a.y -> b.res
program c \$QW/res_add.chx r=1 c=4 bm=1
table c.z \$W/tab.txt 3 $4
link b.y -> c.res
answer a.y
answer b.y
answer c.y
EOM
}
model pick "by ix.y 0" "by ix.y 1" "by ix.y 2"
same "three picked blocks: 2, then 0, then 1, each added to the one before" "$(ask "$T/pick.model" "")" "3e1 3.1e1 3.2e1 3.3e1 4e1 4.2e1 4.4e1 4.6e1 6e1 6.3e1 6.6e1 6.9e1"
model slot "by ix.y 1" "by ix.y 1" "by ix.y 1"
same "the slot is read, not the first value: block 0 three times" "$(ask "$T/slot.model" "")" "1e1 1.1e1 1.2e1 1.3e1 2e1 2.2e1 2.4e1 2.6e1 3e1 3.3e1 3.6e1 3.9e1"
model mixed "by ix.y 0" "" "by ix.y 2"
same "a table that steps beside two that are picked: block 0 at the first request" "$(ask "$T/mixed.model" "")" "3e1 3.1e1 3.2e1 3.3e1 4e1 4.2e1 4.4e1 4.6e1 6e1 6.3e1 6.6e1 6.9e1"
# the picks change from step to step: the index itself is a table that steps (1 2 0, then 0 0 2), and two steps add up in a carried sum
printf '1 2 0  0 0 2\n' > "$T/idxs.txt"
cat > "$T/steps.model" <<EOM
text QW = examples/qwen
text W = .
program ix \$QW/res_add.chx r=1 c=3 bm=1
table ix.z \$W/idxs.txt 2
resident ix.res \$W/z3.txt
program a \$QW/res_add.chx r=1 c=4 bm=1
table a.z \$W/tab.txt 3 by ix.y 0
resident a.res \$W/z4.txt
carry a.y -> a.res
program b \$QW/res_add.chx r=1 c=4 bm=1
table b.z \$W/tab.txt 3 by ix.y 2
link a.y -> b.res
steps
answer a.y
answer b.y
EOM
same "two steps: block 1 then block 0 summed, and the last step's second pick (block 2) on top" "$(ask "$T/steps.model" "2")" "3e1 3.2e1 3.4e1 3.6e1 6e1 6.3e1 6.6e1 6.9e1"
# a table of 4-bit codes and one of f16 scales, picked together: dq_q4 on the block the value names
awk 'BEGIN { for (b = 0; b < 3; b++) for (i = 0; i < 64; i++) printf "%d ", (b * 16 + i * 5) % 100 - 50; printf "\n" }' > "$T/codes.txt"
printf '0.5 1 2 4  0.25 8 1.5 3  1 1 1 1\n' > "$T/scales.txt"; printf '1 1\n' > "$T/ones.txt"; printf '1\n' > "$T/one.txt"; printf '0\n' > "$T/z1.txt"
q4() { # the block
cat > "$T/q4.model" <<EOM
text QW = examples/qwen
text CHX = tests/chx
text W = .
program ix \$QW/res_add.chx r=1 c=1 bm=1
resident ix.z \$W/$1
resident ix.res \$W/z1.txt
program d \$CHX/dq_q4.chx n=2 k=64 bn=2 bk=64
table d.wq \$W/codes.txt 3 by ix.y 0
table d.ws \$W/scales.txt 3 by ix.y 0
resident d.wg \$W/ones.txt
answer d.w
EOM
}
want() { awk -v b="$1" 'BEGIN { split("0.5 1 2 4 0.25 8 1.5 3 1 1 1 1", sc, " "); for (r = 0; r < 2; r++) for (c = 0; c < 64; c++) { v = (b * 16 + (r * 32 + int(c / 2)) * 5) % 100 - 50; if (v < 0) v += 256; code = (c % 2 == 0) ? v % 16 : int(v / 16); printf "%.9g\n", (code - 8) * sc[b * 4 + r * 2 + int(c / 32) + 1] } }'; }
printf '1\n' > "$T/one1.txt"; q4 one1.txt
same "codes and scales picked together: block 1 dequantised" "$(ask "$T/q4.model" "" | tr ' ' '\n' | grep -v '^$' | awk '{printf "%.9g\n", $1}' | cksum)" "$(want 1 | cksum)"
printf '2\n' > "$T/two1.txt"; q4 two1.txt
same "and block 2" "$(ask "$T/q4.model" "" | tr ' ' '\n' | grep -v '^$' | awk '{printf "%.9g\n", $1}' | cksum)" "$(want 2 | cksum)"
# refusals: a value that is not a block of the table
for v in "3 0 1" "-1 0 1" "1.5 0 1" "1e30 0 1"; do
  printf '%s\n' "$v" > "$T/bad.txt"; model bad "by ix.y 0" "by ix.y 1" "by ix.y 2" bad.txt
  refused "a pick of ${v%% *}" "$T/bad.model" "a table of 3 blocks was picked by a value that is not one of them"
done
# refusals of the session file
model late "by c.y 0" "by ix.y 1" "by ix.y 2"
refused "picked by a program listed after it" "$T/late.model" "line 5: \`a\` is picked by \`c\`, which is not listed before it"
model past "by ix.y 3" "by ix.y 1" "by ix.y 2"
refused "a slot past the output's values" "$T/past.model" "line 5: \`y\` has 3 values: the slot is past them"
model input "by ix.z 0" "by ix.y 1" "by ix.y 2"
refused "picked by a parameter a kernel only reads" "$T/input.model" "line 5: \`z\` is not an f32 output: a table is picked by a value a kernel wrote"
model noprog "by nobody.y 0" "by ix.y 1" "by ix.y 2"
out=$(ask "$T/noprog.model" ""); case "$out" in "error[session]: line 5:"*) echo "ok   picked by a program that is not there: refused" ;; *) echo "FAIL picked by a program that is not there: $out"; fail=1 ;; esac
model frac "by ix.y 0.5" "by ix.y 1" "by ix.y 2"
refused "a slot that is not a whole number" "$T/frac.model" "line 5: a table is picked \`by PROGRAM.OUT SLOT\`, the slot a whole number from 0"
model words "by ix.y" "by ix.y 1" "by ix.y 2"
refused "by with no slot" "$T/words.model" "line 5: table PROGRAM.PARAM FILE BLOCKS [f32|f16|i8] [hold | span | by PROGRAM.OUT SLOT]"
[ "$fail" -eq 0 ] && echo "pick: ok"
exit $fail
