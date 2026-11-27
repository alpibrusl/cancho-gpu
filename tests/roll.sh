#!/bin/sh
# tests/roll.sh [session binary] -- a roll, its windows, and a table that holds (docs/resident.md section 33; docs/kolibri.md section 19.2), on the CPU session (the gate) or, given its
# binary, the Metal one (by hand):
#     store P.OUT ROLL BLOCKS      the output of P is block j of the roll in repetition j
#     window Q.PARAM ROLL          Q.PARAM, n blocks long, is the n blocks that end with block j (the first n while there are fewer)
#     table P.PARAM FILE BLOCKS hold    block min(j, BLOCKS - 1), where a table takes j mod BLOCKS
# One program adds the block of a table to zeros and stores it; another adds a window of the roll to zeros and answers it, so an answer is the roll's blocks as they are. Small whole
# numbers: the answers are exact on both sessions. Then `seek` over a roll, and every refusal.
set -u
cd "$(dirname "$0")/.."
B="${1:-build/cancho-gpu-session}"
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'st=$?; rm -rf "$T"; exit $st' EXIT
fail=0
CHX=$(pwd)/tests/chx
printf '1 1\n2 2\n3 3\n4 4\n5 5\n6 6\n' > "$T/tab.txt"
printf '0 0\n' > "$T/z2.txt"; printf '0 0 0\n' > "$T/z3.txt"; printf '0 0 0 0\n' > "$T/z4.txt"; printf '0 0 0 0 0 0 0 0\n' > "$T/z8.txt"
# a session: the writer `a` (a block of two from the table of six, stored in the roll R of $1 blocks), then the lines given, then `steps` and the answer of `b`
sess() { # file, the roll's blocks, the lines after the writer
  f=$1; nb=$2; shift 2
  { printf 'program a %s/step_add.chx n=2\nresident a.w %s/z2.txt\ntable a.d %s/tab.txt 6\nstore a.wo R %s\n' "$CHX" "$T" "$T" "$nb"; for l in "$@"; do printf '%s\n' "$l"; done; printf 'steps\nanswer b.wo\n'; } > "$T/$f"
}
ask() { s=$1; shift; printf '%s\n' "$@" | "$B" "$T/$s" 2> "$T/err.txt" | awk '{ for (i = 1; i <= NF; i++) printf "%s%d", (i > 1 ? " " : ""), $i; printf ";" }'; }
same() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: \`$2\`, wanted \`$3\` $(head -c 300 "$T/err.txt")"; fail=1; fi; }
sess whole.session 4 "program b $CHX/step_add.chx n=8" "resident b.w $T/z8.txt" "window b.d R"
same "a roll of 4 blocks read whole: a block more at every step, zeros where none is written" "$(ask whole.session 1 1 1 1)" "1 1 0 0 0 0 0 0;1 1 2 2 0 0 0 0;1 1 2 2 3 3 0 0;1 1 2 2 3 3 4 4;"
same "a request of several steps: the roll after the last" "$(ask whole.session 3)" "1 1 2 2 3 3 0 0;"
sess slide.session 6 "program b $CHX/step_add.chx n=4" "resident b.w $T/z4.txt" "window b.d R"
same "a window of 2 blocks of a roll of 6: the first two while there are fewer, then the two that end with the step's" "$(ask slide.session 1 1 1 1 1 1)" "1 1 0 0;1 1 2 2;2 2 3 3;3 3 4 4;4 4 5 5;5 5 6 6;"
sess one.session 6 "program b $CHX/step_add.chx n=2" "resident b.w $T/z2.txt" "window b.d R"
same "a window of one block is the step's own" "$(ask one.session 1 1 1)" "1 1;2 2;3 3;"
sess two.session 6 "program b0 $CHX/step_add.chx n=2" "resident b0.w $T/z2.txt" "window b0.d R" "program b $CHX/step_add.chx n=4" "link b0.wo -> b.w" "window b.d R"
printf 'program a %s/step_add.chx n=2\nresident a.w %s/z2.txt\ntable a.d %s/tab.txt 6\nstore a.wo R 6\nprogram b0 %s/step_add.chx n=2\nresident b0.w %s/z2.txt\nwindow b0.d R\nprogram b %s/step_add.chx n=2\nlink b0.wo -> b.w\nwindow b.d R\nsteps\nanswer b.wo\n' "$CHX" "$T" "$T" "$CHX" "$T" "$CHX" > "$T/two.session"
same "two readers of one roll: each has the step's block (their sum)" "$(ask two.session 1 1 1)" "2 2;4 4;6 6;"
# seek: the blocks stay; a step run again writes its block again; a block after the step sought is still there, which is why what reads a roll masks
same "seek back over a roll: the step's block is written again, the earlier ones are as they were" "$(ask slide.session 4 "seek 2" 1 1)" "3 3 4 4;2;2 2 3 3;3 3 4 4;"
same "seek to 0 after four steps: the window is the first two blocks, the second still the earlier run's" "$(ask slide.session 4 "seek 0" 1)" "3 3 4 4;0;1 1 2 2;"
same "seek forward over blocks never written: zeros" "$(ask slide.session 1 "seek 4" 1)" "1 1 0 0;4;0 0 5 5;"
# hold
printf 'program b %s/step_add.chx n=2\nresident b.w %s/z2.txt\ntable b.d %s/tab.txt 6 hold\nsteps\nanswer b.wo\n' "$CHX" "$T" "$T" > "$T/hold.session"
printf 'program b %s/step_add.chx n=2\nresident b.w %s/z2.txt\ntable b.d %s/tab.txt 6\nsteps\nanswer b.wo\n' "$CHX" "$T" "$T" > "$T/round.session"
same "a table that holds: its last block from there on" "$(ask hold.session 5 1 1 1 "seek 100" 1 "seek 4" 1)" "5 5;6 6;6 6;6 6;100;6 6;4;5 5;"
same "and one that does not goes round" "$(ask round.session 5 1 1 1)" "5 5;6 6;1 1;2 2;"
# a repetition past the roll's last block
out=$(printf '4\n1\n1\n' | "$B" "$T/whole.session" 2>&1 > "$T/out.txt" | head -c 300)
case "$out" in "error[session]: the roll \`R\` has 4 blocks: repetition 4 is past them"*) [ "$(wc -l < "$T/out.txt" | tr -d ' ')" = 1 ] && echo "ok   a step past the roll's last block: refused, and the session ends" || { echo "FAIL a step past the roll: the session went on"; fail=1; } ;; *) echo "FAIL a step past the roll's last block: \`$out\`"; fail=1 ;; esac
out=$(printf 'seek 9\n1\n' | "$B" "$T/slide.session" 2>&1 > /dev/null | head -c 300)
case "$out" in "error[session]: the roll \`R\` has 6 blocks: repetition 9 is past them"*) echo "ok   a seek past the roll, then a step: refused" ;; *) echo "FAIL a seek past the roll: \`$out\`"; fail=1 ;; esac
# what a session file may not say
refused() { # name, the refusal's text, the session's lines
  name=$1; want=$2; shift 2
  for l in "$@"; do printf '%s\n' "$l"; done > "$T/r.session"
  out=$(printf '1\n' | "$B" "$T/r.session" 2>&1 > "$T/out.txt" | head -c 400)
  case "$out" in "error[session]: "*"$want"*) [ -s "$T/out.txt" ] && { echo "FAIL $name: refused, and answered"; fail=1; } || echo "ok   refused: $name" ;; *) echo "FAIL $name: \`$out\`"; fail=1 ;; esac
}
A="program a $CHX/step_add.chx n=2"; AW="resident a.w $T/z2.txt"; AD="table a.d $T/tab.txt 6"; ST="store a.wo R 6"
Bp="program b $CHX/step_add.chx n=4"; BW="resident b.w $T/z4.txt"; WI="window b.d R"
refused "store with no blocks" "store PROGRAM.OUT ROLL BLOCKS" "$A" "$AW" "$AD" "store a.wo R" "$Bp" "$BW" "$WI" steps "answer b.wo"
refused "a roll of 0 blocks" "positive whole number" "$A" "$AW" "$AD" "store a.wo R 0" "$Bp" "$BW" "$WI" steps "answer b.wo"
refused "a roll of x blocks" "positive whole number" "$A" "$AW" "$AD" "store a.wo R x" "$Bp" "$BW" "$WI" steps "answer b.wo"
refused "store of what is no PROGRAM.OUT" "store takes PROGRAM.OUT" "$A" "$AW" "$AD" "store awo R 6" "$Bp" "$BW" "$WI" steps "answer b.wo"
refused "store of a program that is not there" "is not a program" "$A" "$AW" "$AD" "store c.wo R 6" "$Bp" "$BW" "$WI" steps "answer b.wo"
refused "store of a parameter the kernel reads" "a roll stores an output" "$A" "$AW" "$AD" "store a.w R 6" "$Bp" "$BW" "$WI" steps "answer b.wo"
refused "a roll stored twice" "is stored twice" "$A" "$AW" "$AD" "$ST" "program a2 $CHX/step_add.chx n=2" "resident a2.w $T/z2.txt" "table a2.d $T/tab.txt 6" "store a2.wo R 6" "$Bp" "$BW" "$WI" steps "answer b.wo"
refused "an output stored in two rolls" "is stored twice" "$A" "$AW" "$AD" "$ST" "store a.wo S 6" "$Bp" "$BW" "$WI" steps "answer b.wo"
refused "window with a word too many" "window PROGRAM.PARAM ROLL" "$A" "$AW" "$AD" "$ST" "$Bp" "$BW" "window b.d R 2" steps "answer b.wo"
refused "a window of a roll that is not one" "is not a roll of the session" "$A" "$AW" "$AD" "$ST" "$Bp" "$BW" "window b.d Q" steps "answer b.wo"
refused "a window that is an output" "cannot be a window" "$A" "$AW" "$AD" "$ST" "$Bp" "$BW" "resident b.d $T/z4.txt" "window b.wo R" steps "answer b.wo"
refused "a window supplied another way too" "is supplied twice" "$A" "$AW" "$AD" "$ST" "$Bp" "$BW" "resident b.d $T/z4.txt" "$WI" steps "answer b.wo"
refused "a window that is no whole number of blocks" "a whole number of the roll's blocks" "$A" "$AW" "$AD" "$ST" "program b $CHX/step_add.chx n=3" "resident b.w $T/z3.txt" "$WI" steps "answer b.wo"
refused "a window longer than the roll" "no more than the roll has" "$A" "$AW" "$AD" "store a.wo R 1" "$Bp" "$BW" "$WI" steps "answer b.wo"
refused "a reader listed before the roll's program" "not listed before its reader" "$Bp" "$BW" "$WI" "$A" "$AW" "$AD" "$ST" steps "answer b.wo"
refused "a stored output that is also linked" "read through windows" "$A" "$AW" "$AD" "$ST" "program b $CHX/step_add.chx n=2" "link a.wo -> b.w" "resident b.d $T/z2.txt" steps "answer b.wo"
refused "a stored output that is also answered" "read through windows" "$A" "$AW" "$AD" "$ST" "$Bp" "$BW" "$WI" steps "answer a.wo"
refused "a stored output that is also carried" "read through windows" "$A" "$AW" "$AD" "$ST" "carry a.wo -> a.w" "$Bp" "$BW" "$WI" steps "answer b.wo"
refused "a roll in a session with no steps" "a session file with \`steps\`" "$A" "$AW" "resident a.d $T/z2.txt" "$ST" "$Bp" "$BW" "$WI" "answer b.wo"
refused "hold on a picked table" "table PROGRAM.PARAM FILE BLOCKS" "$A" "$AW" "$AD" "$Bp" "$BW" "table b.d $T/tab.txt 3 by a.wo 0 hold" steps "answer b.wo"
refused "hold where the blocks are" "positive whole number" "$A" "$AW" "table a.d $T/tab.txt hold" steps "answer a.wo"
if [ $fail -ne 0 ]; then echo "FAIL roll"; exit 1; fi
echo "roll: ok"
