#!/bin/sh
# tests/phase.sh [session binary] -- phases (docs/resident.md section 35; docs/kolibri.md section 20), on the CPU session (the gate) or, given its binary, the Metal one (by hand):
#     phase NAME STRIDE            the programs after the line run for a request `NAME N ...` alone, and a repetition of theirs is STRIDE repetitions of the session
#     table P.PARAM FILE BLOCKS span    the parameter is STRIDE blocks of the table, the repetition's first
# and what a roll, a window and a chosen mask are in a phase of stride S: S blocks stored, the window that ends with the last of them, S masks.
# The session's own programs write a roll a block a repetition and read a window of two; the phase `two`, of stride 2, writes two blocks from a spanned table and reads a window of
# three. Each set counts its own runs in a carried sum, so that a set that ran for the other's request shows. Small whole numbers: exact on both sessions. Then every refusal.
set -u
cd "$(dirname "$0")/.."
B="${1:-build/cancho-gpu-session}"
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'st=$?; rm -rf "$T"; exit $st' EXIT
fail=0
CHX=$(pwd)/tests/chx
i=1; while [ $i -le 12 ]; do echo "$i $i"; i=$((i + 1)); done > "$T/tab.txt"
printf '0\n' > "$T/z1.txt"; printf '1\n' > "$T/one.txt"; printf '0 0\n' > "$T/z2.txt"; printf '0 0 0 0\n' > "$T/z4.txt"; printf '0 0 0 0 0 0\n' > "$T/z6.txt"
OWN="program a $CHX/step_add.chx n=2
resident a.w $T/z2.txt
table a.d $T/tab.txt 12
store a.wo R 12
program b $CHX/step_add.chx n=4
resident b.w $T/z4.txt
window b.d R
program c $CHX/step_add.chx n=1
resident c.w $T/z1.txt
resident c.d $T/one.txt
carry c.wo -> c.w"
TWO="phase two 2
program pa $CHX/step_add.chx n=4
resident pa.w $T/z4.txt
table pa.d $T/tab.txt 12 span
store pa.wo R 12
program pb $CHX/step_add.chx n=6
resident pb.w $T/z6.txt
window pb.d R
program pc $CHX/step_add.chx n=1
resident pc.w $T/z1.txt
resident pc.d $T/one.txt
carry pc.wo -> pc.w"
END="steps
answer b.wo
answer c.wo
answer pb.wo
answer pc.wo"
printf '%s\n%s\n%s\n' "$OWN" "$TWO" "$END" > "$T/p.session"
ask() { s=$1; shift; printf '%s\n' "$@" | "$B" "$T/$s" 2> "$T/err.txt" | awk '{ for (i = 1; i <= NF; i++) printf "%s%d", (i > 1 ? " " : ""), $i; printf ";" }'; }
same() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: \`$2\`, wanted \`$3\` $(head -c 300 "$T/err.txt")"; fail=1; fi; }
same "a step, a repetition of the phase (two blocks, a window of three that ends with the second), a step: each request answers its own programs' answers, and each set has run only for its own" \
  "$(ask p.session 1 "two 1" 1)" "1 1 0 0 1;1 1 2 2 3 3 1;3 3 4 4 2;"
same "two repetitions of the phase are four of the session: the window after the last, and the step after them is the fifth" "$(ask p.session "two 2" 1)" "2 2 3 3 4 4 2;4 4 5 5 1;"
same "seek, then the phase: its blocks from there (the one before them was never written); the count after a repetition is two further" "$(ask p.session "seek 3" "two 1" "seek 100" "seek 5" "two 1" 1)" "3;0 0 4 4 5 5 1;100;5;5 5 6 6 7 7 2;7 7 8 8 1;"
same "the first window of the phase, while there are fewer blocks than it has: the roll's first three" "$(ask p.session "two 1")" "1 1 2 2 0 0 1;"
same "the phase after steps reads what they stored, and a step what the phase stored" "$(ask p.session 2 "two 1" 1)" "1 1 2 2 2;2 2 3 3 4 4 1;4 4 5 5 3;"
same "back over blocks the phase wrote: a step reads the phase's block before its own, and the phase after it the step's" "$(ask p.session "two 2" "seek 1" 1 "two 1")" "2 2 3 3 4 4 2;1;1 1 2 2 1;2 2 3 3 4 4 3;"
# the last repetition the table and the roll hold, and the one past it
same "the phase's last repetition: blocks 10 and 11 of 12" "$(ask p.session "seek 10" "two 1")" "10;0 0 11 11 12 12 1;"
out=$(printf 'seek 11\ntwo 1\n1\n' | "$B" "$T/p.session" 2>&1 > "$T/out.txt" | head -c 300)
case "$out" in "error[session]: a spanned table has 12 blocks: repetition 12 is past them"*) [ "$(wc -l < "$T/out.txt" | tr -d ' ')" = 1 ] && echo "ok   a repetition whose second block is past the table: refused, and the session ends" || { echo "FAIL past the table: the session went on"; fail=1; } ;; *) echo "FAIL a repetition past the spanned table: \`$out\`"; fail=1 ;; esac
same "a step at the last position, where the phase's two blocks would pass its table and the roll: the phase's tables are not the step's" "$(ask p.session "seek 11" 1)" "11;0 0 12 12 1;"
# without the spanned table, the roll's own refusal: the phase's second block is past it
printf '%s\n' "$OWN" "phase two 2" "program pa $CHX/step_add.chx n=4" "resident pa.w $T/z4.txt" "resident pa.d $T/z4.txt" "store pa.wo R 12" "program pb $CHX/step_add.chx n=6" "resident pb.w $T/z6.txt" "window pb.d R" "steps" "answer b.wo" "answer pb.wo" > "$T/q.session"
out=$(printf 'seek 11\ntwo 1\n' | "$B" "$T/q.session" 2>&1 > /dev/null | head -c 300)
case "$out" in "error[session]: the roll \`R\` has 12 blocks: repetition 12 is past them"*) echo "ok   a repetition whose second block is past the roll: refused" ;; *) echo "FAIL a repetition past the roll: \`$out\`"; fail=1 ;; esac
# inputs: a request gives the numbers of its phase's inputs, and no others
printf '%s\n' "program a $CHX/step_add.chx n=1" "resident a.w $T/z1.txt" "input a.d" "phase two 2" "program pa $CHX/step_add.chx n=2" "resident pa.w $T/z2.txt" "input pa.d" "steps" "answer a.wo" "answer pa.wo" > "$T/in.session"
same "a request's numbers are its phase's inputs" "$(ask in.session "1 5" "two 1 7 8" "1 6")" "5;7 8;6;"
out=$(printf 'two 1 7\n' | "$B" "$T/in.session" 2>&1 > /dev/null | head -c 300)
case "$out" in "error[session]: the request holds 1 numbers; the session takes 2"*) echo "ok   a number too few for the phase's inputs: refused" ;; *) echo "FAIL a number too few: \`$out\`"; fail=1 ;; esac
out=$(printf '1 7 8\n' | "$B" "$T/in.session" 2>&1 > /dev/null | head -c 300)
case "$out" in "error[session]: the request holds more than the 1 numbers the session takes"*) echo "ok   the phase's numbers given to the session's own programs: refused" ;; *) echo "FAIL the phase's numbers to a step: \`$out\`"; fail=1 ;; esac
out=$(printf 'three 1\n' | "$B" "$T/in.session" 2>&1 > /dev/null | head -c 300)
case "$out" in "error[session]: \`three\` is not a count of steps"*) echo "ok   a request that names no phase: refused" ;; *) echo "FAIL a request that names no phase: \`$out\`"; fail=1 ;; esac
out=$(printf 'two\n' | "$B" "$T/in.session" 2>&1 > /dev/null | head -c 300)
case "$out" in "error[session]: the request starts with the count of steps, and holds nothing"*) echo "ok   a phase's name and no count: refused" ;; *) echo "FAIL a phase's name and no count: \`$out\`"; fail=1 ;; esac
# a table that holds and one that goes round, in a phase: the block of the count
printf '%s\n' "program a $CHX/step_add.chx n=1" "resident a.w $T/z1.txt" "resident a.d $T/z1.txt" "phase two 2" "program ph $CHX/step_add.chx n=2" "resident ph.w $T/z2.txt" "table ph.d $T/tab.txt 12 hold" "program pr $CHX/step_add.chx n=2" "resident pr.w $T/z2.txt" "table pr.d $T/tab.txt 12" "steps" "answer a.wo" "answer ph.wo" "answer pr.wo" > "$T/h.session"
same "in a phase a table that holds and one that goes round take the block of the count" "$(ask h.session "two 1" "two 1" "seek 11" "two 1" "two 1")" "1 1 1 1;3 3 3 3;11;12 12 12 12;12 12 2 2;"
# a chosen mask in a phase of stride 2: two masks a repetition, each awk's mask of its own repetition (the rule and the scores of tests/choose.sh)
scores() { awk -v n="$1" 'BEGIN { for (s = 0; s < n; s++) { for (c = 0; c < n; c++) { v = (s % 7 == 3) ? 4 : ((s % 7 == 5) ? c : ((s % 7 == 6) ? n - c : (c * 7 + s * 13 + int(c * s / 3)) % 5)); printf "%d ", v } printf "\n" } }'; }
want() { # n, top, first, last, the scores' file: a line a step
  awk -v n="$1" -v top="$2" -v first="$3" -v last="$4" '{ j = NR - 1; for (c = 0; c < n; c++) m[c] = (c <= j) ? 0 : -1000000000
    if (j + 1 > top + first + last) { k = 0; for (c = first; c <= j - last; c++) { m[c] = -1000000000; idx[k] = c; val[k] = $(c + 1); k++ }
      for (a = 1; a < k; a++) { i = idx[a]; v = val[a]; b = a - 1; while (b >= 0 && val[b] < v) { idx[b + 1] = idx[b]; val[b + 1] = val[b]; b-- } idx[b + 1] = i; val[b + 1] = v }
      for (a = 0; a < top; a++) m[idx[a]] = 0 }
    for (c = 0; c < n; c++) printf "%s%d", (c ? " " : ""), m[c]; printf ";" }' "$5"; }
scores 16 > "$T/sc.txt"; awk 'BEGIN { for (c = 0; c < 32; c++) printf "0 "; printf "\n" }' > "$T/z32.txt"
chosen() { # stride-2 choose with top, first, last
  printf '%s\n' "program o $CHX/step_add.chx n=1" "resident o.w $T/z1.txt" "resident o.d $T/z1.txt" "phase two 2" "program a $CHX/step_add.chx n=32" "resident a.w $T/z32.txt" "table a.d $T/sc.txt 16 span" \
    "program b $CHX/step_add.chx n=32" "resident b.w $T/z32.txt" "choose b.d by a.wo $1 $2 $3" "steps" "answer o.wo" "answer b.wo" > "$T/c.session"
}
chosen 3 1 2
got=$(i=0; while [ $i -lt 8 ]; do echo "two 1"; i=$((i + 1)); done | "$B" "$T/c.session" 2> "$T/err.txt" | awk '{ for (i = 1; i <= NF; i++) { printf "%s%d", ((i - 1) % 16 ? " " : ""), $i; if (i % 16 == 0) printf ";" } }')
if [ "$got" = "$(want 16 3 1 2 "$T/sc.txt")" ] && [ -n "$got" ]; then echo "ok   a chosen mask in a phase of stride 2: eight repetitions' sixteen masks, awk's for each repetition of the session"; else echo "FAIL the chosen masks: $(head -c 200 "$T/err.txt") got $(echo "$got" | cut -c1-300)"; fail=1; fi
got=$(printf 'seek 5\ntwo 1\n' | "$B" "$T/c.session" 2> "$T/err.txt" | awk 'NF > 1 { for (i = 1; i <= NF; i++) { printf "%s%d", ((i - 1) % 16 ? " " : ""), $i; if (i % 16 == 0) printf ";" } }')
if [ "$got" = "$(want 16 3 1 2 "$T/sc.txt" | tr ';' '\n' | sed -n '6,7p' | tr '\n' ';')" ]; then echo "ok   from an odd repetition: the masks of repetitions 5 and 6, the first of which reads no scores and the second does"; else echo "FAIL the masks from repetition 5: $got"; fail=1; fi
# the same edge where the second mask's choice is not the first entries (there a session that had read no scores would choose the same): the 2 largest, the first 1 and the last 1,
# from repetition 3, whose mask reads no scores, in a session that has read none yet
chosen 2 1 1
got=$(printf 'seek 3\ntwo 1\n' | "$B" "$T/c.session" 2> "$T/err.txt" | awk 'NF > 1 { for (i = 1; i <= NF; i++) { printf "%s%d", ((i - 1) % 16 ? " " : ""), $i; if (i % 16 == 0) printf ";" } }')
wantm=$(want 16 2 1 1 "$T/sc.txt" | tr ';' '\n' | sed -n '4,5p' | tr '\n' ';')
if [ "$got" = "$wantm" ] && [ "$(echo "$wantm" | cut -d';' -f2 | cut -d' ' -f1-5)" = "0 -1000000000 0 0 0" ]; then echo "ok   a session's first request, from repetition 3: the second mask is chosen by scores the first did not need (entries 2 and 3, not 1 and 2)"; else echo "FAIL the masks from repetition 3: $got, wanted $wantm"; fail=1; fi
chosen 3 1 2
out=$(printf 'seek 15\ntwo 1\n' | "$B" "$T/c.session" 2>&1 > /dev/null | head -c 300)
case "$out" in "error[session]: a spanned table has 16 blocks: repetition 16 is past them"*) echo "ok   a repetition whose second mask is past the scores: refused" ;; *) echo "FAIL past the masks: \`$out\`"; fail=1 ;; esac
# what a session file may not say
refused() { # name, the refusal's text, the session's lines
  name=$1; wanted=$2; shift 2
  for l in "$@"; do printf '%s\n' "$l"; done > "$T/r.session"
  out=$(printf '1\n' | "$B" "$T/r.session" 2>&1 > "$T/out.txt" | head -c 400)
  case "$out" in "error[session]: "*"$wanted"*) [ -s "$T/out.txt" ] && { echo "FAIL $name: refused, and answered"; fail=1; } || echo "ok   refused: $name" ;; *) echo "FAIL $name: \`$out\`"; fail=1 ;; esac
}
refused "phase with no stride" "phase NAME STRIDE" "$OWN" "phase two" "program pa $CHX/step_add.chx n=4" steps "answer b.wo"
refused "a stride of 0" "whole number from 1" "$OWN" "phase two 0" "program pa $CHX/step_add.chx n=4" steps "answer b.wo"
refused "a stride that is a fraction" "whole number from 1" "$OWN" "phase two 1.5" "program pa $CHX/step_add.chx n=4" steps "answer b.wo"
refused "a stride that is a word" "whole number from 1" "$OWN" "phase two x" "program pa $CHX/step_add.chx n=4" steps "answer b.wo"
refused "a phase named by a number" "cannot name a phase" "$OWN" "phase 2 2" "program pa $CHX/step_add.chx n=4" steps "answer b.wo"
refused "a phase named seek" "cannot name a phase" "$OWN" "phase seek 2" "program pa $CHX/step_add.chx n=4" steps "answer b.wo"
refused "a phase named checkpoint" "cannot name a phase" "$OWN" "phase checkpoint 2" "program pa $CHX/step_add.chx n=4" steps "answer b.wo"
refused "a phase named twice" "is named twice" "$OWN" "$TWO" "phase two 2" "program pd $CHX/step_add.chx n=1" "resident pd.w $T/z1.txt" "resident pd.d $T/z1.txt" "$END" "answer pd.wo"
refused "a phase before any program" "follows no program" "phase two 2" "program pa $CHX/step_add.chx n=1" "resident pa.w $T/z1.txt" "resident pa.d $T/z1.txt" steps "answer pa.wo"
refused "a phase with no program between it and the next" "follows no program" "$OWN" "phase one 1" "phase two 2" "program pa $CHX/step_add.chx n=1" "resident pa.w $T/z1.txt" "resident pa.d $T/z1.txt" steps "answer b.wo" "answer pa.wo"
refused "a last phase with no program" "has no program" "$OWN" "phase two 2" steps "answer b.wo"
refused "a phase with no answer" "the phase \`two\` has no answer" "$OWN" "$TWO" steps "answer b.wo"
refused "the session's own programs with no answer" "before the first phase, have no answer" "$OWN" "$TWO" steps "answer pb.wo"
refused "a phase in a session with no steps" "a session file with \`steps\`" "program a $CHX/step_add.chx n=1" "resident a.w $T/z1.txt" "resident a.d $T/z1.txt" "phase two 2" "program pa $CHX/step_add.chx n=1" "resident pa.w $T/z1.txt" "resident pa.d $T/z1.txt" "answer a.wo" "answer pa.wo"
X="program x $CHX/step_add.chx n=1"; XW="resident x.w $T/z1.txt"; XD="resident x.d $T/z1.txt"; Y="program y $CHX/step_add.chx n=1"; YW="resident y.w $T/z1.txt"; YD="resident y.d $T/z1.txt"
refused "a link from the session's own program into a phase" "a link joins \`x\` and \`y\`, which are of two phases" "$X" "$XW" "$XD" "phase two 2" "$Y" "$YW" "link x.wo -> y.d" steps "answer x.wo" "answer y.wo"
refused "a carry from a phase to the session's own program" "a carry joins \`y\` and \`x\`, which are of two phases" "$X" "$XW" "$XD" "phase two 2" "$Y" "$YW" "$YD" "carry y.wo -> x.w" steps "answer x.wo" "answer y.wo"
refused "an alias across phases" "an alias joins \`x\` and \`y\`, which are of two phases" "$X" "$XW" "$XD" "program x2 $CHX/step_add.chx n=1" "link x.wo -> x2.w" "resident x2.d $T/z1.txt" "phase two 2" "$Y" "$YW" "$YD" "program y2 $CHX/step_add.chx n=1" "link y.wo -> y2.w" "resident y2.d $T/z1.txt" "alias x.wo y.wo" steps "answer x2.wo" "answer y2.wo"
refused "a table picked by a program of another phase" "a pick joins \`x\` and \`y\`, which are of two phases" "$X" "$XW" "$XD" "phase two 2" "$Y" "$YW" "table y.d $T/tab.txt 24 by x.wo 0" steps "answer x.wo" "answer y.wo"
refused "a mask chosen by a program of another phase" "a choice joins \`x\` and \`y\`, which are of two phases" "$X" "$XW" "$XD" "phase two 2" "$Y" "$YW" "choose y.d by x.wo 1 0 0" steps "answer x.wo" "answer y.wo"
refused "a roll stored twice in a phase" "is stored twice" "$OWN" "$TWO" "program pd $CHX/step_add.chx n=4" "resident pd.w $T/z4.txt" "resident pd.d $T/z4.txt" "store pd.wo R 12" "$END"
refused "a roll stored in a phase with other blocks" "with other blocks, or blocks of another size" "$OWN" "phase two 2" "program pa $CHX/step_add.chx n=4" "resident pa.w $T/z4.txt" "resident pa.d $T/z4.txt" "store pa.wo R 10" "program pb $CHX/step_add.chx n=6" "resident pb.w $T/z6.txt" "window pb.d R" steps "answer b.wo" "answer pb.wo"
refused "a roll stored in a phase with blocks of another size" "with other blocks, or blocks of another size" "$OWN" "phase two 2" "program pa $CHX/step_add.chx n=6" "resident pa.w $T/z6.txt" "resident pa.d $T/z6.txt" "store pa.wo R 12" "program pb $CHX/step_add.chx n=6" "resident pb.w $T/z6.txt" "window pb.d R" steps "answer b.wo" "answer pb.wo"
refused "a stored output that is no whole number of the stride" "a stored output is a block for each of its phase's stride" "$OWN" "phase three 3" "program pa $CHX/step_add.chx n=4" "resident pa.w $T/z4.txt" "resident pa.d $T/z4.txt" "store pa.wo R 12" steps "answer b.wo" "answer pa.wo"
refused "a window of a roll its phase does not store" "is not stored in the window's phase" "$OWN" "phase two 2" "program pb $CHX/step_add.chx n=6" "resident pb.w $T/z6.txt" "window pb.d R" steps "answer b.wo" "answer pb.wo"
refused "span on a picked table" "table PROGRAM.PARAM FILE BLOCKS" "$OWN" "phase two 2" "$Y" "$YW" "$YD" "program pa $CHX/step_add.chx n=4" "resident pa.w $T/z4.txt" "table pa.d $T/tab.txt 12 by y.wo 0 span" steps "answer b.wo" "answer pa.wo"
refused "span and hold" "positive whole number" "$OWN" "phase two 2" "program pa $CHX/step_add.chx n=4" "resident pa.w $T/z4.txt" "table pa.d $T/tab.txt hold span" steps "answer b.wo" "answer pa.wo"
refused "a spanned parameter that is no whole number of the stride" "a spanned parameter is a whole number of blocks" "$OWN" "phase three 3" "program pa $CHX/step_add.chx n=4" "resident pa.w $T/z4.txt" "table pa.d $T/tab.txt 12 span" steps "answer b.wo" "answer pa.wo"
refused "a chosen mask that is no whole number of the stride" "a chosen mask is a row for each of its phase's stride" "$OWN" "phase three 3" "program pa $CHX/step_add.chx n=4" "resident pa.w $T/z4.txt" "resident pa.d $T/z4.txt" "program pb $CHX/step_add.chx n=4" "resident pb.w $T/z4.txt" "choose pb.d by pa.wo 1 0 0" steps "answer b.wo" "answer pb.wo"
if [ $fail -ne 0 ]; then echo "FAIL phase"; exit 1; fi
echo "phase: ok"
