#!/bin/sh
# tests/choose.sh [session binary] -- a mask the session chooses (docs/resident.md section 34; docs/kolibri.md section 19.5), on the CPU session (the gate) or, given its binary, the
# Metal one (by hand):
#     choose P.PARAM by Q.OUT TOP FIRST LAST
# P.PARAM is written when P is about to run in repetition j: entry c is 0 when c <= j and c < FIRST, or c > j - LAST, or among the TOP largest of Q.OUT over the other entries to j
# (a tie to the lower c); every other entry is -1e9. One program copies a row of a table (the scores of the step); another adds the chosen mask to zeros and answers it. The scores are
# small whole numbers with many ties, a row a step, and awk says the mask by sorting: another way to the same rule than the session's selection. Then every refusal.
#     choose P.PARAM by Q.OUT TOP FIRST LAST index
# (docs/kolibri.md section 23) writes the same choice as a list: the entries the mask leaves at 0, their numbers rising, and -1 in the rest of P.PARAM's room. Held to the list
# awk makes of its own mask.
set -u
cd "$(dirname "$0")/.."
B="${1:-build/cancho-gpu-session}"
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'st=$?; rm -rf "$T"; exit $st' EXIT
fail=0
CHX=$(pwd)/tests/chx
# the scores: N numbers a step for N steps, from a few values so that ties are the rule; a row of one value, a rising row and a falling one among them
scores() { awk -v n="$1" 'BEGIN { for (s = 0; s < n; s++) { for (c = 0; c < n; c++) { v = (s % 7 == 3) ? 4 : ((s % 7 == 5) ? c : ((s % 7 == 6) ? n - c : (c * 7 + s * 13 + int(c * s / 3)) % 5)); printf "%d ", v } printf "\n" } }'; }
zeros() { awk -v n="$1" 'BEGIN { for (c = 0; c < n; c++) printf "0 "; printf "\n" }'; }
# awk's mask of every step: the candidates sorted by score, the larger first and of equals the earlier
want() { # n, top, first, last, the scores' file: a line a step
  awk -v n="$1" -v top="$2" -v first="$3" -v last="$4" '{ j = NR - 1; for (c = 0; c < n; c++) m[c] = (c <= j) ? 0 : -1000000000
    if (j + 1 > top + first + last) { k = 0; for (c = first; c <= j - last; c++) { m[c] = -1000000000; idx[k] = c; val[k] = $(c + 1); k++ }
      for (a = 1; a < k; a++) { i = idx[a]; v = val[a]; b = a - 1; while (b >= 0 && val[b] < v) { idx[b + 1] = idx[b]; val[b + 1] = val[b]; b-- } idx[b + 1] = i; val[b + 1] = v }
      for (a = 0; a < top; a++) m[idx[a]] = 0 }
    for (c = 0; c < n; c++) printf "%s%d", (c ? " " : ""), m[c]; printf ";" }' "$5"; }
run() { # n, top, first, last: the session answers the mask of each of n steps
  n=$1; scores "$n" > "$T/sc.txt"; zeros "$n" > "$T/z.txt"
  printf 'program a %s/step_add.chx n=%s\nresident a.w %s/z.txt\ntable a.d %s/sc.txt %s\nprogram b %s/step_add.chx n=%s\nresident b.w %s/z.txt\nchoose b.d by a.wo %s %s %s\nsteps\nanswer b.wo\n' "$CHX" "$n" "$T" "$T" "$n" "$CHX" "$n" "$T" "$2" "$3" "$4" > "$T/c.session"
  i=0; while [ $i -lt "$n" ]; do echo 1; i=$((i + 1)); done | "$B" "$T/c.session" 2> "$T/err.txt" | awk '{ for (i = 1; i <= NF; i++) printf "%s%d", (i > 1 ? " " : ""), $i; printf ";" }' > "$T/got.txt"
  want "$n" "$2" "$3" "$4" "$T/sc.txt" > "$T/want.txt"
  if cmp -s "$T/got.txt" "$T/want.txt" && [ -s "$T/got.txt" ]; then echo "ok   $5"; else echo "FAIL $5: $(head -c 300 "$T/err.txt") got $(cut -c1-200 "$T/got.txt") wanted $(cut -c1-200 "$T/want.txt")"; fail=1; fi
}
run 16 3 1 2 "16 steps of 16 scores, the 3 largest with the first 1 and the last 2: awk's mask at every step"
run 16 1 0 0 "the one largest, none at the start or the end"
run 16 5 0 3 "the 5 largest and the last 3"
run 16 4 4 0 "the 4 largest and the first 4"
run 16 14 1 1 "as many taken as there are (16 of 16): every entry to the step, the scores never read"
run 16 30 0 0 "more largest than entries"
run 32 6 2 5 "32 steps of 32 scores, the 6 largest, the first 2, the last 5"
run 64 9 3 7 "64 steps of 64 scores, the 9 largest, the first 3, the last 7"
# the cases must have what they are for: in the first, steps where the edge of the choice is a tie (the score of the last taken is also the score of one not taken) and steps where it is not
scores 16 > "$T/sc.txt"; want 16 3 1 2 "$T/sc.txt" | tr ';' '\n' > "$T/want_lines.txt"
ties=$(paste -d'|' "$T/sc.txt" "$T/want_lines.txt" | awk -F'|' '{ j = NR - 1; if (j + 1 <= 6) next; n = split($1, s, " "); split($2, m, " "); lo = 1e9; hi = -1e9; for (c = 2; c <= j - 1; c++) { if (m[c] == 0 && s[c] < lo) lo = s[c]; if (m[c] != 0 && s[c] > hi) hi = s[c] } if (lo == hi) t++; else u++ } END { print t + 0, u + 0 }')
case "$ties" in "0 "*|*" 0") echo "FAIL the scores of the first case have $ties steps with and without a tie at the edge: one of the two is not tested"; fail=1 ;; *) echo "ok   of its steps that choose, ${ties% *} have a tie at the edge and ${ties#* } have none" ;; esac
# with a seek, the mask is the repetition's again; and a request of several steps answers the last one's
scores 16 > "$T/sc.txt"; zeros 16 > "$T/z.txt"
printf 'program a %s/step_add.chx n=16\nresident a.w %s/z.txt\ntable a.d %s/sc.txt 16\nprogram b %s/step_add.chx n=16\nresident b.w %s/z.txt\nchoose b.d by a.wo 3 1 2\nsteps\nanswer b.wo\n' "$CHX" "$T" "$T" "$CHX" "$T" > "$T/c.session"
want 16 3 1 2 "$T/sc.txt" | tr ';' '\n' > "$T/want_lines.txt"
got=$(printf '12\nseek 9\n1\nseek 2\n1\n' | "$B" "$T/c.session" 2> "$T/err.txt" | awk 'NF > 1 { for (i = 1; i <= NF; i++) printf "%s%d", (i > 1 ? " " : ""), $i; printf ";" }')
if [ "$got" = "$(sed -n 12p "$T/want_lines.txt");$(sed -n 10p "$T/want_lines.txt");$(sed -n 3p "$T/want_lines.txt");" ]; then echo "ok   a request of 12 steps answers the twelfth's mask; after seek 9 the tenth's, after seek 2 the third's"; else echo "FAIL seeks: $got $(head -c 200 "$T/err.txt")"; fail=1; fi
out=$(printf '16\n1\n' | "$B" "$T/c.session" 2>&1 > /dev/null | head -c 300)
case "$out" in "error[session]: a chosen mask has 16 entries: repetition 16 is past them"*) echo "ok   a step past the mask's entries: refused" ;; *) echo "FAIL a step past the mask: \`$out\`"; fail=1 ;; esac
# what a session file may not say
refused() { # name, the refusal's text, the session's lines
  name=$1; wanted=$2; shift 2
  for l in "$@"; do printf '%s\n' "$l"; done > "$T/r.session"
  out=$(printf '1\n' | "$B" "$T/r.session" 2>&1 > "$T/out.txt" | head -c 400)
  case "$out" in "error[session]: "*"$wanted"*) [ -s "$T/out.txt" ] && { echo "FAIL $name: refused, and answered"; fail=1; } || echo "ok   refused: $name" ;; *) echo "FAIL $name: \`$out\`"; fail=1 ;; esac
}
A="program a $CHX/step_add.chx n=16"; AW="resident a.w $T/z.txt"; AD="table a.d $T/sc.txt 16"; Bp="program b $CHX/step_add.chx n=16"; BW="resident b.w $T/z.txt"; CH="choose b.d by a.wo 3 1 2"
zeros 8 > "$T/z8.txt"
refused "choose with a word missing" "choose PROGRAM.PARAM by PROGRAM.OUT TOP FIRST LAST" "$A" "$AW" "$AD" "$Bp" "$BW" "choose b.d by a.wo 3 1" steps "answer b.wo"
refused "choose without by" "choose PROGRAM.PARAM by PROGRAM.OUT TOP FIRST LAST" "$A" "$AW" "$AD" "$Bp" "$BW" "choose b.d of a.wo 3 1 2" steps "answer b.wo"
refused "none of the largest taken" "whole numbers" "$A" "$AW" "$AD" "$Bp" "$BW" "choose b.d by a.wo 0 1 2" steps "answer b.wo"
refused "a count that is a fraction" "whole numbers" "$A" "$AW" "$AD" "$Bp" "$BW" "choose b.d by a.wo 3 1.5 2" steps "answer b.wo"
refused "a count that is negative" "whole numbers" "$A" "$AW" "$AD" "$Bp" "$BW" "choose b.d by a.wo 3 1 -2" steps "answer b.wo"
refused "a count that is a word" "whole numbers" "$A" "$AW" "$AD" "$Bp" "$BW" "choose b.d by a.wo x 1 2" steps "answer b.wo"
refused "a mask that is no PROGRAM.PARAM" "choose takes PROGRAM.PARAM and PROGRAM.OUT" "$A" "$AW" "$AD" "$Bp" "$BW" "choose bd by a.wo 3 1 2" steps "answer b.wo"
refused "a mask of a program that is not there" "is not a program" "$A" "$AW" "$AD" "$Bp" "$BW" "choose c.d by a.wo 3 1 2" steps "answer b.wo"
refused "a mask that is an output" "a mask is chosen for one" "$A" "$AW" "$AD" "$Bp" "$BW" "resident b.d $T/z.txt" "choose b.wo by a.wo 3 1 2" steps "answer b.wo"
refused "a mask supplied another way too" "is supplied twice" "$A" "$AW" "$AD" "$Bp" "$BW" "resident b.d $T/z.txt" "$CH" steps "answer b.wo"
refused "scores that are no output" "chosen by numbers a kernel wrote" "$A" "$AW" "$AD" "$Bp" "$BW" "choose b.d by a.w 3 1 2" steps "answer b.wo"
refused "scores of a program listed after the mask's" "not listed before it" "$Bp" "$BW" "choose b.d by a.wo 3 1 2" "$A" "$AW" "$AD" steps "answer b.wo"
refused "scores of another length than the mask" "the mask has 8 entries and the output that chooses it 16" "$A" "$AW" "$AD" "program b $CHX/step_add.chx n=8" "resident b.w $T/z8.txt" "$CH" steps "answer b.wo"
refused "choose in a session with no steps" "a session file with \`steps\`" "$A" "$AW" "resident a.d $T/z.txt" "$Bp" "$BW" "$CH" "answer b.wo"
# --- the choice as a list
runlist() { # n, top, first, last, room, name: the session answers the list of each of n steps
  n=$1; scores "$n" > "$T/sc.txt"; zeros "$n" > "$T/z.txt"; zeros "$5" > "$T/zr.txt"
  printf 'program a %s/step_add.chx n=%s\nresident a.w %s/z.txt\ntable a.d %s/sc.txt %s\nprogram b %s/step_add.chx n=%s\nresident b.w %s/zr.txt\nchoose b.d by a.wo %s %s %s index\nsteps\nanswer b.wo\n' "$CHX" "$n" "$T" "$T" "$n" "$CHX" "$5" "$T" "$2" "$3" "$4" > "$T/c.session"
  i=0; while [ $i -lt "$n" ]; do echo 1; i=$((i + 1)); done | "$B" "$T/c.session" 2> "$T/err.txt" | awk '{ for (i = 1; i <= NF; i++) printf "%s%d", (i > 1 ? " " : ""), $i; printf ";" }' > "$T/got.txt"
  want "$n" "$2" "$3" "$4" "$T/sc.txt" | tr ';' '\n' | awk -v room="$5" 'NF { k = 0; out = ""; for (c = 1; c <= NF; c++) if ($c == 0) { out = out (k ? " " : "") (c - 1); k++ } for (; k < room; k++) out = out (k ? " " : "") "-1"; printf "%s;", out }' > "$T/want.txt"
  if cmp -s "$T/got.txt" "$T/want.txt" && [ -s "$T/got.txt" ]; then echo "ok   $6"; else echo "FAIL $6: $(head -c 300 "$T/err.txt") got $(cut -c1-200 "$T/got.txt") wanted $(cut -c1-200 "$T/want.txt")"; fail=1; fi
}
runlist 16 3 1 2 6 "as a list, 16 steps: the 3 largest, the first and the last 2, in rising order, and -1 where there are fewer; room for exactly the 6"
runlist 16 3 1 2 9 "the same with room for 9: the rest is -1"
runlist 32 6 2 5 16 "32 steps, the 6 largest, the first 2, the last 5, room for 16"
runlist 16 5 0 3 8 "the 5 largest and the last 3"
runlist 16 1 0 0 1 "the one largest: a list of one"
zeros 5 > "$T/z5.txt"; zeros 6 > "$T/z6.txt"
refused "a list with less room than can be chosen" "with room for the largest taken, those at the start and those at the end" "$A" "$AW" "$AD" "program b $CHX/step_add.chx n=5" "resident b.w $T/z5.txt" "choose b.d by a.wo 3 1 2 index" steps "answer b.wo"
refused "an eighth word that is not index" "choose PROGRAM.PARAM by PROGRAM.OUT TOP FIRST LAST [index]" "$A" "$AW" "$AD" "program b $CHX/step_add.chx n=6" "resident b.w $T/z6.txt" "choose b.d by a.wo 3 1 2 list" steps "answer b.wo"
refused "a ninth word" "choose PROGRAM.PARAM by PROGRAM.OUT TOP FIRST LAST [index]" "$A" "$AW" "$AD" "program b $CHX/step_add.chx n=6" "resident b.w $T/z6.txt" "choose b.d by a.wo 3 1 2 index index" steps "answer b.wo"
out=$(printf '16\n1\n' | "$B" "$T/c.session" 2>&1 > /dev/null | head -c 300)
case "$out" in "error[session]: a chosen mask has 16 entries: repetition 16 is past them"*) echo "ok   a step past the scores of a list: refused" ;; *) echo "FAIL a step past a list's scores: \`$out\`"; fail=1 ;; esac
if [ $fail -ne 0 ]; then echo "FAIL choose"; exit 1; fi
echo "choose: ok"
