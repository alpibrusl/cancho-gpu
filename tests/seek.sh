#!/bin/sh
# tests/seek.sh [session binary] -- the request `seek N` (docs/kolibri.md section 17, S1) on the CPU session (the gate) or, given its binary, the Metal one (by hand): the next repetition is
# repetition N, so a table is read from block N mod its blocks on; no program runs and what was carried stays. One program adds a table's block to a row: without a carry the answer is the
# block, with one it is the sum so far. The values are small whole numbers, so the answers are exact on both sessions. Then every refusal.
set -eu
cd "$(dirname "$0")/.."
B="${1:-build/cancho-gpu-session}"
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
fail=0
CHX=$(pwd)/tests/chx
printf '1 1\n10 10\n100 100\n' > "$T/tab.txt"
printf '0 0\n' > "$T/w.txt"
printf 'program a %s/step_add.chx n=2\nresident a.w %s/w.txt\ntable a.d %s/tab.txt 3\nsteps\nanswer a.wo\n' "$CHX" "$T" "$T" > "$T/plain.session"
printf 'program a %s/step_add.chx n=2\nresident a.w %s/w.txt\ntable a.d %s/tab.txt 3\ncarry a.wo -> a.w\nsteps\nanswer a.wo\n' "$CHX" "$T" "$T" > "$T/carry.session"
printf 'program a %s/step_add.chx n=2\nresident a.w %s/w.txt\ninput a.d\nanswer a.wo\n' "$CHX" "$T" > "$T/nosteps.session"
# the answers of a session to the lines given, as whole numbers, a request a line
ask() { s=$1; shift; printf '%s\n' "$@" | "$B" "$s" 2> "$T/err.txt" | awk '{ for (i = 1; i <= NF; i++) printf "%s%d", (i > 1 ? " " : ""), $i; printf ";" }'; }
same() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: \`$2\`, wanted \`$3\` $(head -c 200 "$T/err.txt")"; fail=1; fi; }
same "a table read in order, without a seek" "$(ask "$T/plain.session" 1 1 1 1)" "1 1;10 10;100 100;1 1;"
same "seek 0 starts the table again, and answers the repetition number" "$(ask "$T/plain.session" 1 1 "seek 0" 1 1)" "1 1;10 10;0;1 1;10 10;"
same "seek to a later repetition" "$(ask "$T/plain.session" 1 "seek 2" 1 1)" "1 1;2;100 100;1 1;"
same "seek past the table's blocks wraps as the count does" "$(ask "$T/plain.session" "seek 5" 1 "seek 3000000" 1)" "5;100 100;3000000;1 1;"
same "a request of several steps after a seek" "$(ask "$T/plain.session" 3 "seek 1" 2)" "100 100;1;100 100;"
same "a seek to where the session is changes nothing" "$(ask "$T/plain.session" 1 "seek 1" 1)" "1 1;1;10 10;"
same "what was carried stays over a seek" "$(ask "$T/carry.session" 1 1 "seek 0" 1 1 1)" "1 1;11 11;0;12 12;22 22;122 122;"
same "a seek before any step" "$(ask "$T/carry.session" "seek 1" 1 1)" "1;10 10;110 110;"
same "two seeks in a row: the last one counts" "$(ask "$T/plain.session" "seek 2" "seek 1" 1)" "2;1;10 10;"
same "  seek with blanks around it" "$(ask "$T/plain.session" "   seek   2  " 1)" "2;100 100;"
refused() { # name, session, line, the refusal's text
  out=$(printf '1\n%s\n1\n' "$3" | "$B" "$2" 2>&1 > "$T/out.txt" | head -c 400) && rc=0 || rc=$?
  case "$out" in "error[session]: $4"*) if [ "$(wc -l < "$T/out.txt" | tr -d ' ')" -le 1 ]; then echo "ok   $1: refused"; else echo "FAIL $1: refused, and the session went on"; fail=1; fi ;; *) echo "FAIL $1: \`$out\`"; fail=1 ;; esac
}
refused "seek with no number" "$T/plain.session" "seek" "seek N takes one repetition number"
refused "seek with two numbers" "$T/plain.session" "seek 1 2" "seek N takes one repetition number"
refused "seek to a negative repetition" "$T/plain.session" "seek -1" "\`-1\` is not a repetition number"
refused "seek to a fraction" "$T/plain.session" "seek 1.5" "\`1.5\` is not a repetition number"
refused "seek to a word" "$T/plain.session" "seek x" "\`x\` is not a repetition number"
refused "seek to 10^9" "$T/plain.session" "seek 1000000000" "\`1000000000\` is not a repetition number"
out=$(printf 'seek 0\n' | "$B" "$T/nosteps.session" 2>&1 | head -c 300) || true
case "$out" in "error[session]: seek is for a session file with \`steps\`"*) echo "ok   seek in a session with no steps: refused" ;; *) echo "FAIL seek in a session with no steps: \`$out\`"; fail=1 ;; esac
if [ $fail -ne 0 ]; then echo "FAIL seek"; exit 1; fi
echo "seek: ok"
