#!/bin/sh
# tests/kolibri_cmd.sh -- the command of Kolibri 1.1 (scripts/kolibri, docs/kolibri.md section 16) without the model and without a GPU:
#  1. `kolibri_gpu prompt` writes, for every case of tests/kolibri_prompt/ (X.user, X.effort and perhaps X.system), the bytes of golden/kolibri/X.txt: what the model's own chat template
#     gives through jinja2, recorded by hand;
#  2. `kolibri_gpu ends` on the toy GGUF of tests/kolibri_toy.sh, `kolibri_gpu say` on quoted lines, and their refusals, each with its rule;
#  3. scripts/kolibri itself with the toy GGUF, a tokenizer that is a stub (it prints the ids it is told to, and "detokenizes" ids to themselves joined by `-`) and a session that is a stub
#     (it answers the ids it is told to, a step each, as a float's text): what it asks of the session, where it stops, what it prints, and what it refuses.
# What the model answers is not here: that is by hand, on the Metal GPU.
# KOLIBRI_CMD=<a script in scripts/> runs another script: the mutants of the command, by hand.
set -eu
cd "$(dirname "$0")/.."
C="${CANCHO:-cancho}"
CANCHO="$C" scripts/build-kolibri.sh
B=build/kolibri_gpu
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
fail=0
. tests/kolibri_toy.sh
mkdir "$T/want"
toy "$T/toy.gguf" none
refuse() { # name, rule, status, the command
  name=$1; rule=$2; want=$3; shift 3
  status=0; "$@" > "$T/out.txt" 2> "$T/err.txt" || status=$?
  if [ "$status" -eq "$want" ] && grep -q "^error\[$rule\]" "$T/err.txt"; then echo "ok   refused ($rule): $name"; else echo "FAIL $name: status $status, wanted $want and error[$rule]: $(head -c 300 "$T/err.txt")"; fail=1; fi
}

# 1. the template
n=0; bad=0
for u in tests/kolibri_prompt/*.user; do
  b=${u%.user}; name=$(basename "$b"); n=$((n + 1))
  if [ -f "$b.system" ]; then "$B" prompt "$u" "$(cat "$b.effort")" "$b.system" > "$T/p.txt" 2> "$T/err.txt" || true; else "$B" prompt "$u" "$(cat "$b.effort")" > "$T/p.txt" 2> "$T/err.txt" || true; fi
  cmp -s "$T/p.txt" "golden/kolibri/$name.txt" || { echo "FAIL prompt $name: not the template's bytes: $(head -c 200 "$T/err.txt")"; bad=1; fail=1; }
done
[ "$n" -ge 9 ] || { echo "FAIL only $n cases of the template"; bad=1; fail=1; }
[ $bad -eq 0 ] && echo "ok   prompt: $n cases, each the bytes of the model's own template"
refuse "an effort the template has not" kolibri-usage 1 "$B" prompt tests/kolibri_prompt/plain.user highest
refuse "a text that is not there" kolibri-read 1 "$B" prompt "$T/nothing.txt" high
refuse "a system message that is not there" kolibri-read 1 "$B" prompt tests/kolibri_prompt/plain.user high "$T/nothing.txt"
refuse "prompt with no effort" kolibri-usage 2 "$B" prompt tests/kolibri_prompt/plain.user

# 2. the end tokens, and a quoted line's bytes
if [ "$("$B" ends "$T/toy.gguf" 2> "$T/err.txt")" = "5 6 136" ]; then echo "ok   ends: the toy's eos, its padding token and the most steps"; else echo "FAIL ends: $("$B" ends "$T/toy.gguf" 2>&1 | head -c 200)"; fail=1; fi
toy "$T/v.gguf" noeos
refuse "a file with no eos token" kolibri-meta 1 "$B" ends "$T/v.gguf"
toy "$T/v.gguf" notop
refuse "ends on a file with no indexer.top_k" kolibri-meta 1 "$B" ends "$T/v.gguf"
printf 'dec "a \\"b\\" \\\\ c\\x0a\\xc3\\xa4\\x00z"\n' > "$T/q.txt"
printf 'a "b" \\ c\n\303\244\000z' > "$T/qwant.txt"
if "$B" say "$T/q.txt" > "$T/qgot.txt" 2> "$T/err.txt" && cmp -s "$T/qgot.txt" "$T/qwant.txt"; then echo "ok   say: a quote, a backslash, a newline, two bytes of UTF-8 and a zero byte"; else echo "FAIL say: $(od -c "$T/qgot.txt" | head -3) $(head -c 200 "$T/err.txt")"; fail=1; fi
printf 'dec ""\n' > "$T/q.txt"
if "$B" say "$T/q.txt" > "$T/qgot.txt" 2> "$T/err.txt" && [ ! -s "$T/qgot.txt" ]; then echo "ok   say: an empty text"; else echo "FAIL say of an empty text"; fail=1; fi
printf 'dec "a\\q"\n' > "$T/q.txt";      refuse "an escape that is not written" kolibri-read 1 "$B" say "$T/q.txt"
printf 'dec "a\\x4"\n' > "$T/q.txt";     refuse "a hexadecimal escape cut short" kolibri-read 1 "$B" say "$T/q.txt"
printf 'dec "a\\xZZb"\n' > "$T/q.txt";   refuse "a hexadecimal escape that is not one" kolibri-read 1 "$B" say "$T/q.txt"
printf 'dec "a\\xC3\\xA4"\n' > "$T/q.txt"; refuse "a hexadecimal escape in capitals, which is not written" kolibri-read 1 "$B" say "$T/q.txt"
printf 'dec "a"b"\n' > "$T/q.txt";       refuse "a quote inside the line" kolibri-read 1 "$B" say "$T/q.txt"
printf 'dec "a\\"\n' > "$T/q.txt";       refuse "a backslash at the end" kolibri-read 1 "$B" say "$T/q.txt"
printf 'dec "a\n' > "$T/q.txt";          refuse "a line with no closing quote" kolibri-read 1 "$B" say "$T/q.txt"
printf 'ids 3: 1 2 3\n' > "$T/q.txt";    refuse "a line that is not a decoded one" kolibri-read 1 "$B" say "$T/q.txt"
printf 'dec "a"' > "$T/q.txt";           refuse "a line with no end" kolibri-read 1 "$B" say "$T/q.txt"
: > "$T/q.txt";                          refuse "an empty file" kolibri-read 1 "$B" say "$T/q.txt"
refuse "a file that is not there" kolibri-read 1 "$B" say "$T/nothing.txt"

# 3. the script. The stubs: a tokenizer that prints STUB_IDS for any text (and keeps the text), and prints ids back joined by `-`; a session that answers STUB_ANS, an id a step
cat > "$T/tok" <<'EOS'
#!/bin/sh
[ -z "${STUB_TOK_FAILS:-}" ] || { echo "error[tok]: the stub was told to fail"; exit 1; }
case "$1" in
  tokenize) cp "$3" "$STUB_DIR/tokenized.txt"; echo "tokenizer ids=64"; echo "ids $(echo $STUB_IDS | wc -w | tr -d ' '): $STUB_IDS"; echo 'dec "x"' ;;
  detokenize) printf 'dec "'; tr -s ' \n' '--' < "$3" | sed 's/-*$//' | tr -d '\n'; printf '"\n' ;;
esac
EOS
cat > "$T/session" <<'EOS'
#!/bin/sh
echo "$*" > "$STUB_DIR/args.txt"
[ -z "${STUB_SESSION_FAILS:-}" ] || { echo "error[session]: the stub was told to fail" >&2; exit 3; }
s=0; : > "$STUB_DIR/requests.txt"
while read -r line; do
  echo "$line" >> "$STUB_DIR/requests.txt"
  # a request is `N ID MASK [U]`: the answer is the id of the last of its steps
  s=$((s + ${line%% *}))
  echo $STUB_ANS | awk -v s="$s" '{ if ($s ~ /^[0-9]+$/) printf "%e\n", $s; else print $s }'
done
EOS
chmod +x "$T/tok" "$T/session"
mkdir "$T/w" "$T/cache" "$T/stub"; : > "$T/w/head.f32"
run() { # name, status wanted, the script's arguments: its standard output in $T/o.txt, its line in $T/e.txt
  name=$1; want=$2; shift 2
  status=0; KOLIBRI_GGUF="$T/toy.gguf" KOLIBRI_WEIGHTS="${WDIR:-$T/w}" KOLIBRI_SESSION="$T/session" KOLIBRI_TOK="$T/tok" CANCHO_MEDIA_CACHE="$T/cache" STUB_DIR="$T/stub" "${KOLIBRI_CMD:-scripts/kolibri}" "$@" > "$T/o.txt" 2> "$T/e.txt" || status=$?
  [ "$status" -eq "$want" ] || { echo "FAIL $name: status $status, wanted $want: $(head -c 300 "$T/e.txt")"; fail=1; return 1; }
}
is() { # name, file, the text it must be
  if [ "$(cat "$2")" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: $(head -c 200 "$2"), wanted $3 ($(head -c 200 "$T/e.txt"))"; fail=1; fi
}
has() { # name, file, a text it must have
  if grep -q -- "$3" "$2"; then echo "ok   $1"; else echo "FAIL $1: $(head -c 300 "$2")"; fail=1; fi
}
export STUB_IDS="3 0 7"
# the prompt's three steps answer 1, 1 and (the last, the first new token) 9; then 4; then the eos
export STUB_ANS="1 1 9 4 5 8 8 8"
set -- $("$B" draws 1 5)
run "to the eos" 0 "a question" && { is "the answer is the ids before the eos" "$T/o.txt" "9-4"; is "a request a step: the prompt's ids with a mask of 1, then the id the step before chose, each with the seed's number for its step" "$T/stub/requests.txt" "1 3 1 $1
1 0 1 $2
1 7 1 $3
1 0 0 $4
1 0 0 $5"; has "it says where it stopped" "$T/e.txt" "2 new tokens.*stopped at the end token 5"; has "the session samples, for as many steps as the toy sees whole (the prompt and 256 more would be past them)" "$T/stub/args.txt" "T=192 NS=136 GEN=1 SAMP=1 SAVE=0"; }
[ ! -e "$T/cache/gpu.lock" ] || { echo "FAIL the lock is left behind"; fail=1; }
export STUB_ANS="1 1 9 6 5"
run "to the padding token" 0 "a question" && { is "the padding token ends an answer too" "$T/o.txt" "9"; has "and is named" "$T/e.txt" "stopped at the end token 6"; }
export STUB_ANS="1 1 5 9 9"
run "an answer of nothing" 0 "a question" && { is "an end token first: an empty line" "$T/o.txt" ""; has "no new token" "$T/e.txt" "0 new tokens"; }
export STUB_ANS="1 1 9 4 8 8 8 8"
run "to the limit" 0 --max 2 --greedy "a question" && { is "the limit ends an answer, with all its ids" "$T/o.txt" "9-4"; is "the prompt's three steps and one past them, with no number to draw with" "$T/stub/requests.txt" "1 3 1
1 0 1
1 7 1
1 0 0"; has "the limit is named" "$T/e.txt" "stopped at the limit of 2 new tokens"; has "the session takes the arg max; its tables are for what the toy sees whole, whatever the run's length" "$T/stub/args.txt" "T=192 NS=136 GEN=1 SAMP=0 SAVE=0"; }
run "a limit under what the model sees whole" 0 --max 100 "a question" && has "the same tables for a longer run" "$T/stub/args.txt" "T=192 NS=136 "
export STUB_IDS="7"
export STUB_ANS="9 4 5"
run "a prompt of one token" 0 --greedy "a" && { is "its first step is the first new token" "$T/o.txt" "9-4"; is "the prompt's one step answers the first new token" "$T/stub/requests.txt" "1 7 1
1 0 0
1 0 0"; }
export STUB_IDS="3 0 7"
export STUB_ANS="1 1 9 5"
# the template, the raw text, a system message, a file
run "the template" 0 -o "$T/r1" "Was ist die Hauptstadt von Frankreich?" && { cmp -s "$T/stub/tokenized.txt" golden/kolibri/plain.txt && echo "ok   what is tokenized is the template around the text, effort high" || { echo "FAIL the tokenized text is not the template's"; fail=1; }; }
run "no reasoning" 0 --effort none "Was ist die Hauptstadt von Frankreich?" && { cmp -s "$T/stub/tokenized.txt" golden/kolibri/none.txt && echo "ok   --effort none: the template with reasoning disabled" || { echo "FAIL --effort none"; fail=1; }; }
run "a system message" 0 --effort none --system tests/kolibri_prompt/system.system "Wie heißt du?" && { cmp -s "$T/stub/tokenized.txt" golden/kolibri/system.txt && echo "ok   --system: the template with the system message" || { echo "FAIL --system"; fail=1; }; }
run "raw" 0 --raw "Berlin ist" && is "--raw: the text as it is" "$T/stub/tokenized.txt" "Berlin ist"
run "a file" 0 -f tests/kolibri_prompt/lines.user && { cmp -s "$T/stub/tokenized.txt" golden/kolibri/lines.txt && echo "ok   -f: the file's bytes in the template" || { echo "FAIL -f"; fail=1; }; }
run "a text that starts with a dash" 0 --raw -- "-x" && is "-- before a text that looks like an option" "$T/stub/tokenized.txt" "-x"
# the seed reaches the uniform numbers
run "seed 1" 0 -o "$T/r2" --seed 1 "a question" || true
run "seed 2" 0 -o "$T/r3" --seed 2 "a question" || true
if cmp -s "$T/r1/draws.txt" "$T/r2/draws.txt" && ! cmp -s "$T/r2/draws.txt" "$T/r3/draws.txt" && [ -s "$T/r3/draws.txt" ]; then echo "ok   the seed is 1 unless given, and another seed is other numbers"; else echo "FAIL the seed does not reach the numbers drawn with"; fail=1; fi
run "seed 2 again" 0 --seed 2 "a question" && has "and the seed's numbers are the requests'" "$T/stub/requests.txt" "^1 3 1 $("$B" draws 2 1)\$"
[ -f "$T/r1/answer_ids.txt" ] && [ -f "$T/r1/prompt.txt" ] && echo "ok   -o keeps the run's files" || { echo "FAIL -o keeps nothing"; fail=1; }
# the weights, written when they are not there
WDIR="$T/w2"
run "the first run" 0 "a question" && { [ -s "$T/w2/head.f32" ] && [ -f "$T/w2/a0.qkv.q4" ] && grep -q "writing the weights" "$T/e.txt" && echo "ok   weights that are not there are written, and it says so" || { echo "FAIL the weights are not written"; fail=1; }; }
WDIR="$T/w"
# the lock: a live job's is respected, a dead one's is not
echo $$ > "$T/cache/gpu.lock"
run "another job" 2 "a question" && has "a running GPU job stops it" "$T/e.txt" "another GPU job is running"
rm "$T/cache/gpu.lock"
sh -c 'echo $$' > "$T/cache/gpu.lock"
run "a dead job's lock" 0 "a question" && echo "ok   a lock whose job is gone is taken over"
# what it refuses
run "no text" 2 && has "no text is a usage mistake" "$T/e.txt" "no text"
run "two texts" 2 "a" "b" && has "two texts" "$T/e.txt" "one text"
run "max 0" 2 --max 0 "a" && has "--max 0" "$T/e.txt" "whole number of new tokens"
run "max x" 2 --max x "a" && has "--max that is no number" "$T/e.txt" "whole number of new tokens"
run "an effort" 2 --effort highest "a" && has "an effort the template has not" "$T/e.txt" "none, low, medium or high"
run "an option" 2 --fast "a" && has "an option that is not one" "$T/e.txt" "no option --fast"
run "raw with a system" 2 --raw --system tests/kolibri_prompt/system.system "a" && has "--raw with --system" "$T/e.txt" "no --system"
run "text and file" 2 -f tests/kolibri_prompt/plain.user "a" && has "a text and a file" "$T/e.txt" "not both"
run "a seed" 2 --seed 12345678 "a" && has "a seed of eight digits" "$T/e.txt" "at most 7 digits"
export STUB_IDS="$(awk 'BEGIN { for (i = 0; i < 137; i++) printf "1 " }')"
run "a prompt too long" 2 "a" && has "a prompt of more tokens than the step holds" "$T/e.txt" "137 tokens, more than the 136"
export STUB_IDS="3 0 7"
export STUB_SESSION_FAILS=1
run "a session that fails" 3 "a" && has "the failure of the session is passed on, with its words" "$T/e.txt" "the stub was told to fail"
unset STUB_SESSION_FAILS
export STUB_TOK_FAILS=1
run "a tokenizer that fails" 1 "a" && has "the refusal of the tokenizer is passed on" "$T/e.txt" "the stub was told to fail"
unset STUB_TOK_FAILS
export STUB_ANS="1 1 x"
run "an answer that is no number" 1 "a" && has "an answer that is no id stops it" "$T/e.txt" "which is no id"
[ "$fail" -eq 0 ] && echo "kolibri_cmd: ok"
exit $fail
