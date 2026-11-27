#!/bin/sh
# tests/kolibri_serve.sh -- the server of Kolibri 1.1 (kolibri/kserve.cho, docs/kolibri.md section 18) without the model and without a GPU: `cancho-gpu-kolibri-cpu`, the server on the CPU
# session's engine, with a session that is no model. Its "model" is two tables: the id a step answers is `next[the step's id] + position mod 2`, so an answer depends on the id the step was
# given and on where the session is, and awk says what it must be. The tokenizer is the toy of tests/tokenizer/ (its ids below 256 are bytes, so the answers are letters), the end tokens 858
# (`<|im_end|>`) and 859. Requests go through tests/httpc.cho, and what the server asked of the session is read from its `--trace`.
#  1. the routes, the key, the refusals, each with its status;
#  2. the prompt of every request of tests/kolibri_chat/ is the text of golden/kolibri_chat/ (the model's own chat template through jinja2, recorded by hand): the ids the session was given
#     are the tokenizer's ids of that text;
#  3. an answer: the ids after the prompt are awk's, it stops at either end token or at the limit, the counts are right, the head comes chunked;
#  4. the prefix kept: what the session is asked when a request repeats, continues, or parts from the last one; with a ring shorter than the context, when going back is taken and when not;
#  5. sampling: the numbers on the lines are `kolibri_gpu draws`' of the request's seed, at the steps' positions;
#  6. a session that refuses stops the server, and the body says so;
#  7. passes (docs/kolibri.md section 20): with `--pass 4` and a session that has the phase, a prompt's tokens go four a line while there are four and one more, from wherever the
#     session starts, and the answer is the one without passes.
#  8. tools (docs/kolibri.md section 21): with a session that answers a table's ids by position, so that an answer is any text: the calls of an answer in the body, its
#     `finish_reason`, what is no call, and the turn sent back kept by the session.
# What the model answers is not here: that is by hand, on the Metal GPU.
# KSERVE=<binary> tests another build of the server and builds nothing (build/kolibri_gpu and build/httpc are then there already): the mutants, by hand.
cd "$(dirname "$0")/.."
ROOT=$(pwd -P)
C="${CANCHO:-cancho}"
if [ -z "${KSERVE:-}" ]; then
  CANCHO="$C" scripts/build-kolibri-serve.sh cpu > /dev/null || { echo "FAIL kolibri_serve: build"; exit 1; }
  CANCHO="$C" scripts/build-kolibri.sh || { echo "FAIL kolibri_serve: build kolibri_gpu"; exit 1; }
  "$C" build tests/httpc.cho src/fileio.cho src/mem.cho src/text.cho --std -o build/httpc 2>&1 | grep -v "^ld: warning" || true
fi
[ -x build/httpc ] || { echo "FAIL kolibri_serve: build httpc"; exit 1; }
B=${KSERVE:-build/cancho-gpu-kolibri-cpu}
H=build/httpc
TOK=tests/tokenizer/toy-qwen2.gguf
T=$(cd "$(mktemp -d)" && pwd -P)
pid=""
stop() { [ -z "$pid" ] || { kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; }; pid=""; }
trap 'st=$?; stop; rm -rf "$T"; exit $st' EXIT
fail=0
CTX=2048
# the session that stands for the model
cat > "$T/toy.model" <<'EOM'
text CHX = tests/chx
text X = .
const V = 864
const NS = 2048
const SAMP = 0
const PASS = 0
text NEXT = next.txt
text PD = zero4.txt
program cur $CHX/row_mix.chx n=1 bn=1
input cur.xt
resident cur.xe $X/zero.txt
input cur.m
if SAMP
    program uu $CHX/step_add.chx n=1
    input uu.d
    resident uu.w $X/zero.txt
end
program nx $CHX/step_add.chx n=1
table nx.d $X/$NEXT $V by cur.y 0
table nx.w $X/pos.txt $NS
carry nx.wo -> cur.xe
if PASS
    phase pass 4
    program pc $CHX/step_add.chx n=4
    input pc.w
    resident pc.d $X/$PD
end
steps
answer nx.wo
if PASS
    answer pc.wo
end
EOM
echo 0 > "$T/zero.txt"; echo "0 0 0 0" > "$T/zero4.txt"; echo "1 1 1 1" > "$T/one4.txt"
# an end token after some letters only: the template ends in a newline, whose id would else end every chat answer at once
NEXT='function nx(i) { return (i % 7 == 3 && i >= 97) ? 858 : 97 + 2 * ((i * 5 + 1) % 12) }'
awk "$NEXT"' BEGIN { for (i = 0; i < 864; i++) print nx(i) }' > "$T/next.txt"
awk 'BEGIN { for (s = 0; s < 2048; s++) print s % 2 }' > "$T/pos.txt"
port=$((20000 + $$ % 20000))
start() { # the server's flags: it listens when this returns
  stop; : > "$T/err"
  tries=0
  while [ $tries -lt 20 ]; do
    "$B" "$TOK" "$T/toy.model" --port $port --ctx $CTX --end 858 --end 859 --trace "$@" CHX="$ROOT/tests/chx" X="${XDIR:-$T}" NS=$CTX 2> "$T/err" > /dev/null &
    pid=$!
    w=0
    while [ $w -lt 200 ] && ! grep -q "^listening\|^error\|^usage" "$T/err" 2>/dev/null; do sleep 0.05; w=$((w + 1)); done
    if grep -q "^listening" "$T/err"; then return 0; fi
    stop; port=$((port + 1)); tries=$((tries + 1))
  done
  echo "FAIL the server does not start: $(head -c 300 "$T/err")"; fail=1; return 1
}
# a request: method, path, a file with the body (or none), then extra header lines. The status in $code, the head in $T/head, the body in $T/body (a chunked body decoded: it must be one
# chunk of the length said, then the end)
ask() {
  method=$1; path=$2; bf=${3:-}; shift 2; [ $# -eq 0 ] || shift
  { printf '%s %s HTTP/1.1\r\nHost: localhost\r\n' "$method" "$path"; for h in "$@"; do printf '%s\r\n' "$h"; done
    if [ -n "$bf" ]; then printf 'Content-Type: application/json\r\nContent-Length: %s\r\n\r\n' "$(wc -c < "$bf" | tr -d ' ')"; cat "$bf"; else printf '\r\n'; fi; } > "$T/req"
  : > "$T/err.mark"; marks=$(wc -l < "$T/err" | tr -d ' ')
  "$H" $port "$T/req" > "$T/resp" 2> /dev/null || true
  tr -d '\r' < "$T/resp" | awk -v H="$T/head" -v B="$T/body" 'BEGIN { printf "" > H; printf "" > B } !inbody { if ($0 == "") inbody = 1; else print > H; next } { print > B }'
  code=$(head -1 "$T/head" | cut -d' ' -f2)
  chunked=0
  if grep -q "^Transfer-Encoding: chunked" "$T/head"; then
    chunked=1
    LC_ALL=C awk 'NR == 1 { n = $0 } NR == 2 { b = $0 } NR == 3 { z = $0 } END { if (NR == 4 && z == "0" && sprintf("%x", length(b)) == n) print b; else print "NOT ONE CHUNK: " NR " lines, said " n ", is " length(b) }' "$T/body" > "$T/body2"; mv "$T/body2" "$T/body"
  fi
}
# what the session was asked for the last request: the lines of the trace after it was sent
asked() { sed -n "$((marks + 1)),\$p" "$T/err" | sed -n 's/^ask //p'; }
field() { sed -n "s/.*\"$1\":\([0-9a-z._\"-]*\)[,}].*/\1/p" "$T/body" | tr -d '"'; }   # a number or a word of the body's JSON
is() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: \`$(printf '%s' "$2" | head -c 300)\`, wanted \`$(printf '%s' "$3" | head -c 300)\`"; fail=1; fi; }
has() { if grep -q -- "$2" "$T/body"; then echo "ok   $1"; else echo "FAIL $1: $(head -c 300 "$T/body")"; fail=1; fi; }
body() { printf '%s' "$1" > "$T/b.json"; }
ids_of() { "$ROOT/build/cancho-gpu" tokenize "$TOK" "$1" | awk '/^ids / { for (i = 3; i <= NF; i++) print $i }'; }   # the tokenizer's ids of a file's text, a line each
# awk's answer to a prompt's ids (a file, a line each) with at most $2 new ids: a line `letters count reason`
sim() { awk "$NEXT"' { p[n++] = $1 } END { id = p[n - 1]; s = n - 1; why = "length"; out = ""; k = 0; while (k < max) { a = nx(id) + s % 2; if (a == 858 || a == 859) { why = "stop"; break } out = out sprintf("%c", a); k++; id = a; s++ } print out, k, why }' max="$2" "$1"; }

# --- 1. the routes
start --greedy --max-new 7 || exit 1
ask GET /health; is "GET /health" "$code $(cat "$T/body")" '200 {"status":"ok"}'
ask GET /v1/models; is "GET /v1/models" "$code $(field id)" "200 kolibri-1.1"
ask GET /api/tags; is "GET /api/tags" "$code $(field name)" "200 kolibri-1.1"
ask GET "/health?x=1"; is "a query after the path is not part of it" "$code" 200
ask GET /nothing; is "a route that is not one" "$code" 404
ask GET /api/generate; is "GET on a route that takes POST" "$code" 405
ask POST /v1/models "$T/zero.txt"; is "POST on a route that takes GET" "$code" 405
body '{"prompt":"hi","raw":true,"options":{"num_predict":2}}'
ask POST /api/generate "$T/b.json"; is "POST /api/generate" "$code $chunked $(field done) $(field eval_count)" "200 1 true 2"
body '{"messages":[{"role":"user","content":"hi"}],"max_tokens":2}'
ask POST /v1/chat/completions "$T/b.json"; is "POST /v1/chat/completions" "$code $chunked $(field object | head -1)" "200 1 chat.completion"
# refusals, each a 400 with its message and nothing asked of the session
refuse() { # name, route, body, a text of the message
  body "$3"; ask POST "$2" "$T/b.json"
  if [ "$code" = 400 ] && [ "$chunked" = 0 ] && grep -q -- "$4" "$T/body" && grep -q '"type":"invalid_request_error"' "$T/body" && [ -z "$(asked)" ]; then echo "ok   refused: $1"; else echo "FAIL $1: status $code, $(head -c 300 "$T/body") $(asked | head -3)"; fail=1; fi
}
G=/api/generate; O=/v1/chat/completions
refuse "a body that is not JSON" $G '{"prompt":' "not JSON"
refuse "a body that is an array" $G '[1]' "must be a JSON object"
refuse "no prompt" $G '{"model":"m"}' "prompt. is missing"
refuse "a prompt that is a number" $G '{"prompt":3}' "must be a string"
refuse "a system that is a number" $G '{"prompt":"a","system":3}' "must be a string"
refuse "num_predict that is a word" $G '{"prompt":"a","options":{"num_predict":"x"}}' "must be a whole number"
refuse "num_predict 0" $G '{"prompt":"a","options":{"num_predict":0}}' "1 or more"
refuse "an empty raw prompt" $G '{"prompt":"","raw":true}' "no tokens"
refuse "an effort that is no effort" $G '{"prompt":"a","think":"deeply"}' "the effort is"
refuse "no messages" $O '{"model":"m"}' "messages. must be an array"
refuse "messages that are empty" $O '{"messages":[]}' "messages. must be an array"
refuse "a message that is no object" $O '{"messages":["hi"]}' "must be an object"
refuse "a message with no role" $O '{"messages":[{"content":"hi"}]}' "needs a role"
refuse "a role that is none" $O '{"messages":[{"role":"robot","content":"hi"}]}' "system, user, assistant or tool"
refuse "no user message" $O '{"messages":[{"role":"system","content":"hi"}]}' "no user message"
U='{"role":"user","content":"hi"}'
refuse "tools that are no array" $O '{"messages":['"$U"'],"tools":{"name":"f"}}' "tools. must be an array"
refuse "a tool that is no object" $O '{"messages":['"$U"'],"tools":[{"name":"f"},"g"]}' "a tool must be an object"
refuse "a tool that is no object, with tool_choice none" $O '{"messages":['"$U"'],"tool_choice":"none","tools":[3]}' "a tool must be an object"
refuse "tool_choice required" $O '{"messages":['"$U"'],"tool_choice":"required","tools":[{"name":"f"}]}' "tool_choice. is auto or none"
refuse "tool_choice that names a function" $O '{"messages":['"$U"'],"tool_choice":{"type":"function","function":{"name":"f"}},"tools":[{"name":"f"}]}' "tool_choice. is auto or none"
refuse "a tool message with no content" $O '{"messages":['"$U"',{"role":"tool","tool_call_id":"1"}]}' "a tool message needs its content"
refuse "a tool message whose content is null" $O '{"messages":['"$U"',{"role":"tool","content":null}]}' "a tool message needs its content"
refuse "tool calls that are no array" $O '{"messages":['"$U"',{"role":"assistant","content":null,"tool_calls":{"name":"f"}}]}' "tool_calls. must be an array"
refuse "a tool call that is no object" $O '{"messages":['"$U"',{"role":"assistant","content":null,"tool_calls":["f"]}]}' "a tool call must be an object"
refuse "a tool call whose function is no object" $O '{"messages":['"$U"',{"role":"assistant","content":null,"tool_calls":[{"function":"f"}]}]}' "function. must be an object"
refuse "a tool call with no name" $O '{"messages":['"$U"',{"role":"assistant","content":null,"tool_calls":[{"function":{"arguments":"{}"}}]}]}' "needs a .name. that is a string"
refuse "a tool call whose name is a number" $O '{"messages":['"$U"',{"role":"assistant","content":null,"tool_calls":[{"function":{"name":3,"arguments":"{}"}}]}]}' "needs a .name. that is a string"
refuse "a tool call with no arguments" $O '{"messages":['"$U"',{"role":"assistant","content":null,"tool_calls":[{"function":{"name":"f"}}]}]}' "needs its .arguments."
refuse "a second tool call that is none" $O '{"messages":['"$U"',{"role":"assistant","content":null,"tool_calls":[{"name":"f","arguments":{}},{"name":"g"}]}]}' "needs its .arguments."
refuse "a string with half a surrogate pair" $O '{"messages":['"$U"'],"tools":[{"name":"\ud83d"}]}' "not JSON"
refuse "a string with the second half of a surrogate pair alone" $O '{"messages":['"$U"'],"tools":[{"name":"\ude00"}]}' "not JSON"
refuse "a stream" $O '{"messages":[{"role":"user","content":"hi"}],"stream":true}' "stream is not built"
refuse "n = 2" $O '{"messages":[{"role":"user","content":"hi"}],"n":2}' "only n = 1"
refuse "an image" $O '{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}' "must be a text"
refuse "a content that is a number" $O '{"messages":[{"role":"user","content":3}]}' "a string or an array"
refuse "a reasoning that is a number" $O '{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b","reasoning":3},{"role":"user","content":"c"}]}' "must be a string"
refuse "max_tokens 0" $O '{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}' "1 or more"
refuse "reasoning_effort that is none of them" $O '{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"some"}' "the effort is"
awk -v n=$((CTX + 1)) 'BEGIN { printf "{\"raw\":true,\"prompt\":\""; for (i = 0; i < n; i++) printf "a "; printf "\"}" }' > "$T/b.json"
ask POST $G "$T/b.json"; if [ "$code" = 400 ] && grep -q "more than the $CTX positions" "$T/body"; then echo "ok   refused: a prompt of more tokens than the session's positions"; else echo "FAIL a prompt too long: $code $(head -c 200 "$T/body")"; fail=1; fi
# a length that is no number, and one over the largest taken. (A body shorter than its length is not here: the server waits for the rest, and has no limit to that wait.)
printf 'POST /api/generate HTTP/1.1\r\nContent-Length: x\r\n\r\n' > "$T/req"; "$H" $port "$T/req" > "$T/resp" 2>/dev/null || true
is "a Content-Length that is no number" "$(head -1 "$T/resp" | tr -d '\r' | cut -d' ' -f2)" 400
printf 'POST /api/generate HTTP/1.1\r\nContent-Length: 99999999\r\n\r\n' > "$T/req"; "$H" $port "$T/req" > "$T/resp" 2>/dev/null || true
is "a body over the largest taken" "$(head -1 "$T/resp" | tr -d '\r' | cut -d' ' -f2)" 413
ask GET /health; is "the server still answers after them" "$code" 200

# --- 2. the template: the ids the session is given are the tokenizer's ids of the recorded text
reset() { printf '%s' '{"prompt":"zz","raw":true,"options":{"num_predict":1}}' > "$T/reset.json"; ask POST $G "$T/reset.json"; }   # a prompt that shares no start with any other, so the next is run from 0
n=0; bad=0
for f in tests/kolibri_chat/*.json; do
  name=$(basename "$f" .json); n=$((n + 1))
  case "$name" in gen_*) route=$G ;; *) route=$O ;; esac
  reset
  ask POST $route "$f"
  asked | awk '$3 == 1 { print $2 }' > "$T/got_ids.txt"
  want=golden/kolibri_chat/$name.txt; [ "$name" != chat_developer ] || want=golden/kolibri_chat/chat_system.txt
  ids_of "$want" > "$T/want_ids.txt"
  if [ "$code" != 200 ] || [ ! -s "$T/want_ids.txt" ] || ! cmp -s "$T/got_ids.txt" "$T/want_ids.txt"; then echo "FAIL the prompt of $name is not the template's: status $code, $(wc -l < "$T/got_ids.txt" | tr -d ' ') ids, wanted $(wc -l < "$T/want_ids.txt" | tr -d ' ') $(head -c 200 "$T/body")"; bad=1; fail=1; fi
  [ "$(field prompt_eval_count)$(field prompt_tokens)" = "$(wc -l < "$T/want_ids.txt" | tr -d ' ')" ] || { echo "FAIL $name: the prompt's count is $(field prompt_eval_count)$(field prompt_tokens)"; bad=1; fail=1; }
done
[ $bad -ne 0 ] || echo "ok   the prompts of $n requests are the model's template's, id for id (chat turns, reasoning kept and dropped, a later system message, parts, the efforts, Ollama's prompt and system, raw; tools, an assistant's calls, tool turns)"

# --- 3. an answer
answer() { # name, the prompt's text, the most new ids: a raw prompt through /api/generate against awk
  printf '%s' "$2" > "$T/p.txt"; ids_of "$T/p.txt" > "$T/p_ids.txt"
  set -- "$1" "$2" "$3" $(sim "$T/p_ids.txt" "$3")
  [ $# -eq 6 ] || set -- "$1" "$2" "$3" "" "$4" "$5"
  reset
  awk -v p="$2" -v n="$3" 'BEGIN { printf "{\"prompt\":\"%s\",\"raw\":true,\"options\":{\"num_predict\":%d}}", p, n }' > "$T/b.json"
  ask POST $G "$T/b.json"
  is "$1: \`$4\`, $5 new, $6" "$(field response) $(field eval_count) $(field done_reason) $(field prompt_eval_count)" "$4 $5 $6 $(wc -l < "$T/p_ids.txt" | tr -d ' ')"
  # the prompt's last step chooses the first new id; each step after it runs on the id before, and the one that answers an end token is run too
  if [ "$6" = stop ]; then own=$5; else own=$(($5 - 1)); fi
  [ "$(asked | awk '$3 == 0' | wc -l | tr -d ' ')" = "$own" ] || { echo "FAIL $1: the session ran $(asked | awk '$3 == 0' | wc -l | tr -d ' ') steps on its own ids, wanted $own"; fail=1; }
}
answer "an answer to the limit" "ab" 5
answer "an answer to an end token" "abc" 40
answer "an answer of one id" "ab" 1
answer "another prompt, another answer" "hello world" 12
answer "a prompt whose first answer is an end token" "e" 9
answer "the other end token" "ae" 9
# the chat route gives the same answer, with its own names
printf 'ab' > "$T/p.txt"; ids_of "$T/p.txt" > "$T/p_ids.txt"
body '{"messages":[{"role":"user","content":"Why?"}],"max_tokens":3}'; reset; ask POST $O "$T/b.json"
ids_of golden/kolibri_chat/chat_effort_none.txt > "$T/p_ids.txt"; set -- $(sim "$T/p_ids.txt" 3)
[ $# -eq 3 ] || set -- "" "$1" "$2"
is "the chat route: content, finish_reason and usage" "$(field content) $(field finish_reason) $(field completion_tokens) $(field prompt_tokens) $(field total_tokens)" "$1 $3 $2 $(wc -l < "$T/p_ids.txt" | tr -d ' ') $(($(wc -l < "$T/p_ids.txt") + $2))"
# the most new ids: cut to the room left (the last step is the session's last position), with a limit past it and with Ollama's -1; and the server's own default when none is given
tried=0; cut=0
for tail in "hello world" "ab" "one two" "x y z" "abc def"; do
  awk -v n=$((CTX - 12)) -v w="$tail" 'BEGIN { for (i = 0; i < n; i++) printf "a"; printf " %s", w }' > "$T/p.txt"; ids_of "$T/p.txt" > "$T/p_ids.txt"; np=$(wc -l < "$T/p_ids.txt" | tr -d ' ')
  room=$((CTX - np + 1)); set -- $(sim "$T/p_ids.txt" $room); [ $# -eq 3 ] || set -- "" "$1" "$2"
  [ "$3" = length ] || continue
  cut=1
  for lim in 5000 -1; do
    awk -v p="$(cat "$T/p.txt")" -v l="$lim" 'BEGIN { printf "{\"raw\":true,\"prompt\":\"%s\",\"options\":{\"num_predict\":%d}}", p, l }' > "$T/b.json"
    reset; ask POST $G "$T/b.json"
    is "a limit of $lim with room for $room: $room new, and the answer ends by its length" "$code $(field eval_count) $(field done_reason) $(field response)" "200 $room length $1"
  done
  break
done
[ $cut -eq 1 ] || { echo "FAIL no prompt of the five runs to the session's last position: the room left is not tested"; fail=1; }
printf 'hello world' > "$T/p.txt"; ids_of "$T/p.txt" > "$T/p_ids.txt"; set -- $(sim "$T/p_ids.txt" 7); [ $# -eq 3 ] || set -- "" "$1" "$2"
body '{"prompt":"hello world","raw":true}'; reset; ask POST $G "$T/b.json"
is "no limit given: the server's (--max-new 7)" "$(field eval_count) $(field done_reason) $(field response)" "$2 $3 $1"
# max_completion_tokens over max_tokens: on the first recorded chat request whose answer runs past two ids
mc=0
for f in tests/kolibri_chat/chat_*.json; do
  ids_of "golden/kolibri_chat/$(basename "$f" .json).txt" > "$T/p_ids.txt"; set -- $(sim "$T/p_ids.txt" 9); [ $# -eq 3 ] || set -- "" "$1" "$2"
  [ "$2" -gt 2 ] && [ "$(basename "$f")" != chat_developer.json ] || continue
  sed 's/"max_tokens":1}/"max_tokens":9,"max_completion_tokens":2}/' "$f" > "$T/b.json"; grep -q max_completion_tokens "$T/b.json" || continue
  reset; ask POST $O "$T/b.json"; mc=1
  is "max_completion_tokens is taken over max_tokens (an answer of $2 ids cut at 2)" "$(field completion_tokens) $(field finish_reason)" "2 length"
  break
done
[ $mc -eq 1 ] || { echo "FAIL no recorded chat request answers more than two ids: max_completion_tokens is not tested"; fail=1; }
body '{"model":"my-name","prompt":"hi","raw":true,"options":{"num_predict":1}}'; ask POST $G "$T/b.json"; is "the model's name is the request's" "$(field model)" "my-name"

# --- 4. the prefix kept
P1='{"prompt":"one two three four five six","raw":true,"options":{"num_predict":4}}'
printf 'one two three four five six' > "$T/p.txt"; ids_of "$T/p.txt" > "$T/p_ids.txt"; np1=$(wc -l < "$T/p_ids.txt" | tr -d ' ')
reset; body "$P1"; ask POST $G "$T/b.json"; first=$(field response); k1=$(field eval_count)
is "a prompt after another that shares nothing: from 0, no seek" "$(asked | head -1 | cut -d' ' -f1) $(asked | awk '$3 == 1' | wc -l | tr -d ' ')" "seek $np1"
ask POST $G "$T/b.json"
is "the same request again: back to its last token's step, and the same answer" "$(asked | head -2 | tr '\n' ';') $(field response)" "seek $((np1 - 1));1 $(tail -1 "$T/p_ids.txt") 1; $first"
# a longer prompt that starts with the last prompt and its answer goes on from where the session is, with no seek
printf 'one two three four five six%s and more' "$first" > "$T/p2.txt"; ids_of "$T/p2.txt" > "$T/p2_ids.txt"; np2=$(wc -l < "$T/p2_ids.txt" | tr -d ' ')
same=$(paste "$T/p2_ids.txt" "$T/p_ids.txt" | awk '$1 != $2 { exit } { n++ } END { print n + 0 }')
awk -v p="$(cat "$T/p2.txt")" 'BEGIN { printf "{\"prompt\":\"%s\",\"raw\":true,\"options\":{\"num_predict\":3}}", p }' > "$T/b.json"
ask POST $G "$T/b.json"; set -- $(sim "$T/p2_ids.txt" 3)
is "a prompt that goes on from the last one and its answer: only the new ids are run, and the answer is awk's" "$(asked | grep -c '^seek') $(asked | awk '$3 == 1' | wc -l | tr -d ' ') $(field response)" "0 $((np2 - np1 - k1 + 1 - 1 + 1 - 1)) $1"
# a prompt that parts from the last one in the middle: back to where they part
printf 'one two three FOUR five' > "$T/p3.txt"; ids_of "$T/p3.txt" > "$T/p3_ids.txt"; np3=$(wc -l < "$T/p3_ids.txt" | tr -d ' ')
part=$(paste "$T/p3_ids.txt" "$T/p2_ids.txt" | awk '$1 != $2 { exit } { n++ } END { print n + 0 }')
body '{"prompt":"one two three FOUR five","raw":true,"options":{"num_predict":3}}'; ask POST $G "$T/b.json"; set -- $(sim "$T/p3_ids.txt" 3)
is "a prompt that parts from the last one: back to where they part ($part of $np3 ids kept), and the answer is awk's" "$(asked | head -1) $(asked | awk '$3 == 1' | wc -l | tr -d ' ') $(field response)" "seek $part $((np3 - part)) $1"
grep -q "a prompt of $np3 tokens, $part of them kept, $2 new" "$T/err" && echo "ok   the server's line says what was kept" || { echo "FAIL the line of the request: $(tail -1 "$T/err")"; fail=1; }
body '{"messages":[{"role":"user","content":"Why?"}],"max_tokens":2}'; reset; ask POST $O "$T/b.json"; ask POST $O "$T/b.json"
is "the chat route says what was kept" "$(field cached_tokens)" "$(($(field prompt_tokens) - 1))"
# a ring shorter than the context: going back is taken while the window before the position is still in the ring. The last k words of a long prompt changed, for k from 1 to 8: how far
# back each is, from the tokenizer; the window is then set so that one of those distances is exactly the furthest taken (ring - window), with a nearer and a further one beside it
LONG='a b c d e f g h i j k l m n o p q r s t u v w x y z'
printf '%s' "$LONG" > "$T/p.txt"; ids_of "$T/p.txt" > "$T/p_ids.txt"; npl=$(wc -l < "$T/p_ids.txt" | tr -d ' ')
awk -v p="$LONG" 'BEGIN { printf "{\"prompt\":\"%s\",\"raw\":true,\"options\":{\"num_predict\":1}}", p }' > "$T/long.json"
: > "$T/dist.txt"
for k in 1 2 3 4 5 6 7 8; do
  awk -v p="$LONG" -v k="$k" 'BEGIN { n = split(p, w, " "); for (i = 1; i <= n; i++) printf "%s%s", (i > n - k ? "Z" : w[i]), (i < n ? " " : "") }' > "$T/p2_$k.txt"
  ids_of "$T/p2_$k.txt" > "$T/p2_ids.txt"
  part=$(paste "$T/p2_ids.txt" "$T/p_ids.txt" | awk '$1 != $2 { exit } { n++ } END { print n + 0 }')
  echo "$k $part $((npl - 1 - part))" >> "$T/dist.txt"
done
# the furthest taken: the third of the distances in order (so that two are nearer), which must leave one further and fit a ring of 16
edge=$(cut -d' ' -f3 "$T/dist.txt" | sort -n -u | sed -n 3p)
[ -n "$edge" ] && [ "$edge" -lt 15 ] && [ "$(awk -v e="$edge" '$3 > e' "$T/dist.txt" | wc -l | tr -d ' ')" -gt 0 ] || { echo "FAIL the long prompt's changed endings give no distance to put the ring's edge at: $(tr '\n' ';' < "$T/dist.txt")"; fail=1; edge=6; }
start --greedy --max-new 7 --ring 16 --window $((16 - edge)) || exit 1
ask POST $G "$T/long.json"; ask POST $G "$T/long.json"
is "a ring of 16 for a window of $((16 - edge)), after $npl steps: the last step again is taken" "$(asked | head -1)" "seek $((npl - 1))"
took=0; left=0; at=0; bad=0
while read -r k part d; do
  awk -v p="$(cat "$T/p2_$k.txt")" 'BEGIN { printf "{\"prompt\":\"%s\",\"raw\":true,\"options\":{\"num_predict\":1}}", p }' > "$T/b2.json"
  ask POST $G "$T/long.json"; ask POST $G "$T/b2.json"
  if [ "$d" -le "$edge" ]; then want="seek $part"; took=$((took + 1)); [ "$d" -ne "$edge" ] || at=1; else want="seek 0"; left=$((left + 1)); fi
  [ "$(asked | head -1)" = "$want" ] || { echo "FAIL the last $k words changed ($d positions back, the furthest taken is $edge): \`$(asked | head -1)\`, wanted \`$want\`"; bad=1; fail=1; }
done < "$T/dist.txt"
if [ $bad -eq 0 ] && [ $took -gt 1 ] && [ $left -gt 0 ] && [ $at -eq 1 ]; then echo "ok   the same ring: back up to $edge positions (ring - window) is taken ($took cases, one of them at exactly $edge), further is from 0 ($left cases)"; else echo "FAIL the ring's cases: $took taken, $left not, at the edge $at"; fail=1; fi
# and while the ring has not wrapped, any position is gone back to: a prompt of no more ids than the ring, then one that parts from it further back than ring - window
start --greedy --ring 16 --window $((16 - edge)) || exit 1
body '{"prompt":"a b c d e f g h","raw":true,"options":{"num_predict":1}}'; ask POST $G "$T/b.json"
printf 'a b c d e f g h' > "$T/p.txt"; ids_of "$T/p.txt" > "$T/s_ids.txt"; printf 'a Z c d e f g h' > "$T/p.txt"; ids_of "$T/p.txt" > "$T/s2_ids.txt"
nps=$(wc -l < "$T/s_ids.txt" | tr -d ' '); part=$(paste "$T/s2_ids.txt" "$T/s_ids.txt" | awk '$1 != $2 { exit } { n++ } END { print n + 0 }')
body '{"prompt":"a Z c d e f g h","raw":true,"options":{"num_predict":1}}'; ask POST $G "$T/b.json"
if [ "$nps" -le 16 ] && [ "$part" -gt 0 ] && [ $((nps - 1 - part)) -gt "$edge" ]; then is "a ring that has not wrapped ($nps steps of 16): $((nps - 1 - part)) positions back is taken, though further than $edge" "$(asked | head -1)" "seek $part"; else echo "FAIL the short prompt is $nps ids and parts at $part: no case of a ring that has not wrapped (the furthest taken is $edge)"; fail=1; fi
start --greedy --max-new 7 --ring 16 --window $((16 - edge)) || exit 1
ask POST $G "$T/long.json"
[ "$npl" -gt 16 ] || { echo "FAIL the long prompt is $npl ids, not more than the ring: the cases above show nothing"; fail=1; }

# --- 5. sampling: a number on every line, the seed's at the step's position
start SAMP=1 || exit 1
body '{"prompt":"one two three","raw":true,"options":{"num_predict":3,"seed":7}}'; ask POST $G "$T/b.json"
nl=$(asked | wc -l | tr -d ' ')
is "a sampling session: every line carries the number of seed 7 for its step" "$(asked | awk 'NF == 4 { print $4 }')" "$(build/kolibri_gpu draws 7 "$nl")"
ask POST $G "$T/b.json"
kept=$(asked | sed -n 's/^seek //p')
is "the same request again: the numbers of the steps it runs, not the seed's first" "$(asked | awk 'NR > 1 { print $4 }')" "$(build/kolibri_gpu draws 7 "$nl" | sed -n "$((kept + 1)),\$p")"
body '{"prompt":"one two three","raw":true,"options":{"num_predict":3,"seed":8}}'; reset; ask POST $G "$T/b.json"
is "another seed, other numbers" "$(asked | awk 'NF == 4 { print $4 }')" "$(build/kolibri_gpu draws 8 "$nl")"
body '{"prompt":"one two three","raw":true,"options":{"num_predict":3}}'; reset; ask POST $G "$T/b.json"; asked | awk 'NF == 4 { print $4 }' > "$T/u1.txt"; reset; ask POST $G "$T/b.json"; asked | awk 'NF == 4 { print $4 }' > "$T/u2.txt"
if [ -s "$T/u1.txt" ] && ! cmp -s "$T/u1.txt" "$T/u2.txt"; then echo "ok   with no seed, two requests draw with different numbers"; else echo "FAIL two requests with no seed drew the same"; fail=1; fi
start --greedy --api-key s3cret || exit 1
ask GET /health; is "with a key: /health needs none" "$code" 200
ask GET /v1/models; is "with a key: none given" "$code" 401
ask GET /v1/models "" "Authorization: Bearer wrong"; is "with a key: a wrong one" "$code" 401
ask GET /v1/models "" "Authorization: Bearer s3cret"; is "with a key: the key" "$code" 200
body '{"prompt":"hi","raw":true,"options":{"num_predict":1}}'; ask POST $G "$T/b.json"; is "with a key: a prompt with none is not run" "$code $(asked | wc -l | tr -d ' ')" "401 0"

# the server's status when it has stopped by itself within ten seconds; 99 when it has not (and it is stopped)
ended() { w=0; while [ $w -lt 100 ] && kill -0 "$pid" 2>/dev/null; do sleep 0.1; w=$((w + 1)); done; if kill -0 "$pid" 2>/dev/null; then stop; st=99; else st=0; wait "$pid" 2>/dev/null || st=$?; pid=""; fi; }
# --- 7. passes: the lines a prompt is sent as. awk says them from the ids, where the session starts and the pass's size: a pass while it leaves a token for the last step
plan() { awk -v k="$2" -v pb=4 '{ id[n++] = $1 } END { while (n - 1 - k >= pb) { printf "pass 1"; for (b = 0; b < pb; b++) printf " %d", id[k + b]; printf "\n"; k += pb } for (; k < n; k++) print 1, id[k], 1 }' "$1"; }
prompt_lines() { asked | grep -v '^seek' | awk '$1 == "pass" || $3 == 1'; }   # the lines of the prompt's tokens, among what was asked
LONG='a b c d e f g h i j k l m n o p q r s t u v w x y z'
start --greedy --max-new 5 --pass 4 PASS=1 || exit 1
printf '%s' "$LONG" > "$T/p.txt"; ids_of "$T/p.txt" > "$T/p_ids.txt"; npl=$(wc -l < "$T/p_ids.txt" | tr -d ' ')
awk -v p="$LONG" 'BEGIN { printf "{\"prompt\":\"%s\",\"raw\":true,\"options\":{\"num_predict\":5}}", p }' > "$T/long.json"
ask POST $G "$T/long.json"; want=$(sim "$T/p_ids.txt" 5)
is "a prompt of $npl tokens in passes of 4: the lines are awk's ($(( (npl - 1) / 4 )) passes, then the steps)" "$(prompt_lines)" "$(plan "$T/p_ids.txt" 0)"
is "its answer is awk's, which knows no pass, and its counts" "$code $npl $(field prompt_eval_count) $(field response) $(field eval_count)" "200 $npl $npl ${want% *}"
[ "$(prompt_lines | grep -c '^pass')" -ge 3 ] && [ "$(prompt_lines | grep -vc '^pass')" -ge 1 ] || { echo "FAIL the long prompt gives $(prompt_lines | grep -c '^pass') passes: the case shows nothing"; fail=1; }
ask POST $G "$T/long.json"
is "the same request again: back to its last token's step, no pass" "$(asked | head -2 | tr '\n' ';') $(asked | grep -c '^pass')" "seek $((npl - 1));1 $(tail -1 "$T/p_ids.txt") 1; 0"
# a prompt that parts from it: from where they part, passes again
for k in 9 5 4; do
  awk -v p="$LONG" -v k="$k" 'BEGIN { n = split(p, w, " "); for (i = 1; i <= n; i++) printf "%s%s", (i > n - k ? "Z" : w[i]), (i < n ? " " : "") }' > "$T/p2.txt"
  ids_of "$T/p2.txt" > "$T/p2_ids.txt"; part=$(paste "$T/p2_ids.txt" "$T/p_ids.txt" | awk '$1 != $2 { exit } { n++ } END { print n + 0 }'); np2=$(wc -l < "$T/p2_ids.txt" | tr -d ' ')
  awk -v p="$(cat "$T/p2.txt")" 'BEGIN { printf "{\"prompt\":\"%s\",\"raw\":true,\"options\":{\"num_predict\":5}}", p }' > "$T/b2.json"
  ask POST $G "$T/long.json"; ask POST $G "$T/b2.json"; want=$(sim "$T/p2_ids.txt" 5)
  is "the last $k words changed ($part of $np2 ids kept): back to there, then awk's lines ($(plan "$T/p2_ids.txt" "$part" | grep -c '^pass') passes), and awk's answer" "$(asked | head -1);$(prompt_lines);$(field response) $(field eval_count)" "seek $part;$(plan "$T/p2_ids.txt" "$part");${want% *}"
  field eval_count >> "$T/left.txt"
done
[ "$(sort -n "$T/left.txt" | tail -1)" -ge 1 ] || { echo "FAIL none of the prompts that part has an answer of a token or more: awk's answer was compared with nothing"; fail=1; }
# the edge: with exactly four tokens to run and one more, a pass; with exactly four, steps
: > "$T/edge.txt"
for pe in "a b c d" "x y z" "one two three" "a b c d e"; do
  printf '%s' "$pe" > "$T/pe.txt"; ids_of "$T/pe.txt" > "$T/pe_ids.txt"; npe=$(wc -l < "$T/pe_ids.txt" | tr -d ' ')
  awk -v p="$(cat "$T/pe.txt")" 'BEGIN { printf "{\"prompt\":\"%s\",\"raw\":true,\"options\":{\"num_predict\":1}}", p }' > "$T/be.json"
  reset; ask POST $G "$T/be.json"
  [ "$(prompt_lines)" = "$(plan "$T/pe_ids.txt" 0)" ] && echo "$npe $(prompt_lines | grep -c '^pass')" >> "$T/edge.txt" || { echo "FAIL a prompt of $npe tokens: $(prompt_lines | tr '\n' ';')"; fail=1; }
done
is "a prompt of 4 tokens is four steps, one of 5 a pass and a step, one of 6 a pass and two" "$(sort -u "$T/edge.txt" | tr '\n' ';')" "4 0;5 1;6 1;"
# sampling: a step after passes draws with the number of its position
start --pass 4 PASS=1 SAMP=1 || exit 1
awk -v p="$LONG" 'BEGIN { printf "{\"prompt\":\"%s\",\"raw\":true,\"options\":{\"num_predict\":3,\"seed\":7}}", p }' > "$T/long.json"
ask POST $G "$T/long.json"
np4=$(( (npl - 1) / 4 * 4 ))
is "a sampling session with passes: the steps after $((np4 / 4)) passes carry the numbers of seed 7 from position $np4 on" "$(asked | awk '$1 != "pass" { print $4 }')" "$(build/kolibri_gpu draws 7 $((npl + 3)) | sed -n "$((np4 + 1)),$((np4 + $(asked | grep -vc '^pass')))p")"
# a session whose pass answers another id than it was given: the server stops
start --greedy --pass 4 PASS=1 PD=one4.txt || exit 1
ask POST $G "$T/long.json"
ended
if [ "$code" = 200 ] && grep -q '"type":"server_error"' "$T/body" && [ "$st" -eq 1 ] && grep -q "^error\[kolibri-session\]: the session did not run the pass asked" "$T/err"; then echo "ok   a session whose pass does not answer its first id: said, and the server stops"; else echo "FAIL a pass that answers another id: status $code, exit $st, $(tail -2 "$T/err")"; fail=1; fi
# a server told of passes whose session has none: the session's refusal
start --greedy --pass 4 || exit 1
ask POST $G "$T/long.json"
ended
if [ "$st" -eq 1 ] && grep -q "^error\[kolibri-session\]: .pass. is not a count of steps" "$T/err"; then echo "ok   --pass with a session that has no such phase: the session's refusal, and the server stops"; else echo "FAIL --pass with no phase: exit $st, $(tail -2 "$T/err")"; fail=1; fi

# --- 8. tools: a session that answers what a table says at each position (its `next` is all zeros), so the answer to a request is the text put after its prompt
mkdir "$T/x2"; cp "$T/zero.txt" "$T/zero4.txt" "$T/x2/"; awk 'BEGIN { for (i = 0; i < 864; i++) print 0 }' > "$T/x2/next.txt"
scripted() { # the golden prompt of the request, the answer's text (a file), then the server's flags: a server whose session answers that text after that prompt, then an end token
  ids_of "$1" > "$T/sp_ids.txt"; ids_of "$2" > "$T/sa_ids.txt"
  snp=$(wc -l < "$T/sp_ids.txt" | tr -d ' '); sna=$(wc -l < "$T/sa_ids.txt" | tr -d ' ')
  { awk -v n=$((snp - 1)) 'BEGIN { for (i = 0; i < n; i++) print 0 }'; cat "$T/sa_ids.txt"; echo 858; awk -v n=$((CTX - snp - sna)) 'BEGIN { for (i = 0; i < n; i++) print 0 }'; } > "$T/x2/pos.txt"
  shift 2
  XDIR="$T/x2" start --greedy --max-new 400 "$@"
}
KC=tests/kolibri_chat; GC=golden/kolibri_chat; KA=tests/kolibri_calls
message() { sed -n 's/.*"message":\(.*\),"finish_reason":"\([a-z_]*\)".*/\2 \1/p' "$T/body"; }   # the finish reason and the message of the body
scripted $GC/chat_tools.txt $KA/two.txt || exit 1
ask POST $O $KC/chat_tools.json
is "an answer of two calls: no content, the calls with the request's number in their ids and their arguments as the model wrote them, and it says tool_calls" "$code $(message)" "200 tool_calls $(sed 's/call_0_/call_1_/g' golden/kolibri_calls/two.json)"
is "its counts" "$(field prompt_tokens) $(field completion_tokens)" "$snp $sna"
# the turn sent back: the template writes the calls as the model wrote them, so the session goes on from after them
ask POST $O $KC/chat_tool_turn.json
is "the calls sent back as tool_calls, with their answers: the session goes on from where it is, after the prompt and the model's own turn, with that turn's end" "$code $(field cached_tokens) $(asked | head -1)" "200 $((snp + sna)) 1 858 1"
ask POST $O $KC/chat_tools.json
is "the next request's calls have its number" "$(message | grep -o 'call_[0-9]*_[0-9]*' | tr '\n' ' ')" "call_3_0 call_3_1 "
# the limit: an answer cut after its calls says length and has them
sed "s/\"max_tokens\":256/\"max_tokens\":$sna/" $KC/chat_tools.json > "$T/cut.json"; grep -q "\"max_tokens\":$sna," "$T/cut.json" || { echo "FAIL the request has no max_tokens to change"; fail=1; }
reset; ask POST $O "$T/cut.json"
is "an answer cut by the limit after its calls: length, with the calls" "$(message | cut -c1-60)" "$(printf '%s' 'length {"role":"assistant","content":null,"tool_calls":[{"id":"call_' | cut -c1-60)"
sed "s/\"max_tokens\":256/\"max_tokens\":$((sna - 1))/" $KC/chat_tools.json > "$T/cut.json"
reset; ask POST $O "$T/cut.json"
is "cut inside the second call: length, the first call, and the rest as text" "$(message | grep -o 'length\|"name":"[a-z_]*"\|"content":"<tool_call>' | tr '\n' ' ')" 'length "content":"<tool_call> "name":"get_weather" '
# a request with no tools block is not read for calls
scripted $GC/chat_tools_none.txt $KA/two.txt || exit 1
ask POST $O $KC/chat_tools_none.json
is "tool_choice none: the answer is its text, whatever it holds" "$code $(message | cut -c1-70)" "200 stop $(printf '%s' '{"role":"assistant","content":"<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {' | cut -c1-65)"
has "and it has no tool_calls" '"finish_reason":"stop"'; if grep -q '"tool_calls"' "$T/body"; then echo "FAIL tool_choice none: the body has tool_calls"; fail=1; fi
# an answer with no call, and one whose block is none
scripted $GC/chat_tools.txt $KA/plain.txt || exit 1
ask POST $O $KC/chat_tools.json
is "tools and an answer with no call: stop, the text as it is" "$(message)" "stop $(cat golden/kolibri_calls/plain.json)"
scripted $GC/chat_tools.txt $KA/arguments_array.txt || exit 1
ask POST $O $KC/chat_tools.json
is "a block that is no call: stop, the text as it is" "$(message)" "stop $(cat golden/kolibri_calls/arguments_array.json)"
# with reasoning: a call inside the reasoning is not one; the content is what is outside the call after it
printf '<think>\nI need the time. <tool_call>{"name":"no","arguments":{}}</tool_call>\n</think>\n\nOne moment.\n<tool_call>\n{"name": "now", "arguments": {}}\n</tool_call>' > "$T/think.txt"
scripted $GC/chat_tools_effort.txt "$T/think.txt" || exit 1
ask POST $O $KC/chat_tools_effort.json
is "with reasoning: the reasoning apart with its tags as text, the content outside the call, one call" "$(message)" 'tool_calls {"role":"assistant","content":"One moment.","reasoning_content":"I need the time. <tool_call>{\"name\":\"no\",\"arguments\":{}}</tool_call>","tool_calls":[{"id":"call_1_0","type":"function","function":{"name":"now","arguments":"{}"}}]}'
# Ollama's route has no tools: an answer there is its text
printf 'hi' > "$T/hi.txt"
body '{"prompt":"hi","raw":true,"tools":[{"name":"f"}],"options":{"num_predict":300}}'
scripted "$T/hi.txt" $KA/two.txt || exit 1
ask POST $G "$T/b.json"
is "/api/generate: an answer with a call is its text" "$(field done_reason) $(grep -c '"tool_calls"' "$T/body") $(grep -c '"response":"<tool_call>' "$T/body")" "stop 0 1"

# --- 6. a session that refuses: a vocabulary of 800 ids where the template's own ids are over it
head -800 "$T/next.txt" > "$T/next800.txt"
start --greedy V=800 NEXT=next800.txt || exit 1
body '{"prompt":"hi","options":{"num_predict":2}}'; ask POST $G "$T/b.json"
st=0; wait "$pid" 2>/dev/null || st=$?; pid=""
if [ "$code" = 200 ] && grep -q '"type":"server_error"' "$T/body" && [ "$st" -eq 1 ] && grep -q "^error\[kolibri-session\]" "$T/err"; then echo "ok   a session that refuses: said in the body, said on standard error, and the server stops with status 1"; else echo "FAIL a session that refuses: status $code, exit $st, $(head -c 200 "$T/body") $(tail -2 "$T/err")"; fail=1; fi
# the command line
for a in "" "$TOK" "$TOK $T/toy.model --port 0" "$TOK $T/toy.model --port x" "$TOK $T/toy.model --frob" "$TOK $T/toy.model --ring 4 --window 10" "$TOK $T/toy.model --end 1 --end 2 --end 3" "$TOK $T/toy.model --pass 1" "$TOK $T/toy.model --pass x"; do
  # shellcheck disable=SC2086
  "$B" $a > /dev/null 2> "$T/u.txt"; st=$?
  if [ $st -eq 2 ] && grep -q "^usage: " "$T/u.txt"; then :; else echo "FAIL the arguments \`$a\`: status $st, $(head -c 200 "$T/u.txt")"; fail=1; fi
done
echo "ok   nine command lines that are no command line: usage, status 2"
"$B" "$T/none.gguf" "$T/toy.model" --port $port > /dev/null 2> "$T/u.txt"; is "a GGUF that is not there" "$? $(cut -d: -f1 "$T/u.txt" | head -1)" "1 error[kolibri-read]"
"$B" "$TOK" "$T/none.model" --port $port --end 858 > /dev/null 2> "$T/u.txt"; is "a model file that is not there" "$? $(cut -d: -f1 "$T/u.txt" | head -1)" "1 error[io]"
"$B" "$TOK" "$T/toy.model" --port $port > /dev/null 2> "$T/u.txt"; is "a GGUF that names no end token, and none given" "$? $(cut -d: -f1 "$T/u.txt" | head -1)" "1 error[kolibri-meta]"
if [ $fail -ne 0 ]; then echo "FAIL kolibri_serve"; exit 1; fi
echo "kolibri_serve: ok"
