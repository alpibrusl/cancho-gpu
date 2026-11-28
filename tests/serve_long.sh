#!/bin/sh
# The 27B server past 1,024 positions (docs/serving.md section 12), by hand: needs a Mac, the model in Ollama's store and a built
# `cancho-gpu-serve`; not in the gate (no model there). Three servers are started one after another on one port:
#
#   1. with `--long-attention` (the long route of attention at every length) and 2. without it, on prompts under 1,024 tokens: the same
#      greedy answers, so the long route computes what the one-tile kernels compute;
#   3. with the prefix cache and 2. again with `--no-prefix-cache`, on prompts of about 11,000 tokens that part from one another at the
#      end, in the middle, and not at all: the same greedy answers, and the tokens reused are the checkpoint the design says;
#   and on /api/generate, an answer in Ollama's form and the refusals in its `{"error": ...}`.
#
#     tests/serve_long.sh [ollama store]
#
# The long text is the first 44,000 bytes of docs/serving.md (about 11,000 tokens). Exit 0 when every check holds.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STORE=${1:-$HOME/.ollama/models}
BIN=${SERVE:-$ROOT/build/cancho-gpu-serve}
PORT=${PORT:-18599}
W=$(mktemp -d "${TMPDIR:-/tmp}/serve_long.XXXXXX")
failed=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: [$2] where [$3]"; failed=1; fi; }

start() {
  "$BIN" "$STORE" "$ROOT/tests/chx" --port "$PORT" "$@" > "$W/server.log" 2>&1 &
  pid=$!
  i=0
  while [ $i -lt 120 ] && ! grep -q 'ready' "$W/server.log"; do
    kill -0 $pid 2>/dev/null || { echo "FAIL the server did not start: $(cat "$W/server.log")"; exit 1; }
    perl -e 'sleep 1'; i=$((i + 1))
  done
}
stop() { kill $pid 2>/dev/null; wait $pid 2>/dev/null; }
# body FILE TEXT [max_tokens]: a greedy chat request of one user message
body() { jq -n --arg c "$2" --argjson n "${3:-48}" '{messages: [{role: "user", content: $c}], temperature: 0, max_tokens: $n}' > "$1"; }
answer() { curl -s "localhost:$PORT/v1/chat/completions" -d @"$1" | jq -r '.choices[0].message.content'; }
reused() { curl -s "localhost:$PORT/v1/chat/completions" -d @"$1" | jq -r '[.usage.prompt_tokens_details.cached_tokens, .choices[0].message.content] | @json'; }

head -c 44000 "$ROOT/docs/serving.md" > "$W/long.txt"
half=$(head -c 22000 "$W/long.txt")
rest=$(tail -c 22000 "$W/long.txt")
long=$(cat "$W/long.txt")
body "$W/s1.json" "Name three colours."
body "$W/s2.json" "Write a cancho function that returns the larger of two ints. Code only."
body "$W/s3.json" "$(head -c 3000 "$W/long.txt")

In one sentence: what is this about?"
body "$W/a.json" "$long

In one sentence: what does section 7 add?"
body "$W/b.json" "$long

In one sentence: what does section 9 measure?"
body "$W/c.json" "$half
(an aside)
$rest

In one sentence: what does section 7 add?"

# 1 and 2: the long route against the one-tile kernels
start --long-attention --no-spec
for f in s1 s2 s3; do answer "$W/$f.json" > "$W/$f.long"; done
stop
start --no-prefix-cache --no-spec
for f in s1 s2 s3; do answer "$W/$f.json" > "$W/$f.short"; check "the long route answers $f as the one-tile kernels do" "$(cat "$W/$f.long")" "$(cat "$W/$f.short")"; done
for f in a b c; do answer "$W/$f.json" > "$W/$f.nocache"; done
stop

# 3: the prefix cache with its checkpoints, against no cache
# (with --no-spec on both sides: with the draft head a greedy answer at 13,000 positions can depend on the requests before it,
# docs/serving.md 13.2, which would make this a check of that and not of the cache)
start --no-spec
a=$(reused "$W/a.json"); b=$(reused "$W/b.json"); c=$(reused "$W/c.json"); b2=$(reused "$W/b.json")
# (printf, not echo, below: sh's echo turns a JSON string's \n into a line break)
check "a first long prompt reuses nothing" "$(printf '%s' "$a" | jq -r '.[0]')" 0
check "it answers as with no cache" "$(printf '%s' "$a" | jq -r '.[1]')" "$(cat "$W/a.nocache")"
nb=$(printf '%s' "$b" | jq -r '.[0]')
nt=$(curl -s "localhost:$PORT/v1/chat/completions" -d @"$W/b.json" | jq -r '.usage.prompt_tokens')
# a checkpoint or the chat's cut, whichever is later: either way under 1,024 positions are read again, at a multiple of the chunk
check "one that parts from it at the end reads under 1,024 positions again" "$([ "$nb" -ge $((nt - 1024)) ] && [ $((nb % 16)) -eq 0 ] && echo yes)" yes
check "it answers as with no cache" "$(printf '%s' "$b" | jq -r '.[1]')" "$(cat "$W/b.nocache")"
nc=$(printf '%s' "$c" | jq -r '.[0]')
check "one edited in the middle reuses the checkpoint before the edit" "$([ "$nc" -ge 4096 ] && [ "$nc" -le 6144 ] && [ $((nc % 1024)) -eq 0 ] && echo yes)" yes
check "it answers as with no cache" "$(printf '%s' "$c" | jq -r '.[1]')" "$(cat "$W/c.nocache")"
check "the first again, after the edit, reuses what it shares with the edit" "$(printf '%s' "$b2" | jq -r '.[0]')" "$nc"
check "and answers as with no cache" "$(printf '%s' "$b2" | jq -r '.[1]')" "$(cat "$W/b.nocache")"

# /api/generate
g() { curl -s -w ' %{http_code}' "localhost:$PORT/api/generate" -d "$1"; }
r=$(g '{"model":"m","prompt":"Name three colours.","stream":false,"think":false,"options":{"temperature":0,"num_predict":48,"num_ctx":16384}}')
check "/api/generate answers in Ollama's form" "$(printf '%s' "$r" | sed 's/ [0-9]*$//' | jq -r '[.model, .done, .done_reason, .response] | @json')" "$(jq -n --arg r "$(cat "$W/s1.short")" '["m", true, "stop", $r] | @json' | jq -r .)"
check "a stream is refused" "$(g '{"prompt":"x","stream":true}')" '{"error":"a stream is not built on /api/generate: send `stream: false`"} 400'
check "a missing prompt is refused" "$(g '{"stream":false}')" '{"error":"`prompt` is missing"} 400'
check "a prompt that is no string is refused" "$(g '{"prompt":5}')" '{"error":"`prompt` must be a string"} 400'
check "an option that is no number is refused" "$(g '{"prompt":"x","options":{"top_p":"high"}}')" '{"error":"`options.top_p` must be a number"} 400'
stop
rm -rf "$W"
exit $failed
