#!/bin/sh
# cancho-gpu-front against two fake runners (front/fake_runner.cho): no model, no GPU (docs/one-server.md section 5).
#
#     front/check.sh            builds with scripts/build-front.sh first (its CANCHO, CANCHO_DIR, WEB_DIR, SCHEMA_DIR)
#     NOBUILD=1 front/check.sh  the binaries in build/ as they are
#
# Checked: the routes the front answers itself; a request forwarded to the runner of the model it names, as it came; the refusals in
# each API's form (no such model, no JSON object, no such route, a wrong method, a missing or wrong key); a swap of runners; a stream
# passed on as it comes, not at its end; a runner that ends without an answer (502), started again by the next request; requests that
# wait their turn; the front's end ending its runner; and its authority report, the one committed in front/authority.txt.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
FRONT=$ROOT/build/cancho-gpu-front
PORT=${PORT:-18710}
[ -n "${NOBUILD:-}" ] || "$ROOT/scripts/build-front.sh" >/dev/null || exit 1
W=$(mktemp -d "${TMPDIR:-/tmp}/front_check.XXXXXX")
mkdir -p "$W/runners"
cp "$ROOT/build/fake-runner" "$W/runners/"
failed=0
n=0
check() { n=$((n + 1)); if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: [$2] where [$3]"; failed=1; fi; }
q() { curl -s -m 20 -w ' %{http_code}' "$@"; }
key=k3y
start() {
  "$FRONT" "$PORT" "$W/runners" "$@" > "$W/front.log" 2>&1 &
  pid=$!
  i=0
  while [ $i -lt 50 ] && ! grep -q listening "$W/front.log"; do perl -e 'select(undef, undef, undef, 0.1)'; i=$((i + 1)); done
}
models="--model alpha fake-runner $((PORT + 1)) alpha --model beta fake-runner $((PORT + 2)) beta"

# shellcheck disable=SC2086
start --api-key "$key" $models
H="Authorization: Bearer $key"
check "/health is open" "$(q localhost:$PORT/health)" '{"status":"ok"} 200'
check "the models, OpenAI's form" "$(q -H "$H" localhost:$PORT/v1/models)" '{"object":"list","data":[{"id":"alpha","object":"model","owned_by":"cancho-gpu"},{"id":"beta","object":"model","owned_by":"cancho-gpu"}]} 200'
check "the models, /api/tags' form" "$(q -H "$H" localhost:$PORT/api/tags)" '{"models":[{"name":"alpha","model":"alpha"},{"name":"beta","model":"beta"}]} 200'
check "no key, OpenAI's form" "$(q localhost:$PORT/v1/chat/completions -d '{"model":"alpha"}')" '{"error":{"message":"send Authorization: Bearer <token>","type":"invalid_request_error","param":null,"code":null}} 401'
check "a wrong key, /api/generate's form" "$(q -H 'Authorization: Bearer nope' localhost:$PORT/api/generate -d '{"model":"alpha"}')" '{"error":"the bearer token is not valid"} 401'
check "a request forwarded to its model's runner, the body as it came" "$(q -H "$H" localhost:$PORT/v1/chat/completions -d '{"model":"alpha","messages":[]}')" '{"runner":"alpha","got":31} 200'
check "/api/generate to the other model: a swap" "$(q -H "$H" localhost:$PORT/api/generate -d '{"model":"beta","prompt":"x"}')" '{"response":"beta","done":true} 200'
check "only one runner is up" "$(pgrep -f "fake-runner --port $((PORT + 1)) " | wc -l | tr -d ' ')" 0
check "a body that names no model goes to the runner that is up" "$(q -H "$H" localhost:$PORT/v1/chat/completions -d '{"messages":[]}')" '{"runner":"beta","got":15} 200'
check "no such model, OpenAI's form" "$(q -H "$H" localhost:$PORT/v1/chat/completions -d '{"model":"gamma"}')" '{"error":{"message":"no such model: GET /v1/models lists them","type":"invalid_request_error","param":null,"code":null}} 404'
check "no such model, /api/generate's form" "$(q -H "$H" localhost:$PORT/api/generate -d '{"model":"gamma"}')" '{"error":"no such model: GET /v1/models lists them"} 404'
check "a body that is no JSON object" "$(q -H "$H" localhost:$PORT/v1/chat/completions -d '[1]')" '{"error":{"message":"the body must be a JSON object","type":"invalid_request_error","param":null,"code":null}} 400'
check "no such route" "$(q -H "$H" localhost:$PORT/nope)" '{"error":{"message":"no such route","type":"invalid_request_error","param":null,"code":null}} 404'
check "a wrong method" "$(q -X DELETE localhost:$PORT/health)" '{"error":{"message":"method not allowed","type":"invalid_request_error","param":null,"code":null}} 405'
# a stream: three events 200 ms apart, read as they come
curl -sN -m 20 -H "$H" "localhost:$PORT/v1/chat/completions" -d '{"model":"alpha","stream":true}' | while IFS= read -r l; do
  [ -n "$l" ] && echo "$(perl -MTime::HiRes=time -e 'printf "%.3f", time') $l"
done > "$W/stream.txt"
check "the stream's events, in order" "$(cut -d' ' -f2- "$W/stream.txt" | tr '\n' '|')" 'data: {"runner":"alpha","n":0}|data: {"runner":"alpha","n":1}|data: {"runner":"alpha","n":2}|data: [DONE]|'
check "they came as they were sent, not at the end" "$(awk 'NR == 1 { a = $1 } { b = $1 } END { print (b - a > 0.45) ? "apart" : "together" }' "$W/stream.txt")" apart
check "a runner that ends without an answer: 502" "$(q -H "$H" localhost:$PORT/v1/chat/completions -d '{"model":"beta","x":"die"}')" '{"error":{"message":"the runner ended without an answer","type":"invalid_request_error","param":null,"code":null}} 502'
check "the next request starts it again" "$(q -H "$H" localhost:$PORT/v1/chat/completions -d '{"model":"beta"}')" '{"runner":"beta","got":16} 200'
# two at once: the second waits for the first (a stream of 0.6 s), and is answered
curl -sN -m 20 -H "$H" "localhost:$PORT/v1/chat/completions" -d '{"model":"beta","stream":true}' > "$W/first.txt" &
cpid=$!
perl -e 'select(undef, undef, undef, 0.1)'
second=$(q -H "$H" localhost:$PORT/v1/chat/completions -d '{"model":"beta"}')
wait $cpid
check "a request that came during a stream waited and was answered" "$second" '{"runner":"beta","got":16} 200'
check "and the stream was whole" "$(grep -c '^data: ' "$W/first.txt")" 4
check "the document is served" "$(curl -s localhost:$PORT/openapi.json -H "$H" | head -c 40)" '{"openapi":"3.1.0","info":{"title":"canc'
kill -TERM $pid
wait $pid 2>/dev/null
perl -e 'select(undef, undef, undef, 0.5)'
check "the front's end ends its runner" "$(pgrep -f "fake-runner --port $((PORT + 2)) " | wc -l | tr -d ' ')" 0

# without a key, every route is open
# shellcheck disable=SC2086
start $models
check "with no key the models are open" "$(q localhost:$PORT/api/tags | tail -c 4)" ' 200'
kill -TERM $pid
wait $pid 2>/dev/null

if [ -n "${CANCHO:-}" ] && [ -d "$ROOT/build/front-deps" ]; then
  "$CANCHO" authority "$ROOT/front/front.cho" "$ROOT"/build/front-deps/*.cho --std 2>/dev/null | sed -n '/^performs/,/^provably/p' | grep -v '^provably' > "$W/authority.txt"
  check "the authority report is front/authority.txt" "$(diff "$W/authority.txt" "$ROOT/front/authority.txt" >/dev/null && echo same)" same
fi
if [ $failed -ne 0 ]; then echo "--- the front's log:"; tail -20 "$W/front.log"; fi
rm -rf "$W"
echo "front/check.sh: $n checks"
exit $failed
