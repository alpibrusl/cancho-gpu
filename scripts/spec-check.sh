#!/bin/sh
# Hold speculative decoding (`cancho-gpu-run35 --spec`) to plain greedy decoding: the same tokens, in the same
# order, on a set of prompts, and report how many drafts the model accepted and how fast each ran. A verify that
# accepts a token the model would not have produced is a wrong answer delivered faster. Needs a Mac and the
# model in Ollama's store; not in the gate.
#
#     CANCHO=<pinned cancho> scripts/spec-check.sh [ollama store] [new tokens]
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STORE=${1:-$HOME/.ollama/models}
NEW=${2:-96}
"$ROOT/scripts/build-run35.sh" >/dev/null
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
fail=0
while IFS= read -r prompt; do
  [ -n "$prompt" ] || continue
  prompt=$(printf '%b' "$prompt")
  "$ROOT/build/cancho-gpu-run35" "$STORE" "$ROOT/tests/chx" "$prompt" "$NEW" --timing >"$WORK/plain" 2>&1
  "$ROOT/build/cancho-gpu-run35" "$STORE" "$ROOT/tests/chx" "$prompt" "$NEW" --spec --timing >"$WORK/spec" 2>&1
  if [ "$(grep '^tokens:' "$WORK/plain")" = "$(grep '^tokens:' "$WORK/spec")" ]; then ok="same tokens"; else ok="TOKENS DIFFER"; fail=1; fi
  pms=$(sed -n 's/.*decode_ms=//p' "$WORK/plain")
  sms=$(sed -n 's/.*decode_ms=//p' "$WORK/spec")
  st=$(grep '^spec:' "$WORK/spec")
  printf '%s | %s | plain %s ms, spec %s ms | %s\n' "$(printf '%s' "$prompt" | head -1 | cut -c1-40)" "$ok" "$pms" "$sms" "$st"
done <<'PROMPTS'
The capital of France is
def quicksort(arr):\n    
Once upon a time, in a small village by the sea,
The three laws of thermodynamics, in plain words:\n1.
Q: Why is the sky blue?\nA:
<|im_start|>user\nName three colours.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
rare words: quixotic obelisk marzipan tundra velcro gazebo ferrule
PROMPTS
[ "$fail" -eq 0 ]
