#!/bin/sh
# tests/qwen3_asr_greedy.sh -- stage A2 of docs/audio.md by the slow road: a greedy answer of Qwen3-ASR's language model, one whole pass of the session a token and no cache. Needs a Metal GPU and the weights
# (`build/qwen3_asr_text_gpu weights`), so it is not in the gate.
#
#     tests/qwen3_asr_greedy.sh <weights dir> <work dir> <prompt ids.txt> <audio rows .f32> [<most tokens, default 128>]
#
# The prompt's ids are copied to <work>/ids.txt and each pass's arg max appended, until <|im_end|> (151645) or <|endoftext|> (151643). Prints the generated ids on one line.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$1; K=$2; PROMPT=$3; ROWS=$4; MOST=${5:-128}
B=$ROOT/build/qwen3_asr_text_gpu
mkdir -p "$K"
tr -s ' \n\t' ' ' < "$PROMPT" > "$K/ids.txt"
out=""
n=0
while [ "$n" -lt "$MOST" ]; do
  sp=$("$B" inputs "$W" "$K" "$K/ids.txt" "$ROWS" | sed -n 's/^SP=\([0-9]*\) .*/\1/p')
  [ -n "$sp" ] || { echo "inputs failed"; exit 1; }
  printf '1\ncheckpoint\n' | "$ROOT/build/cancho-gpu-metal-session" "$ROOT/models/qwen3_asr_text.model" CHX="$ROOT/tests/chx" WORK="$K" WDIR="$W" SP="$sp" > /dev/null 2> "$K/session.err" || { echo "the session failed: $(tail -1 "$K/session.err")"; exit 1; }
  id=$("$B" next "$K" "$K/ids.txt")
  printf ' %s' "$id" >> "$K/ids.txt"
  out="$out $id"
  n=$((n + 1))
  if [ "$id" = 151645 ] || [ "$id" = 151643 ]; then break; fi
done
echo "generated:$out"
