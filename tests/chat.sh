#!/bin/sh
# Hold `cancho-gpu chat` (the chat template of `src/chat.cho`) to the model's own template: every request of
# tests/chat/ rendered by this compiler and by jinja2 (tools/chat_template.py) must agree byte for byte, or
# both must refuse (a `reject_` request). Needs python3 with jinja2 and the model's tokenizer_config.json
# (golden/chat/tokenizer_config.json, the one in Ollama's store).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${CANCHO_GPU:-$ROOT/build/cancho-gpu}
CFG=$ROOT/golden/chat/tokenizer_config.json
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
pass=0
fail=0
for req in "$ROOT"/tests/chat/*.json; do
  name=$(basename "$req" .json)
  ours=0; theirs=0
  "$BIN" chat "$req" >"$WORK/ours" 2>"$WORK/ours.err" || ours=$?
  python3 "$ROOT/tools/chat_template.py" "$CFG" "$req" >"$WORK/theirs" 2>"$WORK/theirs.err" || theirs=$?
  case "$name" in
    reject_*)
      if [ "$ours" -eq 1 ] && [ "$theirs" -eq 1 ]; then pass=$((pass + 1)); else echo "$name: expected both to refuse (ours $ours, jinja2 $theirs)"; fail=$((fail + 1)); fi ;;
    unsupported_*)
      # The template renders it (an image placeholder); this server refuses it by name.
      if [ "$ours" -eq 1 ]; then pass=$((pass + 1)); else echo "$name: expected this server to refuse (ours $ours)"; fail=$((fail + 1)); fi ;;
    *)
      if [ "$ours" -ne 0 ] || [ "$theirs" -ne 0 ]; then echo "$name: refused (ours $ours: $(head -1 "$WORK/ours.err"); jinja2 $theirs: $(head -1 "$WORK/theirs.err"))"; fail=$((fail + 1))
      elif cmp -s "$WORK/ours" "$WORK/theirs"; then pass=$((pass + 1))
      else echo "$name: differs from the template"; diff "$WORK/ours" "$WORK/theirs" | head -8; fail=$((fail + 1)); fi ;;
  esac
done
echo "$pass of $((pass + fail)) chat requests agree with the model's template"
[ "$fail" -eq 0 ]
