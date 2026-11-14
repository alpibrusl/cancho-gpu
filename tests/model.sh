#!/bin/sh
# Model files (docs/resident.md section 31, issue #151): `cancho-gpu model-expand` expands the directives of a model file (const, text, block, use, repeat, if, else, end, `$name`, `${expression}`) into the plain session file.
#
#     tests/model.sh [path to cancho-gpu-session]
#
#  1. every tests/model/ok/X.model (with the definitions in X.args, if there) expands to X.out, byte for byte;
#  2. every tests/model/reject/X.model is refused with the rule `session`, exit 1, nothing on standard output, and the message in X.err;
#  3. the two real models of models/ (the text encoder's 36 layers and the transformer's 32 blocks) expand to files of the recorded SHA-256 (golden/model/): the expansions that tests/text_img_gpu.py and tests/dit_edit_lib.py write,
#     compared byte for byte on a Mac with the checkpoint when they were recorded (docs/resident.md section 31);
#  4. with a CPU session binary: a model file given to it (with its definitions on the command line) answers what its expansion answers.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${CANCHO_GPU:-$ROOT/build/cancho-gpu}
SESSION=${1:-}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
fail=0
n=0
cd "$ROOT"
for f in tests/model/ok/*.model; do
  b=${f%.model}; args=""
  [ -f "$b.args" ] && args=$(cat "$b.args")
  # shellcheck disable=SC2086
  "$BIN" model-expand "$f" $args > "$WORK/out" 2> "$WORK/err" || { echo "$f: refused: $(head -1 "$WORK/err")"; fail=1; continue; }
  cmp -s "$WORK/out" "$b.out" || { echo "$f: the expansion differs from $b.out"; diff "$WORK/out" "$b.out" | head -5; fail=1; }
  n=$((n + 1))
done
for f in tests/model/reject/*.model; do
  b=${f%.model}
  status=0
  "$BIN" model-expand "$f" > "$WORK/out" 2> "$WORK/err" || status=$?
  [ "$status" -eq 1 ] || { echo "$f: exit $status, expected 1"; fail=1; }
  [ ! -s "$WORK/out" ] || { echo "$f: refused but wrote output"; fail=1; }
  grep -q '^error\[session\]' "$WORK/err" || { echo "$f: expected error[session], got: $(head -1 "$WORK/err")"; fail=1; }
  [ "$(head -1 "$WORK/err")" = "$(head -1 "$b.err")" ] || { echo "$f: the message changed: $(head -1 "$WORK/err")"; fail=1; }
  n=$((n + 1))
done
check() {
  name=$1; shift
  got=$("$BIN" model-expand "$ROOT/models/$name.model" "$@" | shasum -a 256 | cut -d' ' -f1)
  want=$(cat "$ROOT/golden/model/$name.$tag.sha256")
  [ "$got" = "$want" ] || { echo "models/$name.model ($tag): expansion $got, recorded $want"; fail=1; }
  n=$((n + 1))
}
tag=ref;     check qwen3vl_text CHX=/chx WORK=/w SP=1088 NOREF=0
tag=noref;   check qwen3vl_text CHX=/chx WORK=/w SP=1088 NOREF=1
tag=step;    check qwen_image_dit CHX=/chx WORK=/w WDIR=/wd NP=4160
if [ -n "$SESSION" ]; then
  "$BIN" model-expand tests/model/chain.model CHX=tests/chx > "$WORK/chain.session"
  a=$(echo 1 | "$SESSION" "$WORK/chain.session")
  b=$(echo 1 | "$SESSION" tests/model/chain.model CHX=tests/chx)
  [ "$a" = "$b" ] && [ -n "$a" ] || { echo "the session on a model file answers '$b', on its expansion '$a'"; fail=1; }
  # 1 2 3 4 / 5 6 7 8 plus three times 10 20 30 40 / 50 60 70 80
  [ "$a" = "3.1e1 6.2e1 9.3e1 1.24e2 1.55e2 1.86e2 2.17e2 2.48e2" ] || { echo "the chain answers '$a'"; fail=1; }
  n=$((n + 1))
fi
echo "$n model-file checks"
exit $fail
