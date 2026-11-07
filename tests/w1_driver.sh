#!/bin/sh
# `cancho-gpu-w1`'s front door (docs/weight-space.md section 14, T9; device/w1-main.cho), against the mock Metal driver (no GPU, no model):
# it compiles and links, a usage mistake is exit 2, and what precedes the first kernel is refused with a rule tag (a directory without a
# tokenizer is `q08-io`). The jobs themselves need the engine on a Mac: the smoke run in docs/weight-space.md 14.12; the Python half
# (the protocol, the records, the seeds, resume) is tests/w1_cancho.py against tests/w1c/fake_engine.py.
#
#     CANCHO=<pinned cancho> tests/w1_driver.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MOCK=1 "$ROOT/scripts/build-w1.sh" >/dev/null 2>&1
BIN=$ROOT/build/cancho-gpu-w1-mock
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/empty"
fail=0
expect() { # <exit code> <text in stderr> <args...>
  want=$1; pat=$2; shift 2
  set +e
  out=$("$BIN" "$@" 2>&1 </dev/null)
  got=$?
  set -e
  case "$out" in *"$pat"*) found=1 ;; *) found=0 ;; esac
  if [ "$got" -ne "$want" ] || [ "$found" -ne 1 ]; then
    echo "FAIL w1_driver: $* -> exit $got, wanted $want with '$pat': $out"; fail=1
  else
    echo "ok   w1_driver: $pat"
  fi
}
S="--temperature 0.6 --top-p 0.95 --top-k 20 --presence 1.5"
expect 2 "usage:"
# shellcheck disable=SC2086
expect 2 "usage:" "$WORK/empty" "$ROOT/tests/chx" f64 64 $S
expect 2 "usage:" "$WORK/empty" "$ROOT/tests/chx" f32 x $S
expect 2 "usage:" "$WORK/empty" "$ROOT/tests/chx" f32 64 --temperature 0.6 --top-p 0.95 --top-k 20
expect 2 "usage:" "$WORK/empty" "$ROOT/tests/chx" f32 64 --temperature 0.6 --top-p 0.95 --top-k 20 --seed 1
expect 2 "usage:" "$WORK/empty" "$ROOT/tests/chx" f32 64 --temperature x --top-p 0.95 --top-k 20 --presence 1.5
expect 2 "usage:" "$WORK/empty" "$ROOT/tests/chx" f32 64 --temperature 0.6 --top-p 1.5 --top-k 20 --presence 1.5
expect 2 "usage:" "$WORK/empty" "$ROOT/tests/chx" f32 64 --temperature 0.6 --top-p 0.95 --top-k 0 --presence 1.5
# shellcheck disable=SC2086
expect 2 "usage:" "$WORK/empty" "$ROOT/tests/chx" f32 64 $S --patch
# shellcheck disable=SC2086
expect 2 "usage:" "$WORK/empty" "$ROOT/tests/chx" f32 64 $S --patch a b
# shellcheck disable=SC2086
expect 1 "q08-io" "$WORK/empty" "$ROOT/tests/chx" f32 64 $S --patch "$WORK/none.patch"
# shellcheck disable=SC2086
expect 2 "usage:" "$WORK/empty" "$ROOT/tests/chx" f32 64 $S --require-evidence yes
# shellcheck disable=SC2086
expect 2 "usage:" "$WORK/empty" "$ROOT/tests/chx" f32 64 $S --patch a --require-evidence maybe
# shellcheck disable=SC2086
expect 1 "q08-io" "$WORK/empty" "$ROOT/tests/chx" f32 64 $S
# shellcheck disable=SC2086
expect 1 "q08-io" "$WORK/no-such-dir" "$ROOT/tests/chx" f32 64 $S
exit $fail
