#!/bin/sh
# The Qwen3.5-0.8B engine's front door (docs/weight-space.md section 14, T6), against the mock Metal driver (no GPU, no model):
# `device/qwen08.cho` and its runner compile and link, a usage mistake is exit 2, and the refusals that precede any kernel are
# refused with their rule tags: a `tmax` that is not a multiple of 64 (`q08-tmax`), a directory without the checkpoint
# (`q08-io`), a checkpoint that is not Qwen3.5-0.8B's dimensions (`q08-engine`: the toy checkpoint of tests/qwen08/fixtures.py).
# What the engine computes needs the real checkpoint and a Mac: `examples/qwen35/qwen08_c34.py` (checks C3 and C4).
#
#     CANCHO=<pinned cancho> tests/qwen08_engine.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MOCK=1 "$ROOT/scripts/build-run08.sh" >/dev/null
BIN=$ROOT/build/cancho-gpu-run08-mock
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
python3 "$ROOT/tests/qwen08/fixtures.py" "$WORK"
fail=0
expect() { # <exit code> <text in stderr> <args...>
  want=$1; pat=$2; shift 2
  set +e
  out=$("$BIN" "$@" 2>&1)
  got=$?
  set -e
  case "$out" in *"$pat"*) found=1 ;; *) found=0 ;; esac
  if [ "$got" -ne "$want" ] || [ "$found" -ne 1 ]; then
    echo "FAIL qwen08_engine: $* -> exit $got, wanted $want with '$pat': $out"; fail=1
  else
    echo "ok   qwen08_engine: $pat"
  fi
}
expect 2 "usage:" 
expect 2 "usage:" "$WORK/ok" "$ROOT/tests/chx" f64 64 1
expect 2 "usage:" "$WORK/ok" "$ROOT/tests/chx" f32 64 1,x
expect 1 "q08-range" "$WORK/ok" "$ROOT/tests/chx" f32 64 ""
expect 1 "q08-tmax" "$WORK/ok" "$ROOT/tests/chx" f32 100 1
expect 1 "q08-io" "$WORK/no-such-dir" "$ROOT/tests/chx" f32 64 1
expect 1 "q08-engine" "$WORK/ok" "$ROOT/tests/chx" f32 64 1
# The greedy runner's front door (T7): flags, the cap against tmax, the batch file, a request prompt (needs tokenizer.json).
expect 2 "usage:" "$WORK/ok" "$ROOT/tests/chx" f32 64 1 --max-new
expect 2 "usage:" "$WORK/ok" "$ROOT/tests/chx" f32 64 1 --max-new x
expect 2 "usage:" "$WORK/ok" "$ROOT/tests/chx" f32 64 1 --force 1,x
expect 2 "usage:" "$WORK/ok" "$ROOT/tests/chx" f32 64 1 --batch "$WORK/b"
expect 1 "q08-range" "$WORK/ok" "$ROOT/tests/chx" f32 64 1,2,3 --max-new 62
expect 1 "q08-range" "$WORK/ok" "$ROOT/tests/chx" f32 64 1,2,3 --force 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62
expect 1 "q08-io" "$WORK/ok" "$ROOT/tests/chx" f32 64 @"$WORK/req.json"
printf '3 1,2,3 4\nnot a line\n' > "$WORK/bad.batch"
expect 2 "a batch line" "$WORK/ok" "$ROOT/tests/chx" f32 64 - --batch "$WORK/bad.batch"
expect 1 "q08-io" "$WORK/ok" "$ROOT/tests/chx" f32 64 - --batch "$WORK/no-such.batch"
printf '3 1,2,3\n' > "$WORK/good.batch"
expect 1 "q08-io" "$WORK/ok" "$ROOT/tests/chx" f32 64 - --batch "$WORK/good.batch"
exit $fail
