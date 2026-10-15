#!/bin/sh
# Run a checkpoint on the Metal GPU with `cancho-gpu-run` and hold it to the
# same checkpoint run by Hugging Face transformers in float32 on the CPU
# (tests/model/reference.py), greedy: the same tokens, and the same five largest
# logits at every position to within 1e-3. Needs a Mac, the checkpoint, python3
# with torch and transformers; not in the gate.
#
#     CANCHO=<pinned cancho> scripts/model-check.sh <model dir> [new tokens]
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:?usage: model-check.sh <model dir> [new tokens]}
NEW=${2:-12}
"$ROOT/scripts/build-run.sh" >/dev/null
fail=0
for prompt in "The capital of France is" "def fibonacci(n):
    " "Once upon a time, in a land far away,"; do
  ours=$("$ROOT/build/cancho-gpu-run" "$MODEL" "$ROOT/tests/lx" "$prompt" "$NEW" --trace)
  ref=$(python3 "$ROOT/tests/model/reference.py" "$MODEL" "$NEW" "$prompt" 2>/dev/null)
  ids=$(echo "$ours" | sed -n 's/^tokens: //p')
  follow=$(python3 "$ROOT/tests/model/reference.py" --follow "$ids" "$MODEL" 0 "$prompt" 2>/dev/null)
  want=$(echo "$ref" | sed -n 's/^tokens: //p')
  if [ "$ids" = "$want" ]; then t="same tokens"; else t="TOKENS DIFFER ($ids against $want)"; fail=1; fi
  worst=$(python3 - "$follow" "$ours" <<'PY'
import sys, re
ref = [l for l in sys.argv[1].splitlines() if l.startswith("step")]
our = [l for l in sys.argv[2].splitlines() if l.startswith("step")]
worst = 0.0
for a, b in zip(ref, our):
    ia = [x.split("=") for x in a.split("top: ")[1].split()]
    ib = [x.split("=") for x in b.split("top: ")[1].split()]
    if [i for i, _ in ia] != [i for i, _ in ib]:
        print("ORDER-DIFFERS"); sys.exit()
    worst = max(worst, max(abs(float(x) - float(y)) for (_, x), (_, y) in zip(ia, ib)))
print(f"{worst:.4f}")
PY
)
  case "$worst" in ORDER*) o="top-5 order differs"; fail=1 ;; *) o="worst top-5 logit difference $worst"; [ "$(echo "$worst" | awk '{print ($1 <= 0.001) ? 1 : 0}')" = 1 ] || fail=1 ;; esac
  echo "$(echo "$prompt" | head -1 | cut -c1-30): $t; $o"
done
exit $fail
