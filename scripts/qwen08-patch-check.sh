#!/bin/sh
# Check C10 (docs/weight-space.md section 14.6 and 14.12) on the real Qwen3.5-0.8B checkpoints: the edit as a patch.
#
#   1. examples/weightspace/w1_patch.py makes the patch from ~/models/qwen35-08b-base and ~/models/qwen35-08b-edited, reads it back
#      and applies it with its own reader: base + patch equals edited in all 488 tensors (sha256 of the raw bytes).
#   2. cancho-gpu-qwen08-load (the mock driver, no GPU) loads the base with `--patch` and every one of the 320 text tensors, f32 and f16,
#      equals the device bytes the edited file gives (Adler-32 and the count of values narrowing changes, tests/qwen08_load.py model ... patch).
#   3. On a Mac with Metal, cancho-gpu-run08 --patch: base + patch prints the same logits as the edited checkpoint, bit for bit, and the
#      edit's effect matches Hugging Face's (examples/qwen35/qwen08_c10.py).
#
# Needs numpy and both checkpoints (step 3: a Mac, pyarrow and ~/models/gsm8k); says so and exits 0 when they are absent. Not in the gate
# (tests/qwen08_load.sh holds the format and the refusals on a toy checkpoint). Files go under build/c10/.
#
#     CANCHO=<pinned cancho> scripts/qwen08-patch-check.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODELS=${MODELS_DIR:-$HOME/models}
BASE=$MODELS/qwen35-08b-base
EDITED=$MODELS/qwen35-08b-edited
if ! python3 -c 'import numpy' 2>/dev/null; then
  echo "skip qwen08-patch-check: numpy is not installed"; exit 0
fi
for d in "$BASE" "$EDITED"; do
  if [ ! -f "$d/config.json" ]; then
    echo "skip qwen08-patch-check: no checkpoint in $d"; exit 0
  fi
done
OUT=$ROOT/build/c10
mkdir -p "$OUT"
python3 "$ROOT/examples/weightspace/w1_patch.py" make "$BASE" "$EDITED" -o "$OUT/w1.patch" --json "$OUT/w1_patch.json"
python3 "$ROOT/examples/weightspace/w1_patch.py" check "$BASE" "$OUT/w1.patch" "$EDITED"
MOCK=1 "$ROOT/scripts/build-qwen08-load.sh" >/dev/null
fail=0
for mode in f32 f16; do
  python3 "$ROOT/tests/qwen08_load.py" model "$ROOT/build/cancho-gpu-qwen08-load-mock" "$BASE" "$mode" "$OUT/w1.patch" "$EDITED" || fail=1
done
if [ "$(uname -s)" != Darwin ]; then
  echo "skip the engine half of qwen08-patch-check: needs Metal (a Mac)"; exit $fail
fi
if ! python3 -c 'import pyarrow' 2>/dev/null || [ ! -f "$MODELS/gsm8k/main/test-00000-of-00001.parquet" ]; then
  echo "skip the engine half of qwen08-patch-check: needs pyarrow and the GSM8K parquet in $MODELS/gsm8k"; exit $fail
fi
"$ROOT/scripts/build-run08.sh" >/dev/null
python3 "$ROOT/examples/qwen35/qwen08_c10.py" --models-dir "$MODELS" --patch "$OUT/w1.patch" --work "$OUT" || fail=1
exit $fail
