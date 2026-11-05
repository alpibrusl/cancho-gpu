#!/bin/sh
# Checks C5, C6 and C7 (docs/weight-space.md section 14.6) on Metal: the Qwen3.5-0.8B engine's last-position logits, its one-token greedy
# argmax on the 200 W1 prompts and its 64-token greedy decoding of W1 prompts 0 to 19 (free-running and teacher-forced), against the
# float64 Hugging Face goldens in golden/hf08/, for both checkpoints and both weight paths. Builds the runner (scripts/build-run08.sh) first.
# Needs a Mac with Metal, numpy, pyarrow, ~/models/qwen35-08b-base, ~/models/qwen35-08b-edited and ~/models/gsm8k; says so and exits 0 when
# they are absent. Not in the gate. Output, batch files and the runner's raw output go under build/c567/.
#
#     CANCHO=<pinned cancho> scripts/qwen08-check.sh [extra args of examples/qwen35/qwen08_c567.py: --model base --mode f16 --checks c5 ...]
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ "$(uname -s)" != Darwin ]; then
  echo "skip qwen08-check: needs Metal (a Mac)"; exit 0
fi
if ! python3 -c 'import numpy, pyarrow' 2>/dev/null; then
  echo "skip qwen08-check: numpy and pyarrow are needed"; exit 0
fi
MODELS=${MODELS_DIR:-$HOME/models}
for d in qwen35-08b-base qwen35-08b-edited; do
  if [ ! -f "$MODELS/$d/config.json" ]; then
    echo "skip qwen08-check: no checkpoint in $MODELS/$d"; exit 0
  fi
done
if [ ! -f "$MODELS/gsm8k/main/test-00000-of-00001.parquet" ]; then
  echo "skip qwen08-check: no GSM8K parquet in $MODELS/gsm8k"; exit 0
fi
"$ROOT/scripts/build-run08.sh" >/dev/null
exec python3 "$ROOT/examples/qwen35/qwen08_c567.py" --models-dir "$MODELS" "$@"
