#!/bin/sh
# The Qwen3.5-0.8B weight loader (docs/weight-space.md section 14, T5), on a toy
# checkpoint with the 0.8B layout (tests/qwen08/fixtures.py) and against the mock
# Metal driver, whose buffers are host memory (no GPU):
#
#   * every text tensor, on the f32 path and the f16 path, is read back from the
#     "device" and equals an oracle written apart from the loader (bf16 by a shift,
#     IEEE half conversion, Adler-32), the F32 tensors stay F32, and the
#     vision and mtp tensors are not loaded; the embedding holds all 65,536 bf16
#     bit patterns;
#   * each refusal fixture is refused with its rule tag, every `q08-` rule in
#     the sources has one (or a stated reason it cannot), the edit patch (T11, `src/qwen08_patch.cho`): base + patch equals the edited toy tensor by
#     tensor, a malformed or mismatched patch is refused with its rule tag, hostile patches never trap;
#   * `--require-evidence` (docs/weight-space-w4.md 15.6): a patch with the evidence `w4_screen.py` writes loads, a missing, malformed, stale or failing
#     evidence is refused with its `q08-evidence-` tag, hostile evidence never traps;
#   * hostile headers and configs never trap.
#
# The real checkpoint is `scripts/qwen08-load-check.sh` (check C2), not the gate.
#
#     CANCHO=<pinned cancho> tests/qwen08_load.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MOCK=1 "$ROOT/scripts/build-qwen08-load.sh" >/dev/null
BIN=$ROOT/build/cancho-gpu-qwen08-load-mock
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
python3 "$ROOT/tests/qwen08/fixtures.py" "$WORK"
if python3 -c 'import numpy' 2>/dev/null; then
  python3 "$ROOT/tests/qwen08/patch_fixtures.py" "$WORK"
else
  echo "skip the patch fixtures: numpy is not installed"
fi
fail=0
python3 "$ROOT/tests/qwen08_load.py" ok "$BIN" "$WORK" || fail=1
python3 "$ROOT/tests/qwen08_load.py" reject "$BIN" "$WORK" || fail=1
python3 "$ROOT/tests/qwen08_load.py" fuzz "$BIN" "$WORK" 300 || fail=1
if python3 -c 'import numpy' 2>/dev/null; then
  python3 "$ROOT/tests/qwen08_load.py" patch "$BIN" "$WORK" || fail=1
  python3 "$ROOT/tests/qwen08_load.py" patch-fuzz "$BIN" "$WORK" 150 || fail=1
  python3 "$ROOT/tests/qwen08_load.py" evidence "$BIN" "$WORK" || fail=1
fi
exit $fail
