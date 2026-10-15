#!/bin/sh
# Hold the batched prefill of `cancho-gpu-run35` to the single-token one: the same prompt
# (long enough for several chunks of eight) run both ways must give the same tokens and
# the same five largest logits at every step to within 1e-3. Needs a Mac and the model
# in Ollama's store; not in the gate.
#
#     CANCHO=<pinned cancho> scripts/qwen35-batch-check.sh [ollama store]
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STORE=${1:-$HOME/.ollama/models}
"$ROOT/scripts/build-run35.sh" >/dev/null
PROMPT="The history of computing is a history of abstraction. Early programmers wrote machine code by hand, then assemblers gave names to instructions, then compilers let people describe what they wanted rather than how to get it. Today a single line of code can"
single=$("$ROOT/build/cancho-gpu-run35" "$STORE" "$ROOT/tests/chx" "$PROMPT" 8 --trace --single 2>&1)
batch=$("$ROOT/build/cancho-gpu-run35" "$STORE" "$ROOT/tests/chx" "$PROMPT" 8 --trace 2>&1)
python3 - "$single" "$batch" <<'PY'
import re, sys
def parse(text):
    toks = [l for l in text.splitlines() if l.startswith("tokens:")]
    steps = [l for l in text.splitlines() if l.startswith("step")]
    return toks, steps
(ts, ss), (tb, sb) = parse(sys.argv[1]), parse(sys.argv[2])
ok = ts == tb and len(ss) == len(sb) and len(ss) > 0
print("same tokens" if ts == tb else f"TOKENS DIFFER: {ts} against {tb}")
worst = 0.0
for a, b in zip(ss, sb):
    la = [(int(i), float(v)) for i, v in re.findall(r"(\d+)=(-?[\d.]+)", a)]
    lb = [(int(i), float(v)) for i, v in re.findall(r"(\d+)=(-?[\d.]+)", b)]
    if [i for i, _ in la] != [i for i, _ in lb]:
        ok = False
        print("TOP FIVE DIFFER:", a, "|", b)
    worst = max([worst] + [abs(x - y) for (_, x), (_, y) in zip(la, lb)])
print(f"{len(ss)} steps, worst top-five logit difference {worst:.4f}")
ok = ok and worst <= 1e-3
sys.exit(0 if ok else 1)
PY
