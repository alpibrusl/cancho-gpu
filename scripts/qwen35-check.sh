#!/bin/sh
# Run Qwen3.8-27B on the Metal GPU with `cancho-gpu-run35` and hold it to the
# reference implementation's golden fixture (crates/lex-rt/tests/data/qwen35_27b_golden.txt:
# an f32 reference over the same NVFP4 weights): the same token at every step, the same
# five tokens in the top five, and the gaps between their logits within 0.01 of the
# reference's log-probability gaps. Needs a Mac, the model in Ollama's store (~/.ollama/models)
# and a reference checkout; not in the gate.
#
#     CANCHO=<pinned cancho> scripts/qwen35-check.sh <reference checkout> [new tokens]
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REF=${1:?usage: qwen35-check.sh <reference checkout> [new tokens]}
GOLDEN=$REF/crates/lex-rt/tests/data/qwen35_27b_golden.txt
"$ROOT/scripts/build-run35.sh" >/dev/null
python3 - "$ROOT" "$GOLDEN" "${2:-8}" <<'PY'
import subprocess, sys

root, golden, new = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompts = ["The capital of France is", "def fibonacci(n):", "Once upon a time, in a small village,"]
cases, cur = [], None
for line in open(golden).read().splitlines():
    if line.startswith("case"):
        cur = []
        cases.append(cur)
    elif line.startswith("step") and cur is not None:
        parts = line.split()
        cur.append((int(parts[1]), [(int(x.split(":")[0]), float(x.split(":")[1])) for x in parts[2:]]))
fail = False
for prompt, steps in zip(prompts, cases):
    n = min(new, len(steps))
    out = subprocess.run([root + "/build/cancho-gpu-run35", __import__("os").path.expanduser("~/.ollama/models"),
                          root + "/tests/chx", prompt, str(n), "--trace"], capture_output=True, text=True).stdout
    ours = [[(int(a.split("=")[0]), float(a.split("=")[1])) for a in l.split("top: ")[1].split()]
            for l in out.splitlines() if l.startswith("step")]
    tokens = [int(x) for l in out.splitlines() if l.startswith("tokens:") for x in l.split()[1:]]
    worst, tokens_ok, sets_ok = 0.0, True, True
    for k in range(n):
        want_next, top = steps[k]
        o = ours[k]
        tokens_ok &= tokens[k] == want_next
        sets_ok &= [a for a, _ in top] == [a for a, _ in o]
        gm, om = dict(top), dict(o)
        for x in gm:
            if x in om:
                worst = max(worst, abs((gm[x] - top[0][1]) - (om[x] - o[0][1])))
    ok = tokens_ok and sets_ok and worst <= 0.01
    fail |= not ok
    print(f"{prompt[:34]!r}: {n} steps, {'same tokens' if tokens_ok else 'TOKENS DIFFER'}, "
          f"{'same top five' if sets_ok else 'TOP FIVE DIFFERS'}, worst logit-gap difference {worst:.4f}")
sys.exit(1 if fail else 0)
PY
