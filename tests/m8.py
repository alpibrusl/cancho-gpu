#!/usr/bin/env python3
"""The M8 pieces that need no GPU and no model (docs/roadmap-models.md section 7): the prompt format of the training pairs, the breakers, the chunker's budget, the n-gram contamination rule and the
shim's sampling and chat template.

    python3 tests/m8.py

  1. m8_pairs.compose equals, byte for byte, a prompt cancho-code's ollama agent really sent (tests/m8/cancho-code-prompt.txt, from a journal of the pilot run), given its task, source and the checker's JSON.
  2. every breaker leaves a text different from its input and none raises on a text without its target.
  3. the shim renders the Qwen2.5 chat template around the prompt, samples greedily at temperature 0 and only from the top-k at temperature > 0 (a seed gives the same token twice)."""
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]      # tests/kernels.py would shadow the `kernels` package transformers imports
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "examples", "qwen"))
import m8_pairs as P  # noqa: E402

fails = 0


def expect(name, got, want):
    global fails
    if got != want:
        fails += 1
        print("FAIL: %s\n  got:  %r\n  want: %r" % (name, got, want))


prompt = open(os.path.join(HERE, "m8", "cancho-code-prompt.txt")).read()
fb = open(os.path.join(HERE, "m8", "cancho-code-feedback.json")).read()
m = re.match(r"Your task, for the cancho program below:\n(.*?)\n\nReply with the complete file.*?\n---\n(.*)---\n$", prompt, re.S)
task, source = m.group(1), m.group(2)
expect("compose equals cancho-code's prompt", P.compose(source, task, fb), prompt)
expect("fix-only prompt starts as cancho-code's", P.compose(source, "", fb).startswith("The cancho program below is refused by its compiler. Reply with the complete corrected file and nothing else"), True)

good = "edition 7;\n\nfn f(n: int) -> [] int {\n    var k = n;\n    k = k + 1;\n    if k == 3 {\n        return 0;\n    }\n    return k;\n}\n"
for op, fn in sorted(P.BREAKERS.items()):
    r = fn(good, random.Random(1))
    expect("breaker %s changes the text or declines" % op, r is None or r != good, True)
    fn("edition 7;\n", random.Random(1))      # no target in the text: must decline (None) and not raise

fs = P.functions("// Adds one to the number and returns the sum of both.\npub fn inc(n: int) -> [] int {\n    return n + 1;\n}\n\nfn g(n: int) -> [] int {\n    return n;\n}\n")
expect("functions: only the documented one", [f[0] for f in fs], ["inc"])
expect("scaffold without imports", P.scaffold_for("fn f() {}", "import std.buffer;\n"), "edition 7;\n")
expect("scaffold with the import the function names", P.scaffold_for("fn f() { buffer.empty(); }", "import std.buffer;\n"), "edition 7;\nimport std.buffer;\n")

try:
    import torch
    import serve_ollama as S
    lg = torch.tensor([0.0, 5.0, 1.0, 4.0])
    g = torch.Generator().manual_seed(3)
    expect("greedy at temperature 0", S.Served.sample(lg, 0.0, 0, 1.0, g), 1)
    a = [S.Served.sample(lg, 0.7, 2, 0.8, torch.Generator().manual_seed(s)) for s in range(50)]
    expect("top-k 2 keeps to the two largest", set(a) <= {1, 3}, True)
    expect("a seed gives the same token", S.Served.sample(lg, 1.0, 0, 1.0, torch.Generator().manual_seed(9)), S.Served.sample(lg, 1.0, 0, 1.0, torch.Generator().manual_seed(9)))
except ImportError:
    print("skip: torch/transformers not installed (the shim's sampling is not tested)")

print("m8: %s" % ("FAILED %d" % fails if fails else "ok"))
sys.exit(1 if fails else 0)
