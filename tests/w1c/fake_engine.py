#!/usr/bin/env python3
"""A stand-in for build/cancho-gpu-w1 (device/w1-main.cho) speaking its protocol, for tests/w1_cancho.py: no GPU, no model.

argv as the real driver's (<model dir> <kernel dir> f32|f16 <tmax> --temperature T --top-p P --top-k K --presence X); the argv is
appended to $FAKE_LOG as one json line at start, and every job's (id, seed, max_new, question) after it. A job's reply is a function of
(seed, question) only, as the real driver's is: the answer to question `q<i>...` boxes i when the seed is even, else 99; a question
containing "STOP" ends with the stop id after 3 tokens, "CUT" runs to max_new (hit_limit), others stop after 5. Environment:
FAKE_DIE_AFTER=n exits with an error after n jobs; FAKE_NOREADY=1 exits at once with an error."""
import json
import os
import sys

log = os.environ.get("FAKE_LOG")


def note(x):
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(x) + "\n")


note({"argv": sys.argv[1:]})
if os.environ.get("FAKE_NOREADY"):
    sys.stderr.write("error[q08-io]: cannot read tokenizer.json in the model directory\n")
    sys.exit(1)
print("ready", flush=True)
done = 0
for line in sys.stdin:
    if os.environ.get("FAKE_DIE_AFTER") and done >= int(os.environ["FAKE_DIE_AFTER"]):
        sys.stderr.write("error[q08-range]: the prompt (1 or more tokens) and the new tokens must fit tmax\n")
        sys.exit(1)
    jid, seed, max_new, req = line.rstrip("\n").split(" ", 3)
    q = json.loads(req)["messages"][0]["content"]
    note({"id": int(jid), "seed": int(seed), "max_new": int(max_new), "question": q})
    i = int(q.split("\n")[0].split("|")[0][1:])
    text = f"héllo 世界 \\boxed{{{i if int(seed) % 2 == 0 else 99}}}"
    if "STOP" in q:
        toks, eos, hit = [11, 22, 248046], 248046, 0
    elif "CUT" in q:
        toks, eos, hit = list(range(int(max_new))), -1, 1
    else:
        toks, eos, hit = [5, 6, 7, 8, 9], -1, 0
        if len(toks) >= int(max_new):
            toks, hit = toks[:int(max_new)], 1
        else:
            toks, eos, hit = toks + [248044], 248044, 0
    print("tokens", jid, ",".join(map(str, toks)))
    print(f"gen {jid} prompt=7 new={len(toks)} hit_limit={hit} eos={eos} gen_ms=3 text={text.encode().hex()}", flush=True)
    done += 1
