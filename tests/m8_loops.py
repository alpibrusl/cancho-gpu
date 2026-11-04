#!/usr/bin/env python3
"""The W0 repeat detector (docs/weight-space.md section 5.1) on planted data: no model, no GPU.

    python3 tests/m8_loops.py

  1. tail_repeat: a planted loop (a 64-character span 4 times, a cut-off loop, a long unit 4 times after a prefix, a single character 256 times) is found; non-loops (3 copies, a 63-character span
     4 times, a repeat in the middle but not the tail, empty, whitespace, non-strings) are not.
  2. a synthetic arm with looping, limit-hit and clean runs is counted exactly; the decision rule's three branches.
  3. malformed runs (no log, binary garbage, a text before the first header, bad JSON lines, a missing directory) are counted and never raise."""
import json
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "examples", "qwen"))
import m8_loops as L  # noqa: E402

fails = 0


def expect(name, got, want):
    global fails
    if got != want:
        fails += 1
        print("FAIL: %s\n  got:  %r\n  want: %r" % (name, got, want))


rng = random.Random(7)


def noise(n):
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789 ;{}()") for _ in range(n))


span = noise(64)
prefix = noise(300)
expect("4 x 64", L.tail_repeat(span * 4), True)
expect("4 x 64 after a prefix", L.tail_repeat(prefix + span * 4), True)
expect("5 x 64 and trailing newline", L.tail_repeat(prefix + span * 5 + "\n\n"), True)
expect("cut-off loop (ends mid copy)", L.tail_repeat(prefix + span * 4 + span[:30]), True)
big = noise(1000)
expect("4 x 1000", L.tail_repeat(prefix + big * 4), True)
expect("4 x 4096 (longest period)", L.tail_repeat(noise(4096) * 4), True)
expect("one character 256 times", L.tail_repeat(prefix + "-" * 256), True)
expect("3 x 64 is not more than 3", L.tail_repeat(prefix + span * 3), False)
expect("4 x 63 is a span too short", L.tail_repeat(prefix + noise(63) * 4), False)
expect("repeat in the middle, clean tail", L.tail_repeat(span * 6 + noise(200)), False)
expect("empty", L.tail_repeat(""), False)
expect("whitespace only", L.tail_repeat(" \n\t " * 100), False)
expect("short", L.tail_repeat("ab" * 10), False)
expect("None", L.tail_repeat(None), False)
expect("bytes", L.tail_repeat(b"x" * 500), False)
expect("clean prose", L.tail_repeat(noise(5000)), False)
expect("5000 distinct-ish chars with a 4 x 64 not at the tail", L.tail_repeat(span * 4 + noise(10)), False)
# mutants: the thresholds are the registered ones
expect("constants", (L.MIN_SPAN, L.MAX_SPAN, L.COPIES), (64, 4096, 4))


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def raw(*replies):
    return "".join("=== reply (stopped because: %s)\n%s\n" % (stop, body) for stop, body in replies)


def ev(*attempts):
    return "".join(json.dumps({"kind": "attempt", "unit": u, "n": i, "signature": sig}) + "\n" for i, (u, sig) in enumerate(attempts))


with tempfile.TemporaryDirectory() as tmp:
    arm = os.path.join(tmp, "X")
    clean = noise(400)
    write(os.path.join(arm, "r-clean", "raw.txt"), raw(("stop", clean), ("stop", noise(500))))
    write(os.path.join(arm, "r-clean", "journal", "events.jsonl"), ev(("u", "e1"), ("u", "e2")))
    write(os.path.join(arm, "r-loop", "raw.txt"), raw(("stop", clean), ("stop", prefix + span * 6), ("stop", clean)))   # R only; third reply repeats the first
    write(os.path.join(arm, "r-limit", "raw.txt"), raw(("length", noise(900)), ("length", prefix + span * 5)))            # L both, R on the second
    write(os.path.join(arm, "r-limit", "journal", "events.jsonl"), ev(("u", "same"), ("u", "same"), ("v", "same"), ("u", "same")))
    write(os.path.join(arm, "r-usage", "usage.jsonl"), '{"stop":"length"}\n{"stop":"stop"}\nnot json\n[1]\n')            # no raw.txt: stop reasons only
    write(os.path.join(arm, "r-none", "note.txt"), "no reply log")
    write(os.path.join(arm, "stray.txt"), "a file, not a run")
    a = L.analyse_arm(arm)
    expect("runs", a["runs"], 4)
    expect("unreadable", a["unreadable"], 1)
    expect("replies", a["replies"], 2 + 3 + 2 + 2)
    expect("limit", a["limit"], 3)
    expect("repeat", a["repeat"], 2)
    expect("either", a["either"], 4)
    expect("no_text", a["no_text"], 2)
    expect("malformed (two bad usage lines)", a["malformed"], 2)
    expect("runs with L or R", a["runs_with_loop_or_limit"], 3)
    expect("repeated reply (r-loop third = first? no: second between)", a["repeated_reply"], 0)
    expect("repeated signature", a["repeated_signature"], 2)
    expect("small", a["small"], True)
    expect("per-run r-limit", (a["per_run"]["r-limit"]["limit"], a["per_run"]["r-limit"]["repeat"]), (2, 1))

    # a verbatim repeated action: two identical consecutive replies
    write(os.path.join(arm, "r-dup", "raw.txt"), raw(("stop", clean), ("stop", clean), ("stop", clean)))
    expect("repeated reply", L.analyse_arm(arm)["per_run"]["r-dup"]["repeated_reply"], 2)

    # malformed: garbage bytes, text before the header, a missing directory, an arm that is a file
    bad = os.path.join(tmp, "bad")
    os.makedirs(os.path.join(bad, "g"))
    with open(os.path.join(bad, "g", "raw.txt"), "wb") as f:
        f.write(bytes(range(256)) * 20)
    write(os.path.join(bad, "h", "raw.txt"), "junk first\n" + raw(("stop", clean)))
    write(os.path.join(bad, "i", "raw.txt"), "")
    write(os.path.join(bad, "j", "raw.txt"), raw(("stop", clean)))
    write(os.path.join(bad, "j", "journal", "events.jsonl"), "{bad\n[]\n{\"kind\":\"attempt\"}\n\n")
    b = L.analyse_arm(bad)
    expect("garbage run counted, no raise", (b["runs"], b["unreadable"]), (4, 0))
    expect("junk before header is malformed", b["per_run"]["h"]["malformed"], 1)
    expect("garbage with no header is malformed", b["per_run"]["g"]["malformed"], 1)
    expect("missing dir", "error" in L.analyse_arm(os.path.join(tmp, "nope")), True)
    expect("arm that is a file", "error" in L.analyse_arm(os.path.join(arm, "stray.txt")), True)

    # decision rule
    def fake(replies, either):
        return {"replies": replies, "either": either, "share_either": either / replies, "small": replies < L.SMALL}
    expect("over 10%", L.decide({"A": fake(1000, 101), "B": fake(1000, 0)}).startswith("over 10%"), True)
    expect("exactly 10% is between", L.decide({"A": fake(1000, 100)}).startswith("between"), True)
    expect("under 2% everywhere", L.decide({"A": fake(1000, 19), "B": fake(500, 0)}).startswith("under 2%"), True)
    expect("2% exactly is between", L.decide({"A": fake(1000, 20)}).startswith("between"), True)
    expect("small arms are not judged", L.decide({"A": fake(50, 50)}).startswith("not judged"), True)

    # the command line
    out = os.path.join(tmp, "o.json")
    expect("main ok", L.main(["X=" + arm, "--json", out]), 0)
    expect("json written", json.load(open(out))["arms"]["X"]["replies"], a["replies"] + 3)
    expect("main usage error", L.main([]), 2)
    expect("main bad arg", L.main(["nodash"]), 2)
    expect("main --json without path", L.main(["X=" + arm, "--json"]), 2)

    # determinism
    expect("deterministic", json.dumps(L.analyse_arm(arm), sort_keys=True), json.dumps(L.analyse_arm(arm), sort_keys=True))

print("m8_loops: %s" % ("ok" if not fails else "%d failed" % fails))
sys.exit(1 if fails else 0)
