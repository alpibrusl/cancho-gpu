#!/usr/bin/env python3
"""W0 (docs/weight-space.md section 5.1): does the M8 model loop? A deterministic repeat detector over the arms' run directories.

    python3 examples/qwen/m8_loops.py A=/path/to/m8-runs/A B=/path/to/m8-runs/B [--json out.json]

An arm directory holds run directories (`A-base64-1/`, ...) each with `raw.txt` (every reply, headed `=== reply (stopped because: X)`), `usage.jsonl` and `journal/events.jsonl`.
Indicators per reply: L (stop reason `length`) and R (the reply ends in a span of >= 64 characters repeated more than 3 times). The definition is pre-registered in the document; this file
only implements it. Standard library only; reads files, runs nothing, never raises on a malformed run (it counts it)."""
import json
import math
import os
import re
import sys

MIN_SPAN = 64          # shortest repeated span, characters
MAX_SPAN = 4096        # longest period looked for
COPIES = 4             # "more than 3 times"
WINDOW = MAX_SPAN * COPIES   # only the last 16,384 characters are looked at
SMALL = 100            # an arm with fewer replies is not judged
HEAD = re.compile(r"^=== reply \(stopped because: ([^)\n]*)\)[ \t]*$", re.M)


def tail_repeat(text):
    """True iff, after trailing whitespace is removed, the last COPIES*p characters are p-periodic for some MIN_SPAN <= p <= MAX_SPAN (the last copy may be cut short)."""
    if not isinstance(text, str):
        return False
    s = text.rstrip()[-WINDOW:]
    n = len(s)
    last = s[-1:]
    for p in range(MIN_SPAN, min(MAX_SPAN, n // COPIES) + 1):
        if s[n - 1 - p] != last:
            continue
        if s[n - (COPIES - 1) * p:] == s[n - COPIES * p:n - p]:
            return True
    return False


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except (OSError, ValueError):
        return None


def replies_from_raw(text):
    """[(stop, reply_text)] from raw.txt, plus the count of malformed sections (text before the first header)."""
    marks = list(HEAD.finditer(text))
    bad = 1 if (marks and text[:marks[0].start()].strip()) or (not marks and text.strip()) else 0
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group(1).strip(), text[m.end() + 1:end]))
    return out, bad


def replies_from_usage(text):
    out, bad = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(d, dict) and isinstance(d.get("stop"), str):
            out.append((d["stop"], None))
        else:
            bad += 1
    return out, bad


def repeated_signatures(text):
    """Consecutive attempts of one unit with the same non-empty refusal signature."""
    last, n = {}, 0
    for line in text.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if not isinstance(d, dict) or d.get("kind") != "attempt":
            continue
        unit, sig = d.get("unit"), d.get("signature")
        if isinstance(sig, str) and sig and last.get(unit) == sig:
            n += 1
        last[unit] = sig if isinstance(sig, str) else None
    return n


def analyse_run(run_dir):
    """One run's counts, or None if the directory has no reply log at all."""
    raw = read_text(os.path.join(run_dir, "raw.txt"))
    usage = read_text(os.path.join(run_dir, "usage.jsonl"))
    malformed = 0
    if raw is not None:
        replies, malformed = replies_from_raw(raw)
        with_text = True
    elif usage is not None:
        replies, malformed = replies_from_usage(usage)
        with_text = False
    else:
        return None
    r = {"replies": len(replies), "limit": 0, "repeat": 0, "either": 0, "no_text": 0, "repeated_reply": 0, "malformed": malformed, "repeated_signature": 0}
    prev = None
    for stop, body in replies:
        lim = stop == "length"
        rep = tail_repeat(body) if with_text else False
        r["limit"] += lim
        r["repeat"] += rep
        r["either"] += lim or rep
        if not with_text:
            r["no_text"] += 1
        elif prev is not None and body.strip() and body == prev:
            r["repeated_reply"] += 1
        prev = body
    ev = read_text(os.path.join(run_dir, "journal", "events.jsonl"))
    if ev is not None:
        r["repeated_signature"] = repeated_signatures(ev)
    return r


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def analyse_arm(arm_dir):
    runs, unreadable = {}, 0
    try:
        names = sorted(os.listdir(arm_dir))
    except OSError:
        return {"error": "cannot read %s" % arm_dir, "runs": 0, "unreadable": 0, "replies": 0}
    for name in names:
        d = os.path.join(arm_dir, name)
        if not os.path.isdir(d):
            continue
        r = analyse_run(d)
        if r is None:
            unreadable += 1
        else:
            runs[name] = r
    tot = {k: sum(r[k] for r in runs.values()) for k in ("replies", "limit", "repeat", "either", "no_text", "repeated_reply", "malformed", "repeated_signature")}
    lo, hi = wilson(tot["either"], tot["replies"])
    tot.update({
        "runs": len(runs), "unreadable": unreadable,
        "runs_with_loop_or_limit": sum(1 for r in runs.values() if r["either"]),
        "share_either": tot["either"] / tot["replies"] if tot["replies"] else 0.0,
        "share_either_ci95": [lo, hi],
        "small": tot["replies"] < SMALL,
        "per_run": runs,
    })
    return tot


def decide(arms):
    judged = [a for a in arms.values() if not a.get("small") and a["replies"] and "error" not in a]
    if not judged:
        return "not judged: no arm has %d replies" % SMALL
    if any(a["share_either"] > 0.10 for a in judged):
        return "over 10%: a looping fault is plausibly present (W1 relevant to M8)"
    if all(a["share_either"] < 0.02 for a in judged):
        return "under 2% in every judged arm: not present (W1 stays a standalone check)"
    return "between 2% and 10%: inconclusive"


def report(arms):
    lines = ["| arm | runs | replies | L (limit) | R (tail repeat) | L or R | share (95% Wilson) | runs with L or R | repeated reply | repeated signature | unreadable runs | malformed |", "|" + "---|" * 12]
    for name, a in arms.items():
        if "error" in a:
            lines.append("| %s | %s |" % (name, a["error"]))
            continue
        lines.append("| %s | %d | %d | %d | %d | %d | %.3f (%.3f to %.3f)%s | %d | %d | %d | %d | %d |" % (
            name, a["runs"], a["replies"], a["limit"], a["repeat"], a["either"], a["share_either"], a["share_either_ci95"][0], a["share_either_ci95"][1],
            " small" if a["small"] else "", a["runs_with_loop_or_limit"], a["repeated_reply"], a["repeated_signature"], a["unreadable"], a["malformed"]))
    return "\n".join(lines)


def main(argv):
    out_json = None
    args = list(argv)
    if "--json" in args:
        i = args.index("--json")
        if i + 1 >= len(args):
            print("--json needs a path", file=sys.stderr)
            return 2
        out_json = args[i + 1]
        del args[i:i + 2]
    arms = {}
    for a in args:
        name, sep, path = a.partition("=")
        if not sep or not name or not path:
            print("usage: m8_loops.py ARM=DIR [ARM=DIR ...] [--json out.json]", file=sys.stderr)
            return 2
        arms[name] = analyse_arm(path)
    if not arms:
        print("usage: m8_loops.py ARM=DIR [ARM=DIR ...] [--json out.json]", file=sys.stderr)
        return 2
    print(report(arms))
    print("\ndecision (section 5): " + decide(arms))
    if out_json:
        with open(out_json, "w") as f:
            json.dump({"arms": arms, "decision": decide(arms)}, f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
