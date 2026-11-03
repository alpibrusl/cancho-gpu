#!/usr/bin/env python3
"""The M8 training mix and the held-out sets (docs/roadmap-models.md section 7), from the files m8_corpus.py wrote.

    python3 examples/qwen/m8_mix.py DATA_DIR [--seed 20261103]

Writes DATA_DIR/train.jsonl (shuffled), heldout_repos.jsonl, heldout_files.jsonl and MIX.json (counts, tokens with Qwen's tokenizer, sha256). Every pool is used at most once (no repetition): the
total is the largest the binding pool allows at the stated weights. The weights make cancho code dominate: cho 50% (raw .cho/.chx), instr 12%, fix 18% (cancho code in all, 80%), md 15%, rs 5%.
A sequence is `prompt + reply` for a pair (the loss is on the reply, `loss_from` marks where it starts) or the whole chunk for a document, at most 1,024 tokens."""
import argparse
import hashlib
import json
import os
import random

WEIGHTS = {"cho": 0.50, "instr": 0.12, "fix": 0.18, "md": 0.15, "rs": 0.05}
FILES = {"cho": "raw_cho", "instr": "pairs_instr", "fix": "pairs_fix", "md": "raw_md", "rs": "raw_rs"}


def load(d, name):
    return [json.loads(l) for l in open(os.path.join(d, name + ".jsonl"))]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--seed", type=int, default=20261103)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pools = {k: load(a.data, f) for k, f in FILES.items()}
    train = {k: [r for r in v if r["split"] == "train"] for k, v in pools.items()}
    avail = {k: sum(r["ntok"] for r in v) for k, v in train.items()}
    total = min(avail[k] / w for k, w in WEIGHTS.items())
    picked, report = [], {"weights": WEIGHTS, "available_tokens": avail, "total_tokens_target": int(total), "pools": {}}
    for k, w in WEIGHTS.items():
        rows = list(train[k])
        rng.shuffle(rows)
        want, got, take = w * total, 0, []
        for r in rows:
            if got >= want:
                break
            take.append(r)
            got += r["ntok"]
        picked += take
        report["pools"][k] = {"records": len(take), "tokens": got, "share": 0.0}
    tot = sum(p["tokens"] for p in report["pools"].values())
    for p in report["pools"].values():
        p["share"] = round(p["tokens"] / tot, 4)
    report["train_tokens"], report["train_records"] = tot, len(picked)
    rng.shuffle(picked)
    out = {"train": picked, "heldout_repos": [r for v in pools.values() for r in v if r["split"] == "heldout_repos"],
           "heldout_files": [r for v in pools.values() for r in v if r["split"] == "heldout_files"]}
    report["files"] = {}
    for name, rows in out.items():
        rows.sort(key=lambda r: r["id"]) if name != "train" else None
        path = os.path.join(a.data, name + ".jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        report["files"][name + ".jsonl"] = {"records": len(rows), "tokens": sum(r["ntok"] for r in rows), "sha256": sha(path),
                                            "by_kind": {k: sum(r["ntok"] for r in rows if r["kind"] == k) for k in ("cho", "md", "rs", "instr", "fix")}}
    json.dump(report, open(os.path.join(a.data, "MIX.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
