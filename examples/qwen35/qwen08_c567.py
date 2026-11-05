#!/usr/bin/env python3
r"""docs/weight-space.md section 14, ticket T7, checks C5, C6 and C7: the Qwen3.5-0.8B engine's logits and greedy tokens against the
float64 Hugging Face goldens of T1 (`golden/hf08/`).

    python3 examples/qwen35/qwen08_c567.py [--runner build/cancho-gpu-run08] [--kdir tests/chx] [--models-dir ~/models]
                                           [--model base|edited ...] [--mode f32|f16 ...] [--checks c5,c6,c7] [--work build/c567]

Needs a Mac with Metal (`scripts/build-run08.sh`), `numpy`, `pyarrow` (the GSM8K parquet, for the W1 prompts) and both checkpoints; not in the
gate (`scripts/qwen08-check.sh` is the front door). One runner process per (checkpoint, weight path, check): the engine is loaded once and
runs every job of a batch file (`device/run08-main.cho`).

* C5: the final logits at the last position of the 8 check prompts (3 W1 prompts, 5 short; the ids are the goldens') against the float64
  golden's top 20: top-5 ids equal in order (a swap whose golden gap is under 1e-3 is reported, not failed), each of the golden's top-5
  logits within 1e-3 absolute.
* C6: the one-token greedy argmax of the 200 W1 prompts against the golden's. The W1 prompts go through the runner's own renderer
  (`src/chat_w1.cho`) and tokenizer, and their ids are held to the golden's sha256 of the ids. A difference whose golden top-2 gap is under
  2e-3 is listed; any other is a failure. Where the golden has the whole top-5 of the first token (prompts 0 to 19, from the continuations),
  the top-5 ids and values are compared too. Note: the golden's first token is id 8160 on all 200 prompts, so the argmax check is weak.
* C7: 64-token greedy decoding of W1 prompts 0 to 19: (a) free-running, the runner's tokens against the golden continuation; a first
  divergence is reported with the golden's top-2 gap there (under 1e-3 is a legitimate near-tie flip, otherwise a failure); (b) teacher
  forced, the runner fed the golden's tokens, its top-5 ids and values at each of the 1,280 steps against the golden's top-5: values within
  1e-3, ids equal (a swap between neighbours whose golden gap is under 1e-3 is reported, not failed).

Exit 0 when every requested check passes.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "examples", "weightspace"))
GOLD = os.path.join(ROOT, "golden", "hf08")
TOL = 1e-3
TIE = 1e-3
TIE6 = 2e-3
failures = []


def say(ok, text):
    print(("ok   " if ok else "FAIL ") + text, flush=True)
    if not ok:
        failures.append(text)


def vals(bits):
    return np.array(bits, dtype=np.uint32).view(np.float32).astype(np.float64)


def parse(out):
    """The jobs of a runner's stdout: {k: {prompt, ids, top: [(id, v)], steps: [(chosen, [(id, v)])], new, hit, eos, timing}}."""
    jobs, cur = {}, None
    for line in out.split("\n"):
        w = line.split(" ")
        if w[0] == "job":
            cur = jobs.setdefault(int(w[1]), {"steps": []})
            cur["n"] = int(w[2].split("=")[1])
        elif w[0] == "ids":
            cur["ids"] = [int(x) for x in w[2].split(",")] if len(w) > 2 else []
        elif w[0] == "top":
            pr = [x.split(":") for x in w[2:]]
            cur["top"] = [(int(a), float(vals([int(b)])[0])) for a, b in pr]
        elif w[0] == "step":
            pr = [x.split(":") for x in w[3:]]
            cur["steps"].append((int(w[2]), [(int(a), float(vals([int(b)])[0])) for a, b in pr]))
        elif w[0] == "done":
            cur["new"] = int(w[2].split("=")[1])
            cur["hit"] = int(w[3].split("=")[1])
            cur["eos"] = int(w[4].split("=")[1])
        elif w[0] == "timing:":
            cur["timing"] = {x.split("=")[0]: int(x.split("=")[1]) for x in w[1:]}
    return jobs


def run_batch(args, model_dir, mode, lines, name):
    os.makedirs(args.work, exist_ok=True)
    name = name.replace("/", "_")
    path = os.path.join(args.work, name + ".batch")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    cmd = [args.runner, model_dir, args.kdir, mode, str(args.tmax), "-", "--batch", path, "--timing"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"runner failed ({' '.join(cmd)}): exit {p.returncode}\n{p.stderr[-2000:]}")
    with open(os.path.join(args.work, name + ".out"), "w") as f:
        f.write(p.stdout)
    jobs = parse(p.stdout)
    if len(jobs) != len(lines):
        sys.exit(f"{name}: {len(jobs)} jobs back for {len(lines)} lines")
    return jobs


def write_requests(args, n):
    import pyarrow.parquet as pq
    import w1_behave
    gsm = os.path.expanduser(args.gsm8k)
    qs = pq.read_table(gsm).column("question").to_pylist()[:n]
    d = os.path.join(args.work, "req")
    os.makedirs(d, exist_ok=True)
    paths = []
    for i, q in enumerate(qs):
        p = os.path.join(d, f"{i:03d}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"messages": [{"role": "user", "content": w1_behave.build_question(q)}]}, f, ensure_ascii=False)
        paths.append(p)
    return paths


def ids_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def top5_match(top, gi, gv):
    """Our top-5 [(id, value)] against the golden's ids and float64 values: (kind, worst |diff| over the ranks whose ids agree).
    kind: "equal"; "tie" when only the fifth place differs (our fifth is not among the golden's five) and its value is within TIE of the
    golden's fifth, so the golden's sixth is within TIE (+ the logits' own error) of its fifth: a flip with the sixth, reported, not failed;
    "swap" when the same five ids come in another order across a golden gap under TIE; "bad" otherwise."""
    ids = [x for x, _ in top]
    vv = [v for _, v in top]
    d = max((abs(a - b) for a, b, x, y in zip(vv, gv, ids, gi) if x == y), default=0.0)
    if ids == gi:
        return ("equal" if d <= TOL else "bad"), d
    if ids[:4] == gi[:4] and abs(vv[4] - gv[4]) < TIE and d <= TOL:
        return "tie", d
    if sorted(ids) == sorted(gi) and min(abs(gv[a] - gv[a + 1]) for a in range(4)) < TIE:
        return "swap", d
    return "bad", d


def c5(args, gold, prompts, model_dir, mode, tag):
    checks = prompts["check"]
    jobs = run_batch(args, model_dir, mode, [f"0 {','.join(map(str, c['ids']))}" for c in checks], f"{tag}_c5")
    worst, swaps, bad = 0.0, [], []
    for k, c in enumerate(checks):
        g = gold["check"]["logits"][k]
        kind, d = top5_match(jobs[k]["top"][:5], g["top20_ids"][:5], g["top20_vals"][:5])
        worst = max(worst, d)
        if kind in ("tie", "swap"):
            swaps.append((k, kind, [x for x, _ in jobs[k]["top"][:5]], g["top20_ids"][:5]))
        elif kind == "bad":
            bad.append((k, [x for x, _ in jobs[k]["top"][:5]], g["top20_ids"][:5], d))
    say(not bad, f"C5 {tag}: 8 check prompts, top-5 ids {'equal in order' if not swaps else 'equal in order except ' + str(swaps)}, worst top-5 logit |diff| {worst:.2e} (limit {TOL:g}); failures: {bad}")


def c6(args, gold, prompts, model_dir, mode, tag, reqs):
    n = len(reqs)
    jobs = run_batch(args, model_dir, mode, [f"0 @{p}" for p in reqs], f"{tag}_c6")
    ref = gold["argmax200"]["f64"]
    gaps = gold["argmax200"]["gap_f64"]
    wrong, tied, hashbad = [], [], []
    for i in range(n):
        if ids_sha(jobs[i]["ids"]) != prompts["w1"][i]["sha256_ids"]:
            hashbad.append(i)
        a = jobs[i]["top"][0][0]
        if a != ref[i]:
            (tied if gaps[i] < TIE6 else wrong).append((i, a, ref[i], gaps[i]))
    first = set(j["top"][0][0] for j in jobs.values())
    say(not hashbad, f"C6 {tag}: the runner's rendered+tokenized ids equal the golden sha256 of the ids for {n - len(hashbad)} of {n} prompts{'' if not hashbad else ': ' + str(hashbad[:10])}")
    say(not wrong, f"C6 {tag}: argmax equal to the golden on {n - len(wrong) - len(tied)} of {n} prompts (first token ids seen: {sorted(first)}); "
                   f"near-tie differences (golden gap < {TIE6:g}): {tied}; unexplained: {wrong}")
    # the whole top-5 where the golden has it
    cmp5, bad5, worst, ties = 0, [], 0.0, []
    for i in range(min(n, 20)):
        g = gold["cont64"][i]
        kind, d = top5_match(jobs[i]["top"][:5], g["top5_ids"][0], g["top5_vals"][0])
        worst = max(worst, d)
        cmp5 += 1
        if kind == "bad":
            bad5.append((i, [x for x, _ in jobs[i]["top"][:5]], g["top5_ids"][0], d))
        elif kind != "equal":
            ties.append((i, kind))
    for i in range(3):
        g = gold["check"]["logits"][i]
        kind, d = top5_match(jobs[i]["top"][:5], g["top20_ids"][:5], g["top20_vals"][:5])
        if kind == "bad":
            bad5.append((i, "check", [x for x, _ in jobs[i]["top"][:5]], g["top20_ids"][:5]))
    say(not bad5, f"C6 {tag}: full top-5 equal to the golden's (ids in order, values within {TOL:g}) on {cmp5 - len(bad5)} of {cmp5} prompts with a golden top-5 (worst |diff| {worst:.2e}); near-tie flips {ties}; failures: {bad5}")
    return jobs


def c7(args, gold, prompts, model_dir, mode, tag, reqs):
    n = 20
    conts = gold["cont64"][:n]
    free = run_batch(args, model_dir, mode, [f"64 @{reqs[i]}" for i in range(n)], f"{tag}_c7free")
    forced = run_batch(args, model_dir, mode, [f"64 @{reqs[i]} {','.join(map(str, conts[i]['ids']))}" for i in range(n)], f"{tag}_c7forced")
    # (a) free running
    div = []
    for i in range(n):
        want = conts[i]["ids"]
        got = [c for c, _ in free[i]["steps"]]
        if got != want[:len(got)] or (len(got) < len(want) and free[i]["eos"] < 0):
            j = next((j for j in range(min(len(got), len(want))) if got[j] != want[j]), min(len(got), len(want)))
            v = conts[i]["top5_vals"][j]
            div.append((i, j, got[j] if j < len(got) else None, want[j] if j < len(want) else None, v[0] - v[1]))
    legit = [d for d in div if d[4] < TIE]
    real = [d for d in div if d[4] >= TIE]
    say(not real, f"C7a {tag}: free-running 64 tokens of W1 prompts 0 to 19: {n - len(div)} of {n} continuations equal to the golden's; "
                  f"divergences (prompt, position, ours, golden, golden top-2 gap): near-tie (< {TIE:g}) {legit}; unexplained {real}")
    # (b) teacher forced
    steps, id_swaps, id_bad, worst = 0, [], [], 0.0
    for i in range(n):
        sts = forced[i]["steps"]
        if len(sts) != 64:
            id_bad.append((i, "steps", len(sts)))
            continue
        for j, (_, top5) in enumerate(sts):
            gi, gv = conts[i]["top5_ids"][j], conts[i]["top5_vals"][j]
            kind, d = top5_match(top5, gi, gv)
            steps += 1
            worst = max(worst, d)
            if kind in ("tie", "swap"):
                id_swaps.append((i, j, kind, [x for x, _ in top5], gi))
            elif kind == "bad":
                id_bad.append((i, j, [x for x, _ in top5], gi, d))
    say(not id_bad, f"C7b {tag}: teacher forced, {steps} of {n * 64} steps compared; top-5 ids and values (within {TOL:g}) equal except {len(id_swaps)} reported near-tie flips {id_swaps}; "
                    f"worst |diff| of a top-5 logit {worst:.2e}; failures {id_bad}")
    ms = [j["timing"]["gen_ms"] / j["timing"]["gen_new"] for j in free.values() if j["timing"]["gen_new"]]
    tot = sum(j["timing"]["gen_ms"] for j in free.values())
    cnt = sum(j["timing"]["gen_new"] for j in free.values())
    say(True, f"C7 {tag}: ms per generated token, end to end (feed, logits readback, host scan for the argmax): {tot / cnt:.2f} mean over {cnt} tokens "
              f"(per prompt {min(ms):.2f} to {max(ms):.2f}); prompt {sum(j['timing']['prompt_ms'] for j in free.values()) / sum(j['n'] for j in free.values()):.2f} ms per prompt token")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default=os.path.join(ROOT, "build/cancho-gpu-run08"))
    ap.add_argument("--kdir", default=os.path.join(ROOT, "tests/chx"))
    ap.add_argument("--models-dir", default=os.path.expanduser("~/models"))
    ap.add_argument("--gsm8k", default="~/models/gsm8k/main/test-00000-of-00001.parquet")
    ap.add_argument("--model", nargs="*", default=["base", "edited"])
    ap.add_argument("--mode", nargs="*", default=["f32", "f16"])
    ap.add_argument("--checks", default="c5,c6,c7")
    ap.add_argument("--tmax", type=int, default=512)
    ap.add_argument("--work", default=os.path.join(ROOT, "build/c567"))
    args = ap.parse_args()
    args.work = os.path.abspath(args.work)
    prompts = json.load(open(os.path.join(GOLD, "prompts.json")))
    want = set(args.checks.split(","))
    reqs = write_requests(args, 200) if want & {"c6", "c7"} else []
    for model in args.model:
        gold = json.load(open(os.path.join(GOLD, model + ".json")))
        model_dir = os.path.join(args.models_dir, "qwen35-08b-" + model)
        for mode in args.mode:
            tag = f"{model}/{mode}"
            if "c5" in want:
                c5(args, gold, prompts, model_dir, mode, tag)
            if "c6" in want:
                c6(args, gold, prompts, model_dir, mode, tag, reqs)
            if "c7" in want:
                c7(args, gold, prompts, model_dir, mode, tag, reqs)
    print("FAILED: " + str(len(failures)) if failures else "all requested checks pass")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
