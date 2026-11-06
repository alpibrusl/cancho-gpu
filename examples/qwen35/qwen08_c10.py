#!/usr/bin/env python3
r"""docs/weight-space.md section 14, ticket T11, check C10 (the behavioural half): the edit applied as a patch inside the engine
has the effect of the edited checkpoint.

    python3 examples/qwen35/qwen08_c10.py [--runner build/cancho-gpu-run08] [--kdir tests/chx] [--models-dir ~/models]
                                          [--mode f32 f16] [--patch FILE] [--work build/c10]

Needs a Mac with Metal (`scripts/build-run08.sh`), `numpy`, `pyarrow` (the GSM8K parquet, for the W1 prompts) and both checkpoints; not in
the gate (`scripts/qwen08-patch-check.sh` is the front door). The patch is made from the two checkpoints (`examples/weightspace/w1_patch.py`)
unless `--patch` names one. For each weight path, one runner process each loads (a) the base, (b) the base with `--patch`, (c) the edited
checkpoint, and runs the same 28 jobs: the 8 check prompts of the goldens (ids) for 16 new tokens and W1 prompts 0 to 19 (rendered and
tokenized by the runner) for 64 new tokens, greedily.

* identity: every line of (b)'s output (the 20 largest logits after the prompt as f32 bits, every step's chosen id and 5 largest logits as bits)
  equals (c)'s. Nothing is compared with a tolerance.
* the edit shows: the jobs on which (a) and (c) differ are counted (a check that could not tell the two checkpoints apart would prove nothing).
* consistency with Hugging Face (`golden/hf08/`, float64): for each check prompt, the change of a logit from the base to the edited checkpoint,
  over the token ids that are in the top 20 of all four lists (ours and the goldens', base and edited), engine's (c) - (a) against the golden's
  edited - base. Limit 1e-3 on the difference of the two changes (14.6, C10).
* the 20 W1 continuations of (b) against the golden's edited continuations (C7a's comparison, now for the patched base).

Exit 0 when every check passes."""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "examples", "qwen35"))
sys.path.insert(0, os.path.join(ROOT, "examples", "weightspace"))
import qwen08_c567 as C  # noqa: E402
import w1_diff as D  # noqa: E402
import w1_patch as W  # noqa: E402

GOLD = os.path.join(ROOT, "golden", "hf08")
TOL = 1e-3
failures = []


def say(ok, text):
    print(("ok   " if ok else "FAIL ") + text, flush=True)
    if not ok:
        failures.append(text)


def run(args, model_dir, mode, lines, name, patch=None):
    os.makedirs(args.work, exist_ok=True)
    path = os.path.join(args.work, name + ".batch")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    cmd = [args.runner, model_dir, args.kdir, mode, str(args.tmax), "-", "--batch", path, "--timing"] + (["--patch", patch] if patch else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"runner failed ({' '.join(cmd)}): exit {p.returncode}\n{p.stderr[-2000:]}")
    with open(os.path.join(args.work, name + ".out"), "w") as f:
        f.write(p.stdout)
    return p.stdout, C.parse(p.stdout)


def job_lines(out):
    """The runner's lines grouped per job, without the timing lines (wall clock)."""
    jobs, cur = [], None
    for line in out.split("\n"):
        if line.startswith("job "):
            cur = []
            jobs.append(cur)
        if cur is not None and not line.startswith("timing:") and line:
            cur.append(line)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default=os.path.join(ROOT, "build/cancho-gpu-run08"))
    ap.add_argument("--kdir", default=os.path.join(ROOT, "tests/chx"))
    ap.add_argument("--models-dir", default=os.path.expanduser("~/models"))
    ap.add_argument("--gsm8k", default="~/models/gsm8k/main/test-00000-of-00001.parquet")
    ap.add_argument("--mode", nargs="*", default=["f32", "f16"])
    ap.add_argument("--patch")
    ap.add_argument("--tmax", type=int, default=512)
    ap.add_argument("--work", default=os.path.join(ROOT, "build/c10"))
    args = ap.parse_args()
    args.work = os.path.abspath(args.work)
    base = os.path.join(args.models_dir, "qwen35-08b-base")
    edited = os.path.join(args.models_dir, "qwen35-08b-edited")
    patch = args.patch
    if not patch:
        os.makedirs(args.work, exist_ok=True)
        patch = os.path.join(args.work, "w1.patch")
        b, e = D.read_checkpoint(base), D.read_checkpoint(edited)
        blob = W.encode(W.make(b, e))
        open(patch, "wb").write(blob)
        rows, ok = W.check(b, blob, e)
        say(ok, f"the patch ({len(blob):,} bytes) applied by w1_patch.py to the base gives the edited bytes in all {len(rows)} tensors")
    prompts = json.load(open(os.path.join(GOLD, "prompts.json")))
    gb = json.load(open(os.path.join(GOLD, "base.json")))
    ge = json.load(open(os.path.join(GOLD, "edited.json")))
    checks = prompts["check"]
    reqs = C.write_requests(args, 20)
    lines = [f"16 {','.join(map(str, c['ids']))}" for c in checks] + [f"64 @{r}" for r in reqs]
    for mode in args.mode:
        ob, jb = run(args, base, mode, lines, f"base_{mode}")
        op, jp = run(args, base, mode, lines, f"base_patch_{mode}", patch)
        oe, je = run(args, edited, mode, lines, f"edited_{mode}")
        lb, lp, le = job_lines(ob), job_lines(op), job_lines(oe)
        same = [i for i in range(len(lines)) if lp[i] == le[i]]
        say(len(same) == len(lines) and len(lp) == len(le), f"C10 {mode}: base + patch and the edited checkpoint print identical lines (top 20 logits as bits, every step's token and top-5 bits) on {len(same)} of {len(lines)} jobs")
        differ = [i for i in range(len(lines)) if lb[i] != le[i]]
        say(len(differ) >= 8, f"C10 {mode}: the base and the edited checkpoint differ on {len(differ)} of {len(lines)} jobs (the check can tell them apart)")
        # Consistency with HF on the 8 check prompts: the change of a logit, ours against the golden's.
        worst, nmin, mx_eff, mx_hf = 0.0, 99, 0.0, 0.0
        for k in range(len(checks)):
            gl_b, gl_e = gb["check"]["logits"][k], ge["check"]["logits"][k]
            hb = dict(zip(gl_b["top20_ids"], gl_b["top20_vals"]))
            he = dict(zip(gl_e["top20_ids"], gl_e["top20_vals"]))
            ob_t = dict(jb[k]["top"])
            op_t = dict(jp[k]["top"])
            common = sorted(set(hb) & set(he) & set(ob_t) & set(op_t))
            nmin = min(nmin, len(common))
            for i in common:
                ours = op_t[i] - ob_t[i]
                theirs = he[i] - hb[i]
                worst = max(worst, abs(ours - theirs))
                mx_eff = max(mx_eff, abs(ours))
                mx_hf = max(mx_hf, abs(theirs))
        say(worst <= TOL and nmin >= 5, f"C10 {mode}: the change of a logit from the base to the edited checkpoint, ours (base+patch minus base) against HF's float64 (edited minus base), "
                                       f"on the token ids in all four top-20 lists ({nmin} or more per check prompt): worst difference of the two changes {worst:.2e} (limit {TOL:g}); "
                                       f"largest change ours {mx_eff:.3f}, HF {mx_hf:.3f}")
        # W1 continuations of the patched base against the golden's edited continuations.
        conts = ge["cont64"][:20]
        n_eq, n_base_eq = 0, 0
        for i in range(20):
            got = [c for c, _ in jp[8 + i]["steps"]]
            want = conts[i]["ids"]
            n_eq += got == want[:len(got)] and (len(got) == len(want) or jp[8 + i]["eos"] >= 0)
            gb_ids = [c for c, _ in jb[8 + i]["steps"]]
            n_base_eq += gb_ids == want[:len(gb_ids)] and (len(gb_ids) == len(want) or jb[8 + i]["eos"] >= 0)
        say(n_eq >= 18, f"C10 {mode}: free-running 64 tokens of W1 prompts 0 to 19, base + patch equals HF's edited continuation on {n_eq} of 20 (the unpatched base: {n_base_eq} of 20)")
    print("FAILED: " + str(len(failures)) if failures else "all C10 checks pass")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
