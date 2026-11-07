#!/usr/bin/env python3
"""docs/weight-space.md section 14.15, ticket T12: the patched engine on the W1 sampler (throughput) and against PyTorch (agreement).

    python3 w1_t12.py throughput --patch a6.chopatch [--n 4] [--max-new 8192] --out OUT.json
    python3 w1_t12.py agree      --patch a6.chopatch --variant-dir CKPT/a6 --out OUT.json [--n 8] [--new 64]

Mac only (Metal, build/cancho-gpu-w1 and build/cancho-gpu-run08), needs the base checkpoint and the GSM8K parquet. The pure parts
(`projection`, `feasibility`, `first_mismatch`, `job_lines`) need nothing and are tested by tests/w1_t12.py. Refusals: `t12-input`,
`t12-engine`, `t12-result`; the CLI exits 2 on them."""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "examples", "qwen35"))

BASE_MEAN_TOKENS, EDITED_MEAN_TOKENS = 7829, 6620      # weight-space-w4.md 13.1
FEASIBLE_H, MARGINAL_H = 10.0, 24.0
GSM8K = os.path.expanduser("~/models/gsm8k/main/test-00000-of-00001.parquet")
BASE = os.path.expanduser("~/models/qwen35-08b-base")


class T12Error(Exception):
    def __init__(self, rule, msg):
        super().__init__(msg)
        self.rule = rule


def projection(mean_tokens, ms_per_token, problems=200):
    """Hours of one arm of `problems` generations of `mean_tokens` tokens at `ms_per_token`."""
    if not (mean_tokens > 0 and ms_per_token > 0 and problems > 0):
        raise T12Error("t12-input", "tokens, ms a token and problems must be positive")
    return problems * mean_tokens * ms_per_token / 3.6e6


def feasibility(hours):
    return "feasible" if hours <= FEASIBLE_H else ("marginal" if hours <= MARGINAL_H else "not feasible")


def job_lines(out):
    """The runner's lines grouped per job without the timing lines (wall clock)."""
    jobs, cur = [], None
    for line in out.split("\n"):
        if line.startswith("job "):
            cur = []
            jobs.append(cur)
        if cur is not None and not line.startswith("timing:") and line:
            cur.append(line)
    return jobs


def first_mismatch(a, b):
    """(job, step) of the first differing chosen id between two lists of id lists, or None."""
    if len(a) != len(b):
        return (min(len(a), len(b)), 0)
    for j, (x, y) in enumerate(zip(a, b)):
        for s in range(max(len(x), len(y))):
            if s >= len(x) or s >= len(y) or x[s] != y[s]:
                return (j, s)
    return None


def throughput(a):
    for path, what in ((a.patch, "patch"), (a.binary, "engine binary"), (BASE, "base checkpoint"), (GSM8K, "GSM8K parquet")):
        if not os.path.exists(path):
            raise T12Error("t12-input", f"{what} {path}: not found")
    import w1_behave as B
    import w1_cancho as C
    probs = B.load_problems(GSM8K, a.n)
    tmax = 8704
    cmd = C.command_for(a.binary, BASE, a.kernels, "f16", tmax) + ["--patch", os.path.abspath(a.patch)]
    stats = []
    gen = C.cancho_generator(cmd, stats)
    rows = []
    try:
        for p in probs:
            text, n, hit, = gen([B.build_question(p["question"])], a.max_new, p["id"])[0]
            s = stats[-1]
            rows.append({"id": p["id"], "prompt_tokens": s["prompt_tokens"], "new": n, "hit_limit": hit, "engine_ms": s["gen_ms"],
                         "ms_per_token": s["gen_ms"] / max(n, 1), "tokens_per_s": 1000.0 * n / max(s["gen_ms"], 1)})
            print(json.dumps(rows[-1]), flush=True)
    except Exception as e:  # noqa: BLE001 -- the engine's message is the evidence
        raise T12Error("t12-engine", str(e))
    finally:
        gen.close()
    capped = [r for r in rows if r["hit_limit"]] or rows
    c = sum(r["engine_ms"] for r in capped) / sum(r["new"] for r in capped)
    res = {"patch": os.path.basename(a.patch), "mode": "f16", "tmax": tmax, "max_new": a.max_new, "problems": rows,
           "tokens_per_s_all": 1000.0 * sum(r["new"] for r in rows) / sum(r["engine_ms"] for r in rows),
           "ms_per_token_capped": c, "n_capped": len([r for r in rows if r["hit_limit"]])}
    for name, t in (("base_like", BASE_MEAN_TOKENS), ("published_edit", EDITED_MEAN_TOKENS)):
        h = projection(t, c)
        res["projection_" + name] = {"mean_tokens": t, "hours_200": h, "verdict": feasibility(h)}
    return res


def agree(a):
    import numpy as np
    import qwen08_c567 as Q
    for path, what in ((a.patch, "patch"), (a.runner, "runner"), (a.variant_dir, "variant directory"), (BASE, "base checkpoint")):
        if not os.path.exists(path):
            raise T12Error("t12-input", f"{what} {path}: not found")
    ns = argparse.Namespace(gsm8k=GSM8K, work=os.path.abspath(a.work), runner=os.path.abspath(a.runner), kdir=os.path.abspath(a.kernels), tmax=512)
    reqs = Q.write_requests(ns, a.n)
    lines = [f"{a.new} @{r}" for r in reqs]

    def run(model_dir, name, patch=None):
        os.makedirs(ns.work, exist_ok=True)
        path = os.path.join(ns.work, name + ".batch")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        cmd = [ns.runner, model_dir, ns.kdir, "f32", str(ns.tmax), "-", "--batch", path, "--timing"] + (["--patch", patch] if patch else [])
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise T12Error("t12-engine", f"{' '.join(cmd)}: exit {p.returncode}: {p.stderr[-500:]}")
        with open(os.path.join(ns.work, name + ".out"), "w") as f:
            f.write(p.stdout)
        return p.stdout, Q.parse(p.stdout)

    o_patch, j_patch = run(BASE, "base_patch", os.path.abspath(a.patch))
    o_var, j_var = run(a.variant_dir, "variant")
    o_base, j_base = run(BASE, "base")
    lp, lv, lb = job_lines(o_patch), job_lines(o_var), job_lines(o_base)
    same = sum(1 for x, y in zip(lp, lv) if x == y)
    differ = sum(1 for x, y in zip(lb, lv) if x != y)
    ids_patch = [[c for c, _ in j_patch[k]["steps"]] for k in range(len(lines))]
    prompts = [j_patch[k]["ids"] for k in range(len(lines))]
    import torch
    import hf08_ref as H
    dev = a.ref_device
    model = H.build_model(a.variant_dir, "f32").to(dev).eval()   # the text tower, transformers' code as shipped, float32
    ref, gaps = [], []
    with torch.no_grad():
        for pr in prompts:
            ids = torch.tensor([pr], device=dev)
            o = model(input_ids=ids, use_cache=True)
            seq, gp = [], []
            for _ in range(a.new):
                lg = o.logits[0, -1].float().cpu()
                top = torch.topk(lg, 2)
                t = int(top.indices[0])
                seq.append(t)
                gp.append(float(top.values[0] - top.values[1]))
                if t in (248044, 248046):
                    break
                o = model(input_ids=torch.tensor([[t]], device=dev), past_key_values=o.past_key_values, use_cache=True)
            ref.append(seq)
            gaps.append(gp)
    mm = first_mismatch(ids_patch, ref)
    res = {"jobs": len(lines), "new": a.new, "identical_lines_patch_vs_variant": same, "base_differs_from_variant": differ,
           "patch_vs_pytorch_equal_jobs": sum(1 for x, y in zip(ids_patch, ref) if x == y), "first_mismatch": mm,
           "ref_device": dev, "tokens_compared": sum(min(len(x), len(y)) for x, y in zip(ids_patch, ref))}
    if mm is not None and mm[0] < len(gaps) and mm[1] < len(gaps[mm[0]]):
        res["gap_at_first_mismatch"] = gaps[mm[0]][mm[1]]
    res["passes"] = same == len(lines) and mm is None
    res["min_gap_over_run"] = float(np.min([g for gp in gaps for g in gp]))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("throughput", "agree"):
        s = sub.add_parser(name)
        s.add_argument("--patch", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--kernels", default=os.path.join(ROOT, "tests", "chx"))
        if name == "throughput":
            s.add_argument("--n", type=int, default=4)
            s.add_argument("--max-new", type=int, default=8192)
            s.add_argument("--binary", default=os.path.join(ROOT, "build", "cancho-gpu-w1"))
        else:
            s.add_argument("--variant-dir", required=True)
            s.add_argument("--n", type=int, default=8)
            s.add_argument("--new", type=int, default=64)
            s.add_argument("--ref-device", default="cpu")
            s.add_argument("--runner", default=os.path.join(ROOT, "build", "cancho-gpu-run08"))
            s.add_argument("--work", default=os.path.join(ROOT, "build", "t12"))
    a = ap.parse_args(argv)
    try:
        res = throughput(a) if a.cmd == "throughput" else agree(a)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != "problems"}, indent=1))
    except T12Error as e:
        print(f"error[{e.rule}]: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
