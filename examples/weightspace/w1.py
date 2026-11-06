#!/usr/bin/env python3
"""docs/weight-space.md section 13 (W1): the reduced reproduction of the Qwen3.5-0.8B weight edit.

    python3 examples/weightspace/w1.py diff   BASE EDITED [--json report.json] [--w1-expect]
    python3 examples/weightspace/w1.py behave --base BASE --edited EDITED --data gsm8k_test.parquet --out DIR --gpu-free
                                              [--n 200] [--max-new 8192] [--batch 8] [--backend transformers|mlx|cancho] [--device mps]
                                              [--cancho-bin build/cancho-gpu-w1] [--kernels tests/chx] [--mode f16|f32] [--tmax N]
    python3 examples/weightspace/w1.py status  --dir DIR [--n 200] [--max-new 8192]
    python3 examples/weightspace/w1.py verdict --dir DIR [--n 200] [--max-new 8192] [--json verdict.json]

BASE and EDITED are local checkpoints: a .safetensors file or a directory holding them. `diff` runs on the CPU, reads
nothing it does not compare and writes only the --json file. `--w1-expect` additionally holds the diff to what the authors
published (README of RikiyaT/Qwen3.5-0.8B-weight-space-programming and exp8_math/runs/p50c/edit/manifest.json): exactly two
tensors differ, 64 columns each, with the published entry counts and sha256. `behave` runs both models on the first --n
GSM8K test problems, applies the section 13 rule and writes the records and the verdict into DIR; it needs the GPU (`--backend cancho`: cancho-gpu's own
Metal engine, build/cancho-gpu-w1, one conversation at a time; w1_cancho.py says what its seeds mean), and
refuses to start without --gpu-free (your statement that nothing else is using it) or when `diff --w1-expect` fails.

`status` and `verdict` read DIR's base_<max-new>.jsonl and edited_<max-new>.jsonl (what `behave` wrote, or what scripts/gcp/w1_cloud.sh pulled
from a VM), need numpy only and no GPU. `status` exits 0 when both cover problems 0..n-1, else 1; `verdict` applies the section 13.3 rule to
them (exit 1 when they do not cover n problems) and says whether the rule asks for the cap check.

Exit status: 0 done (and, with --w1-expect, matched), 1 the expectations or the rule's inputs failed, 2 the input is unusable."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w1_behave as B  # noqa: E402
import w1_cancho as C  # noqa: E402
import w1_diff as D  # noqa: E402

PREFIX = "model.language_model.layers."
CAP_CHECK_N, CAP_CHECK_MAX_NEW = 60, 16384   # docs/weight-space.md 13.3
# What the authors publish for the edit (all [read], from their README and manifest.json, 2026-11-04).
EXPECT = {
    PREFIX + "15.mlp.down_proj.weight": {"changed_entries": 65260, "changed_columns": 64, "changed_rows": 1024, "shape": [1024, 3584],
        "base_sha256": "ef60d2c687bab19464dca9c8bfd7777e46ad226df0c9e25643b0bab04b5a1cca",
        "edited_sha256": "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53"},
    PREFIX + "19.mlp.down_proj.weight": {"changed_entries": 64928, "changed_columns": 64, "changed_rows": 1023, "shape": [1024, 3584],
        "base_sha256": "638eedeb701b94aa861f4438e615c07ec6c60ee703db1bfee2de577bb6498c3e",
        "edited_sha256": "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2"},
}


def check_expect(rep):
    """[(what, ok, detail)] of the report against EXPECT."""
    res = []
    got = {c["name"]: c for c in rep["changed"]}
    res.append(("the changed tensors are exactly the two published", sorted(got) == sorted(EXPECT), sorted(got)))
    res.append(("no tensor added, removed or reshaped", not (rep["only_in_base"] or rep["only_in_edited"] or rep["mismatched"]),
                {k: rep[k] for k in ("only_in_base", "only_in_edited", "mismatched")}))
    for name, want in EXPECT.items():
        c = got.get(name)
        if c is None:
            res.append((name + " changed", False, "identical in both"))
            continue
        for k, v in want.items():
            res.append((f"{name} {k}", c.get(k) == v, f"got {c.get(k)}, published {v}"))
    return res


def cmd_diff(a):
    rep = D.diff_checkpoints(D.read_checkpoint(a.base), D.read_checkpoint(a.edited))
    print(D.render(rep))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
    if not a.w1_expect:
        return 0
    bad = 0
    for what, ok, detail in check_expect(rep):
        print(("ok   " if ok else "FAIL ") + what + ("" if ok else f"  ({detail})"))
        bad += not ok
    return 1 if bad else 0


def pick_generator(a, model_dir):
    if a.backend == "cancho":
        tmax = a.tmax or C.default_tmax(a.max_new)
        return C.cancho_generator(C.command_for(a.cancho_bin, model_dir, a.kernels, a.mode, tmax))
    if a.backend == "mlx":
        return B.mlx_generator(model_dir)
    return B.transformers_generator(model_dir, a.device)


def cmd_behave(a):
    if not a.gpu_free:
        print("refusing: pass --gpu-free to state that no other job (M8's training) is using the GPU", file=sys.stderr)
        return 2
    problems = B.load_problems(a.data, a.n)
    if not a.no_verify:
        rep = D.diff_checkpoints(D.read_checkpoint(a.base), D.read_checkpoint(a.edited))
        bad = [w for w, ok, _ in check_expect(rep) if not ok]
        if bad:
            print("refusing: the edited checkpoint is not the published edit of the base (" + "; ".join(bad[:3]) +
                  "); run `diff --w1-expect`, or pass --no-verify", file=sys.stderr)
            return 1
    os.makedirs(a.out, exist_ok=True)
    if a.backend == "cancho":
        a.batch = 1   # one conversation at a time: problem i is generated with seed --seed + i (w1_cancho.py)
    recs = {}
    for tag, path in (("base", a.base), ("edited", a.edited)):
        gen = pick_generator(a, path)
        try:
            recs[tag] = B.run_behaviour(gen, problems, tag, a.max_new, a.seed, a.batch, os.path.join(a.out, f"{tag}_{a.max_new}.jsonl"))
        finally:
            if hasattr(gen, "close"):
                gen.close()
        del gen
    v = B.verdict(recs["base"], recs["edited"])
    v["protocol"] = {"n": a.n, "max_new": a.max_new, "seed": a.seed, "batch": a.batch, "backend": a.backend, **B.SAMPLING}
    if a.backend == "cancho":
        v["protocol"].update({"mode": a.mode, "seed_model": "per problem: sampler.new(--seed + id), splitmix64 into xorshift64*"})
    with open(os.path.join(a.out, f"verdict_{a.max_new}.json"), "w", encoding="utf-8") as f:
        json.dump(v, f, indent=1)
    print(B.render_verdict(v, f"(cap {a.max_new})"))
    return 0


def cap_check_line(v, max_new):
    """Whether 13.3(d) asks for the cap check; only the main run (not the cap check itself) has the question."""
    if max_new == CAP_CHECK_MAX_NEW:
        return "this is the cap check: read it next to the main run's verdict (13.3)"
    if v["verdict"] in ("reproduces", "partial"):
        return (f"cap check (13.3(d)): ASKED, the main run is '{v['verdict']}': the first {CAP_CHECK_N} problems again at "
                f"--max-new {CAP_CHECK_MAX_NEW}, base and edited (N={CAP_CHECK_N} MAX_NEW={CAP_CHECK_MAX_NEW} scripts/gcp/w1_cloud.sh)")
    return "cap check (13.3(d)): not asked, the main run is not reproduced"


def covered(a):
    """([(tag, finished ids below n)], whether every id below n is done for both) of DIR's logs."""
    got = []
    for tag in ("base", "edited"):
        done = B.load_done(os.path.join(a.dir, f"{tag}_{a.max_new}.jsonl"), tag, a.max_new)
        got.append((tag, done, sum(1 for i in range(a.n) if i in done)))
    return got, all(c == a.n for _, _, c in got)


def cmd_status(a):
    got, full = covered(a)
    print(" ".join(f"{tag} {c}/{a.n}" for tag, _, c in got) + (" complete" if full else " incomplete"))
    return 0 if full else 1


def cmd_verdict(a):
    got, full = covered(a)
    if not full:
        print("no verdict: " + " ".join(f"{tag} {c}/{a.n}" for tag, _, c in got) + " problems finished", file=sys.stderr)
        return 1
    v = B.verdict([got[0][1][i] for i in range(a.n)], [got[1][1][i] for i in range(a.n)])
    v["protocol"] = {"n": a.n, "max_new": a.max_new, **B.SAMPLING, "source": "jsonl in " + a.dir}
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(v, f, indent=1)
    print(B.render_verdict(v, f"(cap {a.max_new})"))
    print(cap_check_line(v, a.max_new))
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diff")
    d.add_argument("base")
    d.add_argument("edited")
    d.add_argument("--json")
    d.add_argument("--w1-expect", action="store_true")
    b = sub.add_parser("behave")
    b.add_argument("--base", required=True)
    b.add_argument("--edited", required=True)
    b.add_argument("--data", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--gpu-free", action="store_true")
    b.add_argument("--no-verify", action="store_true")
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--max-new", type=int, default=8192)
    b.add_argument("--batch", type=int, default=8)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--backend", choices=["transformers", "mlx", "cancho"], default="transformers")
    b.add_argument("--device", default="mps")
    b.add_argument("--cancho-bin", default=os.path.join(HERE, "..", "..", "build", "cancho-gpu-w1"))
    b.add_argument("--kernels", default=os.path.join(HERE, "..", "..", "tests", "chx"))
    b.add_argument("--mode", choices=["f16", "f32"], default="f16")
    b.add_argument("--tmax", type=int, default=0)
    for name in ("status", "verdict"):
        c = sub.add_parser(name)
        c.add_argument("--dir", required=True)
        c.add_argument("--n", type=int, default=200)
        c.add_argument("--max-new", type=int, default=8192)
        if name == "verdict":
            c.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd != "diff" and (a.n < 1 or a.max_new < 1 or getattr(a, "batch", 1) < 1):
        print("--n, --max-new and --batch must be positive", file=sys.stderr)
        return 2
    try:
        return {"diff": cmd_diff, "behave": cmd_behave, "status": cmd_status, "verdict": cmd_verdict}[a.cmd](a)
    except (D.ReadError, B.BehaveError) as e:
        print("error: " + str(e), file=sys.stderr)
        return 2
    except ImportError as e:
        print(f"error: a dependency is missing ({e})", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
