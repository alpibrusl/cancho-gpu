#!/usr/bin/env python3
"""docs/weight-space.md section 13 (W1): the reduced reproduction of the Qwen3.5-0.8B weight edit.

    python3 examples/weightspace/w1.py diff   BASE EDITED [--json report.json] [--w1-expect]
    python3 examples/weightspace/w1.py behave --base BASE --edited EDITED --data gsm8k_test.parquet --out DIR --gpu-free
                                              [--n 200] [--max-new 8192] [--batch 8] [--backend transformers|mlx] [--device mps]

BASE and EDITED are local checkpoints: a .safetensors file or a directory holding them. `diff` runs on the CPU, reads
nothing it does not compare and writes only the --json file. `--w1-expect` additionally holds the diff to what the authors
published (README of RikiyaT/Qwen3.5-0.8B-weight-space-programming and exp8_math/runs/p50c/edit/manifest.json): exactly two
tensors differ, 64 columns each, with the published entry counts and sha256. `behave` runs both models on the first --n
GSM8K test problems, applies the section 13 rule and writes the records and the verdict into DIR; it needs the GPU, and
refuses to start without --gpu-free (your statement that nothing else is using it) or when `diff --w1-expect` fails.

Exit status: 0 done (and, with --w1-expect, matched), 1 the expectations or the rule's inputs failed, 2 the input is unusable."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w1_behave as B  # noqa: E402
import w1_diff as D  # noqa: E402

PREFIX = "model.language_model.layers."
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
    recs = {}
    for tag, path in (("base", a.base), ("edited", a.edited)):
        gen = pick_generator(a, path)
        recs[tag] = B.run_behaviour(gen, problems, tag, a.max_new, a.seed, a.batch, os.path.join(a.out, f"{tag}_{a.max_new}.jsonl"))
        del gen
    v = B.verdict(recs["base"], recs["edited"])
    v["protocol"] = {"n": a.n, "max_new": a.max_new, "seed": a.seed, "batch": a.batch, "backend": a.backend, **B.SAMPLING}
    with open(os.path.join(a.out, f"verdict_{a.max_new}.json"), "w", encoding="utf-8") as f:
        json.dump(v, f, indent=1)
    print(B.render_verdict(v, f"(cap {a.max_new})"))
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
    b.add_argument("--backend", choices=["transformers", "mlx"], default="transformers")
    b.add_argument("--device", default="mps")
    a = ap.parse_args(argv)
    if a.cmd == "behave" and (a.n < 1 or a.max_new < 1 or a.batch < 1):
        print("--n, --max-new and --batch must be positive", file=sys.stderr)
        return 2
    try:
        return cmd_diff(a) if a.cmd == "diff" else cmd_behave(a)
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
