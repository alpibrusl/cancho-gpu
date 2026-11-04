#!/usr/bin/env python3
"""W2b of docs/weight-space.md: the arms Z, R and G(N), and the evaluator that scores any network (any arm) on the two held-out sets.

    python3 examples/weightspace/arms.py train N OUT_DIR [--steps S] [--lr L] [--batch B]      G(N): trains on the CPU session, writes OUT_DIR/params.npz and OUT_DIR/train.json
    python3 examples/weightspace/arms.py eval OUT.json ARM=DIR_OR_npz ...                       scores Z, R and each given network on the interior and band sets (and prints a markdown table)
    python3 examples/weightspace/arms.py validate N [--lr L] [--steps S]                         the learning rate is chosen on a *validation* set (fresh held-in draws), never on the held-out sets

The held-out sets (task.sets(): 2,000 poses each, fixed seeds) are made here and nowhere else: the harness's `forward` draws its own held-in poses and never sees them.
Errors are the mean Euclidean position error in millimetres with a 95% percentile bootstrap interval (10,000 resamples of the 2,000 poses, numpy default_rng(20261104))."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import task  # noqa: E402
import trainer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
SESSION = os.environ.get("CANCHO_GPU_SESSION", os.path.join(ROOT, "build", "cancho-gpu-session"))
VALIDATION_SEED = 20261104005


def opt(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "params.npz")
    z = np.load(path)
    return {n: z[n].astype(np.float32) for n in task.NAMES}


def cmd_train():
    n, out = int(sys.argv[2]), sys.argv[3]
    steps, lr, batch = opt("--steps", 40000), opt("--lr", 1e-3), opt("--batch", 250)
    os.makedirs(out, exist_ok=True)
    p, secs, last = trainer.train(SESSION, os.path.join(out, "session"), n, steps, batch=batch, lr=lr)
    np.savez(os.path.join(out, "params.npz"), **p)
    json.dump({"arm": f"G({n})", "n": n, "steps": steps, "batch": batch, "lr": lr, "train_seconds": secs, "last_minibatch_loss_standardised": last}, open(os.path.join(out, "train.json"), "w"), indent=1)
    print(f"G({n}): {steps} steps of {batch}, lr {lr}: {secs:.0f} s, last minibatch loss {last:.5f}")


def cmd_validate():
    n = int(sys.argv[2])
    steps, batch = opt("--steps", 4000), opt("--batch", 250)
    val = task.sample_held_in(2000, np.random.default_rng(VALIDATION_SEED))
    for lr in ([opt("--lr", 1e-3)] if "--lr" in sys.argv else [3e-4, 1e-3, 3e-3]):
        d = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"ws_val_{n}_{lr}")
        p, secs, _ = trainer.train(SESSION, d, n, steps, batch=batch, lr=lr, log=lambda *_: None)
        print(f"validation  N={n} steps={steps} lr={lr}: mean error {task.errors_mm(p, val).mean():.3f} mm ({secs:.0f} s)")


def cmd_eval():
    out = sys.argv[2]
    nets = {"Z": task.zero_params(), "R": task.init_params()}
    for a in sys.argv[3:]:
        k, v = a.split("=", 1)
        nets[k] = load(v)
    interior, band = task.sets()
    res, rows = {}, []
    for name, p in nets.items():
        r = {}
        for sname, q in (("interior", interior), ("band", band)):
            m, lo, hi = task.bootstrap_ci(task.errors_mm(p, q))
            r[sname] = {"mean_mm": m, "lo_mm": lo, "hi_mm": hi}
        res[name] = r
        rows.append(f"| {name} | {r['interior']['mean_mm']:.2f} [{r['interior']['lo_mm']:.2f}, {r['interior']['hi_mm']:.2f}] | {r['band']['mean_mm']:.2f} [{r['band']['lo_mm']:.2f}, {r['band']['hi_mm']:.2f}] |")
    # the same networks through the session (the harness's forward path) on the first 256 poses of each set: it must agree with the binary64 forward
    worst = 0.0
    for name, p in nets.items():
        for q in (interior[:256], band[:256]):
            got = trainer.run_forward(SESSION, os.path.join(os.environ.get("TMPDIR", "/tmp"), "ws_eval_fwd"), p, q)
            worst = max(worst, float(np.abs(got - task.forward_np(p, q)).max()))
    res["_session_vs_binary64_largest_output_difference_m"] = worst
    json.dump(res, open(out, "w"), indent=1)
    print("| arm | interior mean error, mm [95% CI] | band mean error, mm [95% CI] |\n|---|---|---|")
    print("\n".join(rows))
    print(f"session forward against the binary64 forward, 512 poses per network: largest output difference {worst:.2e} m")


if __name__ == "__main__":
    {"train": cmd_train, "eval": cmd_eval, "validate": cmd_validate}[sys.argv[1]]()
