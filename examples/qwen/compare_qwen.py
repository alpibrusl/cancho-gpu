#!/usr/bin/env python3
"""The exits of roadmap M7.1 (docs/resident.md section 20): Qwen2.5-0.5B's forward pass over the fixed batch (8 sequences of 512) in a Metal session, against transformers.

    python3 examples/qwen/compare_qwen.py WORKDIR [--reps 5] [--spec] [--torch] [--metal build/cancho-gpu-metal-session]

WORKDIR holds (and is filled on the first run with) `weights/` (convert.py), `hf/` (hf_reference.py: transformers float32 on the CPU) and `session/`. Prints: the loss against transformers',
the arg max agreement (and how far apart the two largest logits were where they disagree), the residual stream after every block against transformers' (relative L2 and relative to the largest value),
the time of a forward pass, and with --torch the same batch's forward in torch on MPS in float32 and float16 autocast; --spec also checks the binary64 specification against transformers (26 s, 4 GB)."""
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import data  # noqa: E402
import qwen_ref as R  # noqa: E402
import qwen_session as S  # noqa: E402

arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


def main():
    work = sys.argv[1]
    reps = int(arg("--reps", 5))
    metal = arg("--metal", os.path.join(ROOT, "build", "cancho-gpu-metal-session"))
    wdir, hdir = os.path.join(work, "weights"), os.path.join(work, "hf")
    if not os.path.exists(os.path.join(wdir, "manifest.json")):
        import convert
        src, cfg = convert.find(data.MODEL)
        m = convert.convert(src, cfg, wdir)
        print("converted:", m["report"])
    if not os.path.exists(os.path.join(hdir, "loss.npy")):
        import hf_reference
        hf_reference.run(hdir, 8, 512)
    c = R.config_of(wdir)
    tok = data.fixed_batch(c.nb, c.t)
    assert (np.load(os.path.join(hdir, "tokens.npy")) == tok).all(), "the reference was made on another batch"
    sess = S.write(os.path.join(work, "session"), c, wdir, tok, acts=os.path.join(work, "acts.bin"))
    ss = S.Session(metal, sess)
    loss, am, first = ss.forward(c.n)
    times = [ss.forward(c.n)[2] for _ in range(reps)]
    ss.checkpoint()
    err = ss.close()
    ms = sorted(times)[len(times) // 2] * 1000
    hl = float(np.load(os.path.join(hdir, "loss.npy"))[0])
    ours = R.mean_loss(loss, c)
    print(f"loss: {ours:.6f}   transformers float32: {hl:.6f}   difference {ours - hl:+.2e}  (exit: within 2e-3)")
    href = np.load(os.path.join(hdir, "argmax.npy"))
    agree = float((am == href).mean())
    print(f"arg max of the logits equal at {agree * 100:.2f}% of {c.n} positions ({int((am != href).sum())} differ; exit: at least 99%)")
    margin = np.load(os.path.join(hdir, "margin.npy"))
    print("  the differing positions, transformers' gap between its largest two logits:", [round(float(x), 4) for x in margin[am != href]])
    acts = S.read_acts(os.path.join(work, "acts.bin"), c)
    hs = np.load(os.path.join(hdir, "hidden.npy"))
    print("residual stream after block k against transformers' (relative L2 / relative to the largest value):")
    worst = 0.0
    for k in range(1, c.layers + 1):
        a, b = acts[k - 1].astype(np.float64), hs[k].astype(np.float64)
        l2, mx = np.linalg.norm(a - b) / np.linalg.norm(b), np.abs(a - b).max() / np.abs(b).max()
        worst = max(worst, l2)
        print(f"  {k:2d}: {l2:.2e} / {mx:.2e}")
    print(f"forward pass over {c.n} tokens: {ms:.1f} ms (median of {reps} requests; first request {first * 1000:.0f} ms), {c.n / ms * 1000:.0f} tokens a second")
    if "--torch" in sys.argv:
        for dt in ("float32", "float16"):
            r = subprocess.run([sys.executable, os.path.join(HERE, "torch_forward.py"), "--dtype", dt], capture_output=True, text=True)
            print(r.stdout.strip().split("\n")[-1] if r.stdout else r.stderr[-300:])
    if "--spec" in sys.argv:
        P = R.load_params(wdir, c)
        xs = R.forward(c, P, tok)
        rows, best = R.head(c, P, xs[-1], R.next_targets(tok))
        print(f"binary64 specification: loss {R.mean_loss(rows, c):.6f}, arg max equal to transformers' at {(best == href).mean() * 100:.2f}%; session against the specification: "
              f"loss {ours - R.mean_loss(rows, c):+.2e}, arg max {(am == best).mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
