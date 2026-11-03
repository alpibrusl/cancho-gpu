#!/usr/bin/env python3
"""The exits of roadmap M7.2 (docs/resident.md section 21): the gradient of every LoRA adapter matrix of Qwen2.5-0.5B, in a Metal session, against peft's float32 autograd on the fixed batch.

    python3 examples/qwen/compare_lora.py WORKDIR [--weights DIR] [--loss-scale 16] [--b-std 0.02]

For the adapters as peft initialises them (A random, B = 0: the gradients of A are exactly zero, so only B's are tested) and for adapters with a random B (every gradient non-zero), prints per projection
(over the 24 layers) the cosine and the relative norm error against peft, the worst of each, the loss against peft's, and with --time the milliseconds of a full step (forward, backward and AdamW) and
the peak memory. WORKDIR is filled on first use with peft's answers (peft_reference.py).

    python3 examples/qwen/compare_lora.py WORKDIR --train 40 [--lr 2e-4] [--batches 8]

runs a short AdamW training (one batch a step, taken round and round from different windows of the corpus, linear warm-up then cosine decay) in peft (float32 on the CPU) and in the session from the same
adapters, and prints the loss of both at every step."""
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
import qwen_train as T  # noqa: E402

arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


def stats(ours, ref):
    """Per projection kind, over the layers: the smallest cosine and the largest relative norm error, for A and B; and the same for every single matrix."""
    rows = {}
    for k in sorted(ref):
        a, b = ours[k].ravel(), ref[k].ravel()
        nb = np.linalg.norm(b)
        na = np.linalg.norm(a)
        cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else (1.0 if na == nb else 0.0)
        rel = float(np.linalg.norm(a - b) / nb) if nb > 0 else float(na)
        rows[k] = (cos, rel, float(nb))
    return rows


def train_run(work, wdir, c, steps, lr, k):
    """The same short AdamW run in peft and in the session: the loss before each step, step by step."""
    import json
    import peft_reference as PR
    bt = os.path.join(work, f"train_batches_{k}.npy")
    if not os.path.exists(bt):
        np.save(bt, data.train_batches(k, c.nb, c.t).astype(np.int32))
    batches = np.load(bt).astype(np.int64)
    warm = max(1, steps // 8)
    lr_fn = lambda t: lr * (t / warm if t <= warm else 0.5 * (1 + np.cos(np.pi * (t - warm) / max(1, steps - warm))))
    AB = R.init_lora(c, 16, 7, b_std=0.0)
    fp = os.path.join(work, f"train_peft_{steps}_{lr}_{k}.json")
    if os.path.exists(fp):
        ref = json.load(open(fp))
    else:
        model, handles = PR.build(c, AB)
        t0 = time.time()
        losses, final = PR.train(model, handles, batches, steps, lr, lr_fn=lr_fn)
        print(f"peft: {steps} steps in {time.time() - t0:.0f} s")
        ref = dict(losses=losses)
        np.savez(os.path.join(work, f"train_peft_final_{steps}_{lr}_{k}.npz"), **final)
        json.dump(ref, open(fp, "w"))
        del model, handles
    d = os.path.join(work, "s_train")
    sess = T.write(d, c, wdir, batches, AB, r=16, loss_scale=float(arg("--loss-scale", 16.0)), train=True, steps=steps, lr=lr, lr_fn=lr_fn, save=os.path.join(d, "ad.bin"), lean="--lean" in sys.argv)
    ss = S.Session(arg("--metal", os.path.join(ROOT, "build", "cancho-gpu-metal-session")), sess)
    got, times = [], []
    for s in range(steps):
        out, secs = ss.ask("1")
        got.append(float(out.split()[0]))
        times.append(secs)
    ss.checkpoint()
    ss.close()
    rl = np.array(ref["losses"])
    gl = np.array(got)
    print(f"{'step':>4s} {'peft':>10s} {'session':>10s} {'diff':>10s}")
    for s in range(steps):
        if s < 5 or s % 5 == 0 or s == steps - 1:
            print(f"{s + 1:4d} {rl[s]:10.5f} {gl[s]:10.5f} {gl[s] - rl[s]:+10.5f}")
    print(f"largest |difference| of the loss over {steps} steps: {np.abs(gl - rl).max():.5f} (mean {np.abs(gl - rl).mean():.5f}); loss {rl[0]:.4f} -> {rl[-1]:.4f} (peft), {gl[0]:.4f} -> {gl[-1]:.4f} (session)")
    # the adapters at the end against peft's
    fin = np.load(os.path.join(work, f"train_peft_final_{steps}_{lr}_{k}.npz"))
    fused = T.read_fused(c, 16, os.path.join(d, "ad.bin"))
    cos, rel = [], []
    for l in range(c.layers):
        sp = T.split_params(c, 16, {g: (fused[(l, g, "A")], fused[(l, g, "B")]) for g in T.group_dims(c, 16)}, l)
        for name, v in sp.items():
            a, b = v.ravel(), fin[name].ravel()
            cos.append(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
            rel.append(float(np.linalg.norm(a - b) / np.linalg.norm(b)))
    print(f"adapters after {steps} steps against peft's: smallest cosine {min(cos):.5f}, largest relative norm error {max(rel):.2e}, over {len(cos)} matrices")
    print(f"ms a step (one request a step, the first {times[0] * 1000:.0f} ms compiles): median {sorted(times[1:])[len(times) // 2] * 1000:.0f}")


def timing(work, wdir, c, steps):
    """ms a step of the whole training step and the peak memory footprint of the process (`/usr/bin/time -l`)."""
    tok = data.fixed_batch(c.nb, c.t)
    AB = R.init_lora(c, 16, 7, b_std=0.0)
    d = os.path.join(work, "s_time")
    sess = T.write(d, c, wdir, tok, AB, r=16, loss_scale=float(arg("--loss-scale", 16.0)), train=True, steps=100, lr=1e-4, lean="--lean" in sys.argv)
    p = subprocess.Popen(["/usr/bin/time", "-l", arg("--metal", os.path.join(ROOT, "build", "cancho-gpu-metal-session")), sess], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def ask(line):
        t0 = time.time()
        p.stdin.write(line + "\n")
        p.stdin.flush()
        out = p.stdout.readline()
        return out, time.time() - t0

    first = ask("1")[1]
    for _ in range(2):
        ask("1")
    runs = []
    for _ in range(3):
        out, secs = ask(str(steps))
        runs.append(secs / steps * 1000)
    p.stdin.close()
    p.wait()
    err = p.stderr.read()
    peak = [int(line.split()[0]) for line in err.split("\n") if "peak memory footprint" in line]
    ms = sorted(runs)[1]
    print(f"cancho-gpu session: {ms:.1f} ms a step (median of 3 requests of {steps} steps; runs {', '.join(f'{r_:.1f}' for r_ in runs)}; the first request, {first * 1000:.0f} ms, compiles), "
          f"{c.n / ms * 1000:.0f} tokens a second, peak memory footprint {peak[0] / 1e9 if peak else float('nan'):.2f} GB")
    if "--torch" in sys.argv:
        for dt in ("float16", "float32"):
            r_ = subprocess.run([sys.executable, os.path.join(HERE, "torch_baseline.py"), "--dtype", dt, "--steps", "10"], capture_output=True, text=True)
            print(r_.stdout.strip().split("\n")[-1] if r_.stdout else r_.stderr[-300:])


def main():
    work = sys.argv[1]
    wdir = arg("--weights", data.WEIGHTS)
    ls = float(arg("--loss-scale", 16.0))
    bstd = float(arg("--b-std", 0.02))
    metal = arg("--metal", os.path.join(ROOT, "build", "cancho-gpu-metal-session"))
    os.makedirs(work, exist_ok=True)
    c = R.config_of(wdir)
    if "--time" in sys.argv:
        timing(work, wdir, c, int(arg("--steps", 10)))
        return
    if "--train" in sys.argv:
        train_run(work, wdir, c, int(arg("--train", 40)), float(arg("--lr", 2e-4)), int(arg("--batches", 8)))
        return
    tok = data.fixed_batch(c.nb, c.t)
    if not os.path.exists(os.path.join(work, "grads_randB.npz")):
        subprocess.run([sys.executable, os.path.join(HERE, "peft_reference.py"), work, "--weights", wdir, "--b-std", str(bstd)], check=True)
    ncount = float(c.nb * (c.t - 1))
    worst = {}
    for tag, b_std in (("init", 0.0), ("randB", bstd)):
        ref = np.load(os.path.join(work, f"grads_{tag}.npz"))
        refl = float(ref["loss"][0])
        AB = R.init_lora(c, 16, 7, b_std=b_std)
        d = os.path.join(work, f"s_{tag}")
        sess = T.write(d, c, wdir, tok, AB, r=16, loss_scale=ls, train=False, grads=os.path.join(d, "grads.bin"), lean="--lean" in sys.argv)
        ss = S.Session(metal, sess)
        out, first = ss.ask("1")
        loss = float(out.split()[0])
        times = [ss.ask("1")[1] for _ in range(3)]
        ss.checkpoint()
        ss.close()
        got = T.split_grads(c, 16, T.read_fused(c, 16, os.path.join(d, "grads.bin")), ncount, ls)
        st = stats(got, {k: ref[k] for k in ref.files if k != "loss"})
        print(f"\n== adapters '{tag}' (B {'= 0' if b_std == 0 else f'~ N(0, {b_std}^2)'}): loss {loss:.6f}, peft {refl:.6f}, difference {loss - refl:+.2e}; forward+backward {sorted(times)[1] * 1000:.0f} ms")
        print(f"{'projection':10s} {'cos A (min over layers)':>26s} {'rel A (max)':>12s} {'cos B (min)':>12s} {'rel B (max)':>12s}")
        for p in R.PROJ:
            ca = [st[f"l{l:02d}.{p}.A"] for l in range(c.layers)]
            cb = [st[f"l{l:02d}.{p}.B"] for l in range(c.layers)]
            print(f"{p:10s} {min(x[0] for x in ca):26.6f} {max(x[1] for x in ca):12.2e} {min(x[0] for x in cb):12.6f} {max(x[1] for x in cb):12.2e}")
        if tag == "randB":
            print("per layer, the smallest cosine over the seven projections (A / B) and the largest relative error (A / B):")
            for l in range(c.layers):
                ka = [st[f"l{l:02d}.{p}.A"] for p in R.PROJ]
                kb = [st[f"l{l:02d}.{p}.B"] for p in R.PROJ]
                print(f"  layer {l:2d}: cos {min(x[0] for x in ka):.5f} / {min(x[0] for x in kb):.5f}   rel {max(x[1] for x in ka):.2e} / {max(x[1] for x in kb):.2e}")
        cos_all = [v[0] for k, v in st.items() if not (tag == "init" and k.endswith(".A"))]
        rel_all = [v[1] for k, v in st.items() if not (tag == "init" and k.endswith(".A"))]
        worst[tag] = (min(cos_all), max(rel_all), len(cos_all))
        if tag == "init":
            za = max(float(np.abs(got[k]).max()) for k in got if k.endswith(".A"))
            print(f"(with B = 0 the gradients of A are zero: ours has largest |value| {za:.1e}, peft's {max(float(np.abs(ref[k]).max()) for k in ref.files if k.endswith('.A')):.1e})")
    for tag, (cs, rl, nn_) in worst.items():
        print(f"{tag}: {nn_} matrices, smallest cosine {cs:.6f} (exit: at least 0.995), largest relative norm error {rl:.2e}")


if __name__ == "__main__":
    main()
