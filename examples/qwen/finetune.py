#!/usr/bin/env python3
"""The 300-step fine-tuning run of roadmap M7.3 (docs/resident.md section 22): rank-16 LoRA (alpha 32) on Qwen2.5-0.5B over the corpus of finetune_data.py, in a Metal session of .chx kernels, and the same
batches and initial adapters in peft on MPS (peft_finetune.py).

    python3 examples/qwen/finetune_data.py WORK                                   # the corpus (once)
    python3 examples/qwen/finetune.py WORK train  [--steps 300] [--lr 2e-4]       # cancho-gpu: WORK/cancho.json (the loss of every step, ms a step, peak memory), checkpoints WORK/ck/
    python3 examples/qwen/finetune.py WORK eval                                   # the held-out loss at every checkpoint (adds it to WORK/cancho.json)
    python3 examples/qwen/finetune.py WORK compare [peft_float16 peft_float32]    # the curves side by side and the largest differences

The schedule is a linear warm-up over steps / 10 and a cosine decay to 0, AdamW (0.9, 0.999, 1e-8, no weight decay), batch t of the run is batch t of the corpus (8 sequences of 512, no repeats).
A session's `table` is one file of at most 512 MiB, which holds 60 batches of embeddings (7.3 MB each), so the 300 steps are five sessions of 60 that hand the adapters and Adam's moments over through
`save` / `checkpoint` (qwen_train.write `state`, `first_step`); the handover is exact (tests/qwen.py)."""
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import qwen_ref as R  # noqa: E402
import qwen_session as S  # noqa: E402
import qwen_train as T  # noqa: E402

arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
WDIR = arg("--weights", "/private/tmp/claude-502/t/qwen05")
METAL = arg("--metal", os.path.join(ROOT, "build", "cancho-gpu-metal-session"))
EVERY = int(arg("--every", 30))
R_, ALPHA = 16, 32.0


def lr_schedule(steps, lr):
    warm = max(1, steps // 10)
    return lambda t: lr * (t / warm if t <= warm else 0.5 * (1 + np.cos(np.pi * (t - warm) / max(1, steps - warm))))


def load(work):
    z = np.load(os.path.join(work, "corpus.npz"))
    return z["train"].astype(np.int64).reshape(-1, 8, 512), z["heldout"].astype(np.int64).reshape(-1, 8, 512)


def initial_adapters(c):
    return R.init_lora(c, R_, 7, b_std=0.0)


def spawn(session):
    """The session under /usr/bin/time -l, so that its peak memory footprint is read when it ends."""
    return subprocess.Popen(["/usr/bin/time", "-l", METAL, session], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def ask(p, line):
    t0 = time.perf_counter()
    p.stdin.write(line + "\n")
    p.stdin.flush()
    out = p.stdout.readline()
    if not out:
        raise SystemExit("session: " + p.stderr.read()[-600:])
    return out, time.perf_counter() - t0


def close(p):
    p.stdin.close()
    p.wait()
    err = p.stderr.read()
    m = re.search(r"(\d+)\s+peak memory footprint", err)
    return int(m.group(1)) if m else 0


def split_fused(c, fused):
    AB = {}
    for l in range(c.layers):
        AB.update(T.split_params(c, R_, {g: (fused[(l, g, "A")], fused[(l, g, "B")]) for g in T.group_dims(c, R_)}, l))
    return AB


def train(work, c):
    steps, lr, seg = int(arg("--steps", 300)), float(arg("--lr", 2e-4)), int(arg("--segment", 60))
    batches, _ = load(work)
    assert len(batches) >= steps
    lr_fn = lr_schedule(steps, lr)
    AB = initial_adapters(c)
    ck = os.path.join(work, "ck")
    os.makedirs(ck, exist_ok=True)
    np.savez(os.path.join(ck, "step000.npz"), **AB)
    state, losses, times, peaks = None, [], [], []
    for s0 in range(0, steps, seg):
        n = min(seg, steps - s0)
        d = os.path.join(work, "s_train")
        sess = T.write(d, c, WDIR, batches[s0:s0 + n], AB, r=R_, alpha=ALPHA, loss_scale=16.0, train=True, steps=n, lr=lr, lr_fn=lr_fn, save=os.path.join(d, "ad.bin"), save_state=os.path.join(d, "st"),
                       state=state, first_step=s0)
        p = spawn(sess)
        for j in range(n):
            out, secs = ask(p, "1")
            losses.append(float(out.split()[0]))
            times.append(secs * 1000 if j > 0 else None)  # the first request of a session compiles and warms
            step = s0 + j + 1
            if step % EVERY == 0 or j == n - 1:
                ask(p, "checkpoint")
                shutil.copy(os.path.join(d, "ad.bin"), os.path.join(ck, f"step{step:03d}.bin"))
            if step % 10 == 0:
                print(f"step {step}: loss {losses[-1]:.5f}, {times[-1] or 0:.0f} ms", flush=True)
        peaks.append(close(p))
        fused = T.read_fused(c, R_, os.path.join(d, "ad.bin"))
        m, v = T.read_fused(c, R_, os.path.join(d, "st.m")), T.read_fused(c, R_, os.path.join(d, "st.v"))
        state = {k: (m[k], v[k]) for k in m}
        AB = split_fused(c, fused)
    ts = [t for t in times if t is not None]
    res = dict(losses=losses, ms=times, median_ms=float(np.median(ts)), mean_ms=float(np.mean(ts)), peak_bytes=max(peaks), peaks=peaks, steps=steps, lr=lr)
    json.dump(res, open(os.path.join(work, "cancho.json"), "w"))
    np.savez(os.path.join(work, "cancho_final.npz"), **AB)
    print(f"{steps} steps: median {res['median_ms']:.1f} ms a step ({c.n / res['median_ms'] * 1000:.0f} tokens a second), peak footprint {max(peaks) / 1e9:.2f} GB, loss {losses[0]:.4f} -> {losses[-1]:.4f}")


def heldout_loss(c, tokens, AB, d):
    """The mean loss over the held-out batches ([k, 8, 512]) of a forward pass with the adapters AB (the session of training with no optimiser; one batch a request)."""
    sess = T.write(d, c, WDIR, tokens, AB, r=R_, alpha=ALPHA, loss_scale=16.0, train=False, steps=1)
    p = spawn(sess)
    ls = [float(ask(p, "1")[0].split()[0]) for _ in range(len(tokens))]
    close(p)
    return float(np.mean(ls)), ls


def evaluate(work, c):
    _, held = load(work)
    res = json.load(open(os.path.join(work, "cancho.json")))
    ev = {}
    ck = os.path.join(work, "ck")
    for f in sorted(os.listdir(ck)):
        step = int(f[4:7])
        if f.endswith(".npz"):
            AB = {k: v for k, v in np.load(os.path.join(ck, f)).items()}
        elif f.endswith(".bin"):
            AB = split_fused(c, T.read_fused(c, R_, os.path.join(ck, f)))
        else:
            continue
        if step % EVERY and step != res["steps"]:
            continue
        ev[step], _ = heldout_loss(c, held, AB, os.path.join(work, "s_eval"))
        print(f"step {step}: held-out {ev[step]:.5f}", flush=True)
    res["heldout"] = ev
    json.dump(res, open(os.path.join(work, "cancho.json"), "w"))


def compare(work, names):
    runs = {"cancho": json.load(open(os.path.join(work, "cancho.json")))}
    for n in names:
        runs[n] = json.load(open(os.path.join(work, n + ".json")))
    base = runs["cancho"]
    steps = base["steps"]
    print(f"{'step':>5s} " + " ".join(f"{n:>13s}" for n in runs))
    for s in list(range(1, 6)) + list(range(10, steps + 1, 10)):
        print(f"{s:5d} " + " ".join(f"{r['losses'][s - 1]:13.5f}" for r in runs.values()))
    print("\nheld-out loss")
    hs = sorted(int(k) for k in base["heldout"])
    for s in hs:
        print(f"{s:5d} " + " ".join(f"{r['heldout'][str(s)] if str(s) in r['heldout'] else r['heldout'].get(s, float('nan')):13.5f}" for r in runs.values()))
    names = list(runs)
    print("\nlargest |difference| of the training loss over the steps (mean) / of the held-out loss")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da = np.abs(np.array(runs[a]["losses"]) - np.array(runs[b]["losses"]))
            ha = [abs(float(runs[a]["heldout"][str(s)]) - float(runs[b]["heldout"][str(s)])) for s in hs]
            print(f"  {a} vs {b}: train {da.max():.5f} (mean {da.mean():.5f}, step {int(da.argmax()) + 1}); held-out {max(ha):.5f}")
    for n, r in runs.items():
        print(f"{n}: median {r['median_ms']:.1f} ms a step, {4096 / r['median_ms'] * 1000:.0f} tokens a second, peak {r.get('peak_bytes', 0) / 1e9:.2f} GB")


if __name__ == "__main__":
    work, cmd = sys.argv[1], sys.argv[2]
    cfg = R.config_of(WDIR)
    {"train": lambda: train(work, cfg), "eval": lambda: evaluate(work, cfg), "compare": lambda: compare(work, [a for a in sys.argv[3:] if not a.startswith("--")])}[cmd]()
