#!/usr/bin/env python3
"""The launches and the time of a training step of the ResNet-18-class backbone (docs/resident.md section 26), cancho-gpu on Metal against torch on MPS.

    python3 examples/cnn/bench_rn.py launches                       # the programs of one step by kind (no GPU)
    python3 examples/cnn/bench_rn.py time [--batch 8] [--rounds 5] [--steps 10]    # interleaved rounds; ONE GPU job: check `ps` first
    python3 examples/cnn/bench_rn.py profile [--batch 8]            # CANCHOGPU_PROFILE=2: GPU time by kernel kind

The GPU is shared (docs/resident.md section 25 is queued on it): `time` and `profile` refuse to start if a Metal session, a finetune or an ollama runner is alive."""
import collections
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import resnet_ref as R  # noqa: E402
import rn_session as S  # noqa: E402

SESSION = os.path.join(ROOT, "build", "cancho-gpu-metal-session")


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


BATCH = int(arg("--batch", 8))


def kind(path):
    return re.sub(r"_(stem|s\d+b\d+[abd]|up\d+|\d+)$", "", os.path.basename(path).replace(".chx", ""))


def write(total):
    c = R.Config(n=BATCH, classes=10)
    rng = np.random.RandomState(1)
    imgs = rng.normal(size=(4, c.n, c.h, c.w, c.cin))
    labs = rng.randint(0, 10, size=(4, c.n))
    return c, S.write(tempfile.mkdtemp(), c, imgs, labs, total, params=R.init_params(c, 1))


def launches():
    _, sess = write(1)
    k = collections.Counter(kind(l.split()[2]) for l in open(sess) if l.startswith("program "))
    groups = {"optimiser": ("adam", "adamw_h"), "convolutions": ("convk", "convk_dx", "convk_dw", "wsum", "wsum_mask"),
              "BatchNorm": ("bn_part", "bn_fin", "bn_bwd_fin", "bn_apply", "bn_apply_res", "bn_apply_lin", "bn_bwd_part", "bn_bwd_part_res", "bn_bwd_part_lin", "bn_bwd_apply", "bn_bwd_apply_res", "bn_bwd_apply_lin")}
    total = sum(k.values())
    print(f"{total} programs a step")
    for g, ks in groups.items():
        print(f"  {g}: {sum(k[x] for x in ks)}")
    print("  the rest:", total - sum(k[x] for ks in groups.values() for x in ks))
    print(sorted(k.items(), key=lambda x: -x[1]))


def busy():
    out = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if re.search(r"metal-session|finetune|qwen_train|ollama runner|mlx_lm", l) and "bench_rn" not in l]


def session_proc(sess, env=None, err=None):
    p = subprocess.Popen([SESSION, sess], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err or subprocess.DEVNULL, text=True, env=env)
    p.stdin.write("1\n")
    p.stdin.flush()
    p.stdout.readline()
    return p


def run_steps(p, n):
    t0 = time.perf_counter()
    p.stdin.write(f"{n}\n")
    p.stdin.flush()
    p.stdout.readline()
    return (time.perf_counter() - t0) / n * 1000


def timing():
    import torch
    import resnet_torch as T
    rounds, steps = int(arg("--rounds", 5)), int(arg("--steps", 10))
    if busy():
        sys.exit("the GPU is in use: " + busy()[0])
    c, sess = write(rounds * steps + 2)
    p = session_proc(sess)
    P = R.init_params(c, 1)
    x = torch.tensor(np.random.RandomState(1).normal(size=(c.n, c.h, c.w, c.cin)).astype(np.float32)).permute(0, 3, 1, 2).contiguous().to("mps")
    y = torch.tensor(np.random.RandomState(2).randint(0, 10, size=c.n)).to("mps")
    runners = {"cancho-gpu": lambda: run_steps(p, steps)}
    for name, amp in (("torch-mps-f32", False), ("torch-mps-amp", True)):
        m = T.ResNet(c).float()
        T.load(m, c, P)
        m = m.to("mps").train()
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        def go(m=m, opt=opt, amp=amp):
            torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(steps):
                with torch.autocast(device_type="mps", dtype=torch.float16, enabled=amp):
                    out = m(x)
                loss = torch.nn.functional.cross_entropy(out.float(), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            torch.mps.synchronize()
            return (time.perf_counter() - t0) / steps * 1000
        go()
        runners[name] = go
    times = {n: [] for n in runners}
    for _ in range(rounds):
        for n, fn in runners.items():
            times[n].append(fn())
    p.stdin.close()
    p.wait()
    print(f"batch {BATCH}, {rounds} rounds x {steps} steps, 96x128 input, {sum(v.size for v in P.values())} parameters\n\n| runtime | median ms a step | fastest round | images a second (median) |\n|---|---|---|---|")
    for n, t in times.items():
        print(f"| {n} | {np.median(t):.1f} | {min(t):.1f} | {BATCH / np.median(t) * 1000:.0f} |")
        print(f"  rounds: {' '.join(f'{v:.1f}' for v in t)}", file=sys.stderr)


def profile():
    if busy():
        sys.exit("the GPU is in use: " + busy()[0])
    _, sess = write(12)
    errf = tempfile.mktemp()  # the profile is long: a pipe nobody reads would block the session
    with open(errf, "w") as ef:
        p = session_proc(sess, dict(os.environ, CANCHOGPU_PROFILE="2"), ef)
        run_steps(p, 1)
        mark = os.path.getsize(errf)
        run_steps(p, 5)
        p.stdin.close()
        p.wait()
    print(open(errf).read()[-6000:])


if __name__ == "__main__":
    {"launches": launches, "time": timing, "profile": profile}[sys.argv[1]]()
