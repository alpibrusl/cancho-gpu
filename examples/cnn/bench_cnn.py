#!/usr/bin/env python3
"""The time of a training step of the CNN (docs/resident.md section 18), cancho-gpu against torch, in interleaved rounds so that a machine that is busy for a while (another process on
the GPU) hits all the runtimes alike: each round times STEPS steps of every runtime, and the table gives the median and the fastest round. No evaluation, no curves: compare_cnn.py has those.

    python3 examples/cnn/bench_cnn.py [--rounds 7] [--steps 60] [--batch 128] [--only cancho,torch-mps,torch-mps-amp]"""
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cnn_data as D  # noqa: E402
import cnn_ref as R  # noqa: E402
import cnn_session as S  # noqa: E402
import compare_cnn as C  # noqa: E402

arg = C.arg
ROUNDS, STEPS, BATCH = int(arg("--rounds", 7)), int(arg("--steps", 60)), int(arg("--batch", 128))
ONLY = arg("--only", "cancho,torch-mps,torch-mps-amp").split(",")


def main():
    import torch
    c = R.Config(n=BATCH)
    x, y = D.synthetic(BATCH * 8, 1)
    bx, by = D.batches(x, y, BATCH, 8, 1)
    params = R.init_params(c, 1)
    total = ROUNDS * STEPS + 1
    sess = S.write(os.path.join(C.DIR, "bench"), c, bx, by, total, lr=C.LR, wd=C.WD, params=params)
    p = subprocess.Popen([C.SESSION, sess], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    p.stdin.write("1\n")
    p.stdin.flush()
    p.stdout.readline()
    runners = {}
    for name in ONLY:
        if name == "cancho":
            def step_cancho():
                t0 = time.perf_counter()
                p.stdin.write(f"{STEPS}\n")
                p.stdin.flush()
                p.stdout.readline()
                return (time.perf_counter() - t0) / STEPS * 1000
            runners[name] = step_cancho
        elif name.startswith("torch"):
            device, amp = ("cpu" if "cpu" in name else "mps"), name.endswith("amp")
            P, st, fwd = C.torch_model(c, params, None, device)
            opt = torch.optim.AdamW([{"params": [P[k] for k in R.param_names(c)[0]], "weight_decay": C.WD}, {"params": [P[k] for k in R.param_names(c)[1]], "weight_decay": 0.0}], lr=C.LR, betas=(S.B1, S.B2), eps=S.EPS)
            xs = torch.tensor(bx.astype(np.float32), device=device).permute(0, 1, 4, 2, 3).contiguous()
            ys = torch.tensor(by, device=device)
            sync = torch.mps.synchronize if device == "mps" else (lambda: None)

            def step_torch(fwd=fwd, opt=opt, xs=xs, ys=ys, sync=sync, device=device, amp=amp):
                sync()
                t0 = time.perf_counter()
                for i in range(STEPS):
                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                        logits = fwd(xs[i % 8], True)
                    loss = torch.nn.functional.cross_entropy(logits.float(), ys[i % 8])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                sync()
                return (time.perf_counter() - t0) / STEPS * 1000
            step_torch()  # warm-up
            runners[name] = step_torch
    times = {n: [] for n in runners}
    for r in range(ROUNDS):
        for n, fn in runners.items():
            times[n].append(fn())
    p.stdin.close()
    p.wait()
    print(f"rounds {ROUNDS} x {STEPS} steps, batch {BATCH}\n\n| runtime | median ms a step | fastest round | images a second (median) |\n|---|---|---|---|")
    for n, t in times.items():
        print(f"| {n} | {np.median(t):.1f} | {min(t):.1f} | {BATCH / np.median(t) * 1000:.0f} |")
        print(f"  rounds: {' '.join(f'{v:.1f}' for v in t)}", file=sys.stderr)


main()
