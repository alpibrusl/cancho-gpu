#!/usr/bin/env python3
"""Train the same CNN on the same images with cancho-gpu (a Metal session) and torch (MPS float32 and float16 autocast; CPU with --only torch-cpu) and compare the loss curves, the test
accuracy and the time (docs/resident.md section 18). The model, its start (cnn_ref.init_params), the batches and AdamW are the same for every runtime: the images are shuffled in
`--laps` different orders and the training takes them in that order, lap after lap.

    python3 examples/cnn/compare_cnn.py [--epochs 20] [--train 20000] [--test 4000] [--batch 128] [--lr 2e-3] [--wd 0.01] [--laps 2]
        [--data synthetic | /path/to/cifar-10-batches-py] [--only cancho,torch-mps,torch-mps-amp]"""
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import cnn_data as D  # noqa: E402
import cnn_ref as R  # noqa: E402
import cnn_session as S  # noqa: E402

arg = lambda name, default: sys.argv[sys.argv.index(name) + 1] if name in sys.argv and sys.argv.index(name) + 1 < len(sys.argv) else default
EPOCHS = int(arg("--epochs", 20))
NTRAIN, NTEST, BATCH = int(arg("--train", 20000)), int(arg("--test", 4000)), int(arg("--batch", 128))
LR, WD, LAPS = float(arg("--lr", 2e-3)), float(arg("--wd", 0.01)), int(arg("--laps", 2))
DATA = arg("--data", "synthetic")
NOISE = float(arg("--noise", 1.5))
ONLY = arg("--only", "cancho,torch-mps,torch-mps-amp").split(",")
SESSION = os.environ.get("SESSION", os.path.join(ROOT, "build", "cancho-gpu-metal-session"))
DIR = arg("--dir", os.path.join(ROOT, "build", "cnn"))
SPE = NTRAIN // BATCH  # steps an epoch
STEPS = EPOCHS * SPE


def lr_at(step):
    """Linear warm-up for 3% of the steps, then cosine to 2% of the peak."""
    warm = max(1, STEPS * 3 // 100)
    if step <= warm:
        return LR * step / warm
    prog = (step - warm) / max(1, STEPS - warm)
    return LR * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))


def torch_model(c, params, running, device):
    import torch
    import torch.nn.functional as F
    P = {k: torch.tensor(np.asarray(v, dtype=np.float32), device=device, requires_grad=True) for k, v in params.items()}
    for k in P:
        if k.startswith("w_"):  # [co, 3, 3, ci] -> [co, ci, 3, 3]
            P[k] = P[k].detach().permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    st = {n: [torch.zeros(co, device=device), torch.ones(co, device=device)] for n, _, co in c.layers()}
    if running:
        for n in st:
            st[n] = [torch.tensor(running[n][0], dtype=torch.float32, device=device), torch.tensor(running[n][1], dtype=torch.float32, device=device)]

    def forward(x, train):
        def layer(name, a, res=None):
            z = F.conv2d(a, P["w_" + name], padding=1)
            y = F.batch_norm(z, st[name][0], st[name][1], P["g_" + name], P["b_" + name], train, 0.1, R.EPS)
            return F.relu(y if res is None else y + res)
        a = layer("stem", x)
        for s, nb in enumerate(c.blocks):
            if s > 0:
                a = layer(f"up{s}", F.avg_pool2d(a, 2))
            for b in range(nb):
                a = layer(f"s{s}b{b}b", layer(f"s{s}b{b}a", a), a)
        return a.mean((2, 3)) @ P["wfc"].T
    return P, st, forward


def accuracy(c, params, running, x_test, y_test, device="mps"):
    import torch
    P, st, fwd = torch_model(c, params, running, device)
    xt = torch.tensor(x_test, device=device).permute(0, 3, 1, 2).contiguous()
    yt = torch.tensor(y_test, device=device)
    right, loss = 0, 0.0
    with torch.no_grad():
        for i in range(0, len(xt), 500):
            lg = fwd(xt[i:i + 500], False)
            right += int((lg.argmax(1) == yt[i:i + 500]).sum())
            loss += float(torch.nn.functional.cross_entropy(lg, yt[i:i + 500], reduction="sum"))
    return right / len(xt), loss / len(xt)


def run_cancho(c, params, bx, by, x_test, y_test):
    curves = []
    # the time: a session that answers only the loss
    sess = S.write(os.path.join(DIR, "cancho-time"), c, bx, by, STEPS, lr=LR, wd=WD, params=params, lr_fn=lr_at)
    p = subprocess.Popen([SESSION, sess], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    p.stdin.write("1\n")
    p.stdin.flush()
    if not p.stdout.readline():
        raise SystemExit("cancho session: " + p.stderr.read()[:300])
    times = []
    for e in range(EPOCHS):
        k = SPE - 1 if e == 0 else SPE
        t0 = time.perf_counter()
        p.stdin.write(f"{k}\n")
        p.stdin.flush()
        line = p.stdout.readline()
        times.append((time.perf_counter() - t0) / k * 1000)
    p.stdin.close()
    p.wait()
    ms = float(np.median(times[1:] or times))
    # the curve: the same training with every parameter in the answer after each epoch
    sess = S.write(os.path.join(DIR, "cancho"), c, bx, by, STEPS, lr=LR, wd=WD, params=params, lr_fn=lr_at, dump=True)
    p = subprocess.Popen([SESSION, sess], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for e in range(EPOCHS):
        p.stdin.write(f"{SPE}\n")
        p.stdin.flush()
        nums = [float(v) for v in p.stdout.readline().split()]
        pr, rn = S.parse_dump(c, nums)
        acc, tl = accuracy(c, pr, rn, x_test, y_test)
        curves.append((e + 1, nums[0], tl, acc))
    p.stdin.close()
    p.wait()
    return ms, curves


def run_torch(c, params, bx, by, x_test, y_test, device, amp):
    import torch
    P, st, fwd = torch_model(c, params, None, device)
    decay = [P[k] for k in R.param_names(c)[0]]
    nodecay = [P[k] for k in R.param_names(c)[1]]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": nodecay, "weight_decay": 0.0}], lr=LR, betas=(S.B1, S.B2), eps=S.EPS)
    nb = len(bx)
    xs = torch.tensor(bx.astype(np.float32), device=device).permute(0, 1, 4, 2, 3).contiguous()  # [S, N, C, H, W]
    ys = torch.tensor(by, device=device)
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)
    curves, times, step = [], [], 0
    for e in range(EPOCHS):
        sync()
        t0 = time.perf_counter()
        for _ in range(SPE):
            step += 1
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step)
            opt.param_groups[0]["weight_decay"] = WD * LR / lr_at(step)  # the session's decay is lr_peak * wd a step, not scheduled
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                logits = fwd(xs[(step - 1) % nb], True)
            loss = torch.nn.functional.cross_entropy(logits.float(), ys[(step - 1) % nb])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sync()
        times.append((time.perf_counter() - t0) / SPE * 1000)
        pr = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in P.items()}
        for k in pr:
            if k.startswith("w_"):
                pr[k] = pr[k].transpose(0, 2, 3, 1)
        rn = {n: (v[0].cpu().numpy(), v[1].cpu().numpy()) for n, v in st.items()}
        acc, tl = accuracy(c, pr, rn, x_test, y_test)
        curves.append((e + 1, float(loss.detach()), tl, acc))
    return float(np.median(times[1:] or times)), curves


def main():
    c = R.Config(n=BATCH)
    if DATA == "synthetic":
        xtr, ytr = D.synthetic(NTRAIN, 1, noise=NOISE)
        xte, yte = D.synthetic(NTEST, 2, noise=NOISE)
        what = "synthetic textures"
    else:
        xtr, ytr, xte, yte = D.cifar(DATA)
        what = "CIFAR-10"
    params = R.init_params(c, 1)
    # laps of different shuffles, taken round and round
    perms = [np.random.RandomState(10 + k).permutation(len(xtr))[: SPE * BATCH].reshape(SPE, BATCH) for k in range(LAPS)]
    idx = np.concatenate(perms)
    bx, by = xtr[idx], ytr[idx]
    nparams = sum(int(np.prod(v.shape)) for v in params.values())
    print(f"CNN: widths {c.widths}, blocks {c.blocks}, {nparams / 1e6:.3f} M parameters; {what}: {len(xtr)} train, {len(xte)} test; batch {BATCH}, {EPOCHS} epochs ({STEPS} steps), AdamW {LR}, wd {WD}", flush=True)
    rows = []
    for name in ONLY:
        os.makedirs(DIR, exist_ok=True)
        if name == "cancho":
            ms, curve = run_cancho(c, params, bx, by, xte, yte)
        elif name.startswith("torch"):
            ms, curve = run_torch(c, params, bx, by, xte, yte, "cpu" if "cpu" in name else "mps", name.endswith("amp"))
        else:
            continue
        rows.append((name, ms, curve))
        print(f"{name}: {ms:.1f} ms a step ({BATCH / ms * 1000:.0f} images/s); (epoch, train loss of the last step, test loss, test accuracy): "
              + " ".join(f"({e}, {tl:.3f}, {vl:.3f}, {a:.4f})" for e, tl, vl, a in curve), flush=True)
    print("\n| runtime | ms a step | images a second | training time | test accuracy at the end | best test accuracy |\n|---|---|---|---|---|---|")
    for name, ms, curve in rows:
        print(f"| {name} | {ms:.1f} | {BATCH / ms * 1000:.0f} | {ms * STEPS / 1000:.0f} s | {curve[-1][3]:.4f} | {max(a for _, _, _, a in curve):.4f} |")


if __name__ == "__main__":
    main()
