#!/usr/bin/env python3
"""docs/resident.md section 12: a network trained with Adam inside a session, held to a binary64 numpy implementation of the same steps and to torch.optim.Adam.

    python3 tests/train.py build/cancho-gpu-session [build/cancho-gpu-metal-session]

1. The CPU session trains a 8-4-1 sigmoid network on 256 samples: the loss after 1, 10 and 30 steps and the parameters after 30 equal the float64 reference's to a float32 tolerance.
2. A second request continues the first: 15 steps and then 15 give exactly the loss of 30 at once.
3. (Metal session given) a 32-32-1 network on 4096 samples, 100 steps: the loss after 1, 10 and 100 steps equals the reference's to 1e-3 relative, and the CPU session's at the small size to 1e-4;
   the device trains in float32 and the reference in float64, so they part company later (after 1000 steps the loss is 0.00076 against 0.00108), and this does not claim more than 100.
3a. A mixed-precision network (f16 weights and activations, `mma` kernels, f32 accumulation): near the float64 reference to 2e-3 early, learning, and the Metal session equal to the CPU one to 2e-2.
3b. Minibatches (`batch` in mlp_session.write): the same checks on a table of 8 batches, through two laps.
4. (torch given) torch.optim.Adam in float32 on the same data and start gives the loss of the device after 100 steps to 1e-3 relative.
Needs numpy; without it the test says so and passes."""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "train"))
try:
    import numpy as np
    import mlp_session as M
except ImportError:
    print("skip: numpy is not installed")
    sys.exit(0)

bad = 0


def say(good, text):
    global bad
    print(f"{'ok  ' if good else 'FAIL'} {text}")
    bad += 0 if good else 1


def serve(binary, session, lines, extra=()):
    r = subprocess.run([binary, session, *extra], input=lines, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise SystemExit(f"{binary}: {r.stderr[:300]}")
    return [[float(v) for v in line.split()] for line in r.stdout.splitlines()]


def close(a, b, rel):
    return abs(a - b) <= rel * max(abs(b), 1e-12)


def main():
    cpu = os.path.abspath(sys.argv[1])
    metal = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 and os.path.exists(sys.argv[2]) else None
    d = tempfile.mkdtemp()
    r, k, hd, steps = 256, 8, 4, 30
    ref_losses, ref_params = M.reference(r, k, hd, steps, lr=0.01)
    session = M.write(os.path.join(d, "small"), r, k, hd, steps, lr=0.01, dump=True)
    out = serve(cpu, session, "1\n9\n20\n", ["--target", "nvidia-ada"])
    for line, at in zip(out, (1, 10, 30)):
        say(close(line[0], ref_losses[at - 1], 1e-4), f"CPU session, {at} steps: loss {line[0]:.6f}, float64 reference {ref_losses[at - 1]:.6f}")
    final = np.array(out[-1][1:])
    want = np.concatenate([p.reshape(-1) for p in ref_params])
    say(float(np.abs(final - want).max()) < 2e-4, f"CPU session, parameters after 30 steps: largest difference from the reference {float(np.abs(final - want).max()):.1e}")
    once = serve(cpu, session, "30\n", ["--target", "nvidia-ada"])
    twice = serve(cpu, session, "15\n15\n", ["--target", "nvidia-ada"])
    say(once[0] == twice[1], f"15 steps and then 15 give exactly what 30 steps at once give: {once[0][0]!r} and {twice[1][0]!r}")
    # minibatches: the data is a table of batches and step j trains on batch j mod 8, so the table wraps after 8 steps
    mb = M.write(os.path.join(d, "mb"), 1024, 16, 16, 40, lr=0.01, batch=128)
    mb_ref, _ = M.reference(1024, 16, 16, 40, lr=0.01, batch=128)
    marks = (1, 2, 3, 10, 40)
    out_mb = serve(cpu, mb, "1\n1\n1\n7\n30\n", ["--target", "nvidia-ada"])
    for line, at in zip(out_mb, marks):
        say(close(line[0], mb_ref[at - 1], 1e-4), f"CPU session, minibatches of 128 of 1024, {at} steps: loss {line[0]:.6f}, float64 reference {mb_ref[at - 1]:.6f}")
    if metal:
        out_dev = serve(metal, mb, "1\n1\n1\n7\n30\n")
        say(all(close(a[0], b[0], 1e-4) for a, b in zip(out_dev, out_mb)), f"Metal session on minibatches equals the CPU session's losses to 1e-4: {[round(v[0], 6) for v in out_dev]}")
    # a ReLU network: relu and step in the surface (docs/chx.md)
    rl = M.write(os.path.join(d, "relu"), 1024, 16, 16, 40, lr=0.01, act="relu")
    rl_ref, _ = M.reference(1024, 16, 16, 40, lr=0.01, act="relu")
    out_rl = serve(cpu, rl, "1\n9\n30\n", ["--target", "nvidia-ada"])
    for line, at in zip(out_rl, (1, 10, 40)):
        say(close(line[0], rl_ref[at - 1], 1e-4), f"CPU session, a ReLU network, {at} steps: loss {line[0]:.6f}, float64 reference {rl_ref[at - 1]:.6f}")
    say(rl_ref[39] < 0.6 * rl_ref[0], f"the ReLU network learns: loss {rl_ref[0]:.4f} -> {rl_ref[39]:.4f}")
    if metal:
        out_rl_dev = serve(metal, rl, "1\n9\n30\n")
        say(all(close(a[0], b[0], 1e-4) for a, b in zip(out_rl_dev, out_rl)), f"Metal session on the ReLU network equals the CPU session's losses to 1e-4: {[round(v[0], 6) for v in out_rl_dev]}")
    # a deeper network, minibatches, binary files: dense_fwd, out_fwd and hid_bwd against a float64 numpy implementation (docs/resident.md section 13)
    import deep_session as DS
    ddir = os.path.join(d, "deepdata")
    os.makedirs(ddir)
    rng = np.random.RandomState(1)
    Xd = rng.normal(size=(640, 16)).astype(np.float32)
    np.save(os.path.join(ddir, "X.npy"), Xd)
    np.save(os.path.join(ddir, "y.npy"), (Xd[:, 0] * Xd[:, 1] + Xd[:, 2] > 0).astype(np.float32))
    ds = DS.write(ddir, os.path.join(ddir, "s"), [32, 32], 64, 40, lr=0.01)
    dref, _ = DS.reference(ddir, [32, 32], 64, 40, lr=0.01)
    out_d = serve(cpu, ds, "1\n1\n8\n30\n", ["--target", "nvidia-ada"])
    for line, at in zip(out_d, (1, 2, 10, 40)):
        say(close(line[0], dref[at - 1], 1e-4), f"CPU session, a 16-32-32-1 ReLU network on minibatches of 64, {at} steps: loss {line[0]:.6f}, float64 reference {dref[at - 1]:.6f}")
    if metal:
        out_dd = serve(metal, ds, "1\n1\n8\n30\n")
        say(all(close(a[0], b[0], 1e-4) for a, b in zip(out_dd, out_d)), f"Metal session on the deeper network equals the CPU session's losses to 1e-4: {[round(v[0], 6) for v in out_dd]}")
    # mixed precision on the matrix unit (docs/resident.md section 14): f16 weights and activations, f32 accumulation and master weights
    import mp_session as MPS
    mdir = os.path.join(d, "mpdata")
    os.makedirs(mdir)
    rngm = np.random.RandomState(1)
    Xm = rngm.normal(size=(1280, 64)).astype(np.float32)
    np.save(os.path.join(mdir, "X.npy"), Xm)
    np.save(os.path.join(mdir, "y.npy"), (Xm[:, 0] * Xm[:, 1] + Xm[:, 2] > 0).astype(np.float32))
    ms = MPS.write(mdir, os.path.join(mdir, "s"), [64, 64], 64, 40, lr=0.01)
    mref, _ = MPS.reference(mdir, [64, 64], 64, 40, lr=0.01)
    out_m = serve(cpu, ms, "1\n1\n8\n30\n", ["--target", "apple-m-series"])
    for line, at in zip(out_m, (1, 2, 10)):
        say(close(line[0], mref[at - 1], 2e-3), f"CPU session, a 64-64-64-1 mixed-precision network, {at} steps: loss {line[0]:.6f}, float64 reference {mref[at - 1]:.6f} (f16 weights and activations)")
    say(out_m[3][0] < 0.6 * out_m[0][0], f"the mixed-precision network learns: loss {out_m[0][0]:.4f} -> {out_m[3][0]:.4f} after 40 steps")
    if metal:
        out_md = serve(metal, ms, "1\n1\n8\n30\n")
        say(all(close(a[0], b[0], 2e-2) for a, b in zip(out_md, out_m)), f"Metal session on the mixed-precision network equals the CPU session's losses to 2e-2: {[round(v[0], 6) for v in out_md]}")
    if metal:
        dev_small = serve(metal, session, "1\n9\n20\n")
        say(all(close(a[0], b[0], 1e-4) for a, b in zip(dev_small, out)), f"Metal session on the small network equals the CPU session's loss to 1e-4: {[round(v[0], 6) for v in dev_small]}")
        r, k, hd, steps = 4096, 32, 32, 100
        ref_losses, _ = M.reference(r, k, hd, steps, lr=0.01)
        big = M.write(os.path.join(d, "big"), r, k, hd, steps, lr=0.01)
        out = serve(metal, big, "1\n9\n90\n")
        for line, at in zip(out, (1, 10, 100)):
            say(close(line[0], ref_losses[at - 1], 1e-3), f"Metal session, {r} x {k} x {hd}, {at} steps: loss {line[0]:.6f}, float64 reference {ref_losses[at - 1]:.6f}")
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None:
            x, t = M.data(r, k)
            w1, b1, w2, b2 = M.init(k, hd)
            X, T = torch.tensor(x), torch.tensor(t)
            P = [torch.tensor(a, requires_grad=True) for a in (w1, b1, w2, b2)]
            opt = torch.optim.Adam(P, lr=0.01)
            for _ in range(steps):
                opt.zero_grad()
                e = torch.sigmoid(X @ P[0].T + P[1]) @ P[2] + P[3] - T
                loss = 0.5 * (e * e).sum() / r
                loss.backward()
                opt.step()
            # the session reports the loss of the last step before its update: the torch loss of the 100th step
            say(close(out[-1][0], float(loss.detach()), 1e-3), f"Metal session against torch.optim.Adam (float32), 100 steps: {out[-1][0]:.6f} and {float(loss.detach()):.6f}")
    sys.exit(1 if bad else 0)


main()
