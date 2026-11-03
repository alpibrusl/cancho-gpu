#!/usr/bin/env python3
"""docs/roadmap-models.md M10: the gated delta rule's specification (examples/qwen35/gdn_ref.py), its backward, the cost model (m10_cost.py) and, with torch and
transformers installed, the specification against autograd through HF's own code in binary64 (examples/qwen35/hf_check.py).

    python3 tests/gdn.py

Needs numpy for the specification (the cost model is pure Python and always runs); the torch part skips without torch/transformers. Every check that
can fail has a mutant (a deliberate fault of the specification or the model) that must make it fail."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# tests/kernels.py would shadow the `kernels` package that transformers imports
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, os.path.join(ROOT, "examples", "qwen35"))
bad = 0


def say(ok, text):
    global bad
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    bad += 0 if ok else 1


def catches(name, fault_error, bound):
    """A mutant is caught when its error exceeds the bound that the specification passes."""
    say(fault_error > bound, f"mutant {name}: error {fault_error:.2e} > {bound:.0e}")


def spec_checks():
    import numpy as np
    import gdn_ref as G

    rng = np.random.RandomState(0)
    H, T, dk, dv = 3, 150, 8, 6
    q = rng.normal(size=(H, T, dk)); k = rng.normal(size=(H, T, dk)); v = rng.normal(size=(H, T, dv))
    q /= np.linalg.norm(q, axis=-1, keepdims=True); k /= np.linalg.norm(k, axis=-1, keepdims=True)
    g = -rng.uniform(0.01, 0.5, size=(H, T)); beta = 1 / (1 + np.exp(-rng.normal(size=(H, T))))
    o, S = G.recurrent(q, k, v, g, beta)
    worst = 0.0
    for solve in ("loop", "inv"):
        for C in (64, 16, 7):
            o2, S2, _ = G.chunk_fwd(q, k, v, g, beta, C, solve)
            worst = max(worst, abs(o - o2).max(), abs(S - S2).max())
    say(worst < 1e-12, f"chunked form equals the recurrence at C = 64, 16, 7 (T = 150: padded), forward substitution and inverse: {worst:.1e}")
    for mut in ("no_decay", "solve_transposed"):
        o2, _, _ = G.chunk_fwd(q, k, v, g, beta, 64, "loop", mut=mut)
        catches(f"chunk_fwd {mut}", abs(o - o2).max(), 1e-6)

    # the backward against finite differences
    def fd(mut=None):
        r = np.random.RandomState(1)
        H, T, dk, dv = 2, 21, 5, 4
        a = dict(q=r.normal(size=(H, T, dk)), k=r.normal(size=(H, T, dk)), v=r.normal(size=(H, T, dv)), g=-r.uniform(0.05, 0.5, size=(H, T)),
                 beta=1 / (1 + np.exp(-r.normal(size=(H, T)))), S0=r.normal(size=(H, dk, dv)))
        W, WS = r.normal(size=(H, T, dv)), r.normal(size=(H, dk, dv))

        def f(a):
            o, S, c = G.chunk_fwd(a["q"], a["k"], a["v"], a["g"], a["beta"], 8, "inv", S0=a["S0"])
            return (o * W).sum() + (S * WS).sum(), c
        _, c = f(a)
        gr = G.chunk_bwd(c, W, WS, mut=mut)
        err = 0.0
        for key, name in (("q", "dq"), ("k", "dk"), ("v", "dv"), ("g", "dg"), ("beta", "dbeta"), ("S0", "dS0")):
            x = a[key]
            # a few coordinates only: the full sweep is examples/qwen35/hf_check.py's job
            for idx in [tuple(int(r.randint(s)) for s in x.shape) for _ in range(6)]:
                h = 1e-6
                xp, xm = {**a, key: x.copy()}, {**a, key: x.copy()}
                xp[key][idx] += h
                xm[key][idx] -= h
                num = (f(xp)[0] - f(xm)[0]) / (2 * h)
                err = max(err, abs(num - gr[name][idx]) / max(1.0, abs(num)))
        return err
    e = fd()
    say(e < 1e-6, f"chunk_bwd (dq dk dv dg dbeta dS0, T = 21 with a padded chunk of 8) against finite differences: {e:.1e}")
    for mut in ("dls_unmasked", "gate_sign"):
        catches(f"chunk_bwd {mut}", fd(mut), 1e-4)

    # a whole layer
    def fd_layer(mut=None):
        r = np.random.RandomState(2)
        p = G.make_params(r, hidden=12, nk=2, nv=4, dk=5, dv=3)
        x = r.normal(size=(21, 12))
        W = r.normal(size=(21, 12))
        f = lambda: (G.layer_fwd(p, x, 8, "inv")[0] * W).sum()
        _, s = G.layer_fwd(p, x, 8, "inv", mut=mut)
        dx, gr = G.layer_bwd(p, s, W, mut=mut)
        err = 0.0
        for name, arr, ana in [("x", x, dx)] + [(n, p[n], gr[n]) for n in ("Wqkv", "Wz", "Wb", "Wa", "conv", "A_log", "dt_bias", "norm", "Wout")]:
            for idx in [tuple(int(r.randint(s_)) for s_ in arr.shape) for _ in range(5)]:
                o_ = arr[idx]
                arr[idx] = o_ + 1e-6
                fp = f()
                arr[idx] = o_ - 1e-6
                fm = f()
                arr[idx] = o_
                num = (fp - fm) / 2e-6
                err = max(err, abs(num - ana[idx]) / max(1.0, abs(num)))
        return err
    e = fd_layer()
    say(e < 1e-6, f"layer_bwd (conv, silu, l2norm, gates, rule, gated norm, projections: dx and 9 parameter gradients) against finite differences: {e:.1e}")
    catches("layer_bwd gate_sign", fd_layer("gate_sign"), 1e-4)

    # the repeated-squaring solve is not usable: key cosine about 0.5, beta 0.9
    r = np.random.RandomState(5)
    Hh, C = 2, 64
    kk = 1.0 * r.normal(size=(Hh, 1, 128)) + r.normal(size=(Hh, C, 128))
    kk /= np.linalg.norm(kk, axis=-1, keepdims=True)
    Ls = G.tril((0.9 * kk) @ kk.swapaxes(-1, -2), -1)
    ex = np.linalg.inv(np.eye(C) + Ls)
    L32 = Ls.astype(np.float32)
    I32 = np.eye(C, dtype=np.float32)
    out, p_, s_ = I32 - L32, L32, 1
    while 2**s_ < C:
        p_ = p_ @ p_
        out = out @ (I32 + p_)
        s_ += 1
    a = -L32.copy()
    for i in range(1, C):
        a[:, i, :i] = a[:, i, :i] + (a[:, i, :i, None] * a[:, :i, :i]).sum(-2)
    fs = a + I32
    e_n, e_f = abs(out - ex).max(), abs(fs - ex).max()
    say(e_n > 1.0 and e_f < 1e-5, f"float32 solve at key cosine ~0.5: repeated squaring is off by {e_n:.1e} (powers of Ls reach {abs(p_).max():.0e}); forward substitution by {e_f:.1e}")

    # mixed precision: f16 operands, f32 accumulation, as the launches of examples/qwen35/probes/ run it
    worst = 0.0
    for corr in (0.0, 2.0):
        r = np.random.RandomState(7)
        base = r.normal(size=(2, 1, 128))
        k_ = corr * base + r.normal(size=(2, 256, 128)); k_ /= np.linalg.norm(k_, axis=-1, keepdims=True)
        q_ = r.normal(size=(2, 256, 128)); q_ /= np.linalg.norm(q_, axis=-1, keepdims=True)
        v_ = r.normal(size=(2, 256, 128))
        g_ = -r.uniform(0, 0.04, size=(2, 256)); b_ = 1 / (1 + np.exp(-r.normal(size=(2, 256))))
        o_ref, _, _ = G.chunk_fwd(q_, k_, v_, g_, b_, 64, "inv")
        o_mp = G.chunk_fwd_mp(q_, k_, v_, g_, b_)
        worst = max(worst, float(np.sqrt(((o_mp - o_ref) ** 2).mean() / (o_ref**2).mean())))
        if corr == 2.0:
            o_bad = G.chunk_fwd_mp(q_, k_, v_, g_, b_, solve="neumann")
            bad_err = float(np.sqrt(np.nanmean((o_bad - o_ref) ** 2) / (o_ref**2).mean())) if np.isfinite(o_bad).all() else float("inf")
    say(worst < 3e-3, f"f16-operand forward (solve by forward substitution) is within {worst:.1e} relative rms of binary64 at key correlation 0 and 2")
    catches("f16 forward with the squaring solve", bad_err, 3e-3)


def hf_checks():
    try:
        import torch  # noqa: F401
        import transformers.models.qwen3_5  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"skip hf_check: {type(e).__name__}: torch with transformers >= 5.2 is not installed")
        return
    import hf_check
    hf_check.bad = 0
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = hf_check.main()
    lines = [l for l in buf.getvalue().splitlines() if l.startswith(("ok", "FAIL"))]
    say(rc == 0 and len(lines) == 18, f"gdn_ref against torch autograd through HF's qwen3_5 code in binary64: {len(lines)} comparisons, worst relative error "
        f"{max(float(l.split(':')[-1]) for l in lines if l.split(':')[-1].strip()[0].isdigit()):.1e}")
    if rc:
        print(buf.getvalue())
    # a mutant of the specification (the sign of the gate gradients' column term) must fail the comparison with autograd
    hf_check.bad = 0
    orig = hf_check.G.chunk_bwd
    hf_check.G.chunk_bwd = lambda cache, do, dS=None, mut=None: orig(cache, do, dS, mut="gate_sign")
    with contextlib.redirect_stdout(io.StringIO()):
        rc2 = hf_check.main()
    hf_check.G.chunk_bwd = orig
    say(rc2 != 0, "mutant (gate gradient sign) fails the comparison with autograd")


def cost_checks():
    import m10_cost as M

    c = M.qwen38_27b()
    p = M.params(c)
    # the weight bytes of Ollama's manifest (qwen3.8:27b-mlx, one layer's tensor, header included, bytes); a tensor is n * 0.5625 + a header of about 450 B
    manifest = {"in_proj_qkv": 29491660, "in_proj_z": 17695172, "out_proj": 17695164, "gate": 50135468, "down": 50135460}
    dims = {"in_proj_qkv": 10240 * 5120, "in_proj_z": 6144 * 5120, "out_proj": 5120 * 6144, "gate": 17408 * 5120, "down": 5120 * 17408}

    def worst_gap(bytes_per_weight):
        return max(abs(manifest[k] - dims[k] * bytes_per_weight) / manifest[k] for k in manifest)
    say(worst_gap(M.NVFP4_BYTES) < 1e-4, f"4-bit bytes a tensor equal the manifest's (0.5625 B a weight): worst {worst_gap(M.NVFP4_BYTES):.1e}")
    catches("4 bits without scales (0.5 B a weight)", worst_gap(0.5), 1e-4)
    say(abs(p["total"] / 1e9 - 26.893) < 0.001 and p["gdn"] + p["full"] == 48 * 52428800 + 48 * 31457280 + 48 * 491520 + 48 * 31457280 + 16 * 104857600,
        f"parameters: {p['total']/1e9:.3f} B (blocks {p['blocks']/1e9:.3f}, embedding and head {p['embed']/1e9:.3f} each)")
    cal = M.calibrate_7b()
    err = max(abs(m - g) / g for _, m, g in cal)
    say(err < 0.02, "the accounting reproduces the 7B run's footprint at 4 x 512 and 8 x 512 (21.65, 34.63 GB): " + ", ".join(f"{m:.2f} vs {g:.2f}" for _, m, g in cal))
    bad_cal = M.calibrate_7b(mut="no_logits")
    catches("accounting without the logits", max(abs(m - g) / g for _, m, g in bad_cal), 0.02)
    # monotone in the batch and smaller with recompute
    m1, m4 = sum(M.memory(c, 1).values()), sum(M.memory(c, 4).values())
    r1, r4 = sum(M.memory(c, 1, recompute=True).values()), sum(M.memory(c, 4, recompute=True).values())
    say(m1 < m4 and r4 < m4 and r1 < r4, f"27B memory grows with the batch ({m1:.1f} to {m4:.1f} GB) and recompute shrinks it ({r1:.1f} to {r4:.1f} GB)")
    t = M.step_seconds(c, 4)
    base7 = M.step_seconds(M.qwen25_7b(), 4)
    say(5.0 < base7["total_s"] < 6.0, f"the same time model on the 7B gives {base7['total_s']:.2f} s a 4 x 512 step (measured 5.34)")
    say(t["gdn_core_s"] > 0 and t["total_s"] > t["matmul_s"], "27B step: gated-delta core and dequantisation add to the matrix products")


def main():
    cost_checks()
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("skip specification checks: numpy is not installed")
        return
    spec_checks()
    hf_checks()


if __name__ == "__main__":
    main()
    sys.exit(1 if bad else 0)
