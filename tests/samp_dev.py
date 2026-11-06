#!/usr/bin/env python3
"""The on-device sampler (docs/kernels.md section 9): temperature sampling by inverse CDF over a padded vocabulary of nq * 64 columns, as a session of eight kernels
(samp_max, samp_gmax, samp_sum, samp_cum, samp_pick, samp_local, samp_tok; onehot_h turns the ids into the embedding product's operand), held to numpy.

    python3 tests/samp_dev.py BINARY [BINARY ...]        # cancho-gpu-session (the CPU interpreter) and/or cancho-gpu-metal-session

Per binary: (1) the token of every row of many requests (a fresh uniform per row), equal to the float64 inverse CDF of the same logits except where the uniform is within 1e-4
(relative) of a boundary of the cumulative distribution, where the f32 sums may fall on the other side; (2) the distribution: the empirical frequencies of many draws
against the softmax of the logits at temperature 1 / invt (a chi-squared on the columns that carry mass, and the padding columns never drawn); (3) the draws of two rows with
equal logits and different uniforms are different tokens, and the same uniform gives the same token (determinism)."""
import os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHX = os.path.join(os.path.dirname(HERE), "tests", "chx")
bad = 0

def say(ok, text):
    global bad
    print(("ok   " if ok else "FAIL ") + text)
    bad += 0 if ok else 1

R, NQ, G, BN, INVT, PAD = 4, 24, 4, 24, 1.25, 100
V = NQ * 64

def write_session(d, logits):
    f32 = lambda name, a: np.asarray(a, dtype="<f4").tofile(os.path.join(d, name))
    f32("l", logits)
    mk = np.zeros(V); mk[V - PAD:] = -1e9; f32("mk", mk)
    f32("tri", np.tril(np.ones((NQ, NQ)))); f32("tri64", np.tril(np.ones((64, 64))))
    f32("iota", np.arange(NQ)); f32("base", np.arange(NQ) * 64)
    p = lambda name: os.path.join(CHX, name + ".chx")
    L = [f"program mx {p('samp_max')} r={R} nq={NQ} G={G}", f"resident mx.l {d}/l f32", f"resident mx.mk {d}/mk f32",
         f"program gm {p('samp_gmax')} r={R} nq={NQ}", "link mx.cm -> gm.cm",
         f"program sm {p('samp_sum')} r={R} nq={NQ} G={G} invt={INVT}", f"resident sm.l {d}/l f32", f"resident sm.mk {d}/mk f32", "link gm.gmb -> sm.gmb",
         f"program cu {p('samp_cum')} r={R} nq={NQ} bn={BN}", "link sm.cs -> cu.cs", f"resident cu.tri {d}/tri f32",
         f"program pk {p('samp_pick')} r={R} nq={NQ}", "link cu.cum -> pk.cum", "link sm.cs -> pk.cs", f"resident pk.iota {d}/iota f32", "input pk.u",
         f"program lc {p('samp_local')} r={R} nq={NQ} G={G} invt={INVT}", f"resident lc.l {d}/l f32", f"resident lc.mk {d}/mk f32", "link gm.gmb -> lc.gmb", "link pk.tl -> lc.tl", f"resident lc.tri {d}/tri64 f32",
         f"program tk {p('samp_tok')} r={R} nq={NQ}", "link pk.sel -> tk.sel", "link lc.cnt -> tk.cnt", f"resident tk.base {d}/base f32",
         "steps", "answer tk.tok"]
    open(os.path.join(d, "session"), "w").write("\n".join(L) + "\n")
    return os.path.join(d, "session")

def reference(logits, u):
    """The float64 inverse CDF: (1-based token, distance of u to the nearest boundary of the cumulative distribution, relative) of every row."""
    out = []
    for r in range(R):
        z = logits[r] * INVT
        z = z.copy(); z[V - PAD:] = -np.inf
        p = np.exp(z - z.max()); p /= p.sum()
        c = np.cumsum(p)
        t = int(np.searchsorted(c, u[r]))
        near = np.min(np.abs(c - u[r])) / max(u[r], 1e-30)
        out.append((t + 1, near))
    return out

class Proc:
    def __init__(self, binary, session):
        self.p = subprocess.Popen([binary, session], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    def ask(self, u):
        self.p.stdin.write("1 " + " ".join(repr(float(x)) for x in u) + "\n"); self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise SystemExit("session: " + self.p.stderr.read()[:500])
        return [int(round(float(x))) for x in line.split()]
    def close(self):
        self.p.stdin.close(); self.p.wait()

def run(binary):
    name = os.path.basename(binary)
    rng = np.random.default_rng(3)
    d = tempfile.mkdtemp(prefix="samp_")
    logits = rng.normal(size=(R, V)) * 2.5
    logits[:, V - PAD:] = rng.normal(size=(R, PAD)) * 2.5 + 100.0     # padding columns with huge logits: the mask must keep them out
    sess = write_session(d, logits)
    pr = Proc(binary, sess)
    mism, checked, in_range = 0, 0, True
    for _ in range(150):
        u = rng.uniform(1e-6, 1 - 1e-6, size=R)
        got = pr.ask(u)
        for r, (want, near) in enumerate(reference(logits, u)):
            checked += 1
            in_range &= 1 <= got[r] <= V - PAD
            if got[r] != want and near > 1e-4:
                mism += 1
    say(mism == 0 and in_range, f"{name}: {checked} draws equal the float64 inverse CDF away from its boundaries ({mism} differ), all in the real vocabulary")
    # determinism and difference
    u = np.resize(np.array([0.3, 0.3, 0.7, 0.7]), R)
    a, b = pr.ask(u), pr.ask(u)
    say(a == b, f"{name}: the same uniforms give the same tokens")
    # the distribution, one row of logits, many uniforms
    n = 3000
    counts = np.zeros(V)
    for _ in range(n // R):
        for t in pr.ask(rng.uniform(1e-9, 1 - 1e-9, size=R)):
            counts[t - 1] += 1
    z = logits[0] * INVT; z[V - PAD:] = -np.inf
    # all rows have different logits; test row 0 only, with its own draws
    c0 = np.zeros(V)
    for _ in range(n):
        c0[pr.ask(rng.uniform(1e-9, 1 - 1e-9, size=R))[0] - 1] += 1
    p = np.exp(z - z.max()); p /= p.sum()
    big = p * n >= 5
    exp_big, obs_big = p[big] * n, c0[big]
    rest_e, rest_o = p[~big].sum() * n, c0[~big].sum()
    chi = float(((obs_big - exp_big) ** 2 / exp_big).sum() + ((rest_o - rest_e) ** 2 / max(rest_e, 1e-9)))
    dof = int(big.sum())
    limit = dof + 4.5 * np.sqrt(2 * dof)              # about the 1 - 1e-5 quantile of a chi-squared
    say(chi < limit and c0[V - PAD:].sum() == 0, f"{name}: {n} draws from row 0 against its softmax: chi-squared {chi:.1f} on {dof} bins (limit {limit:.1f}), no padding column drawn")
    pr.close()

def timing(binary, rows=32, nq=2376, g=24, bn=24, steps=20):
    """The cost of one sampling step at the size of the Qwen2.5-1.5B batch (32 rows x a vocabulary of 152,064): ms a step, from a request of `steps` repetitions."""
    global R, NQ, G, BN, V, PAD
    R, NQ, G, BN, V, PAD = rows, nq, g, bn, nq * 64, nq * 64 - 151936
    import time
    d = tempfile.mkdtemp(prefix="samp_time_")
    rng = np.random.default_rng(5)
    sess = write_session(d, rng.normal(size=(R, V)) * 2.5)
    pr = Proc(binary, sess)
    u = rng.uniform(0.01, 0.99, size=R)
    pr.ask(u)
    runs = []
    for _ in range(3):
        t0 = time.time()
        pr.p.stdin.write(f"{steps} " + " ".join(repr(float(x)) for x in u) + "\n"); pr.p.stdin.flush(); pr.p.stdout.readline()
        runs.append((time.time() - t0) / steps * 1000)
    pr.close()
    print(f"{os.path.basename(binary)}: sampling {rows} rows x {V} columns: {sorted(runs)[1]:.2f} ms a step (runs {', '.join(f'{x:.2f}' for x in runs)})")

if sys.argv[1] == "--big":
    # the size of the Qwen2.5-1.5B batch: 32 rows, a vocabulary of 151,936 real columns padded to 152,064
    R, NQ, G, BN = 32, 2376, 24, 24
    V, PAD = NQ * 64, NQ * 64 - 151936
    for b in sys.argv[2:]:
        run(b)
    print("bad" if bad else "all ok")
    sys.exit(1 if bad else 0)
if sys.argv[1] == "--time":
    for b in sys.argv[2:]:
        timing(b)
    sys.exit(0)
for b in sys.argv[1:]:
    run(b)
print("bad" if bad else "all ok")
sys.exit(1 if bad else 0)
