#!/usr/bin/env python3
"""A decode loop on the device (cancho-rl design section 38): a request of N steps in which every step turns the previous ids into a one-hot (onehot_h), multiplies it by a table to
get the next logits (gemm_mma), samples (the seven samp_* kernels, docs/sampler.md) and writes the ids into slot j of a carried buffer (tok_acc), the ids of the step feeding the next
one by `carry`, the uniforms and the slot selectors being `table`s read a block a repetition. The host sends "N" and reads the N ids of every row at the end. The "model" is a bigram table
(the next logits are a row of a matrix chosen by the previous id), so the whole run has an exact numpy counterpart.

    python3 tests/carry_dev.py BINARY [BINARY ...]       # cancho-gpu-session and/or cancho-gpu-metal-session
    python3 tests/carry_dev.py --time BINARY             # ms a step

Held to: (1) the ids of 64 steps for 32 rows, from two requests (48 steps, then 16: a second request continues the tables and the carried state), equal the float64 inverse CDF run step by
step with the same uniforms; (2) the padding columns, whose logits are +100, are never drawn; (3) a fresh session gives the same ids (determinism)."""
import os, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHX = os.path.join(HERE, "chx")
bad = 0
MUT = os.environ.get("CARRY_MUTANT", "")     # "nocarry" or "shift": the test must fail (watched once; not used by the gate)
R, NQ, G, BN, INVT, PAD, S = 32, 24, 4, 24, 1.25, 100, 64
V = NQ * 64

def say(ok, text):
    global bad
    print(("ok   " if ok else "FAIL ") + text)
    bad += 0 if ok else 1

def build(d, seed=7):
    rng = np.random.default_rng(seed)
    Lm = rng.normal(size=(V, V)) * 2.5
    Lm[:, V - PAD:] = rng.normal(size=(V, PAD)) * 2.5 + 100.0
    Lm16 = Lm.astype(np.float16)
    f32 = lambda name, a: np.asarray(a, dtype="<f4").tofile(os.path.join(d, name))
    Lm16.T.astype("<f2").tofile(os.path.join(d, "wT"))               # w[n, k] = Lm[k, n]: the product of a one-hot at k is column k of w
    ids0 = rng.integers(1, V - PAD + 1, size=R)
    f32("ids0", ids0); f32("jrow", np.arange(1, V + 1))
    mk = np.zeros(V); mk[V - PAD:] = -1e9; f32("mk", mk)
    f32("tri", np.tril(np.ones((NQ, NQ)))); f32("tri64", np.tril(np.ones((64, 64))))
    f32("iota", np.arange(NQ)); f32("base", np.arange(NQ) * 64)
    u = rng.uniform(1e-6, 1 - 1e-6, size=(S, R)); f32("utab", u)
    f32("seltab", np.roll(np.eye(S), 1, axis=1) if MUT == "shift" else np.eye(S)); f32("keeptab", 1 - np.eye(S)); f32("acc0", np.zeros((R, S)))
    src = open(os.path.join(CHX, "..", "chx", "ported", "gemm_mma.chx")).read()
    def sched(ln):
        if ln.startswith("schedule gemm_mma for apple-m-series"):
            return "schedule gemm_mma for apple-m-series { threads 128 warps 1 4 bm %d bn 64 bk 32 }" % R
        if ln.startswith("schedule gemm_mma for nvidia-ada"):
            return "schedule gemm_mma for nvidia-ada { threads 128 warps 2 2 bm %d bn 64 bk 64 }" % R
        return ln
    open(os.path.join(d, "gemm_s.chx"), "w").write("\n".join(sched(ln) for ln in src.split("\n")))
    p = lambda name: os.path.join(CHX, name + ".chx")
    L0 = [f"program oh {os.path.normpath(os.path.join(HERE, os.pardir, 'examples', 'qwen', 'onehot_h.chx'))} r={R} c={V} bc=128", f"resident oh.ids {d}/ids0 f32", f"resident oh.jrow {d}/jrow f32",
         f"program mm {d}/gemm_s.chx m={R} n={V} k={V}", "link oh.y -> mm.a", f"resident mm.w {d}/wT f16",
         f"program mx {p('samp_max')} r={R} nq={NQ} G={G}", "link mm.y -> mx.l", f"resident mx.mk {d}/mk f32",
         f"program gm {p('samp_gmax')} r={R} nq={NQ}", "link mx.cm -> gm.cm",
         f"program sm {p('samp_sum')} r={R} nq={NQ} G={G} invt={INVT}", "link mm.y -> sm.l", f"resident sm.mk {d}/mk f32", "link gm.gmb -> sm.gmb",
         f"program cu {p('samp_cum')} r={R} nq={NQ} bn={BN}", "link sm.cs -> cu.cs", f"resident cu.tri {d}/tri f32",
         f"program pk {p('samp_pick')} r={R} nq={NQ}", "link cu.cum -> pk.cum", "link sm.cs -> pk.cs", f"resident pk.iota {d}/iota f32", f"table pk.u {d}/utab {S} f32",
         f"program lc {p('samp_local')} r={R} nq={NQ} G={G} invt={INVT}", "link mm.y -> lc.l", f"resident lc.mk {d}/mk f32", "link gm.gmb -> lc.gmb", "link pk.tl -> lc.tl", f"resident lc.tri {d}/tri64 f32",
         f"program tk {p('samp_tok')} r={R} nq={NQ}", "link pk.sel -> tk.sel", "link lc.cnt -> tk.cnt", f"resident tk.base {d}/base f32",
         f"program ac {p('tok_acc')} r={R} s={S}", f"resident ac.acc {d}/acc0 f32", "link tk.tok -> ac.ids", f"table ac.sel {d}/seltab {S} f32", f"table ac.keep {d}/keeptab {S} f32",
         ]
    L = L0 + ([] if MUT == "nocarry" else ["carry tk.tok -> oh.ids"]) + ["carry ac.y -> ac.acc", "steps", "answer ac.y"]
    open(os.path.join(d, "session"), "w").write("\n".join(L) + "\n")
    return os.path.join(d, "session"), Lm16.astype(np.float64), ids0, u

def reference(Lm, ids0, u, n):
    """The ids of n steps by the float64 inverse CDF, step by step, and the smallest relative distance of a uniform to a boundary of its cumulative distribution."""
    prev = ids0.copy(); out = np.zeros((R, n), dtype=np.int64); near = np.inf
    for j in range(n):
        for r in range(R):
            z = Lm[prev[r] - 1] * INVT; z = z.copy(); z[V - PAD:] = -np.inf
            p = np.exp(z - z.max()); p /= p.sum(); c = np.cumsum(p)
            out[r, j] = int(np.searchsorted(c, u[j, r])) + 1
            near = min(near, float(np.min(np.abs(c - u[j, r])) / u[j, r]))
        prev = out[:, j].copy()
    return out, near

class Proc:
    def __init__(self, binary, session):
        self.p = subprocess.Popen([binary, session], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    def ask(self, n):
        self.p.stdin.write(f"{n}\n"); self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            self.p.wait()
            raise SystemExit(f"session: exit {self.p.returncode} " + self.p.stderr.read()[:600])
        return np.array([int(round(float(x))) for x in line.split()]).reshape(R, S)
    def close(self):
        self.p.stdin.close(); self.p.wait()

def run(binary):
    name = os.path.basename(binary)
    d = tempfile.mkdtemp(prefix="carry_")
    sess, Lm, ids0, u = build(d)
    want, near = reference(Lm, ids0, u, S)
    pr = Proc(binary, sess)
    a = pr.ask(48)
    say(np.array_equal(a[:, :48], want[:, :48]), f"{name}: one request of 48 steps gives the 48 ids of each of {R} rows, equal to the float64 inverse CDF run step by step (smallest relative distance of a uniform to a boundary {near:.1e})")
    b = pr.ask(16)
    say(np.array_equal(b, want), f"{name}: a second request of 16 steps continues the tables and the carried state: all {S} slots equal the reference")
    say(int(b.max()) <= V - PAD and int(b.min()) >= 1, f"{name}: no padding column drawn, all ids in 1..{V - PAD}")
    pr.close()
    pr2 = Proc(binary, sess)
    c = pr2.ask(S)
    pr2.close()
    say(np.array_equal(c, want) and np.array_equal(c, b), f"{name}: a fresh session with one request of {S} steps gives the same ids")

def timing(binary):
    d = tempfile.mkdtemp(prefix="carry_time_")
    sess, *_ = build(d)
    pr = Proc(binary, sess)
    pr.ask(S)
    runs = []
    for _ in range(3):
        t0 = time.time(); pr.ask(S); runs.append((time.time() - t0) / S * 1000)
    pr.close()
    print(f"{os.path.basename(binary)}: {sorted(runs)[1]:.2f} ms a step of the toy loop (14 programs a step; runs {', '.join(f'{x:.2f}' for x in runs)})")

if sys.argv[1] == "--time":
    for b in sys.argv[2:]:
        timing(b)
    sys.exit(0)
for b in sys.argv[1:]:
    run(b)
print("bad" if bad else "all ok")
sys.exit(1 if bad else 0)
