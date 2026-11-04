#!/usr/bin/env python3
"""docs/weight-space.md section 4 and 7: the harness for weights written by a model (examples/weightspace/), the review checklist, the sandbox and the gradient arm.

    python3 tests/weightspace.py [build/cancho-gpu-session]

Without the session binary the forward path and the trainer are not tested (the rest is).

  1. The reference forward kinematics (task.fk) equals, to 1e-12 m, 25 poses of cancho-robot's src/kinematics.cho recorded in tests/weightspace/fk_fixture.json (examples/weightspace/check_fk.py
     compares 2,500 with the live program: 2e-16 m); the band and the held-in region are disjoint, the held-out sets have 2,000 poses each and are reproducible.
  2. The review checklist catches a planted violator that fits a tensor by `lstsq` (written through the harness, read back from its log), reports a borderline case instead of dropping it, passes a closed
     form, and catches a hill-climbing loop; and each rule is what catches its planted case (with the rule disabled the plant passes: the test can fail).
  3. `forward` beyond the budget is refused and does not run; the count survives reopening the run; budget 0 refuses the first call; n_poses out of range is refused; the answer holds one scalar and
     nothing from which the poses or per-pose errors could be read; the log has every call.
  4. The sandbox refuses a read and a write outside its scratch directory, the harness's own log, a network connection (to a listener the test opens: nothing arrives), a subprocess, a blocked import, and
     stops an endless loop; a good program still runs. Two things it does NOT stop are pinned as known behaviour (docstring of sandbox.py): `os.path.exists` on an outside path, and reading the Python installation.
  5. (session given) `forward` through the CPU session equals the binary64 forward to 1e-6 mm of score; the Adam session equals a binary64 numpy implementation of the same 30 steps (loss and parameters)
     and a mutant of the tanh derivative's kernel is caught."""
import json
import os
import socket
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, os.path.join(ROOT, "examples", "weightspace"))
try:
    import numpy as np
except ImportError:
    print("skip: numpy is not installed")
    sys.exit(0)
import harness as H  # noqa: E402
import review  # noqa: E402
import sandbox  # noqa: E402
import task  # noqa: E402
import trainer  # noqa: E402

fails = 0


def expect(name, ok, detail=""):
    global fails
    print("%s %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  " + str(detail)))
    fails += 0 if ok else 1


tmp = tempfile.mkdtemp(prefix="ws_test_")


def new_run(name, budget=3, **kw):
    kw.setdefault("backend", "numpy")
    return H.Harness(os.path.join(tmp, name), budget=budget, draw_seed=7, **kw)


# 1 -- the reference ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
fx = json.load(open(os.path.join(HERE, "weightspace", "fk_fixture.json")))
d = np.abs(task.fk(np.array(fx["radians"])) - np.array(fx["position"])).max()
expect("task.fk equals cancho-robot's kinematics.cho on %d recorded poses (largest difference %.1e m)" % (len(fx["radians"]), d), d < 1e-12, d)
interior, band = task.sets()
expect("held-out sets: 2,000 poses each", len(interior) == 2000 and len(band) == 2000)
expect("interior is outside the band, band is inside it", not task.in_band(interior).any() and task.in_band(band).all())
expect("the sets are reproducible", np.array_equal(interior, task.sets()[0]) and np.array_equal(band, task.sets()[1]))
expect("the band reaches the pan hole and the high wrist_flex third", (band[:, 0] >= task.PAN_HOLE[0]).any() and (band[:, 3] >= task.WF_HIGH).any() and band[:, 0].max() > task.PAN_HOLE[1] and band[:, 3].min() < task.WF_HIGH)
expect("the trained arms nest (G(1e3) is a prefix of G(1e4))", np.array_equal(task.train_set(1000)[0], task.train_set(10000)[0][:1000]))
m, lo, hi = task.bootstrap_ci(np.arange(100.0))
expect("bootstrap interval contains the mean and is reproducible", lo < 49.5 < hi and m == 49.5 and task.bootstrap_ci(np.arange(100.0)) == (m, lo, hi), (lo, hi))

# 2 -- the review checklist --------------------------------------------------------------------------------------------------------------------------------------------------------------------
VIOLATOR = """
import numpy as np
rng = np.random.default_rng(1)
X = rng.uniform(-1, 1, size=(2000, SHAPE[1]))
Y = np.sin(X @ rng.normal(size=(SHAPE[1], SHAPE[0])))
W, *_ = np.linalg.lstsq(X, Y, rcond=None)          # the planted violation: the weights are FOUND by least squares against computed outputs
tensor = W.T
"""
CLOSED = """
import numpy as np
# closed form: a rotation about z by a fixed angle, written explicitly (allowed); and a function tabulated at fixed knots
a = np.pi / 2
R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
knots = np.linspace(-1, 1, 5)
tensor = np.zeros(SHAPE)
tensor.flat[:5] = np.tanh(knots)
tensor.flat[5:9] = R.reshape(-1)
"""
BORDER = """
import numpy as np
M = np.array([[2.0, 0.0], [0.0, 4.0]])
Minv = np.linalg.inv(M)                              # an inverse of a fixed matrix: a closed form, but also how a fit is written
tensor = np.zeros(SHAPE)
tensor.flat[:4] = Minv.reshape(-1)
"""
HILL = """
import numpy as np
rng = np.random.default_rng(0)
tensor = np.zeros(SHAPE)
def error(t):
    return float(np.abs(np.tanh(t).sum() - 3.0))
best = error(tensor)
for i in range(500):
    cand = tensor + 0.01 * rng.normal(size=SHAPE)
    if error(cand) < best:                           # keeps the best of random trials against a computed error
        tensor, best = cand, error(cand)
"""
GRAD = """
import numpy as np
tensor = np.ones(SHAPE)
lr = 0.1
for step in range(100):
    grad = 2 * (tensor - 3.0)
    tensor -= lr * grad                              # a parameter moved down the gradient of a computed loss
"""
h = new_run("review", budget=1, init="zeros")
for src in (VIOLATOR, CLOSED, BORDER, HILL, GRAD):
    r = h.write("w1", src)
    expect("planted program runs in the sandbox", r["ok"] or src in (VIOLATOR,), r)
rows, verdict = review.review_run(h.dir)
by = {i: [(v, rule) for v, rule, *_ in fs] for i, (_, _, _, fs, _) in enumerate(rows)}
expect("the lstsq violator is VIOLATING (rule fit-call)", ("violating", "fit-call") in by[0], by[0])
expect("the closed form is clean", by[1] == [], by[1])
expect("the inverse of a fixed matrix is reported BORDERLINE, not dropped", by[2] == [("borderline", "linalg")], by[2])
expect("the hill climber is VIOLATING (rule random-search)", ("violating", "random-search") in by[3], by[3])
expect("the gradient loop is VIOLATING (rule update-loop)", ("violating", "update-loop") in by[4], by[4])
expect("the run verdict is violating, and the checker's exit code is 1", verdict == "violating" and review.main([h.dir]) == 1)
for rule, i in (("fit-call", 0), ("random-search", 3), ("update-loop", 4)):
    f = review.review_source([VIOLATOR, CLOSED, BORDER, HILL, GRAD][i], disabled={rule})
    expect("mutant: with rule %s off the plant passes (so the test can fail)" % rule, not any(v == "violating" for v, *_ in f), f)
expect("an import of scipy.optimize is violating", review.review_source("from scipy.optimize import minimize\ntensor = 0")[0][0] == "violating")
expect("code that does not parse is borderline, not clean", review.review_source("def (:")[0][0] == "borderline")
expect("eval is borderline (cannot be read statically)", ("borderline", "dynamic") in [(v, r) for v, r, *_ in review.review_source("tensor = eval('np.zeros(3)')")])
h2 = new_run("review_clean", budget=1)
h2.write("w1", CLOSED)
expect("a run of clean code is clean, exit 0", review.review_run(h2.dir)[1] == "clean" and review.main([h2.dir]) == 0)

def unlock(h):
    """Marks the run as written without a logged write (for the tests that are about the budget, not about writing)."""
    st = h._state()
    st["stored"] = 1
    h._save_state(st)
    return h


# 2b -- forward before any successful write (docs/weight-space.md 12.4: 6 of 6 first calls scored the untouched network) -------------------------------------------------------------------------
h = new_run("unwritten", budget=3)
r = h.forward(8)
expect("forward before any write is refused (forward.unwritten) and spends no pass", not r["ok"] and r["rule"] == "forward.unwritten" and h.status()["passes_used"] == 0, r)
w = h.write("w1", "W1 = np.zeros(SHAPE)\n")
expect("a write that does not define `tensor` stores nothing and says so (NOTHING WAS STORED, stored false)", not w["ok"] and w["stored"] is False and w["error"].startswith("NOTHING WAS STORED"), w)
expect("a failed write does not unlock forward", h.forward(8)["rule"] == "forward.unwritten" and h.status()["passes_used"] == 0)
w = h.write("w1", "tensor = np.zeros(SHAPE)\n")
expect("a successful write reports stored true", w["ok"] and w["stored"] is True, w)
r = h.forward(8)
expect("after one successful write forward answers and spends one pass", r["ok"] and h.status()["passes_used"] == 1, r)
expect("the unlock survives reopening the run", H.Harness(h.dir, draw_seed=7).forward(8)["ok"])
expect("an old run's state without the counter is treated as unwritten", (lambda hh: (hh._save_state({"used": 0, "writes": 0, "calls": 0}), hh.forward(1))[1]["rule"] == "forward.unwritten")(new_run("legacy", budget=1)))

# 3 -- the budget ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
h = unlock(new_run("budget", budget=2))
r1, r2, r3 = h.forward(10), h.forward(10), h.forward(10)
expect("two forward calls within the budget of 2 answer", r1["ok"] and r2["ok"], (r1, r2))
expect("the third is refused (forward.budget) and the count does not move", not r3["ok"] and r3["rule"] == "forward.budget" and h.status()["passes_used"] == 2, r3)
expect("reopening the run does not reset the count", not H.Harness(h.dir, draw_seed=7).forward(5)["ok"] and H.Harness(h.dir).status()["passes_used"] == 2)
expect("budget 0 refuses the first call", unlock(new_run("b0", budget=0)).forward(1)["rule"] == "forward.budget")
h = unlock(new_run("range", budget=5))
expect("n_poses 0, 257 and a float are refused (forward.n) without spending a pass", all(h.forward(n).get("rule") == "forward.n" for n in (0, 257, 3.5)) and h.status()["passes_used"] == 0)
r = h.forward(64)
expect("a forward answers exactly one number (plus counters), no per-pose values", set(r) == {"ok", "score_mm", "n_poses", "passes_used", "passes_left"} and all(isinstance(v, (int, float, bool)) for v in r.values()), r)
zero_score = task.errors_mm(task.zero_params(), task.sample_held_in(64, np.random.default_rng([7, 1]))).mean()
expect("the score is the mean error in mm of the stored network on held-in poses (zero network: %.3f mm)" % r["score_mm"], abs(r["score_mm"] - zero_score) < 1e-6, (r, zero_score))
log = [json.loads(l) for l in open(os.path.join(h.dir, "log.jsonl"))]
expect("every call is in the log, with its arguments", [x["call"] for x in log] == ["init", "forward", "forward", "forward", "forward"] and log[-1]["args"] == {"n_poses": 64}, [x["call"] for x in log])
expect("the draw's seed is in the log's private part and not in what the call returned", "seed" in log[-1]["private"] and "seed" not in json.dumps(r))
h.show("b3", 0, 3)
expect("show and describe are logged", [json.loads(l)["call"] for l in open(os.path.join(h.dir, "log.jsonl"))][-1] == "show" and "5 -> 64" not in h.describe() and "tanh" in h.describe())
expect("the log keeps the token counts the driver gives", (h.forward(1, tokens={"in": 5, "out": 7}), json.loads(open(os.path.join(h.dir, "log.jsonl")).readlines()[-1]).get("tokens"))[1] == {"in": 5, "out": 7})

# 4 -- the sandbox -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
h = new_run("sandbox", budget=1, timeout=3.0)
secret = os.path.join(tmp, "secret.txt")
open(secret, "w").write("CANARY-4242")
r = h.write("b3", "tensor = np.eye(3)[0]\n")
expect("a good program stores its tensor as exact float32", r["ok"] and np.array_equal(h.params()["b3"], np.array([1, 0, 0], dtype=np.float32)), r)
r = h.write("b3", "tensor = np.zeros(SHAPE) + CURRENT['b3']\n")
expect("CURRENT holds the stored tensors", r["ok"] and np.array_equal(h.params()["b3"], np.array([1, 0, 0], dtype=np.float32)), r)
r = h.write("b3", "print(open(%r).read())\ntensor = np.zeros(3)\n" % secret)
expect("a read of a file outside the scratch directory is refused (sandbox.file), the content never leaves", not r["ok"] and r["rule"] == "sandbox.file" and "CANARY" not in json.dumps(r).replace("CANARY-4242'", ""), r)
r = h.write("b3", "open(%r, 'w').write('x')\ntensor = np.zeros(3)\n" % os.path.join(tmp, "written.txt"))
expect("a write outside is refused (sandbox.file) and no file appears", not r["ok"] and r["rule"] == "sandbox.file" and not os.path.exists(os.path.join(tmp, "written.txt")), r)
r = h.write("b3", "print(open(%r).read())\ntensor = np.zeros(3)\n" % os.path.join(h.dir, "log.jsonl"))
expect("the harness's own log is unreadable from the sandbox", not r["ok"] and r["rule"] == "sandbox.file", r)
r = h.write("b3", "import os\nprint(os.listdir(%r))\ntensor = np.zeros(3)\n" % ROOT)
expect("listing the repository (build/, the session binary) is refused", not r["ok"] and r["rule"] == "sandbox.file", r)
r = h.write("b3", "open('inside.txt','w').write('1')\nprint(open('inside.txt').read())\ntensor = np.zeros(3)\n")
expect("a file inside the scratch directory can be written and read", r["ok"] and r["output"].strip() == "1", r)
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen(1)
listener.settimeout(1.5)
port = listener.getsockname()[1]
for code in ("import socket\ns = socket.create_connection(('127.0.0.1', %d))\ntensor = np.zeros(3)\n" % port,
             "import urllib.request\nurllib.request.urlopen('http://127.0.0.1:%d/')\ntensor = np.zeros(3)\n" % port,
             "import importlib\nimportlib.import_module('_socket')\ntensor = np.zeros(3)\n"):
    r = h.write("b3", code)
    expect("a network call is refused (%s)" % r.get("rule"), not r["ok"] and r["rule"] == "sandbox.import", r)
try:
    conn = listener.accept()[0]
    expect("a connection reached the test's listener", False)
except socket.timeout:
    expect("nothing connected to the listener the test opened", True)
for code, rule in (("import subprocess\nsubprocess.run(['true'])\ntensor = np.zeros(3)\n", "sandbox.import"), ("import os\nos.system('true')\ntensor = np.zeros(3)\n", "sandbox.process"),
                   ("import os\nos.fork()\ntensor = np.zeros(3)\n", "sandbox.process"), ("import scipy.optimize\ntensor = np.zeros(3)\n", "sandbox.import"),
                   ("import ctypes\ntensor = np.zeros(3)\n", "sandbox.import")):
    r = h.write("b3", code)
    expect("%s is refused (%s)" % (code.split(chr(10))[1][:30], rule), not r["ok"] and r["rule"] == rule, r)
t0 = time.time()
r = h.write("b3", "while True:\n    pass\n")
expect("an endless loop is stopped by the time limit (sandbox.time) in %.1f s" % (time.time() - t0), not r["ok"] and r["rule"] == "sandbox.time" and time.time() - t0 < 10, r)
h.write("b3", "tensor = np.array([1.0, 2.0, 3.0])\n")
r = h.write("b3", "tensor = np.zeros(4)\n")
expect("a tensor of the wrong shape is refused and nothing is stored", not r["ok"] and np.array_equal(h.params()["b3"], np.array([1, 2, 3], dtype=np.float32)), r)
r = h.write("b3", "tensor = np.array([1.0, np.nan, 0.0])\n")
expect("NaN is refused (write.finite)", not r["ok"] and r["rule"] == "write.finite", r)
r = h.write("zz", "tensor = 0")
expect("an unknown tensor is refused (write.tensor)", r.get("rule") == "write.tensor", r)
# the two behaviours the sandbox does not claim to prevent
r = h.write("b3", "import os\ntensor = np.zeros(3) + float(os.path.exists('/etc/hosts'))\n")
expect("KNOWN LIMITATION: whether an outside path exists can be learned (os.stat is not audited) -- %s" % ("learned" if r["ok"] else "refused: the platform sandbox stops it too"), r["ok"] or r["rule"] == "sandbox.file", r)
r = h.write("b3", "import sys\ntensor = np.zeros(3) + float(len(open(sys.prefix + '/pyvenv.cfg').read()) >= 0 if __import__('os').path.exists(sys.prefix + '/pyvenv.cfg') else 0)\n")
expect("KNOWN LIMITATION: the Python installation is readable (numpy needs it)", r["ok"], r)

# 5 -- the CPU session --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
session = sys.argv[1] if len(sys.argv) > 1 else None
if session and os.path.exists(session):
    session = os.path.abspath(session)
    p = task.init_params()
    hs = H.Harness(os.path.join(tmp, "fwd_session"), budget=2, init="random", backend="session", session=session, draw_seed=11)
    hn = H.Harness(os.path.join(tmp, "fwd_numpy"), budget=2, init="random", backend="numpy", draw_seed=11)
    unlock(hs), unlock(hn)
    a, b = hs.forward(200), hn.forward(200)
    expect("forward through the CPU session equals the binary64 forward (%.6f and %.6f mm)" % (a["score_mm"], b["score_mm"]), abs(a["score_mm"] - b["score_mm"]) < 1e-3, (a, b))
    q, pos = task.train_set(1000)
    init = task.init_params()
    s, _ = trainer.train_session(os.path.join(tmp, "train"), q, pos, init, 250, 30)
    import subprocess
    out = subprocess.run([session, s, "--target", "nvidia-ada"], input="1\n9\n20\n", capture_output=True, text=True).stdout.splitlines()
    ref_loss, ref_p = trainer.reference(q, pos, init, 250, 30)
    v = [float(x) for x in out[0].split()]
    expect("Adam session: loss of step 1 equals the binary64 reference (%.6f, %.6f)" % (v[0] + v[4] + v[8], ref_loss[0]), abs(v[0] + v[4] + v[8] - ref_loss[0]) < 1e-5 * ref_loss[0])
    flat = np.array([float(x) for x in out[-1].split()][9:])
    dmax = np.abs(flat - task.flatten(ref_p)).max()
    expect("Adam session: all 4,611 parameters after 30 steps equal the reference's (largest difference %.1e)" % dmax, dmax < 1e-5, dmax)
    mut = os.path.join(tmp, "mutant")
    os.makedirs(mut)
    for f in ("tanh_fwd.chx", "lin_out.chx", "tanh_bwd.chx"):
        txt = open(os.path.join(ROOT, "examples", "weightspace", f)).read()
        if f == "tanh_bwd.chx":
            assert "let hm = nh + 1.0" in txt
            txt = txt.replace("let hm = nh + 1.0", "let hm = nh + 2.0")           # the derivative (1 + h)(2 - h) for (1 + h)(1 - h)
        open(os.path.join(mut, f), "w").write(txt)
    sm, _ = trainer.train_session(os.path.join(tmp, "train_mut"), q, pos, init, 250, 30, kernels=mut)
    pm = subprocess.run([session, sm, "--target", "nvidia-ada"], input="30\n", capture_output=True, text=True)
    outm = pm.stdout.splitlines() or [pm.stderr]
    dm = np.abs(np.array([float(x) for x in outm[-1].split()][9:]) - task.flatten(ref_p)).max()
    expect("mutant: a wrong tanh derivative is caught (largest difference %.1e)" % dm, dm > 1e-3, dm)
else:
    print("skip: no session binary given (the CPU session forward and the Adam session are not tested)")

print("weightspace: %s" % ("FAILED %d" % fails if fails else "ok"))
sys.exit(1 if fails else 0)
