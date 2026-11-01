#!/usr/bin/env python3
"""Gate 7 of docs/run.md: a least-squares fit trained by cancho-opt's BFGS whose
loss and gradient are computed by cancho-gpu kernels (examples/train/).

    OPT=<cancho-opt>/build/opt python3 scripts/check_train.py [--bin PROGRAM] [--target T]

`--bin` is the program the objective calls (default `$CANCHO_GPU`, else
`build/cancho-gpu`): `build/cancho-gpu-metal-device` runs the same fit through the
Metal device program (docs/run.md section 7), whose one target is implied, so
`--target -` (the default for a name containing "device"). `--target` defaults to
`nvidia-ada` for `cancho-gpu`. `--runs N` repeats the fit to show the spread.

Not in scripts/gate.sh: it needs a built `opt` (cancho-opt, its own pinned
compiler). It can fail: the last check flips the gradient's sign and requires the
fit not to reach the answer."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "train"))
import lsq_objective as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bin", default=os.environ.get("CANCHO_GPU", os.path.join(ROOT, "build", "cancho-gpu")))
ap.add_argument("--target", default=None)
ap.add_argument("--runs", type=int, default=1)
ap.add_argument("--session", action="store_true", help="BFGS with --persistent through a resident session (docs/resident.md): --bin is cancho-gpu-session")
ARGS = ap.parse_args()
OPT = os.environ.get("OPT", "")
BIN = os.path.abspath(ARGS.bin)
TARGET = ARGS.target or ("-" if "device" in os.path.basename(BIN) else "nvidia-ada")
if not os.path.exists(OPT):
    raise SystemExit("set OPT to a built cancho-opt `opt`")
PY = sys.executable
failed = 0


def check(name, good, detail=""):
    global failed
    print(f"{'ok  ' if good else 'FAIL'} {name} {detail}")
    failed += 0 if good else 1


def normal_equations():
    x, t = L.data()
    a = [[sum(x[i][p] * x[i][q] for i in range(L.R)) for q in range(L.K)] + [sum(x[i][p] * t[i] for i in range(L.R))]
         for p in range(L.K)]
    for i in range(L.K):
        piv = max(range(i, L.K), key=lambda r: abs(a[r][i]))
        a[i], a[piv] = a[piv], a[i]
        for r in range(L.K):
            if r != i:
                f = a[r][i] / a[i][i]
                a[r] = [p - f * q for p, q in zip(a[r], a[i])]
    w = [a[i][L.K] / a[i][i] for i in range(L.K)]
    loss = 0.5 * sum((sum(p * q for p, q in zip(x[i], w)) - t[i]) ** 2 for i in range(L.R)) / L.R
    return w, loss


def fit(program):
    start = time.time()
    persistent = ["--persistent"] if ARGS.session else []
    r = subprocess.run([OPT, "minimize", "--method", "bfgs", "--x0", "0,0,0", *persistent, "--"] + program, capture_output=True, text=True)
    elapsed = time.time() - start
    # A fit that does not converge exits 1 and still prints its result; only an
    # error object is a failure of the run itself.
    try:
        out = json.loads(r.stdout)
    except ValueError:
        raise SystemExit(f"opt: {r.stdout}{r.stderr}")
    if "error" in out:
        raise SystemExit(f"opt: {r.stdout}{r.stderr}")
    return out, elapsed


w_ne, loss_ne = normal_equations()
objective = [PY, os.path.join(ROOT, "examples", "train", "lsq_objective.py"), BIN, ROOT, TARGET]
if ARGS.session:
    sys.path.insert(0, os.path.join(ROOT, "examples", "train"))
    import lsq_session
    session_dir = tempfile.mkdtemp()
    session_file = lsq_session.write(session_dir, L.R, L.K)
    objective = [BIN, session_file] + ([] if "metal" in os.path.basename(BIN) else ["--target", "nvidia-ada"])
res, secs = fit(objective)
times = [secs]
for _ in range(ARGS.runs - 1):
    again, more = fit(objective)
    assert again["nfev"] == res["nfev"]
    times.append(more)
dist = max(abs(a - b) for a, b in zip(res["x"], w_ne))
check("bfgs converged with kernels as the objective", res["converged"], f"nfev {res['nfev']}")
check("it reaches the normal-equations solution", dist < 1e-4 and abs(res["fun"] - loss_ne) < 1e-8,
      f"largest difference in w {dist:.1e}, in the loss {abs(res['fun'] - loss_ne):.1e}")
ms = sorted(1000 * s_ / res["nfev"] for s_ in times)
if ARGS.session:
    print(f"     cost through a resident session: the whole fit, 12 evaluations, takes {1000 * min(times):.1f} to {1000 * max(times):.1f} ms over {len(times)} run(s), "
          f"{ms[0]:.2f} to {ms[-1]:.2f} ms an evaluation, start included (this machine, not a benchmark)")
else:
    print(f"     cost with {os.path.basename(BIN)}: {min(times):.2f} to {max(times):.2f} s for {res['nfev']} evaluations, "
          f"{ms[0]:.0f} to {ms[-1]:.0f} ms each over {len(times)} run(s) "
          f"(two {os.path.basename(BIN)} processes and a Python one; this machine, not a benchmark)")

# It can fail: the same program with the gradient's sign flipped.
with tempfile.TemporaryDirectory() as d:
    bad = os.path.join(d, "flipped.py")
    if ARGS.session:
        with open(bad, "w") as f:
            f.write("import subprocess, sys\n"
                    f"p = subprocess.Popen({objective!r}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)\n"
                    "for line in sys.stdin:\n"
                    "    p.stdin.write(line); p.stdin.flush()\n"
                    "    v = p.stdout.readline().split()\n"
                    "    print(v[0], *['-' + g if not g.startswith('-') else g[1:] for g in v[1:]], flush=True)\n")
    else:
        with open(bad, "w") as f:
            f.write("import subprocess, sys\n"
                    f"r = subprocess.run([{PY!r}, {objective[1]!r}, {BIN!r}, {ROOT!r}, {TARGET!r}], input=sys.stdin.read(), capture_output=True, text=True, check=True)\n"
                    "v = r.stdout.split()\nprint(v[0], *['-' + g if not g.startswith('-') else g[1:] for g in v[1:]])\n")
    res_bad, _ = fit([PY, bad])
dist_bad = max(abs(a - b) for a, b in zip(res_bad["x"], w_ne))
check("a flipped gradient does not reach it", not (res_bad["converged"] and dist_bad < 1e-4),
      f"converged {res_bad['converged']}, difference in w {dist_bad:.1e}")
sys.exit(1 if failed else 0)
