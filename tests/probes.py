#!/usr/bin/env python3
"""docs/resident.md section 26.2: the probes of what the .chx surface can and cannot express for a ResNet (docs/roadmap-models.md M9).

    python3 tests/probes.py build/cancho-gpu [build/cancho-gpu-session]

Every file in examples/cnn/probes/ names, on its first line, what the compiler must answer (`// PROBE <rule tag>` for a refusal, `// PROBE ok` for a program it accepts, `// PROBE numeric`
for one that is also run and compared with numpy); the second line has the constants. A probe that is refused with another tag, or accepted when a refusal was recorded, fails: the
record of what the surface can say is held to the compiler, as the reject fixtures are. Needs numpy for the numeric probes."""
import glob
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES = os.path.join(ROOT, "examples", "cnn", "probes")
bad = 0


def say(good, text):
    global bad
    print(f"{'ok  ' if good else 'FAIL'} {text}")
    bad += 0 if good else 1


def main(gpu, session):
    for path in sorted(glob.glob(os.path.join(PROBES, "*.chx"))):
        lines = open(path).read().split("\n")
        want = lines[0].replace("// PROBE ", "").strip()
        args = lines[1].replace("// ARGS ", "").split()
        name = os.path.basename(path)
        r = subprocess.run([gpu, "check", path] + args, capture_output=True, text=True)
        got = "ok"
        if r.returncode != 0:
            got = r.stderr.split("error[")[1].split("]")[0] if "error[" in r.stderr else "?"
        if want == "numeric":
            say(got == "ok", f"{name}: accepted")
            numeric(name, path, args, session)
        else:
            say(got == want, f"{name}: {'accepted' if got == 'ok' else 'refused as ' + got}" + ("" if got == want else f", recorded as {want}"))


def numeric(name, path, args, session):
    try:
        import numpy as np
    except ImportError:
        print("skip: numpy is not installed")
        return
    if name != "max_select.chx" or session is None:
        return
    consts = dict(a.split("=") for a in args)
    r, c = int(consts["r"]), int(consts["c"])
    rng = np.random.RandomState(5)
    # f32 values with ties and signs: the select must be exactly the maximum, and its mask must be 1 only where a > b strictly (the first of equals wins)
    a = rng.normal(size=(r, c)).astype(np.float32)
    b = rng.normal(size=(r, c)).astype(np.float32)
    b[::3] = a[::3]
    d = tempfile.mkdtemp()
    for n_, v in (("a", a), ("b", b)):
        open(os.path.join(d, n_), "wb").write(v.astype("<f4").tobytes())
    spec = os.path.join(d, "s")
    open(spec, "w").write(f"program k {path} " + " ".join(args) + f"\nresident k.a {d}/a f32\nresident k.b {d}/b f32\nanswer k.m\nanswer k.s\n")
    out = subprocess.run([session, spec, "--target", "apple-m-series"], input="\n", capture_output=True, text=True, timeout=120)
    rows = [np.array([float(v) for v in line.split()]) for line in out.stdout.splitlines()]
    m, s = rows[0][: r * c].reshape(r, c), rows[0][r * c:].reshape(r, c)
    say(np.array_equal(m, np.maximum(a, b)), "max_select: step-select gives exactly np.maximum (bit for bit, ties included)")
    say(np.array_equal(s, (a > b).astype(float)), "max_select: its mask is 1 only where a > b strictly (ties keep the first, as torch's max-pool does)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    sys.exit(1 if bad else 0)
