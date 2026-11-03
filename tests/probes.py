#!/usr/bin/env python3
"""docs/resident.md section 26.2: the probes of what the .chx surface can and cannot express for a ResNet (docs/roadmap-models.md M9), and, in
examples/qwen35/probes/, for the training form of a gated-delta layer (docs/roadmap-models.md M10).

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
PROBES = [os.path.join(ROOT, "examples", "cnn", "probes"), os.path.join(ROOT, "examples", "qwen35", "probes")]
bad = 0


def say(good, text):
    global bad
    print(f"{'ok  ' if good else 'FAIL'} {text}")
    bad += 0 if good else 1


def main(gpu, session):
    for path in sorted(p for d in PROBES for p in glob.glob(os.path.join(d, "*.chx"))):
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
    if session is None:
        return
    if name in GDN_NUMERIC:
        gdn_numeric(name, path, args, session)
        return
    if name != "max_select.chx":
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


GDN_NUMERIC = ("decay_mask.chx", "solve_fwdsub.chx", "state_scan_matmul.chx")
# deliberate faults of the two probes: each must make its numeric check fail
GDN_MUTANTS = {
    "decay_mask.chx": [("no clamp (the upper triangle reaches exp)", "  let cl  = df - pos\n", "  let cl  = df + pos * 0\n"),
                       ("difference the wrong way round", "  let df  = dc - dr\n", "  let df  = dr - dc\n")],
    "solve_fwdsub.chx": [("one-hot row of the wrong step", "let e    = load eye[i, 0 ; 1, 64]", "let e    = load eye[63 - i, 0 ; 1, 64]"),
                         ("row of Ls taken from the wrong end", "let lrow = load ls[c * 64 + i, 0 ; 1, 64]", "let lrow = load ls[c * 64 + 63 - i, 0 ; 1, 64]")],
    "state_scan_matmul.chx": [("the decay applied to the update too", "    yield sd + ups\n", "    yield (sd + ups) * ev\n"),
                              ("v_new = w S - u", "    let vn  = ut - ws\n", "    let vn  = ws - ut\n"),
                              ("the output from the old state only", "    store o1 + o2 -> o[(h * nc + i) * 64, 0]\n", "    store o1 -> o[(h * nc + i) * 64, 0]\n")],
}


def run_session(session, path, args, files, answers):
    d = tempfile.mkdtemp()
    spec = os.path.join(d, "s")
    lines = [f"program k {path} " + " ".join(args)]
    for n_, (arr, kind) in files.items():
        fp = os.path.join(d, n_)
        open(fp, "wb").write(arr.astype("<f4").tobytes())
        lines.append(f"resident k.{n_} {fp} f32")
    lines += [f"answer k.{a}" for a in answers]
    open(spec, "w").write("\n".join(lines) + "\n")
    out = subprocess.run([session, spec, "--target", "apple-m-series"], input="\n", capture_output=True, text=True, timeout=300)
    import numpy as np
    return [np.array([float(v) for v in line.split()]) for line in out.stdout.splitlines()]


def gdn_expected_and_run(name, path, args, session):
    import numpy as np
    consts = dict(a.split("=") for a in args)
    nc = int(consts["nc"])
    nv = int(consts.get("nv", 0))
    rng = np.random.RandomState(11)
    if name == "decay_mask.chx":
        a = rng.normal(size=(nc * 64, 64))
        g = -rng.uniform(0.5, 3.0, size=(nc, 64))  # gc falls to about -100: the unclamped upper triangle is exp(+100), past f32
        gc = np.cumsum(g, 1)
        mask = np.tril(np.ones((64, 64)))
        rows = run_session(session, path, args, {"a": (a, 0), "gcc": (gc.reshape(nc * 64, 1), 0), "gcr": (gc, 0), "mask": (mask, 0)}, ["y"])
        want = np.concatenate([a[c * 64:(c + 1) * 64] * mask * np.exp(np.minimum(gc[c][:, None] - gc[c][None, :], 0)) for c in range(nc)])
        got = rows[0].reshape(nc * 64, 64) if rows and rows[0].size == nc * 64 * 64 else None
    elif name == "state_scan_matmul.chx":
        # the chunk factors of gdn_ref (one head, 128 keys, 64 values) are the kernel's inputs; its output is the specification's o and final state
        sys.path.insert(0, os.path.join(ROOT, "examples", "qwen35"))
        import gdn_ref
        T = nc * 64
        q = rng.normal(size=(1, T, 128)); k = rng.normal(size=(1, T, 128)); v = rng.normal(size=(1, T, 64))
        q /= np.linalg.norm(q, axis=-1, keepdims=True); k /= np.linalg.norm(k, axis=-1, keepdims=True)
        g = -rng.uniform(0.01, 0.4, size=(1, T)); beta = 1 / (1 + np.exp(-rng.normal(size=(1, T))))
        o_ref, S_ref, c = gdn_ref.chunk_fwd(q, k, v, g, beta, 64, "inv")
        gc = c["gc"][0]
        tri = lambda x: np.where(np.tril(np.ones((64, 64), bool)), x, 0.0)
        aqk = np.concatenate([tri(c["qs"][0, i] @ c["k"][0, i].T * c["D"][0, i]) for i in range(nc)])
        qg = np.concatenate([c["qs"][0, i] * np.exp(gc[i])[:, None] for i in range(nc)])
        kdt = np.concatenate([(c["k"][0, i] * np.exp(gc[i, -1] - gc[i])[:, None]).T for i in range(nc)])
        el = np.stack([np.full(64, np.exp(gc[i, -1])) for i in range(nc)])
        rows = run_session(session, path, args, {"u": (c["u"][0].reshape(T, 64), 0), "w": (c["w"][0].reshape(T, 128), 0), "aqk": (aqk, 0), "qg": (qg, 0),
                                                 "kdt": (kdt, 0), "el": (el, 0)}, ["o", "sf"])
        # the session prints every answer on one line, in order
        if len(rows) != 1 or rows[0].size != T * 64 + 128 * 64:
            return False
        err = max(np.abs(rows[0][: T * 64].reshape(T, 64) - o_ref[0]).max(), np.abs(rows[0][T * 64:].reshape(128, 64) - S_ref[0]).max())
        return bool(np.isfinite(err) and err < 1e-4)
    else:
        ls = np.tril(rng.normal(size=(nc, 64, 64)) * 0.25, -1)
        R = rng.normal(size=(nc, 64, nv))
        rt = np.concatenate([R[c].T for c in range(nc)])
        eye = np.eye(64)
        rows = run_session(session, path, args, {"ls": (ls.reshape(nc * 64, 64), 0), "rt": (rt, 0), "eye": (eye, 0)}, ["xt"])
        want = np.concatenate([(np.linalg.inv(np.eye(64) + ls[c]) @ R[c]).T for c in range(nc)])
        got = rows[0].reshape(nc * nv, 64) if rows and rows[0].size == nc * nv * 64 else None
    if got is None or not np.all(np.isfinite(got)):
        return False
    return bool(np.abs(got - want).max() <= 1e-4 * max(1.0, np.abs(want).max()))


def gdn_numeric(name, path, args, session):
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        print("skip: numpy is not installed")
        return
    say(gdn_expected_and_run(name, path, args, session), f"{name}: the interpreter's answer equals numpy's")
    src = open(path).read()
    for what, old, new in GDN_MUTANTS[name]:
        assert old in src, (name, old)
        d = tempfile.mkdtemp()
        mp = os.path.join(d, name)
        open(mp, "w").write(src.replace(old, new))
        say(not gdn_expected_and_run(name, mp, args, session), f"{name}: mutant ({what}) fails the check")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    sys.exit(1 if bad else 0)
