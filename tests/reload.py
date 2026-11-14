#!/usr/bin/env python3
"""docs/resident.md sections 29 and 30: the Metal session's requests `reload P.PARAM ...` (re-reads a binary resident file onto its buffer) and `copy P.OUT -> Q.PARAM ...` (an output copied on the device to a resident parameter).

    python3 tests/reload.py [build/cancho-gpu-metal-session]

A session with one program (the first two rows of a [4, 64] matrix) whose input is a resident f32 file; the file is rewritten between requests. Checks: the answer follows the file only after `reload`
(before it the device still holds the old data); two files (a second program on its own file) reload by name in one request; a name that is not a program, not a parameter, not resident, an output, or no
name at all is refused with the rule `session`, which ends the session as every refusal of a request does. `copy`: the output of one program becomes the input of another after the request that made it, the second program's answer follows, and eight malformed or illegal copies are refused. Needs a Mac.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SES = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(ROOT, "build/cancho-gpu-metal-session")
CHX = os.path.join(ROOT, "tests/chx/dit_rows2.chx")
tmp = tempfile.mkdtemp()
fa, fb = os.path.join(tmp, "a.f32"), os.path.join(tmp, "b.f32")
rng = np.random.default_rng(3)
wr = lambda p, a: np.asarray(a, dtype="<f4").tofile(p)
sess = os.path.join(tmp, "s.session")
open(sess, "w").write("\n".join([
    f"program pa {CHX} m=4 n=64 bn=64", f"resident pa.x {fa} f32",
    f"program pb {CHX} m=4 n=64 bn=64", f"resident pb.x {fb} f32",
    "steps", "answer pa.y", "answer pb.y", ""]))
fails = []


def check(name, ok):
    print("%s %s" % ("ok  " if ok else "FAIL", name))
    if not ok:
        fails.append(name)


def start():
    return subprocess.Popen([SES, sess], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)


def ask(p, line):
    p.stdin.write(line + "\n")
    p.stdin.flush()
    return p.stdout.readline().split()


def rows(a):
    return np.concatenate([a[0], a[1]])


a0, b0 = rng.standard_normal((4, 64)), rng.standard_normal((4, 64))
wr(fa, a0); wr(fb, b0)
p = start()
got = np.array(ask(p, "1"), dtype=np.float64)
check("first request answers the files as uploaded", np.allclose(got, np.concatenate([rows(a0.astype('f4')), rows(b0.astype('f4'))]), atol=0))
a1, b1 = rng.standard_normal((4, 64)), rng.standard_normal((4, 64))
wr(fa, a1); wr(fb, b1)
got = np.array(ask(p, "1"), dtype=np.float64)
check("a rewritten file changes nothing without reload", np.allclose(got[:128], rows(a0.astype('f4')), atol=0))
check("reload of one parameter answers 1", ask(p, "reload pa.x") == ["1"])
got = np.array(ask(p, "1"), dtype=np.float64)
want = np.concatenate([rows(a1.astype("f4")), rows(b0.astype("f4"))])
check("after reload pa.x only pa follows the file", np.allclose(got, want, atol=0))
check("reload of two answers 2", ask(p, "reload pa.x pb.x") == ["2"])
got = np.array(ask(p, "1"), dtype=np.float64)
check("both follow after reloading both", np.allclose(got, np.concatenate([rows(a1.astype('f4')), rows(b1.astype('f4'))]), atol=0))
p.stdin.close(); p.wait()
for bad in ("reload", "reload nope.x", "reload pa.nope", "reload pa.y", "reload pa", "reload pa.x.y"):
    p = start()
    ask(p, "1")
    out = ask(p, bad)
    err = p.stderr.read()
    p.stdin.close(); p.wait()
    check("refused: %r (rule session)" % bad, out == [] and "error[session]" in err and p.returncode != 0)
# ---- copy
fc, fd = os.path.join(tmp, "c.f32"), os.path.join(tmp, "d.f32")
c0, d0 = rng.standard_normal((2, 64)), rng.standard_normal((2, 64))
wr(fc, c0); wr(fd, d0)
sess2 = os.path.join(tmp, "s2.session")
open(sess2, "w").write("\n".join([
    f"program pa {CHX} m=2 n=64 bn=64", f"resident pa.x {fc} f32",
    f"program pb {CHX} m=2 n=64 bn=64", f"resident pb.x {fd} f32",
    f"program pc {CHX} m=2 n=64 bn=64", "link pb.y -> pc.x",
    "steps", "answer pa.y", "answer pb.y", "answer pc.y", ""]))
sess_save, sess = sess, sess2
p = start()
got = np.array(ask(p, "1"), dtype=np.float64)
check("copy: before the copy pb holds its own file", np.allclose(got[128:256], d0.astype("f4").ravel(), atol=0))
check("copy: one pair answers 1", ask(p, "copy pa.y -> pb.x") == ["1"])
got = np.array(ask(p, "1"), dtype=np.float64)
check("copy: pb now computes from pa's output", np.allclose(got[128:256], c0.astype("f4").ravel(), atol=0) and np.allclose(got[:128], c0.astype("f4").ravel(), atol=0))
p.stdin.close(); p.wait()
for bad in ("copy", "copy pa.y", "copy pa.y -> ", "copy pa.y => pb.x", "copy nope.y -> pb.x", "copy pa.x -> pb.x", "copy pa.y -> pb.y", "copy pa.y -> pc.x"):
    p = start()
    ask(p, "1")
    out = ask(p, bad)
    err = p.stderr.read()
    p.stdin.close(); p.wait()
    check("refused: %r (rule session)" % bad, out == [] and "error[session]" in err and p.returncode != 0)
sess = sess_save
sys.exit(1 if fails else 0)
