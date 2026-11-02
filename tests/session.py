#!/usr/bin/env python3
"""Gates 1, 4 (CPU half) and 5 of docs/resident.md section 5: `cancho-gpu-session` on the CPU interpreter.

    session.py build/cancho-gpu-session [--mutants]       # --mutants needs CANCHO=the compiler and builds with scripts/build-session.sh
    session.py build/cancho-gpu-metal-session --metal     # the same checks on a Metal GPU (a Mac); no --target, and the device sums in
                                                          # another order, so the numbers are held to 1e-4 and not 1e-6

1. A hand-checked answer (a 2 x 2 fit), and the least-squares fit at 16 x 3, 256 x 16 and 1024 x 64 against a float64 reference
   written here from the mathematics: the loss to 1e-6 relative, the gradient to 1e-5 of its peak (the kernel sums in f32).
2. Resident means once: the data file is deleted after the first request and the second still answers.
3. 2,000 requests in one session keep answering (memory is given back), and the last equals the first.
4. The end: no input is no output and status 0; a last line without its newline is answered.
5. Every refusal is the rule `session` (or `usage` / `io` where `runcmd` says so), status 1, and nothing on standard output.
`--mutants` builds the program with one deliberate bug at a time and requires the checks to fail on each."""
import math
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "train"))
import lsq_objective as L  # noqa: E402
import lsq_session  # noqa: E402

METAL = "--metal" in sys.argv
TARGET = [] if METAL else ["--target", "nvidia-ada"]
TOL = 100.0 if METAL else 1.0  # the device sums in another order
bad = 0
QUIET = False


def say(good, text):
    global bad
    if not QUIET:
        print(f"{'ok  ' if good else 'FAIL'} {text}")
    bad += 0 if good else 1
    return good


def talk(binary, session, lines, extra=TARGET, text=None):
    """Run the program on the session file, feeding `lines` (a string); (status, stdout lines, stderr)."""
    r = subprocess.run([binary, session, *extra], input=lines, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout.split("\n")[:-1] if r.stdout else [], r.stderr


def reference(r, k, w):
    L.R, L.K = r, k
    x, t = L.data()
    e = [sum(a * b for a, b in zip(row, w)) - ti for row, ti in zip(x, t)]
    return 0.5 * sum(v * v for v in e) / r, [sum(e[i] * x[i][c] for i in range(r)) / r for c in range(k)]


def check(binary, quiet=False):
    global bad, QUIET
    bad = 0
    QUIET = quiet
    d = tempfile.mkdtemp()
    try:
        # 1. a hand-checked answer: x = I, t = 1 2, w = 0 gives e = -1 -2, loss 1.25, gradient -0.5 -1
        hand = os.path.join(d, "hand")
        os.makedirs(hand)
        open(os.path.join(hand, "x"), "w").write("1 0 0 1\n")
        open(os.path.join(hand, "xt"), "w").write("1 0 0 1\n")
        open(os.path.join(hand, "t"), "w").write("1 2\n")
        session = lsq_session.write(hand, 2, 2)
        # lsq_session.write generates its own data; overwrite it with the hand-made one
        for name, text in (("x", "1 0 0 1\n"), ("xt", "1 0 0 1\n"), ("t", "1 2\n")):
            open(os.path.join(hand, name), "w").write(text)
        status, out, err = talk(binary, session, "0 0\n")
        got = [float(v) for v in out[0].split()] if out else []
        say(status == 0 and len(got) == 3 and max(abs(a - b) for a, b in zip(got, [1.25, -0.5, -1.0])) <= 1e-6 * TOL, f"a hand-checked answer: {out} (want 1.25 -0.5 -1.0)")
        status, out, err = talk(binary, session, "1 2\n")
        got = [float(v) for v in out[0].split()] if out else []
        # w = 1 2: e = 0 0, loss 0, gradient 0
        say(status == 0 and len(got) == 3 and max(abs(v) for v in got) <= 1e-6 * TOL, f"the exact solution gives zero loss and gradient: {out}")

        # the fit against a float64 reference
        for r, k in ((16, 3), (256, 16), (1024, 64)):
            dd = os.path.join(d, f"fit{r}")
            session = lsq_session.write(dd, r, k)
            ws = [[0.0] * k, [0.1 * ((i % 5) - 2) for i in range(k)], [0.3 * ((i % 3) - 1) for i in range(k)]]
            status, out, err = talk(binary, session, "".join(" ".join(repr(v) for v in w) + "\n" for w in ws))
            good = status == 0 and len(out) == len(ws)
            worst_loss = worst_grad = 0.0
            if good:
                for w, line in zip(ws, out):
                    nums = [float(v) for v in line.split()]
                    loss, grad = reference(r, k, w)
                    good = good and len(nums) == k + 1
                    if len(nums) == k + 1:
                        worst_loss = max(worst_loss, abs(nums[0] - loss) / abs(loss))
                        peak = max(abs(v) for v in grad)
                        worst_grad = max(worst_grad, max(abs(a - b) for a, b in zip(nums[1:], grad)) / peak)
            say(good and worst_loss < 1e-6 * TOL and worst_grad < 1e-5 * TOL, f"the fit at {r} x {k} against float64: loss {worst_loss:.1e} relative, gradient {worst_grad:.1e} of its peak" + ("" if good else f" ({status}, {err[:80]!r})"))

        # 2. resident means once
        dd = os.path.join(d, "fit16")
        session = os.path.join(dd, "session")
        proc = subprocess.Popen([binary, session, *TARGET], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        proc.stdin.write("0.1 0.2 0.3\n")
        proc.stdin.flush()
        first = proc.stdout.readline()
        for name in ("x", "xt", "t"):
            os.remove(os.path.join(dd, name))
        proc.stdin.write("0.1 0.2 0.3\n")
        proc.stdin.flush()
        second = proc.stdout.readline()
        proc.stdin.close()
        proc.wait(timeout=30)
        say(first != "" and first == second and proc.returncode == 0, "the data is read once: the files were deleted and the second request still answers the same")
        session = lsq_session.write(dd, 16, 3)

        # 3. many requests
        many = "0.1 0.2 0.3\n" * 2000
        status, out, err = talk(binary, session, many)
        say(status == 0 and len(out) == 2000 and len(set(out)) == 1, f"2000 requests in one session: {len(out)} answers, {len(set(out))} distinct, status {status}")

        # memory is given back: the peak resident size after 20,000 requests is close to the one after 2,000
        def peak(n):
            code = ("import resource, subprocess, sys; "
                    "subprocess.run([sys.argv[1], sys.argv[2]] + sys.argv[4:], input='0.1 0.2 0.3\\n' * int(sys.argv[3]), capture_output=True, text=True); "
                    "print(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)")
            return int(subprocess.run([sys.executable, "-c", code, binary, session, str(n), *TARGET], capture_output=True, text=True, timeout=300).stdout)
        small, large = peak(2000), peak(20000)
        # Memory is given back on the device path too: it was not until the shim was built with ARC and each entry point put in an
        # autorelease pool (about 1.9 KB a request leaked before: docs/native.md N5).
        say(large < small * 1.25 + 4096, f"memory is given back: peak resident size {small} after 2000 requests, {large} after 20000 (units as the platform's ru_maxrss)")

        # 4. the end
        status, out, err = talk(binary, session, "")
        say(status == 0 and out == [], f"no input is no output: status {status}, {out}")
        status, out, err = talk(binary, session, "0.1 0.2 0.3")
        say(status == 0 and len(out) == 1, f"a last line without its newline is answered: {out}")

        # 5. the refusals
        base = open(os.path.join(dd, "session")).read()

        def refuse(what, text, tag="session", lines="0 0 0\n", extra=TARGET, edit=None):
            path = os.path.join(d, "bad.session")
            body = text if text is not None else base
            for old, new in ([edit] if isinstance(edit, tuple) else (edit or [])):
                body = body.replace(old, new)
            open(path, "w").write(body)
            status, out, err = talk(binary, path, lines, extra)
            good = status == 1 and out == [] and err.startswith(f"error[{tag}]")
            say(good, f"refused as {tag}: {what}" + ("" if good else f" (status {status}, stdout {out}, stderr {err[:100]!r})"))

        refuse("an unknown directive", base + "frobnicate x\n")
        refuse("no program", "# nothing\n")
        refuse("no answer", base.split("answer")[0])
        refuse("a program named twice", base, edit=("program bwd", "program fwd"))
        refuse("a program that is not there", base, edit=("link fwd.e -> bwd.dyt", "link nope.e -> bwd.dyt"))
        refuse("a parameter that is not there", base, edit=("link fwd.e -> bwd.dyt", "link fwd.zz -> bwd.dyt"))
        refuse("a resident parameter the kernel writes", base, edit=("resident fwd.t ", "resident fwd.e "))
        refuse("a parameter supplied twice", base, edit=("input fwd.w", "input fwd.w\ninput fwd.x"))
        refuse("a parameter nothing supplies", base, edit=("input fwd.w", ""))
        refuse("a link from an input", base, edit=("link fwd.e -> bwd.dyt", "link fwd.w -> bwd.dyt"))
        refuse("a link to an output", base, edit=("link fwd.e -> bwd.dyt", "link fwd.e -> bwd.dw"))
        refuse("a link backwards", base, edit=("link fwd.e -> bwd.dyt", "link bwd.dw -> fwd.t"))
        refuse("a link from a program to itself", base, edit=[("resident fwd.t ", "# resident fwd.t "), ("input fwd.w", "input fwd.w\nlink fwd.e -> fwd.t")])
        refuse("a link between different sizes", base, edit=("link fwd.e -> sq.xt", "link fwd.e -> bwd.xt"))
        refuse("an answer that is an input", base, edit=("answer sq.dw", "answer fwd.x"))
        refuse("a scale with a zero denominator", base, edit=("scale 1/16", "scale 1/0"))
        refuse("a scale that is not a number", base, edit=("scale 1/16", "scale x"))
        refuse("a program file that is not there", base, edit=("examples/train/lsq_residual.chx", "examples/train/none.chx"))
        refuse("a constant that is not name=value", base, edit=("r=16 k=3", "r=16 k"), tag="usage")
        if not METAL:
            refuse("a target that is not one", base, extra=["--target", "nvidia-lovelace"])
        refuse("a resident file that is not there", base, edit=(os.path.join(dd, "t"), os.path.join(dd, "gone")), tag="io")
        refuse("a request with too few numbers", None, lines="0 0\n")
        refuse("a request with too many numbers", None, lines="0 0 0 0\n")
        refuse("a request that is not numbers", None, lines="0 0 x\n")
        r = subprocess.run([binary, os.path.join(d, "no-such-file"), *TARGET], capture_output=True, text=True)
        say(r.returncode == 1 and r.stdout == "" and r.stderr.startswith("error[io]"), "refused as io: a session file that is not there")
        r = subprocess.run([binary], capture_output=True, text=True)
        say(r.returncode == 2 and r.stdout == "", "no arguments: usage, status 2")

        training_checks(binary, d)
    finally:
        shutil.rmtree(d)
    return bad == 0


def f32(x):
    import struct
    return struct.unpack("f", struct.pack("f", x))[0]


def training_checks(binary, d):
    """docs/resident.md section 12: `steps`, `carry` and `table`, on a kernel that adds a table's block to a carried parameter."""
    n, blocks = 5, 3
    tdir = os.path.join(d, "train")
    os.makedirs(tdir)
    w0 = [f32(i + 0.3) for i in range(n)]  # not dyadic, so that every byte of a binary word matters
    table = [[f32((j + 1) * 0.5 + i * 0.25 + 0.1) for i in range(n)] for j in range(blocks)]
    open(os.path.join(tdir, "w0"), "w").write(" ".join(map(repr, w0)) + "\n")
    open(os.path.join(tdir, "table"), "w").write(" ".join(repr(v) for row in table for v in row) + "\n")
    kernel = os.path.join(ROOT, "tests/chx/step_add.chx")
    body = f"""program add {kernel} n={n}
resident add.w {os.path.join(tdir, 'w0')}
table add.d {os.path.join(tdir, 'table')} {blocks}
carry add.wo -> add.w
steps
answer add.wo
"""
    path = os.path.join(tdir, "session")
    open(path, "w").write(body)

    def want(reps_total):
        w = list(w0)
        for j in range(reps_total):
            w = [f32(a + b) for a, b in zip(w, table[j % blocks])]
        return w

    def numbers(line):
        return [float(v) for v in line.split()]

    status, out, err = talk(binary, path, "5\n")
    say(status == 0 and len(out) == 1 and numbers(out[0]) == want(5), f"5 steps carry the parameter through the table, which wraps after {blocks}: {out}")
    status, out, err = talk(binary, path, "1\n")
    say(status == 0 and len(out) == 1 and numbers(out[0]) == want(1), f"one step is one block: {out}")
    status, out, err = talk(binary, path, "2\n3\n")
    say(status == 0 and len(out) == 2 and numbers(out[1]) == want(5), f"a second request continues where the first stopped (2 then 3 steps = 5): {out}")
    status, out, err = talk(binary, path, "7\n")
    say(status == 0 and len(out) == 1 and numbers(out[0]) == want(7), f"7 steps, two laps and a block: {out}")
    for text, why in (("0\n", "0 steps"), ("-1\n", "a negative count"), ("1.5\n", "a count that is not whole"), ("x\n", "a count that is not a number"), ("\n", "no count"), ("2 3\n", "a number after the count when the session takes none")):
        status, out, err = talk(binary, path, text)
        say(status == 1 and out == [] and err.startswith("error[session]"), f"refused as session: {why}")

    def refuse(what, edit, tag="session"):
        bad_path = os.path.join(tdir, "bad.session")
        text = body
        for old, new in edit:
            text = text.replace(old, new)
        open(bad_path, "w").write(text)
        status, out, err = talk(binary, bad_path, "1\n")
        say(status == 1 and out == [] and err.startswith(f"error[{tag}]"), f"refused as {tag}: {what}" + ("" if status == 1 else f" (status {status}, {out}, {err[:80]!r})"))

    refuse("a carry to a parameter that is not resident", [("resident add.w", "# resident add.w\ninput add.w")])
    refuse("a carry from an input", [("carry add.wo -> add.w", "carry add.d -> add.w")])
    refuse("a carry to an output", [("carry add.wo -> add.w", "carry add.wo -> add.wo")])
    refuse("a table whose file holds another number of blocks", [(f"table add.d {os.path.join(tdir, 'table')} {blocks}", f"table add.d {os.path.join(tdir, 'table')} {blocks + 1}")], tag="usage")
    refuse("a table with zero blocks", [(f"{os.path.join(tdir, 'table')} {blocks}", f"{os.path.join(tdir, 'table')} 0")])
    refuse("steps given twice", [("steps\n", "steps\nsteps\n")])
    refuse("a table for a parameter that is also resident", [("table add.d", "resident add.d " + os.path.join(tdir, "w0") + "\ntable add.d")])
    refuse("a carry between different sizes", [("n=5", "n=5"), ("carry add.wo -> add.w", "carry add.wo -> add.d")])
    # binary files (`f32` after the file): little-endian float32 words, no cap and nothing to parse
    import struct
    open(os.path.join(tdir, "w0.bin"), "wb").write(b"".join(struct.pack("<f", v) for v in w0))
    open(os.path.join(tdir, "table.bin"), "wb").write(b"".join(struct.pack("<f", v) for row in table for v in row))
    bin_body = body.replace(os.path.join(tdir, "w0"), os.path.join(tdir, "w0.bin")).replace(f"{os.path.join(tdir, 'table')} {blocks}", f"{os.path.join(tdir, 'table.bin')} {blocks} f32").replace("resident add.w " + os.path.join(tdir, "w0.bin"), "resident add.w " + os.path.join(tdir, "w0.bin") + " f32")
    open(os.path.join(tdir, "bin.session"), "w").write(bin_body)
    status, out, err = talk(binary, os.path.join(tdir, "bin.session"), "7\n")
    say(status == 0 and len(out) == 1 and numbers(out[0]) == want(7), f"binary resident and table files give the text files' answer: {out}")

    def refuse_bin(what, edit, tag="usage"):
        path_bad = os.path.join(tdir, "bad-bin.session")
        text = bin_body
        for old, new in edit:
            text = text.replace(old, new)
        open(path_bad, "w").write(text)
        status, out, err = talk(binary, path_bad, "1\n")
        say(status == 1 and out == [] and err.startswith(f"error[{tag}]"), f"refused as {tag}: {what}" + ("" if status == 1 else f" (status {status}, {out}, {err[:80]!r})"))

    open(os.path.join(tdir, "short.bin"), "wb").write(b"\0" * 12)
    open(os.path.join(tdir, "long.bin"), "wb").write(b"\0" * (4 * n + 4))
    refuse_bin("a binary file too short", [(os.path.join(tdir, "w0.bin"), os.path.join(tdir, "short.bin"))])
    refuse_bin("a binary file too long", [(os.path.join(tdir, "w0.bin"), os.path.join(tdir, "long.bin"))])
    refuse_bin("a binary file that is not there", [(os.path.join(tdir, "w0.bin"), os.path.join(tdir, "gone.bin"))], tag="io")
    refuse_bin("a flag that is not f32", [(os.path.join(tdir, "w0.bin") + " f32", os.path.join(tdir, "w0.bin") + " f64")], tag="session")
    # binary16 files (`f16` after the file) for an f16 parameter
    vals = [f32(v * 0.37 - 1.1) for v in range(8)]
    h16 = [struct.unpack("<e", struct.pack("<e", v))[0] for v in vals]
    open(os.path.join(tdir, "h.bin"), "wb").write(b"".join(struct.pack("<e", v) for v in vals))
    wbody = f"program wd {os.path.join(ROOT, 'tests/chx/widen.chx')} n=8\nresident wd.x {os.path.join(tdir, 'h.bin')} f16\nanswer wd.y\n"
    open(os.path.join(tdir, "widen.session"), "w").write(wbody)
    status, out, err = talk(binary, os.path.join(tdir, "widen.session"), "\n")
    say(status == 0 and len(out) == 1 and numbers(out[0]) == h16, f"an f16 binary file is read as binary16 words: {out}")
    open(os.path.join(tdir, "widen-bad.session"), "w").write(wbody.replace("h.bin", "short16.bin"))
    open(os.path.join(tdir, "short16.bin"), "wb").write(b"\0" * 14)
    status, out, err = talk(binary, os.path.join(tdir, "widen-bad.session"), "\n")
    say(status == 1 and out == [] and err.startswith("error[usage]"), "refused as usage: an f16 binary file of the wrong size")
    open(os.path.join(tdir, "widen-f32.session"), "w").write(wbody.replace(" f16\n", " f32\n"))
    status, out, err = talk(binary, os.path.join(tdir, "widen-f32.session"), "\n")
    say(status == 1 and out == [] and err.startswith("error[usage]"), "refused as usage: an f32 flag on an f16 parameter")
    # without `steps` a request is just its inputs, once
    open(os.path.join(tdir, "once.session"), "w").write(body.replace("steps\n", ""))
    status, out, err = talk(binary, os.path.join(tdir, "once.session"), "\n\n")
    say(status == 0 and len(out) == 2 and numbers(out[0]) == want(1) and numbers(out[1]) == want(2), f"without steps each request is one repetition, and the table and the carry still advance: {out}")


MUTANTS = [
    ("src/session.cho", "fm[m[gd + 2] + j] = interp.convert(dt, fm[m[gs + 2] + j]);", "fm[m[gd + 2] + j] = interp.convert(dt, 0.0);", "a link copies nothing"),
    ("src/session.cho", "fm[m[g + 2] + j] * scale_a / scale_b", "fm[m[g + 2] + j] * scale_a", "the denominator of a scale is ignored"),
    ("src/session.cho", "fm[m[g + 2] + j] * scale_a / scale_b", "fm[m[g + 2] + j] / scale_a * scale_b", "a scale is applied the wrong way round"),
    ("src/session.cho", "if m[link + 2] == q {", "if m[link + 2] == q + 1 {", "a link is copied after the program that reads it"),
    ("session/main.cho", "mem.release_to(m, mark);", "m[mem.text_slot()] = tmark;", "memory is not given back after a request"),
    ("session/main.cho", "if c < 0 && n == 0 {", "if c < 0 {", "a last line without its newline is dropped"),
    ("src/session.cho", "while pos < n && space(text.at(t, line, pos)) {\n        pos = pos + 1;\n    }\n    if pos < n {", "while pos < n && space(text.at(t, line, pos)) {\n        pos = pos + 1;\n    }\n    if pos < 0 {", "a request with too many numbers is accepted"),
    ("src/session.cho", "if sp >= dp {", "if sp > dp {", "a link from a program to itself is accepted"),
    ("src/session.cho", "return (truncate(v), text.sub(line, pos, n));", "return (truncate(v) + 1, text.sub(line, pos, n));", "the count of steps is one too many"),
    ("src/session.cho", "m[sess + s_count()] = m[sess + s_count()] + 1;\n        j = j + 1;", "j = j + 1;", "the table position does not advance"),
    ("src/session.cho", "fm[m[gd + 2] + e] = fm[m[gs + 2] + e];", "fm[m[gd + 2] + e] = fm[m[gd + 2] + e];", "a carry copies nothing"),
    ("src/session.cho", "(m[sess + s_count()] % m[tb + 3])", "(m[sess + s_count()] % m[tb + 3] + 1)", "the table is read one block late"),
    ("src/runcmd.cho", "f32.from_bits32(int_of(t[b]) | int_of(t[b + 1]) << 8", "f32.from_bits32(int_of(t[b]) | int_of(t[b + 1]) << 7", "a binary word's second byte is shifted wrongly"),
    ("src/runcmd.cho", "if fsize != 4 * n {", "if fsize < 4 * n {", "a binary file longer than the parameter is accepted"),
    ("src/runcmd.cho", "f32.from_bits16(int_of(t[b]) | int_of(t[b + 1]) << 8)", "f32.from_bits16(int_of(t[b]) | int_of(t[b + 1]) << 7)", "an f16 word's second byte is shifted wrongly"),
    ("src/session.cho", "if m[mem.get(m, supplied, dp) + da] != 1 {", "if m[mem.get(m, supplied, dp) + da] < 0 {", "a carry to a parameter that is not resident is accepted"),
]


def mutants(cancho):
    ok = True
    for path, old, new, what in MUTANTS:
        d = tempfile.mkdtemp()
        try:
            srcs = [os.path.join(ROOT, "src", f) for f in sorted(os.listdir(os.path.join(ROOT, "src"))) if f.endswith(".cho") and f != "main.cho"] + [os.path.join(ROOT, "session", "main.cho")]
            built = []
            for s in srcs:
                rel = os.path.relpath(s, ROOT)
                dest = os.path.join(d, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                text = open(s).read()
                if rel == path:
                    if text.count(old) != 1:
                        print(f"FAIL mutant '{what}': the text it changes occurs {text.count(old)} times in {rel}")
                        ok = False
                    text = text.replace(old, new, 1)
                open(dest, "w").write(text)
                built.append(dest)
            out = os.path.join(d, "session-bin")
            r = subprocess.run([cancho, "build"] + built + ["--std", "-o", out], capture_output=True, text=True, env=dict(os.environ))
            if r.returncode != 0:
                print(f"FAIL mutant '{what}' does not build: {r.stderr[-200:]}")
                ok = False
                continue
            survived = check(out, quiet=True)
            if survived:
                print(f"FAIL mutant '{what}' SURVIVED: the check does not notice it")
                ok = False
            else:
                print(f"ok   mutant killed: {what}")
        finally:
            shutil.rmtree(d)
    return ok


if __name__ == "__main__":
    good = check(os.path.abspath(sys.argv[1]))
    if "--mutants" in sys.argv:
        good = mutants(os.environ.get("CANCHO", "cancho")) and good
    sys.exit(0 if good else 1)
