#!/usr/bin/env python3
"""Gates 1 and 2 of docs/run.md section 7: `--in` / `--out` on the device
programs, against a mock driver (no GPU).

    device_io.py <cuda|metal> <program> <mock log variable> [VAR=value ...]

The mock runs no kernel, so a run with `--in` has to refuse `device-mismatch`
(that is what shows the comparison can fail) and print nothing on stdout; what
the "driver" received is in its log, `sums`: the FNV-1a 64 of each argument
buffer at the launch. Those are compared here with the f32 / f16 / i8 encoding
of the numbers written to the files, so a wrong order, a dropped sign, a skipped
upload or a wrong rounding is a failure. Then the one run that passes (inputs
whose answer is zero, which a mock that writes nothing gets right) prints the
`--out` line, and then each refusal of section 3, for this program.
"""
import os
import struct
import subprocess
import sys
import tempfile

KIND, BIN, LOGVAR = sys.argv[1:4]
ENV = dict(os.environ)
for kv in sys.argv[4:]:
    k, v = kv.split("=", 1)
    ENV[k] = v
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failed = 0


def check(name, good, detail=""):
    global failed
    print(f"{'ok  ' if good else 'FAIL'} [{KIND}] {name} {detail if not good else ''}")
    failed += 0 if good else 1


def fnv(data):
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) % (1 << 64)
    return f"{h:016x}"


def encode(dtype, values):
    if dtype == "f32":
        return b"".join(struct.pack("<f", v) for v in values)
    if dtype == "f16":
        return b"".join(struct.pack("<e", v) for v in values)
    if dtype == "i8":
        return b"".join(struct.pack("<b", int(v)) for v in values)
    raise ValueError(dtype)


def lcg(seed, n):
    out = []
    for _ in range(n):
        seed = (seed * 1664525 + 1013904223) % 4294967296
        out.append(round((seed >> 9) / 8388608.0 - 0.5, 5))
    return out


def run(args, log=None):
    if log and os.path.exists(log):
        os.remove(log)
    env = dict(ENV)
    if log:
        env[LOGVAR] = log
    r = subprocess.run([BIN, "run"] + args, capture_output=True, text=True, env=env)
    return r


def sums(log):
    with open(log) as f:
        for line in f:
            if line.startswith("sums "):
                return line.split()[1].split(",")
    return None


# name, dtype, element count, is it written; in parameter order.
CASES = [
    ("tests/chx/bwd_dx.chx", ["r=2", "n=2", "k=3"],
     [("dy", "f32", 4, False), ("w", "f32", 6, False), ("dx", "f32", 6, True)]),
    ("tests/chx/bwd_dw.chx", ["r=4", "n=3", "k=2"],
     [("dyt", "f32", 12, False), ("xt", "f32", 8, False), ("dw", "f32", 6, True)]),
    # f16 activations, i8 weights and scales, an f32 scale: every dtype the uploads encode.
    ("tests/chx/ported/gemm_fp4.chx", ["m=128", "n=128", "k=64"],
     [("x", "f16", 128 * 64, False), ("wq", "i8", 128 * 32, False), ("ws", "i8", 128 * 4, False),
      ("wg", "f32", 128, False), ("y", "f32", 128 * 128, True)]),
]

with tempfile.TemporaryDirectory() as d:
    log = os.path.join(d, "log")
    for case, (kernel, consts, params) in enumerate(CASES):
        args, want = [os.path.join(ROOT, kernel)] + consts, []
        for i, (name, dtype, n, written) in enumerate(params):
            if written:
                want.append(fnv(bytes(n * {"f32": 4, "f16": 2, "i8": 1}[dtype])))
                continue
            if dtype == "i8":
                values = [((i * 7 + j * 5) % 15) - 7 for j in range(n)]
                text = " ".join(str(v) for v in values)
            else:
                values = lcg(100 * case + i, n)
                text = "\n".join(" ".join(repr(v) for v in values[j:j + 8]) for j in range(0, n, 8))
            path = os.path.join(d, f"{case}_{name}")
            with open(path, "w") as f:
                f.write(text + "\n")
            args += ["--in", name, path]
            want.append(fnv(encode(dtype, values)))
        out = [p[0] for p in params if p[3]][0]
        r = run(args + ["--out", out], log)
        tag = os.path.basename(kernel)
        check(f"{tag}: the zeros of the mock are refused as device-mismatch",
              r.returncode == 1 and "error[device-mismatch]" in r.stderr, r.stderr[-200:])
        check(f"{tag}: nothing on stdout after a mismatch", r.stdout == "", repr(r.stdout[:60]))
        got = sums(log) if os.path.exists(log) else None
        check(f"{tag}: the driver received the encoded inputs", got == want, f"got {got}, want {want}")

    # The reference ran on the files' numbers, not the LCG: bwd_dx's known answer.
    dy, w = os.path.join(d, "dy"), os.path.join(d, "w")
    open(dy, "w").write("1 2\n3 4\n")
    open(w, "w").write("1 0 2\n0 1 3\n")
    kx = os.path.join(ROOT, "tests/chx/bwd_dx.chx")
    r = run([kx, "r=2", "n=2", "k=3", "--in", "dy", dy, "--in", "w", w, "--out", "dx"], log)
    check("the reference is computed from the files (peak 18.0)", "peak 18.0" in r.stderr, r.stderr[-200:])
    check("with --out the summaries are on stderr, not stdout", "reference: " in r.stderr and r.stdout == "")
    # The one run that passes: dy all zeros, so dx is zeros and so is the mock's answer.
    zdy = os.path.join(d, "zdy")
    open(zdy, "w").write("0 0\n0 0\n")
    r = run([kx, "r=2", "n=2", "k=3", "--in", "dy", zdy, "--in", "w", w, "--out", "dx"], log)
    check("a passing comparison prints the device's --out", r.returncode == 0 and r.stdout == "0.0 0.0 0.0 0.0 0.0 0.0\n",
          f"exit {r.returncode} {r.stdout!r} {r.stderr[-120:]}")
    r = run([kx, "r=2", "n=2", "k=3", "--in", "dy", zdy, "--in", "w", w, "--out", "dx", "--out", "dx"], log)
    check("two --out, a line each", r.stdout == "0.0 0.0 0.0 0.0 0.0 0.0\n" * 2, repr(r.stdout))

    # Refusals (docs/run.md section 3), each with its rule, status 1, nothing on stdout.
    for name, text in [("three", "1 2 3\n"), ("nan", "1 2 3 x\n"), ("five", "1 2 3 4 5\n"), ("plus", "1 +2\n3 4\n")]:
        open(os.path.join(d, name), "w").write(text)
    D = lambda n: os.path.join(d, n)
    base = [kx, "r=2", "n=2", "k=3"]
    both = ["--in", "dy", dy, "--in", "w", w]
    refusals = [
        ("usage", "a read-only parameter not given", 'read by the kernel', base + ["--in", "dy", dy, "--out", "dx"]),
        ("usage", "too few numbers", 'has 3 numbers', base + ["--in", "dy", D("three"), "--in", "w", w, "--out", "dx"]),
        ("usage", "too many numbers", 'more than the 4 numbers', base + ["--in", "dy", D("five"), "--in", "w", w, "--out", "dx"]),
        ("usage", "a token that is not a number", 'is not a number', base + ["--in", "dy", D("nan"), "--in", "w", w, "--out", "dx"]),
        ("usage", "an explicit plus sign", 'is not a number', base + ["--in", "dy", D("plus"), "--in", "w", w, "--out", "dx"]),
        ("usage", "--in names an unknown parameter", 'which is not a parameter', base + ["--in", "nosuch", dy] + both + ["--out", "dx"]),
        ("usage", "--in names a writable parameter", 'which the kernel writes', base + ["--in", "dx", dy] + both + ["--out", "dx"]),
        ("usage", "--in given twice", 'twice', base + ["--in", "dy", dy] + both + ["--out", "dx"]),
        ("usage", "--out names a read-only parameter", 'only reads', base + both + ["--out", "dy"]),
        ("usage", "--out names an unknown parameter", 'which is not a parameter', base + both + ["--out", "nosuch"]),
        ("usage", "--in without its file", '--in needs a parameter name and a file', base + ["--in", "dy"]),
        ("usage", "--out without its name", '--out needs a parameter name', base + both + ["--out"]),
        ("usage", "--target is not offered", '`--target` is not name=value', base + both + ["--target", "nvidia-ada", "--out", "dx"]),
        ("usage", "a constant that is not name=value", '`junk` is not name=value', base + ["junk"] + both + ["--out", "dx"]),
        ("io", "an input file that does not exist", 'cannot read', base + ["--in", "dy", D("absent"), "--in", "w", w, "--out", "dx"]),
        ("io", "a source file that does not exist", 'cannot read', [D("absent.chx")] + both),
    ]
    for rule, label, needle, argv in refusals:
        r = run(argv, log)
        good = r.returncode == 1 and f"error[{rule}]" in r.stderr and needle in r.stderr and r.stdout == ""
        check(f"refused: {label} ({rule})", good, f"exit {r.returncode}, stderr {r.stderr[:100]!r}, stdout {r.stdout[:40]!r}")
        check(f"  ... and nothing reached the device: {label}", not os.path.exists(log))

sys.exit(1 if failed else 0)
