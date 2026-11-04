#!/usr/bin/env python3
"""The cases the interpreter cannot hold, through the emulator without a
reference (docs/device.md section 5b). `cancho-gpu-device run` compares with
the interpreter, which cannot hold the model-sized cases (error[interp]); this
runs their emitted kernel on the emulating driver at the real size with
random inputs and checks what needs no reference: no illegal or misaligned
access, no deadlock, no bytes written outside a buffer, bit-identical results
from the two schedules (no race), and an output that is finite and not zero.

    python3 tests/cuda_emu_big.py                   # every `noref` case of tests/cuda-emu-known.txt
    python3 tests/cuda_emu_big.py tests/chx/matvec_fp4_rows.chx n=17408 k=5120

Slow (minutes a case: the grids are real); not in the gate.
"""
import array
import os
import random
import re
import subprocess
import struct
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "emu"))
import emupy  # noqa: E402

ROOT = emupy.ROOT
SIZE = {"f32": 4, "f16": 2, "i8": 1}


def params(chx, consts):
    text = Path(chx).read_text()
    ps = []
    for kind, name, ty, dims in re.findall(r"^\s*(in|out|inout)\s+(\w+)\s*:\s*(\w+)\[([^\]]*)\]", text, re.M):
        n = 1
        for d in dims.split(","):
            n *= eval(d.strip().replace("/", "//"), {}, dict(consts))
        ps.append((kind, name, ty, n))
    return ps


def fill(ty, n, rnd):
    if ty == "i8":
        return bytes(rnd.getrandbits(8) for _ in range(min(n, 1 << 16))) * (n // (1 << 16) + 1)
    pool = [rnd.uniform(-1, 1) for _ in range(4096)]
    if ty == "f32":
        a = array.array("f", pool)
        return (a.tobytes() * (n // 4096 + 1))[: 4 * n]
    h = b"".join(struct.pack("<e", x) for x in pool)
    return (h * (n // 4096 + 1))[: 2 * n]


def finite(ty, data):
    if ty == "f32":
        v = array.array("f", data)
        return all(x - x == 0 for x in v), any(v)
    v = struct.unpack("<%de" % (len(data) // 2), data)
    return all(x - x == 0 for x in v), any(v)


def one(chx, consts_args, passes):
    consts = {k: int(v) for k, v in (a.split("=") for a in consts_args if not a.startswith("--"))}
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run([str(ROOT / "build/cancho-gpu"), "emit", str(ROOT / chx), d] + consts_args, capture_output=True, text=True)
        m = re.search(r"(\S+\.cu): (\d+) threads, grid (\d+)x(\d+)", out.stdout)
        if not m:
            return False, "emit: " + out.stderr.strip()[:200]
        src = Path(m.group(1)).read_text()
        threads, gx, gy = int(m.group(2)), int(m.group(3)), int(m.group(4))
        entry = re.search(r'__global__ void (\w+)\(', src).group(1)
    rnd = random.Random(1)
    ps = params(ROOT / chx, consts)
    bufs = [(fill(ty, n, rnd) if kind != "out" else bytes(n * SIZE[ty])) for kind, _, ty, n in ps]
    t0 = time.time()
    r, outs = emupy.run(src, entry, (gx, gy), threads, bufs, passes=passes)
    dt = time.time() - t0
    if r.rc != 0:
        return False, f"{r.err} ({dt:.0f}s) {r.stderr.strip()[:200]}"
    if "emu-race" in r.stderr:
        return False, "race: " + r.stderr.strip()[:300]
    for (kind, name, ty, n), data in zip(ps, outs):
        if kind != "in":
            ok, nonzero = finite(ty, data)
            if not ok or not nonzero:
                return False, f"output {name} is {'not finite' if not ok else 'all zero'}"
    return True, f"grid {gx}x{gy} x {threads} threads, {sum(len(b) for b in bufs) // (1 << 20)} MiB, {dt:.0f}s for {passes} schedules" + (f"; {r.stderr.strip()[:120]}" if r.stderr.strip() else "")


def main():
    passes = int(os.environ.get("CANCHOGPU_EMU_PASSES", "2"))
    args = sys.argv[1:]
    if args:
        jobs = [(args[0], args[1:])]
    else:
        jobs = []
        for line in (ROOT / "tests/cases.txt").read_text().splitlines():
            if line and not line.startswith("#"):
                f, *rest = line.split()
                name = subprocess.run([str(ROOT / "scripts/case-name.sh"), f] + rest, capture_output=True, text=True).stdout.strip()
                known = [l.split("|") for l in (ROOT / "tests/cuda-emu-known.txt").read_text().splitlines() if "|" in l and not l.startswith("#")]
                if any(k[0] == name and k[1] == "noref" for k in known):
                    jobs.append((f, rest))
    bad = 0
    for f, rest in jobs:
        ok, msg = one(f, rest, passes)
        print(("ok   " if ok else "FAIL ") + f"{Path(f).stem} {' '.join(rest)}: {msg}", flush=True)
        bad += not ok
    print(f"{len(jobs) - bad} of {len(jobs)} model-sized kernels clean on the emulator")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
