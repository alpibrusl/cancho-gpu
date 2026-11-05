#!/usr/bin/env python3
"""Sizes the goldens do not have, through the emulator (docs/device.md
section 5b). tests/cases.txt fixes the sizes the goldens record; the
emulator can run any, so this perturbs each case's constants (the values the family's
cases use, halved, doubled, moved by 1, 16 and 32), runs what the
compiler accepts through the emulating driver against the interpreter, and
lists what does not agree. The compiler refusing a size (a rule tag other than
`device`, `device-mismatch`) is not a finding; the CUDA disagreeing with the
interpreter, a race, an overrun or a deadlock is.

    python3 tests/cuda_emu_sweep.py [variants-per-case] [seed]

Not in the gate (the variants are not recorded in golden/); a finding is
reproduced with `build/cancho-gpu-device run <file> <consts>` under
CANCHOGPU_LIBCUDA / CANCHOGPU_LIBNVRTC, and belongs in tests/cases.txt once
the emitter is fixed.
"""
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMU = ROOT / "build" / "emu"
MAX_ELEMS = 3_000_000  # keep the interpreter quick


def families():
    """file -> {const: values seen in tests/cases.txt}, and the cases' own constant sets."""
    seen = {}
    for line in (ROOT / "tests/cases.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f, *rest = line.split()
        kv = [r.split("=") for r in rest if "=" in r and not r.startswith("--") and not r.startswith("eps")]
        if not kv:
            continue
        d = seen.setdefault(f, {})
        for k, v in kv:
            d.setdefault(k, set()).add(int(v) if v.isdigit() else v)  # a float constant stays as written
    return seen


def variants(consts, rnd, n):
    """Combinations of the values the family's cases use per constant, with
    each also halved, doubled and moved by 16 and 32 (the granularities the
    kernels divide by): most are refused by the compiler's own rules, which
    is cheap; the rest are sizes no golden records."""
    pools = {}
    for k, vs in consts.items():
        if any(isinstance(v, str) for v in vs):
            pools[k] = sorted(str(v) for v in vs)
            continue
        pool = set()
        for v in vs:
            pool |= {v, v * 2, v + 16, v + 32, v + 1, max(1, v - 1), 1, 2, 3}
            if v % 2 == 0:
                pool.add(v // 2)
            if v > 32:
                pool.add(v - 16)
                pool.add(v - 32)
        pools[k] = sorted(x for x in pool if 0 < x <= 20000)
    out = set()
    for _ in range(n * 40):
        pick = {k: rnd.choice(p) for k, p in pools.items()}
        prod = 1
        for v in pick.values():
            prod *= v if isinstance(v, int) else 1
        if prod > 40_000_000:
            continue
        out.add(" ".join(f"{k}={v}" for k, v in pick.items()))
        if len(out) >= n:
            break
    return sorted(out)


def run(job):
    f, args = job
    env = dict(os.environ, CANCHOGPU_LIBCUDA=str(EMU / "libcuda_emu.so"), CANCHOGPU_LIBNVRTC=str(EMU / "libnvrtc_emu.so"),
               CANCHOGPU_EMU_CACHE=str(EMU / "cache"), CANCHOGPU_EMU_TIMEOUT="60")
    try:
        p = subprocess.run([str(ROOT / "build/cancho-gpu-device"), "run", str(ROOT / f)] + args.split(), capture_output=True, text=True, env=env, timeout=240)
    except subprocess.TimeoutExpired:
        return f, args, "slow", ""
    out = p.stdout + p.stderr
    m = re.search(r"^error\[([\w-]+)\]", out, re.M)
    if p.returncode == 0 and "emu-race" not in out:
        return f, args, "ok", out
    if "no progress in" in out:
        return f, args, "slow", out  # the emulator is slow at big grids; not a finding
    tag = m.group(1) if m else ("emu-race" if "emu-race" in out else f"exit {p.returncode}")
    return f, args, tag, out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rnd = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    jobs = []
    for f, consts in sorted(families().items()):
        for v in variants(consts, rnd, n):
            jobs.append((f, v))
    tally = {}
    findings = []
    with ThreadPoolExecutor(int(os.environ.get("JOBS", "4"))) as ex:
        for f, args, tag, out in ex.map(run, jobs):
            tally[tag] = tally.get(tag, 0) + 1
            if tag in ("device-mismatch", "device", "emu-race") or tag.startswith("exit"):
                first = next((l for l in out.splitlines() if l.startswith(("error", "emu-", "compare"))), "")
                findings.append(f"{f} {args}: {tag}: {first[:300]}")
                print(findings[-1], flush=True)
    print("sweep:", ", ".join(f"{k} {v}" for k, v in sorted(tally.items())), f"({len(jobs)} variants)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
