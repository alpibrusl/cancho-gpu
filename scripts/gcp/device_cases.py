#!/usr/bin/env python3
"""Step 4 of docs/cloud.md: every case of tests/cases.txt through the REAL GPU.

    python3 scripts/gcp/device_cases.py --bin build/cancho-gpu-device [--timeout 300] [--jobs 4] [--json FILE]

For each case (a `.chx` file and its constants; the `--run` lines are the
interpreter's own record and are skipped, so are cases the golden record shows
have no CUDA lowering) it runs `cancho-gpu-device run`, which runs the reference
interpreter, compiles the emitted CUDA with NVRTC, launches it and holds the
outputs to the interpreter at the program's own threshold (2e-2 of the output's
peak, docs/device.md section 3). Prints a pass/fail table. A case whose CPU
reference is too slow (the big GEMMs) is TIMEOUT, which is not a failure of the
GPU and is counted apart. Exit 0 when no case is a MISMATCH or an ERROR.

The device program is the thing under test: this script adds no tolerance of its
own. Standard library only."""
import argparse
import concurrent.futures
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def cases(path):
    seen, out = set(), []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if "--run" in parts:
            continue
        key = tuple(parts)
        if key in seen:
            continue
        seen.add(key)
        out.append(parts)
    return out


def case_name(parts):
    # scripts/case-name.sh: the file's name without `.chx`, then its constants, joined by `_`.
    return "_".join([os.path.basename(parts[0])[: -len(".chx")]] + parts[1:])


def run_case(binary, parts, timeout):
    t0 = time.time()
    try:
        p = subprocess.run([binary, "run", os.path.join(ROOT, parts[0])] + parts[1:], capture_output=True,
                           text=True, timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "detail": f"over {timeout}s (the CPU reference is the slow part)", "seconds": timeout}
    sec = round(time.time() - t0, 1)
    err = p.stderr.strip().splitlines()
    if p.returncode == 0:
        m = re.search(r"worst difference ([0-9.e+-]+) of peak ([0-9.e+-]+)", p.stdout + p.stderr)
        return {"status": "PASS", "detail": m.group(0) if m else "", "seconds": sec}
    first = err[0] if err else f"exit {p.returncode}, no message"
    if "error[device-mismatch]" in first:
        return {"status": "MISMATCH", "detail": first, "seconds": sec}
    return {"status": "ERROR", "detail": first, "seconds": sec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--cases", default=os.path.join(ROOT, "tests", "cases.txt"))
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--filter", default="", help="only cases whose name contains this")
    ap.add_argument("--json")
    a = ap.parse_args()
    todo, no_cuda = [], []
    for parts in cases(a.cases):
        name = case_name(parts)
        if a.filter and a.filter not in name:
            continue
        if not glob.glob(os.path.join(ROOT, "golden", name, "*.cu")):
            no_cuda.append(name)
            continue
        todo.append((name, parts))
    print(f"{len(todo)} cases through the device ({len(no_cuda)} skipped: no CUDA lowering recorded), "
          f"{a.jobs} at a time, {a.timeout}s each")
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_case, a.bin, parts, a.timeout): name for name, parts in todo}
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            r["case"] = futs[f]
            rows.append(r)
            print(f"{r['status']:8} {r['case']:60} {r['seconds']:>6}s  {r['detail']}", flush=True)
    rows.sort(key=lambda r: r["case"])
    count = {s: sum(1 for r in rows if r["status"] == s) for s in ("PASS", "MISMATCH", "ERROR", "TIMEOUT")}
    print("\n== per-case table (sorted) ==")
    for r in rows:
        print(f"{r['status']:8} {r['case']:60} {r['detail']}")
    print(f"\nPASS {count['PASS']}  MISMATCH {count['MISMATCH']}  ERROR {count['ERROR']}  TIMEOUT {count['TIMEOUT']}"
          f"  (of {len(rows)})")
    if a.json:
        with open(a.json, "w") as h:
            json.dump({"counts": count, "skipped_no_cuda": no_cuda, "rows": rows}, h, indent=1)
    return 1 if count["MISMATCH"] or count["ERROR"] or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
