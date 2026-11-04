#!/usr/bin/env python3
"""Step `shuffle_repro` of docs/cloud.md: the two NVFP4 lookup-shuffle bugs the CUDA
emulator found (docs/device.md 5b), on the real GPU.

    python3 scripts/gcp/shuffle_repro.py --bin build/cancho-gpu-device [--timeout 60] [--json FILE]

Each line of scripts/gcp/shuffle_repro.txt runs in ITS OWN process through
`cancho-gpu-device run` (interpreter vs the GPU, the program's 2e-2 threshold), with
a per-case timeout after which the process group is killed and the case is TIMEOUT.
`nvidia-smi` is queried between cases; if it does not answer, or the GPU reports
an Xid since the step began, the case that preceded it is recorded as WEDGED and the
step stops cleanly (the VM is deleted anyway).

Statuses: PASS, MISMATCH (error[device-mismatch]), TIMEOUT, ERROR (error[device]:
driver, NVRTC or launch), REFUSED (the compiler's own rule, not a finding), WEDGED.
Exit 1 when anything but PASS/REFUSED occurred: on THIS step FAIL means "the bug
was seen on hardware". Standard library only."""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def cases(path):
    out = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if line:
            out.append(line.split())
    return out


def run_one(binary, parts, timeout):
    t0 = time.time()
    p = subprocess.Popen([binary, "run", os.path.join(ROOT, parts[0])] + parts[1:], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, cwd=ROOT, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        try:
            p.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            pass
        return {"status": "TIMEOUT", "detail": f"no answer in {timeout}s (a hung kernel, or the CPU reference)", "seconds": timeout}
    sec = round(time.time() - t0, 1)
    m = re.search(r"worst difference ([0-9.e+-]+) of peak ([0-9.e+-]+)", out + err)
    if p.returncode == 0:
        return {"status": "PASS", "detail": m.group(0) if m else "", "seconds": sec,
                "err": m.group(1) if m else "", "peak": m.group(2) if m else ""}
    first = (err.strip().splitlines() or [f"exit {p.returncode}, no message"])[0]
    tag = re.match(r"error\[([a-z-]+)\]", first)
    if tag and tag.group(1) == "device-mismatch":
        mm = re.search(r"([0-9.e+-]+) .*peak ([0-9.e+-]+)", first)
        return {"status": "MISMATCH", "detail": first[:200], "seconds": sec,
                "err": mm.group(1) if mm else "", "peak": mm.group(2) if mm else ""}
    if tag and tag.group(1) == "device":
        return {"status": "ERROR", "detail": first[:200], "seconds": sec}
    return {"status": "REFUSED", "detail": first[:120], "seconds": sec}


def gpu_alive():
    """(ok, text): does nvidia-smi answer, and has the driver logged an Xid (read from
    the kernel log when allowed)?"""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False, "nvidia-smi did not answer"
    if r.returncode:
        return False, "nvidia-smi: " + r.stderr.strip()[:120]
    return True, r.stdout.strip()


def xids():
    try:
        r = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return 0
    return len(re.findall(r"NVRM: Xid", r.stdout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--cases", default=os.path.join(ROOT, "scripts", "gcp", "shuffle_repro.txt"))
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--no-smi", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    check_gpu = not a.no_smi
    x0 = xids()
    rows, wedged = [], False
    for parts in cases(a.cases):
        name = " ".join(parts)
        r = run_one(a.bin, parts, a.timeout)
        r["case"] = name
        if check_gpu:
            ok, text = gpu_alive()
            r["gpu_after"] = text
            if not ok or xids() > x0:
                r["status"] = "WEDGED" if r["status"] in ("TIMEOUT", "ERROR", "MISMATCH") or not ok else r["status"]
                r["detail"] += f" | GPU after: {text}; Xid events since start: {xids() - x0}"
                wedged = True
        rows.append(r)
        print(f"{r['status']:9} {name:58} {r['seconds']:>5}s  err {r.get('err', ''):>10} peak {r.get('peak', ''):>8}  {r['detail']}", flush=True)
        if wedged:
            print("the GPU is not answering after this case: recorded, stopping the step here", flush=True)
            break
    count = {s: sum(1 for r in rows if r["status"] == s) for s in ("PASS", "MISMATCH", "TIMEOUT", "ERROR", "REFUSED", "WEDGED")}
    print("\n== table ==")
    for r in rows:
        print(f"{r['status']:9} {r['case']:58} max error {r.get('err', '-'):>10} of peak {r.get('peak', '-'):>8}")
    print("\n" + "  ".join(f"{k} {v}" for k, v in count.items()) + f"  (of {len(rows)} run)")
    bad = count["MISMATCH"] + count["TIMEOUT"] + count["ERROR"] + count["WEDGED"]
    print(f"BUG SEEN ON HARDWARE in {bad} case(s)" if bad else "no case misbehaved on this GPU (not a proof: the guide calls these undefined)")
    if a.json:
        with open(a.json, "w") as h:
            json.dump({"counts": count, "rows": rows}, h, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
