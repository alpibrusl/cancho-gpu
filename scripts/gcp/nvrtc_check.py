#!/usr/bin/env python3
"""Step 3 of docs/cloud.md: does the CUDA this compiler emits compile under the
REAL NVRTC, for compute_89? No GPU is needed to answer, and none is used.

    python3 scripts/gcp/nvrtc_check.py [--arch compute_89] [--json FILE] [--ptxas] FILE.cu ...

One line per file, then the first compile errors of each failing one. With
--ptxas, and `ptxas` on PATH (the CUDA toolkit), the PTX is also assembled for
sm_89 and its registers and spills reported: ptxas states them directly, which
is what Metal never does. Exit 0 when every file compiles, 1 otherwise, 2 for a
usage error or no NVRTC found. Standard library only (ctypes)."""
import argparse
import ctypes
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


def find_nvrtc():
    names = ["libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12"]
    dirs = [os.environ.get("CUDA_HOME", ""), "/usr/local/cuda", "/usr/local/cuda-12", "/usr/local/cuda-13"]
    cands = []
    for d in dirs:
        if d:
            cands += glob.glob(os.path.join(d, "lib64", "libnvrtc.so*"))
    # The pip wheels (nvidia-cuda-nvrtc) put it beside the interpreter's site-packages.
    for p in sys.path:
        cands += glob.glob(os.path.join(p, "nvidia", "*", "lib", "libnvrtc.so*"))
    for c in names + cands:
        try:
            return ctypes.CDLL(c), c
        except OSError:
            pass
    return None, None


def include_dirs():
    out = []
    for d in [os.environ.get("CUDA_HOME", ""), "/usr/local/cuda", "/usr"]:
        if d and os.path.exists(os.path.join(d, "include", "cuda_fp16.h")):
            out.append(os.path.join(d, "include"))
    for p in sys.path:
        out += glob.glob(os.path.join(p, "nvidia", "*", "include"))
    return out


def compile_one(lib, src, arch, incs):
    c = ctypes
    prog = c.c_void_p()
    if lib.nvrtcCreateProgram(c.byref(prog), src, b"k.cu", 0, None, None):
        return None, "nvrtcCreateProgram failed"
    opts = [("--gpu-architecture=" + arch).encode()] + [("-I" + i).encode() for i in incs]
    arr = (c.c_char_p * len(opts))(*opts)
    rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
    n = c.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, c.byref(n))
    log = c.create_string_buffer(n.value)
    lib.nvrtcGetProgramLog(prog, log)
    ptx = None
    if rc == 0:
        lib.nvrtcGetPTXSize(prog, c.byref(n))
        buf = c.create_string_buffer(n.value)
        lib.nvrtcGetPTX(prog, buf)
        ptx = buf.value
    lib.nvrtcDestroyProgram(c.byref(prog))
    return ptx, log.value.decode(errors="replace")


def ptxas_report(ptx):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "k.ptx")
        with open(p, "wb") as f:
            f.write(ptx)
        r = subprocess.run(["ptxas", "-v", "--gpu-name", "sm_89", p, "-o", os.path.join(d, "k.cubin")],
                           capture_output=True, text=True)
    text = r.stderr
    regs = re.search(r"Used (\d+) registers", text)
    spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", text)
    return {
        "ptxas_rc": r.returncode,
        "registers": int(regs.group(1)) if regs else None,
        "spill_bytes": int(spill.group(1)) + int(spill.group(2)) if spill else None,
        "ptxas_error": text.strip().splitlines()[-1] if r.returncode else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="compute_89")
    ap.add_argument("--json")
    ap.add_argument("--ptxas", action="store_true")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    lib, where = find_nvrtc()
    if lib is None:
        print("no libnvrtc found (CUDA_HOME, /usr/local/cuda/lib64, pip nvidia-cuda-nvrtc)", file=sys.stderr)
        return 2
    major, minor = ctypes.c_int(), ctypes.c_int()
    lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    incs = include_dirs()
    use_ptxas = a.ptxas and shutil.which("ptxas") is not None
    print(f"NVRTC {major.value}.{minor.value} ({where}), arch {a.arch}, includes {incs}, ptxas {'yes' if use_ptxas else 'no'}")
    rows, bad = [], 0
    for f in sorted(a.files):
        name = os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f)
        with open(f, "rb") as h:
            ptx, log = compile_one(lib, h.read(), a.arch, incs)
        row = {"file": name, "ok": ptx is not None, "log": log if ptx is None else ""}
        if ptx is not None and use_ptxas:
            row.update(ptxas_report(ptx))
        rows.append(row)
        extra = ""
        if "registers" in row:
            extra = f"  regs {row['registers']}, spill {row['spill_bytes']} B"
        print(f"{'ok  ' if row['ok'] else 'FAIL'} {name}{extra}")
        if not row["ok"]:
            bad += 1
            for line in log.strip().splitlines()[:6]:
                print("      " + line)
    spilled = [r for r in rows if r.get("spill_bytes")]
    print(f"{len(rows) - bad} of {len(rows)} compile under NVRTC {major.value}.{minor.value} for {a.arch}"
          + (f"; {len(spilled)} spill registers" if use_ptxas else ""))
    if a.json:
        with open(a.json, "w") as h:
            json.dump({"nvrtc": f"{major.value}.{minor.value}", "arch": a.arch, "rows": rows}, h, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
