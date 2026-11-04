#!/usr/bin/env python3
"""Checks on the emitted CUDA that need no hardware and no compiler
(docs/device.md section 5b): over every golden/*/*.cu,

  * static __shared__ per block is at most 48 KB. NVRTC refuses a larger
    static allocation; the Ada opt-in to 99 KB needs dynamic shared memory
    and cuFuncSetAttribute, which the shim never asks for. The kernels'
    own header comment states a budget of 101376 B (99 KB): a kernel
    between the two would be one the compiler accepts and the target
    cannot compile;
  * what the header comment says the kernel uses (`tg mem`) is what it
    declares;
  * threads per block are at most 1024 (and the launch is 1-D);
  * the source uses nothing sm_89 lacks: Hopper's clusters, TMA, wgmma,
    setmaxnreg, Blackwell's tcgen05 and fp4/fp6 types, inline PTX at all
    (the emitter writes none, so any is new and unreviewed), and only the
    two headers the emulator and the toolkit both provide;
  * exactly one entry, `extern "C" __global__ void`, every parameter a
    `__restrict__` pointer (the shim passes only device pointers).

    python3 tests/cuda_static.py [golden-dir]

Prints one line per finding and a summary; exit 1 on any finding.
"""
import re
import sys
from pathlib import Path

STATIC_SHARED_LIMIT = 48 * 1024
OPT_IN_LIMIT = 99 * 1024
MAX_THREADS = 1024
SIZES = {"float": 4, "float2": 8, "float4": 16, "uint": 4, "int": 4, "uchar": 1, "__half": 2, "uint4": 16, "double": 8}
NOT_SM89 = [
    (r"__cluster_dims__|cluster_group|cooperative_groups::this_cluster|barrier\.cluster", "thread-block clusters (sm_90)"),
    (r"\bwgmma\b|wgmma\.", "wgmma (sm_90)"),
    (r"cp\.async\.bulk|cuda::memcpy_async.*tensor|CUtensorMap|cp\.reduce\.async\.bulk", "TMA / bulk async copies (sm_90)"),
    (r"setmaxnreg|elect\.sync|fence\.proxy\.async", "sm_90 PTX"),
    (r"tcgen05|\btmem\b|__nv_fp4|__nv_fp6|e2m1x2|e3m2|e2m3", "Blackwell (sm_100) features"),
    (r"\basm\b|__asm__", "inline PTX (the emitter writes none)"),
    (r"__nv_bfloat16_raw.*cvt\.rn\.satfinite", "sm_100 conversions"),
]
ALLOWED_INCLUDES = {"cuda_fp16.h", "mma.h"}


def check(path):
    src = path.read_text()
    out = []
    name = path.parent.name
    head = "\n".join(src.splitlines()[:8])
    m = re.search(r"// launch : (\d+) threadgroups x (\d+) threads", head)
    if not m:
        out.append("no `// launch :` header")
    else:
        if int(m.group(2)) > MAX_THREADS:
            out.append(f"{m.group(2)} threads per block (max {MAX_THREADS})")
    shared = 0
    arena = scratch = 0
    for decl in re.finditer(r"__shared__\s+(?:__align__\(\d+\)\s+)?(\w+)\s+(\w+)\[(\d+)\]", src):
        t, n, c = decl.group(1), decl.group(2), int(decl.group(3))
        if t not in SIZES:
            out.append(f"__shared__ of unknown type {t}")
            continue
        b = SIZES[t] * c
        shared += b
        if n.startswith("arena"):
            arena += b
        else:
            scratch += b
    if len(re.findall(r"__shared__", src)) != len(re.findall(r"__shared__\s+(?:__align__\(\d+\)\s+)?\w+\s+\w+\[\d+\]", src)):
        out.append("a __shared__ declaration the checker cannot size")
    if shared > STATIC_SHARED_LIMIT:
        out.append(f"{shared} B of static __shared__ (the limit is {STATIC_SHARED_LIMIT}; "
                   f"{'the 99 KB opt-in needs dynamic shared memory and cuFuncSetAttribute, which the shim never does' if shared <= OPT_IN_LIMIT else 'past even the 99 KB opt-in'})")
    mem = re.search(r"// tg mem : (\d+) B tiles \+ (\d+) B scratch of (\d+) B", head)
    if mem:
        tiles, scr = int(mem.group(1)), int(mem.group(2))
        if arena < tiles or scratch < scr:
            out.append(f"header says {tiles} B tiles + {scr} B scratch but the source declares {arena} + {scratch}")
    for rx, what in NOT_SM89:
        if re.search(rx, src):
            out.append(f"uses {what}")
    for inc in re.findall(r"#include\s*[<\"]([^>\"]+)[>\"]", src):
        if inc not in ALLOWED_INCLUDES:
            out.append(f"includes <{inc}>")
    entries = re.findall(r'extern "C" __global__ void (\w+)\(([^)]*)\)', src)
    if len(entries) != 1 or src.count("__global__") != 1:
        out.append(f"{len(entries)} extern \"C\" __global__ entries")
    else:
        for p in entries[0][1].split(","):
            if "*" not in p or "__restrict__" not in p:
                out.append(f"parameter `{p.strip()}` is not a __restrict__ pointer")
    return name, shared, out


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "golden")
    files = sorted(root.glob("*/*.cu"))
    bad = 0
    top = (0, "")
    for f in files:
        name, shared, findings = check(f)
        top = max(top, (shared, name))
        for x in findings:
            print(f"{name}: {x}")
            bad += 1
    print(f"{len(files)} kernels checked statically; most static __shared__: {top[0]} B ({top[1]}); {bad} findings")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
