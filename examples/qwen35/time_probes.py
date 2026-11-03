#!/usr/bin/env python3
"""Times the probe kernels of examples/qwen35/probes/ on the Metal device at the model's sizes (docs/roadmap-models.md M10, section 'probes'), with
zero-filled inputs (a kernel's time here does not depend on its values). Not in the gate: it needs a Mac and the GPU.

    python3 examples/qwen35/time_probes.py build/cancho-gpu-metal-session [--batch 4] [--reps 5]

CANCHOGPU_PROFILE=2 makes the shim time each launch alone; the mean of `--reps` launches is printed. One GPU job: check `ps` first."""
import argparse
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PR = os.path.join(ROOT, "examples", "qwen35", "probes")


def zeros(d, name, count, kind):
    p = os.path.join(d, f"{name}.{kind}")
    with open(p, "wb") as f:
        f.write(b"\0" * (count * (2 if kind == "f16" else 4)))
    return p


def run(session, prog, args, inputs, reps):
    """inputs: name -> (count, kind). Returns the shim's profile line for the single program: (launches, total ms, ms a launch)."""
    d = tempfile.mkdtemp()
    lines = [f"program k {os.path.join(PR, prog)} " + " ".join(f"{a}={b}" for a, b in args.items())]
    for n, (count, kind) in inputs.items():
        lines.append(f"resident k.{n} {zeros(d, n, count, kind)} {kind}")
    out = [l.split()[1].rstrip(":") for l in open(os.path.join(PR, prog)) if l.startswith("  out ")][0]
    lines.append(f"answer k.{out} scale 0/1")
    spec = os.path.join(d, "s")
    open(spec, "w").write("\n".join(lines) + "\nsteps\n")
    env = dict(os.environ, CANCHOGPU_PROFILE="2")
    r = subprocess.run([session, spec], input=f"{reps}\n", stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env, timeout=600)
    m = re.findall(r"prof (\S+)\s+(\d+) launches\s+([0-9.]+) ms", r.stderr)
    if not m:
        print(r.stderr[-400:])
        return None
    name, n, ms = m[-1]
    return int(n), float(ms), float(ms) / int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    nchunks = a.batch * 48 * 8  # sequences x value heads x chunks of 64 at T = 512
    nb = a.batch * 48 * 2  # (sequence, head, half of the 128 values)
    cases = [
        ("kkt_mma.chx", dict(nc=nchunks), dict(kb=(nchunks * 64 * 128, "f16"), k=(nchunks * 64 * 128, "f16"))),
        ("decay_mask.chx", dict(nc=nchunks), dict(a=(nchunks * 64 * 64, "f32"), gcc=(nchunks * 64, "f32"), gcr=(nchunks * 64, "f32"), mask=(64 * 64, "f32"))),
        ("solve_fwdsub.chx", dict(nc=nchunks, nv=128), dict(ls=(nchunks * 64 * 64, "f32"), rt=(nchunks * 128 * 64, "f32"), eye=(64 * 64, "f32"))),
        ("solve_fwdsub.chx", dict(nc=nchunks, nv=64), dict(ls=(nchunks * 64 * 64, "f32"), rt=(nchunks * 64 * 64, "f32"), eye=(64 * 64, "f32"))),
        ("neumann_mma_step.chx", dict(nc=nchunks), dict(p=(nchunks * 64 * 64, "f16"), pi=(nchunks * 64 * 64, "f16"))),
        ("frag_add_window.chx", dict(nc=nchunks), dict(a=(nchunks * 64 * 64, "f16"), b=(nchunks * 64 * 64, "f16"), x=(nchunks * 64 * 64, "f32"))),
        ("neumann_matmul_chain.chx", dict(nc=nchunks), dict(n=(nchunks * 64 * 64, "f32"))),
        ("scan_mma_a.chx", dict(nb=nb), dict(s=(nb * 128 * 64, "f32"), up=(nb * 128 * 64, "f32"), el=(nb * 64, "f32"), w=(nb * 64 * 128, "f16"))),
        ("scan_mma_bo.chx", dict(nb=nb), dict(u=(nb * 64 * 64, "f32"), p=(nb * 64 * 64, "f32"), s=(nb * 128 * 64, "f32"), qg=(nb * 64 * 128, "f16"), aqk=(nb * 64 * 64, "f16"))),
        ("scan_mma_bu.chx", dict(nb=nb), dict(u=(nb * 64 * 64, "f32"), p=(nb * 64 * 64, "f32"), kd=(nb * 64 * 128, "f16"))),
        ("state_scan_matmul.chx", dict(nc=8, nb=nb), dict(u=(nb * 8 * 64 * 64, "f32"), w=(nb * 8 * 64 * 128, "f32"), aqk=(nb * 8 * 64 * 64, "f32"), qg=(nb * 8 * 64 * 128, "f32"),
                                                         kdt=(nb * 8 * 128 * 64, "f32"), el=(nb * 8 * 64, "f32"))),
    ]
    for prog, args, inputs in cases:
        res = run(a.session, prog, args, inputs, a.reps)
        print(f"{prog:28s} {args} -> " + (f"{res[2]:.3f} ms a launch ({res[0]} launches, {res[1]:.3f} ms)" if res else "no timing"))


if __name__ == "__main__":
    main()
