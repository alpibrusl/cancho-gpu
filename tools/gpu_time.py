#!/usr/bin/env python3
"""GPU time of one .chx kernel on Metal, in isolation: a session with the kernel, its inputs resident (random numbers), 40 requests,
the median of the shim's per-kernel profile (CANCHOGPU_PROFILE=2). usage: gpu_time.py KERNEL.chx name=value ... [-- param:count ...]"""
import os, random, re, statistics, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kernel = sys.argv[1]
consts = [a for a in sys.argv[2:] if "=" in a and ":" not in a]
sizes = dict(a.split(":") for a in sys.argv[2:] if ":" in a)
d = tempfile.mkdtemp()
rnd = random.Random(1)
lines = [f"program k {os.path.abspath(kernel)} " + " ".join(consts)]
outs = []
for name, n in sizes.items():
    if name.startswith("out_"):
        outs.append(name[4:])
        continue
    path = os.path.join(d, name)
    open(path, "w").write(" ".join(repr(rnd.uniform(-0.5, 0.5)) for _ in range(int(n))) + "\n")
    lines.append(f"resident k.{name} {path}")
lines += [f"answer k.{o}" for o in outs]
open(os.path.join(d, "session"), "w").write("\n".join(lines) + "\n")
env = dict(os.environ, CANCHOGPU_PROFILE="2")
r = subprocess.run([ROOT + "/build/cancho-gpu-metal-session", os.path.join(d, "session")], input="\n" * 40, capture_output=True, text=True, env=env)
if r.returncode:
    print("session failed:", r.stderr[:300]); sys.exit(1)
ts = [float(m.group(1)) for m in re.finditer(r"^prof \S+ +\d+ launches +([\d.]+) ms", r.stderr, re.M)][5:]
out = r.stdout.split("\n")[0].split()
print(f"GPU time {statistics.median(ts)*1000:.1f} us (median of {len(ts)}); first answers {out[:3]}")
