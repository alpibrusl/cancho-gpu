#!/usr/bin/env python3
"""Where a session's device memory goes (docs/resident.md section 23.4): the bytes of every program output (aliased outputs share their root's buffer), resident file and table of a session file, from
the declared shapes of the kernels, grouped by what they are.

    python3 examples/qwen/session_memory.py SESSION [--top 25]

Computed, not measured: it adds up what the session file declares and is held to the process's peak footprint by the caller."""
import os
import re
import sys
from collections import defaultdict

BYTES = {"f32": 4, "f16": 2, "i32": 4}


def kernel_outputs(path, args):
    """[(name, bytes)] of the outputs of the kernel in `path` with the integer arguments `args`."""
    text = open(path).read()
    outs = []
    for m in re.finditer(r"^\s*out\s+(\w+)\s*:\s*(\w+)\[([^\]]*)\]", text, re.M):
        name, ty, dims = m.groups()
        n = 1
        for e in dims.split(","):
            n *= int(eval(e.strip().replace("/", "//"), {}, dict(args)))
        outs.append((name, n * BYTES[ty]))
    return outs


def parse(session):
    base = os.path.dirname(session)
    progs, alias, res, tabs = {}, {}, [], []
    for line in open(session):
        w = line.split()
        if not w or w[0].startswith("#"):
            continue
        if w[0] == "program":
            progs[w[1]] = (w[2], {k: v for k, v in (x.split("=") for x in w[3:])})
        elif w[0] == "alias":
            alias[w[2]] = w[1]
        elif w[0] in ("resident", "table"):
            res.append((w[0], w[1], w[2]))
    return base, progs, alias, res


def main():
    session = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
    base, progs, alias, res = parse(session)
    root = lambda o: root(alias[o]) if o in alias else o
    sizes = {}
    for pn, (kpath, kv) in progs.items():
        args = {}
        for k, v in kv.items():
            try:
                args[k] = int(v)
            except ValueError:
                pass
        for on, b in kernel_outputs(kpath if os.path.isabs(kpath) else os.path.join(base, kpath), args):
            sizes[f"{pn}.{on}"] = b
    buf = defaultdict(int)
    for o, b in sizes.items():
        buf[root(o)] = max(buf[root(o)], b)
    kinds = defaultdict(int)
    for o, b in buf.items():
        kinds[re.sub(r"_?\d+", "", o.split(".")[0]) + "." + o.split(".")[1]] += b
    total_out = sum(buf.values())
    files = defaultdict(int)
    for kind, target, path in res:
        files[kind] += os.path.getsize(path) if os.path.exists(path) else 0
    uniq = {}
    for kind, target, path in res:
        if os.path.exists(path):
            uniq[path] = os.path.getsize(path)
    print(f"program outputs: {total_out / 1e9:.2f} GB in {len(buf)} buffers (of {len(sizes)} outputs; {len(alias)} aliases)")
    print(f"resident files: {sum(s for k, p, _ in res if k == 'resident' for s in [uniq.get(_, 0)]) / 1e9:.2f} GB in {sum(1 for k, _, _ in res if k == 'resident')} declarations "
          f"({sum(uniq.values()) / 1e9:.2f} GB distinct files)")
    print(f"tables: {files['table'] / 1e9:.2f} GB")
    print("largest kinds of output (the kernel's output summed over layers):")
    for k, v in sorted(kinds.items(), key=lambda x: -x[1])[:top]:
        print(f"  {k:28s} {v / 1e9:7.3f} GB")


if __name__ == "__main__":
    main()
