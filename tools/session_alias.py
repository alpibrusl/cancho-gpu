#!/usr/bin/env python3
"""Buffer reuse for a session file by liveness (docs/resident.md section 21.4): `alias_lines(lines)` returns the `alias P.OUT Q.OUT` directives that let every program output live in the buffer of an earlier
output of the same type and size that nothing reads any more.

A program's output joins a group when the group's last member belongs to an earlier program and all that member's readers (the programs it is `link`ed into) come before this program; an output that is `save`d,
`answer`ed or has no reader is never reused and never reuses (the session's own checks, rule `session`, refuse the rest). Greedy, in program order: it is a sound reuse, not an optimal one."""
import os
import re

BYTES = {"f32": 4, "f16": 2, "i32": 4}


def kernel_outputs(path, args):
    text = open(path).read()
    outs = []
    for m in re.finditer(r"^\s*out\s+(\w+)\s*:\s*(\w+)\[([^\]]*)\]", text, re.M):
        name, ty, dims = m.groups()
        n = 1
        for e in dims.split(","):
            n *= int(eval(e.strip().replace("/", "//"), {}, dict(args)))
        outs.append((name, ty, n * BYTES[ty]))
    return outs


def alias_lines(lines):
    order, outs, last_read, pinned = [], {}, {}, set()
    for line in lines:
        w = line.split()
        if not w:
            continue
        if w[0] == "program":
            args = {}
            for kv in w[3:]:
                k, v = kv.split("=", 1)
                try:
                    args[k] = int(v)
                except ValueError:
                    pass
            i = len(order)
            order.append(w[1])
            for on, ty, nb in kernel_outputs(w[2], args):
                outs[f"{w[1]}.{on}"] = (i, ty, nb)
        elif w[0] == "link":
            src, dst = w[1], w[3]
            last_read[src] = max(last_read.get(src, -1), order.index(dst.split(".")[0]) if dst.split(".")[0] in order else len(order))
        elif w[0] in ("save", "carry"):
            pinned.add(w[1])
        elif w[0] == "answer":
            pinned.add(w[1])
    out, groups = [], {}  # key -> list of [last member]
    for name, (i, ty, nb) in sorted(outs.items(), key=lambda kv: kv[1][0]):
        if name in pinned or name not in last_read:
            continue
        free = None
        for g in groups.setdefault((ty, nb), []):
            m = g[0]
            if outs[m][0] < i and last_read[m] < i:
                free = g
                break
        if free is None:
            groups[(ty, nb)].append([name])
        else:
            out.append(f"alias {free[0]} {name}")
            free[0] = name
    return out
