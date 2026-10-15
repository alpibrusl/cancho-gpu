#!/usr/bin/env python3
"""The oracle of `cancho-gpu safetensors`: the `safetensors` library (its
Rust reader, which validates the file as the format defines it), printing
the canonical dump of `src/safetensors_dump.cho` (docs/runtime.md §8).

    safetensors_dump.py <file.safetensors>

Shapes and dtypes come from the library; the byte ranges, which it does not
expose, from Python's own `json` module over the raw header. A file the
library refuses prints `REFUSED <its message>` and exits 1.
"""
import json
import os
import struct
import sys

from safetensors import safe_open


def quoted(s):
    out = '"'
    for b in s.encode():
        if b in (34, 92):
            out += "\\" + chr(b)
        elif b < 32 or b >= 127:
            out += "\\x%02x" % b
        else:
            out += chr(b)
    return out + '"'


path = sys.argv[1]
try:
    f = safe_open(path, framework="pt")
except Exception as e:  # the library's own refusal
    print("REFUSED", str(e).replace("\n", " "))
    sys.exit(1)
raw = open(path, "rb").read()
n = struct.unpack("<Q", raw[:8])[0]
header = json.loads(raw[8:8 + n])
meta = header.pop("__metadata__", None) or {}
print("safetensors header=%d tensors=%d data=%d" % (n, len(header), os.path.getsize(path) - 8 - n))
for k in sorted(meta, key=lambda s: s.encode()):
    print("meta %s %s" % (k, quoted(meta[k])))
for name in sorted(f.keys(), key=lambda s: s.encode()):
    sl = f.get_slice(name)
    lo, hi = header[name]["data_offsets"]
    print("tensor %s %s shape=[%s] range=[%d,%d]" % (name, sl.get_dtype(), ",".join(str(d) for d in sl.get_shape()), lo, hi))
