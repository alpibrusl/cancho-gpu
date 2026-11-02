#!/usr/bin/env python3
"""The oracle of `cancho-gpu gguf`: an independent reader of the GGUF format that prints the canonical dump of
`src/gguf_dump.cho` (docs/runtime.md section 4).

    gguf_dump.py <file.gguf> [--sum]

It is written from the format's public description (the header, the key-value store, the tensor directory, the aligned data
section) and shares no code with `cancho-gpu` or with any Rust reader. It replaced the dumper `tools/gguf_dump.rs` (lex-rt's
reader) and agrees with every golden that dumper made: `scripts/regen-gguf-golden.sh`, `docs/native.md` N2f.

It handles the files `cancho-gpu` accepts (versions 2 and 3) and prints `error: ...` and exits 1 on anything it cannot read;
what `cancho-gpu` refuses and with which rule is `tests/gguf/reject/`'s business, not this tool's.
"""
import mmap
import struct
import sys

# value types of the key-value store
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
# tensor types the reader knows the layout of: type id -> (name, values per block, bytes per block)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 30: ("BF16", 1, 2), 8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144),
         14: ("Q6_K", 256, 210), 142: ("PQ2_0", 128, 34), 143: ("PTQ1_0", 128, 28)}


class Reader:
    def __init__(self, data):
        self.d, self.p = data, 0

    def take(self, n):
        if n < 0 or self.p + n > len(self.d):
            raise ValueError("truncated")
        out = self.d[self.p:self.p + n]
        self.p += n
        return out

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return self.take(self.unpack("<Q"))


def value(r, ty, depth=0):
    if depth > 8:
        raise ValueError("too deep")
    if ty == STR:
        return ("str", r.string())
    if ty == BOOL:
        return ("bool", r.unpack("<B") != 0)
    if ty == F32:
        # widened to binary64: the dump shows the 64-bit pattern
        return ("float", struct.unpack("<Q", struct.pack("<d", r.unpack("<f")))[0])
    if ty == F64:
        return ("float", r.unpack("<Q"))
    if ty == ARR:
        et, n = r.unpack("<I"), r.unpack("<Q")
        return ("array", [value(r, et, depth + 1) for _ in range(n)])
    fmt = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", I64: "<q"}.get(ty)
    if ty == U64:
        v = r.unpack("<Q")
        return ("int", v - (1 << 64) if v >= 1 << 63 else v)  # a u64 above i64::MAX shows as negative
    if fmt is None:
        raise ValueError("value type")
    return ("int", r.unpack(fmt))


def quoted(b):
    out = ['"']
    for x in b:
        if x in (0x22, 0x5C):
            out.append("\\" + chr(x))
        elif x < 32 or x >= 127:
            out.append("\\x%02x" % x)
        else:
            out.append(chr(x))
    out.append('"')
    return "".join(out)


def show(v):
    kind, x = v
    if kind == "int":
        return "int %d" % x
    if kind == "float":
        return "float 0x%016x" % x
    if kind == "bool":
        return "bool " + ("true" if x else "false")
    if kind == "str":
        return "str " + quoted(x)
    first = ", ".join(show(e) for e in x[:4])
    return "array n=%d [%s%s]" % (len(x), first, ", ..." if len(x) > 4 else "")


def fnv32(b):
    h = 2166136261
    for x in b:
        h = ((h ^ x) * 16777619) & 0xFFFFFFFF
    return h


def main():
    path = sys.argv[1]
    with_sum = "--sum" in sys.argv[2:]
    f = open(path, "rb")
    data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if f.seek(0, 2) else b""  # a real model is gigabytes: map it, never read it all
    try:
        r = Reader(data)
        if r.take(4) != b"GGUF":
            raise ValueError("magic")
        version = r.unpack("<I")
        if version not in (2, 3):
            raise ValueError("version")
        n_tensors, n_kv = r.unpack("<Q"), r.unpack("<Q")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = value(r, r.unpack("<I"))
        tensors = {}
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.unpack("<Q") for _ in range(r.unpack("<I"))]
            ty, off = r.unpack("<I"), r.unpack("<Q")
            tensors[name] = (dims, ty, off)
        align = meta.get(b"general.alignment", ("int", 32))[1]
        start = (r.p + align - 1) // align * align
    except (ValueError, struct.error) as e:
        print("error: %s" % e, file=sys.stderr)
        sys.exit(1)

    print("gguf v%d tensors=%d kv=%d alignment=%d" % (version, len(tensors), len(meta), align))
    for k in sorted(meta):
        print("kv %s %s" % (k.decode("utf-8", "replace"), show(meta[k])))
    for name in sorted(tensors):
        dims, ty, off = tensors[name]
        elems = 1
        for d in dims:
            elems *= d
        known = TYPES.get(ty)
        tname = known[0] if known else "other(%d)" % ty
        size = elems // known[1] * known[2] if known else None
        line = "tensor %s type=%s dims=[%s] bytes=%s" % (name.decode("utf-8", "replace"), tname, ",".join(str(d) for d in dims),
                                                       size if known else "unsupported")
        if with_sum and known:
            line += " sum=%08x" % fnv32(data[start + off:start + off + size])
        print(line)


main()
