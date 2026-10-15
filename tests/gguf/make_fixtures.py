#!/usr/bin/env python3
"""Writes the GGUF fixtures of tests/gguf/ (and, with --big DIR, the one
too large to commit). The files are committed; this is how they were made.

    python3 tests/gguf/make_fixtures.py          # rewrites tests/gguf/*.gguf
    python3 tests/gguf/make_fixtures.py --big D  # D/gguf-string.gguf, 17 MiB
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def s(x):
    b = x if isinstance(x, bytes) else x.encode()
    return struct.pack("<Q", len(b)) + b


def kv(key, ty, payload):
    return s(key) + struct.pack("<I", ty) + payload


def u8(x): return kv_t(0, struct.pack("<B", x))
def kv_t(ty, payload): return (ty, payload)


def val(ty, v):
    if ty == 0: return struct.pack("<B", v)
    if ty == 1: return struct.pack("<b", v)
    if ty == 2: return struct.pack("<H", v)
    if ty == 3: return struct.pack("<h", v)
    if ty == 4: return struct.pack("<I", v)
    if ty == 5: return struct.pack("<i", v)
    if ty == 6: return struct.pack("<f", v) if not isinstance(v, bytes) else v
    if ty == 7: return struct.pack("<B", 1 if v else 0)
    if ty == 8: return s(v)
    if ty == 10: return struct.pack("<Q", v)
    if ty == 11: return struct.pack("<q", v)
    if ty == 12: return struct.pack("<d", v)
    raise ValueError(ty)


def arr(et, items):
    body = b"".join(items)
    return struct.pack("<IQ", et, len(items)) + body


def tensor(name, dims, gty, offset):
    return s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", gty, offset)


def pad(n, a):
    return (-n) % a


def file(version, kvs, tensors, data, align=32, magic=b"GGUF"):
    head = magic + struct.pack("<IQQ", version, len(tensors), len(kvs)) + b"".join(kvs) + b"".join(tensors)
    return head + b"\0" * pad(len(head), align) + data


def pattern(n, seed):
    return bytes((i * 7 + seed) & 255 for i in range(n))


def tiny(version=3, align=32, explicit_align=True):
    kvs = [
        kv("general.architecture", 8, val(8, "llama")),
        kv("u8", 0, val(0, 200)),
        kv("i8", 1, val(1, -5)),
        kv("u16", 2, val(2, 65535)),
        kv("i16", 3, val(3, -32768)),
        kv("u32", 4, val(4, 4294967295)),
        kv("i32", 5, val(5, -2147483648)),
        kv("u64.big", 10, val(10, 18446744073709551615)),
        kv("i64", 11, val(11, -9223372036854775808)),
        kv("f32.one", 6, val(6, 1.0)),
        kv("f32.third", 6, val(6, 1 / 3)),
        kv("f32.sub", 6, struct.pack("<I", 0x00000001)),
        kv("f32.sub2", 6, struct.pack("<I", 0x007fffff)),
        kv("f32.neg0", 6, struct.pack("<I", 0x80000000)),
        kv("f32.inf", 6, struct.pack("<I", 0x7f800000)),
        kv("f32.ninf", 6, struct.pack("<I", 0xff800000)),
        kv("f32.qnan", 6, struct.pack("<I", 0x7fc00000)),
        kv("f32.max", 6, struct.pack("<I", 0x7f7fffff)),
        kv("f64.pi", 12, val(12, 3.141592653589793)),
        kv("bool.t", 7, val(7, True)),
        kv("bool.f", 7, val(7, False)),
        kv("str.plain", 8, val(8, "hello")),
        kv("str.escapes", 8, s(b'q"uote \\ back\ttab\n\x01 \xc3\xb1 end')),
        kv("str.empty", 8, val(8, "")),
        kv("arr.empty", 9, struct.pack("<IQ", 4, 0)),
        kv("arr.ints", 9, arr(5, [val(5, i * 1000 - 3) for i in range(7)])),
        kv("arr.strs", 9, arr(8, [val(8, w) for w in ["a", "bb", "ccc", "", "eeeee"]])),
        kv("arr.floats", 9, arr(6, [val(6, x) for x in (0.5, -2.0)])),
        kv("arr.bools", 9, arr(7, [val(7, b) for b in (True, False, True)])),
        kv("arr.nested", 9, arr(9, [arr(4, [val(4, 1), val(4, 2)]), arr(8, [val(8, "x")]), arr(4, [])])),
    ]
    if explicit_align:
        kvs.append(kv("general.alignment", 4, val(4, align)))
    specs = [
        ("token_embd.weight", [4, 3], 0, 48),
        ("blk.0.attn_norm.weight", [8], 1, 16),
        ("blk.0.ffn.weight", [64], 8, 68),
        ("output.weight", [256], 12, 144),
        ("rope.freqs", [2, 2], 30, 8),
        ("unknown.kind", [5], 99, 0),
    ]
    tensors, data, off = [], b"", 0
    for i, (name, dims, gty, n) in enumerate(specs):
        tensors.append(tensor(name, dims, gty, off))
        chunk = pattern(n, i + 1)
        data += chunk + b"\0" * pad(len(chunk), align)
        off += len(chunk) + pad(len(chunk), align)
    return file(version, kvs, tensors, data, align)


def refusals():
    ok = lambda: tiny()
    out = {}
    out["gguf-magic"] = b"GGUX" + ok()[4:]
    out["gguf-version-old"] = file(1, [], [], b"")
    out["gguf-version-new"] = file(4, [], [], b"")
    out["gguf-truncated"] = ok()[:300]
    out["gguf-truncated-count"] = b"GGUF" + struct.pack("<IQQ", 3, 0, 1 << 40)
    out["gguf-value-type"] = file(3, [kv("k", 77, b"\0" * 8)], [], b"")
    out["gguf-value-type-array"] = file(3, [kv("k", 9, struct.pack("<IQ", 77, 1) + b"\0" * 8)], [], b"")
    deep = struct.pack("<IQ", 4, 0)
    for _ in range(9):
        deep = struct.pack("<IQ", 9, 1) + deep
    out["gguf-depth"] = file(3, [kv("k", 9, deep)], [], b"")
    out["gguf-duplicate"] = file(3, [kv("k", 0, b"\1"), kv("k", 0, b"\2")], [], b"")
    out["gguf-duplicate-tensor"] = file(3, [], [tensor("t", [1], 0, 0), tensor("t", [1], 0, 32)], b"\0" * 64)
    out["gguf-alignment-zero"] = file(3, [kv("general.alignment", 4, val(4, 0))], [], b"")
    out["gguf-alignment-type"] = file(3, [kv("general.alignment", 8, val(8, "32"))], [], b"")
    out["gguf-block"] = file(3, [], [tensor("t", [33], 8, 0)], b"\0" * 64)
    out["gguf-tensor-offset"] = file(3, [], [tensor("t", [1], 0, 4)], b"\0" * 64)
    out["gguf-dims-count"] = file(3, [], [tensor("t", [1] * 9, 0, 0)], b"\0" * 64)
    out["gguf-dims-size"] = file(3, [], [tensor("t", [1 << 40, 1 << 40], 0, 0)], b"")
    return out


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--big":
        name = b"k" * (17 * 1024 * 1024)
        data = file(3, [kv(name, 0, b"\1")], [], b"")
        with open(os.path.join(sys.argv[2], "gguf-string.gguf"), "wb") as f:
            f.write(data)
        sys.exit(0)
    with open(os.path.join(HERE, "tiny.gguf"), "wb") as f:
        f.write(tiny())
    with open(os.path.join(HERE, "tiny-v2-align64.gguf"), "wb") as f:
        f.write(tiny(version=2, align=64))
    with open(os.path.join(HERE, "tiny-default-align.gguf"), "wb") as f:
        f.write(tiny(explicit_align=False))
    os.makedirs(os.path.join(HERE, "reject"), exist_ok=True)
    for name, data in refusals().items():
        with open(os.path.join(HERE, "reject", name + ".gguf"), "wb") as f:
            f.write(data)
