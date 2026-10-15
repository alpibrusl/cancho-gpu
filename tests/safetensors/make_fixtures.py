#!/usr/bin/env python3
"""Writes the safetensors fixtures of tests/safetensors/ (committed; this is
how they were made).

    python3 tests/safetensors/make_fixtures.py

A safetensors file is an 8-byte little-endian header length N, N bytes of
JSON (padded with spaces), then the tensor bytes. Each refusal fixture is
named for the rule `cancho-gpu safetensors` refuses it with.
"""
import json
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
         "I32": 4, "U32": 4, "F32": 4, "F64": 8, "I64": 8, "U64": 8}


def pattern(n, seed):
    return bytes((i * 11 + seed) & 255 for i in range(n))


def build(header, data=b"", pad=8, raw_header=None):
    h = raw_header if raw_header is not None else json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    h += b" " * ((-len(h)) % pad)
    return struct.pack("<Q", len(h)) + h + data


def layout(specs, order=None):
    """specs: [(name, dtype, shape)] in data order -> (header dict, data)."""
    header, data = {}, b""
    for i, (name, dt, shape) in enumerate(specs):
        n = SIZES[dt]
        for d in shape:
            n *= d
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [len(data), len(data) + n]}
        data += pattern(n, i + 1)
    if order:
        header = {k: header[k] for k in order}
    return header, data


def tiny():
    specs = [("w.f32", "F32", [2, 3]), ("w.bf16", "BF16", [4]), ("w.u32", "U32", [2]),
             ("scalar", "F16", []), ("w.i8", "I8", [3, 1]), ("empty", "U8", [0]), ("w.f64", "F64", [1])]
    header, data = layout(specs, order=["w.u32", "scalar", "w.f32", "empty", "w.i8", "w.bf16", "w.f64"])
    # Keys out of name order and out of offset order; the metadata between them.
    items = list(header.items())
    items.insert(2, ("__metadata__", {"format": "pt", "note": 'q"uote \\ back é'}))
    return build(dict(items), data)


def mlx_like():
    specs = [("layer.weight", "U32", [8, 4]), ("layer.scales", "F16", [8, 1]), ("layer.biases", "F16", [8, 1]),
             ("norm.weight", "BF16", [32])]
    header, data = layout(specs)
    header["__metadata__"] = {"format": "mlx"}
    return build(header, data)


def extra_field():
    """Accepted: a field the format does not define is ignored."""
    return build({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4], "extra": 1}}, b"\0" * 4)


def refusals():
    out = {}
    h, d = layout([("a", "F32", [2]), ("b", "F32", [2])])
    out["st-truncated"] = b"\x08\0\0"
    out["st-header-length-cut"] = build(h, d)[:20]
    out["st-header-length"] = struct.pack("<Q", 10**6) + b"{}" + b" " * 6
    out["st-header-size"] = struct.pack("<Q", 200_000_000) + b"{}"
    out["st-json-brace"] = build(None, b"", raw_header=b"[1,2]")
    out["st-json"] = build(None, b"", raw_header=b'{"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},}')
    h2 = {"a": {"shape": [1], "data_offsets": [0, 4]}}
    out["st-entry-dtype-missing"] = build(h2, b"\0" * 4)
    out["st-dtype"] = build({"a": {"dtype": "F33", "shape": [1], "data_offsets": [0, 4]}}, b"\0" * 4)
    out["st-shape"] = build({"a": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}}, b"\0" * 4)
    out["st-shape-type"] = build({"a": {"dtype": "F32", "shape": ["x"], "data_offsets": [0, 4]}}, b"\0" * 4)
    out["st-offsets-count"] = build({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0]}}, b"\0" * 4)
    out["st-offsets-order"] = build({"a": {"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}}, b"\0" * 4)
    out["st-size"] = build({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}, b"\0" * 4)
    out["st-overlap"] = build({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                               "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}}, b"\0" * 12)
    out["st-hole"] = build({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                            "b": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]}}, b"\0" * 12)
    out["st-data-short"] = build({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, b"\0" * 4)
    out["st-data-long"] = build({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, b"\0" * 8)
    out["st-metadata"] = build({"__metadata__": {"k": 1}}, b"")
    out["st-metadata-type"] = build({"__metadata__": ["x"]}, b"")
    out["st-numel"] = build({"a": {"dtype": "F32", "shape": [1 << 40, 1 << 40], "data_offsets": [0, 4]}}, b"\0" * 4)
    out["st-duplicate"] = build(None, b"\0" * 4, raw_header=b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}')
    return out


if __name__ == "__main__":
    for name, data in (("tiny", tiny()), ("mlx-like", mlx_like()), ("extra-field", extra_field())):
        with open(os.path.join(HERE, name + ".safetensors"), "wb") as f:
            f.write(data)
    for name, data in refusals().items():
        with open(os.path.join(HERE, "reject", name + ".safetensors"), "wb") as f:
            f.write(data)
