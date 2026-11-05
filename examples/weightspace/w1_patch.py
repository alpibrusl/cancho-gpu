#!/usr/bin/env python3
"""docs/weight-space.md section 14.12 (T11): the W1 edit as a patch, written from the two published
checkpoints and checked without the loader that applies it.

    w1_patch.py make    BASE EDITED -o PATCH [--json report.json]
    w1_patch.py inspect PATCH
    w1_patch.py check   BASE PATCH EDITED          # base + patch equals edited, tensor by tensor, sha256

A patch (`src/qwen08_patch.cho` has the same description) is a column patch: for each tensor that
differs it holds the BF16 columns that differ, with the edited tensor's bits for every row of those
columns, and the CRC-32 and Adler-32 of the base tensor and of the edited tensor, so a loader can
refuse a patch made for another base and can prove that it made the tensor the patch promises. It is
derived from the two checkpoints' bytes, not from the authors' `edit.npz`. All integers little-endian:

    "CHOPATCH", u32 version = 1, u32 tensors (1..64), then per tensor
    u32 name length (1..255), name (zero-padded to 4), u32 rows, u32 cols (1..2^24 each), u32 ncols (1..cols),
    u32 base crc32, base adler32, edited crc32, edited adler32, u32 column index x ncols (strictly increasing,
    each < cols), u16 bf16 bits x (ncols * rows) column by column (zero-padded to 4),
    and at the end the u32 CRC-32 of every byte before it.

`decode` refuses a malformed patch with the rule tag the Cancho loader uses (`PatchError.rule`), in the
same order, so the tests can compare the two readers on the same bytes. The reader does not know the
loader's code. numpy only (the checkpoint reader is w1_diff.py's)."""
import argparse
import json
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w1_diff as D  # noqa: E402

MAGIC = b"CHOPATCH"
VERSION = 1
MAX_TENSORS = 64
MAX_DIM = 1 << 24
MAX_FILE = 1 << 24


class PatchError(Exception):
    """A patch that cannot be made or read; `rule` is the Cancho loader's rule tag for the refusal."""

    def __init__(self, rule, msg):
        super().__init__(f"{rule}: {msg}")
        self.rule = rule


def _pad4(b):
    return b + b"\0" * (-len(b) % 4)


def tensor_hashes(raw):
    raw = bytes(raw)
    return zlib.crc32(raw), zlib.adler32(raw)


def encode(records):
    """The patch bytes of `records`, a list of dicts with name, rows, cols, cols_idx (list of column
    indices), values (uint16 array [len(cols_idx), rows], column by column), base_hash and edited_hash
    (crc32, adler32). Nothing is checked: the tests use it to write malformed patches with a valid CRC."""
    out = MAGIC + struct.pack("<II", VERSION, len(records))
    for r in records:
        name = r["name"].encode()
        out += struct.pack("<I", len(name)) + _pad4(name)
        idx = list(r["cols_idx"])
        out += struct.pack("<7I", r["rows"], r["cols"], len(idx), *r["base_hash"], *r["edited_hash"])
        out += b"".join(struct.pack("<I", c) for c in idx)
        out += _pad4(np.asarray(r["values"], dtype="<u2").tobytes())
    return out + struct.pack("<I", zlib.crc32(out))


def decode(buf):
    """The records of a patch, checked as the loader checks a file (magic, version, CRC-32, layout, indices)."""
    n = len(buf)
    if n < 20:
        raise PatchError("q08-patch-truncated", "too short for a patch")
    if buf[:8] != MAGIC:
        raise PatchError("q08-patch-magic", "does not start with CHOPATCH")
    (ver,) = struct.unpack_from("<I", buf, 8)
    if ver != VERSION:
        raise PatchError("q08-patch-version", f"version {ver}")
    if zlib.crc32(buf[:-4]) != struct.unpack_from("<I", buf, n - 4)[0]:
        raise PatchError("q08-patch-checksum", "the CRC-32 does not match the trailer")
    (nt,) = struct.unpack_from("<I", buf, 12)
    if not 1 <= nt <= MAX_TENSORS:
        raise PatchError("q08-patch-format", f"{nt} tensors")
    e, p, out = n - 4, 16, []
    for k in range(nt):
        if p + 4 > e:
            raise PatchError("q08-patch-truncated", f"inside tensor {k}")
        (nl,) = struct.unpack_from("<I", buf, p)
        if not 1 <= nl <= 255:
            raise PatchError("q08-patch-format", f"tensor {k} has a name of {nl} bytes")
        pad = -nl % 4
        if p + 4 + nl + pad + 28 > e:
            raise PatchError("q08-patch-truncated", f"inside tensor {k}")
        if any(buf[p + 4 + nl:p + 4 + nl + pad]):
            raise PatchError("q08-patch-format", f"padding after the name of tensor {k}")
        q = p + 4 + nl + pad
        rows, cols, nc, bc, ba, ec, ea = struct.unpack_from("<7I", buf, q)
        if not (1 <= rows <= MAX_DIM and 1 <= cols <= MAX_DIM and nc >= 1):
            raise PatchError("q08-patch-format", f"tensor {k} has a shape or column count outside 1..2^24")
        if nc > cols:
            raise PatchError("q08-patch-range", f"tensor {k} replaces more columns than it has")
        idx_at = q + 28
        if idx_at + 4 * nc > e:
            raise PatchError("q08-patch-truncated", f"inside tensor {k}")
        val_at = idx_at + 4 * nc
        vbytes = 2 * nc * rows
        vpad = vbytes + (-vbytes % 4)
        if val_at + vpad > e:
            raise PatchError("q08-patch-truncated", f"inside tensor {k}")
        if any(buf[val_at + vbytes:val_at + vpad]):
            raise PatchError("q08-patch-format", f"padding after the values of tensor {k}")
        idx = list(struct.unpack_from(f"<{nc}I", buf, idx_at))
        for j, c in enumerate(idx):
            if c >= cols:
                raise PatchError("q08-patch-range", f"tensor {k} names column {c}")
            if j and c == idx[j - 1]:
                raise PatchError("q08-patch-duplicate", f"tensor {k} names column {c} twice")
            if j and c < idx[j - 1]:
                raise PatchError("q08-patch-order", f"tensor {k}: column {c} out of order")
        name = bytes(buf[p + 4:p + 4 + nl]).decode("utf-8", "replace")
        if any(r["name"] == name for r in out):
            raise PatchError("q08-patch-twice", f"`{name}` twice")
        vals = np.frombuffer(bytes(buf[val_at:val_at + vbytes]), dtype="<u2").reshape(nc, rows)
        out.append({"name": name, "rows": rows, "cols": cols, "cols_idx": idx, "values": vals,
                    "base_hash": (bc, ba), "edited_hash": (ec, ea)})
        p = val_at + vpad
    if p != e:
        raise PatchError("q08-patch-format", "bytes after the last tensor")
    return out


def apply_to(base_u16, rec):
    """The patched tensor (uint16 [rows, cols]) of the base tensor and a decoded record."""
    out = np.array(base_u16, dtype=np.uint16, copy=True)
    out[:, rec["cols_idx"]] = rec["values"].T
    return out


def make(base, edited):
    """The records for every tensor that differs between two checkpoints ({name: Tensor}). A patch holds
    BF16 matrices only; two checkpoints that differ in another way are refused."""
    nb, ne = set(base), set(edited)
    if nb != ne:
        raise PatchError("q08-patch-unknown", f"the checkpoints do not have the same tensors ({len(nb ^ ne)} differ)")
    recs = []
    for name in sorted(nb):
        tb, te = base[name], edited[name]
        if tb.dtype != te.dtype or tb.shape != te.shape:
            raise PatchError("q08-patch-shape", f"`{name}` has another dtype or shape in the edited checkpoint")
        if D._equal_bytes(tb.raw, te.raw):
            continue
        if tb.dtype != "BF16" or len(tb.shape) != 2:
            raise PatchError("q08-patch-unknown", f"`{name}` differs and is a {tb.dtype} {list(tb.shape)}, not a BF16 matrix")
        ub, ue = tb.unsigned(), te.unsigned()
        cols = np.flatnonzero((ub != ue).any(0))
        recs.append({"name": name, "rows": tb.shape[0], "cols": tb.shape[1], "cols_idx": [int(c) for c in cols],
                     "values": np.ascontiguousarray(ue[:, cols].T), "base_hash": tensor_hashes(tb.raw),
                     "edited_hash": tensor_hashes(te.raw)})
    if not recs:
        raise PatchError("q08-patch-format", "the checkpoints are identical: there is nothing to patch")
    if len(recs) > MAX_TENSORS:
        raise PatchError("q08-patch-format", f"{len(recs)} tensors differ, a patch holds {MAX_TENSORS}")
    return recs


def check(base, patch_bytes, edited):
    """Base + patch against edited, every tensor, by sha256 of the raw bytes. `(rows, ok)`: a row per
    tensor `(name, patched?, equal?)`."""
    recs = {r["name"]: r for r in decode(patch_bytes)}
    rows = []
    for name in recs:
        if name not in base:
            raise PatchError("q08-patch-unknown", f"the patch names `{name}`, which the base does not have")
        t = base[name]
        if t.dtype != "BF16" or t.shape != (recs[name]["rows"], recs[name]["cols"]):
            raise PatchError("q08-patch-shape", f"`{name}` is not the [rows, cols] the patch was made for")
        if tensor_hashes(t.raw) != recs[name]["base_hash"]:
            raise PatchError("q08-patch-base", f"`{name}` in the base is not the tensor the patch was made from")
    import hashlib
    ok = True
    for name in sorted(base):
        if name not in edited:
            rows.append((name, name in recs, False))
            ok = False
            continue
        if name in recs:
            raw = apply_to(base[name].unsigned(), recs[name]).astype("<u2").tobytes()
        else:
            raw = bytes(base[name].raw)
        eq = hashlib.sha256(raw).digest() == hashlib.sha256(bytes(edited[name].raw)).digest()
        rows.append((name, name in recs, eq))
        ok = ok and eq
    return rows, ok


def cmd_make(a):
    base, edited = D.read_checkpoint(a.base), D.read_checkpoint(a.edited)
    recs = make(base, edited)
    blob = encode(recs)
    if len(blob) > MAX_FILE:
        raise PatchError("q08-patch-io", f"the patch is {len(blob)} bytes, over {MAX_FILE}")
    # Never trust the writer: read it back and apply it.
    rows, ok = check(base, blob, edited)
    if not ok:
        raise PatchError("q08-patch-result", "base + patch does not equal edited: " + ", ".join(n for n, _, e in rows if not e)[:300])
    with open(a.out, "wb") as f:
        f.write(blob)
    rep = {"patch": os.path.abspath(a.out), "bytes": len(blob), "crc32": zlib.crc32(blob),
           "tensors": [{"name": r["name"], "rows": r["rows"], "cols": r["cols"], "columns": len(r["cols_idx"]),
                        "weights": len(r["cols_idx"]) * r["rows"], "first_columns": r["cols_idx"][:8]} for r in recs],
           "weights": sum(len(r["cols_idx"]) * r["rows"] for r in recs), "tensors_compared": len(rows)}
    print(f"wrote {a.out}: {len(blob):,} bytes, {len(recs)} tensors, {rep['weights']:,} weights replaced "
          f"({', '.join(r['name'] for r in recs)})")
    print(f"read back: base + patch equals edited for all {len(rows)} tensors (sha256 of the raw bytes)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


def cmd_inspect(a):
    with open(a.patch, "rb") as f:
        blob = f.read(MAX_FILE + 1)
    recs = decode(blob)
    print(f"{a.patch}: {len(blob):,} bytes, version {VERSION}, crc32 {zlib.crc32(blob):08x}, {len(recs)} tensors")
    for r in recs:
        print(f"  {r['name']} [{r['rows']}, {r['cols']}]: {len(r['cols_idx'])} columns, {len(r['cols_idx']) * r['rows']:,} weights; "
              f"base crc32/adler32 {r['base_hash'][0]:08x}/{r['base_hash'][1]:08x}, edited {r['edited_hash'][0]:08x}/{r['edited_hash'][1]:08x}")
    return 0


def cmd_check(a):
    with open(a.patch, "rb") as f:
        blob = f.read(MAX_FILE + 1)
    rows, ok = check(D.read_checkpoint(a.base), blob, D.read_checkpoint(a.edited))
    bad = [n for n, _, e in rows if not e]
    print(f"{len(rows)} tensors, {sum(1 for _, p, _ in rows if p)} patched, {len(bad)} differ from the edited checkpoint's bytes")
    for n in bad[:20]:
        print(f"  differs: {n}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("base")
    m.add_argument("edited")
    m.add_argument("-o", "--out", required=True)
    m.add_argument("--json")
    i = sub.add_parser("inspect")
    i.add_argument("patch")
    c = sub.add_parser("check")
    c.add_argument("base")
    c.add_argument("patch")
    c.add_argument("edited")
    a = ap.parse_args(argv)
    try:
        return {"make": cmd_make, "inspect": cmd_inspect, "check": cmd_check}[a.cmd](a)
    except (PatchError, D.ReadError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
