"""docs/weight-space.md section 13 (W1): the weight diff between two checkpoints, on the CPU, numpy only.

A small safetensors reader (no `safetensors`, no torch: bfloat16 is read as the top half of a float32) that
refuses a malformed file with `ReadError` instead of raising anything else, and `diff_checkpoints`, which
reports for every tensor whether it changed (compared as raw bytes, so NaN payloads and -0.0 count), and for
the tensors that did: the changed entries, rows and columns, the norms of the difference, the singular values
of the difference on its support (the rank at three tolerances) and the sha256 of both tensors' raw bytes.

A checkpoint is a `.safetensors` file or a directory of them (any name ending in `.safetensors`, so the
Hugging Face `model.safetensors-00001-of-00001.safetensors` is found). Nothing is written."""
import glob
import hashlib
import json
import mmap
import os
import struct

import numpy as np

MAX_HEADER = 100 * 1024 * 1024
# dtype -> (bytes per element, unsigned integer type of that size, numeric view or None)
DTYPES = {
    "BOOL": (1, np.uint8, np.uint8),
    "U8": (1, np.uint8, np.uint8), "I8": (1, np.uint8, np.int8),
    "U16": (2, np.uint16, np.uint16), "I16": (2, np.uint16, np.int16),
    "U32": (4, np.uint32, np.uint32), "I32": (4, np.uint32, np.int32),
    "U64": (8, np.uint64, np.uint64), "I64": (8, np.uint64, np.int64),
    "F16": (2, np.uint16, np.float16), "BF16": (2, np.uint16, "bf16"),
    "F32": (4, np.uint32, np.float32), "F64": (8, np.uint64, np.float64),
    "F8_E4M3": (1, np.uint8, None), "F8_E5M2": (1, np.uint8, None),
}
RANK_TOLERANCES = (0.1, 0.01, 0.001)
SVD_LIMIT = 40_000_000   # elements of the support submatrix above which the SVD is skipped (reported)
CHUNK = 1 << 26


class ReadError(Exception):
    """A checkpoint that cannot be read; the message says which file and why."""


class Tensor:
    def __init__(self, name, dtype, shape, raw, path):
        self.name, self.dtype, self.shape, self.raw, self.path = name, dtype, shape, raw, path

    @property
    def size(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def unsigned(self):
        """The elements as unsigned integers of their width (a copy: the mapped bytes may be unaligned)."""
        return np.frombuffer(bytes(self.raw), dtype=DTYPES[self.dtype][1]).reshape(self.shape)

    def as_float64(self):
        """The values as float64, or None for a dtype this module does not interpret (F8)."""
        view = DTYPES[self.dtype][2]
        if view is None:
            return None
        u = self.unsigned()
        if view == "bf16":
            with np.errstate(invalid="ignore"):   # a NaN payload is reported by the caller, not warned about
                return (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        return u.view(view).astype(np.float64)


def _check_header(path, header, data_len):
    if not isinstance(header, dict):
        raise ReadError(f"{path}: the header is not a JSON object")
    spans = []
    for name, ent in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(ent, dict) or {"dtype", "shape", "data_offsets"} - set(ent):
            raise ReadError(f"{path}: tensor {name!r} lacks dtype, shape or data_offsets")
        if not isinstance(ent["dtype"], str) or ent["dtype"] not in DTYPES:
            raise ReadError(f"{path}: tensor {name!r} has the unknown dtype {ent['dtype']!r}")
        shape, off = ent["shape"], ent["data_offsets"]
        if not isinstance(shape, list) or not all(type(d) is int and d >= 0 for d in shape):
            raise ReadError(f"{path}: tensor {name!r} has an invalid shape")
        if not isinstance(off, list) or len(off) != 2 or not all(type(o) is int and o >= 0 for o in off):
            raise ReadError(f"{path}: tensor {name!r} has invalid data_offsets")
        n = 1
        for d in shape:
            n *= d
        if off[1] < off[0] or off[1] > data_len:
            raise ReadError(f"{path}: tensor {name!r} lies outside the file (offsets {off}, {data_len} bytes of data)")
        if off[1] - off[0] != n * DTYPES[ent["dtype"]][0]:
            raise ReadError(f"{path}: tensor {name!r}: {off[1] - off[0]} bytes for shape {shape} of {ent['dtype']}")
        spans.append((off[0], off[1], name))
    spans.sort()
    for (a0, a1, an), (b0, b1, bn) in zip(spans, spans[1:]):
        if b0 < a1 and b1 > b0 and a1 > a0:
            raise ReadError(f"{path}: tensors {an!r} and {bn!r} overlap")


def read_file(path):
    """{name: Tensor} of one .safetensors file, validated."""
    try:
        size = os.path.getsize(path)
        if size < 8:
            raise ReadError(f"{path}: {size} bytes, too short for a safetensors file")
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            if hlen > MAX_HEADER or 8 + hlen > size:
                raise ReadError(f"{path}: header length {hlen} does not fit the file ({size} bytes)")
            try:
                header = json.loads(f.read(hlen).decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as e:
                raise ReadError(f"{path}: the header is not JSON ({e})")
            data_len = size - 8 - hlen
            _check_header(path, header, data_len)
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if data_len else b""
    except OSError as e:
        raise ReadError(f"{path}: {e}")
    out = {}
    base = 8 + hlen
    for name, ent in header.items():
        if name == "__metadata__":
            continue
        b, e = ent["data_offsets"]
        out[name] = Tensor(name, ent["dtype"], tuple(ent["shape"]), memoryview(mm)[base + b:base + e] if e > b else memoryview(b""), path)
    return out


def read_checkpoint(path):
    """{name: Tensor} of a file or of every *.safetensors file in a directory."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not files:
            raise ReadError(f"{path}: no *.safetensors file in this directory")
    elif os.path.isfile(path):
        files = [path]
    else:
        raise ReadError(f"{path}: no such file or directory")
    out = {}
    for f in files:
        for name, t in read_file(f).items():
            if name in out:
                raise ReadError(f"{f}: tensor {name!r} is also in {out[name].path}")
            out[name] = t
    return out


def _equal_bytes(a, b):
    n = len(a)
    if n != len(b):
        return False
    for i in range(0, n, CHUNK):
        if bytes(a[i:i + CHUNK]) != bytes(b[i:i + CHUNK]):
            return False
    return True


def _sha(t):
    h = hashlib.sha256()
    for i in range(0, len(t.raw), CHUNK):
        h.update(bytes(t.raw[i:i + CHUNK]))
    return h.hexdigest()


def _rank_report(sub):
    if sub.size == 0:
        return {"svd": "empty support"}
    if sub.size > SVD_LIMIT:
        return {"svd": f"skipped: the support is {sub.shape[0]} x {sub.shape[1]}"}
    try:
        s = np.linalg.svd(sub, compute_uv=False)
    except np.linalg.LinAlgError as e:
        return {"svd": f"failed: {e}"}
    if s[0] == 0.0:
        return {"svd": "zero"}
    e2 = s * s
    return {"support_shape": [int(sub.shape[0]), int(sub.shape[1])],
            "singular_values": [float(x) for x in s[:8]],
            "rank_at": {str(t): int((s > t * s[0]).sum()) for t in RANK_TOLERANCES},
            "top1_energy": float(e2[0] / e2.sum()),
            "stable_rank": float(e2.sum() / e2[0])}


def diff_tensor(tb, te):
    """The report of one tensor present in both checkpoints with the same dtype and shape."""
    ub, ue = tb.unsigned(), te.unsigned()
    mask = ub != ue
    rep = {"name": tb.name, "dtype": tb.dtype, "shape": list(tb.shape), "size": tb.size,
           "changed_entries": int(mask.sum()), "base_sha256": _sha(tb), "edited_sha256": _sha(te)}
    rep["changed_fraction"] = rep["changed_entries"] / tb.size if tb.size else 0.0
    if mask.ndim == 2:
        rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        rep["changed_rows"], rep["changed_columns"] = int(len(rows)), int(len(cols))
        rep["columns"] = [int(c) for c in cols[:128]]
        rep["columns_truncated"] = bool(len(cols) > 128)
    fb, fe = tb.as_float64(), te.as_float64()
    if fb is None:
        rep["numeric"] = f"not interpreted: dtype {tb.dtype}"
        return rep
    d = fe - fb
    if not np.isfinite(d[mask]).all() or not np.isfinite(fb).all():
        rep["numeric"] = "not computed: a non-finite value in the tensor or the difference"
        return rep
    nb, nd = float(np.linalg.norm(fb)), float(np.linalg.norm(d))
    rep.update(base_norm=nb, delta_norm=nd, relative_change=nd / nb if nb else None,
               max_abs_delta=float(np.abs(d).max()) if d.size else 0.0)
    if mask.ndim == 2 and rep["changed_entries"]:
        sub = d[np.ix_(rows, cols)]
        rep.update(_rank_report(sub))
        col_e = (d * d).sum(0)
        top = np.sort(col_e)[::-1][:rep["changed_columns"]].sum()
        rep["energy_in_changed_columns"] = float(top / col_e.sum()) if col_e.sum() else None
    return rep


def diff_checkpoints(base, edited):
    """The whole report. `base`, `edited`: {name: Tensor}."""
    nb, ne = set(base), set(edited)
    rep = {"tensors_base": len(nb), "tensors_edited": len(ne), "only_in_base": sorted(nb - ne),
           "only_in_edited": sorted(ne - nb), "mismatched": [], "unchanged": 0, "changed": []}
    total = 0
    for name in sorted(nb & ne):
        tb, te = base[name], edited[name]
        total += tb.size
        if tb.dtype != te.dtype or tb.shape != te.shape:
            rep["mismatched"].append({"name": name, "base": [tb.dtype, list(tb.shape)], "edited": [te.dtype, list(te.shape)]})
        elif _equal_bytes(tb.raw, te.raw):
            rep["unchanged"] += 1
        else:
            rep["changed"].append(diff_tensor(tb, te))
    rep["total_weights"] = total
    rep["changed_weights"] = sum(c["changed_entries"] for c in rep["changed"])
    rep["changed_fraction"] = rep["changed_weights"] / total if total else 0.0
    return rep


def render(rep):
    """The report as text."""
    L = [f"tensors: {rep['tensors_base']} in the base, {rep['tensors_edited']} in the edited; {rep['unchanged']} identical, "
         f"{len(rep['changed'])} changed, {len(rep['mismatched'])} with another dtype or shape, "
         f"{len(rep['only_in_base'])} only in the base, {len(rep['only_in_edited'])} only in the edited",
         f"weights changed: {rep['changed_weights']:,} of {rep['total_weights']:,} ({100 * rep['changed_fraction']:.4f}%)"]
    for k in ("only_in_base", "only_in_edited"):
        for n in rep[k][:20]:
            L.append(f"  {k}: {n}")
    for m in rep["mismatched"][:20]:
        L.append(f"  mismatched: {m['name']} base {m['base']} edited {m['edited']}")
    for c in rep["changed"]:
        L.append(f"- {c['name']} {c['dtype']} {c['shape']}: {c['changed_entries']:,} entries ({100 * c['changed_fraction']:.2f}%)"
                 + (f", {c['changed_rows']} rows, {c['changed_columns']} columns" if "changed_columns" in c else ""))
        if "delta_norm" in c:
            rc = c["relative_change"]
            L.append(f"    |delta| {c['delta_norm']:.6g}, relative to |base| {'n/a' if rc is None else format(rc, '.4g')}, max |delta| {c['max_abs_delta']:.6g}")
        elif "numeric" in c:
            L.append(f"    {c['numeric']}")
        if "rank_at" in c:
            L.append(f"    delta on its support {c['support_shape']}: rank {c['rank_at']} (singular values above tol * the largest), "
                     f"top-1 energy {c['top1_energy']:.4f}, stable rank {c['stable_rank']:.3f}; "
                     f"{100 * c['energy_in_changed_columns']:.2f}% of the delta energy in the changed columns")
            L.append("    singular values: " + " ".join(f"{x:.4g}" for x in c["singular_values"]))
        elif "svd" in c:
            L.append(f"    svd: {c['svd']}")
        L.append(f"    sha256 base {c['base_sha256']} edited {c['edited_sha256']}")
    return "\n".join(L)
