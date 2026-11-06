#!/usr/bin/env python3
"""docs/weight-space-w4.md section 13: the W4 variants as loadable checkpoints, so that `w1.py behave` can run them.

    python3 examples/weightspace/w4_ckpt.py list
    python3 examples/weightspace/w4_ckpt.py write  --variant x1e-1 --out DIR [--work build/w4] [--base ~/models/qwen35-08b-base]
                                                   [--edited ~/models/qwen35-08b-edited] [--no-patch]
    python3 examples/weightspace/w4_ckpt.py verify --variant x1e-1 --out DIR [--work build/w4] [--windows 4] [--max-len 256]

Variants (the names of 11 to 13): `x5e-2`, `x1e-1`, `x2e-1` (the published edit projected off the directions of K0 whose
singular value is above that fraction of the largest, w4_lib.null_project, K0 from WORK/fit_moments.npz), `e15`, `e19` (the
published edit on one layer only), `a1`, `a3`, `a6`, `a9` (the rank-1 form `dW[:, j] = -alpha (u . W_j) u` on the 64 published
columns, u the top left singular vector of the published edit; `a6` is the published edit's own form, 11.1).

`write` copies BASE's directory (config, tokenizer, index) and the safetensors files byte for byte, then overwrites only the bytes of
the tensors the variant changes, in the base's dtype (bf16: round to nearest even of base + intended edit). It reads the result
back, compares every other tensor's bytes to the base's, and writes `w4_manifest.json` (the cosine of what was written with the
intended edit, the written/intended norm ratio, the changed-weight count, per-tensor sha256) and, unless --no-patch, `variant.chopatch`
(CHOPATCH v1, w1_patch.py's writer, read back and applied). The directory appears only when complete. numpy only for `write`; `verify`
needs torch and transformers (examples/qwen35/hf08_ref.py) and the corpus of WORK.

Exit status: 0 done, 1 a check failed, 2 the input is unusable (every refusal prints its rule tag `w4c-*`)."""
import argparse
import glob
import hashlib
import json
import mmap
import os
import shutil
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import w1_diff as D  # noqa: E402
import w1_patch as P  # noqa: E402
import w4_lib as L  # noqa: E402

PREFIX = "model.language_model.layers."
LAYERS = (15, 19)
NAME_FMT = PREFIX + "{}.mlp.down_proj.weight"
CUTS = {"x5e-2": 5e-2, "x1e-1": 1e-1, "x2e-1": 2e-1}
ALPHAS = {"a1": 1.0, "a3": 3.0, "a6": 6.0, "a9": 9.0}
SINGLE = {"e15": 15, "e19": 19}
VARIANTS = list(CUTS) + list(SINGLE) + list(ALPHAS)
COS_CLAIM = 0.998   # docs/weight-space-w4.md 11.3: bf16 keeps at least this cosine with the intended edit
HOME = os.path.expanduser("~")
BASE_DIR = os.path.join(HOME, "models", "qwen35-08b-base")
EDITED_DIR = os.path.join(HOME, "models", "qwen35-08b-edited")
MANIFEST, PATCH = "w4_manifest.json", "variant.chopatch"


class CkptError(Exception):
    """A refusal; `rule` is its tag."""

    def __init__(self, rule, msg):
        super().__init__(f"{rule}: {msg}")
        self.rule = rule


# ------------------------------------------------------------------ the intended edits (float64, pure)

def parse_variant(name):
    if not isinstance(name, str) or name not in VARIANTS:
        raise CkptError("w4c-variant", f"unknown variant {str(name)[:40]!r}; one of {', '.join(VARIANTS)}")
    return name


def layer_of(name):
    """The layers a variant touches."""
    return (SINGLE[name],) if name in SINGLE else LAYERS


def _check_pub(pub, layers):
    for l in layers:
        e = pub.get(l) if isinstance(pub, dict) else None
        if not e or "w" not in e or "dw" not in e:
            raise CkptError("w4c-input", f"the published edit of layer {l} is missing")
        w, dw = np.asarray(e["w"]), np.asarray(e["dw"])
        if w.ndim != 2 or w.shape != dw.shape:
            raise CkptError("w4c-input", f"layer {l}: base {w.shape} and edit {dw.shape} are not the same matrix")
        if not np.isfinite(dw).all() or not np.any(dw):
            raise CkptError("w4c-input", f"layer {l}: the published edit is empty or not finite")


def intended_edits(name, pub, moments=None):
    """{layer: dW float64} for a variant. `pub` = {layer: {"w": base [out, in], "dw": published edit}}; `moments` = {"n": positions,
    layer: second moment C [in, in]} (needed by the x* variants)."""
    parse_variant(name)
    layers = layer_of(name)
    _check_pub(pub, layers)
    out = {}
    for l in layers:
        w, dw = np.asarray(pub[l]["w"], dtype=np.float64), np.asarray(pub[l]["dw"], dtype=np.float64)
        if name in CUTS:
            if not moments or l not in moments or "n" not in moments:
                raise CkptError("w4c-moments", f"{name} needs the second moment of layer {l}")
            c, n = np.asarray(moments[l], dtype=np.float64), int(moments["n"])
            if c.shape != (w.shape[1], w.shape[1]) or not np.isfinite(c).all() or n < 1:
                raise CkptError("w4c-moments", f"layer {l}: the second moment is {c.shape}, not [{w.shape[1]}, {w.shape[1]}], or has no positions")
            ev, q = L.eig_of(c)
            out[l], _ = L.null_project(dw, ev, q, n, CUTS[name])
        elif name in ALPHAS:
            _, u, _ = L.rank1(dw)
            cols = np.flatnonzero(np.abs(dw).sum(0) > 0)
            d = np.zeros_like(dw)
            d[:, cols] = -ALPHAS[name] * np.outer(u, u @ w[:, cols])
            out[l] = d
        else:
            out[l] = dw
    return out


# ------------------------------------------------------------------ dtype

def decode(u, dtype):
    """float64 values of the raw elements `u` (uint16/uint32 array) of a safetensors dtype."""
    if dtype == "BF16":
        return (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if dtype == "F16":
        return u.view(np.float16).astype(np.float64)
    if dtype == "F32":
        return u.view(np.float32).astype(np.float64)
    raise CkptError("w4c-dtype", f"dtype {dtype} is not one a variant is written in (BF16, F16, F32)")


def encode(x, dtype):
    """Raw little-endian bytes of float values `x` in `dtype` (bf16: nearest, ties to even)."""
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    if dtype == "BF16":
        return (L.bf16_round(x32).view(np.uint32) >> 16).astype("<u2").tobytes()
    if dtype == "F16":
        return x32.astype("<f2").tobytes()
    if dtype == "F32":
        return x32.astype("<f4").tobytes()
    raise CkptError("w4c-dtype", f"dtype {dtype} is not one a variant is written in (BF16, F16, F32)")


# ------------------------------------------------------------------ the checkpoint

def _layout(base_dir):
    """{tensor name: (file, dtype, shape, data offset in file, nbytes)} of a base directory; refuses what it cannot rewrite."""
    if not os.path.isdir(base_dir):
        raise CkptError("w4c-base", f"{base_dir}: not a directory")
    files = sorted(glob.glob(os.path.join(base_dir, "*.safetensors")))
    if not files:
        raise CkptError("w4c-base", f"{base_dir}: no *.safetensors file")
    lay = {}
    for f in files:
        try:
            D.read_file(f)   # validates the header against the file size
            with open(f, "rb") as fh:
                (hlen,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(hlen).decode("utf-8"))
        except (D.ReadError, OSError, ValueError) as e:
            raise CkptError("w4c-base", str(e)[:300])
        for k, v in header.items():
            if k != "__metadata__":
                lay[k] = (f, v["dtype"], tuple(v["shape"]), 8 + hlen + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0])
    return lay


def _read_raw(layout, name):
    f, dtype, shape, off, n = layout[name]
    if dtype not in ("BF16", "F16", "F32"):
        raise CkptError("w4c-dtype", f"`{name}` is {dtype}: a variant is written in BF16, F16 or F32")
    with open(f, "rb") as fh:
        fh.seek(off)
        raw = fh.read(n)
    if len(raw) != n:
        raise CkptError("w4c-base", f"{f}: `{name}` is cut short")
    try:
        return np.frombuffer(raw, dtype=np.uint16 if dtype != "F32" else np.uint32).reshape(shape)
    except ValueError:
        raise CkptError("w4c-base", f"`{name}`: {n} bytes do not make a {dtype} tensor of shape {list(shape)}")


def load_published(base_dir, edited_dir):
    """{layer: {"w", "dw"}} in float64 from two real checkpoints (the edit is edited - base on the two published tensors)."""
    lb, le = _layout(base_dir), _layout(edited_dir)
    pub = {}
    for l in LAYERS:
        n = NAME_FMT.format(l)
        if n not in lb or n not in le:
            raise CkptError("w4c-base", f"`{n}` is not in both checkpoints")
        if lb[n][1] != le[n][1] or lb[n][2] != le[n][2]:
            raise CkptError("w4c-base", f"`{n}` differs in dtype or shape between the checkpoints")
        wb = decode(_read_raw(lb, n), lb[n][1])
        pub[l] = {"w": wb, "dw": decode(_read_raw(le, n), le[n][1]) - wb}
    return pub


def load_moments(work):
    path = os.path.join(work, "fit_moments.npz")
    try:
        z = np.load(path)
        m = {"n": int(z["n"])}
        for l in LAYERS:
            m[l] = z[f"c{l}"]
        return m
    except (OSError, KeyError, ValueError) as e:
        raise CkptError("w4c-moments", f"{path}: {str(e)[:200]} (run w4_collateral.py fit)")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def write_checkpoint(base_dir, out_dir, variant, pub, moments=None, patch=True, tensor_name=NAME_FMT.format, edits=None):
    """Write the variant's checkpoint to `out_dir` (which must not exist or be empty) and return the manifest. Pure file work.
    `edits` ({layer: dW float64}, docs/weight-space-w4-search.md 5) replaces the named variant's intended edits; `variant` is then a label."""
    if edits is None:
        parse_variant(variant)
        edits = intended_edits(variant, pub, moments)
    else:
        _check_pub(pub, tuple(edits))
        for l, dw in edits.items():
            if not np.isfinite(dw).all() or not np.any(dw):
                raise CkptError("w4c-input", f"layer {l}: the edit is empty or not finite")
    lay = _layout(base_dir)
    out_dir = os.path.abspath(out_dir)
    if os.path.exists(out_dir) and (not os.path.isdir(out_dir) or os.listdir(out_dir)):
        raise CkptError("w4c-out", f"{out_dir} exists and is not an empty directory")
    tmp = out_dir + ".partial"
    if os.path.exists(tmp):
        raise CkptError("w4c-out", f"{tmp} exists (a cut-off earlier write); remove it")
    plan = {}
    for l, dw in edits.items():
        n = tensor_name(l)
        if n not in lay:
            raise CkptError("w4c-base", f"`{n}` is not in the base")
        f, dtype, shape, off, nb = lay[n]
        if tuple(dw.shape) != tuple(shape):
            raise CkptError("w4c-input", f"`{n}`: the edit is {dw.shape}, the base tensor {shape}")
        base_u = _read_raw(lay, n)
        wb = decode(base_u, dtype)
        if not np.array_equal(wb, np.asarray(pub[l]["w"], dtype=np.float64)):
            raise CkptError("w4c-input", f"`{n}`: the base weights given are not the base checkpoint's")
        raw = encode(wb + dw, dtype)
        plan[n] = (raw, dtype, shape, wb, dw, base_u)
    os.makedirs(tmp)
    try:
        for p in sorted(os.listdir(base_dir)):
            src = os.path.join(base_dir, p)
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(tmp, p))
        by_file = {}
        for n, (raw, *_r) in plan.items():
            by_file.setdefault(lay[n][0], []).append((lay[n][3], raw))
        for f, items in by_file.items():
            with open(os.path.join(tmp, os.path.basename(f)), "r+b") as fh:
                for off, raw in items:
                    fh.seek(off)
                    fh.write(raw)
        man = _manifest(base_dir, tmp, variant, plan, lay, moments)
        if patch:
            man["patch"] = _write_patch(base_dir, tmp)
        with open(os.path.join(tmp, MANIFEST), "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=1)
        if not man["checks"]["ok"]:
            raise CkptError("w4c-result", "the read-back check failed: " + "; ".join(man["checks"]["failures"])[:300])
        if os.path.isdir(out_dir):
            os.rmdir(out_dir)   # empty, checked above
        os.replace(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return man


def _manifest(base_dir, out_dir, variant, plan, lay, moments):
    """Read the written files back with the independent reader and compare."""
    back = D.read_checkpoint(out_dir)
    base = D.read_checkpoint(base_dir)
    fails = []
    tensors, total_changed = [], 0
    dot = n1 = n2 = 0.0
    for name in sorted(base):
        if name not in back:
            fails.append(f"`{name}` missing from the written checkpoint")
            continue
        same = D._equal_bytes(base[name].raw, back[name].raw)
        if name not in plan:
            if not same or base[name].shape != back[name].shape or base[name].dtype != back[name].dtype:
                fails.append(f"`{name}` changed and should not have")
            continue
        raw, dtype, shape, wb, dw, base_u = plan[name]
        got = back[name]
        if bytes(got.raw) != raw or got.dtype != dtype or got.shape != tuple(shape):
            fails.append(f"`{name}` is not what was meant to be written")
        written = decode(got.unsigned(), dtype) - wb
        d1, a, b = float((written * dw).sum()), float((written ** 2).sum()), float((dw ** 2).sum())
        dot, n1, n2 = dot + d1, n1 + a, n2 + b
        diff = got.unsigned() != base_u
        total_changed += int(diff.sum())
        tensors.append({"name": name, "dtype": dtype, "shape": list(shape), "cosine": d1 / np.sqrt(a * b) if a > 0 and b > 0 else None,
                        "norm_ratio_written_over_intended": float(np.sqrt(a / b)) if b > 0 else None,
                        "changed_weights": int(diff.sum()), "changed_columns": int(diff.any(0).sum()), "changed_rows": int(diff.any(1).sum()),
                        "intended_nonzero_weights": int((dw != 0).sum()), "sha256": _sha(bytes(got.raw))})
    extra = sorted(set(back) - set(base))
    if extra:
        fails.append(f"{len(extra)} tensors the base does not have")
    cos = dot / np.sqrt(n1 * n2) if n1 > 0 and n2 > 0 else None
    for t in tensors:
        if t["cosine"] is None or t["cosine"] < COS_CLAIM:
            t["below_claim"] = True
    return {"variant": variant, "format": "w4_ckpt v1", "base": os.path.abspath(base_dir),
            "cutoff": CUTS.get(variant), "alpha": ALPHAS.get(variant), "fit_positions": (moments or {}).get("n") if variant in CUTS else None,
            "dtype": tensors[0]["dtype"] if tensors else None, "tensors": tensors, "changed_weights": total_changed,
            "cosine_with_intended": cos, "cosine_claim": COS_CLAIM, "cosine_meets_claim": bool(cos is not None and all(not t.get("below_claim") for t in tensors)),
            "tensors_compared": len(base), "unchanged_tensors_identical": not any("changed and should not" in f for f in fails),
            "checks": {"ok": not fails, "failures": fails}}


def _write_patch(base_dir, out_dir):
    """CHOPATCH v1 of base -> written; the refusal of the writer is recorded, not raised."""
    try:
        base, edited = D.read_checkpoint(base_dir), D.read_checkpoint(out_dir)
        recs = P.make(base, edited)
        blob = P.encode(recs)
        if len(blob) > P.MAX_FILE:
            raise P.PatchError("q08-patch-io", f"the patch is {len(blob)} bytes, over {P.MAX_FILE}")
        rows, ok = P.check(base, blob, edited)
        if not ok:
            raise P.PatchError("q08-patch-result", "base + patch does not equal the written checkpoint")
        with open(os.path.join(out_dir, PATCH), "wb") as fh:
            fh.write(blob)
        return {"file": PATCH, "bytes": len(blob), "tensors": len(recs), "weights": sum(len(r["cols_idx"]) * r["rows"] for r in recs),
                "read_back_equals_written": True}
    except (P.PatchError, D.ReadError) as e:
        return {"file": None, "refused": str(e)[:300]}


# ------------------------------------------------------------------ commands

def cmd_list(_a):
    for v in VARIANTS:
        what = (f"null-space projection, relative singular value cutoff {CUTS[v]:g}, both layers" if v in CUTS else
                f"published edit on layer {SINGLE[v]} only" if v in SINGLE else f"rank-1 form, alpha {ALPHAS[v]:g}, 64 published columns, both layers")
        print(f"{v:6s} {what}")
    return 0


def cmd_write(a):
    parse_variant(a.variant)
    pub = load_published(a.base, a.edited)
    moments = load_moments(a.work) if a.variant in CUTS else None
    man = write_checkpoint(a.base, a.out, a.variant, pub, moments, patch=not a.no_patch)
    print(f"wrote {a.out}: {a.variant}, {man['changed_weights']:,} weights changed in {man['dtype']}, cosine with the intended edit "
          f"{man['cosine_with_intended']:.6f} ({'meets' if man['cosine_meets_claim'] else 'BELOW'} {COS_CLAIM}); "
          + ("patch " + (f"{man['patch']['bytes']:,} bytes" if man["patch"].get("file") else "refused: " + man["patch"]["refused"]) if "patch" in man else "no patch"))
    for t in man["tensors"]:
        print(f"  {t['name']}: cosine {t['cosine']:.6f}, norm ratio {t['norm_ratio_written_over_intended']:.4f}, {t['changed_weights']:,} changed")
    return 0


def cmd_verify(a):
    """Reload the written directory and compare its logits with the in-memory (float32, unrounded) variant on a few windows."""
    import torch
    sys.path.insert(0, os.path.join(ROOT, "examples", "qwen35"))
    import hf08_ref as H
    parse_variant(a.variant)
    man_path = os.path.join(a.out, MANIFEST)
    try:
        with open(man_path, encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError) as e:
        raise CkptError("w4c-out", f"{man_path}: {str(e)[:200]}")
    pub = load_published(a.base, a.edited)
    edits = intended_edits(a.variant, pub, load_moments(a.work) if a.variant in CUTS else None)
    with open(os.path.join(a.work, "corpus.json"), encoding="utf-8") as f:
        ws = json.load(f)["eval"]
    ws = [w for w in ws if len(w["ids"]) >= 64][:a.windows]
    if not ws:
        raise CkptError("w4c-input", "no evaluation window of at least 64 tokens in the corpus")
    base, mem, back = (H.build_model(d, "f32").to(a.device).eval() for d in (a.base, a.base, a.out))
    mp = dict(mem.named_parameters())
    for l, dw in edits.items():
        mp[f"model.layers.{l}.mlp.down_proj.weight"].data.copy_(torch.from_numpy((pub[l]["w"] + dw).astype(np.float32)))
    rows = []
    with torch.no_grad():
        for w in ws:
            ids = torch.tensor([w["ids"][:a.max_len]], device=a.device)
            lg = [m.lm_head(m.model(ids, use_cache=False).last_hidden_state[0]) for m in (base, mem, back)]
            tg = ids[0, 1:]
            kl_var = L.token_stats(lg[0][:-1], lg[1][:-1], tg.cpu().numpy())[0]
            kl_rt = L.token_stats(lg[1][:-1], lg[2][:-1], tg.cpu().numpy())[0]
            kl_back = L.token_stats(lg[0][:-1], lg[2][:-1], tg.cpu().numpy())[0]
            rows.append({"tokens": int(ids.shape[1]), "kl_base_vs_inmemory_variant": float(kl_var.mean()),
                         "kl_inmemory_vs_reloaded": float(kl_rt.mean()), "kl_base_vs_reloaded": float(kl_back.mean()),
                         "max_abs_logit_diff_inmemory_vs_reloaded": float((lg[1] - lg[2]).abs().max())})
    res = {"variant": a.variant, "windows": rows, "device": a.device, "model_precision": "f32 (checkpoint bf16 values loaded into f32)",
           "mean_kl_inmemory_vs_reloaded": float(np.mean([r["kl_inmemory_vs_reloaded"] for r in rows])),
           "mean_kl_base_vs_inmemory_variant": float(np.mean([r["kl_base_vs_inmemory_variant"] for r in rows])),
           "mean_kl_base_vs_reloaded": float(np.mean([r["kl_base_vs_reloaded"] for r in rows]))}
    man["verify"] = res
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    print(json.dumps(res, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("write", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--variant", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--work", default=os.path.join(ROOT, "build", "w4"))
        p.add_argument("--base", default=BASE_DIR)
        p.add_argument("--edited", default=EDITED_DIR)
        if name == "write":
            p.add_argument("--no-patch", action="store_true")
        else:
            p.add_argument("--windows", type=int, default=4)
            p.add_argument("--max-len", type=int, default=256)
            p.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    try:
        return {"list": cmd_list, "write": cmd_write, "verify": cmd_verify}[a.cmd](a)
    except (CkptError, D.ReadError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (OSError, ImportError) as e:
        print(f"error: w4c-io: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
