#!/usr/bin/env python3
"""docs/weight-space-w4.md section 13: examples/weightspace/w4_ckpt.py on a toy checkpoint. No model, no download, no GPU.
Needs numpy; without it this prints a skip and exits 0 (the CI runner has none).

    python3 tests/w4_ckpt.py

  1. A bf16 toy base (two layer tensors [6, 16] under the real names and one other tensor) and a planted published edit (rank 1, remove -6 times
     the u component on 4 columns): `e15` changes layer 15 only; every other byte of every file, the header, the index and the config are
     the base's; the changed tensors hold exactly bf16(base + intended); the manifest's cosine, norm ratio and changed count are recomputed here.
  2. `a6` of the planted edit reproduces it (cosine 1); `a3` is half of it; the support is the 4 columns; the formula is checked entry by entry.
  3. `x*`: the written edit is annihilated by the retained directions of K0 (to bf16 resolution), a higher cutoff keeps more of the edit, a
     K0 that excludes the edit leaves it whole, and a left projection (a mutant) would fail the same test.
  4. The same writer on an f32 and an f16 toy base (the base's dtype is kept).
  5. The patch: read back, base + patch equals the written tensors; a tensor outside the patch format is recorded as refused, not raised.
  6. The refusals, each with its rule tag, and the CLI's exit status 2: unknown variant, a non-empty output, a second moment of the wrong
     size / missing / not finite, an edit that is empty or not finite, a base that is not a directory / has no safetensors / is truncated /
     has a corrupt header / holds an unsupported dtype; a cut-off write leaves no `.partial` and no output.
  7. Random bytes and truncations of the toy file never raise anything but the tagged refusal."""
import io
import contextlib
import json
import os
import shutil
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, os.path.join(ROOT, "examples", "weightspace"))
try:
    import numpy as np
except ImportError:
    print("skip: numpy is not installed")
    sys.exit(0)
import w4_ckpt as K  # noqa: E402
import w4_lib as L  # noqa: E402

fails = 0


def expect(name, ok, detail=""):
    global fails
    print("%s %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  " + str(detail)))
    fails += 0 if ok else 1


def raises(rule, fn):
    try:
        fn()
    except K.CkptError as e:
        return e.rule == rule
    except Exception as e:   # anything else is a panic
        return "unexpected %s: %s" % (type(e).__name__, e)
    return "no refusal"


R, C_IN = 6, 16
N15, N19 = K.NAME_FMT.format(15), K.NAME_FMT.format(19)
OTHER = "model.language_model.embed_tokens.weight"


def to_bf16_bits(x):
    return (L.bf16_round(np.asarray(x, dtype=np.float32)).view(np.uint32) >> 16).astype("<u2")


def write_st(path, tensors, dtypes=None):
    """tensors {name: float array}; dtypes {name: "BF16"|"F32"|"F16"|"I64"}; a safetensors file."""
    dtypes = dtypes or {}
    hdr, blobs, off = {}, [], 0
    for n, a in tensors.items():
        dt = dtypes.get(n, "BF16")
        if dt == "BF16":
            raw = to_bf16_bits(a).tobytes()
        elif dt == "F32":
            raw = np.asarray(a, dtype="<f4").tobytes()
        elif dt == "F16":
            raw = np.asarray(a, dtype="<f2").tobytes()
        else:
            raw = np.asarray(a, dtype="<i8").tobytes()
        hdr[n] = {"dtype": dt, "shape": list(np.shape(a)), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def make_base(d, dtype="BF16", other_dtype=None):
    rng = np.random.default_rng(3)
    os.makedirs(d)
    w = {N15: rng.normal(size=(R, C_IN)) * 0.5, N19: rng.normal(size=(R, C_IN)) * 0.5, OTHER: rng.normal(size=(5, 4))}
    dts = {N15: dtype, N19: dtype, OTHER: other_dtype or dtype}
    w = {k: (L.bf16_round(v.astype(np.float32)) if dtype == "BF16" else v.astype(np.float16) if dtype == "F16" else v.astype(np.float32)).astype(np.float64)
         for k, v in w.items()}
    write_st(os.path.join(d, "model.safetensors-00001-of-00001.safetensors"), w, dts)
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model.safetensors-00001-of-00001.safetensors" for k in w}}, f)
    with open(os.path.join(d, "config.json"), "w") as f:
        f.write('{"toy": true}')
    return {l: w[K.NAME_FMT.format(l)] for l in K.LAYERS}


def planted(w_base, alpha=6.0):
    """{layer: {"w", "dw"}}: dw[:, j] = -alpha (u . W_j) u on columns 2, 5, 9, 12 with a fixed u."""
    rng = np.random.default_rng(9)
    pub = {}
    for l, w in w_base.items():
        u = rng.normal(size=R)
        u /= np.linalg.norm(u)
        dw = np.zeros_like(w)
        for j in (2, 5, 9, 12):
            dw[:, j] = -alpha * (u @ w[:, j]) * u
        pub[l] = {"w": w, "dw": dw, "u": u}
    return pub


def cos(a, b):
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def tensors_of(d):
    import w1_diff as D
    return D.read_checkpoint(d)


tmp = tempfile.mkdtemp(prefix="w4ckpt_")
try:
    base_dir = os.path.join(tmp, "base")
    wb = make_base(base_dir)
    pub = planted(wb)
    bfile = os.path.join(base_dir, "model.safetensors-00001-of-00001.safetensors")
    raw_base = open(bfile, "rb").read()

    # ---- 1. e15: only layer 15 changes; all else byte-identical
    out = os.path.join(tmp, "e15")
    man = K.write_checkpoint(base_dir, out, "e15", pub)
    raw_out = open(os.path.join(out, os.path.basename(bfile)), "rb").read()
    (hl,) = struct.unpack("<Q", raw_base[:8])
    expect("header and size are the base's", raw_out[:8 + hl] == raw_base[:8 + hl] and len(raw_out) == len(raw_base))
    base_t, out_t = tensors_of(base_dir), tensors_of(out)
    expect("e15 changes layer 15 only", [n for n in base_t if bytes(base_t[n].raw) != bytes(out_t[n].raw)] == [N15])
    want = to_bf16_bits(pub[15]["w"] + pub[15]["dw"])
    expect("the changed tensor is bf16(base + intended), bit for bit", np.array_equal(out_t[N15].unsigned(), want))
    expect("config and index copied", open(os.path.join(out, "config.json")).read() == '{"toy": true}' and
           json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"][N15])
    ch = int((to_bf16_bits(pub[15]["w"]) != want).sum())
    expect("the manifest's changed count and cosine are recomputed here", man["changed_weights"] == ch and
           abs(man["cosine_with_intended"] - cos(L.bf16_round((pub[15]["w"] + pub[15]["dw"]).astype(np.float32)).astype(np.float64) - pub[15]["w"], pub[15]["dw"])) < 1e-12,
           (man["changed_weights"], ch))
    expect("the manifest on disk is the returned one", json.load(open(os.path.join(out, K.MANIFEST)))["changed_weights"] == ch)
    expect("no .partial left", not os.path.exists(out + ".partial"))
    expect("the read-back check passed", man["checks"]["ok"] and man["unchanged_tensors_identical"])
    expect("e19 touches layer 19 only", [t["name"] for t in K.write_checkpoint(base_dir, os.path.join(tmp, "e19"), "e19", pub)["tensors"]] == [N19])

    # ---- 2. alpha
    ws = {}
    for a_name, alpha in K.ALPHAS.items():
        m = K.write_checkpoint(base_dir, os.path.join(tmp, a_name), a_name, pub, patch=False)
        ws[a_name] = tensors_of(os.path.join(tmp, a_name))[N15].as_float64() - pub[15]["w"]
    expect("a6 of a planted alpha=6 edit reproduces it", cos(ws["a6"], pub[15]["dw"]) > 0.9999, cos(ws["a6"], pub[15]["dw"]))
    expect("a3 is half of a6 (to bf16 resolution)", np.linalg.norm(2 * ws["a3"] - ws["a6"]) / np.linalg.norm(ws["a6"]) < 0.02)
    u = pub[15]["u"]
    f = np.zeros_like(pub[15]["w"])
    for j in (2, 5, 9, 12):
        f[:, j] = -9.0 * (u @ pub[15]["w"][:, j]) * u
    expect("a9 equals -9 (u . W_j) u entry by entry on the support", np.linalg.norm(ws["a9"] - f) / np.linalg.norm(f) < 0.02)
    expect("a-variants are zero off the 4 columns", np.abs(np.delete(ws["a9"], [2, 5, 9, 12], 1)).max() == 0)

    # ---- 3. null space
    rng = np.random.default_rng(5)
    q, _ = np.linalg.qr(rng.normal(size=(C_IN, C_IN)))
    spec = np.array([1.0] * 3 + [0.0225] * 3 + [0.0049] * 3 + [1e-6] * (C_IN - 9))   # relative singular values (sqrt): 1, .15, .07, .001
    n_pos = 100
    cm = (q * (spec * n_pos)) @ q.T
    mom = {"n": n_pos, 15: cm, 19: cm}
    got = {}
    for v, cut in K.CUTS.items():
        m = K.write_checkpoint(base_dir, os.path.join(tmp, v), v, pub, mom, patch=False)
        got[v] = tensors_of(os.path.join(tmp, v))[N15].as_float64() - pub[15]["w"]
    r = {v: L.retained(L.eig_of(cm)[0], n_pos, cut) for v, cut in K.CUTS.items()}
    expect("the toy spectrum retains 9, 6 and 3 directions at 5e-2, 1e-1, 2e-1", (r["x5e-2"], r["x1e-1"], r["x2e-1"]) == (9, 6, 3), r)
    ev, qq = L.eig_of(cm)
    for v, cut in K.CUTS.items():
        rr = L.retained(ev, n_pos, cut)
        res = np.abs(got[v] @ qq[:, :rr]).max() / np.abs(pub[15]["dw"]).max()
        expect(f"{v}: the written edit is annihilated by the {rr} retained directions", res < 0.02, res)
    nrm = [np.linalg.norm(got[v]) for v in ("x5e-2", "x1e-1", "x2e-1")]
    expect("a higher cutoff keeps more of the edit", nrm[0] <= nrm[1] + 1e-9 <= nrm[2] + 2e-9, nrm)
    mut = pub[15]["dw"] - qq[:R, :6] @ (qq[:R, :6].T @ pub[15]["dw"])
    expect("a left projection (mutant) fails the null-space test", np.abs(mut @ qq[:, :6]).max() / np.abs(pub[15]["dw"]).max() > 0.02)
    support = np.zeros((C_IN, C_IN))
    support[[0, 1, 2], [0, 1, 2]] = 1.0 * n_pos   # K0 on columns 0..2 only; the edit's columns 2, 5, 9, 12: 2 is partly under
    mom2 = {"n": n_pos, 15: support, 19: support}
    m2 = K.intended_edits("x1e-1", pub, mom2)
    expect("a K0 that excludes the edit's columns leaves them whole", np.allclose(m2[15][:, [5, 9, 12]], pub[15]["dw"][:, [5, 9, 12]]))
    expect("and removes the part on the excited column", np.abs(m2[15][:, 2]).max() < 1e-12)

    # ---- 4. dtypes
    for dt in ("F32", "F16"):
        d = os.path.join(tmp, "base_" + dt)
        w = make_base(d, dt)
        p = planted(w)
        m = K.write_checkpoint(d, os.path.join(tmp, "o_" + dt), "e15", p, patch=False)
        o = tensors_of(os.path.join(tmp, "o_" + dt))
        expect(f"{dt}: the dtype is kept, cosine near 1", o[N15].dtype == dt and m["cosine_with_intended"] > 0.9999 and m["dtype"] == dt, m["cosine_with_intended"])
    # an unsupported dtype outside the edited tensors is fine, inside is refused
    d = os.path.join(tmp, "base_odd")
    w = make_base(d, "BF16", other_dtype="I64")
    expect("an int64 tensor outside the edit is copied untouched", K.write_checkpoint(d, os.path.join(tmp, "o_odd"), "e19", planted(w), patch=False)["checks"]["ok"])

    # ---- 5. patch
    man = K.write_checkpoint(base_dir, os.path.join(tmp, "xp"), "x1e-1", pub, mom)
    expect("the patch is written and read back", man["patch"].get("file") == K.PATCH and man["patch"]["read_back_equals_written"], man["patch"])
    import w1_patch as P
    blob = open(os.path.join(tmp, "xp", K.PATCH), "rb").read()
    rows, ok = P.check(tensors_of(base_dir), blob, tensors_of(os.path.join(tmp, "xp")))
    expect("base + patch equals the written checkpoint, tensor by tensor", ok and len(rows) == 3)
    odd_out = K.write_checkpoint(d, os.path.join(tmp, "o_odd2"), "e15", planted(w))   # I64 outside, BF16 matrices inside: patchable
    expect("a patch is made when only BF16 matrices differ", odd_out["patch"].get("file") == K.PATCH)

    # ---- 6. refusals
    base2 = os.path.join(tmp, "r_out")
    cases = [
        ("unknown variant", "w4c-variant", lambda: K.write_checkpoint(base_dir, base2, "x9", pub)),
        ("variant not a string", "w4c-variant", lambda: K.write_checkpoint(base_dir, base2, None, pub)),
        ("x* without moments", "w4c-moments", lambda: K.write_checkpoint(base_dir, base2, "x1e-1", pub)),
        ("moments of the wrong size", "w4c-moments", lambda: K.write_checkpoint(base_dir, base2, "x1e-1", pub, {"n": 5, 15: np.eye(3), 19: np.eye(3)})),
        ("moments with a NaN", "w4c-moments", lambda: K.write_checkpoint(base_dir, base2, "x1e-1", pub, {"n": 5, 15: np.full((C_IN, C_IN), np.nan), 19: np.eye(C_IN)})),
        ("moments with no positions", "w4c-moments", lambda: K.write_checkpoint(base_dir, base2, "x1e-1", pub, {"n": 0, 15: np.eye(C_IN), 19: np.eye(C_IN)})),
        ("an empty edit", "w4c-input", lambda: K.write_checkpoint(base_dir, base2, "e15", {15: {"w": wb[15], "dw": np.zeros_like(wb[15])}})),
        ("an edit with a NaN", "w4c-input", lambda: K.write_checkpoint(base_dir, base2, "e15", {15: {"w": wb[15], "dw": np.full_like(wb[15], np.nan)}})),
        ("an edit of the wrong shape", "w4c-input", lambda: K.write_checkpoint(base_dir, base2, "e15", {15: {"w": wb[15], "dw": np.ones((2, 2))}})),
        ("a missing layer", "w4c-input", lambda: K.write_checkpoint(base_dir, base2, "a6", {15: pub[15]})),
        ("base weights that are not the base's", "w4c-input", lambda: K.write_checkpoint(base_dir, base2, "e15", {15: {"w": wb[15] + 1.0, "dw": pub[15]["dw"]}})),
        ("an output that is not empty", "w4c-out", lambda: K.write_checkpoint(base_dir, os.path.join(tmp, "e15"), "e15", pub)),
        ("a leftover .partial", "w4c-out", lambda: (os.makedirs(base2 + ".partial"), K.write_checkpoint(base_dir, base2, "e15", pub))),
        ("a base that is not a directory", "w4c-base", lambda: K.write_checkpoint(bfile, base2 + "2", "e15", pub)),
    ]
    for name, rule, fn in cases:
        res = raises(rule, fn)
        expect("refused " + name + " (" + rule + ")", res is True, res)
    shutil.rmtree(base2 + ".partial", ignore_errors=True)
    expect("a refused write leaves no output", not os.path.exists(base2) and not os.path.exists(base2 + ".partial"))
    empty = os.path.join(tmp, "empty")
    os.makedirs(empty)
    expect("a base with no safetensors is refused", raises("w4c-base", lambda: K.write_checkpoint(empty, base2 + "3", "e15", pub)) is True)
    # an existing EMPTY output directory is fine
    os.makedirs(os.path.join(tmp, "emptyout"))
    expect("an empty output directory is accepted", K.write_checkpoint(base_dir, os.path.join(tmp, "emptyout"), "e15", pub, patch=False)["checks"]["ok"])
    # unsupported dtype in an edited tensor
    d8 = os.path.join(tmp, "base_i8")
    w8 = make_base(d8, "BF16")
    write_st(os.path.join(d8, "model.safetensors-00001-of-00001.safetensors"), {N15: np.ones((R, C_IN)), N19: np.ones((R, C_IN)), OTHER: np.ones((5, 4))},
             {N15: "I64", N19: "BF16", OTHER: "BF16"})
    expect("an edited tensor of an unsupported dtype is refused", raises("w4c-dtype", lambda: K.write_checkpoint(d8, base2 + "4", "e15", {15: {"w": np.ones((R, C_IN)), "dw": pub[15]["dw"]}})) is True)
    # corrupt bases
    for name, mut in (("truncated", lambda b: b[:len(b) // 2]), ("header length past the file", lambda b: struct.pack("<Q", 1 << 40) + b[8:]),
                      ("header not JSON", lambda b: b[:8] + b"\xff" * (len(b) - 8)), ("under 8 bytes", lambda b: b[:5])):
        dc = os.path.join(tmp, "c_" + name.replace(" ", "_"))
        shutil.copytree(base_dir, dc)
        open(os.path.join(dc, os.path.basename(bfile)), "wb").write(mut(raw_base))
        expect("a base that is " + name + " is refused with its tag", raises("w4c-base", lambda: K.write_checkpoint(dc, dc + "_o", "e15", pub)) is True)
        expect("... and writes nothing", not os.path.exists(dc + "_o") and not os.path.exists(dc + "_o.partial"))

    # ---- CLI: exit status 2 with the rule tag, no traceback
    def cli(args):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = K.main(args)
        return code, err.getvalue()
    code, err = cli(["write", "--variant", "nope", "--out", os.path.join(tmp, "cli1"), "--base", base_dir, "--edited", base_dir])
    expect("CLI: unknown variant exits 2 with the tag", code == 2 and "w4c-variant" in err, (code, err))
    code, err = cli(["write", "--variant", "x1e-1", "--out", os.path.join(tmp, "cli2"), "--base", base_dir, "--edited", base_dir, "--work", os.path.join(tmp, "nowork")])
    expect("CLI: an identical edited checkpoint or a missing work dir exits 2 with a tag", code == 2 and "w4c-" in err, (code, err))
    code, err = cli(["write", "--variant", "e15", "--out", os.path.join(tmp, "cli3"), "--base", os.path.join(tmp, "missing"), "--edited", base_dir])
    expect("CLI: a missing base exits 2", code == 2 and "w4c-base" in err, (code, err))
    code, err = cli(["list"])
    expect("CLI: list exits 0", code == 0)

    # ---- 7. fuzz: nothing but the tagged refusal
    rng = np.random.default_rng(11)
    bad, tried = 0, 0
    for i in range(60):
        dc = os.path.join(tmp, f"f{i}")
        shutil.copytree(base_dir, dc)
        b = bytearray(raw_base)
        if i % 3 == 0:
            b = b[:int(rng.integers(0, len(b)))]
        else:
            for _ in range(int(rng.integers(1, 6))):
                b[int(rng.integers(0, min(len(b), 8 + hl)))] = int(rng.integers(0, 256))
        open(os.path.join(dc, os.path.basename(bfile)), "wb").write(bytes(b))
        tried += 1
        try:
            K.write_checkpoint(dc, dc + "_o", "e15", pub, patch=False)
        except K.CkptError:
            pass
        except Exception as e:
            bad += 1
            print("   unexpected", type(e).__name__, e)
        shutil.rmtree(dc + "_o.partial", ignore_errors=True)
    expect(f"{tried} corrupted bases raise only the tagged refusal", bad == 0, bad)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("w4_ckpt: %s" % ("ok" if not fails else "%d FAILED" % fails))
sys.exit(1 if fails else 0)
