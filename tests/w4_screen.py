#!/usr/bin/env python3
"""docs/weight-space-w4.md section 15: examples/weightspace/w4_screen.py on toy data. No model, no download, no GPU.
Needs numpy; without it this prints a skip and exits 0 (the CI runner has none).

    python3 tests/w4_screen.py

  1. The gates on planted logits: an identical variant passes with KL 0; a small perturbation passes; a large one fails kl_mean, kl_p99 and
     ppl_all; a domain-only shift fails kl_domain while the mean passes; the KL is the float64 sum of a direct computation; each `ok`
     equals w4_lib.gate's; NaN logits fail `finite` and still serialize; shape, vocabulary and empty inputs are refused with their tags.
  2. Hashes: the tree hash is stable, changes with one byte, and refuses a directory without safetensors.
  3. Patch to checkpoint: materialize reproduces the edited file byte for byte; a patch of another base is refused (`w4s-patch`);
     patch_matches accepts the right edited checkpoint and refuses another (`w4s-patch-mismatch`).
  4. The evidence and the verifier: a built evidence verifies; every tag of 15.4 fires on its planted fault (missing, not JSON, wrong schema,
     patch changed with and without a valid CRC, patch null, loosened threshold, absent gate, ok disagreeing with value, verdict disagreeing,
     failed gate with exit 1, another base); exit statuses of the CLI.
  5. E2c: planted records give the exit shares and the paired gain; missing records are refused (`w4s-e2c`).
  6. `run` end to end with a stub for the model pass: patch only, edited only and both write evidence that verifies; the refusals of `run`.
  7. Random corruption of the evidence and of the patch never raises anything but the tagged refusal."""
import contextlib
import io
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
import w1_diff as D  # noqa: E402
import w1_patch as P  # noqa: E402
import w4_lib as L  # noqa: E402
import w4_screen as S  # noqa: E402

fails = 0


def expect(name, ok, detail=""):
    global fails
    print("%s %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  " + str(detail)))
    fails += 0 if ok else 1


def refused(rule, fn):
    """True when fn raises ScreenError(rule); the reason otherwise."""
    try:
        fn()
    except S.ScreenError as e:
        return True if e.rule == rule else f"wrong tag {e.rule}: {e}"
    except Exception as e:   # anything else is a panic
        return f"unexpected {type(e).__name__}: {e}"
    return "no refusal"


def cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = S.main(list(argv))
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


def write_st(path, tensors):
    hdr, blobs, off = {}, [], 0
    for n, a in tensors.items():
        raw = (L.bf16_round(np.asarray(a, dtype=np.float32)).view(np.uint32) >> 16).astype("<u2").tobytes()
        hdr[n] = {"dtype": "BF16", "shape": list(np.shape(a)), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def make_ckpt(d, tensors):
    os.makedirs(d)
    write_st(os.path.join(d, "model.safetensors"), tensors)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"text_config": {"vocab_size": 50}}, f)


tmp = tempfile.mkdtemp(prefix="w4_screen_test_")
try:
    # ---------------------------------------------------------------- 1. the gates
    rng = np.random.default_rng(7)
    V, T = 50, 40

    def seqs(shift_fn, doms=("en", "code"), n=6):
        r = np.random.default_rng(11)
        out = []
        for i in range(n):
            lb = r.normal(size=(T, V)) * 2
            lv = shift_fn(lb, doms[i % 2], r)
            out.append((doms[i % 2], lb, lv, r.integers(0, V, T)))
        return out
    ident = S.screen_windows(seqs(lambda lb, d, r: lb.copy()))
    gi = S.gate_table(ident)
    expect("identical variant: KL 0 and every gate passes", ident["all"]["kl"]["mean"] == 0 and all(g["ok"] for g in gi.values()), gi)
    small = S.screen_windows(seqs(lambda lb, d, r: lb + r.normal(size=lb.shape) * 0.02))
    gs = S.gate_table(small)
    expect("a small perturbation passes", all(g["ok"] for g in gs.values()) and 0 < small["all"]["kl"]["mean"] < 0.01, (gs, small["all"]["kl"]))
    big = S.screen_windows(seqs(lambda lb, d, r: lb + r.normal(size=lb.shape) * 2.0))
    gb = S.gate_table(big)
    expect("a large perturbation fails kl_mean, kl_domain, kl_p99", not gb["kl_mean"]["ok"] and not gb["kl_domain"]["ok"] and not gb["kl_p99"]["ok"], gb)
    expect("... and its values and thresholds are recorded", gb["kl_mean"]["threshold"] == L.GATE["kl_mean_hi"] and gb["kl_mean"]["value"] > 0.02)
    # a shift on one domain only, scaled so the pooled upper bound stays under 0.02 but the domain mean is over 0.05
    one = S.screen_windows(seqs(lambda lb, d, r: lb + (r.normal(size=lb.shape) * 0.9 if d == "code" else 0), n=40))
    g1 = S.gate_table(one)
    expect("a one-domain shift fails kl_domain (worst domain over 0.05)", not g1["kl_domain"]["ok"] and one["domains"]["code"]["kl"]["mean"] > 0.05, (one["domains"]["code"]["kl"], g1["kl_domain"]))
    ref = L.gate(big)
    expect("every ok equals w4_lib.gate's", all(gb[k]["ok"] == ref[k][0] for k in gb) and all(g1[k]["ok"] == L.gate(one)[k][0] for k in g1))
    # direct float64 KL of one sequence
    lb, lv = np.random.default_rng(1).normal(size=(T, V)), np.random.default_rng(2).normal(size=(T, V))
    tg = np.arange(T) % V
    pb = np.exp(lb - lb.max(1, keepdims=True))
    pb /= pb.sum(1, keepdims=True)
    qv = np.exp(lv - lv.max(1, keepdims=True))
    qv /= qv.sum(1, keepdims=True)
    direct = float((pb * (np.log(pb) - np.log(qv))).sum(1).mean())
    one_seq = S.screen_windows([("en", lb, lv, tg)])
    expect("the mean KL is the direct float64 value", abs(one_seq["all"]["kl"]["mean"] - direct) < 1e-9, (one_seq["all"]["kl"]["mean"], direct))
    nan = S.screen_windows(seqs(lambda lb, d, r: np.where(np.arange(V) == 3, np.nan, lb)))
    gn = S.gate_table(nan)
    expect("NaN logits fail `finite` and the evidence still serializes", not gn["finite"]["ok"] and S.build_evidence(nan, {"sha256": "0" * 64}, {"sha256": "0" * 64}, None, {}, "t", V) is not None)
    json.dumps(S.build_evidence(nan, {"sha256": "0" * 64}, {"sha256": "0" * 64}, None, {}, "t", V), allow_nan=False)
    expect("a vocabulary mismatch is refused", refused("w4s-model", lambda: S.screen_windows([("en", lb, lv[:, :40], tg)])) is True)
    expect("a target outside the vocabulary is refused", refused("w4s-corpus", lambda: S.screen_windows([("en", lb, lv, tg + V)])) is True)
    expect("no sequence is refused", refused("w4s-corpus", lambda: S.screen_windows([])) is True)

    # ---------------------------------------------------------------- 2. hashes
    r2 = np.random.default_rng(5)
    base_t = {"a.weight": r2.normal(size=(6, 16)), "b.weight": r2.normal(size=(6, 16)), "c": r2.normal(size=(5, 4))}
    edit_t = {k: v.copy() for k, v in base_t.items()}
    edit_t["a.weight"][:, [2, 5, 9]] += 1.0
    edit_t["b.weight"][:, [0, 15]] -= 0.5
    bd, ed = os.path.join(tmp, "base"), os.path.join(tmp, "edited")
    make_ckpt(bd, base_t)
    make_ckpt(ed, edit_t)
    h1, h2, h3 = S.tree_hash(bd), S.tree_hash(bd), S.tree_hash(ed)
    expect("the tree hash is stable and differs between base and edited", h1 == h2 and h1["sha256"] != h3["sha256"] and set(h1["files"]) == {"model.safetensors", "config.json"})
    flip = os.path.join(tmp, "flip")
    shutil.copytree(bd, flip)
    with open(os.path.join(flip, "model.safetensors"), "r+b") as f:
        f.seek(-1, 2)
        b = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([b[0] ^ 1]))
    expect("one flipped byte changes the tree hash", S.tree_hash(flip)["sha256"] != h1["sha256"])
    os.makedirs(os.path.join(tmp, "empty"))
    expect("a directory without safetensors is refused", refused("w4s-input", lambda: S.tree_hash(os.path.join(tmp, "empty"))) is True)
    expect("a missing directory is refused", refused("w4s-input", lambda: S.tree_hash(os.path.join(tmp, "nope"))) is True)

    # ---------------------------------------------------------------- 3. patch to checkpoint
    blob = P.encode(P.make(D.read_checkpoint(bd), D.read_checkpoint(ed)))
    pp = os.path.join(tmp, "x.chopatch")
    open(pp, "wb").write(blob)
    mat = os.path.join(tmp, "mat")
    S.materialize(bd, blob, mat)
    expect("materialize reproduces the edited safetensors byte for byte", S.tree_hash(mat)["files"]["model.safetensors"] == h3["files"]["model.safetensors"])
    S.patch_matches(bd, blob, ed)
    expect("patch_matches accepts the edited checkpoint", True)
    expect("patch_matches refuses another edited checkpoint", refused("w4s-patch-mismatch", lambda: S.patch_matches(bd, blob, flip)) is True)
    other = os.path.join(tmp, "other")
    make_ckpt(other, {k: v + 0.5 for k, v in base_t.items()})
    expect("a patch of another base is refused", refused("w4s-patch", lambda: S.materialize(other, blob, os.path.join(tmp, "mat2"))) is True)
    expect("a corrupt patch is refused", refused("w4s-patch", lambda: S.materialize(bd, blob[:-9], os.path.join(tmp, "mat3"))) is True)
    expect("a non-empty output directory is refused", refused("w4s-out", lambda: S.materialize(bd, blob, bd)) is True)

    # ---------------------------------------------------------------- 4. evidence and verifier
    def evidence(summary=small, patch=True):
        return S.build_evidence(summary, S.tree_hash(bd), S.tree_hash(ed), S.patch_identity(pp, blob) if patch else None,
                                {"corpus_sha256": "1" * 64, "manifest_sha256": None, "windows": 6, "positions": 240, "max_len": 40, "domains": ["code", "en"]}, "toy", V)
    ev = evidence()
    side = pp + ".evidence.json"
    S.write_json(side, ev)
    expect("a passing evidence verifies", S.verify(pp)["verdict"] == "pass" and S.verify(pp, base_dir=bd) is not None)
    expect("the evidence has the schema, the tool, the date, the hashes and the corpus", ev["schema"] == S.SCHEMA and ev["tool"]["version"] and ev["date"].endswith("Z") and
           ev["patch"]["sha256"] and ev["base"]["sha256"] == h1["sha256"] and ev["corpus"]["corpus_sha256"] and set(ev["gates"]) == set(S.GATES) and ev["not_evaluated"] == ["rep4", "new_loops"])
    expect("the patch's crc32 is the trailer", ev["patch"]["crc32"] == struct.unpack("<I", blob[-4:])[0])

    def with_evidence(mod):
        e = json.loads(json.dumps(ev))
        mod(e)
        S.write_json(side, e)

    os.remove(side)
    expect("missing evidence", refused("w4s-evidence-missing", lambda: S.verify(pp)) is True)
    open(side, "w").write("{not json")
    expect("evidence that is not JSON", refused("w4s-evidence-format", lambda: S.verify(pp)) is True)
    open(side, "w").write("[1]")
    expect("evidence that is not an object", refused("w4s-evidence-format", lambda: S.verify(pp)) is True)
    with_evidence(lambda e: e.update(schema="other/1"))
    expect("wrong schema", refused("w4s-evidence-format", lambda: S.verify(pp)) is True)
    with_evidence(lambda e: e["base"].update(sha256="xyz"))
    expect("a hash that is not 64 hex digits", refused("w4s-evidence-format", lambda: S.verify(pp)) is True)
    with_evidence(lambda e: e["gates"]["kl_mean"].update(ok="yes"))
    expect("a gate with a non-boolean ok", refused("w4s-evidence-format", lambda: S.verify(pp)) is True)
    S.write_json(side, ev)
    pp2 = os.path.join(tmp, "y.chopatch")
    open(pp2, "wb").write(blob)
    shutil.copy(side, pp2 + ".evidence.json")
    expect("the same bytes under another name verify (the name is not the identity)", S.verify(pp2) is not None)
    # a changed patch: change one value and keep the CRC valid
    recs = P.decode(blob)
    recs[0]["values"] = recs[0]["values"].copy()
    recs[0]["values"][0, 0] ^= 1
    open(pp2, "wb").write(P.encode(recs))
    expect("a patch changed with a valid CRC", refused("w4s-evidence-patch", lambda: S.verify(pp2)) is True)
    open(pp2, "wb").write(blob[:100] + bytes([blob[100] ^ 1]) + blob[101:])
    expect("a patch changed without a valid CRC", refused("w4s-patch", lambda: S.verify(pp2)) is True)
    open(pp2, "wb").write(b"CHOP")
    expect("a patch that is too short", refused("w4s-patch", lambda: S.verify(pp2)) is True)
    expect("a patch that does not exist", refused("w4s-patch-io", lambda: S.verify(os.path.join(tmp, "none.chopatch"))) is True)
    with_evidence(lambda e: e.update(patch=None))
    expect("evidence for a pair, not a patch", refused("w4s-evidence-patch", lambda: S.verify(pp)) is True)
    with_evidence(lambda e: e["patch"].update(bytes=e["patch"]["bytes"] + 1))
    expect("evidence with the wrong size", refused("w4s-evidence-patch", lambda: S.verify(pp)) is True)

    def loosen(e):
        g = e["gates"]["kl_mean"]
        g["threshold"] = 5.0
    with_evidence(loosen)
    expect("a loosened threshold is refused even though the value passes it", refused("w4s-evidence-thresholds", lambda: S.verify(pp)) is True)
    with_evidence(lambda e: e["gates"].pop("ppl_domain"))
    expect("an absent registered gate", refused("w4s-evidence-thresholds", lambda: S.verify(pp)) is True)
    with_evidence(lambda e: e["gates"]["kl_p99"].update(threshold=None))
    expect("a registered gate without a threshold", refused("w4s-evidence-thresholds", lambda: S.verify(pp)) is True)
    with_evidence(lambda e: e["gates"]["kl_mean"].update(ok=False))
    expect("ok disagreeing with value and threshold", refused("w4s-evidence-verdict", lambda: S.verify(pp)) is True)
    with_evidence(lambda e: e.update(verdict="fail"))
    expect("verdict disagreeing with the gates", refused("w4s-evidence-verdict", lambda: S.verify(pp)) is True)
    S.write_json(side, evidence(big))
    expect("a recorded failed gate is refused", refused("w4s-gate-failed", lambda: S.verify(pp)) is True)
    bad = evidence(big)
    bad["verdict"] = "pass"
    S.write_json(side, bad)
    expect("a failed gate under a verdict of pass", refused("w4s-evidence-verdict", lambda: S.verify(pp)) is True)
    forged = evidence(big)
    forged["gates"]["kl_mean"].update(ok=True)
    S.write_json(side, forged)
    expect("a failed gate flagged ok", refused("w4s-evidence-verdict", lambda: S.verify(pp)) is True)
    S.write_json(side, ev)
    expect("another base", refused("w4s-evidence-base", lambda: S.verify(pp, base_dir=ed)) is True)
    expect("a base that is not a directory", refused("w4s-input", lambda: S.verify(pp, base_dir=os.path.join(tmp, "nope"))) is True)
    rc, out, err = cli("verify", pp)
    expect("CLI: verified exits 0 and states the limits", rc == 0 and "verified" in out and "not signed" in out, (rc, out, err))
    S.write_json(side, evidence(big))
    rc, out, err = cli("verify", pp)
    expect("CLI: a failed gate exits 1 with its tag", rc == 1 and "w4s-gate-failed" in err, (rc, err))
    os.remove(side)
    rc, out, err = cli("verify", pp)
    expect("CLI: missing evidence exits 2 with its tag", rc == 2 and "w4s-evidence-missing" in err, (rc, err))
    S.write_json(side, ev)

    # ---------------------------------------------------------------- 5. E2c
    def rec(i, ms):
        return {"id": i, "nll": 0.0002, "nll_period": 0.0002, "samples": [{"seed": k, "m": m, "toks": []} for k, m in enumerate(ms)]}
    ed2 = os.path.join(tmp, "e2c")
    os.makedirs(ed2)
    with open(os.path.join(ed2, "records_base.jsonl"), "w") as f:
        f.writelines(json.dumps(rec(i, [256] * 4)) + "\n" for i in range(6))
    with open(os.path.join(ed2, "records_edited.jsonl"), "w") as f:
        f.writelines(json.dumps(rec(i, [10, 20, 256, 30] if i % 2 else [5, 6, 7, 8])) + "\n" for i in range(6))
    e2 = S.e2c_pair(ed2)
    expect("E2c: exit shares and paired gain", e2["loops"] == 6 and e2["E_base"] == 0 and abs(e2["E_edited"] - 0.875) < 1e-12 and abs(e2["G_E"][0] - 0.875) < 1e-12 and e2["gate"] is None, e2)
    expect("E2c: missing records are refused", refused("w4s-e2c", lambda: S.e2c_pair(os.path.join(tmp, "nope"))) is True)
    open(os.path.join(ed2, "records_edited.jsonl"), "w").write('{"id": 1}\n')
    expect("E2c: malformed records are refused", refused("w4s-e2c", lambda: S.e2c_pair(ed2)) is True)

    # ---------------------------------------------------------------- 6. run, end to end with a stub model pass
    work = os.path.join(tmp, "work")
    os.makedirs(work)
    wins = [{"domain": "en" if i % 2 else "code", "ids": [int(x) for x in np.random.default_rng(i).integers(0, V, 30)]} for i in range(6)]
    json.dump({"fit": [], "eval": wins}, open(os.path.join(work, "corpus.json"), "w"))
    json.dump({"salt": "x"}, open(os.path.join(work, "manifest.json"), "w"))

    def stub(base_dir, edited_dir, windows, device="cpu", max_len=512):
        assert os.path.isfile(os.path.join(edited_dir, "model.safetensors"))
        r = np.random.default_rng(3)
        for w in windows:
            n = min(len(w["ids"]), max_len) - 1
            lb = r.normal(size=(n, V))
            yield w["domain"], lb, lb + r.normal(size=lb.shape) * 0.01, np.array(w["ids"][1:n + 1])
    real = S.collect
    S.collect = stub
    try:
        for kind, extra in (("patch only", ["--patch", pp]), ("edited only", ["--edited", ed]), ("both", ["--patch", pp, "--edited", ed])):
            out = os.path.join(tmp, "ev_%s.json" % kind.replace(" ", "_"))
            rc, so, se = cli("run", "--base", bd, "--corpus", work, "--out", out, "--label", "toy", *extra)
            e = json.load(open(out)) if os.path.exists(out) else {}
            expect(f"run ({kind}) writes a passing evidence", rc == 0 and e.get("verdict") == "pass" and e["corpus"]["windows"] == 6 and e["corpus"]["manifest_sha256"] and
                   e["edited"]["sha256"] == h3["sha256"] and e["model"]["vocab"] == 50 and (e["patch"] is not None) == ("--patch" in extra), (rc, se, so))
        shutil.copy(os.path.join(tmp, "ev_patch_only.json"), pp2 + ".evidence.json")
        open(pp2, "wb").write(blob)
        expect("the evidence of a patch-only run verifies the patch", S.verify(pp2, base_dir=bd)["verdict"] == "pass")
        rc, so, se = cli("run", "--base", bd, "--edited", ed, "--corpus", work, "--out", os.path.join(tmp, "ev_e2c.json"), "--e2c-records", ed2.replace("e2c", "e2c_none"))
        expect("run refuses unusable E2c records (w4s-e2c, exit 2) and writes nothing", rc == 2 and "w4s-e2c" in se and not os.path.exists(os.path.join(tmp, "ev_e2c.json")), (rc, se))
        shutil.rmtree(ed2)
        os.makedirs(ed2)
        for n in ("base", "edited"):
            open(os.path.join(ed2, f"records_{n}.jsonl"), "w").writelines(json.dumps(rec(i, [256] * 4 if n == "base" else [3] * 4)) + "\n" for i in range(4))
        out = os.path.join(tmp, "ev_e2c.json")
        rc, so, se = cli("run", "--base", bd, "--edited", ed, "--corpus", work, "--out", out, "--e2c-records", ed2)
        e = json.load(open(out))
        expect("run with E2c records reports the proxy and does not gate on it", rc == 0 and e["e2c"]["G_E"][0] == 1.0 and e["e2c"]["gate"] is None and e["verdict"] == "pass", (rc, se))
    finally:
        S.collect = real
    base_args = ["--base", bd, "--edited", ed, "--corpus", work, "--out", os.path.join(tmp, "z.json")]
    for what, argv, tag in (("no edited and no patch", ["--base", bd, "--corpus", work, "--out", os.path.join(tmp, "z.json")], "w4s-input"),
                            ("a base that is not a directory", ["--base", os.path.join(tmp, "nope"), "--edited", ed, "--corpus", work, "--out", os.path.join(tmp, "z.json")], "w4s-input"),
                            ("no corpus", ["--base", bd, "--edited", ed, "--corpus", os.path.join(tmp, "nope"), "--out", os.path.join(tmp, "z.json")], "w4s-corpus"),
                            ("a patch that is not the base's", ["--base", other, "--patch", pp, "--corpus", work, "--out", os.path.join(tmp, "z.json")], "w4s-patch"),
                            ("a patch that does not make the edited checkpoint", ["--base", bd, "--edited", flip, "--patch", pp, "--corpus", work, "--out", os.path.join(tmp, "z.json")], "w4s-patch-mismatch"),
                            ("an unwritable output", base_args[:-1] + [os.path.join(tmp, "nodir", "z.json")], "w4s-out"),
                            ("--max-len 1", base_args + ["--max-len", "1"], "w4s-input")):
        rc, so, se = cli("run", *argv)
        expect(f"run refuses {what}", rc == 2 and (tag in se or tag == "w4s-out") and "Traceback" not in se, (rc, se))
    json.dump({"eval": [{"domain": "en", "ids": [1]}]}, open(os.path.join(work, "corpus.json"), "w"))
    rc, so, se = cli("run", *base_args)
    expect("run refuses a window of one id", rc == 2 and "w4s-corpus" in se, se)
    open(os.path.join(work, "corpus.json"), "w").write("{")
    rc, so, se = cli("run", *base_args)
    expect("run refuses a corrupt corpus", rc == 2 and "w4s-corpus" in se, se)

    # ---------------------------------------------------------------- 7. corruption never panics
    good_ev = open(side, "rb").read()
    r7 = np.random.default_rng(99)
    panics = 0
    for k in range(300):
        b = bytearray(good_ev)
        for _ in range(int(r7.integers(1, 4))):
            b[int(r7.integers(0, len(b)))] = int(r7.integers(0, 256))
        if k % 3 == 0:
            b = b[:int(r7.integers(0, len(b)))]
        open(side, "wb").write(bytes(b))
        try:
            S.verify(pp)
        except S.ScreenError:
            pass
        except Exception:
            panics += 1
    expect("300 corrupted evidence files: only tagged refusals", panics == 0, panics)
    open(side, "wb").write(good_ev)
    panics = 0
    for k in range(200):
        b = bytearray(blob)
        for _ in range(int(r7.integers(1, 4))):
            b[int(r7.integers(0, len(b)))] = int(r7.integers(0, 256))
        if k % 3 == 0:
            b = b[:int(r7.integers(0, len(b)))]
        open(pp2, "wb").write(bytes(b))
        shutil.copy(side, pp2 + ".evidence.json")
        try:
            S.verify(pp2)
        except S.ScreenError:
            pass
        except Exception:
            panics += 1
        try:
            S.materialize(bd, bytes(b), os.path.join(tmp, "fz%d" % k))
        except S.ScreenError:
            pass
        except Exception:
            panics += 1
    expect("200 corrupted patches: only tagged refusals", panics == 0, panics)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("w4_screen: %s" % ("ok" if not fails else "%d failed" % fails))
sys.exit(1 if fails else 0)
