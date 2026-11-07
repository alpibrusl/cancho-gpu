#!/usr/bin/env python3
"""docs/weight-space-w4.md section 15: screen an edit against the registered collateral gates and write ONE evidence file; verify a patch's.

    python3 w4_screen.py run    --base DIR (--edited DIR | --patch FILE) --corpus WORKDIR --out EVIDENCE.json
                                [--label NAME] [--device cpu|mps] [--max-len 512] [--windows N]
                                [--loops loops.json [--e2c-dir DIR]] [--e2c-records DIR]
    python3 w4_screen.py verify PATCH [--evidence FILE] [--base DIR]

`run` hashes the checkpoints (and the patch), scores the corpus' eval split with the base and the edited model (KL, perplexity: the gates of
4.3 through w4_lib.gate) and, with a loops file or records, E2c's exit share (reported, never gated). `verify` refuses a patch whose
sidecar evidence is missing, belongs to another file, has loosened thresholds or records a failed gate. The evidence is NOT signed (15.1).
The model passes (`collect`, the E2c run) need torch and transformers and are not run by the tests; everything else is numpy.
Exit status: 0 done, 1 a gate failed, 2 an input or evidence is unusable. Every refusal prints its tag `w4s-*` (15.4, 15.5)."""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w1_diff as D  # noqa: E402
import w1_patch as P  # noqa: E402
import w4_ckpt as K  # noqa: E402
import w4_e2c as E  # noqa: E402
import w4_lib as L  # noqa: E402

SCHEMA = "cancho-patch-evidence/1"
TOOL = {"name": "w4_screen.py", "version": "1", "rules": "docs/weight-space-w4.md 4.3, 14, 15"}
NOT_EVALUATED = ["rep4", "new_loops"]
MAX_EVIDENCE = 1 << 20
# gate name -> (key of w4_lib.GATE, w4_lib.gate's clause)
GATES = {"kl_mean": "kl_mean_hi", "kl_domain": "kl_domain_mean", "kl_p99": "kl_tok_p99", "ppl_all": "ppl_hi", "ppl_domain": "ppl_domain",
         "finite": None}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ScreenError(Exception):
    """A refusal; `rule` is its tag."""

    def __init__(self, rule, msg):
        super().__init__(f"{rule}: {msg}")
        self.rule = rule


# ------------------------------------------------------------------ hashes

def sha256_file(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
    except OSError as e:
        raise ScreenError("w4s-input", f"{path}: {e}")
    return h.hexdigest()


def tree_hash(directory):
    """{'sha256': tree hash, 'files': {name: sha256}} of a checkpoint directory: config.json, the safetensors index and every *.safetensors."""
    if not os.path.isdir(directory):
        raise ScreenError("w4s-input", f"{directory}: not a directory")
    names = sorted(set(glob.glob(os.path.join(directory, "*.safetensors"))) |
                   {p for p in (os.path.join(directory, "config.json"), os.path.join(directory, "model.safetensors.index.json")) if os.path.isfile(p)})
    if not any(n.endswith(".safetensors") for n in names):
        raise ScreenError("w4s-input", f"{directory}: no *.safetensors file")
    files = {os.path.basename(n): sha256_file(n) for n in names}
    lines = "".join(f"{k} {v}\n" for k, v in sorted(files.items()))
    return {"sha256": hashlib.sha256(lines.encode()).hexdigest(), "files": files}


def read_patch(path):
    try:
        with open(path, "rb") as f:
            blob = f.read(P.MAX_FILE + 1)
    except OSError as e:
        raise ScreenError("w4s-patch-io", f"{path}: {e}")
    if len(blob) > P.MAX_FILE:
        raise ScreenError("w4s-patch-io", f"{path}: over {P.MAX_FILE} bytes")
    return blob


def patch_identity(path, blob):
    """The evidence's `patch` object. `crc32` is the patch's own trailer (the CRC-32 of every byte before it): the CRC-32 of the whole file
    is a constant, and the cancho loader compares the trailer."""
    if len(blob) < 20 or zlib.crc32(blob[:-4]) != int.from_bytes(blob[-4:], "little"):
        raise ScreenError("w4s-patch", "q08-patch-checksum: the CRC-32 does not match the trailer (or the file is too short for a patch)")
    return {"file": os.path.basename(path), "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(), "crc32": zlib.crc32(blob[:-4])}


# ------------------------------------------------------------------ patch -> checkpoint

def materialize(base_dir, blob, out_dir):
    """Write `out_dir` (empty or absent): the base directory with the patch's columns written. Refuses a patch that is not the base's."""
    try:
        recs = P.decode(blob)
        lay = K._layout(base_dir)
    except P.PatchError as e:
        raise ScreenError("w4s-patch", str(e))
    except K.CkptError as e:
        raise ScreenError("w4s-input", str(e))
    plan = []
    for r in recs:
        n = r["name"]
        if n not in lay or lay[n][1] != "BF16" or tuple(lay[n][2]) != (r["rows"], r["cols"]):
            raise ScreenError("w4s-patch", f"q08-patch-unknown: `{n}` is not a BF16 [{r['rows']}, {r['cols']}] tensor of the base")
        try:
            base_u = K._read_raw(lay, n)
        except K.CkptError as e:
            raise ScreenError("w4s-input", str(e))
        if P.tensor_hashes(base_u.astype("<u2").tobytes()) != r["base_hash"]:
            raise ScreenError("w4s-patch", f"q08-patch-base: `{n}` in the base is not the tensor the patch was made from")
        new = P.apply_to(base_u, r).astype("<u2")
        if P.tensor_hashes(new.tobytes()) != r["edited_hash"]:
            raise ScreenError("w4s-patch", f"q08-patch-result: `{n}` patched is not the tensor the patch says it makes")
        plan.append((lay[n][0], lay[n][3], new.tobytes()))
    try:
        os.makedirs(out_dir, exist_ok=True)
        if os.listdir(out_dir):
            raise ScreenError("w4s-out", f"{out_dir} is not empty")
        for p in sorted(os.listdir(base_dir)):
            if os.path.isfile(os.path.join(base_dir, p)):
                shutil.copyfile(os.path.join(base_dir, p), os.path.join(out_dir, p))
        for f, off, raw in plan:
            with open(os.path.join(out_dir, os.path.basename(f)), "r+b") as fh:
                fh.seek(off)
                fh.write(raw)
    except OSError as e:
        raise ScreenError("w4s-out", str(e))


def patch_matches(base_dir, blob, edited_dir):
    try:
        rows, ok = P.check(D.read_checkpoint(base_dir), blob, D.read_checkpoint(edited_dir))
    except (P.PatchError, D.ReadError) as e:
        raise ScreenError("w4s-patch", str(e))
    if not ok:
        raise ScreenError("w4s-patch-mismatch", "base + patch is not the edited checkpoint: " + ", ".join(n for n, _, e in rows if not e)[:300])


# ------------------------------------------------------------------ the gates (pure)

def screen_windows(items):
    """The summary of w4_lib.gate from an iterable of (domain, logits_base [n, V], logits_variant [n, V], targets [n]), one item a sequence.
    Positions are scored as w4_collateral does: KL(base || variant), NLL of the target under each, paired difference d."""
    acc, bad = {}, False
    for dom, lb, lv, tg in items:
        lb, lv = np.asarray(lb), np.asarray(lv)
        if lb.ndim != 2 or lb.shape != lv.shape or len(tg) != lb.shape[0] or lb.shape[0] < 1:
            raise ScreenError("w4s-model", f"logits of shape {lb.shape} and {lv.shape} for {len(tg)} targets: the models disagree (vocabulary?)")
        if int(np.max(tg)) >= lb.shape[1] or int(np.min(tg)) < 0:
            raise ScreenError("w4s-corpus", "a target id is outside the vocabulary")
        kl, nb, nv, _ent, ag = L.token_stats(lb, lv, tg)
        bad = bad or not (np.isfinite(kl).all() and np.isfinite(nv).all() and np.isfinite(nb).all())
        for scope in ("all", dom):
            a = acc.setdefault(scope, {"kl": [], "d": [], "skl": [], "sd": [], "n": [], "nb": 0.0, "ag": []})
            a["kl"].append(kl)
            a["skl"].append(float(kl.sum()))
            a["sd"].append(float((nv - nb).sum()))
            a["n"].append(len(kl))
            a["nb"] += float(nb.sum())
            a["ag"].append(ag)
    if "all" not in acc:
        raise ScreenError("w4s-corpus", "no sequence was scored")

    def one(a):
        kl = np.concatenate(a["kl"])
        n, skl, sd = np.array(a["n"]), np.array(a["skl"]), np.array(a["sd"])
        with np.errstate(all="ignore"):
            ci = L.ratio_ci(skl, n)
            dd = L.ratio_ci(sd, n)
        if ci is None or dd is None:
            raise ScreenError("w4s-corpus", "no scored positions")
        ok = bool(np.isfinite(kl).all()) and not bad
        return {"positions": int(n.sum()), "sequences": int(n.size), "kl": ci, "kl_tok": L.dist(kl),
                "dppl": {"mean": float(np.expm1(dd["mean"])), "lo": float(np.expm1(dd["lo"])), "hi": float(np.expm1(dd["hi"]))},
                "top1_agree": float(np.concatenate(a["ag"]).mean()), "finite": ok}
    out = {"all": one(acc.pop("all")), "domains": {}}
    for k in sorted(acc):
        out["domains"][k] = one(acc[k])
    out["all"]["finite"] = out["all"]["finite"] and not bad
    return out


def _num(x):
    x = float(x)
    return x if np.isfinite(x) else None


def gate_table(summary, g=L.GATE):
    """{gate: {ok, value, threshold, detail}} for the screened gates; `ok` is w4_lib.gate's, value and threshold the numbers it compared."""
    res = L.gate(summary, None, g)
    a = summary["all"]
    dm = [(s["kl"]["mean"], d) for d, s in summary["domains"].items() if s.get("kl")]
    dp = [(s["dppl"]["mean"], d) for d, s in summary["domains"].items() if s.get("dppl")]
    val = {"kl_mean": a["kl"]["hi"], "kl_domain": max(dm)[0] if dm else 0.0, "kl_p99": a["kl_tok"]["p99"], "ppl_all": a["dppl"]["hi"],
           "ppl_domain": max(dp)[0] if dp else 0.0, "finite": bool(a.get("finite", True))}
    out = {}
    for name, key in GATES.items():
        ok, detail = res[name]
        v = val[name]
        out[name] = {"ok": bool(ok), "value": v if isinstance(v, bool) else _num(v), "threshold": None if key is None else g[key], "detail": detail}
        if out[name]["value"] is None:
            out[name]["ok"] = False
    return out


def e2c_pair(directory):
    """E2c for one edit from records_base.jsonl and records_edited.jsonl in `directory`: the exit shares and the paired gain (14.3)."""
    recs = {}
    for name in ("base", "edited"):
        p = os.path.join(directory, f"records_{name}.jsonl")
        if not os.path.isfile(p):
            raise ScreenError("w4s-e2c", f"{p}: missing")
        try:
            recs[name] = E.load_records(p)
        except E.E2cError as e:
            raise ScreenError("w4s-e2c", str(e))
    g, ids = E.gains(recs["edited"], recs["base"])
    if len(ids) < 2:
        raise ScreenError("w4s-e2c", "fewer than two loops in common between base and edited")
    lo, hi = E.bootstrap(g["E"])
    pb, pe = E.per_loop(recs["base"]), E.per_loop(recs["edited"])
    return {"loops": len(ids), "E_base": float(np.mean([pb[i][0] for i in ids])), "E_edited": float(np.mean([pe[i][0] for i in ids])),
            "G_E": [float(np.mean(g["E"])), lo, hi], "gate": None,
            "note": "reported, not gated: 14.4's calibration holds for the published Qwen3.5-0.8B loops only (14.7)"}


# ------------------------------------------------------------------ the evidence

def _clean(x):
    """Non-finite floats become null (JSON has none); a value that is null fails its gate."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def build_evidence(summary, base, edited, patch, corpus, label, vocab, e2c=None, now=None):
    gates = gate_table(summary)
    now = now or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    keep = lambda s: {"positions": s["positions"], "sequences": s["sequences"], "kl": s["kl"], "kl_tok": s["kl_tok"], "dppl": s["dppl"]}
    return {"schema": SCHEMA, "verdict": "pass" if all(g["ok"] for g in gates.values()) else "fail", "date": now, "tool": TOOL,
            "model": {"label": label, "vocab": vocab}, "base": base, "edited": edited, "patch": patch, "corpus": corpus, "gates": gates,
            "not_evaluated": NOT_EVALUATED, "summary": _clean({"all": keep(summary["all"]), "domains": {k: keep(v) for k, v in summary["domains"].items()}}),
            "e2c": e2c}


def write_json(path, obj):
    tmp = path + ".partial"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    except (OSError, ValueError) as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise ScreenError("w4s-out", f"{path}: {e}")


# ------------------------------------------------------------------ the verifier

def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)


def load_evidence(path):
    if not os.path.isfile(path):
        raise ScreenError("w4s-evidence-missing", f"{path}: no evidence for this patch")
    try:
        with open(path, "rb") as f:
            raw = f.read(MAX_EVIDENCE + 1)
        if len(raw) > MAX_EVIDENCE:
            raise ValueError("over 1 MiB")
        ev = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as e:
        raise ScreenError("w4s-evidence-format", f"{path}: {e}")
    return ev


def _format(ev):
    def bad(msg):
        raise ScreenError("w4s-evidence-format", msg)
    if not isinstance(ev, dict):
        bad("not a JSON object")
    if ev.get("schema") != SCHEMA:
        bad(f"schema is not {SCHEMA}")
    if ev.get("verdict") not in ("pass", "fail"):
        bad("verdict is neither pass nor fail")
    for k in ("base", "edited"):
        v = ev.get(k)
        if not isinstance(v, dict) or not isinstance(v.get("sha256"), str) or not HEX64.match(v["sha256"]):
            bad(f"{k}.sha256 is not 64 hex digits")
    p = ev.get("patch")
    if p is not None and (not isinstance(p, dict) or not isinstance(p.get("sha256"), str) or not HEX64.match(p["sha256"]) or
                          not all(isinstance(p.get(k), int) and not isinstance(p.get(k), bool) and p[k] >= 0 for k in ("bytes", "crc32"))):
        bad("patch is not {bytes, crc32, sha256}")
    gates = ev.get("gates")
    if not isinstance(gates, dict):
        bad("gates is not an object")
    for n, g in gates.items():
        if not isinstance(g, dict) or not isinstance(g.get("ok"), bool) or not (g.get("value") is None or isinstance(g.get("value"), (bool, int, float))):
            bad(f"gate {n!r} is malformed")
        if not (g.get("threshold") is None or _is_num(g["threshold"])):
            bad(f"gate {n!r} has a threshold that is not a number")


def verify(patch_path, evidence_path=None, base_dir=None):
    """Returns the evidence when it is valid and every evaluated gate passed; raises ScreenError otherwise."""
    blob = read_patch(patch_path)
    ev = load_evidence(evidence_path or patch_path + ".evidence.json")
    _format(ev)
    p = ev["patch"]
    if p is None:
        raise ScreenError("w4s-evidence-patch", "the evidence is for a pair of checkpoints, not for a patch (patch is null)")
    ident = patch_identity(patch_path, blob)
    if any(p[k] != ident[k] for k in ("bytes", "sha256", "crc32")):
        raise ScreenError("w4s-evidence-patch", "the evidence was made for another file: " + ", ".join(k for k in ("bytes", "sha256", "crc32") if p[k] != ident[k]) + " differ")
    for name, key in GATES.items():
        g = ev["gates"].get(name)
        if g is None:
            raise ScreenError("w4s-evidence-thresholds", f"registered gate `{name}` is absent from the evidence")
        if key is not None and not (_is_num(g.get("threshold")) and g["threshold"] <= L.GATE[key]):
            raise ScreenError("w4s-evidence-thresholds", f"gate `{name}` records threshold {g.get('threshold')!r}, looser than the registered {L.GATE[key]}")
    for name, g in ev["gates"].items():
        key = GATES.get(name)
        v = g.get("value")
        want = (v is True) if key is None else (_is_num(v) and _is_num(g.get("threshold")) and v <= g["threshold"])
        if g["ok"] != bool(want):
            raise ScreenError("w4s-evidence-verdict", f"gate `{name}`: ok={g['ok']} disagrees with value {v!r} and threshold {g.get('threshold')!r}")
    ok = all(g["ok"] for g in ev["gates"].values())
    if (ev["verdict"] == "pass") != ok:
        raise ScreenError("w4s-evidence-verdict", f"verdict {ev['verdict']!r} disagrees with the gates")
    if base_dir is not None and tree_hash(base_dir)["sha256"] != ev["base"]["sha256"]:
        raise ScreenError("w4s-evidence-base", f"{base_dir} is not the base the evidence was made against")
    failed = sorted(n for n, g in ev["gates"].items() if not g["ok"])
    if failed:
        raise ScreenError("w4s-gate-failed", "the evidence records failed gates: " + ", ".join(failed))
    return ev


# ------------------------------------------------------------------ the model passes (torch; not run by the tests)

def _load_model(model_dir, device):
    try:
        import torch
        import transformers as T
    except ImportError as e:
        raise ScreenError("w4s-model", f"the model pass needs torch and transformers ({e})")
    try:
        try:
            m = T.AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, trust_remote_code=False)
        except (ValueError, KeyError, AttributeError):
            m = T.AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.float32, trust_remote_code=False)
    except (OSError, ValueError, ImportError) as e:
        raise ScreenError("w4s-model", f"{model_dir}: {e}")
    return m.to(device).eval()


def collect(base_dir, edited_dir, windows, device="cpu", max_len=512):
    """Yield (domain, logits_base, logits_edited, targets) for each corpus window, one full forward each, truncated to `max_len` tokens."""
    import torch
    mb, me = _load_model(base_dir, device), _load_model(edited_dir, device)
    with torch.no_grad():
        for w in windows:
            ids = w["ids"][:max_len]
            x = torch.tensor([ids], device=device)
            lb = mb(input_ids=x, use_cache=False).logits[0, :-1].float().cpu().numpy()
            le = me(input_ids=x, use_cache=False).logits[0, :-1].float().cpu().numpy()
            yield w["domain"], lb, le, np.array(ids[1:])


def corpus_windows(work, limit=None):
    cj = os.path.join(work, "corpus.json")
    try:
        with open(cj, encoding="utf-8") as f:
            ws = json.load(f)["eval"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ScreenError("w4s-corpus", f"{cj}: no readable `eval` split ({e})")
    good = [w for w in ws if isinstance(w, dict) and isinstance(w.get("domain"), str) and isinstance(w.get("ids"), list) and len(w["ids"]) >= 2
            and all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in w["ids"])]
    if not good or len(good) != len(ws):
        raise ScreenError("w4s-corpus", f"{cj}: the eval split is empty or has a window that is not {{domain, ids}} with at least 2 ids")
    return good[:limit] if limit else good


def corpus_identity(work, windows, max_len):
    man = os.path.join(work, "manifest.json")
    return {"corpus_sha256": sha256_file(os.path.join(work, "corpus.json")), "manifest_sha256": sha256_file(man) if os.path.isfile(man) else None,
            "windows": len(windows), "positions": sum(min(len(w["ids"]), max_len) - 1 for w in windows), "max_len": max_len,
            "domains": sorted({w["domain"] for w in windows})}


def run(a):
    if (a.edited is None) == (a.patch is None) and not (a.edited and a.patch):
        raise ScreenError("w4s-input", "give --edited DIR, --patch FILE, or both")
    if not os.path.isdir(os.path.dirname(os.path.abspath(a.out))):
        raise ScreenError("w4s-out", f"{a.out}: the directory does not exist")
    base = tree_hash(a.base)
    blob = read_patch(a.patch) if a.patch else None
    tmp = None
    try:
        edited_dir = a.edited
        if a.edited and blob is not None:
            patch_matches(a.base, blob, a.edited)
        elif blob is not None:
            tmp = tempfile.mkdtemp(prefix="w4_screen_")
            edited_dir = os.path.join(tmp, "edited")
            materialize(a.base, blob, edited_dir)
        edited = tree_hash(edited_dir)
        windows = corpus_windows(a.corpus, a.windows)
        summary = screen_windows(collect(a.base, edited_dir, windows, a.device, a.max_len))
        vocab = _vocab(a.base)
        e2c = None
        if a.loops:
            outdir = a.e2c_dir or a.out + ".e2c"
            try:
                E.run("base", a.base, a.loops, outdir, a.device, E.SEEDS)
                E.run("edited", edited_dir, a.loops, outdir, a.device, E.SEEDS)
            except E.E2cError as e:
                raise ScreenError("w4s-e2c", str(e))
            e2c = e2c_pair(outdir)
        elif a.e2c_records:
            e2c = e2c_pair(a.e2c_records)
        ev = build_evidence(summary, base, edited, patch_identity(a.patch, blob) if blob is not None else None,
                            corpus_identity(a.corpus, windows, a.max_len), a.label, vocab, e2c)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    write_json(a.out, ev)
    return ev


def _vocab(model_dir):
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            c = json.load(f)
        v = (c.get("text_config") or c).get("vocab_size")
        return v if isinstance(v, int) and not isinstance(v, bool) else None
    except (OSError, ValueError, AttributeError):
        return None


def render(ev):
    lines = [f"verdict: {ev['verdict'].upper()}  ({ev['date']}, {ev['tool']['name']} {ev['tool']['version']})"]
    for n, g in ev["gates"].items():
        lines.append(f"  {'ok  ' if g['ok'] else 'FAIL'} {n}: {g['value']} vs {g['threshold']}")
    lines.append("  not evaluated: " + ", ".join(ev.get("not_evaluated", [])))
    if ev.get("e2c"):
        lines.append(f"  e2c (reported only): E {ev['e2c']['E_base']:.3f} -> {ev['e2c']['E_edited']:.3f}, G_E {ev['e2c']['G_E'][0]:+.3f}")
    lines.append("  limits: the evidence is not signed; it shows the gates were run on this file, not that the edit is good (15.1)")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base", required=True)
    r.add_argument("--edited")
    r.add_argument("--patch")
    r.add_argument("--corpus", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--label", default="unlabelled")
    r.add_argument("--device", default="cpu")
    r.add_argument("--max-len", type=int, default=512)
    r.add_argument("--windows", type=int)
    r.add_argument("--loops")
    r.add_argument("--e2c-dir")
    r.add_argument("--e2c-records")
    v = sub.add_parser("verify")
    v.add_argument("patch")
    v.add_argument("--evidence")
    v.add_argument("--base")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "run":
            if a.max_len < 2 or (a.windows is not None and a.windows < 1):
                raise ScreenError("w4s-input", "--max-len must be at least 2 and --windows at least 1")
            ev = run(a)
            print(render(ev))
            return 0 if ev["verdict"] == "pass" else 1
        ev = verify(a.patch, a.evidence, a.base)
        print("verified: " + a.patch)
        print(render(ev))
        return 0
    except ScreenError as e:
        print(f"error[{e.rule}]: {e}", file=sys.stderr)
        return 1 if e.rule == "w4s-gate-failed" else 2
    except (E.E2cError, K.CkptError) as e:
        print(f"error[{e.rule}]: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
