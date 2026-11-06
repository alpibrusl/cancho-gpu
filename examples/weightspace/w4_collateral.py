#!/usr/bin/env python3
"""docs/weight-space-w4.md: E2-lite (the collateral of the published Qwen3.5-0.8B edit) and E2b (its null-space projection), offline.

    python3 examples/weightspace/w4_collateral.py e1a     [--out results/w4_e1a.json]
    python3 examples/weightspace/w4_collateral.py corpus  --work DIR
    python3 examples/weightspace/w4_collateral.py fit     --work DIR [--device mps]
    python3 examples/weightspace/w4_collateral.py project --work DIR
    python3 examples/weightspace/w4_collateral.py eval    --work DIR [--device mps] [--variants edited,p1e-3,...]
    python3 examples/weightspace/w4_collateral.py repeat  --work DIR [--device mps]
    python3 examples/weightspace/w4_collateral.py report  --work DIR --out examples/weightspace/results/w4_collateral.json
    python3 examples/weightspace/w4_collateral.py all     --work DIR --out ...

The stages in order: `corpus` (local text only, split by a hash of the unit: section 3.1), `fit` (the second moment of the inputs of
layer 15 and 19's mlp.down_proj on the fit split, float64), `project` (eigenproblem, the three cutoffs, survival of the edit, the
200 random-support controls, bf16 survival), `eval` (every variant against the base on the eval split: KL, NLL, the activation of the
edited neurons, the effect amplitude), `repeat` (50 greedy continuations of 256 tokens), `report` (the gates). `e1a` is weights only.

Model: the text tower of ~/models/qwen35-08b-base and ~/models/qwen35-08b-edited in float32 (examples/qwen35/hf08_ref.build_model, the
shipped transformers code), every window scored by ONE full forward (transformers' gated-delta layer ignores a cached state unless the new
chunk is one token, docs/weight-space.md 14.12). The checkpoints are held to the published sha256 by w1.EXPECT before anything runs.
No download, no network; the GPU is not used (MPS is the Mac's own, `--device cpu` is the reference)."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "examples", "qwen35"))
import numpy as np  # noqa: E402
import w4_lib as L  # noqa: E402

HOME = os.path.expanduser("~")
WS = os.path.join(HOME, "Workspace")
AL = os.path.join(WS, "alpibrusl")
BASE_DIR = os.path.join(HOME, "models", "qwen35-08b-base")
EDITED_DIR = os.path.join(HOME, "models", "qwen35-08b-edited")
GSM8K = os.path.join(HOME, "models", "gsm8k", "main", "test-00000-of-00001.parquet")
COMFY = os.path.join(AL, "mymodel", "venv", "lib", "python3.12", "site-packages", "comfyui_embedded_docs", "docs")
COMFY_LANGS = ["en", "zh", "zh-TW", "ja", "ko", "ru", "fr", "es", "pt-BR", "tr", "ar", "fa"]
LAYERS = (15, 19)
D_MLP = 3584
MAX_WINDOW, MIN_WINDOW, MAX_PER_DOC, UNIT_TOKENS = 2048, 32, 2, 16384
EXCL = {".git", "node_modules", "target", "__pycache__", "venv", ".venv", "site-packages", "build", "dist", "results", "golden", "vendor", "third_party", ".claude", ".lex", "lexsys", ".cargo"}
EOS = (248044, 248046)
GEN_LEN, N_GENERAL, N_MATH = 256, 30, 20
REP_VARIANTS = ["edited", "p1e-2", "x1e-1"]   # p1e-3 and p1e-4 are the base itself (the projection removes the whole edit, 11.3): not generated
ALL_CUTS = tuple(L.CUTOFFS) + (2e-2, 5e-2, 1e-1, 2e-1, 5e-1)   # the registered three, then the post-hoc extension of 12.1

# (domain, quota tokens per split, extensions, roots, kind)
DOMAINS = [
    ("en_md", 24000, {".md"}, [os.path.join(ROOT, "docs"), os.path.join(AL, "cancho", "docs"), os.path.join(AL, "lex-lang"), os.path.join(AL, "acli"),
                               os.path.join(AL, "agents"), os.path.join(AL, "colibri", "docs"), os.path.join(AL, "cancho-forge"), os.path.join(AL, "cancho-code"),
                               os.path.join(WS, "ds4")], "en"),
    ("code_py", 14000, {".py"}, [os.path.join(ROOT, d) for d in ("examples", "tests", "tools", "scripts")] +
     [os.path.join(AL, d) for d in ("acli", "agents", "agentcmp", "cancho-code", "colibri")], "code"),
    ("code_rs", 14000, {".rs"}, [os.path.join(AL, d) for d in ("lex-lang", "acli", "cancho")], "code"),
    ("code_cho", 10000, {".cho", ".chx"}, [os.path.join(ROOT, d) for d in ("src", "device", "tests", "examples")] +
     [os.path.join(AL, d) for d in ("cancho-code", "cancho-forge")], "code"),
    ("code_c", 8000, {".c", ".h", ".cu", ".metal", ".mm"}, [os.path.join(ROOT, d) for d in ("shim", "device", "tests")] +
     [os.path.join(AL, "colibri"), os.path.join(WS, "ds4")], "code"),
    ("code_web", 6000, {".js", ".ts", ".sh"}, [os.path.join(AL, d) for d in ("agents", "acli", "lex-lang", "colibri")] + [os.path.join(ROOT, "scripts")], "code"),
    ("data", 6000, {".json", ".yaml", ".yml", ".toml"}, [ROOT] + [os.path.join(AL, d) for d in ("acli", "agents", "colibri", "lex-lang")], "data"),
    ("es", 16000, {".md", ".txt"}, [os.path.join(AL, d) for d in ("content", "creative", "lex-www", "lex-robot-teclado", "lex-robot-live", "proposals")], "es"),
    ("it", 10000, {".md", ".txt"}, [os.path.join(WS, "ds4", "speed-bench", "promessi_sposi.txt"), os.path.join(AL, "colibri", "README.it.md"),
                                    os.path.join(AL, "colibri", "docker", "README.IT.md")], "it"),
]
COMFY_QUOTA, MATH_QUOTA = 5000, 12000
REPO_TEXT = {"en_md", "code_py", "code_rs", "code_cho", "code_c", "code_web", "data"}   # domains whose contexts may be printed
PRIVATE = {"es", "it"}                                                                   # no path, no text in any output


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ the published edit

def published():
    """{layer: {"w": base float64 [1024, 3584], "dw": float64, "cols": [..]}} after holding both checkpoints to w1.EXPECT's sha256."""
    import w1
    import w1_diff as D
    tb, te = D.read_checkpoint(BASE_DIR), D.read_checkpoint(EDITED_DIR)
    out = {}
    for layer in LAYERS:
        name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
        want = w1.EXPECT[name]
        gb, ge = D._sha(tb[name]), D._sha(te[name])
        if gb != want["base_sha256"] or ge != want["edited_sha256"]:
            raise SystemExit(f"{name}: the checkpoints are not the published ones (base {gb}, edited {ge})")
        wb, we = tb[name].as_float64(), te[name].as_float64()
        dw = we - wb
        out[layer] = {"w": wb, "dw": dw, "cols": [int(c) for c in np.nonzero(np.abs(dw).sum(0) > 0)[0]], "sha_base": gb, "sha_edited": ge}
    return out


# ------------------------------------------------------------------ E1a

def cmd_e1a(a):
    pub = published()
    res = {"layers": {}}
    for layer in LAYERS:
        p = pub[layer]
        r = L.e1a(p["w"], p["dw"])
        r["sha_base"], r["sha_edited"] = p["sha_base"], p["sha_edited"]
        proj_path = os.path.join(a.work, "proj.npz")
        if os.path.exists(proj_path):   # exploratory, after the fit stage: which weight-and-activation scores recover the published support (post hoc, not registered: 12)
            rms = np.load(proj_path)[f"rms{layer}"]
            cols = np.array(p["cols"])
            au = np.abs(L.rank1(p["dw"])[1] @ p["w"])
            ex = {}
            for name, sc in (("rms", rms), ("abs_uW_times_rms", au * rms), ("abs_uW_times_rms2", au * rms ** 2)):
                rk = L.rank_of(cols, sc)
                ex[name] = {"in_top64": int((rk <= 64).sum()), "in_top128": int((rk <= 128).sum()), "median_rank": float(np.median(rk))}
            r["exploratory_with_activation_scale"] = ex
        res["layers"][str(layer)] = r
        log(f"layer {layer}: slope {r['slope_c']:.3f} rho {r['pearson_rho']:.3f}  top64/top96 by |u.W_j|: {r['rank_by_abs_uW']['in_top64']}/{r['rank_by_abs_uW']['in_top96']}"
            f"  median rank {r['rank_by_abs_uW']['median']}  spearman(index, |u.W|) {r['spearman_index_vs']['abs_uW']:.3f} (membership {r['spearman_index_vs']['membership']:.3f})  H1 {r['h1']} H2 {r['h2']}")
    if a.out:
        save_json(a.out, res)
    return res


# ------------------------------------------------------------------ the corpus

def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_DIR)


def walk(roots, exts):
    seen = set()
    for r in roots:
        if os.path.isfile(r):
            if os.path.splitext(r)[1].lower() in exts and r not in seen:
                seen.add(r)
                yield r
            continue
        for dp, dn, fn in os.walk(r):
            dn[:] = sorted(d for d in dn if d not in EXCL and "venv" not in d)
            for f in sorted(fn):
                p = os.path.join(dp, f)
                if os.path.splitext(f)[1].lower() in exts and p not in seen:
                    seen.add(p)
                    yield p


def read_text(path, kind):
    try:
        size = os.path.getsize(path)
        if size < 300 or size > (64_000 if kind == "data" else 2_000_000):
            return None
        with open(path, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if "\x00" in text or max((len(l) for l in text.split("\n")), default=0) > 2000:
        return None
    base = os.path.basename(path)
    if kind == "en" and (L.is_spanish(text) or L.has_cjk(text) or base.startswith("weight-space") or "README.zh" in base or ".zh-" in base):
        return None
    if kind == "es" and not L.is_spanish(text):
        return None
    return text


def tidy(path):
    return path.replace(HOME, "~")


def build_domain(tok, name, quota, files, kind, rng, manifest):
    got = {"fit": 0, "eval": 0}
    out = {"fit": [], "eval": []}
    seen_md5 = set()
    order = rng.permutation(len(files))
    n_read = 0
    for fi in order:
        path = files[fi]
        text = read_text(path, kind)
        if text is None:
            continue
        md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
        if md5 in seen_md5:
            continue
        seen_md5.add(md5)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        n_read += 1
        rel = tidy(path)
        units = [(rel, ids)] if len(ids) <= UNIT_TOKENS else [(f"{rel}#{k}", ids[k * UNIT_TOKENS:(k + 1) * UNIT_TOKENS]) for k in range((len(ids) + UNIT_TOKENS - 1) // UNIT_TOKENS)]
        for uid, uids in units:
            sp = L.split_of(uid)
            if got[sp] >= quota:
                continue
            wins = L.cut_windows(uids, MAX_WINDOW, MIN_WINDOW)
            if not wins:
                continue
            pick = sorted(rng.permutation(len(wins))[:MAX_PER_DOC])
            used = 0
            for k in pick:
                w = wins[k]
                room = quota - got[sp]
                if room < MIN_WINDOW:
                    break
                w = w[:room]
                out[sp].append({"domain": name, "unit": uid, "ids": w})
                got[sp] += len(w)
                used += len(w)
            if used:
                manifest.append({"domain": name, "split": sp, "unit": (hashlib.sha256(uid.encode()).hexdigest()[:16] if name in PRIVATE else uid),
                                 "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "tokens": used})
        if min(got.values()) >= quota:
            break
    return out, got, n_read


def math_windows(tok, lo=200, n_tok=MATH_QUOTA):
    import pyarrow.parquet as pq
    import w1_behave as B
    rows = pq.read_table(GSM8K).to_pylist()
    out, got = [], 0
    for i in range(lo, len(rows)):
        q, ans = rows[i]["question"], rows[i]["answer"]
        sol, final = ans.split("####")
        import re
        sol = re.sub(r"<<[^>]*>>", "", sol).strip()
        prompt = tok.apply_chat_template([{"role": "user", "content": B.build_question(q)}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        resp = sol + "\n</think>\n\n\\boxed{" + final.strip() + "}<|im_end|>"
        pi = tok(prompt, add_special_tokens=False)["input_ids"]
        ri = tok(resp, add_special_tokens=False)["input_ids"]
        if got + len(pi) + len(ri) > n_tok:
            break
        out.append({"domain": "math_chat", "unit": f"gsm8k/{i}", "ids": pi + ri, "prompt_len": len(pi)})
        got += len(pi) + len(ri)
    return out, got


def cmd_corpus(a):
    tok = tokenizer()
    os.makedirs(a.work, exist_ok=True)
    corpus = {"fit": [], "eval": []}
    manifest, counts = [], {}
    for name, quota, exts, roots, kind in DOMAINS:
        files = list(walk([r for r in roots if os.path.exists(r)], exts))
        rng = np.random.default_rng([L.SEED, zlib.crc32(name.encode())])
        out, got, n_read = build_domain(tok, name, quota, files, kind, rng, manifest)
        for sp in out:
            corpus[sp] += out[sp]
        counts[name] = {"candidates": len(files), "read": n_read, "fit": got["fit"], "eval": got["eval"], "quota": quota}
        log(f"{name}: {len(files)} candidate files, {got['fit']} fit and {got['eval']} eval tokens (quota {quota})")
    # comfyui: one unit is a node, across all languages
    if os.path.isdir(COMFY):
        nodes = sorted(d for d in os.listdir(COMFY) if os.path.isfile(os.path.join(COMFY, d, "en.md")))
        order = np.random.default_rng([L.SEED, zlib.crc32(b"comfy")]).permutation(len(nodes))
        for lang in COMFY_LANGS:
            got = {"fit": 0, "eval": 0}
            for k in order:
                node = nodes[k]
                path = os.path.join(COMFY, node, lang + ".md")
                if not os.path.isfile(path):
                    continue
                uid = "comfy/" + node
                sp = L.split_of(uid)
                if got[sp] >= COMFY_QUOTA:
                    continue
                with open(path, encoding="utf-8") as f:
                    text = f.read()
                ids = tok(text, add_special_tokens=False)["input_ids"]
                w = L.cut_windows(ids, MAX_WINDOW, MIN_WINDOW)[:1]
                if not w:
                    continue
                w = w[0][:COMFY_QUOTA - got[sp]]
                if len(w) < MIN_WINDOW:
                    continue
                corpus[sp].append({"domain": "comfy_" + lang, "unit": f"{uid}/{lang}", "ids": w})
                got[sp] += len(w)
                manifest.append({"domain": "comfy_" + lang, "split": sp, "unit": f"{uid}/{lang}", "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "tokens": len(w)})
                if min(got.values()) >= COMFY_QUOTA:
                    break
            counts["comfy_" + lang] = {"candidates": len(nodes), "fit": got["fit"], "eval": got["eval"], "quota": COMFY_QUOTA}
            log(f"comfy_{lang}: {got['fit']} fit and {got['eval']} eval tokens")
    else:
        log("no comfyui_embedded_docs on this machine: the multilingual domains are empty")
    mw, mt = math_windows(tok)
    corpus["eval"] += mw
    counts["math_chat"] = {"fit": 0, "eval": mt, "quota": MATH_QUOTA, "problems": len(mw)}
    log(f"math_chat: {len(mw)} problems, {mt} tokens")
    tot = {sp: sum(len(w["ids"]) for w in corpus[sp]) for sp in corpus}
    log("totals:", tot, "windows:", {sp: len(corpus[sp]) for sp in corpus})
    save_json(os.path.join(a.work, "corpus.json"), corpus)
    save_json(os.path.join(a.work, "manifest.json"), {"salt": L.SALT, "seed": L.SEED, "counts": counts, "totals": tot, "files": manifest})
    return counts


# ------------------------------------------------------------------ the model

def load_model(device):
    import torch
    import hf08_ref as H
    m = H.build_model(BASE_DIR, "f32").to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def hidden(model, ids_t, hooks=None):
    import torch
    with torch.no_grad():
        return model.model(ids_t, use_cache=False).last_hidden_state[0]


def down(model, layer):
    return model.model.layers[layer].mlp.down_proj


class Tap:
    """Pre-hooks on down_proj of the given layers that keep the last input (the 3,584-d activation) of each."""

    def __init__(self, model, layers):
        self.a, self.h = {}, []
        for l in layers:
            self.h.append(down(model, l).register_forward_pre_hook(self._mk(l)))

    def _mk(self, l):
        def f(_m, args):
            self.a[l] = args[0][0].detach()
        return f

    def close(self):
        for h in self.h:
            h.remove()


def windows(work, split):
    return load_json(os.path.join(work, "corpus.json"))[split]


def cmd_fit(a):
    import torch
    ws = windows(a.work, "fit")
    model = load_model(a.device)
    tap = Tap(model, LAYERS)
    mom = {l: L.Moment(D_MLP) for l in LAYERS}
    t0 = time.time()
    for i, w in enumerate(ws):
        ids = torch.tensor([w["ids"]], device=a.device)
        hidden(model, ids)
        for l in LAYERS:
            x = tap.a[l].float()
            mom[l].add_gram((x.T @ x).cpu().double().numpy(), x.shape[0])
        if i % 40 == 0:
            log(f"fit window {i}/{len(ws)} ({time.time() - t0:.0f}s)")
    tap.close()
    np.savez(os.path.join(a.work, "fit_moments.npz"), n=mom[LAYERS[0]].n, **{f"c{l}": mom[l].c for l in LAYERS})
    log("fit positions:", mom[LAYERS[0]].n)


# ------------------------------------------------------------------ the projection (E2b)

def cmd_project(a):
    z = np.load(os.path.join(a.work, "fit_moments.npz"))
    n = int(z["n"])
    pub = published()
    rng = np.random.default_rng([L.SEED, 7])
    res = {"n_fit_positions": n, "layers": {}}
    arrays = {}
    for l in LAYERS:
        c = z[f"c{l}"]
        t0 = time.time()
        ev, q = L.eig_of(c)
        sigma, u, v = L.rank1(pub[l]["dw"])
        dw = pub[l]["dw"]
        diag = np.diag(c) / n
        rms = np.sqrt(diag)
        pct = rms.argsort().argsort() / (D_MLP - 1)
        m_rms = float(np.median(rms))
        cols = np.array(pub[l]["cols"])
        entry = {"eig_max": float(ev[0]), "eig_min": float(ev[-1]), "eig_noise_floor_note": "float32 accumulation: relative 1e-7",
                 "singular_ratio_min": float(np.sqrt(max(ev[-1], 0) / ev[0])) if ev[0] > 0 else None,
                 "m_rms": m_rms, "edited_rms_percentile_median": float(np.median(pct[cols])), "edited_rms_over_median": float(np.median(rms[cols]) / m_rms),
                 "cutoffs": {}}
        arrays[f"v{l}"] = v
        arrays[f"u{l}"] = u
        arrays[f"sigma{l}"] = np.float64(sigma)
        arrays[f"rms{l}"] = rms
        arrays[f"m_rms{l}"] = np.float64(m_rms)
        # 200 random supports: published values in random order
        ctrl = {}
        for cut in ALL_CUTS:
            dwc, r = L.null_project(dw, ev, q, n, cut)
            sf = float(np.linalg.norm(dwc) / np.linalg.norm(dw))
            vc = L.null_project(v[None, :], ev, q, n, cut)[0][0]
            arrays[f"vc{l}_{cut:g}"] = vc
            arrays[f"dwc{l}_{cut:g}"] = dwc.astype(np.float32)
            nz = np.sort(np.abs(v))[::-1][:64]
            sfs = []
            for _ in range(200):
                vr = np.zeros(D_MLP)
                vr[rng.choice(D_MLP, 64, replace=False)] = rng.permutation(nz) * rng.choice([-1, 1], 64)
                vr /= np.linalg.norm(vr)
                sfs.append(float(np.linalg.norm(L.null_project(vr[None, :], ev, q, n, cut)[0])))
            ctrl[cut] = sfs
            wb32 = pub[l]["w"].astype(np.float32)
            rd = L.bf16_round(wb32 + dwc.astype(np.float32)).astype(np.float64) - pub[l]["w"]   # what a bf16 checkpoint of the variant holds
            surv = float(np.linalg.norm(rd) / max(np.linalg.norm(dwc), 1e-30))
            cosv = float((rd * dwc).sum() / max(np.linalg.norm(rd) * np.linalg.norm(dwc), 1e-30))
            relerr = float(np.linalg.norm(rd - dwc) / max(np.linalg.norm(dwc), 1e-30))
            entry["cutoffs"][f"{cut:g}"] = {
                "retained": r, "S_F": sf,
                "S_F_random_supports": {"mean": float(np.mean(sfs)), "p5": float(np.percentile(sfs, 5)), "p95": float(np.percentile(sfs, 95)),
                                        "published_percentile": float((np.array(sfs) < sf).mean())},
                "bf16_survival": surv, "bf16_cosine": cosv, "bf16_rel_error": relerr}
        wb32 = pub[l]["w"].astype(np.float32)
        entry["bf16_survival_published"] = float(np.linalg.norm(L.bf16_round(wb32 + pub[l]["dw"].astype(np.float32)).astype(np.float64) - pub[l]["w"]) / np.linalg.norm(pub[l]["dw"]))
        entry["eig_seconds"] = time.time() - t0
        res["layers"][str(l)] = entry
        log(f"layer {l}: " + ", ".join(f"c={k}: r={v['retained']} S_F={v['S_F']:.4f}" for k, v in entry["cutoffs"].items()))
    np.savez(os.path.join(a.work, "proj.npz"), **arrays)
    save_json(os.path.join(a.work, "proj.json"), res)
    return res


# ------------------------------------------------------------------ variants

CUT_OF = {"p1e-2": 1e-2, "p1e-3": 1e-3, "p1e-4": 1e-4, "x2e-2": 2e-2, "x5e-2": 5e-2, "x1e-1": 1e-1, "x2e-1": 2e-1, "x5e-1": 5e-1}   # x*: not registered (12.1)
VARIANTS = ["edited", "e15", "e19"] + list(CUT_OF) + ["rand0", "rand1", "q8", "q4"]


def variant_weights(name, pub, proj, model):
    """Yield (parameter name, float32 array) for every tensor the variant replaces."""
    import torch.nn as nn

    def pn(l):
        return f"model.layers.{l}.mlp.down_proj.weight"

    def f32(l, dw):
        return (pub[l]["w"] + dw).astype(np.float32)
    if name == "edited":
        for l in LAYERS:
            yield pn(l), f32(l, pub[l]["dw"])
    elif name in ("e15", "e19"):
        l = int(name[1:])
        yield pn(l), f32(l, pub[l]["dw"])
    elif name in CUT_OF:
        for l in LAYERS:
            yield pn(l), f32(l, proj[f"dwc{l}_{CUT_OF[name]:g}"].astype(np.float64))
    elif name.startswith("rand"):
        rng = np.random.default_rng([L.SEED, 1000 + int(name[4:])])
        for l in LAYERS:
            dwr, _ = L.random_edit(pub[l]["dw"], 64, rng)
            yield pn(l), f32(l, dwr)
    elif name in ("q8", "q4"):
        bits, group = (8, None) if name == "q8" else (4, 32)
        for mn, m in model.model.layers.named_modules():
            if isinstance(m, nn.Linear):
                w = m.weight.detach().cpu().numpy()
                yield f"model.layers.{mn}.weight", L.fake_quant(w, bits, group)
    else:
        raise ValueError(name)


class Swapper:
    def __init__(self, base, work, device):
        self.bp, self.wp, self.touched, self.device = dict(base.named_parameters()), dict(work.named_parameters()), [], device

    def set(self, items):
        import torch
        for k in self.touched:
            self.wp[k].data.copy_(self.bp[k].data)
        self.touched = []
        for k, arr in items:
            self.wp[k].data.copy_(torch.from_numpy(arr).to(self.device))
            self.touched.append(k)


# ------------------------------------------------------------------ eval (E2-lite, E2b)

def seg_domains(w, T):
    """Per scored position t = 0..T-2 (predicting ids[t+1]): the domain name and the part (0 or 1)."""
    if w["domain"] == "math_chat":
        pl = w["prompt_len"]
        part = (np.arange(T - 1) + 1 >= pl).astype(int)
        return ["math_chat_prompt" if p == 0 else "math_chat_resp" for p in part], part
    return [w["domain"]] * (T - 1), np.zeros(T - 1, dtype=int)


def summarize(arr, doms):
    """The summary of one variant: overall and per domain (kl, kl_tok, kl_seq, nll, dppl, top1 agreement, finite)."""
    kl, nb, nv, ag, seg, dom = arr["kl"], arr["nb"], arr["nv"], arr["agree"], arr["seg"], arr["dom"]
    d = nv - nb

    def one(mask):
        if not mask.any():
            return None
        sg = seg[mask]
        u, inv = np.unique(sg, return_inverse=True)
        cnt = np.bincount(inv).astype(np.float64)
        skl = np.bincount(inv, weights=kl[mask])
        sd = np.bincount(inv, weights=d[mask])
        sb = np.bincount(inv, weights=nb[mask])
        dd = L.ratio_ci(sd, cnt)
        return {
            "positions": int(mask.sum()), "sequences": int(u.size),
            "kl": L.ratio_ci(skl, cnt), "kl_tok": L.dist(kl[mask]), "kl_seq": L.dist(skl / cnt),
            "ppl_base": float(np.exp(sb.sum() / cnt.sum())), "ppl_var": float(np.exp((sb.sum() + sd.sum()) / cnt.sum())),
            "dppl": {"mean": float(np.expm1(dd["mean"])), "lo": float(np.expm1(dd["lo"])), "hi": float(np.expm1(dd["hi"]))},
            "top1_agree": float(ag[mask].mean()),
            "finite": bool(np.isfinite(kl[mask]).all() and np.isfinite(nv[mask]).all())}
    out = {"all": one(np.ones(kl.shape, dtype=bool)), "domains": {}}
    for k, name in enumerate(doms):
        s = one(dom == k)
        if s:
            out["domains"][name] = s
    return out


def cmd_eval(a):
    import torch
    ws = windows(a.work, "eval")
    pub = published()
    proj = np.load(os.path.join(a.work, "proj.npz"))
    pj = load_json(os.path.join(a.work, "proj.json"))
    base, work = load_model(a.device), load_model(a.device)
    sw = Swapper(base, work, a.device)
    # domains
    dom_names = []
    for w in ws:
        T = len(w["ids"])
        for d in seg_domains(w, T)[0][:1] + (["math_chat_resp"] if w["domain"] == "math_chat" else []):
            if d not in dom_names:
                dom_names.append(d)
    dom_idx = {d: i for i, d in enumerate(dom_names)}
    # --- base pass: cache hidden, accumulate the activation statistics of the edited neurons
    tap = Tap(base, LAYERS)
    hb = []
    nd = len(dom_names)
    stat = {l: {"n": np.zeros(nd), "sq": np.zeros((nd, D_MLP)), "fire": np.zeros((nd, D_MLP)), "z2": np.zeros((nd, 1 + len(ALL_CUTS)))} for l in LAYERS}
    vmat = {}
    for l in LAYERS:
        cols = [proj[f"v{l}"]] + [proj[f"vc{l}_{c:g}"] for c in ALL_CUTS]
        vmat[l] = torch.from_numpy(np.stack(cols, 1)).float().to(a.device)
    thr = {l: 0.1 * float(proj[f"m_rms{l}"]) for l in LAYERS}
    pos_arrays = {"seg": [], "dom": [], "tgt": [], "pos": [], "win": []}
    seg_counter = 0
    t0 = time.time()
    for i, w in enumerate(ws):
        ids = torch.tensor([w["ids"]], device=a.device)
        T = ids.shape[1]
        h = hidden(base, ids)
        hb.append(h)
        names, part = seg_domains(w, T)
        di = np.array([dom_idx[n] for n in names])
        pos_arrays["seg"].append(seg_counter + part)
        seg_counter += 2 if w["domain"] == "math_chat" else 1
        pos_arrays["dom"].append(di)
        pos_arrays["tgt"].append(np.array(w["ids"][1:]))
        pos_arrays["pos"].append(np.arange(1, T))
        pos_arrays["win"].append(np.full(T - 1, i))
        # activation statistics over all T positions, attributed to the window's domain(s): position t is in the part of the token it predicts t+1;
        # for simplicity a position belongs to the domain of its own token (the math window's prompt/response boundary is the same)
        if w["domain"] == "math_chat":
            pdom = np.array([dom_idx["math_chat_prompt"] if t < w["prompt_len"] else dom_idx["math_chat_resp"] for t in range(T)])
        else:
            pdom = np.full(T, dom_idx[w["domain"]])
        for l in LAYERS:
            x = tap.a[l].float()
            z = (x @ vmat[l]) ** 2
            sq = (x ** 2).cpu().double().numpy()
            fire = (x.abs() >= thr[l]).cpu().double().numpy()
            zz = z.cpu().double().numpy()
            for d in np.unique(pdom):
                m = pdom == d
                stat[l]["n"][d] += m.sum()
                stat[l]["sq"][d] += sq[m].sum(0)
                stat[l]["fire"][d] += fire[m].sum(0)
                stat[l]["z2"][d] += zz[m].sum(0)
        if i % 40 == 0:
            log(f"base window {i}/{len(ws)} ({time.time() - t0:.0f}s)")
    tap.close()
    arrays = {k: np.concatenate(v) for k, v in pos_arrays.items()}
    np.savez(os.path.join(a.work, "eval_positions.npz"), **arrays)
    # the neuron/effect statistics
    nstat = {"domains": dom_names, "layers": {}}
    for l in LAYERS:
        cols = np.array(pub[l]["cols"])
        s = stat[l]
        n_all = s["n"].sum()
        gen = [dom_idx[d] for d in dom_names if d not in ("math_chat_prompt", "math_chat_resp")]
        fire_all = s["fire"][gen].sum(0) / s["n"][gen].sum()
        rms_all = np.sqrt(s["sq"][gen].sum(0) / s["n"][gen].sum())
        rms_pct = rms_all.argsort().argsort() / (D_MLP - 1)
        notin = np.ones(D_MLP, bool)
        notin[cols] = False
        e = {"n_positions": int(n_all), "general_positions": int(s["n"][gen].sum()),
             "edited_fire_rate": {"median": float(np.median(fire_all[cols])), "p10": float(np.percentile(fire_all[cols], 10)), "p90": float(np.percentile(fire_all[cols], 90)),
                                  "others_median": float(np.median(fire_all[notin]))},
             "edited_rms_over_fit_median": float(np.median(rms_all[cols]) / float(proj[f"m_rms{l}"])),
             "edited_rms_percentile_median_eval": float(np.median(rms_pct[cols])),
             "C_fire": bool(np.median(fire_all[cols]) >= 0.25),
             "S_E": {}}
        for k, c in enumerate(ALL_CUTS):
            row = {}
            for d, name in enumerate(dom_names):
                if s["z2"][d, 0] > 0:
                    row[name] = float(np.sqrt(s["z2"][d, k + 1] / s["z2"][d, 0]))
            pooled_gen = float(np.sqrt(s["z2"][gen, k + 1].sum() / s["z2"][gen, 0].sum()))
            resp = row.get("math_chat_resp")
            row["_general_pooled"] = pooled_gen
            row["_selectivity_resp_over_general"] = (resp / pooled_gen) if resp is not None and pooled_gen > 0 else None
            e["S_E"][f"{c:g}"] = row
        e["fire_rate_by_domain_edited_median"] = {name: float(np.median(s["fire"][d][cols] / max(s["n"][d], 1))) for d, name in enumerate(dom_names) if s["n"][d] > 0}
        nstat["layers"][str(l)] = e
        log(f"layer {l}: edited fire-rate median {e['edited_fire_rate']['median']:.3f} (others {e['edited_fire_rate']['others_median']:.3f}), C_fire {e['C_fire']}, rms/median {e['edited_rms_over_fit_median']:.2f}")
    save_json(os.path.join(a.work, "neurons.json"), nstat)
    # --- variants
    names = a.variants.split(",") if a.variants else VARIANTS
    summ = {}
    sp = os.path.join(a.work, "eval_summary.json")
    if os.path.exists(sp):
        summ = load_json(sp)
    lm_head = base.lm_head
    for vn in names:
        t0 = time.time()
        sw.set(variant_weights(vn, pub, proj, base))
        kl_l, nb_l, nv_l, en_l, ag_l = [], [], [], [], []
        for i, w in enumerate(ws):
            ids = torch.tensor([w["ids"]], device=a.device)
            T = ids.shape[1]
            hv = hidden(work, ids)
            for s0 in range(0, T - 1, 256):
                s1 = min(s0 + 256, T - 1)
                tg = torch.tensor(w["ids"][s0 + 1:s1 + 1], device=a.device)
                kl, nb, nv, en, ag = L.token_stats(lm_head(hb[i][s0:s1]), lm_head(hv[s0:s1]), tg)
                kl_l.append(kl); nb_l.append(nb); nv_l.append(nv); en_l.append(en); ag_l.append(ag)
        arr = dict(arrays, kl=np.concatenate(kl_l), nb=np.concatenate(nb_l), nv=np.concatenate(nv_l), ent=np.concatenate(en_l), agree=np.concatenate(ag_l))
        np.savez(os.path.join(a.work, f"eval_{vn}.npz"), kl=arr["kl"], nv=arr["nv"], nb=arr["nb"], ent=arr["ent"], agree=arr["agree"])
        summ[vn] = summarize(arr, dom_names)
        summ[vn]["seconds"] = time.time() - t0
        s = summ[vn]["all"]
        log(f"{vn}: mean KL {s['kl']['mean']:.5f} [{s['kl']['lo']:.5f}, {s['kl']['hi']:.5f}]  dppl {s['dppl']['mean'] * 100:.3f}%  top1 {s['top1_agree']:.4f}  ({time.time() - t0:.0f}s)")
        save_json(sp, summ)
    return summ


# ------------------------------------------------------------------ repetition

def greedy_ids(model, ids, n, device):
    import torch
    out = []
    with torch.no_grad():
        o = model(torch.tensor([ids], device=device), use_cache=True, logits_to_keep=1)
        for _ in range(n):
            t = int(o.logits[0, -1].argmax())
            if t in EOS:
                return out, True
            out.append(t)
            o = model(torch.tensor([[t]], device=device), past_key_values=o.past_key_values, use_cache=True, logits_to_keep=1)
    return out, False


def repeat_prompts(work, tok):
    import pyarrow.parquet as pq
    import w1_behave as B
    rows = pq.read_table(GSM8K).to_pylist()
    prompts = []
    for i in range(N_MATH):
        p = tok.apply_chat_template([{"role": "user", "content": B.build_question(rows[i]["question"])}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        prompts.append({"group": "math", "name": f"gsm8k/{i}", "ids": tok(p, add_special_tokens=False)["input_ids"]})
    ev = windows(work, "eval")
    rng = np.random.default_rng([L.SEED, 99])
    fam = ["en_md", "code_py", "code_rs", "code_cho", "code_c", "code_web", "data", "es", "it"]
    by = {}
    for w in ev:
        by.setdefault(w["domain"], []).append(w)
    chosen = []
    for d in fam:
        pool = by.get(d, [])
        for k in rng.permutation(len(pool))[:2]:
            chosen.append(pool[k])
    for lang in COMFY_LANGS:
        pool = by.get("comfy_" + lang, [])
        if pool:
            chosen.append(pool[int(rng.integers(len(pool)))])
    for w in chosen[:N_GENERAL]:
        prompts.append({"group": "general", "name": w["domain"] + "/" + (hashlib.sha256(w["unit"].encode()).hexdigest()[:8] if w["domain"] in PRIVATE else w["unit"]), "ids": w["ids"][:48]})
    return prompts


def cmd_repeat(a):
    tok = tokenizer()
    pub = published()
    proj = np.load(os.path.join(a.work, "proj.npz"))
    prompts = repeat_prompts(a.work, tok)
    base, work = load_model(a.device), load_model(a.device)
    sw = Swapper(base, work, a.device)
    res = {"prompts": [{"group": p["group"], "name": p["name"], "n_prompt": len(p["ids"])} for p in prompts], "variants": {}}
    names = ["base"] + [v for v in (a.variants.split(",") if a.variants else REP_VARIANTS) if v != "base"]
    for vn in names:
        t0 = time.time()
        sw.set([] if vn == "base" else variant_weights(vn, pub, proj, base))
        rows = []
        for j, p in enumerate(prompts):
            g, stopped = greedy_ids(work, p["ids"], GEN_LEN, a.device)
            rows.append({"n": len(g), "stopped": stopped, "rep4": L.rep4(g), "loop": L.tail_loop(g), "head": g[:24], "sha": hashlib.sha256(np.array(g, dtype="<i8").tobytes()).hexdigest()[:16]})
            if j % 10 == 0:
                log(f"{vn}: prompt {j}/{len(prompts)} ({time.time() - t0:.0f}s)")
        res["variants"][vn] = rows
    save_json(os.path.join(a.work, "repeat.json"), res)
    return res


def rep_summary(res, vn):
    """Paired comparison with the base: general and math groups."""
    base, var, pr = res["variants"]["base"], res["variants"][vn], res["prompts"]
    out = {}
    for grp in ("general", "math"):
        idx = [i for i, p in enumerate(pr) if p["group"] == grp]
        pair = [(base[i]["rep4"], var[i]["rep4"]) for i in idx if base[i]["rep4"] is not None and var[i]["rep4"] is not None]
        d = [y - x for x, y in pair]
        out[grp] = {"prompts": len(idx), "scored_pairs": len(pair),
                    "rep4_base": float(np.mean([x for x, _ in pair])) if pair else None, "rep4_var": float(np.mean([y for _, y in pair])) if pair else None,
                    "rep4_delta": float(np.mean(d)) if d else 0.0,
                    "loops_base": sum(1 for i in idx if base[i]["loop"]), "loops_var": sum(1 for i in idx if var[i]["loop"]),
                    "new_loops": sum(1 for i in idx if var[i]["loop"] and not base[i]["loop"]), "lost_loops": sum(1 for i in idx if base[i]["loop"] and not var[i]["loop"]),
                    "new_loop_prompts": [pr[i]["name"] for i in idx if var[i]["loop"] and not base[i]["loop"]],
                    "lost_loop_prompts": [pr[i]["name"] for i in idx if base[i]["loop"] and not var[i]["loop"]],
                    "identical_continuations": sum(1 for i in idx if base[i]["sha"] == var[i]["sha"]),
                    "short_continuations_base": sum(1 for i in idx if base[i]["rep4"] is None), "short_continuations_var": sum(1 for i in idx if var[i]["rep4"] is None)}
    return out


# ------------------------------------------------------------------ where the KL concentrates, and the report

def concentration(work, vn, tok, ws, dom_names):
    z = np.load(os.path.join(work, f"eval_{vn}.npz"))
    p = np.load(os.path.join(work, "eval_positions.npz"))
    kl, ent, pos, tgt, win, dom = z["kl"], z["ent"], p["pos"], p["tgt"], p["win"], p["dom"]
    order = np.sort(kl)[::-1]
    tot = kl.sum()
    out = {"share_top1pct": float(order[:max(1, len(kl) // 100)].sum() / tot), "share_top10pct": float(order[:max(1, len(kl) // 10)].sum() / tot)}
    out["by_position"] = {name: {"mean_kl": float(kl[m].mean()), "n": int(m.sum())} for name, m in
                          (("1-15", pos < 16), ("16-127", (pos >= 16) & (pos < 128)), ("128+", pos >= 128)) if m.any()}
    uniq = np.unique(tgt)
    strs = {int(t): tok.decode([int(t)]) for t in uniq}

    def klass(s):
        if s.strip() == "":
            return "whitespace_newline"
        if s.strip().isdigit():
            return "digit"
        if not any(ch.isalnum() for ch in s):
            return "punct_symbol"
        return "alphanumeric"
    cls = np.array([klass(strs[int(t)]) for t in tgt])
    out["by_target_class"] = {c: {"mean_kl": float(kl[cls == c].mean()), "share_of_total": float(kl[cls == c].sum() / tot), "n": int((cls == c).sum())} for c in np.unique(cls)}
    edges = np.quantile(ent, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, ent, side="right") - 1, 0, 9)
    out["by_entropy_decile"] = [{"decile": int(k), "entropy_upper": float(edges[k + 1]), "mean_kl": float(kl[dec == k].mean())} for k in range(10) if (dec == k).any()]
    out["spearman_entropy_kl"] = L.spearman(ent[::7], kl[::7])
    agg = {}
    for t, k in zip(tgt, kl):
        e = agg.setdefault(int(t), [0.0, 0])
        e[0] += k
        e[1] += 1
    top = sorted(agg.items(), key=lambda kv: -kv[1][0])[:30]
    out["top_tokens_by_total_kl"] = [{"token": strs[t], "total_kl": v[0], "count": v[1], "mean_kl": v[0] / v[1], "share": v[0] / tot} for t, v in top]
    ctx = []
    for k in np.argsort(-kl):
        d = dom_names[dom[k]]
        if d not in REPO_TEXT:
            continue
        w = ws[int(win[k])]
        t = int(pos[k])
        ctx.append({"domain": d, "kl": float(kl[k]), "context": tok.decode(w["ids"][max(0, t - 4):t]), "target": strs[int(tgt[k])]})
        if len(ctx) >= 20:
            break
    out["top_contexts_repo_text"] = ctx
    return out


def cmd_report(a):
    tok = tokenizer()
    summ = load_json(os.path.join(a.work, "eval_summary.json"))
    proj = load_json(os.path.join(a.work, "proj.json"))
    nstat = load_json(os.path.join(a.work, "neurons.json"))
    manifest = load_json(os.path.join(a.work, "manifest.json"))
    ws = windows(a.work, "eval")
    rp = os.path.join(a.work, "repeat.json")
    rep = load_json(rp) if os.path.exists(rp) else None
    rows = {}
    for vn, s in summ.items():
        rs = rep_summary(rep, vn)["general"] if rep and vn in rep["variants"] else None
        g = L.gate(s, rs)
        rows[vn] = {"summary": s, "gates": {k: {"ok": bool(ok), "detail": d} for k, (ok, d) in g.items()}, "pass": L.passes(g), "strict": bool(L.strict_ok(s)), "repetition_general": rs}
    out = {"date": time.strftime("%Y-%m-%d"), "machine": platform.platform(), "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
           "versions": {"python": platform.python_version(), "numpy": np.__version__}, "git": sh("git", "rev-parse", "--short", "HEAD"),
           "corpus": {"totals": manifest["totals"], "counts": manifest["counts"]}, "projection": proj, "neurons": nstat, "variants": rows}
    try:
        import torch, transformers
        out["versions"].update(torch=torch.__version__, transformers=transformers.__version__)
    except ImportError:
        pass
    if rep:
        out["repetition"] = {vn: rep_summary(rep, vn) for vn in rep["variants"] if vn != "base"}
        out["repetition_raw"] = {"prompts": rep["prompts"], "variants": {vn: [{k: r[k] for k in ("n", "stopped", "rep4", "loop")} for r in rows_] for vn, rows_ in rep["variants"].items()}}
    dom_names = nstat["domains"]
    out["concentration"] = {vn: concentration(a.work, vn, tok, ws, dom_names) for vn in ("edited", "x5e-2") if vn in summ}
    save_json(a.out, out)
    save_json(os.path.join(os.path.dirname(os.path.abspath(a.out)), "w4_collateral_manifest.json"), manifest)
    log("wrote", a.out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["e1a", "corpus", "fit", "project", "eval", "repeat", "report", "all"])
    ap.add_argument("--work", default=os.path.join(ROOT, "build", "w4"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--variants", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "all":
        for c in ("corpus", "fit", "project", "eval", "repeat", "report"):
            a.cmd = c
            run(a)
    else:
        run(a)


def run(a):
    {"e1a": cmd_e1a, "corpus": cmd_corpus, "fit": cmd_fit, "project": cmd_project, "eval": cmd_eval, "repeat": cmd_repeat, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
