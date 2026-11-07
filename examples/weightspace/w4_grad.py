#!/usr/bin/env python3
"""docs/weight-space-w4-r4.md: can the loop-fixing edit be found by a gradient rule (the proposal's R4)?

    python3 w4_grad.py select  --base-jsonl .../base_8192.jsonl --work build/r4          # the 60 loops, the split
    python3 w4_grad.py capture --work build/r4 [--model ~/models/qwen35-08b-base] [--pairs 15:mlp.down_proj 19:mlp.down_proj] [--device mps]
    python3 w4_grad.py analyse --work build/r4 --out results/w4_r4/analysis.json            # numpy only
    python3 w4_grad.py build   --work build/r4 --out-dir build/ckpt                         # numpy only: r4p, r4g and the controls
    python3 w4_grad.py behave  --name NAME (--model-dir DIR | --edits NPZ) --work build/r4 --out results/w4_r4 [--reuse results/w4_e2c]
    python3 w4_grad.py kl      --variant NAME=pub|dir:PATH|edits:NPZ ... --corpus corpus.json --out results/w4_r4/kl.json
    python3 w4_grad.py score   --dir results/w4_r4 [--json OUT]

`capture` is model-agnostic: a Hugging Face model name or directory and a list of `layer:module` pairs (a module path inside the decoder
layer whose weight is a 2-d matrix); the decoder layers are found by name. Per loop it stores, for each pair and loss, the weight
gradient, the sum and second moment of the output-side gradient and the squared input (docs 3.1). Every refusal prints its rule tag
`w4g-*` (`w4e-*`, `w4c-*` from the modules it uses); the CLI exits 2 on them."""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import w4_ckpt as K  # noqa: E402
import w4_e2c as E  # noqa: E402
import w4_grad_lib as G  # noqa: E402
import w4_lib as L  # noqa: E402
import w4_search_lib as S  # noqa: E402

HOME = os.path.expanduser("~")
BASE_DIR, EDITED_DIR = K.BASE_DIR, K.EDITED_DIR
PAIRS = [(15, "mlp.down_proj"), (19, "mlp.down_proj")]
LAYERS = (15, 19)
LM_FIRST, LM_LAST = 256, 1024    # the specificity control: generated tokens 256 .. 1023 (docs 3.4)
N_LOOPS = 60
N_PERM = 3
CORPUS = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "alpibrusl", "cancho-gpu-w4", "build", "w4", "corpus.json")


# ---------------------------------------------------------------- finding the layers (model-agnostic)

def find_layers(model, prefix=None):
    """The decoder layers of a Hugging Face model: the longest ModuleList called `layers`, `h` or `blocks` outside any module whose
    path mentions a vision or audio tower; `prefix` (a dotted path) overrides the search."""
    import torch.nn as nn
    if prefix:
        mod = model
        for part in prefix.split("."):
            mod = getattr(mod, part, None)
            if mod is None:
                raise G.GradError("w4g-model", f"no module `{prefix}` in the model")
        return mod
    best = None
    for name, mod in model.named_modules():
        last = name.rsplit(".", 1)[-1]
        if isinstance(mod, nn.ModuleList) and last in ("layers", "h", "blocks") and not re.search(r"vision|visual|audio|image", name):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None or len(best) == 0:
        raise G.GradError("w4g-model", "no decoder layers found; pass --layers-prefix")
    return best


def module_of(layers, layer, path):
    if not (isinstance(layer, int) and 0 <= layer < len(layers)):
        raise G.GradError("w4g-pair", f"layer {layer} is not in 0..{len(layers) - 1}")
    mod = layers[layer]
    for part in path.split("."):
        mod = getattr(mod, part, None)
        if mod is None:
            raise G.GradError("w4g-pair", f"layer {layer} has no module `{path}`")
    w = getattr(mod, "weight", None)
    if w is None or w.dim() != 2:
        raise G.GradError("w4g-pair", f"`{path}` of layer {layer} has no 2-d weight")
    return mod


def parse_pairs(items):
    out = []
    for it in items:
        m = re.fullmatch(r"(\d+):([A-Za-z0-9_.]+)", it)
        if not m:
            raise G.GradError("w4g-pair", f"`{it}` is not layer:module")
        out.append((int(m.group(1)), m.group(2)))
    if not out:
        raise G.GradError("w4g-pair", "no (layer, module) pair")
    return out


# ---------------------------------------------------------------- the capture (torch)

class Tap:
    """Hooks on the target modules: the input of each (a) and the gradient at its output (g), both [T, dim] for the batch of one."""

    def __init__(self, mods):
        self.mods, self.a, self.g, self.h = mods, {}, {}, []
        for key, mod in mods.items():
            self.h.append(mod.register_forward_pre_hook(self._pre(key)))
            self.h.append(mod.register_forward_hook(self._fwd(key)))

    def _pre(self, key):
        def f(_m, args):
            self.a[key] = args[0].detach().reshape(-1, args[0].shape[-1])
        return f

    def _fwd(self, key):
        def f(_m, _args, out):
            if out.requires_grad:
                out.register_hook(lambda g: self.g.__setitem__(key, g.detach().reshape(-1, g.shape[-1])))
        return f

    def close(self):
        for h in self.h:
            h.remove()


def losses_of(logits, targets, kinds):
    """{kind: scalar tensor}: `p` = mean log p_t, `o` = mean (log p_t - log(1 - p_t)) with log(1 - p_t) as the logsumexp of the other logits."""
    import torch
    z = logits[:-1]
    t = torch.as_tensor(targets, device=z.device).long()
    zt = z.gather(1, t[:, None])[:, 0]
    out = {}
    if "p" in kinds:
        out["p"] = (zt - torch.logsumexp(z, -1)).mean()
    if "o" in kinds:
        other = torch.logsumexp(z.scatter(1, t[:, None], float("-inf")), -1)
        out["o"] = (zt - other).mean()
    return out


def grad_job(model, layers, pairs, ids, first, kinds=("p", "o")):
    """One teacher-forced forward on `ids` with the loss over the tokens ids[first + 1:] and one backward a loss.
    Returns {kind: {(layer, module): {"g": [out, in] float32, "m", "q", "r", "n", "check": relative gap of the autograd gradient to sum g a^T}}}."""
    import torch
    if len(ids) < first + 2 or first < 0:
        raise G.GradError("w4g-job", f"{len(ids)} tokens cannot be scored from position {first}")
    dev = next(model.parameters()).device
    mods = {p: module_of(layers, p[0], p[1]) for p in pairs}
    for p in model.parameters():
        p.requires_grad_(False)
    for mod in mods.values():
        mod.weight.requires_grad_(True)
        mod.weight.grad = None
    tap = Tap(mods)
    try:
        keep = len(ids) - first
        try:
            lg = model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=keep).logits[0].float()
        except TypeError:
            lg = model(input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0, first:].float()
        ls = losses_of(lg, ids[first + 1:], kinds)
        out = {}
        for i, kind in enumerate(kinds):
            tap.g.clear()
            for mod in mods.values():
                mod.weight.grad = None
            ls[kind].backward(retain_graph=i < len(kinds) - 1)
            res = {}
            for key, mod in mods.items():
                if key not in tap.g or mod.weight.grad is None:
                    raise G.GradError("w4g-job", f"no gradient reached {key}")
                g, a, w = tap.g[key].float(), tap.a[key].float(), mod.weight.grad.float()
                gw = g.T @ a
                check = float((gw - w).norm() / max(float(w.norm()), 1e-30))
                res[key] = {"g": w.cpu().numpy(), "m": g.sum(0).double().cpu().numpy(), "q": (g.T @ g).double().cpu().numpy(),
                            "r": (a ** 2).sum(0).double().cpu().numpy(), "n": int(g.shape[0]), "check": check, "loss": float(ls[kind].detach())}
            out[kind] = res
        return out
    finally:
        tap.close()
        for mod in mods.values():
            mod.weight.requires_grad_(False)
            mod.weight.grad = None


def save_job(path, res):
    arrs = {}
    for kind, byp in res.items():
        for (layer, _mod), d in byp.items():
            for k in ("g", "m", "q", "r"):
                arrs[f"{k}{layer}_{kind}"] = d[k]
            arrs[f"n{layer}_{kind}"] = np.array(d["n"])
            arrs[f"check{layer}_{kind}"] = np.array(d["check"])
            arrs[f"loss{layer}_{kind}"] = np.array(d["loss"])
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


# ---------------------------------------------------------------- the work files

def read_json(path, rule):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise G.GradError(rule, f"{path}: {str(e)[:200]}")


def load_loops(work):
    return read_json(os.path.join(work, "loops.json"), "w4g-loops")


def halves_of(loops):
    tr, te = G.split_loops(len(loops))
    return {"train": [loops[i] for i in tr], "test": [loops[i] for i in te]}


def loop_job(lp):
    ctx = lp["prompt"] + lp["ids"][:lp["cut"]]
    want = lp["ids"][lp["cut"]:lp["cut"] + E.NEXT]
    return ctx + want, len(ctx) - 1


def perm_job(lp, other):
    """Loop lp's context scored on the loop-continuing tokens of loop `other` (docs 3.4)."""
    ctx = lp["prompt"] + lp["ids"][:lp["cut"]]
    return ctx + other["ids"][other["cut"]:other["cut"] + E.NEXT], len(ctx) - 1


def lm_job(lp):
    seq = lp["prompt"] + lp["ids"][:LM_LAST]
    return seq, len(lp["prompt"]) + LM_FIRST - 1


def cmd_select(a):
    os.makedirs(a.work, exist_ok=True)
    loops = E.select(a.base_jsonl, a.model_dir, a.gsm8k, want=a.n)
    if len(loops) < 4:
        raise G.GradError("w4g-loops", f"only {len(loops)} loops qualify")
    with open(os.path.join(a.work, "loops.json"), "w", encoding="utf-8") as f:
        json.dump(loops, f)
    h = halves_of(loops)
    for name, ls in h.items():
        with open(os.path.join(a.work, f"{name}_loops.json"), "w", encoding="utf-8") as f:
            json.dump(ls, f)
    meta = [{k: v for k, v in lp.items() if k not in ("ids", "prompt")} for lp in loops]
    split = {"train": [lp["id"] for lp in h["train"]], "test": [lp["id"] for lp in h["test"]]}
    print(f"{len(loops)} loops; train {len(split['train'])} test {len(split['test'])}; split {json.dumps(split)}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "loops_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"loops": meta, "split": split}, f, indent=1)


def cmd_capture(a):
    pairs = parse_pairs(a.pairs)
    loops = load_loops(a.work)
    gdir = os.path.join(a.work, "grad")
    os.makedirs(gdir, exist_ok=True)
    import torch
    model = E.load_model(a.model, a.device) if os.path.isdir(a.model) else _load_named(a.model, a.device)
    layers = find_layers(model, a.layers_prefix)
    for p in pairs:
        module_of(layers, *p)
    t0 = time.time()
    pos = {lp["id"]: i for i, lp in enumerate(loops)}
    h = halves_of(loops)
    other = {}   # (draw k, loop id) -> the loop whose continuation it is scored on (a derangement inside each half)
    for k in range(N_PERM):
        for ls in h.values():
            for lp, j in zip(ls, G.derangement(len(ls), G.SEED + k)):
                other[(k, lp["id"])] = ls[int(j)]
    jobs = [("loop", lambda lp: loop_job(lp), ("p", "o")), ("lm", lambda lp: lm_job(lp), ("p",))]
    jobs += [(f"perm{k}", (lambda lp, k=k: perm_job(lp, other[(k, lp["id"])])), ("p", "o")) for k in range(N_PERM)]
    for i, lp in enumerate(loops):
        for kind, mk, kinds in jobs:
            path = os.path.join(gdir, f"{kind}_{lp['id']}.npz")
            if os.path.exists(path):
                continue
            ids, first = mk(lp)
            res = grad_job(model, layers, pairs, ids, first, kinds)
            save_job(path, res)
            first_key = next(iter(res[kinds[0]]))
            print(f"loop {lp['id']} ({i + 1}/{len(loops)}) {kind}: {len(ids)} tokens, loss {res[kinds[0]][first_key]['loss']:.5f}, "
                  f"check {max(d['check'] for d in res[kinds[0]].values()):.1e} [{time.time() - t0:.0f}s]", flush=True)
            if a.device == "mps":
                torch.mps.empty_cache()


def _load_named(name, device):
    import torch
    import transformers as T
    try:
        try:
            m = T.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
        except (ValueError, KeyError, OSError, AttributeError):
            m = T.AutoModelForImageTextToText.from_pretrained(name, dtype=torch.float32)
    except (OSError, ValueError) as e:
        raise G.GradError("w4g-model", f"{name}: {str(e)[:200]}")
    return m.to(device).eval()


# ---------------------------------------------------------------- analyse (numpy)

def load_sums(work, loops, kind, layer, loss, keep=False):
    """The Sums of the loops' stored gradients for a layer and loss (and, if `keep`, the per-loop G and m for the sign-flip null)."""
    sums, gs, ms = None, [], []
    for lp in loops:
        path = os.path.join(work, "grad", f"{kind}_{lp['id']}.npz")
        try:
            z = np.load(path)
            g, m, q, r, n = z[f"g{layer}_{loss}"], z[f"m{layer}_{loss}"], z[f"q{layer}_{loss}"], z[f"r{layer}_{loss}"], int(z[f"n{layer}_{loss}"])
        except (OSError, KeyError, ValueError) as e:
            raise G.GradError("w4g-input", f"{path}: {str(e)[:200]} (run capture)")
        if sums is None:
            sums = G.Sums(g.shape[0], g.shape[1])
        sums.add(g, m, q, r, n)
        if keep:
            gs.append(g)
            ms.append(m)
    return sums, gs, ms


def cmd_analyse(a):
    loops = load_loops(a.work)
    h = halves_of(loops)
    pub = K.load_published(BASE_DIR, EDITED_DIR)
    res = {"layers": {}, "n_loops": {k: len(v) for k, v in h.items()}, "split": {k: [lp["id"] for lp in v] for k, v in h.items()}}
    per_half = {hn: {} for hn in G.HALVES}
    null_max, cand = {}, {}
    for layer in LAYERS:
        w, dw = pub[layer]["w"], pub[layer]["dw"]
        _, u_pub, _ = L.rank1(dw)
        cols_pub = S.published_columns(dw)
        rnd_cos = S.random_cosines(u_pub, 200, G.SEED)
        rnd_ov = S.random_overlaps(cols_pub, w.shape[1], 200, G.SEED, k=int(cols_pub.size))
        lr = {"random": {"cos_max": float(rnd_cos.max()), "cos_median": float(np.median(rnd_cos)), "overlap_max": int(rnd_ov.max()),
                         "overlap_median": float(np.median(rnd_ov))}, "halves": {}, "published_columns": int(cols_pub.size)}
        null_dir = {}   # dir/loss -> the largest |cos| any null reaches
        for hn in G.HALVES:
            by_loss = {}
            for loss in G.LOSSES:
                by_loss[loss], _, _ = load_sums(a.work, h[hn], "loop", layer, loss)
            an = G.analyse_half(by_loss, w, u_pub, cols_pub, int(cols_pub.size))
            per_half[hn][layer] = an
            lm_sums, _, _ = load_sums(a.work, h[hn], "lm", layer, "p")
            lm_dirs = G.directions(lm_sums)
            half = {"rows": an["rows"], "perm": {}, "oracle": {}, "along_u_pub": {}, "specificity": {}}
            for loss in G.LOSSES:
                pr = []
                for k in range(N_PERM):
                    ps, _, _ = load_sums(a.work, h[hn], f"perm{k}", layer, loss)
                    dk = G.directions(ps)
                    c1 = G.column_rules(ps, w, dk["GU1"], int(cols_pub.size))["C1"]
                    pr.append({"cos": {d: S.abscos(u, u_pub) for d, u in dk.items()}, "C1_overlap": S.overlap(c1, cols_pub)})
                    for d in G.DIRS:
                        null_dir[f"{d}/{loss}"] = max(null_dir.get(f"{d}/{loss}", 0.0), pr[-1]["cos"][d])
                half["perm"][loss] = pr
                rules = G.column_rules(by_loss[loss], w, u_pub, int(cols_pub.size))
                half["oracle"][loss] = {r: S.overlap(c, cols_pub) for r, c in rules.items()}
                half["along_u_pub"][loss] = {"u_pub": G.along(by_loss[loss].g, u_pub), "chance": float(np.sqrt(1.0 / w.shape[0])),
                                             "GU1": G.along(by_loss[loss].g, an["dirs"][f"GU1/{loss}"])}
            for d, u in lm_dirs.items():
                half["specificity"][d] = {"cos_pub": S.abscos(u, u_pub), "cos_loop_p": S.abscos(u, an["dirs"][f"{d}/p"]),
                                          "cos_loop_o": S.abscos(u, an["dirs"][f"{d}/o"])}
                for loss in G.LOSSES:
                    null_dir[f"{d}/{loss}"] = max(null_dir.get(f"{d}/{loss}", 0.0), half["specificity"][d]["cos_pub"])
            lr["halves"][hn] = half
            if hn == "train":
                cand[layer] = {"u": an["dirs"]["GU1/p"], "cols": an["cols"]["GU1/p"]["C1"], "g": by_loss["p"].g.astype(np.float32)}
            print(f"layer {layer} {hn}: done", flush=True)
        lr["stability"] = {k: {"cos_train_test": S.abscos(per_half["train"][layer]["dirs"][k], per_half["test"][layer]["dirs"][k]),
                               "C1_overlap_train_test": S.overlap(per_half["train"][layer]["cols"][k]["C1"], per_half["test"][layer]["cols"][k]["C1"])}
                           for k in per_half["train"][layer]["dirs"]}
        for loss in G.LOSSES:
            for d in G.DIRS:
                k = f"{d}/{loss}"
                null_max[(k, layer)] = max(float(rnd_cos.max()), null_dir.get(k, 0.0))
        res["layers"][str(layer)] = lr
    table = G.merge_halves(per_half, LAYERS)
    beats = G.beats_of(table, null_max, LAYERS)
    verdict = G.decide(table, beats, LAYERS)
    res["table"] = {f"{d}|{r}": {str(l): list(v) for l, v in by.items()} for (d, r), by in sorted(table.items())}
    res["null_max"] = {f"{k}|{l}": v for (k, l), v in sorted(null_max.items())}
    res["beats"] = {f"{k}|{l}": v for (k, l), v in sorted(beats.items())}
    pk = ("GU1/p", "C1")
    res["primary"] = {"pair": "GU1/p + C1", "cos_overlap_by_layer": {str(l): list(table[pk][l]) for l in LAYERS}}
    res["verdict"] = {"status": verdict["status"], "recovered": [list(p) for p in verdict["recovered"]], "partial_a": verdict["partial_a"],
                      "partial_b": [list(p) for p in verdict["partial_b"]], "best": list(verdict["best"]) if verdict["best"] else None}
    best = verdict["best"]
    if best:
        res["best_pair"] = {str(l): list(table[best][l]) for l in LAYERS}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    np.savez(os.path.join(a.work, "cand.npz"), **{f"{k}{l}": v for l, d in cand.items() for k, v in d.items()})
    print(f"verdict: {verdict['status']}; primary (GU1/p, C1) min over halves {res['primary']['cos_overlap_by_layer']}; best {best} {res.get('best_pair')}")
    print(f"wrote {a.out}")


# ---------------------------------------------------------------- build (numpy): the candidates and controls

def cmd_build(a):
    try:
        z = np.load(os.path.join(a.work, "cand.npz"))
    except (OSError, ValueError, KeyError) as e:
        raise G.GradError("w4g-input", f"{a.work}/cand.npz: {str(e)[:200]} (run analyse)")
    pub = K.load_published(BASE_DIR, EDITED_DIR)
    ep, eg, cp, cg, info = {}, {}, {}, {}, {"layers": {}}
    for l in LAYERS:
        w, dwp = pub[l]["w"], pub[l]["dw"]
        target = float(np.linalg.norm(dwp))
        u, cols, g = z[f"u{l}"], z[f"cols{l}"], z[f"g{l}"]
        ur = G.random_unit(w.shape[0], l)
        ep[l], eg[l] = G.edit_p(w, u, cols, target), G.edit_g(g, u, cols, target)
        cp[l], cg[l] = G.edit_p(w, ur, cols, target), G.edit_g(g, ur, cols, target)
        info["layers"][str(l)] = {"target_norm": target, "columns": [int(c) for c in cols], "overlap_with_published": S.overlap(cols, S.published_columns(dwp)),
                                  "cos_u_pub": S.abscos(u, L.rank1(dwp)[1]), "cos_random_u_pub": S.abscos(ur, L.rank1(dwp)[1]),
                                  "edit_cos_with_published": {"P": G.edit_cosine(ep[l], dwp), "G": G.edit_cosine(eg[l], dwp),
                                                              "ctrlP": G.edit_cosine(cp[l], dwp), "ctrlG": G.edit_cosine(cg[l], dwp)}}
    np.savez(os.path.join(a.work, "edits_ctrlp.npz"), **{f"dw{l}": v for l, v in cp.items()})
    np.savez(os.path.join(a.work, "edits_ctrlg.npz"), **{f"dw{l}": v for l, v in cg.items()})
    for name, ed in (("r4p", ep), ("r4g", eg)):
        out = os.path.join(a.out_dir, name)
        man = K.write_checkpoint(BASE_DIR, out, name, pub, edits=ed)
        info[name] = {"dir": out, "checks": man["checks"]["ok"], "cos_written_vs_intended": man.get("cos_min"), "keys": sorted(man.keys())}
        print(f"{name}: written to {out}, read-back ok {man['checks']['ok']}", flush=True)
    with open(os.path.join(a.work, "build.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    for l in LAYERS:
        i = info["layers"][str(l)]
        print(f"layer {l}: overlap {i['overlap_with_published']}/64, |cos u_pub| {i['cos_u_pub']:.3f} (random {i['cos_random_u_pub']:.3f}), edit cosines {i['edit_cos_with_published']}")


# ---------------------------------------------------------------- behave (MPS): E2c on the held-out loops

def edit_weights(spec_path, layers_n=LAYERS):
    z = np.load(spec_path)
    pub = K.load_published(BASE_DIR, EDITED_DIR)
    return {l: L.bf16_round((pub[l]["w"] + z[f"dw{l}"]).astype(np.float32)) for l in layers_n}


def cmd_behave(a):
    test = read_json(os.path.join(a.work, "test_loops.json"), "w4g-loops")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"records_{a.name}.jsonl")
    ids = {lp["id"] for lp in test}
    if a.reuse and not os.path.exists(path):
        src = os.path.join(a.reuse, f"records_{a.name}.jsonl")
        if os.path.exists(src):
            rows = [r for r in E.load_records(src) if r["id"] in ids]
            with open(path, "w", encoding="utf-8") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
            print(f"{a.name}: reused {len(rows)} records of {src}")
    if bool(a.model_dir) == bool(a.edits):
        raise G.GradError("w4g-input", "exactly one of --model-dir and --edits")
    if a.model_dir:
        E.run(a.name, a.model_dir, os.path.join(a.work, "test_loops.json"), a.out, a.device, tuple(a.seeds))
        return
    done = {r["id"] for r in E.load_records(path)} if os.path.exists(path) else set()
    todo = [lp for lp in test if lp["id"] not in done]
    if not todo:
        print(f"{a.name}: all {len(test)} loops done")
        return
    import torch
    model = E.load_model(BASE_DIR, a.device)
    layers = find_layers(model)
    for l, wnew in edit_weights(a.edits).items():
        module_of(layers, l, "mlp.down_proj").weight.data.copy_(torch.from_numpy(wnew).to(a.device))
    t0 = time.time()
    for i, lp in enumerate(todo):
        rec = E.measure_loop(model, lp, tuple(a.seeds), a.device)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{a.name} loop {lp['id']} ({i + 1}/{len(todo)}) E={np.mean([s['m'] < E.NEXT for s in rec['samples']]):.2f} [{time.time() - t0:.0f}s]", flush=True)


# ---------------------------------------------------------------- kl (MPS): the collateral

def variant_tensors(spec, device_free=True):
    """{layer: float32 [out, in]} of the down_proj weights a variant spec names: `pub`, `dir:PATH` (a written checkpoint), `edits:NPZ`."""
    pub = K.load_published(BASE_DIR, EDITED_DIR)
    if spec == "pub":
        return {l: (pub[l]["w"] + pub[l]["dw"]).astype(np.float32) for l in LAYERS}
    kind, _, path = spec.partition(":")
    if kind == "edits" and path:
        return edit_weights(path)
    if kind == "dir" and path:
        lay = K._layout(path)
        out = {}
        for l in LAYERS:
            n = K.NAME_FMT.format(l)
            if n not in lay:
                raise G.GradError("w4g-input", f"`{n}` is not in {path}")
            out[l] = K.decode(K._read_raw(lay, n), lay[n][1]).astype(np.float32)
        return out
    raise G.GradError("w4g-input", f"variant spec `{spec}` is not pub, dir:PATH or edits:NPZ")


def cmd_kl(a):
    import torch
    sys.path.insert(0, os.path.join(ROOT, "examples", "qwen35"))
    import hf08_ref as H
    import w4_collateral as C
    specs = []
    for v in a.variant:
        name, _, spec = v.partition("=")
        if not name or not spec:
            raise G.GradError("w4g-input", f"`{v}` is not NAME=SPEC")
        specs.append((name, spec))
    ws = C.windows(os.path.dirname(a.corpus), "eval") if os.path.basename(a.corpus) == "corpus.json" else None
    if ws is None:
        raise G.GradError("w4g-input", f"{a.corpus} is not a corpus.json")
    dev = a.device
    base, work = (H.build_model(BASE_DIR, "f32").to(dev).eval() for _ in range(2))
    for m in (base, work):
        for p in m.parameters():
            p.requires_grad_(False)
    dom_names = []
    for w in ws:
        T = len(w["ids"])
        for d in C.seg_domains(w, T)[0][:1] + (["math_chat_resp"] if w["domain"] == "math_chat" else []):
            if d not in dom_names:
                dom_names.append(d)
    dom_idx = {d: i for i, d in enumerate(dom_names)}
    hb, seg, dom, counter = [], [], [], 0
    t0 = time.time()
    for w in ws:
        ids = torch.tensor([w["ids"]], device=dev)
        T = ids.shape[1]
        hb.append(C.hidden(base, ids))
        names, part = C.seg_domains(w, T)
        dom.append(np.array([dom_idx[n] for n in names]))
        seg.append(counter + part)
        counter += 2 if w["domain"] == "math_chat" else 1
    base_w = {l: C.down(work, l).weight.data.clone() for l in LAYERS}
    summ = {}
    if os.path.exists(a.out):
        summ = read_json(a.out, "w4g-input")
    print(f"base pass {time.time() - t0:.0f}s, {len(ws)} windows", flush=True)
    for name, spec in specs:
        t1 = time.time()
        for l, wnew in variant_tensors(spec).items():
            C.down(work, l).weight.data.copy_(torch.from_numpy(wnew).to(dev))
        kl_l, nb_l, nv_l, ag_l = [], [], [], []
        for i, w in enumerate(ws):
            ids = torch.tensor([w["ids"]], device=dev)
            T = ids.shape[1]
            hv = C.hidden(work, ids)
            for s0 in range(0, T - 1, 256):
                s1 = min(s0 + 256, T - 1)
                tg = torch.tensor(w["ids"][s0 + 1:s1 + 1], device=dev)
                kl, nb, nv, _en, ag = L.token_stats(base.lm_head(hb[i][s0:s1]), work.lm_head(hv[s0:s1]), tg)
                kl_l.append(kl)
                nb_l.append(nb)
                nv_l.append(nv)
                ag_l.append(ag)
        arr = {"kl": np.concatenate(kl_l), "nb": np.concatenate(nb_l), "nv": np.concatenate(nv_l), "agree": np.concatenate(ag_l),
               "seg": np.concatenate(seg), "dom": np.concatenate(dom)}
        s = C.summarize(arr, dom_names)
        summ[name] = {"all": {k: s["all"][k] for k in ("positions", "sequences", "kl", "dppl", "top1_agree", "finite")},
                      "domains": {d: {"kl": v["kl"]["mean"], "dppl": v["dppl"]["mean"]} for d, v in s["domains"].items()}, "seconds": time.time() - t1}
        k = s["all"]["kl"]
        print(f"{name}: mean KL {k['mean']:.5f} [{k['lo']:.5f}, {k['hi']:.5f}] dppl {s['all']['dppl']['mean'] * 100:.3f}% top1 {s['all']['top1_agree']:.4f} ({time.time() - t1:.0f}s)", flush=True)
        for l in LAYERS:
            C.down(work, l).weight.data.copy_(base_w[l])
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=1)


# ---------------------------------------------------------------- score

def score(directory):
    recs = {}
    try:
        names = sorted(os.listdir(directory))
    except OSError as e:
        raise G.GradError("w4g-input", f"{directory}: {e}")
    for fn in names:
        if fn.startswith("records_") and fn.endswith(".jsonl"):
            recs[fn[len("records_"):-len(".jsonl")]] = E.load_records(os.path.join(directory, fn))
    if "base" not in recs:
        raise G.GradError("w4g-input", f"{directory}: no records_base.jsonl")
    kl = read_json(os.path.join(directory, "kl.json"), "w4g-input") if os.path.exists(os.path.join(directory, "kl.json")) else {}
    base = recs["base"]
    out = {"n_loops": len(base), "models": {}}
    for name, rows in recs.items():
        pl = E.per_loop(rows)
        row = {"n_loops": len(pl), "E": float(np.mean([v[0] for v in pl.values()])), "nll": float(np.mean([v[1] for v in pl.values()])),
               "leave64": float(np.mean([v[4] for v in pl.values()])), "match": float(np.mean([v[3] for v in pl.values()]))}
        if name != "base":
            g, ids = E.gains(rows, base)
            if len(ids) < 2:
                raise G.GradError("w4g-input", f"{name}: fewer than two loops in common with the base")
            lo, hi = E.bootstrap(g["E"])
            row["G_E"] = [float(np.mean(g["E"])), lo, hi]
        if name in kl:
            k = kl[name]["all"]["kl"]
            row["kl"] = [k["mean"], k["lo"], k["hi"]]
            row["dppl"] = kl[name]["all"]["dppl"]["mean"]
        out["models"][name] = row
    ms = out["models"]
    for cand, ctrl in (("r4p", "ctrlp"), ("r4g", "ctrlg")):
        if cand in ms and "kl" in ms[cand]:
            f = G.found(ms[cand]["E"], ms[cand]["kl"][2])
            out[cand] = {"found": f, "specific": bool(f and ctrl in ms and G.specific(ms[cand]["E"], ms[ctrl]["E"]))}
    return out


def render(res):
    lines = [f"test loops: {res['n_loops']}", "", "| model | E | leave<64 | match prefix | NLL (256) | G_E [95% CI] | mean KL [95% CI] | dPPL |", "|---|---|---|---|---|---|---|---|"]
    for name, r in res["models"].items():
        ge = "" if "G_E" not in r else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*r["G_E"])
        kl = "" if "kl" not in r else "{:.5f} [{:.5f}, {:.5f}]".format(*r["kl"])
        dp = "" if "dppl" not in r else f"{100 * r['dppl']:.3f}%"
        lines.append(f"| {name} | {r['E']:.3f} | {r['leave64']:.3f} | {r['match']:.1f} | {r['nll']:.4f} | {ge} | {kl} | {dp} |")
    for c in ("r4p", "r4g"):
        if c in res:
            lines.append(f"{c}: FOUND {res[c]['found']}, SPECIFIC {res[c]['specific']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--base-jsonl", required=True)
    s.add_argument("--model-dir", default=BASE_DIR)
    s.add_argument("--gsm8k", default=os.path.join(HOME, "models", "gsm8k", "main", "test-00000-of-00001.parquet"))
    s.add_argument("--work", required=True)
    s.add_argument("--n", type=int, default=N_LOOPS)
    s.add_argument("--out")
    c = sub.add_parser("capture")
    c.add_argument("--work", required=True)
    c.add_argument("--model", default=BASE_DIR)
    c.add_argument("--pairs", nargs="+", default=[f"{l}:{m}" for l, m in PAIRS])
    c.add_argument("--layers-prefix")
    c.add_argument("--device", default="mps")
    n = sub.add_parser("analyse")
    n.add_argument("--work", required=True)
    n.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--work", required=True)
    b.add_argument("--out-dir", required=True)
    h = sub.add_parser("behave")
    h.add_argument("--name", required=True)
    h.add_argument("--model-dir")
    h.add_argument("--edits")
    h.add_argument("--work", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--reuse")
    h.add_argument("--device", default="mps")
    h.add_argument("--seeds", type=int, nargs="+", default=list(E.SEEDS))
    k = sub.add_parser("kl")
    k.add_argument("--variant", nargs="+", required=True)
    k.add_argument("--corpus", default=CORPUS)
    k.add_argument("--out", required=True)
    k.add_argument("--device", default="mps")
    r = sub.add_parser("score")
    r.add_argument("--dir", required=True)
    r.add_argument("--json")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "select":
            cmd_select(a)
        elif a.cmd == "capture":
            cmd_capture(a)
        elif a.cmd == "analyse":
            cmd_analyse(a)
        elif a.cmd == "build":
            cmd_build(a)
        elif a.cmd == "behave":
            cmd_behave(a)
        elif a.cmd == "kl":
            cmd_kl(a)
        else:
            res = score(a.dir)
            print(render(res))
            if a.json:
                with open(a.json, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=1)
    except (G.GradError, E.E2cError, K.CkptError, S.SearchError) as e:
        print(f"error[{getattr(e, 'rule', 'w4g')}]: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
