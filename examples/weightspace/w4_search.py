#!/usr/bin/env python3
"""docs/weight-space-w4-search.md: can the loop-fixing edit be rediscovered without the published direction or columns?

    python3 examples/weightspace/w4_search.py capture --data DIR/base_8192.jsonl [--work build/w4s] [--threads 4]   # torch, CPU only
    python3 examples/weightspace/w4_search.py analyse [--work build/w4s] [--out examples/weightspace/results/w4_search.json]  # numpy only
    python3 examples/weightspace/w4_search.py write   [--work build/w4s] --out-dir build/ckpt                      # numpy only; only if the rule says so

`capture` runs the base's text tower in float32 on the CPU over 24 windows of the base's own generations (read-only) and stores, per
layer 15 and 19, the input and output of `mlp.down_proj` at the counted positions. `analyse` computes the candidates, the nulls, the scores and
the verdict of the pre-registered rule (section 4). `write` writes at most two checkpoints, only when the verdict is RECOVERED or PARTIAL
(section 5). Exit status: 0 done, 1 a check failed, 2 the input is unusable (every refusal prints its rule tag `w4s-*` or `w4c-*`)."""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import w4_ckpt as K  # noqa: E402
import w4_lib as L  # noqa: E402
import w4_search_lib as S  # noqa: E402

LAYERS = (15, 19)
N_LOOP, N_NON, WIN, COUNT = 8, 8, 1024, 512
DATA_KINDS = {"D1": "data", "D2": "data", "D3": "data", "D4": "data", "W1": "weights", "W2": "weights"}


# ------------------------------------------------------------------ choosing the windows (section 2; pure)

def choose_windows(rows, tokenize):
    """[(label, kind, row id, token ids of the window, first counted position)] for the 8 loop, 8 non-loop and 8 early windows.
    `rows` are the jsonl records; `tokenize(text) -> ids`."""
    try:
        rows = sorted(rows, key=lambda r: int(r["id"]))
    except (KeyError, TypeError, ValueError):
        raise S.SearchError("w4s-data", "a row has no numeric id")
    loops, non, early = [], [], []
    for r in rows:
        if not isinstance(r, dict) or "text" not in r or "hit_limit" not in r:
            raise S.SearchError("w4s-data", "a row lacks `text` or `hit_limit`")
        if r["hit_limit"] and len(loops) < N_LOOP:
            ids = tokenize(r["text"])
            rep = L.rep4(ids[-COUNT:])
            if len(ids) >= WIN and rep is not None and rep >= 0.5:
                loops.append(("loop", "tail", r["id"], ids[-WIN:], WIN - COUNT))
                early.append(("non", "early", r["id"], ids[:WIN], COUNT))
        elif not r["hit_limit"] and len(non) < N_NON:
            ids = tokenize(r["text"])
            if len(ids) >= 700:
                non.append(("non", "finished", r["id"], ids[:WIN], COUNT))
    if len(loops) < N_LOOP or len(non) < N_NON:
        raise S.SearchError("w4s-data", f"found {len(loops)} loop and {len(non)} non-loop texts, wanted {N_LOOP} and {N_NON}")
    return loops + non + early


# ------------------------------------------------------------------ capture (torch, CPU)

class _Stop(Exception):
    pass


def cmd_capture(a):
    import torch
    sys.path.insert(0, os.path.join(ROOT, "examples", "qwen35"))
    import hf08_ref as H
    import transformers as T
    torch.set_num_threads(a.threads)
    try:
        with open(a.data, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError) as e:
        raise S.SearchError("w4s-data", f"{a.data}: {str(e)[:200]}")
    tok = T.AutoTokenizer.from_pretrained(a.base)
    wins = choose_windows(rows, lambda t: tok(t, add_special_tokens=False)["input_ids"])
    model = H.build_model(a.base, "f32").eval()   # CPU float32
    got = {}

    def hook(layer):
        def f(_m, inp, out):
            st = got["start"]
            got[layer] = (inp[0][0, st:].detach().float().numpy().copy(), out[0, st:].detach().float().numpy().copy())
            if layer == LAYERS[-1]:
                raise _Stop()
        return f
    hs = [model.model.layers[l].mlp.down_proj.register_forward_hook(hook(l)) for l in LAYERS]
    store = {l: {"a": [], "y": []} for l in LAYERS}
    meta = []
    t0 = time.time()
    try:
        for i, (label, kind, rid, ids, start) in enumerate(wins):
            got.clear()
            got["start"] = start
            try:
                with torch.no_grad():
                    model.model(torch.tensor([ids]), use_cache=False)
            except _Stop:
                pass
            for l in LAYERS:
                store[l]["a"].append(got[l][0])
                store[l]["y"].append(got[l][1])
            meta.append({"label": label, "kind": kind, "id": rid, "tokens": len(ids), "counted": int(len(ids) - start)})
            print(f"window {i + 1}/{len(wins)} {label}/{kind} row {rid}: {len(ids)} tokens, {meta[-1]['counted']} counted [{time.time() - t0:.0f}s]", flush=True)
    finally:
        for h in hs:
            h.remove()
    os.makedirs(a.work, exist_ok=True)
    arrs = {}
    for l in LAYERS:
        arrs[f"a{l}"] = np.concatenate(store[l]["a"])
        arrs[f"y{l}"] = np.concatenate(store[l]["y"])
    arrs["off"] = np.cumsum([0] + [m["counted"] for m in meta])
    arrs["label"] = np.array([m["label"] for m in meta])
    np.savez(os.path.join(a.work, "capture.npz"), **arrs)
    with open(os.path.join(a.work, "capture.json"), "w", encoding="utf-8") as f:
        json.dump({"data": os.path.abspath(a.data), "windows": meta, "seconds": time.time() - t0, "threads": a.threads}, f, indent=1)
    print(f"captured {len(meta)} windows, {int(arrs['off'][-1])} positions per layer")
    return 0


# ------------------------------------------------------------------ analysis (numpy; section 3 and 4)

def split_windows(z, layer):
    """[(label, y, a)] per window from a capture."""
    off, lab = z["off"], z["label"]
    return [(str(lab[i]), z[f"y{layer}"][off[i]:off[i + 1]], z[f"a{layer}"][off[i]:off[i + 1]]) for i in range(len(lab))]


def dd_diff(mom, li, ni):
    gl, _, n_l = S.pool(mom, li)
    gn, _, n_n = S.pool(mom, ni)
    return gl / n_l - gn / n_n


def analyse_layer(wins, w, dw, n_perm=50, n_rand=200):
    """Everything of sections 3.1 to 3.3 for one layer. `wins` = [(label, y, a)] loop windows first; `w`, `dw` = [out, in] base and published edit.
    Returns (report dict, {direction: unit vector}, {(direction, rule): columns}, {(direction, rule): (|cos|, overlap)})."""
    nl = sum(1 for x in wins if x[0] == "loop")
    nn = len(wins) - nl
    if nl < 2 or nn < 2 or [x[0] for x in wins] != ["loop"] * nl + ["non"] * nn:
        raise S.SearchError("w4s-data", "need at least 2 windows of each class, loop windows first")
    _, u_pub, _ = L.rank1(dw)
    cols_pub = S.published_columns(dw)
    mom = [S.window_moments(x[1]) for x in wins]
    li, ni = np.arange(nl), nl + np.arange(nn)

    def dirs_of(lidx, nidx):
        gl, sl, n_l = S.pool(mom, lidx)
        gn, sn, n_n = S.pool(mom, nidx)
        return S.data_directions(gl, sl, n_l, gn, sn, n_n)
    dd = dirs_of(li, ni)
    dirs = dict(dd)
    dirs.update(S.weight_directions(w))
    rep = {"cos": {k: S.abscos(v, u_pub) for k, v in dirs.items()}}
    rc = S.random_cosines(u_pub, n_rand)
    rep["random_cos"] = {"median": float(np.median(rc)), "p99": float(np.percentile(rc, 99)), "max": float(rc.max())}
    perm_max = {k: 0.0 for k in dd}
    perm_all = {k: [] for k in dd}
    for pl, pn in S.label_permutations(nl, nn, n_perm):
        pd = dirs_of(pl, pn)
        for k, v in pd.items():
            c = S.abscos(v, u_pub)
            perm_all[k].append(c)
            perm_max[k] = max(perm_max[k], c)
    rep["permutation_null"] = {k: {"median": float(np.median(v)), "max": perm_max[k], "n": len(v)} for k, v in perm_all.items()}
    (ha_l, ha_n), (hb_l, hb_n) = S.halves(nl, nn)
    da, db = dirs_of(ha_l, ha_n), dirs_of(hb_l, hb_n)
    rep["half_vs_half_cos"] = {k: S.abscos(da[k], db[k]) for k in dd}
    rep["half_cos_with_pub"] = {k: [S.abscos(da[k], u_pub), S.abscos(db[k], u_pub)] for k in dd}
    rep["subspace_capture"] = {str(k): {"u_pub": v[0], "chance": v[1]} for k, v in S.subspace_capture(w, u_pub).items()}
    # exploratory, added after the first run (docs/weight-space-w4-search.md 7.3): what u_pub is made of
    gall, sall, nall = S.pool(mom, np.arange(len(wins)))
    rep["exploratory"] = {"cos_with_mean_output": S.abscos(sall / nall, u_pub),
                          "capture_in_top_eigenspace_of_all_positions": {str(k): {"u_pub": v[0], "chance": v[1]} for k, v in S.eig_capture(gall / nall, u_pub).items()},
                          "capture_in_top_eigenspace_of_loop_minus_non": {str(k): {"u_pub": v[0], "chance": v[1]} for k, v in
                                                                         S.eig_capture(dd_diff(mom, li, ni), u_pub).items()}}
    # neurons
    a_l = np.concatenate([x[2] for x in wins[:nl]])
    a_n = np.concatenate([x[2] for x in wins[nl:]])
    rms, dmean = S.neuron_stats(a_l, a_n)
    rep["positions"] = {"loop": int(a_l.shape[0]), "non": int(a_n.shape[0])}
    k = int(cols_pub.size)   # 64 for the published edit
    c5, v5 = S.neuron_gev(a_l, a_n, k=k)
    cols = {("R5", "-"): c5}
    ncol = w.shape[1]
    ro = S.random_overlaps(cols_pub, ncol, n_rand, k=k)
    rep["random_overlap"] = {"mean": float(ro.mean()), "p99": float(np.percentile(ro, 99)), "max": int(ro.max())}
    rep["rms_alone_overlap"] = S.overlap(S.top_cols(rms, k), cols_pub)
    rep["R5_overlap"] = S.overlap(c5, cols_pub)
    rep["published_columns"] = int(cols_pub.size)
    table, rows = {}, {}
    cand_dirs = dict(dirs)
    cand_dirs["PUB"] = u_pub   # the oracle reference
    for name, u in cand_dirs.items():
        rules = S.column_rules(w, u, rms=rms, dmean=dmean, k=k)
        for rn, cs in rules.items():
            e = S.candidate_edit(w, u, cs)
            ov = S.overlap(cs, cols_pub)
            rows[f"{name}/{rn}"] = {"cos": S.abscos(u, u_pub), "overlap": ov, "edit_cos": S.edit_cosine(e, dw)}
            cols[(name, rn)] = cs
            if name != "PUB":
                table[(name, rn)] = (S.abscos(u, u_pub), ov)
        if name != "PUB":
            e = S.candidate_edit(w, u, c5)   # R5 needs no direction: its columns with this direction
            ov = S.overlap(c5, cols_pub)
            rows[f"{name}/R5"] = {"cos": S.abscos(u, u_pub), "overlap": ov, "edit_cos": S.edit_cosine(e, dw)}
            cols[(name, "R5")] = c5
            table[(name, "R5")] = (S.abscos(u, u_pub), ov)
    rep["pairs"] = rows
    return rep, cand_dirs, cols, table


def verdict(per_layer):
    """Section 4 on {layer: (report, dirs, cols, table)}."""
    table, beats = {}, {}
    for l, (rep, _d, _c, tb) in per_layer.items():
        for pair, v in tb.items():
            table.setdefault(pair, {})[l] = v
        for k, pn in rep["permutation_null"].items():
            beats[(k, l)] = rep["cos"][k] > pn["max"]
    return S.decide(table, DATA_KINDS, beats, layers=tuple(per_layer))


def cmd_analyse(a):
    z = np.load(os.path.join(a.work, "capture.npz"))
    pub = K.load_published(a.base, a.edited)
    per = {}
    for l in LAYERS:
        t0 = time.time()
        per[l] = analyse_layer(split_windows(z, l), pub[l]["w"], pub[l]["dw"], a.perms)
        print(f"layer {l} analysed [{time.time() - t0:.0f}s]", flush=True)
    v = verdict(per)
    out = {"design": "docs/weight-space-w4-search.md", "alpha": S.ALPHA, "rule": S.R, "verdict": {k: (v[k] if k != "best" else list(v[k]) if v[k] else None) for k in v},
           "layers": {str(l): per[l][0] for l in LAYERS}}
    out["verdict"]["recovered"] = [list(p) for p in v["recovered"]]
    out["verdict"]["partial_b"] = [list(p) for p in v["partial_b"]]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    cand = {}
    for l in LAYERS:
        for k, u in per[l][1].items():
            cand[f"u_{k}_{l}"] = u
        for (d, r), cs in per[l][2].items():
            cand[f"c_{d}_{r}_{l}"] = cs
    np.savez(os.path.join(a.work, "candidates.npz"), **cand)
    print(json.dumps(out["verdict"], indent=1))
    return 0


def cmd_write(a):
    with open(a.results, encoding="utf-8") as f:
        res = json.load(f)
    status = res["verdict"]["status"]
    if status == "NOT RECOVERED":
        raise S.SearchError("w4s-rule", "the pre-registered rule says NOT RECOVERED: no checkpoint is written")
    best = res["verdict"]["best"]
    z = np.load(os.path.join(a.work, "candidates.npz"))
    pub = K.load_published(a.base, a.edited)
    d, r = best
    u = {l: z[f"u_{d}_{l}"] for l in LAYERS}
    cols = {l: z[f"c_{d}_{r}_{l}"] for l in LAYERS}
    s1 = {l: S.candidate_edit(pub[l]["w"], u[l], cols[l]) for l in LAYERS}
    direction_ok = all(res["layers"][str(l)]["cos"][d] >= S.R["found_cos"] for l in LAYERS)
    if direction_ok:
        s2 = {l: S.candidate_edit(pub[l]["w"], u[l], S.published_columns(pub[l]["dw"])) for l in LAYERS}
        s2_what = "found direction, published columns"
    else:
        s2 = {l: S.candidate_edit(pub[l]["w"], L.rank1(pub[l]["dw"])[1], cols[l]) for l in LAYERS}
        s2_what = "published direction, found columns"
    os.makedirs(a.out_dir, exist_ok=True)
    for name, edits, what in (("s1", s1, f"{d} + {r}, found direction and columns"), ("s2", s2, s2_what)):
        man = K.write_checkpoint(a.base, os.path.join(a.out_dir, name), f"search-{name}", pub, edits=edits, patch=not a.no_patch)
        print(f"wrote {name} ({what}): cosine with intended {man['cosine_with_intended']:.6f}, {man['changed_weights']:,} weights changed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("capture", "analyse", "write"):
        p = sub.add_parser(name)
        p.add_argument("--work", default=os.path.join(ROOT, "build", "w4s"))
        p.add_argument("--base", default=K.BASE_DIR)
        p.add_argument("--edited", default=K.EDITED_DIR)
        if name == "capture":
            p.add_argument("--data", required=True)
            p.add_argument("--threads", type=int, default=4)
        elif name == "analyse":
            p.add_argument("--out", default=os.path.join(HERE, "results", "w4_search.json"))
            p.add_argument("--perms", type=int, default=50)
        else:
            p.add_argument("--results", default=os.path.join(HERE, "results", "w4_search.json"))
            p.add_argument("--out-dir", default=os.path.join(ROOT, "build", "ckpt"))
            p.add_argument("--no-patch", action="store_true")
    a = ap.parse_args(argv)
    try:
        return {"capture": cmd_capture, "analyse": cmd_analyse, "write": cmd_write}[a.cmd](a)
    except (S.SearchError, K.CkptError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (OSError, ImportError, KeyError, ValueError) as e:
        print(f"error: w4s-io: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
