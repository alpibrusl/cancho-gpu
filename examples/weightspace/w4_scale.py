#!/usr/bin/env python3
"""docs/weight-space-w4-r4.md section 9 (R4b): choose the scale of the r4p form on the train loops against a KL target, score once on the held-out loops.

    python3 w4_scale.py grid   --work build/r4b                       # numpy: the 18 (fraction, budget) edits
    python3 w4_scale.py kl     --work build/r4b [--names a b ...]     # MPS: the collateral KL of each (resumable)
    python3 w4_scale.py behave --work build/r4b --name NAME --split train|test [--edits NPZ]   # MPS: E2c on a split
    python3 w4_scale.py select --work build/r4b                       # numpy: the eligible set and the selected config
    python3 w4_scale.py final  --work build/r4b --out results/w4_r4b  # numpy: the verdict from the records
    python3 w4_scale.py write  --work build/r4b --out-dir DIR         # numpy: the selected checkpoint (w4_ckpt)

The pure parts (config naming, the edit, the selection with its tie rules, the verdict) are held by tests/w4_scale.py. Every refusal
prints a rule tag `w4s-*` (or `w4g-*`, `w4e-*` from the modules it uses); the CLI exits 2 on them."""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w4_ckpt as K  # noqa: E402
import w4_e2c as E  # noqa: E402
import w4_grad as W  # noqa: E402
import w4_grad_lib as G  # noqa: E402
import w4_lib as L  # noqa: E402

FRACTIONS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
BUDGETS = (16, 32, 64)
KL_MAX = 0.05
E_MIN = 0.5
CTRL_MAX = 0.1
LAYERS = W.LAYERS


class ScaleError(Exception):
    """A refusal; `rule` is its tag."""

    def __init__(self, rule, msg):
        super().__init__(f"{rule}: {msg}")
        self.rule = rule


# ---------------------------------------------------------------- pure parts

def config_name(f, k):
    return f"f{f:g}_k{int(k)}"


def configs(fractions=FRACTIONS, budgets=BUDGETS):
    """The grid, in a fixed order: budgets outer, fractions inner."""
    for k in budgets:
        if int(k) != k or k < 1:
            raise ScaleError("w4s-config", f"budget {k} is not a positive integer")
    for f in fractions:
        if not (np.isfinite(f) and f > 0):
            raise ScaleError("w4s-config", f"fraction {f} is not positive")
    return [(float(f), int(k)) for k in budgets for f in fractions]


def rank_columns(g, cols):
    """`cols` ordered by the train gradient's column norm, largest first (ties: the smaller index)."""
    g = np.asarray(g, dtype=np.float64)
    cols = G._check_cols(cols, g.shape[1])
    norms = np.linalg.norm(g[:, cols], axis=0)
    order = np.lexsort((cols, -norms))
    return cols[order]


def edit_fk(w, u, ranked, target, f, k):
    """Form P at fraction `f` of the r4p coefficient on the top `k` of the ranked columns. The r4p coefficient is the one that makes the
    edit on ALL ranked columns have Frobenius norm `target`; (f, k) = (1, len(ranked)) is therefore edit_p itself."""
    if not (np.isfinite(f) and f > 0):
        raise ScaleError("w4s-config", f"fraction {f} is not positive")
    if int(k) != k or not 1 <= k <= len(ranked):
        raise ScaleError("w4s-config", f"budget {k} is not in 1..{len(ranked)}")
    full = G.edit_p(w, u, ranked, target)
    keep = np.zeros(full.shape[1], dtype=bool)
    keep[np.asarray(ranked[:int(k)], dtype=np.int64)] = True
    return f * full * keep[None, :]


def ctrl_edit(candidate, w, ur):
    """The random-direction control at the candidate's columns: the same columns, a random unit `ur`, the candidate's Frobenius norm."""
    cols = np.flatnonzero(np.any(candidate != 0, axis=0))
    if cols.size == 0:
        raise ScaleError("w4s-degenerate", "the candidate edit is empty")
    return G.edit_p(w, ur, cols, float(np.linalg.norm(candidate)))


def eligible(kl_rows, kl_max=KL_MAX):
    """{name: kl_hi} -> the names whose upper end is finite and <= kl_max."""
    out = []
    for name, hi in kl_rows.items():
        if not np.isfinite(hi):
            raise ScaleError("w4s-finite", f"{name}: the KL bound is not finite")
        if hi <= kl_max:
            out.append(name)
    return out


def select(train_e, kl_hi, fk):
    """9.3: among names in `train_e`, the largest train E; ties go to the lower KL upper end, then the smaller f, then the smaller k.
    `fk` maps a name to (f, k). Returns the name; refuses on an empty set."""
    if not train_e:
        raise ScaleError("w4s-none", "no eligible configuration was run on the train loops")
    for n, e in train_e.items():
        if not np.isfinite(e) or n not in kl_hi or n not in fk:
            raise ScaleError("w4s-input", f"{n}: missing or non-finite")
    return sorted(train_e, key=lambda n: (-train_e[n], kl_hi[n], fk[n][0], fk[n][1]))[0]


def verdict(e_test, kl_hi, e_ctrl):
    """9.4: FOUND iff held-out E >= 0.5, the selected KL upper end <= 0.05 and the control's held-out E < 0.1."""
    for x in (e_test, kl_hi, e_ctrl):
        if not np.isfinite(x):
            raise ScaleError("w4s-finite", "E or KL is not finite")
    return {"effect": bool(e_test >= E_MIN), "kl": bool(kl_hi <= KL_MAX), "specific": bool(e_ctrl < CTRL_MAX),
            "found": bool(e_test >= E_MIN and kl_hi <= KL_MAX and e_ctrl < CTRL_MAX)}


def exit_share(records):
    """Mean over loops of the per-loop exit share (E2c's E)."""
    if not records:
        raise ScaleError("w4s-input", "no records")
    return float(np.mean([v[0] for v in E.per_loop(records).values()]))


# ---------------------------------------------------------------- files

def read_json(path, rule="w4s-input"):
    return W.read_json(path, rule)


def load_cand(work):
    try:
        z = np.load(os.path.join(work, "cand.npz"))
        return {l: (z[f"u{l}"], z[f"cols{l}"], z[f"g{l}"]) for l in LAYERS}
    except (OSError, ValueError, KeyError) as e:
        raise ScaleError("w4s-input", f"{work}/cand.npz: {str(e)[:200]}")


def edit_path(work, name):
    return os.path.join(work, f"edits_{name}.npz")


def cmd_grid(a):
    pub = K.load_published(W.BASE_DIR, W.EDITED_DIR)
    cand = load_cand(a.work)
    grid = {}
    ranked = {}
    for l in LAYERS:
        u, cols, g = cand[l]
        ranked[l] = rank_columns(g, cols)
    for f, k in configs():
        name = config_name(f, k)
        dws = {l: edit_fk(pub[l]["w"], cand[l][0], ranked[l], float(np.linalg.norm(pub[l]["dw"])), f, k) for l in LAYERS}
        np.savez(edit_path(a.work, name), **{f"dw{l}": dws[l] for l in LAYERS})
        grid[name] = {"f": f, "k": k, "frobenius": {str(l): float(np.linalg.norm(dws[l])) for l in LAYERS}}
    # references: the published edit and e19 as edit files, for the KL and E2c reference rows
    np.savez(edit_path(a.work, "pubedit"), **{f"dw{l}": pub[l]["dw"] for l in LAYERS})
    np.savez(edit_path(a.work, "e19"), dw15=np.zeros_like(pub[15]["dw"]), dw19=pub[19]["dw"])
    with open(os.path.join(a.work, "grid.json"), "w", encoding="utf-8") as fh:
        json.dump({"grid": grid, "ranked": {str(l): [int(c) for c in ranked[l]] for l in LAYERS}}, fh, indent=1)
    print(f"{len(grid)} configurations written to {a.work}")


def cmd_kl(a):
    grid = read_json(os.path.join(a.work, "grid.json"))["grid"]
    names = a.names or (list(grid) + ["pubedit", "e19"])
    out = os.path.join(a.work, "kl.json")
    done = read_json(out) if os.path.exists(out) else {}
    todo = [n for n in names if n not in done]
    if not todo:
        print("all KL rows exist")
        return 0
    return W.main(["kl", "--device", a.device, "--out", out, "--variant"] + [f"{n}=edits:{edit_path(a.work, n)}" for n in todo])


def cmd_behave(a):
    loops = read_json(os.path.join(a.work, f"{a.split}_loops.json"))
    outdir = os.path.join(a.work, f"e2c_{a.split}")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"records_{a.name}.jsonl")
    if a.reuse and not os.path.exists(path):
        src = os.path.join(a.reuse, f"records_{a.reuse_name or a.name}.jsonl")
        ids = {lp["id"] for lp in loops}
        if os.path.exists(src):
            rows = [r for r in E.load_records(src) if r["id"] in ids]
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("".join(json.dumps(r) + "\n" for r in rows))
            print(f"{a.name}: reused {len(rows)} records of {src}")
    done = {r["id"] for r in E.load_records(path)} if os.path.exists(path) else set()
    todo = [lp for lp in loops if lp["id"] not in done]
    if not todo:
        print(f"{a.name}: all {len(loops)} loops done")
        return 0
    import torch
    model = E.load_model(W.BASE_DIR, a.device)
    layers = W.find_layers(model)
    for l, wnew in W.edit_weights(a.edits or edit_path(a.work, a.name)).items():
        W.module_of(layers, l, "mlp.down_proj").weight.data.copy_(torch.from_numpy(wnew).to(a.device))
    t0 = time.time()
    for i, lp in enumerate(todo):
        rec = E.measure_loop(model, lp, tuple(E.SEEDS), a.device)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{a.name}/{a.split} loop {lp['id']} ({i + 1}/{len(todo)}) E={np.mean([s['m'] < E.NEXT for s in rec['samples']]):.2f} [{time.time() - t0:.0f}s]", flush=True)
    return 0


def kl_hi_of(kl):
    return {n: float(v["all"]["kl"]["hi"]) for n, v in kl.items()}


def split_e(work, split, names):
    out = {}
    for n in names:
        p = os.path.join(work, f"e2c_{split}", f"records_{n}.jsonl")
        if os.path.exists(p):
            rows = E.load_records(p)
            n_loops = len(read_json(os.path.join(work, f"{split}_loops.json")))
            if len(rows) == n_loops:
                out[n] = exit_share(rows)
    return out


def cmd_select(a):
    g = read_json(os.path.join(a.work, "grid.json"))["grid"]
    kl = kl_hi_of(read_json(os.path.join(a.work, "kl.json")))
    missing = [n for n in g if n not in kl]
    if missing:
        raise ScaleError("w4s-input", f"no KL for {missing}")
    el = eligible({n: kl[n] for n in g})
    te = split_e(a.work, "train", g)
    fk = {n: (g[n]["f"], g[n]["k"]) for n in g}
    res = {"eligible": el, "train_e": te, "kl_hi": {n: kl[n] for n in g}, "kl_mean": {n: read_json(os.path.join(a.work, "kl.json"))[n]["all"]["kl"]["mean"] for n in g}}
    todo = [n for n in el if n not in te]
    if todo:
        res["todo_train"] = todo
        print("eligible without train records:", " ".join(todo))
    elif el:
        res["selected"] = select({n: te[n] for n in el}, kl, fk)
        print("selected", res["selected"])
    else:
        res["selected"] = None
        print("no eligible configuration: NOT FOUND, no loop is run")
    with open(os.path.join(a.work, "select.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    for n in g:
        print(f"{n}: KL mean {res['kl_mean'][n]:.4f} hi {kl[n]:.4f} {'eligible' if n in el else '-'} train E {te.get(n, float('nan')):.3f}")
    return 0


def cmd_ctrl(a):
    sel = read_json(os.path.join(a.work, "select.json"))["selected"]
    if not sel:
        raise ScaleError("w4s-none", "nothing selected")
    pub = K.load_published(W.BASE_DIR, W.EDITED_DIR)
    z = np.load(edit_path(a.work, sel))
    out = {}
    for l in LAYERS:
        out[f"dw{l}"] = ctrl_edit(z[f"dw{l}"], pub[l]["w"], G.random_unit(pub[l]["w"].shape[0], l))
    np.savez(edit_path(a.work, "ctrl_" + sel), **out)
    print("ctrl_" + sel, "written")
    return 0


def cmd_final(a):
    sel_d = read_json(os.path.join(a.work, "select.json"))
    sel = sel_d["selected"]
    res = {"selected": sel, "selection": sel_d}
    if sel:
        recs = {n: E.load_records(os.path.join(a.work, "e2c_test", f"records_{n}.jsonl"))
                for n in (sel, "ctrl_" + sel, "pubedit", "e19") if os.path.exists(os.path.join(a.work, "e2c_test", f"records_{n}.jsonl"))}
        kl_all = read_json(os.path.join(a.work, "kl.json"))
        res["test_e"] = {n: exit_share(r) for n, r in recs.items()}
        res["test_n_loops"] = {n: len(r) for n, r in recs.items()}
        res["kl"] = {n: kl_all[n]["all"]["kl"] for n in (sel, "pubedit", "e19") if n in kl_all}
        res["dppl"] = {n: kl_all[n]["all"]["dppl"]["mean"] for n in (sel, "pubedit", "e19") if n in kl_all}
        if sel in res["test_e"] and "ctrl_" + sel in res["test_e"]:
            res["verdict"] = verdict(res["test_e"][sel], res["kl"][sel]["hi"], res["test_e"]["ctrl_" + sel])
    else:
        res["verdict"] = {"found": False, "reason": "no eligible configuration"}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "final.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "selection"}, indent=1))
    return 0


def cmd_write(a):
    sel = read_json(os.path.join(a.work, "select.json"))["selected"]
    if not sel:
        raise ScaleError("w4s-none", "nothing selected")
    pub = K.load_published(W.BASE_DIR, W.EDITED_DIR)
    z = np.load(edit_path(a.work, sel))
    man = K.write_checkpoint(W.BASE_DIR, a.out_dir, "r4b_" + sel, pub, edits={l: z[f"dw{l}"] for l in LAYERS})
    print(f"written to {a.out_dir}, read-back ok {man['checks']['ok']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("grid", "select", "ctrl"):
        sub.add_parser(n).add_argument("--work", required=True)
    k = sub.add_parser("kl")
    k.add_argument("--work", required=True)
    k.add_argument("--names", nargs="+")
    k.add_argument("--device", default="mps")
    b = sub.add_parser("behave")
    b.add_argument("--work", required=True)
    b.add_argument("--name", required=True)
    b.add_argument("--split", choices=("train", "test"), required=True)
    b.add_argument("--edits")
    b.add_argument("--reuse")
    b.add_argument("--reuse-name")
    b.add_argument("--device", default="mps")
    f = sub.add_parser("final")
    f.add_argument("--work", required=True)
    f.add_argument("--out", required=True)
    w = sub.add_parser("write")
    w.add_argument("--work", required=True)
    w.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    try:
        return {"grid": lambda: cmd_grid(a), "kl": lambda: cmd_kl(a), "behave": lambda: cmd_behave(a), "select": lambda: cmd_select(a),
                "ctrl": lambda: cmd_ctrl(a), "final": lambda: cmd_final(a), "write": lambda: cmd_write(a)}[a.cmd]() or 0
    except (ScaleError, G.GradError, E.E2cError, K.CkptError) as e:
        print(f"error[{getattr(e, 'rule', 'w4s')}]: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
