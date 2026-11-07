#!/usr/bin/env python3
"""docs/weight-space-w4.md section 14: E2c, the loop-continuation proxy.

    python3 w4_e2c.py select --base-jsonl results/w1_cloud/data/base_8192.jsonl [--model-dir ~/models/qwen35-08b-base] --out DIR
    python3 w4_e2c.py run    --name NAME --model-dir DIR --loops DIR/loops.json --out DIR [--device mps] [--seeds 0 1 2 3]
    python3 w4_e2c.py score  --dir DIR [--json OUT.json]

`select` finds the base's loops and the cut (14.2). `run` measures one model on them (14.3): the sampled continuation (W1's sampler)
and the teacher-forced NLL; it writes DIR/records_NAME.jsonl, one record a loop, and resumes. `score` applies the calibration rule
(14.4) and the recommendation rule (14.5). The pure parts need only numpy; `select` and `run` import torch and transformers.
Refusals carry `w4e-input`, `-loops`, `-model`, `-out`, `-records`; the CLI exits 2 on them."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PMIN, PMAX, MINSPAN, CUT_OFFSET, NEXT = 8, 400, 1500, 1000, 256
N_LOOPS, SEEDS = 40, (0, 1, 2, 3)
BOOT_SEED, BOOT_N = 20261106, 10_000
CALIB = ("edited", "e19", "x1e-1")
ALPHAS = ("a1", "a3", "a6", "a9")
KL4 = {"a1": 0.00042, "a3": 0.00368, "a6": 0.01423, "a9": 0.03256}   # section 13's four-window KL
KL_GATE = 0.02
STOPS = (248044, 248046)


class E2cError(Exception):
    """A refusal; `rule` is its tag (w4e-input, -loops, -model, -out, -records)."""

    def __init__(self, rule, msg):
        super().__init__(msg)
        self.rule = rule


# ---------------------------------------------------------------- pure parts

def find_loop(ids, pmin=PMIN, pmax=PMAX, minspan=MINSPAN):
    """(period, span) of the repeating tail of `ids`: the smallest p in [pmin, pmax] with r(p) + p >= minspan, r(p) the number of
    trailing positions i with ids[i] == ids[i-p]; None when there is none."""
    a = np.asarray(ids, dtype=np.int64)
    n = len(a)
    for p in range(pmin, min(pmax, n - 1) + 1):
        neq = np.nonzero(a[p:] != a[:-p])[0]
        r = len(a) - p if len(neq) == 0 else len(a) - p - 1 - int(neq[-1])
        if r + p >= minspan:
            return p, r + p
    return None


def cut_of(n, span, offset=CUT_OFFSET, nxt=NEXT):
    """The cut c = loop start + offset; refuses (None) when 256 tokens do not remain."""
    c = n - span + offset
    return c if c + nxt <= n else None


def match_prefix(a, b):
    """Length of the common prefix of two id sequences."""
    m = 0
    for x, y in zip(a, b):
        if x != y:
            break
        m += 1
    return m


def sample_step(logits, seen, u, temperature=0.6, top_k=20, top_p=0.95, presence=1.5):
    """One draw in W1's order: the presence penalty on the logits of tokens already generated, temperature, top-k, softmax over the
    kept, top-p (keep a token while the mass before it is below top_p), then the uniform `u` against the renormalized cumulative."""
    z = np.asarray(logits, dtype=np.float64) - presence * np.asarray(seen, dtype=np.float64)
    z = z / temperature
    k = min(top_k, len(z))
    part = np.argpartition(-z, k - 1)[:k]
    idx = part[np.argsort(-z[part], kind="stable")]
    zs = z[idx]
    w = np.exp(zs - zs.max())
    w = w / w.sum()
    before = np.cumsum(w) - w
    keep = before < top_p
    w = w[keep]
    idx = idx[keep]
    w = w / w.sum()
    cum = np.cumsum(w)
    j = int(np.searchsorted(cum, u, side="right"))
    return int(idx[min(j, len(idx) - 1)])


def bootstrap(values, stat=np.mean, resamples=BOOT_N, seed=BOOT_SEED):
    """95% percentile interval of stat(values) over resamples of the loops."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(v)
    out = np.array([stat(v[rng.integers(0, n, n)]) for _ in range(resamples)])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def per_loop(records):
    """{loop id: (exit share, mean NLL over NEXT, mean NLL over the period, mean match prefix, share leaving within 64)}."""
    out = {}
    for r in records:
        ms = [s["m"] for s in r["samples"]]
        out[r["id"]] = (float(np.mean([m < NEXT for m in ms])), r["nll"], r["nll_period"], float(np.mean(ms)),
                        float(np.mean([m < 64 for m in ms])))
    return out


def gains(model_records, base_records):
    """Per-loop paired gains of a model over the base on the loops both have: ({'E': [...], 'N': [...]}, ids)."""
    pm, pb = per_loop(model_records), per_loop(base_records)
    ids = sorted(set(pm) & set(pb))
    return {"E": np.array([pm[i][0] - pb[i][0] for i in ids]), "N": np.array([pm[i][1] - pb[i][1] for i in ids])}, ids


def calibration(G, metric):
    """14.4 on one metric. G: {model: (gain, lo, hi)}. Returns (passes, [(clause, ok, text)])."""
    ge, g19, gx = G["edited"][0], G["e19"][0], G["x1e-1"][0]
    clauses = [
        ("a", ge > g19 > gx, f"G(edited) {ge:+.4f} > G(e19) {g19:+.4f} > G(x1e-1) {gx:+.4f}"),
        ("b", -0.1 * ge <= gx <= 0.5 * g19, f"{-0.1 * ge:+.4f} <= G(x1e-1) {gx:+.4f} <= {0.5 * g19:+.4f}"),
        ("c", G["edited"][1] > 0, f"lower end of G(edited)'s interval {G['edited'][1]:+.4f} > 0"),
    ]
    return all(c[1] for c in clauses), clauses


def recommend(G):
    """14.5: (R1, R2, note) from {alpha model: gain}, the published edit's gain under 'edited' and KL4."""
    ge = G["edited"]
    have = [a for a in ALPHAS if a in G]
    if len(have) < 2:
        raise E2cError("w4e-records", "fewer than two alpha variants scored")
    reach = [a for a in have if G[a] >= 0.5 * ge]
    if not reach:
        top = sorted(have, key=lambda a: -G[a])
        return top[0], top[1], "no alpha reaches half of the published edit's gain: the two with the largest gain"
    r1 = min(reach, key=lambda a: int(a[1:]))
    rest = [a for a in have if a != r1 and KL4[a] <= KL_GATE]
    if not rest:
        raise E2cError("w4e-records", "no alpha left inside the KL gate for the second arm")
    r2 = max(rest, key=lambda a: G[a])
    return r1, r2, "R1 the smallest alpha with at least half the published gain; R2 the largest gain inside the KL gate"


# ---------------------------------------------------------------- records

def read_jsonl(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError) as e:
        raise E2cError("w4e-records", f"{path}: {e}")


def check_record(r):
    try:
        ok = (isinstance(r["id"], int) and isinstance(r["nll"], float) and isinstance(r["nll_period"], float)
              and np.isfinite(r["nll"]) and np.isfinite(r["nll_period"]) and len(r["samples"]) > 0
              and all(isinstance(s["m"], int) and 0 <= s["m"] <= NEXT for s in r["samples"]))
    except (KeyError, TypeError):
        ok = False
    if not ok:
        raise E2cError("w4e-records", f"a record is malformed: {str(r)[:120]}")


def load_records(path):
    rows = read_jsonl(path)
    for r in rows:
        check_record(r)
    return rows


# ---------------------------------------------------------------- score

def score(directory):
    recs = {}
    try:
        names = sorted(os.listdir(directory))
    except OSError as e:
        raise E2cError("w4e-records", f"{directory}: {e}")
    for fn in names:
        if fn.startswith("records_") and fn.endswith(".jsonl"):
            recs[fn[len("records_"):-len(".jsonl")]] = load_records(os.path.join(directory, fn))
    if "base" not in recs:
        raise E2cError("w4e-records", f"{directory}: no records_base.jsonl")
    base = recs["base"]
    out = {"models": {}, "n_loops": len(base)}
    G = {"E": {}, "N": {}}
    for name, rows in recs.items():
        pl = per_loop(rows)
        row = {"n_loops": len(pl), "E": float(np.mean([v[0] for v in pl.values()])), "nll": float(np.mean([v[1] for v in pl.values()])),
               "nll_period": float(np.mean([v[2] for v in pl.values()])), "match": float(np.mean([v[3] for v in pl.values()])),
               "leave64": float(np.mean([v[4] for v in pl.values()]))}
        if name != "base":
            g, ids = gains(rows, base)
            if len(ids) < 2:
                raise E2cError("w4e-records", f"{name}: fewer than two loops in common with the base")
            for m in ("E", "N"):
                lo, hi = bootstrap(g[m])
                G[m][name] = (float(np.mean(g[m])), lo, hi)
                row["G_" + m] = [G[m][name][0], lo, hi]
        out["models"][name] = row
    out["calibration"] = {}
    chosen = None
    if all(k in G["E"] for k in CALIB):
        for m in ("E", "N"):
            ok, cl = calibration(G[m], m)
            out["calibration"][m] = {"passes": ok, "clauses": [[c[0], bool(c[1]), c[2]] for c in cl]}
            if ok and chosen is None:
                chosen = m
    out["metric"] = chosen
    out["informative"] = chosen is not None
    if chosen and all(a in G[chosen] for a in ALPHAS):
        gm = {k: v[0] for k, v in G[chosen].items()}
        r1, r2, note = recommend(gm)
        out["recommend"] = {"R1": r1, "R2": r2, "note": note}
    return out


def render(res):
    lines = [f"loops: {res['n_loops']}", "", "| model | E | leave<64 | match prefix | NLL (256) | NLL (period) | G_E [95% CI] | G_N [95% CI] |", "|---|---|---|---|---|---|---|---|"]
    for name, r in res["models"].items():
        ge = "" if "G_E" not in r else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*r["G_E"])
        gn = "" if "G_N" not in r else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*r["G_N"])
        lines.append(f"| {name} | {r['E']:.3f} | {r['leave64']:.3f} | {r['match']:.1f} | {r['nll']:.4f} | {r['nll_period']:.4f} | {ge} | {gn} |")
    lines.append("")
    for m, c in res["calibration"].items():
        lines.append(f"calibration on {m}: {'PASSES' if c['passes'] else 'FAILS'}")
        lines += [f"  ({x[0]}) {'ok' if x[1] else 'FAIL'}: {x[2]}" for x in c["clauses"]]
    lines.append("proxy informative: " + (f"yes, on {res['metric']}" if res["informative"] else "NO"))
    if "recommend" in res:
        lines.append("recommend: {R1}, {R2} ({note})".format(**res["recommend"]))
    return "\n".join(lines)


# ---------------------------------------------------------------- select and run (torch)

def _tokenizer(model_dir):
    import transformers as T
    try:
        return T.AutoTokenizer.from_pretrained(model_dir)
    except (OSError, ValueError) as e:
        raise E2cError("w4e-model", f"{model_dir}: {e}")


def prompt_ids(tok, question):
    import w1_behave as B
    text = tok.apply_chat_template([{"role": "user", "content": B.build_question(question)}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def select(base_jsonl, model_dir, gsm8k, want=N_LOOPS):
    import w1_behave as B
    tok = _tokenizer(model_dir)
    rows = read_jsonl(base_jsonl)
    probs = B.load_problems(gsm8k, 200)
    loops = []
    for r in rows:
        if not r.get("hit_limit") or not isinstance(r.get("text"), str):
            continue
        ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        fl = find_loop(ids)
        if fl is None:
            continue
        p, span = fl
        c = cut_of(len(ids), span)
        if c is None:
            continue
        loops.append({"id": r["id"], "period": p, "span": span, "n": len(ids), "start": len(ids) - span, "cut": c,
                      "prompt": prompt_ids(tok, probs[r["id"]]["question"]), "ids": ids[:c + NEXT]})
        if len(loops) == want:
            break
    if not loops:
        raise E2cError("w4e-loops", f"{base_jsonl}: no looping generation qualifies")
    return loops


def load_model(model_dir, device):
    import torch
    import transformers as T
    try:
        try:
            m = T.AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
        except (ValueError, KeyError, OSError, AttributeError):
            m = T.AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.float32)
    except (OSError, ValueError) as e:
        raise E2cError("w4e-model", f"{model_dir}: {e}")
    return m.to(device).eval()


def measure_loop(model, loop, seeds, device):
    import time
    import torch
    T0 = time.perf_counter()
    prompt, ids, c, p = loop["prompt"], loop["ids"], loop["cut"], loop["period"]
    ctx = prompt + ids[:c]
    want = ids[c:c + NEXT]
    with torch.no_grad():
        full = torch.tensor([ctx + want], device=device)
        lg = model(input_ids=full, use_cache=False, logits_to_keep=NEXT + 1).logits[0].float()
        lp = torch.log_softmax(lg[:-1], dim=-1)
        nll_t = -lp[torch.arange(NEXT), torch.tensor(want, device=device)].cpu().numpy()
        T1 = time.perf_counter()
        o = model(input_ids=torch.tensor([ctx], device=device), use_cache=True, logits_to_keep=1)
        cache = o.past_key_values
        T2 = time.perf_counter()
        B = len(seeds)
        cache.reorder_cache(torch.zeros(B, dtype=torch.long, device=device))   # row 0 repeated B times
        logits = o.logits[:, -1].float().repeat(B, 1).cpu().numpy()
        seen = np.zeros((B, logits.shape[1]), dtype=bool)
        seen[:, np.array(ids[:c])] = True
        gens = [torch.Generator().manual_seed(1000 * loop["id"] + k) for k in seeds]
        toks = [[] for _ in seeds]
        for step in range(NEXT):
            nxt = []
            for b in range(B):
                u = float(torch.rand((), generator=gens[b]))
                t = sample_step(logits[b], seen[b], u)
                toks[b].append(t)
                seen[b, t] = True
                nxt.append(t)
            if step + 1 == NEXT:
                break
            if device == "mps" and step % 32 == 31:
                torch.mps.empty_cache()   # without it 256 steps took 210 s instead of 33 s (found by the smoke: the allocator thrashes on the growing key/value cache)
            o = model(input_ids=torch.tensor(nxt, device=device).unsqueeze(1), past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            logits = o.logits[:, -1].float().cpu().numpy()
    T3 = time.perf_counter()
    print(f"  secs nll {T1 - T0:.1f} prefill {T2 - T1:.1f} generate {T3 - T2:.1f}", flush=True)
    samples = []
    for k, t in zip(seeds, toks):
        m = match_prefix(t, want)
        samples.append({"seed": 1000 * loop["id"] + k, "m": m, "toks": t[:min(NEXT, m + 16)]})
    return {"id": loop["id"], "period": p, "cut": c, "nll": float(nll_t.mean()), "nll_period": float(nll_t[:p].mean()), "samples": samples}


def run(name, model_dir, loops_path, out, device, seeds):
    if not os.path.isdir(model_dir):
        raise E2cError("w4e-model", f"{model_dir}: not a directory")
    try:
        with open(loops_path, encoding="utf-8") as f:
            loops = json.load(f)
    except (OSError, ValueError) as e:
        raise E2cError("w4e-loops", f"{loops_path}: {e}")
    try:
        os.makedirs(out, exist_ok=True)
    except OSError as e:
        raise E2cError("w4e-out", f"{out}: {e}")
    path = os.path.join(out, f"records_{name}.jsonl")
    done = {r["id"] for r in load_records(path)} if os.path.exists(path) else set()
    todo = [lp for lp in loops if lp["id"] not in done]
    if not todo:
        print(f"{name}: all {len(loops)} loops done")
        return
    import time
    model = load_model(model_dir, device)
    t0 = time.time()
    for i, lp in enumerate(todo):
        rec = measure_loop(model, lp, seeds, device)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{name} loop {lp['id']} ({i + 1}/{len(todo)}) E={np.mean([s['m'] < NEXT for s in rec['samples']]):.2f} nll={rec['nll']:.3f} "
              f"[{time.time() - t0:.0f}s]", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--base-jsonl", required=True)
    s.add_argument("--model-dir", default=os.path.expanduser("~/models/qwen35-08b-base"))
    s.add_argument("--gsm8k", default=os.path.expanduser("~/models/gsm8k/main/test-00000-of-00001.parquet"))
    s.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--model-dir", required=True)
    r.add_argument("--loops", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--device", default="mps")
    r.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    c = sub.add_parser("score")
    c.add_argument("--dir", required=True)
    c.add_argument("--json")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "select":
            loops = select(a.base_jsonl, a.model_dir, a.gsm8k)
            os.makedirs(a.out, exist_ok=True)
            with open(os.path.join(a.out, "loops.json"), "w", encoding="utf-8") as f:
                json.dump(loops, f)
            meta = [{k: v for k, v in lp.items() if k not in ("ids", "prompt")} for lp in loops]
            with open(os.path.join(a.out, "loops_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=1)
            print(f"{len(loops)} loops; periods {sorted(m['period'] for m in meta)}; cuts {sorted(m['cut'] for m in meta)}")
        elif a.cmd == "run":
            run(a.name, a.model_dir, a.loops, a.out, a.device, tuple(a.seeds))
        else:
            res = score(a.dir)
            print(render(res))
            if a.json:
                with open(a.json, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=1)
    except E2cError as e:
        print(f"error[{e.rule}]: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
