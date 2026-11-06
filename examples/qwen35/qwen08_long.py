#!/usr/bin/env python3
r"""docs/weight-space.md section 14, ticket T10, check C7b: long-context teacher-forced agreement of the Qwen3.5-0.8B engine.

    python3 examples/qwen35/qwen08_long.py seq  --kind gen|text --model base|edited --length N --out seq.json
    python3 examples/qwen35/qwen08_long.py ref  --model base|edited --prec f64|f32 --seq seq.json --out ref_f64.npz [--prefill 64]
    python3 examples/qwen35/qwen08_long.py check --model base|edited --mode f32|f16 --seq seq.json --ref ref_f64.npz [--floor ref_f32.npz] [--tmax N]

* `seq` makes a token sequence.  `gen`: W1 prompt 0 (the T1 check prompt) followed by the model's own free-running continuation under
  `transformers`' float32 code (the W1 sampler: temperature 0.6, top-p 0.95, top-k 20, seed fixed, EOS ignored so the length exists).
  `text`: W1 prompt 1 followed by GSM8K test problems 200.. with their reference solutions (real text) concatenated until `length` tokens.
* `ref` runs `transformers` (text tower, CPU) over the whole sequence, in float64 (`hf64.py`: the reference) or float32 (the shipped
  code: the floor), one token at a time with the cache after a short prefill (transformers continues the delta state only for one-token steps), and
  stores the 20 largest logits (ids and values) of EVERY position.  Nothing is sampled: every position of the sequence is compared.
* `check` runs the engine under `--force` (teacher forcing) over the sequence and compares, at every position, the engine's top-5 with the
  float64 reference's top-5 (ids equal in order, a swap of neighbours whose reference gap is under 1e-3 reported, values within the
  tolerance), and the float32 floor (float32 vs float64) in the same terms.  Prints the worst per length window.

Not in the gate (Metal and the HF stack are needed); `scripts/qwen08-check.sh` does not run it (hours of CPU for the references).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
HOME = os.path.expanduser("~")
MODELS = {"base": os.path.join(HOME, "models", "qwen35-08b-base"), "edited": os.path.join(HOME, "models", "qwen35-08b-edited")}
GSM8K_TEST = os.path.join(HOME, "models", "gsm8k", "main", "test-00000-of-00001.parquet")
GOLD = os.path.join(ROOT, "golden", "hf08")
TOPK = 20
sys.path.insert(0, HERE)


def w1_prompt(index):
    with open(os.path.join(GOLD, "prompts.json"), encoding="utf-8") as f:
        p = json.load(f)
    return next(c["ids"] for c in p["check"] if c["index"] == index)


def say(s):
    print(s, flush=True)


# ------------------------------------------------------------------ seq

def make_seq(a):
    import hf08_ref as H
    import torch
    if a.kind == "gen":
        ids = list(w1_prompt(0))
        H.setup(a.threads)
        torch.manual_seed(a.seed)
        model = H.build_model(MODELS[a.model], "f32")
        t0 = time.time()
        with torch.no_grad():
            o = model(torch.tensor([ids]), use_cache=True, logits_to_keep=1)
            while len(ids) < a.length:
                row = o.logits[0, -1] / 0.6
                v, i = torch.topk(row, 20)
                p = torch.softmax(v, -1)
                cum = torch.cumsum(p, -1)
                keep = (cum - p) < 0.95
                p = p * keep
                t = int(i[torch.multinomial(p / p.sum(), 1)])
                ids.append(t)
                o = model(torch.tensor([[t]]), past_key_values=o.past_key_values, use_cache=True)
                if len(ids) % 200 == 0:
                    say(f"gen {len(ids)} tokens, {time.time() - t0:.0f} s")
        prompt_len = len(w1_prompt(0))
    else:
        import pyarrow.parquet as pq
        import transformers as T
        tok = T.AutoTokenizer.from_pretrained(MODELS[a.model])
        ids = list(w1_prompt(1))
        prompt_len = len(ids)
        rows = pq.read_table(GSM8K_TEST).to_pylist()
        k = 200  # problems 0..199 are the W1 prompts
        while len(ids) < a.length:
            r = rows[k]
            ids += tok("\n\nQuestion: " + r["question"] + "\nAnswer: " + r["answer"], add_special_tokens=False)["input_ids"]
            k += 1
        ids = ids[:a.length]
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"kind": a.kind, "model": a.model, "prompt_len": prompt_len, "ids": ids}, f)
    say(f"wrote {a.out}: {len(ids)} tokens (prompt {prompt_len})")


# ------------------------------------------------------------------ ref

def make_ref(a):
    import hf08_ref as H
    import torch
    with open(a.seq, encoding="utf-8") as f:
        ids = json.load(f)["ids"]
    H.setup(a.threads)
    model = H.build_model(MODELS[a.model], a.prec)
    dt = torch.float64 if a.prec == "f64" else torch.float32
    top_i = np.zeros((len(ids), TOPK), dtype=np.int32)
    top_v = np.zeros((len(ids), TOPK), dtype=np.float64)
    cache = None
    t0 = time.time()
    # One forward over the first `prefill` tokens (no earlier state), then one token at a time with the cache: transformers' gated-delta
    # layer only continues from a stored state when the new part is a single token (a longer continuation silently ignores the state:
    # measured, a 512-token second chunk is wrong by up to 8 in a logit). The lm_head is applied to blocks of hidden states.
    hid = []
    flushed = 0

    def flush(upto):
        nonlocal flushed
        h = torch.cat(hid, dim=0)
        lg = h @ model.lm_head.weight.T
        v, i = torch.topk(lg, TOPK, dim=-1)
        top_i[flushed:upto] = i.numpy()
        top_v[flushed:upto] = v.double().numpy()
        flushed = upto
        hid.clear()

    with torch.no_grad():
        p = min(a.prefill, len(ids))
        o = model.model(input_ids=torch.tensor([ids[:p]]), use_cache=True)
        cache = o.past_key_values
        hid.append(o.last_hidden_state[0].to(dt))
        while p < len(ids):
            o = model.model(input_ids=torch.tensor([[ids[p]]]), past_key_values=cache, use_cache=True)
            hid.append(o.last_hidden_state[0].to(dt))
            p += 1
            if p % 256 == 0:
                flush(p)
                say(f"ref {a.prec} {p}/{len(ids)} positions, {time.time() - t0:.0f} s")
        flush(p)
    np.savez(a.out, ids=top_i, vals=top_v, prec=a.prec)
    say(f"wrote {a.out}")


# ------------------------------------------------------------------ check

def parse_steps(out):
    """The engine's `step <j> <chosen> <id>:<f32 bits> x5` lines -> ids [n, 5] and values [n, 5] (float64 of the f32)."""
    ids, vals = [], []
    for line in out.split("\n"):
        w = line.split(" ")
        if w[0] == "step":
            pr = [x.split(":") for x in w[3:8]]
            ids.append([int(p[0]) for p in pr])
            vals.append([int(p[1]) for p in pr])
    return np.array(ids, dtype=np.int64), np.array(vals, dtype=np.uint32).view(np.float32).astype(np.float64)


def compare(ids_a, vals_a, ids_b, vals_b, tie=1e-3):
    """Per position: (worst |value difference| over the 5 places, ids equal in order, equal up to a swap of neighbours whose
    reference (b) gap is below `tie`)."""
    n = len(ids_a)
    err = np.abs(vals_a[:, :5] - vals_b[:, :5]).max(axis=1)
    same = (ids_a[:, :5] == ids_b[:, :5]).all(axis=1)
    soft = same.copy()
    for k in np.nonzero(~same)[0]:
        d = np.nonzero(ids_a[k, :5] != ids_b[k, :5])[0]
        gaps_ok = all(
            (j == 0 or abs(vals_b[k, j] - vals_b[k, j - 1]) < tie) or (j == 4 or abs(vals_b[k, j + 1] - vals_b[k, j]) < tie) for j in d)
        soft[k] = gaps_ok and set(ids_a[k, :5]) == set(ids_b[k, :5]) or (set(ids_a[k, :4]) == set(ids_b[k, :4]) and gaps_ok)
    return err, same, soft


def windows(n, marks):
    lo = 0
    for hi in marks + [n]:
        if hi > n:
            hi = n
        if hi > lo:
            yield lo, hi
        lo = hi


def make_check(a):
    with open(a.seq, encoding="utf-8") as f:
        seq = json.load(f)
    ids = seq["ids"]
    plen = seq["prompt_len"]
    tmax = a.tmax
    ref = np.load(a.ref)
    exe = os.path.join(ROOT, "build", "cancho-gpu-run08")
    # Teacher forcing: the prompt is ids[:plen]; the forced ids are ids[plen:]; step j predicts ids[plen + j]; the top after the
    # prompt predicts ids[plen], so the step lines cover positions plen - 1 .. len - 2 (the last forced id is fed to no one: a step
    # predicts the token being forced, and the final forced token's own successor is not asked).
    forced = ids[plen:]
    cmd = [exe, MODELS[a.model], os.path.join(ROOT, "tests", "chx"), a.mode, str(tmax), ",".join(map(str, ids[:plen])), "--force",
           ",".join(map(str, forced)), "--timing"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if r.returncode != 0:
        say("runner failed: " + r.stderr[-2000:])
        return 1
    with open(a.out_raw, "w", encoding="utf-8") if a.out_raw else open(os.devnull, "w") as f:
        f.write(r.stdout)
    sids, svals = parse_steps(r.stdout)
    tm = [l for l in r.stdout.split("\n") if l.startswith("timing:")]
    # step j predicts token at index plen + j, i.e. the logits at position plen + j - 1.
    n = len(sids)
    pos = np.arange(plen - 1, plen - 1 + n)
    rid, rv = ref["ids"][pos], ref["vals"][pos]
    err, same, soft = compare(sids, svals, rid, rv)
    say(f"{a.model} {a.mode}: {n} positions {pos[0]}..{pos[-1]}, wall {wall:.0f} s; {tm[0] if tm else ''}")
    ok = True
    marks = [p for p in (1024, 2048, 2046, 4096, 6144, 8192, 12288, 16384) if p < pos[-1]]
    marks = sorted(set(marks))
    say(f"{'positions':>16} {'ids equal':>10} {'up to tie':>10} {'worst |dv|':>11} {'mean |dv|':>11}")
    for lo, hi in windows(n, [m - pos[0] for m in marks]):
        sl = slice(lo, hi)
        say(f"{pos[lo]:>7}..{pos[hi - 1]:<7} {int(same[sl].sum()):>5}/{hi - lo:<5} {int(soft[sl].sum()):>5}/{hi - lo:<5} "
            f"{err[sl].max():>11.2e} {err[sl].mean():>11.2e}")
    say(f"overall: ids equal {int(same.sum())}/{n}; up to a tie swap {int(soft.sum())}/{n}; worst {err.max():.3e} at position {pos[err.argmax()]}")
    if a.floor:
        fl = np.load(a.floor)
        ferr, fsame, fsoft = compare(fl["ids"][pos], fl["vals"][pos], rid, rv)
        say(f"float32 floor (transformers float32 vs float64 reference): ids equal {int(fsame.sum())}/{n}; up to tie {int(fsoft.sum())}/{n}; "
            f"worst {ferr.max():.3e} at {pos[ferr.argmax()]}")
        for lo, hi in windows(n, [m - pos[0] for m in marks]):
            say(f"  floor {pos[lo]:>7}..{pos[hi - 1]:<7} worst {ferr[lo:hi].max():.2e}  engine worst {err[lo:hi].max():.2e}")
        # the engine against the float32 reference too
        eerr, esame, esoft = compare(sids, svals, fl["ids"][pos], fl["vals"][pos])
        say(f"engine vs float32 reference: ids equal {int(esame.sum())}/{n}; worst {eerr.max():.3e}")
    bad = np.nonzero(~soft)[0]
    if len(bad):
        say(f"{len(bad)} positions with ids differing beyond a tie swap; first {pos[bad[:10]].tolist()}")
        ok = False
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seq")
    s.add_argument("--kind", choices=["gen", "text"], required=True)
    s.add_argument("--model", default="base")
    s.add_argument("--length", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--threads", type=int, default=0)
    s.add_argument("--out", required=True)
    r = sub.add_parser("ref")
    r.add_argument("--model", default="base")
    r.add_argument("--prec", choices=["f64", "f32"], required=True)
    r.add_argument("--seq", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--prefill", type=int, default=64)
    r.add_argument("--threads", type=int, default=0)
    c = sub.add_parser("check")
    c.add_argument("--model", default="base")
    c.add_argument("--mode", choices=["f32", "f16"], required=True)
    c.add_argument("--seq", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--floor")
    c.add_argument("--tmax", type=int, default=8704)
    c.add_argument("--out-raw", dest="out_raw")
    a = ap.parse_args()
    if a.cmd == "seq":
        make_seq(a)
    elif a.cmd == "ref":
        make_ref(a)
    else:
        sys.exit(make_check(a))


if __name__ == "__main__":
    main()
