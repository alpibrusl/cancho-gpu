"""docs/weight-space.md section 13 (W1): the behavioural check. Grading, the statistic and the verdict need numpy only
(and are tested, tests/w1.py); generation needs torch + transformers, or mlx-lm, and a free GPU.

The protocol is section 6's, fixed before any run: GSM8K test problems, one sample each, thinking on, temperature 0.6,
top-p 0.95, top-k 20, presence penalty 1.5, the prompt of the authors' evaluate.py (the question, then "Please reason step by
step, and put your final answer within \\boxed{}."), graded by the last \\boxed{} answer."""
import json
import os
import re

import numpy as np

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}
BOOT_SEED = 20261104
BOOT_N = 10_000
ACC_RULE = "edited accuracy exceeds the base's by more than the half-width of the base's 95% interval"
L_RULE_POINTS = 20.0


class BehaveError(Exception):
    """A problem file or a result that cannot be used; the message says why."""


def build_question(q):
    return q + SUFFIX


def last_boxed(text):
    """The content of the last \\boxed{...} in `text` (braces balanced), or None."""
    i = text.rfind("\\boxed{")
    while i >= 0:
        j, depth = i + len("\\boxed{"), 1
        k = j
        while k < len(text) and depth:
            depth += (text[k] == "{") - (text[k] == "}")
            k += 1
        if depth == 0:
            return text[j:k - 1]
        i = text.rfind("\\boxed{", 0, i)   # an unclosed one (cut off): try the one before it
    return None


def _number(s):
    s = re.sub(r"\\text\{[^}]*\}|\\[,!; ]|\\\$|[$ ,]|\\%|%", "", s.strip())
    s = re.sub(r"^\\(?:d?frac)\{(-?[\d.]+)\}\{(-?[\d.]+)\}$", lambda m: repr(float(m.group(1)) / float(m.group(2))) if float(m.group(2)) else "x", s)
    try:
        return float(s)
    except ValueError:
        return None


def grade(text, gold):
    """(prediction or None, correct). Correct iff the last \\boxed{} is the same number as `gold` (GSM8K answers are
    integers). Not math-verify: a boxed expression that is not a plain number or a fraction is wrong here."""
    pred = last_boxed(text)
    if pred is None:
        return None, False
    a, b = _number(pred), _number(gold)
    return pred, a is not None and b is not None and abs(a - b) < 1e-6


def load_problems(path, n):
    """The first n problems of a GSM8K file (.jsonl with question/answer, or the dataset's parquet)."""
    if not os.path.isfile(path):
        raise BehaveError(f"{path}: no such file")
    rows = []
    try:
        if path.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
            except ImportError:
                raise BehaveError("reading parquet needs pyarrow; give a .jsonl instead")
            t = pq.read_table(path).to_pydict()
            rows = [{"question": q, "answer": a} for q, a in zip(t.get("question", []), t.get("answer", []))]
        else:
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError) as e:
        raise BehaveError(f"{path}: {e}")
    out = []
    for i, r in enumerate(rows[:n]):
        if not isinstance(r, dict) or not isinstance(r.get("question"), str) or not isinstance(r.get("answer"), str):
            raise BehaveError(f"{path}: row {i} has no question and answer strings")
        out.append({"id": i, "question": r["question"], "gold": r["answer"].split("####")[-1].strip().replace(",", "")})
    if len(out) < n:
        raise BehaveError(f"{path}: {len(out)} problems, {n} asked for")
    return out


def bootstrap(values, resamples=BOOT_N, seed=BOOT_SEED):
    """(mean, lo, hi): mean of 0/1 values with a 95% percentile bootstrap interval."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        raise BehaveError("no values")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(resamples, v.size))].mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(base, edited):
    """The pre-registered rule. `base`, `edited`: lists of records with `correct` and `hit_limit` over the same problems.
    Reproduces iff the accuracy gain exceeds the half-width of the base's interval AND the token-limit rate falls by more
    than 20 points; partial if exactly one holds; else not reproduced."""
    if not base or not edited or len(base) != len(edited):
        raise BehaveError("the two runs must cover the same non-empty set of problems")
    ba, ea = bootstrap([r["correct"] for r in base]), bootstrap([r["correct"] for r in edited])
    bl, el = bootstrap([r["hit_limit"] for r in base]), bootstrap([r["hit_limit"] for r in edited])
    half = (ba[2] - ba[1]) / 2
    gain = ea[0] - ba[0]
    drop = round((bl[0] - el[0]) * 100, 9)
    acc_ok, l_ok = gain > half, drop > L_RULE_POINTS
    name = "reproduces" if acc_ok and l_ok else "partial" if acc_ok or l_ok else "not reproduced at this protocol"
    return {"n": len(base), "base_accuracy": ba, "edited_accuracy": ea, "base_limit_rate": bl, "edited_limit_rate": el,
            "accuracy_gain": gain, "base_half_width": half, "accuracy_rule_holds": bool(acc_ok),
            "limit_drop_points": drop, "limit_rule_holds": bool(l_ok), "verdict": name}


def render_verdict(v, label=""):
    f = lambda t: f"{100 * t[0]:.1f}% [{100 * t[1]:.1f}, {100 * t[2]:.1f}]"
    return "\n".join([
        f"W1 verdict{(' ' + label) if label else ''}: {v['verdict']}  (n = {v['n']})",
        f"  accuracy    base {f(v['base_accuracy'])}  edited {f(v['edited_accuracy'])}  gain {100 * v['accuracy_gain']:+.1f} points "
        f"(rule: more than the base's half-width {100 * v['base_half_width']:.1f}: {'holds' if v['accuracy_rule_holds'] else 'fails'})",
        f"  token limit base {f(v['base_limit_rate'])}  edited {f(v['edited_limit_rate'])}  drop {v['limit_drop_points']:+.1f} points "
        f"(rule: more than {L_RULE_POINTS:.0f}: {'holds' if v['limit_rule_holds'] else 'fails'})"])


def run_behaviour(gen, problems, model_tag, max_new, seed, batch, log_path=None):
    """Generate for every problem, `batch` at a time, appending each finished record to `log_path` (a jsonl; a rerun skips the
    ids already in it). `gen(prompts, max_new, seed) -> [(text, n_tokens, hit_limit)]`. Returns the records in problem order."""
    done = {}
    if log_path and os.path.isfile(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue   # a line cut off by a crash: that problem is run again
                if r.get("model") == model_tag and r.get("max_new") == max_new and isinstance(r.get("id"), int):
                    done[r["id"]] = r
    todo = [p for p in problems if p["id"] not in done]
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        outs = gen([build_question(p["question"]) for p in chunk], max_new, seed + chunk[0]["id"])
        if len(outs) != len(chunk):
            raise BehaveError(f"the generator returned {len(outs)} answers for {len(chunk)} prompts")
        for p, (text, n_tok, hit) in zip(chunk, outs):
            pred, ok = grade(text, p["gold"])
            r = {"model": model_tag, "id": p["id"], "max_new": max_new, "gold": p["gold"], "pred": pred, "correct": ok,
                 "n_tokens": int(n_tok), "hit_limit": bool(hit), "text": text}
            done[p["id"]] = r
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r) + "\n")
    return [done[p["id"]] for p in problems]


# ---- generation backends: need the GPU, not run by any test ----

def transformers_generator(model_dir, device):
    """A `gen` over transformers: the model in bfloat16 on `device` (mps, cuda or cpu). Untested before the weights exist."""
    import torch
    import transformers as T

    tok = T.AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    try:
        model = T.AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    except (ValueError, KeyError, OSError, AttributeError):  # Qwen3_5Config (the multimodal wrapper) has no vocab_size: the text-only class cannot be built from it
        model = T.AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16)
    model = model.to(device).eval()
    eos = {tok.eos_token_id} | set(np.atleast_1d(getattr(model.generation_config, "eos_token_id", None) or []).tolist())
    eos.discard(None)

    class Presence(T.LogitsProcessor):
        def __init__(self, prompt_len, penalty):
            self.prompt_len, self.penalty = prompt_len, penalty

        def __call__(self, input_ids, scores):
            g = input_ids[:, self.prompt_len:]
            if g.shape[1] == 0:
                return scores
            seen = torch.zeros_like(scores, dtype=torch.bool)
            seen.scatter_(1, g, True)
            return scores - self.penalty * seen

    def gen(questions, max_new, seed):
        prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=True) for q in questions]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        n0 = enc["input_ids"].shape[1]
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=SAMPLING["temperature"],
                                 top_p=SAMPLING["top_p"], top_k=SAMPLING["top_k"],
                                 logits_processor=T.LogitsProcessorList([Presence(n0, SAMPLING["presence_penalty"])]))
        res = []
        for row in out[:, n0:].tolist():
            cut = next((i for i, t in enumerate(row) if t in eos), None)
            n = len(row) if cut is None else cut + 1
            res.append((tok.decode(row[:n], skip_special_tokens=True), n, cut is None and n >= max_new))
        return res

    return gen


def mlx_generator(model_dir):
    """A `gen` over mlx-lm, one prompt at a time. Untested before the weights exist; the presence penalty covers every
    generated token (the context size is set to the cap), as vLLM's does."""
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    model, tok = load(model_dir)
    sampler = make_sampler(temp=SAMPLING["temperature"], top_p=SAMPLING["top_p"], top_k=SAMPLING["top_k"])

    def gen(questions, max_new, seed):
        res = []
        for i, q in enumerate(questions):
            mx.random.seed(seed + i)
            prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=True)
            procs = make_logits_processors(presence_penalty=SAMPLING["presence_penalty"], presence_context_size=max_new)
            text, n, reason = "", 0, None
            for r in stream_generate(model, tok, prompt=prompt, max_tokens=max_new, sampler=sampler, logits_processors=procs):
                text += r.text
                n = r.generation_tokens
                reason = r.finish_reason
            res.append((text, n, reason == "length"))
        return res

    return gen
