#!/usr/bin/env python3
"""The M8 training corpus (docs/roadmap-models.md section 7): cancho sources, documents, instruction pairs and compile-fix pairs, with a contamination check against the benchmark.

    CANCHO=<compiler> python3 examples/qwen/m8_corpus.py OUT_DIR [--limit-pairs N] [--seed 20261103]

Deterministic: the files are read from the `main` commit of each repository (recorded in MANIFEST.json), sorted by path; every random choice comes from `--seed`. Writes to OUT_DIR (outside git; the script,
the sha256 of every file and the counts are what the repository keeps):
  raw_cho.jsonl raw_md.jsonl raw_rs.jsonl   continued-pretraining chunks of at most CHUNK Qwen tokens, cut at line boundaries, each headed by `// <repo>/<path>`
  pairs_instr.jsonl pairs_fix.jsonl         {prompt, reply} in cancho-code's format (m8_pairs.py); the compiler accepted every reply, and refused every broken prompt's file
  excluded.json, contamination.json, MANIFEST.json, stats.json
Every record has `split`: train | heldout_repos (whole repositories fixed before training: HELD_REPOS) | heldout_files (about 5% of the other files, by hash of the path).

Hard exclusions: everything under cancho-code/eval/ (references, oracles, plans, briefs, results, runs, services); a file that implements one of the benchmark libraries (STRONG names below).
Contamination: the 8-grams of Qwen tokens of every benchmark artefact (briefs, plans, references, oracles, audits, tests) against every chunk and pair; an 8-gram that occurs in COMMON_MIN or more different
documents is an idiom (`import std.buffer;`, the `main` boilerplate) and does not count; a chunk with more than OVERLAP_FRAC of its 8-grams in the benchmark set, or a run of RUN_MAX consecutive ones, is dropped."""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_pairs as P  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))     # the directory that holds the repositories
REPOS = ["cancho", "cancho-code", "cancho-forge", "cancho-hdl", "cancho-ui", "cancho-web", "cancho-www", "cancho-oci", "cancho-cache", "cancho-afm", "cancho-robot", "cancho-opt", "cancho-gpu", "lex-gpu"]
HELD_REPOS = ["cancho-hdl", "cancho-opt", "cancho-cache"]
TOKENIZER = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots")
CHUNK = 512
NGRAM, COMMON_MIN, OVERLAP_FRAC, RUN_MAX = 8, 20, 0.15, 24
DROP_FAMILIES = ("brief", "plan", "reference", "oracle", "tests", "facts", "item-other", "results")      # eval-other (services, libs, probes, multi) is excluded as a source, not counted as overlap
MAX_FILE = 120_000
SKIP_PARTS = ("/vendor/", "/journal/", "/blobs/", "/runs/", "/target/", "/build/", "node_modules/", "tests/vectors/", "/.git/", "package-lock")
ITEMS = ["basics", "tokenbucket", "roman", "base64", "duration", "editdistance", "sortedmap", "jsonptr", "expr", "glob", "semver", "ini", "csv"]
# names that mean "this file implements a benchmark library": the whole file is excluded. The generic ones (gcd, clamp, ...) are left to the n-gram check, because std/math.cho has many other functions.
STRONG = {"base64": r"base64", "semver": r"semver", "csv": r"\bcsv_|_csv\b", "glob": r"\bglob_|_glob\b|glob_match", "ini": r"\bini_|_ini\b", "jsonptr": r"json_?pointer|jsonptr",
          "roman": r"\broman\b|roman_", "duration": r"duration_(parse|format)|parse_duration|format_duration", "editdistance": r"levenshtein|edit_distance|editdistance",
          "sortedmap": r"sorted_?map", "tokenbucket": r"token_?bucket", "expr": r"parse_expr|eval_expr|tokenize_expr|fn evaluate\b",
          "basics": r"fn (gcd|lcm|isqrt|pow_mod|is_leap_year)\b"}
GENERIC_ONLY = {"basics"}      # reported, not excluded as a whole file


def git(repo, *args, raw=False):
    p = subprocess.run(["git", "-C", os.path.join(ROOT, repo)] + list(args), capture_output=True)
    if p.returncode != 0:
        raise SystemExit("git %s failed in %s: %s" % (args, repo, p.stderr.decode()[:200]))
    return p.stdout if raw else p.stdout.decode()


def read_blobs(repo, paths):
    """{path: bytes} for the paths at the `main` commit, by one `git cat-file --batch`."""
    p = subprocess.run(["git", "-C", os.path.join(ROOT, repo), "cat-file", "--batch"], input="".join("main:%s\n" % x for x in paths).encode(), capture_output=True)
    out, buf, i = {}, p.stdout, 0
    for x in paths:
        nl = buf.index(b"\n", i)
        head = buf[i:nl].split()
        if len(head) < 3 or head[1] != b"blob":
            i = nl + 1
            continue
        n = int(head[2])
        out[x] = buf[nl + 1:nl + 1 + n]
        i = nl + 1 + n + 1
    return out


def kind_of(path):
    if path.endswith((".cho", ".chx")):
        return "cho"
    if path.endswith(".md"):
        return "md"
    if path.endswith(".rs"):
        return "rs"
    return None


def tokenizer():
    from tokenizers import Tokenizer
    d = sorted(os.listdir(TOKENIZER))[0]
    return Tokenizer.from_file(os.path.join(TOKENIZER, d, "tokenizer.json"))


def chunks(tok, header, text):
    """[(chunk text, token count)] of at most CHUNK tokens each, cut at line starts when possible; every chunk begins with `header`."""
    out = []
    lines = text.split("\n")
    cur, cur_n = [], 0
    h = header + "\n"
    hn = len(tok.encode(h).ids)
    for ln in lines:
        n = len(tok.encode(ln + "\n").ids)
        if n + hn > CHUNK:      # one line longer than a chunk: cut it by tokens
            ids = tok.encode(ln).ids
            step = CHUNK - hn - 8
            if cur:
                out.append(h + "\n".join(cur))
                cur, cur_n = [], 0
            for k in range(0, len(ids), step):
                out.append(h + tok.decode(ids[k:k + step]))
            continue
        if cur_n + n + hn > CHUNK and cur:
            out.append(h + "\n".join(cur))
            cur, cur_n = [], 0
        cur.append(ln)
        cur_n += n
    if cur and any(c.strip() for c in cur):
        out.append(h + "\n".join(cur))
    return [(c, len(tok.encode(c).ids)) for c in out if c.strip()]


def grams(ids):
    return [hash(tuple(ids[i:i + NGRAM])) for i in range(max(0, len(ids) - NGRAM + 1))]


def split_of(repo, path):
    if repo in HELD_REPOS:
        return "heldout_repos"
    return "heldout_files" if int(hashlib.sha1(("%s/%s" % (repo, path)).encode()).hexdigest(), 16) % 20 == 0 else "train"


def pair_work(arg):
    """One instruction candidate: (the instruction record and its fix records, None) or (None, why it was dropped). Runs in a worker process with its own scratch directory."""
    cancho, seed, (repo, path, split, text, name, sig, desc, body) = arg
    with tempfile.TemporaryDirectory() as wd:
        fn = P.strip_pub(body)
        sc = P.scaffold_for(fn, text)
        reply = sc + "\n" + fn + "\n"
        refused, _ = P.check(cancho, reply, wd)
        if refused is None:
            return None, "no answer"
        if refused.get("refused"):
            return None, "not standalone"
        task = P.task_block(name, name, sig, desc)
        src = "%s/%s" % (repo, path)
        rec = {"id": "%s::%s" % (src, name), "kind": "instr", "split": split, "source": src, "prompt": P.compose(sc, task, ""), "reply": reply}
        fixes = []
        rng = random.Random("%s/%s" % (seed, rec["id"]))
        for prompt2, good, op, tags in P.fix_pairs(cancho, name, sc, fn, task, rng, wd):
            fixes.append({"id": "%s::%s" % (rec["id"], op), "kind": "fix", "split": split, "source": src, "prompt": prompt2, "reply": good, "op": op, "tags": tags})
        return (rec, fixes), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--limit-pairs", type=int, default=0, help="at most this many instruction candidates are checked (0: all)")
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cancho = os.environ.get("CANCHO")
    if not cancho:
        sys.exit("m8_corpus: set CANCHO to the compiler")
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    tok = tokenizer()
    manifest = {"seed": a.seed, "repos": {}, "chunk_tokens": CHUNK, "held_repos": HELD_REPOS, "compiler": subprocess.run([cancho, "--version"], capture_output=True, text=True).stdout.strip(),
                "ngram": NGRAM, "common_min": COMMON_MIN, "overlap_frac": OVERLAP_FRAC, "run_max": RUN_MAX}

    # 1. the benchmark artefacts, by family, from cancho-code's main
    bench_paths = [p for p in git("cancho-code", "ls-tree", "-r", "--name-only", "main").split("\n") if p.startswith("eval/") and p.endswith((".txt", ".json", ".py", ".cho", ".md"))]
    blobs = read_blobs("cancho-code", bench_paths)

    def family(p):
        b = os.path.basename(p)
        parts = p.split("/")
        if len(parts) == 3 and parts[1] in ITEMS:
            return {"brief.txt": "brief", "plan.json": "plan", "reference.cho": "reference", "oracle.py": "oracle", "facts.json": "facts"}.get(b, "item-other")
        if p.startswith("eval/reference/"):
            return "reference" if not re.search(r"test", b) else "tests"
        if re.match(r"eval/audit_(%s)\.py" % "|".join(ITEMS), p):
            return "oracle"
        if p.startswith("eval/results/"):
            return "results"
        return "eval-other"

    bench_grams = defaultdict(set)      # family -> hashes
    all_bench = set()
    fam_tokens = Counter()
    for p, b in blobs.items():
        if len(b) > 3_000_000:
            continue
        ids = tok.encode(b.decode("utf-8", "replace")).ids
        f = family(p)
        fam_tokens[f] += len(ids)
        for g in grams(ids):
            bench_grams[f].add(g)
            if f in DROP_FAMILIES:
                all_bench.add(g)
    manifest["benchmark"] = {"files": len(blobs), "tokens_by_family": dict(fam_tokens), "ngrams": len(all_bench)}

    # 2. the candidate documents
    docs, excluded = [], {"eval": 0, "skip_path": 0, "too_large": 0, "binary": 0, "duplicate": 0, "implements": []}
    seen = {}
    for repo in REPOS:
        manifest["repos"][repo] = git(repo, "rev-parse", "main").strip()
        paths = sorted(p for p in git(repo, "ls-tree", "-r", "--name-only", "main").split("\n") if p and kind_of(p))
        ok = []
        for p in paths:
            if repo == "cancho-code" and p.startswith("eval/"):
                excluded["eval"] += 1
            elif any(s in "/" + p for s in SKIP_PARTS):
                excluded["skip_path"] += 1
            else:
                ok.append(p)
        for p, b in read_blobs(repo, ok).items():
            if len(b) > MAX_FILE:
                excluded["too_large"] += 1
                continue
            try:
                t = b.decode("utf-8")
            except UnicodeDecodeError:
                excluded["binary"] += 1
                continue
            h = hashlib.sha256(b).hexdigest()
            if h in seen:
                excluded["duplicate"] += 1
                continue
            seen[h] = "%s/%s" % (repo, p)
            hit = [it for it in ITEMS if kind_of(p) == "cho" and it not in GENERIC_ONLY and re.search(STRONG[it], t + "\n" + p)]
            gen = [it for it in GENERIC_ONLY if kind_of(p) == "cho" and re.search(STRONG[it], t)]
            if hit:
                excluded["implements"].append({"file": "%s/%s" % (repo, p), "items": hit})
                continue
            docs.append({"repo": repo, "path": p, "kind": kind_of(p), "text": t, "split": split_of(repo, p), "generic_hit": gen})

    # 3. chunks, with the contamination count
    recs = []
    for d in docs:
        if d["kind"] == "rs" and not (d["repo"] == "cancho" and d["path"].startswith("crates/") or d["repo"] in ("cancho-ui", "lex-gpu")):
            continue
        hdr = ("// %s/%s" if d["kind"] != "md" else "<!-- %s/%s -->") % (d["repo"], d["path"])
        for k, (c, n) in enumerate(chunks(tok, hdr, d["text"])):
            recs.append({"id": "%s/%s#%d" % (d["repo"], d["path"], k), "kind": d["kind"], "split": d["split"], "source": "%s/%s" % (d["repo"], d["path"]), "text": c, "ntok": n})

    pairs_i, pairs_f = [], []
    cands = []
    for d in docs:
        if d["kind"] != "cho":
            continue
        for name, sig, desc, body in P.functions(d["text"]):
            if len(desc) <= 700:
                cands.append((d["repo"], d["path"], d["split"], d["text"], name, sig, desc, body))
    if a.limit_pairs:
        rng.shuffle(cands)
        cands = cands[:a.limit_pairs]
    manifest["instruction_candidates"] = len(cands)
    failed = Counter()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res, why in ex.map(pair_work, [(cancho, a.seed, c) for c in cands], chunksize=8):
            if res is None:
                failed[why] += 1
            else:
                pairs_i.append(res[0])
                pairs_f.extend(res[1])
    manifest["instruction_dropped"] = dict(failed)

    # 4. lengths and the contamination filter
    for r in pairs_i + pairs_f:
        r["ntok"] = len(tok.encode(r["prompt"] + r["reply"]).ids) + 12
    pairs_i = [r for r in pairs_i if r["ntok"] <= 1024]
    pairs_f = [r for r in pairs_f if r["ntok"] <= 1024]
    items = recs + pairs_i + pairs_f
    gram_cache = []
    for r in items:
        ids = tok.encode(r.get("text") or (r["prompt"] + "\n" + r["reply"])).ids
        gs = grams(ids)
        gram_cache.append(gs)
    docs_of = defaultdict(set)
    for r, gs in zip(items, gram_cache):
        for g in set(gs):
            docs_of[g].add(r["source"])
    report = {"dropped": Counter(), "dropped_tokens": 0, "by_family": Counter(), "examples": []}
    kept = []
    for r, gs in zip(items, gram_cache):
        hits = [i for i, g in enumerate(gs) if g in all_bench and len(docs_of[g]) < COMMON_MIN]
        run = best = 0
        hs = set(hits)
        for i in range(len(gs)):
            run = run + 1 if i in hs else 0
            best = max(best, run)
        frac = len(hits) / max(1, len(gs))
        if frac > OVERLAP_FRAC or best >= RUN_MAX:
            fams = [f for f, s in bench_grams.items() if any(gs[i] in s for i in hits)]
            report["dropped"][r["kind"]] += 1
            report["dropped_tokens"] += r["ntok"]
            for f in fams:
                report["by_family"][f] += 1
            if len(report["examples"]) < 40:
                report["examples"].append({"id": r["id"], "kind": r["kind"], "frac": round(frac, 3), "longest_run": best, "families": fams})
            continue
        r["overlap_frac"] = round(frac, 4)
        kept.append(r)
    report["dropped"] = dict(report["dropped"])
    report["by_family"] = dict(report["by_family"])
    report["generic_name_hits"] = sorted({"%s/%s" % (d["repo"], d["path"]) + " " + ",".join(d["generic_hit"]) for d in docs if d["generic_hit"]})[:100]
    report["items_before"], report["items_after"] = len(items), len(kept)

    # 5. write
    files = {}
    for name, sel in (("raw_cho", lambda r: r["kind"] == "cho"), ("raw_md", lambda r: r["kind"] == "md"), ("raw_rs", lambda r: r["kind"] == "rs"),
                      ("pairs_instr", lambda r: r["kind"] == "instr"), ("pairs_fix", lambda r: r["kind"] == "fix")):
        rows = [r for r in kept if sel(r)]
        path = os.path.join(a.out, name + ".jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        files[name + ".jsonl"] = {"sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "records": len(rows), "tokens": sum(r["ntok"] for r in rows),
                                  "by_split": {s: {"records": sum(1 for r in rows if r["split"] == s), "tokens": sum(r["ntok"] for r in rows if r["split"] == s)} for s in ("train", "heldout_repos", "heldout_files")}}
    manifest["files"] = files
    fixes = [r for r in kept if r["kind"] == "fix"]
    manifest["fix_operators"] = dict(Counter(r["op"] for r in fixes))
    manifest["fix_rule_tags"] = dict(Counter(t for r in fixes for t in r["tags"]).most_common(40))
    json.dump(manifest, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1, sort_keys=True)
    json.dump(excluded, open(os.path.join(a.out, "excluded.json"), "w"), indent=1)
    json.dump(report, open(os.path.join(a.out, "contamination.json"), "w"), indent=1)
    print(json.dumps({"files": files, "excluded": {k: (v if not isinstance(v, list) else len(v)) for k, v in excluded.items()}, "contamination": {k: report[k] for k in ("dropped", "dropped_tokens", "by_family", "items_before", "items_after")}}, indent=1))


if __name__ == "__main__":
    main()
