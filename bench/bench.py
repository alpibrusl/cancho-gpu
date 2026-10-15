#!/usr/bin/env python3
"""Benchmark cancho-gpu-run against the other ways of running the same
checkpoints on this machine (docs/benchmark.md says what is measured and why).

    bench/bench.py [--models qwen2.5-0.5b,qwen3.8-27b] [--runs 5] [--new 128]
                   [--json results.json] [--markdown docs/benchmark.md] [--quick]

For each model and each runtime that can run it: one warm-up run (dropped), then
`runs` runs of the same raw prompt (no chat template), greedy, `new` tokens.
Per run it records the prefill rate, the decode rate and the peak memory the
runtime reports, and the output text; per runtime the load time. Runtimes are
measured one at a time with the others idle (Ollama's models are unloaded
first), because a loaded 27B changes everyone's numbers.

Needs a Mac with Metal, the checkpoints in the Hugging Face cache and, for the
Ollama rows, the model in Ollama; each missing runtime is skipped and said so.
"""
import argparse
import glob
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = os.path.expanduser("~/.cache/huggingface/hub")

# Long enough that prefill is more than a rounding error, one line, plain text.
PROMPT = ("The history of computing is a history of abstraction. Early programmers wrote machine "
          "code by hand, then assemblers gave names to instructions, then compilers let people "
          "describe what they wanted rather than how to get it. Today a single line of code can")


def snapshot(repo):
    found = glob.glob(os.path.join(HF, f"models--{repo.replace('/', '--')}", "snapshots", "*"))
    return found[0] if found else None


MODELS = {
    "qwen2.5-0.5b": {
        "title": "Qwen2.5-0.5B-Instruct (bf16, 0.49 B parameters)",
        "dir": snapshot("Qwen/Qwen2.5-0.5B-Instruct"),
        "runtimes": ["cancho-gpu", "mlx-lm", "transformers-cpu"],
        "reference": "transformers-cpu",
    },
    "qwen3.8-27b": {
        "title": "Qwen3.8-27B (4-bit)",
        "dir": snapshot("mlx-community/Qwen3.8-27B-4bit"),
        "ollama": "qwen3.8:27b-mlx",
        "runtimes": ["ollama", "mlx-lm", "cancho-gpu"],
        # Not the same weights (Ollama's are NVFP4, MLX's affine 4-bit): the column is only how far two
        # 4-bit versions of one model agree.
        "reference": "mlx-lm",
    },
}


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def median(xs):
    return statistics.median(xs) if xs else None


# ---------------------------------------------------------------------------
# Runtimes. Each answers {"load_s": float, "runs": [{prefill_tps, decode_tps, mem_gb, text, ...}]}
# or {"skipped": reason}.
# ---------------------------------------------------------------------------

def run_cancho(model, new, runs):
    binary = os.path.join(ROOT, "build", "cancho-gpu-run")
    if model["dir"] is None or not os.path.exists(binary):
        return {"skipped": "build/cancho-gpu-run is missing (scripts/build-run.sh) or the checkpoint is"}
    if "qwen2.5-0.5b" not in model["dir"].lower() and "Qwen2.5-0.5B" not in model["dir"]:
        return {"skipped": "not supported yet: the runner handles the Qwen2 architecture; Qwen3.8 needs NVFP4 weights and "
                           "linear-attention kernels (docs/inference.md section 6)"}
    out, loads = [], []
    for i in range(runs + 1):
        r = sh(["/usr/bin/time", "-l", binary, model["dir"], os.path.join(ROOT, "tests", "lx"), PROMPT, str(new), "--timing"])
        if r.returncode != 0:
            return {"skipped": f"cancho-gpu-run failed: {r.stderr.strip().splitlines()[:2]}"}
        text, _, rest = r.stdout.partition("\ntokens: ")
        ids = [int(x) for x in rest.splitlines()[0].split()]
        t = dict(re.findall(r"(\w+)=(\d+)", rest.splitlines()[1]))
        rss = int(re.search(r"(\d+)\s+maximum resident set size", r.stderr).group(1))
        if i == 0:
            continue  # warm-up
        loads.append(int(t["load_ms"]) / 1000)
        out.append({"prefill_tps": int(t["prefill_tokens"]) / max(int(t["prefill_ms"]), 1) * 1000,
                    "decode_tps": int(t["decode_tokens"]) / max(int(t["decode_ms"]), 1) * 1000,
                    "mem_gb": rss / 1e9, "text": text.lstrip("\n"), "ids": ids,
                    "prefill_tokens": int(t["prefill_tokens"]), "decode_tokens": int(t["decode_tokens"])})
    return {"load_s": median(loads), "runs": out,
            "note": "memory is the process's resident set; the weights live in Metal buffers that may not be counted"}


def run_python(script, model, new, runs, timeout):
    if model["dir"] is None:
        return {"skipped": "the checkpoint is not in the Hugging Face cache"}
    r = sh([sys.executable, os.path.join(ROOT, "bench", script), model["dir"], PROMPT, str(new), str(runs)], timeout=timeout)
    if r.returncode != 0:
        return {"skipped": f"{script} failed: {r.stderr.strip().splitlines()[-1:]}"}
    d = json.loads(r.stdout.strip().splitlines()[-1])
    for x in d["runs"]:
        x["prefill_tps"] = x.pop("prompt_tps")
        x["mem_gb"] = x.pop("peak_gb", None)
    return d


def run_mlx(model, new, runs):
    return run_python("mlx_run.py", model, new, runs, 3600)


def run_hf(model, new, runs):
    return run_python("hf_run.py", model, new, runs, 7200)


def ollama_stop_all():
    ps = sh(["ollama", "ps"]).stdout.splitlines()[1:]
    for line in ps:
        if line.strip():
            sh(["ollama", "stop", line.split()[0]])


def run_ollama(model, new, runs):
    name = model.get("ollama")
    if not name or sh(["which", "ollama"]).returncode != 0:
        return {"skipped": "Ollama is not installed or has no such model"}

    def ask():
        body = json.dumps({"model": name, "prompt": PROMPT, "raw": True, "stream": False, "keep_alive": "10m",
                           "options": {"num_predict": new, "temperature": 0, "seed": 1}}).encode()
        req = urllib.request.Request("http://localhost:11434/api/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as resp:
            return json.loads(resp.read())
    ollama_stop_all()
    try:
        first = ask()  # cold: the load and the warm-up
    except Exception as e:  # noqa: BLE001
        return {"skipped": f"Ollama failed: {e}"}
    out = []
    for _ in range(runs):
        d = ask()
        # Ollama keeps the prompt's keys and values between requests, so after the first
        # a request evaluates a handful of tokens: the prefill rate is the cold request's.
        out.append({"prefill_tps": first["prompt_eval_count"] / max(first["prompt_eval_duration"], 1) * 1e9,
                    "decode_tps": d["eval_count"] / max(d["eval_duration"], 1) * 1e9,
                    "mem_gb": None, "text": d["response"], "prefill_tokens": d["prompt_eval_count"],
                    "decode_tokens": d["eval_count"]})
    ollama_stop_all()
    return {"load_s": first["load_duration"] / 1e9, "runs": out,
            "note": "Ollama caches the prompt between requests, so its prefill rate is that of the first, cold request "
                    "(one measurement, not a median); memory is not reported by the API"}


RUNNERS = {"cancho-gpu": run_cancho, "mlx-lm": run_mlx, "transformers-cpu": run_hf, "ollama": run_ollama}
LABELS = {"cancho-gpu": "cancho-gpu (Metal, f16 matrices)", "mlx-lm": "MLX-LM (Metal)",
          "transformers-cpu": "transformers (CPU, float32)", "ollama": "Ollama (Metal)"}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def machine():
    chip = sh(["sysctl", "-n", "machdep.cpu.brand_string"]).stdout.strip()
    ram = int(sh(["sysctl", "-n", "hw.memsize"]).stdout.strip()) / 2**30
    versions = {}
    for mod in ("mlx", "mlx_lm", "transformers", "torch"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    ver = sh(["ollama", "--version"]).stdout.strip().splitlines()
    return {"chip": chip, "ram_gib": round(ram), "macos": platform.mac_ver()[0], "python": platform.python_version(),
            "ollama": ver[-1] if ver else None, **versions,
            "cancho_gpu_commit": sh(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"]).stdout.strip()}


def fmt(x, digits=1):
    return "-" if x is None else f"{x:.{digits}f}"


def span(values):
    return "-" if not values else f"{median(values):.1f} ({min(values):.1f}-{max(values):.1f})"


def table(results, mkey):
    res = results[mkey]
    ref_name = MODELS[mkey]["reference"]
    ref = res.get(ref_name, {})
    ref_run = ref["runs"][0] if ref.get("runs") else None
    lines = ["| runtime | load s | prefill tok/s | decode tok/s | peak memory GB | agrees with reference |",
             "|---|---|---|---|---|---|"]
    for rt in MODELS[mkey]["runtimes"]:
        r = res.get(rt)
        if r is None:
            continue
        if "skipped" in r:
            lines.append(f"| {LABELS[rt]} | not run: {r['skipped']} | | | | |")
            continue
        runs = r["runs"]
        mem = [x["mem_gb"] for x in runs if x["mem_gb"] is not None]
        if ref_run is None:
            agree = "no reference for this model"
        elif rt == ref_name:
            agree = "reference"
        elif runs[0].get("ids") and ref_run.get("ids"):
            agree = f"{common_prefix(runs[0]['ids'], ref_run['ids'])} of {len(ref_run['ids'])} tokens"
        else:
            a, b = runs[0]["text"].lstrip(), ref_run["text"].lstrip()
            agree = f"{common_prefix(a, b)} of {len(b)} characters"
        lines.append(f"| {LABELS[rt]} | {fmt(r['load_s'])} | {span([x['prefill_tps'] for x in runs])} | "
                     f"{span([x['decode_tps'] for x in runs])} | {fmt(median(mem)) if mem else '-'} | {agree} |")
    return "\n".join(lines)


def markdown(results, args, mach):
    out = ["# Benchmark", "",
           f"> Generated by `bench/bench.py` on {time.strftime('%Y-%m-%d')} ({mach['chip']}, {mach['ram_gib']} GiB, macOS "
           f"{mach['macos']}, cancho-gpu `{mach['cancho_gpu_commit']}`). Every number below was measured on that machine "
           f"by that script; re-run it to refresh them.", ""]
    out += ["## What is measured", "",
            f"The same raw prompt (about 60 tokens, no chat template), greedy, **{args.new} new tokens**, per runtime: "
            f"one warm-up run (dropped) then **{args.runs} runs**. The cells are the median with the minimum and maximum "
            f"in brackets.", "",
            "* **load s**: from starting to being ready to generate (for cancho-gpu: tokenizer, kernel compilation and "
            "the weights; for Ollama: the cold load its API reports; for MLX and transformers: reading the checkpoint).",
            "* **prefill tok/s**: prompt tokens over the time to the first generated token (it includes that token's own "
            "computation). cancho-gpu has no batched prefill, so this is its decode rate; the others batch.",
            "* **decode tok/s**: generated tokens after the first over the time they took, as the runtime or this "
            "script measures it (`--timing` for cancho-gpu, MLX's own report, Ollama's `eval_duration`, the difference of "
            "two `generate` calls for transformers).",
            "* **peak memory GB**: what the runtime reports (MLX: peak Metal memory; cancho-gpu: the process's resident "
            "set, which may not include its Metal buffers). Ollama's API reports none.",
            "* **agrees with reference**: the length of the common prefix of the generated token ids (or, "
            "for Ollama, which does not return them, of the text) with the reference runtime's. Different quantisations of one model legitimately diverge.", "",
            "Runtimes ran one at a time with the others idle; Ollama's models were unloaded first. The GPU is shared "
            "with whatever else the machine is doing, which is the main source of the spread.", "",
            "**Compare rows within one table, not across invocations.** The spread inside an invocation is small, "
            "but two full invocations minutes apart on the same machine differed by up to 1.8x for the same "
            "runtime (MLX-LM decoding the 27B: 12.1 tok/s in one, 21.3 in the other; transformers on the CPU: 7.6 and "
            "13.0). Whatever changed between them (clocks, thermal state, what else was resident) was not controlled, "
            "so a ratio between two rows is evidence and a number from last week is not.", ""]
    for mkey in args.models:
        out += [f"## {MODELS[mkey]['title']}", "", table(results, mkey), ""]
        notes = {rt: results[mkey][rt].get("note") for rt in MODELS[mkey]["runtimes"] if results[mkey].get(rt, {}).get("note")}
        for rt, n in notes.items():
            out.append(f"* {LABELS[rt]}: {n}.")
        out.append("")
    out += ["## Not measured", "",
            "* Accuracy beyond the agreement column: `scripts/model-check.sh` holds cancho-gpu's logits to transformers'.",
            "* Long contexts: cancho-gpu's context is 256 positions.",
            "* Batched or concurrent requests; energy; a second machine.",
            "* cancho-gpu on Qwen3.8-27B: it does not run it yet (`docs/inference.md` section 6)."]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2.5-0.5b,qwen3.8-27b")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--json")
    ap.add_argument("--markdown")
    ap.add_argument("--quick", action="store_true", help="2 runs of 32 tokens, to see that it works")
    ap.add_argument("--from-json", help="render the markdown from a results file instead of running")
    args = ap.parse_args()
    if args.quick:
        args.runs, args.new = 2, 32
    args.models = args.models.split(",")
    if args.from_json:
        saved = json.load(open(args.from_json))
        mach = saved["machine"]
        args.runs, args.new = saved["args"]["runs"], saved["args"]["new"]
        md = markdown(saved["results"], args, mach)
        if args.markdown:
            with open(args.markdown, "w") as f:
                f.write(md)
        print(md)
        return
    ollama_stop_all()
    results = {}
    for mkey in args.models:
        results[mkey] = {}
        for rt in MODELS[mkey]["runtimes"]:
            runs = args.runs if mkey != "qwen3.8-27b" else min(args.runs, 3)
            print(f"[{mkey}] {rt} ...", file=sys.stderr, flush=True)
            results[mkey][rt] = RUNNERS[rt](MODELS[mkey], args.new, runs)
            r = results[mkey][rt]
            print("   ", r["skipped"] if "skipped" in r else f"decode {span([x['decode_tps'] for x in r['runs']])} tok/s",
                  file=sys.stderr, flush=True)
    mach = machine()
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"machine": mach, "args": vars(args), "prompt": PROMPT, "results": results}, f, indent=1)
    md = markdown(results, args, mach)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(md)
    print(md)


if __name__ == "__main__":
    main()
