#!/usr/bin/env python3
r"""docs/weight-space.md section 14, ticket T6, checks C3 and C4: the Qwen3.5-0.8B engine's layer outputs against Hugging Face.

    python3 examples/qwen35/qwen08_c34.py [--model ~/models/qwen35-08b-base] [--runner build/cancho-gpu-run08] [--kdir tests/chx]
                                          [--mode f32|f16] [--tmax 1024] [--prompts all|N,...] [--cache DIR] [--no-run]

Needs a Mac with Metal (the runner, `scripts/build-run08.sh`), `torch`, `transformers` (>= 5.2) and `numpy`; not in the gate.

The reference is `transformers`' own `modeling_qwen3_5` code (`hf64.py`: its float32 casts turned into float64) running the text tower of the
checkpoint on the CPU in binary64, with a forward hook on every decoder layer and on the final norm; it is written once to `--cache` (default
`build/c34/`, one .npz a prompt, and a second set for the shipped float32 code, whose gap to binary64 is the reference's own floor, 14.6). It does
NOT use the T1 goldens (`examples/qwen35/hf08_ref.py`), which may not exist yet; it is a layer-by-layer check of one short prompt set:
the first three GSM8K test problems rendered as W1 prompts (the model's chat template, thinking on), five short texts and one long text of 700 tokens
(prompt 8: attention over 256, 512 and 1,024 cache positions).

For each prompt the runner is fed the token ids and dumps (`--trace`) the embedding, every layer's output at every position, and the final
norm; the logits' 20 largest come with it. The pass criteria are those of 14.6, fixed before any run:

* C3: layer 0, every position, max |ours - reference| / max |reference| <= 1e-4 (f32 weights);
* C4: the same for every layer 0 to 23 (layer 3 is the first attention layer; 15 and 19 are the edited ones), and not more than 4 times the
  reference's own float32 floor (shipped float32 code against binary64, the same ratio).

Both ratios are printed for every layer, with the worst position, so that a failure is localised (layer, position) by the first line that fails.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "weightspace"))

SHORT = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n",
    "Once upon a time, in a small village by the sea,",
    "1 + 1 = 2, 2 + 2 = 4, 3 + 3 =",
    "Q: What is the largest planet in the solar system?\nA:",
]


def prompts(model, n_w1=3):
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    import w1_behave
    tok = AutoTokenizer.from_pretrained(model)
    gsm = os.path.expanduser("~/models/gsm8k/main/test-00000-of-00001.parquet")
    out = []
    for q in pq.read_table(gsm).column("question").to_pylist()[:n_w1]:
        text = tok.apply_chat_template([{"role": "user", "content": w1_behave.build_question(q)}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        out.append(tok(text, add_special_tokens=False)["input_ids"])
    for s in SHORT:
        out.append(tok(s, add_special_tokens=False)["input_ids"])
    # one long prompt (about 700 tokens: the first 12 GSM8K problems one after the other), so that the attention kernels compiled for 256 and
    # 512 positions and for tmax all run, and the delta state is carried far
    longp = " ".join(w1_behave.build_question(q) for q in pq.read_table(gsm).column("question").to_pylist()[:12])
    out.append(tok(longp, add_special_tokens=False)["input_ids"][:700])
    return out


def load_text_model(model, f64):
    import torch
    from safetensors import safe_open
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    if f64:
        import hf64
        M = hf64.load()
    tc = json.load(open(model + "/config.json"))["text_config"]
    cfg = Qwen3_5TextConfig(**{k: v for k, v in tc.items() if k not in ("model_type", "dtype", "mtp_num_hidden_layers", "mtp_use_dedicated_embeddings", "mamba_ssm_dtype")})
    cfg.tie_word_embeddings = True
    net = M.Qwen3_5ForCausalLM(cfg)
    net = net.double() if f64 else net.float()
    path = model + "/model.safetensors"
    if not os.path.exists(path):
        path = model + "/model.safetensors-00001-of-00001.safetensors"
    sd = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.language_model."):
                v = f.get_tensor(k)
                sd["model." + k[len("model.language_model."):]] = v.double() if f64 else v.float()
    missing, unexpected = net.load_state_dict(sd, strict=False)
    net.tie_weights()
    assert not unexpected and all(m == "lm_head.weight" for m in missing), (missing, unexpected)
    return net.eval()


def reference(net, ids):
    import torch
    outs = {}
    hooks = []
    for i, layer in enumerate(net.model.layers):
        hooks.append(layer.register_forward_hook(lambda mod, a, o, i=i: outs.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach())))
    hooks.append(net.model.norm.register_forward_hook(lambda mod, a, o: outs.__setitem__("norm", o.detach())))
    emb = []
    hooks.append(net.model.embed_tokens.register_forward_hook(lambda mod, a, o: emb.append(o.detach())))
    with torch.no_grad():
        logits = net(torch.tensor([ids])).logits[0]
    for h in hooks:
        h.remove()
    rows = [emb[0][0]] + [outs[i][0] for i in range(24)] + [outs["norm"][0]]
    return torch.stack(rows, 1).double().numpy(), logits[-1].double().numpy()  # [n, 26, 1024], [vocab]


def cached(cache, name, build):
    path = os.path.join(cache, name + ".npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["hs"], z["logits"]
    hs, lg = build()
    np.savez(path, hs=hs, logits=lg)
    return hs, lg


def run_runner(args, ids):
    cmd = [args.runner, args.model, args.kdir, args.mode, str(args.tmax), ",".join(map(str, ids)), "--trace", "--timing"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"runner failed ({p.returncode}): {p.stderr.strip()[:2000]}")
    n = len(ids)
    hs = np.zeros((n, 26, 1024), np.float32)
    top = None
    timing = ""
    for line in p.stdout.split("\n"):
        if line.startswith("trace "):
            _, pos, slot, *hexes = line.split(" ")
            hs[int(pos), int(slot)] = np.frombuffer(bytes.fromhex("".join(hexes)), dtype=">u4").astype(np.uint32).view(np.float32)
        elif line.startswith("top "):
            top = [(int(a), np.array([int(b)], np.int64).astype(np.uint32).view(np.float32)[0]) for a, b in (x.split(":") for x in line.split(" ")[2:])]
        elif line.startswith("timing"):
            timing = line
    return hs, top, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/qwen35-08b-base"))
    ap.add_argument("--runner", default=os.path.join(ROOT, "build/cancho-gpu-run08"))
    ap.add_argument("--kdir", default=os.path.join(ROOT, "tests/chx"))
    ap.add_argument("--mode", default="f32")
    ap.add_argument("--tmax", type=int, default=1024)
    ap.add_argument("--prompts", default="all")
    ap.add_argument("--cache", default=os.path.join(ROOT, "build/c34"))
    ap.add_argument("--no-run", action="store_true", help="only write the reference")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    allp = prompts(args.model)
    pick = range(len(allp)) if args.prompts == "all" else [int(x) for x in args.prompts.split(",")]
    net64 = net32 = None
    worst = np.zeros(26)
    worst_floor = np.zeros(26)
    worst_pos = {}
    top_bad = []
    timing = []
    for pi in pick:
        ids = allp[pi]
        if len(ids) > args.tmax:
            sys.exit(f"prompt {pi} has {len(ids)} tokens; --tmax {args.tmax}")

        def b64():
            nonlocal net64
            net64 = net64 or load_text_model(args.model, True)
            return reference(net64, ids)

        def b32():
            nonlocal net32
            net32 = net32 or load_text_model(args.model, False)
            return reference(net32, ids)

        ref, rlg = cached(args.cache, f"p{pi}_f64", b64)
        ref32, rlg32 = cached(args.cache, f"p{pi}_f32", b32)
        if args.no_run:
            continue
        ours, top, tm = run_runner(args, ids)
        timing.append(tm)
        for s in range(26):
            peak = np.abs(ref[:, s]).max()
            e = np.abs(ours[:, s].astype(np.float64) - ref[:, s]).max(axis=1)
            fl = np.abs(ref32[:, s].astype(np.float64) - ref[:, s]).max() / peak
            r = e.max() / peak
            if r > worst[s]:
                worst[s] = r
                worst_pos[s] = (pi, int(e.argmax()))
            worst_floor[s] = max(worst_floor[s], fl)
        want = np.argsort(-rlg)[:5]
        got = [i for i, _ in top[:5]]
        gv = {i: v for i, v in top}
        dv = max(abs(float(gv.get(int(i), np.nan)) - float(rlg[i])) for i in want)
        print(f"prompt {pi} ({len(ids)} tokens): top-5 ids {'equal' if got == [int(i) for i in want] else 'DIFFER ' + str(got) + ' vs ' + str(list(map(int, want)))}, top-5 logit max |diff| {dv:.2e}; {tm}")
        if got != [int(i) for i in want]:
            top_bad.append(pi)
    if args.no_run:
        print("reference written to", args.cache)
        return
    names = ["embed"] + [f"layer {i}" + (" (attention)" if i % 4 == 3 else "") for i in range(24)] + ["final norm"]
    print(f"\nmode {args.mode}, tmax {args.tmax}: worst over {len(list(pick))} prompts, max|err| / max|ref| (reference float32 floor beside it)")
    bad = 0
    for s in range(26):
        ok = worst[s] <= 1e-4
        c4 = ok and (1 <= s <= 24) and worst[s] <= 4 * max(worst_floor[s], 1e-30)
        flag = "ok  " if ok else "FAIL"
        extra = "" if not (1 <= s <= 24) else ("" if c4 else "  (above 4x the float32 floor)")
        print(f"{flag} {names[s]:<22} {worst[s]:.3e}   floor {worst_floor[s]:.3e}   worst at prompt/pos {worst_pos.get(s)}{extra}")
        bad += 0 if ok else 1
    print(f"C3 (layer 0): {'pass' if worst[1] <= 1e-4 else 'FAIL'}; C4 (layers 0 to 23 <= 1e-4): {'pass' if all(worst[1:25] <= 1e-4) else 'FAIL'}; "
          f"C4 (and <= 4x the float32 floor): {'pass' if all(worst[1:25] <= 4 * np.maximum(worst_floor[1:25], 1e-30)) else 'FAIL'}")
    sys.exit(1 if bad or top_bad else 0)


main()
