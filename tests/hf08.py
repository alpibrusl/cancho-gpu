#!/usr/bin/env python3
"""examples/qwen35/hf08_ref.py and golden/hf08/ (docs/weight-space.md section 14, T1).

    python3 tests/hf08.py            # the gate: no model needed; torch/transformers parts skip with a message when absent
    python3 tests/hf08.py --models   # also regenerate a subset from ~/models/qwen35-08b-* and compare (minutes, ~7 GB; skips if absent)

  1. The committed goldens are well formed: the shapes C3 to C6 need (200 prompts, 8 check prompts x 26 outputs, 20 x 64 continuation
     steps, two models), the two checkpoints' rendered prompts are the same, every number is finite, the files stay small, and the
     float32-against-float64 argmax disagreements are exactly the ones listed (each with a top-2 gap, none silently dropped).
  2. The edit shows in the goldens where the design says it should (layers before 15 identical between base and edited, later ones not).
  3. The pipeline on a tiny random Qwen3.5 (4 layers, 3 linear + 1 attention): deterministic (two runs write byte-identical text), the
     float32 floor is small, the cached continuation agrees with the uncached forward, the rank/format helpers are exact.
  4. With --models: the real checkpoints reproduce the committed goldens (hf08_ref.py --check)."""
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLD = os.path.join(ROOT, "golden", "hf08")
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, os.path.join(ROOT, "examples", "qwen35"))

bad = 0


def say(ok, text):
    global bad
    print(("ok   " if ok else "FAIL ") + text)
    bad += 0 if ok else 1


def finite(x):
    if isinstance(x, float):
        return math.isfinite(x)
    if isinstance(x, list):
        return all(finite(v) for v in x)
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    return True


def load(name):
    with open(os.path.join(GOLD, name), encoding="utf-8") as f:
        return json.load(f)


def goldens():
    if not os.path.isdir(GOLD):
        say(False, "golden/hf08 is missing")
        return
    P = load("prompts.json")
    say(len(P["w1"]) == 200 and all(len(e["sha256_text"]) == 64 and e["n_tokens"] > 20 for e in P["w1"]), "prompts.json: 200 W1 prompts, hashed")
    say(len(P["check"]) == 8 and P["check_w1_problems"] == [0, 1, 2] and P["check"][0]["ids"] and P["n_cont"] == 20 and P["cont_len"] == 64,
        "prompts.json: 8 check prompts (3 W1 + 5 short), 20 x 64 continuation")
    say(all(P["w1"][i]["n_tokens"] == len(P["check"][i]["ids"]) for i in range(3)), "prompts.json: the 3 W1 check prompts are W1 problems 0 to 2")
    M = {n: load(n + ".json") for n in ("base", "edited")}
    for n, g in M.items():
        say(finite(g), f"{n}.json: every number is finite")
        say(len(os.listdir(GOLD)) == 5 and os.path.getsize(os.path.join(GOLD, n + ".json")) < 600_000, f"{n}.json: small ({os.path.getsize(os.path.join(GOLD, n + '.json'))} bytes)")
        lay = g["check"]["layers"]
        say(len(lay) == 8 and all(len(e["outputs"]) == 26 for e in lay), f"{n}: 8 check prompts x (embedding, 24 layers, final norm)")
        say(all(len(e["outputs"][k]["norms"]) == e["n_tokens"] == len(P["check"][p]["ids"]) for p, e in enumerate(lay) for k in range(26)),
            f"{n}: a norm for every position of every output")
        say([o["at"] for o in lay[0]["outputs"]] == ["emb"] + list(range(24)) + ["norm"] ,
            f"{n}: layer 3 (attention) and layers 15, 19 (edited) are present")
        say(all(len(e["top20_ids"]) == 20 and e["row_size"] == 248320 for e in g["check"]["logits"]) and len(g["check"]["logits"]) == 8, f"{n}: top-20 logits for 8 prompts")
        a = g["argmax200"]
        say(len(a["f64"]) == len(a["f32"]) == len(a["gap_f64"]) == 200 and all(0 <= t < 248320 for t in a["f64"]), f"{n}: 200 greedy next tokens")
        listed = [i for i in range(200) if a["f64"][i] != a["f32"][i]]
        say(listed == [d["index"] for d in a["disagree"]] and all(d["gap_f64"] == a["gap_f64"][d["index"]] for d in a["disagree"]),
            f"{n}: the float32/float64 argmax disagreements are exactly the listed ones ({len(listed)})")
        say(all(x >= 0 for x in a["gap_f64"]) and a["min_gap_f64"] == min(a["gap_f64"]), f"{n}: top-2 gaps (smallest {a['min_gap_f64']})")
        c = g["cont64"]
        say(len(c) == 20 and all(len(e["ids"]) == 64 == len(e["top5_ids"]) == len(e["top5_vals"]) for e in c), f"{n}: 20 x 64 continuation with top-5 per step")
        say(all(e["top5_ids"][k][0] == e["ids"][k] for e in c for k in range(64)), f"{n}: each step's token is its logits' top-1")
        say(all(e["top5_vals"][k] == sorted(e["top5_vals"][k], reverse=True) for e in c for k in range(64)), f"{n}: top-5 values are sorted")
        say(c[0]["ids"][0] == a["f64"][0], f"{n}: the continuation's first token is the 1-token argmax of the same prompt")
        say(all(e["f32_first_difference"] is None or e["f32_ids"][e["f32_first_difference"]] != e["ids"][e["f32_first_difference"]] for e in c), f"{n}: float32 continuation divergences are real")
        say(all(0 <= o["floor_rel"] < 1e-3 for e in lay for o in e["outputs"]), f"{n}: the float32 floor is recorded and small (worst {max(o['floor_rel'] for e in lay for o in e['outputs']):.2e})")
    # the edit: layers before 15 do not see it at all
    b, e = M["base"]["check"]["layers"], M["edited"]["check"]["layers"]
    same = lambda p, k: b[p]["outputs"][k]["sha256_f32"] == e[p]["outputs"][k]["sha256_f32"]
    say(all(same(p, k) for p in range(8) for k in range(0, 16)), "the edit: embedding and layers 0 to 14 are bit-equal between base and edited")
    say(any(not same(p, 1 + 19) for p in range(8)), "the edit: layer 19's output differs somewhere between base and edited")
    say(any(not same(p, 1 + 15) for p in range(8)), "the edit: layer 15's output differs somewhere between base and edited")
    say(all(not same(p, 26 - 1) for p in range(8)), "the edit: the final norm output differs for every check prompt")
    say(sum(x["ids"] != y["ids"] for x, y in zip(M["base"]["cont64"], M["edited"]["cont64"])) > 0, "the edit: the 64-token continuations differ between base and edited on some prompt")


def tiny():
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError:
        print("skip tiny pipeline: torch/transformers are not installed")
        return
    import hf08_ref as H
    import hf64
    mod64 = hf64.load()
    import transformers.models.qwen3_5.modeling_qwen3_5 as mod32

    def cfg(mod):
        c = mod.Qwen3_5TextConfig(
            vocab_size=64, hidden_size=24, intermediate_size=40, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=6, linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"], max_position_embeddings=256, tie_word_embeddings=True,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0], "mrope_interleaved": True})
        c.attn_output_gate = True
        c._attn_implementation = "eager"
        return c

    def load(prec):
        mod, dt = (mod64, torch.float64) if prec == "f64" else (mod32, torch.float32)
        torch.manual_seed(7)
        m = mod.Qwen3_5ForCausalLM(cfg(mod)).to(dt).eval()
        torch.manual_seed(7)
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(torch.randn(p.shape, dtype=torch.float64) * 0.3)
        return m

    H.setup(1)
    x = torch.randn(2, 40, 17, dtype=torch.float64)
    w = torch.randn(40, 1, 4, dtype=torch.float64)
    bb = torch.randn(40, dtype=torch.float64)
    say(all((H.depthwise_conv1d(x, w, bb, padding=pd, groups=40) - H._conv1d(x, w, bb, padding=pd, groups=40)).abs().max() < 1e-14 for pd in (0, 3))
        and (H.depthwise_conv1d(x, w[:, :, :3].contiguous(), None, padding=2, groups=40) - H._conv1d(x, w[:, :, :3].contiguous(), None, padding=2, groups=40)).abs().max() < 1e-14,
        "the depthwise convolution equals torch's grouped one (1e-14, float64; padding 0 and 3, with and without bias)")
    say(torch.nn.functional.conv1d(torch.randn(1, 4, 9), torch.randn(6, 4, 3)).shape == (1, 6, 7), "an ordinary convolution still goes to torch")
    spec = {"check": [[1, 5, 9, 3, 7, 2, 11, 4], [3, 3, 8]], "many": [[1, 2, 3], [4, 5, 6, 7], [9, 9], [2, 40, 41, 5]], "cont": [[1, 2, 3], [7, 8]],
            "cont_len": 6, "eos": [0]}
    m64 = load("f64")
    fast = H.hidden_states(m64, spec["check"][0])
    torch.nn.functional.conv1d = H._conv1d
    try:
        slow = H.hidden_states(m64, spec["check"][0])
    finally:
        torch.nn.functional.conv1d = H.depthwise_conv1d
    say(max(float((u - v).abs().max()) for u, v in zip(fast[1], slow[1])) < 1e-12, "tiny: the whole model with the fast convolution equals torch's grouped one (1e-12)")
    g1, full1 = H.compute_model(load, spec)
    g2, full2 = H.compute_model(load, spec)
    say(H.dumps({"x": g1}) == H.dumps({"x": g2}), "tiny: two runs write byte-identical text")
    say(all(o["floor_rel"] < 1e-4 for e in g1["check"]["layers"] for o in e["outputs"]), "tiny: the float32 floor is below 1e-4")
    say(len(g1["check"]["layers"][0]["outputs"]) == 6 and [o["at"] for o in g1["check"]["layers"][0]["outputs"]] == ["emb", 0, 1, 2, 3, "norm"], "tiny: embedding, 4 layers, final norm")
    say(len(g1["check"]["layers"][1]["outputs"][1]["norms"]) == 3, "tiny: a norm per position")
    # the cached continuation equals the uncached forward at every step
    m = load("f64")
    ids = spec["cont"][0]
    out, tops = H.greedy(m, ids, 6)
    ok = True
    seq = list(ids)
    for k in range(6):
        row = H.next_logits(m, seq)
        ti, tv = H.top_k(row, 5)
        ok = ok and ti == tops[k][0] and max(abs(a - b) for a, b in zip(tv, tops[k][1])) < 1e-9 and ti[0] == out[k]
        seq.append(out[k])
    say(ok, "tiny: the cached greedy continuation equals the uncached forward at all 6 steps (1e-9)")
    say(g1["cont64"][0]["ids"] == out, "tiny: the golden's continuation is that continuation")
    say(H.r(1 / 3) == 0.3333333333 and H.r(123456789012.0) == 123456789000.0 and H.rl([0.5]) == [0.5], "round-to-10-significant helper")
    ids = [1, 2, 3]
    say(H.sha_ids(ids) != H.sha_ids([1, 2, 4]) and len(H.sha_ids(ids)) == 64, "sha of ids")
    say(H.dumps({"a": [1, 2], "b": {"c": 1}}) == '{\n"a":[1,2],\n"b":{"c":1}\n}\n', "dumps: one top-level key a line")


def models():
    H = os.path.join(ROOT, "examples", "qwen35", "hf08_ref.py")
    for d in ("~/models/qwen35-08b-base", "~/models/qwen35-08b-edited", "~/models/gsm8k/main/test-00000-of-00001.parquet"):
        if not os.path.exists(os.path.expanduser(d)):
            print(f"skip --models: {d} is not there")
            return
    rc = subprocess.call([sys.executable, H, "--check"])
    say(rc == 0, "the real checkpoints reproduce the committed goldens (hf08_ref.py --check)")


goldens()
tiny()
if "--models" in sys.argv:
    models()
print("hf08: FAILED" if bad else "hf08: ok")
sys.exit(1 if bad else 0)
