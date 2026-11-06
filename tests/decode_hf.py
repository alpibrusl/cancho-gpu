#!/usr/bin/env python3
"""The batched decode (`qsw decode`) on the real Qwen2.5-1.5B-Instruct against transformers' greedy generation (float32, CPU): two chat prompts of different lengths are fed one token a step and
then the session free-runs with a temperature so low (invt 1e6) that the sampler is an arg max; the tokens it generates are compared with `generate(do_sample=False)` on the same prompts.
Needs a Metal GPU, the converted weights (examples/qwen/convert.py) and the Hugging Face cache of the model.

    python3 tests/decode_hf.py build/qsw build/cancho-gpu-metal-session WEIGHTS_DIR [--new 24]

Reported per prompt: how many of the generated tokens agree and where the first difference is. f16 products can flip a near-tie, so the check is that the first `--new 12` tokens agree for both prompts and the logits' arg max agrees on most of the rest."""
import os, struct, subprocess, sys, tempfile
import numpy as np

# transformers imports a package called `kernels`, which tests/kernels.py would shadow if this directory stayed first on the path
sys.path[:] = [x for x in sys.path if os.path.abspath(x or ".") != os.path.dirname(os.path.abspath(__file__))]

arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
QSW, BIN, W = sys.argv[1], sys.argv[2], sys.argv[3]
NEW = int(arg("--new", 24))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bad = 0

def say(ok, text):
    global bad
    print(("ok   " if ok else "FAIL ") + text, flush=True)
    bad += 0 if ok else 1

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float32).eval()
msgs = ["Write a cancho function that returns the square of an integer.", "What is the capital of France? Answer in one short sentence, then explain why in two."]
prompts = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"] for m in msgs]
lens = [len(p) for p in prompts]
want = []
for p in prompts:
    with torch.no_grad():
        out = model.generate(torch.tensor([p]), do_sample=False, repetition_penalty=1.0, max_new_tokens=NEW, pad_token_id=tok.pad_token_id)   # Qwen's generation_config has repetition_penalty 1.05, which greedy generation applies
    want.append(out[0, len(p):].tolist())
print("prompt lengths", lens, flush=True)
NB, S = len(prompts), max(lens) + NEW
d = tempfile.mkdtemp(prefix="dechf_"); out_dir = os.path.join(d, "out"); os.makedirs(out_dir)
T = np.zeros((NB, S), dtype="<i4")
for r, p in enumerate(prompts):
    T[r, :len(p)] = p
T.tofile(os.path.join(d, "tokens.i32")); np.array(lens, dtype="<i4").tofile(os.path.join(d, "lens.i32"))
bits = lambda x: struct.unpack("<q", struct.pack("<d", x))[0]
open(os.path.join(d, "cfg.txt"), "w").write("".join(f"{k} {v}\n" for k, v in dict(layers=28, d=1536, h=12, hkv=2, dh=128, ff=8960, v=151936, t=256, nb=NB, tied=1, r=16, steps=S, eps=bits(1e-6), theta=bits(1e6), invt=bits(1000000.0)).items()))
r = subprocess.run([QSW, "decode", "--config", os.path.join(d, "cfg.txt"), "--wdir", W, "--out", out_dir, "--root", ROOT, "--tokens", os.path.join(d, "tokens.i32"), "--lens", os.path.join(d, "lens.i32")], capture_output=True, text=True)
say(r.returncode == 0, f"qsw decode wrote the session{'' if r.returncode == 0 else ': ' + r.stderr[:300]}")
if r.returncode:
    sys.exit(1)
p = subprocess.Popen([BIN, os.path.join(out_dir, "session")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
p.stdin.write(f"{S}\n"); p.stdin.flush()
line = p.stdout.readline()
if not line:
    p.wait(); print("session:", p.returncode, p.stderr.read()[:500]); sys.exit(1)
p.stdin.close(); p.wait()
ids = np.array([int(round(float(x))) for x in line.split()]).reshape(NB, S) - 1
tot_ok = 0
for rr in range(NB):
    got = ids[rr, lens[rr] - 1:lens[rr] - 1 + NEW].tolist()
    same = [a == b for a, b in zip(got, want[rr])]
    first = same.index(False) if False in same else NEW
    tot_ok += sum(same)
    print(f"     prompt {rr}: {sum(same)} of {NEW} tokens agree, first difference at {first if first < NEW else 'none'}")
    print("       decode :", repr(tok.decode(got)))
    print("       hf     :", repr(tok.decode(want[rr])))
    say(first >= min(12, NEW), f"prompt {rr}: the first {min(12, NEW)} generated tokens agree with transformers' greedy generation")

# the same model teacher-forced on the sequences transformers generated: the logits of every position against transformers' own, which says whether a difference above is a flipped near-tie
seqs = [prompts[r] + want[r] for r in range(NB)]
S2 = max(len(q) for q in seqs)
d2 = tempfile.mkdtemp(prefix="dechf2_"); o2 = os.path.join(d2, "out"); os.makedirs(o2)
T2 = np.zeros((NB, S2), dtype="<i4")
for r, q in enumerate(seqs):
    T2[r, :len(q)] = q
T2.tofile(os.path.join(d2, "tokens.i32")); np.array([len(q) for q in seqs], dtype="<i4").tofile(os.path.join(d2, "lens.i32"))
open(os.path.join(d2, "cfg.txt"), "w").write("".join(f"{k} {v}\n" for k, v in dict(layers=28, d=1536, h=12, hkv=2, dh=128, ff=8960, v=151936, t=256, nb=NB, tied=1, r=16, steps=S2, eps=bits(1e-6), theta=bits(1e6), invt=bits(1000000.0)).items()))
r2 = subprocess.run([QSW, "decode", "--config", os.path.join(d2, "cfg.txt"), "--wdir", W, "--out", o2, "--root", ROOT, "--tokens", os.path.join(d2, "tokens.i32"), "--lens", os.path.join(d2, "lens.i32"), "--logits"], capture_output=True, text=True)
assert r2.returncode == 0, r2.stderr
with torch.no_grad():
    hf_logits = [model(torch.tensor([q])).logits[0].numpy() for q in seqs]
p2 = subprocess.Popen([BIN, os.path.join(o2, "session")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
V = 151936; vp = (V + 1535) // 1536 * 1536
cos_min, agree, total, flips = 1.0, 0, 0, []
samp_agree, samp_flips = 0, []
for j in range(S2):
    p2.stdin.write("1\n"); p2.stdin.flush()
    ln = p2.stdout.readline()
    if not ln:
        p2.wait(); print("session:", p2.returncode, p2.stderr.read()[:400]); sys.exit(1)
    vals = np.array(ln.split(), dtype=np.float64)
    got, drawn = vals[:NB * vp].reshape(NB, vp)[:, :V], vals[NB * vp:].astype(np.int64) - 1
    for r in range(NB):
        if j >= len(seqs[r]):
            continue
        ref = hf_logits[r][j].astype(np.float64)
        cos_min = min(cos_min, float(got[r] @ ref / np.linalg.norm(got[r]) / np.linalg.norm(ref)))
        total += 1
        if drawn[r] == got[r].argmax():
            samp_agree += 1
        else:
            samp_flips.append((r, j, int(drawn[r]), int(got[r].argmax()), float(got[r][drawn[r]]), float(got[r].max())))
        if got[r].argmax() == ref.argmax():
            agree += 1
        else:
            top = np.sort(ref)[-2:]
            flips.append((r, j, float(top[1] - top[0]), float(ref.max() - ref.min())))
p2.stdin.close(); p2.wait()
print(f"     teacher-forced on the generated sequences: {total} positions, the arg max agrees on {agree}, the smallest cosine {cos_min:.6f}")
for r, j, gap, rng_ in flips:
    print(f"       sequence {r} position {j}: transformers' top two logits differ by {gap:.3f} (their range is {rng_:.1f})")
print(f"     the sampler (invt 1e6) drew the arg max of its own logits at {samp_agree} of {total} positions"); [print("       sampler flip", f) for f in samp_flips[:10]]
say(cos_min > 0.999, f"the logits of every position have cosine at least {cos_min:.6f} with transformers'")
say(all(g < 0.5 for _, _, g, _ in flips), "every flipped arg max is a near-tie in transformers' own logits (gap under 0.5)")
print("bad" if bad else "all ok")
sys.exit(1 if bad else 0)
