#!/usr/bin/env python3
"""Prefill, then batched decode, on the real Qwen2.5-1.5B-Instruct against transformers' greedy generation: `qsw write --no-train --kv-export` writes a forward-only session of the training forward
pass over the distinct prompts that saves every layer's keys and values in the layout of the decode caches (eight copies of each prompt's, `kv_import`); `qsw decode --kv-dir --pos0` then
starts each sequence at the last token of its prompt with those caches; the 8 samples of a prompt must all generate transformers' greedy tokens. Needs a Metal GPU, the converted weights and the cache of the model.

    python3 tests/prefill_hf.py build/qsw build/cancho-gpu-metal-session WEIGHTS_DIR [--new 24]"""
import os, struct, subprocess, sys, tempfile, time
import numpy as np

sys.path[:] = [x for x in sys.path if os.path.abspath(x or ".") != os.path.dirname(os.path.abspath(__file__))]
arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
QSW, BIN, W = sys.argv[1], sys.argv[2], sys.argv[3]
NEW = int(arg("--new", 24))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = 256
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
        out = model.generate(torch.tensor([p]), do_sample=False, repetition_penalty=1.0, max_new_tokens=NEW, pad_token_id=tok.pad_token_id)
    want.append(out[0, len(p):].tolist())
NP = len(prompts)
bits = lambda x: struct.unpack("<q", struct.pack("<d", x))[0]
d = tempfile.mkdtemp(prefix="prefill_")
common = dict(layers=28, d=1536, h=12, hkv=2, dh=128, ff=8960, v=151936, t=T, tied=1, r=16, eps=bits(1e-6), theta=bits(1e6))
cfg = lambda path, **kw: open(path, "w").write("".join(f"{k} {v}\n" for k, v in {**common, **kw}.items()))

# 1. the prefill
kv = os.path.join(d, "kv"); out1 = os.path.join(d, "pre"); os.makedirs(kv); os.makedirs(out1)
tk = np.zeros((1, NP, T), dtype="<i4")
for r, p in enumerate(prompts):
    tk[0, r, :len(p)] = p
tk.tofile(os.path.join(d, "pre.tok"))
words = sum(16 * (i + o) for i, o in ((1536, 1536), (1536, 256), (1536, 256), (1536, 1536), (1536, 8960), (1536, 8960), (8960, 1536))) * 28
np.zeros(words, dtype="<f8").tofile(os.path.join(d, "zero.ad"))
cfg(os.path.join(d, "pre.cfg"), nb=NP, steps=1)
r1 = subprocess.run([QSW, "write", "--config", os.path.join(d, "pre.cfg"), "--wdir", W, "--out", out1, "--root", ROOT, "--tokens", os.path.join(d, "pre.tok"), "--adapters", os.path.join(d, "zero.ad"), "--no-train", "--kv-export", kv], capture_output=True, text=True)
say(r1.returncode == 0, f"qsw wrote the prefill session{'' if r1.returncode == 0 else ': ' + r1.stderr[:300]}")
if r1.returncode:
    sys.exit(1)
p1 = subprocess.Popen([BIN, os.path.join(out1, "session")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
t0 = time.time(); p1.stdin.write("1\n"); p1.stdin.flush(); l1 = p1.stdout.readline(); first = time.time() - t0
if not l1:
    p1.wait(); print("prefill session:", p1.returncode, p1.stderr.read()[:500]); sys.exit(1)
t0 = time.time(); p1.stdin.write("1\n"); p1.stdin.flush(); p1.stdout.readline(); again = time.time() - t0
t0 = time.time(); p1.stdin.write("checkpoint\n"); p1.stdin.flush(); p1.stdout.readline(); saved = time.time() - t0
p1.stdin.close(); p1.wait()
nfiles = len(os.listdir(kv))
say(nfiles == 56, f"the prefill saved {nfiles} cache files (keys and values of 28 layers); first request {first:.1f} s, a second {again:.2f} s, the checkpoint (writing them) {saved:.1f} s")

# 2. the decode from those caches
NB = NP * 8
S = NEW
T2 = np.zeros((NB, S), dtype="<i4")
for r in range(NB):
    T2[r, 0] = prompts[r // 8][-1]
T2.tofile(os.path.join(d, "dec.tok")); np.ones(NB, dtype="<i4").tofile(os.path.join(d, "dec.lens"))
np.array([lens[r // 8] - 1 for r in range(NB)], dtype="<i4").tofile(os.path.join(d, "dec.pos"))
out2 = os.path.join(d, "dec"); os.makedirs(out2)
cfg(os.path.join(d, "dec.cfg"), nb=NB, steps=S, invt=bits(1000000.0))
r2 = subprocess.run([QSW, "decode", "--config", os.path.join(d, "dec.cfg"), "--wdir", W, "--out", out2, "--root", ROOT, "--tokens", os.path.join(d, "dec.tok"), "--lens", os.path.join(d, "dec.lens"), "--pos0", os.path.join(d, "dec.pos"), "--kv-dir", kv], capture_output=True, text=True)
say(r2.returncode == 0, f"qsw wrote the decode session{'' if r2.returncode == 0 else ': ' + r2.stderr[:300]}")
if r2.returncode:
    sys.exit(1)
p2 = subprocess.Popen([BIN, os.path.join(out2, "session")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
t0 = time.time(); p2.stdin.write(f"{S}\n"); p2.stdin.flush(); line = p2.stdout.readline(); secs = time.time() - t0
if not line:
    p2.wait(); print("decode session:", p2.returncode, p2.stderr.read()[:500]); sys.exit(1)
p2.stdin.close(); p2.wait()
ids = np.array([int(round(float(x))) for x in line.split()]).reshape(NB, S) - 1
print(f"     {S} decode steps for {NB} sequences: {secs:.2f} s (the first request compiles)", flush=True)
for pr in range(NP):
    rows = [ids[pr * 8 + c, :NEW].tolist() for c in range(8)]
    ok = [row == want[pr] for row in rows]
    print(f"     prompt {pr}: {sum(ok)} of 8 copies generate transformers' {NEW} greedy tokens; copy 0: {tok.decode(rows[0])!r}")
    say(all(ok), f"prompt {pr}: all 8 samples, started from the prefilled caches, equal transformers' greedy generation")
print("bad" if bad else "all ok")
sys.exit(1 if bad else 0)
