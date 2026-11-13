#!/usr/bin/env python3
"""D5 of docs/diffusion.md: the text encoder of Qwen-Image-2.1-Turbo (36 layers of the language model of Qwen3-VL) on a Metal GPU in one session, held to the float64 encoder (tools/text_ref.py) and to torch's BF16 hidden states.

    text_gpu.py GOLDEN_DIR [REF_DIR] [--keep DIR] [--save OUT.npy]

REF_DIR is the output of `python tools/text_ref.py GOLDEN_DIR REF_DIR` (the float64 residual stream after each layer; optional). Needs a Mac, the checkpoint, D0's goldens (`text_ids`, `drop_idx`, `text_embeds`), numpy, torch and safetensors (the venv of the
first image edits has them), `build/cancho-gpu-metal-session`, and about 18 GB of disk for the f16 weight files. Not in `scripts/gate.sh`.

It needs **no new kernel**: the encoder is the DiT's block with other constants. Per layer: rmsnorm_rows over the 4096 channels (weight, eps 1e-6), dit_to_f16, three gemm_mma (q, k, v), rmsnorm_rows per head (q and k: weight [128]), dit_rope (rotate-half with per-position tables:
the text model's rotary embedding is that, theta 5e6, so the DiT's permutation of channels is not needed), dit_scores / dit_stats / dit_pv (causal mask), dit_to_f16, gemm_mma (o), dit_add, rmsnorm_rows, dit_to_f16, gemm_mma (gate, up), dit_swiglu, gemm_mma (down), dit_add. **Grouped attention** (32 query heads, 8 key/value heads)
is the same kernels with the key and value projections' rows repeated four times (head i reads key/value head i // 4), which costs 50 MB of f16 weights a layer and no kernel. The 35 tokens are padded to 64 rows (the matrix unit's tile); padded rows are zeros and are not keys.
The pipeline takes the residual stream after the 36 layers **before the final norm**, from row `drop_idx` on: so does this. On the host: the embedding lookup (35 rows of a 151,936 x 4096 table), the rotary tables, the weights as f16 files.
"""
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import text_ref as tr  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gold = sys.argv[1]
refdir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
keep = sys.argv[sys.argv.index("--keep") + 1] if "--keep" in sys.argv else None
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
CHX = os.path.join(ROOT, "tests/chx")
NEG = -1e9
D, H, KV, HD, NL, SP = 4096, 32, 8, 128, 36, 64
ids = np.load(os.path.join(gold, "text_ids.npy"))[0].astype(int)
drop = int(np.load(os.path.join(gold, "drop_idx.npy"))[0])
want = np.load(os.path.join(gold, "text_embeds.npy")).astype(np.float64)[0]
N = len(ids)
work = keep or tempfile.mkdtemp()
os.makedirs(work, exist_ok=True)
P = lambda n: os.path.join(work, n)
f16 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f2")).tofile(path)
f32 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f4")).tofile(path)
Wf = lambda name: tr._f.get_tensor("model.language_model." + name).float().numpy()

emb = tr._f.get_slice("model.language_model.embed_tokens.weight")
x0 = np.zeros((SP, D))
x0[:N] = np.stack([emb[int(t):int(t) + 1][0].double().numpy() for t in ids])
f32(P("x0.f32"), x0)
cos, sin = tr.rope_tables(SP)
f32(P("cs.f32"), cos)
f32(P("sn.f32"), sin)
idx = np.arange(SP)
f32(P("mask.f32"), np.where((idx[:, None] >= idx[None, :]) & (idx < N)[None, :], 0.0, NEG))

t0 = time.time()
rep = lambda m: np.repeat(m.reshape(KV, HD, D), H // KV, axis=0).reshape(H * HD, D)
for i in range(NL):
    p = "layers.%d." % i
    f16(P("wq%d.f16" % i), Wf(p + "self_attn.q_proj.weight"))
    f16(P("wk%d.f16" % i), rep(Wf(p + "self_attn.k_proj.weight")))
    f16(P("wv%d.f16" % i), rep(Wf(p + "self_attn.v_proj.weight")))
    f16(P("wo%d.f16" % i), Wf(p + "self_attn.o_proj.weight"))
    f16(P("wg%d.f16" % i), Wf(p + "mlp.gate_proj.weight"))
    f16(P("wu%d.f16" % i), Wf(p + "mlp.up_proj.weight"))
    f16(P("wd%d.f16" % i), Wf(p + "mlp.down_proj.weight"))
    for n, key in (("l1", "input_layernorm"), ("l2", "post_attention_layernorm")):
        f32(P("%s%d.f32" % (n, i)), Wf(p + key + ".weight")[None])
    f32(P("qn%d.f32" % i), Wf(p + "self_attn.q_norm.weight")[None])
    f32(P("kn%d.f32" % i), Wf(p + "self_attn.k_norm.weight")[None])
print("weights written: %d layers in %.0f s" % (NL, time.time() - t0), flush=True)

prog = []
n_prog = [0]


def program(name, kernel, args):
    n_prog[0] += 1
    prog.append(f"program {name} {CHX}/{kernel}.chx {args}")


def res(target, fname, flag="f32"):
    prog.append(f"resident {target} {P(fname)} {flag}")


def link(a, b):
    prog.append(f"link {a} -> {b}")


SHOW = [4, 12, 24, 36]
for i in range(NL):
    n = lambda s: "L%d_%s" % (i, s)
    prev = None if i == 0 else "L%d_ad2.z" % (i - 1)
    program(n("n1"), "rmsnorm_rows", f"r={SP} n={D} eps=0.000001")
    if prev:
        link(prev, n("n1") + ".x")
    else:
        res(n("n1") + ".x", "x0.f32")
    res(n("n1") + ".w", "l1%d.f32" % i)
    program(n("c1"), "dit_to_f16", f"s={SP} n={D}"); link(n("n1") + ".y", n("c1") + ".x")
    for g, wf in (("gq", "wq"), ("gk", "wk"), ("gv", "wv")):
        program(n(g), "ported/gemm_mma", f"m={SP} n={D} k={D}")
        link(n("c1") + ".y", n(g) + ".a"); res(n(g) + ".w", "%s%d.f16" % (wf, i), "f16")
    for nm, src, wf in (("nq", "gq", "qn"), ("nk", "gk", "kn")):
        program(n(nm), "rmsnorm_rows", f"r={SP * H} n={HD} eps=0.000001")
        link(n(src) + ".y", n(nm) + ".x"); res(n(nm) + ".w", "%s%d.f32" % (wf, i))
    for rn, src in (("rq", "nq"), ("rk", "nk")):
        program(n(rn), "dit_rope", f"s={SP} nh={H} h=64")
        link(n(src) + ".y", n(rn) + ".x"); res(n(rn) + ".cs", "cs.f32"); res(n(rn) + ".sn", "sn.f32")
    program(n("sc"), "dit_scores", f"nh={H} nq={SP} t={SP} qs=32 ts=64 d={HD}")
    link(n("rq") + ".y", n("sc") + ".q"); link(n("rk") + ".y", n("sc") + ".k"); res(n("sc") + ".mask", "mask.f32")
    program(n("st"), "dit_stats", f"nh={H} nq={SP} t={SP} qs=32 cs=64"); link(n("sc") + ".sc", n("st") + ".sc")
    program(n("pv"), "dit_pv", f"nh={H} nq={SP} t={SP} qs=32 cs=64 dw=32 d={HD}")
    link(n("sc") + ".sc", n("pv") + ".sc"); link(n("st") + ".m", n("pv") + ".m"); link(n("st") + ".z", n("pv") + ".z"); link(n("gv") + ".y", n("pv") + ".v")
    program(n("c2"), "dit_to_f16", f"s={SP} n={D}"); link(n("pv") + ".o", n("c2") + ".x")
    program(n("go"), "ported/gemm_mma", f"m={SP} n={D} k={D}"); link(n("c2") + ".y", n("go") + ".a"); res(n("go") + ".w", "wo%d.f16" % i, "f16")
    program(n("ad1"), "dit_add", f"s={SP} n={D} bn=1024")
    if prev:
        link(prev, n("ad1") + ".a")
    else:
        res(n("ad1") + ".a", "x0.f32")
    link(n("go") + ".y", n("ad1") + ".b")
    program(n("n2"), "rmsnorm_rows", f"r={SP} n={D} eps=0.000001"); link(n("ad1") + ".z", n("n2") + ".x"); res(n("n2") + ".w", "l2%d.f32" % i)
    program(n("c3"), "dit_to_f16", f"s={SP} n={D}"); link(n("n2") + ".y", n("c3") + ".x")
    for g, wf in (("gg", "wg"), ("gu", "wu")):
        program(n(g), "ported/gemm_mma", f"m={SP} n={3 * D} k={D}")
        link(n("c3") + ".y", n(g) + ".a"); res(n(g) + ".w", "%s%d.f16" % (wf, i), "f16")
    program(n("sw"), "dit_swiglu", f"s={SP} n={3 * D} bn=1024"); link(n("gg") + ".y", n("sw") + ".g"); link(n("gu") + ".y", n("sw") + ".u")
    program(n("gd"), "ported/gemm_mma", f"m={SP} n={D} k={3 * D}"); link(n("sw") + ".y", n("gd") + ".a"); res(n("gd") + ".w", "wd%d.f16" % i, "f16")
    program(n("ad2"), "dit_add", f"s={SP} n={D} bn=1024"); link(n("ad1") + ".z", n("ad2") + ".a"); link(n("gd") + ".y", n("ad2") + ".b")
prog.append("steps")
for b in SHOW:
    prog.append("answer L%d_ad2.z" % (b - 1))
open(P("text.session"), "w").write("\n".join(prog) + "\n")

t0 = time.time()
r = subprocess.run([SESSION, P("text.session")], input="1\n", capture_output=True, text=True, timeout=7200)
secs = time.time() - t0
if r.returncode != 0 or not r.stdout.strip():
    print("FAIL the session did not answer: status %d, %s" % (r.returncode, r.stderr.strip()[:800]))
    sys.exit(1)
nums = np.array(r.stdout.split(), dtype=np.float64).reshape(len(SHOW), SP, D)[:, :N]
print("session: %d programs, answered in %.1f s (Metal; includes the compile and the upload of %.1f GB of weights)" % (n_prog[0], secs, NL * 0.4358))
ok = True
if refdir and os.path.exists(os.path.join(refdir, "layers_f64.npy")):
    f64 = np.load(os.path.join(refdir, "layers_f64.npy")).astype(np.float64)
    for j, b in enumerate(SHOW):
        d = np.abs(nums[j] - f64[b - 1])
        rms = float(np.sqrt((f64[b - 1] ** 2).mean()))
        print("residual after layer %2d: mean abs error %.4g (%.3g of the rms %.3g), max abs error %.4g" % (b, d.mean(), d.mean() / rms, rms, d.max()))
    exact = f64[NL - 1][drop:]
else:
    exact = None
got = nums[-1][drop:]
if "--save" in sys.argv:
    np.save(sys.argv[sys.argv.index("--save") + 1], got.astype(np.float32))  # the kept rows [21, 4096]: the DiT session's text embeddings (D6)
rms = float(np.sqrt((want ** 2).mean()))
rows = [("GPU encoder vs torch BF16 (text_embeds)", got, want)]
if exact is not None:
    rows = [("GPU encoder vs float64 encoder", got, exact)] + rows + [("(float64 encoder vs torch BF16, for scale)", exact, want)]
for label, a, b in rows:
    d = np.abs(a - b)
    print("%-44s mean abs error %.4g (%.3g of the rms %.3g), max abs error %.4g" % (label, d.mean(), d.mean() / rms, rms, d.max()))
ok = float(np.abs(got - (exact if exact is not None else want)).mean()) / rms < 0.05
print("%s the text encoder on the GPU (mean error under 5%% of the rms of the hidden states)" % ("ok  " if ok else "FAIL"))
sys.exit(0 if ok else 1)
