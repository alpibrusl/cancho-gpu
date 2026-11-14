#!/usr/bin/env python3
"""D9c of docs/diffusion.md: the text encoder of Qwen-Image-2.1-Turbo reading an image, on a Metal GPU in one session, held to the float64 run of tools/text_img_ref.py.

    text_img_gpu.py REF_DIR WEIGHTS_DIR [--mutate pos1d|no-deepstack] [--time N] [--no-ref] [--save OUT.npy]

`--no-ref`: REF_DIR holds only ids.npy, positions.npy, embeds_in.npy and deep_f64.npy (built from the GPU's own vision tower by tests/edit_pipeline_gpu.py); nothing is compared, `--save` writes the rows from `drop_idx`.

REF_DIR is the output of `python tools/text_img_ref.py REF_DIR [--size HxW]` (ids, mRoPE positions, the input embeddings with the vision tower's rows in place, the deepstack maps, the float64 residual stream after
each layer). WEIGHTS_DIR holds the f16 weight files (written on the first run, 15.7 GB, and reused). Needs a Mac, the checkpoint, and `build/cancho-gpu-metal-session`. Not in `scripts/gate.sh`.

It is D5's session (tests/text_gpu.py) with no new kernel: the same 36 layers over N rows padded to 64, with (1) the rotary tables per row from the mRoPE positions (dit_rope already reads a table per row), (2) the input
rows the host built (token embeddings, the tower's merged embeddings at the image-pad rows), and (3) three `dit_add` programs after layers 0, 1 and 2 that add the deepstack maps (zero except at the image-pad rows).
Held to: the float64 residual stream, mean abs error at most 5% of its rms (D5's bound for the language model: torch's BF16 is at 2.1%, the f16 GPU path below it).
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_ref as tr  # noqa: E402
import text_img_ref as ti  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
refdir, work = sys.argv[1], sys.argv[2]
MUT = sys.argv[sys.argv.index("--mutate") + 1] if "--mutate" in sys.argv else None  # a deliberate host-side bug, which the comparison must refuse
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
CHX = os.path.join(ROOT, "tests/chx")
NEG = -1e9
D, H, KV, HD, NL = 4096, 32, 8, 128, 36
ids = np.load(os.path.join(refdir, "ids.npy"))
pos = np.load(os.path.join(refdir, "positions.npy"))
x_in = np.load(os.path.join(refdir, "embeds_in.npy")).astype(np.float64)
deep = np.load(os.path.join(refdir, "deep_f64.npy"))
NOREF = "--no-ref" in sys.argv
after = None if NOREF else np.load(os.path.join(refdir, "after_f64.npy"))
pad_rows = np.nonzero(ids == ti.IMG_PAD)[0]
drop = 14
N = len(ids)
SP = (N + 63) // 64 * 64
os.makedirs(work, exist_ok=True)
P = lambda n: os.path.join(work, n)
f16 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f2")).tofile(path)
f32 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f4")).tofile(path)
Wf = lambda name: tr._f.get_tensor("model.language_model." + name).float().numpy()
print("%d tokens (%d image pads) padded to %d rows" % (N, len(pad_rows), SP), flush=True)

x0 = np.zeros((SP, D))
x0[:N] = x_in
f32(P("x0.f32"), x0)
if MUT == "pos1d":
    pos = np.arange(N)[None].repeat(3, 0)  # the text model's ordinary 1D positions: what D5 did, wrong for an image
cos, sin = ti.tables(pos)
cs, sn = np.ones((SP, 64)), np.zeros((SP, 64))
cs[:N], sn[:N] = cos, sin
f32(P("cs.f32"), cs)
f32(P("sn.f32"), sn)
for l in range(3):
    d = np.zeros((SP, D))
    if MUT != "no-deepstack":
        d[pad_rows] = deep[l]
    f32(P("ds%d.f32" % l), d)
idx = np.arange(SP)
f32(P("mask.f32"), np.where((idx[:, None] >= idx[None, :]) & (idx < N)[None, :], 0.0, NEG))

t0 = time.time()
NL_W = 0 if os.path.exists(P("w_done")) else NL
rep = lambda m: np.repeat(m.reshape(KV, HD, D), H // KV, axis=0).reshape(H * HD, D)
for i in range(NL_W):
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
if NL_W:
    open(P("w_done"), "w").write("ok")
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
    prev = None if i == 0 else ("L%d_dsa.z" if i - 1 < 3 else "L%d_ad2.z") % (i - 1)
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
    if i < 3:  # deepstack: add the i-th vision feature map at the image-pad rows
        program(n("dsa"), "dit_add", f"s={SP} n={D} bn=1024"); link(n("ad2") + ".z", n("dsa") + ".a"); res(n("dsa") + ".b", "ds%d.f32" % i)
    if i > 0 and "--no-alias" not in sys.argv:
        # layer i's activations live in layer i - 1's buffers (docs/resident.md 21.4, as in D9i): the session file's checks prove nothing reads them any more. The residual stream (ad2.z and dsa.z,
        # which the answers read) keeps its own. Without this the session touches 14 GB of fresh activation buffers on its first request (D9k)
        for sfx, o in (("n1", "y"), ("c1", "y"), ("gq", "y"), ("gk", "y"), ("gv", "y"), ("nq", "y"), ("nk", "y"), ("rq", "y"), ("rk", "y"), ("sc", "sc"), ("st", "m"), ("st", "z"), ("pv", "o"), ("c2", "y"),
                       ("go", "y"), ("ad1", "z"), ("n2", "y"), ("c3", "y"), ("gg", "y"), ("gu", "y"), ("sw", "y"), ("gd", "y")):
            prog.append("alias L%d_%s.%s L%d_%s.%s" % (i - 1, sfx, o, i, sfx, o))
TIME = int(sys.argv[sys.argv.index("--time") + 1]) if "--time" in sys.argv else 0
if TIME:
    program("probe", "dit_rows2", f"m={SP} n={D} bn=1024"); link("L%d_ad2.z" % (NL - 1), "probe.x")
if NOREF and not TIME:
    # the pipeline asks for the last layer only: a probe answers (one line), the layer goes out as a binary file on `checkpoint` instead of as 1088 x 4096 numbers of text (D12)
    program("probe", "dit_rows2", f"m={SP} n={D} bn=1024"); link("L%d_ad2.z" % (NL - 1), "probe.x")
    prog.append(f"save L{NL - 1}_ad2.z {P('last.f32')} f32")
prog.append("steps")
if NOREF and not TIME:
    prog.append("answer probe.y")
elif TIME:
    prog.append("answer probe.y")
else:
    for b in SHOW:
        prog.append("answer L%d_%s.z" % (b - 1, "dsa" if b - 1 < 3 else "ad2"))
open(P("text.session"), "w").write("\n".join(prog) + "\n")

if TIME:
    import session_time
    ts = session_time.timed(SESSION, P("text.session"), TIME)
    print("TIME text encoder reading an image, N tokens padded to a multiple of 64: first request %.3f s (device, compile, upload); then median %.4f s (min %.4f, max %.4f) over %d" % (ts[0], sorted(ts[1:])[len(ts[1:]) // 2], min(ts[1:]), max(ts[1:]), len(ts) - 1))
    sys.exit(0)
t0 = time.time()
r = subprocess.run([SESSION, P("text.session")], input="1\ncheckpoint\n" if NOREF else "1\n", capture_output=True, text=True, timeout=7200)
secs = time.time() - t0
if r.returncode != 0 or not r.stdout.strip():
    print("FAIL the session did not answer: status %d, %s" % (r.returncode, r.stderr.strip()[:800]))
    sys.exit(1)
if NOREF:
    nums = np.fromfile(P("last.f32"), dtype="<f4").reshape(1, SP, D)[:, :N].astype(np.float64)
else:
    nums = np.array(r.stdout.split(), dtype=np.float64).reshape(len(SHOW), SP, D)[:, :N]
print("session: %d programs, answered in %.1f s (Metal; includes the compile and the upload of %.1f GB of weights)" % (n_prog[0], secs, NL * 0.4358))
got = nums[-1][drop:]
if "--save" in sys.argv:
    np.save(sys.argv[sys.argv.index("--save") + 1], got.astype(np.float32))
if NOREF:
    print("text encoder done (no reference): %d rows kept" % got.shape[0])
    sys.exit(0)
rms = float(np.sqrt((after[NL - 1][drop:] ** 2).mean()))
txt_rows = np.ones(N - drop, dtype=bool)
txt_rows[np.nonzero(ids[drop:] == ti.IMG_PAD)[0]] = False
for j, b in enumerate(SHOW):
    d = np.abs(nums[j] - after[b - 1])
    r = float(np.sqrt((after[b - 1] ** 2).mean()))
    print("residual after layer %2d: mean abs error %.4g (%.3g of the rms %.3g), max abs error %.4g" % (b, d.mean(), d.mean() / r, r, d.max()))
got = nums[-1][drop:]
exact = after[NL - 1][drop:]
d = np.abs(got - exact)
print("%-40s mean abs error %.4g (%.3g of the rms %.3g), max %.4g" % ("output, all rows", d.mean(), d.mean() / rms, rms, d.max()))
dt = d[txt_rows]
print("%-40s mean abs error %.4g (%.3g of the rms): the rows the transformer reads" % ("output, text rows only", dt.mean(), dt.mean() / rms))
if "--save" in sys.argv:
    np.save(sys.argv[sys.argv.index("--save") + 1], got.astype(np.float32))
ok = float(dt.mean()) / rms < 0.05
print("%s the text encoder reading an image on the GPU (text rows: mean error under 5%% of the rms)" % ("ok  " if ok else "FAIL"))
sys.exit(0 if ok else 1)
