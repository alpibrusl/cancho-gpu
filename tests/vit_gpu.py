#!/usr/bin/env python3
"""D9b of docs/diffusion.md: the Qwen3-VL vision tower of Qwen-Image-2.1-Turbo on a Metal GPU, as four short sessions, held to the float64 tower (tools/vit_ref.py).

    vit_gpu.py WORK_DIR [--size N | --size HxW] [--ref DIR] [--time] [--mutate no-perm|keys-padded|scale-128]

Needs a Mac, the checkpoint, numpy, torch and safetensors (the venv of the first image edits), and `build/cancho-gpu-metal-session`. Not in `scripts/gate.sh`.

The image is synthetic (tools/vit_ref.py `synthetic`; the tower's arithmetic does not depend on the picture). `--ref DIR` reads patches, merged_f64.npy and deep_f64.npy saved by `vit_ref.py --save DIR`
(the float64 tower at 4,096 patches takes minutes); without it the float64 tower is computed here.

The layout, from the shapes (docs/diffusion.md D9a's plan): the patches are cut on the host (768 values each: the two time slices of the 3D convolution are the same pixels, so their kernels are summed), N
is padded with zero rows to a multiple of 256 (the merged rows, N / 4, to the matrix unit's 64), the padded rows are not keys (the mask says so) and are dropped from the comparison. **Each head's 72
channels are padded to 128 with zeros in the weights** (rows of q, k and v, columns of the output projection: exact), the second 36 channels of q and k placed at 64 to 99 so that dit_rope16's rotate-half
pairs (i, i + 64) are the model's pairs (i, i + 36); the score scale is 72^-1/2 (in dit_rope16's `sc` for q). The MLP's 4,304 is padded to 4,352 (zero rows of fc1 and zero columns of fc2: gelu(0) = 0).
The attention is D8's, unchanged: dit_scores_mma, dit_stats_m, dit_pv_mma at 16 heads and d = 128.
Sessions (a session keeps every output buffer, and one layer's scores are 16 x N x N x 4 B, 1.07 GB at N = 4,096): layers 0 to 8 and the first deepstack merger; 9 to 16 and the second; 17 to 24 and the third;
25 and 26 and the final merger. The residual stream goes from one to the next as a binary file.
"""
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import vit_ref as vr  # noqa: E402

arg = lambda name, default=None: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
work = sys.argv[1]
TIME = "--time" in sys.argv
MUT = arg("--mutate")  # a deliberate host-side bug, which the comparison must refuse
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
CHX = os.path.join(ROOT, "tests/chx")
NEG = -1e9
D, NH, HD, PHD, FF, FFP, DEPTH, DEEP = 1152, 16, 72, 128, 4304, 4352, 27, vr.DEEP
QD = NH * PHD  # 2048
os.makedirs(work, exist_ok=True)
P = lambda n: os.path.join(work, n)
f16 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f2")).tofile(path)
f32 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f4")).tofile(path)

sz = arg("--size", "256x384")
IH, IW = (int(a) for a in (sz.split("x") if "x" in sz else (sz, sz)))
img = vr.synthetic(IH, IW)
pat, h, w = vr.patches(img)
N = h * w
NP = (N + 255) // 256 * 256
MP = NP // 4
print("image %d x %d: %d patches (padded to %d), %d merged tokens (padded to %d)" % (IH, IW, N, NP, N // 4, MP), flush=True)

# ---- the float64 reference
if arg("--ref"):
    rd = arg("--ref")
    assert np.abs(np.load(rd + "/patches.npy") - pat).max() < 1e-12, "--ref was saved for another image"
    want, want_deep = np.load(rd + "/merged_f64.npy"), np.load(rd + "/deep_f64.npy")
else:
    t0 = time.time()
    want, want_deep, _, _ = vr.tower(img)
    print("float64 tower in %.0f s" % (time.time() - t0), flush=True)

# ---- the data that does not change
# the patch matrix: the 768 values of the first time slice (channels, rows, columns); the second slice is the same pixels
pm = np.zeros((NP, 768))
pm[:N] = pat.reshape(N, 3, 2, 256)[:, :, 0, :].reshape(N, 768)
f16(P("patch.f16"), pm)
wpe = vr.W("patch_embed.proj.weight").sum(2).reshape(D, 768)  # [D, C, T, P, P] summed over T
f16(P("wpe.f16"), wpe)
posb = np.zeros((NP, D))
posb[:N] = vr.pos_table(h, w) + vr.W("patch_embed.proj.bias")
f32(P("posb.f32"), posb)
cos, sin = vr.rope_tables(h, w)
cs, sn = np.ones((NP, 64)), np.zeros((NP, 64))
cs[:N, :36], sn[:N, :36] = cos[:, :36], sin[:, :36]
f32(P("cs.f32"), cs)
f32(P("sn.f32"), sn)
mask = np.zeros((NP, NP))
if MUT != "keys-padded":
    mask[:, N:] = NEG
f32(P("mask.f32"), mask)
f32(P("ones_n.f32"), np.ones((NP, 1)))
f32(P("ones_m.f32"), np.ones((MP, 1)))
pad_to = np.arange(HD) if MUT == "no-perm" else np.array(list(range(36)) + list(range(64, 100)))  # the position of a channel of q or k: first half stays, second half at 64


def pad_rows(wt, idx):  # [NH * 72, in] -> [NH * 128, in]
    o = np.zeros((NH, PHD, wt.shape[-1]))
    o[:, idx, :] = wt.reshape(NH, HD, -1)
    return o.reshape(NH * PHD, -1)


def pad_vec(v, idx):
    o = np.zeros((NH, PHD))
    o[:, idx] = v.reshape(NH, HD)
    return o.reshape(1, -1)


if not os.path.exists(P("w_done")):
    t0 = time.time()
    ident = np.arange(HD)
    for b in range(DEPTH):
        W = lambda k: vr.W("blocks.%d.%s" % (b, k))
        qkv, qkvb = W("attn.qkv.weight"), W("attn.qkv.bias")
        for i, (nm, idx) in enumerate((("q", pad_to), ("k", pad_to), ("v", ident))):
            f16(P("w%s%d.f16" % (nm, b)), pad_rows(qkv[i * D:(i + 1) * D], idx))
            f32(P("b%s%d.f32" % (nm, b)), pad_vec(qkvb[i * D:(i + 1) * D], idx))
        po = np.zeros((D, NH, PHD))
        po[:, :, :HD] = W("attn.proj.weight").reshape(D, NH, HD)
        f16(P("wo%d.f16" % b), po.reshape(D, QD))
        f32(P("bo%d.f32" % b), W("attn.proj.bias")[None])
        for k in ("norm1", "norm2"):
            f32(P("%sw%d.f32" % (k, b)), W(k + ".weight")[None])
            f32(P("%sb%d.f32" % (k, b)), W(k + ".bias")[None])
        f1 = np.zeros((FFP, D))
        f1[:FF] = W("mlp.linear_fc1.weight")
        f16(P("w1_%d.f16" % b), f1)
        b1 = np.zeros((1, FFP))
        b1[0, :FF] = W("mlp.linear_fc1.bias")
        f32(P("b1_%d.f32" % b), b1)
        f2 = np.zeros((D, FFP))
        f2[:, :FF] = W("mlp.linear_fc2.weight")
        f16(P("w2_%d.f16" % b), f2)
        f32(P("b2_%d.f32" % b), W("mlp.linear_fc2.bias")[None])
    for nm in ["merger"] + ["deepstack_merger_list.%d" % i for i in range(3)]:
        tag = nm.replace("deepstack_merger_list.", "ds")
        f32(P("%s_nw.f32" % tag), vr.W(nm + ".norm.weight")[None])
        f32(P("%s_nb.f32" % tag), vr.W(nm + ".norm.bias")[None])
        f16(P("%s_w1.f16" % tag), vr.W(nm + ".linear_fc1.weight"))
        f32(P("%s_b1.f32" % tag), vr.W(nm + ".linear_fc1.bias")[None])
        f16(P("%s_w2.f16" % tag), vr.W(nm + ".linear_fc2.weight"))
        f32(P("%s_b2.f32" % tag), vr.W(nm + ".linear_fc2.bias")[None])
    open(P("w_done"), "w").write("ok")
    print("weights written in %.0f s" % (time.time() - t0), flush=True)


class Session:
    def __init__(self, name):
        self.name, self.lines = name, []

    def program(self, n, kernel, args):
        self.lines.append(f"program {n} {CHX}/{kernel}.chx {args}")

    def res(self, target, fname, flag="f32"):
        self.lines.append(f"resident {target} {P(fname)} {flag}")

    def link(self, a, b):
        self.lines.append(f"link {a} -> {b}")

    def save(self, out, fname, flag):
        self.lines.append(f"save {out} {P(fname)} {flag}")

    def run(self, probe_of, probe_m, probe_n):
        self.program("probe", "dit_rows2", f"m={probe_m} n={probe_n} bn={next(b for b in (1024, 512, 384, 256, 128, 64, 32, 16, 8, 4, 2, 1) if probe_n % b == 0)}")
        self.link(probe_of, "probe.x")
        open(P(self.name + ".session"), "w").write("\n".join(self.lines) + "\nsteps\nanswer probe.y\n")
        t0 = time.time()
        p = subprocess.Popen([SESSION, P(self.name + ".session")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        reqs = []
        for line in (["1\n", "1\n"] if TIME else ["1\n"]) + ["checkpoint\n"]:
            t1 = time.perf_counter()
            p.stdin.write(line)
            p.stdin.flush()
            out = p.stdout.readline()
            reqs.append(time.perf_counter() - t1)
            if not out:
                raise SystemExit("session %s stopped: %s" % (self.name, p.stderr.read()[:800]))
        p.stdin.close()
        p.wait()
        return reqs, time.time() - t0


def merger(s, tag, src, nm, dst, postshuffle):
    """A merger on the f32 stream `src` ([NP, 1152]): its layer norm, two products with a GELU between, the bias; saved as `dst` ([MP, 4096] f32)."""
    if postshuffle:
        s.program(nm + "_ln", "vit_ln", f"s={MP} n=4608")  # the rows are four adjacent patches: the same memory read as [MP, 4608]
    else:
        s.program(nm + "_ln", "vit_ln", f"s={NP} n={D}")
    s.link(src, nm + "_ln.x"); s.res(nm + "_ln.w", tag + "_nw.f32"); s.res(nm + "_ln.b", tag + "_nb.f32")
    s.program(nm + "_g1", "ported/gemm_mma", f"m={MP} n=4608 k=4608"); s.link(nm + "_ln.y", nm + "_g1.a"); s.res(nm + "_g1.w", tag + "_w1.f16", "f16")
    s.program(nm + "_ge", "vit_bias_gelu_erf", f"s={MP} n=4608 bn=512 m1=-1"); s.link(nm + "_g1.y", nm + "_ge.x"); s.res(nm + "_ge.b", tag + "_b1.f32")
    s.program(nm + "_g2", "ported/gemm_mma", f"m={MP} n=4096 k=4608"); s.link(nm + "_ge.y", nm + "_g2.a"); s.res(nm + "_g2.w", tag + "_w2.f16", "f16")
    s.program(nm + "_b2", "vae_bias", f"r={MP} c=4096"); s.link(nm + "_g2.y", nm + "_b2.x"); s.res(nm + "_b2.b", tag + "_b2.f32"); s.res(nm + "_b2.m", "ones_m.f32")
    s.save(nm + "_b2.z", dst, "f32")


def layers(name, lo, hi, first, deep, final):
    s = Session(name)
    prev = None
    if first:
        s.program("pe", "ported/gemm_mma", f"m={NP} n={D} k=768"); s.res("pe.a", "patch.f16", "f16"); s.res("pe.w", "wpe.f16", "f16")
        s.program("pa", "dit_add", f"s={NP} n={D} bn=128"); s.link("pe.y", "pa.a"); s.res("pa.b", "posb.f32")
        prev = "pa.z"
    for b in range(lo, hi):
        n = lambda t: "b%d_%s" % (b, t)

        def src(target):
            if prev:
                s.link(prev, target)
            else:
                s.res(target, "x_in.bin")
        s.program(n("ln1"), "vit_ln", f"s={NP} n={D}"); src(n("ln1") + ".x"); s.res(n("ln1") + ".w", "norm1w%d.f32" % b); s.res(n("ln1") + ".b", "norm1b%d.f32" % b)
        for nm in ("q", "k", "v"):
            s.program(n("g" + nm), "ported/gemm_mma", f"m={NP} n={QD} k={D}"); s.link(n("ln1") + ".y", n("g" + nm) + ".a"); s.res(n("g" + nm) + ".w", "w%s%d.f16" % (nm, b), "f16")
            s.program(n("b" + nm), "vae_bias", f"r={NP} c={QD}"); s.link(n("g" + nm) + ".y", n("b" + nm) + ".x"); s.res(n("b" + nm) + ".b", "b%s%d.f32" % (nm, b)); s.res(n("b" + nm) + ".m", "ones_n.f32")
        for rn, srcn, sc_ in (("rq", "bq", repr(1 / math.sqrt(PHD if MUT == "scale-128" else HD))), ("rk", "bk", "1.0")):
            s.program(n(rn), "dit_rope16", f"s={NP} nh={NH} h=64 sc={sc_}"); s.link(n(srcn) + ".z", n(rn) + ".x"); s.res(n(rn) + ".cs", "cs.f32"); s.res(n(rn) + ".sn", "sn.f32")
        s.program(n("sc"), "dit_scores_mma", f"nh={NH} nq={NP} t={NP} d={PHD}"); s.link(n("rq") + ".y", n("sc") + ".q"); s.link(n("rk") + ".y", n("sc") + ".k")
        s.program(n("st"), "dit_stats_m", f"nh={NH} nq={NP} t={NP} qs=32 cs=64"); s.link(n("sc") + ".sc", n("st") + ".sc"); s.res(n("st") + ".mask", "mask.f32")
        s.program(n("vh"), "dit_to_f16", f"s={NP} n={QD}"); s.link(n("bv") + ".z", n("vh") + ".x")
        s.program(n("pv"), "dit_pv_mma", f"nh={NH} nq={NP} t={NP} d={PHD}")
        s.link(n("sc") + ".sc", n("pv") + ".sc"); s.res(n("pv") + ".mask", "mask.f32"); s.link(n("st") + ".m", n("pv") + ".m"); s.link(n("st") + ".z", n("pv") + ".z"); s.link(n("vh") + ".y", n("pv") + ".v")
        s.program(n("c1"), "dit_to_f16", f"s={NP} n={QD}"); s.link(n("pv") + ".o", n("c1") + ".x")
        s.program(n("go"), "ported/gemm_mma", f"m={NP} n={D} k={QD}"); s.link(n("c1") + ".y", n("go") + ".a"); s.res(n("go") + ".w", "wo%d.f16" % b, "f16")
        s.program(n("r1"), "vae_bias_res", f"r={NP} c={D}"); s.link(n("go") + ".y", n("r1") + ".x"); s.res(n("r1") + ".b", "bo%d.f32" % b); s.res(n("r1") + ".m", "ones_n.f32"); src(n("r1") + ".res")
        s.program(n("ln2"), "vit_ln", f"s={NP} n={D}"); s.link(n("r1") + ".z", n("ln2") + ".x"); s.res(n("ln2") + ".w", "norm2w%d.f32" % b); s.res(n("ln2") + ".b", "norm2b%d.f32" % b)
        s.program(n("f1"), "ported/gemm_mma", f"m={NP} n={FFP} k={D}"); s.link(n("ln2") + ".y", n("f1") + ".a"); s.res(n("f1") + ".w", "w1_%d.f16" % b, "f16")
        s.program(n("ge"), "vit_bias_gelu", f"s={NP} n={FFP} bn=256"); s.link(n("f1") + ".y", n("ge") + ".x"); s.res(n("ge") + ".b", "b1_%d.f32" % b)
        s.program(n("f2"), "ported/gemm_mma", f"m={NP} n={D} k={FFP}"); s.link(n("ge") + ".y", n("f2") + ".a"); s.res(n("f2") + ".w", "w2_%d.f16" % b, "f16")
        s.program(n("r2"), "vae_bias_res", f"r={NP} c={D}"); s.link(n("f2") + ".y", n("r2") + ".x"); s.res(n("r2") + ".b", "b2_%d.f32" % b); s.res(n("r2") + ".m", "ones_n.f32"); s.link(n("r1") + ".z", n("r2") + ".res")
        prev = n("r2") + ".z"
        if b in DEEP:
            merger(s, "ds%d" % DEEP.index(b), prev, "ds%d" % DEEP.index(b), "deep%d.bin" % DEEP.index(b), True)
    if final:
        merger(s, "merger", prev, "mg", "merged.bin", False)
    s.save(prev, "x_out_%s.bin" % name, "f32")
    return s.run(prev, NP, D)


plan = [("s1", 0, 9, True, False), ("s2", 9, 17, False, False), ("s3", 17, 25, False, False), ("s4", 25, 27, False, True)]
total = 0.0
xin = None
for name, lo, hi, first, final in plan:
    if xin:
        os.replace(P(xin), P("x_in.bin"))
    reqs, whole = layers(name, lo, hi, first, None, final)
    steady = reqs[-2] if TIME else reqs[0]
    total += steady
    print("session %s (layers %d to %d): %.2f s %s, %.1f s with the start" % (name, lo, hi - 1, steady, "steady state" if TIME else "first request (includes the compile and upload)", whole), flush=True)
    xin = "x_out_%s.bin" % name
if TIME:
    print("TIME the tower (steady state, weights resident): %.2f s" % total)

got = np.fromfile(P("merged.bin"), dtype="<f4").reshape(MP, 4096)[:N // 4].astype(np.float64)
gd = np.stack([np.fromfile(P("deep%d.bin" % i), dtype="<f4").reshape(MP, 4096)[:N // 4].astype(np.float64) for i in range(3)])
rms = float(np.sqrt((want ** 2).mean()))
e = np.abs(got - want)
print("GPU vs float64: merged mean abs error %.3g (%.3g of the rms %.3g), max %.3g" % (e.mean(), e.mean() / rms, rms, e.max()))
for i in range(3):
    d = np.abs(gd[i] - want_deep[i])
    print("   deepstack %d: mean abs error %.3g (%.3g of its rms %.3g), max %.3g" % (i, d.mean(), d.mean() / np.sqrt((want_deep[i] ** 2).mean()), np.sqrt((want_deep[i] ** 2).mean()), d.max()))
BOUND = 5e-3  # a tenth of torch's BF16 floor on this tower (D9a: 4.6e-2 of the rms): the GPU must be at least ten times closer to float64 than torch's BF16 is
ok = e.mean() / rms <= BOUND and all(np.abs(gd[i] - want_deep[i]).mean() / np.sqrt((want_deep[i] ** 2).mean()) <= BOUND for i in range(3))
print("%s the vision tower on the GPU: mean abs error within %.0e of the rms (a tenth of torch's BF16 floor, 4.6e-2 in D9a)" % ("ok  " if ok else "FAIL", BOUND))
sys.exit(0 if ok else 1)
