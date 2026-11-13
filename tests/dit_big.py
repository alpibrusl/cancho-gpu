#!/usr/bin/env python3
"""D7 of docs/diffusion.md: the eight denoising steps of Qwen-Image-2.1-Turbo at any size on a Metal GPU, as many short sessions, because the 32 blocks of one session do not fit in memory at 1024 x 1024.

    dit_big.py GOLDEN_DIR WORK_DIR [--text TEXT.npy] [--steps N] [--group G] [--time] [--ref TRAJ_DIR]

Why sessions: a session allocates a buffer for every output of every program and gives none back, so 32 blocks keep 32 attention score matrices of 32 x 4160 x 4160 x 4 B = 2.2 GB each at 1024 x 1024 (71 GB in all), and a table
(the way a session reads a different weight each repetition) is capped at the session's 512 MiB byte pool, against 33.5 MB to 100 MB of f16 weights a layer. So a step is: one **front** session (txt_in, img_in, the timestep
embedding, the modulation), `32 / G` **group** sessions of G blocks (default 4), and one **back** session (norm_out, proj_out, the Euler update); the residual stream, the modulation, the embedding's SiLU and the latents go from one to the next through binary files
(`save` and `checkpoint`, then `resident ... f32`). The kernels are exactly those of tests/dit_loop_gpu.py (D3), whose single-session run at 256 x 256 this must reproduce; the weights are written once (14.3 GB) and each group session uploads its own 1.75 GB.

`--time`: each session is driven interactively (tests/session_time.py's way) and the second of two identical requests is timed: the steady state with the weights on the device (the first request's time is the cold start). A step's time is then the sum of its sessions'
steady-state request times: the front, `32 / G` groups and the back. The host's file writing and reading between sessions, and each session's start (device, compile, upload), are reported separately and are not in that sum.
"""
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dit_block_ref as ref  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gold, work = sys.argv[1], sys.argv[2]
arg = lambda name, default=None: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
NSTEPS = int(arg("--steps", 8))
G = int(arg("--group", 4))
TIME = "--time" in sys.argv
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
CHX = os.path.join(ROOT, "tests/chx")
NEG = -1e9
NB, D, H, HD = 32, 4096, 32, 128
perm = np.array(list(range(0, HD, 2)) + list(range(1, HD, 2)))
os.makedirs(work, exist_ok=True)
P = lambda n: os.path.join(work, n)
ld = lambda n: np.load(os.path.join(gold, n + ".npy")).astype(np.float64)
f16 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f2")).tofile(path)
f32 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f4")).tofile(path)

_, _, _, rope, tmask = ref.goldens(gold)
S = rope.shape[0]
SP = (S + 63) // 64 * 64
NIMG = S - 21
SIDE = int(round(math.sqrt(NIMG)))
text = np.load(arg("--text")).astype(np.float64) if "--text" in sys.argv else ld("text_embeds")[0]
sigmas, timesteps = ld("sigmas"), ld("timesteps")
print("tokens %d (21 text + %d image, %d x %d), padded to %d" % (S, NIMG, SIDE, SIDE, SP), flush=True)

# ---- the data that does not change between steps
sel = np.zeros((SP, 1))
sel[:S, 0] = tmask
f32(P("sel.f32"), sel)
cs, sn = np.ones((SP, 64)), np.zeros((SP, 64))
cs[:S], sn[:S] = rope[..., 0], rope[..., 1]
f32(P("cs.f32"), cs)
f32(P("sn.f32"), sn)
idx = np.arange(SP)
tm = np.zeros(SP, dtype=bool)
tm[:S] = tmask
f32(P("mask.f32"), np.where(((idx[:, None] >= idx[None, :]) | (tm[:, None] & tm[None, :])) & (idx < S)[None, :], 0.0, NEG))
txt = np.zeros((SP, D))
txt[:text.shape[0]] = text
f32(P("txt.f32"), txt)
f32(P("tw1.f32"), (ref.weight("txt_in.text_norm.weight").double().numpy() + 1)[None])
lat = np.zeros((SP, 64))
lat[21:S] = ld("step0_in")[0]
f32(P("lat.f32"), lat)
f16(P("lat.f16"), lat)
freqs = np.exp(-math.log(10000) * np.arange(128) / 128)
for name, key in (("wti", "txt_in.in_layer.weight"), ("wto", "txt_in.out_layer.weight"), ("wim", "img_in.weight"), ("wt1", "time_text_embed.timestep_embedder.linear_1.weight"),
                  ("wt2", "time_text_embed.timestep_embedder.linear_2.weight"), ("wmo", "modulation.1.weight"), ("wno", "norm_out.linear.weight"), ("wpj", "proj_out.weight")):
    f16(P(name + ".f16"), ref.weight(key).float().numpy())
if not os.path.exists(P("w_done")):
    t0 = time.time()
    head_perm = lambda m: m.reshape(H, HD, -1)[:, perm, :].reshape(H * HD, -1)
    for b in range(NB):
        t = lambda k: ref.weight("transformer_blocks.%d.%s" % (b, ref.BLOCK0[k])).float().numpy()
        f16(P("wq%d.f16" % b), head_perm(t("q")))
        f16(P("wk%d.f16" % b), head_perm(t("k")))
        for k, n in (("v", "wv"), ("o", "wo"), ("g", "wg"), ("p", "wp"), ("out", "wout")):
            f16(P("%s%d.f16" % (n, b)), t(k))
        f32(P("nq%d.f32" % b), t("nq")[perm][None])
        f32(P("nk%d.f32" % b), t("nk")[perm][None])
    open(P("w_done"), "w").write("ok")
    print("weights written: %d blocks in %.0f s" % (NB, time.time() - t0), flush=True)


# ---- sessions
class Session:
    def __init__(self, name):
        self.name, self.lines, self.saves = name, [], []

    def program(self, n, kernel, args):
        self.lines.append(f"program {n} {CHX}/{kernel}.chx {args}")

    def res(self, target, fname, flag="f32"):
        self.lines.append(f"resident {target} {P(fname)} {flag}")

    def link(self, a, b):
        self.lines.append(f"link {a} -> {b}")

    def save(self, out, fname, flag):
        self.lines.append(f"save {out} {P(fname)} {flag}")

    def run(self, probe_of, probe_m, probe_n, tag):
        """Add a probe as the answer, run one request and the checkpoint; returns (seconds of the request, seconds of the whole session incl. start)."""
        self.program("probe", "dit_rows2", f"m={probe_m} n={probe_n} bn={min(probe_n, 1024)}")
        self.link(probe_of, "probe.x")
        text = "\n".join(self.lines) + "\nsteps\nanswer probe.y\n"
        path = P(self.name + ".session")
        open(path, "w").write(text)
        t0 = time.time()
        p = subprocess.Popen([SESSION, path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
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


def front(i):
    s = Session("front")
    s.program("tn", "rmsnorm_rows", f"r={SP} n={D} eps=0.000001"); s.res("tn.x", "txt.f32"); s.res("tn.w", "tw1.f32")
    s.program("tnc", "dit_to_f16", f"s={SP} n={D}"); s.link("tn.y", "tnc.x")
    s.program("ti", "ported/gemm_mma", f"m={SP} n={D} k={D}"); s.link("tnc.y", "ti.a"); s.res("ti.w", "wti.f16", "f16")
    s.program("tg", "dit_gelu", f"s={SP} n={D} bn=1024"); s.link("ti.y", "tg.x")
    s.program("to", "ported/gemm_mma", f"m={SP} n={D} k={D}"); s.link("tg.y", "to.a"); s.res("to.w", "wto.f16", "f16")
    s.program("ii", "ported/gemm_mma", f"m={SP} n={D} k=64"); s.res("ii.a", "lat.f16", "f16"); s.res("ii.w", "wim.f16", "f16")
    s.program("jn", "dit_add", f"s={SP} n={D} bn=1024"); s.link("to.y", "jn.a"); s.link("ii.y", "jn.b")
    s.program("t1", "ported/gemm_mma", f"m=64 n={D} k=256"); s.res("t1.a", "emb%d.f16" % i, "f16"); s.res("t1.w", "wt1.f16", "f16")
    s.program("ts", "dit_silu", f"s=64 n={D} bn=1024"); s.link("t1.y", "ts.x")
    s.program("t2", "ported/gemm_mma", f"m=64 n={D} k={D}"); s.link("ts.y", "t2.a"); s.res("t2.w", "wt2.f16", "f16")
    s.program("tsil", "dit_silu", f"s=64 n={D} bn=1024"); s.link("t2.y", "tsil.x")
    s.program("mo", "ported/gemm_mma", f"m=64 n={4 * D} k={D}"); s.link("tsil.y", "mo.a"); s.res("mo.w", "wmo.f16", "f16")
    s.program("ms", "dit_modsplit", f"m=64 n={D} bn=1024"); s.link("mo.y", "ms.md")
    s.save("jn.z", "x_in.bin", "f32")
    for o in ("s1", "g1", "s2", "g2"):
        s.save("ms." + o, "ms_%s.bin" % o, "f32")
    s.save("tsil.y", "tsil.bin", "f16")
    return s.run("jn.z", SP, D, "front")


def group(g, xfile, outfile):
    s = Session("group")
    prev = None
    for b in range(g * G, (g + 1) * G):
        n = lambda t: "b%d_%s" % (b, t)
        s.program(n("lnm1"), "dit_lnmod", f"s={SP} n={D}")
        if prev:
            s.link(prev, n("lnm1") + ".x")
        else:
            s.res(n("lnm1") + ".x", xfile)
        s.res(n("lnm1") + ".sc", "ms_s1.bin"); s.res(n("lnm1") + ".sel", "sel.f32")
        for gg, wf in (("gq", "wq"), ("gk", "wk"), ("gv", "wv")):
            s.program(n(gg), "ported/gemm_mma", f"m={SP} n={D} k={D}")
            s.link(n("lnm1") + ".y", n(gg) + ".a"); s.res(n(gg) + ".w", "%s%d.f16" % (wf, b), "f16")
        for nm, src, wf in (("nq", "gq", "nq"), ("nk", "gk", "nk")):
            s.program(n(nm), "rmsnorm_rows", f"r={SP * H} n={HD} eps=0.000001")
            s.link(n(src) + ".y", n(nm) + ".x"); s.res(n(nm) + ".w", "%s%d.f32" % (wf, b))
        for rn, src in (("rq", "nq"), ("rk", "nk")):
            s.program(n(rn), "dit_rope", f"s={SP} nh={H} h=64")
            s.link(n(src) + ".y", n(rn) + ".x"); s.res(n(rn) + ".cs", "cs.f32"); s.res(n(rn) + ".sn", "sn.f32")
        s.program(n("sc"), "dit_scores", f"nh={H} nq={SP} t={SP} qs=32 ts=64 d={HD}")
        s.link(n("rq") + ".y", n("sc") + ".q"); s.link(n("rk") + ".y", n("sc") + ".k"); s.res(n("sc") + ".mask", "mask.f32")
        s.program(n("st"), "dit_stats", f"nh={H} nq={SP} t={SP} qs=32 cs=64"); s.link(n("sc") + ".sc", n("st") + ".sc")
        s.program(n("pv"), "dit_pv", f"nh={H} nq={SP} t={SP} qs=32 cs=64 dw=32 d={HD}")
        s.link(n("sc") + ".sc", n("pv") + ".sc"); s.link(n("st") + ".m", n("pv") + ".m"); s.link(n("st") + ".z", n("pv") + ".z"); s.link(n("gv") + ".y", n("pv") + ".v")
        s.program(n("c1"), "dit_to_f16", f"s={SP} n={D}"); s.link(n("pv") + ".o", n("c1") + ".x")
        s.program(n("go"), "ported/gemm_mma", f"m={SP} n={D} k={D}"); s.link(n("c1") + ".y", n("go") + ".a"); s.res(n("go") + ".w", "wo%d.f16" % b, "f16")
        s.program(n("gr1"), "dit_gate_res", f"s={SP} n={D}")
        if prev:
            s.link(prev, n("gr1") + ".x")
        else:
            s.res(n("gr1") + ".x", xfile)
        s.link(n("go") + ".y", n("gr1") + ".y"); s.res(n("gr1") + ".g", "ms_g1.bin"); s.res(n("gr1") + ".sel", "sel.f32")
        s.program(n("lnm2"), "dit_lnmod", f"s={SP} n={D}"); s.link(n("gr1") + ".z", n("lnm2") + ".x"); s.res(n("lnm2") + ".sc", "ms_s2.bin"); s.res(n("lnm2") + ".sel", "sel.f32")
        for gg, wf in (("gg", "wg"), ("gp", "wp")):
            s.program(n(gg), "ported/gemm_mma", f"m={SP} n={3 * D} k={D}")
            s.link(n("lnm2") + ".y", n(gg) + ".a"); s.res(n(gg) + ".w", "%s%d.f16" % (wf, b), "f16")
        s.program(n("sw"), "dit_swiglu", f"s={SP} n={3 * D} bn=1024"); s.link(n("gg") + ".y", n("sw") + ".g"); s.link(n("gp") + ".y", n("sw") + ".u")
        s.program(n("gd"), "ported/gemm_mma", f"m={SP} n={D} k={3 * D}"); s.link(n("sw") + ".y", n("gd") + ".a"); s.res(n("gd") + ".w", "wout%d.f16" % b, "f16")
        s.program(n("gr2"), "dit_gate_res", f"s={SP} n={D}")
        s.link(n("gr1") + ".z", n("gr2") + ".x"); s.link(n("gd") + ".y", n("gr2") + ".y"); s.res(n("gr2") + ".g", "ms_g2.bin"); s.res(n("gr2") + ".sel", "sel.f32")
        prev = n("gr2") + ".z"
    s.save(prev, outfile, "f32")
    return s.run(prev, SP, D, "group")


def back(i, xfile):
    s = Session("back")
    s.program("no", "ported/gemm_mma", f"m=64 n={D} k={D}"); s.res("no.a", "tsil.bin", "f16"); s.res("no.w", "wno.f16", "f16")
    s.program("nr", "dit_rows2", f"m=64 n={D} bn=1024"); s.link("no.y", "nr.x")
    s.program("fl", "dit_lnmod", f"s={SP} n={D}"); s.res("fl.x", xfile); s.link("nr.y", "fl.sc"); s.res("fl.sel", "sel.f32")
    s.program("pj", "ported/gemm_mma", f"m={SP} n=64 k={D}"); s.link("fl.y", "pj.a"); s.res("pj.w", "wpj.f16", "f16")
    s.program("eu", "dit_euler", f"s={SP} n=64"); s.res("eu.lat", "lat.f32"); s.link("pj.y", "eu.pr"); s.res("eu.sel", "sel.f32"); s.res("eu.dt", "dt%d.f32" % i)
    s.program("euh", "dit_to_f16", f"s={SP} n=64"); s.link("eu.z", "euh.x")
    s.save("eu.z", "lat_new.f32", "f32")
    s.save("euh.y", "lat_new.f16", "f16")
    return s.run("eu.z", SP, 64, "back")


f64 = None
if "--ref" in sys.argv and os.path.exists(os.path.join(arg("--ref"), "latents_f64.npy")):
    f64 = np.load(os.path.join(arg("--ref"), "latents_f64.npy")).astype(np.float64)
tot = {"front": [], "group": [], "back": []}
walls = []
for i in range(NSTEPS):
    t_step = time.time()
    e = np.zeros((64, 256))
    a = np.array([timesteps[i] / 1000.0, 0.0])[:, None] * 1000.0 * freqs[None]
    e[:2] = np.concatenate([np.cos(a), np.sin(a)], axis=-1)
    f16(P("emb%d.f16" % i), e)
    f32(P("dt%d.f32" % i), np.array([[sigmas[i + 1] - sigmas[i]]]))
    r, w = front(i)
    tot["front"].append(r[-3] if TIME else r[0])
    x = "x_in.bin"
    for g in range(NB // G):
        out = "x_g%d.bin" % g
        r, w = group(g, x, out)
        tot["group"].append(r[-3] if TIME else r[0])
        x = out
    r, w = back(i, x)
    tot["back"].append(r[-3] if TIME else r[0])
    os.replace(P("lat_new.f32"), P("lat.f32"))
    os.replace(P("lat_new.f16"), P("lat.f16"))
    new = np.fromfile(P("lat.f32"), dtype="<f4").reshape(SP, 64)[21:S].astype(np.float64)
    np.save(P("latents_%d.npy" % i), new)
    want = ld("latents_%d" % i)[0]
    msg = "step %d: %.0f s wall; latents rms %.4g; vs torch BF16: mean abs %.3g, max %.3g" % (i, time.time() - t_step, np.sqrt((want ** 2).mean()), np.abs(new - want).mean(), np.abs(new - want).max())
    if f64 is not None and i < len(f64):
        msg += "; vs float64: mean abs %.3g, max %.3g (torch vs float64: %.3g)" % (np.abs(new - f64[i]).mean(), np.abs(new - f64[i]).max(), np.abs(want - f64[i]).mean())
    print(msg, flush=True)
    if TIME and i == 0:
        print("TIME steady state of one step's sessions: front %.3f s + %d groups x %.3f s + back %.3f s = %.3f s" % (tot["front"][0], NB // G, np.median(tot["group"]), tot["back"][0], tot["front"][0] + (NB // G) * np.median(tot["group"]) + tot["back"][0]), flush=True)
np.save(P("lat_final.npy"), new)
print("final latents saved: %s" % P("lat_final.npy"))
