"""D9e/D9g of docs/diffusion.md: the transformer with a condition image on a Metal GPU as a prefix pass and a cached step, as a library (tests/dit_edit_gpu.py holds it to float64; tests/edit_pipeline_gpu.py runs eight
steps of it). See dit_edit_gpu.py for the design.

    e = Edit(work, wdir, np_, npr, ntg, gp, gs, time=False, mut=None)
    e.static(cs_pre, sn_pre, cs_tgt, sn_tgt, mask_pre, mask_step)   # the tables that do not change between steps
    e.modulation(mod)                                                # mod [2, 16384]: row 0 the sampled timestep's, row 1 t = 0's; rewritten each step
    e.prefix(x_pre)       # x_pre [NP, 4096]: the prefix rows after the front end; saves each block's keys and values
    x = e.step(x_tgt)     # x_tgt [NP, 4096]: the target rows after the front end; returns [NP, 4096] after the 32 blocks
"""
import math
import os
import subprocess
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
CHX = os.path.join(ROOT, "tests/chx")
NEG = -1e9
NB, D, H, HD = 32, 4096, 32, 128
f16 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f2")).tofile(path)
f32 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f4")).tofile(path)


def statics(rope, allowed, pre, tgt, np_, mut=None):
    """The rotary tables (padded rows: cos 1, sin 0) and the two masks of an edit, from the joint sequence's rope [S, 64, 2], allowed [S, S], and the index sets of the prefix and the target rows."""
    npr, ntg = len(pre), len(tgt)
    cs_sn = []
    for rows, n in ((pre, npr), (tgt, ntg)):
        cs, sn = np.ones((np_, 64)), np.zeros((np_, 64))
        cs[:n], sn[:n] = rope[rows][..., 0], rope[rows][..., 1]
        cs_sn += [cs, sn]
    ap = allowed[np.ix_(pre, pre)].copy()
    if mut == "cond-causal":  # the condition image made causal instead of bidirectional inside itself
        ap = np.tril(np.ones_like(ap))
    mp = np.full((np_, np_), NEG)
    mp[:npr, :npr][ap] = 0.0
    mp[npr:, :npr] = 0.0  # a padded query row sees the whole prefix (a row with no key would be nan)
    ms = np.full((np_, 2 * np_), NEG)
    ms[:, :npr] = 0.0
    ms[:, np_:np_ + ntg] = 0.0
    return cs_sn[0], cs_sn[1], cs_sn[2], cs_sn[3], mp, ms


class Session:
    def __init__(self, work, name, timed):
        self.work, self.name, self.lines, self.timed = work, name, [], timed

    def program(self, n, kernel, args):
        self.lines.append(f"program {n} {CHX}/{kernel}.chx {args}")

    def res(self, target, path, flag="f32"):
        self.lines.append(f"resident {target} {path} {flag}")

    def link(self, a, b):
        self.lines.append(f"link {a} -> {b}")

    def save(self, out, path, flag):
        self.lines.append(f"save {out} {path} {flag}")

    def run(self, probe_of, probe_m, probe_n):
        self.program("probe", "dit_rows2", f"m={probe_m} n={probe_n} bn=1024")
        self.link(probe_of, "probe.x")
        path = os.path.join(self.work, self.name + ".session")
        open(path, "w").write("\n".join(self.lines) + "\nsteps\nanswer probe.y\n")
        p = subprocess.Popen([SESSION, path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        reqs = []
        for line in (["1\n", "1\n"] if self.timed else ["1\n"]) + ["checkpoint\n"]:
            t1 = time.perf_counter()
            p.stdin.write(line)
            p.stdin.flush()
            out = p.stdout.readline()
            reqs.append(time.perf_counter() - t1)
            if not out:
                raise SystemExit("session %s stopped: %s" % (self.name, p.stderr.read()[:800]))
        p.stdin.close()
        p.wait()
        return reqs[-2] if self.timed else reqs[0]


class Edit:
    def __init__(self, work, wdir, np_, npr, ntg, gp, gs, time=False, mut=None):
        self.work, self.wdir, self.NP, self.NPR, self.NTG, self.GP, self.GS, self.timed, self.mut = work, wdir, np_, npr, ntg, gp, gs, time, mut
        os.makedirs(work, exist_ok=True)
        self.P = lambda n: os.path.join(work, n)
        self.W = lambda n: os.path.join(wdir, n)
        self.secs = {"prefix": 0.0, "step": 0.0}

    def static(self, cs_pre, sn_pre, cs_tgt, sn_tgt, mask_pre, mask_step):
        P, NP = self.P, self.NP
        f32(P("cs_pre.f32"), cs_pre); f32(P("sn_pre.f32"), sn_pre); f32(P("cs_tgt.f32"), cs_tgt); f32(P("sn_tgt.f32"), sn_tgt)
        f32(P("sel_pre.f32"), np.ones((NP, 1)) if self.mut == "prefix-sampled-t" else np.zeros((NP, 1)))  # the prefix reads the t = 0 row of the modulation
        f32(P("sel_tgt.f32"), np.ones((NP, 1)))
        f32(P("mask_pre.f32"), mask_pre)
        f32(P("mask_step.f32"), mask_step)

    def modulation(self, mod):
        for i, nm in enumerate(("s1", "g1", "s2", "g2")):
            f32(self.P("ms_%s.bin" % nm), mod[:, i * D:(i + 1) * D])

    def _blocks(self, name, lo, hi, xfile, prefix):
        P, W, NP, mut = self.P, self.W, self.NP, self.mut
        s = Session(self.work, name, self.timed)
        suf = "pre" if prefix else "tgt"
        prev = None
        for b in range(lo, hi):
            n = lambda t: "b%d_%s" % (b, t)
            sel_f = P("sel_%s.f32" % suf)
            s.program(n("lnm1"), "dit_lnmod", f"s={NP} n={D}")
            if prev:
                s.link(prev, n("lnm1") + ".x")
            else:
                s.res(n("lnm1") + ".x", xfile)
            s.res(n("lnm1") + ".sc", P("ms_s1.bin")); s.res(n("lnm1") + ".sel", sel_f)
            for gg, wf in (("gq", "wq"), ("gk", "wk"), ("gv", "wv")):
                s.program(n(gg), "ported/gemm_mma", f"m={NP} n={D} k={D}")
                s.link(n("lnm1") + ".y", n(gg) + ".a"); s.res(n(gg) + ".w", W("%s%d.f16" % (wf, b)), "f16")
            for nm, src, wf in (("nq", "gq", "nq"), ("nk", "gk", "nk")):
                s.program(n(nm), "rmsnorm_rows", f"r={NP * H} n={HD} eps=0.000001")
                s.link(n(src) + ".y", n(nm) + ".x"); s.res(n(nm) + ".w", W("%s%d.f32" % (wf, b)))
            for rn, src, sc_ in (("rq", "nq", repr(1 / math.sqrt(HD))), ("rk", "nk", "1.0")):
                s.program(n(rn), "dit_rope16", f"s={NP} nh={H} h=64 sc={sc_}")
                s.link(n(src) + ".y", n(rn) + ".x"); s.res(n(rn) + ".cs", P("cs_%s.f32" % suf)); s.res(n(rn) + ".sn", P("sn_%s.f32" % suf))
            s.program(n("vh"), "dit_to_f16", f"s={NP} n={D}"); s.link(n("gv") + ".y", n("vh") + ".x")
            if prefix:
                s.save(n("rk") + ".y", P("kpre%d.f16" % b), "f16")
                s.save(n("vh") + ".y", P("vpre%d.f16" % b), "f16")
                kk, vv, t, maskf = n("rk") + ".y", n("vh") + ".y", NP, "mask_pre.f32"
            else:
                kk, vv, t, maskf = n("ck") + ".y", n("cv") + ".y", 2 * NP, "mask_step.f32"
                for cn, src, pf in (("ck", "rk", "kpre"), ("cv", "vh", "vpre")):
                    s.program(n(cn), "dit_cat", f"s={NP} n={D} bn=1024")
                    if mut == "no-cat":  # the cached prefix left out: the target's own twice
                        s.link(n(src) + ".y", n(cn) + ".a")
                    else:
                        s.res(n(cn) + ".a", P("%s%d.f16" % (pf, b)), "f16")
                    s.link(n(src) + ".y", n(cn) + ".b")
            s.program(n("sc"), "dit_scores_mma", f"nh={H} nq={NP} t={t} d={HD}")
            s.link(n("rq") + ".y", n("sc") + ".q"); s.link(kk, n("sc") + ".k")
            s.program(n("st"), "dit_stats_m", f"nh={H} nq={NP} t={t} qs=32 cs=64"); s.link(n("sc") + ".sc", n("st") + ".sc"); s.res(n("st") + ".mask", P(maskf))
            s.program(n("pv"), "dit_pv_mma", f"nh={H} nq={NP} t={t} d={HD}")
            s.link(n("sc") + ".sc", n("pv") + ".sc"); s.res(n("pv") + ".mask", P(maskf)); s.link(n("st") + ".m", n("pv") + ".m"); s.link(n("st") + ".z", n("pv") + ".z"); s.link(vv, n("pv") + ".v")
            s.program(n("c1"), "dit_to_f16", f"s={NP} n={D}"); s.link(n("pv") + ".o", n("c1") + ".x")
            s.program(n("go"), "ported/gemm_mma", f"m={NP} n={D} k={D}"); s.link(n("c1") + ".y", n("go") + ".a"); s.res(n("go") + ".w", W("wo%d.f16" % b), "f16")
            s.program(n("gr1"), "dit_gate_res", f"s={NP} n={D}")
            if prev:
                s.link(prev, n("gr1") + ".x")
            else:
                s.res(n("gr1") + ".x", xfile)
            s.link(n("go") + ".y", n("gr1") + ".y"); s.res(n("gr1") + ".g", P("ms_g1.bin")); s.res(n("gr1") + ".sel", sel_f)
            s.program(n("lnm2"), "dit_lnmod", f"s={NP} n={D}"); s.link(n("gr1") + ".z", n("lnm2") + ".x"); s.res(n("lnm2") + ".sc", P("ms_s2.bin")); s.res(n("lnm2") + ".sel", sel_f)
            for gg, wf in (("gg", "wg"), ("gp", "wp")):
                s.program(n(gg), "ported/gemm_mma", f"m={NP} n={3 * D} k={D}")
                s.link(n("lnm2") + ".y", n(gg) + ".a"); s.res(n(gg) + ".w", W("%s%d.f16" % (wf, b)), "f16")
            s.program(n("sw"), "dit_swiglu", f"s={NP} n={3 * D} bn=1024"); s.link(n("gg") + ".y", n("sw") + ".g"); s.link(n("gp") + ".y", n("sw") + ".u")
            s.program(n("gd"), "ported/gemm_mma", f"m={NP} n={D} k={3 * D}"); s.link(n("sw") + ".y", n("gd") + ".a"); s.res(n("gd") + ".w", W("wout%d.f16" % b), "f16")
            s.program(n("gr2"), "dit_gate_res", f"s={NP} n={D}")
            s.link(n("gr1") + ".z", n("gr2") + ".x"); s.link(n("gd") + ".y", n("gr2") + ".y"); s.res(n("gr2") + ".g", P("ms_g2.bin")); s.res(n("gr2") + ".sel", sel_f)
            prev = n("gr2") + ".z"
        outf = P("x_%s_%d.bin" % (suf, hi))
        s.save(prev, outf, "f32")
        return outf, s.run(prev, NP, D)

    def prefix(self, x_pre):
        f32(self.P("xpre.f32"), x_pre)
        x = self.P("xpre.f32")
        self.secs["prefix"] = 0.0
        for lo in range(0, NB, self.GP):
            x, secs = self._blocks("pre%d" % lo, lo, lo + self.GP, x, True)
            self.secs["prefix"] += secs

    def step(self, x_tgt, report=None):
        f32(self.P("xtgt.f32"), x_tgt)
        x = self.P("xtgt.f32")
        self.secs["step"] = 0.0
        for lo in range(0, NB, self.GS):
            x, secs = self._blocks("stp%d" % lo, lo, lo + self.GS, x, False)
            self.secs["step"] += secs
            if report:
                report(lo + self.GS, np.fromfile(x, dtype="<f4").reshape(self.NP, D)[:self.NTG].astype(np.float64))
        return np.fromfile(x, dtype="<f4").reshape(self.NP, D).astype(np.float64)
