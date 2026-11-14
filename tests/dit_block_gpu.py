#!/usr/bin/env python3
"""D2c of docs/diffusion.md: transformer block 0 of Qwen-Image-2.1-Turbo on a Metal GPU, as one session of 20 kernel programs, held to the float64 block and to torch's BF16 output.

    dit_block_gpu.py GOLDEN_DIR [--keep DIR] [--mutate no-perm|sel-inverted|keys-padded]

Needs a Mac, the checkpoint (`Qwen/Qwen-Image-2.1-Turbo`, in the Hugging Face cache), D0's goldens (`tools/diffusion_golden.py`), numpy, torch and safetensors (the venv of the first
image edits has them), and `build/cancho-gpu-metal-session` (`scripts/build-metal-session.sh`). So it is not in `scripts/gate.sh`, like `scripts/metal-kernels.sh`.

The session (written here, in the repository's way, from Python: `examples/gpt/gpt_session.py` is the model) is the block of tools/dit_block_ref.py, kernel by kernel:
dit_lnmod -> gemm_mma (q, k, v) -> rmsnorm_rows (per head) -> dit_rope -> dit_scores, dit_stats, dit_pv -> dit_to_f16 -> gemm_mma (o) -> dit_gate_res -> dit_lnmod -> gemm_mma (gate, up)
-> dit_swiglu -> gemm_mma (down) -> dit_gate_res. The 277 tokens (21 text + 256 image) are padded to 320 (the matrix unit's tile is 64): the padded rows are zeros, are not keys (the mask says so)
and are dropped from the comparison. The weights go up as f16 files (narrowing BF16 to f16 changes 0.0097% of them: D2a), with the output channels of to_q and to_k and the norm weights
permuted within each head, even channels first, so that the rotate-half kernel is the model's interleaved RoPE (tests/dit_block.py, `permutation`).

Held to (D2a's numbers): the float64 block, mean abs error at most 1e-3 of the output's rms; torch's BF16 block, mean abs error at most 4e-4 of the rms and max abs error at most 24.
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import dit_block_ref as ref  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gold = sys.argv[1]
keep = sys.argv[sys.argv.index("--keep") + 1] if "--keep" in sys.argv else None
MUT = sys.argv[sys.argv.index("--mutate") + 1] if "--mutate" in sys.argv else None  # a deliberate host-side bug, which the comparison must refuse
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
CHX = os.path.join(ROOT, "tests/chx")
NEG = -1e9
D, H, HD, S, SP = 4096, 32, 128, 277, 320  # hidden, heads, head dim, tokens, padded tokens
perm = np.arange(HD) if MUT == "no-perm" else np.array(list(range(0, HD, 2)) + list(range(1, HD, 2)))

x, want, mod, rope, tmask = ref.goldens(gold)
w = {k: ref.load64("transformer_blocks.0." + v) for k, v in ref.BLOCK0.items()}
exact = ref.block(x.copy(), w, mod, rope, tmask)


def head_perm_rows(m):  # [4096 out, 4096 in]: within each head's 128 output rows, even channels first
    return m.reshape(H, HD, -1)[:, perm, :].reshape(H * HD, -1)


def f16(path, a):
    np.ascontiguousarray(a.astype("<f2")).tofile(path)


def f32(path, a):
    np.ascontiguousarray(a.astype("<f4")).tofile(path)


work = keep or tempfile.mkdtemp()
os.makedirs(work, exist_ok=True)
P = lambda n: os.path.join(work, n)
xp = np.zeros((SP, D))
xp[:S] = x
f32(P("x.f32"), xp)
sel = np.zeros((SP, 1))
sel[:S, 0] = ~tmask if MUT == "sel-inverted" else tmask
f32(P("sel.f32"), sel)
for i, name in enumerate(("sc1", "g1", "sc2", "g2")):
    f32(P(name + ".f32"), mod[:, i * D:(i + 1) * D])
f16(P("wq.f16"), head_perm_rows(w["q"]))
f16(P("wk.f16"), head_perm_rows(w["k"]))
f16(P("wv.f16"), w["v"])
f16(P("wo.f16"), w["o"])
f16(P("wg.f16"), w["g"])
f16(P("wp.f16"), w["p"])
f16(P("wout.f16"), w["out"])
f32(P("nq.f32"), w["nq"][perm][None])
f32(P("nk.f32"), w["nk"][perm][None])
cs, sn = np.ones((SP, 64)), np.zeros((SP, 64))
cs[:S], sn[:S] = rope[..., 0], rope[..., 1]
f32(P("cs.f32"), cs)
f32(P("sn.f32"), sn)
idx = np.arange(SP)
valid = idx < S
tm = np.zeros(SP, dtype=bool)
tm[:S] = tmask
allowed = (idx[:, None] >= idx[None, :]) | (tm[:, None] & tm[None, :])
if MUT != "keys-padded":
    allowed &= valid[None, :]
f32(P("mask.f32"), np.where(allowed, 0.0, NEG))

prog = []


def program(name, kernel, args):
    prog.append(f"program {name} {CHX}/{kernel}.chx {args}")


def res(target, fname, flag="f32"):
    prog.append(f"resident {target} {P(fname)} {flag}")


def link(a, b):
    prog.append(f"link {a} -> {b}")


program("lnm1", "dit_lnmod", f"s={SP} n={D}")
res("lnm1.x", "x.f32"); res("lnm1.sc", "sc1.f32"); res("lnm1.sel", "sel.f32")
for n, wf in (("gq", "wq"), ("gk", "wk"), ("gv", "wv")):
    program(n, "ported/gemm_mma", f"m={SP} n={D} k={D}")
    link("lnm1.y", n + ".a"); res(n + ".w", wf + ".f16", "f16")
program("nq", "rmsnorm_rows", f"r={SP * H} n={HD} eps=0.000001"); link("gq.y", "nq.x"); res("nq.w", "nq.f32")
program("nk", "rmsnorm_rows", f"r={SP * H} n={HD} eps=0.000001"); link("gk.y", "nk.x"); res("nk.w", "nk.f32")
for n, src in (("rq", "nq"), ("rk", "nk")):
    program(n, "dit_rope", f"s={SP} nh={H} h=64")
    link(src + ".y", n + ".x"); res(n + ".cs", "cs.f32"); res(n + ".sn", "sn.f32")
program("sc", "dit_scores", f"nh={H} nq={SP} t={SP} qs=32 ts=64 d={HD}")
link("rq.y", "sc.q"); link("rk.y", "sc.k"); res("sc.mask", "mask.f32")
program("st", "dit_stats", f"nh={H} nq={SP} t={SP} qs=32 cs=64"); link("sc.sc", "st.sc")
program("pv", "dit_pv", f"nh={H} nq={SP} t={SP} qs=32 cs=64 dw=32 d={HD}")
link("sc.sc", "pv.sc"); link("st.m", "pv.m"); link("st.z", "pv.z"); link("gv.y", "pv.v")
program("c1", "dit_to_f16", f"s={SP} n={D}"); link("pv.o", "c1.x")
program("go", "ported/gemm_mma", f"m={SP} n={D} k={D}"); link("c1.y", "go.a"); res("go.w", "wo.f16", "f16")
program("gr1", "dit_gate_res", f"s={SP} n={D}")
res("gr1.x", "x.f32"); link("go.y", "gr1.y"); res("gr1.g", "g1.f32"); res("gr1.sel", "sel.f32")
program("lnm2", "dit_lnmod", f"s={SP} n={D}"); link("gr1.z", "lnm2.x"); res("lnm2.sc", "sc2.f32"); res("lnm2.sel", "sel.f32")
for n, wf in (("gg", "wg"), ("gp", "wp")):
    program(n, "ported/gemm_mma", f"m={SP} n={3 * D} k={D}")
    link("lnm2.y", n + ".a"); res(n + ".w", wf + ".f16", "f16")
program("sw", "dit_swiglu", f"s={SP} n={3 * D} bn=1024"); link("gg.y", "sw.g"); link("gp.y", "sw.u")
program("gd", "ported/gemm_mma", f"m={SP} n={D} k={3 * D}"); link("sw.y", "gd.a"); res("gd.w", "wout.f16", "f16")
program("gr2", "dit_gate_res", f"s={SP} n={D}")
link("gr1.z", "gr2.x"); link("gd.y", "gr2.y"); res("gr2.g", "g2.f32"); res("gr2.sel", "sel.f32")
prog.append("steps")
prog.append("answer gr2.z")
open(P("block.session"), "w").write("\n".join(prog) + "\n")

t0 = time.time()
r = subprocess.run([SESSION, P("block.session")], input="1\n", capture_output=True, text=True, timeout=3600)
secs = time.time() - t0
if r.returncode != 0 or not r.stdout.strip():
    print("FAIL the session did not answer: status %d, %s" % (r.returncode, r.stderr.strip()[:600]))
    sys.exit(1)
got = np.array(r.stdout.split(), dtype=np.float64).reshape(SP, D)[:S]
print("session: %d programs, answered in %.1f s (Metal, includes the upload of 436 MB of weights and the compile)" % (sum(1 for l in prog if l.startswith("program")), secs))

rms = float(np.sqrt((exact ** 2).mean()))


def row(label, a, b):
    d = np.abs(a - b)
    print("%-40s mean abs error %.4g (%.3g of the rms %.3g), max abs error %.4g" % (label, d.mean(), d.mean() / rms, rms, d.max()))
    return d.mean() / rms, d.max()


m1, _ = row("session vs float64 block", got, exact)
m2, mx2 = row("session vs torch BF16 (the golden)", got, want)
m3, mx3 = row("(torch BF16 vs float64, from D2a)", want, exact)
ok = m1 <= 1e-3 and m2 <= 4e-4 and mx2 <= 24
print("%s the block on the GPU: float64 %s, BF16 golden %s" % ("ok  " if ok else "FAIL", "within 1e-3" if m1 <= 1e-3 else "OUT", "within the BF16 floor" if (m2 <= 4e-4 and mx2 <= 24) else "OUT"))
sys.exit(0 if ok else 1)
