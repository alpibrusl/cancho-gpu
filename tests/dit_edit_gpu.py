#!/usr/bin/env python3
"""D9e of docs/diffusion.md: one denoising step of Qwen-Image-2.1-Turbo's transformer **with a condition image** on a Metal GPU, as a prefix pass and a cached step, held to the float64 step of tools/dit_edit_ref.py.

    dit_edit_gpu.py DE_DIR WORK_DIR --weights DIR [--gp N] [--gs N] [--time] [--mutate cond-causal|prefix-sampled-t|no-cat]

DE_DIR is the output of `tools/dit_edit_ref.py` (the joint sequence after the front end, the rotary angles, the mask, the modulation, and the float64 residual after every block). `--weights` is the directory of f16 block
weights that tests/dit_big.py writes (head-permuted, so dit_rope16's rotate-half is the model's rotary embedding). Needs a Mac, the checkpoint, and `build/cancho-gpu-metal-session`. Not in `scripts/gate.sh`.

The design (docs/diffusion.md D9, read from `QwenImage21Transformer2DModel.forward` with a KV cache): the prefix, every row that is not a target token (the text rows and the condition image's tokens), is modulated from t = 0 and never
attends to a target token, so its keys and values are the same at every step. **The prefix pass** runs those rows through the 32 blocks with the block-causal mask and saves each block's keys (after the per-head norm and the rotary
embedding, f16) and values (f16). **A step** runs only the target's rows: per block the same programs as tests/dit_big.py, except that the keys and values are `dit_cat` of the cached prefix's and the target's own, so the
scores are [32 x N_target, N_prefix + N_target] (rows padded to a multiple of 64, both parts to the same count); the mask only hides padded keys (a target row sees the whole prefix and the whole target block).
The host does the front end (txt_in, img_in, the timestep embedding, the modulation, the rotary tables: in float64, from DE_DIR) and the back end (norm_out and proj_out on the target rows).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import dit_block_ref as ref  # noqa: E402
import dit_step_ref as step  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dit_edit_lib import Edit, statics  # noqa: E402

arg = lambda name, default=None: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
de, work = sys.argv[1], sys.argv[2]
WDIR = arg("--weights")
GP, GS = int(arg("--gp", 32)), int(arg("--gs", 32))
TIME = "--time" in sys.argv
MUT = arg("--mutate")  # a deliberate host-side bug, which the comparison must refuse
D = 4096
ld = lambda n: np.load(os.path.join(de, n + ".npy"))

x0, rope, sel, allowed, mod, temb = ld("x0"), ld("rope"), ld("sel"), ld("allowed"), ld("mod"), ld("temb")
blocks = ld("blocks_f64")
want_out = ld("out_f64")
pre, tgt = np.nonzero(~sel)[0], np.nonzero(sel)[0]
NPR, NTG = len(pre), len(tgt)
pad64 = lambda n: (n + 63) // 64 * 64
NP = max(pad64(NPR), pad64(NTG))
print("prefix %d rows, target %d rows, both padded to %d; keys %d" % (NPR, NTG, NP, 2 * NP), flush=True)

xp = np.zeros((NP, D))
xp[:NPR] = x0[pre]
xt = np.zeros((NP, D))
xt[:NTG] = x0[tgt]
e = Edit(work, WDIR, NP, NPR, NTG, GP, GS, time=TIME, mut=MUT)
e.static(*statics(rope, allowed, pre, tgt, NP, MUT))
e.modulation(mod)
e.prefix(xp)
print("prefix pass: %.2f s (%s)" % (e.secs["prefix"], "steady state, weights resident" if TIME else "first requests: includes each session's compile and upload"), flush=True)


def report(upto, got):
    wb = blocks[upto - 1][tgt].astype(np.float64)
    r = float(np.sqrt((wb ** 2).mean()))
    print("   after block %2d: target rows vs float64: mean abs error %.4g (%.3g of the rms %.3g), max %.4g" % (upto, np.abs(got - wb).mean(), np.abs(got - wb).mean() / r, r, np.abs(got - wb).max()), flush=True)


full = e.step(xt, report)
print("cached step: %.2f s (%s)" % (e.secs["step"], "steady state, weights resident" if TIME else "first requests"))
got = full[:NTG]
y = step.back(got, temb, np.ones(NTG, dtype=bool))
d = np.abs(y - want_out)
rms = float(np.sqrt((want_out ** 2).mean()))
print("step output [%d, 64] vs float64: mean abs error %.4g (%.3g of the rms %.3g), max %.4g" % (NTG, d.mean(), d.mean() / rms, rms, d.max()))
BOUND = 5e-3
ok = d.mean() / rms <= BOUND
print("%s the edit step on the GPU: output within %.0e of the rms of the float64 step" % ("ok  " if ok else "FAIL", BOUND))
sys.exit(0 if ok else 1)
