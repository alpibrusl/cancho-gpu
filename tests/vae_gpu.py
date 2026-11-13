#!/usr/bin/env python3
"""D4c of docs/diffusion.md: the VAE decoder of Qwen-Image-2.1-Turbo on a Metal GPU, as one session of kernel programs, held to the float64 decoder (tools/vae_ref.py) and to D0's BF16 decode.

    vae_gpu.py GOLDEN_DIR [--keep DIR] [--z LATENT.npy] [--save-rgb OUT.npy] [--time N] [--no-ref]

--no-ref: do not compute the float64 decoder (it takes minutes at 1024 x 1024); only the GPU image is written.

--time N: time N requests (decodes) one at a time (tests/session_time.py), a probe as the answer.

--z gives the latent [64, h, w] to decode (after the mean/std of D6) instead of D0's `vae_in`; the comparison with D0's `vae_out` is then not meaningful and is skipped.

Needs a Mac, the checkpoint, D0's goldens (`vae_in`, `vae_out`), numpy, torch and safetensors (the venv of the first image edits has them), and `build/cancho-gpu-metal-session`. Not in `scripts/gate.sh`.

The layout and the kernels are tests/vae_pieces.py's (docs/diffusion.md D4b): every feature map is a matrix of rows, one a pixel of a bordered image (rows rounded up to 64), f32 between the blocks, f16 with guard rows into a
convolution. Per resolution (16, 32, 64, 128, 256 pixels a side for a 256 x 256 image) there is one layout and one interior mask. A residual block is vae_norm_silu, a convolution, vae_bias, vae_norm_silu, a convolution,
and vae_bias_res (with a 1 x 1 shortcut convolution when the channels change); an up block ends with vae_up2, the 3 x 3 convolution at the doubled size and vae_up2_epi with the shortcut, which is four 1 x 1 convolutions of the block's
input with 0/1 selection matrices (DupUp3D's channel rearrangement, built here from its definition: output channel o, sub-position (a, b) reads input channel ((o ft + ft - 1) 4 + 2a + b) // reps, reps = out ft 4 / in, ft = 2 for
the first three up blocks, 1 for the fourth). The mid block's attention is vae_norm, a 1 x 1 convolution to q, k, v, vae_split16, one matrix product for the scores (gemm_mma), vae_mask_add, dit_stats and dit_pv with one head of 1152, a 1 x 1
convolution and vae_bias_res. 3 x 3 convolutions are conv3.chx (channels multiples of 32) or conv3_k16.chx (the 144 and the padded output of 16).

What stays on the host in this slice: the latent as a bordered f16 matrix (D6 will lay D3's latents out on the device and undo their mean and std), the weights as f16 files, the selection matrices, and the final clamp to [-1, 1].
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
import vae_ref as vr  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gold = sys.argv[1]
keep = sys.argv[sys.argv.index("--keep") + 1] if "--keep" in sys.argv else None
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
CHX = os.path.join(ROOT, "tests/chx")
CONV3 = os.path.join(ROOT, "examples/cnn/conv3.chx")
NEG = -1e9

ZGIVEN = "--z" in sys.argv
z = np.load(sys.argv[sys.argv.index("--z") + 1]).astype(np.float64) if ZGIVEN else np.load(os.path.join(gold, "vae_in.npy")).astype(np.float64)[0, :, 0]  # [64, h, w]
want = None if "--z" in sys.argv else np.load(os.path.join(gold, "vae_out.npy")).astype(np.float64)[0, :, 0]
H0, W0 = z.shape[1:]
work = keep or tempfile.mkdtemp()
os.makedirs(work, exist_ok=True)
P = lambda n: os.path.join(work, n)


def lay(h, w):
    wp = w + 2
    rp = (h + 2) * wp
    return {"h": h, "w": w, "wp": wp, "R": (rp + 63) // 64 * 64, "g": wp + 1}


def f16(path, a):
    np.ascontiguousarray(np.asarray(a).astype("<f2")).tofile(path)


def f32(path, a):
    np.ascontiguousarray(np.asarray(a).astype("<f4")).tofile(path)


_made = {}


def mask_file(L):
    key = "mask%d" % L["h"]
    if key not in _made:
        m = np.zeros(L["R"])
        for y in range(L["h"]):
            m[(y + 1) * L["wp"] + 1:(y + 1) * L["wp"] + 1 + L["w"]] = 1.0
        f32(P(key + ".f32"), m[:, None])
        _made[key] = True
    return key + ".f32"


def conv_w3(name):  # [O, C, 3, 3] -> [O, 9 C], tap-major (tap 3 dy + dx)
    w = vr.W(name + ".weight")
    return w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1)


def conv_w1(name):
    w = vr.W(name + ".weight")
    return w.reshape(w.shape[0], w.shape[1])


prog = []
n_prog = 0


def program(name, kernel, args):
    global n_prog
    n_prog += 1
    path = kernel if kernel.startswith("/") else f"{CHX}/{kernel}.chx"
    prog.append(f"program {name} {path} {args}")


def res(target, fname, flag="f32"):
    prog.append(f"resident {target} {P(fname)} {flag}")


def link(a, b):
    prog.append(f"link {a} -> {b}")


uid = [0]


def fresh(prefix):
    uid[0] += 1
    return "%s%d" % (prefix, uid[0])


def conv3(src, name, cin, cout, L, weights, pad_out=None):
    """3 x 3 convolution of the f16 guarded `src` (a program.output name); returns the f32 output name. `weights` [cout, 9 cin]."""
    n = fresh("cv")
    wf = n + ".f16"
    f16(P(wf), weights)
    k16 = cin % 32 != 0 or cout % 32 != 0
    program(n, "conv3_k16" if k16 else CONV3, f"rp={L['R']} cin={cin} cout={cout} wp={L['wp']} g={L['g']}")
    link(src, n + ".x")
    res(n + ".w", wf, "f16")
    return n + ".y"


def conv1(src, cin, cout, L, weights):
    n = fresh("c1")
    f16(P(n + ".f16"), weights)
    program(n, "conv1", f"r={L['R']} cin={cin} cout={cout} g={L['g']}")
    link(src, n + ".x")
    res(n + ".w", n + ".f16", "f16")
    return n + ".y"


def vecfile(name, vec):
    f32(P(name + ".f32"), np.asarray(vec)[None])
    return name + ".f32"


def bias(src, b, c, L):
    n = fresh("bi")
    program(n, "vae_bias", f"r={L['R']} c={c}")
    link(src, n + ".x")
    res(n + ".b", vecfile(n + "b", b)); res(n + ".m", mask_file(L))
    return n + ".z"


def norm(src, gamma, c, L, silu=True):
    n = fresh("nm")
    program(n, "vae_norm_silu" if silu else "vae_norm", f"r={L['R']} c={c} g={L['g']} sc={math.sqrt(c)!r}")
    link(src, n + ".x")
    res(n + ".gamma", vecfile(n + "g", gamma)); res(n + ".mask", mask_file(L))
    return n + ".y"


def pad16(src, c, L):
    n = fresh("pd")
    program(n, "vae_pad16", f"r={L['R']} c={c} g={L['g']}")
    link(src, n + ".x")
    return n + ".y"


def bias_res(src, b, res_src, c, L):
    n = fresh("br")
    program(n, "vae_bias_res", f"r={L['R']} c={c}")
    link(src, n + ".x"); link(res_src, n + ".res")
    res(n + ".b", vecfile(n + "b", b)); res(n + ".m", mask_file(L))
    return n + ".z"


def resnet(x, cin, cout, L, p):
    a = conv3(norm(x, vr.W(p + ".norm1.gamma").reshape(-1), cin, L), None, cin, cout, L, conv_w3(p + ".conv1"))
    a = bias(a, vr.W(p + ".conv1.bias"), cout, L)
    a = conv3(norm(a, vr.W(p + ".norm2.gamma").reshape(-1), cout, L), None, cout, cout, L, conv_w3(p + ".conv2"))
    if cin != cout:
        s = conv1(pad16(x, cin, L), cin, cout, L, conv_w1(p + ".conv_shortcut"))
        h = bias(s, vr.W(p + ".conv_shortcut.bias"), cout, L)
    else:
        h = x
    return bias_res(a, vr.W(p + ".conv2.bias"), h, cout, L)


def attention(x, c, L, p):
    nrm = norm(x, vr.W(p + ".norm.gamma").reshape(-1), c, L, silu=False)
    qkv = bias(conv1(nrm, c, 3 * c, L, conv_w1(p + ".to_qkv")), vr.W(p + ".to_qkv.bias"), 3 * c, L)
    sp = fresh("sp")
    program(sp, "vae_split16", f"r={L['R']} c={c} sc={1 / math.sqrt(c)!r}")
    link(qkv, sp + ".qkv")
    sc = fresh("sc")
    program(sc, "ported/gemm_mma", f"m={L['R']} n={L['R']} k={c}")
    link(sp + ".q", sc + ".a"); link(sp + ".k", sc + ".w")
    ma = fresh("ma")
    program(ma, "vae_mask_add", f"n={L['R']}")
    link(sc + ".y", ma + ".sc")
    mk = np.where(np.asarray([1.0 if i in interior_rows(L) else 0.0 for i in range(L["R"])]) > 0, 0.0, NEG)
    res(ma + ".mk", vecfile(ma + "mk", mk))
    st = fresh("st")
    program(st, "dit_stats", f"nh=1 nq={L['R']} t={L['R']} qs=32 cs=64")
    link(ma + ".z", st + ".sc")
    pv = fresh("pv")
    program(pv, "dit_pv", f"nh=1 nq={L['R']} t={L['R']} qs=32 cs=64 dw=32 d={c}")
    link(ma + ".z", pv + ".sc"); link(st + ".m", pv + ".m"); link(st + ".z", pv + ".z"); link(sp + ".v", pv + ".v")
    o16 = pad16(pv + ".o", c, L)
    pr = conv1(o16, c, c, L, conv_w1(p + ".proj"))
    return bias_res(pr, vr.W(p + ".proj.bias"), x, c, L)


_rows = {}


def interior_rows(L):
    if L["h"] not in _rows:
        _rows[L["h"]] = {(y + 1) * L["wp"] + 1 + x for y in range(L["h"]) for x in range(L["w"])}
    return _rows[L["h"]]


def selection(cin, cout, ft, a, b):
    reps = cout * ft * 4 // cin
    sel = np.zeros((cout, cin))
    for o in range(cout):
        sel[o, ((o * ft + ft - 1) * 4 + 2 * a + b) // reps] = 1.0
    return sel


# ---- the decoder
L = lay(H0, W0)
zr = np.zeros((L["R"], 64))
for y in range(H0):
    for xx in range(W0):
        zr[(y + 1) * L["wp"] + xx + 1] = z[:, y, xx]
zg = np.zeros((L["R"] + 2 * L["g"], 64))
zg[L["g"]:L["g"] + L["R"]] = zr
f16(P("z.f16"), zg)
pq = fresh("c1")
f16(P(pq + ".f16"), conv_w1("post_quant_conv"))
program(pq, "conv1", f"r={L['R']} cin=64 cout=64 g={L['g']}")
res(pq + ".x", "z.f16", "f16"); res(pq + ".w", pq + ".f16", "f16")
x = bias(pq + ".y", vr.W("post_quant_conv.bias"), 64, L)
x = bias(conv3(pad16(x, 64, L), None, 64, 1152, L, conv_w3("decoder.conv_in")), vr.W("decoder.conv_in.bias"), 1152, L)
x = resnet(x, 1152, 1152, L, "decoder.mid_block.resnets.0")
x = attention(x, 1152, L, "decoder.mid_block.attentions.0")
x = resnet(x, 1152, 1152, L, "decoder.mid_block.resnets.1")
outs = [1152, 1152, 576, 288, 144]
ins = [1152, 1152, 1152, 576, 288]
temporal = [True, True, True, False, False]
for i in range(5):
    p = "decoder.up_blocks.%d." % i
    x0, cin, cout = x, ins[i], outs[i]
    for r in range(3):
        x = resnet(x, cin if r == 0 else cout, cout, L, p + "resnets.%d" % r)
    if i < 4:
        L2 = lay(L["h"] * 2, L["w"] * 2)
        up = fresh("up")
        program(up, "vae_up2", f"h={L['h']} w={L['w']} c={cout} r={L['R']} r2={L2['R']} g2={L2['g']}")
        link(x, up + ".x")
        cv = conv3(up + ".y", None, cout, cout, L2, conv_w3(p + "upsampler.resample.1"))
        x0h = pad16(x0, cin, L)
        ft = 2 if temporal[i] else 1
        s = [conv1(x0h, cin, cout, L, selection(cin, cout, ft, a, b)) for a in range(2) for b in range(2)]
        ep = fresh("ep")
        program(ep, "vae_up2_epi", f"h={L['h']} w={L['w']} c={cout} r={L['R']} r2={L2['R']}")
        link(cv, ep + ".cv")
        res(ep + ".b", vecfile(ep + "b", vr.W(p + "upsampler.resample.1.bias")))
        for name, src in zip(("s00", "s01", "s10", "s11"), s):
            link(src, ep + "." + name)
        x, L = ep + ".z", L2
x = norm(x, vr.W("decoder.norm_out.gamma").reshape(-1), 144, L)
wout = np.zeros((16, 9 * 144))
wout[:4] = conv_w3("decoder.conv_out")
y = conv3(x, None, 144, 16, L, wout)
TIME = int(sys.argv[sys.argv.index("--time") + 1]) if "--time" in sys.argv else 0
if TIME:
    program("probe", "dit_rows2", f"m={L['R']} n=16 bn=16"); link(y, "probe.x")
prog.append("steps")
prog.append("answer probe.y" if TIME else "answer " + y)
open(P("vae.session"), "w").write("\n".join(prog) + "\n")

if TIME:
    import session_time
    ts = session_time.timed(SESSION, P("vae.session"), TIME)
    rest = ts[1:]
    print("TIME VAE decode, %d x %d: first request %.3f s (device, compile, upload); then median %.4f s (min %.4f, max %.4f) over %d" % (L["h"], L["w"], ts[0], sorted(rest)[len(rest) // 2], min(rest), max(rest), len(rest)))
    sys.exit(0)
t0 = time.time()
r = subprocess.run([SESSION, P("vae.session")], input="1\n", capture_output=True, text=True, timeout=7200)
secs = time.time() - t0
if r.returncode != 0 or not r.stdout.strip():
    print("FAIL the session did not answer: status %d, %s" % (r.returncode, r.stderr.strip()[:800]))
    sys.exit(1)
out = np.array(r.stdout.split(), dtype=np.float64).reshape(L["R"], 16)
bias_out = vr.W("decoder.conv_out.bias")
img = np.zeros((4, L["h"], L["w"]))
for yy in range(L["h"]):
    for xx in range(L["w"]):
        img[:, yy, xx] = out[(yy + 1) * L["wp"] + xx + 1, :4] + bias_out
img = np.clip(img, -1.0, 1.0)
print("session: %d programs, answered in %.1f s (Metal; includes the compile and the upload of the weights)" % (n_prog, secs))
exact = img if "--no-ref" in sys.argv else vr.decode(z)
if want is None:
    np.save(P("decoded_f64.npy"), exact)
    want = exact
rms = float(np.sqrt((want ** 2).mean()))
ok = True
for label, a, b in ((("GPU decoder vs float64 decoder", img, exact),) if ZGIVEN else (("GPU decoder vs float64 decoder", img, exact), ("GPU decoder vs torch BF16 (D0's vae_out)", img, want), ("(float64 decoder vs torch BF16, for scale)", exact, want))):
    d = np.abs(a - b)
    print("%-44s mean abs error %.4g (%.3g of the output rms %.3g), max abs error %.4g" % (label, d.mean(), d.mean() / rms, rms, d.max()))
d = np.abs(img - exact)
ok = d.mean() / rms < 5e-3
rgb = ((img[:3] + 1) * 127.5).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
np.save(P("decoded_rgb.npy"), rgb)
if "--save-rgb" in sys.argv:
    np.save(sys.argv[sys.argv.index("--save-rgb") + 1], rgb)
print("%s the decoder on the GPU (mean error against the float64 decoder under 0.5%% of the output's rms); the 8-bit RGB image is %s" % ("ok  " if ok else "FAIL", P("decoded_rgb.npy")))
sys.exit(0 if ok else 1)
