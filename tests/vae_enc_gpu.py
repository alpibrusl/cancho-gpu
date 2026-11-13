#!/usr/bin/env python3
"""D9d of docs/diffusion.md: the VAE encoder of Qwen-Image-2.1-Turbo on a Metal GPU, as one session of kernel programs, held to the float64 encoder (tools/vae_enc_ref.py).

    vae_enc_gpu.py WORK_DIR [--size N | HxW] [--time N] [--mutate no-zero-frame|no-s2d-tap]

Needs a Mac, the checkpoint, numpy, torch and safetensors, and `build/cancho-gpu-metal-session`. Not in `scripts/gate.sh`. The image is synthetic (RGB gradient plus noise, alpha 1: what `preprocess` gives an RGBA image).

The layout and most kernels are the decoder's (tests/vae_gpu.py, D4c): every feature map is a matrix of rows, one a pixel of a bordered image (rows rounded up to 64), f32 between the blocks, f16 with guard rows into a convolution; the residual
blocks, the mid block's attention (one head of 768) and the 1 x 1 convolutions are the decoder's programs. What an encoder adds, in two kernels (D9d, tests/vae_pieces.py): a **stride-2 3 x 3 convolution** is `vae_s2d`
(each 2 x 2 group of pixels becomes 4c channels) and a stride-1 3 x 3 convolution at half size whose weights are the original taps placed at the offsets 0 and +1 (the others zero); the **average-down shortcut** is `vae_avg2` (the 2 x 2 mean)
and a 1 x 1 convolution with a 0/1 matrix: for the temporal blocks (1 to 3) output channel 2c is zero and 2c + 1 is input channel c's mean (a single frame is padded in front by a zero frame), for block 0 the identity, and block 4 adds its
input as it is. The input has 4 channels, padded to 32 with zeros for conv3.chx (channels are multiples of 32); the output is the 128 channels of `quant_conv`, of which the first 64 are the mean.
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
import vae_enc_ref as ve  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
work = sys.argv[1]
MUT = sys.argv[sys.argv.index("--mutate") + 1] if "--mutate" in sys.argv else None  # a deliberate host-side bug, which the comparison must refuse
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
CHX = os.path.join(ROOT, "tests/chx")
CONV3 = os.path.join(ROOT, "examples/cnn/conv3.chx")
NEG = -1e9
os.makedirs(work, exist_ok=True)
P = lambda n: os.path.join(work, n)
sz = sys.argv[sys.argv.index("--size") + 1] if "--size" in sys.argv else "128x192"
IH, IW = (int(a) for a in (sz.split("x") if "x" in sz else (sz, sz)))
img = ve.synthetic(IH, IW)  # [4, H, W] in [-1, 1]


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



# ---- the encoder
dims = [96, 96, 192, 384, 768, 768]
temporal = [False, True, True, True, False]
L = lay(IH, IW)
xr = np.zeros((L["R"] + 2 * L["g"], 32))
for y in range(IH):
    for xx in range(IW):
        xr[L["g"] + (y + 1) * L["wp"] + xx + 1, :4] = img[:, y, xx]
f16(P("img.f16"), xr)
w0 = np.zeros((96, 9, 32))
w0[:, :, :4] = conv_w3("encoder.conv_in").reshape(96, 9, 4)
c0 = fresh("cv")
f16(P(c0 + ".f16"), w0.reshape(96, -1))
program(c0, CONV3, f"rp={L['R']} cin=32 cout=96 wp={L['wp']} g={L['g']}")
res(c0 + ".x", "img.f16", "f16"); res(c0 + ".w", c0 + ".f16", "f16")
x = bias(c0 + ".y", vr.W("encoder.conv_in.bias"), 96, L)


def s2d_weights(name, c, cout):
    """The 3 x 3 stride-2 weights [cout, c, 3, 3] as a stride-1 3 x 3 convolution over 4c channels (a, b, channel): tap (di + 1, dj + 1) holds the original tap (2 di + a, 2 dj + b) when it exists."""
    w = vr.W(name + ".weight")
    out = np.zeros((cout, 3, 3, 2, 2, c))
    for di in (0, 1):
        for dj in (0, 1):
            for a in range(2):
                for b in range(2):
                    dy, dx = 2 * di + a, 2 * dj + b
                    if dy <= 2 and dx <= 2:
                        out[:, di + 1, dj + 1, a, b, :] = w[:, :, dy, dx]
    if MUT == "no-s2d-tap":  # the last tap (2, 2) dropped
        out[:, 2, 2, 0, 0, :] = 0
    return out.reshape(cout, 9 * 4 * c)


for i in range(5):
    p = "encoder.down_blocks.%d." % i
    x0, cin, cout = x, dims[i], dims[i + 1]
    for r in range(2):
        x = resnet(x, cin if r == 0 else cout, cout, L, p + "resnets.%d" % r)
    if i < 4:
        L2 = lay(L["h"] // 2, L["w"] // 2)
        sd = fresh("sd")
        program(sd, "vae_s2d", f"h={L['h']} w={L['w']} c={cout} r={L['R']} r2={L2['R']} g2={L2['g']}")
        link(x, sd + ".x")
        cv = conv3(sd + ".y", None, 4 * cout, cout, L2, s2d_weights(p + "downsampler.resample.1", cout, cout))
        av = fresh("av")
        program(av, "vae_avg2", f"h={L['h']} w={L['w']} c={cin} r={L['R']} r2={L2['R']} g2={L2['g']}")
        link(x0, av + ".x")
        sel = np.zeros((cout, cin))
        for o in range(cout):
            if temporal[i]:
                if o % 2 == 1 or MUT == "no-zero-frame":
                    sel[o, o // 2 if MUT != "no-zero-frame" else (o // 2 if temporal[i] else o)] = 1.0
            else:
                sel[o, o] = 1.0
        if MUT == "no-zero-frame":  # the zero frame forgotten: every channel o takes the mean of c = o // 2 (the pad frame treated as real)
            sel = np.zeros((cout, cin))
            for o in range(cout):
                sel[o, o // 2] = 1.0
        sh = conv1(av + ".y", cin, cout, L2, sel)
        x = bias_res(cv, vr.W(p + "downsampler.resample.1.bias"), sh, cout, L2)
        L = L2
    else:
        ad = fresh("ad")
        program(ad, "dit_add", f"s={L['R']} n={cout} bn=256")
        link(x, ad + ".a"); link(x0, ad + ".b")
        x = ad + ".z"
x = resnet(x, 768, 768, L, "encoder.mid_block.resnets.0")
x = attention(x, 768, L, "encoder.mid_block.attentions.0")
x = resnet(x, 768, 768, L, "encoder.mid_block.resnets.1")
x = norm(x, vr.W("encoder.norm_out.gamma").reshape(-1), 768, L)
x = bias(conv3(x, None, 768, 128, L, conv_w3("encoder.conv_out")), vr.W("encoder.conv_out.bias"), 128, L)
x = bias(conv1(pad16(x, 128, L), 128, 128, L, conv_w1("quant_conv")), vr.W("quant_conv.bias"), 128, L)
TIME = int(sys.argv[sys.argv.index("--time") + 1]) if "--time" in sys.argv else 0
if TIME:
    program("probe", "dit_rows2", f"m={L['R']} n=128 bn=128"); link(x, "probe.x")
prog.append("steps")
prog.append("answer probe.y" if TIME else "answer " + x)
open(P("enc.session"), "w").write("\n".join(prog) + "\n")

if TIME:
    import session_time
    ts = session_time.timed(SESSION, P("enc.session"), TIME)
    rest = ts[1:]
    print("TIME VAE encode, %d x %d: first request %.3f s (device, compile, upload); then median %.4f s (min %.4f, max %.4f) over %d" % (IH, IW, ts[0], sorted(rest)[len(rest) // 2], min(rest), max(rest), len(rest)))
    sys.exit(0)
t0 = time.time()
r = subprocess.run([SESSION, P("enc.session")], input="1\n", capture_output=True, text=True, timeout=7200)
secs = time.time() - t0
if r.returncode != 0 or not r.stdout.strip():
    print("FAIL the session did not answer: status %d, %s" % (r.returncode, r.stderr.strip()[:800]))
    sys.exit(1)
out = np.array(r.stdout.split(), dtype=np.float64).reshape(L["R"], 128)
print("session: %d programs, answered in %.1f s (Metal; includes the compile and the upload)" % (n_prog, secs))
got = np.zeros((64, L["h"], L["w"]))
for yy in range(L["h"]):
    for xx in range(L["w"]):
        got[:, yy, xx] = out[(yy + 1) * L["wp"] + xx + 1, :64]
t0 = time.time()
want = ve.encode(img)
print("float64 encoder in %.0f s" % (time.time() - t0))
rms = float(np.sqrt((want ** 2).mean()))
d = np.abs(got - want)
print("GPU vs float64 (posterior mean, %d x %d latent): mean abs error %.4g (%.3g of the rms %.3g), max %.4g" % (L["h"], L["w"], d.mean(), d.mean() / rms, rms, d.max()))
dn = np.abs(ve.normalise(got) - ve.normalise(want))
print("   normalised by the latents' mean and std (what the transformer reads): mean abs error %.3g, max %.3g" % (dn.mean(), dn.max()))
BOUND = 1.5e-3  # a fifth of torch's BF16 floor at 128 x 192 (7.7e-3 of the rms, docs/diffusion.md D9d); chosen after seeing 6.7e-4 and 1.2e-3, so it is a regression guard, not an a priori criterion
ok = d.mean() / rms <= BOUND
print("%s the VAE encoder on the GPU: mean abs error within %.0e of the rms" % ("ok  " if ok else "FAIL", BOUND))
sys.exit(0 if ok else 1)
