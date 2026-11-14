#!/usr/bin/env python3
"""V2 of docs/diffusion.md section 6: the VAE decoder of Wan2.2-TI2V-5B (AutoencoderKLWan) on a Metal GPU, as one session of kernel programs, held to the float64 decoder of tools/wan_vae_ref.py.

    wan_vae_gpu.py WORK_DIR [--frames F] [--size HxW] [--seed S] [--z LATENT.npy] [--no-ref] [--save VIDEO.npy] [--mutate NAME]

Needs a Mac, the first shards of `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (the VAE) in the Hugging Face cache, numpy and safetensors (the venv of the first image edits), and `build/cancho-gpu-metal-session`.
Not in `scripts/gate.sh`. The latent is [48, F, h, w]; the video is [3, 1 + 4 (F - 1), 16 h, 16 w] in [-1, 1].

The layout is tests/vae_gpu.py's (the 2-D decoder of the first image edits), the clip stacked in it: every feature map is a matrix of rows, one a pixel of a bordered image, the frames one after the other (s = (h + 2) (w + 2)
rows a frame, the whole rounded up to 64 rows), f32 between the blocks, f16 with g guard rows into a convolution, with g >= 2 s + wp + 1: the causal 3 x 3 x 3 convolutions (conv3t.chx) read the two frames before the
current one as shifted windows of the same matrix, and the guard's zeros are the two frames a causal convolution pads in front of the clip. The temporal doubling of an up block (the model's `upsample3d`) adds a **zero frame at
the head** of the layout (the doubled clip has 2 n slots for its 2 n - 1 frames, slot 0 zero): it is the causal window's zero pad for the next convolution, so nothing else has to know about it; every kernel that could write
into it (the masks, `fa`, `fm`) writes zeros. The first latent frame is not doubled and is hidden from `time_conv` (the model's cache holds "Rep" there); `vae_tint.chx` interleaves the two halves of `time_conv`'s
2 c channels in time. Up blocks 0 and 1 double time and space, block 2 only space, block 3 neither. Attention (the middle block, one head of 1024 channels over the positions of one frame) is over all the frames' rows
with a block-diagonal mask (a matrix of n x n floats: small clips only).

`--mutate` makes one deliberate error that the comparison must refuse: `sees-first` (`time_conv` also sees the first latent frame), `swap-halves` (the two halves of its output exchanged), `visible-zero-frame` (the head slot is
data: the masks do not zero it and the shortcut's `fa` writes into it; `fa` alone is **not** a mutant that fails: every reader of another frame reads through a mask, so the zero frame is zero twice over).
What stays on the host: the latent as a bordered f16 matrix, the weights as f16 files, the selection matrices of the shortcuts, the masks, and the unpatchify and the clamp of the output.
"""
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import wan_vae_ref as wv  # noqa: E402

arg = lambda name, default=None: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
work = sys.argv[1]
F = int(arg("--frames", 3))
H0, W0 = (int(v) for v in arg("--size", "4x4").split("x"))
SEED, MUT = int(arg("--seed", 0)), arg("--mutate")
os.makedirs(work, exist_ok=True)
P = lambda n: os.path.join(work, n)
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
CHX = os.path.join(ROOT, "tests/chx")
CONV3 = os.path.join(ROOT, "examples/cnn/conv3.chx")
NEG = -1e9
W = wv.W

f16 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f2")).tofile(path)
f32 = lambda path, a: np.ascontiguousarray(np.asarray(a).astype("<f4")).tofile(path)


def lay(h, w, slots, junk):
    wp = w + 2
    s = (h + 2) * wp
    return {"h": h, "w": w, "wp": wp, "S": s, "slots": slots, "junk": junk, "R": (slots * s + 63) // 64 * 64, "g": 2 * s + wp + 1}


def interior(L):
    """The rows of one frame that are pixels."""
    m = np.zeros(L["S"])
    for y in range(L["h"]):
        m[(y + 1) * L["wp"] + 1:(y + 1) * L["wp"] + 1 + L["w"]] = 1.0
    return m


_made = set()


def put(name, a):
    if name not in _made:
        f32(P(name + ".f32"), a)
        _made.add(name)
    return name + ".f32"


def mask_file(L, hide=0):
    """[R, 1]: 1 on the pixels of the frames from `hide` on (the first `junk` slots are the zero frames, always 0)."""
    h0 = hide if MUT == "visible-zero-frame" else max(hide, L["junk"])
    name = "mask_%d_%d_%d_%d_%d" % (L["h"], L["w"], L["slots"], L["junk"], h0)
    m = np.zeros(L["R"])
    fr = interior(L)
    for f in range(h0, L["slots"]):
        m[f * L["S"]:(f + 1) * L["S"]] = fr
    return put(name, m[:, None])


def conv_w3t(name):  # [O, C, 3, 3, 3] -> [O, 27 C], tap-major (dt, dy, dx), channels last
    w = W(name + ".weight")
    return w.transpose(0, 2, 3, 4, 1).reshape(w.shape[0], -1)


def conv_w3(name):  # [O, C, 3, 3] -> [O, 9 C]
    w = W(name + ".weight")
    return w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1)


def conv_w1(name):  # [O, C, 1, 1(, 1)] -> [O, C]
    w = W(name + ".weight")
    return w.reshape(w.shape[0], w.shape[1])


prog = []
n_prog = 0
uid = [0]


def fresh(prefix):
    uid[0] += 1
    return "%s%d" % (prefix, uid[0])


def program(name, kernel, args):
    global n_prog
    n_prog += 1
    path = kernel if kernel.startswith("/") else f"{CHX}/{kernel}.chx"
    prog.append(f"program {name} {path} {args}")


res = lambda target, fname, flag="f32": prog.append(f"resident {target} {P(fname)} {flag}")
link = lambda a, b: prog.append(f"link {a} -> {b}")


def vecfile(name, vec):
    f32(P(name + ".f32"), np.asarray(vec)[None])
    return name + ".f32"


def conv3t(src, cin, cout, L, weights):
    n = fresh("cv")
    f16(P(n + ".f16"), weights)
    k16 = cin % 32 != 0 or cout % 32 != 0
    program(n, "conv3t_k16" if k16 else "conv3t", f"rp={L['R']} cin={cin} cout={cout} wp={L['wp']} s={L['S']} g={L['g']}")
    link(src, n + ".x")
    res(n + ".w", n + ".f16", "f16")
    return n + ".y"


def conv3(src, cin, cout, L, weights):
    n = fresh("cs")
    f16(P(n + ".f16"), weights)
    program(n, "conv3_k16" if (cin % 32 or cout % 32) else CONV3, f"rp={L['R']} cin={cin} cout={cout} wp={L['wp']} g={L['g']}")
    link(src, n + ".x")
    res(n + ".w", n + ".f16", "f16")
    return n + ".y"


def conv1(src, cin, cout, L, weights):
    n = fresh("c1")
    f16(P(n + ".f16"), weights)
    program(n, "conv1", f"r={L['R']} cin={cin} cout={cout} g={L['g']}")
    link(src, n + ".x")
    res(n + ".w", n + ".f16", "f16")
    return n + ".y"


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
    a = conv3t(norm(x, W(p + ".norm1.gamma").reshape(-1), cin, L), cin, cout, L, conv_w3t(p + ".conv1"))
    a = bias(a, W(p + ".conv1.bias"), cout, L)
    a = conv3t(norm(a, W(p + ".norm2.gamma").reshape(-1), cout, L), cout, cout, L, conv_w3t(p + ".conv2"))
    if cin != cout:
        h = bias(conv1(pad16(x, cin, L), cin, cout, L, conv_w1(p + ".conv_shortcut")), W(p + ".conv_shortcut.bias"), cout, L)
    else:
        h = x
    return bias_res(a, W(p + ".conv2.bias"), h, cout, L)


def attention(x, c, L, p):
    nrm = norm(x, W(p + ".norm.gamma").reshape(-1), c, L, silu=False)
    qkv = bias(conv1(nrm, c, 3 * c, L, conv_w1(p + ".to_qkv")), W(p + ".to_qkv.bias"), 3 * c, L)
    sp = fresh("sp")
    program(sp, "vae_split16", f"r={L['R']} c={c} sc={1 / math.sqrt(c)!r}")
    link(qkv, sp + ".qkv")
    sc = fresh("sc")
    program(sc, "ported/gemm_mma", f"m={L['R']} n={L['R']} k={c}")
    link(sp + ".q", sc + ".a"); link(sp + ".k", sc + ".w")
    ma = fresh("ma")
    program(ma, "vae_mask_add2", f"n={L['R']}")
    link(sc + ".y", ma + ".sc")
    mk = np.full((L["R"], L["R"]), NEG, dtype=np.float32)
    fr = interior(L) > 0
    for f in range(L["slots"]):
        mk[f * L["S"]:(f + 1) * L["S"], f * L["S"]:(f + 1) * L["S"]][:, fr] = 0.0  # a frame's rows (pixels and border) see that frame's pixels
    name = "attn_mask_%d_%d_%d" % (L["h"], L["w"], L["slots"])
    if name not in _made:
        f32(P(name + ".f32"), mk)
        _made.add(name)
    res(ma + ".mk", name + ".f32")
    st = fresh("st")
    program(st, "dit_stats", f"nh=1 nq={L['R']} t={L['R']} qs=32 cs=64")
    link(ma + ".z", st + ".sc")
    pv = fresh("pv")
    program(pv, "dit_pv", f"nh=1 nq={L['R']} t={L['R']} qs=32 cs=64 dw=32 d={c}")
    link(ma + ".z", pv + ".sc"); link(st + ".m", pv + ".m"); link(st + ".z", pv + ".z"); link(sp + ".v", pv + ".v")
    pr = conv1(pad16(pv + ".o", c, L), c, c, L, conv_w1(p + ".proj"))
    return bias_res(pr, W(p + ".proj.bias"), x, c, L)


def selection(cin, cout, ft, k, a, b):
    """DupUp3D's channel rearrangement for time slot k and sub-position (a, b): output channel o reads input channel ((o ft + k) 4 + 2a + b) // reps, reps = out ft 4 / in."""
    reps = cout * ft * 4 // cin
    sel = np.zeros((cout, cin))
    for o in range(cout):
        sel[o, ((o * ft + k) * 4 + 2 * a + b) // reps] = 1.0
    return sel


def upblock(x, x0, L, cin, cout, p, temporal, last):
    """After the three residual blocks (x, at layout L, `cout` channels): the upsample and the shortcut of the block's input x0 (cin channels). Returns the new matrix name and layout."""
    if not temporal:
        L2 = lay(L["h"] * 2, L["w"] * 2, L["slots"], L["junk"])
        up = fresh("up")
        program(up, "vae_up2v", f"n={L['slots']} h={L['h']} w={L['w']} c={cout} s={L['S']} s2={L2['S']} r={L['R']} r2={L2['R']} g2={L2['g']}")
        link(x, up + ".x")
        cv = conv3(up + ".y", cout, cout, L2, conv_w3(p + "upsampler.resample.1"))
        x0h = pad16(x0, cin, L)
        s = [conv1(x0h, cin, cout, L, selection(cin, cout, 1, 0, a, b)) for a in range(2) for b in range(2)]
        ep = fresh("ep")
        program(ep, "vae_up2_epiv", f"n={L['slots']} h={L['h']} w={L['w']} c={cout} s={L['S']} s2={L2['S']} r={L['R']} r2={L2['R']}")
        link(cv, ep + ".cv")
        res(ep + ".b", vecfile(ep + "b", W(p + "upsampler.resample.1.bias")))
        res(ep + ".fa", put("fa_%d_%d_%s" % (L["slots"], L["junk"], MUT == "visible-zero-frame"), np.array([0.0 if MUT != "visible-zero-frame" else 1.0] * L["junk"] + [1.0] * (L["slots"] - L["junk"]))[:, None]))
        for name, src in zip(("s00", "s01", "s10", "s11"), s):
            link(src, ep + "." + name)
        return ep + ".z", L2
    n = L["slots"] - L["junk"]  # the frames that are real
    so = L["S"] if L["junk"] else 0
    hide = 0 if MUT == "sees-first" else L["junk"] + 1
    xh = fresh("pm")
    program(xh, "vae_pad16m", f"r={L['R']} c={cout} g={L['g']}")
    link(x, xh + ".x")
    res(xh + ".m", mask_file(L, hide))
    tw = W(p + "upsampler.time_conv.weight")[:, :, :, 0, 0].transpose(0, 2, 1).reshape(2 * cout, 3 * cout)  # [2c, c, 3] -> [2c, 3c], tap-major
    tb = W(p + "upsampler.time_conv.bias")
    if MUT == "swap-halves":
        tw = np.concatenate([tw[cout:], tw[:cout]])
        tb = np.concatenate([tb[cout:], tb[:cout]])
    ytc = fresh("ct")
    f16(P(ytc + ".f16"), tw)
    program(ytc, "conv_t3", f"rp={L['R']} cin={cout} cout={2 * cout} s={L['S']} g={L['g']}")
    link(xh + ".y", ytc + ".x")
    res(ytc + ".w", ytc + ".f16", "f16")
    Lt = lay(L["h"], L["w"], 2 * n, 1)
    ti = fresh("ti")
    program(ti, "vae_tint", f"n={n} c={cout} s={L['S']} so={so} r={L['R']} r2={Lt['R']}")
    link(ytc + ".y", ti + ".y2"); link(x, ti + ".x3")
    res(ti + ".b", vecfile(ti + "b", tb))
    fm = np.array([0.0] + [1.0] * (n - 1))[:, None]
    res(ti + ".fm", put("fm_%d" % n, fm))
    res(ti + ".mk", put("mk1_%d_%d" % (L["h"], L["w"]), interior(L)[:, None]))
    L2 = lay(L["h"] * 2, L["w"] * 2, 2 * n, 1)
    up = fresh("up")
    program(up, "vae_up2v", f"n={2 * n} h={L['h']} w={L['w']} c={cout} s={L['S']} s2={L2['S']} r={Lt['R']} r2={L2['R']} g2={L2['g']}")
    link(ti + ".z", up + ".x")
    cv = conv3(up + ".y", cout, cout, L2, conv_w3(p + "upsampler.resample.1"))
    x0h = pad16(x0, cin, L)
    s = [conv1(x0h, cin, cout, L, selection(cin, cout, 2, k, a, b)) for k in range(2) for a in range(2) for b in range(2)]
    ep = fresh("ep")
    program(ep, "vae_up3_epi", f"n={n} h={L['h']} w={L['w']} c={cout} s={L['S']} s2={L2['S']} so={so} r={L['R']} r2={L2['R']}")
    link(cv, ep + ".cv")
    res(ep + ".b", vecfile(ep + "b", W(p + "upsampler.resample.1.bias")))
    res(ep + ".fa", put("fm1_%d" % n if MUT == "visible-zero-frame" else "fm_%d" % n, np.ones((n, 1)) if MUT == "visible-zero-frame" else fm))
    for i, src in enumerate(s):
        link(src, ep + ".s%d%d%d" % (i // 4, (i // 2) % 2, i % 2))
    return ep + ".z", L2


# ---- the decoder
z = np.load(arg("--z")).astype(np.float64) if "--z" in sys.argv else wv.latent(F, H0, W0, SEED)  # [48, F, h, w]; --z: a latent as the pipeline hands it to the decoder (z * std + mean)
F, H0, W0 = z.shape[1:]
L = lay(H0, W0, F, 0)
zr = np.zeros((L["R"] + 2 * L["g"], 64))
for f in range(F):
    for y in range(H0):
        for xx in range(W0):
            zr[L["g"] + f * L["S"] + (y + 1) * L["wp"] + xx + 1, :48] = z[:, f, y, xx]
f16(P("z.f16"), zr)
pq = fresh("c1")
wq = np.zeros((64, 64))
wq[:48, :48] = conv_w1("post_quant_conv")
f16(P(pq + ".f16"), wq)
program(pq, "conv1", f"r={L['R']} cin=64 cout=64 g={L['g']}")
res(pq + ".x", "z.f16", "f16"); res(pq + ".w", pq + ".f16", "f16")
x = bias(pq + ".y", np.concatenate([W("post_quant_conv.bias"), np.zeros(16)]), 64, L)
wi = np.zeros((1024, 27, 64))
wi[:, :, :48] = conv_w3t("decoder.conv_in").reshape(1024, 27, 48)
x = bias(conv3t(pad16(x, 64, L), 64, 1024, L, wi.reshape(1024, -1)), W("decoder.conv_in.bias"), 1024, L)
x = resnet(x, 1024, 1024, L, "decoder.mid_block.resnets.0")
x = attention(x, 1024, L, "decoder.mid_block.attentions.0")
x = resnet(x, 1024, 1024, L, "decoder.mid_block.resnets.1")
outs = [1024, 1024, 512, 256]
ins = [1024, 1024, 1024, 512]
temporal = [True, True, False, False]
for i in range(4):
    p = "decoder.up_blocks.%d." % i
    x0, cin, cout = x, ins[i], outs[i]
    for r in range(3):
        x = resnet(x, cin if r == 0 else cout, cout, L, p + "resnets.%d" % r)
    if i < 3:
        x, L = upblock(x, x0, L, cin, cout, p, temporal[i], i == 2)
x = norm(x, W("decoder.norm_out.gamma").reshape(-1), 256, L)
wout = np.zeros((16, 27 * 256))
wout[:12] = conv_w3t("decoder.conv_out")
y = conv3t(x, 256, 16, L, wout)
program("probe", "dit_rows2", f"m={L['R']} n=16 bn=16")
link(y, "probe.x")
prog.append(f"save {y} {P('y.f32')} f32")
prog.append("steps")
prog.append("answer probe.y")
open(P("wan_vae.session"), "w").write("\n".join(prog) + "\n")

print("latent [48, %d, %d, %d]: decoded layout %d x %d, %d slots; %d programs" % (F, H0, W0, L["h"], L["w"], L["slots"], n_prog), flush=True)
t0 = time.time()
r = subprocess.run([SESSION, P("wan_vae.session")], input="1\ncheckpoint\n", capture_output=True, text=True, timeout=7200)
secs = time.time() - t0
if r.returncode != 0 or not os.path.exists(P("y.f32")):
    print("FAIL the session did not answer: status %d, %s" % (r.returncode, r.stderr.strip()[:800]))
    sys.exit(1)
out = np.fromfile(P("y.f32"), dtype="<f4").reshape(L["R"], 16).astype(np.float64)
bo = W("decoder.conv_out.bias")
T = L["slots"] - L["junk"]
v12 = np.zeros((12, T, L["h"], L["w"]))
for f in range(L["junk"], L["slots"]):
    for yy in range(L["h"]):
        row = f * L["S"] + (yy + 1) * L["wp"] + 1
        v12[:, f - L["junk"], yy] = (out[row:row + L["w"], :12] + bo).T
c, t, h, w = v12.shape
got = np.clip(v12.reshape(3, 2, 2, t, h, w).transpose(0, 3, 4, 2, 5, 1).reshape(3, t, h * 2, w * 2), -1.0, 1.0)
print("session: %d programs answered in %.1f s (Metal; includes the compile and the upload of the weights); video %s" % (n_prog, secs, list(got.shape)), flush=True)
if "--save" in sys.argv:
    np.save(arg("--save"), got)
if "--no-ref" in sys.argv:
    sys.exit(0)
want = wv.decode(z)
rms = float(np.sqrt((want ** 2).mean()))
d = np.abs(got - want)
BOUND = 5e-3  # as the 2-D decoder's (tests/vae_gpu.py): written before the first run
ok = bool(d.mean() / rms <= BOUND)
print("GPU decoder vs float64: mean abs error %.4g (%.3g of the output rms %.3g), max %.4g" % (d.mean(), d.mean() / rms, rms, d.max()))
print("%s the Wan VAE decoder on the GPU (mean error against the float64 decoder within %.0e of the output's rms)" % ("ok  " if ok else "FAIL", BOUND))
sys.exit(0 if ok else 1)
