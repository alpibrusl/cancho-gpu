#!/usr/bin/env python3
"""docs/resident.md section 26 (roadmap M9, V1): the ResNet-18-class backbone's kernels held to float64 numpy, then the whole network to its specification and to torch.

    python3 tests/resnet.py build/cancho-gpu-session [build/cancho-gpu-metal-session]

1. The specification (examples/cnn/resnet_ref.py) equals torch's float64 autograd on a ResNet-18 of the same structure (the 4-block configuration and the full 11.2 M parameters at 96 x 128).
2. Each new kernel on its own, on the CPU interpreter and (given) on Metal, against numpy on f16-representable inputs: `convk` in the four windows the network uses (3x3, the 2x2 and 1x1 of a
   stride-2 convolution on a space-to-depth image, the 4x4 of the 7x7 stem) against an explicit zero-padded convolution of the image (which is the claim that a tap two pixels past the edge is absorbed
   by the border), its two gradients, `wsum_mask`, `s2d_fwd`/`s2d_bwd`, `maxpool_fwd`/`maxpool_bwd` (with ties) against the image-level max-pool, the global average and the packing of a
   non-square frame, and the BatchNorm kernels without the ReLU. Then one deliberate change to each of nine kernels, which these checks must catch.
3. The whole network (stem, max-pool, four stages with three stride-2 blocks and their 1x1 shortcuts; 64 x 96 images): the loss and the gradients of all tensors against the specification that rounds
   to binary16 where the device does, and against torch float32; then 25 AdamW steps, whose loss must fall, the CPU and Metal sessions following each other.
Needs numpy (and torch for 1 and the torch comparisons); without numpy the test says so and passes."""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CNN = os.path.join(ROOT, "examples", "cnn")
sys.path.insert(0, CNN)
try:
    import numpy as np
    import resnet_ref as R
    import rn_session as S
except ImportError:
    print("skip: numpy is not installed")
    sys.exit(0)
try:
    import resnet_torch as T
    import torch
except ImportError:
    T = None

bad = 0
G = S.G


def say(good, text):
    global bad
    print(f"{'ok  ' if good else 'FAIL'} {text}")
    bad += 0 if good else 1


def serve(binary, session, lines, cpu):
    r = subprocess.run([binary, session] + (["--target", "apple-m-series"] if cpu else []), input=lines, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise SystemExit(f"{binary}: {r.stderr[:400]}")
    return [np.array([float(v) for v in line.split()]) for line in r.stdout.splitlines()]


def f16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float64)


def guarded(x):
    """[n, h, w, c] -> the padded, guarded f16 matrix [g + n (h+2)(w+2) + g, c] the kernels read."""
    n, h, w, c = x.shape
    out = np.zeros((G + n * (h + 2) * (w + 2) + G, c))
    out[G:-G] = padded(x)
    return out


def padded(x):
    n, h, w, c = x.shape
    xp = np.zeros((n, h + 2, w + 2, c))
    xp[:, 1:-1, 1:-1] = x
    return xp.reshape(-1, c)


def interior(m, n, h, w, c):
    return m.reshape(n, h + 2, w + 2, c)[:, 1:-1, 1:-1]


class Rig:
    """One-program sessions: write the inputs as files, run, read the answers."""

    def __init__(self, binaries):
        self.dir = tempfile.mkdtemp()
        self.binaries = binaries
        self.k = 0

    def run(self, kernel, consts, inputs, outputs, sched=None, edit=None, binaries=None):
        """inputs: {name: (array, "f32" | "f16")}; answers one flat array (the outputs concatenated) for each binary. `edit` = (old, new) changes the kernel's text first (a mutant)."""
        self.k += 1
        d = os.path.join(self.dir, str(self.k))
        os.makedirs(d)
        src = open(os.path.join(CNN, kernel)).read()
        if edit:
            assert edit[0] in src, (kernel, edit[0])
            src = src.replace(*edit)
        if sched:
            src = "\n".join((l.split("{")[0] + "{ " + sched + " }") if l.startswith("schedule ") and "apple-m-series" in l else l for l in src.split("\n"))
        kpath = os.path.join(d, kernel)
        open(kpath, "w").write(src)
        lines = [f"program k {kpath} " + " ".join(f"{a}={b}" for a, b in consts.items())]
        for name, (arr, ty) in inputs.items():
            S.put(os.path.join(d, name), arr, "<f2" if ty == "f16" else "<f4")
            lines.append(f"resident k.{name} {os.path.join(d, name)} {ty}")
        lines += [f"answer k.{o}" for o in outputs]
        open(os.path.join(d, "session"), "w").write("\n".join(lines) + "\n")
        return [serve(b, os.path.join(d, "session"), "\n", cpu="metal" not in b)[0] for b in (binaries or self.binaries)]


def split(out, sizes):
    o, r = 0, []
    for s in sizes:
        r.append(out[o:o + s])
        o += s
    return r


def close(got, want, rel):
    return float(np.abs(got - want).max()) <= rel * max(float(np.abs(want).max()), 1e-12)


def win_conv(x, w, oy, ox):
    """The explicit convolution the flat shifted reads must equal: out(y, x) = sum over taps of x(y + dy - oy, x + dx - ox)[:cin] w[:, dy, dx], zero outside the image. x [n,h,w,cx], w [co,kh,kw,cin]."""
    n, h, wd, _ = x.shape
    co, kh, kw, cin = w.shape
    y = np.zeros((n, h, wd, co))
    for dy in range(kh):
        for dx in range(kw):
            sy, sx = dy - oy, dx - ox
            ys, ye = max(0, -sy), min(h, h - sy)
            xs, xe = max(0, -sx), min(wd, wd - sx)
            y[:, ys:ye, xs:xe] += np.einsum("nhwc,oc->nhwo", x[:, ys + sy:ye + sy, xs + sx:xe + sx, :cin], w[:, dy, dx])
    return y


def win_conv_b(dz, x, w, oy, ox):
    n, h, wd, cx = x.shape
    co, kh, kw, cin = w.shape
    gx, gw = np.zeros((n, h, wd, cx)), np.zeros_like(w)
    for dy in range(kh):
        for dx in range(kw):
            sy, sx = dy - oy, dx - ox
            ys, ye = max(0, -sy), min(h, h - sy)
            xs, xe = max(0, -sx), min(wd, wd - sx)
            gx[:, ys + sy:ye + sy, xs + sx:xe + sx, :cin] += np.einsum("nhwo,oc->nhwc", dz[:, ys:ye, xs:xe], w[:, dy, dx])
            gw[:, dy, dx] = np.einsum("nhwo,nhwc->oc", dz[:, ys:ye, xs:xe], x[:, ys + sy:ye + sy, xs + sx:xe + sx, :cin])
    return gx, gw


def s2d_ref(x):
    n, h, w, c = x.shape
    return x.reshape(n, h // 2, 2, w // 2, 2, c).transpose(0, 1, 3, 2, 4, 5).reshape(n, h // 2, w // 2, 4 * c)


def s2d_back_ref(g):
    n, h, w, c4 = g.shape
    return g.reshape(n, h, w, 2, 2, c4 // 4).transpose(0, 1, 3, 2, 4, 5).reshape(n, 2 * h, 2 * w, c4 // 4)


def sels(wb):
    out = []
    for px in range(2):
        s = np.zeros((wb, 2 * wb))
        s[np.arange(wb), 2 * np.arange(wb) + px] = 1.0
        out.append(s)
    return out


def specification_equals_torch():
    if T is None:
        print("skip: torch is not installed (the specification is not compared with it)")
        return
    for c in (R.Config(h=32, w=64, widths=(4, 4, 6, 8), blocks=(1, 2, 1, 1), classes=5, n=4), R.Config(h=96, w=128, n=2)):
        P = R.init_params(c, 3)
        rng = np.random.RandomState(4)
        for k in P:
            P[k] = P[k] + (0.3 * rng.normal(size=P[k].shape) if k.startswith(("g_", "b_")) else 0.05 * rng.normal(size=P[k].shape))
        x, y = rng.normal(size=(c.n, c.h, c.w, c.cin)), rng.randint(0, c.classes, size=c.n)
        loss, g = R.forward(c, P, x, y, grads=True)
        lt, gt, _ = T.step(c, P, x, y)
        worst = max(float(np.abs(g[k] - gt[k]).max() / (np.abs(gt[k]).max() + 1e-30)) for k in g)
        say(abs(loss - lt) < 1e-9 and worst < 1e-9, f"the specification equals torch float64 autograd on a ResNet-18 structure {c.h}x{c.w} widths {c.widths} ({sum(v.size for v in P.values())} parameters): loss {loss:.8f} / {lt:.8f}, worst gradient difference {worst:.1e}")


def kernels(binaries):
    rig = Rig(binaries)
    rng = np.random.RandomState(7)
    names = [os.path.basename(b) for b in binaries]
    n, h, w, co, g = 2, 6, 8, 32, G
    wp, rp = w + 2, n * (h + 2) * (w + 2)
    for kind, (kh, kw, oy, ox, cx, cin) in {"3x3": (3, 3, 1, 1, 16, 16), "2x2 of a stride-2 3x3": (2, 2, 1, 1, 32, 32), "1x1 of a stride-2 shortcut (reads 8 of 32 channels)": (1, 1, 0, 0, 32, 8), "4x4 of the 7x7 stem": (4, 4, 2, 2, 16, 16)}.items():
        x = f16(rng.normal(size=(n, h, w, cx)))
        wt = f16(rng.normal(size=(co, kh, kw, cin)))
        consts = dict(rp=rp, cx=cx, cin=cin, cout=co, wp=wp, g=g, kh=kh, kw=kw, oy=oy, ox=ox)
        bk = min(cin, 16)
        outs = rig.run("convk.chx", consts, {"x": (guarded(x), "f16"), "w": (wt.reshape(co, -1), "f16")}, ["y"], f"threads 128 warps 2 2 bm 32 bn 32 bk {bk}")
        ref = win_conv(x, wt, oy, ox)
        for nm, o in zip(names, outs):
            say(close(interior(o.reshape(rp, co), n, h, w, co), ref, 1e-4), f"{nm}: convk, the {kind} window, equals an explicit zero-padded convolution of the image")
        dz = f16(rng.normal(size=(n, h, w, co)))
        gx_ref, gw_ref = win_conv_b(dz, x, wt, oy, ox)
        outs = rig.run("convk_dx.chx", consts, {"dz": (guarded(dz), "f16"), "w": (wt.reshape(co, -1), "f16")}, ["gx"], f"threads {64 if cin < 16 else 128} warps 2 {1 if cin < 16 else 2} bm 32 bn {min(cin, 16)} bk 16")
        for nm, o in zip(names, outs):
            gx = interior(o.reshape(rp, cx), n, h, w, cx)
            say(close(gx[..., :cin], gx_ref[..., :cin], 1e-4) and not np.abs(gx[..., cin:]).any(), f"{nm}: convk_dx, the {kind} window, equals the adjoint (columns beyond cin stay zero)")
        rs, ns = 32, rp // 32
        tn = min(cin, 16)
        wr, wc = 2, (2 if tn % 16 == 0 else 1)
        sched = f"threads {32 * wr * wc} warps {wr} {wc} bm 16 bn {tn} bk 8"
        outs = rig.run("convk_dw.chx", dict(consts, rs=rs), {"dz": (guarded(dz), "f16"), "x": (guarded(x), "f16")}, ["part"], sched)
        mask = (rng.uniform(size=(1, kh * kw * cin)) > 0.3).astype(float)
        for bi, (nm, o) in enumerate(zip(names, outs)):
            part = o.reshape(ns, co, kh * kw * cin)
            say(close(part.sum(0).reshape(co, kh, kw, cin), gw_ref, 1e-4), f"{nm}: convk_dw, the {kind} window, summed over {ns} pieces equals the weight gradient")
            sm = rig.run("wsum_mask.chx", dict(ns=ns, r=co, k=kh * kw * cin, bm=16, bn=S.block(kh * kw * cin, 128)), {"part": (part.reshape(-1), "f32"), "mk": (mask, "f32")}, ["dw"], binaries=[binaries[bi]])[0]
            say(close(sm.reshape(co, -1), gw_ref.reshape(co, -1) * mask, 1e-4), f"{nm}: wsum_mask adds the pieces and zeroes the masked weights")
    # space to depth
    h2, w2, c = 3, 8, 8
    img = f16(rng.normal(size=(n, 2 * h2, 2 * w2, c)))
    for wb in (4, 8):
        s0, s1 = sels(wb)
        outs = rig.run("s2d_fwd.chx", dict(n=n, h=h2, w=w2, c=c, g=g, wb=wb), {"x": (guarded(img), "f16"), "sel0": (s0, "f32"), "sel1": (s1, "f32")}, ["s"])
        rps = n * (h2 + 2) * (w2 + 2)
        for nm, o in zip(names, outs):
            o = o.reshape(rps + 2 * g, 4 * c)
            full = o[g:g + rps].reshape(n, h2 + 2, w2 + 2, 4 * c)
            say(close(full[:, 1:-1, 1:-1], s2d_ref(img), 1e-3) and not full[:, 0].any() and not full[:, -1].any() and not full[:, :, 0].any() and not full[:, :, -1].any() and not np.abs(o[:g]).any(),
                f"{nm}: s2d_fwd (blocks of {wb} pixels) is the space-to-depth image, zero on the border")
        gs = rng.normal(size=(n, h2, w2, 4 * c))
        outs = rig.run("s2d_bwd.chx", dict(n=n, h=h2, w=w2, c=c, wb=wb), {"gs": (padded(gs), "f32"), "selt0": (s0.T.copy(), "f32"), "selt1": (s1.T.copy(), "f32")}, ["gi"])
        for nm, o in zip(names, outs):
            gi = o.reshape(n, 2 * h2 + 2, 2 * w2 + 2, c)
            say(close(gi[:, 1:-1, 1:-1], s2d_back_ref(gs), 1e-5) and not np.abs(gi[:, 0]).any() and not np.abs(gi[:, :, 0]).any() and not np.abs(gi[:, -1]).any() and not np.abs(gi[:, :, -1]).any(),
                f"{nm}: s2d_bwd (blocks of {wb}) is the inverse permutation, zero on the border")
    # max-pooling: the image-level reference, with ties among positive values (and, in the second case, random ReLU outputs)
    hp_, wq, cm = 6, 8, 8
    for label, vals in (("ties among positive values", rng.choice([0.5, 1.0, 1.5], size=(n, 2 * hp_, 2 * wq, cm))), ("ReLU outputs", np.maximum(rng.normal(size=(n, 2 * hp_, 2 * wq, cm)), 0))):
        a_img = f16(vals)
        s_img = s2d_ref(a_img)
        pooled, idx = R.maxpool_f(a_img)
        rpm = n * (hp_ + 2) * (wq + 2)
        mk = padded(np.ones((n, hp_, wq, 1)))[:, 0]
        outs = rig.run("maxpool_fwd.chx", dict(rp=rpm, c=cm, wp=wq + 2, g=g, bm=16), {"s": (guarded(s_img), "f16"), "mk": (mk, "f32")}, ["a", "idx"])
        for nm, o in zip(names, outs):
            a, ix = split(o, [(rpm + 2 * g) * cm] * 2)
            a, ix = a.reshape(rpm + 2 * g, cm), ix.reshape(rpm + 2 * g, cm)
            ixi = ix[g:g + rpm].reshape(n, hp_ + 2, wq + 2, cm)
            ok_idx = np.array_equal(ixi[:, 1:-1, 1:-1], idx) if label.startswith("ties") else True
            say(close(interior(a[g:g + rpm], n, hp_, wq, cm), pooled, 1e-6) and ok_idx and (ixi[:, 0] == -1).all() and not np.abs(a[:g]).any(), f"{nm}: maxpool_fwd on {label}: the maximum{' and the first-maximum index' if ok_idx else ''} equal the image-level max-pool, border marked -1")
        gr = rng.normal(size=(n, hp_, wq, cm))
        dx_img = R.maxpool_b(gr, idx, a_img.shape)
        ixg = np.zeros((rpm + 2 * g, cm))
        ixg[g:g + rpm] = -1.0
        ixg[g:g + rpm].reshape(n, hp_ + 2, wq + 2, cm)[:, 1:-1, 1:-1] = idx
        outs = rig.run("maxpool_bwd.chx", dict(rp=rpm, c=cm, wp=wq + 2, g=g, bm=16), {"gr": (guarded(gr), "f32"), "idx": (ixg, "f32")}, ["gs"])
        for nm, o in zip(names, outs):
            gsk = interior(o.reshape(rpm, 4 * cm), n, hp_, wq, 4 * cm)
            if label.startswith("ties"):
                say(close(gsk, s2d_ref(dx_img), 1e-5), f"{nm}: maxpool_bwd on {label}: the gradient reaches exactly the winners the image-level max-pool's backward gives")
    # the global average, the packing of a frame, the guarded sum
    hh, ww, cc = 3, 4, 32
    a_img = f16(rng.normal(size=(n, hh, ww, cc)))
    vm = np.zeros((hh + 2, ww + 2))
    vm[1:-1, 1:-1] = 1.0 / (hh * ww)
    rc = next(b for b in range(20, 0, -1) if ((hh + 2) * (ww + 2)) % b == 0)
    outs = rig.run("gap2_fwd.chx", dict(n=n, h=hh, w=ww, c=cc, g=g, rc=rc), {"a": (guarded(a_img), "f16"), "vm": (vm.reshape(-1), "f32")}, ["f"])
    gf = rng.normal(size=(n, cc))
    outs2 = rig.run("gap2_bwd.chx", dict(n=n, h=hh, w=ww, c=cc, rc=rc), {"gf": (gf, "f32"), "mc": (vm.reshape(-1), "f32")}, ["gi"])
    want = np.zeros((n, hh + 2, ww + 2, cc))
    want[:, 1:-1, 1:-1] = gf[:, None, None, :] / (hh * ww)
    for nm, o, o2 in zip(names, outs, outs2):
        say(close(o.reshape(n, cc), f16(a_img.mean((1, 2))), 1e-3) and close(o2.reshape(-1, cc), want.reshape(-1, cc), 1e-5), f"{nm}: gap2_fwd and gap2_bwd on a {hh}x{ww} map are the mean over the valid pixels and its spread")
    n2, hf, wf = 2, 6, 10
    ia, ib = f16(rng.normal(size=(n2, hf, wf, 3))), f16(rng.normal(size=(n2, hf, wf, 3)))
    cmat = np.zeros((3, 4))
    cmat[:, :3] = np.eye(3)
    want = np.zeros((2 * n2, hf + 2, wf + 2, 4))
    want[:, 1:-1, 1:-1, :3] = np.concatenate([ia, ib])
    outs = rig.run("pack_hw.chx", dict(n2=n2, h=hf, w=wf, cin=3, cp=4, g=g), {"xa": (ia.reshape(-1, 3), "f16"), "xb": (ib.reshape(-1, 3), "f16"), "cmat": (cmat, "f32")}, ["o"])
    ra, rb_ = rng.normal(size=(64, 8)), rng.normal(size=(64, 8))
    outs2 = rig.run("add2g.chx", dict(r=64, c=8, bm=16, g=g), {"a": (ra, "f32"), "b": (rb_, "f32")}, ["y"])
    for nm, o, o2 in zip(names, outs, outs2):
        say(close(o.reshape(-1, 4)[g:-g], want.reshape(-1, 4), 1e-3) and not np.abs(o[:g * 4]).any() and not np.abs(o[-g * 4:]).any(), f"{nm}: pack_hw pads the borders and channels of a 6x10 frame, two half batches")
        o2 = o2.reshape(-1, 8)
        say(close(o2[g:g + 64], ra + rb_, 1e-6) and not np.abs(o2[:g]).any() and not np.abs(o2[g + 64:]).any(), f"{nm}: add2g adds into a guarded matrix")
    # BatchNorm without the ReLU
    c, m_ = 32, 4 * h * w
    rpb = n * (h + 2) * (w + 2)
    mkb = padded(np.ones((n, h, w, 1)))[:, 0]
    z = rng.normal(2.0, 3.0, size=(rpb, c))
    zv = z[mkb > 0]
    mean, istd = zv.mean(0), 1 / np.sqrt(zv.var(0) + 1e-5)
    gamma, beta = rng.normal(1, 0.3, size=c), rng.normal(0, 0.3, size=c)
    sc, sh = gamma * istd, beta - mean * gamma * istd
    outs = rig.run("bn_apply_lin.chx", dict(rp=rpb, c=c, bm=16, g=g), {"z": (z, "f32"), "sc": (sc, "f32"), "sh": (sh, "f32"), "mk": (mkb, "f32")}, ["a"])
    gr = rng.normal(size=(rpb, c))
    d = gr * mkb[:, None]
    xhat = (z - mean) * istd
    dbeta, dgamma = d.sum(0), (d * xhat).sum(0)
    for nm, o in zip(names, outs):
        o = o.reshape(rpb + 2 * g, c)
        say(close(o[g:g + rpb], f16((z * sc + sh) * mkb[:, None]), 1e-3) and not np.abs(o[:g]).any(), f"{nm}: bn_apply_lin is z scale + shift, masked, with negative values kept")
    rb2 = 32
    o = rig.run("bn_bwd_part_lin.chx", dict(rp=rpb, c=c, rb=rb2), {"gr": (gr, "f32"), "z": (z, "f32"), "mean": (mean, "f32"), "istd": (istd, "f32"), "v": (mkb, "f32")}, ["p1", "p2"])
    o3 = rig.run("bn_bwd_apply_lin.chx", dict(rp=rpb, c=c, bm=16, g=g, m=m_), {"gr": (gr, "f32"), "z": (z, "f32"), "mean": (mean, "f32"), "istd": (istd, "f32"), "gamma": (gamma, "f32"), "dbeta": (dbeta, "f32"), "dgamma": (dgamma, "f32"), "mk": (mkb, "f32")}, ["dz"])
    want = gamma * istd * (d - dbeta / m_ - xhat * dgamma / m_) * mkb[:, None]
    for nm, a, b3 in zip(names, o, o3):
        p1, p2 = split(a, [rpb // rb2 * c] * 2)
        say(close(p1.reshape(-1, c).sum(0), dbeta, 1e-4) and close(p2.reshape(-1, c).sum(0), dgamma, 1e-4), f"{nm}: bn_bwd_part_lin sums d and d xhat with d = gr")
        say(close(b3.reshape(rpb + 2 * g, c)[g:g + rpb], f16(want), 2e-3), f"{nm}: bn_bwd_apply_lin: dz = gamma istd (gr - dbeta/m - xhat dgamma/m)")


def mutants(binary):
    """Each new kernel with one deliberate change that its test must catch (CPU interpreter)."""
    rig = Rig([binary])
    rng = np.random.RandomState(9)
    n, h, w, co, g = 2, 6, 8, 32, G
    wp, rp = w + 2, n * (h + 2) * (w + 2)
    cx = cin = 16
    x, wt, dz = f16(rng.normal(size=(n, h, w, cx))), f16(rng.normal(size=(co, 4, 4, cin))), f16(rng.normal(size=(n, h, w, co)))
    consts = dict(rp=rp, cx=cx, cin=cin, cout=co, wp=wp, g=g, kh=4, kw=4, oy=2, ox=2)
    gx_ref, gw_ref = win_conv_b(dz, x, wt, 2, 2)
    verdicts = []
    o = rig.run("convk.chx", consts, {"x": (guarded(x), "f16"), "w": (wt.reshape(co, -1), "f16")}, ["y"], "threads 128 warps 2 2 bm 32 bn 32 bk 16", edit=("+ dx - oy * wp - ox,", "+ dx - oy * wp,"))[0]
    verdicts.append(("convk", "the column shift `- ox` dropped", not close(interior(o.reshape(rp, co), n, h, w, co), win_conv(x, wt, 2, 2), 1e-3)))
    o = rig.run("convk_dx.chx", consts, {"dz": (guarded(dz), "f16"), "w": (wt.reshape(co, -1), "f16")}, ["gx"], "threads 128 warps 2 2 bm 32 bn 16 bk 16", edit=("oy * wp + ox - dy * wp - dx", "oy * wp + ox - dy * wp"))[0]
    verdicts.append(("convk_dx", "the column shift `- dx` dropped", not close(interior(o.reshape(rp, cx), n, h, w, cx), gx_ref, 1e-3)))
    ns = rp // 32
    o = rig.run("convk_dw.chx", dict(consts, rs=32), {"dz": (guarded(dz), "f16"), "x": (guarded(x), "f16")}, ["part"], "threads 128 warps 2 2 bm 16 bn 16 bk 8", edit=("+ dx - oy * wp - ox, jn", "+ dx - oy * wp, jn"))[0]
    verdicts.append(("convk_dw", "the column shift `- ox` dropped", not close(o.reshape(ns, co, 16 * cin).sum(0).reshape(co, 4, 4, cin), gw_ref, 1e-3)))
    part = rng.normal(size=(ns, co, 4 * 4 * cin))
    o = rig.run("wsum_mask.chx", dict(ns=ns, r=co, k=256, bm=16, bn=128), {"part": (part.reshape(-1), "f32"), "mk": ((np.arange(256) % 3 == 0).astype(float)[None], "f32")}, ["dw"], edit=("load mk[0, j * bn ; 1, bn]", "load mk[0, 0 ; 1, bn]"))[0]
    verdicts.append(("wsum_mask", "the mask read from the first columns for every block", not close(o.reshape(co, 256), part.sum(0) * (np.arange(256) % 3 == 0), 1e-4)))
    h2, w2, c = 3, 8, 8
    img = f16(rng.normal(size=(n, 2 * h2, 2 * w2, c)))
    s0, s1 = sels(4)
    o = rig.run("s2d_fwd.chx", dict(n=n, h=h2, w=w2, c=c, g=g, wb=4), {"x": (guarded(img), "f16"), "sel0": (s0, "f32"), "sel1": (s1, "f32")}, ["s"], edit=("matmul &a1 f0", "matmul &a0 f0"))[0]
    rps2 = n * (h2 + 2) * (w2 + 2)
    verdicts.append(("s2d_fwd", "two channel groups taken from the same column parity", not close(interior(o.reshape(rps2 + 2 * g, 4 * c)[g:g + rps2], n, h2, w2, 4 * c), s2d_ref(img), 1e-3)))
    gs = rng.normal(size=(n, h2, w2, 4 * c))
    o = rig.run("s2d_bwd.chx", dict(n=n, h=h2, w=w2, c=c, wb=4), {"gs": (padded(gs), "f32"), "selt0": (s0.T.copy(), "f32"), "selt1": (s1.T.copy(), "f32")}, ["gi"], edit=("let p0 = load selt0\n  let p1 = load selt1", "let p0 = load selt1\n  let p1 = load selt0"))[0]
    verdicts.append(("s2d_bwd", "the two column parities swapped", not close(o.reshape(n, 2 * h2 + 2, 2 * w2 + 2, c)[:, 1:-1, 1:-1], s2d_back_ref(gs), 1e-3)))
    hp_, wq, cm = 6, 8, 8
    a_img = f16(rng.choice([0.5, 1.0, 1.5], size=(n, 2 * hp_, 2 * wq, cm)))
    pooled, idx = R.maxpool_f(a_img)
    rpm = n * (hp_ + 2) * (wq + 2)
    mk = padded(np.ones((n, hp_, wq, 1)))[:, 0]
    o = rig.run("maxpool_fwd.chx", dict(rp=rpm, c=cm, wp=wq + 2, g=g, bm=16), {"s": (guarded(s2d_ref(a_img)), "f16"), "mk": (mk, "f32")}, ["a", "idx"], edit=("let ix8 = e8 - w8", "let ix8 = e8 + w8"))[0]
    ix = split(o, [(rpm + 2 * g) * cm] * 2)[1].reshape(rpm + 2 * g, cm)[g:g + rpm].reshape(n, hp_ + 2, wq + 2, cm)[:, 1:-1, 1:-1]
    verdicts.append(("maxpool_fwd", "the winner's index updated with the wrong sign at the last candidate", not np.array_equal(ix, idx)))
    o = rig.run("maxpool_fwd.chx", dict(rp=rpm, c=cm, wp=wq + 2, g=g, bm=16), {"s": (guarded(s2d_ref(a_img)), "f16"), "mk": (mk, "f32")}, ["a", "idx"], edit=("let s5 = step d5", "let s5 = step d5 * 0"))[0]
    a = split(o, [(rpm + 2 * g) * cm] * 2)[0].reshape(rpm + 2 * g, cm)
    verdicts.append(("maxpool_fwd", "the fifth candidate never allowed to win", not close(interior(a[g:g + rpm], n, hp_, wq, cm), pooled, 1e-6)))
    gr = rng.normal(size=(n, hp_, wq, cm))
    ixg = np.zeros((rpm + 2 * g, cm))
    ixg[g:g + rpm] = -1.0
    ixg[g:g + rpm].reshape(n, hp_ + 2, wq + 2, cm)[:, 1:-1, 1:-1] = idx
    o = rig.run("maxpool_bwd.chx", dict(rp=rpm, c=cm, wp=wq + 2, g=g, bm=16), {"gr": (guarded(gr), "f32"), "idx": (ixg, "f32")}, ["gs"], edit=("let dd4 = ix4 - 4", "let dd4 = ix4 - 3"))[0]
    verdicts.append(("maxpool_bwd", "the centre candidate matched against the wrong index", not close(interior(o.reshape(rpm, 4 * cm), n, hp_, wq, 4 * cm), s2d_ref(R.maxpool_b(gr, idx, a_img.shape)), 1e-3)))
    for kernel, what, caught in verdicts:
        say(caught, f"mutant: {kernel} with {what} is caught")


def parse_grads(c, out, o=1):
    got, _ = S.parse_flat(c, out, o)
    return {k: v / c.n / S.LOSS_SCALE for k, v in got.items()}


def compare(got, ref):
    worst, cosmin, wname = 0.0, 1.0, ""
    for k, r in ref.items():
        gg = got[k].reshape(r.shape)
        e = float(np.linalg.norm(gg - r) / np.linalg.norm(r))
        if e > worst:
            worst, wname = e, k
        cosmin = min(cosmin, float((gg * r).sum() / np.linalg.norm(gg) / np.linalg.norm(r)))
    return worst, cosmin, wname


def network(binaries):
    d = tempfile.mkdtemp()
    c = R.Config(h=64, w=96, widths=(8, 16, 16, 32), blocks=(1, 2, 1, 1), classes=10, n=8)
    rng = np.random.RandomState(5)
    imgs = rng.normal(size=(25, c.n, c.h, c.w, c.cin))
    lab = rng.randint(0, 10, size=(25, c.n))
    P = R.init_params(c, 1)
    R.QUANT = True
    sess = S.write(os.path.join(d, "net"), c, imgs, lab, 25, grads=True)
    x16 = imgs[0].astype(np.float16).astype(np.float64)
    loss, g = R.forward(c, P, x16, lab[0], grads=True)
    R.QUANT = False
    if T is not None:
        lt, gt, _ = T.step(c, P, imgs[0].astype(np.float16).astype(np.float64), lab[0], torch.float32)
    nt = len(R.param_names(c)[0]) + len(R.param_names(c)[1])
    for b in binaries:
        cpu = "metal" not in b
        out = serve(b, sess, "1\n", cpu)[0]
        nm = os.path.basename(b)
        say(abs(out[0] - loss) <= 1e-3 * loss, f"{nm}: a ResNet-18 structure (stem 7x7 stride 2, max-pool, 4 stages, 3 stride-2 blocks with 1x1 shortcuts, {len(c.layers())} convolutions), loss {out[0]:.5f}, f16-rounding specification {loss:.5f}")
        got = parse_grads(c, out)
        worst, cosmin, wname = compare(got, g)
        say(worst < 0.2 and cosmin > 0.98, f"{nm}: the gradients of all {nt} tensors against the f16-rounding specification: relative error at most {worst:.1e} ({wname}), cosine at least {cosmin:.4f}")
        if T is not None:
            worst, cosmin, wname = compare(got, gt)
            say(abs(out[0] - lt) <= 2e-3 * lt and worst < 0.2 and cosmin > 0.98, f"{nm}: against torch float32 on the same weights and batch: loss {out[0]:.5f} / {lt:.5f}, gradients' relative error at most {worst:.1e} ({wname}), cosine at least {cosmin:.4f}")
    lab2 = (imgs.mean((2, 3, 4)) > 0).astype(int)  # a label the images carry: whether the image is brighter than average
    sess2 = S.write(os.path.join(d, "learn"), c, imgs, lab2, 25, lr=3e-3, wd=0.01)
    outs = [serve(b, sess2, "1\n4\n5\n15\n", "metal" not in b) for b in binaries]
    for b, o in zip(binaries, outs):
        say(o[-1][0] < 0.5 * o[0][0], f"{os.path.basename(b)}: 25 AdamW steps lower the loss {o[0][0]:.3f} -> {o[-1][0]:.3f}")
    if len(outs) > 1:
        say(all(abs(a[0] - b_[0]) <= 5e-2 * max(a[0], 0.1) for a, b_ in zip(outs[0], outs[1])), f"the Metal session follows the CPU session: {[round(float(v[0]), 3) for v in outs[0]]} and {[round(float(v[0]), 3) for v in outs[1]]}")


def main():
    binaries = [os.path.abspath(sys.argv[1])]
    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
        binaries.append(os.path.abspath(sys.argv[2]))
    specification_equals_torch()
    kernels(binaries)
    mutants(binaries[0])
    network(binaries)
    sys.exit(1 if bad else 0)


main()
