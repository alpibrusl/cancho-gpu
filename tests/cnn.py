#!/usr/bin/env python3
"""docs/resident.md section 18: the convolutional network's kernels held to float64 numpy, then the whole network to its specification.

    python3 tests/cnn.py build/cancho-gpu-session [build/cancho-gpu-metal-session]

1. Each kernel of examples/cnn on its own, on the CPU interpreter and (given) on Metal, against a float64 numpy computation from the same f16-representable inputs: the three
   convolutions (forward, input gradient, weight gradient and its sum), BatchNorm's statistics and their finish, the apply and the backward (with and without the residual), the
   pooling and the global average and their gradients. No ReLU decision can differ between the kernel and the reference here, so these are tight (1e-4 relative).
2. The whole network (2 stages and a small image, examples/cnn/cnn_ref.py): the loss against the specification that rounds to binary16 where the device does, and the gradients of all
   its tensors against that specification's hand-written backward, by cosine and relative error; a ReLU whose input is within the f16 rounding of zero can fall on either side, which
   moves a gradient by a few percent whatever the batch (the rate of such elements is the rounding over the spread), so the bound is loose (0.2) and the cosine tight (0.98).
3. 25 AdamW steps: the CPU and Metal sessions give each other's losses and the loss falls.
Needs numpy; without it the test says so and passes."""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "cnn"))
try:
    import numpy as np
    import cnn_data as D
    import cnn_ref as R
    import cnn_session as S
except ImportError:
    print("skip: numpy is not installed")
    sys.exit(0)

bad = 0
CNN = os.path.join(ROOT, "examples", "cnn")
G = S.G


def say(good, text):
    global bad
    print(f"{'ok  ' if good else 'FAIL'} {text}")
    bad += 0 if good else 1


def serve(binary, session, lines, cpu):
    r = subprocess.run([binary, session] + (["--target", "apple-m-series"] if cpu else []), input=lines, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise SystemExit(f"{binary}: {r.stderr[:400]}")
    return [np.array([float(v) for v in line.split()]) for line in r.stdout.splitlines()]


def f16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float64)


def guarded(x):
    """[n, h, w, c] -> the padded, guarded f16 matrix [g + n (h+2)(w+2) + g, c] the kernels read."""
    n, h, w, c = x.shape
    xp = np.zeros((n, h + 2, w + 2, c))
    xp[:, 1:-1, 1:-1] = x
    out = np.zeros((G + xp.shape[0] * xp.shape[1] * xp.shape[2] + G, c))
    out[G:-G] = xp.reshape(-1, c)
    return out


def padded(x):
    n, h, w, c = x.shape
    xp = np.zeros((n, h + 2, w + 2, c))
    xp[:, 1:-1, 1:-1] = x
    return xp.reshape(-1, c)


def interior(m, n, h, c):
    return m.reshape(n, h + 2, h + 2, c)[:, 1:-1, 1:-1]


class Rig:
    """One-program sessions: write the inputs as files, run, read the answers."""

    def __init__(self, binaries):
        self.dir = tempfile.mkdtemp()
        self.binaries = binaries
        self.k = 0

    def run(self, kernel, consts, inputs, outputs, sched=None, extra=()):
        """inputs: {name: (array, "f32" | "f16")}; outputs: names; answers {name: flat array} for each binary."""
        self.k += 1
        d = os.path.join(self.dir, str(self.k))
        os.makedirs(d)
        src = open(os.path.join(CNN, kernel)).read()
        if sched:
            src = "\n".join((l.split("{")[0] + "{ " + sched + " }") if l.startswith("schedule ") and "apple-m-series" in l else l for l in src.split("\n"))
        kpath = os.path.join(d, kernel)
        open(kpath, "w").write(src)
        lines = [f"program k {kpath} " + " ".join(f"{a}={b}" for a, b in consts.items())]
        for name, (arr, ty) in inputs.items():
            S.put(os.path.join(d, name), arr, "<f2" if ty == "f16" else "<f4")
            lines.append(f"resident k.{name} {os.path.join(d, name)} {ty}")
        lines += list(extra)
        lines += [f"answer k.{o}" for o in outputs]
        open(os.path.join(d, "s"), "w").write("\n".join(lines) + "\n")
        res = []
        for b in self.binaries:
            out = serve(b, os.path.join(d, "s"), "\n", cpu="metal" not in b)[0]
            res.append(out)
        return res


def split(out, sizes):
    o, r = 0, []
    for s in sizes:
        r.append(out[o:o + s])
        o += s
    return r


def close(got, want, rel):
    return float(np.abs(got - want).max()) <= rel * max(float(np.abs(want).max()), 1e-12)


def kernels(binaries):
    rig = Rig(binaries)
    rng = np.random.RandomState(7)
    n, h, ci, co = 4, 6, 16, 32
    wp, rp = h + 2, n * (h + 2) ** 2
    x = f16(rng.normal(size=(n, h, h, ci)))
    w = f16(rng.normal(size=(co, 3, 3, ci)))
    # forward
    xp = np.zeros((n, h + 2, h + 2, ci))
    xp[:, 1:-1, 1:-1] = x
    ref = R.conv_f(x, w)
    outs = rig.run("conv3.chx", dict(rp=rp, cin=ci, cout=co, wp=wp, g=G), {"x": (guarded(x), "f16"), "w": (w.reshape(co, -1), "f16")}, ["y"], "threads 128 warps 2 2 bm 32 bn 32 bk 16")
    for b, o in zip(binaries, outs):
        say(close(interior(o.reshape(rp, co), n, h, co), ref, 1e-4), f"{os.path.basename(b)}: conv3 against numpy")
    # input gradient: dz is zero on the border
    dz = f16(rng.normal(size=(n, h, h, co)))
    _, gx_ref = R.conv_b(dz, x, w)
    outs = rig.run("conv3_dx.chx", dict(rp=rp, cin=ci, cout=co, wp=wp, g=G), {"dz": (guarded(dz), "f16"), "w": (w.reshape(co, -1), "f16")}, ["gx"], "threads 128 warps 2 2 bm 32 bn 16 bk 16")
    for b, o in zip(binaries, outs):
        say(close(interior(o.reshape(rp, ci), n, h, ci), gx_ref, 1e-4), f"{os.path.basename(b)}: conv3_dx (the gradient with respect to the input) against numpy")
    # weight gradient, in pieces and summed
    dw_ref, _ = R.conv_b(dz, x, w)
    rs = 64
    ns = rp // rs
    inputs = {"dz": (guarded(dz), "f16"), "x": (guarded(x), "f16")}
    outs = rig.run("conv3_dw.chx", dict(rp=rp, cin=ci, cout=co, wp=wp, g=G, rs=rs), inputs, ["part"], "threads 128 warps 2 2 bm 16 bn 16 bk 8")
    for b, o in zip(binaries, outs):
        part = o.reshape(ns, co, 9 * ci)
        say(close(part.sum(0).reshape(co, 3, 3, ci), dw_ref, 1e-4), f"{os.path.basename(b)}: conv3_dw against numpy ({ns} pieces, tiles looped over)")
        sm = rig.run("wsum.chx", dict(ns=ns, r=co, k=9 * ci, bm=16, bn=48), {"part": (part.reshape(-1), "f32")}, ["dw"])
        say(close(sm[binaries.index(b)].reshape(co, 3, 3, ci), dw_ref, 1e-4), f"{os.path.basename(b)}: wsum adds the pieces")
    # BatchNorm forward
    c = 32
    z = rng.normal(2.0, 3.0, size=(rp, c))
    mk = padded(np.ones((n, h, h, 1)))[:, 0]
    rb, nb = 64, rp // 64
    cb = 2
    outs = rig.run("bn_part.chx", dict(rp=rp, c=c, rb=rb), {"z": (z, "f32"), "v": (mk, "f32")}, ["s1", "s2"])
    zv = z[mk > 0]
    m = len(zv)
    res = []  # the partials come back concatenated: s1 then s2, each [nb, c]
    for bi, b in enumerate(binaries):
        s1, s2 = split(outs[bi], [nb * c, nb * c])
        say(close(s1.reshape(nb, c).sum(0), zv.sum(0), 1e-5) and close(s2.reshape(nb, c).sum(0), (zv ** 2).sum(0), 1e-5), f"{os.path.basename(b)}: bn_part sums the valid rows' z and z^2")
        res.append((s1, s2))
    gamma, beta = rng.normal(1, 0.3, size=c), rng.normal(0, 0.3, size=c)
    rm, rv = rng.normal(size=c), rng.uniform(0.5, 2, size=c)
    mean_ref, var_ref = zv.mean(0), zv.var(0)
    istd_ref = 1 / np.sqrt(var_ref + 1e-5)
    fins = []
    for bi, b in enumerate(binaries):
        s1, s2 = res[bi]
        o = rig.run("bn_fin.chx", dict(nb=nb, c=c, cb=cb, bc=32, m=m, eps=1e-5, mom=0.1),
                    {"s1": (s1, "f32"), "s2": (s2, "f32"), "ones": (np.ones(cb), "f32"), "gamma": (gamma, "f32"), "beta": (beta, "f32"), "rm": (rm, "f32"), "rv": (rv, "f32")},
                    ["sc", "sh", "mean", "istd", "rmo", "rvo"])[bi]
        sc, sh, mean, istd, rmo, rvo = split(o, [c] * 6)
        ok = (close(mean, mean_ref, 1e-4) and close(istd, istd_ref, 1e-4) and close(sc, gamma * istd_ref, 1e-4) and close(sh, beta - mean_ref * gamma * istd_ref, 1e-3)
              and close(rmo, 0.9 * rm + 0.1 * mean_ref, 1e-4) and close(rvo, 0.9 * rv + 0.1 * var_ref * m / (m - 1), 1e-4))
        say(ok, f"{os.path.basename(b)}: bn_fin gives mean, 1/std, scale, shift and the running statistics of numpy (the unbiased variance in the running one)")
        fins.append((sc, sh, mean, istd))
    # apply, with and without the residual
    resid = f16(rng.normal(size=(n, h, h, c)))
    mcol = mk
    for with_res in (False, True):
        for bi, b in enumerate(binaries):
            sc, sh, _, _ = fins[bi]
            inputs = {"z": (z, "f32"), "sc": (sc, "f32"), "sh": (sh, "f32"), "mk": (mcol, "f32")}
            if with_res:
                inputs["res"] = (guarded(resid), "f16")
            o = rig.run("bn_apply_res.chx" if with_res else "bn_apply.chx", dict(rp=rp, c=c, bm=16, g=G), inputs, ["a"])[bi].reshape(rp + 2 * G, c)
            want = z * sc + sh + (padded(resid) if with_res else 0)
            want = np.maximum(want, 0) * mk[:, None]
            say(close(o[G:G + rp], f16(want), 1e-3) and not np.abs(o[:G]).any() and not np.abs(o[G + rp:]).any(),
                f"{os.path.basename(b)}: {'bn_apply_res' if with_res else 'bn_apply'}: relu(z scale + shift{' + skip' if with_res else ''}), zero on the border and in the guards")
    # backward
    gr = rng.normal(size=(rp, c))
    for with_res in (False, True):
        for bi, b in enumerate(binaries):
            sc, sh, mean, istd = fins[bi]
            pre = z * sc + sh + (padded(resid) if with_res else 0)
            d = gr * (pre > 0) * mk[:, None]
            xhat = (z - mean) * istd
            dbeta_ref, dgamma_ref = d.sum(0), (d * xhat).sum(0)
            rb2 = 64
            inputs = {"gr": (gr, "f32"), "z": (z, "f32"), "sc": (sc, "f32"), "sh": (sh, "f32"), "mean": (mean, "f32"), "istd": (istd, "f32"), "v": (mk, "f32")}
            consts = dict(rp=rp, c=c, rb=rb2)
            if with_res:
                inputs["res"] = (guarded(resid), "f16")
                consts["g"] = G
            o = rig.run("bn_bwd_part_res.chx" if with_res else "bn_bwd_part.chx", consts, inputs, ["p1", "p2"])[bi]
            p1, p2 = split(o, [rp // rb2 * c] * 2)
            say(close(p1.reshape(-1, c).sum(0), dbeta_ref, 1e-4) and close(p2.reshape(-1, c).sum(0), dgamma_ref, 1e-4), f"{os.path.basename(b)}: {'bn_bwd_part_res' if with_res else 'bn_bwd_part'} sums d and d xhat over the valid rows")
            cb2 = 2
            o2 = rig.run("bn_bwd_fin.chx", dict(nb=rp // rb2, c=c, cb=cb2, bc=32), {"p1": (p1, "f32"), "p2": (p2, "f32"), "ones": (np.ones(cb2), "f32")}, ["dbeta", "dgamma"])[bi]
            db, dg = split(o2, [c, c])
            say(close(db, dbeta_ref, 1e-4) and close(dg, dgamma_ref, 1e-4), f"{os.path.basename(b)}: bn_bwd_fin adds the blocks")
            inputs = {"gr": (gr, "f32"), "z": (z, "f32"), "sc": (sc, "f32"), "sh": (sh, "f32"), "mean": (mean, "f32"), "istd": (istd, "f32"), "gamma": (gamma, "f32"),
                      "dbeta": (db, "f32"), "dgamma": (dg, "f32"), "mk": (mk, "f32")}
            consts = dict(rp=rp, c=c, bm=16, g=G, m=m)
            if with_res:
                inputs["res"] = (guarded(resid), "f16")
            o3 = rig.run("bn_bwd_apply_res.chx" if with_res else "bn_bwd_apply.chx", consts, inputs, ["dz", "skip"] if with_res else ["dz"])[bi]
            dzk = o3[: (rp + 2 * G) * c].reshape(rp + 2 * G, c)
            want = gamma * istd * (d - dbeta_ref / m - xhat * dgamma_ref / m) * mk[:, None]
            ok = close(dzk[G:G + rp], f16(want), 2e-3) and not np.abs(dzk[:G]).any() and not np.abs(dzk[G + rp:]).any()
            if with_res:
                ok = ok and close(o3[(rp + 2 * G) * c:].reshape(rp, c), gr * (pre > 0), 1e-5)
            say(ok, f"{os.path.basename(b)}: {'bn_bwd_apply_res' if with_res else 'bn_bwd_apply'}: dz = gamma istd (d - dbeta/m - xhat dgamma/m), masked{' and the skip gradient' if with_res else ''}")
    # the input: two half batches of unpadded images into the padded, guarded matrix, channels padded to 8
    h3, n2 = 6, 2
    ia, ib = f16(rng.normal(size=(n2, h3, h3, 3))), f16(rng.normal(size=(n2, h3, h3, 3)))
    cmat = np.zeros((3, 8))
    cmat[:, :3] = np.eye(3)
    em = np.zeros((h3 + 2, h3))
    em[np.arange(1, h3 + 1), np.arange(h3)] = 1.0
    both = np.concatenate([ia, ib])
    want = np.zeros((2 * n2, h3 + 2, h3 + 2, 8))
    want[:, 1:-1, 1:-1, :3] = both
    for bi, b in enumerate(binaries):
        o = rig.run("pack.chx", dict(n2=n2, h=h3, cin=3, cp=8, g=G), {"xa": (ia.reshape(-1, 3), "f16"), "xb": (ib.reshape(-1, 3), "f16"), "cmat": (cmat, "f32"), "em": (em, "f32")}, ["o"])[bi].reshape(-1, 8)
        say(close(o[G:-G], want.reshape(-1, 8), 1e-3) and not np.abs(o[:G]).any() and not np.abs(o[-G:]).any(), f"{os.path.basename(b)}: pack pads the borders and the channels of two half batches")
    # pooling and the global average
    hh, cc = 8, 32
    a_img = f16(rng.normal(size=(n, hh, hh, cc)))
    h2 = hh // 2
    pm = np.zeros((h2 + 2, hh + 2))
    for xx in range(h2):
        pm[xx + 1, 2 * xx + 1] = pm[xx + 1, 2 * xx + 2] = 0.25
    pool_ref = f16(R.pool_f(a_img))
    gp = rng.normal(size=(n, h2, h2, cc))
    pb_ref = R.pool_b(gp)
    for bi, b in enumerate(binaries):
        o = rig.run("pool_fwd.chx", dict(n=n, h=hh, c=cc, g=G), {"a": (guarded(a_img), "f16"), "pm": (pm, "f32")}, ["o"])[bi].reshape(-1, cc)
        rpo = n * (h2 + 2) ** 2
        say(close(interior(o[G:G + rpo], n, h2, cc), pool_ref, 1e-3) and not np.abs(o[:G]).any(), f"{os.path.basename(b)}: pool_fwd is the 2x2 average, the border stays zero")
        o = rig.run("pool_bwd.chx", dict(n=n, h=hh, c=cc), {"gr": (padded(gp), "f32"), "pt": (pm.T.copy(), "f32")}, ["gi"])[bi].reshape(-1, cc)
        say(close(interior(o, n, hh, cc), pb_ref, 1e-5), f"{os.path.basename(b)}: pool_bwd spreads a quarter of each gradient over its 2x2")
        rc = 20
        vm = np.zeros((hh + 2, hh + 2))
        vm[1:-1, 1:-1] = 1.0 / (hh * hh)
        o = rig.run("gap_fwd.chx", dict(n=n, h=hh, c=cc, g=G, rc=rc), {"a": (guarded(a_img), "f16"), "vm": (vm.reshape(-1), "f32")}, ["f"])[bi]
        say(close(o.reshape(n, cc), f16(a_img.mean((1, 2))), 1e-3), f"{os.path.basename(b)}: gap_fwd is the mean over the valid pixels")
        gf = rng.normal(size=(n, cc))
        o = rig.run("gap_bwd.chx", dict(n=n, h=hh, c=cc, rc=rc), {"gf": (gf, "f32"), "mc": (vm.reshape(-1), "f32")}, ["gi"])[bi].reshape(-1, cc)
        want = np.zeros((n, hh + 2, hh + 2, cc))
        want[:, 1:-1, 1:-1] = gf[:, None, None, :] / (hh * hh)
        say(close(o, want.reshape(-1, cc), 1e-5), f"{os.path.basename(b)}: gap_bwd spreads each gradient over the valid pixels")


def mutants(binary):
    """Each kernel family with one deliberate change that its test above must catch (CPU interpreter)."""
    global CNN
    rig = Rig([binary])
    rng = np.random.RandomState(9)
    n, h, ci, co = 4, 6, 16, 32
    wp, rp = h + 2, n * (h + 2) ** 2
    x, w, dz = f16(rng.normal(size=(n, h, h, ci))), f16(rng.normal(size=(co, 3, 3, ci))), f16(rng.normal(size=(n, h, h, co)))
    dw_ref, gx_ref = R.conv_b(dz, x, w)
    cases = [("conv3.chx", "- wp - 1, p * bk", "- wp, p * bk", dict(rp=rp, cin=ci, cout=co, wp=wp, g=G), {"x": (guarded(x), "f16"), "w": (w.reshape(co, -1), "f16")}, "y", R.conv_f(x, w), ci, "threads 128 warps 2 2 bm 32 bn 32 bk 16"),
             ("conv3_dx.chx", "+ wp + 1 - dy * wp - dx", "+ wp + 1 - dy * wp", dict(rp=rp, cin=ci, cout=co, wp=wp, g=G), {"dz": (guarded(dz), "f16"), "w": (w.reshape(co, -1), "f16")}, "gx", gx_ref, ci, "threads 128 warps 2 2 bm 32 bn 16 bk 16")]
    for kernel, old, new, consts, inputs, out, ref, ch, sched in cases:
        src = open(os.path.join(CNN, kernel)).read()
        assert old in src
        mutated = os.path.join(rig.dir, "mut_" + kernel)
        open(mutated, "w").write(src.replace(old, new))
        os.makedirs(os.path.join(rig.dir, "m"), exist_ok=True)
        # run the mutated text through the rig by temporarily pointing CNN at it
        keep, CNN = CNN, rig.dir
        os.replace(mutated, os.path.join(rig.dir, kernel))
        try:
            o = rig.run(kernel, consts, inputs, [out], sched)[0]
        finally:
            CNN = keep
        got = interior(o.reshape(rp, ch if out == "gx" else co), n, h, ch if out == "gx" else co)
        say(not close(got, ref, 1e-3), f"mutant: {kernel} with `{old}` changed to `{new}` is caught")


def network(binaries):
    d = tempfile.mkdtemp()
    c = R.Config(h=8, cin=3, widths=(16, 16, 32), blocks=(1, 1, 1), classes=10, n=16)
    rng = np.random.RandomState(5)
    imgs = rng.normal(size=(25, c.n, c.h, c.h, c.cin))
    lab = rng.randint(0, 10, size=(25, c.n))
    P = R.init_params(c, 1)
    R.QUANT = True
    sess = S.write(os.path.join(d, "net"), c, imgs, lab, 25, grads=True)
    loss, g = R.forward(c, P, imgs[0], lab[0], grads=True)
    mats, vecs = S.names(c)
    Pp = S.pad_params(c, P)
    for b in binaries:
        cpu = "metal" not in b
        out = serve(b, sess, "1\n", cpu)[0]
        say(abs(out[0] - loss) <= 1e-3 * loss, f"{os.path.basename(b)}: a 7-convolution network, loss {out[0]:.5f}, f16-rounding specification {loss:.5f}")
        worst, cosmin, o = 0.0, 1.0, 1
        for name in mats + vecs:
            size = Pp[name].size
            got = out[o:o + size] / c.n / S.LOSS_SCALE
            o += size
            ref = g[name]
            if name == "w_stem":
                got = got.reshape(c.widths[0], 3, 3, 8)[..., :3]
            elif name == "wfc":
                got = got.reshape(S.VP, -1)[:c.classes]
            got = got.reshape(ref.shape)
            worst = max(worst, float(np.linalg.norm(got - ref) / np.linalg.norm(ref)))
            cosmin = min(cosmin, float((got * ref).sum() / np.linalg.norm(got) / np.linalg.norm(ref)))
        say(worst < 0.2 and cosmin > 0.98, f"{os.path.basename(b)}: the gradients of all {len(mats + vecs)} tensors: relative error at most {worst:.1e}, cosine at least {cosmin:.4f}")
    R.QUANT = False
    # training: the loss falls, and the sessions agree
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
    kernels(binaries)
    mutants(binaries[0])
    network(binaries)
    sys.exit(1 if bad else 0)


main()
