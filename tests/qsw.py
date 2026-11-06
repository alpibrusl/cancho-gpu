#!/usr/bin/env python3
"""The session writer in cancho (tools/qsw, docs/qsw.md) against the Python one (examples/qwen/qwen_train.py `write`), byte for byte, on a tiny fake Qwen2 model.

    python3 tests/qsw.py BUILD_QSW [--keep DIR] [--full]

BUILD_QSW is the built tool (scripts/build-qsw.sh). The Python writer is the specification here: it is what trained the 1.5B model, and a session file is data (kernel paths, sizes,
tables, floats printed as Python prints them, f16/f32 words rounded once) -- so "the same" is checkable exactly: the same file names, the same bytes. Each case writes a session
with both writers from the same inputs and compares every file. `--full` runs only the 1.5B-shaped case (about 1,150 files, a 466 MB padded output layer written twice)."""
import filecmp
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "qwen"))
try:
    import numpy as np
    import qwen_ref as R
    import qwen_session as S
    import qwen_train as T
except ImportError:
    print("skip: numpy is not installed")
    sys.exit(0)

bad = 0


def say(good, text):
    global bad
    print(f"{'ok  ' if good else 'FAIL'} {text}")
    bad += 0 if good else 1


def bits(x):
    """A float as the decimal integer of its binary64 bits: the tool's config has no decimal reading in it."""
    return struct.unpack("<q", struct.pack("<d", x))[0]


def config_text(c, r, steps, extra):
    kv = dict(layers=c.layers, d=c.d, h=c.h, hkv=c.hkv, dh=c.dh, ff=c.ff, v=c.v, t=c.t, nb=c.nb, tied=int(c.tied), r=r, steps=steps)
    fl = dict(eps=c.eps, theta=c.theta)
    fl.update(extra.pop("floats", {}))
    kv.update(extra)
    return "\n".join([f"{k} {v}" for k, v in kv.items()] + [f"{k} {bits(v)}" for k, v in fl.items()]) + "\n"


def case(name, c, steps=2, r=16, wt=None, save=True, seed=3, keep=None, sparse=False, train=True, grads=False, **kw):
    """One comparison. `kw` are qwen_train.write's keywords (alpha, lr, wd, loss_scale, betas, eps_adam, tune, reuse) given to both writers. `wt` is a per-position weight ([nb * t]) for the
    tool; the Python writer has no such input, so with `wt` every file but `wscale` is compared with Python's and `wscale` with the rule (f32 of weight x loss scale)."""
    d = tempfile.mkdtemp(prefix="qsw_")
    try:
        wdir = os.path.join(d, "w")
        if sparse:
            # the writers read only the embedding table of the weights (every other weight file is named in the session, not read): a table of zeros as a sparse file
            os.makedirs(wdir)
            with open(os.path.join(wdir, "emb.f16"), "wb") as f:
                f.truncate(2 * c.v * c.d)
        else:
            S.write_files(wdir, c, R.init_params(c, seed=seed))
        rng = np.random.default_rng(seed + 10)
        toks = rng.integers(0, c.v, size=(steps, c.nb, c.t)).astype(np.int32)
        AB = R.init_lora(c, r=r, b_std=0.02)
        # the session names its files by path, so both writers write to the same directory, one after the other
        out_dir = os.path.join(d, "out")
        py_dir = os.path.join(d, "py")
        os.makedirs(os.path.join(d, "in"))
        tok_path = os.path.join(d, "in", "tokens.i32")
        toks.astype("<i4").tofile(tok_path)
        ad_path = os.path.join(d, "in", "adapters.f64")
        with open(ad_path, "wb") as f:
            for nm in R.lora_names(c):
                f.write(np.asarray(AB[nm], dtype="<f8").reshape(-1).tobytes())
        cfg_path = os.path.join(d, "in", "config.txt")
        floats = {k_: float(kw[k_]) for k_ in ("alpha", "lr", "wd", "loss_scale", "eps_adam") if k_ in kw}
        if "betas" in kw:
            floats["b1"], floats["b2"] = kw["betas"]
        extra = dict(floats=floats)
        if "tune" in kw and "bc" in kw["tune"]:
            extra["bc"] = kw["tune"]["bc"]
        if "reuse" in kw:
            extra["reuse"] = int(kw["reuse"])
        open(cfg_path, "w").write(config_text(c, r, steps, extra))
        T.write(out_dir, c, wdir, toks, AB, r=r, steps=steps, save=os.path.join(d, "ad.bin") if save else None, train=train, grads=os.path.join(d, "grads.bin") if grads else None, **kw)
        os.rename(out_dir, py_dir)
        os.makedirs(out_dir)
        cmd = [sys.argv[1], "write", "--config", cfg_path, "--wdir", wdir, "--out", out_dir, "--root", ROOT, "--tokens", tok_path, "--adapters", ad_path]
        if save:
            cmd += ["--save", os.path.join(d, "ad.bin")]
        if not train:
            cmd += ["--no-train"]
        if grads:
            cmd += ["--grads", os.path.join(d, "grads.bin")]
        if wt is not None:
            wt_path = os.path.join(d, "in", "wt.f64")
            np.asarray(wt, dtype="<f8").tofile(wt_path)
            cmd += ["--weights", wt_path]
        import time
        t0 = time.perf_counter()
        done = subprocess.run(["/usr/bin/time", "-l"] + cmd if sparse else cmd, capture_output=True, text=True)
        if sparse:
            peak = [ln.split()[0] for ln in done.stderr.splitlines() if "maximum resident set size" in ln]
            print(f"     the tool: {time.perf_counter() - t0:.1f} s, peak {int(peak[0]) / 2**20:.0f} MiB" if peak else "")
        if done.returncode != 0:
            say(False, f"{name}: the tool failed ({done.returncode}): {done.stderr.strip()[:300]}")
            return
        py_files = sorted(os.listdir(py_dir))
        ch_files = sorted(os.listdir(out_dir))
        say(py_files == ch_files, f"{name}: the same {len(py_files)} files" + ("" if py_files == ch_files else f" (only python: {sorted(set(py_files) - set(ch_files))[:6]}, only cancho: {sorted(set(ch_files) - set(py_files))[:6]})"))
        tabled = wt is not None and np.size(wt) > c.nb * c.t
        skip = {"wscale"} if wt is not None else set()
        if tabled:
            # one block of weights for each step: `wscale` is a table read a block a step; the session differs from the Python writer's in that one line
            skip.add("session")
            want_s = open(os.path.join(py_dir, "session")).read().replace(f"resident cg.wt {out_dir}/wscale f32", f"table cg.wt {out_dir}/wscale {steps} f32")
            say(want_s == open(os.path.join(out_dir, "session")).read(), f"{name}: the session is the Python writer's with `table cg.wt .../wscale {steps} f32` for the resident line")
        differ = [f for f in py_files if f in ch_files and f not in skip and not filecmp.cmp(os.path.join(py_dir, f), os.path.join(out_dir, f), shallow=False)]
        say(not differ, f"{name}: every file is byte for byte the Python writer's" + ("" if not differ else f" (differ: {differ[:8]})"))
        if wt is not None:
            want = (np.asarray(wt, dtype=np.float64) * kw.get("loss_scale", 16.0)).astype("<f4").tobytes()
            say(open(os.path.join(out_dir, "wscale"), "rb").read() == want, f"{name}: wscale is f32(weight x loss scale)")
        if differ and "session" in differ:
            a = open(os.path.join(py_dir, "session")).read().split("\n")
            b = open(os.path.join(out_dir, "session")).read().split("\n")
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"  first session line that differs, {i + 1}:\n   python: {x}\n   cancho: {y}")
                    break
            else:
                print(f"  session line counts: python {len(a)}, cancho {len(b)}")
        if keep:
            shutil.copytree(d, keep, dirs_exist_ok=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def encoders(probe):
    """The encoders alone against numpy and Python: f32 and f16 words of binary64 values (ties, subnormals, overflow, random bit patterns), `repr` of a float, `%g` of a count."""
    import random
    random.seed(1)
    vals = [1.0, 0.1, 1e-6, 1e-8, 1e16, 1e15, 123456789.0, 0.5, 1 / math.sqrt(64), 1 / math.sqrt(128), 1e-4, 1e-5, 0.0001234, 16.0, 32.0, 2.0 ** -24, 2.0 ** -25, 65504.0, 65520.0, 65519.9,
            1e-9, 6.1e-5, 5.9e-8, 3.3e-8, 2.9e-8, 1e21, 5e-324, 1.7976931348623157e308, 1e22, 123456789012345680.0, 0.001, 1e100]
    for _ in range(20000):
        x = struct.unpack("<d", struct.pack("<Q", random.getrandbits(64)))[0]
        if x == x and abs(x) != float("inf"):
            vals.append(x)
        vals.append(random.gauss(0, 1) * 10 ** random.randint(-9, 6))
        vals.append(random.randint(-2000, 2000) / 2 ** random.randint(0, 14))     # exact ties in f16 and f32
        vals.append(random.randint(1, 2 ** 24) * 2.0 ** random.randint(-40, 30))  # values at the f32 boundary
    vals = [v for v in vals if not (v == 0 and math.copysign(1, v) < 0)]
    arr = np.array(vals)
    d = tempfile.mkdtemp(prefix="qsw_enc_")
    try:
        path = os.path.join(d, "bits")
        open(path, "w").write("\n".join(str(int(np.float64(v).view(np.int64))) for v in arr) + "\n")
        with np.errstate(over="ignore"):
            want = "".join(f"{int(a)} {int(b)} {repr(float(v))}\n" for a, b, v in zip(arr.astype("<f4").view("<u4"), arr.astype("<f2").view("<u2"), arr))
        got = subprocess.run([probe, path], capture_output=True, text=True).stdout
        say(got == want, f"encoders: f32 word, f16 word and repr of {len(vals)} binary64 values equal numpy's and Python's")
        counts = [0, 1, 999999, 1000000, 1000001, 1234567, 1234565, 1234575, 12345678, 99999949, 99999950, 999999499999, 999999500000, 4294967296, 10 ** 15]
        open(path, "w").write("\n".join(str(x) for x in counts) + "\n")
        got = subprocess.run([probe, path, "g"], capture_output=True, text=True).stdout
        say(got == "".join(f"{x:g}\n" for x in counts), "encoders: `%g` of whole numbers equals Python's")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def refusals(tool):
    """Every refusal names a rule (`qsw.<rule>` on standard error, exit status 2) and none is a crash: the inputs of a good case, broken one way at a time."""
    d = tempfile.mkdtemp(prefix="qsw_ref_")
    try:
        c = R.Config(layers=1, d=128, h=4, hkv=2, dh=32, ff=256, v=320, t=64, nb=2)
        wdir = os.path.join(d, "w")
        S.write_files(wdir, c, R.init_params(c, seed=3))
        AB = R.init_lora(c, r=16, b_std=0.02)
        os.makedirs(os.path.join(d, "in"))
        toks = np.random.default_rng(1).integers(0, c.v, size=(1, c.nb, c.t)).astype("<i4")
        toks.tofile(os.path.join(d, "in", "tok"))
        with open(os.path.join(d, "in", "ad"), "wb") as f:
            for nm in R.lora_names(c):
                f.write(np.asarray(AB[nm], dtype="<f8").reshape(-1).tobytes())
        open(os.path.join(d, "in", "cfg"), "w").write(config_text(c, 16, 1, {}))
        out = os.path.join(d, "out")
        os.makedirs(out)

        def run(rule, mutate=None, drop=None, name=None, extra=()):
            args = {"--config": os.path.join(d, "in", "cfg"), "--wdir": wdir, "--out": out, "--root": ROOT, "--tokens": os.path.join(d, "in", "tok"), "--adapters": os.path.join(d, "in", "ad")}
            if mutate:
                args.update(mutate)
            if drop:
                args.pop(drop)
            cmd = [tool, "write"] + [x for kv in args.items() for x in kv] + list(extra)
            done = subprocess.run(cmd, capture_output=True, text=True)
            say(done.returncode == 2 and f"qsw: {rule}:" in done.stderr, f"refusal {rule}: {name or ''} (exit {done.returncode}, {done.stderr.strip()[:90]!r})")

        def cfg_with(text, line):
            p = os.path.join(d, "in", "cfg_" + line.split()[0])
            open(p, "w").write(text + line + "\n")
            return p

        base = config_text(c, 16, 1, {})
        run("qsw.usage", drop="--tokens", name="a flag missing")
        run("qsw.input", {"--config": os.path.join(d, "none")}, name="no configuration file")
        run("qsw.input", {"--adapters": os.path.join(d, "in", "tok")}, name="adapters of the wrong size")
        run("qsw.config", {"--config": cfg_with(base, "r 8")}, name="rank 8")
        run("qsw.config", {"--config": cfg_with(base, "steps 0")}, name="no steps")
        run("qsw.shape", {"--config": cfg_with(base, "t 60")}, name="a sequence that is not a multiple of 64")
        run("qsw.shape", {"--config": cfg_with(base, "dh 16")}, name="dh that is not d / h")
        np.array([c.v] + [0] * (c.nb * c.t - 1), dtype="<i4").tofile(os.path.join(d, "in", "tok_bad"))
        run("qsw.input", {"--tokens": os.path.join(d, "in", "tok_bad")}, name="a token outside the vocabulary")
        np.zeros(5, dtype="<i4").tofile(os.path.join(d, "in", "tok_short"))
        run("qsw.input", {"--tokens": os.path.join(d, "in", "tok_short")}, name="tokens that are not whole batches")
        run("qsw.write", {"--out": os.path.join(d, "no_such_dir")}, name="an output directory that does not exist")
        run("qsw.kernel", {"--root": d}, name="a root without the kernels")
        open(os.path.join(wdir, "l00.wqkv.q"), "w").write("")
        run("qsw.quant", name="a quantised base")
        os.remove(os.path.join(wdir, "l00.wqkv.q"))
        run("qsw.input", {"--weights": os.path.join(d, "none")}, name="a weights file that is not there")
        run("qsw.usage", {"--save": os.path.join(d, "ad.bin")}, name="--save with --no-train", extra=["--no-train"])
        np.zeros(3, dtype="<f8").tofile(os.path.join(d, "in", "wt_bad"))
        run("qsw.input", {"--weights": os.path.join(d, "in", "wt_bad")}, name="weights that are neither one block nor one for each batch")
        done = subprocess.run([tool], capture_output=True, text=True)
        say(done.returncode == 2 and "qsw.usage" in done.stderr, "refusal qsw.usage: no command")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def full():
    """The shape of the 1.5B (Qwen2.5-1.5B: 28 layers, d 1536, 12 heads over 2, ff 8960, vocabulary 151,936) at a batch of 8 sequences of 512: about 30,000 program lines, 466 MB of padded output layer."""
    c = R.Config(layers=28, d=1536, h=12, hkv=2, dh=128, ff=8960, v=151936, t=512, nb=8)
    case("Qwen2.5-1.5B shapes, 1 step", c, steps=1, sparse=True)


def split_case(name, c, r=16, loss_scale=None, alpha=None):
    """`qsw split` against qwen_train.split_grads: random f32 fused gradients (with zeros, a negative zero, subnormals and large values), the per-projection binary64 words equal."""
    d = tempfile.mkdtemp(prefix="qsw_split_")
    try:
        G = T.group_dims(c, r)
        words = sum(gd["r2"] * gd["inp"] + gd["N"] * gd["r2"] for gd in G.values()) * c.layers
        rng = np.random.default_rng(11)
        raw = (rng.normal(size=words) * 10.0 ** rng.integers(-3, 3, size=words)).astype("<f4")
        raw[::97] = 0.0
        raw[1::193] = np.float32(-0.0)
        raw[2::211] = np.float32(1.2e-42)
        raw[3::223] = np.float32(-3.4e-39)
        raw[4::229] = np.float32(3.0e38)
        gp = os.path.join(d, "g.f32"); raw.tofile(gp)
        floats = {}
        if loss_scale is not None:
            floats["loss_scale"] = loss_scale
        if alpha is not None:
            floats["alpha"] = alpha
        open(os.path.join(d, "cfg"), "w").write(config_text(c, r, 1, dict(floats=floats)))
        out = os.path.join(d, "o.f64")
        done = subprocess.run([sys.argv[1], "split", "--config", os.path.join(d, "cfg"), "--grads", gp, "--out", out], capture_output=True, text=True)
        if done.returncode != 0:
            say(False, f"{name}: the tool failed ({done.returncode}): {done.stderr.strip()[:300]}")
            return
        ls = 16.0 if loss_scale is None else loss_scale
        al = 32.0 if alpha is None else alpha
        want = T.split_grads(c, r, T.read_fused(c, r, gp), float(c.nb * (c.t - 1)), ls, alpha=al)
        want_bytes = b"".join(np.asarray(want[nm], dtype="<f8").reshape(-1).tobytes() for nm in R.lora_names(c))
        got = open(out, "rb").read()
        say(got == want_bytes, f"{name}: {len(got) // 8} words, byte for byte split_grads' ({'equal' if got == want_bytes else sum(1 for i in range(0, min(len(got), len(want_bytes)), 8) if got[i:i + 8] != want_bytes[i:i + 8])} {'' if got == want_bytes else 'differ'})")
        bad = os.path.join(d, "short"); open(bad, "wb").write(raw.tobytes()[:-4])
        r2 = subprocess.run([sys.argv[1], "split", "--config", os.path.join(d, "cfg"), "--grads", bad, "--out", out], capture_output=True, text=True)
        say(r2.returncode == 2 and "qsw.input" in r2.stderr, f"{name}: refusal qsw.input for a gradients file of the wrong size")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def kv_export_case(tool):
    """`qsw write --no-train --kv-export DIR` (docs/decode.md): the prefill session of the training forward pass that saves every layer's keys and values for a batched decode. Written for a small
    model here (what it computes is held by tests/prefill_hf.py, which needs a Metal GPU): the session has the forward pass and no backward pass or head, one kv_import program for the keys and one for
    the values of each layer, a save of each, and `--kv-export` without `--no-train` is refused."""
    d = tempfile.mkdtemp(prefix="qsw_kv_")
    try:
        c = R.Config(layers=2, d=128, h=4, hkv=2, dh=32, ff=256, v=320, t=64, nb=2)
        wdir, out, kv = os.path.join(d, "w"), os.path.join(d, "out"), os.path.join(d, "kv")
        S.write_files(wdir, c, R.init_params(c, seed=3))
        os.makedirs(out); os.makedirs(kv); os.makedirs(os.path.join(d, "in"))
        AB = R.init_lora(c, r=16, b_std=0.02)
        with open(os.path.join(d, "in", "ad"), "wb") as f:
            for nm in R.lora_names(c):
                f.write(np.asarray(AB[nm], dtype="<f8").reshape(-1).tobytes())
        np.random.default_rng(1).integers(0, c.v, size=(1, c.nb, c.t)).astype("<i4").tofile(os.path.join(d, "in", "tok"))
        open(os.path.join(d, "in", "cfg"), "w").write(config_text(c, 16, 1, {}))
        cmd = [tool, "write", "--config", os.path.join(d, "in", "cfg"), "--wdir", wdir, "--out", out, "--root", ROOT, "--tokens", os.path.join(d, "in", "tok"), "--adapters", os.path.join(d, "in", "ad")]
        r1 = subprocess.run(cmd + ["--kv-export", kv], capture_output=True, text=True)
        say(r1.returncode == 2 and "qsw.usage" in r1.stderr, f"refusal qsw.usage: --kv-export without --no-train (exit {r1.returncode})")
        r2 = subprocess.run(cmd + ["--no-train", "--kv-export", kv], capture_output=True, text=True)
        say(r2.returncode == 0, f"the prefill session is written{'' if r2.returncode == 0 else ': ' + r2.stderr.strip()[:200]}")
        text = open(os.path.join(out, "session")).read() if os.path.exists(os.path.join(out, "session")) else ""
        lines = text.split("\n")
        say(sum(1 for x in lines if x.startswith("program kk_")) == 2 and sum(1 for x in lines if x.startswith("program kw_")) == 2, "one kv_import program for the keys and one for the values of each layer")
        say(sum(1 for x in lines if x.startswith("save ")) == 4 and f"save kk_1.y {kv}/kcache01 f32" in text, "a save of each, to kcacheNN and vcacheNN in the export directory")
        say(not any(x.startswith("program ce") or x.startswith("program lm ") or x.startswith("program gB") or "aA00" in x for x in lines), "no head, loss, backward pass or optimiser in a prefill session")
        say(os.path.exists(os.path.join(out, "probe")) and os.path.getsize(os.path.join(out, "probe")) == 4, "the one-element probe the session answers")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def decode_cases(tool):
    """`qsw decode` (docs/decode.md): its refusals by rule, and that a small valid configuration writes the session and the tables. (What the session computes is held by tests/decode_dev.py and
    tests/decode_hf.py, which need a Metal GPU.)"""
    d = tempfile.mkdtemp(prefix="qsw_dec_")
    try:
        wdir, out = os.path.join(d, "w"), os.path.join(d, "out")
        os.makedirs(wdir); os.makedirs(out); os.makedirs(os.path.join(d, "in"))
        with open(os.path.join(wdir, "emb.f16"), "wb") as f:
            f.truncate(2 * 64 * 1536)
        steps, nb = 4, 2
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype="<i4").tofile(os.path.join(d, "in", "tok"))
        np.array([3, 4], dtype="<i4").tofile(os.path.join(d, "in", "lens"))
        base = dict(layers=1, d=1536, h=12, hkv=2, dh=128, ff=8960, v=64, t=64, nb=nb, tied=1, steps=steps)
        def cfg(**kw):
            b = dict(base); b.update(kw)
            p = os.path.join(d, "in", "cfg_" + "_".join(f"{k}{v}" for k, v in kw.items()))
            open(p, "w").write("".join(f"{k} {v}\n" for k, v in b.items()))
            return p
        def run(rule, config, name, tok="tok", lens="lens", extra=()):
            cmd = [tool, "decode", "--config", config, "--wdir", wdir, "--out", out, "--root", ROOT, "--tokens", os.path.join(d, "in", tok), "--lens", os.path.join(d, "in", lens)] + list(extra)
            done = subprocess.run(cmd, capture_output=True, text=True)
            if rule is None:
                return done
            say(done.returncode == 2 and f"qsw: {rule}:" in done.stderr, f"refusal {rule}: decode, {name} (exit {done.returncode}, {done.stderr.strip()[:80]!r})")
        say(subprocess.run([tool, "decode", "--config", cfg()], capture_output=True, text=True).returncode == 2, "refusal qsw.usage: decode with flags missing")
        run("qsw.shape", cfg(h=8), "heads that are not Qwen2.5-1.5B's")
        run("qsw.shape", cfg(t=60), "a cache that is not a multiple of 64")
        run("qsw.shape", cfg(nb=33), "more than 32 sequences")
        np.array([1, 2, 3, 99], dtype="<i4").tofile(os.path.join(d, "in", "tok_bad"))
        np.array([[1, 2, 3, 4], [5, 6, 7, 99]], dtype="<i4").tofile(os.path.join(d, "in", "tok_bad"))
        run("qsw.input", cfg(), "a token outside the vocabulary", tok="tok_bad")
        np.array([3, 9], dtype="<i4").tofile(os.path.join(d, "in", "lens_bad"))
        run("qsw.input", cfg(), "a prompt longer than the steps", lens="lens_bad")
        np.array([3], dtype="<i4").tofile(os.path.join(d, "in", "lens_short"))
        run("qsw.input", cfg(), "a lengths file of the wrong size", lens="lens_short")
        np.array([0, 100], dtype="<i4").tofile(os.path.join(d, "in", "pos_bad"))
        run("qsw.input", cfg(), "positions that run past the cache", extra=["--pos0", os.path.join(d, "in", "pos_bad")])
        done = run(None, cfg(), "valid")
        files = set(os.listdir(out))
        want = {"session", "cos", "sin", "mask", "sel", "keep", "tftab", "mtab", "ids0", "utab", "zcache", "acc0", "accsel", "acckeep", "mk", "tri", "tri64", "iota", "base", "jrow", "embt", "wlm.f16"}
        say(done.returncode == 0 and want <= files, f"a valid small configuration writes the session and its {len(want) - 1} data files{'' if done.returncode == 0 else ': ' + done.stderr.strip()[:200]}")
        text = open(os.path.join(out, "session")).read() if "session" in files else ""
        say(all(x in text for x in ("program kwk_0", "carry kwk_0.y -> kwk_0.c", "carry mix.y -> oh.ids", "carry ac.y -> ac.acc", "answer ac.y")), "the session carries the caches, the ids and the id buffer, and answers the ids")
        # the adapters: the session without them has no adapter programs; with them each of the four projections of a layer is a base product, the two small products, a cast and an add,
        # reading the Ah and Bh files of its group in the adapters' directory, and the consumers still read the projection's own name
        say(not any(x in text for x in ("qkvb_0", "Ah00_", "Bh00_")), "without --adapters the session has no adapter programs")
        done = run(None, cfg(), "adapters", extra=["--adapters", "/ad"])
        lt = open(os.path.join(out, "session")).read()
        shape = []
        for pn, gn in (("qkv", "qkv"), ("op", "o"), ("gu", "gu"), ("dp", "dn")):
            shape.append(all(x in lt for x in (f"program {pn}b_0 ", f"program {pn}m_0 ", f"program {pn}c_0 ", f"program {pn}l_0 ", f"program {pn}_0 ", f"resident {pn}m_0.w /ad/Ah00_{gn} f16", f"resident {pn}l_0.w /ad/Bh00_{gn} f16", f"link {pn}b_0.y -> {pn}_0.z", f"link {pn}l_0.y -> {pn}_0.res")))
        say(done.returncode == 0 and all(shape), f"with --adapters each of the four projections is a base product, the adapter's two small products and an add{'' if done.returncode == 0 else ': ' + done.stderr.strip()[:200]}")
        say("link qkv_0.y -> post_0.y" in lt and "link ra_0.y -> rb_0.res" in lt, "the consumers of a projection still read it by its own name")
        run("qsw.usage", cfg(), "an adapters flag without a value", extra=["--adapters"])
        sizes = {"mask": 4 * steps * nb * 64, "sel": 4 * steps * nb * 64, "tftab": 4 * steps * nb, "cos": 4 * steps * nb * 64, "embt": 2 * 1536 * 1536}
        say(all(os.path.getsize(os.path.join(out, k)) == v for k, v in sizes.items()), "the tables have the sizes of steps x sequences x positions (and the transposed embedding d x padded vocabulary)")
        tf = np.fromfile(os.path.join(out, "tftab"), dtype="<f4").reshape(steps, nb)
        mt = np.fromfile(os.path.join(out, "mtab"), dtype="<f4").reshape(steps, nb)
        say(np.array_equal(tf, np.array([[2, 6], [3, 7], [0, 8], [0, 0]]) + (np.array([[2, 6], [3, 7], [0, 8], [0, 0]]) > 0)) and np.array_equal(mt, [[1, 1], [1, 1], [0, 1], [0, 0]]), "the prompt tables: block j holds the token (plus one) and the flag that step j + 1 reads while inside its prompt")
        sel = np.fromfile(os.path.join(out, "sel"), dtype="<f4").reshape(steps, nb, 64)
        say(all(sel[j, r].argmax() == j and sel[j, r].sum() == 1 for j in range(steps) for r in range(nb)), "the cache selector of step j is the one-hot of position j")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    keep = sys.argv[sys.argv.index("--keep") + 1] if "--keep" in sys.argv else None
    encoders(os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "qsw_probe"))
    if "--full" in sys.argv:
        full()
        print("bad" if bad else "all ok")
        return 1 if bad else 0
    refusals(sys.argv[1])
    decode_cases(sys.argv[1])
    kv_export_case(sys.argv[1])
    c = R.Config(layers=2, d=128, h=4, hkv=2, dh=32, ff=256, v=320, t=64, nb=2)
    case("tiny, 2 steps", c, steps=2, keep=keep)
    case("tiny, 1 step, no sharing of layers' buffers", c, steps=1, reuse=False)
    case("tiny, other optimiser settings", c, steps=3, alpha=16.0, lr=3e-4, wd=0.01, loss_scale=64.0, betas=(0.8, 0.99), eps_adam=1e-7)
    case("3 layers, 8 heads over 2 (groups of 4), exact vocabulary block", R.Config(layers=3, d=256, h=8, hkv=2, dh=32, ff=192, v=384, t=64, nb=2), steps=2)
    case("head of 128, untied output layer", R.Config(layers=2, d=256, h=2, hkv=1, dh=128, ff=256, v=300, t=64, nb=2, tied=False), steps=2)
    case("a 1.5B-shaped layer, small vocabulary", R.Config(layers=2, d=1536, h=12, hkv=2, dh=128, ff=8960, v=512, t=64, nb=2), steps=1)
    case("a sequence of 512 (the rotary tables up to angle 511 rad), head of 128", R.Config(layers=1, d=256, h=2, hkv=1, dh=128, ff=256, v=320, t=512, nb=1), steps=1)
    n = c.nb * c.t
    wc = np.ones(n)
    wc[c.t - 1::c.t] = 0.0
    case("explicit weights equal to the default ones reproduce the files", c, steps=2, wt=wc)
    case("per-position weights (advantage x mask)", c, steps=2, wt=np.where(np.arange(n) % c.t < 20, 0.0, np.random.default_rng(5).normal(size=n)))
    case("gradients saved while training", c, steps=2, grads=True)
    case("no training, gradients saved (the policy-gradient check)", c, steps=1, save=False, train=False, grads=True)
    case("no training, nothing saved", c, steps=1, save=False, train=False)
    case("no training, gradients saved, per-position weights", c, steps=1, save=False, train=False, grads=True, wt=np.where(np.arange(n) % c.t < 20, 0.0, np.random.default_rng(6).normal(size=n)))
    case("per-step weights (a table of two blocks)", c, steps=2, save=False, train=False, grads=True, wt=np.random.default_rng(8).normal(size=2 * n))
    split_case("split: tiny", c)
    split_case("split: 3 layers, 8 heads over 2, loss scale 64 and alpha 16", R.Config(layers=3, d=256, h=8, hkv=2, dh=32, ff=192, v=384, t=64, nb=2), loss_scale=64.0, alpha=16.0)
    split_case("split: head of 128", R.Config(layers=2, d=256, h=2, hkv=1, dh=128, ff=256, v=300, t=64, nb=2, tied=False))
    case("no training, gradients saved, 3 layers", R.Config(layers=3, d=256, h=8, hkv=2, dh=32, ff=192, v=384, t=64, nb=2), steps=1, save=False, train=False, grads=True)
    print("bad" if bad else "all ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
