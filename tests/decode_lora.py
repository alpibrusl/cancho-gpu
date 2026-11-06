#!/usr/bin/env python3
"""`qsw decode --adapters DIR` (docs/decode.md, LoRA): the batched decode session with the LoRA adapters against the float64 forward pass with adapters (qwen_ref.lora_forward).
The adapter files (Ah, Bh: f16, the scale folded into Bh) are the ones `qsw write` writes for the training session, so the decode reads the very files the prefill and the training do.
Like decode_dev.py: a random Qwen2.5-1.5B-shaped model of 2 layers, `nb` sequences teacher-forced, the logits of every step against the reference. The adapters have a B that is not
small (b_std 0.05), so a decode that left them out is far from the reference: the check also measures the reference without adapters against the same logits, which must be well above
the tolerance. Needs a Metal GPU.

    python3 tests/decode_lora.py build/qsw build/cancho-gpu-metal-session [--layers 2] [--steps 12] [--nb 2] [--v 1024]"""
import os, struct, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "examples", "qwen"))
import qwen_ref as R, qwen_session as S  # noqa: E402

arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
QSW, BIN = sys.argv[1], sys.argv[2]
LAYERS, T, NB, V = int(arg("--layers", 2)), int(arg("--steps", 12)), int(arg("--nb", 2)), int(arg("--v", 1024))
bad = 0


def say(ok, text):
    global bad
    print(("ok   " if ok else "FAIL ") + text)
    bad += 0 if ok else 1


def bits(x):
    return struct.unpack("<q", struct.pack("<d", x))[0]


c = R.Config(layers=LAYERS, d=1536, h=12, hkv=2, dh=128, ff=8960, v=V, t=T, nb=NB)
P = R.init_params(c, seed=5)
AB = R.init_lora(c, r=16, seed=11, b_std=0.05)
d = tempfile.mkdtemp(prefix="decode_lora_")
wdir, out, adir = os.path.join(d, "w"), os.path.join(d, "out"), os.path.join(d, "ad")
S.write_files(wdir, c, P)
os.makedirs(out)
os.makedirs(adir)
rng = np.random.default_rng(2)
tokens = rng.integers(0, V, size=(NB, T)).astype("<i4")
tokens.tofile(os.path.join(d, "tokens.i32"))
np.full(NB, T, dtype="<i4").tofile(os.path.join(d, "lens.i32"))
common = dict(layers=LAYERS, d=1536, h=12, hkv=2, dh=128, ff=8960, v=V, t=64, tied=1, r=16, eps=bits(c.eps), theta=bits(c.theta))
open(os.path.join(d, "cfg.txt"), "w").write("".join(f"{k} {v}\n" for k, v in dict(common, nb=NB, steps=T).items()))

# the adapter files, as the training session's writer makes them (one batch of one sequence is enough: only Ah and Bh are used here)
open(os.path.join(d, "wcfg.txt"), "w").write("".join(f"{k} {v}\n" for k, v in dict(common, nb=1, steps=1).items()))
rng.integers(0, V, size=(1, 1, 64)).astype("<i4").tofile(os.path.join(d, "wtok.i32"))
with open(os.path.join(d, "ad.f64"), "wb") as f:
    for nm in R.lora_names(c):
        f.write(np.asarray(AB[nm], dtype="<f8").reshape(-1).tobytes())
done = subprocess.run([QSW, "write", "--config", os.path.join(d, "wcfg.txt"), "--wdir", wdir, "--out", adir, "--root", ROOT, "--tokens", os.path.join(d, "wtok.i32"), "--adapters", os.path.join(d, "ad.f64"), "--no-train"], capture_output=True, text=True)
say(done.returncode == 0, f"qsw write wrote the adapter files ({len(os.listdir(adir))} files){'' if done.returncode == 0 else ': ' + done.stderr.strip()[:300]}")
if done.returncode != 0:
    sys.exit(1)

done = subprocess.run([QSW, "decode", "--config", os.path.join(d, "cfg.txt"), "--wdir", wdir, "--out", out, "--root", ROOT, "--tokens", os.path.join(d, "tokens.i32"), "--lens", os.path.join(d, "lens.i32"), "--adapters", adir, "--logits"], capture_output=True, text=True)
say(done.returncode == 0, f"qsw decode --adapters wrote the session ({len(os.listdir(out))} files){'' if done.returncode == 0 else ': ' + done.stderr.strip()[:300]}")
if done.returncode != 0:
    sys.exit(1)
vp = (V + 1535) // 1536 * 1536

# the references: the whole sequence at once, with the adapters and without
xs, _ = R.lora_forward(c, P, AB, tokens, 2.0)
h = R.rms(xs[-1], P["norm"], c.eps)
ref = (h @ R.out_matrix(P).T).reshape(NB, T, V)
xb = R.forward(c, P, tokens)
base = (R.rms(xb[-1], P["norm"], c.eps) @ R.out_matrix(P).T).reshape(NB, T, V)
gap = float(np.abs(ref - base).max() / np.abs(ref).max())

p = subprocess.Popen([BIN, os.path.join(out, "session")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
worst, agree, decided, cos_min = 0.0, 0, 0, 1.0
for j in range(T):
    p.stdin.write("1\n")
    p.stdin.flush()
    line = p.stdout.readline()
    if not line:
        p.wait()
        print("session:", p.returncode, p.stderr.read()[:600])
        sys.exit(1)
    vals = np.array(line.split(), dtype=np.float64)
    got = vals[:NB * vp].reshape(NB, vp)[:, :V]
    want = ref[:, j]
    err = float(np.abs(got - want).max() / np.abs(want).max())
    cos = float(np.min([got[b] @ want[b] / np.linalg.norm(got[b]) / np.linalg.norm(want[b]) for b in range(NB)]))
    worst, cos_min = max(worst, err), min(cos_min, cos)
    for b in range(NB):
        top2 = np.sort(want[b])[-2:]
        if top2[1] - top2[0] > 1e-2 * (want[b].max() - want[b].min()):
            decided += 1
            agree += int(got[b].argmax() == want[b].argmax())
    print(f"     step {j}: max difference {err:.2e} of the largest logit, cosine {cos:.6f}", flush=True)
p.stdin.close()
p.wait()
say(gap > 0.1, f"the adapters matter: the reference without them differs from the one with them by {gap:.2e} of the largest logit (must exceed 0.1)")
say(worst < 5e-3, f"{LAYERS} layers, {NB} sequences, {T} steps: the largest difference of any step is {worst:.2e} of the largest logit (limit 5e-3), the smallest cosine {cos_min:.6f}")
say(agree == decided, f"the arg max agrees on {agree} of {decided} clear-cut positions")
print("bad" if bad else "all ok")
sys.exit(1 if bad else 0)
