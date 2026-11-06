#!/usr/bin/env python3
"""The batched decode session (`qsw decode`, docs/decode.md) against the float64 forward pass (examples/qwen/qwen_ref.py): a random Qwen2.5-1.5B-shaped model of 2 layers and a small
vocabulary, `nb` sequences whose prompts are fed one token a step (teacher forcing, all steps) with the session answering the logits of every step (`--logits`, one request a step), compared
with the logits the full forward pass gives at the same positions. Needs a Metal GPU.

    python3 tests/decode_dev.py build/qsw build/cancho-gpu-metal-session [--layers 2] [--steps 12] [--nb 2] [--v 1024]

Per step: the largest difference over the logits relative to their largest, the cosine, and whether the arg max agrees; the error of f16 weights and f16 activations in every product is expected
(about 1e-3), so the check is that no step is worse than 5e-3 and the arg max agrees wherever the reference's top two logits differ by more than 1e-2 of their range."""
import os, subprocess, sys, tempfile
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

c = R.Config(layers=LAYERS, d=1536, h=12, hkv=2, dh=128, ff=8960, v=V, t=T, nb=NB)
P = R.init_params(c, seed=5)
d = tempfile.mkdtemp(prefix="decode_")
wdir, out = os.path.join(d, "w"), os.path.join(d, "out")
S.write_files(wdir, c, P)
os.makedirs(out)
rng = np.random.default_rng(2)
tokens = rng.integers(0, V, size=(NB, T)).astype("<i4")
tokens.tofile(os.path.join(d, "tokens.i32")); np.full(NB, T, dtype="<i4").tofile(os.path.join(d, "lens.i32"))
def bits(x):
    import struct
    return struct.unpack("<q", struct.pack("<d", x))[0]
open(os.path.join(d, "cfg.txt"), "w").write("".join(f"{k} {v}\n" for k, v in dict(layers=LAYERS, d=1536, h=12, hkv=2, dh=128, ff=8960, v=V, t=64, nb=NB, tied=1, r=16, steps=T, eps=bits(c.eps), theta=bits(c.theta)).items()))
done = subprocess.run([QSW, "decode", "--config", os.path.join(d, "cfg.txt"), "--wdir", wdir, "--out", out, "--root", ROOT, "--tokens", os.path.join(d, "tokens.i32"), "--lens", os.path.join(d, "lens.i32"), "--logits"], capture_output=True, text=True)
say(done.returncode == 0, f"qsw decode wrote the session ({len(os.listdir(out))} files){'' if done.returncode == 0 else ': ' + done.stderr.strip()[:300]}")
if done.returncode != 0:
    sys.exit(1)
vp = (V + 1535) // 1536 * 1536

# the reference: the whole sequence at once
xs = R.forward(c, P, tokens)
h = R.rms(xs[-1], P["norm"], c.eps)
ref = (h @ R.out_matrix(P).T).reshape(NB, T, V)

p = subprocess.Popen([BIN, os.path.join(out, "session")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
worst, agree, decided, cos_min = 0.0, 0, 0, 1.0
sampler_ok = True
for j in range(T):
    p.stdin.write("1\n"); p.stdin.flush()
    line = p.stdout.readline()
    if not line:
        p.wait(); print("session:", p.returncode, p.stderr.read()[:600]); sys.exit(1)
    vals = np.array(line.split(), dtype=np.float64)
    got, drawn = vals[:NB * vp].reshape(NB, vp)[:, :V], vals[NB * vp:].astype(np.int64) - 1
    want = ref[:, j]
    err = float(np.abs(got - want).max() / np.abs(want).max())
    cos = float(np.min([got[b] @ want[b] / np.linalg.norm(got[b]) / np.linalg.norm(want[b]) for b in range(NB)]))
    worst, cos_min = max(worst, err), min(cos_min, cos)
    for b in range(NB):
        sampler_ok &= bool(drawn[b] == got[b].argmax())
    for b in range(NB):
        top2 = np.sort(want[b])[-2:]
        if top2[1] - top2[0] > 1e-2 * (want[b].max() - want[b].min()):
            decided += 1; agree += int(got[b].argmax() == want[b].argmax())
    print(f"     step {j}: max difference {err:.2e} of the largest logit, cosine {cos:.6f}", flush=True)
p.stdin.close(); p.wait()
say(worst < 5e-3, f"{LAYERS} layers, {NB} sequences, {T} steps: the largest difference of any step is {worst:.2e} of the largest logit (limit 5e-3), the smallest cosine {cos_min:.6f}")
say(sampler_ok, "the id the sampler drew (invt 1.25 by default: not an arg max)") if False else None
say(agree == decided, f"the arg max agrees on {agree} of {decided} clear-cut positions")
print("bad" if bad else "all ok")
sys.exit(1 if bad else 0)
