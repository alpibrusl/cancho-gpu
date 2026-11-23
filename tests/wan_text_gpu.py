#!/usr/bin/env python3
"""V3 of docs/diffusion.md section 6: the text of a Wan2.2 prompt on a Metal GPU: umT5-XXL's encoder (24 blocks) as one session, models/umt5_enc.model.

    wan_text_gpu.py "PROMPT" OUT.npy [--wdir DIR]            the stage of `media video`: tools/wan_text.py's output, with no torch
    wan_text_gpu.py --check [--wdir DIR] [--ref DIR]         the GPU's rows against torch's (needs torch and transformers), with mutants

Writes the encoder's last hidden state of the prompt's tokens, [n, 4096] float32, as tools/wan_text.py does (torch, BF16 on MPS: the stage this replaces, kept as the other side of `--check`).

The session runs the prompt's own n tokens padded to SP rows, SP the next multiple of 64, not the 512 the pipelines pad to: padded keys are masked, so the n real rows do not depend on the padding (measured: the design in
docs/diffusion.md). Two kernels are umT5's own (`dit_scores_bias`: the scores with a bias per head and no scale; `dit_geglu`: gelu_new(g) * u); the rest are the DiT's (`rmsnorm_rows`, `dit_to_f16`, `gemm_mma`, `dit_stats`, `dit_pv`, `dit_add`).
On the host: the tokeniser (`tokenizers`, the snapshot's tokenizer.json), the n rows of the embedding table (read from the checkpoint's shard), and each block's relative position bias expanded from its [32, 64] table to
[64 * SP, SP] with the key mask added. `--wdir` (default ~/.cache/cancho-media/umt5-f16) holds the f16 weight files, written on first use from the checkpoint's BF16 safetensors (9.3 GB, 6 s on an M4 Max; refused when the disk has
less than 11 GB free). Needs a Mac, numpy, `tokenizers`, `build/cancho-gpu-metal-session` and `text_encoder/*`, `tokenizer/*` of `Wan-AI/Wan2.2-TI2V-5B-Diffusers` in the Hugging Face cache. Not in `scripts/gate.sh`.

`--check`: for four prompts (1, 3, 19 and 102 tokens) the GPU's rows against torch float32 on the CPU, with torch BF16 on MPS as the scale. The bound, written before the first run: on every row the GPU's max abs error is under
BF16's for that row and its cosine above BF16's, and the prompt's mean abs error is under a quarter of BF16's. Three mutants must fail it (`--mutate no-bias|transposed|reversed`: the relative bias dropped with the key mask kept,
the bias read at bucket(i - j), the token rows fed in reverse order). Printed, not bounded: the rows at SP = 128 against SP = 64, the tokeniser's ids and the bucket function against transformers'. `--ref DIR` keeps torch's rows
between runs.
"""
import glob
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION = os.path.join(ROOT, "build/cancho-gpu-metal-session")
MODEL = os.path.join(ROOT, "models/umt5_enc.model")
D, FF, H, NL, NBUCKET, MAXDIST, T = 4096, 10240, 64, 24, 32, 128, 512
NEG = -1e9
CHECK = sys.argv[1:2] == ["--check"]
OPTS = sys.argv[2:] if CHECK else sys.argv[3:]  # the stage's first two arguments are the prompt and the output, whatever they look like
arg = lambda name, default=None: OPTS[OPTS.index(name) + 1] if name in OPTS[:-1] else default
WDIR = os.path.expanduser(arg("--wdir", "~/.cache/cancho-media/umt5-f16"))


def fail(msg):
    print("wan_text_gpu: " + msg, file=sys.stderr)
    sys.exit(1)


def snapshot():
    s = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/*")))
    if not s or not os.path.isdir(s[-1] + "/text_encoder") or not os.path.isfile(s[-1] + "/tokenizer/tokenizer.json"):
        fail("the text encoder is not in the Hugging Face cache: `hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers`")
    return s[-1]


def tensors(snap):
    """name -> (file, offset, shape) of the checkpoint's tensors, from the safetensors headers (8 bytes of length, then JSON); every one is BF16."""
    out = {}
    for f in sorted(glob.glob(snap + "/text_encoder/*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            if v["dtype"] != "BF16":
                fail("%s is %s in %s: this reader takes BF16" % (k, v["dtype"], f))
            out[k] = (f, 8 + n + v["data_offsets"][0], tuple(v["shape"]))
    return out


def bf16(tens, name, rows=None):
    """A tensor (or the given rows of a matrix) as float32: BF16 is the top 16 bits of a float32."""
    if name not in tens:
        fail("the checkpoint has no tensor " + name)
    f, off, shape = tens[name]
    u = np.memmap(f, dtype="<u2", mode="r", offset=off, shape=shape)
    if rows is not None:
        u = u[np.asarray(rows)]
    return (np.asarray(u).astype(np.uint32) << 16).view(np.float32)


def write_weights(snap):
    """The f16 files of models/umt5_enc.model, once. nf.f32 is written last: its presence says the directory is whole."""
    if os.path.isfile(os.path.join(WDIR, "nf.f32")):
        return
    parent = os.path.dirname(WDIR.rstrip("/")) or "/"
    os.makedirs(parent, exist_ok=True)
    free = shutil.disk_usage(parent).free
    if free < 11e9:
        fail("the f16 weight files (9.3 GB) are written to %s on first use and only %.0f GB are free" % (WDIR, free / 1e9))
    os.makedirs(WDIR, exist_ok=True)
    tens = tensors(snap)
    t0 = time.time()
    Wp = lambda n: os.path.join(WDIR, n)
    tables = []
    for i in range(NL):
        a, f = "encoder.block.%d.layer.0." % i, "encoder.block.%d.layer.1." % i
        for nm, key in (("wq%d", a + "SelfAttention.q"), ("wk%d", a + "SelfAttention.k"), ("wv%d", a + "SelfAttention.v"), ("wo%d", a + "SelfAttention.o"),
                        ("w0_%d", f + "DenseReluDense.wi_0"), ("w1_%d", f + "DenseReluDense.wi_1"), ("w2_%d", f + "DenseReluDense.wo")):
            bf16(tens, key + ".weight").astype("<f2").tofile(Wp(nm % i + ".f16"))
        bf16(tens, a + "layer_norm.weight")[None].astype("<f4").tofile(Wp("n1_%d.f32" % i))
        bf16(tens, f + "layer_norm.weight")[None].astype("<f4").tofile(Wp("n2_%d.f32" % i))
        tables.append(bf16(tens, a + "SelfAttention.relative_attention_bias.weight"))
    np.stack(tables).astype("<f4").tofile(Wp("rb.f32"))  # [24, 32 buckets, 64 heads]
    bf16(tens, "encoder.final_layer_norm.weight")[None].astype("<f4").tofile(Wp("nf.f32"))
    print("weights written to %s in %.0f s" % (WDIR, time.time() - t0), flush=True)


def buckets(rel):
    """transformers' `UMT5Attention._relative_position_bucket` for an encoder (bidirectional, 32 buckets, max distance 128), rel = key position - query position: 16 buckets a sign, offsets 0 to 7 a bucket each,
    8 to 127 in 8 logarithmic ones, 15 beyond. The logarithm is float32 there and here (a boundary such as 32 depends on it); `--check` compares every offset from -511 to 511 with theirs."""
    half, exact = NBUCKET // 2, NBUCKET // 4
    r = np.abs(rel)
    with np.errstate(divide="ignore"):
        lg = np.log(r.astype(np.float32) / np.float32(exact)) / np.float32(math.log(MAXDIST / exact)) * np.float32(half - exact)
    large = np.minimum(exact + np.where(r > 0, lg, 0).astype(np.int64), half - 1)
    return (rel > 0) * half + np.where(r < exact, r, large)


def tokenizer(snap):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(snap + "/tokenizer/tokenizer.json")
    tk.no_padding()
    tk.enable_truncation(T)
    return tk


def encode(snap, ids, pad=None, mutate=None):
    """The encoder's last hidden state of the tokens `ids`, [n, 4096] float32, from one session; and the seconds the session took."""
    n = len(ids)
    sp = pad or (n + 63) // 64 * 64
    assert 1 <= n <= sp and sp % 64 == 0
    work = tempfile.mkdtemp(prefix="umt5-")
    try:
        P = lambda nm: os.path.join(work, nm)
        x0 = np.zeros((sp, D), dtype="<f4")
        order = np.argsort(ids, kind="stable")  # the table's rows in file order, then put back
        x0[order] = bf16(tensors(snap), "shared.weight", rows=np.asarray(ids)[order])
        if mutate == "reversed":
            x0[:n] = x0[:n][::-1]
        x0.tofile(P("x0.f32"))
        tab = np.fromfile(os.path.join(WDIR, "rb.f32"), dtype="<f4").reshape(NL, NBUCKET, H)
        idx = np.arange(sp)
        rel = idx[None, :] - idx[:, None]  # [query i, key j]: j - i
        bk = buckets(rel.T if mutate == "transposed" else rel)
        keymask = np.where(idx < n, 0.0, NEG).astype(np.float32)[None, None, :]
        for i in range(NL):
            b = np.zeros((H, sp, sp), dtype=np.float32) if mutate == "no-bias" else tab[i][bk].transpose(2, 0, 1)  # [head, query, key]
            (b + keymask).astype("<f4").tofile(P("rb%d.f32" % i))
        t0 = time.time()
        r = subprocess.run([SESSION, MODEL, "CHX=" + os.path.join(ROOT, "tests/chx"), "WORK=" + work, "WDIR=" + WDIR, "SP=%d" % sp, "NL=%d" % NL], input="1\ncheckpoint\n", capture_output=True, text=True, timeout=3600)
        secs = time.time() - t0
        if r.returncode != 0 or not os.path.isfile(P("last.f32")):
            fail("the session did not answer: status %d, %s" % (r.returncode, r.stderr.strip()[-800:]))
        h = np.fromfile(P("last.f32"), dtype="<f4").reshape(sp, D)[:n].copy()
        if not np.isfinite(h).all():
            fail("the encoder's rows are not finite (an activation outside f16?)")
        return h, secs
    finally:
        shutil.rmtree(work, ignore_errors=True)


def stage():
    prompt, out = sys.argv[1], sys.argv[2]
    if not os.path.isfile(SESSION):
        fail("build/cancho-gpu-metal-session is not built: scripts/build-metal-session.sh")
    t0 = time.time()
    snap = snapshot()
    write_weights(snap)
    ids = tokenizer(snap).encode(prompt).ids
    t1 = time.time()
    h, secs = encode(snap, ids)
    np.save(out, h)
    print("prompt of %d tokens (%d rows) -> %s %s, rms %.4g; tokeniser and tables %.1f s, session %.1f s (the compile and the upload of 9.3 GB of weights included)"
          % (len(ids), (len(ids) + 63) // 64 * 64, out, list(h.shape), float(np.sqrt((h ** 2).mean())), t1 - t0, secs))


PROMPTS = ["A red fox running through fresh snow in a pine forest at sunrise, cinematic", "A cat", "",
           "An elderly fisherman in a yellow raincoat mends a net on the deck of a small wooden boat while gulls circle overhead; the sea is grey and choppy, the light is low and golden, and in the distance a lighthouse blinks "
           "through the mist as the camera slowly pulls back to reveal the whole harbour, its stone walls wet with spray, crates of fish stacked by the quay and a child waving from the pier. 日本語のテキストも少し。"]


def rowstats(a, ref):
    d = np.abs(a.astype(np.float64) - ref)
    cos = (a * ref).sum(1) / np.sqrt((a.astype(np.float64) ** 2).sum(1) * (ref ** 2).sum(1))
    return d.mean(), d.max(1), cos


def check():
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]  # transformers probes for a package named `kernels`, and tests/kernels.py is not it
    import torch
    from transformers import AutoTokenizer, UMT5EncoderModel
    snap = snapshot()
    write_weights(snap)
    refdir = arg("--ref")
    tok = AutoTokenizer.from_pretrained(snap + "/tokenizer")
    tk = tokenizer(snap)
    ok = True
    enc = {}

    def torch_rows(i, kind):
        path = os.path.join(refdir, "%s_%d.npy" % (kind, i)) if refdir else None
        if path and os.path.isfile(path):
            return np.load(path)
        if kind not in enc:
            enc.clear()
            enc[kind] = UMT5EncoderModel.from_pretrained(snap + "/text_encoder", torch_dtype=torch.float32 if kind == "f32" else torch.bfloat16).to("cpu" if kind == "f32" else "mps").eval()
        t = tok([PROMPTS[i]], padding="max_length", max_length=T, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        dev = "cpu" if kind == "f32" else "mps"
        with torch.no_grad():
            h = enc[kind](t.input_ids.to(dev), t.attention_mask.to(dev)).last_hidden_state[0, : int(t.attention_mask.sum())].float().cpu().numpy()
        if path:
            os.makedirs(refdir, exist_ok=True)
            np.save(path, h)
        return h

    f32 = [torch_rows(i, "f32").astype(np.float64) for i in range(len(PROMPTS))]
    b16 = [torch_rows(i, "bf16") for i in range(len(PROMPTS))]
    m = UMT5EncoderModel.from_pretrained(snap + "/text_encoder", torch_dtype=torch.bfloat16) if "f32" not in enc and "bf16" not in enc else list(enc.values())[0]
    rel = np.arange(-(T - 1), T)
    theirs = m.encoder.block[0].layer[0].SelfAttention._relative_position_bucket(torch.arange(-(T - 1), T)).numpy()
    same = bool((buckets(rel) == theirs).all())
    ok &= same
    print("%s the bucket of every offset from %d to %d is transformers'" % ("ok  " if same else "FAIL", rel[0], rel[-1]))
    enc.clear()
    del m
    ids = []
    for i, p in enumerate(PROMPTS):
        mine = tk.encode(p).ids
        t = tok([p], padding="max_length", max_length=T, truncation=True, add_special_tokens=True, return_attention_mask=True)
        hf = t["input_ids"][0][: sum(t["attention_mask"][0])]
        ok &= mine == hf
        print("%s prompt %d: %d tokens, the ids of `tokenizers` are AutoTokenizer's" % ("ok  " if mine == hf else "FAIL", i, len(mine)))
        ids.append(mine)

    def judge(i, h, quiet=False):
        """The bound of the design: every row nearer float32 than BF16's row is (max abs error and cosine), the mean abs error under a quarter of BF16's."""
        gm, gmax, gcos = rowstats(h, f32[i])
        bm, bmax, bcos = rowstats(b16[i], f32[i])
        good = bool((gmax < bmax).all() and (gcos > bcos).all() and gm < 0.25 * bm)
        rms = float(np.sqrt((f32[i] ** 2).mean()))
        line = ("mean abs error %.3g (%.3g of the rms; BF16 %.3g: %.1f times); per-row max abs error worst %.3g, median %.3g (BF16 %.3g, %.3g); cosine min %.7f (BF16 %.7f); rows worse than BF16's: %d by max, %d by cosine"
                % (gm, gm / rms, bm, bm / gm if gm > 0 else float("inf"), gmax.max(), np.median(gmax), bmax.max(), np.median(bmax), gcos.min(), bcos.min(), int((gmax >= bmax).sum()), int((gcos <= bcos).sum())))
        return good, line

    rows64 = None
    for i in range(len(PROMPTS)):
        h, secs = encode(snap, ids[i])
        if i == 0:
            rows64 = h
        good, line = judge(i, h)
        ok &= good
        print("%s prompt %d, %d tokens in %d rows, session %.1f s: %s" % ("ok  " if good else "FAIL", i, len(ids[i]), (len(ids[i]) + 63) // 64 * 64, secs, line), flush=True)
    for mut, what in (("no-bias", "the relative bias dropped, the key mask kept"), ("transposed", "the bias read at bucket(i - j)"), ("reversed", "the token rows fed in reverse order")):
        h, _ = encode(snap, ids[0], mutate=mut)
        good, line = judge(0, h)
        ok &= not good
        print("%s mutant, %s: %s" % ("killed  " if not good else "SURVIVED", what, line), flush=True)
    h128, _ = encode(snap, ids[0], pad=128)
    d = np.abs(h128.astype(np.float64) - rows64)
    print("     prompt 0 in 128 rows against 64 rows (not bounded): max abs difference %.3g, mean %.3g" % (d.max(), d.mean()))
    print("%s the umT5 encoder on the GPU against torch float32, four prompts and three mutants" % ("ok  " if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if CHECK:
    check()
elif len(sys.argv) >= 3:
    stage()
else:
    fail("usage: wan_text_gpu.py \"PROMPT\" OUT.npy [--wdir DIR] | --check [--wdir DIR] [--ref DIR]")
