#!/usr/bin/env python3
"""docs/resident.md section 27 (roadmap M7.5): a Qwen2 LoRA session over a 4-bit base, held to the binary64 specification of examples/qwen/qwen_ref.py on the dequantised weights.

    python3 tests/qwen_quant.py build/cancho-gpu-session [build/cancho-gpu-metal-session]

1. The packing (examples/qwen/nvfp4.py): the decoder, which reads only the three byte files, equals the encoder's own dequantisation once rounded to f16, for G = 16, 32 and 64; a block of rows is
   packed exactly as the whole matrix; a zero row and an outlier survive.
2. The converter's `--quant nvfp4:G` writes the same files as quantising the converter's f16 files (needs torch and safetensors; skipped without).
3. A Qwen-shaped model (2 layers, 4 query heads over 2 key/value heads of 32) whose four matrices a block are the files `.q`, `.s`, `.g`: the session's dequantise kernel (`dq_fp4.chx`, into one scratch shared by
   the layers and by the forward and the backward) and the unchanged products give the loss and the gradient of every adapter matrix of the specification on the dequantised weights, on the CPU interpreter
   and, when given, the GPU; for G = 16 and 32, with tiles that cut k (so that the scale window moves).
4. Mutants of the dequantise kernel (the row scale of the first block for every block; the scale window not moving with k; the code window not moving with k) and of the writer (the forward's scratch read by
   the backward unchanged is not a mutant: the session file refuses an alias that would overwrite a value a later program reads): each makes the gradients differ from the specification's."""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "qwen"))
try:
    import numpy as np
    import nvfp4
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


def packing_checks():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((70, 192)) * 0.04
    w[4, 9] = 1.5
    w[11] = 0.0
    w[20, :16] = 0.0
    for G in (16, 32, 64):
        q, s, g = nvfp4.pack_rows(w, G, block=16)
        qq, ss, gg = nvfp4.pack_rows(w, G, block=4096)
        dec = nvfp4.unpack_rows(q, s, g, G)
        enc = nvfp4.nvfp4(w, G)[0]
        e = np.abs(dec - enc).max()
        same16 = np.array_equal(dec.astype(np.float16), enc.astype(np.float16))
        say(same16 and e < 1e-14 and np.array_equal(q, qq) and np.array_equal(s, ss) and np.array_equal(g, gg) and not dec[11].any(),
            f"nvfp4 G = {G}: the decoder (bytes only) equals the encoder's dequantisation (largest difference {e:.1e}, equal once rounded to f16), blocks of 16 rows pack as the whole matrix, a zero row stays zero")
    q, s, g = nvfp4.pack_rows(w, 16)
    s2 = s.copy()
    s2[0, 0] = 0x7F
    try:
        nvfp4.unpack_rows(q, s2, g, 16)
        say(False, "nvfp4: a NaN scale byte is refused")
    except ValueError:
        say(True, "nvfp4: a NaN scale byte is refused")


def converter_quant_check(d):
    try:
        import torch
        from safetensors.torch import save_file
        import convert
    except ImportError:
        print("skip: the converter check needs torch and safetensors")
        return
    c = R.Config(layers=2, d=128, h=4, hkv=2, dh=32, ff=256, v=320, t=64, nb=2)
    P = R.init_params(c, seed=3)
    hf = {"model.embed_tokens.weight": P["emb"], "model.norm.weight": P["norm"]}
    for l in range(c.layers):
        q, a, n = f"model.layers.{l}.", f"model.layers.{l}.self_attn.", f"l{l:02d}."
        wqkv, bqkv, wgu = P[n + "wqkv"], P[n + "bqkv"], P[n + "wgu"]
        hf.update({q + "input_layernorm.weight": P[n + "ln1"], q + "post_attention_layernorm.weight": P[n + "ln2"],
                   a + "q_proj.weight": wqkv[:c.d], a + "k_proj.weight": wqkv[c.d:c.d + c.dkv], a + "v_proj.weight": wqkv[c.d + c.dkv:],
                   a + "q_proj.bias": bqkv[:c.d], a + "k_proj.bias": bqkv[c.d:c.d + c.dkv], a + "v_proj.bias": bqkv[c.d + c.dkv:],
                   a + "o_proj.weight": P[n + "wo"], q + "mlp.gate_proj.weight": wgu[:c.ff], q + "mlp.up_proj.weight": wgu[c.ff:], q + "mlp.down_proj.weight": P[n + "wd"]})
    hf = {k: torch.tensor(v, dtype=torch.float32).to(torch.bfloat16).contiguous() for k, v in hf.items()}
    cfg = {"num_hidden_layers": 2, "hidden_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 256, "vocab_size": 320, "rms_norm_eps": 1e-6, "rope_theta": 1e6}
    src = os.path.join(d, "fake.safetensors")
    save_file(hf, src)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    plain = convert.convert(src, os.path.join(d, "config.json"), os.path.join(d, "plain"))
    quant = convert.convert(src, os.path.join(d, "config.json"), os.path.join(d, "q"), quant="nvfp4:16")
    same = True
    for l in range(2):
        for m in ("wqkv", "wo", "wgu", "wd"):
            w16 = np.fromfile(os.path.join(d, "plain", f"l{l:02d}.{m}.f16"), dtype="<f2")
            n_, k_ = R.shapes(R.from_hf(cfg))[f"l{l:02d}.{m}"]
            q, s, g = nvfp4.pack_rows(w16.reshape(n_, k_), 16)
            for suffix, part in (("q", q), ("s", s), ("g", g)):
                same &= np.fromfile(os.path.join(d, "q", f"l{l:02d}.{m}.{suffix}"), dtype=part.dtype).tobytes() == part.tobytes()
            same &= not os.path.exists(os.path.join(d, "q", f"l{l:02d}.{m}.f16"))
    say(same and quant["quant"] == {"format": "nvfp4", "group": 16} and "quant" not in plain, "converter --quant nvfp4:16: every matrix of a block is the .q/.s/.g of its f16 values, no .f16 is written, the manifest names the format")
    # a sharded checkpoint with an untied output layer (Qwen2.5-7B): two shards and an index; head.f16 is lm_head, emb.f16 the table, and the blocks are what the single-file conversion wrote
    names = sorted(hf)
    head = torch.tensor(np.random.default_rng(9).standard_normal((320, 128)), dtype=torch.float32).to(torch.bfloat16).contiguous()
    allt = dict(hf)
    allt["lm_head.weight"] = head
    names = sorted(allt)
    half = len(names) // 2
    sh = os.path.join(d, "sharded")
    os.makedirs(sh, exist_ok=True)
    save_file({k: allt[k] for k in names[:half]}, os.path.join(sh, "model-00001-of-00002.safetensors"))
    save_file({k: allt[k] for k in names[half:]}, os.path.join(sh, "model-00002-of-00002.safetensors"))
    json.dump({"metadata": {}, "weight_map": {k: "model-00001-of-00002.safetensors" if k in names[:half] else "model-00002-of-00002.safetensors" for k in names}}, open(os.path.join(sh, "model.safetensors.index.json"), "w"))
    json.dump(dict(cfg, tie_word_embeddings=False), open(os.path.join(sh, "config.json"), "w"))
    found = convert.find(sh)
    conv = convert.convert(found[0], found[1], os.path.join(d, "sharded_out"))
    rd = lambda f: open(os.path.join(d, f), "rb").read()
    same_blocks = all(rd(f"plain/l{l:02d}.{m}.f16") == rd(f"sharded_out/l{l:02d}.{m}.f16") for l in range(2) for m in ("wqkv", "wo", "wgu", "wd"))
    say(found[0].endswith("model.safetensors.index.json") and same_blocks and rd("sharded_out/emb.f16") == rd("plain/emb.f16")
        and np.array_equal(np.fromfile(os.path.join(d, "sharded_out", "head.f16"), dtype="<f2"), head.float().numpy().astype(np.float16).reshape(-1)) and not os.path.exists(os.path.join(d, "plain", "head.f16")),
        "converter: a checkpoint in two shards with an index, and an untied lm_head, converts to the single file's blocks and embedding table plus head.f16 (the tied checkpoint writes none)")
    for bad_q in ("int4:64", "nvfp4:24", "nvfp4"):
        try:
            convert.convert(src, os.path.join(d, "config.json"), os.path.join(d, "bad"), quant=bad_q)
            say(False, f"converter --quant {bad_q} is refused")
        except SystemExit as e:
            say("can be read by a session" in str(e), f"converter --quant {bad_q} is refused")


def quantised_dir(d, c, P, G):
    """The weights directory of the model P, with the four matrices of every block packed."""
    wdir = os.path.join(d, f"w{G}")
    S.write_files(wdir, c, {k: v for k, v in P.items() if k.split(".")[-1] not in ("wqkv", "wo", "wgu", "wd")})
    for name, w in P.items():
        if name.split(".")[-1] in ("wqkv", "wo", "wgu", "wd"):
            for suffix, part in zip("qsg", nvfp4.pack_rows(w, G)):
                part.tofile(os.path.join(wdir, f"{name}.{suffix}"))
    json.dump(dict(quant=dict(format="nvfp4", group=G)), open(os.path.join(wdir, "manifest.json"), "w"))
    return wdir


def grads_of(binary, extra, d, c, wdir, tok, AB, mut=(), tune=None, ls=64.0):
    t = dict(bc=128, kmut=list(mut))
    t.update(tune or {})
    sess = T.write(os.path.join(d, "g"), c, wdir, tok, AB, r=16, loss_scale=ls, train=False, grads=os.path.join(d, "grads.bin"), tune=t)
    ss = S.Session(binary, sess, extra)
    out, _ = ss.ask("1")
    ss.checkpoint()
    ss.close()
    return float(out.split()[0]), T.split_grads(c, 16, T.read_fused(c, 16, os.path.join(d, "grads.bin")), float(c.nb * (c.t - 1)), ls), sess


def compare(got, ref):
    rel, cos = 0.0, 1.0
    for k in ref:
        a, b = got[k].ravel(), ref[k].ravel()
        if np.linalg.norm(b) == 0:
            rel = max(rel, float(np.linalg.norm(a)))
            continue
        rel = max(rel, float(np.linalg.norm(a - b) / np.linalg.norm(b)))
        cos = min(cos, float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)) if np.linalg.norm(a) > 0 else 0.0)
    return rel, cos


def session_checks(cpu, metal, d):
    c = R.Config(layers=2, d=128, h=4, hkv=2, dh=32, ff=256, v=320, t=64, nb=2)
    P = R.init_params(c, seed=1)
    P["emb"] = P["emb"] * 0.3
    tok = np.random.default_rng(0).integers(0, c.v, (c.nb, c.t))
    AB = R.init_lora(c, 16, 7, b_std=0.05)
    tol, ctol = 4e-3, 0.9995
    tune = dict(dq=dict(bn=32, bk=64))  # k = 128 and 256 are cut into 2 and 4 tiles, n = 128 .. 512 into 4 .. 16
    for G in (16, 32):
        wdir = quantised_dir(d, c, P, G)
        Pq = R.load_params(wdir, c)
        loss, ref = R.lora_loss_grads(c, Pq, AB, tok)
        _, ref_plain = R.lora_loss_grads(c, P, AB, tok)
        far = max(np.linalg.norm(ref[k] - ref_plain[k]) / np.linalg.norm(ref_plain[k]) for k in ref if np.linalg.norm(ref_plain[k]) > 0)
        say(far > 0.02, f"G = {G}: the quantised base is not the f16 one (the adapter gradients differ by {far:.2f} relative), so a session that passes below really runs the 4-bit weights")
        for binary, extra, label in ((cpu, ["--target", "nvidia-ada"], "CPU session"), (metal, [], "Metal session")):
            if not binary:
                continue
            dd = os.path.join(d, f"s{G}{label[0]}")
            os.makedirs(dd, exist_ok=True)
            got_loss, got, sess = grads_of(binary, extra, dd, c, wdir, tok, AB, tune=tune)
            rel, cos = compare(got, ref)
            say(rel < tol and cos > ctol and abs(got_loss - loss) < 3e-3 * abs(loss),
                f"{label}, 4-bit base (nvfp4:{G}): loss {got_loss:.5f} against the specification's {loss:.5f} on the dequantised weights; all {len(ref)} adapter gradients equal its (largest relative norm error {rel:.1e}, smallest cosine {cos:.5f})")
            if G == 16 and label.startswith("CPU"):
                text = open(sess).read()
                nres = text.count(" i8\n")
                nal = text.count("\nalias dqf_") + text.count("\nalias dqb_")
                say(nres == 2 * 2 * 4 * c.layers and nal == 4 * (2 * c.layers - 1) and not any(f"{m}.f16" in text for m in ("wqkv", "wo", "wgu", "wd")),
                    f"the session file reads {nres} i8 files (q and s of 4 matrices, read by the forward and the backward of {c.layers} layers), chains the dequantise scratch of each matrix kind through the layers and the backward "
                    f"({nal} alias lines) and reads no f16 copy of a block matrix")
    # the 7B's configuration: a 4-bit base under an untied head
    cu = R.Config(layers=2, d=128, h=4, hkv=2, dh=32, ff=256, v=320, t=64, nb=2, tied=False)
    Pu = R.init_params(cu, seed=8)
    Pu["emb"] = Pu["emb"] * 0.3
    Pu["head"] = Pu["head"] * 0.3
    os.makedirs(os.path.join(d, "untied"), exist_ok=True)
    wu = quantised_dir(os.path.join(d, "untied"), cu, Pu, 16)
    Pq_u = R.load_params(wu, cu)
    _, ref_u = R.lora_loss_grads(cu, Pq_u, AB, tok)
    for binary, extra, label in ((cpu, ["--target", "nvidia-ada"], "CPU session"), (metal, [], "Metal session")):
        if not binary:
            continue
        dd = os.path.join(d, "su" + label[0])
        os.makedirs(dd, exist_ok=True)
        _, got, _ = grads_of(binary, extra, dd, cu, wu, tok, AB, tune=tune)
        rel, cos = compare(got, ref_u)
        say(rel < tol and cos > ctol, f"{label}, 4-bit base under an untied head: all {len(ref_u)} adapter gradients equal the specification's (largest relative norm error {rel:.1e}, smallest cosine {cos:.5f})")
    # the forward session of M7.1 over the same files: the loss of every position and the arg max of its logits, and the residual stream after the last block, against the specification's
    wdir = quantised_dir(d, c, P, 16)
    Pq = R.load_params(wdir, c)
    xs = R.forward(c, Pq, tok)
    rows, best = R.head(c, Pq, xs[-1], R.next_targets(tok), chunk=100)
    for binary, extra, label in ((cpu, ["--target", "nvidia-ada"], "CPU session"), (metal, [], "Metal session")):
        if not binary:
            continue
        dd = os.path.join(d, "f" + label[0])
        os.makedirs(dd, exist_ok=True)
        sess = S.write(dd, c, wdir, tok, acts=os.path.join(dd, "acts.bin"), tune=dict(bc=128, dq=dict(bn=32, bk=64)))
        ss = S.Session(binary, sess, extra)
        loss, am, _ = ss.forward(c.n)
        ss.checkpoint()
        ss.close()
        valid = np.arange(c.n) % c.t != c.t - 1
        e = float(np.abs(loss[valid] - rows[valid]).max())
        act = S.read_acts(os.path.join(dd, "acts.bin"), c)[-1]
        r = float(np.abs(act - xs[-1]).max() / np.abs(xs[-1]).max())
        say(e < 1e-2 and r < 3e-3 and (am == best).mean() > 0.99, f"{label}, 4-bit base, forward session: the loss of each position is within {e:.1e} of the specification's, the residual stream after the last block {r:.1e} of its largest value, arg max equal at {(am == best).mean() * 100:.1f}% of {c.n} positions")
    _, ref = R.lora_loss_grads(c, Pq, AB, tok)
    for what, mut in (("the row scales of the first block of rows for every block", [("dq_fp4.chx", "load wg[i * bn ; bn]", "load wg[0 ; bn]")]),
                      ("the scale window does not move with k", [("dq_fp4.chx", "j * (bk / 16)", "0")]),
                      ("the code window does not move with k", [("dq_fp4.chx", "j * (bk / 2)", "0")])):
        dd = os.path.join(d, "m" + str(abs(hash(what)) % 10 ** 6))
        os.makedirs(dd, exist_ok=True)
        try:
            _, got, _ = grads_of(cpu, ["--target", "nvidia-ada"], dd, c, wdir, tok, AB, mut=mut, tune=tune)
            rel, cos = compare(got, ref)
            say(rel > 5 * tol or cos < 0.99, f"dequantise-kernel mutant ({what}): the gradients are {rel:.1e} from the specification's (cosine {cos:.4f}), so the check fails")
        except SystemExit as e:
            say(True, f"dequantise-kernel mutant ({what}): the session refuses it ({str(e)[:80]})")


def main():
    cpu = os.path.abspath(sys.argv[1])
    metal = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 and os.path.exists(sys.argv[2]) else None
    d = tempfile.mkdtemp()
    packing_checks()
    converter_quant_check(d)
    session_checks(cpu, metal, d)
    print(f"{'qwen_quant: ok' if not bad else f'qwen_quant: {bad} FAILED'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
