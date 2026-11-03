#!/usr/bin/env python3
"""docs/resident.md section 20 (roadmap M7.1): the forward pass of a Qwen2 model in a session, held to the binary64 specification of examples/qwen/qwen_ref.py.

    python3 tests/qwen.py build/cancho-gpu-session [build/cancho-gpu-metal-session]

1. The converter (examples/qwen/convert.py) on a two-layer fake checkpoint written in bf16: every file has the shape and the values it should (read back through qwen_ref.load_params), a weight
   that does not fit f16 is refused, and the values f16 flushes to zero are counted.
2. The cross-entropy as three kernels (ce_stats, ce_comb) over a vocabulary of 300 padded to 384 and cut into chunks of 128: the loss of each row and the arg max of its logits equal numpy's,
   for targets in every chunk, for an arg max in the padding's neighbourhood and for the largest logit in each chunk.
3. A Qwen-shaped model (2 layers, 4 query heads over 2 key/value heads of 32, SwiGLU, biases on q, k, v, rotary embedding, causal mask; random weights, 2 sequences of 64): the loss of each of the
   128 positions, the arg max of its logits and the residual stream after each block equal the specification's (f16 products, f32 accumulation: 3e-3 relative), on the CPU interpreter and, when given, on the GPU.
4. Mutants of the specification (rotary pairs adjacent elements instead of halves; no bias on q/k/v; a head reading key/value head h % 2 instead of h // 2; no causal mask; gate and up exchanged;
   RMSNorm with the mean taken off): each differs from the session's residual stream by more than the tolerance, so the check in 3 would fail on a kernel that did that.
Needs numpy; without it the test says so and passes. The converter check needs torch and safetensors and is skipped without them."""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "qwen"))
try:
    import numpy as np
    import qwen_ref as R
    import qwen_session as S
except ImportError:
    print("skip: numpy is not installed")
    sys.exit(0)

bad = 0
TOL = 3e-3


def say(good, text):
    global bad
    print(f"{'ok  ' if good else 'FAIL'} {text}")
    bad += 0 if good else 1


def converter_check(d):
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
    # bf16 has an 8-bit significand: round the fake weights to it first, so that the comparison is exact
    hf = {k: torch.tensor(v, dtype=torch.float32).to(torch.bfloat16) for k, v in hf.items()}
    cfg = {"num_hidden_layers": 2, "hidden_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 256, "vocab_size": 320, "rms_norm_eps": 1e-6, "rope_theta": 1e6}
    src = os.path.join(d, "fake.safetensors")
    save_file({k: v.contiguous() for k, v in hf.items()}, src)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    out = os.path.join(d, "converted")
    man = convert.convert(src, os.path.join(d, "config.json"), out)
    back = R.load_params(out, R.from_hf(cfg))
    q = hf["model.layers.1.self_attn.k_proj.weight"].float().numpy().astype(np.float64)
    say(np.array_equal(back["l01.wqkv"][c.d:c.d + c.dkv], q.astype(np.float16).astype(np.float64)), "converter: the k rows of layer 1's fused q/k/v matrix are the checkpoint's k_proj, as f16")
    say(np.array_equal(back["l00.wgu"][c.ff:], hf["model.layers.0.mlp.up_proj.weight"].float().numpy().astype(np.float16)), "converter: the up rows follow the gate rows in the fused gate/up matrix")
    say(np.array_equal(back["l00.bqkv"][:c.d], hf["model.layers.0.self_attn.q_proj.bias"].float().numpy()) and np.array_equal(back["norm"], hf["model.norm.weight"].float().numpy()),
        "converter: biases and norm gains are f32 and equal to the checkpoint's")
    say(man["report"]["overflow"] == 0, f"converter: report says {man['report']['values']} matrix values, none overflowing")
    # a weight that does not fit f16 is refused
    big = dict(hf)
    big["model.layers.0.mlp.down_proj.weight"] = hf["model.layers.0.mlp.down_proj.weight"].clone()
    big["model.layers.0.mlp.down_proj.weight"][0, 0] = 70000.0
    save_file({k: v.contiguous() for k, v in big.items()}, os.path.join(d, "big.safetensors"))
    try:
        convert.convert(os.path.join(d, "big.safetensors"), os.path.join(d, "config.json"), os.path.join(d, "big"))
        say(False, "converter: a weight of 70000 is refused")
    except SystemExit as e:
        say("does not fit f16" in str(e), "converter: a weight of 70000 is refused (it does not fit f16)")
    # the flush is counted
    tiny = dict(hf)
    w = hf["model.layers.0.mlp.down_proj.weight"].clone()
    w[0, 0] = 1e-9
    w[0, 1] = 1e-9
    tiny["model.layers.0.mlp.down_proj.weight"] = w
    save_file({k: v.contiguous() for k, v in tiny.items()}, os.path.join(d, "tiny.safetensors"))
    m2 = convert.convert(os.path.join(d, "tiny.safetensors"), os.path.join(d, "config.json"), os.path.join(d, "tiny"))
    say(m2["report"]["flushed_to_zero"] >= 2, f"converter: values below 2^-25 are counted as flushed to zero ({m2['report']['flushed_to_zero']})")


def run(binary, session, extra, lines):
    r = subprocess.run([binary, session, *extra], input=lines, capture_output=True, text=True, errors="replace", timeout=900)
    if r.returncode != 0:
        raise SystemExit(f"{binary}: {r.stderr[:400]}")
    return r.stdout.split("\n")[:-1]


def ce_check(binary, extra, d, label):
    rng = np.random.default_rng(5)
    r, v, bc = 8, 300, 128
    vp = 384
    lg = rng.standard_normal((r, vp)) * 3.0
    lg[:, v:] = rng.standard_normal((r, vp - v)) * 3.0 + 50.0  # the padding columns hold huge values: only the mask keeps them out
    tgt = np.array([0, 127, 128, 255, 256, 299, 17, 200])
    lg[1, 40] = lg[1, :v].max() + 5.0   # an arg max in the first chunk
    lg[3, 299] = lg[3, :v].max() + 5.0  # an arg max in the last real column
    lg[4, 130] = lg[4, :v].max() + 5.0  # in the middle chunk
    S.put(os.path.join(d, "lg"), lg)
    S.put(os.path.join(d, "tgt"), tgt + 1)
    S.put(os.path.join(d, "jrow"), np.arange(1, vp + 1))
    S.put(os.path.join(d, "maskv"), np.where(np.arange(vp) < v, 0.0, -1e9))
    nq = vp // bc
    q = lambda f: os.path.join(S.HERE, f)
    L = [f"program cs {q('ce_stats.chx')} r={r} c={vp} bc={bc} nq={nq}", f"resident cs.lg {d}/lg f32", f"resident cs.tgt {d}/tgt f32", f"resident cs.jrow {d}/jrow f32", f"resident cs.maskv {d}/maskv f32",
         f"program cc {q('ce_comb.chx')} r={r} nq={nq} bm=2"]
    L += [f"link cs.{o} -> cc.{o}" for o in ("cm", "cs", "ct", "ci")] + ["steps", "answer cc.loss", "answer cc.am"]
    open(os.path.join(d, "ce.session"), "w").write("\n".join(L) + "\n")
    out = run(binary, os.path.join(d, "ce.session"), extra, "1\n")
    got = np.array(out[0].split(), dtype=np.float64)
    real = lg[:, :v]
    m = real.max(1)
    want = np.log(np.exp(real - m[:, None]).sum(1)) + m - real[np.arange(r), tgt]
    say(float(np.abs(got[:r] - want).max()) < 2e-5 * max(1.0, float(np.abs(want).max())), f"{label}: the three-kernel cross-entropy equals numpy's on 8 rows over 3 chunks, targets in every chunk (largest difference {np.abs(got[:r] - want).max():.1e})")
    say(np.array_equal(got[r:].astype(int) - 1, real.argmax(1)), f"{label}: the arg max of each row is numpy's (the padding columns hold 50 and are masked), {[int(x) for x in got[r:] - 1]}")


def forward_check(binary, extra, d, label, c, P, tok, xs, rows, best):
    wdir = os.path.join(d, "w")
    S.write_files(wdir, c, P)
    sess = S.write(os.path.join(d, "s"), c, wdir, tok, acts=os.path.join(d, "acts.bin"), tune=dict(bc=128))
    ss = S.Session(binary, sess, extra)
    loss, am, _ = ss.forward(c.n)
    ss.checkpoint()
    ss.close()
    acts = S.read_acts(os.path.join(d, "acts.bin"), c)
    rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
    errs = [rel(acts[l], xs[l + 1]) for l in range(c.layers)]
    say(max(errs) < TOL, f"{label}: the residual stream after each of the {c.layers} blocks equals the specification's to {max(errs):.1e} (relative to its largest value; tolerance {TOL:g})")
    lerr = float(np.abs(loss - rows).max() / np.abs(rows).max())
    say(lerr < TOL, f"{label}: the loss of each of the {c.n} positions equals the specification's to {lerr:.1e}; the mean over the positions with a next token is {R.mean_loss(loss, c):.4f} against {R.mean_loss(rows, c):.4f}")
    say(float((am == best).mean()) >= 0.98, f"{label}: the arg max of the logits equals the specification's at {(am == best).mean() * 100:.1f}% of the positions")
    return acts


def mutants(c, P, tok, acts):
    """Each mutant of the specification must be far from the session's residual stream (the check above would fail)."""
    last = acts[-1]
    base = R.forward(c, P, tok)[-1]
    rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
    say(rel(last, base) < TOL, f"mutants: the unmutated specification is within tolerance of the session ({rel(last, base):.1e})")
    saved = (R.rope, R.attention, R.rms, R.layer)

    def run_with(**kw):
        try:
            if "rope" in kw:
                R.rope = kw["rope"]
            if "attention" in kw:
                R.attention = kw["attention"]
            if "rms" in kw:
                R.rms = kw["rms"]
            Q = kw.get("P", P)
            return R.forward(c, Q, tok)[-1]
        finally:
            R.rope, R.attention, R.rms, R.layer = saved

    def rope_adjacent(x, cs, sn, heads, cc):
        y = x.reshape(cc.nb, cc.t, heads, cc.dh // 2, 2)
        x1, x2 = y[..., 0], y[..., 1]
        co, si = cs[None, :, None, :], sn[None, :, None, :]
        return np.stack([x1 * co - x2 * si, x2 * co + x1 * si], -1).reshape(cc.n, heads * cc.dh)

    def attn_wrong_group(q, k, v, cc, mask=True, wrong=True):
        qh, kh, vh = q.reshape(cc.nb, cc.t, cc.h, cc.dh), k.reshape(cc.nb, cc.t, cc.hkv, cc.dh), v.reshape(cc.nb, cc.t, cc.hkv, cc.dh)
        out = np.zeros_like(qh)
        mk = np.triu(np.full((cc.t, cc.t), -np.inf), 1) if mask else 0.0
        for h in range(cc.h):
            g = h % cc.hkv if wrong else h // cc.group
            s = np.einsum("bid,bjd->bij", qh[:, :, h], kh[:, :, g]) / np.sqrt(cc.dh) + mk
            s -= s.max(-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(-1, keepdims=True)
            out[:, :, h] = np.einsum("bij,bjd->bid", p, vh[:, :, g])
        return out.reshape(cc.n, cc.h * cc.dh)

    nobias = dict(P)
    for l in range(c.layers):
        nobias[f"l{l:02d}.bqkv"] = np.zeros_like(P[f"l{l:02d}.bqkv"])
    swapped = dict(P)
    for l in range(c.layers):
        w = P[f"l{l:02d}.wgu"]
        swapped[f"l{l:02d}.wgu"] = np.concatenate([w[c.ff:], w[:c.ff]])
    cases = {
        "rotary pairs adjacent elements": run_with(rope=rope_adjacent),
        "no bias on q, k, v": run_with(P=nobias),
        "a head reads key/value head h % 2": run_with(attention=attn_wrong_group),
        "no causal mask": run_with(attention=lambda q, k, v, cc: attn_wrong_group(q, k, v, cc, mask=False, wrong=False)),
        "gate and up exchanged": run_with(P=swapped),
        "RMSNorm with the mean taken off": run_with(rms=lambda x, g, eps: (x - x.mean(-1, keepdims=True)) / np.sqrt(((x - x.mean(-1, keepdims=True)) ** 2).mean(-1, keepdims=True) + eps) * g),
    }
    for name, got in cases.items():
        e = rel(last, got)
        say(e > 5 * TOL, f"mutant ({name}): differs from the session by {e:.1e}, so the check would fail")


def kernel_mutants(binary, extra, d, c, P, tok, xs):
    """A session built with one deliberate bug in a generated kernel must fail the residual-stream check of forward_check."""
    wdir = os.path.join(d, "cpu", "w")
    for name, mut in (("the rotation's sign", [("let lo = p1 - p2", "let lo = p1 + p2")]),
                      ("the key/value copy for the second head of a group goes to the first", [("* 2 + 1) * t", "* 2 + 0) * t")])):
        dd = os.path.join(d, "mut")
        os.makedirs(dd, exist_ok=True)
        sess = S.write(dd, c, wdir, tok, acts=os.path.join(dd, "acts.bin"), tune=dict(bc=128, mutations=mut))
        ss = S.Session(binary, sess, extra)
        ss.forward(c.n)
        ss.checkpoint()
        ss.close()
        acts = S.read_acts(os.path.join(dd, "acts.bin"), c)
        e = float(np.abs(acts[-1] - xs[-1]).max() / np.abs(xs[-1]).max())
        say(e > 5 * TOL, f"kernel mutant ({name}): the session's residual stream is {e:.1e} from the specification's, so the check fails")


def main():
    cpu = os.path.abspath(sys.argv[1])
    metal = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 and os.path.exists(sys.argv[2]) else None
    d = tempfile.mkdtemp()
    converter_check(d)
    os.makedirs(os.path.join(d, "ce"), exist_ok=True)
    ce_check(cpu, ["--target", "nvidia-ada"], os.path.join(d, "ce"), "CPU session")
    if metal:
        ce_check(metal, [], os.path.join(d, "ce"), "Metal session")
    c = R.Config(layers=2, d=128, h=4, hkv=2, dh=32, ff=256, v=320, t=64, nb=2)
    P = R.init_params(c, seed=1)
    tok = np.random.default_rng(0).integers(0, c.v, (c.nb, c.t))
    xs = R.forward(c, P, tok)
    rows, best = R.head(c, P, xs[-1], R.next_targets(tok), chunk=100)
    os.makedirs(os.path.join(d, "cpu"), exist_ok=True)
    acts = forward_check(cpu, ["--target", "nvidia-ada"], os.path.join(d, "cpu"), "CPU session", c, P, tok, xs, rows, best)
    if metal:
        os.makedirs(os.path.join(d, "gpu"), exist_ok=True)
        forward_check(metal, [], os.path.join(d, "gpu"), "Metal session", c, P, tok, xs, rows, best)
    mutants(c, P, tok, acts)
    kernel_mutants(cpu, ["--target", "nvidia-ada"], d, c, P, tok, xs)
    print(f"{'qwen: ok' if not bad else f'qwen: {bad} FAILED'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
