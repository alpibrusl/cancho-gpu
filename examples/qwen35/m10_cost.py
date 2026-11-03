#!/usr/bin/env python3
"""The cost model of docs/roadmap-models.md M10: parameters, adapters, saved activations, memory and step time of a LoRA fine-tune of a Qwen3.5/3.8-class
model with an NVFP4 base on the cancho-gpu training stack, from the real shapes. Pure Python (no numpy), so the gate can run it anywhere.

    python3 examples/qwen35/m10_cost.py            # the 27B at 1 x 512, 2 x 512 and 4 x 512, with and without recompute
    python3 examples/qwen35/m10_cost.py --7b       # the same accounting on Qwen2.5-7B, held to resident.md section 28's measurements

Marks: the shapes are [read] from config.json, the weight bytes are [measured] against Ollama's manifest (tests/gdn.py holds them), the per-token
activation lists are derived from the backward equations (gdn_ref.py) and the kernels of examples/qwen/qwen_train.py, and calibrated on the 7B run
(the same function reproduces its 15.17 GB of program outputs); times are FLOPs over the 7B run's measured 10.8 TFLOPS plus the probe timings of
time_probes.py. An estimate until a run replaces it."""
import sys

NVFP4_BYTES = 0.5625  # 4.5 bits a weight: e2m1 code 0.5 B + an e4m3 scale per 16 values 1/16 B
TFLOPS = 10.8  # resident.md section 28.5: the 7B step, 57.9 TFLOP in 5.34 s, everything included
DEQ_MS_PER_GB = 55.0 / 3.67  # section 28.5: 55 ms a step for the 7B's 3.67 GB of 4-bit blocks


# per layer at 4 x 512 (1,536 chunks of 64 tokens, 384 scan blocks); [measured] with time_probes.py unless marked
SOLVE_KEY = "solve (I+Ls)^-1, fwd substitution, nv = 64"
GDN_FWD_MS = {"kkt (beta k)k^T": 0.30, "decay mask": 0.11, SOLVE_KEY: 5.69, "u = M vb, w = M (kb e^g): two products": 0.60,
              "Aqk = q k^T D": 0.41, "state scan, 8 chunks x (a + bo + bu) = 8 x 0.57": 4.57, "scan launch gaps, 24 x ~10 us [estimate]": 0.25,
              "conv, silu, l2norm, gates, gated norm, the f16 casts [estimate]": 2.0}


def qwen38_27b():
    c = dict(name="Qwen3.8-27B", hidden=5120, ff=17408, vocab=248320, layers=64, interval=4, nq=24, nkv=4, dh=256, nk=16, nv=48, dk=128, dv=128, kern=4)
    c["types"] = ["full" if (i + 1) % c["interval"] == 0 else "gdn" for i in range(c["layers"])]
    return c


def qwen25_7b():
    c = dict(name="Qwen2.5-7B", hidden=3584, ff=18944, vocab=152064, layers=28, nq=28, nkv=4, dh=128)
    c["types"] = ["dense"] * 28
    return c


def linears(c, kind):
    """[(name, out, in)] of the linear layers of one layer's token mixer (the MLP is common)."""
    h = c["hidden"]
    if kind == "gdn":
        kd, vd = c["nk"] * c["dk"], c["nv"] * c["dv"]
        return [("qkv", 2 * kd + vd, h), ("z", vd, h), ("b", c["nv"], h), ("a", c["nv"], h), ("out", h, vd)]
    if kind == "full":
        return [("q+gate", c["nq"] * c["dh"] * 2, h), ("k", c["nkv"] * c["dh"], h), ("v", c["nkv"] * c["dh"], h), ("o", h, c["nq"] * c["dh"])]
    return [("qkv", (c["nq"] + 2 * c["nkv"]) * c["dh"], h), ("o", h, c["nq"] * c["dh"])]  # Qwen2


def mlp(c):
    return [("gate", c["ff"], c["hidden"]), ("up", c["ff"], c["hidden"]), ("down", c["hidden"], c["ff"])]


def params(c):
    """Parameter counts by group (matrix values; norms and the 3.4 M of small vectors are ignored)."""
    out = {"gdn": 0, "full": 0, "dense": 0, "mlp": 0}
    for t in c["types"]:
        out[t] += sum(o * i for _, o, i in linears(c, t))
        out["mlp"] += sum(o * i for _, o, i in mlp(c))
    out["embed"] = c["vocab"] * c["hidden"]
    out["head"] = c["vocab"] * c["hidden"]
    out["blocks"] = out["gdn"] + out["full"] + out["dense"] + out["mlp"]
    out["total"] = out["blocks"] + out["embed"] + out["head"]
    return out


def lora_params(c, r=16, skip=("b", "a")):
    """Adapter parameters at rank r on every linear of the blocks (peft's all-linear), less the named tiny projections."""
    n = 0
    for t in c["types"]:
        for name, o, i in linears(c, t) + mlp(c):
            if name not in skip:
                n += r * (o + i)
    return n


def act_bytes_per_token(c, kind, recompute_in_layer=False):
    """Bytes a layer keeps for its backward, per token (f16 unless noted; the lean step of resident.md section 23.4: the gate/up product only as f16,
    the SwiGLU output recomputed). Returns (list, total). Scores/probabilities are per sequence of length T and returned separately."""
    h, ff = c["hidden"], c["ff"]
    L = [("residual stream after the mixer (f32)", 4 * h), ("residual stream out (f32)", 4 * h), ("norm1 out", 2 * h), ("norm2 out", 2 * h), ("gate/up f16 copy", 2 * 2 * ff)]
    if kind == "dense":
        nq, nkv, dh = c["nq"], c["nkv"], c["dh"]
        L += [("fused qkv out", 2 * (nq + 2 * nkv) * dh), ("q heads-major (rope'd)", 2 * nq * dh), ("k per query head", 2 * nq * dh), ("v per query head", 2 * nq * dh),
              ("attention out heads-major", 2 * nq * dh), ("attention out (o input)", 2 * nq * dh)]
    elif kind == "full":
        nq, nkv, dh = c["nq"], c["nkv"], c["dh"]
        L += [("q+gate out", 2 * 2 * nq * dh), ("k, v out", 2 * 2 * nkv * dh), ("q after norm+rope, heads-major", 2 * nq * dh), ("k per query head", 2 * nq * dh),
              ("v per query head", 2 * nq * dh), ("attention out heads-major", 2 * nq * dh), ("gated out (o input)", 2 * nq * dh), ("sigmoid(gate)", 2 * nq * dh)]
    else:
        kd, vd, nv = c["nk"] * c["dk"], c["nv"] * c["dv"], c["nv"]
        cd = 2 * kd + vd
        L += [("in_proj_qkv out, pre-conv", 2 * cd), ("conv+silu out (q0 k0 v0)", 2 * cd), ("z", 2 * vd), ("b, a, beta, g (f32)", 4 * 4 * nv),
              # chunk factors, per head and chunk of 64 tokens: M [64,64], w and u [64,128], Aqk [64,64] (f16): per token / 64
              ("M (f16)", 2 * nv * 64), ("w (f16)", 2 * nv * c["dk"]), ("u (f16)", 2 * nv * c["dv"]), ("Aqk (f16)", 2 * nv * 64),
              ("state at each chunk start [nv,128,128] f32 / 64 tokens", 4 * nv * c["dk"] * c["dv"] // 64), ("rule out o (f16)", 2 * vd), ("gated-norm out h (out_proj input)", 2 * vd)]
    return L, sum(b for _, b in L)


def probs_bytes_per_token(c, kind, T):
    if kind in ("dense", "full"):
        return 2 * c["nq"] * T  # f16 probabilities of the T keys, per head
    return 0


def memory(c, nb, T=512, recompute=False, r=16, head_slabs=4, quant=True, head_f16=False, emb_resident=False, overhead=1.6e9, mut=None):
    """Peak device bytes of a step (GB, 1e9) by item."""
    n = nb * T
    p = params(c)
    items = {}
    wbytes = (NVFP4_BYTES if quant else 2.0)
    items["blocks, 4-bit (resident)"] = p["blocks"] * wbytes
    items["head (4-bit resident)" if not head_f16 else "head (f16 resident)"] = p["head"] * (2.0 if head_f16 or not quant else NVFP4_BYTES)
    if emb_resident:
        items["embedding (f16 resident)"] = p["embed"] * 2.0
    n_lora = lora_params(c, r)
    items["adapters: f32 master + 2 moments + f16 copy (14 B)"] = 14.0 * n_lora
    # one dequantisation scratch per matrix kind, shared by all layers and by the forward and the backward (section 27.1)
    kinds = {}
    for t in set(c["types"]):
        for name, o, i in linears(c, t) + mlp(c):
            kinds[(t if name not in ("gate", "up", "down") else "mlp", name)] = o * i * 2
    gu = c["ff"] * c["hidden"] * 2
    scratch = sum(v for (t, name), v in kinds.items() if name not in ("gate", "up")) + 2 * gu  # gate and up are one fused kernel's output
    if quant:
        items["dequantisation scratch (one buffer per matrix kind)"] = scratch
        if not head_f16:
            items[f"head scratch ({head_slabs} slabs)"] = p["head"] * 2.0 / head_slabs
    saved = 0.0
    for t in c["types"]:
        if recompute:
            continue
        _, b = act_bytes_per_token(c, t)
        saved += n * (b + probs_bytes_per_token(c, t, T))
    if recompute:
        # per-layer checkpointing: each layer's input (f32 residual) is kept; one layer's full set is alive at a time (the largest kind)
        saved = n * 4 * c["hidden"] * c["layers"]
        worst = max(n * (act_bytes_per_token(c, t)[1] + probs_bytes_per_token(c, t, T)) for t in set(c["types"]))
        items["one layer's activations (recomputed)"] = worst
    items["saved activations" if not recompute else "checkpointed layer inputs (f32)"] = saved
    if mut != "no_logits":
        items["logits f32"] = 4.0 * n * c["vocab"]
        items["logits gradient f16"] = 2.0 * n * c["vocab"]
    items["embedding rows of the step (f16 table) + targets"] = 2.0 * n * c["hidden"] + 4.0 * n
    # backward temporaries aliased across layers: the largest layer's dP, dS, d(act), d(gate/up), dx ... about the layer's forward set again
    items["backward temporaries (aliased across layers)"] = 0.5 * max(n * (act_bytes_per_token(c, t)[1] + probs_bytes_per_token(c, t, T)) for t in set(c["types"]))
    items["process overhead (calibrated on the 7B run, unexplained)"] = overhead
    return {k: v / 1e9 for k, v in items.items()}


# resident.md section 28.5, measured: the peak footprint of the 7B QLoRA session, 4-bit blocks, f16 head and embedding files
MEASURED_7B = {4: 21.65, 8: 34.63}


def calibrate_7b(**kw):
    """The same accounting on Qwen2.5-7B as that session was run: [(nb, model GB, measured GB)]."""
    c = qwen25_7b()
    return [(nb, sum(memory(c, nb, head_f16=True, emb_resident=True, **kw).values()), g) for nb, g in sorted(MEASURED_7B.items())]


def step_seconds(c, nb, T=512, recompute=False, gdn_core_ms=None):
    """FLOP time at the measured rate, the dequantisation, and the gated-delta core's launches (probe timings, scaled by chunks)."""
    n = nb * T
    p = params(c)
    flop = 4.0 * n * (p["blocks"] + p["head"])
    # attention scores: 2 products of T x T x dh per head forward, the same again for dP/dQ/dK/dV in the backward (about 4 more): 3 x 2 x 2 T dh nq per token a layer
    n_full = sum(1 for t in c["types"] if t in ("full", "dense"))
    flop += n_full * n * (3 * 4 * T * c["dh"] * c["nq"] / 2)
    t_mm = flop / (TFLOPS * 1e12)
    if recompute:
        t_mm += 2.0 * n * (p["blocks"] + p["head"] - p["head"]) / (TFLOPS * 1e12)  # a second forward over the blocks (the head is not recomputed)
    wgb = p["blocks"] * NVFP4_BYTES / 1e9 + p["head"] * NVFP4_BYTES / 1e9
    t_dq = wgb * DEQ_MS_PER_GB / 1e3 * (1.5 if recompute else 1.0)  # forward + backward (55 ms at 7B); a recompute is a third pass
    n_gdn = sum(1 for t in c["types"] if t == "gdn")
    t_gdn = 0.0
    if n_gdn:
        # per layer at 4 x 512 (1,536 chunks of 64): forward 14 ms, backward about 2.2x the forward without the solve (estimate), recompute = another forward
        chunks = n / 64.0 * c["nv"]  # (head, chunk) pairs; 1,536 at 4 x 512
        fwd = gdn_core_ms if gdn_core_ms is not None else sum(GDN_FWD_MS.values()) * chunks / 1536.0
        bwd = 2.2 * (fwd - GDN_FWD_MS[SOLVE_KEY] * chunks / 1536.0)
        t_gdn = n_gdn * (fwd + bwd + (fwd if recompute else 0.0)) / 1e3
    return dict(matmul_s=t_mm, dequant_s=t_dq, gdn_core_s=t_gdn, total_s=t_mm + t_dq + t_gdn, tokens=n)


def fmt_mem(m):
    return "\n".join(f"    {k:60s} {v:7.2f} GB" for k, v in m.items()) + f"\n    {'TOTAL':60s} {sum(m.values()):7.2f} GB"


def main():
    if "--7b" in sys.argv:
        c = qwen25_7b()
        for nb in (4, 8):
            m = memory(c, nb)
            print(f"{c['name']}, {nb} x 512:\n{fmt_mem(m)}")
        print(calibrate_7b())
        return
    c = qwen38_27b()
    p = params(c)
    print(f"parameters: blocks {p['blocks']/1e9:.3f} B (gdn linears {p['gdn']/1e9:.3f}, attention {p['full']/1e9:.3f}, mlp {p['mlp']/1e9:.3f}), embedding {p['embed']/1e9:.3f}, head {p['head']/1e9:.3f}, total {p['total']/1e9:.3f} B")
    print(f"LoRA r16 on all linears but b, a: {lora_params(c)/1e6:.1f} M; with b, a: {lora_params(c, skip=())/1e6:.1f} M")
    for kind in ("gdn", "full"):
        L, tot = act_bytes_per_token(c, kind)
        print(f"{kind} layer keeps {tot/1e3:.1f} KB a token (+ {probs_bytes_per_token(c, kind, 512)/1e3:.1f} KB of probabilities at T = 512)")
    for rc in (False, True):
        for nb in (1, 2, 4):
            m = memory(c, nb, recompute=rc)
            s = step_seconds(c, nb, recompute=rc)
            print(f"\n{nb} x 512, {'recompute' if rc else 'no recompute'}: {sum(m.values()):.1f} GB, step {s['total_s']:.1f} s ({s['tokens']/s['total_s']:.0f} tokens/s; matmul {s['matmul_s']:.1f}, dequant {s['dequant_s']:.2f}, gated-delta core {s['gdn_core_s']:.2f})")
            if "-v" in sys.argv:
                print(fmt_mem(m))


if __name__ == "__main__":
    main()
