#!/usr/bin/env python3
"""gdn_ref.py against torch autograd through transformers' own qwen3_5 code in binary64 (hf64.py). Needs torch and transformers (>= 5.2).

    python3 examples/qwen35/hf_check.py

1. the delta rule: HF's torch_chunk_gated_delta_rule (patched to float64) forward and the gradients of q, k, v, g, beta, against chunk_fwd / chunk_bwd;
2. a whole Qwen3_5GatedDeltaNet layer: forward, dx and the gradient of all nine parameter tensors;
3. the same layer INSIDE a tiny random Qwen3_5ForCausalLM (4 layers: 3 linear, 1 full attention; vocabulary 64), the layer's input and output gradient
   captured by hooks during a real loss.backward(), so the specification is held to the inputs a trained network produces."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gdn_ref as G  # noqa: E402
import hf64  # noqa: E402

bad = 0


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def say(ok, text):
    global bad
    print(("ok   " if ok else "FAIL ") + text)
    bad += 0 if ok else 1


def tiny_config(m, layers=4, hidden=24, nk=2, nv=4, dk=8, dv=6):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    cfg = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=hidden, intermediate_size=40, num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_conv_kernel_dim=4, linear_key_head_dim=dk, linear_value_head_dim=dv, linear_num_key_heads=nk, linear_num_value_heads=nv,
        layer_types=["linear_attention"] * (layers - 1) + ["full_attention"], max_position_embeddings=256, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0], "mrope_interleaved": True},
    )
    cfg.attn_output_gate = True
    return cfg


def layer_params_from(mod):
    sd = {k: v.detach().double().numpy() for k, v in mod.state_dict().items()}
    nk, nv, dk, dv, kern = mod.num_k_heads, mod.num_v_heads, mod.head_k_dim, mod.head_v_dim, mod.conv_kernel_size
    return dict(Wqkv=sd["in_proj_qkv.weight"], Wz=sd["in_proj_z.weight"], Wb=sd["in_proj_b.weight"], Wa=sd["in_proj_a.weight"],
                conv=sd["conv1d.weight"][:, 0, :], A_log=sd["A_log"], dt_bias=sd["dt_bias"], norm=sd["norm.weight"], Wout=sd["out_proj.weight"],
                dims=(nk, nv, dk, dv, kern))


GRADNAMES = {"Wqkv": "in_proj_qkv.weight", "Wz": "in_proj_z.weight", "Wb": "in_proj_b.weight", "Wa": "in_proj_a.weight", "A_log": "A_log",
             "dt_bias": "dt_bias", "norm": "norm.weight", "Wout": "out_proj.weight"}


def check_rule(m):
    import torch
    rng = np.random.RandomState(3)
    H, T, dk, dv = 3, 150, 8, 6
    q = rng.normal(size=(H, T, dk)); k = rng.normal(size=(H, T, dk)); v = rng.normal(size=(H, T, dv))
    k /= np.linalg.norm(k, axis=-1, keepdims=True); q /= np.linalg.norm(q, axis=-1, keepdims=True)
    g = -rng.uniform(0.01, 0.6, size=(H, T)); beta = 1 / (1 + np.exp(-rng.normal(size=(H, T))))
    W = rng.normal(size=(H, T, dv))
    # HF's layout is [batch, T, heads, dim] and [batch, T, heads]
    ts = {n: torch.tensor((a.transpose(1, 0, 2) if a.ndim == 3 else a.T.copy())[None], dtype=torch.float64, requires_grad=True)
          for n, a in dict(q=q, k=k, v=v, g=g, beta=beta).items()}
    o, _ = m.torch_chunk_gated_delta_rule(ts["q"], ts["k"], ts["v"], g=ts["g"], beta=ts["beta"], chunk_size=64)
    (o * torch.tensor(W.transpose(1, 0, 2)[None])).sum().backward()
    o_np, _, cache = G.chunk_fwd(q, k, v, g, beta, 64, "loop")
    say(rel(o_np.transpose(1, 0, 2), o.detach().numpy()[0]) < 1e-12, f"rule forward vs HF float64: {rel(o_np.transpose(1, 0, 2), o.detach().numpy()[0]):.1e}")
    gr = G.chunk_bwd(cache, W)
    for n, key in (("q", "dq"), ("k", "dk"), ("v", "dv"), ("g", "dg"), ("beta", "dbeta")):
        a = ts[n].grad.numpy()[0]
        a = a.transpose(1, 0, 2) if a.ndim == 3 else a.T
        say(rel(gr[key], a) < 1e-10, f"rule {key} vs autograd: {rel(gr[key], a):.1e}")


def check_layer(m):
    import torch
    cfg = tiny_config(m)
    torch.manual_seed(0)
    layer = m.Qwen3_5GatedDeltaNet(cfg, 0).double()
    with torch.no_grad():
        for n_, p_ in layer.named_parameters():
            p_.copy_(torch.randn_like(p_) * 0.3 if n_ not in ("A_log", "dt_bias", "norm.weight") else p_)
        layer.A_log.copy_(torch.log(torch.rand(layer.A_log.shape, dtype=torch.float64) * 3 + 0.5))
        layer.dt_bias.copy_(torch.randn(layer.dt_bias.shape, dtype=torch.float64) * 0.3 + 0.5)
        layer.norm.weight.copy_(1 + 0.2 * torch.randn(layer.norm.weight.shape, dtype=torch.float64))
    x = torch.randn(1, 150, cfg.hidden_size, dtype=torch.float64, requires_grad=True)
    Wy = torch.randn(1, 150, cfg.hidden_size, dtype=torch.float64)
    y = layer(x)
    (y * Wy).sum().backward()
    p = layer_params_from(layer)
    yn, s = G.layer_fwd(p, x.detach().numpy()[0], 64, "loop")
    say(rel(yn, y.detach().numpy()[0]) < 1e-11, f"layer forward vs HF float64 (T=150, 3 chunks): {rel(yn, y.detach().numpy()[0]):.1e}")
    dx, gr = G.layer_bwd(p, s, Wy.numpy()[0])
    say(rel(dx, x.grad.numpy()[0]) < 1e-10, f"layer dx vs autograd: {rel(dx, x.grad.numpy()[0]):.1e}")
    for n, hn in GRADNAMES.items():
        a = dict(layer.named_parameters())[hn].grad.numpy()
        say(rel(gr[n], a) < 1e-10, f"layer d{n} vs autograd: {rel(gr[n], a):.1e}")
    a = layer.conv1d.weight.grad.numpy()[:, 0, :]
    say(rel(gr["conv"], a) < 1e-10, f"layer dconv vs autograd: {rel(gr['conv'], a):.1e}")


def check_in_model(m):
    import torch
    cfg = tiny_config(m)
    torch.manual_seed(1)
    model = m.Qwen3_5ForCausalLM(cfg).double()
    with torch.no_grad():
        for n_, p_ in model.named_parameters():
            if "A_log" in n_:
                p_.copy_(torch.log(torch.rand(p_.shape, dtype=torch.float64) * 3 + 0.5))
            elif "dt_bias" in n_:
                p_.copy_(torch.randn(p_.shape, dtype=torch.float64) * 0.3 + 0.5)
    cap = {}
    lay = model.model.layers[1].linear_attn
    lay.register_forward_hook(lambda mod, i, o: (o.register_hook(lambda g: cap.update(dy=g.detach().clone())), None)[1])
    lay.register_forward_pre_hook(lambda mod, a, kw: cap.update(x=(a[0] if a else kw["hidden_states"]).detach().clone()), with_kwargs=True)
    ids = torch.randint(0, 64, (2, 70))
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    p = layer_params_from(lay)
    worst = 0.0
    for b in range(2):
        yn, s = G.layer_fwd(p, cap["x"].numpy()[b], 64, "loop")
        dx, gr = G.layer_bwd(p, s, cap["dy"].numpy()[b])
        if b == 0:
            tot = {k: gr[k].copy() for k in gr}
        else:
            for k in gr:
                tot[k] += gr[k]
    # parameter gradients of the layer inside the model are summed over the batch
    for n, hn in GRADNAMES.items():
        a = dict(lay.named_parameters())[hn].grad.numpy()
        worst = max(worst, rel(tot[n], a))
    a = lay.conv1d.weight.grad.numpy()[:, 0, :]
    worst = max(worst, rel(tot["conv"], a))
    say(worst < 1e-9, f"layer 1 of a tiny random Qwen3_5ForCausalLM (loss {out.loss.item():.4f}): all nine parameter gradients from the captured dy, worst {worst:.1e}")


def main():
    m = hf64.load()
    check_rule(m)
    check_layer(m)
    check_in_model(m)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
