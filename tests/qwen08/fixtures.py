#!/usr/bin/env python3
"""Tiny Qwen3.5-style checkpoints for `tests/qwen08_load.sh` (nothing is committed;
this writes them).

    python3 tests/qwen08/fixtures.py <dir>

Writes `<dir>/ok/` (a `config.json` and a `model.safetensors` with the layout of
the 0.8B model at toy sizes: 4 layers, the last full attention, a gated
`q_proj`, the F32 `A_log` and `linear_attn.norm.weight`, a tied head, and a
vision and an mtp tensor that are not loaded) and `<dir>/reject/<rule>[-variant]/`,
one directory a refusal of `src/qwen08_load.cho`, named for the rule it must be
refused with. The embedding holds every one of the 65,536 bf16 bit patterns, so
the f16 narrowing is compared with Python's IEEE half on all of them.
"""
import copy
import json
import os
import struct
import sys

CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "tie_word_embeddings": True,
    "text_config": {
        "attn_output_gate": True,
        "full_attention_interval": 4,
        "head_dim": 4,
        "hidden_size": 8,
        "intermediate_size": 16,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_value_head_dim": 4,
        "num_attention_heads": 2,
        "num_hidden_layers": 4,
        "num_key_value_heads": 1,
        "tie_word_embeddings": True,
        "vocab_size": 8192,
    },
}
P = "model.language_model."
H, I, V = 8, 16, 8192


def layout(cfg=CONFIG, vocab=V):
    """[(name, dtype, shape)] for the text tower, then two tensors that are not loaded."""
    out = [(P + "embed_tokens.weight", "BF16", [vocab, H])]
    for l in range(4):
        q = P + f"layers.{l}."
        out += [(q + "input_layernorm.weight", "BF16", [H]), (q + "post_attention_layernorm.weight", "BF16", [H]),
                (q + "mlp.gate_proj.weight", "BF16", [I, H]), (q + "mlp.up_proj.weight", "BF16", [I, H]),
                (q + "mlp.down_proj.weight", "BF16", [H, I])]
        if l == 3:
            out += [(q + "self_attn.q_proj.weight", "BF16", [16, H]), (q + "self_attn.k_proj.weight", "BF16", [4, H]),
                    (q + "self_attn.v_proj.weight", "BF16", [4, H]), (q + "self_attn.o_proj.weight", "BF16", [H, 8]),
                    (q + "self_attn.q_norm.weight", "BF16", [4]), (q + "self_attn.k_norm.weight", "BF16", [4])]
        else:
            out += [(q + "linear_attn.A_log", "F32", [2]), (q + "linear_attn.norm.weight", "F32", [4]),
                    (q + "linear_attn.dt_bias", "BF16", [2]), (q + "linear_attn.conv1d.weight", "BF16", [24, 1, 4]),
                    (q + "linear_attn.in_proj_qkv.weight", "BF16", [24, H]), (q + "linear_attn.in_proj_z.weight", "BF16", [8, H]),
                    (q + "linear_attn.in_proj_a.weight", "BF16", [2, H]), (q + "linear_attn.in_proj_b.weight", "BF16", [2, H]),
                    (q + "linear_attn.out_proj.weight", "BF16", [H, 8])]
    out.append((P + "norm.weight", "BF16", [H]))
    out += [("model.visual.patch_embed.weight", "BF16", [3]), ("mtp.fc.weight", "BF16", [2, H])]
    return out


SIZE = {"BF16": 2, "F32": 4}
state = 20261105


def rnd():
    global state
    state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return state >> 33


def payload(name, dtype, shape):
    n = 1
    for d in shape:
        n *= d
    if name == P + "embed_tokens.weight":
        return b"".join(struct.pack("<H", i & 65535) for i in range(n))
    if dtype == "BF16":
        # Mostly small magnitudes, so that the subnormal range is crossed.
        return b"".join(struct.pack("<H", (rnd() & 0x8000) | ((rnd() % 160) << 7) | (rnd() & 127)) for _ in range(n))
    return b"".join(struct.pack("<I", rnd() & 0xFFFFFFFF) for _ in range(n))


def safetensors(specs, data=None):
    """specs [(name, dtype, shape)]; the file bytes."""
    header, blob = {}, b""
    for i, (name, dtype, shape) in enumerate(specs):
        b = payload(name, dtype, shape) if data is None or name not in data else data[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(blob), len(blob) + len(b)]}
        blob += b
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((-len(h)) % 8)
    return struct.pack("<Q", len(h)) + h + blob


def write(d, cfg, specs, raw_config=None, data=None):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        f.write(raw_config if raw_config is not None else json.dumps(cfg, indent=2))
    if specs is not None:
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(safetensors(specs, data))


def payloads(specs):
    return {name: payload(name, dtype, shape) for name, dtype, shape in specs}


def bf16_matrix_edit(raw, rows, cols, columns, keep_rows=()):
    """`raw` (the bytes of a BF16 [rows, cols] matrix) with every row of the given columns replaced by
    other bf16 bits, except the rows in `keep_rows`, which keep their value (an entry the edit
    leaves alone inside a replaced column)."""
    b = bytearray(raw)
    for c in columns:
        for r in range(rows):
            if r in keep_rows:
                continue
            at = 2 * (r * cols + c)
            v = (struct.unpack_from("<H", b, at)[0] ^ (1 + rnd() % 65534)) & 0xFFFF
            struct.pack_into("<H", b, at, v)
    return bytes(b)


def main(root):
    base = layout()
    pl = payloads(base)
    write(os.path.join(root, "ok"), CONFIG, base, data=pl)
    # The edited toy (T11): two down_proj matrices of the 0.8B layout's kind ([hidden, inter] = [8, 16]) with
    # columns replaced, one of them leaving a row of its replaced columns as it was; the rest is the same bytes.
    ed = dict(pl)
    d1, d3 = P + "layers.1.mlp.down_proj.weight", P + "layers.3.mlp.down_proj.weight"
    ed[d1] = bf16_matrix_edit(pl[d1], H, I, [0, 5, 6, 15], keep_rows=(2,))
    ed[d3] = bf16_matrix_edit(pl[d3], H, I, [7])
    write(os.path.join(root, "edited"), CONFIG, base, data=ed)
    # A checkpoint whose embedding spans several of the patch checker's chunks (1,048,576 values), edited in two columns.
    big_cfg = copy.deepcopy(CONFIG)
    big_cfg["text_config"]["vocab_size"] = 131072
    big = layout(vocab=131072)
    bpl = payloads(big)
    bed = dict(bpl)
    emb = P + "embed_tokens.weight"
    bed[emb] = bf16_matrix_edit(bpl[emb], 131072, H, [1, 6])
    write(os.path.join(root, "big"), big_cfg, big, data=bpl)
    write(os.path.join(root, "big-edited"), big_cfg, big, data=bed)
    r = os.path.join(root, "reject")

    def cfg_with(f):
        c = copy.deepcopy(CONFIG)
        f(c["text_config"], c)
        return c

    def drop(name):
        return [s for s in base if s[0] != name]

    def swap(name, dtype=None, shape=None):
        return [(n, dtype or d, shape or s) if n == name else (n, d, s) for n, d, s in base]

    # No weights; no config.
    write(os.path.join(r, "q08-io"), CONFIG, None)
    os.makedirs(os.path.join(r, "q08-io-config"), exist_ok=True)
    with open(os.path.join(r, "q08-io-config", "model.safetensors"), "wb") as f:
        f.write(safetensors(base))
    write(os.path.join(r, "q08-config-json"), None, base, raw_config="{")
    write(os.path.join(r, "q08-config"), cfg_with(lambda t, c: t.pop("hidden_size")), base)
    write(os.path.join(r, "q08-config-zero"), cfg_with(lambda t, c: t.update(hidden_size=0)), base)
    write(os.path.join(r, "q08-config-huge"), cfg_with(lambda t, c: t.update(intermediate_size=1 << 60)), base)
    write(os.path.join(r, "q08-config-text"), cfg_with(lambda t, c: c.pop("text_config")), base)
    write(os.path.join(r, "q08-config-string"), cfg_with(lambda t, c: t.update(num_attention_heads="two")), base)
    write(os.path.join(r, "q08-layers"), cfg_with(lambda t, c: t.update(layer_types=["linear_attention"] * 4)), base)
    write(os.path.join(r, "q08-layers-count"), cfg_with(lambda t, c: t.update(layer_types=["linear_attention"])), base)
    write(os.path.join(r, "q08-unsupported"), cfg_with(lambda t, c: t.update(tie_word_embeddings=False)), base)
    write(os.path.join(r, "q08-missing"), CONFIG, drop(P + "layers.2.mlp.up_proj.weight"))
    write(os.path.join(r, "q08-missing-final-norm"), CONFIG, drop(P + "norm.weight"))
    write(os.path.join(r, "q08-dtype"), CONFIG, swap(P + "layers.0.linear_attn.A_log", dtype="BF16"))
    write(os.path.join(r, "q08-dtype-f32-for-bf16"), CONFIG, swap(P + "layers.1.mlp.down_proj.weight", dtype="F32"))
    write(os.path.join(r, "q08-shape"), CONFIG, swap(P + "layers.2.linear_attn.in_proj_qkv.weight", shape=[8, 24]))
    write(os.path.join(r, "q08-shape-embed"), CONFIG, swap(P + "embed_tokens.weight", shape=[V // 2, H * 2]))
    write(os.path.join(r, "q08-unexpected"), CONFIG, base + [(P + "layers.0.extra.weight", "BF16", [2])])
    write(os.path.join(r, "q08-unexpected-prefix"), CONFIG, base + [("model.other.weight", "BF16", [2])])
    write(os.path.join(r, "q08-untied"), CONFIG, base + [("lm_head.weight", "BF16", [V, H])])
    # The refusals of safetensors.cho pass through with their own tags.
    d = os.path.join(r, "st-truncated")
    write(d, CONFIG, base)
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.write(b"abc")


if __name__ == "__main__":
    main(sys.argv[1])
