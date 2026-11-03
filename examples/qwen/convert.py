#!/usr/bin/env python3
"""Qwen2 safetensors (bf16) -> the binary files of a session (docs/resident.md section 20, roadmap M7.1).

    python3 examples/qwen/convert.py [MODEL_DIR_OR_SAFETENSORS] OUTDIR [--quant nvfp4:G]

Matrices are written as little-endian f16 (bf16 narrows to f16 exactly for every value inside f16's range; the converter counts the values outside it and the ones f16 flushes to
zero, and refuses a weight that overflows), vectors (norm gains and biases) as f32. Per layer l (two digits), the files are
    lLL.ln1.f32 [d]   lLL.wqkv.f16 [d + 2 dkv, d] (q, k, v rows one after another)   lLL.bqkv.f32 [d + 2 dkv]   lLL.wo.f16 [d, d]
    lLL.ln2.f32 [d]   lLL.wgu.f16 [2 ff, d] (gate rows, then up rows)                lLL.wd.f16 [d, ff]
and norm.f32 [d], emb.f16 [vocabulary, d] (the input embedding and, tied, the output layer). manifest.json holds the configuration and the report.
With `--quant nvfp4:G` (docs/resident.md section 26) the four matrices of a block are written, from their f16 values, as the three files `NAME.q`, `NAME.s`, `NAME.g` of nvfp4.py instead of `NAME.f16`, and the manifest
says `"quant": {"format": "nvfp4", "group": G}`; a session then keeps 0.56 bytes a weight resident and dequantises a layer at a time."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

F16_MAX = 65504.0
F16_MIN_SUB = 2.0 ** -24


def load_tensors(path):
    """name -> float32 numpy array (bf16 is widened exactly)."""
    import torch
    from safetensors import safe_open
    out = {}
    with safe_open(path, "pt") as s:
        for k in s.keys():
            out[k] = s.get_tensor(k).to(torch.float32).numpy()
    return out


def find(arg):
    if os.path.isfile(arg):
        return arg, os.path.join(os.path.dirname(arg), "config.json")
    snaps = glob.glob(os.path.join(arg, "snapshots", "*")) or [arg]
    return os.path.join(snaps[0], "model.safetensors"), os.path.join(snaps[0], "config.json")


class Report:
    def __init__(self):
        self.n = 0
        self.overflow = 0
        self.flushed = 0
        self.subnormal = 0
        self.max_abs = 0.0

    def add(self, w):
        a = np.abs(w)
        np.seterr(over="ignore")  # the cast of an overflowing weight to f16 is counted, and then refused by the caller
        self.n += w.size
        self.max_abs = max(self.max_abs, float(a.max()))
        self.overflow += int((a > F16_MAX).sum())
        nz = a > 0
        self.flushed += int((nz & (a.astype(np.float16) == 0)).sum())
        self.subnormal += int(((a < 2.0 ** -14) & (a.astype(np.float16) != 0)).sum())

    def dict(self):
        return dict(values=self.n, max_abs=self.max_abs, overflow=self.overflow, flushed_to_zero=self.flushed, f16_subnormal=self.subnormal)


def convert(src, config, outdir, quant=None):
    cfg = json.load(open(config))
    T = load_tensors(src)
    os.makedirs(outdir, exist_ok=True)
    rep = Report()
    wrote = []

    qg = int(quant.partition(":")[2]) if quant and quant.partition(":")[2].isdigit() else 0
    if quant and (quant.partition(":")[0] != "nvfp4" or qg not in (16, 32, 64)):
        raise SystemExit(f"{quant}: only nvfp4:16, nvfp4:32 and nvfp4:64 can be read by a session")

    def put(name, arr, dt):
        arr = np.ascontiguousarray(arr)
        if dt == "f16":
            rep.add(arr)
            if (np.abs(arr) > F16_MAX).any():
                raise SystemExit(f"{name}: a weight does not fit f16 (largest {np.abs(arr).max()})")
            arr = arr.astype("<f2")
            if quant and name.split(".")[-1] in ("wqkv", "wo", "wgu", "wd"):
                import nvfp4
                for suffix, part in zip("qsg", nvfp4.pack_rows(arr, qg)):
                    part.tofile(os.path.join(outdir, f"{name}.{suffix}"))
                wrote.append(name)
                return
        else:
            arr = arr.astype("<f4")
        arr.tofile(os.path.join(outdir, f"{name}.{dt}"))
        wrote.append(name)

    put("emb", T["model.embed_tokens.weight"], "f16")
    put("norm", T["model.norm.weight"], "f32")
    for l in range(cfg["num_hidden_layers"]):
        q = f"model.layers.{l}."
        a = q + "self_attn."
        n = f"l{l:02d}."
        put(n + "ln1", T[q + "input_layernorm.weight"], "f32")
        put(n + "wqkv", np.concatenate([T[a + "q_proj.weight"], T[a + "k_proj.weight"], T[a + "v_proj.weight"]]), "f16")
        put(n + "bqkv", np.concatenate([T[a + "q_proj.bias"], T[a + "k_proj.bias"], T[a + "v_proj.bias"]]), "f32")
        put(n + "wo", T[a + "o_proj.weight"], "f16")
        put(n + "ln2", T[q + "post_attention_layernorm.weight"], "f32")
        put(n + "wgu", np.concatenate([T[q + "mlp.gate_proj.weight"], T[q + "mlp.up_proj.weight"]]), "f16")
        put(n + "wd", T[q + "mlp.down_proj.weight"], "f16")
    man = dict(config=cfg, report=rep.dict(), files=wrote)
    if quant:
        man["quant"] = dict(format="nvfp4", group=qg)
    json.dump(man, open(os.path.join(outdir, "manifest.json"), "w"), indent=1)
    return man


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    quant = None
    if "--quant" in args:
        i = args.index("--quant")
        quant = args[i + 1]
        del args[i:i + 2]
    default = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B")
    src, out = (default, args[0]) if len(args) == 1 else (args[0], args[1])
    s, c = find(src)
    m = convert(s, c, out, quant)
    r = m["report"]
    print(f"{len(m['files'])} files in {out}; {r['values']} matrix values, largest |w| {r['max_abs']:.4g} (f16 max {F16_MAX:g}): "
          f"{r['overflow']} overflow, {r['flushed_to_zero']} nonzero values flushed to zero (below 2^-25), {r['f16_subnormal']} become f16 subnormals")
