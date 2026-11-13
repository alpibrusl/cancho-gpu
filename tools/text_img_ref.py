#!/usr/bin/env python3
"""D9c of docs/diffusion.md: the text encoder of Qwen-Image-2.1-Turbo reading an image, in float64: the language model of Qwen3-VL over a prompt that carries a reference image's vision tokens.

    python tools/text_img_ref.py OUTDIR [--size N | HxW] [--prompt TEXT] [--torch32]

Composes tools/vit_ref.py (the vision tower, D9a) and tools/text_ref.py (the 36 layers, D5a), and adds what an image needs, written from `modeling_qwen3_vl.py`:

* the token ids come from the real processor (the pipeline's `prompt_template_ti2i`: the `<|image_pad|>` of the template becomes one token per 2 x 2 patches); everything after is computed here;
* the input embeddings are the token embeddings with the image-pad rows replaced by the vision tower's merged embeddings, in order;
* **mRoPE**: the position of a text token is its running index on all three axes; an image block's tokens get the frame position `start`, the row `start + r` and the column `start + c`, and the text after
  it resumes at `start + max(rows, columns)`; per rotary frequency j of the 64 (theta 5e6) the angle is position * inv_freq[j] with the position taken from the height axis for j = 1, 4, ..., 58, from the width axis
  for j = 2, 5, ..., 59 and from the frame axis otherwise (sections 24, 20, 20, interleaved);
* **deepstack**: after layers 0, 1 and 2 the three deepstack maps are added to the residual stream at the image-pad rows.

The result is what the pipeline hands the transformer: the residual stream after the 36 layers, before the final norm, from row `drop_idx` (14) on, with a mask of the image-pad rows.
Writes OUTDIR/{ids,positions,embeds_in,after_f64,out_f64,pad_mask}.npy; `--torch32` also runs transformers' own model in float32 on the CPU (35 GB) on the processor's pixels and prints the difference.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_ref as tr  # noqa: E402
import vit_ref as vr  # noqa: E402

SYS = "Comprehend and analyze the provided prompt."
TEMPLATE = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{{}}<|im_end|>\n<|im_start|>assistant\n"
IMG_PAD = 151655
SECTIONS = (24, 20, 20)


def positions(types, grid_hw, merge=2):
    """[3, N] mRoPE positions of one sequence; `types` is 0 for text and 1 for image tokens, `grid_hw` the image's patch grid."""
    out, cur, i = [], 0, 0
    while i < len(types):
        j = i
        while j < len(types) and types[j] == types[i]:
            j += 1
        if types[i] == 0:
            out.append(np.arange(cur, cur + j - i)[None].repeat(3, 0))
            cur += j - i
        else:
            gh, gw = grid_hw[0] // merge, grid_hw[1] // merge
            rr, cc = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
            out.append(np.stack([np.full(gh * gw, cur), rr.reshape(-1) + cur, cc.reshape(-1) + cur]))
            cur += max(gh, gw)
        i = j
    return np.concatenate(out, 1)


def tables(pos):
    """cos, sin [N, 64] from [3, N] positions: the interleaved mRoPE of Qwen3-VL's text model."""
    inv = 1.0 / tr.THETA ** (np.arange(0, tr.HD, 2) / tr.HD)
    axis = np.zeros(64, dtype=int)  # which axis each frequency reads: 0 frame, 1 height, 2 width
    for ax in (1, 2):
        axis[ax:SECTIONS[ax] * 3:3] = ax
    p = pos[axis, :].T  # [N, 64]
    ang = p * inv[None]
    return np.cos(ang), np.sin(ang)


def build(prompt, img):
    """The processor's ids for the template with this image, the patch grid, and the vision tower's outputs in float64."""
    from PIL import Image
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(vr.CKPT + "processor")
    pil = Image.fromarray(img.astype(np.uint8))
    o = proc(text=[TEMPLATE.format(prompt)], images=[pil], padding=True, return_tensors="pt")
    ids = o["input_ids"][0].numpy().astype(int)
    grid = o["image_grid_thw"][0].tolist()
    return ids, grid, o


def main():
    out = sys.argv[1]
    sz = sys.argv[sys.argv.index("--size") + 1] if "--size" in sys.argv else "256x384"
    n, m = (int(a) for a in (sz.split("x") if "x" in sz else (sz, sz)))
    prompt = sys.argv[sys.argv.index("--prompt") + 1] if "--prompt" in sys.argv else "Make the sky purple."
    img = np.round(vr.synthetic(n, m))
    ids, grid, o = build(prompt, img)
    assert grid[1:] == [n // 16, m // 16], "the processor resized the image: %s" % grid
    merged, deep, _, _ = vr.tower(img)
    types = (ids == IMG_PAD).astype(int)
    assert types.sum() == merged.shape[0], (types.sum(), merged.shape)
    pos = positions(types, grid[1:])
    cos, sin = tables(pos)
    emb = tr._f.get_slice("model.language_model.embed_tokens.weight")
    x = np.stack([emb[int(t):int(t) + 1][0].double().numpy() for t in ids])
    pad = np.nonzero(types)[0]
    x[pad] = merged
    x0 = x.copy()
    after, peaks = [], []
    for i in range(tr.NL):
        x = tr.layer(x, i, cos, sin, {})
        if i < 3:
            x[pad] += deep[i]
        after.append(x.astype(np.float32))
    drop = 14
    os.makedirs(out, exist_ok=True)
    np.save(out + "/ids.npy", ids)
    np.save(out + "/positions.npy", pos)
    np.save(out + "/embeds_in.npy", x0.astype(np.float32))
    np.save(out + "/after_f64.npy", np.stack(after))
    np.save(out + "/out_f64.npy", x[drop:])
    np.save(out + "/pad_mask.npy", types[drop:].astype(bool))
    np.save(out + "/image_u8.npy", img.astype(np.uint8))
    print("%d tokens (%d image pads, %d text), grid %s; output rows %d (from %d); rms %.4g" % (len(ids), len(pad), len(ids) - len(pad), grid, len(ids) - drop, drop, np.sqrt((x[drop:] ** 2).mean())))
    if "--torch32" in sys.argv:
        import torch
        from transformers import Qwen3VLForConditionalGeneration
        mdl = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", subfolder="text_encoder", torch_dtype=torch.float32, low_cpu_mem_usage=True)
        lm = getattr(mdl.model, "language_model", mdl.model)
        h = lm.norm.register_forward_hook(lambda mod, a, out_: a[0])
        kw = {k: o[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids") if k in o}
        kw["pixel_values"] = kw["pixel_values"].float()
        with torch.no_grad():
            hs = mdl(**kw, output_hidden_states=True).hidden_states[-1][0].double().numpy()
        h.remove()
        d = np.abs(x[drop:] - hs[drop:])
        rms = float(np.sqrt((x[drop:] ** 2).mean()))
        txt = ~types[drop:].astype(bool)
        print("float64 vs torch float32 (CPU): mean abs error %.3g (%.3g of the rms), max %.3g; on the text rows only %.3g" % (d.mean(), d.mean() / rms, d.max(), np.abs(x[drop:][txt] - hs[drop:][txt]).mean()))


if __name__ == "__main__":
    main()
