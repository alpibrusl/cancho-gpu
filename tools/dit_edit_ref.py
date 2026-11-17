#!/usr/bin/env python3
"""D9e of docs/diffusion.md: the transformer of Qwen-Image-2.1-Turbo with a condition image, one denoising step in float64, written from transformer_qwenimage21.py (`QwenImage21Transformer2DModel.forward` with
`img_shapes` of a condition image and the target, `img_mask`, no KV cache), held to the real transformer in torch float32 on the CPU.

    python tools/dit_edit_ref.py REF_DIR OUT_DIR [--torch32] [--t T]

REF_DIR is the output of `tools/text_img_ref.py` (the prompt embeddings after the language model: `out_f64.npy`, from `drop_idx` on, and `pad_mask.npy`, which of those rows are image pads) *for an image of the same size as the one
encoded to the condition latents*; the condition latents here are `tools/vae_enc_ref.py`'s mean, normalised, as the pipeline does, of that same synthetic image, and the target latents are fixed noise.

What an edit changes in the step (all [read] from the source, D9 design):
* the joint sequence is the language model's rows with **each image-pad row replaced by four slots**, in which the condition tokens (`img_in` of the condition latents, in raster order) are dropped in order, and the target's
  tokens appended at the end; the text rows go through `txt_in`;
* the rotary positions (`QwenImage21Rope`): text advances a shared position on all three axes, an image block holds the frame axis at the position reached so far (advancing by max(rows, columns) after it) and takes
  zero-centred row and column positions;
* the mask is `(q >= kv) or the same image block` (each image block bidirectional, text causal), no padding;
* the modulation rows: the target's tokens take the sampled timestep's, every other token the t = 0 row.
Writes OUT_DIR/{x0,rope,sel,allowed,cond,target,prompt,pad,meta}.npy, blocks_f64.npy ([32, S, 4096] float32) and out_f64.npy ([N_target, 64]); `--torch32` also runs the real transformer.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dit_block_ref as ref  # noqa: E402
import dit_step_ref as step  # noqa: E402
import vae_enc_ref as ve  # noqa: E402

AXES = (16, 56, 56)
THETA = 10000.0
SLOT = 4


def rope_angles(shapes, pad_exp):
    """QwenImage21Rope.forward in numpy: shapes [(h, w), ...] (condition images first, the target last), pad_exp the bool image-token mask over the joint sequence. Returns [S, 64, 2] (cos, sin)."""
    S = len(pad_exp)
    frame, hidx, widx = [], [], []
    cursor, position = 0, 0
    img_pos = [i for i in range(S) if pad_exp[i]]
    for (h, w) in shapes:
        start = next(i for i in range(cursor, S) if pad_exp[i])
        text_len = start - cursor
        frame.extend(range(position, position + text_len))
        position += text_len
        cursor = start + h * w
        frame.extend([position] * (h * w))
        if MUT != "no-advance":  # the cursor advances by max(h, w) after each image block: the next block's frame position
            position += max(h, w)
        hidx.extend(r for r in range(-(h - h // 2), h // 2) for _ in range(w))
        widx.extend(c for _ in range(h) for c in range(-(w - w // 2), w // 2))
    if cursor < S:
        frame.extend(range(position, position + S - cursor))
    frame = np.array(frame)
    hi, wi = frame.copy(), frame.copy()
    hi[img_pos], wi[img_pos] = hidx, widx
    ang = []
    for idx, dim in zip((frame, hi, wi), AXES):
        inv = 1.0 / THETA ** (np.arange(0, dim, 2) / dim)
        ang.append(idx[:, None] * inv[None])
    a = np.concatenate(ang, -1)  # [S, 64]
    return np.stack([np.cos(a), np.sin(a)], -1)


def assemble(prompt, pad, cond, target, t):
    """The joint sequence after the front end (float64): x0 [S, 4096], mod [2, 16384], temb, rope [S, 64, 2], sel [S] (target tokens), allowed [S, S], and the bookkeeping."""
    w = ref.load64
    tw = w("txt_in.text_norm.weight")
    tn = prompt / np.sqrt((prompt ** 2).mean(-1, keepdims=True) + 1e-6) * (tw + 1)
    txt = step.gelu_tanh(tn @ w("txt_in.in_layer.weight").T) @ w("txt_in.out_layer.weight").T  # [L, 4096]
    img = np.concatenate([cond, target], 0) @ w("img_in.weight").T  # [Nc + Nt, 4096]
    ntgt = len(target)
    img_mask = np.concatenate([pad, np.ones(ntgt // SLOT, dtype=bool)])
    repeats = np.where(img_mask, SLOT, 1)
    pad_exp = np.repeat(img_mask, repeats)
    S = len(pad_exp)
    x = np.zeros((S, 4096))
    text_pos = np.nonzero(~pad_exp)[0]
    x[text_pos] = txt[~pad]  # text rows in order
    x[np.nonzero(pad_exp)[0]] = img
    rope = rope_angles(SHAPES, pad_exp)
    block_len = [h * w for (h, w) in SHAPES]  # the conditions in order, then the target: any number of pictures (docs/two-images.md)
    assert sum(block_len) == len(cond) + ntgt, "SHAPES holds %d image tokens, the conditions and the target %d" % (sum(block_len), len(cond) + ntgt)
    blk = np.arange(len(block_len))
    if MUT == "one-block":  # the conditions as ONE block: they would attend to each other bidirectionally
        blk = np.where(blk < len(block_len) - 1, 0, 1)
    ids = np.full(S, -1)
    ids[np.nonzero(pad_exp)[0]] = np.repeat(blk, block_len)
    sel = np.zeros(S, dtype=bool)
    sel[np.nonzero(pad_exp)[0][-ntgt:]] = True
    idx = np.arange(S)
    allowed = (idx[:, None] >= idx[None, :]) | ((ids[:, None] == ids[None, :]) & (ids[:, None] >= 0))
    freqs = np.exp(-math.log(10000) * np.arange(128) / 128)
    args = np.array([t, 0.0])[:, None] * 1000.0 * freqs[None]
    emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    temb = step.silu(emb @ w("time_text_embed.timestep_embedder.linear_1.weight").T) @ w("time_text_embed.timestep_embedder.linear_2.weight").T
    mod = step.silu(temb) @ w("modulation.1.weight").T
    return x, mod, temb, rope, sel, allowed, pad_exp


SHAPES = None
MUT = None  # a deliberate bug the comparison must refuse: one-block, no-advance


def main_random():
    """docs/two-images.md check 1: the float64 joint step for TWO condition pictures of different shapes against torch's `QwenImage21Transformer2DModel`, on random inputs (the transformer does not read what the rows are).
    `python tools/dit_edit_ref.py --random [--mutate one-block|no-advance] [--torch32]`"""
    global SHAPES, MUT
    MUT = sys.argv[sys.argv.index("--mutate") + 1] if "--mutate" in sys.argv else None
    rng = np.random.default_rng(5)
    shapes_c = [(8, 6), (6, 10)]  # two pictures of different shape, in latent tokens
    tgt_hw = (8, 8)
    SHAPES = shapes_c + [tgt_hw]
    t = 0.978453
    runs = [8, 3, 6]  # text rows before the first picture, between the two, and after
    npad = [h * w // SLOT for h, w in shapes_c]
    pad = np.concatenate([np.zeros(runs[0], bool), np.ones(npad[0], bool), np.zeros(runs[1], bool), np.ones(npad[1], bool), np.zeros(runs[2], bool)])
    prompt = rng.normal(size=(len(pad), 4096))
    cond = rng.normal(size=(sum(h * w for h, w in shapes_c), 64))
    target = rng.normal(size=(tgt_hw[0] * tgt_hw[1], 64))
    x, mod, temb, rope, sel, allowed, pad_exp = assemble(prompt, pad, cond, target, t)
    print("joint sequence %d: %d text rows, conditions %s, target %s; mutant %s" % (len(x), (~pad_exp).sum(), shapes_c, tgt_hw, MUT), flush=True)
    xs = x.copy()
    for i in range(32):
        wts = {k: ref.load64("transformer_blocks.%d.%s" % (i, v)) for k, v in ref.BLOCK0.items()}
        xs = ref.block(xs, wts, mod, rope, sel, None, allowed)
    y = step.back(xs, temb, sel)[sel]
    print("float64 step: output rms %.4g" % np.sqrt((y ** 2).mean()))
    import torch
    from diffusers import QwenImage21Transformer2DModel
    m = QwenImage21Transformer2DModel.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", subfolder="transformer", torch_dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True).eval()
    img_mask = torch.tensor(np.concatenate([pad, np.ones(len(target) // SLOT, dtype=bool)]))[None]
    with torch.no_grad():
        o = m(hidden_states=torch.tensor(np.concatenate([cond, target], 0), dtype=torch.float32)[None], timestep=torch.tensor([t], dtype=torch.float32),
              encoder_hidden_states=torch.tensor(prompt, dtype=torch.float32)[None], encoder_hidden_states_mask=None, img_shapes=[[(1, hh, ww) for hh, ww in SHAPES]], img_mask=img_mask, return_dict=False)[0]
    ty = o[0, -len(target):].double().numpy()
    d = np.abs(y - ty)
    print("float64 vs torch float32 (the target rows): mean abs error %.3g (%.3g of the rms %.3g), max %.3g" % (d.mean(), d.mean() / np.sqrt((y ** 2).mean()), np.sqrt((y ** 2).mean()), d.max()))


def main():
    global SHAPES
    if "--random" in sys.argv:
        return main_random()
    refdir, out = sys.argv[1], sys.argv[2]
    t = float(sys.argv[sys.argv.index("--t") + 1]) if "--t" in sys.argv else 0.978453
    prompt = np.load(os.path.join(refdir, "out_f64.npy")).astype(np.float64)
    pad = np.load(os.path.join(refdir, "pad_mask.npy"))
    img_u8 = np.load(os.path.join(refdir, "image_u8.npy")).astype(np.float64)  # [H, W, 3] pixels
    h, w = img_u8.shape[:2]
    rgba = np.concatenate([img_u8.transpose(2, 0, 1) / 127.5 - 1.0, np.ones((1, h, w))], 0)
    mean = ve.encode(rgba)  # [64, h/16, w/16]
    cond = ve.normalise(mean).reshape(64, -1).T  # raster tokens [(h/16)(w/16), 64]
    gh, gw = h // 16, w // 16
    SHAPES = [(gh, gw), (gh, gw)]
    target = np.random.default_rng(11).normal(size=(gh * gw, 64))
    x, mod, temb, rope, sel, allowed, pad_exp = assemble(prompt, pad, cond, target, t)
    S = len(x)
    print("joint sequence %d tokens: %d text, %d condition, %d target" % (S, (~pad_exp).sum(), len(cond), len(target)), flush=True)
    os.makedirs(out, exist_ok=True)
    after = []
    xs = x.copy()
    for i in range(32):
        wts = {k: ref.load64("transformer_blocks.%d.%s" % (i, v)) for k, v in ref.BLOCK0.items()}
        xs = ref.block(xs, wts, mod, rope, sel, None, allowed)
        after.append(xs.astype(np.float32))
        if i % 8 == 7:
            print("block %2d done, residual max %.1f" % (i, np.abs(xs).max()), flush=True)
    y = step.back(xs, temb, sel)[sel]  # the target's rows: [N_target, 64]
    for n, a in (("x0", x), ("rope", rope), ("sel", sel), ("allowed", allowed), ("cond", cond), ("target", target), ("prompt", prompt), ("pad", pad), ("pad_exp", pad_exp), ("mod", mod), ("temb", temb), ("blocks_f64", np.stack(after)), ("out_f64", y)):
        np.save(os.path.join(out, n + ".npy"), a)
    np.save(os.path.join(out, "meta.npy"), np.array([t, gh, gw]))
    print("float64 step: output rms %.4g" % np.sqrt((y ** 2).mean()))
    if "--torch32" in sys.argv:
        import torch
        from diffusers import QwenImage21Transformer2DModel
        m = QwenImage21Transformer2DModel.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", subfolder="transformer", torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
        img_mask = torch.tensor(np.concatenate([pad, np.ones(len(target) // SLOT, dtype=bool)]))[None]
        with torch.no_grad():
            o = m(hidden_states=torch.tensor(np.concatenate([cond, target], 0), dtype=torch.float32)[None], timestep=torch.tensor([t], dtype=torch.float32),
                  encoder_hidden_states=torch.tensor(prompt, dtype=torch.float32)[None], encoder_hidden_states_mask=None, img_shapes=[[(1, hh, ww) for hh, ww in SHAPES]], img_mask=img_mask, return_dict=False)[0]
        ty = o[0, -len(target):].double().numpy()
        # `o` is over the joint sequence; its target rows are the last N_target
        d = np.abs(y - ty)
        print("float64 vs torch float32 (the target rows): mean abs error %.3g (%.3g of the rms %.3g), max %.3g" % (d.mean(), d.mean() / np.sqrt((y ** 2).mean()), np.sqrt((y ** 2).mean()), d.max()))


if __name__ == "__main__":
    main()
