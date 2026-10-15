# Running a model: `cancho-gpu-run`

> **Status: Qwen2.5-0.5B-Instruct runs on a Metal GPU, written in cancho, and
> agrees with Hugging Face transformers; Qwen3.8-27B does not run yet (§6).**

```sh
CANCHO=$CANCHO scripts/build-run.sh
build/cancho-gpu-run <model dir> tests/lx "The capital of France is" 12
# " Paris. It is the largest city in Europe and the third"
```

A model dir is a Hugging Face checkpoint of the **Qwen2** architecture
(`config.json`, `tokenizer.json`, `model.safetensors`, bf16). Arguments after
the new-token count: `--trace` (print the residual stream's first values at
chosen points and the five largest logits at every sampled position), then
`temperature top_p top_k seed` for sampling (greedy by default).

## 1. What is on the GPU, and what is not

Everything a token costs but four small things runs as `.lx` kernels, compiled
by this repository's compiler at start-up for `apple-m-series` and launched
through the Metal shim (`docs/device.md`). Per layer, in order:

| Kernel | What | Constants (Qwen2.5-0.5B) |
|---|---|---|
| `lex-gpu/rmsnorm.lx` | RMSNorm | `n=896 eps=1e-6` |
| `matvec_bias.lx` | q, k and v projections with their biases | `n=896/128/128 k=896` |
| `rope.lx` | rotary embedding, full width (`p=0`) | `r=14` (q), `r=2` (k), `h=32` |
| `attention_d64.lx` | softmax attention, 14 query heads over 2 KV heads | `nkv=2 g=7 d=64 t=tmax=256` |
| `matvec_res.lx` | o_proj and down_proj, added to the residual stream | `n=896 k=896/4864` |
| `matvec_rows.lx` | gate and up projections; the tied output projection | `n=4864 k=896`; `n=151936` |
| `swiglu.lx` | `silu(gate) * up` | `n=4864` |

On the host, in cancho: the embedding row of the token (one `file_pread` of
1,792 bytes from the checkpoint), the rotary tables for the position
(`cos` and `sin` of `pos * theta^(-2j/64)`, computed in binary64 and rounded to
f32), the attention mask (opened one position at a time), the **write of the
new key and value into the cache** (a download of 512 bytes and two uploads; the
`kv_append.lx` kernel exists but its position is a launch constant, so using it
would recompile a kernel for every position), and the sampling of the logits
(`sampler.cho`). The weights go up once, bf16 widened to f32 (a shift: bf16 is
the top half of an f32) in 4M-value chunks read with `file_pread`, so 0.5 B
parameters are 2 GB of device memory and the 15 GB checkpoints this is aimed at
never pass through the pool whole.

The prompt is tokenized by `bpe.cho` and fed **one token at a time**, so
prefill costs what decoding does. There is no chat template.

## 2. How it is held to something

`scripts/model-check.sh <model dir>` (needs a Mac, the checkpoint and python3 with
torch and transformers; not in the gate) runs three prompts through
`cancho-gpu-run` and through the same checkpoint in **float32 on the CPU by
transformers** (`tests/model/reference.py`, greedy, no repetition penalty), and
requires the same tokens and, at every position, the same five largest logits to
within 1e-3. The reference is also run teacher-forced on the GPU run's own tokens,
so a near-tie that flips a token cannot hide a difference in the logits.

**Measured** (Apple M4 Max, 12 new tokens each): the three prompts
(`The capital of France is`, a Python `def`, `Once upon a time, ...`) give **the
same 12 tokens**, and the worst difference in a top-5 logit over every position is
**1e-4**.

The kernels are held to what they compute like the rest (`tests/kernels.py`, five
more references: a matrix-vector product with and without a bias or residual, SwiGLU,
attention for `d=64`) and to the reference compiler's output (13 more cases in
`tests/cases.txt`, 172 in all).

## 3. Speed, once

One run, one machine: **about 35 ms a token (28 tokens a second)** for
Qwen2.5-0.5B on an M4 Max, from the difference of two runs (2 and 130 new tokens:
1.06 s and 5.59 s, the first including loading 2 GB of weights, about 1 s). That
is far from what the hardware can do and the reasons are known: every launch waits
for the GPU, the cache write goes through the host, and the kernels the compiler
lowers read each element from device memory with no staging (`0 B of tiles`).
Nothing here has been tuned.

## 4. Limits

* The context is **256 positions**, a constant of the compiled kernels (the mask
  makes it a capacity), and the prompt plus the new tokens must fit.
* The architecture is Qwen2 only: dense, grouped-query attention with biases on
  q, k and v, RoPE over the whole head, RMSNorm, SwiGLU, tied embeddings.
* bf16 checkpoints only, widened to f32; no quantised format is read.
* Raw text in, no chat template, no stop on a stop string (it stops on
  `<|im_end|>` and `<|endoftext|>`).

## 5. What it needed from the rest of the repository

All of §2-§4 of `docs/runtime.md`: the safetensors header, the tokenizer, the
sampler; `fileio.read_at` (a positioned read of any length); a shim that can write
into a buffer at an offset and hold a few thousand of them (`lxg_upload_at`, the
buffer limit raised from 256: this model has 290 weight tensors).

## 6. Toward Qwen3.8-27B

What is missing, in the order it blocks:

1. **The weights' format.** The 15 GB Hugging Face checkpoint is MLX affine 4-bit
   (scales and biases per group), which the `.lx` surface has no dequantisation
   for; Ollama's `qwen3.8:27b-mlx` is NVFP4, one safetensors blob a tensor, which
   it does (`dequant_fp4`, `matvec_fp4.lx`). So: read the Ollama store (a manifest
   of digests) and a `matvec` over NVFP4 codes with their scales.
2. **48 of its 64 layers are linear attention** (a gated delta rule over a state
   matrix with a short causal convolution): kernels not yet written, and the state
   to keep between tokens.
3. **The 16 attention layers**: `rope.lx`, `kv_append.lx`, `attention.lx` and
   `attn_gate.lx` exist (`docs/kernels.md`); the per-head RMSNorm of q and k is
   `rmsnorm.lx` over heads; the "1 + w" norm convention and `A_log` handling are
   host-side changes at load.
4. **Context.** `attention.lx` holds a head's scores in one tile (1364 positions on
   Metal); longer needs split-KV, which cannot be written in the surface yet
   (`docs/kernels.md` §4).
5. **Memory and speed**: 15 to 18 GB of weights on the device, and the launch
   overhead of §3 multiplied by 64 layers.
