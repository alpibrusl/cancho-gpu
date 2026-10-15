# Qwen3.8-27B in `.lx`

> **Status: every kernel of the model's decode step is written in `.lx`, held to the
> reference compiler and to a plain-Python model of what it computes; the runner that
> strings them together on a Metal GPU is next (§4).** Nothing here has run the
> model yet.

The model is `qwen3.8:27b-mlx`: 64 layers, hidden 5120, NVFP4 weights (4-bit E2M1 codes,
an E4M3 scale per 16 values, one f32 scale per tensor), `model_type: qwen3_5`. It
already runs here under Ollama and MLX-LM (`docs/benchmark.md`); the reference this
port is held to is `scripts/qwen_ref.py` of the reference implementation (f32 numpy over the
same Ollama blobs, whose golden fixture is `crates/lex-rt/tests/data/qwen35_27b_golden.txt`)
and, for tokens, the reference runtime itself (greedy, `The capital of France is`:
` Paris`, `.`, newline, `The` ... at 25 tokens a second on this machine).

## 1. The layers

Every fourth layer is **grouped attention** (16 of them): 24 query heads of 256 and 4
key/value heads, RoPE over the first 64 dimensions of each head (theta 10^7), a per-head
RMSNorm of q and k, and a sigmoid gate that `q_proj` carries beside the queries (its output
is 24 x [q 256, gate 256]). The other 48 are **gated delta** layers: a four-tap causal
depthwise convolution over the `[q, k, v]` projection, then a recurrent state
`S[48 heads, 128, 128]` (f32, 3 MB a layer whatever the context) updated by the delta rule

    S = S g;   S += (v - S k) beta k^T;   y = S q

then a per-head RMSNorm of `y` gated by `silu(z)`. Every layer ends in the same SwiGLU MLP
(5120, 17408, 5120). Norm weights are stored as deltas from 1 (`x * (1 + w)`), except the
gated one; the host adds 1 when it loads them.

## 2. The kernels

| Kernel | What | Shapes |
|---|---|---|
| `matvec_fp4_rows.lx`, `matvec_fp4_res.lx` | `x w^T` over NVFP4 weights, and added to the residual | `n=17408 k=5120` (gate, up), `n=5120 k=17408` (down), `n=248320 k=5120` (head), ... |
| `rmsnorm.lx` | the layer norms | `n=5120` |
| `rmsnorm_rows.lx` | the per-head norm of q and k | `r=24` and `4`, `n=256` |
| `rope.lx`, `kv_write.lx`, `attention.lx`, `attn_gate.lx` | partial RoPE, the cache write, attention, the output gate | `h=32 p=192`; `nkv=4 d=256`; `g=6` |
| `conv_step.lx` | the four-tap convolution and silu, rewriting its window | `c=10240` |
| `qk_norm.lx` | the queries' and keys' RMS normalisation and scaling | `nk=16 rows=80` |
| `gates.lx` | `beta = sigmoid(b)`, `g = exp(-A softplus(a + dt_bias))` | `h=48` |
| `delta_step.lx` | the delta rule for the three value heads of each key head | `nk=16` |
| `gated_norm.lx` | the norm gated by `silu(z)` | `h=48 n=128` |
| `swiglu.lx` | `silu(gate) * up` | `n=17408` |

Two shapes needed finding. The delta state is a 128 x 128 f32 tile, 64 KiB, which does
not fit a threadgroup (32 KiB on Metal, 48 KiB on CUDA): but the rule is independent for
each of the 128 value rows, so `delta_step` takes `br` rows of each of the three heads in
a threadgroup (`br=16`, 16 x 8 threadgroups), and the compilers refused the whole-state
version identically. And the surface has no integer division of a grid index, so the three
value heads that share a key head are written out, the shared `q` and `k` rows borrowed
twice and moved on the third use.

## 3. How they are held to something

* The 200 cases of `tests/cases.txt` now include these kernels at the model's sizes (emit
  only) and small runs: CUDA C and Metal **byte-identical with the reference compiler's**
  on every one, and its interpreter's `--run` summaries identical, including the f32
  order of the recurrent update.
* `tests/kernels.py` (in the gate) runs each small case through the interpreter and a
  plain-Python model written from the equations above and from the NVFP4 definition
  (`e2m1 * e4m3 * global`, the low nibble first): 20 more checks, all agree, on both
  targets' summaries.

## 4. What is left

1. **The weights**: the Ollama store (a manifest of digests, one safetensors blob a tensor,
   `U32` codes with a `.scale` and a `.global_scale`), read with `file_pread` and uploaded as
   the codes' bytes, the scale bytes and the tensor's scale spread over its rows; the
   BF16 `in_proj_a`/`in_proj_b`, convolution (transposed to tap-major), `A_log`
   (exponentiated), `dt_bias` and norms as f32; the embedding row dequantised on the host.
2. **The runner**: the per-layer launch sequence for both layer kinds, the states and
   caches, sampling (`device/run-main.cho` is the Qwen2 one).
3. **Checking it** against the reference's golden log-probabilities and the reference
   runtime's tokens, layer by layer where they disagree.
4. **Speed and context**: 14.5 GB read a token against the 5 to 9 ms a token of the 0.5B
   model's profile; and attention's one-tile scores cap the context at 1364 positions on Metal
   (`docs/kernels.md` §4).
