# Qwen3.8-27B in `.chx`

> **Status: Qwen3.8-27B runs on a Metal GPU in cancho and agrees with the reference
> implementation's golden log-probabilities (§4); about 22 tokens a second, against the
> reference runtime's 25 and Ollama's 42 on this machine (§5).**

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
| `matvec_fp4_rows.chx`, `matvec_fp4_res.chx` | `x w^T` over NVFP4 weights, and added to the residual | `n=17408 k=5120` (gate, up), `n=5120 k=17408` (down), `n=248320 k=5120` (head), ... |
| `rmsnorm.chx` | the layer norms | `n=5120` |
| `rmsnorm_rows.chx` | the per-head norm of q and k | `r=24` and `4`, `n=256` |
| `rope.chx`, `kv_row.chx`, `attention.chx`, `attn_gate.chx` | partial RoPE, the cache write, attention, the output gate | `h=32 p=192`; `nkv=4 d=256`; `g=6` |
| `conv_step.chx` | the four-tap convolution and silu, rewriting its window; 256 channels to a threadgroup | `c=10240` |
| `qk_norm.chx` | the queries' and keys' RMS normalisation and scaling | `nk=16 rows=80` |
| `gates.chx` | `beta = sigmoid(b)`, `g = exp(-A softplus(a + dt_bias))` | `h=48` |
| `delta_step.chx` | the delta rule for the three value heads of each key head | `nk=16` |
| `gated_norm.chx` | the norm gated by `silu(z)` | `h=48 n=128` |
| `swiglu.chx` | `silu(gate) * up` | `n=17408` |

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

## 4. The runner and how it is checked

`device/run35-main.cho` (`scripts/build-run35.sh`, `build/cancho-gpu-run35 <ollama store> tests/chx
"<prompt>" <new tokens> [--trace] [--timing]`) reads Ollama's manifest and runs the 64 layers as the
kernels above on a Metal GPU. The dimensions are the model's own, written as constants, so it runs this
model and no other.

* **Weights.** An NVFP4 tensor's codes and scales go up as they lie in the blob (16 MiB at a time with
  `file_pread`; the layout is the one the kernels read: low nibble first), and its one f32 scale is written
  once for each row. The bf16 norms are read as f32 with 1 added (all but the gated one), `A_log` is
  exponentiated, the convolution weight is transposed to tap-major, `in_proj_a` and `in_proj_b` go up as f16.
  The embedding is bf16: one row is read and widened per token. 14.6 GB are resident.
* **A token**: per layer a norm; for a gated-delta layer the `qkv`, `z`, `a`, `b` projections, the
  convolution (its window ping-pongs between two buffers), `qk_norm`, `gates`, the delta step (in place on
  the layer's 3 MB state, reading the convolution's value part through a buffer offset, the shim's
  `cgpu_arg_at`), the gated norm and `out_proj` added to the residual; for an attention layer `q`, `k`, `v`,
  the per-head norms, RoPE, the cache write, attention, the gate and `o_proj`; then the norm and the SwiGLU
  MLP. One command buffer a layer, committed without waiting; the CPU waits once, for the logits.
* **Checked** (`scripts/qwen35-check.sh <reference checkout>`, not in the gate: it needs the model and a Mac):
  against the reference's golden fixture for three prompts, **eight steps each give the same token, the
  same five tokens at the top, and logit gaps within 1e-4 of the reference's log-probability gaps** (its
  own f32 reference; the fixture's tolerance for the reference runtime is 0.02). On `The capital of France
  is`, twelve greedy tokens (` Paris`, `.`, newline, `The`, ` capital`, ` of`, ` Germany`, ` is`, ...) are the
  reference runtime's, token for token.
* **Found on the way**: the embedding is not quantised (it is bf16, 2.5 GB); an attention layer's
  weight list must be padded to the gated-delta layer's width so the MLP's slots line up (a trap, not a wrong
  answer, which is how it was found); and the `tokenizer.json` layer of the store is plain JSON, not a
  safetensors blob.

## 5. Speed

Steady decode on this machine (M4 Max, the weights read from the page cache): **41 ms a token, 24 tokens a
second**; the weights take 2.8 s to load. Prefill takes the prompt eight tokens at a time (below), about 75 tokens a second on a 49-token prompt (22 token by token, with `--single`). In
the same benchmark invocation (`docs/benchmark.md`) Ollama decodes the same weights at about 42 tokens a
second and MLX-LM (a different 4-bit quantisation) at about 28; the reference runtime measured 25 (39 ms a
token) when it was run for its tokens.

Where a token goes, measured by leaving kernels out of the real, pipelined run (`CANCHOGPU_SKIP=<name>`, which
makes the output wrong and the timing true; `CANCHOGPU_PROFILE=2` times each kernel alone, which adds about 60
us of launch overhead to each and so overstates small kernels): the NVFP4 matrix-vector products are about
31 of the 41 ms and already read at about 430 GB/s, against 546 for the chip; the rest is some 900 small
launches at about 10 us each. What brought it from 52 ms to 41, all in `.chx` and the runner:

* `kv_row.chx` writes only the new cache row (the position is the cache buffer's byte offset) instead of
  rewriting the whole cache (7.7 ms a token down to 0.1), and `conv_step.chx` spreads its channels over
  threadgroups (6.6 ms down to 0.3);
* `matvec_fp4_swiglu.chx` does the gate and up projections and the activation in one launch;
* the encoder dispatches concurrently and a barrier separates launches that depend on each other
  (`cgpu_overlap` marks the ones that do not: the q/k/v and the qkv/z/a/b projections and so on).

What did not help: threads or rows per threadgroup (nine schedules, the same time), and the emitter's
wide-load text on Metal (faster for one kernel alone, 450 against 340 GB/s, slower in the run). A
hand-written Metal kernel (`experimental/metal-bw/bw.mm`) reaches about 450 GB/s at best on the 17408x5120
shape, a few percent over the emitted text in the run, so the matvec is within about 20% of the chip's
bandwidth and no emitter change is planned for it (issue 24).

### Batched prefill

While at least eight prompt tokens remain before the last, the runner takes eight at once: the three
heavy products (`matvec_fp4_batch`, `_res_batch`, `_swiglu_batch`, `docs/kernels.md` section 6) read the weights once
for the eight, the per-row kernels (the norms, the output gate) run with the eight tokens' heads as more rows, and the
parts that are sequential in the token (the convolution and the delta state, the cache writes, attention, RoPE)
are launched once a token with buffer offsets. A chunk of eight takes about 55 ms (a decode step is 41), so
inside a chunk prefill runs at about 145 tokens a second; the last prompt tokens (up to eight, and the one whose
logits are wanted) go one at a time. `--single` turns it off; `scripts/qwen35-batch-check.sh` holds the two to
the same tokens and top-five logits (worst difference 1e-4 on a 49-token prompt). The speculative check (next section) needs a pass over two
tokens at about the cost of one, which these matrix-unit kernels do not give (an 8-token pass is ~63 ms), so it
has its own few-row kernels.

### Speculative decoding

A plain step is at the memory floor, so more tokens a second has to mean more tokens a pass. The checkpoint ships
a multi-token-prediction head (`mtp.*`, 239 MB: `fc` over `[norm(embedding), norm(hidden)]`, one attention layer of
the model's own shape, `mtp.norm`, and the model's `lm_head`). It reads the model's final-normed hidden state and
the token chosen from it, and drafts the token after that. `cancho-gpu-run35 --spec` (and the server, unless
`--no-spec`) runs rounds of: draft with the head; check `[last, draft, draft]` in one pass over the weights (three rows);
draw the token after `last` from the first row with the request's sampler, and, while each is the draft, the next
one from the following row for free (up to three tokens a round). Drawing from the model's own rows with the request's sampler leaves the
output distributed as if no draft had been made: at temperature 0 the tokens are exactly those of plain decoding.

* **Acceptance** (`--mtp-accept`, greedy, 64 tokens): 95.2% on prose, 96.8% on code; in `--spec` runs 46 of 49
  rounds on a question, 42 of 52 on random words. Two things about the head are inferred, not documented
  (`lex-gpu/docs/qwen.md`): every norm of it is stored as a delta from 1, and the embedding comes first in the
  concatenation into `fc`; the wrong choice accepts 0.4%.
* **The check costs 1.15 passes** for two tokens (47 ms against 41 for one): the few-row kernels
  (`matvec_fp4_*_few.chx`, 256 threads, 32 weight rows) read the weights once. Their schedule was chosen inside
  the running verify: one kernel timed alone pointed to 1024 threads, which made the verify 30% slower.
* **Rollback is a pointer swap.** The 48 delta layers' memory is a state updated in place, so the verify writes
  token 0's state and token 1's into two spare buffers and `commit` keeps one of the three (nothing copied); an
  attention layer needs nothing (positions are overwritten) and the convolution window already alternates between
  two buffers by position.
* **Equivalence:** `scripts/spec-check.sh` requires the same tokens as plain greedy decoding, in the same order,
  on seven prompts (96 tokens each: prose, code, a chat turn, random words); they are the same, at 1.28-1.46x the
  speed. `tests/serve_check.py` holds the server's speculative answers to the engine's.
* **Speed:** 33-35 tokens a second through the server (31 at temperature 1) against 24 without the head; a round
  is about 5 ms of draft, 47 of verify and 2.6 of sampling on the CPU for 1.9 tokens. The head's cache is warmed
  through prefill (without it acceptance falls from about 0.9 to about 0.5): a position costs about 1 ms more.
* **A second draft** (a chain: the head again, on its own final-normed output and the token it just drafted) is
  accepted, given the first, 53% of the time on prose, 67% on an explanation, 92% on code (`--mtp-accept`
  prints it; the raw residual instead of the normed output is 1-3 points worse). Checking three rows gives 2.2-2.9
  tokens a round for 3.3 rounds in 2.5; but a round then costs about 25% more (the third row adds about 10 ms
  to the verify, the second draft about 6, and a warm step per kept row), so the net speed is -1% on prose and
  +9% on code (alternating runs, 96 tokens, five each). The tokens are the same as plain greedy.
* **Costs:** 0.8 GB more memory (the head 239 MB, two spare delta states 600 MB) and about 2 s more to start (the
  few-row kernels are compiled at load).

## 6. Limits and what is left

* **Speed**: 14.6 GB are read a token, so the floor at the copy bandwidth is about 27 ms; the kernels are
  at 41. What is left is the 900 small launches (more fusion, or fewer kernels per layer) and a faster batch kernel;
  the matvec itself is near what the chip gives (§5).
* **Context**: 1024 positions (a constant of the compiled kernels; attention is compiled for 256, 512 and 1024 of
  them and uses the smallest that covers the conversation), and attention's one-tile scores cap it at 1364 on
  Metal (`docs/kernels.md` §4). The model's context is 262,144.
* **`cancho-gpu-run35` takes raw text**, greedy; the chat template, thinking mode and sampling are the server's
  (`docs/serving.md`), over the same engine (`device/qwen35.cho`: `load`, `reset`, `feed`, `feed_one`).
* **Positions**: text positions only (the model's interleaved m-RoPE reduces to plain RoPE when the three
  position streams are equal, which is all this feeds it); the multi-token-prediction head is used only as above.
