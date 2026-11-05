# `qsw`: the Qwen2 LoRA training session writer, in cancho

> **Status: slice 1 built and held to the Python writer byte for byte.** The session of a Qwen2 LoRA training run
> (`examples/qwen/qwen_train.py` `write`) is written by `tools/qsw`, a cancho program, with no Python in the path. It
> exists for the pure-cancho RL library (alpibrusl/cancho-rl, `docs/design.md` §26), whose training loop writes a session
> per iteration and needs a per-position weight in the loss. Claims are marked **[measured]** (with the command),
> **[read]** (from the source, not run) or **[open]**.

## 1. What it is, and what it is not

`qwen_train.write` is not a computation: it turns a model's shape, a batch of tokens and the adapters into a directory of
data files, four generated kernels and one `session` text (about 600 lines for two layers, 30,000 for the 1.5B). That makes
"the same" checkable exactly, so the check is the strongest one there is: **the same file names and the same bytes**
(`tests/qsw.py`). The Python writer stays in the tree as the specification and as the writer of everything this slice does
not cover.

```
qsw write --config FILE --wdir DIR --out DIR --root DIR --tokens FILE --adapters FILE [--save FILE] [--weights FILE]
```

| input | what it is |
|---|---|
| `--config` | `key value` lines: `layers d h hkv dh ff v t nb` (required), `tied r steps bc reuse` (defaults 1, 16, 1, 2304, 1). Floats (`eps theta alpha lr wd loss_scale b1 b2 eps_adam`; defaults as `write`'s) are given as the **decimal integer of their binary64 bits**, so no decimal reading, and no rounding of one, stands between the Python float and this one's. The last line for a key wins. |
| `--wdir` | the converted weights (`emb.f16`, `head.f16` when untied; the other files are named in the session, not read). An absolute path without a trailing slash. |
| `--out` | an **existing** directory (cancho has no path-based `mkdir`: `dir_mkdir` needs a `Dir`), an absolute path without a trailing slash. |
| `--root` | this repository (where the kernels are found, in `write`'s order: `examples/qwen`, `examples/train`, `tests/chx/ported`, `tests/chx`). |
| `--tokens` | int32 little-endian `[steps, nb, t]`; one batch per step, taken round and round as in `write`. |
| `--adapters` | binary64 little-endian, in `qwen_ref.lora_names` order (layer, then q k v o gate up down, A then B). |
| `--save` | where the session saves the trained adapters on a `checkpoint` (f32, fused, as `write`'s `save`). |
| `--weights` | binary64 `[nb t]`: the weight of each position in the loss (§3). Without it, 1 at every position but the last of a sequence, which is `write`'s. |

Refusals print `qsw: <rule>: <message>` on standard error and exit 2; the rules are `qsw.usage`, `qsw.config`, `qsw.shape`,
`qsw.input`, `qsw.kernel`, `qsw.quant` and `qsw.write`, each with a case in `tests/qsw.py`. No input reaches a trap.

## 2. Held to the Python writer **[measured]**

`scripts/build-qsw.sh && python3 tests/qsw.py build/qsw` (in the gate; about 30 s, most of it Python's):

* **The encoders alone** (`tools/qsw/enc.cho`): the f32 word, the f16 word and Python's `repr` of 80,015 binary64 values,
  random bit patterns, exact ties in f16 and f32, subnormals and overflow, against numpy's `astype` and Python. `%g` of
  whole numbers (the count in `answer ls.dw scale 1/…`), including the carry (999,999,500,000 is `1e+12`).
* **Nine sessions**, each written from the same inputs to the same directory (the session names its files by path), every
  file compared; seven against the Python writer: two layers with 1 and 2 steps, with and without the layers' shared buffers; other
  optimiser settings (alpha, lr, weight decay, loss scale, betas, Adam's epsilon); three layers with 8 heads over 2 and an
  exact vocabulary block; a head of 128 with an untied output layer; a 1.5B-shaped layer (d 1,536, ff 8,960); a sequence of
  512 (rotary angles up to 511 rad). **All files equal**: 70 to 149 files each. The other two are the weights of §3.
* **The 1.5B** (`python3 tests/qsw.py build/qsw --full`, not in the gate): Qwen2.5-1.5B's shapes (28 layers, d 1,536, 12
  heads over 2, ff 8,960, vocabulary 151,936, a batch of 8 × 512): **1,150 files, every one equal**. The tool took **2.0 s
  and a peak of 1.85 GB** (the padded output layer is 466 MB, written once, and the table is read once).
* **The refusals**: 14 cases, each broken one way at a time.
* **Mutants, all killed**: the order of two optimiser carries swapped (8 failures); f16 ties rounded up (1 failure, in the
  encoder check: it survived the sessions, whose random doubles are never ties, which is why the encoder check exists);
  the weights file ignored (2 failures).

What makes the exact comparison possible, and what would break it: a float printed in a session is `repr` of a binary64
(`enc.put_repr` on `std.fmt`'s shortest digits, positional from 1e-4 to 1e16); every f32 and f16 in a file is rounded **once**
from binary64 (`enc.word_of`, nearest even; rounding through f32 first would differ); the rotary tables are `cos` and `sin`
of `i θ^(-j/hf)` with `std.math`'s `pow`, `sin` and `cos`, which agree with numpy's after the f32 rounding at every table
tried, **up to 512 positions and θ = 10^6 [measured]**. They are not specified to agree: a table at which they differ is a
failure of the comparison, not a bug in either, and the first one found will say so here.

## 3. The per-position weight

`ce_grad.chx` computes `dl = (exp(logit - lse) - onehot(target)) wt` as f16, with `wt` the row's weight read from the
resident file `wscale` **[read]**; `write` fills it with `loss_scale` for a position with a next token and 0 for the last
of a sequence. `qsw --weights` fills it with `weight × loss_scale` instead, so a policy gradient is the same session with
the advantage of each token in the weight (0 for the prompt). **No kernel changes [read, and measured here only as the
file]**: with `--weights` every file but `wscale` equals the Python writer's, `wscale` is f32(weight × loss scale), and a
weights file equal to the default one reproduces the whole directory byte for byte.

Three things a caller must know, none of them a kernel matter:

* The optimiser divides the gradient by `nb (t − 1) × loss_scale`, a constant written into the program text (`gs`), whatever
  the weights are. A mean over `N` weighted positions is therefore weights scaled by `nb (t − 1) / N`. Adam is nearly
  invariant to the scale; its epsilon is not, and neither is the f16 range of `dl` **[open: the range of weight ×
  loss_scale × the softmax residual has not been measured on a policy-gradient batch]**.
* The loss the session answers (`ls.dw`) is the mean cross-entropy over the positions of `wcount` (the default mask), not
  the weighted one: it is the language-model loss of the batch, a monitor, not the policy objective.
* `wscale` is resident, one vector for all the steps of a session. A different weight per step is a different session (the
  per-iteration session cost of cancho-rl's design §28 is still to be measured with this writer).

## 4. Not covered (slice 1)

* **A 4-bit base** (`quant.py pack` files): refused with `qsw.quant`. `write`'s `dq_*` programs and its scratch aliases.
* **`train=False`**, the `lean` variant, the `tune` and `kmut` overrides, `probes`, `grads`.
* **Continuation from a checkpoint**: `state`, `first_step`, `save_state` (the moments are zero and the schedule starts at
  step 1).
* **The converter** (`convert.py`: safetensors to the weight files) and **the adapters' initialiser**: the writer takes
  files; making them is still Python.
* **`--out` is not created**, and relative paths are not resolved (the session names files by absolute path).
* **The driver** that runs the session (`cancho-gpu-metal-session`, already cancho) from a loop that rewrites the session
  each iteration: cancho-rl's, not this repository's.

## 5. Layout

`tools/qsw/` (each file under 2,000 lines): `enc.cho` (the encoders and Python's text), `tpl.cho` (the line templates,
`$key$` as the f-strings' `{key}`), `kern.cho` (the generated kernels and the scheduled copies of a library kernel),
`data.cho` (the data files), `session.cho` (the programs of the forward pass, the backward pass and the optimiser, the
aliases), `cfg.cho`, `rd.cho`, `main.cho`, and `probe.cho` (the encoders under test). Built with `scripts/build-qsw.sh`
into `build/qsw` and `build/qsw_probe`.
