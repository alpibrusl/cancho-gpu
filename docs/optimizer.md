# Optimiser completeness (roadmap G6): clipping, decay groups, the schedule, accumulation

> **Status: designed first, then built; §6 holds what was measured.** Claims are marked
> **[measured]** (with the command), **[read]** (from the source), **[choice]** or **[open]**.
> The oracle is `torch` (`torch.optim.AdamW`, `torch.nn.utils.clip_grad_norm_`, `torch.optim.lr_scheduler`),
> and a binary64 numpy implementation where torch is float32 (`CLAUDE.md`: no Rust oracle).
> Every kernel is `.chx` (`examples/optim/`); no new Rust and no new language feature.

## 1. What exists, read from the source

* **AdamW** is in `examples/gpt/adamw_h.chx` (f32 masters, an f16 copy for the matrix unit), `examples/qwen/adamw_{e,f,l}.chx`
  and `examples/train/adam.chx` (no decay). **[read]**
* **Decay groups** exist only as a choice of kernel: the GPT's matrices run `adamw_h` and its vectors run `adam` (no decay), the
  torch comparison's parameter groups. **[read]**
* **Decay is `lwd = lr * wd`, a constant of the program.** `torch.optim.AdamW` decays by the *current* learning rate,
  `w <- w (1 - lr_t wd)`. With a schedule the two differ by `lr_t / lr_peak`: during warm-up the kernels decay too much and
  at the end of a cosine to 10% they decay ten times too much. `compare_gpt.py` passes `lr_fn` (the schedule goes in the
  `sched` table) but `lwd` stays at the peak. **[read]; the size of the effect is measured in §6.**
* **The schedule** is a host function (`compare_gpt.lr_at`, `gpt_session.schedule`) written to a `table` of
  `[lr_t / (1 - b1^t), 1 / sqrt(1 - b2^t)]` per step; each example has its own copy of the warm-up and cosine. **[read]**
* **Clipping:** none. **Accumulation:** none: a repetition of a session is one optimiser step; a bigger batch
  needs a bigger `c.b` (and the memory of it). `resident.md` has "accumulating" only in the sense of a kernel adding into a
  buffer, not of micro-batching. **[read: `grep -n -i "clip\|accum\|micro" docs examples`]**

## 2. The design

One session repetition is one **micro-step**. A window of `N` repetitions is one optimiser step. Everything that varies
per repetition goes in a `table`, which exists already (`resident.md` section 12), so there is no new directive.

**Kernels** (`examples/optim/`):

| kernel | computes |
|---|---|
| `gacc.chx` | `acc' = acc * keep + grad * sc`; `keep` (a table, one number a repetition) is 0 on the first micro-step of a window, 1 after; `sc = 1/N` makes the sum a mean |
| `gradsq.chx` | `o = acc + gs^2 * sum(grad^2)`; chained tensor to tensor through `link`, the chain is the squared global norm |
| `clip_scale.chx` | `cs = min(1, max_norm / (sqrt(ss) + 1e-6))`, torch's formula and its 1e-6 |
| `adamw_g.chx`, `adamw_gh.chx` (f16 copy) | decoupled AdamW; reads the scale `cs` and the 4 numbers of its group's table row `[rate, isq, dec, apply]` |

* **Clip:** the gradient is multiplied by `cs` *before* the moments, as torch does it (`clip_grad_norm_` then `step`). Adam is
  nearly invariant to a gradient scale (only `eps` breaks it), so the tests compare the **moments `m` and `v`**, not only the
  weights, and use an `eps` that is not negligible. **[choice]**
* **Decay groups:** each group has its own table, whose `dec = 1 - lr_t * wd_group` (1 for no decay) is computed on the host in
  binary64, so decay follows the schedule exactly as torch's does and a group is a column of data, not a kernel choice. **[choice]**
* **Schedule:** `examples/optim/schedule.py`, `warmup_cosine(peak, warmup, total, floor)`: `lr(t)` for the optimiser step
  `t = 1 .. total` equal to `SequentialLR(LinearLR, CosineAnnealingLR)`: step `t <= warmup` gives `peak t / warmup`; step
  `t > warmup` gives `floor + (peak - floor) (1 + cos(pi (t - warmup - 1) / (total - warmup))) / 2`. The two tests below pin
  the first cosine step to `peak` and the last to within one step of `floor`. `compare_gpt.lr_at` is left as it is (its first
  cosine step is already below the peak: a different, older, convention) so recorded results stay valid. **[choice]**
* **Accumulation:** repetition `j` (from 0) of window `t = j // N + 1` has the table rows `keep = (j % N != 0)`,
  `apply = (j % N == N - 1)`, and the Adam row of step `t`. When `apply` is 0 the Adam kernel returns `w, m, v` unchanged:
  `out = old * (1 - apply) + new * apply`, which is exact for 0 and 1 in f32 (a `old + apply (new - old)` form would not be). **[choice]**
* **Loss scaling** is not touched: `gs` stays the Adam kernel's gradient scale (the GPT's `1 / (n LOSS_SCALE)`), and the
  clip norm is the norm of the *scaled* (true) gradient, `gradsq` applying the same `gs`. **[choice]**

## 3. Exit criteria (fixed before the code)

1. **Clip.** Over a set of 5 tensors of unequal sizes, the new weights, `m` and `v` equal `clip_grad_norm_(max_norm)` +
   `torch.optim.AdamW` (float32) within 2e-6 relative to the largest value, in three cases: norm above `max_norm`
   (clipped), norm below (no scaling: `cs` is exactly 1), all gradients zero (no NaN, the weights only decay). The
   scale `cs` itself is within 1e-6 relative of torch's `clip_coef`.
2. **AdamW groups and schedule.** 40 steps with two groups (decay 0.1 and 0), a warm-up + cosine schedule, the same
   gradients: weights equal `torch.optim.AdamW` with two param groups and the real `lr_scheduler`s within 1e-5 relative of
   the weights' range; the scheduler's `lr` equals `schedule.py`'s to 1e-12 at every step.
3. **Accumulation.** (a) `N = 4` micro-batches of a small least-squares/GPT model, accumulated, give the update of one batch
   `4 x` as large within float tolerance (1e-5 relative); (b) a chain of 5 windows equals torch's
   `loss / N` backward `N` times then `step`, 5 times, with clipping on, within 1e-5.
4. **Mutants the tests must kill** (each makes one deliberate change to a copy of a kernel or of the generator, §5):
   the clip factor without the square root (`ss` where `sqrt(ss)` goes); the decay applied to the wrong group; the
   accumulation not divided (`sc = 1`); the window not reset (`keep` always 1); the clip applied after the moments;
   decay by the peak rate instead of `lr_t`; an `apply` that is always 1.
5. **No existing result moves**: `scripts/gate.sh` passes with no golden changed; the GPT and Qwen sessions still
   use their own kernels unless asked to use the new path (§4).

## 4. Where it plugs in

`examples/optim/optim_session.py` writes a session for a list of parameters `(name, shape, group)`, the per-micro-step
gradients as a `table` (so the optimiser is tested alone and exactly: the gradient is data, not a model). The GPT session
takes `opt=` (§6) so the same kernels train a transformer: `gacc` after each gradient, the `gradsq` chain, `clip_scale`,
then `adamw_gh` or `adamw_g` per tensor.

## 5. The mutants

`tests/optim.py --mutants` copies the kernel directory, changes one line of one kernel (or one table of the generator),
runs the same checks and requires them to fail.

## 6. Results

Built: `examples/optim/` (`gacc`, `gradsq`, `clip_scale`, `adamw_g`, `adamw_gh`, `schedule.py`, `optim_session.py`, `gpt_opt.py`),
`gpt_session.write(..., opt=dict(accum, max_norm, wd))`, `tests/optim.py` (in `scripts/gate.sh`, with `--mutants`).
Run: `python3 tests/optim.py build/cancho-gpu-session [build/cancho-gpu-metal-session] [--mutants]`. The CPU session and the Metal
session (M4 Max) give the same table to the digit shown below. Against `torch 2.12` float32, `AdamW` and `clip_grad_norm_`:

| check (criterion §3) | weights | first moment | second moment | clip coefficient |
|---|---|---|---|---|
| 1. norm above `max_norm` (3 steps, eps 1e-3) | 2e-8 | 3e-7 | 3e-7 | 0.1462933 vs 0.1462934 |
| 1. norm below `max_norm` | 2e-8 | 8e-8 | 1e-7 | exactly 1 |
| 1. all gradients zero | 0 | 0 | 0 | 1, no NaN |
| 1. norm of the size of the formula's 1e-6 | 0 | 1.5e-7 | 3.8e-7 | 0.2235838 = torch's |
| 2. two groups (0.3 / 0), warm-up + cosine, 40 steps | 9e-8 | 2e-7 | 7e-8 | |
| 3a. 4 micro-steps against one step on the mean gradient | 2e-8 | 0 | 0 to 1e-7 | |
| 3b. 5 windows of 4, clip 1.0, decay, schedule | 5e-8 | 2e-7 | 4e-7 | 0.1526007 = torch's |

(Errors are the largest difference relative to the largest value of the tensor.) **[measured]**

* **The schedule** equals `SequentialLR(LinearLR(start 1/warmup, total_iters warmup - 1), CosineAnnealingLR(T_max total - warmup, eta_min))`
  at all 40 steps to 1e-17. **[measured]**
* **A GPT** (2 layers, d 128, `tests/optim.py gpt_checks`): the new path with one micro-step, no clip and a constant rate gives
  the weights of `adamw_h` / `adam` after 2 steps within 9e-7 (CPU) and 1.1e-6 (Metal); 3 steps of 2 micro-batches of one
  sequence equal 3 steps of one batch of two sequences with the clip active (clip 0.3, decay, schedule) to **5e-3 of the norm of
  the move** (f16 rounding of the activations), while the unclipped run differs by 10x the move. These two are run in a regime
  where the update is linear in the gradient (eps 1e3, rate 30): with `eps = 1e-8` Adam moves every element by about `lr`
  whatever its size, so a gradient at the f16 noise flips sign between two runs that differ in a last bit, and neither the clip (which
  Adam absorbs) nor the big batch could be told from rounding. **[measured; a finding about what can be tested, not about the kernels]**
* **The old decay was not torch's under a schedule.** `adamw_h` decays by the constant `lr_peak * wd`; torch by `lr_t * wd`.
  For `compare_gpt.py`'s schedule (1,500 steps, peak 1e-3, wd 0.1, 5% warm-up, cosine to 10%) the weights' total decay factor is
  **0.861 with the constant, 0.921 with torch's** `exp(-wd sum lr_t)`: 6.5% of the weight, on the matrices, over the run. This
  does not change the recorded curves (they stay as recorded); the new path (`opt=`) is the one that decays as torch does. **[measured, arithmetic of the two formulas]**
* **`1 - beta2` in f32 is not torch's.** `1 - 0.999` computed in f32 inside the old kernels is off by 4.7e-5 relative, so their
  second moment differs from torch's by about 1e-5 (measured as 1.3e-5 before `ob1`/`ob2` were passed as host-side constants). The new
  kernels take `ob1`, `ob2` from the host in binary64 and match torch to 4e-7. The old kernels are left alone. **[measured]**
* **The clip's `min(1, c)`** first written `c - step(c - 1)(c - 1)` lost the 1 in f32 for `c = 1e30` (the no-clip case): the
  first run of the test gave a coefficient 0. Written `1 + (c - 1) step(1 - c)`. The test found it. **[measured]**

**Mutants** (`--mutants`, each a deliberate change, each killed): clip factor without the square root; clip not applied; clip applied twice;
the norm without `gs`; the formula's 1e-6 dropped (killed only by the case with a norm near 1e-6); accumulation not divided;
accumulator never reset; update on every micro-step; decay of the wrong group; decay at the first step's rate; no decay. The
mutants run on the optimiser session, not on the GPT. **[measured]**

**Open.**
* The Qwen LoRA sessions (`adamw_l`, with its mask and f16 scale) and the CNN's are not moved to the new kernels; only the GPT
  session takes `opt=`. The CNN's torch comparison also uses a warm-up + cosine and the same constant-`lwd` shape.
* `gradsq` reads a tensor with one threadgroup; at 60 M parameters that is a long serial walk. Not timed. A two-stage reduction
  (per-block partials, then a sum) is the next step if the clip shows in a step time. **[open]**
* A long run (`compare_gpt.py` at 1,500 steps) with the new path against torch, curve against curve, is not done: it is a GPU
  hour of the shared machine and the claim is carried by the exact small tests above.
* Gradient accumulation holds one `[n]` f32 buffer a parameter (the 59.95 M GPT: 240 MB), measured nowhere. **[open]**
* bf16/fp16 loss scaling dynamics (skip a step on overflow) are not part of this; `gs` is a constant.
