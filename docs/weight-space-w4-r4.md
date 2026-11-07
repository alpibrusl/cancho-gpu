# W4 R4: can the loop-fixing edit be found by a gradient rule? (design written before any gradient was measured, 2026-11-07)

> **Status.** Sections 1 to 6 are the design and the pre-registered rules; they were written and committed **before** any gradient was captured, any candidate computed or any candidate edit run (the commit that adds this file precedes the code and the run in the branch history). What was done before writing it: the 60 loops were selected (a data step, no model output looked at), one forward and backward on loop 0 was timed (4.9 s, memory 3.8 GB; the gradient norms of that probe were printed by the smoke and are not used for anything), and the code layout below was planned. Section 7 onwards (added after the run) holds the results; a rule in sections 1 to 6 is not edited after a result is seen, a deviation goes to section 8. Marks as in `weight-space.md`: **[read]**, **[measured]**, **[estimate]**, **[choice]**, **[open]**.
> Follows `weight-space-w4-search.md` (NOT RECOVERED: the spectral and contrastive rules gave `|cos|` with the published `u` at most 0.44 and column overlap at most 34 of 64) and is the proposal's R4 (`weight-space-w4.md` 6), run now that the Mac GPU is free. It uses the E2c proxy of `weight-space-w4.md` 14 (passes its calibration: published edit `E` = 0.975, `e19` 0.362, `x1e-1` 0.000) and the collateral measure of its sections 11 to 13.

## 1. The question

The published edit is `dW[:, j] = -6.000 (u . W_j) u` on 64 columns of `mlp.down_proj` in layers 15 and 19 of Qwen3.5-0.8B-Base (`weight-space-w4.md` 11.1 **[measured]**). The base repeats a loop token for token (`E` = 0 on the 40 E2c loops: it leaves none). **Given only the base and some of its own loops, does the gradient of "the loop continues" with respect to `down_proj` find (a) the direction `u` and (b) the 64 columns, and, whether or not it does, is the edit it builds out of its own answer a loop-breaker that leaves the model's other behaviour alone?** `u_pub` and the published columns are used **only to score**, except where a section says otherwise (the scale of 3.5, marked).

## 2. Materials (fixed)

* **Model.** `~/models/qwen35-08b-base` (`Qwen3_5ForConditionalGeneration`), the bf16 checkpoint widened to float32, on MPS (the Mac's own GPU), the shipped transformers code with the torch fallback for the gated-delta layers, exactly as E2c (`w4_e2c.load_model`). Only this model is run; the capture and gradient code takes a model name or directory and a list of `(layer, module)` pairs (here `(15, "mlp.down_proj")`, `(19, "mlp.down_proj")`), and finds the decoder layers by name, so another family is a command line.
* **Loops.** `w4_e2c.py select` with `want = 60`: the first 60 qualifying looping generations of `results/w1_cloud/data/base_8192.jsonl` (read-only) in problem order, by E2c's rule (14.2: `hit_limit`, an exact repeating tail of period 8 to 400 and span at least 1,500, 256 tokens remaining after the cut). The first 40 are E2c's 40 (checked: same problem ids). Context = chat prompt (thinking on, the authors' suffix) + `ids[:c]`, `c` = loop start + 1,000; the **loop-continuing tokens** are `ids[c:c+256]` (the base's own next 256). Contexts are 3,000 to 7,900 tokens.
* **Split.** The 60 loops in list order; **train = the loops at even positions (0, 2, ...), test = the loops at odd positions** (30 and 30). **[choice]** a rule that uses nothing but the position. Everything "found" (directions, columns, scales) uses train only for the candidate; test is for the held-out confirmation (4.1) and the behaviour (4.2). 20 of the 30 test loops are among E2c's 40 (their base and published-edit records are reused, the other 10 are run).
* **Published edit (scoring only).** `K.load_published(base, edited)`: layers 15 and 19, `dW = edited - base`, `u_pub` = the top left singular vector of `dW` (`w4_lib.rank1`), the 64 published columns = its nonzero columns, `||dW||_F` per layer.

## 3. The candidates

### 3.1 The losses and the gradient
Per loop, the context and the 256 loop-continuing tokens are one teacher-forced forward in float32; `p_t` is the model's probability of the loop's token at the 256 counted positions. Two losses, both to be **reduced**:
* **Lp** = `mean_t log p_t` (the task's: the log-probability of the loop continuing; saturated: the base's mean NLL is 0.0002, so the gradient per token is `(1 - p_t) * grad(log-odds)` and weights the uncertain tokens, the ones where a loop could break). **This is the primary loss.**
* **Lo** = `mean_t [log p_t - log(1 - p_t)]` (the log-odds; the same direction per token without the `(1 - p_t)` weight; reported as the unsaturated alternative).

Only the weights of the two target modules require gradients, so autograd runs the backward from the loss through layers 23 to 15 only. Per loop `j`, layer `l`, loss `X`: `G_j = dX/dW` (1,024 x 3,584, = `sum_t g_t a_t^T`, `g_t` the gradient at the module's output, `a_t` its input; computed by autograd and checked against the hook-side sum), `m_j = sum_t g_t` (the sum of the output-side gradients over all positions), `Q_j = sum_t g_t g_t^T`, `r_j = sum_t a_t^2` (per column, with `n_j` positions). For a set of loops `H` the sums over `H`: `G_H`, `m_H`, `Q_H`, and `rms_H = sqrt(sum r / sum n)`.

### 3.2 Directions `u` (unit, sign irrelevant), per layer, per loss, per half `H` (train, test)
* **GU1**: the top left singular vector of `G_H` (the output-side factor of the summed gradient). **Primary.**
* **GU2**: `m_H` normalised (the mean of the output-side gradient).
* **GU3**: the top eigenvector (largest eigenvalue, uncentred) of `Q_H` (the dominant axis of the output-side gradients, a gradient analogue of the search's D4).

### 3.3 Column rules (each returns 64 of 3,584), per half
* **C1**: the 64 largest `||G_H[:, j]||` (needs no `u`; the task's (b)). **Primary.**
* **C2**: the 64 largest `|u . G_H[:, j]|`. **C3**: the 64 largest `|u . W_j|` (the search's R1 with a found `u`). **C4**: the 64 largest `|u . W_j| rms_j` (the search's Rrms, the best rule there).

3 directions x 2 losses x 4 rules = 24 (direction, rule) pairs per half. **Primary pair: (GU1, C1) with Lp.**

### 3.4 Controls
* **Random** (the equal-budget null): 200 random unit vectors (their `|cos|` with `u_pub`), 200 random sets of 64 of 3,584 columns (their overlap), seed `w4_lib.SEED + 5`.
* **Permutation null**: 50 **sign-flip permutations** of the half's loops: draw `s_j` in {+1, -1} per loop (all equal excluded) and recompute the direction from `sum_j s_j G_j` (GU1) or `sum_j s_j m_j` (GU2); a signal shared by the loops adds up, a loop-specific or noise one does not. Its `|cos|` with `u_pub` and the overlap of C1 on `sum_j s_j G_j` are the null. (GU3 is a sum of positive semidefinite terms; it has no sign-flip null and is held against the random one only.)
* **Specificity control** (does the loop matter, or is this the layer's generic gradient axis?): the same three directions from the gradient of the **ordinary language-modelling loss** `mean log p(true next token)` over positions 256 to 1,022 of the first 1,024 tokens of the same texts (prompt + the base's generation before the cut; most of them are before their loop, not all). Reported as `|cos|` with `u_pub` and with the loop gradient's directions. It is a diagnostic, not a clause of the rule.
* **Oracle references** (not candidates): C2, C3, C4 with `u_pub` instead of the found `u`, and `u_pub`'s `|cos|` with itself's rank-1 approximations of `G_H` (the share of `G_H`'s Frobenius norm that lies along `u_pub`: `||u_pub^T G_H|| / ||G_H||`, against the chance `sqrt(1/1024)`).

### 3.5 The candidate edit (behaviour; section 4.2)
For the primary pair's direction `u` and columns `S` (both from **train**), in each layer `l`:
* **Form P (the published form):** `dW[:, j] = -alpha_l (u . W_j) u` for `j` in `S`, zero elsewhere.
* **Form G (the gradient step):** `dW[:, j] = -s_l (u . G_j) u` for `j` in `S`, zero elsewhere, `G` = the summed train gradient of Lp. (`u^T G` is a row vector; this is the rank-1, 64-column restriction of one gradient-descent step on Lp.)
* **The scale (one scale only, no tuning):** `alpha_l` and `s_l` are chosen so that `||dW_l||_F` equals the **published edit's** `||dW_pub,l||_F` in that layer. **[choice, uses the published edit's norm; this is the one place the published checkpoint decides something, and it fixes the budget, not the direction]**. The sign of `u` does not matter in either form.
* **Random-direction control (same budget):** the same two forms with `u` replaced by a random unit vector (seed `w4_lib.SEED + 6`, per layer) and the **same columns and the same scale rule**.
Written values are the base's bf16 plus `dW`, rounded to bf16 (`w4_ckpt.write_checkpoint(edits=...)`: the same writer, read-back check and manifest as `weight-space-w4.md` 13). At most **two** checkpoints are written: `r4p` (form P) and `r4g` (form G) of the primary pair. The controls are evaluated in memory with the same bf16 rounding (`bf16_round(W + dW)`); nothing else is written.

### 3.6 Exploratory (not part of any rule)
**S3**: three steps of descent on Lp over the train loops with the dense weight of the two layers (each step the gradient scaled so that the step's Frobenius norm summed over the two layers is a third of the published edit's), the accumulated update's top left singular vector and column norms scored as above. Only if time allows; it has no held-out half. Reported as exploratory.

## 4. The pre-registered rules

### 4.1 Rule (a): is the published recipe recovered?
Each (direction, loss, rule) is computed on the train half and, independently, on the test half. A pair's table entry in a layer is the **minimum over the two halves** of `|cos|` with `u_pub` and of the overlap with the published 64. With this table the rule of `weight-space-w4-search.md` 4 applies unchanged (`w4_search_lib.decide`, all directions of kind "data"):
* **RECOVERED**: some pair has, in both layers, `|cos| >= 0.9` and `overlap >= 48` of 64 (so on both halves).
* **PARTIAL**: not RECOVERED, and either **(A)** some direction has `|cos| >= 0.7` in both layers and beats its null in both layers (`beats` = the min-over-halves `|cos|` exceeds the maximum of the 200 random cosines and, for GU1 and GU2, of the 50 sign-flip permutations of both halves; GU3 against the random maximum only); or **(B)** some pair has `|cos| >= 0.5` in both layers (and beats its null) and `overlap >= 32` of 64 in both layers.
* **NOT RECOVERED** otherwise.
The thresholds are the search's, for the reasons given there (0.9 because `x1e-1` and `e19` show that an edit half an amplitude away does not reproduce the behaviour; 48 of 64 is H2's). 24 pairs against nulls whose maxima are reported; the thresholds are far above them.

### 4.2 Rule (b): does the gradient-built edit work, whatever its cosine? (built regardless of the cosine; the task's `cos < 0.7` clause is a permission, and it is used)
The two checkpoints of 3.5 are scored on the **30 test loops**, exactly as E2c (14.3: 4 samples of 256 tokens each with W1's sampler, `E` = the share of samples that are not identical to the base's continuation) and on the collateral corpus of `weight-space-w4.md` 3 (the 276 eval windows, 179,845 tokens, `corpus.json` of the W4 worktree, read-only), the KL(base || variant) per token, pooled, with its 95% interval over windows (`w4_collateral.summarize`, `ratio_ci`).
* **FOUND** (a candidate edit counts as found) iff **`E >= 0.5` on the held-out test loops AND the upper end of the mean-KL interval `<= 0.05` nats.** (0.05 is the gate's per-domain mean-KL limit, looser than the all-corpus gate of 0.02 that the published edit meets with 0.0205 mean; stated so as not to read "found" as "gate-passing".)
* **SPECIFIC** (found *because of the direction*) iff, additionally, the random-direction control of the same form (same columns, same scale, same bf16 rounding) has `E <= half of the candidate's E`. A candidate that is FOUND and not SPECIFIC means: at this norm and on these columns any direction breaks the loop, which would be a different result (and is reported as such).
* Neither is tuned: one scale, one pair, two forms, the thresholds above. No candidate other than the primary pair is run.
* Reference rows on the same 30 loops: the base, the published edit (reused records for the 20 loops E2c has, run for the 10 it does not), and the published edit's KL on the same corpus code (to check the KL code against 0.0205, `weight-space-w4.md` 12).

### 4.3 What the answer means (fixed now, for the product question "find an edit for a new model")
* **ALIVE (strong):** RECOVERED, or FOUND and SPECIFIC with form P or G: a gradient on a handful of the model's own loops produces a loop-breaking edit; a next model family is worth a run.
* **ALIVE (weak):** FOUND but not SPECIFIC, or PARTIAL: the gradient carries some of the structure or the budget does the work; the controls say which.
* **DEAD on this evidence** (for this recipe's rank-1 form with 64 columns, this model, these layers): NOT RECOVERED and not FOUND. Not a proof about other recipes: the dense few-step edit (3.6) is the pointer if it does anything.

## 5. Code and tests

`examples/weightspace/w4_grad_lib.py` (numpy only: the directions, rules, sign-flip and random nulls, the table and `decide` input, the two edit forms and their scales, the split) and `examples/weightspace/w4_grad.py` (torch: layer finder by name, capture of gradients for `(layer, module)` pairs on a model and a list of `(context ids, target ids, loss)` jobs, `analyse`, `build`, `behave`, `kl`, `score`; resumable, one record or one file per loop; every refusal a `w4g-*` rule tag; the CLI exits 2 on them). `tests/w4_grad.py`: planted cases for the pure parts (a planted rank-1 gradient is found by GU1 and C1, sign never matters, the nulls, the forms' formulas entry by entry and their norm match, the split, the rule on hand-made tables, refusals) and, when torch is available, the capture against finite differences and against `weight.grad` on a toy two-layer model with the same layer layout; skips cleanly without numpy (and the torch part without torch), verified with shadow `numpy.py` / `torch.py` stubs on `PYTHONPATH`; wired into `scripts/gate.sh`.

## 6. What the run cannot show

* One model, two layers, 60 loops of one sample per problem; the loops of GSM8K prompts only. The cut is 1,000 tokens inside the loop: the gradient says how to break an **established** loop, not how to stop entering one; E2c measures the same half (14.1).
* `u_pub` is the authors' direction; a different direction with a different effect is possible. The cosine and overlap are weight-space facts; the behaviour is E2c's proxy (a necessary condition, `weight-space-w4.md` 14.7) and the 0.05 KL clause is not the full gate: **FOUND does not mean the W1 arm would pass**.
* Float32 on MPS from the bf16 checkpoint; a gradient of a saturated loss at 0.0002 nats is tiny and the float32 backward through the torch fallback of the gated-delta layers has not been checked against float64 (the direction is what is used, and the unit tests check the capture against finite differences on a toy).
* The specificity control's texts are the base's first 1,024 generated tokens: not guaranteed loop-free.
