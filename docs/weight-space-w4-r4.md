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
* **Permutation null (target permutation)**: 3 draws per half of a **derangement** of the half's loops (seed `w4_lib.SEED + 5 + k`, `k` = 0, 1, 2): loop `j`'s context is scored on the loop-continuing tokens of loop `pi(j)`, `pi(j) != j` (the same contexts, the same losses, the same estimators, the tokens the model assigns low probability to). The directions and C1 columns recomputed on these gradients are the permutation null: what the estimator returns when the loss is not about **this** loop's continuation. (Amended before any measurement, see section 8: a sign-flip permutation of the per-loop gradients was registered first and dropped because a direction shared by the loops survives sign flips of rank-1 gradients, so it cannot be beaten by a real signal; the unit test 3b shows it.)
* **Specificity control** (does the loop matter, or is this the layer's generic gradient axis?): the same three directions from the gradient of the **ordinary language-modelling loss** `mean log p(true next token)` over the base's generated tokens 256 to 1,023 of the same texts (given the prompt and the generation before them; most of them are before their loop, not all). It is the loop/non-loop contrast of the search (its D1 to D4 against relabelled windows) in gradient form, and it is part of the null (4.1).
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
* **PARTIAL**: not RECOVERED, and either **(A)** some direction has `|cos| >= 0.7` in both layers and beats its null in both layers (`beats`: the min-over-halves `|cos|` of the direction exceeds the maximum of its null: the 200 random cosines, the 3 target-permutation draws of both halves, and the specificity control's `|cos|` of the same direction type (GU1, GU2, GU3) in both halves); or **(B)** some pair has `|cos| >= 0.5` in both layers (and beats its null) and `overlap >= 32` of 64 in both layers.
* **NOT RECOVERED** otherwise.
The thresholds are the search's, for the reasons given there (0.9 because `x1e-1` and `e19` show that an edit half an amplitude away does not reproduce the behaviour; 48 of 64 is H2's). 24 pairs against nulls whose maxima are reported; the thresholds are far above the random ones, and the permutation and specificity maxima are what a direction has to beat on top.

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

`examples/weightspace/w4_grad_lib.py` (numpy only: the directions, rules, derangements and random nulls, the table and `decide` input, the two edit forms and their scales, the split) and `examples/weightspace/w4_grad.py` (torch: layer finder by name, capture of gradients for `(layer, module)` pairs on a model and a list of `(context ids, target ids, loss)` jobs, `analyse`, `build`, `behave`, `kl`, `score`; resumable, one record or one file per loop; every refusal a `w4g-*` rule tag; the CLI exits 2 on them). `tests/w4_grad.py`: planted cases for the pure parts (a planted rank-1 gradient is found by GU1 and C1, sign never matters, the nulls (derangements, never constant), the forms' formulas entry by entry and their norm match, the split, the rule on hand-made tables, refusals) and, when torch is available, the capture against finite differences and against `weight.grad` on a toy two-layer model with the same layer layout; skips cleanly without numpy (and the torch part without torch), verified with shadow `numpy.py` / `torch.py` stubs on `PYTHONPATH`; wired into `scripts/gate.sh`.

## 6. What the run cannot show

* One model, two layers, 60 loops of one sample per problem; the loops of GSM8K prompts only. The cut is 1,000 tokens inside the loop: the gradient says how to break an **established** loop, not how to stop entering one; E2c measures the same half (14.1).
* `u_pub` is the authors' direction; a different direction with a different effect is possible. The cosine and overlap are weight-space facts; the behaviour is E2c's proxy (a necessary condition, `weight-space-w4.md` 14.7) and the 0.05 KL clause is not the full gate: **FOUND does not mean the W1 arm would pass**.
* Float32 on MPS from the bf16 checkpoint; a gradient of a saturated loss at 0.0002 nats is tiny and the float32 backward through the torch fallback of the gated-delta layers has not been checked against float64 (the direction is what is used, and the unit tests check the capture against finite differences on a toy).
* The specificity control's texts are the base's first 1,024 generated tokens: not guaranteed loop-free.

## 7. Results (measured 2026-11-07; nothing above this line, except the amendments of section 8, was edited after the run)

Run: `w4_grad.py capture` (60 loops x [loop gradient with Lp and Lo, ordinary-LM gradient, 3 target-permutation gradients]; 300 files, 1,952 s on the Mac GPU, float32; the hook-side `g^T a` equals autograd's `weight.grad` exactly on every job), `analyse` (CPU), `build`, `behave` (E2c on the 30 test loops), `kl` (the 276 eval windows). Data: `examples/weightspace/results/w4_r4/` (`analysis.json`, `records_*.jsonl`, `kl.json`, `score.json`, `loops_meta.json` with the split). The KL code reproduces the published edit's mean KL of `weight-space-w4.md` 12: 0.02053 [0.01932, 0.02188] **[measured]**.

### 7.1 Rule (a): **NOT RECOVERED**
`|cos|` with `u_pub` (minimum over the train and test halves; layer 15 / layer 19) and overlap with the published 64 (minimum over halves):

| direction (loss) | cos | overlap C1 (the gradient's column norm) | C2 | C3 | C4 |
|---|---|---|---|---|---|
| GU1 (Lp), primary | 0.220 / 0.190 | 21 / 11 | 20 / 11 | 9 / 3 | 17 / 10 |
| GU2 (Lp) | 0.208 / 0.182 | 21 / 11 | 22 / 11 | 8 / 4 | 17 / 9 |
| GU3 (Lp) | **0.252 / 0.214** | 21 / 11 | 21 / 11 | 9 / 4 | 16 / 9 |
| GU1 / GU2 (Lo) | 0.093 / 0.109, 0.089 / 0.103 | 26 / 16 | 25 / 15 | 13 / 4 | 18 / 12 |
| GU3 (Lo) | 0.006 / 0.017 | 26 / 16 | 13 / 16 | 1 / 1 | 5 / 9 |

* **No cosine reaches 0.7 (best 0.252 / 0.214); no overlap reaches 32 (best 26 / 16); 48 of 64 is not approached.** Neither PARTIAL clause holds: (A) needs 0.7, (B) needs 0.5 and 32. The registered rule gives NOT RECOVERED. The best pair by the registered ranking is (GU3, Lp, C2).
* **Nulls.** Random unit vectors: median 0.023 / 0.024, maximum 0.089 / 0.085 of 200; random column sets: maximum 4 of 200 (chance 1.3). Target-permutation draws (Lp): GU1 `|cos|` 0.04 to 0.08 (layer 15), 0.09 to 0.12 (layer 19); their C1 overlap 20 to 21 / 11 to 12. The specificity control (ordinary-LM gradient): GU1 0.01 to 0.02 / 0.12 to 0.14, GU3 0.07 to 0.08 / 0.01. So the Lp directions are above the nulls (0.19 to 0.25 against null maxima of 0.09 to 0.19 once the random, permuted-target and ordinary-LM maxima are pooled; `analysis.json` `beats`), **a real but small alignment: the same estimator on other targets gives 0.04 to 0.14**.
* **Column ranking by the permuted-target gradient reaches the same 20 to 21 / 11 to 12 overlap as the real one.** The 21 / 11 of C1 is therefore not loop-specific: it is what the column norms of the gradient of any continuation score on these contexts give (about the oracle value of the rule with the true `u`: R1 31 / 18, Rrms 42 / 24 in this data).
* **Stability.** Train against test: `|cos|` of the directions 0.97 to 1.00 and 58 to 63 of 64 shared C1 columns: the gradient estimates are reproducible; they are not the published `u`.
* **What the gradient holds along `u_pub`.** The share of the summed Lp gradient's Frobenius norm along `u_pub` is 0.20 to 0.22 (layer 15), 0.17 to 0.19 (layer 19), against chance 0.031 (7 times chance); along its own GU1: 0.89. For Lo: 0.09 / 0.11.

### 7.2 Rule (b): the primary pair built into an edit, on the 30 held-out loops (E2c) and the collateral corpus
Both candidates: `u` = GU1 (Lp) of the train gradient, the 64 columns of C1, scale = the published edit's Frobenius norm per layer. Columns overlap the published ones 21 / 11; `|cos|` of `u` with `u_pub` 0.242 / 0.212 (random control 0.031 / 0.020); the candidates' Frobenius cosines with the published `dW`: P 0.021 / 0.062, G 0.039 / 0.085 (controls about 0).

| model | `E` (exit share) | `G_E` [95% CI] | mean KL (nats) [95% CI] | dPPL | FOUND | SPECIFIC |
|---|---|---|---|---|---|---|
| base | 0.000 | | | | | |
| published edit (reused for 20 loops, run for 10) | 0.983 | +0.983 [+0.958, +1.000] | 0.0205 [0.0193, 0.0219] | 1.33% | | |
| **r4p** (form P) | **0.892** | +0.892 [+0.775, +0.983] | **0.1547** [0.1460, 0.1639] | 18.1% | **no** | no (by the KL) |
| **r4g** (form G) | 1.000 | +1.000 | **1.0325** [0.979, 1.090] | 165% | **no** | no |
| ctrlp (random `u`, form P) | 0.000 | +0.000 | 0.0098 [0.0093, 0.0103] | 0.9% | | |
| ctrlg (random `u`, form G) | 1.000 | +1.000 | 0.0668 [0.0642, 0.0695] | 6.8% | | |

* **FOUND fails for both on the KL clause (0.05):** `r4p` leaves the loop in 89% of samples on held-out loops, with a KL 7.5 times the published edit's and a perplexity change of 18%; `r4g` leaves in all, at a KL of 1.03 (the model is broken: mean NLL of the loop's continuation 3.1).
* **Form P is direction-specific**: with a random `u` on the same columns and norm, `E` is 0.000 (and KL 0.0098); with the gradient's `u` (cos 0.24 with the published one) it is 0.892. The gradient direction and columns carry the loop-breaking, and they carry it at a large cost. **Form G is not direction-specific**: the random-`u` control also gives `E` = 1.000 (KL 0.067): a rank-1 step along `u (u^T G)` on the gradient's own 64 columns breaks the loops whatever `u`, because `G_j` is already the loop's gradient on those columns.
* No scale was tried other than the registered one, so nothing says a smaller `r4p` would pass the KL clause; a post-hoc scale search on these test loops would not be held-out and was not done **[open]**.

### 7.3 Reading, and the product question
1. **The published recipe is not rediscovered by the gradient**: direction cosine about 0.2 (3 to 10 times the nulls, far from 0.7), 21 of 64 and 11 of 64 columns (chance 1.3), identical to what the permuted-target gradient gives for the columns. R4's rank-1 form with a column norm rule does not find `u_pub` or its columns. That makes the second spectral/contrastive failure (`weight-space-w4-search.md`) a gradient failure too.
2. **But the gradient does find *an* edit that breaks the base's established loops, held-out**: `r4p` leaves 89% of samples (the published edit 98%), and this is direction-specific (random `u` 0%). The edit is not the authors' and not a free one: at the published norm its collateral is 7.5 times higher (KL 0.155) and fails the registered 0.05.
3. **Product "find an edit for a new model", on this evidence: alive in the weak sense, not in the strong one.** A handful of the model's own loops (30 train) and 30 s of gradient per loop give an edit with a loop-exit effect on unseen loops; they do not give one that meets a collateral budget at the published edit's norm, and they do not rediscover the recipe. What would decide it is a scale (or a column budget) chosen on train loops against a KL target, then scored once on held-out loops, and the same on a second model family; neither was run. The W1 (accuracy) effect of `r4p` is not measured; E2c is a necessary-condition proxy.

## 8. Deviations and limits
* **Amended before any measurement (commit after 7c5ff43):** (i) the permutation null: sign flips of per-loop gradients were registered first and replaced by target permutation (3.4), after the unit test 3b showed that sign flips of rank-1 gradients return the same direction, so a real shared signal could never beat them; (ii) the specificity control was moved from diagnostic into the null (4.1); (iii) the control's counted tokens are said to be generated tokens 256 to 1,023 (after the prompt), not positions of the whole sequence. No gradient had been computed on the 60 loops at the time of these edits.
* **Run deviations.** The first capture attempt failed on an MPS float64 conversion before saving anything and was rerun from the start. The KL run named the published variant `pub`; `kl.json` key was renamed to `edited` by hand before scoring. The S3 exploratory (3.6) was not run. The permutation draws' mean log-probability is -0.4 to -1.1 (loops of other problems are similar text), so the "permuted-target" null is a mismatched but plausible continuation. `analysis.json` carries per-half rows for all 24 pairs.

## 9. R4b: choose the scale of the r4p form on the train loops against a KL target, score once on the held-out loops (pre-registered 2026-11-07, committed before any scaled edit is evaluated)

> **Status.** Sections 9.1 to 9.5 are written and committed before any scaled candidate has been run on anything. What exists at this point: the section 7 results (`r4p` at the published norm: `E` 0.892 on the 30 test loops, mean KL 0.155; random `u` does nothing) and the train gradient `cand.npz` of the section 7 run (direction `u` and summed Lp gradient `G` per layer, computed from the 30 train loops only). A rule here is not edited after a result is seen; a deviation goes to 9.7. Marks as above.

### 9.1 Question
Section 7.3 left one decision open: can the r4p form meet a collateral target without being tuned on the loops it is judged on, and still break held-out loops? If not, the "find an edit for a new model" line ends on this model.

### 9.2 Candidates (a grid, 18)
Form P with the section 7 direction `u` (GU1 of Lp, train) and the section 7 columns, ranked by the train gradient's column norm `||G_j||` (C1). A configuration is (fraction `f`, budget `k`) with `f` in {0.1, 0.2, 0.3, 0.5, 0.7, 1.0} and `k` in {16, 32, 64}: the `k` columns of largest `||G_j||` (the top `k` of the 64 of section 7; ranking uses the train gradient only), and `dW[:, j] = -f alpha_l (u . W_j) u` for those columns, with `alpha_l` the coefficient of the section 7 `r4p` edit in layer `l` (the one that makes the 64-column edit's Frobenius norm equal the published edit's). So `f` is the per-column strength relative to `r4p`, `k` the number of columns, and (1.0, 64) is `r4p` itself (a check: it must reproduce KL 0.1547 and, on the test loops, `E` 0.892 up to the bf16 rounding and sampler determinism). Values are bf16-rounded as in 3.5. Nothing else is tried: no other `u`, columns or form.

### 9.3 Selection (train only)
1. **KL.** Each of the 18 is scored on the collateral corpus (the 276 eval windows of section 4.2; token-pooled mean KL(base || variant) with its 95% interval over windows). The corpus contains no loop of the 60 and is not part of the train or test split, so the KL of a configuration is the same number for selection and for the final report; it is used for selection, the held-out loops are not. A configuration is **eligible** iff the upper end of its mean-KL interval is `<= 0.05` nats.
2. **Effect on the train loops.** Each eligible configuration is run on the **30 train loops** as E2c (4 samples of 256 tokens, W1's sampler, `E` = the share of samples not identical to the base's continuation; the same code as section 7). (Configurations that are not eligible are not run on the train loops in the selection, to save about 11 minutes each; they cannot be selected.)
3. **Selected configuration** = the eligible one with the largest train `E`; ties (equal `E`) go to the lower KL upper end, then to the smaller `f`, then to the smaller `k`. If no configuration is eligible the answer is NOT FOUND and no loop is run.
4. The selection curve reported: KL against `f` for each `k` (all 18), and train `E` against `f` for each `k` (eligible ones; the others are not measured by the rule, and if run afterwards for the curve they are labelled exploratory and cannot change the selection).

### 9.4 The one held-out scoring
The selected configuration is run **once** on the **30 held-out (test) loops** (E2c, as above), together with its **random-direction control** at the same configuration: the same columns, the same bf16 rounding, `u` replaced by the random unit vector of section 3.5 (seed `w4_lib.SEED + 6`, per layer), and the edit scaled so that its Frobenius norm per layer equals the candidate's. Reference rows on the same 30 loops: the published edit (records of section 7) and `e19` (the published edit on layer 19 only; the 20 loops of E2c reuse its records, the other 10 are run), each with its KL on the same corpus.

**FOUND** iff all of: held-out `E >= 0.5`; the mean-KL upper end of the selected configuration `<= 0.05` (from 9.3); the random-`u` control's held-out `E < 0.1`. Otherwise **NOT FOUND**. No second configuration is scored on the held-out loops under any circumstance, including when the selected one fails.

### 9.5 Meaning, fixed now
* **FOUND:** a gradient-built, scale-selected rank-1 edit breaks held-out loops within a collateral budget of 0.05 nats and the effect is direction-specific; the single selected checkpoint is written (`w4_ckpt.write_checkpoint`) with its screening evidence (`w4_screen.py`) so that a patch carries it. W1 is not run, and FOUND does not mean W1 would pass (0.05 is looser than the gate's 0.02 all-corpus KL).
* **NOT FOUND:** the "find an edit for a new model" line ends on this model with this recipe (gradient direction, C1 columns, rank-1 form P): the loop-breaking and the collateral cost are not separable by scale or column budget on these grids. Not a statement about other recipes.

### 9.6 Code and tests
`examples/weightspace/w4_scale.py` (the configs, the edits, the selection and the verdict as pure numpy functions; CLI commands `grid`, `behave`, `kl`, `select`, `final`; it reuses `w4_grad.py`'s MPS runner and KL code), `tests/w4_scale.py` (hand-made tables for the selection and tie rules, the verdict, the edit formulas and the (1.0, 64) identity with `edit_p`; skips without numpy), wired into `scripts/gate.sh`.

### 9.7 Results (measured 2026-11-07; 9.1 to 9.6 above were not edited after the run)
Run: `w4_scale.py grid` (the 18 edits; check: (1.0, 64) equals `edit_p` exactly, entry for entry), `kl` (the 18 plus the published edit and `e19`; the (1.0, 64) row reproduces section 7's r4p KL 0.1547 and the published edit's 0.02053), `behave --split train` on the 12 eligible configurations (30 loops, 4 samples, the E2c code), `select`, then the one held-out scoring. Data: `examples/weightspace/results/w4_r4b/` (`grid.json`, `kl.json`, `select.json`, `final.json`, `e2c_train/`, `e2c_test/`).

**The selection curve (train loops E, collateral KL; mean [upper end of the 95% interval])**

| f | k=16: KL | k=32: KL | k=64: KL | train E (k=16 / 32 / 64) |
|---|---|---|---|---|
| 0.1 | 0.0013 [0.0014] | 0.0012 [0.0013] | 0.0013 [0.0013] | 0 / 0 / 0 |
| 0.2 | 0.0054 [0.0056] | 0.0051 [0.0053] | 0.0052 [0.0055] | 0 / 0 / 0 |
| 0.3 | 0.0123 [0.0128] | 0.0117 [0.0123] | 0.0120 [0.0127] | 0 / 0 / 0 |
| 0.5 | 0.0353 [0.0370] | 0.0343 [0.0361] | 0.0354 [0.0374] | 0 / 0 / **0.008** |
| 0.7 | 0.0713 [0.0750] | 0.0707 [0.0747] | 0.0728 [0.0771] | not eligible; 0.175 (k=64, exploratory) |
| 1.0 | 0.1500 [0.1578] | 0.1492 [0.1578] | 0.1547 [0.1639] | not eligible; 0.900 (k=64, exploratory) |

* **Eligible (KL upper end <= 0.05):** the 12 configurations with `f <= 0.5`. **The effect is a cliff, not a slope**: train `E` is 0 for every eligible configuration except one loop of 30 at (0.5, 64) (`E` 0.008, loop 14 exiting in one sample of four). The two ineligible, exploratory `k=64` rows (run after the selection, cannot change it): `E` 0.175 at `f` = 0.7 (KL upper end 0.077) and 0.900 at `f` = 1.0 (0.164). `E` becomes appreciable only where the KL is already above the target (between `f` 0.5 and 0.7 the KL doubles and `E` is still 0.18); there is no point on the grid with `E` >= 0.5 and KL <= 0.05. The column budget changes almost nothing: at `f` = 1.0, 16 columns cost the same KL as 64 (0.150 against 0.155), so the loop-breaking and the cost both ride on the largest-gradient columns, whatever the rest do.
* **Selected on the train loops:** (f, k) = **(0.5, 64)** (largest train `E`, 0.008, KL 0.0354 [0.0337, 0.0374], dPPL 4.4%).

**The one held-out scoring (30 test loops, E2c)**

| model | held-out `E` | mean KL [95%] | dPPL |
|---|---|---|---|
| selected (0.5, 64) | **0.042** | 0.0354 [0.0335, 0.0374] | 4.4% |
| random-`u` control, same columns and norm | 0.000 | | |
| published edit (reference) | 0.983 | 0.0205 [0.0193, 0.0219] | 1.3% |
| `e19` (published, layer 19 only; reference) | 0.333 | 0.0080 [0.0077, 0.0084] | 1.1% |
| r4p (1.0, 64), section 7 (reference) | 0.892 | 0.1547 [0.1460, 0.1639] | 18.1% |

* **Verdict: NOT FOUND.** Clause by clause: held-out `E` >= 0.5 **fails** (0.042); KL upper end <= 0.05 holds (0.0374); control `E` < 0.1 holds (0.000, but it is vacuous when the candidate itself does nothing). No checkpoint was written and no screening evidence produced (nothing to carry). W1 was not run.
* The published edit exits 98% of held-out samples at KL 0.0205, and even `e19` alone (one layer, half the budget in the published form) reaches 0.333 at KL 0.0080. The gradient-built form at a collateral of 0.035 reaches 0.042: **at equal or lower KL than the published edit's neighbours, the gradient direction and columns buy about nothing**; they only break loops once the cost is 5 to 8 times the published edit's.
* **What this ends:** with this model, the gradient direction (GU1 of Lp from 30 train loops), its C1 columns, the rank-1 form P, and scale or column budget as the only knobs, there is no configuration that breaks held-out loops under a 0.05 collateral. The "find an edit for a new model" line, in the form this document tested, is **closed on this model** (section 4.3's DEAD, now with the scale/budget escape tried and failed). It does not say that no edit exists (the published one does it at 0.0205), only that this gradient recipe does not find it. This also upgrades section 7.3's "alive in the weak sense": the weak sense was that the gradient finds *an* edit that breaks loops; it does, but only at a cost the budget excludes.

### 9.8 Deviations and limits
* None to the registered rule: all 12 eligible configurations were run on the train loops and the selection followed 9.3. The 6 ineligible ones were not run for the selection; two of them (`f` = 0.7 and 1.0, k = 64) were run afterwards for the curve (labelled exploratory above); the other four were not run for `E`.
* The KL corpus is shared by selection and the final report (stated in 9.3), so the held-out half is the loops only. The E2c `E` is a proxy (necessary condition) and the selection used 30 loops with 4 samples each: the cliff is large (0.008 against 0.9), well outside that noise.
* Wall-clock on the shared Mac was 2 to 5 times slower than in section 7 at times because other jobs ran; this affects no number.
