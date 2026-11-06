# W4 edit search: can the loop-fixing edit be rediscovered without the published direction or column list? (design written before any measurement, 2026-11-06)

> **Status.** Sections 1 to 6 are the design and the pre-registered success rule; they were written **before** any activation was captured or any candidate computed (the commit that adds this file precedes the code and the run in the branch history). Section 7 onwards (added after the run) holds the results; a rule in sections 1 to 6 is not edited after a result is seen, a deviation goes to section 8. Marks as in `weight-space.md`: **[read]**, **[measured]**, **[estimate]**, **[choice]**, **[open]**.
> It is the weights-and-activations half of `weight-space-w4.md` section 6 (E1), reduced to what a CPU and a handful of the base's own generations can do. Known from `weight-space-w4.md` 11.1 **[measured]**: the published edit is exactly `dW[:, j] = -6.000 (u . W_j) u` on 64 columns of `mlp.down_proj` in layers 15 and 19 (`u` the left singular vector of the edit, `W_j` column `j` of the base); `x1e-1` does not reproduce the behaviour (13.1) and `e19` alone is partial (13.2). Nothing here runs W1 behaviour (it costs L4 money): the result is a candidate edit and numbers.

## 1. The question

Given only the base model and, for the data arm, a few of its own generations that loop (and a few that do not), can a procedure with no access to the published checkpoint output (a) a direction `u` in the 1,024-d residual space per layer with `|cos(u_found, u_pub)|` close to 1, and (b) 64 of the 3,584 columns of `down_proj` that overlap the published 64? The published `u_pub` and columns are used **only to score**, never to choose anything except where a section says so (the checkpoint choice of 5, marked).

## 2. Data (read-only; CPU; few tokens)

* **Source:** `results/w1_cloud/data/base_8192.jsonl` of the main checkout (200 base generations, field `hit_limit` marks loops: 179 do, 21 do not; **[read]**, never written or moved). Texts are the model's own generations without the prompt; they are re-tokenised with the base tokenizer (the ids differ from the generated ones only where the text round trip is not exact). **[choice]** no prompt is prepended: the contrast is about the loop, and the prompt would put 100 to 300 tokens of identical text in every window.
* **Loop windows (8):** the first 8 rows by `id` (numeric) with `hit_limit` true and `rep4` (`w4_lib.rep4`) of the last 512 tokens at least 0.5. Window = the last 1,024 tokens of the text; the counted positions are the last 512 (the loop is established by then).
* **Non-loop windows (16):** (i) the first 8 rows by `id` with `hit_limit` false and at least 700 tokens: window = the first 1,024 tokens, counted positions 512 to the end of the window; (ii) the first 1,024 tokens of each of the 8 loop texts, counted positions 512 to 1,023 (the same texts before their loops). **[choice]** counting from 512 equalises the depth in the window of both classes.
* **Forward:** the base's text tower in float32 on the CPU (`examples/qwen35/hf08_ref.py build_model`), `torch` limited to 4 threads, no GPU or MPS, stopped after layer 19 (a hook raises, which saves a fifth of the work). Per layer 15 and 19, hooks on `mlp.down_proj` capture, at the counted positions, its **input** `a` (3,584-d) and its **output** `y` (1,024-d, the input to the residual add). 24 windows of 1,024 tokens, one forward each, **[estimate]** a few minutes in total. Captured arrays are written under `build/w4s/` (not committed).
* **Counts expected:** about 4,096 loop and 8,192 non-loop positions per layer. This is small beside the 1,024-d and 3,584-d problems: stability is measured (3.4), not assumed.

## 3. The candidates

Notation as in `weight-space-w4.md` 6: `W` is the base `down_proj` [1,024 x 3,584], `W_j` its column `j`, `C_L`, `C_N` the (uncentred) second moments of `y` over loop and non-loop positions, `d = 1,024`.

### 3.1 Directions `u` (unit, sign irrelevant), per layer

Data:
* **D1 contrastive eigenvector** (the proposal's, 6.2): top eigenvector of `C_L v = lambda (C_N + 1e-4 tr(C_N)/d I) v`. **This is the primary candidate.**
* **D2 difference of means:** `mean_L(y) - mean_N(y)`, normalised.
* **D3 difference of second moments:** the eigenvector of `C_L - C_N` with the largest `|eigenvalue|`.
* **D4 loop-only:** the top eigenvector of `C_L` (what dominates the looping positions, ignoring the contrast).

Weights only (no activations, no text):
* **W1** top left singular vector of `W`.
* **W2** top left singular vector of `W` with its columns scaled to unit norm.
* **Subspace capture (not a candidate, a diagnostic):** `|| P_k u_pub ||` for `P_k` the projector on the top-`k` left singular vectors of `W`, `k` in {1, 4, 16, 64}, against the chance value `sqrt(k/d)` for a random unit vector.

Controls (what a cosine means here): 200 random unit vectors, whose `|cos|` with `u_pub` is the null (median about 0.02); and **label permutations**: 50 random permutations of the window labels (loop/non-loop at window level, keeping 8 loop and 16 non-loop), the D1 to D4 directions recomputed, `|cos|` with `u_pub` recorded. A data candidate "beats the null" iff its `|cos|` exceeds the maximum over the 50 permutations. **Stability:** the windows split into two halves (alternate windows within each class), D1 to D4 computed on each half, `|cos|` between the halves.

### 3.2 Column rules (each takes `u` and returns 64 of 3,584)

* **R1** the 64 largest `|u . W_j|`. **R2** the 64 largest `|u . W_j| / ||W_j||`. (Weights only; these are rules R1 and R2 of `weight-space-w4.md` 6.3.)
* **Rrms** the 64 largest `|u . W_j| rms_j`, `rms_j` the root mean square of neuron `j`'s input over all counted positions (the post-hoc rule of 11.6 item 3, which gave 36 and 21 of 64 with the published `u` on the fit corpus). Needs activations, not the contrast.
* **R3** the 64 largest `|mean_L(a_j) - mean_N(a_j)| |u . W_j|` (6.3 R3).
* **R5** (no `u`): the top eigenvector `v` of the same generalised problem as D1 in the 3,584-d `a` space (`C_L^a`, `C_N^a + 1e-4 tr/d_a I`; the matrices have rank at most 4,096 and 8,192, so the ridge is what makes it defined), the 64 largest `|v_j|`.
* **R0** random 64, 200 draws (chance: 1.14 overlapping columns; the distribution is reported).
* **Oracle references** (not candidates): every rule above with `u_pub` instead of a found `u`, which separates "the direction was wrong" from "the rule cannot find the columns". With `u_pub` and the fit-corpus `rms`, R1 gives 31 / 18 and Rrms 36 / 21 (11.1); the run reproduces R1's 31 / 18 exactly (it is weights only) as a check of the code, and its Rrms differs because the windows are not the fit corpus.

### 3.3 Scores

Per layer and per (direction, rule): `|cos(u_found, u_pub)|`; `overlap` = number of the 64 chosen columns that are among the published 64; `edit cosine` = Frobenius cosine between the candidate edit (`dW[:, j] = -6 (u_found . W_j) u_found` on the chosen columns, `alpha = 6`, the recipe's constant) and the published `dW`. The last is the single number that says how close the candidate *edit* is.

## 4. The success rule (fixed now; **[choice]**, with the reason)

All of it is evaluated for **every** (direction, rule) pair; 5 data directions and weights directions times at most 6 rules is at most 36 pairs, and the thresholds are far above what chance reaches (the null of 3.1 is reported next to them), so the number of pairs is not a multiplicity worth correcting.

* **RECOVERED** (the recipe is rediscovered): some pair has, **in both layers**, `|cos| >= 0.9` and `overlap >= 48` of 64. (0.9 because `x1e-1` and `e19` show that an edit as far as half the published amplitude from it does not reproduce the behaviour; 48 of 64 is `weight-space-w4.md` H2's threshold.)
* **PARTIAL** (not RECOVERED, but something survives), if either: **(A)** some **data** direction has `|cos| >= 0.7` in both layers (the "direction found" rule of 6.2) **and** beats its permutation null in both layers; or **(B)** some pair has `|cos| >= 0.5` in both layers (and beats the null if it is a data direction) **and** `overlap >= 32` of 64 in both layers.
* **NOT RECOVERED** otherwise: the numbers are reported, **no checkpoint is written**.
* **What does not count:** a weights-only direction (W1, W2) can satisfy the cosine clause of RECOVERED or (B) (it needs no null: it is a deterministic function of `W`); it cannot satisfy (A), which is about finding a direction in data.
* **The overlap clause for a direction that is not `u_pub`** is evaluated with that direction's own rule output; the oracle references never count.

## 5. Checkpoints (only if 4 says RECOVERED or PARTIAL; at most 2)

Written with `w4_ckpt.write_checkpoint` (its `edits` argument, added for this: the same bf16 write, read-back check, manifest and patch as section 13 of `weight-space-w4.md`), into `build/ckpt/` of the worktree, never committed:
* **`s1`**: the **best** pair by the minimum over the two layers of `|cos|`, ties by the minimum overlap; `alpha = 6`, its own `u` and its own 64 columns, both layers. **The choice of "best" uses the published `u` and columns (it is a ranking by score), so `s1` is the best of the candidates, not what an unaided search would have picked.** The unaided pick is the primary candidate D1 with the rule Rrms; if that pair is not the best, the report says so and gives its numbers.
* **`s2`** (diagnostic hybrid, an oracle mixture): if `s1`'s direction has `|cos| >= 0.7` in both layers, the same direction with the **published** columns; otherwise the **published** direction with `s1`'s columns. This isolates which half of the recipe is missing. It is not a result about search, it is the control for what `s1` could do.
Behaviour of either is not run here (L4 money); 9 of `weight-space-w4.md` 13 describes how `w1.py behave` is pointed at a directory.

## 6. What the run cannot show

* Texts are the model's own generations without prompt, re-tokenised; 8 loop windows from one run of one sample per problem. The loops of the 179 are not all the same loop.
* A direction can have a large cosine with `u_pub` and a different effect (and the reverse); the cosine and overlap are the weights-space facts, the behaviour is the L4's.
* The contrast is of MLP outputs at two layers; the proposal's R4 (gradient attribution) needs backward passes through a 0.8B model and is **not run** here (CPU cost; the other agent owns the GPU).
* Nothing here says the authors' procedure was a contrast at all; the 6.000 and the support are the facts.

## 7. Results (measured 2026-11-06; nothing above this line was edited after the run)

Run: `w4_search.py capture` (74 s on 4 CPU threads, no GPU or MPS; 24 windows of 1,024 tokens, 4,096 loop and 8,192 non-loop counted positions per layer; loop rows 0,1,2,3,4,7,8,9, finished rows 6,18,25,26,50,55,61,69; sanity **[measured]**: captured `y` equals `a @ W^T` to 1.6e-6), then `w4_search.py analyse` (46 s). Data: `examples/weightspace/results/w4_search.json` (the activations stay in `build/w4s/`, not committed). Tests: `tests/w4_search.py` (gate; skips without numpy, checked with a shadow `numpy.py`).

### 7.1 Verdict by the rule of section 4: **NOT RECOVERED**. No checkpoint was written (`w4_search.py write` refuses with `w4s-rule`, as registered).

### 7.2 Directions: `|cos(u_found, u_pub)|` (layer 15 / layer 19)

| candidate | cos | permutation-null max (50) | half vs half |
|---|---|---|---|
| D1 contrastive eigenvector (the primary) | 0.140 / 0.105 | 0.089 / 0.103 | 0.53 / 0.51 |
| D2 difference of means | 0.350 / 0.439 | 0.346 / 0.445 | 0.95 / 0.98 |
| D3 difference of second moments | 0.398 / 0.439 | 0.403 / 0.449 | 0.96 / 0.99 |
| D4 loop-only top eigenvector | 0.405 / 0.415 | 0.409 / 0.406 | 0.97 / 0.99 |
| W1 top singular vector of `W` | 0.101 / 0.308 | n/a | n/a |
| W2 the same, unit-norm columns | 0.168 / 0.310 | n/a | n/a |
| random unit vector (max of 200) | 0.090 / 0.092 | | |

* **No candidate reaches 0.7, let alone 0.9.** The best is 0.41 / 0.44.
* **The contrast direction D1, the proposal's, is noise-level**: 0.14 / 0.105 against a permutation maximum of 0.089 / 0.103 (it barely beats the null in layer 15, not in layer 19) and unstable between two halves of the windows (cos 0.5). With 8 loop windows from one run the generalised eigenproblem in 1,024-d is not determined.
* **D2 to D4 are stable (0.95 to 0.99 between halves) and reach 0.35 to 0.44, but so do the same estimators on randomly relabelled windows** (null maxima 0.35 to 0.45): they find the **dominant direction of the layer's MLP output at any position**, not a loop direction. Only D2 in layer 15 exceeds its null maximum (0.350 against 0.346), and no data candidate does so in both layers: clause (A) fails on the null as well as on the 0.7.
* The weights alone: the top singular vector of `W` has cos 0.10 / 0.31 with `u_pub`. The span of the top 1 / 4 / 16 / 64 left singular vectors holds 0.10 / 0.29 / 0.32 / 0.37 (layer 15) and 0.31 / 0.32 / 0.33 / 0.36 (layer 19) of `u_pub`, against chance 0.03 / 0.06 / 0.125 / 0.25: `u_pub` is a little more spectral than chance and is not a leading singular direction.

### 7.3 What `u_pub` is made of (exploratory, added after the first run; not registered, read as pointers)
* `|cos|` of `u_pub` with the mean MLP output over all counted positions: 0.357 / 0.300. Share of `u_pub` in the top 1 / 4 / 16 / 64 eigenspace of the all-position second moment: 0.40 / 0.41 / 0.44 / 0.50 (layer 15), 0.34 / 0.48 / 0.50 / 0.53 (layer 19); chance 0.03 / 0.06 / 0.125 / 0.25. So `u_pub` has a component of about 0.35 to 0.4 on the layer's dominant output direction and **most of it (cos^2 about 0.8) lies elsewhere**: not recoverable from the activations' principal axes either.

### 7.4 Columns: overlap with the published 64 (layer 15 / layer 19), chance 1.3 / 1.2 (p99 4, max 5 of 200 draws)

| direction \ rule | R1 | R2 | Rrms | R3 | R5 |
|---|---|---|---|---|---|
| D1 (primary) | 5 / 1 | 6 / 2 | **16 / 9** | 17 / 9 | 2 / 0 |
| D2 | 22 / 17 | 24 / 17 | 31 / 21 | 30 / 19 | 2 / 0 |
| D3 | 29 / 18 | 32 / 16 | 33 / 18 | 33 / 19 | 2 / 0 |
| D4 | 30 / 17 | 34 / 17 | 33 / 19 | 33 / 19 | 2 / 0 |
| W1 | 3 / 16 | 3 / 17 | 20 / 21 | 20 / 20 | 2 / 0 |
| W2 | 7 / 16 | 8 / 18 | 21 / 20 | 25 / 20 | 2 / 0 |
| **oracle: `u_pub`** | 31 / 18 | 33 / 20 | 43 / 22 | 40 / 24 | |
| rms alone (no `u`) | 23 / 12 | | | | |

* **48 of 64 is not approached by any pair** (best 34 / 21). The oracle row shows why: **even with the true `u`, these rules reach 31 to 43 of 64 in layer 15 and 18 to 24 in layer 19** (R1's 31 / 18 reproduces 11.1 exactly: a check of the code). The column list is a second thing that none of these scores recovers; the direction error comes on top.
* The 30 to 34 of layer 15 for D3 / D4 is the dominant output direction's own alignment (it is as good as the oracle R1 there), not a sign of recovered selection: layer 19, where that direction is not what `u_pub` is, gives 16 to 19. rms alone gives 23 / 12, most of what Rrms adds.
* The neuron-space contrast R5 (the proposal's) recovers 2 / 0 columns, chance level. **Edit cosine** (Frobenius cosine between the candidate edit at alpha 6 and the published one): best data pair 0.26 / 0.29 (D4 or D3 with R3 / D2 with Rrms), W1 with Rrms 0.006 / 0.149; the unaided pick (D1 with Rrms) 0.026 / 0.023; the oracle (true `u`, Rrms) 0.79 / 0.67. The best pair by the registered ranking is **D4 with Rrms** (cos 0.405 / 0.415, overlap 33 / 19), which is not the unaided pick.

### 7.5 Reading
1. **The recipe was not rediscovered from the base's weights and a few of its own loops.** The proposal's contrastive direction (D1) is at noise level and unstable at this sample size; the stable data directions are the layer's dominant output axis, which the published `u` only partly shares (cos about 0.4) and which does not depend on the loop labels; the pure-weights directions are no better.
2. **What survives** is small and honest: (i) the published `u` is not a leading singular direction of `W` nor of the activations, so a spectral shortcut does not find it; (ii) loudness of the neuron (rms) plus alignment with any direction near `u` recovers a third to a half of layer 15's columns and about a third of layer 19's, but even the true `u` does not do better than 43 / 22; (iii) the contrast machinery is not adequate at 8 windows: whether it would work with the full 179 loops and more tokens, or with R4 (gradients), is **[open]** and not tested here.
3. **Cost of the answer:** 74 s of capture and 46 s of analysis on 4 CPU threads; the next step, if any, is more windows (all 179 loops, 4 to 5 windows each, about an hour of CPU) and the gradient rule R4, both still CPU; no L4 is justified by these numbers.

## 8. Deviations and limits
* None of sections 1 to 6 was changed after the run. Window rows were fixed by the rule: the loop rows skip row 5 (a `hit_limit` row whose last 512 tokens failed the `rep4` >= 0.5 test; row 6 is a finished row).
* `w4_ckpt.write_checkpoint` gained an `edits=` argument (for section 5; unused by the verdict, tested in `tests/w4_search.py`), and `analyse_layer` takes the number of columns from the published support (64), as the registered rule means.
* 7.3 is post hoc (added after seeing that D2 to D4 behave like their nulls). The permutation null relabels whole windows; the 8 early windows come from the same texts as the loop windows, so the null is conservative for D2 to D4 only if the loop texts share a style, which was not examined.
* R4 (gradient attribution) was not run (section 6). The non-loop and loop windows are from different texts, so some of the contrast is topic and not loop, which would make D1 worse, not better.
