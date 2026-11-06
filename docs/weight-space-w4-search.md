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
