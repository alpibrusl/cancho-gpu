# M11: can a model write the weights of a network we then run in cancho? (design, written before any run)

> **Status: a design.** Nothing here is built or run. Marks as in `roadmap-models.md`: **[measured]**, **[read]** (from a document or source, not run), **[estimate]**, **[choice]**, **[open]**. The decision rules in sections 5 to 7 are fixed now and are not edited after a result is seen; facts that turn out different are corrected in place, with the date.
> Order against the other open work (maintainer, 2026-11-04): M8's training and the MLX 27B step (`roadmap-models.md` 9.8, unknown 2) come first and own the GPU. **Everything in W0 and W2 below runs on the CPU path of the session, so it does not compete for the GPU**; W1 does, and waits.

## 1. The claim, and how well we know it

**What was read.** "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2026-11-02, `rikiyat.github.io/blog/weight-space-programming/`). **[read]** The page was read through a page-to-text summariser, **twice, not in the original and not from the logs**; every figure below is as that summary gave it and must be checked against the authors' logs (`github.com/RikiyaT/lm_understand-logs`) before this document relies on it.
Their model is "GPT-6 Astra" (the maintainer: the latest GPT model). Their claims, in three kinds:
* **Read**: from weights as text, name the game a 17,253-weight policy was trained on (4 of 4); the language of 7.2 M-parameter character transformers from the middle blocks only (3 of 3), with the copying heads found (no false ones).
* **Fix**: add a small side path to a stalled control policy (0 of 256 episodes → 1,024 of 1,024, 3.5% of the weights changed); and in Qwen3.5-0.8B (thinking on) subtract one direction written by two MLPs from 64 output columns per layer, 131,072 weights (0.016%): MATH-500 7.9% → 48.1%, GSM8K 13.8% → 67.1%, IFEval 21.0% → 62.3%, token-limit hits 93% → 33%. The same method on Gemma 3 270M, which rarely loops (8.5% of answers hit the limit), gained at most about one point.
* **Write**: weights of a 600→32→32→6 network for MinAtar games from nothing but the rules and the source, in one reply (Breakout 517.4 against PPO's 25.3 at 500 M steps; Seaquest and Asterix below or near PPO-500M); a 1.48 M-weight Go network written as a 336-line Python program, with forward-pass budgets of 0 to 50 (each pass: up to 128 games against a practice opponent, or 256 positions).
The rule that makes these claims mean something: **the model may not train.** It reasons about what the weights compute and writes them; numpy or torch on CPU is allowed for analysis, a forward pass is a counted, scored evaluation.

**What it does not say** (the post's own limits, **[read]**): editing unmasks an ability that is hidden, it does not add one (the Gemma control); reading from weights alone failed for a sycophancy fine-tune until an attention-pattern tool was added; sample sizes are small and some numbers are one run or the best of a few; the Astra model is not one we can run.

## 2. Why this is a question for cancho, and what is not

What cancho-gpu has that the experiments need **[read from this repository]**: a session whose forward program is a `.chx` program over weights read from binary f32/f16 files (the `resident`/`input` directives, `session` in `run.md`), a deterministic answer printed by `answer`, an interpreter (CPU) path that gives the same forward on the same files, and a training stack that fits the same architecture by gradient descent (M1 to M7). So **one forward program and one set of files serve both sides of the comparison the post never makes: weights written by reasoning against weights found by gradient descent, same architecture, same scorer.** The post compares against PPO networks trained elsewhere; here the trained arm is ours and the forward is exact.

What is not a question for us: reading experiments on networks we train (the post's Exp 1 to 3) say little about the roadmap; one is kept as an optional appendix (W3) only because the harness makes it nearly free.

## 3. What can honestly be tested, and what cannot

* **Not testable: anything about Astra.** We have no access to it **[open: if the maintainer runs it, the harness below produces the same logs]**. The writer we have is Claude, in a sandboxed agent. A result says what *a* frontier model can do in this harness, not whether the post's numbers are right.
* **Testable without a writer model: whether the post's most striking number reproduces** (W1): the edited Qwen3.5-0.8B and its base are public.
* **Testable with our writer: whether weight-writing works at all for a task with an exact oracle in our own repositories** (W2).

## 4. The harness (the one piece of code; nothing else is built before W0)

`examples/weightspace/harness.py` (Python, in cancho-gpu; **[choice]**), driven by an agent whose only tools are the harness's:
* `show(tensor, start, count)` → the weights as text, and `describe()` → the architecture (shapes, activations) and the task statement.
* `write(tensor, python_source)` → runs the source in a **sandbox** (CPU, numpy, no network, no access to `build/` or the session binaries, no files outside the run directory, a time limit) and stores the tensor it returns as a binary f32 file. The agent never receives the scorer's samples.
* `forward(n_poses)` → **counted**: runs the session over `n_poses` held-in samples drawn fresh each call and returns only the **scalar score** (mean error), never the per-sample outputs. A budget B is fixed per arm; calls beyond it are refused. Budget accounting is the harness's, not the agent's.
* The agent's python has no `cancho-gpu*` binary, so it cannot reach the session except through `forward`.
* **Every call is logged** (the full text exchanged, the code run, the score, token counts), in the format of the authors' logs where they can be matched **[open until the logs are read]**.

**Compliance is by review, as in the post, not by construction, and that is a weakness stated now.** A sandbox cannot stop an agent from computing the answer in numpy and then writing weights that merely reproduce the computed table by least squares, which is training under another name. The rule, fixed here: a run **violates** if any parameter is set by solving or searching for it against computed outputs of the target (`lstsq`, `scipy.optimize`, autograd, a loop that adjusts a parameter to reduce a computed error). Setting a parameter by a closed formula from the structure of the problem (an explicit rotation, a tabulated function at fixed knots) is allowed. Borderline runs are reported **as borderline, with the code**, not dropped silently. Violating runs are excluded from the headline and counted. The scalar scores from `forward` are the only feedback; at B = 50 it is a weak search signal, which is the intended handicap.

## 5. W0: free diagnostic before spending anything (CPU, no GPU, no download)

The post's repair works on a model that **loops**. Our M8 tuned model may loop too. From the journals and model outputs of the M8 arms (`cancho-code/eval/results/`, the runs under `m8-runs/`): the share of model replies that hit `--max-tokens` and the share whose tail repeats a span of at least 64 characters more than 3 times (a repeat detector written for this, tested on a planted repeat). Reported per arm. **Decision**: if more than 10% of any arm's replies loop or hit the limit, a looping fault is plausibly present and W1's method is relevant to M8; if under 2% in every arm, it is not, and W1 stays a standalone check. Between is reported as inconclusive. Cost: minutes.

## 6. W1: does the post's Qwen3.5-0.8B edit reproduce? (needs the GPU and two downloads)

**Needs the maintainer's permission to download** the base Qwen3.5-0.8B and the edited model `RikiyaT/Qwen3.5-0.8B-weight-space-programming` (sizes unchecked, **[estimate]** about 1.6 GB each in bf16). Run after M8 and the MLX 27B step, with MLX if `mlx-lm` 0.31.3 loads the architecture **[open: not tried]**, else transformers on MPS.
* **Protocol** (a reduced version of the post's: the post's 32,768-token cap with 4 samples over 500 problems is about 65 M generated tokens for a model that hits the cap 93% of the time, which at an assumed 200 tokens a second is days **[estimate]**): GSM8K test, the first 200 problems, one sample each; thinking on; temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5 (the post's); cap 8,192 tokens; both models, same seeds. The reduced cap favours finding an effect (a looping model loses more answers to a lower cap), so a second run at 16,384 tokens on a 60-problem subset checks that the effect is not made by the cap.
* **Statistic and rule (pre-registered)**: accuracy A and token-limit rate L, each with a 95% bootstrap interval (10,000 resamples of the 200 problems, numpy seed 20261104). **Reproduces** iff the edited model's accuracy exceeds the base's by more than the base's interval half-width **and** L falls by more than 20 points. If only accuracy rises or only L falls, reported as partial with both numbers. If neither, "not reproduced at this protocol", which says nothing about the post's full protocol.
* **Also**: the edit is a few lines of weight arithmetic. If the authors' logs give the layers and the direction, apply it to our base ourselves and compare with their published model (a check that the published weights are what the logs describe): the number of weights that differ and their positions.

## 7. W2: weights written by reasoning against weights found by gradient descent, on a task with an exact oracle (CPU)

**Task.** The SO-101 forward kinematics of `cancho-robot` (`src/kinematics.cho`, generated from `tests/fixtures/urdf/so101_new_calib.urdf`; differential tests C4 hold it to 0.1 mm against placo **[read]**): the five joint angles (radians) → the gripper's position (x, y, z, metres). **[choice]** Not a product: the exact function exists and the controller keeps it. It is a testbed because the oracle is exact and free, the structure (rotations about fixed axes, link offsets) is textbook for a model to program, and it is small enough to inspect.
**Network.** 5 → 64 → 64 → 3, tanh **[choice]** (tanh gives smooth nonlinearities from which products of sines and cosines can be built by a writer; ReLU would force a piecewise-linear table). About 4.6 k weights, so the weights fit in text, and the writer may also write Python that generates them.
**Held-out sets (fixed before any run).** Poses sampled from the URDF joint limits, with the M9 split: an **interior** set, and a **band** set (a held-out interval of `shoulder_pan` and the extreme third of `wrist_flex`), 2,000 poses each, fresh seeds, never shown to any arm or to `forward` (which draws its own).
**Metric.** Mean Euclidean position error in millimetres, per set, with a 95% bootstrap interval (10,000 resamples, seed 20261104).

**Arms.**
| arm | what | notes |
|---|---|---|
| Z | all weights zero | the null; the error of predicting the mean |
| R | random init (the trainer's) | a second null |
| G(N) | the same network trained by cancho-gpu's training stack on N sampled poses, N in {10^3, 10^4, 10^5, 10^6}, Adam, the M1 recipe | the gradient-descent curve; uses the CPU session path, or the GPU if free |
| W(B) | the writer's network, B forward passes, B in {0, 10, 50}, 3 independent runs each | the weight-writing arm |
| T | the writer is asked to write the *kinematics as Python* and not a network | a control: if T is exact and W(0) is not, the bottleneck is encoding, not understanding |

**Pre-registered reading.**
* **W(B) beats gradient descent at N** iff the median over its 3 runs of the interior error is below G(N)'s lower interval bound. We report the largest N beaten. Because W has 3 runs, no confidence statement is made on W; the three values are shown.
* **W is useful** iff some W(B) has interior error under 5 mm and band error within 2x of the interior error (band-error ratio shows whether the structure was captured or a table was memorised). **W is a failure** iff all W(B) have errors over Z's. Between: reported with the values and the code.
* **Cost is reported either way**: tokens, wall time, forward passes used, and the gradient arms' samples and seconds. A win in error that costs a long agent run is not a win to be sold as cheap.
* Outcomes the post's story predicts and ours may not: W(0) already good (the writer knows the physics) and gains from forward passes small; or W bad because tanh networks cannot represent exact rotations cheaply, and G wins at 10^5. Both are acceptable results.

**What a success would mean for cancho-robot, honestly.** Nothing yet. It would show a model can program a small network from a spec and a sandbox, which bears on whether the pose estimator's early layers (edge and capsule detectors, `roadmap-models.md` 8) could be initialised by a writer rather than from random weights. That is the next step, **not part of this milestone [open]**, and it needs the synthetic renderer of section 8.3 (V3), which does not exist.

## 8. W3 (optional appendix): reading

If W2's harness exists, `show` over the weights of two of our own networks (the M1 GPT trained on Shakespeare against one on another corpus; the 6-layer ones, 10.8 M parameters) asks "which corpus" with no tools. Pre-registered at 3 networks and 2 attempts; chance is 1 in 3. Reported, not acted on. Skipped if W2 is a failure.

## 9. Order and cost

| step | needs | cost | gate |
|---|---|---|---|
| W0 repeat detector on M8 journals | M8's results | minutes | decision in section 5 |
| W2 harness, then arms Z, R, G | CPU session | harness ~400 lines **[estimate]**; G(10^6) minutes | none until arms exist |
| W2 arm T, then W(0), W(10), W(50), 3 runs each | an agent with the harness | 12 agent runs **[estimate]**; each tens of minutes | section 7 |
| W1 reproduction | M8 done, MLX step done, download permission | hours of generation | section 6 |
| W3 | W2 | small | none |

The harness and its tests (a planted violator that fits by `lstsq` must be caught by the review checklist; a `forward` beyond the budget must be refused; the sandbox must refuse a file outside the run directory and a network call) are written before any agent run. No compiler change is needed or planned.

## 10. What this document does not claim

* Nothing about GPT-6 Astra or the authors' numbers beyond the one reproduction in W1, which is reduced.
* That a model writing weights is useful: W2 compares error and cost on one task.
* That one task generalises: the kinematics function is a favourable case (exact, structured, known to the writer).
* That compliance with "no training" is enforced: it is reviewed (section 4).
