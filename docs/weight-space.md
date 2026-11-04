# M11: can a model write the weights of a network we then run in cancho? (design, written before any run)

> **Status: a design; W2a (the harness) and W2b's arms Z, R and G(N) are built and measured, section 11 (2026-11-04). No writer arm (W, T) and no W0/W1 has been run.** Marks as in `roadmap-models.md`: **[measured]**, **[read]** (from a document or source, not run), **[estimate]**, **[choice]**, **[open]**. The decision rules in sections 5 to 7 are fixed now and are not edited after a result is seen; facts that turn out different are corrected in place, with the date.
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

### 5.1 W0 made exact: the detector's definition (pre-registered 2026-11-04, before the detector was run on any M8 output)

Written before any loop number was computed from a real run. Code: `examples/qwen/m8_loops.py`; tests: `tests/m8_loops.py` (planted loops, planted non-loops, malformed journals). The definition below is not edited after a result is seen; a fact that turns out different is corrected here, with the date.

* **Unit.** A *reply* is one model reply in a run: one `=== reply (stopped because: X)` section of the run directory's `raw.txt` (cancho-code's log of every sample the model returned; the arm runner is `cancho-code/eval/m8_arm.py`, the run layout is `<arm dir>/<run dir>/`). If a run has no `raw.txt`, its replies are the lines of `usage.jsonl` (stop reason only; no text, so no repeat test: counted in `replies_without_text`). A run dir with neither is `unreadable` and is counted, not raised on. Malformed sections or lines are counted (`malformed`) and skipped.
* **Indicator L (hit the limit).** The stop reason is `length` (`usage.jsonl`: `"stop":"length"`).
* **Indicator R (tail repeat).** After removing trailing whitespace, and looking only at the last 16,384 characters, the reply **ends in a span of at least 64 characters repeated more than 3 times**: there is a period p, 64 <= p <= 4,096, such that the last 4p characters are p-periodic (the last copy may be cut short: a cut-off loop ends mid-copy, so "4 copies" is counted as 4p characters of periodic tail). A run of one repeated character 256 or more long therefore counts (period 64); that is a degenerate loop and is meant to count. Exact characters, case and whitespace included; no fuzzy matching.
* **Reply loops-or-limits** = L or R. This is the statistic of the decision. Its arm value is (replies with L or R) / (replies), with a 95% Wilson interval. L and R are also reported separately.
* **Reported, not in the decision.** (a) The share of *runs* with at least one L-or-R reply (replies inside a run are not independent: a cut-off raises the bound for the next round, `cancho-agent`'s 2,048, 4,096, 8,192, ...); (b) `repeated_reply`: replies byte-identical to the previous reply in the same run (a repeated *action*); (c) `repeated_signature`: consecutive attempts of one unit in `journal/events.jsonl` with the same non-empty refusal signature (the model made the same error again). (b) and (c) are reported because the issue's question is "repeats the same action/text"; they do not move the decision.
* **Decision (section 5, unchanged):** more than 10% of any arm's replies in L-or-R: a looping fault is plausibly present; under 2% in every arm: not; between: inconclusive. Arms with fewer than 100 replies are marked `small` and the rule is not applied to them (reported only).
* **Deterministic and CPU-only:** standard library only, no model, no network, no GPU. Run: `python3 examples/qwen/m8_loops.py A=<dir of arm A> B=<dir of arm B> ... [--json out.json]`.
* **What a result would not show.** That a base or tuned model loops under `--max-tokens 2048` says the *sampling setup* produces cut-offs; it does not by itself show the post's mechanism. Reading beyond the three numbers is for the study, not this detector.

## 6. W1: does the post's Qwen3.5-0.8B edit reproduce? (needs the GPU and two downloads)

**Needs the maintainer's permission to download** the base Qwen3.5-0.8B and the edited model `RikiyaT/Qwen3.5-0.8B-weight-space-programming` (sizes unchecked, **[estimate]** about 1.6 GB each in bf16). Run after M8 and the MLX 27B step, with MLX if `mlx-lm` 0.31.3 loads the architecture **[open: not tried]**, else transformers on MPS.
* **Protocol** (a reduced version of the post's: the post's 32,768-token cap with 4 samples over 500 problems is about 65 M generated tokens for a model that hits the cap 93% of the time, which at an assumed 200 tokens a second is days **[estimate]**): GSM8K test, the first 200 problems, one sample each; thinking on; temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5 (the post's); cap 8,192 tokens; both models, same seeds. The reduced cap favours finding an effect (a looping model loses more answers to a lower cap), so a second run at 16,384 tokens on a 60-problem subset checks that the effect is not made by the cap.
* **Statistic and rule (pre-registered)**: accuracy A and token-limit rate L, each with a 95% bootstrap interval (10,000 resamples of the 200 problems, numpy seed 20261104). **Reproduces** iff the edited model's accuracy exceeds the base's by more than the base's interval half-width **and** L falls by more than 20 points. If only accuracy rises or only L falls, reported as partial with both numbers. If neither, "not reproduced at this protocol", which says nothing about the post's full protocol.
* **Also**: the edit is a few lines of weight arithmetic. If the authors' logs give the layers and the direction, apply it to our base ourselves and compare with their published model (a check that the published weights are what the logs describe): the number of weights that differ and their positions.

## 7. W2: weights written by reasoning against weights found by gradient descent, on a task with an exact oracle (CPU)

**Task.** The SO-101 forward kinematics of `cancho-robot` (`src/kinematics.cho`, generated from `tests/fixtures/urdf/so101_new_calib.urdf`; differential tests C4 hold it to 0.1 mm against placo **[read]** *(corrected 2026-11-04: the test, `tests/differential_kinematics.py`, gates at 1e-9 m, a nanometre; 0.1 mm was the proposal the measurement replaced **[read]**; section 11.3 checks the numpy reference below against `kinematics.cho` itself)*): the five joint angles (radians) → the gripper's position (x, y, z, metres). **[choice]** Not a product: the exact function exists and the controller keeps it. It is a testbed because the oracle is exact and free, the structure (rotations about fixed axes, link offsets) is textbook for a model to program, and it is small enough to inspect.
**Network.** 5 → 64 → 64 → 3, tanh **[choice]** (tanh gives smooth nonlinearities from which products of sines and cosines can be built by a writer; ReLU would force a piecewise-linear table). About 4.6 k weights, so the weights fit in text, and the writer may also write Python that generates them.
**Held-out sets (fixed before any run).** Poses sampled from the URDF joint limits, with the M9 split: an **interior** set, and a **band** set (a held-out interval of `shoulder_pan` and the extreme third of `wrist_flex`), 2,000 poses each, fresh seeds, never shown to any arm or to `forward` (which draws its own).
**Metric.** Mean Euclidean position error in millimetres, per set, with a 95% bootstrap interval (10,000 resamples, seed 20261104).

**Arms.**
| arm | what | notes |
|---|---|---|
| Z | all weights zero | the null; the error of predicting the mean *(corrected 2026-11-04: all-zero weights output exactly (0, 0, 0) m, the base frame's origin, not the mean position; the error of the mean position is 308.3 mm interior, 271.5 band, section 11.5)* |
| R | random init (the trainer's) | a second null |
| G(N) | the same network trained by cancho-gpu's training stack on N sampled poses, N in {10^3, 10^4, 10^5, 10^6}, Adam, the M1 recipe *(2026-11-04: the schedule is M1's, 5% warm-up and a cosine; the learning rate is 1e-2, chosen on a validation set, not M1's 1e-3; section 11.5)* | the gradient-descent curve; uses the CPU session path, or the GPU if free |
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
| W2 harness, then arms Z, R, G | CPU session | harness ~400 lines **[estimate; measured 2026-11-04: 591 lines for the harness, sandbox and review checklist, 1,222 with the task, trainer, arms and kernels]**; G(10^6) minutes **[measured: 347 s on the CPU session, section 11.5]** | none until arms exist |
| W2 arm T, then W(0), W(10), W(50), 3 runs each | an agent with the harness | 12 agent runs **[estimate]**; each tens of minutes | section 7 |
| W1 reproduction | M8 done, MLX step done, download permission | hours of generation | section 6 |
| W3 | W2 | small | none |

The harness and its tests (a planted violator that fits by `lstsq` must be caught by the review checklist; a `forward` beyond the budget must be refused; the sandbox must refuse a file outside the run directory and a network call) are written before any agent run. No compiler change is needed or planned.

## 10. What this document does not claim

* Nothing about GPT-6 Astra or the authors' numbers beyond the one reproduction in W1, which is reduced.
* That a model writing weights is useful: W2 compares error and cost on one task.
* That one task generalises: the kinematics function is a favourable case (exact, structured, known to the writer).
* That compliance with "no training" is enforced: it is reviewed (section 4).

## 11. W2a and W2b (arms Z, R, G) built and measured, 2026-11-04

Everything in this section was produced by the files of `examples/weightspace/` and `tests/weightspace.py` (in the gate), on an Apple M4 Max, numpy 2.4.6, Python 3.14.5, the compiler `a49ea779` as built locally (`-dirty`, so `--ignore-compiler-rev`). The M8 training job owned the GPU and some CPU during all of it; **nothing here used the GPU** (the interpreter session, `build/cancho-gpu-session`, and numpy). The decision rules of sections 5 to 7 are unchanged.

### 11.1 What was built

| file | what |
|---|---|
| `task.py`, `so101_chain.py`, `gen_chain.py` | the task: the SO-101 chain copied from the URDF (with its sha256), the numpy float64 reference `fk`, the pose space and its split, the fixed seeds, the network's shapes, `bootstrap_ci` |
| `check_fk.py` | task.fk against cancho-robot's `kinematics.cho` (builds its test driver into a temporary directory; cancho-robot is read, not written) |
| `sandbox.py` | the restricted subprocess of `write` |
| `harness.py` | `describe`, `show`, `write`, `forward`, the state, the JSONL log; a command line for an agent that drives it from a shell |
| `review.py` | the review checklist: a static check over the logged code |
| `tanh_fwd.chx`, `lin_out.chx`, `tanh_bwd.chx`, `trainer.py` | the gradient arm: a tanh regression network trained with Adam on the CPU session, and the forward program the harness's `forward` uses |
| `arms.py`, `results/` | the arms Z, R, G(N), the evaluator, the trained parameters and the numbers below |
| `tests/weightspace.py` | the tests, wired into `scripts/gate.sh` after `tests/probes.py` |

### 11.2 Choices made while building, fixed before any arm was run

* **The split of section 7 ("the M9 split") as numbers [choice].** Box: the URDF's limits of the five moving joints (`shoulder_pan` +-1.91986, `shoulder_lift` +-1.74533, `elbow_flex` +-1.69, `wrist_flex` +-1.65806, `wrist_roll` -2.74385..2.84121 rad). **Band:** `shoulder_pan` in [0.64, 1.28] rad (a sixth of its range) **or** `wrist_flex` above a third of its upper limit, 0.55269 rad (the "extreme third" read as the top third of the range, one end, which is 1/3 of the range; reading it as both ends would make the band two thirds of the wrist range). Held-in region: the box minus the band. `forward` and the G arms draw only from the held-in region; the **interior** set is fresh held-in draws, the **band** set fresh band draws; both uniform within their region. 2,000 poses each, numpy `default_rng` seeds 20261104001 (interior) and 20261104002 (band); training stream seed 20261104003 (the arms nest: G(10^3) is a prefix of G(10^4)); initialisation seed 20261104004. The two held-out sets are made by `arms.py eval` only; the harness does not import them.
* **`forward(n_poses)`:** one call is one pass of the budget, 1 <= n_poses <= 256 (the post's pass is "up to 256 positions"); the count is saved before the pass runs; it returns the mean error in mm and the counters and nothing else (the draw's seed is in the log's `private` part, which the agent has no path to). It runs the CPU session on 256 padded rows (so the cost does not depend on n_poses).
* **What the writer is told [choice, open for the maintainer]:** `describe()` gives the task, the network, the rules of section 4 and, by default (`--spec chain`), the URDF chain table and the joint limits. With `--spec none` it gives only the limits. W(0) with the table is "write from the spec"; without it, W(0) can only be a guess, so the default is the former. The writer is *not* told the band or the held-in region ("poses within the joint limits").
* **Initial weights** of a writer run: zeros by default (`--init random` for R's).
* **G's recipe [choice]:** Adam (beta 0.9, 0.999, eps 1e-8, no weight decay), batches of 250 (so that N/250 is whole for every N), **40,000 steps for every N**, 5% linear warm-up then cosine to zero (M1's shape), loss e^T e / 2b on targets standardised per axis (mean and standard deviation of the training set), folded exactly into the last layer afterwards so the network is an ordinary one in metres. R's initialisation is `deep_session.init`'s (Glorot-uniform weights, zero biases), and every G starts from it.
* **Learning rate, chosen on a validation set, never on the held-out sets** (2,000 fresh held-in draws, seed 20261104005), mean error in mm: N = 10^4, 4,000 steps: 3e-4 38.78, 1e-3 16.96, 3e-3 9.30; N = 10^4, 20,000 steps: 1e-3 3.93, 3e-3 2.38, 1e-2 1.73; **N = 10^5, 40,000 steps: 3e-3 1.38, 1e-2 1.25, 3e-2 1.95**. Chosen: **1e-2** for every N (M1's 1e-3 is 2.3 times worse at N = 10^4 and 20,000 steps, 3.93 against 1.73 mm; it was not run at N = 10^5). The choice favours the gradient arm; a W that beats G beats a tuned G.

### 11.3 The reference is checked against `kinematics.cho` [measured]

`python3 examples/weightspace/check_fk.py ../cancho-robot <cancho> 2000`: `tests/kinematics_driver.cho` + `src/kinematics.cho` + `src/so101.cho` built with the local compiler and run on the zero pose, 2,000 poses inside the box and 500 beyond it (+-3.5 rad), compared with the numpy chain on the same angles (kinematics.cho takes degrees and multiplies by pi/180; the check feeds the radians it really used): **largest difference of any position component 2.2e-16 m**. The zero pose is (0.391361, -0.0000092, 0.226470) m in both. 25 of those poses and the driver's outputs are recorded in `tests/weightspace/fk_fixture.json`, which the gate compares with `task.fk` (1.1e-16 m), so the gate needs no other checkout. Not checked by me: placo itself (cancho-robot's C4 does it, to 1e-9 m).

### 11.4 What the tests hold (`tests/weightspace.py`, 5 groups; mutants noted)

* A planted violator written through the harness (it fits a tensor by `np.linalg.lstsq` against sampled outputs) is read back from the log and **caught by the review checklist** (`fit-call`); a closed form is clean; the inverse of a fixed matrix is reported **borderline** (`linalg`), not dropped; a hill climber (`random-search`) and a gradient loop (`update-loop`) are caught; and with each rule switched off its plant passes (so the tests can fail). The checklist is a heuristic over the source text, not a proof (its docstring lists what it cannot see).
* `forward` beyond the budget is refused (`forward.budget`), does not run, and the count survives reopening the run; budget 0 refuses the first call; n_poses out of range is refused without spending a pass; the answer is one scalar and counters.
* The sandbox refuses a file read and a file write outside its scratch directory, the harness's own log, a listing of the repository, a network call (`import socket`, `urllib`, `_socket`; a listener the test opens sees no connection), a subprocess, `os.system`, `os.fork`, `ctypes`, `scipy.optimize`, and kills an endless loop at the time limit (each with a `sandbox.*` rule tag); a good program runs. It is run with and without the macOS profile (`WS_NO_OS_SANDBOX=1`, which is what Linux has).
* **What the sandbox does not stop**, pinned in the tests as known behaviour: a program can learn whether an outside path exists (`os.stat` is not audited), it can read the whole Python installation, burn its CPU time, and anything that reaches the OS without a Python audit event is not seen. It is a restricted subprocess, **not a security boundary** (docstring of `sandbox.py`).
* The forward through the CPU session equals the binary64 forward (score 794.105604 vs 794.105605 mm on a random network; the largest output difference over the evaluated networks, 4.4e-7 m); the Adam session's loss at step 1 equals a binary64 numpy implementation of the same training and **all 4,611 parameters after 30 steps agree to 1.6e-7**; a mutant of `tanh_bwd.chx` (the derivative (1+h)(2-h) for (1+h)(1-h)) is 2.5e-2 off and is caught.
* On CI the numpy-dependent steps of the gate skip when numpy is not installed (`ci-numpy`, #83), so there this file prints "skip"; the numbers here are from a machine with numpy.

### 11.5 Arms Z, R and G(N), 2,000 poses per set [measured]

Mean Euclidean position error, mm, with the 95% percentile bootstrap interval (10,000 resamples of the 2,000 poses, `default_rng(20261104)`). One training run per N (the interval is over poses, not over training seeds). G: 40,000 Adam steps of 250, lr 1e-2, the four runs started together on the CPU session (a step is 8.7 ms alone; running four together made no measurable difference, 346 to 348 s each), the M8 job and the gate running beside them. "Wall" is the session's: process start, loading the tables and 40,000 steps; generating 10^6 poses and their kinematics in numpy took 0.8 s more.

| arm | wall | interior, mm [95% CI] | band, mm [95% CI] | band / interior |
|---|---|---|---|---|
| Z (all weights zero) | 0 | 371.42 [366.03, 376.64] | 294.66 [288.34, 300.86] | 0.8 |
| (constant at the mean training position, for reference) | 0 | 308.28 [305.35, 311.21] | 271.55 [268.06, 274.95] | 0.9 |
| R (Glorot-uniform init) | 0 | 769.27 [757.65, 780.92] | 735.84 [723.00, 749.05] | 1.0 |
| G(10^3) | 347 s | 2.70 [2.58, 2.82] | 13.89 [13.22, 14.56] | 5.1 |
| G(10^4) | 346 s | 1.48 [1.42, 1.53] | 8.61 [8.17, 9.06] | 5.8 |
| G(10^5) | 348 s | 1.28 [1.24, 1.32] | 8.28 [7.83, 8.74] | 6.5 |
| G(10^6) | 347 s | 1.29 [1.25, 1.34] | 9.11 [8.63, 9.60] | 7.1 |

Reading it (without applying the W rules of section 7, which have no W to apply to):
* **The CPU session path is not too slow for 10^6 samples**: the step does not depend on N (tables of minibatches), so every arm costs 40,000 x 8.7 ms; nothing was cut and nothing is labelled as reduced. What is not done is more steps for larger N: at the same step count G(10^5) and G(10^6) are the same within the intervals (1.28 and 1.29 mm interior), so the curve flattens by 10^5 for this recipe, and **it may be the step count or the learning rate, not the data**, that sets the floor; not tested.
* **G(N) beats Z and R by two orders of magnitude and is under the "useful" line of 5 mm on the interior at every N, but not on the band**: the band error is 5 to 7 times the interior error, so by the rule of section 7 (band within 2x of interior) **no G arm captures the structure outside the data**: the networks interpolate. A writer that captures the rotation structure would show a ratio near 1, which is what the band set is for.
* The thresholds W has to beat, by the section 7 rule (median interior error below G(N)'s lower bound): 2.58 (10^3), 1.42 (10^4), 1.24 (10^5), 1.25 (10^6) mm.
* R is worse than Z (769 against 371 mm): a random tanh network outputs values of order one metre where the arm moves within about half a metre.
* G(10^3) has 40,000 steps over 4 minibatches (10,000 epochs) and still generalises to 2.7 mm interior; it was not early-stopped (the recipe is fixed; early stopping on a validation set would help it).

### 11.6 What the design had wrong, found while building (corrected in place above, with this date)

1. "Differential tests C4 hold it to 0.1 mm against placo": the test gates at 1e-9 m (section 7, first paragraph).
2. "Z ... the error of predicting the mean": all-zero weights predict the origin (section 7 table). Z is a weaker null than the mean predictor (371 against 308 mm).
3. "Adam, the M1 recipe": M1's learning rate (1e-3) is 2.3 times worse than 1e-2 at N = 10^4 and 20,000 steps (not run at 10^5); the shape (warm-up, cosine) is M1's (section 7 table).
4. "harness ~400 lines": 591 for the harness, sandbox and checklist; 1,222 with everything else; "G(10^6) minutes": 5.8 minutes (section 9 table).
5. The training stack had no tanh, no regression and no multi-output kernels: three small kernels were written (`tanh_fwd.chx`, `lin_out.chx`, `tanh_bwd.chx`); no compiler change was needed. Two surface facts learned: an operation that borrows and moves the same tile is refused (`moved-while-borrowed`), and so is a `1 - h*h`, so the derivative is written (1 + h)(1 - h) with the second factor as `0 - h` plus one; and a local named like a parameter (`b` for a tile in a kernel with a parameter `b`) is a leak error (the same trap as cancho#236, here in `.chx`).
6. Section 4 says the agent "never receives the scorer's samples" and "forward draws fresh held-in samples": true, but a `describe()` that gives the URDF table lets a writer compute the same function itself in its sandbox; the score then only confirms. That is intended (the post's writers have the rules), and it makes W(0) with `--spec chain` the strong form of the arm and T (kinematics as Python) the control for the encoding. Whether to give the table is the choice recorded in 11.2.

### 11.7 Not done

* No writer arm (W, T), no agent run: the harness is built and tested, nothing has driven it. Per-pose logging, the authors' log format (section 4, **[open]**) and token counting are the driver's: the harness logs the counts the driver passes with `--tokens-in/--tokens-out` and nothing else.
* The checklist and the sandbox have not been tried against an agent trying to evade them; the tests use planted programs written by the author of the checklist, which is the weakest kind of test for a heuristic.
* No second training seed, no longer G runs, no validation of G beyond the learning rate.
* Metal / GPU: not used (by instruction). The gate's GPU-less steps all ran; see the report for what CI will run.

## 12. The writer arms W(0), W(10), W(50) and the control T, 2026-11-04 [measured]

Ten separate agents (Claude Sonnet 5.5, one fresh agent per run, background, each with only the harness commands as its tools and the rules of section 4 in its prompt), `--spec chain` (the writer is given the URDF chain table: the strong form of the arm, section 11.2), `--init zeros`, runs created by the maintainer's session so that no agent could set its own budget. Logs, final weights, the writers' scratch sources and the scores are in `examples/weightspace/results/writers/` (`scores.json`; each run's `log.jsonl`). Scoring: the final stored weights of each run on the two held-out sets of section 11 (never shown to any writer), mean position error in mm with a 95% bootstrap interval (10,000 resamples, seed 20261104), exactly as for G.

### 12.1 Results

| run | forward calls (score each returned) | interior mm | band mm | review | agent tokens |
|---|---|---|---|---|---|
| W0_1 | 0 | 10.58 [10.48, 10.67] | 9.58 [9.46, 9.71] | clean | 92 k |
| W0_2 | 0 | 4.98 [4.88, 5.09] | 4.84 [4.74, 4.94] | borderline: only unparsable first writes; stored code clean; local checks (12.3, item 3) | 89 k |
| W0_3 | 0 | 354.45 [343.54, 365.56] | 380.06 [370.33, 389.85] | clean | 67 k |
| W10_1 | 2 (380.7 on the zero network, 0.41) | 0.43 [0.42, 0.44] | 0.40 [0.39, 0.40] | clean | 98 k |
| W10_2 | 2 (372.6 on the zero network, 0.286) | 0.29 [0.29, 0.30] | 0.27 [0.26, 0.27] | clean | 113 k |
| W10_3 | 3 (372.9, 378.0, both on the zero network, then 0.368) | 0.37 [0.36, 0.38] | 0.31 [0.30, 0.31] | borderline: only unparsable first writes; stored code clean | 99 k |
| W50_1 | 2 (369.2 on the zero network, 0.977) | 0.98 [0.95, 1.02] | 0.73 [0.70, 0.76] | checklist clean; borderline by the writer's own report (12.3, item 4) | 99 k |
| W50_2 | 2 (402.5 on the zero network, 2.00 on 128 poses) | 2.14 [2.03, 2.26] | 2.08 [1.98, 2.18] | clean | 125 k |
| W50_3 | 2 (365.6 on the zero network, 0.623) | 0.63 [0.61, 0.64] | 0.53 [0.52, 0.54] | clean | 98 k |
| T_1 (kinematics as Python, no network, no forward) | 0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | n/a | 63 k |

Each writer agent took 1.5 to 10 minutes of wall time (the figures are from the agents' own durations, **[measured, not controlled]**). The gradient arm G took 5.8 minutes of CPU per run (section 11.5), and a person needed no time at all after the harness existed; the writer's cost is tokens, not wall time.
The scores the agents saw in `forward` (256 fresh held-in poses) agree with the held-out scores within the sampling noise of 256 poses.

### 12.2 The pre-registered reading (section 7), applied
* **W(B) beats G(N)** iff the median interior error of its 3 runs is below G(N)'s lower interval bound (2.58, 1.42, 1.24, 1.25 mm for N = 1e3, 1e4, 1e5, 1e6). **W(0)**: median 10.58 mm, beats none. **W(10)**: median 0.37 mm, beats G at **every N** (and is 3.4 times better than G(1e5), the best gradient arm, in interior error). **W(50)**: median 0.98 mm, beats G at every N. If W50_1 is excluded as borderline (12.3, item 4), W(50)'s median of the two remaining runs (0.63, 2.14) is 1.39 mm: it then beats G(1e3) and G(1e4) and not G(1e5) or G(1e6). Three runs per arm, so **no confidence statement is made on W**: the values above are all there is.
* **W is useful** iff some W(B) has interior error under 5 mm and band error within 2x of interior. **Yes**: W10_1, W10_2, W10_3, W50_1, W50_3 (and W50_2 at 2.14 mm); every written network has band error at or below its interior error (ratio 0.7 to 1.0), where **every gradient-trained network has 5 to 7 times**. W0_2 is at 4.98 [4.88, 5.09]: on the threshold, not counted either way.
* **W is a failure** iff all W(B) are over Z's error: no.
* **T is exact and W(0) is not (the section 7 control)**: the writer can write the kinematics exactly as Python in one reply with no oracle (0.00 mm to two decimals on both sets). So where W(0) is worse (10.58, 4.98, 354) the cost is in *encoding* the function in a 5-64-64-3 tanh network (products from tanh curvature, sinusoids as smoothed staircases) and in unchecked construction errors, not in understanding the arm.

### 12.3 What the numbers do not show (read before quoting them)
1. **Forward passes were used as a sanity check, not as a search.** Every W(10) and W(50) writer used 2 or 3 calls; 6 of 6 spent their first call on **the all-zero network** (scores 365 to 402 mm), because their writes had failed silently from their point of view (the source must define a variable named `tensor`; the first six writes of all nine runs, W0 included, failed for that reason). So "W(50)" is W(10) with budget left over, and **the W(10)/W(50) difference is the spread between runs, not an effect of the budget**: W50_2's weaker 2.14 mm comes from table ranges that stop short of reachable angles (the writer's own account), not from having more calls. What the data does support is that the runs with a forward score caught a unit that was still zero (a harness defect, 12.4) and a sign error (W10_3 found it by comparing with its own FK, not with the harness).
2. **W(0) against W(10) is not a clean test of feedback.** The three W(0) runs made different design choices (W0_1 averaged the wrist roll away, W0_2 neglected its coupling, W0_3 chose an additive-separable form that ignores that the pan multiplies the rest), and all six W(10)/W(50) runs modelled the roll. With 3 runs per arm that is as consistent with run-to-run variance in design as with the effect of two forward calls. Nothing here measures the value of feedback.
3. **Every writer built its own numpy forward kinematics and evaluated its construction on its own sampled poses** (local errors of 0.34 to 4.8 mm; they agree with the harness scores). The checklist reads only code sent through `write` and cannot see this. The rule forbids setting a parameter by fitting to computed outputs; the writers say they used the local error to find bugs (a gain applied to constants, a wrong sign, a mislabelled constant) and to **choose between a few hand-made designs** (knot counts, smoothing widths, one of several epsilon values). Choosing between a handful of designs by a computed error is a small manual search; the rule does not decide it and **this section does not hide it**. The designs are all closed-form constructions; none contains a fitted weight, and the unlogged scratch sources are in the results directory for anyone to read.
4. **W50_1 is borderline by its own report**: its layer-1 interpolation solves a square linear system against sin/cos values at the knots (`np.linalg.solve(M, f(x))`, scratch `basis.py`) and it ran `lstsq` once on a cosine to check a basis's accuracy (no weight written from it). The target of that solve is a textbook sinusoid, not an output of the robot's FK, so it reads as "a function tabulated at fixed points", which the rule allows; but it is written the way a fit is. The logged `write` sources are literal arrays, so the checklist saw them clean. Reported in both ways in 12.2.
5. **The writer had the answer's structure in its prompt** (`--spec chain`, the URDF table) and the gradient arms had only samples. The comparison is "a model that knows the arm's geometry and a network trained on samples", which is not a comparison of two ways to learn the same thing. Without the table (`--spec none`) the writer would have to guess the arm's structure from nothing; **not run**.
6. **G is one recipe**: 40,000 steps, one seed, learning rate chosen on validation (section 11.2). That G(1e5) and G(1e6) are equal suggests the floor is the optimiser's, not the data's; a longer or better-conditioned run might close part of the interior gap. "Beats gradient descent" here means beats **this** gradient run, at 5.8 CPU-minutes.
7. **One task, one writer model, 3 runs per arm.** The kinematics function is the favourable case (exact, structured, known to the writer), as section 10 said. The task has an exact answer that fits the network family's strong point; this is not evidence for tasks without one.

### 12.4 Defects in the harness found by the runs (not fixed here)
* **Fixed 2026-11-04 (the next change after #93): `forward` is now refused, without spending a pass, until one write has succeeded (`forward.unwritten`), and a failed write answers `NOTHING WAS STORED` with `stored: false`; `describe` says so too. The runs of section 12 were made before the fix and their tables stand.** The defect was: a write that does not define `tensor` leaves the stored weights unchanged, and a later `forward` silently scores them (the zero network, ~370 mm). It happened in 9 of 9 runs on the first six writes, and 6 of 6 first forward calls were spent on it. The fix is in the harness (the error message of the failed write is shown by the CLI only when you run it by hand; `forward` should refuse or warn when no tensor has been written since `init`), and the budgets of these runs were wasted by it. The W(10) runs had 8, 8 and 7 calls left; the W(50) runs 48, 48, 48. Because the writers did not use their remaining calls, re-running with the fix is **not** expected to change the table, but that is an expectation, not a measurement.
* The agents' scratch FK implementations are outside the log: a future harness could offer a counted `local_eval` so the evaluation a writer makes of its own design is logged and budgeted instead of invisible.

### 12.5 What this says for the next step
Nothing yet for cancho-robot (section 7's last paragraph stands). For the question the milestone was built to ask: a model **can** write a 5-64-64-3 network that computes the SO-101's forward kinematics to 0.3 to 1 mm, with no gradient step, and the written network's error is the same inside and outside the held-out band, which gradient descent on samples did not achieve (5 to 7 times worse in the band). The cost is an agent run (about 100 k tokens, a few minutes) per network, against 5.8 CPU-minutes for the trained one. Whether that trade is useful depends on the task having a structure a writer can read off a specification, and the pose estimator's early layers (edge and capsule detectors on a synthetic frame) are the candidate that has one; that needs the renderer of `roadmap-models.md` 8.3 (V3) and is another milestone.
