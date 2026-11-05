# M11: can a model write the weights of a network we then run in cancho? (design, written before any run)

> **Status: a design; W2a (the harness) and W2b's arms Z, R and G(N) are built and measured, section 11 (2026-11-04). No writer arm (W, T) and no W0/W1 has been run.** W1's plan, with the download list for approval, and its script are section 13 (2026-11-04); nothing of it has been run on a model. **Section 14 (2026-11-05) is the design for W1 on cancho-gpu's own inference stack (audit, checks, tickets); nothing of it is built.** Marks as in `roadmap-models.md`: **[measured]**, **[read]** (from a document or source, not run), **[estimate]**, **[choice]**, **[open]**. The decision rules in sections 5 to 7 are fixed now and are not edited after a result is seen; facts that turn out different are corrected in place, with the date.
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

**Needs the maintainer's permission to download** the base Qwen3.5-0.8B and the edited model `RikiyaT/Qwen3.5-0.8B-weight-space-programming` (sizes **[measured 2026-11-04, from the Hub's file metadata, nothing downloaded]**: the weights file of each is 1,746,942,600 and 1,746,943,032 bytes, 1.63 GiB; the estimate of about 1.6 GB was right in GiB; the whole list is section 13.2). Run after M8 and the MLX 27B step, with MLX if `mlx-lm` 0.31.3 loads the architecture **[open: the installed `mlx_lm` 0.31.3 and `transformers` 5.2.0 both have a `qwen3_5` module (import checked 2026-11-04); loading the weights is not tried]**, else transformers on MPS.
* **Protocol** (a reduced version of the post's: the post's 32,768-token cap with 4 samples over 500 problems is about 65 M generated tokens for a model that hits the cap 93% of the time, which at an assumed 200 tokens a second is days **[estimate]**): GSM8K test, the first 200 problems, one sample each; thinking on; temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5 (the post's); cap 8,192 tokens; both models, same seeds. The reduced cap favours finding an effect (a looping model loses more answers to a lower cap), so a second run at 16,384 tokens on a 60-problem subset checks that the effect is not made by the cap.
* **Statistic and rule (pre-registered)**: accuracy A and token-limit rate L, each with a 95% bootstrap interval (10,000 resamples of the 200 problems, numpy seed 20261104). **Reproduces** iff the edited model's accuracy exceeds the base's by more than the base's interval half-width **and** L falls by more than 20 points. If only accuracy rises or only L falls, reported as partial with both numbers. If neither, "not reproduced at this protocol", which says nothing about the post's full protocol.
* **Also**: the edit is a few lines of weight arithmetic. If the authors' logs give the layers and the direction, apply it to our base ourselves and compare with their published model (a check that the published weights are what the logs describe): the number of weights that differ and their positions.
  *(2026-11-04: the logs give the edit as a bit-exact diff with the sha256 of both tensors; section 13.1. The check becomes `diff --w1-expect`: the two checkpoints are compared and held to the published counts and hashes, so nothing is re-applied.)*

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

## 13. W1 prepared: what is reproduced, the rule, the cost and the download list, 2026-11-04 (pre-registered; nothing here has been run on a model)

Written before any model file is on this machine and before any generation. The rule in 13.3 is fixed now and is not edited after a result is seen; a fact that turns out different is corrected in place with its date. **Needs the maintainer's approval of the download list in 13.2, and a free GPU** (M8's training owns it; the script refuses to start without `--gpu-free`).
What was done to prepare it: the blog post, the model pages' file metadata (Hub API, sizes only), the edited model's README and the authors' `exp8_math/runs/p50c/edit/manifest.json`, `code/apply_edit.py`, `code/make_edit_diff.py`, `code/evaluate.py` and `eval/RESULTS_excerpt.md` were **read as text** (all **[read]**); no model file, dataset or `edit.npz` was fetched, and the GPU was not used.

### 13.1 What is reproduced
**The edit, as the authors' own files describe it [read].** Base `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`, edited `RikiyaT/Qwen3.5-0.8B-weight-space-programming` @ `d1671a178705bcab7644806d23c535517f35723b`. Exactly two tensors differ, both `down_proj` of the language model's MLP, bfloat16, shape [1024, 3584] (rows = hidden, columns = MLP units):

| tensor | changed entries | changed columns | changed rows | norm of delta / norm of weight |
|---|---|---|---|---|
| `model.language_model.layers.15.mlp.down_proj.weight` | 65,260 | 64 | 1,024 | 0.1064 |
| `model.language_model.layers.19.mlp.down_proj.weight` | 64,928 | 64 | 1,023 | 0.0811 |

The post's 131,072 weights are 2 x 64 x 1,024; the manifest counts 130,188 because entries that round back to the same bfloat16 value are not counted. The post says "one direction subtracted from 64 columns": a rank-1 delta supported on 64 columns. The authors give neither the rank nor the magnitudes, and their diff file calls the form "sparse" (under 30% of entries), which is not a rank statement. **Two things are reproduced, separately:**
1. **The weights** (CPU, minutes, no GPU): `examples/weightspace/w1.py diff BASE EDITED --w1-expect` reads both checkpoints, reports every tensor that differs, and checks them against the table above and against the published sha256 of the base and edited tensors (README and manifest). It also reports what the authors do not: the rank of the delta on its support (at tolerances 0.1, 0.01, 0.001 of the largest singular value), its top-1 energy, and the share of the delta in the changed columns. **Pre-registered expectation, not part of the pass/fail [estimate]:** top-1 energy at least 0.9 and rank 1 at tolerance 0.1 in both tensors (the bfloat16 rounding of the result puts noise of about 2^-9 of each entry in the delta, so higher ranks at 0.001 are expected and mean nothing). If it is not so, the post's description of the edit as one direction is loose, and that is reported.
2. **The behaviour** (GPU, hours): does the edited model score higher and loop less than the base, on a reduced protocol. This is section 6's, with the details it left open fixed in 13.3.

### 13.2 Download list (for the maintainer's approval; nothing downloaded yet)
Sizes in bytes are the Hub's file sizes, read 2026-11-04. Both repositories are Apache-2.0 and not gated. Pinned to the revisions above.

| item | file | bytes |
|---|---|---|
| base | `model.safetensors-00001-of-00001.safetensors` | 1,746,942,600 |
| base | `tokenizer.json` 12,807,982; `vocab.json` 6,722,759; `merges.txt` 3,353,259; `tokenizer_config.json` 16,709; `chat_template.jinja` 7,755; `config.json` 2,907; `model.safetensors.index.json` 50,900; `preprocessor_config.json` 390; `video_preprocessor_config.json` 385 | 22,963,046 |
| edited | `model.safetensors-00001-of-00001.safetensors` | 1,746,943,032 |
| edited | the same nine small files (same sizes as the base's) | 22,963,046 |
| data | `openai/gsm8k`, `main/test-00000-of-00001.parquet` (1,319 problems, the first 200 used) | 419,088 |
| **total** | 21 files | **3,540,230,812 (3.30 GiB)** |

Not downloaded: the edited repository's `eval/` (four files, 9.4 MB: the authors' MATH-500 samples), `README.md`, `LICENSE`, and the authors' `edit.npz` (387,619 bytes; not needed, the hashes are in the script). The commands, to be run only after approval:
```
hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir ~/models/qwen35-08b-base --exclude README.md --exclude LICENSE --exclude .gitattributes
hf download RikiyaT/Qwen3.5-0.8B-weight-space-programming --revision d1671a178705bcab7644806d23c535517f35723b --local-dir ~/models/qwen35-08b-edited --exclude "eval/*" --exclude README.md --exclude LICENSE --exclude .gitattributes
hf download openai/gsm8k main/test-00000-of-00001.parquet --repo-type dataset --local-dir ~/models/gsm8k
```
Disk: 3.3 GiB for the files, plus under 100 MB of results; the free space on this machine is 168 GB (2026-11-04). No Hugging Face cache is used (`--local-dir`).

**Corrected 2026-11-04 after the download**: the commands as first written passed the names after one `--exclude`, which `hf` read as files to fetch (it downloaded only LICENSE and README); each pattern needs its own `--exclude`. The commands above are the corrected ones. All 21 files were then downloaded (approved by the maintainer 2026-11-04) and their sizes match the table.

**W1 weight diff, run 2026-11-04** (`diff --w1-expect`, CPU, seconds; `results/w1_diff.json`): exactly the two published tensors changed (layers 15 and 19 `mlp.down_proj`), 130,188 of 873,438,784 weights (0.0149%), with 65,260 and 64,928 changed entries in 64 columns each, and every `--w1-expect` check passes, sha256 of base and edited included. Each delta on its support is **rank 1** (top-1 energy 1.0000, second singular value about 1e-3 against 1.749 and 1.399), so the post's "one direction subtracted from 64 columns" holds as measured. This is the weight half of W1; the behavioural half has not been run.

### 13.3 Protocol and the rule (fixed now)
* **Run, in this order.** (a) `diff --w1-expect`; if it fails, the edited checkpoint is not the published edit and `behave` refuses to start (override `--no-verify`, reported as such). (b) A smoke run (`--n 8 --max-new 512`) to check that the weights load and to measure tokens per second; it is not a result. (c) The main run. (d) The cap check, only if (c) is "reproduces" or "partial".
* **Main run.** GSM8K test, problems 0 to 199 of the parquet's order, **one sample each**, thinking on (`enable_thinking=True` in the chat template), temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5 over the generated tokens, **cap 8,192 new tokens**, base and edited with the same per-batch seeds (seed 0 + the first problem's index; the two models consume the random numbers differently, so "same seeds" means the same seeding and not the same draws). The prompt is the authors': the question, a newline, "Please reason step by step, and put your final answer within \boxed{}."
* **Grading.** The last `\boxed{...}` in the generation, compared as a number with the gold after `####` (commas, dollar signs, percent signs, `\text{}` units and a plain `\frac{a}{b}` are accepted). Not math-verify, which the authors used; for GSM8K's integer answers the difference should be small and is **[unmeasured]**. A generation without a `\boxed{}` is wrong. **Hit the token limit**: the generation ended at the cap without an end-of-sequence token.
* **Statistic.** Accuracy A and token-limit rate L, each with a 95% percentile bootstrap interval (10,000 resamples of the 200 problems, `numpy.random.default_rng(20261104)`).
* **Rule (as section 6, made exact).** **Reproduces** iff (accuracy) the edited A minus the base A is **strictly greater than** the half-width of the base's A interval **and** (loops) the base's L minus the edited L is **strictly greater than 20 points**. Exactly one of the two: **partial**, with both numbers reported. Neither: **not reproduced at this protocol**, which says nothing about the authors' full protocol (32,768 tokens, 4 to 16 samples, MATH-500 and eight other benchmarks).
* **Cap check.** The first 60 problems again with cap 16,384, base and edited, and the same rule applied to those 60 (own intervals). Reading: main "reproduces" and the cap check "reproduces": **the effect does not depend on the 8,192 cap**; main "reproduces" and the cap check does not: **reproduces at 8,192 only (cap-dependent)**; main "partial": the cap check is reported next to it and changes nothing in the verdict.
* **What the post and its re-run report, for reference, not as targets [read].** Base/edited at 32,768 tokens: GSM8K 13.8/67.1 %, MATH-500 7.9/48.1 %, token limit on MATH-500 93.0/33.1 %. An independent re-run of the published checkpoint (README): MATH-500 51.0 %, 31.6 % at the limit; a 25-problem smoke run of the base 5.0 %, 95 % at the limit. At our lower cap the base can only lose more answers, and with 200 single samples the base's interval will be several points wide; the rule is on differences, not on matching these numbers.

### 13.4 Hardware, time, memory (estimates; the smoke run in 13.3(b) replaces them)
* **Diff**: CPU, numpy only. Compares the tensors as raw bytes in 64 MB chunks, converts only the two that differ; memory under 1 GB; **seconds to a minute** (a synthetic 1,024 x 3,584 pair with 64 changed columns took 0.33 s, **[measured]**, in memory; the 1.75 GB files from disk are **[estimate]**, I/O bound).
* **Behaviour**: this Mac (Apple silicon, 64 GB unified memory), bfloat16, `transformers` on MPS (`--backend mlx` is written and not tried). Weights 1.75 GB per model, one loaded at a time; the key-value cache is only in the 6 full-attention layers: 2 heads x 256 x 2 (K, V) x 2 bytes x 6 = 12 KB per token, so batch 8 at 16,384 tokens is 1.6 GB; **peak well under 8 GB [estimate]**. Memory is not the constraint; the GPU being free is.
* **Tokens to generate [estimate]**: main run, base about 0.93 x 8,192 per problem x 200 = 1.5 M, edited about 0.8 M; cap check, 60 problems at 16,384: 0.9 M and 0.4 M. About **3.6 M tokens** in all. At an assumed 100 to 500 tokens per second (batched, the linear-attention layers in plain PyTorch; **unmeasured, the range is wide**) that is **2 to 10 hours** of GPU time; the smoke run says which. The run is resumable (a rerun skips the finished problems in `DIR/*.jsonl`).

### 13.5 Command, once the download is approved and the GPU is free
```
python3 examples/weightspace/w1.py diff ~/models/qwen35-08b-base ~/models/qwen35-08b-edited --w1-expect --json results/w1_diff.json
python3 examples/weightspace/w1.py behave --base ~/models/qwen35-08b-base --edited ~/models/qwen35-08b-edited \
    --data ~/models/gsm8k/main/test-00000-of-00001.parquet --out results/w1_smoke --n 8 --max-new 512 --gpu-free
python3 examples/weightspace/w1.py behave --base ... --edited ... --data ... --out results/w1 --n 200 --max-new 8192 --gpu-free
python3 examples/weightspace/w1.py behave --base ... --edited ... --data ... --out results/w1 --n 60 --max-new 16384 --gpu-free   # only if the rule says so
```
**On a Spot L4 in Google Cloud, in restartable steps:** `scripts/gcp/w1_cloud.sh` (docs/cloud.md, "W1 on a Spot L4"; `SMOKE=1` is the smoke run (b)), with `w1.py status` and `w1.py verdict` reading the pulled jsonl.

`tests/w1.py` (in the gate) holds the diff, the grading, the bootstrap and the rule to synthetic checkpoints and a fake generator: a planted rank-1 edit on 4 columns is found as that and a rank-2 plant is not called rank 1; 400 random corruptions of a valid file raise only `ReadError`; the 20-point and half-width boundaries are strict; a rerun after a crash resumes. **Not tested**: the two generation backends (`transformers_generator`, `mlx_generator`), which need the weights; their first test is the smoke run. The loading path (`AutoModelForCausalLM`, falling back to `AutoModelForImageTextToText` for the multimodal release) is a guess until then **[open]**.

### 13.6 What this would and would not show
* A "reproduces" is evidence that the published edit changes the behaviour of the published base on one benchmark at a reduced cap with one sample per problem, under our sampler and grader. It says nothing about how the edit was found (the claim about the writer), nor about M8's model (W0 decides that).
* A weight diff that matches the hashes shows the checkpoint is the one described; it does not show the description of the mechanism ("one direction") is the cause of the behaviour. Ablating the edit (restoring one of the two layers and rerunning) would; not planned.
* Sampling differences (HF on MPS instead of vLLM, batched seeds) can move the numbers; a "partial" or "not reproduced" is read with that in mind and is not a refutation of the post.

## 14. W1 in cancho: the same experiment on cancho-gpu's own inference stack, 2026-11-05 (design; nothing here has been run on a model)

> Written before any code. Marks as above. Everything marked **[measured]** below was produced on 2026-11-05 on the CPU (header reads, a tokenizer comparison, a template render, an import check) with **no GPU use and no generation**; the commands are given. Everything about speed and memory of the new engine is **[estimate]**, from numbers measured on the 27B and on Qwen2.5-0.5B, and is replaced in place by the smoke run (14.8, check C11). Section 13 stays as it is: it is the plan for the Python-stack reference run (`transformers`/PyTorch, `scripts/gcp/w1_cloud.sh`, running on Google Cloud as this is written; its results are not in this document). This section is about what the maintainer asked for next: **the same experiment with the model run by cancho-gpu's own kernels** (Metal on this Mac), so that the claim "the published edit changes the model's behaviour" is held on two independent implementations, one of them ours.

### 14.1 What is reproduced, and what is not
* **Reproduced:** section 13.3 unchanged: GSM8K test problems 0 to 199, one sample each, thinking on, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5 over the generated tokens, cap 8,192 new tokens, base and edited, the authors' prompt, grading by the last `\boxed{}`, the same bootstrap (seed 20261104) and the same verdict rule (`examples/weightspace/w1_behave.py`: `grade`, `bootstrap`, `verdict`, tested by `tests/w1.py`). **The rule is not edited by this section.**
* **New:** the generator. It is a cancho program on cancho-gpu's `.chx` kernels. Nothing is installed from Python: if a kernel the Python stack gets from flash-linear-attention is needed, it is written in `.chx` (14.3 says none is, for this protocol).
* **Not reproduced and not claimed:** token-by-token equality with the authors' runs (different random streams: `sampler.cho` is xorshift64*, torch's is not; sampled text cannot match, only distributions can: C9); the authors' 32,768-token, multi-sample protocol; MATH-500; anything about the writer (W2).
* **What a second implementation adds, and what it does not:** it separates "the edit changes the behaviour" from "the Python stack's numerics (bf16 on MPS/CUDA, a particular fused delta-rule kernel) produce the behaviour". It adds no statistical power: 200 single samples have the same interval width in both.

### 14.2 The model, as read from `config.json` and the safetensors header **[measured]**
Command: `python3` reading `~/models/qwen35-08b-base/config.json` and the safetensors header of both checkpoints (no tensor data except the weight-range count below).
* `Qwen3_5ForConditionalGeneration` (text tower + a 12-block vision tower, 100,592,896 parameters, **not used**), `model_type qwen3_5`. Text: 24 layers, hidden 1024, MLP 3584 (SwiGLU), vocabulary 248,320 (the tokenizer has 248,044 entries and 26 added tokens), **tied** input and output embedding (no `lm_head.weight` in the file), RMSNorm eps 1e-6, `(1 + w)` norms (`Qwen3_5RMSNorm`, **[read]** `transformers` 5.2.0 `modeling_qwen3_5.py`).
* **Layer pattern:** `linear_attention` ×3 then `full_attention`, six times: attention at layers 3, 7, 11, 15, 19, 23 (`full_attention_interval` 4); 18 gated-delta layers. **The two edited tensors (layers 15 and 19, `mlp.down_proj`) are in attention layers.**
* **Full attention:** 8 query heads, 2 key/value heads, head_dim 256, `attn_output_gate` true (`q_proj` is [4096, 1024] = 8 × [query 256, gate 256]; `k_proj`, `v_proj` [512, 1024]; `o_proj` [1024, 2048]), per-head q/k RMSNorm, scaling 1/16. RoPE: `rope_theta` 10^7, `partial_rotary_factor` 0.25 (64 of 256 dimensions: 32 pairs, 192 passed through), `mrope_interleaved` with sections [11, 11, 10] (reduces to plain RoPE when the three position streams are equal, which text-only input gives: **[read]** `docs/qwen35.md` §6 for the 27B, same code path in `transformers`).
* **Gated delta:** 16 key heads and 16 value heads of 128 (**one value head per key head**; the 27B has 16 and 48), conv kernel 4 over the `[q, k, v]` projection (6,144 channels = 2,048 + 2,048 + 2,048), `in_proj_qkv` [6144, 1024], `in_proj_z` [2048, 1024], `in_proj_a`/`in_proj_b` [16, 1024], `out_proj` [1024, 2048], `A_log` and `linear_attn.norm.weight` stored **F32** (everything else BF16), `dt_bias` BF16, `mamba_ssm_dtype` float32.
* **Sizes:** 873,438,784 parameters in the file: text 752,393,024 (of which the embedding 254,279,680), vision 100,592,896, multi-token-prediction head 20,452,864 (`mtp.*`, unused). Weights file 1,746,942,600 bytes.
* **End of sequence:** `config.json` says `eos_token_id` 248044 (`<|endoftext|>`); the tokenizer's `eos_token` is `<|im_end|>` 248046; no `generation_config.json` is in either model directory. `w1_behave.transformers_generator` stops on the union of both; so must the cancho runner (`cancho-gpu-run` for Qwen2 also stops on both, **[read]** `docs/inference.md` §4).
* **Prompt:** the template's output for the W1 message is `<|im_start|>user\n{question}\nPlease reason step by step, and put your final answer within \boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n` (jinja2 render of the model's `chat_template.jinja`, `enable_thinking=True`, **[measured]**). Over the 200 problems this is **50 to 160 tokens, mean 87** (`cancho-gpu tokenize`, **[measured]**), so the longest context is 160 + 8,192 = **8,352 positions** (16,544 for the cap check).
* **bf16 against f16:** of the 752,390,432 bf16 text weights in the base checkpoint, **3,044,570 (0.40%) are nonzero with magnitude under 2^-14** (f16 subnormals: fewer mantissa bits) and 15,944 are under 2^-25 (would flush to zero); the largest magnitude is 13.06. So narrowing bf16 to f16, which `docs/inference.md` §1 calls exact "for every value in its range", is exact for all but those 0.4%. Whether they matter is C5's question, and the answer is not assumed.

### 14.3 The audit: what cancho-gpu has and lacks for Qwen3.5-0.8B generation
Status: **T** exists and tested (for what it is used for now); **U** exists but untested for this model (a parameter or a shape away, or written for a sibling); **M** missing. Evidence is a file or a measurement; "27B" is Qwen3.8-27B (`docs/qwen35.md`).

| component | status | evidence |
|---|---|---|
| safetensors reader (HF single-file checkpoint, bf16 and F32) | T | `src/safetensors.cho`, `tests/safetensors.sh`; `device/run-main.cho` loads a Qwen2.5 HF directory |
| weight loader for **this** checkpoint | **M** | `run-main.cho` knows Qwen2 names; `device/qwen35.cho` knows the 27B's Ollama store and NVFP4 and hard-codes its dimensions (hidden 5120, 64 layers, `weight_bf16` refuses F32: lines 453-459; here `A_log` and `linear_attn.norm.weight` are F32) |
| bf16 weights on the device | U | the language has `f16`, `f32`, `i8` only (`docs/chx.md` §4); f16 narrowing is the Qwen2.5 path (`docs/inference.md` §1) and is not exact for 0.40% of these weights (14.2); an f32 widening path (exact, 3.0 GB) would need the loader to offer both |
| matrix-vector products, f16 weights, f32 activations | U | `matvec_rows_f16`, `matvec_res_f16` (`tests/chx/`), in `tests/kernels.py` at small shapes and in the Qwen2.5 runner at n=151,936; the shapes here (n=248,320 head, 6,144, 3,584, k=2,048 / 3,584) are not instantiated |
| RMSNorm with `(1 + w)`, per-head q/k norm, gated norm | T / U | `ported/rmsnorm.chx`, `qnorm.chx`, `rmsnorm_rows.chx`, `gated_norm.chx` run the 27B; here n=1024, r=8 and 2, h=16 |
| partial RoPE (32 pairs + 192 pass-through), tables on the host | T | `rope.chx` at h=32, p=192 is the 27B's shape too; r=8 and r=2 are new parameters |
| attention output gate | T | `attn_gate_q.chx` (27B r=24; here r=8) |
| KV-cache row write at a position | T | `kv_row.chx` (position = buffer offset, `cgpu_arg_at`) in the 27B runner |
| decode attention, d=256, **g=4** query heads per key head | U | `attention.chx`, `attention_scores.chx`, `attention_pv.chx` are used at g=6; `tests/kernels.py` has small cases (`attention_scores`/`attention_pv`) |
| **attention over 8,352 (and 16,544) positions** | **M** | the head's `[g, t]` scores are one tile in `attention_pv`: ceiling 1,364 at g=6 on Metal (`docs/kernels.md` §4b); the same arithmetic gives **2,048 at g=4** (32 KiB / (4 × 4 bytes); **[estimate]**, not compiled); the runner's cache is `tmax` 1,024 (27B) / 256 (Qwen2.5). `docs/kernels.md` §4 lists what the surface lacks for a split-KV merge |
| causal depthwise conv step (window of 3, silu) | T / U | `conv_step.chx` (27B c=10,240, `tests/kernels.py` c=512); here c=6,144 (a multiple of its 256-channel tile) |
| `qk_norm`, `gates` | T / U | literals fit dk=128, which is also this model's; `rows` = 16 + 16 + 16 = 48 and h=16 are new parameters |
| **delta rule, one value head per key head** | **M** | `delta_step.chx` and `delta_step3.chx` are written for three value heads per key head (`nk * 384` rows); `gdn_ref.py` (numpy) is the specification for any ratio |
| a gated-delta layer end to end on the GPU | T (27B) / U | 12 greedy tokens of `The capital of France is` equal the reference runtime's and eight steps × three prompts agree with its golden log-probabilities to 1e-4 (`docs/qwen35.md` §4); the specification is held to `transformers`' own code in binary64 in the gate (`tests/gdn.py`, `examples/qwen35/hf_check.py`) |
| the decode engine for this model | **M** | `qwen35.cho` (1,669 lines of the 2,000 allowed; hard-coded to the 27B, NVFP4, Ollama store, `tmax` 1,024, MTP) and `run-main.cho` (Qwen2, 256 positions) run other models; a new module is needed, built from both |
| host-driven loop, one command buffer a layer, one wait a token | T | `qwen35.cho` `feed_one` (27B: 36.5 ms a token), `run-main.cho` (Qwen2.5-0.5B: 6 to 12 ms a token, `docs/inference.md` §3) |
| prefill | T | token by token is what `run-main.cho` does; 50 to 160 tokens a prompt makes batched prefill unnecessary here (14.8) |
| tokenizer (byte-level BPE, NFC, Qwen3 pre-tokenizer regex, added tokens) | T for this file | **[measured]** `cancho-gpu tokenize` on this `tokenizer.json` against `tools/tokenizer_dump.py` (Hugging Face `tokenizers` 0.22.2) over 400 records (the 200 rendered W1 prompts and the 200 gold solutions): ids and decoded text **identical**. The file's sha256 (`5f9e4d4901a9...`) is not the 27B's (`0997f410c57a...`), and `tests/tokenizer_hf.py` does not list it yet |
| chat template | **M** (small) | `src/chat.cho` is the **27B's** template: `cancho-gpu chat tests/chat/think_on.json` renders a system message "Reasoning effort is set to xhigh...", which this model's template does not (**[measured]**, jinja2 render above). The W1 prompt is one fixed user message; it needs its own renderer held to jinja2 |
| sampler: temperature, top-k, top-p, xorshift64* stream | T | `src/sampler.cho`, `tests/sampler.sh` against `tools/sampler_dump.py`; temperature, then top-k, then top-p renormalised, which is the order of `transformers`' warpers |
| **presence penalty** | **M** (small) | not in `sampler.cho` or `chat.cho`. Order to match (`w1_behave.Presence`, and vLLM's): subtract the penalty from the logit of every token generated so far, once, before temperature |
| sampling cost over 248,320 logits a token | unmeasured | top-k ≤ 64 is one pass over the row in cancho; the readback is 1 MB a token; neither has been timed for this vocabulary on a decode loop |
| stop tokens, cap, `hit_limit` | M (small) | `run-main.cho` stops on the two ids; "ended at the cap without an end-of-sequence" is `w1_behave`'s `hit_limit` |
| grading, bootstrap, verdict, resume | T | `examples/weightspace/w1_behave.py`, `tests/w1.py` |
| a cancho backend for `w1.py behave` | **M** (small) | `pick_generator` knows `transformers` and `mlx`; the `gen(prompts, max_new, seed) -> [(text, n_tokens, hit_limit)]` interface is the seam |
| GSM8K input | U | the parquet needs `pyarrow` (Python); `load_problems` also takes `.jsonl`, which a cancho program can read with `std.json` |
| the edit applied inside cancho (base + patch) | **M** | no patch format exists: `w1.py diff` reports counts, hashes and ranks, not the entries (14.5) |
| batched or concurrent decode | M (large) | the engine holds one conversation (`docs/serving.md` §4) |
| **CUDA** | **M** (large) | the kernels emit for `nvidia-ada` and the device cases ran on an L4 (`docs/device.md`); the CUDA shim has `open`, `compile`, `function`, `alloc`, `upload`, `download`, `arg`, `launch` only: no asynchronous launch, no buffer offset, no zero fill, no sync (the Metal shim has `cgpu_launch_async`, `cgpu_arg_at`, `cgpu_zero`, `cgpu_sync`, `cgpu_overlap`), and `qwen35.cho` imports the Metal module. Out of scope here |
| a flash-linear-attention-like (chunked) delta kernel | **not needed** | prefill is 50 to 160 tokens and runs through the recurrent step like decode; the chunked form matters for long prompts, and W1 has none. The long teacher-forced check (C7b) feeds 8,400 tokens one at a time (about the cost of 8,400 decode steps) |
| the multi-token-prediction head, vision tower | not needed | unused by the protocol |
| a reference to check against | U | `torch` 2.12.0 and `transformers` 5.2.0 import on this Mac (**[measured]**) and have `qwen3_5`; `examples/qwen35/hf64.py` recompiles `modeling_qwen3_5` in binary64; no script yet loads this checkpoint |

**What the 27B work proves for the gated-delta layers, and what it does not.** `docs/qwen35.md` is a **forward and generation** result: the 27B decodes on Metal with its gated-delta layers (the recurrent state, `conv_step`, `delta_step`, `gated_norm`), greedily and sampled, 36.5 ms a token. `docs/roadmap-models.md` section 9 (M10) is a **fine-tuning feasibility** design: the specification of the rule and its backward in numpy, held to `transformers` in binary64, kernel probes, a cost model, and an MLX baseline; nothing was trained, and it adds no generation. Neither gives a result for this model's shapes, for bf16 weights, for a tied head, for g=4, for a ratio of one value head per key head, or for any context past 1,364 positions.

**Sizing the missing pieces** (small: one session, tens to a few hundred lines; medium: one session with a probe and a decision; large: several sessions or a change outside this repository). No line counts are invented; the one comparison is the existing 750-line `run-main.cho` for a Qwen2 decoder.
* Small: the one-value-head delta kernel and its references; the 0.8B kernel instances; the template renderer; the presence penalty; stop/cap/`hit_limit`; the `w1.py` backend; the jsonl export of GSM8K.
* Medium: the weight loader (two dtypes, F32 tensors, tied head, patch option); the engine module; **attention past 2,048 positions** (14.5 lists the in-language route, untested, and its fallback).
* Large: batched decode; CUDA.

### 14.4 The shortest verified path to a first generated token
`T1` the reference goldens → `T3` the kernels at this model's shapes → `T5` the loader → `T6` the engine, checked at each step by C1 to C5 (layer-0 hidden state, then all layers, then the logits), and `T7` for the first greedy token (C6). It needs neither long-context attention (`T4`), nor the sampler (`T8`), nor the template (`T2`): a prompt of token ids shorter than 2,048 positions is enough for the first token and for C1 to C7. The first token is not "generated" until C5 holds: a token with wrong layers can look plausible.

### 14.5 Design
**The generation loop.** Options, with the one chosen:
1. **Host-driven decode loop with a device KV cache and device recurrent state (chosen).** The `qwen35.cho` pattern: weights, caches and states are device buffers; per token the host writes the embedding row (a `file_pread` of 2 KiB, `run-main.cho`; the head reads the same table as its matrix, so the table is on the device once), the RoPE table row and the mask; encodes the 24 layers as one command buffer a layer; waits once for the logits; samples on the host; appends. Attention layers write one cache row at a buffer offset (`kv_row`), delta layers update their `[16, 128, 128]` f32 state in place. A conversation is reset with `cgpu_zero`.
2. *One session* (`docs/resident.md`): rejected. A session is a fixed kernel graph run once per request line; it has no cache write at an offset (`kv_write.chx` rewrites the whole cache with a one-hot: 205 MB a token at 8,352 positions, **[estimate]**), no sampler, and `docs/resident.md` §16 itself serves without a cache.
3. *Everything on the device, sampler included* (no host round trip per token): deferred. It is a speed question, answered only if the smoke run says the host sampler matters (C11).
4. *Two processes at once* (base and edited): possible with option 1 and a free GPU; whether two decode streams overlap or contend is **[open]**, measured in C11, not assumed.

**Weight precision.** The loader offers `f32` (exact widening of bf16, 3.0 GB) and `f16` (1.5 GB, 0.40% of weights approximate, 14.2). C5 runs both: `f32` isolates kernel error from narrowing error; `f16` is the speed path. The path used for the main run is the one that passes C5 to C7 and is named in the result; if only `f32` passes, the time estimate in 14.8 doubles on its bandwidth term.

**Attention beyond 2,048 positions (T4).** The in-language route, **untested [open]**: `attention_scores` already writes the scores to a global scratch for any `t`. What is missing is the softmax over a row longer than a tile. (a) the row maximum with a loop over position chunks that carries a 1-D tile, with `max(a, b)` spelt `a + relu(b - a)` (`relu` is in the language; the result may differ from the maximum by one rounding, which a softmax shift tolerates because it cancels, but the claim needs the float64 test); (b) the normaliser the same way; (c) `attention_pv` as a loop over chunks that accumulates `exp(s - m) / z` times `v` into one carried `[g, dw]` tile, with `m` and `z` loaded as `[g, 1]` columns (the broadcast is tested: `tests/chx/colbcast.chx`). Three or four small kernels, no new language feature. If the probe shows it cannot be said, the options are a language change (the maintainer's call, `docs/kernels.md` §4) or a hand-written Metal kernel (the precedent is `tests/chx/head_ops.metal`, which `docs/qwen35.md` calls the first thing to go when the surface can say it); neither is chosen here.

**The edit.** (1) *As published, loaded as is* (primary): the edited directory, same loader as the base. (2) *Optional*: base + patch, so that the edit is applied inside cancho. The patch is written once, in Python, from the two checkpoints (`w1_diff.py` already finds the 130,188 entries): per tensor, `(row, column, new bf16 bits)`. The loader overwrites those entries of the base tensor before widening. **What this tests:** that the patch path reproduces the published tensors byte for byte (C10). **What it does not:** the authors' procedure; the patch is derived from their output, not from their `edit.npz` (387,619 bytes, not downloaded, format **[read]** only as "sparse"). The behavioural run uses (1); (2) is a check, not a third arm.

**Sampler, grader, harness reuse.** The cancho runner implements only: render the prompt, tokenize, prefill, loop (penalty, temperature, top-k, top-p, draw), stop, detokenize. It reads problems from a jsonl and prints one jsonl record per problem (`model`, `id`, `max_new`, `text`, `n_tokens`, `hit_limit`). Grading, bootstrap, verdict, resume stay `w1_behave.py`/`w1.py` unchanged: a `--backend cancho` starts the runner once (the model stays loaded; batch 1) and implements `gen(prompts, max_new, seed)`; the seed for problem `i` is `seed + i`, as now. The per-problem seeding means "same seeds" is the same seeding, not the same draws, as in 13.3.

### 14.6 Acceptance checks, in this order (each must pass before the next is trusted)
Reference: `transformers` 5.2.0 on the CPU, this checkpoint, **float64** where stated and **float32** otherwise (the precision `docs/inference.md` §2 used for Qwen2.5). The float32-against-float64 gap of the reference itself is recorded in T1 as the **floor**; no pass criterion below is tighter than what `torch` float32 itself achieves. The thresholds are **[choice]**, fixed here before the checks exist, from what the earlier checks held (Qwen2.5: logits 1e-3 asked, 1e-4 found; 27B: 1e-4).

| # | check | pass |
|---|---|---|
| C1 | every new or re-instanced kernel at this model's shapes against a float64 reference written from the mathematics, CPU interpreter and Metal, with mutants (`tests/kernels.py`): one-value-head delta step, `qk_norm` rows 48, `gates` h=16, `gated_norm`, attention g=4, RoPE r=8/2, the matvecs at n=248,320 / 6,144 / 3,584 | within the repository's reduction tolerances (1e-5 for sums of a few hundred terms, 1e-4 for 4,096), every mutant fails |
| C2 | weights on the device equal the file: for every tensor, the device buffer downloaded and compared with numpy's widening (`f32`) or `astype(float16)` (`f16`), and the F32 tensors read as F32 | bit-equal; the count of values that differ from the bf16 originals (`f16` path) equals 14.2's count |
| C3 | layer 0 (gated-delta) output, 8 prompts (3 W1 prompts, 5 short), every position, float64 reference, `f32` weights | max abs error / max abs reference ≤ 1e-4 |
| C4 | the output of every layer 0 to 23 (layer 3 first: attention with gate, partial RoPE, q/k norm; then 15 and 19, the edited layers), same prompts, `f32` weights | ≤ 1e-4 at every layer, and not more than 4× the reference's own float32 floor |
| C5 | final logits at the last position, the same prompts, both checkpoints, `f32` then `f16` weights | top-5 ids equal in order (a swap with a float32-reference gap below 1e-3 is reported, not failed); each top-5 logit within 1e-3 absolute |
| C6 | **one-token greedy argmax equal to the reference's on all 200 W1 prompts**, both checkpoints | 0 unexplained differences; one whose reference top-2 gap is below 2e-3 is listed and explained, never silently allowed |
| C7 | 64-token greedy decoding, 20 W1 prompts, both checkpoints: free-running tokens, and teacher-forced (cancho fed the reference's tokens) | teacher-forced logits within 1e-3 at all 1,280 positions; a free-running divergence only at a near-tie of the reference (gap below 1e-3) |
| C7b | long context: the reference's own greedy continuation of one W1 prompt to 8,400 tokens (float32, CPU), cancho teacher-forced over all of it | logits within 2e-3 at positions 1,024, 2,048, 4,096, 8,192 and 8,400; no NaN; the delta state's norm finite at the end. Needs T4. A reference that takes hours is run once and its logits at those positions stored |
| C8 | template and tokenizer: the 200 prompts rendered by cancho are byte-equal to the jinja2 render of the model's template; ids equal to Hugging Face `tokenizers`; decode equal | all 200 (the tokenizer half is already **[measured]** equal, 14.3) |
| C9 | sampler: (a) the presence penalty bit-equal to a numpy reference on 1,000 random rows with random seen sets; (b) the distribution: from one real logits row of C5 with a seen set, 200,000 draws at temperature 0.6, top-p 0.95, top-k 20, penalty 1.5, against the exact distribution computed by `transformers`' processors on the same row | the support (the nucleus ids) equal; chi-square at 0.001 not rejected with the seed fixed in the test, so it cannot flake |
| C10 | the edit: base + patch gives the edited file's two tensors byte for byte; and the logit change from base to edited on 8 prompts in cancho matches `transformers`'s (the edit's effect, not only its weights) | bytes equal; difference vectors within 1e-3 |
| C11 | smoke: `--n 8 --max-new 512`, base and edited | completes, no NaN, texts start in reasoning, `hit_limit` as defined; **records tokens a second, resident memory and the host sampler's share of a step**, which replace 14.8's estimates in place |
| C12 | the main run, then the verdict by `w1.py verdict` with 13.3's rule; the cap check only as 13.3(d) says | the verdict is reported whatever it is |
| agreement | the cancho result beside the Python-stack run's (section 13): accuracy and token-limit rate of each model, with intervals | **[choice]** "the stacks agree" iff, for each of the four (model × {A, L}), the two 95% intervals overlap. Not a pass/fail of W1; if they do not overlap, the first suspects are C7b and the sampler |

C1, C8 (renderer), C9 and the T2/T3/T8 tests need no model and go in `scripts/gate.sh`. C2 to C7b, C10 and C11 need the weights and a Mac and, like `scripts/qwen35-check.sh`, are scripts outside the gate (`scripts/qwen08-check.sh`). A check that disagrees is a finding to be explained, not a threshold to loosen.

### 14.7 Tickets, ordered by dependency (each small enough for one agent session)
Every ticket ends with `CANCHO=<pinned cancho> scripts/gate.sh` passing, and no source file over 2,000 lines (`device/qwen35.cho` is at 1,669: do not grow it).

| id | ticket | size | depends on | done when |
|---|---|---|---|---|
| T1 | `examples/qwen35/hf08_ref.py`: load this checkpoint in `transformers` on the CPU (float64 and float32, text tower only), run the 8 check prompts and the 200 W1 prompts; write per-layer outputs for the 8, last-position logits (top-20 and the full row for 3), greedy argmax for the 200, a 64-token greedy continuation and the logits along it for 20, and the float32-against-float64 floor (the two precisions' argmax must agree on the 200, else the disagreeing prompts are listed) | small | none | goldens stored (small ones in `golden/`, large ones named by sha256 and regenerated by the script) |
| T2 | the W1 prompt renderer (a fixed one-user-message, thinking-on template of this model, beside the 27B's in `src/chat.cho`, selected by an explicit template argument) held to jinja2 on the 200 problems; add this `tokenizer.json` to `tests/tokenizer_hf.py` (skipped when the file is missing) | small | none | C8 **Done (2026-11-05):** `src/chat_w1.cho`, `cancho-gpu chat <request.json> qwen35-08b-w1`, `tests/chat_w1.py` (in the gate): 200 of 200 prompts byte-equal to `apply_chat_template`, ids and decode of the 200 equal to `tokenizers`; C8 holds, **[measured]**. || T3 | `tests/chx/delta_step1.chx` (one value head per key head), its float64 reference and mutants in `tests/kernels.py`, `tests/cases.txt` entries; cases for every other kernel at this model's shapes (emit at size, small `--run`) | small | none | C1 |
| T4 | long-context attention: first a probe in the style of `examples/qwen35/probes/` for 14.5's three-kernel route at g=4, d=256, t up to 8,704 and 16,640; then the kernels with float64 references and mutants; if the surface cannot say it, **stop and report** the choice in 14.5 to the maintainer | medium | none | C1 for the long kernels, or a written decision |
| T5 | loader for this checkpoint (config.json dimensions, F32 and BF16 tensors, `f32` and `f16` paths, tied head, optional patch file), a device-buffer dump for C2 | medium | none (check needs T1 only for C2's numpy side) | C2 |
| T6 | `device/qwen08.cho`: the engine (18 delta layers, 6 attention layers, states, cache at `tmax` a parameter up to 2,048 at first, RoPE tables, `feed_one`, a `--trace` of every layer's output) | medium | T3, T5 | C3, C4 |
| T7 | `cancho-gpu-run08`, a greedy runner (prompt ids or text, stops on 248044/248046, `hit_limit`, `--timing`) and `scripts/qwen08-check.sh` for C5 to C7 | small | T1, T6 | C5, C6, C7: **the first verified generated tokens** |
| T8 | presence penalty in `src/sampler.cho` and the C9 tests (numpy reference in `tools/sampler_dump.py`; the distribution test); time the host sampler on 248,320 logits | small | none | C9; the sampler's cost recorded |
| T9 | `cancho-gpu-w1` driver (jsonl in and out, per-problem seed, stops, cap, `hit_limit`, resumable) and `w1.py behave --backend cancho` (a subprocess kept alive, tested with a fake runner in `tests/w1.py`); `tools`-side jsonl export of GSM8K | small | T2, T7, T8 | a fake-runner test in the gate; a real 2-problem run |
| T10 | `tmax` to 8,704 (and 16,640 for the cap check), cache sizing, the long teacher-forced check | small | T4, T6 | C7b |
| T11 | the edit patch: writer (Python, from the two checkpoints), loader support, and C10 | small | T5, T6 | C10 |
| T12 | the smoke run and the estimates of 14.8 corrected in place (needs the GPU free and the maintainer's go) | small | T9, T10 | C11 |
| T13 | the main run, both models, the verdict, the agreement reading, and 13.3(d)'s cap check if the rule asks (needs the GPU free) | small (hours of run) | T12 | C12 |
| later | batched decode (several conversations a weight read); device-side sampling; a CUDA engine (the shim needs asynchronous launch, buffer offsets, zero fill and sync first) | large | T12's numbers | only if T12 says the run is too slow |

Parallel from the start: T1, T2, T3, T4, T5, T8. The first generated token is after T1, T3, T5, T6, T7 (14.4).

### 14.8 Memory and time (estimates, replaced by C11)
* **Device memory [estimate]:** weights 1.5 GB (`f16`; 752,393,024 × 2 bytes) or 3.0 GB (`f32`), of which the tied table 0.51 / 1.02 GB; the six attention layers' cache, f32, `6 × 2 × 2 × 256 × 4 = 24,576` bytes a position: **205 MB at 8,352 positions, 406 MB at 16,544**; the 18 delta states `18 × 16 × 128 × 128 × 4` = 18.9 MB; scores scratch and activations under 10 MB. About **2 GB** (`f16`) to **3.5 GB** (`f32`) plus the 1 MB logits row; the Mac has 64 GB. Memory is not the constraint.
* **Speed [estimate].** Bytes per token: 1.5 GB (`f16`). The 27B reads 14.6 GB in 36.5 ms, 400 GB/s (**[measured]**, `docs/qwen35.md` §5), so the weights cost 3.8 ms. The 27B's ~930 launches cost 8.1 ms, 8.7 µs each; this model has 24 layers where that had 64, so about 350 launches, 3 ms. Sum **6.8 ms** a token; Qwen2.5-0.5B, also 24 layers and 1 GB of `f16` weights, was measured at 6.3 ms at best and 6 to 12 across runs (`docs/inference.md` §3). So **7 to 12 ms a token, 85 to 140 tokens a second**, before the host sampler (unmeasured) and before long-context attention (K and V read at up to 205 MB a token at the end of a long generation, 0.5 ms at 400 GB/s if the kernels reach the bandwidth, which is not known). `f32` weights roughly add the bandwidth term again.
* **Tokens [estimate]:** 13.4's: main run about 2.3 M (base 1.5 M, edited 0.8 M); with the cap check about 3.6 M. At 85 to 140 tokens a second that is **about 5 to 8 hours for the main run, 7 to 12 with the cap check**, one stream at a time; prefill (87 tokens a prompt) adds about 5 minutes over 400 problems. These are unmeasured token counts times an unmeasured speed.
* **Not claimed:** that two streams overlap, that the host sampler is negligible, that the long-context kernels reach the bandwidth.

### 14.9 What would make this not worth doing
* **The reference run finds nothing.** If the Python-stack run (section 13) says "not reproduced" and nothing suggests its numerics are at fault, the cancho run is only a numerical cross-check of the stack; the checks C1 to C7 are still worth having, the 5 to 8 hours are not.
* **Long-context attention cannot be said in `.chx` and neither a language change nor a hand-written Metal kernel is acceptable** (T4). Without it the cap is 2,048 positions; a base model that loops fills 2,048 almost always, so the protocol's second statistic (token-limit rate at 8,192) is unmeasurable. Stop at C7 and say so.
* **A check will not hold.** If C5 to C7 cannot be brought to the thresholds on `f32` weights, the engine is wrong or the specification is, and no behavioural run means anything. If only `f16` fails, the `f32` path costs about twice the time.
* **The smoke run (C11) is slower than about 30 tokens a second.** The main run is then over 21 hours; cut `n` or stop, and decide on batching (a large ticket) first.
* **The GPU is not free.** M8's training and the MLX 27B step own it (header); the run waits.
* **What it would not show, even if everything works:** nothing about how the edit was found; nothing at the authors' protocol; no more statistical power than the Python run; and a "reproduces" that agrees with the Python stack's is two implementations of one pipeline (same checkpoint, same grader, same prompt), so it does not protect against an error in the grader or the prompt that both share.
