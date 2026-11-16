# cancho-studio: a prompt-rewriting layer (design, 2026-11-16)

> **Status: design; nothing is built and nothing about the *quality* of rewriting is known.** Claims are tagged **[read]** (source or document, not run), **[measured]** (run, with the command) or **[estimate]**. Cancho native: no Python, no Rust. The owner asked (2026-11-16) for a layer that reads the prompt, and for an edit the picture, and rewrites the prompt "to improve performance", with `qwen3.8:27b` as the default model. That the rewrite *improves* anything is the hypothesis this design is built to test, not a premise (section 8).

## 1. What it does

Before a job runs, a local vision-language model reads the person's prompt and, for an edit, the picture, and writes a prompt for the generator. The job runs with the rewritten prompt. **Both prompts are kept and shown**; the original is never lost. Every failure of the layer (the model not running, too slow, an unusable answer) runs the job with the original prompt and says why: the layer can make a job better or leave it as it was, never block it.

## 2. What is there today

* **[measured]** (2026-11-16, `curl localhost:11434/api/tags`, `/api/show`, `/api/ps`, `lsof`) Ollama 0.40.16 runs on this machine with `qwen3.8:27b-mlx`: 27.8 B parameters, `nvfp4`, family `qwen3_5`, 64 blocks, context 262,144, capabilities completion, **vision**, tools, thinking, **16 GiB** on disk. Also `qwen3.5:9b` and `qwen3.5:4b` (both with vision), `qwen3:8b`, `nomic-embed-text`. Nothing is loaded now (`/api/ps` is empty). The machine has **64 GB** of memory.
* **[measured]** **Ollama listens on `*:11434`, every interface**, not on the loopback only (`lsof -nP -iTCP:11434 -sTCP:LISTEN`: `ollama *:11434`). This design sends the picture only to `127.0.0.1`, but the service itself is open to the local network. Binding it to the loopback (`OLLAMA_HOST=127.0.0.1`) is the owner's to decide and is outside this change; it is noted because the layer's privacy argument ("the picture does not leave the machine") is about *our* connection, not about who else can talk to that service.
* **[read, and seen in `studio/check.sh`'s output]** A job's `<id>.argv` is the argument list of `media`: for an edit `edit -o <out> [--seed S] -- <image path> <prompt>`, for a video `video -o <out> [--frames N] [--size WxH] [--seed S] -- <prompt>`. The prompt is **always the last entry**, the picture's path (edit only) the one before it.
* **[read]** What reads the prompt downstream: the video model's text encoder is umT5, text only (`tools/wan_text.py`); the edit model's text encoder is a **vision-language model that already sees the picture** (`models/qwen3vl_text.model` with its vision tower, docs/diffusion.md D9). So a rewrite has a clearer job for video (turn a short request into the descriptive text the model was trained on, **a belief, not checked here**) than for an edit, where the encoder already has the picture and a longer instruction may add noise. The design does not assume either; section 8 measures.
* **[measured]** The pinned compiler (a49ea779) accepts `edition 10;` and `narrow(net, "127.0.0.1:11434")` (a program with that line builds and its report says it touches no network because nothing dials). **Not yet measured:** the report row of a program that does dial, `net_out("127.0.0.1:11434")` (`docs/net-bounds.md` 3A says exact hosts are built in edition 10); that is the first thing slice R1 checks.
* **[read]** `examples/base64/base64.cho` in the cancho repository is a base64 codec written in cancho; the picture goes to Ollama as base64 in JSON (`images`), so an encoder is needed (to be ported into this repository's tests/studio code, not called out to a tool).

## 3. Decisions

**D1. It runs in the worker, not in the server.** A 27 B model reading a picture takes far longer than a request should, and the server answers everything on one thread. The worker already runs one job at a time and is where long things belong. The server only records the choice (D2).

**D2. The choice is per job.** `POST /api/jobs` and `POST /edit` take an optional `rewrite`: `"on"` or `"off"` (a form field in `/edit`; a `Select` in the cancho-ui page). Absent, the server's default applies (`--rewrite on|off` when the server starts; **proposed default `off`**, see section 9). The server writes the choice to `<data>/<id>.rewrite` (`on` or `off`) next to `<id>.argv`. `<id>.argv` is never changed: it is the record of what was asked.

**D3. The step.** For a job whose `<id>.rewrite` is `on` and a worker started with a rewriter, before it spawns `media`:

1. read `<id>.argv`; the prompt is its last entry; for an edit the picture's path is the entry before it, read as bytes (at most 16 MiB: the limit uploads already have; a larger file skips the picture and says so);
2. build one request to Ollama's `POST /api/chat` with `stream: false`, `think: false`, `keep_alive: 0` (the model unloads when the answer is done: D6), `options` with a low temperature and a cap on `num_predict`, **no `tools`**, a system message from section 5 and a user message holding the prompt (and, for an edit, `images: [<base64>]`);
3. send it over a plain TCP connection to `127.0.0.1:11434` and read the answer, bounded to 1 MiB, under a deadline (default 600 s: it includes loading the model);
4. take `message.content`, trim it, remove one pair of surrounding quotes or a code fence, replace newlines with spaces; **accept it only if it is 1 to 1,000 bytes with no NUL and no other control character**, else fall back (D4);
5. write `<id>.prompt.orig` (the original prompt) and `<id>.prompt.new`, append one line to `<id>.log` (`studio: prompt rewritten by <model> in <s> s`) and spawn `media` with the argument list of `<id>.argv` whose last entry is the new prompt.

**D4. Fail open.** Connection refused, a deadline, a non-200 answer, JSON that does not parse, no `message.content`, an empty or over-long or unclean result: the job runs with the original prompt and `<id>.log` says `studio: prompt not rewritten: <reason>`. A reason is a fixed phrase; none contains text from the model.

**D5. Transparency.** `GET /jobs/N` shows the original and, if there is one, the rewritten prompt and the model's name. The rewritten text is the model's output and so **untrusted**: it is HTML-escaped (`& < > " '`) where it is shown; `studio/results.cho` has no escape function today (it prints only state words made of letters, digits and spaces) and this slice adds one with its own test.

**D6. One heavy thing at a time.** The step ends, and the model is unloaded by `keep_alive: 0`, before `media` starts, because the worker is sequential. **[estimate]** The 27 B model is 16 GiB; the edit pipeline holds the 14 GB transformer and a 15.7 GB text encoder (docs/diffusion.md, `docs/cli.md`) on a 64 GB machine: together about 46 GB plus the system, tight; one at a time is safe. **[to measure in R2]** that `/api/ps` is empty when `media` starts, and the time the load adds to a job.

**D7. Authority.** The worker's report gains one outbound row, bounded to `127.0.0.1:11434` by `narrow(net, "127.0.0.1:11434")` (edition 10), and nothing else: no other host, no foreign code. The file read of the picture is the `fs_read` it already has. The picture is sent only to that address; the host is not an argument, so it cannot be pointed elsewhere without changing the program. (The URL fetch of `docs/studio-fetch.md` is a different program, the server, and is the one that holds an unbounded `net_out`.)

**D8. What the model's text can do.** It is a string for a diffusion model. It is passed to `media` as one argument by `std.process` (which refuses a NUL), never to a shell, never executed, never used as a path. A picture or prompt that tells the model to say something else (prompt injection) can at worst make a worse prompt; the layer sends no tools and acts on nothing the model says except to use it as the prompt, capped at 1,000 bytes.

**D9. The model.** Default `qwen3.8:27b-mlx`; `studio_worker ... --rewriter NAME` names another (`qwen3.5:9b` is the lighter one with vision). The name is not checked against Ollama's list when the worker starts: a wrong one fails at the call and falls open (D4).

**D10. Privacy.** The picture's bytes go to `127.0.0.1:11434` and nowhere else; they are not logged and not written anywhere new; their path is not sent. The prompts are the person's own text and are kept (`<id>.prompt.orig`, `<id>.prompt.new`, the log).

## 4. What is built before anything is claimed

The step is checked **against a stub of Ollama's `/api/chat`** (a small cancho program that answers a canned JSON, or a scripted failure), so the checks need no GPU and no model: the effective argument list; the original kept; each failure of D4; a prompt with quotes, backslashes, `;` and UTF-8 passing as data; the 1,000-byte and control-character rules; the report row. Then the real model, once, to measure (R2).

## 5. The two system prompts (drafts, to be tested, not claims)

Kept as plain files `studio/rewrite/edit.txt` and `studio/rewrite/video.txt` (data, read at start; a missing file turns the layer off with a message). Drafts:

* edit: "You write the instruction for an image-editing model. You are given a picture and the person's request. Write one imperative instruction, in English, at most 60 words, that says exactly what to change and what to keep as it is. Do not describe the whole picture. Add nothing the person did not ask for. Reply with the instruction only."
* video: "You write the text prompt for a text-to-video model. Expand the person's request into one paragraph of at most 120 words: the subject, the action, the setting, the camera, the light and the style. Keep every fact of the request and add no new subject. Reply with the prompt only."

Whether to keep the person's language or write English, and whether `think: true` helps, are measured in R2, not decided here.

## 6. Slices, each with a gate that can fail

| slice | what | held to |
|---|---|---|
| R0 | this document | review |
| R1 | the worker's step in cancho (an HTTP request over `tcp_connect_start`, a base64 encoder, the JSON build and parse with `std.json`, the sanitiser, the effective argument list), checked against a stub; the `narrow` row measured | the checks of section 4; mutants (accept an unclean answer, skip the length bound, send to another host, lose the original, ignore a failure) must each fail one |
| R2 | the real model once: cold and warm latency, memory (`/api/ps` during and after), the examples recorded, the system prompts iterated **on a development set that the evaluation of R5 does not use** | numbers measured and written down; nothing about quality yet |
| R3 | the server and the pages: `rewrite` in `POST /api/jobs` and `/edit`, `<id>.rewrite`, the escaped display in `/jobs/N`, the OpenAPI document | `studio/check.sh` rows; the whole flow in a browser once |
| R4 | comparison: `rewrite: "ab"` makes two jobs with the same seed, one with the original prompt and one rewritten, and a page `/compare/N` shows them side by side **without saying which is which** (the order is random and recorded) | the page and the recording are checked; the pairs are made |
| R5 | the evaluation of section 8, by the owner | the result written down as it comes out |

## 7. What it costs (to be measured, not guessed)

Each rewritten job pays the model's load and generation, before the job's own 44 s (video) or 90 s (edit) (`docs/cli.md`, `docs/diffusion.md`). **Corrected in place:** an earlier version of this section estimated "tens of seconds to minutes". **[measured]** (section 10, once each, 2026-11-16) a call took **3 to 7 s** in all, of which the model's load was 3.4 s, 1.1 s and 1.1 s and the generation 33 to 100 tokens. That was with the model's files very probably in the operating system's cache (they had been read before; not checked), so **the cold start from disk is still unmeasured** and the estimate stands for it.

## 8. How "improves" will be decided

Written before any rewritten result exists. **Null hypothesis: rewriting does not make the owner prefer the result.**

* A set of **20 cases** (prompt, and for an edit a picture of the owner's choosing, never committed), a mix of video and edit, written before R2's prompts are tuned and kept apart from the development set.
* For each case, **the same seed**, two jobs: the original prompt and the rewritten one (R4). The owner sees them side by side in an order unknown to the owner and picks the better one, or "no difference".
* **Decision rule, fixed now, one for each kind** (10 video cases, 10 edit cases; a kind is judged alone, because the two text encoders differ): the layer is switched on by default **for a kind** only if the rewritten result wins **at least 9 of its 10** (a tie counts as not winning). Under the null a win is as likely as a loss, so the chance of 9 or 10 wins is (10 + 1) / 1,024, about **0.011** one-sided (**[computed by hand from the binomial, not run]**; 8 of 10 would be 56 / 1,024, about 0.055, which is why it is not the bar). Otherwise that kind stays off. The results are written down either way, with the counts.
* An automatic judge (the same Qwen) is **not** used as proof: it would favour its own wording. It may be used to look for failures, never to decide.
* What this does not show even if it passes: that the rewrite helps for pictures and prompts unlike the 20, or that a different seed would agree.

## 9. Not decided

* **Default on or off.** The owner chose the model; whether the layer is on by default is a separate question, and the rule of section 8 says off until measured. Proposed: the server's default is `off`, the page offers a choice, and R5 may change it.
* **Language.** Keep the person's language or always write English.
* **A large picture.** The picture is sent as it is (up to 16 MiB, so about 21 MB of base64 in the JSON); there is no image decoder in cancho to shrink it, and Ollama resizes on its side. R2 measures whether that is a problem.
* **Whether Ollama should be bound to the loopback** (section 2).

## 10. An exploratory run with imprecise prompts (measured, 2026-11-16; not the evaluation)

Three requests, **one run each**, to Ollama's `/api/chat` with `qwen3.8:27b-mlx`, the two draft system prompts of section 5 exactly as written, `think: false`, `keep_alive: 0`, temperature 0.3, at most 400 tokens. The picture of the edit is **synthetic** (128 x 96: a green-to-orange gradient with a white rectangle, made for this), not anyone's photograph. This is an anecdote of three: it shows what the layer does and what its drafts get wrong; it says nothing about whether the results are better (section 8 is how that is decided).

| kind | the person's prompt | model's load | answer in all | tokens | what it wrote |
|---|---|---|---|---|---|
| video | "something nice with water" | 3.4 s | 7.3 s | 88 | a clear stream over smooth stones in a sunlit forest, a slow pan across the surface, ripples, dappled light, a peaceful, meditative style |
| video | "a dog" | 1.1 s | 3.5 s | 100 | a golden retriever on a sun-dappled park bench, tail wagging, a steady medium shot, warm afternoon light, shallow depth of field, "No other subjects appear.", the dog "remains still, gazing directly into the lens" |
| edit | "make it better" | 1.1 s | 3.2 s | 33 | "Replace the plain white rectangle with a soft, semi-transparent frosted-glass panel featuring a subtle drop shadow. Keep the existing green-to-orange gradient background completely unchanged." |

`/api/ps` was empty before and after: `keep_alive: 0` unloaded the model (D6's "[to measure]", measured for this model).

**What it shows**

* **The model sees the picture.** The edit's answer names the white rectangle and the green-to-orange gradient, both true of the synthetic image: the vision path works through `images`.
* **It is fast** when warm (3 to 7 s a call; section 7).
* **For a vague request it invents, and commits.** "Something nice with water" became a forest stream; "a dog" became a golden retriever on a bench; "make it better" became one specific change (frosted glass), chosen by the model, not asked for. That is the layer's purpose for a vague prompt, and also its risk: the person did not choose any of it. The page must show the rewritten prompt (D5) so that it can be seen.

**What the drafts get wrong, to fix in R2 on the development set (nothing has been changed yet)**

1. **The video draft asks for no motion**, and "a dog" came out as a dog that "remains still". A text-to-video model needs the action; the draft must ask for what moves.
2. **A rule of the system prompt leaked into the answer** ("No other subjects appear."): the model repeated the instruction "add no new subject" as content. The rule should be stated so that it cannot be echoed, or removed if it does not help.
3. **"Keep every fact ... and add no new subject"** is in tension with what a vague request needs; whether to allow invention for a vague prompt is a decision (section 9), not something the draft settles by accident.

**Not shown:** whether any of these three, run through the generator, is better than the original prompt; the cold start from disk; the behaviour with a real photograph or a long prompt; the variation between runs (one run each at temperature 0.3).

## 11. Round 2: drafts v2, two pictures, and the development set (written before the run)

**The drafts of section 5 are version 1.** Version 2 is in `studio/rewrite/video.txt`, `edit.txt` and `edit2.txt` (the last for two pictures), written to answer the three faults of section 10: the video draft now asks for **an action and how it moves**; the sentence that was echoed ("add no new subject") is gone, replaced by "no sentences about what is absent"; and for a vague request all three say to **pick one concrete reading**, which makes the invention a stated policy and not an accident.

**Two pictures.** The torch pipeline of the edit model accepts a list of condition pictures: **[read]** `pipeline_qwenimage21.py` puts `<image1>` ... `<imageN>` placeholders in its template (lines 250-290), resizes each picture, encodes each with the VAE and concatenates the latents along the sequence (line 482), and gives the transformer a shape for each (`img_shapes`). `tools/edit_torch.py` calls it with a one-element list. **`tests/edit_pipeline_gpu.py`, the cancho-gpu path, is single-picture** (one vision-tower pass, one VAE encoding, one prefix of one picture's tokens): taking two would change four of its five stages and is a project of its own, **not part of this round**. What this round can test is the *rewriter's* side: Ollama's `/api/chat` takes several `images` in one message, so the layer can read two pictures and write one instruction that names which is which (`edit2.txt`).

**The development set** (made before any v2 result; **separate from the 20 cases of section 8**, which are not written yet; the pictures are synthetic: A, a green-to-orange gradient with a white rectangle; B, a blue sky gradient above a dark green band with a red disc at the upper right):

* video (5): "a car", "something scary in the woods", "kids playing", "the ocean", "a cool ad for coffee";
* one picture (4): A "make it look nicer", A "change the colours", B "remove the thing", B "make it dramatic";
* two pictures (3): (A, B) "put the red thing in the other one", (A, B) "combine them", (B, A) "use the look of the second on the first".

**The checks, written before the run.** For every answer: it is 1 to 1,000 bytes with no control character (the rule of D3); it contains **none of** "No other", "no new", "not ", "without" (a crude detector of the echo and of sentences about what is absent: **a heuristic, it will have false positives**); for a video, it contains **at least one** motion word from a fixed list (walk, run, move, pan, flow, drift, sway, swirl, splash, ripple, roll, fly, jump, play, danc, glid, rush, crash, spin, turn, rise, fall, wave, trac, push, orbit: **a heuristic, a word is not an action**); the time of the call. The answers are also **read by a person** (here: the author), and what that reading found is written next to the numbers. Version 1 is run on the same set (the two-picture prompts only with version 2: version 1 has no draft for them).

### 11.1 Results (measured, 2026-11-16, one run of each, `qwen3.8:27b-mlx`, temperature 0.3, `think: false`)

19 calls: 5 video prompts with each version (10), 4 one-picture prompts with each version (8), 3 two-picture prompts with `edit2.txt` (3). **3 to 4 s a call** for video, **2 to 3 s** for a picture, **3 to 4 s** for two; `/api/ps` was empty before and after. Every answer was 1 to 1,000 bytes with no control character, and **within its word limit** (version 2 video 66 to 94 words of 100; edit 19 to 35 of 50; two pictures 25 to 46 of 60; checked by counting, not by the model's word).

**What the automatic checks said, and where they were wrong.** The movement check found a word in all ten video answers, **including version 1's stationary car**: "pans" (the camera) counts. It cannot tell the subject's action from the camera's, so it is **not evidence of anything** and is withdrawn as a check; the reading below replaces it. The check for sentences about what is absent found one in version 1 ("a car": "No other subjects are present") and **missed another**: version 2's "the ocean" ends "with no visible land or objects other than the natural elements" (the pattern did not contain "no visible"), which also contradicts its own "the sandy beach".

**What the reading found (by the author; a person, not a measurement of quality)**

| | version 1 | version 2 |
|---|---|---|
| video, the subject acts | the car "sits stationary" (**fault of section 10**); the others act or the scene moves (waves, children, a figure "moves slowly", a bean that falls) | **5 of 5** act (the car accelerates; a figure takes a "jerky, lurching step"; children chase; waves roll; coffee pours, steam rises) |
| video, a sentence about what is absent | **2 of 5** ("No other subjects are present"; "No sudden movements") | **1 of 5** ("the ocean": "no visible land or objects ...") |
| video, the request kept | yes | yes; but "a cool ad for coffee" came out as a plainer mug-and-pour scene than version 1's shattering bean (**a matter of taste, not scored**) |
| one picture | each names things that are in the picture (the white rectangle, the gradient, the red circle) and keeps the rest | **no clear difference**: the same changes chosen; "make it dramatic" asked for three changes (storm clouds, a twilight purple sky, a darker ground) and called the red disc "a red sun" (a guess); the white rectangle became "square" (it is 30 x 36 pixels) |
| two pictures | n/a | **3 of 3 correct about which is which**, including when the pictures are swapped (B first, A second: "image1" was B, "image2" was A, as given) |

**Conclusions, and their limits**

1. **The faults of section 10 are mostly gone in version 2**: a still subject (1 of 5 before, 0 of 5 now) and the echoed rule ("No other subjects", 1 before, 0 now). **A new instance of a sentence about absence appeared** in one answer (the ocean), so "no sentences about what is absent" is **not obeyed every time**, with 5 video prompts; one run each at one temperature says nothing of how often.
2. **For an edit, version 2 is not shown to be better than version 1**; they look alike. The second draft's changes may simply not matter for an edit (the generator's encoder already sees the picture).
3. **Two pictures: the rewriter reads both and keeps them apart** (3 of 3, on synthetic pictures with obvious content: a gradient with a rectangle, a sky with a disc; it would be harder with two photographs, **not tried**). **That the rewritten instruction would make the generator do the right thing is untested**, and the generator path of cancho-gpu takes one picture (section 11).
4. **Nothing here shows the rewritten prompts give better pictures or videos.** That is still section 8's question, for the 20 cases not yet written.

## 12. R1 built (measured, 2026-11-16): the worker's step

**What exists.** `studio/rewrite.cho` (the call: base64, the JSON body from `std.json`'s writer, an HTTP/1.0 request over `tcp_connect` on a `Poller` under a deadline, the answer bounded to 1 MiB, the cleaning rule), `studio/rewrite_step.cho` (what the worker does for a job), the worker's flags (`--rewriter <model> <absolute prompts dir>`, after `--every` if there is one), `studio/rewrite_cli.cho` (the call as a command, for checks and for use by hand) and `studio/rewrite_stub.cho` (a stand-in for Ollama's `/api/chat`, for the checks). The worker is now edition 10, which changes no other meaning (docs/net-bounds.md 3A: exact hosts).

**[measured] The authority is what D7 said.** `cancho authority` on the worker and on `rewrite_cli` lists `net_out("127.0.0.1:11434")` and **no other network row and no foreign code**. The host and port are in the program, not arguments: pointing the worker at another host means changing its text.

**[measured] Against the real model, through this code** (`rewrite_cli ask`, no Python, no `curl`): version 2 of the video draft turned "a dog" into a golden retriever that *sprints across a meadow chasing a frisbee*, with a tracking camera, in 3.5 s (the still dog of section 10 is gone, **one run**); the edit draft read the synthetic picture, sent as base64 by this code's encoder, and named its white rectangle and its green-to-orange gradient; a model name Ollama does not have gave code 4 and no text; `/api/ps` was empty after each call. The base64 encoder equals `/usr/bin/base64` on 11 sizes (0 to 7, 100, 1,000 and 65,537 bytes).

**[measured] The checks** (`studio/rewrite_check.sh`: the real worker, the stand-in, a fake `media` that prints its arguments; no GPU, no model): **62 checks and 4 mutants, all as designed.** They cover: a rewritten video and edit (the prompt is the last argument and nothing else changes; the picture's path is not sent; the request names the model, has `stream: false`, `think: false`, `keep_alive: 0`, **no `tools`**, the picture for an edit and none for a video); `<id>.argv` byte-for-byte unchanged; the original and the new prompt kept; a prompt with quotes, a backslash, `;`, `$(...)` and UTF-8 passing through as **one argument with every byte intact and nothing executed**; a job with no `<id>.rewrite`, or a worker with no `--rewriter`, making **no request at all**; the seven failures of D4 (an error status, JSON that is not JSON, JSON with no `message`, an empty answer, an answer over 1,000 bytes, an answer with a control character, no answer before the deadline) and nothing listening: each runs the job with the original prompt, finished, with the reason in the log; quotes and a code fence taken off an answer; a job with no `--`, an edit with no picture entry, an unreadable picture and a missing system prompt: not rewritten, and the job still runs. **Mutants** (each built and run, each refused by a check): an unclean answer accepted, the length bound removed, a failure that still uses the rewritten list, the original not kept.

**What the checks do not show**

* **The test programs are not the real worker byte for byte**: they are built from copies in which `sed` replaces the port (11434 to 18160, so that the real Ollama is never reached) and the deadline (600 s to 2 s). The script checks that the copies equal the real files after the replacement is undone (three `cmp`s), so the difference is those two literals and nothing else.
* **The bound on the answer's size (1 MiB, code 7) is not tested**: the stand-in is an `http.server`, whose output room is 64 KiB per connection (measured earlier, docs/studio.md), so it cannot send more. The mutant "the answer bound is removed" would **survive**; it was not run for that reason.
* **The deadline is the same 600 s for every job** (a constant in `job`); no flag sets it.
* **The rewriter is not in the studio's pages yet**: nothing writes `<id>.rewrite` (that is R3); the checks write it by hand. The log note is appended **after** the job's output (like the tidying note), not before it, because the child writes the log from offset 0.
* **No picture larger than a few hundred bytes went through the real model**, and none through the stand-in beyond 2,000 bytes; the 16 MiB limit is the arithmetic of `max_picture`, not a measurement.
* **The checks are a shell script**, as `studio/check.sh` is. A cancho version is a later job.
