# cancho-studio: an editing session (design, 2026-11-19)

> **Status: designed, not built.** Claims say what was read in this repository (`studio/upload.cho`, `studio/results.cho`, `studio/jobs.cho`, `docs/studio-variations.md` section 3, `docs/studio-cases.md`) and what was not run. It completes `docs/studio-variations.md` V2 ("a result as the next input"), which was designed there and never built.

## 1. What is asked

"Several rounds of editing; see the original and the edited pictures; recover the prompts used in a job; a **refine** task where the user **selects which pictures to include**; a *session of edition*."

## 2. What exists today (read, not guessed)

* A job is `<id>.argv`, `<id>.log`, `<id>.state` and `out/<id>/`. The prompt, seed, steps and model are in `<id>.argv`; `job_page` already prints them (`param_rows`) except the prompt text, which is `out/<id>/prompt.txt`.
* `POST /edit` keeps the person's pictures as `upload-N.*` and `upload-N-2.*` and the job page shows them beside the result (#224). Nothing says that job 12 was made *from the result of* job 9: every job is an island, and a second round means downloading the result and uploading it again.
* A job takes **at most two input pictures** (`image` and `also`); a third pushes the attention rows over what the kernels index (the refusal in `media`: 3 pictures at 1,048,576 pixels is about 12,416 rows against about 11,500). That limit bounds "select the images to include": **one or two**, not any number.

## 3. The model: a session is a tree of jobs, with no new kind of object

No session record is needed. Two small files per job, written by the server (the only writer of `*.argv`, as today), after the job exists:

* `<id>.from`: one line per input picture, `JOB NAME` for a result of an earlier job (`9 edited.png`) or `upload` for a file the person sent with this job. At most two lines.
* `<id>.root`: the id of the first job of the session. A job made from nothing earlier is its own root; a job made from job M has M's root.

A **session is the set of jobs with one root**. Rounds are the jobs ordered by id; the *tree* is whatever `.from` says (a refine of round 2 made after round 3 is a branch, and is shown as one). Nothing is copied: a refine job's input is `out/M/edited.png`, **named by the server from a job id** and checked (done, an edit, the file exists), never a path from the client (`docs/studio-variations.md` section 3).

*Why not copy the picture into `upload-N.*`?* A copy would keep the job whole if `out/M` is deleted, at the price of a second 1-2 MB file per round. Chosen: **link, no copy**, because `out/` is never deleted by the studio; a job whose source is gone shows "source removed" and its refine is refused. Reversible if the data directory is ever pruned.

## 4. Pages and routes

| route | what | built from |
|---|---|---|
| `GET /sessions` | one row per root: thumbnail of the first picture and of the latest result, round count, state of the last round, newest first | the `.root` files |
| `GET /sessions/ROOT` | **the session page**: a row per round, in order: *input pictures* -> *result*, the prompt, seed, steps, model, state, a link to the job; a branch is indented under the round it came from. At the bottom the **refine form** | `.from`, `.root`, `argv`, `prompt.txt` |
| `GET /edit?from=JOB` | the edit form with the prompt, seed, steps and model **prefilled** from that job ("use this prompt again"), and that job's result as the picture | `argv` |
| `POST /edit` | gains `from_job` (+ `from_name`) and `from_job2`: pictures chosen from earlier results instead of (or with) an upload | `upload.cho` |

The job page (`/jobs/N`) gains: the prompt in full (it is not shown today), "Refine this result" (to the session page, this result preselected) and "Use this prompt again" (`/edit?from=N`); and a link to its session. `/jobs` rows link to the session.

### The refine form (the "select the images" part)

Under the session's last round: every picture of the session as a **thumbnail with a checkbox** (the originals first, then each round's result), at most two checked; a third checkbox is disabled by a line of JavaScript and refused by the server regardless (422, the limit above). Plus an optional upload of a new picture (it counts as one of the two), the instruction box with the cases' tips, and Start. Without JavaScript the form still works: the server refuses more than two with a one-line reason. Same style as the other pages (`docs/studio-design.md`): a server-written page, no cancho-ui island, because a file picker with thumbnails is exactly what the island cannot do (`docs/studio.md`, "Editing a picture chosen in the browser").

## 5. Refusals (each with a tag in the answer, none queues anything)

`from_job` not digits (422); not an edit or not done or no result file (422); more than two pictures in all (422); `from_name` not in the closed list `image.png`, `edited.png` and the uploads (422: never a path); a refused refine leaves no `.from` or `.root`. A crash between the job and its two files leaves a job outside any session, never a session naming a missing job (the order as for `.set` and `.pair`).

## 6. Slices (each its own PR, each with its gate)

1. **Prompt and reuse.** The job page shows the full prompt; `/edit?from=N` prefills; the data is already on disk. No new files. Checks: the prompt with a quote/UTF-8/`<` is escaped; a missing job is 404.
2. **Refine one result.** `from_job` in `POST /edit` and `POST /api/jobs`, `.from` and `.root`, "Refine this result" on the job page. Checks as section 5, and the argv holds the `out/M/...` path.
3. **The session page and `/sessions`**, with the tree.
4. **The picker** (two pictures chosen, with an upload counting as one).
5. *(later)* name a session, delete one (its uploads and results, asking first).

Slices 1 and 2 give what was asked most directly; 3 and 4 make it feel like a session.

## 7. What would show this wrong, and what is not known

* **Quality across rounds is not measured.** The result is a PNG (lossless), but the model works at about 1024 x 1024 and a round resizes its input (`tests/edit_pipeline_gpu.py`): a picture edited five times may drift or soften. Before slice 4 is called done: **one real session of five rounds on the GPU**, the first picture and the last looked at side by side, and the finding written here, good or bad.
* **The output size.** An edit's result is not always the input's size (seen: square results from non-square photos). Chained rounds would compound it; a fix belongs in `media` (restore the input's aspect), not in the studio, and is a separate piece.
* **Memory:** the server holds a request in memory (16 MiB ceiling, measured 11.8 MB after one 10 MB upload); a refine without an upload is a few hundred bytes. Not a concern.
* **Concurrent rounds.** Several refines queued from the same round are fine (the worker runs one job at a time); the tree shows them as branches.
* **Authority:** no new capability; the studio server still reads and writes only the data directory (`cancho authority` on `studio` should show the same rows; to be run when built).
