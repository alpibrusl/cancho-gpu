# cancho-studio: an editing session (design, 2026-11-19)

> **Status: slices 1 and 2 are built and checked (section 8); 3 to 5 are designed, not built.** Claims say what was read in this repository (`studio/upload.cho`, `studio/results.cho`, `studio/jobs.cho`, `docs/studio-variations.md` section 3, `docs/studio-cases.md`) and what was not run. It completes `docs/studio-variations.md` V2 ("a result as the next input"), which was designed there and never built.

## 1. What is asked

"Several rounds of editing; see the original and the edited pictures; recover the prompts used in a job; a **refine** task where the user **selects which pictures to include**; a *session of edition*."

## 2. What exists today (read, not guessed)

* A job is `<id>.argv`, `<id>.log`, `<id>.state` and `out/<id>/`. The prompt, seed, steps and model are in `<id>.argv`; `job_page` already prints the prompt ("Prompt as written", and the rewritten one) and the parameters (`param_rows`). *Correction (2026-11-19, found when building slice 1): the first version of this document said the job page did not show the prompt in full. It does.*
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

The job page (`/jobs/N`) gains: "Refine this result" (to the session page, this result preselected) and "Use this prompt again" (`/edit?from=N`); and a link to its session. `/jobs` rows link to the session.

### The refine form (the "select the images" part)

Under the session's last round: every picture of the session as a **thumbnail with a checkbox** (the originals first, then each round's result), at most two checked; a third checkbox is disabled by a line of JavaScript and refused by the server regardless (422, the limit above). Plus an optional upload of a new picture (it counts as one of the two), the instruction box with the cases' tips, and Start. Without JavaScript the form still works: the server refuses more than two with a one-line reason. Same style as the other pages (`docs/studio-design.md`): a server-written page, no cancho-ui island, because a file picker with thumbnails is exactly what the island cannot do (`docs/studio.md`, "Editing a picture chosen in the browser").

## 5. Refusals (each with a tag in the answer, none queues anything)

`from_job` not digits (422); not an edit or not done or no result file (422); more than two pictures in all (422); `from_name` not in the closed list `image.png`, `edited.png` and the uploads (422: never a path); a refused refine leaves no `.from` or `.root`. A crash between the job and its two files leaves a job outside any session, never a session naming a missing job (the order as for `.set` and `.pair`).

## 6. Slices (each its own PR, each with its gate)

1. **Prompt and reuse.** "Use this prompt again" on the job page opens `/edit?again=N` with the prompt, seed, steps and model filled in (the data is on disk already). No new files. An id that is not digits, or a job that does not exist, is ignored (the plain form), not an error: a link to nothing should still give the form.
2. **Refine one result.** `from_job` in `POST /edit` and `POST /api/jobs`, `.from` and `.root`, "Refine this result" on the job page. Checks as section 5, and the argv holds the `out/M/...` path.
3. **The session page and `/sessions`**, with the tree.
4. **The picker** (two pictures chosen, with an upload counting as one).
5. *(later)* name a session, delete one (its uploads and results, asking first).
6. **The gallery** (asked 2026-11-19), section 9.

Slices 1 and 2 give what was asked most directly; 3 and 4 make it feel like a session.

## 7. What would show this wrong, and what is not known

* **Quality across rounds is not measured.** The result is a PNG (lossless), but the model works at about 1024 x 1024 and a round resizes its input (`tests/edit_pipeline_gpu.py`): a picture edited five times may drift or soften. Before slice 4 is called done: **one real session of five rounds on the GPU**, the first picture and the last looked at side by side, and the finding written here, good or bad.
* **The output size.** An edit's result is not always the input's size (seen: square results from non-square photos). Chained rounds would compound it; a fix belongs in `media` (restore the input's aspect), not in the studio, and is a separate piece.
* **Memory:** the server holds a request in memory (16 MiB ceiling, measured 11.8 MB after one 10 MB upload); a refine without an upload is a few hundred bytes. Not a concern.
* **Concurrent rounds.** Several refines queued from the same round are fine (the worker runs one job at a time); the tree shows them as branches.
* **Authority:** no new capability; the studio server still reads and writes only the data directory (`cancho authority` on `studio` should show the same rows; to be run when built).

## 8. Slices 1 and 2: built (2026-11-19)

**What exists.** `GET /edit?again=N` (slice 1: prompt, seed, steps and model of job N filled in, escaped; a picture is still asked for). `GET /edit?from=N` (slice 2: when N is done and made `edited.png` or `image.png`, the form shows that picture as the first picture and carries `from=N` in a hidden field; otherwise the plain form). `POST /edit` takes `from` in place of `image` (both together: 422; `from` not digits, no such job, not done, no result file: 422). The first picture is `<data>/out/N/edited.png` (or `image.png`), **built by the server from the id, nothing copied**; `image2` may still be uploaded. Every job the form makes (one, a variations set, an `ab` pair) gets `<id>.root` (the first id of the session, or that of the job it was refined from) and `<id>.from` (`N edited.png`, or `upload`, one line per picture). The job page shows a refined job's first picture from `.from`, and offers "Refine this result" (done, with a picture) and "Use this prompt again" (any job with arguments that is not a video).

**Checked** (`studio/check.sh`, 19 new checks, fake `media`): the prompt `make it <b>red</b> & "bold"` comes back escaped; seed 7, steps 4 and the model come back; non-digit and missing ids give the plain form (200); a not-done job is not offered; the job page links match the state; the refine form shows the result, carries `from` and asks for no upload of the first picture; the refine job's argument list holds `out/N/edited.png` and no `upload-` file was made; `.from`/`.root` of a refine, of a third round (same session), of a refine with a second upload (two lines, `--also`), of variations (both jobs) and of an `ab` pair (both jobs); four refusals leave no file behind; `from=../x` is refused.

**Once for real** (the studio on this branch, port 8092, a copy of a data directory with the earlier jobs, the GPU worker on `scripts/media`): job 11 (an edit, done) was refined with `from=11` and the instruction "Make the lighting a little warmer, keep everything else unchanged.", seed 3: job 12 was queued with `out/11/edited.png` as its picture (`.from` = `11 edited.png`, `.root` = `11`: job 11 predates `.root`, so it is its own session), ran to `done` in about two minutes, and its page shows **Picture 1 (result of an earlier job)** beside the **Result**, the prompt, the parameters and the two buttons. Looked at: the person and the jacket are the same, the background went from grey to a warm beige: the change asked for, and nothing else seen to change. One round only; five are still to be measured.

**Not done / not measured:** the session page and the list (slice 3), the picker (4); the quality over five rounds and the output-size drift (section 7) are **still to be measured**; the `cancho authority` report of the studio is unchanged in kind (it lists the same capabilities; I did not diff it against main).

## 9. The gallery (asked 2026-11-19, designed, not built)

**Asked for:** "the concept of a gallery, and a tag label: uploaded, generated, edited".

**What it is.** `GET /gallery`: every picture the studio holds, newest first, as a grid of tiles. Each tile has a **tag**, the picture, the job it belongs to (a link to `/jobs/N`) and, for a picture that can be refined, a *Refine* link (`/edit?from=N`). A filter row above the grid (`All`, `Uploaded`, `Generated`, `Edited`; `?tag=`) and 48 tiles a page (`?page=`).

**The tag comes from the file, so nothing new is stored and the old jobs get one too:**

| tag | which file | where |
|---|---|---|
| **uploaded** | `upload-N.*` (the first picture) and `upload-N-2.*` (the second) | `<data>/`, served as `/files/N/input1.*`, `input2.*` (#224) |
| **generated** | `out/N/image.png` (a picture made from a prompt, `kind` image) | `/files/N/image.png` |
| **edited** | `out/N/edited.png` (an edit, also a refine) | `/files/N/edited.png` |

An upload used by a refine is **not** listed twice: a refine's first picture is an earlier result, so it has no upload of its own. Frames and video files are not pictures of the gallery. A job's result appears only when the job is `done`; a failed job shows nothing.

**Where it meets the session.** The picker of slice 4 offers the gallery's pictures of that session (and may offer any picture: the limit is two). A tile's tag and job link tell *which round and which kind* a picture is, which is what a person needs to choose among them. The session page of slice 3 is the gallery filtered by session root, with the rounds in order.

**Refusals and limits.** `tag` is one of the three words (else 422); `page` is digits (else 422); a page past the end is empty, not an error; both need the token like any route. The listing reads the data directory (names only); no file is read until the browser asks for a picture.

**What would show this wrong, and what is not known.**
* **Weight.** The tile shows the picture itself (the studio makes no thumbnails: a decoder and a resizer exist in cancho, `docs/png-cho.md`, but a thumbnail store is its own piece). A result is about 1 MB; 48 of them is about 50 MB per page. Measured when built (the page's size and the time to first tiles in the browser pane); if it is too heavy, the page size drops, or thumbnails become slice 7.
* **Order across kinds.** Newest first means by job id; an upload and its result share an id and appear next to each other, upload first. Said here so it is not a surprise.
* **A job with both a result and uploads** shows up to three tiles (two uploaded, one edited); that is deliberate: each is a picture one can reuse.

**Checks planned (fake `media`, as the others):** one job of each kind and a two-picture edit give the right tags and counts; the filter returns only that tag; a refine's first picture is not listed as uploaded; a not-done and a failed job add nothing; `tag=bad` and `page=x` give 422; an empty data directory gives an empty grid with a sentence; no token gives 401.
