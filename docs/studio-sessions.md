# cancho-studio: an editing session (design, 2026-11-19)

> **Status: slices 1 and 2 (section 8), 6 the gallery (section 9), and 3 the sessions page with the two-place menu (section 11) are built and checked; 4 (the picker) and 5, and the library upload of section 10.3, are designed, not built.** Claims say what was read in this repository (`studio/upload.cho`, `studio/results.cho`, `studio/jobs.cho`, `docs/studio-variations.md` section 3, `docs/studio-cases.md`) and what was not run. It completes `docs/studio-variations.md` V2 ("a result as the next input"), which was designed there and never built.

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
6. **The gallery** (asked 2026-11-19), section 9: built.
7. **Menu, sessions, library, picker** (asked 2026-11-19), section 10.

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

## 9. The gallery (asked 2026-11-19; built and checked, see the end of this section)

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

**Built and checked (2026-11-19, `studio/gallery.cho`).** `GET /gallery[?tag=][&page=]` as designed: the tag from the file, newest job first (within a job: the uploads, then the result), 48 tiles a page with *Newer* and *Older*, a filter row, the tile's job and prompt, *Refine* on a result. The tiles reuse the job grid's styles; the tags are coloured (edited and generated in the accent colours, uploaded grey). A *Gallery* place is in the header of every page. `/gallery` is in `studio/openapi.json` (a check requires every route to be). 14 new checks in `studio/check.sh` (fake `media`): the number of tiles of each tag equals the files (uploads; done jobs' `edited.png`; done jobs' `image.png`), read across all pages; a failed job's `edited.png` is not listed; a refine's first picture is not an upload; a tile's `src` is the file the job page serves; a result offers Refine and an upload does not; 4 refusals (unknown tag, page not digits, 5 digits, mixed); a page past the end is empty with a sentence, status 200; no token 401; with 50 more uploads a page holds exactly 48 and offers Older, page 2 offers Newer, the pages together hold every upload once, and the newest upload leads page 1.

**Measured, looked at** (the studio on this branch in the browser pane, a copy of the earlier jobs' data): 25 tiles, 25 pictures, **13.6 MB** in all (about 0.5 MB a tile; uploads are JPEGs of a few hundred KB and results are PNGs of about 1 MB), so a full page of 48 would be about 26 MB, **half of the 50 MB estimated above**. The pictures arrive as the page is scrolled (`loading=lazy`); in the first screenshot the tiles were still empty, a second one a moment later showed them. The tags read clearly; the picture fills its tile, cropped to 16:9 (a portrait photo loses its top or bottom in the tile; the job page shows it whole). **Not done:** thumbnails (a smaller file per tile) would cut the weight, not needed at this size; filtering by session waits for slice 3; the gallery does not list the frames of a video, nor a video.

## 10. One menu, sessions, a library and a picker (asked 2026-11-19, designed)

**Asked for:** slice 3 (the session page); in the gallery, upload a picture; everywhere a picture is needed, choose it **from the gallery or upload one**; and merge the menu (Studio, Edit, Cases, Comparisons, and now Gallery and Jobs) into one or two places with a better UX.

### 10.1 The menu: two places and tabs

Six items in the header is too many, and they are two ideas: *making* and *looking back*. Two places, each with a row of tabs under the header:

| place | tabs (the old items) |
|---|---|
| **Create** (`/`) | Generate (the old Studio page: video and image) - Edit a picture - Cases |
| **Library** (`/gallery`) | Gallery - Sessions - Jobs - Comparisons |

No route changes: every old URL still works, only the header differs (the open tab is marked, the place that holds it is marked). The page of `/` is the cancho-ui island and takes its header from `results.decorate`, so it gets the same header. *Why not one page with everything:* the Generate page is a cancho-ui island and the rest are server-written pages; merging them would move one into the other for no gain. The tabs give the same effect (one place, one click) and keep each page as it is.

### 10.2 Slice 3: sessions

`GET /sessions` lists the sessions, newest first: a card per root with the first picture, the latest result, the number of rounds and the state of the last. `GET /sessions/ROOT` is the session: a row per round (inputs, result, prompt, seed, state, links), a round that came from an earlier round indented under it, and the refine form underneath (the picker of 10.4). A session is the jobs with one `.root` (section 3). A job with no `.root` file (made before slice 2, or by the API) is its own session of one round; `/sessions` lists those too, so nothing is missing.

### 10.3 The library: a picture with no job

A person should be able to put a picture in the gallery without making a job. **`<data>/lib/L<n>.<jpg|png|webp>`**, written by `POST /gallery` (multipart, `image`, up to 15 MB; the extension from the first bytes, as `/edit` does); the number is the line count of an append-only `<data>/lib.index` (the queue's way; the server is the only writer). Served as `/files/L<n>/picture.<ext>` (the `file_info` of the other pictures, with an id that is `L` and digits and the closed names `picture.jpg|png|webp`). Listed by the gallery with the tag **uploaded** and the job link replaced by "library". Refusals: not a picture (422), none (422), over the limit (413), no token (401), a foreign Origin (403): the gate `/edit` already has.

### 10.4 The picker

Every place that asks for a picture (the Edit form, each Cases card, the refine form) gets, per picture, **the file input and, beside it, "or choose from the gallery"**: a `<details>` with the most recent pictures as radio tiles (name `pick1` / `pick2`). No JavaScript. The value is `j/N/NAME` (a job's picture: `edited.png`, `image.png`, `input1.*`, `input2.*`) or `l/L<n>` (a library picture). **The server resolves it**: N digits, NAME in the closed list, the file there (and, for a result, the job done); the path is built by the server; nothing is copied. A file uploaded in the same request wins over a pick of the same slot. `from` (a refine) stays and means `pick1=j/N/<its result>`.

`.from` then names every picture: `N NAME` for a job's file, `L<n>` for the library, `upload` for a file sent with the job. The job page shows them all. A pick of a result stays in that result's session (`.root` of its job); a pick of an upload or of the library starts a session.

*Costs, said now:* the file inputs can no longer be `required` (either an upload or a pick will do), so **the browser will not stop an empty submit; the server's one-line 422 is what the person sees**. Cases has six forms and two pickers each: 12 recent pictures per picker (not 24) keep that page's markup to about 100 KB; the images are `loading=lazy` inside a closed `<details>`, so none loads until opened.

### 10.5 Order of work and what would show it wrong

A) the menu and slice 3; B) the library upload and the picker. Each its own PR with its gate. The menu change must not break a page: the existing checks that read headers (`aria-current`, the places) are updated where they counted the old links, and **each update is named in the PR**, not hidden. For B: a pick of a failed job's file, of a name outside the list (`input3.png`, `../x`), of a library id that is not in the index, and a pick of a video job all 422 and leave no file; a pick plus an upload of the same slot uses the upload.

## 11. The menu and the sessions page: built (2026-11-19)

**The menu (10.1).** Two places in the header, **Create** and **Library**, and a row of tabs under it for the open place: Create has Generate, Edit a picture and Cases; Library has Gallery, Sessions, Jobs and Comparisons. Every old URL works. The header is written in one place (`results.header`), so the cancho-ui page of `/` gets it too (`results.decorate`). **Existing checks that read the header were changed, and each is named here:** the "design: ... marks X as the open place" checks (5 pages) now expect two marked links, the place and the tab; "main page: the header is there once and marks Studio" now says Create and Generate; "main page: the header comes before the app's root" allows the tab row between; "site: / is index.html, with ... added" also strips the tab row. No check was deleted or loosened beyond that.

**Sessions (10.2).** `GET /sessions`: a card per session (the latest result, or the first picture; the number of rounds; the latest job's prompt and state), the most recently worked first, at most 60. `GET /sessions/ROOT`: a card per round, oldest first, with its pictures (a refine's first picture is the earlier result), result, prompt and state, a refine indented 24 px under the round it came from (up to six levels), and under them a form: the finished results as radio choices with a thumbnail (the newest checked), the instruction and a seed, posting `from` to `/edit`. A job with no `.root` that is an `edit` or `image` job is its own session of one round; a video is no session. `/sessions/abc` is 422, a root with no job 404. Both routes are in `studio/openapi.json`.

**Checked** (`studio/check.sh`, 11 new checks, fake `media`): the card's round count equals the jobs with that root; an API edit is a session of one round; a video is not; the list is capped at 60; refusals 422/404/401; the rounds come oldest first with margins 0, 24 and 48 px for a root, its refine and the refine of that; the refine's first picture is the earlier result; the form offers the finished results with the newest checked; the Library and its Sessions tab are marked.

**Looked at** (the studio on this branch in the browser pane, real jobs 11 and 12): the two places and the tabs read clearly; session 11 shows round 1 (the two pictures, the result, the instruction), round 2 indented with its first picture captioned as the result of an earlier job, and the refine form with the radio of round 2 checked. The radio buttons were oversized at first (the global input style applied to them) and are fixed. **Not run:** a phone-sized view.

**Not built:** the library upload (10.3) and the picker (10.4), which this change prepares (the Library place and tabs) but does not include.
