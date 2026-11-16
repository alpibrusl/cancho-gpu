# cancho-studio: variations of one request, and a result as the next input (design, 2026-11-16)

> **Status: V1 (variations) is built and checked (section 6); V2 (a result as the next input) is designed and not built.** Claims say what was read in this repository and what was not run. The owner asked (2026-11-16) for "multiple variations of an image, image combinations, ..." to make the studio more professional; this document takes the first two things that need **no change outside `studio/`** and says plainly why combinations are not among them.

## 1. What is asked, and what the generators can do today

| wish | what it needs | can be built here now |
|---|---|---|
| several variations of one picture or video | the same request run N times with different seeds, shown together | **yes**: the worker already takes `--seed`, and the queue already runs jobs one after another |
| keep working on a result (an edit of an edit, a video from a picture) | a finished job's output as the next job's input | **yes** for an edit (the image path is a file of the server's own data directory); a video from a picture waits on the sampler (docs/wan-vae-encoder.md section 11: it holds the first frame, the continuation is not yet right) |
| combine two images in one request | the edit generator reading **two** pictures | **no**: `tests/edit_pipeline_gpu.py` and the edit session take one condition image. That is a change in the generator (the prefix of the sequence, the rotary positions and the mask of two images), **a separate project with its own gates**, not a studio feature |

## 2. V1: `variations`

`POST /api/jobs` takes `"variations": N` (an integer 2 to 8, absent for one job). The server makes **N jobs of the same request** exactly as `create_one` makes one, with the seeds `s, s+1, ..., s+N-1` where `s` is the request's `seed` (default 1 when `variations` is given: the seeds must be known, not the generator's default, so that a variation can be asked for again). The jobs are queued in order and run one at a time like any others; **the time is N times a job's**: an edit measured at about 90 s (docs/cli.md), so 4 variations are about 6 minutes, and the page says so before the user starts, not after.

* **The set** is recorded in `<first id>.set`: the ids, in order, separated by a space, one line. (As `<id>.pair` records a pair, and like it, written after every job exists, so a crash between leaves jobs and no set, never a set naming a missing job.) The answer is `202 {"id":FIRST,"ids":[...],"set":"/sets/FIRST"}`.
* **`rewrite: "ab"` and `variations` are exclusive** (422): a pair is already two jobs of one request, and a set of pairs is a different experiment.
* **The page `GET /sets/FIRST`** is HTML the server writes (as `/jobs/N` and `/compare/N` are): a grid of the N results, each labelled with its seed and state, each a link to `/jobs/ID`; it refreshes every 5 s while any job is queued or running, and shows the prompt once. Same auth as every route; an id that is not a set is 404, one that is not digits 422.
* **Nothing new for the worker**: the jobs are ordinary jobs. A failed variation shows `failed C` in its cell and does not stop the others.

## 3. V2: a result as the input

`POST /api/jobs` for `kind: "edit"` takes `"from_job": M` instead of `"image"`: the input is `<data>/out/M/edited.png`, **chosen by the server from a job id** (1 to 9 digits, as every id), never a path from the client. Refused (422) when M has no `edited.png` (not done, failed, or a video's job). The set page shows a "use as input" form under every done edit result, which asks for an instruction and posts to the same API (a plain `<form>`, no JavaScript, as `/edit` is).

## 4. What would show this wrong

* `variations: 4` makes four jobs with four different seeds and one set file; a refused request makes **none** (a bad prompt in a request of 4 must not leave 1 queued).
* The seeds of a set are `s..s+N-1`; the same request twice gives the same seeds (and, on the generator's side, the same pictures: **to be measured once**, not assumed: whether two runs of one seed give identical bytes).
* A `from_job` that names a video job, a missing job or a failed one is refused; the path built never leaves `<data>/out/`.
* The page of a set with one job failed, one running and two done shows each as it is.
* Mutants the checks must refuse: seeds all equal; the set file written before the jobs; `from_job` accepting a non-digit id; the refresh left on when everything is done.

## 5. Not decided

* Whether the set page should also offer "variations of this result" in one step (a V2 form with a count). Proposed: yes, as the same form with a `variations` field, if V2 is simple.
* Whether a set can be given a name. Not proposed: the first id is the name.
* Combining two images: **not scheduled here**; it is a generator task (section 1).

## 6. V1 built (checked 2026-11-16)

`studio/jobs.cho` (`variations_of`, `create_set`; `create_one` takes the seed to use), `studio/sets.cho` (the page), a route in `studio/studio.cho`, `studio/openapi.json` (the path, the field, the answer's `ids` and `set`, and the `pair` field the answer of a pair already had and the document lacked), and 24 checks in `studio/check.sh` (now 294 in all, with the two OpenAPI rows of `/sets`): ids consecutive and in the answer, the seeds 5, 6, 7 in the argument lists and nothing else different between them, the set file (ids, then the first seed), the queue order, the default first seed 1, six malformed counts refused (1, 9, a string, 2.5, 0, -2), the exclusion with `rewrite: ab`, the seed ceiling (2147483646 with 3 refused, 2147483645 accepted), **a request that fails validation makes no job** (queue length unchanged), the page's seed labels, the refresh while jobs wait and its absence when all are done, a failed variation shown beside done ones, `/sets` of a job that is not the first of a set 404, of a non-numeric id 422, and without the token 401.
**Three mutants the checks refuse, each run once on the whole script and reverted:** the seeds all equal (2 checks fail: `5 5 5` and `1 1`), the page refreshing always (1 check), the seed ceiling removed (3 checks). No GPU was involved: the worker was the script's fake `media`.

**Not shown:** that a set of real edits finishes (the worker runs jobs as it always did; a set of 4 edits would take about 6 minutes by the 90 s measured for one: **not run**); whether one seed twice gives identical bytes (section 4: **not measured**); the page was checked by its HTML, **not opened in a browser**; the prompt is not shown on the set page (the cells link to each job's page).
