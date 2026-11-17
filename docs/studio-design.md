# cancho-studio: the visual design, and whether islands help (2026-11-16)

> **Status: the design is applied to the pages the server writes (section 5) and to the cancho-ui page of `/` (section 6); islands are read, not built.** Claims are tagged **[read]** (from a file, not run) or **[not run]**. The owner supplied a design (section 1) and asked whether the page could use islands "as in cancho-ota" (section 3), and said to look at how they are used in code, not only in the docs; section 3 is from the code.

## 1. The design given (read 2026-11-16)

A Design canvas (an Artifact of type Design, four artboards: `Main` studio `GET /`, `Job` detail `GET /jobs/N`, `Edit` `GET /edit`, `Compare` `GET /compare/N`), **[read]** from its files `project/*.dc.html` and `project/canvas.json`. What it says, as tokens and structure:

| thing | value |
|---|---|
| ground, surface, raised | `#121314`, `#1B1D1F`, `#24272A` |
| lines | `#2A2D30` (cards), `#3A3E42` (inputs), `#4A4F54` (dashed drop zone) |
| text, muted | `#ECEAE4`, `#B5B2A9` |
| accent, accent hover | `#F2B544`, `#FFD27A` (a running job's border and badge; the Start button) |
| error | `#FFB4A0` (border and text of a refusal) |
| type | IBM Plex Sans 15 px / 1.45; IBM Plex Mono for ids, parameters, badges and the status; headings 22 px (page), 20 px (section), 13 px muted (card titles) |
| radii | cards 12 to 14, inputs 8 to 10, badges 999, buttons 10 |
| controls | height 44 px (48 px for the main action), a segmented control for kind and for rewrite (`Off`, `On`, `A/B`) |
| shell | one header on every page: brand, nav (Studio, Edit a picture, Comparisons), and a status "GPU busy · 1 running · 1 queued" with a dot; content `max-width` 1216 px |
| Studio | a composer card (kind, prompt, frames, size, a **number** seed, rewrite, Start) over a grid of job cards: a 16:9 thumbnail, a state badge, `#id`, the prompt in two lines, a mono meta line, a progress bar while running |
| Job | a result column (player, download buttons, log) beside a details column (prompt as written, the rewritten prompt, parameters, files) |
| Edit | a dashed drop zone with a preview, an "advanced: a path on this machine" fold, the instruction, seed, rewrite, an error alert |
| Compare | the pair side by side, three choice buttons, the tally |

The design's own notes say which parts need the server to change: thumbnails need a name in the closed list of `/files`; a stage line and progress need the worker to write a stage and `GET /api/jobs/N` to return it with the seconds; a local preview of the chosen picture needs a few lines of script; the page would be written by the server, and the cancho-ui page comes back "when cancho-ui has a run-time Link, a dynamic Image/Media src and net.upload".

**What can be applied now with no new capability** (a plan, **not done**): the stylesheet (dark, these tokens, the type stack with the system font as fallback), the shared header with the queue counts, the job list as cards with state badges and prompts, the job page in two columns, the `/edit` form and the comparison page restructured. **Not now:** thumbnails, stage and progress (the worker and the closed list of files), the local preview (script), "Run again" and "Reuse prompt" (new behaviour), and IBM Plex **from Google Fonts** (the design links it; the studio is local and offline by default, so the stack names it and falls back to the system font; an installed IBM Plex is used).

## 2. What the pages are, which decides section 3

`/jobs`, `/jobs/N`, `/sets/N`, `/compare` and `/edit` are **HTML written by the server** (`studio/results.cho`, `sets.cho`, `compare.cho`, `upload.cho`). `/` is the **cancho-ui page** (`studio/ui/studio.cui`), built by `cancho-ui` with its own style. A cancho-ui island can be in the second only.

## 3. Islands, read in cancho-ota's code

**Where.** Not in cancho-ota's main checkout `ui/trip.cui` (no `island` in it): in a worktree of it, `.claude/worktrees/agent-a0700ff22a54540db` (commit `4dd5bed`), files `ui/trip.cui` lines 33 and 45, `ui/assets/strip.js` (123 lines), `strip.css`, `chatmedia.js`, `chatmedia.css`. **Whether that branch is merged was not checked.**

**The pattern [read].**
```
island Strip {
    script: "assets/strip.js" sha256: "c1bcdd66…";
    style: "assets/strip.css" sha256: "3e66ab84…";
    label: "The legs and items of the trip, as cards you can swipe. …";
    width: 400;  height: 212;
    props { cards: String; current: String; p0: String; …; p11: String; photo_for: String; editable: Bool; }
    events { Picked: Int; PickedLeg: Int; Photo: String; }
}
// in the view:  Strip { cards: cards current: cur_id p0: pp0 … editable: adm on: Picked on: PickedLeg on: Photo }
```
Inside the frame the script gets `cancho.props`, `cancho.onprops(fn)` and `cancho.emit(name, value)`; it reaches **nothing else** (docs: `cancho-ui/docs/islands.md`: no network, no storage, no cookies, no parent access).
* **Data in is text.** The cards are records `id|kind|route|when|price|status|state` separated by `;`. A photo enters as base64 in up to 12 props `p0..p11` of at most 4000 characters (a frame takes no string longer than 4096, and at most 16 props).
* **Data out is messages.** The island picks a file with its own `<input type=file>` **inside the frame (it works)**, shrinks it with `createImageBitmap` and a canvas (`toBlob`, down to 240 px and quality 0.45 until the base64 is at most 46,000 characters), and sends it as `cancho.emit('Photo', '<n>|<1 if last>|<up to 4000 characters>')`; the page reassembles the pieces and sends them to the server.
* **Pictures are drawn on a canvas**, "a data: URL would not load here: the frame allows no images".
* **The island has its own CSS**, which is how it gets a look the stock page cannot (cancho-ota's design notes).
* **Limits cancho-ota recorded:** an island must sit outside every `If` and cannot come and go; because of it the chat's media island stayed "at the foot of the page" (its design.md 30.67). The server holds a request of 64 KB, so pictures are about 34 KB.

**What that means for the studio [reasoned from the code above, not run].**
* **Sending the picture to edit through an island: no.** The edit pipeline resizes its input to about 1024 x 1024 pixels (`tests/edit_pipeline_gpu.py`); an island's picture is about 34 KB, a thumbnail. `/edit`, a multipart form, stays the way to upload.
* **Showing results in an island: no.** The frame has no network and no `<img>`; a result would have to cross as base64 under the same 34 KB, and the server has no way to make a thumbnail.
* **A card strip of jobs with progress: fits the pattern, with two gaps.** The cancho-ui page knows only the job it started (there is no `GET /api/jobs` list), and a tap cannot open `/jobs/N` (a `Link` in cancho-ui is a literal; docs/studio.md, "What cancho-ui would need").
* **A local preview of a chosen picture, a seed or variations control with its own style:** possible and small; they change how `/` looks, not what the studio can do. They cannot go on the server-written pages (section 2).

**Conclusion.** Islands are a way to style and animate parts of `/`; they do not unblock uploads or results, and the pages that matter most are the server's. Order proposed: **(1)** apply the design to the server-written pages (section 1, "applied now"); **(2)** only then, if `/` is to look like them, one experimental island (the job cards) and the missing list API. **Not run:** whether the installed `cancho-ui` compiles an `island` for this project.

## 4. Pending decisions (the owner's)

1. Dark only, as the design is, or dark and light by the browser's setting (the stylesheet of docs/studio-variations.md section 7 does both today).
2. Whether the worker should write a stage (`N.stage`) and the server add a thumbnail name to the closed list, so that the cards can have thumbnails and progress.
3. Whether `/` stays a cancho-ui page or is written by the server too, as the design proposes.

## 5. Applied (2026-11-16)

**What changed.** `results.css()` is the design's tokens (dark only: the design has no light scheme; the earlier light and dark stylesheet of docs/studio-variations.md section 7 is replaced). `results.open` / `results.close` write the start and end of every server-written page: the head with the stylesheet and the viewport, the optional refresh, the title, the header (brand, **Studio, Edit a picture, Comparisons, Jobs**, the open place marked `aria-current=page`) and the queue's status, `GPU busy &middot; R running &middot; Q queued` or `GPU idle` (`results.counts`, from the 200 newest jobs of `queue`).
* **`/jobs`**: cards, newest first (a grid of 260 px or more): the state badge (`queued`, `running`, `done`, `failed 3`), `#id`, the prompt in two lines, the kind, a moving bar while it runs (still when the browser asks for reduced motion). **A finished edit is its own thumbnail** (the whole PNG, lazily loaded, for the 24 newest only: the server cannot shrink a picture); a video has none, only a label.
* **`/jobs/N`**: the result beside the details: **Prompt as written**, **Rewritten by the local model** when there is one, **Parameters** (the kind and every `--name value` of the job's argument list, then the rewrite setting), **Files** (the ones that exist, and the log); a failed job shows an alert in place of a result.
* **`/sets/N`, `/compare`, `/compare/N`, `/edit`**: the same header and classes; the comparison's three choices are buttons under the pair (`Left is better`, `No difference`, `Right is better`); the edit form has a dashed drop zone for the picture and the instruction as a text area.
* **Fonts.** The stack is `'IBM Plex Sans', system-ui, ...` and `'IBM Plex Mono', ui-monospace, ...`: an installed IBM Plex is used, **nothing is loaded from Google Fonts** (the design links it; the studio is local and offline by default). `studio/check.sh` holds that the stylesheet has no `@import` and no `url(`.

**Checked.** `studio/check.sh`: 341 ok, 0 failed (the `UI` page checks included): 12 new checks of the design (the header and the open place on five pages, the status idle with one waiting job and busy with one running, the running badge and bar, the parameters in order, the thumbnail, the file list, the failed alert, the drop zone and text area, the font stack and no outside file), and the older checks that read the old HTML moved to the new (the card's badge and `#id`, `Prompt as written`, the order of the three choices). **Seen in the browser pane** (dark, 800 px wide): `/jobs` with eight jobs (done edits with thumbnails, running, queued, failed), a done edit's page, a failed job's page, `/edit`, a set of three, a pair in `/compare/N`.
**Two things that were not the design's fault and were fixed in the checks.** A fixed `sleep 1` after starting a server, and the worker tests' fixed sleeps, failed when the machine was busy (another session's compile took the load average to 17 and 22): they now wait for the server's `listening` line and for the state file.

**Not done, as section 1 said:** thumbnails for videos, a stage line and a real progress bar (the worker writes none), the picture preview (script), **Run again** and **Reuse prompt**, and the Studio page `/` (still the cancho-ui page, with its own look: the design's composer and job grid are not built; the nav's "Studio" leads to it). Not seen in a browser: a phone width, the light scheme (there is none), and a job list of 200.

## 6. The main page, `/` (measured, 2026-11-17)

**What was missing.** Section 5 applied the design to the pages the server writes; **`/` is the cancho-ui page** (built by `cancho-ui`, its own markup and script), and it kept cancho-ui's default look: the owner saw "the same page" and asked whether the form's style had changed. It had not.
**What was done.** The server now serves `/` with **the stylesheet linked before `</head>` (after the page's own `<style>`, so that it wins) and the header right after `<body>`** (`results.decorate`: a page that does not have both tags is served as it is). The page's JavaScript and markup are untouched; `style.css` has rules for its elements, written under `#root` so that they outrank the page's own (`.cui-field input` set a transparent, borderless field): a dark card for the form, fields with a border, a focus ring, the unused fields dimmed, the Start button disabled in grey. The heading is "New job" (the header already carries the brand).
**Checked.** `studio/check.sh` **390 ok, 0 failed** with `UI` set, 385 without (the page checks need the cancho-ui binary): the stylesheet linked once, after the page's own style and before `</head>`; the header once, marking Studio as the open place and before the app's root; the app's root and script where they were; `/` equals the built `index.html` with those two additions removed; the stylesheet holds the page's rules; and a page with no `<head>` and `<body>` tags is served as it is. **Seen in the browser pane** (800 px wide, the form's first fields and a scroll further).
**Not done:** the header's counts are those of the moment the page loaded (the page does not refresh them; the page's own line "GPU busy: R running, Q queued" under *Recent jobs* does, every 5 s); the narrow (phone) layout was not looked at.

**Recent jobs (2026-11-17, replaced the same day).** First built as a list of text cards in the cancho-ui page itself (PR #202: the page read `GET /api/jobs`); **corrected here:** that version had no thumbnails and its cards were not links, because cancho-ui's `Image` takes a literal `src` only and a `Link` cannot name a server route. It was replaced by **a frame**: the server serves `GET /recent`, the same cards as `/jobs` (thumbnails of finished pictures, state badge, `#id`, prompt, kind) for the 12 newest, with no header, refreshing itself every 5 s, and `<base target=_top>` so **a card is a link that opens the job in the whole window**; `results.decorate` puts `<iframe class=recent src=/recent>` before `</body>` of `/`. The page itself is as it was before #202 (it still does not read `/api/jobs`, which stays for scripts). **Progress:** a running job's card reads its log (no change to the worker): the last `== stage` line, and `steps N` followed by `step i:` (an edit or a picture; i from 0) or `step i of N:` (the video sampler) give "stage &middot; step d of N" and a bar d/N wide; a log with none of these keeps the moving bar. **Measured:** `studio/check.sh` 401 ok, 0 failed with UI (the three log shapes, the frame once and last, `/recent` without a header and the same ids as `/jobs`, OpenAPI); seen in the browser pane (a running image job: "transformer &middot; step 4 of 8", bar at half). **Not done:** the frame's height is fixed (760 px, 900 px on a phone; it scrolls), and the progress of the stages before the steps (the model's start-up shows only its stage name). *Corrected 2026-11-17 (later):* **a video's card now has a thumbnail**: not a new file, the card holds `<video muted playsinline preload=metadata src=/files/N/video.mp4#t=0.1>`, which the browser draws as its first frame (seen with a 640 x 352 clip; for the 24 newest done jobs, as for pictures). **The phone layout was looked at**, in the pane at 375 x 812: `/`, `/jobs`, `/jobs/1`, `/edit`, none wider than the screen (`scrollWidth` 375); the first look found 32 px gutters (too wide) and a frame too short, so under 600 px the gutters are 16 px, the grid is one column and the frame 900 px. Not seen: a real phone, a tablet width, the comparison pages, a set of variations at that width.
