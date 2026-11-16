# cancho-studio: the visual design, and whether islands help (2026-11-16)

> **Status: a design to apply and a reading of islands; nothing here is built.** Claims are tagged **[read]** (from a file, not run) or **[not run]**. The owner supplied a design (section 1) and asked whether the page could use islands "as in cancho-ota" (section 3), and said to look at how they are used in code, not only in the docs; section 3 is from the code.

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
