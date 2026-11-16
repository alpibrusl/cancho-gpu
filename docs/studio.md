# cancho-studio: a web app to make and edit images and videos (design, 2026-11-16)

> **Status: design, nothing built.** Written from reading `cancho-web`, `cancho-ui` and `cancho`'s `std.process`; the claims below say what was read and what was not run.

## Goal

A page on `localhost` where one types a prompt (and, for an edit, names a photo), starts a job, sees it progress, and opens the result. It is the `media` command line (`docs/cli.md`) with a browser in front: the models, the limits and the one-GPU-job lock are the CLI's, unchanged.

## Parts

* **Server, `cancho-web`** (`studio.cho`): a typed API declared once (and so its OpenAPI document).
  * `POST /api/jobs` `{kind: "video"|"edit", prompt, image_path?, frames?, size?, seed?}` -> `202 {id}`; `422` as `problem+json` for what `media` would refuse (sizes not multiples of 32, over the measured limits).
  * `GET /api/jobs` and `GET /api/jobs/{id}`: state (`queued|running|done|failed`), the stage line the CLI last printed, seconds elapsed, the files.
  * `GET /files/{id}/{name}`: `edited.png`, `video.mp4`, `video.gif`, with the right `Content-Type`; names are from a closed list, never a path from the client.
  * `GET /api/models`: the registry of `tools/media.py`.
  * One job at a time (the CLI's `gpu.lock` already refuses a second); the server queues, it does not run two.
* **Worker**: the server spawns `scripts/media` with `exec_spawn` (`std.process`: an `Exec` narrowed to the script's prefix, arguments as a `\0` list, output captured with a bound and a deadline). The authority report then names the one program it can start. The prompt travels as an argument entry, never through a shell.
* **Page, `cancho-ui`** (`studio.cui`): a form (`TextField`, `Button`), a job list refreshed by `every N Event`, `net.post` to create a job, `net.get` to read it. `requires Network; requires NetworkWrite;`.

## What was read that decides the design

1. **`net.post` sends JSON text and `FileField` reads a bounded text file** (`cancho-ui/docs/file-upload.md`: "not done (binary, multipart)"). A photo cannot be uploaded from the page. **Decision for the first version: `edit` takes a path on the machine the server runs on** (this is a local tool; the same as the CLI). The server refuses a path that is not a regular readable image file, and never copies it anywhere but its own working directory. Upload is a later slice and needs a cancho-ui change.
2. **`Image`, `Media` and `Link` take a literal `src` / `to`, checked when the page is built** (read on cancho-ui `main`, 12e57c6, 2026-11-16: `crates/cancho-ui-ir/src/media.rs` "`src:` of an Image is a text literal, the path of a file of the site", `media-src-invalid`, `asset-missing`, 20 MB cap; `structure.rs` "`to:` of a Link is a string literal ... a text that reads state is not a link"). A job's result, which exists only after the page was compiled, cannot be shown or linked from a cancho-ui page. **Decision, route 3: the results page is HTML the server writes** (`GET /jobs/{id}`: the prompt, the stage, an `<img>` or a `<video>` of `/files/{id}/...`). The cancho-ui page (form and job list) links to it by a navigation the browser does, not by a `Link`: see "What cancho-ui would need". cancho-ota-ui was checked: its console (`ui/trip.cui`) is text only, so it shows nothing to copy for pictures; its server / worker split is what this design takes (the server queues and is the one writer, a worker alone does the expensive work).
3. **cancho has no `exec` that the compiler lets a library hide** (`std/process.cho`): the caller spawns. That is what we want here.
4. **`cancho-web` is on the `http.server` package**, one thread and a `Poller`, no FFI. A long job must not block the poll loop: the server starts the child detached (`exec_spawn_detached`), and each request reads the child's status file, so no request waits on the GPU.

## What cancho-ui would need to work with photos

Each is a gap read in the source, not a measured failure; none is built. In the order they would unblock the app:

1. **A dynamic source for `Image` and `Media`**: `src:` from a `String` in state (a `net.get` answer or a job's `result` field), with the rules the literal has today kept as run-time ones: a closed set of extensions, a same-origin path only (as `net.get` is), `alt:` still mandatory, `ratio:` still declared before load. Build-time `asset-missing` cannot apply; a failed load needs a state of its own (`on Error`).
2. **Binary upload**: `net.upload(url, file)` as `docs/file-upload.md` sketches (bytes as the body, never read into a JavaScript string, a handle type that is not a `String`, its own capability line), with `FileField` accepting `image/*` and giving the handle plus name and size. Today `FileField` reads UTF-8 text and `net.post` sends JSON.
3. **A `Link` whose `to:` is a run-time string**, same-origin only, for a file the page cannot embed (a download of `video.mp4`).
4. **A local preview of the chosen photo** (`Image { src: <the handle> }`), so the page shows what is about to be sent: a revocable object URL, never leaving the page.
5. **Replay support**: `cancho-ui test` traces would need scripted answers for the upload and the dynamic source, as `net.post` has.

Until 1 and 2 exist: `edit` takes a path on the server's machine, and the result is shown by the server's own HTML page.

## Safety

* Bound to `127.0.0.1` only; a bearer token on every route (`web.guard`), read from a file with mode `0600`, not from the command line.
* The prompt is data: a `\0` is refused (`std.process.add` returns 22). Output directories are `<jobs>/<id>/`, `id` made by the server.
* `edit`: the photo is read, never committed or uploaded; the server logs the path's basename, not the prompt.
* The page requires `Network` and `NetworkWrite` and nothing else; `cancho-ui authority` lists them.

## Built so far (2026-11-16)

* **Slice 1**: `studio/studio.cho`, the loopback server (page, `/api/models`).
* **Worker, slice 2a**: `studio/worker.cho` (+ `studio/files.cho`): `studio_worker <script> <home> <job-dir>` starts the script on the argument list in `<job-dir>/argv` (NUL-separated, as the server will write it), output to `<job-dir>/log`, and writes `<job-dir>/state` (`running`, then `done` or `failed N`). One job per process; the queue (the server writing `argv` and starting workers one at a time) is not built. `studio/check.sh` runs it against a fake script, no GPU: 9 checks pass (arguments with shell metacharacters arrive as data; exit 7 is `failed 7`; a missing `argv` writes nothing; a relative job directory is refused). **Not checked:** the worker running the real `scripts/media`; the authority report (the worker holds `Exec("")`, not narrowed to the script's prefix: a gap).

## Slices

1. Server with `/api/models` and `/api/jobs` (fake worker that sleeps), tested over real sockets as `cancho-web/examples/users` is. Gate: OpenAPI document checked in.
2. Worker spawn of `scripts/media` for `video`; one real job through the API.
3. `edit` by server-side path.
4. Page: form, list, links; checked with `cancho-ui check` and `test` with a replay trace.
5. Result shown inside the cancho-ui page (needs cancho-ui items 1 and 4), then browser upload (item 2).

## Not decided

* Where it lives: a directory `studio/` in this repository (it needs `scripts/media` next to it) or its own repository. Proposed: here.
* The cancho-ui changes above are proposals; none is filed yet.
* `image` (text to image from a free prompt) is not wired in the CLI (`docs/cli.md`), so the page does not offer it.
