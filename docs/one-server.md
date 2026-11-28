# One server for every model (design, written before the code, 2026-11-28)

> **Status: built against fake runners (section 7); designed first, 2026-11-28.** Marks: **[read]** (in the code or a document of this or another repository), **[measured]**, **[estimate]**,
> **[choice]**. A claim that turns out false is corrected here, in place.

## 1. Why

cancho-gpu serves two language models, each with a server of its own **[read]**:

| | Qwen3.8-27B | Kolibri 1.1 |
|---|---|---|
| program | `build/cancho-gpu-serve` (`device/serve-main.cho`, 1,093 lines) | `build/cancho-gpu-kolibri` (`kolibri/kserve.cho`, 1,216 lines, and two mains) |
| routes | `/v1/chat/completions` (with a stream), `/api/generate`, `/v1/models`, `/health` | `/v1/chat/completions` (tools, no stream), `/api/generate`, `/v1/models`, `/api/tags`, `/health` |
| kept between requests | the prompt's prefix: delta-state cuts and checkpoints (`docs/serving.md` 7, 8, 12) | the session's ids, on disk with `--cache` (`docs/kolibri.md` 28) |
| HTTP | by hand on `tcp_listen` and `conn_read` | by hand on `tcp_listen` and `conn_read` |

Each reads requests its own way and has its own refusals, answer formats and `--api-key`; every route shared by both was written twice
(`/api/generate`, this week). A client has to know which port holds which model. Only one fits on the GPU at a time: of 64 GB, the 27B
with its caches takes about 30 **[estimate: `docs/serving.md` 12.2]** and Kolibri's weights alone 41 **[read: `docs/kolibri.md`]**.

What decides the shape is cancho's own rule, that a program's authority is what its report says (`cancho authority`) and is no more
than it needs: **the program that answers the network should not be able to touch the GPU, and the program that holds the GPU should
not face the network.** Today each server is both. The APIs are kept because clients speak them (OpenAI's chat completions; the
`/api/generate` of `cancho-code`'s agent), not as a design to follow.

## 2. The shape

```
client --HTTP-->  front (cancho-web)  --HTTP on 127.0.0.1-->  runner: cancho-gpu-serve   (Qwen3.8-27B)
                  routes, checks, key,                        runner: cancho-gpu-kolibri  (Kolibri 1.1)
                  which runner is up                          (one at a time)
```

* **The front** (`front/front.cho`, `build/cancho-gpu-front`) is a cancho-web service: every route declared once (so in an OpenAPI document), each refusal in the form of its API
  (`cancho-web/docs/design.md` §13, built for this), the bearer key held by `web.guard`, the models listed from one table. Its authority
  is the network in, connections out to 127.0.0.1, and starting and stopping its runners; **no `ffi`** [choice].
* **A runner** is today's server of that model, at first unchanged, listening on a loopback port the front gives it. The front starts
  it (`exec_spawn_detached` **[read: `std.process`; `docs/studio.md` 4 uses it]**), waits for its `/health`, forwards it requests, and
  ends its process group before starting another one.
* **A request is forwarded as it came** [choice]: the front reads `model` from the body to know where it goes, and sends the body on.
  The answer comes back as it is, a stream as a stream. Each engine keeps its own template, tools and refusals; the front holds only what
  every model shares. So a model the front does not know, a body that is no JSON object, a missing key, a wrong method or route are the
  front's refusals, and everything about a prompt is the runner's.

**Processes, not one program with both engines** [choice]. Authority is per program, so the split above needs two. And memory: the two
engines do not fit together on this machine, a process that ends gives back all of its device memory, and an engine unloaded inside a
program that keeps running would need every buffer freed, which the shim cannot do (`cgpu_alloc` only grows,
`shim/canchogpu-metal.mm`). A runner that crashes takes nothing else down: the front answers 502 and starts it again.

## 3. What exists, read

* **`http.server` waits without blocking** (`cancho/docs/http-server.md` section 10): `hold` turns a request into a ticket and the loop
  goes on; `answer` answers it later. The front holds a request while its runner works.
* **It polls the application's own handles** (same section): `poller`, `first_token`, `foreign`. The connection to a runner is watched in
  the server's own poller: one loop serves the clients and reads the runners.
* **It streams an answer on a socket server** (section 14, built): `stream` and `space`. With cancho-web's `stream_head` and `chunk`
  (§13), a runner's events are passed on as they come.
* **cancho-web reads a request body whole** (to its buffer; a 413 past it): a primer of 38 KB is far under any buffer that needs
  choosing **[read: `docs/studio.md` measured 16 MiB as a ceiling, not a cost]**.

## 4. What to build

1. **The front** (`front/front.cho` and its `cancho.toml`, built against the `web` store): the table of models (a name, the runner's
   program and arguments, a loopback port), the routes, the forwarding, the runner's life (start, health, stop; again after a crash; a
   swap when another model is asked for), one request at a time to a runner (as the runners are).
2. **The runners: nothing at first.** What the front makes redundant comes out of them later (their API keys, their listing routes).

## 5. Held to

* **The same bytes.** For each route and each model, a request through the front answers what the runner answers directly (greedy):
  `tests/serve_long.sh` and `tests/kolibri_serve.sh`'s cases sent both ways.
* **A swap.** A request for the other model ends the first runner and starts the second; the first's memory is given back (the footprint
  of the processes, measured); a request for the first after that starts it again.
* **A runner that dies** mid-answer: a 502 (or the stream ends without its last chunk), and the next request starts it again.
* **The front's refusals** in each API's form; the runner's passed on as they are.
* **Authority**: `cancho authority` on the front names the network, the loopback connection and the process calls, and no `ffi`.
* **What it costs**: the time to a first token through the front against direct (one loopback hop: **[estimate]** well under a
  millisecond); a swap (Qwen loads in 2 to 4 s **[measured, `docs/serving.md` 12]**; Kolibri is to be measured).
* Without the model, in the gate: the front against a fake runner (a cancho program that answers fixed bytes), its routes, refusals,
  forwarding, streaming and restarts.

## 6. Not in it

More than one request on the GPU at a time; two models loaded together; models other than these two; TLS (terminated in front, as
cancho-web says).

## 7. As built, and what was shown (2026-11-28)

```sh
scripts/build-front.sh          # needs checkouts of cancho (packages/http-server), cancho-web and cancho-schema: front/deps/*.lock pin them
build/cancho-gpu-front 8080 front/runners --env HOME=$HOME --env PATH=/usr/bin:/bin \
    --model qwen3.8:27b-mlx qwen3.8 18520 --model kolibri-1.1 kolibri 18490 --cache $HOME/.cache/cancho-media/kolibri-cache.bin
```

* **`front/front.cho`** (one program, 1,249 lines): the API of section 2 declared with `web`, `dispatch` and `guard` before
  anything else; a request for a model is held (`http.server`'s `hold`), its runner started if another is up (the other ended first),
  connected to on 127.0.0.1, sent the route and the body as they came, and its answer passed back: a whole answer with the front's own
  head (the connection kept open when the client asked), a stream (the runner's `text/event-stream`) re-framed as chunks with
  `web.stream_head` and `web.chunk` as it comes. Up to 8 more wait their turn in an arena of 2 MiB a request; past that, a 503 with
  `Retry-After`. INT and TERM end the runner with the front (`signals("INT,TERM")`), so no model is left on the GPU. A runner's
  standard error is copied to the front's.
* **`front/runners/`**: the two runners, `qwen3.8` (`cancho-gpu-serve` on this checkout's kernels and Ollama's store) and `kolibri`
  (`scripts/kolibri-serve`), each a script of three lines.
* **Its authority** (`front/authority.txt`, held by `front/check.sh`): `net_in("127.0.0.1:")`, `net_out("127.0.0.1:")`, `exec_under`,
  `exec_detached`, `child_signal`, `pipe_read`, `fs_read("")` (the open of the runner directory), `signals("INT,TERM")`, and **no
  foreign code**. It listens on loopback only [choice]: a client on another machine reaches it through something in front.
* **`front/check.sh`**, 24 checks against two fake runners (`front/fake_runner.cho`, which answers fixed bytes and streams three events
  200 ms apart): the open and keyed routes, forwarding with the body byte for byte, a swap (one runner up after it), the refusals in
  OpenAI's form and `/api/generate`'s, a stream whose events arrive apart (not at its end), a runner that ends without an answer (502)
  and is started again, a request that waits its turn behind a stream, the runner ended with the front, the authority report. **Seven
  runs: six passed; one failed the request that waits behind a stream** (its answer timed out, the stream was empty), which was not
  reproduced in the five runs after it; not explained.
* **Found: the compiler in cancho at this pin traps on a `Net` narrowed to `"127.0.0.1"`** at `tcp_listen_on`, where
  `cancho/docs/native-sockets.md` 12.3 says it means the same as `"127.0.0.1:"` (any port on 127.0.0.1); the Rust compiler takes
  both (measured with a program of 20 lines: `"127.0.0.1"` exit 132, `"127.0.0.1:"` and the Rust compiler's build of either, 10). The
  front narrows to `"127.0.0.1:"`. For cancho.
* **Not yet run: the real runners through the front** (section 5's same bytes, a swap's cost, the footprint); the GPU was busy with
  cancho-code's evaluation.
