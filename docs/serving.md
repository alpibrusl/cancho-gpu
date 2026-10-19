# Serving: an OpenAI-compatible chat endpoint over the 27B

`cancho-gpu-serve` answers `POST /v1/chat/completions` with Qwen3.8-27B running on a Metal GPU, whole or as
server-sent events. It is `device/serve-main.cho` over the engine of `device/qwen35.cho`, with the chat
template and the JSON in `src/chat.cho`.

```sh
scripts/build-serve.sh
build/cancho-gpu-serve ~/.ollama/models tests/chx --port 8080 --api-key secret

curl localhost:8080/v1/chat/completions -H 'Authorization: Bearer secret' -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name three colours."}],"stream":true}'
```

| flag | |
|---|---|
| `--port N` | default 8080 |
| `--api-key KEY` | every route but `/health` needs `Authorization: Bearer KEY` |
| `--model NAME` | the id in answers and `/v1/models` (default `qwen3.8:27b-mlx`) |
| `--think` | the model reasons first unless the request says otherwise |
| `--no-spec` | no draft head: one token a pass (25 tok/s) and 2.7 GB less memory; by default the model's draft head is used (a chain of two drafts, three rows checked a pass) |

**It listens on every address of the host** (`tcp_listen` binds them all; there is no way to ask for loopback),
so on a network use `--api-key`; without one it says so at startup.

## 1. What it takes

`GET /health`, `GET /v1/models`, and `POST /v1/chat/completions` with:

* `messages`: roles `system` (first only), `user`, `assistant` (with an optional `reasoning_content`); a
  `content` that is a string or an array of `{"type":"text","text":...}` parts.
* `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stream`,
  `stream_options.include_usage`. Defaults are the model's own (temperature 1, top_p 0.95, top_k 20);
  `temperature: 0` is greedy.
* Thinking: `enable_thinking` (bool), `think` (bool, or an effort), `reasoning_effort` (`low`, `medium`, `high`).
  **Off unless one of them turns it on** (the template's own default is on, which a client that has never heard
  of thinking pays for in tokens; `--think` flips the default). On, the answer carries `reasoning_content`
  (the text before `</think>`) beside `content`, in the stream as deltas of that name.

Refused, with a 400 that names why: tools and tool messages, images and audio, `n` other than 1, a system
message that is not first, no user message, a body that is not JSON, a prompt that does not fit the context.

## 2. The chat template

The model's own (the `chat_template` of its `tokenizer_config.json`, which is Jinja), written out in
`src/chat.cho` for the requests above. `tests/chat.sh` renders 23 requests, refusals included, with this
compiler (`cancho-gpu chat request.json`) and with jinja2 through `tools/chat_template.py` (the way Hugging
Face loads it) and requires the same bytes; it is in the gate. The one place they differ on purpose: the
template renders an image part as a placeholder, this server refuses it.

## 3. Context, and what it costs

The cache holds 1024 positions, and attention is compiled for 256, 512 and 1024 of them, the smallest that
covers the conversation, so a short chat does not scan a long cache. The prompt must leave room for at least
one token; `max_tokens` is cut to the room left and the answer then ends with `finish_reason: "length"`.
Longer needs attention split across threadgroups, which the surface language cannot say yet (`docs/kernels.md`
section 4).

Each request starts from nothing: the conversation is prefilled again, eight tokens at a time. There is no
prefix cache. Measured on an M4 Max (one run, the machine otherwise quiet): a short prompt's first token
arrives in about 0.35 s and the stream runs at 23.5 tokens a second; a 330-token prompt takes about 5 s before
its first token.

## 4. How it is built

* **One request at a time.** The GPU holds one conversation's states and cache, so the engine is reset
  (`qwen35.reset`: the convolution windows and delta states zeroed with `cgpu_zero`, the mask closed) and used
  by one request; the next waits in the listen queue. A connection carries one request (`Connection: close`).
* **Memory.** The server runs for days in a bump allocator that never frees, so after each request it gives
  back what the request took (`mem.release_to`: the words zeroed again, the text pool rewound). A kernel is a
  record in that memory, so a kernel compiled while handling a request would be given back with it: the batch
  kernels are compiled at start, not on first use (the first version compiled them lazily and every request
  after the first ran kernel 0; `tests/serve_check.py` repeats a request for that).
* **Streaming.** Each token is decoded with the others of its phase and what is ready is sent: whole
  characters only (a token can end inside a UTF-8 sequence), and trailing whitespace is held until something
  follows it. A write that fails (the client went away) ends the generation.
* **Authority.** `cancho authority` on the server says `net_in("")` (a listener on any port: `Net` cannot be
  narrowed to a port that is a command-line flag), `conn_accept`, `conn_read`, `conn_write`, and the device
  library as `ffi("libmetalgpu")`, unbounded as such. No `net_out`, no `io_write`. The compiler itself is
  unchanged (the gate holds it to no network).

## 5. Held to

`tests/serve_check.py` (a Mac and the model; not in the gate): for three requests (a question, a system
message with a second turn, thinking on) the greedy answer of the server is the greedy answer of the engine
fed the rendered prompt directly (`cancho-gpu-run35`); the stream is the same answer delta by delta; a request
repeated after another gives the same answer; a client that leaves mid-stream does not stop the server; a
prompt past the context, bad JSON, tools, an image, `n = 2`, `max_tokens: 0`, a missing or wrong key, a GET on
the chat route and an unknown route are each refused with their status. 22 checks, all passing on the first
full run after the allocator bug above was fixed.

No official OpenAI client was used (none was installed here, and the checks speak HTTP directly); a client that
sends only what section 1 lists should work.

## 6. Not here

Concurrent requests and batching them (one GPU conversation at a time), a prefix cache across turns, tools,
images, stop sequences, log-probabilities, a read timeout on a connection (a client that connects and sends
nothing holds the server), TLS, and a smaller memory footprint (14.6 GB of weights).
