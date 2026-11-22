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
| `--no-prefix-cache` | turn off the prefix cache, which is on by default: it keeps the delta state at the end of each conversation's history and reuses it on the next turn (section 7) |
| `--cache-slots N` | conversations the prefix cache keeps, 1 to 16 (default 4; section 8). Each slot is 468 MB of device memory, touched as it is used |
| `--no-spec` | no draft head: one token a pass (about 27 tok/s) and 2.7 GB less memory; by default the model's draft head is used (a chain of two drafts, three rows checked a pass): streaming a 128-token reply measured 35 tok/s with it against 24 without (one request; prose, code and a sampled answer at temperature 1 each 34.7-35.6 with it; 20.6-25 without; the machine had other jobs, load average 14) |

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

A request that does not continue the last conversation starts from nothing: the conversation is prefilled, sixteen tokens at a time and then eight (section 8.7; the prefix cache of section 7 keeps the last conversation's state; Ollama on the same requests is section 8). Measured on an M4 Max (one run, the machine otherwise quiet): a short prompt's first token
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

Concurrent requests and batching them (one GPU conversation at a time), more than one cached conversation (one slot, section 7; several are designed in section 8), tools,
images, stop sequences, log-probabilities, a read timeout on a connection (a client that connects and sends
nothing holds the server), TLS, and a smaller memory footprint (14.6 GB of weights).

## 7. A prefix cache (design, written before the code)

> **Status: a design, step 1 (the probe, 7.9) and step 2 (built, held to 7.6 and measured in 7.10) and step 3 (on by default; `--no-prefix-cache` turns it off) are done.** Marks as elsewhere: **[measured]** (a command is
> named), **[read]** (from the source or a document, not run), **[estimate]** (arithmetic from measured numbers),
> **[open]**.

### 7.1 The problem, measured

Every request is prefilled from nothing (section 3), and a chat client sends the whole conversation again each
turn. **[measured]** on an M4 Max with the 27B (a scratch server, `max_tokens: 1`, `temperature: 0`, one prompt of
repeated filler sentences, the whole request timed over HTTP; the same request sent twice):

| prompt tokens | first request | the same request again | prefill rate |
|---|---|---|---|
| 18 | 0.27 s | 0.21 s | 68 tok/s (fixed cost dominates) |
| 158 | 1.50 s | 1.50 s | 105 tok/s |
| 578 | 4.99 s | 5.00 s | 116 tok/s |

The repeat is no faster, so nothing is reused today, and the cost is linear in the prompt at about 110 tokens a
second. A prompt of 1,000 tokens (most of the 1,024 context) therefore waits about 9 s **[estimate]** before its
first token, on every turn, while the 128 tokens of the answer take about 4 s at 35 tok/s. (The 1,000-token row was
not run: the filler prompt I tried past 578 tokens was refused for the context, as section 1 says.)

The aim: the second and later turns of one conversation prefill only what is new (the last answer and the new user
message), so the first token of a turn arrives in about 0.3 to 1 s whatever the history **[estimate]**: new tokens
over 110 tok/s, plus the restore below.

### 7.2 What makes this less simple than "keep the KV cache"

Qwen3.8 alternates three gated-delta layers with one attention layer (`is_attention(l)`: `l % 4 == 3` **[read]**;
64 layers: 48 delta, 16 attention). The two kinds hold state differently:

* An **attention** layer's keys and values are a cache indexed by position, and the mask says how many rows are
  readable. Rewinding to position `c` is closing the mask above `c`; the rows above are overwritten before they are
  read again (`qwen35.reset` says so). **Truncation is free.**
* A **delta** layer's state is a 128 x 128 matrix per head, and a convolution window, folded from every token so
  far. **It cannot be rewound**: the state after token `c` cannot be had from the state after token `c + 50`.
  Reusing a prefix means keeping a copy of the state taken at exactly that position.

So the cache is a **snapshot** of the delta and convolution state at a chosen position, plus the token ids it
covers. Sizes **[read, from the allocations in `qwen35.load`]**: a delta layer holds 4 x 48 x 128 x 128 = 3,145,728
bytes of state and two convolution windows of 4 x 3 x 10,240 = 122,880 bytes each (the engine takes them in turn,
so both are copied); 48 layers make **162.8 MB**. The keys and values (16 layers x 2 x 4,194,304 bytes = 134 MB at
the full 1,024 rows) are **not** copied: they stay where they are, and the snapshot is only used while they still
hold this conversation (7.4). Against 14.6 GB of weights, a snapshot is 1.1%.

### 7.3 Where to cut

The snapshot is taken at a position `c` that a later request is likely to share. The template decides which
(`tools/chat_template.py` and `cancho-gpu chat` agree on these **[measured: `cancho-gpu chat` on five requests]**):

* thinking off (the default), turn 1: `...<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
  and turn 2, which re-sends turn 1 with the answer: the whole turn-1 prompt, then `4<|im_end|>\n<|im_start|>user\n...`.
  **The whole previous prompt is a prefix of the next one.**
* thinking on: turn 1 ends `assistant\n<think>\n`; in turn 2 the same assistant turn is rendered
  `assistant\n<think>\neasy\n</think>\n\n4<|im_end|>` (when the client sends `reasoning_content` back; without it the
  template renders an empty think block, a different text). So the prefix that is always shared is up to and
  including **`<|im_start|>assistant\n`** of the last turn; what follows depends on the mode and on the client.

**Rule:** `b` is the index just after the last `<|im_start|>assistant\n` in the prompt's ids (the special token's id,
then `assistant` and `\n`: three tokens, `248045 74455 198` **[measured, 7.9]**), and the
cut is `c = 8 * floor(b / 8)`, a multiple of `qwen35.mb()`. (Since 8.7 the chunk is 16 and so is the multiple; the reasons are the same.) Two reasons for the multiple of 8:

1. A fresh prefill feeds chunks of 8 tokens from position 0 (`qwen35spec.feed`), and the chunks use the
   matrix-unit kernels while single steps use another set. A resumed prefill that starts at a multiple of 8 forms the
   same chunks as the fresh one, so the arithmetic is the same and **the resumed state should equal the fresh state
   bit for bit [open: step 2 compares them]**. Cutting at an arbitrary `b` would change which tokens go through
   which kernel.
2. Cost: up to 7 tokens (about 60 ms) are fed again on a hit.

The draft head (when the engine has it, `has_head`) fills its own cache alongside the prefill (`warm`): each chunk's
last row is paired with the token after it, `ids[pos + 8]`. At the cut that token comes from the request that took
the snapshot, and a later request may differ there, so **one head position can hold a pair a fresh prefill would not**.
That changes which tokens the head drafts, not which are accepted (the main model checks every draft), so the answer
is unchanged and only the acceptance rate could move **[open: measured in step 2 as the acceptance over a fixed
dialogue, with and without the cache]**.

### 7.4 One slot, and when it is valid

One snapshot, one slot, describing **the most recent request**: its ids `[0, c)` (at most 1,024 integers, kept in a
fixed area allocated at start, because the bump allocator is rewound after each request), the delta and
convolution state at `c`, and a flag. Decisions:

* On a request, `ids` is computed. If the slot is valid and `ids[0..c_slot)` equals the slot's ids and
  `c_slot < n_prompt` (at least the last token is left to be fed, for the logits), it is a **hit**: the snapshot is
  copied back into the engine, the mask is opened to `c_slot`, and `feed` runs from `c_slot`. Otherwise a **miss**:
  `reset` and a full prefill, exactly as today.
* Either way the request then takes its own snapshot at its own `c` (prefill to `c`, copy, prefill the rest). On a hit
  the new `c` is at or after the old one, so one conversation keeps extending its slot.
* The keys and values of `[0, c)` are the live ones. They stay valid only if no other request has run since the
  snapshot, and since every request rewrites the slot to describe itself, the slot is always about the most recent
  request: **a request from a different conversation is a miss and takes the slot**. Two chats taking turns
  therefore never hit (7.7).
* **Failure safety.** The flag is cleared when a request starts and set only after its snapshot is complete, so a
  request that fails, or a client that leaves, cannot leave a half-written slot that a later request trusts.

### 7.5 The copy

The device library has no device-to-device copy **[read: `device/metal.cho` lists `cgpu_alloc`, `cgpu_upload(_at)`,
`cgpu_download(_at)`, `cgpu_zero` and the launch calls]**. Three ways, in order of effort: through the host (download
then upload; two crossings of 163 MB), a copy kernel in `.chx` compiled at start, or a blit in `shim/canchogpu-metal.mm`.
**Decided by step 1 (7.9): through the host.** Saving or restoring all 48 layers takes 3 to 5 ms either way, so a
new kernel buys nothing and the cache needs no change to the device library or to `tests/chx/`.

### 7.6 What it is held to

* **Same tokens.** For each of: two turns, three turns, thinking on and off, a system message, a long history near
  the context, `temperature: 0` and a fixed `seed`: the answer with the cache is the answer with it off
  (`--no-prefix-cache`), delta by delta in a stream. Preferably the logits of the first generated token are equal to
  the last bit **[open]**; the least accepted is equal tokens.
* **Misses are right.** A request after an unrelated one, a request that is an edit of an earlier message (a prefix
  that is shorter than `c_slot`), a regeneration of the same prompt, and a request that failed halfway each give the
  answer a fresh server gives.
* **Usage.** `prompt_tokens` stays the whole prompt; the tokens reused are reported as
  `prompt_tokens_details.cached_tokens`, as OpenAI's API does, and in the server's log line.
* **A number.** The first-token time of turn 2 and turn 3 of a fixed dialogue (history of about 150, 300 and 600
  tokens) with the cache against without, in the table of 7.1. The success bar, stated now: a hit's first token in
  **at most 25% of the miss time** at a 578-token history, and a miss (the snapshot's cost on a request that does not
  hit) at most **100 ms** slower than today.

### 7.7 Not in this design

* **Several conversations.** A slot per conversation needs the keys and values copied too (134 MB at full context
  **[read]**, less for a short prefix), so an LRU of two to four slots costs 0.3 GB each. Not first: one chat at a
  time is what this server does today.
* **A shared system prompt** (a second cut at the end of the system message) for many conversations with one long
  instruction. It needs several slots (section 8).
* **Contexts above 1,024** (`kernels.md` section 4) and **prefill speed** (the matrix-unit prefill of section 3).
  Both are separate, and a cache helps the second regardless: it makes the prefill happen once per token rather than
  once per turn.
* The cache was built off by default behind `--prefix-cache`; after step 3 it is **on by default** and `--no-prefix-cache` turns it off.

### 7.8 Steps

1. **Probe. Done (7.9).** The ids of the turn markers; the through-host copy and a `.chx` copy kernel of 162.8 MB
   timed; the primitive chosen. No server change.
2. **The slot. Done (7.10)** behind `--prefix-cache`: snapshot, restore, resume, the failure flag, `cached_tokens`; the
   checks of 7.6 in `tests/prefix_check.py`. Exit: equal tokens, and the decode rate measured.
3. **Measure and decide the default. Done:** on by default, because 7.10 met every bar of 7.6 (hit 8-15% of the miss time, miss cost zero within noise, same tokens, same decode rate); the open items (bit-equal logits, decode on prose, one slot) are listed there. Correcting 7.1's estimates in place where they are wrong.
4. Later, on its own design: several slots (section 8).

### 7.9 Step 1: the probe (measured)

`scripts/build-probe-copy.sh` builds `cancho-gpu-probe-copy` (`device/probe-copy-main.cho`, with `tests/chx/copy_rows.chx`
as the one kernel it needs), and `cancho-gpu-probe-copy ~/.ollama/models tests/chx` loads the 27B and prints the
following **[measured]** (M4 Max, one run of four passes, the machine otherwise quiet).

**Token ids.** The marker `<|im_start|>assistant\n` is **3 tokens**, `248045 74455 198` (`<|im_start|>` is one token,
not spelled out). On the two-turn dialogue of 7.3:

* thinking off: turn 1 is 16 tokens and turn 2 is 37; **the 16 of turn 1 are exactly the first 16 of turn 2**.
* thinking on, with `reasoning_content` sent back: turn 1 is 14 tokens; **the 14 are the first 14 of turn 2**
  (turn 2 goes on `248068 198 43518 198 248069 271 ...`, the think block with the reasoning in it).

So the rule of 7.3 gives, for turn 1, `b = 12` and `c = 8`. On a short prompt rounding to 8 gives up a third of it; on a
real history it is 7 tokens out of hundreds.

**Copying the 48 delta layers' states (162.8 MB, 144 buffers) between two sets of buffers:**

| | save (engine to snapshot) | restore |
|---|---|---|
| through the host (a download, then an upload, of each buffer) | 4 to 5 ms | 3 to 5 ms |
| a copy kernel, `copy_rows.chx` (all launches encoded, one wait) | 3 to 4 ms (GPU 2.7 ms, first pass 4.3) | 3 to 5 ms (GPU 2.6 ms, first pass 4.6) |

Both were checked by a round trip: a pattern of valid floats written into the buffers of layer 0 and layer 62,
copied to the snapshot, the engine's buffers zeroed, copied back, and read: **0 differing bytes** by each path.

**Decision: through the host.** The two take the same time to the millisecond, because the memory is unified; the
kernel would be a new file in `tests/chx/` for nothing. The server will use `cgpu_download_at` and `cgpu_upload_at`,
which exist. This is under the 100 ms bar of 7.6 by a factor of 20, and 0.2 to 0.3% of the 1.5 to 5 s a hit saves at
158 to 578 tokens. The 7.6 bar for a miss (at most 100 ms slower than today) is therefore met by the copy alone:
a miss pays one save, about 5 ms.

What the probe does not settle: whether a resumed prefill equals a fresh one bit for bit, and the head's acceptance
(step 2).

### 7.10 Step 2: the slot, built and measured

**What was built.** the prefix cache on `cancho-gpu-serve` (built behind `--prefix-cache`; on by default since step 3, `--no-prefix-cache` turns it off). `device/qwen35.cho`: `snap_new`, `snap_copy` (the
through-host copy of 7.9) and `open_prefix` (the masks of the engine and of the draft head, open for `[0, c)` and
closed above). `device/serve-main.cho`: the slot record (valid, `c`, its ids in 1,024 words allocated at start, the
snapshot, a scratch, the tokens reused), `cut_of` (the rule of 7.3: the last `248045 74455 198`, rounded down to a
multiple of 8, and never the whole prompt), `prefill_cached` (hit or miss as in 7.4; the slot is marked invalid when a
request starts and valid only after its snapshot is complete, or, on a hit that needs no new snapshot, after its
prefill). `src/chat.cho`: `usage.prompt_tokens_details.cached_tokens` in the whole answer and in the streamed usage
chunk (always present, 0 without the cache, as OpenAI's API has it); the server's log line gains ` cached=N` with the
flag. Without the flag nothing else changes: `tests/serve_check.py` (22 checks) passes, and `tests/chat.sh` (23
requests) passes.

**Held to** `tests/prefix_check.py` (a Mac and the model; not in the gate): two servers, one without the cache and one
with it, are sent the same requests with the same history; **all 68 checks passed** on the final run (also the table below, to within 0.02 s):

* **The same tokens.** Four conversations (3 to 4 turns of 190 to 800 tokens: plain; with a system message; thinking
  on with `reasoning_content` sent back; sampled at temperature 1 with seed 7), an interleaved second conversation,
  a repeated request, an edit of the first message, a request after a refusal and one after a client left mid-stream:
  **every answer is the same, token for token**, including the sampled ones (not a comparison of logits, but with a
  fixed seed at temperature 1 a difference in the last bits would show as a different token sooner or later; it did
  not in the 22 answers of up to 40 tokens compared).
* **The expected reuse.** Turn 1 reuses 0; a later turn reuses the previous prompt up to the cut (for example 184 of 399,
  392 of 605, 600 of 811 tokens: the cut leaves out the last assistant opener and rounds down to a multiple of 8);
  a second conversation in between takes the slot, so the first one's next turn reuses 0 and the one after reuses;
  a repeat reuses 680 of 688; an edited first message reuses 0; a refused request leaves the slot alone.

**First token, and the decode rate** (one run each; the history is the conversation so far and the request ends with a
new user message; the first token is `max_tokens: 1`; the rate is tokens from the first to the last streamed chunk of a
128-token answer, median of three):

| case | first token without | with the cache | reused | decode tok/s without | with |
|---|---|---|---|---|---|
| plain, turn 5, 811 tokens of history | 7.57 s | **0.61 s** | 800 | 54.7 | 55.3 |
| system message, turn 4, 700 | 6.59 s | **0.54 s** | 696 | 54.7 | 55.3 |
| thinking on, turn 4, 604 | 6.06 s | **0.89 s** | 600 | 55.9 | 56.6 |
| sampled, turn 4, 605 | 5.81 s | **0.61 s** | 600 | 56.0 | 56.6 |

**The bars of 7.6.** (1) A hit's first token in at most 25% of the miss time at about 600 tokens of history: **8 to
15%** (0.54 to 0.89 s against 5.8 to 7.6 s). Met. (2) A miss at most 100 ms slower: eight requests of about 330 tokens
each, the servers taking turns to go first, measured **188 ms faster** with the cache. That is not the cache: with both servers
started *without* it the second server is also faster, by 166 ms (a control, `prefix_check.py miss control`), so the
process started second is, for a reason not looked into, faster and the cache's own cost is about **-22 ms, which is
zero within the noise**. Met. (3) The decode rate is the same to 1%, so the draft head's one divergent pair (7.3)
costs nothing that this measures; the counting answer it was measured on is predictable text, where a draft head
accepts most of its guesses, so a loss on prose would show less here than there **[open]**.

**Corrections to the design.** 7.1 estimated a first token of "0.3 to 1 s" for a hit; measured 0.54 to 0.89 s at 600 to 800
tokens of history (the 0.2 s fixed cost and the new tokens at about 110 tok/s: the last answer, the new user message
and the up to 7 tokens re-fed). It guessed 4 ms for a copy (7.9) and a miss pays one; that held.

**Not settled.** The logits were not compared bit for bit (only the tokens they produce). The slot holds one
conversation (7.7; several are designed in section 8). Step 3 made it the default.

## 8. Several slots, and Ollama on this machine (design, written before the code; steps 1 to 3 built)

> **Status: steps 1 to 3 of 8.6 are built and measured (8.7): the slots meet their bar; the prefill went from 119 to
> 153 tok/s and does not meet its bar of 235. Steps 4 and 5 are not started.** Marks as in section 7. The aim is stated as a comparison: the same
> requests to Ollama 0.40.16 (its MLX runner, the same NVFP4 weights from the same store) and to `cancho-gpu-serve`
> on this machine (M4 Max, 64 GB), one at a time, with `tests/ollama_compare.py`.

### 8.1 Where we are, measured

`tests/ollama_compare.py both` **[measured, one run, 2026-11-21, thinking off, temperature 0, Ollama unloaded before
our server starts]**. First token is wall clock, streamed.

| case | prompt tokens | Ollama | ours | ours / Ollama |
|---|---|---|---|---|
| cold prompt, first token | 131 | 0.72 s | 1.20 s | 1.7 |
| cold prompt, first token | 403 | 1.65 s | 3.51 s | 2.1 |
| cold prompt, first token | 743 | 2.80 s | 6.57 s | 2.3 |
| decode, 256 tokens of a story | 44 | 47.1, 48.8 tok/s | 40.2, 40.3 tok/s | 0.84 |
| conversation, turns 2 to 4 (112 new tokens a turn), first token | 312 to 536 | 0.45 to 0.57 s | 1.24 to 1.26 s | 2.5 |
| conversation, decode of a 24-token answer | | 56 to 79 tok/s | 48 to 54 tok/s | 0.7 to 0.9 |
| two chats taking turns, turns 2 and 3, first token | 302, 369 | 0.33 to 0.45 s | 2.73 to 3.17 s | 7 to 8 |
| a 540-token system message, three different questions, first token | 545 | 2.07, 2.13, **0.16 s** | 4.68, 4.68, 4.85 s | 2.2, 2.2, 30 |

What the rows say:

* **Cold prefill is the largest gap and it is in every row.** Ollama prefills 743 tokens in 2.64 s by its own counter,
  281 tok/s; ours runs at 113 to 119. Our warm turn is slow for the same reason: the cache leaves 120 tokens to
  prefill and they take a second.
* **Ollama keeps more than one conversation.** Both of two interleaved chats hit; ours has one slot and both miss.
* **Ollama does not reuse a shared system message at once.** The first two questions each pay the whole prompt; the
  third takes 0.16 s. (Its runner seems to learn the branch point from the second request **[open: its source not
  read]**.) A snapshot taken at the end of the system message on the first request would hit on the second: this is
  the one row where a design can be ahead of it without being faster.
* **Decode is 84% of Ollama's on prose**, as `docs/native.md` N4c measured (88% there); on short predictable answers
  Ollama's depth controller takes it further ahead.
* Not in the table, and a larger difference than any row: **our context is 1,024 tokens; Ollama's is whatever
  `num_ctx` says, up to 262,144.**

### 8.2 Where our prefill goes

`CANCHOGPU_PROFILE=2 build/cancho-gpu-run35 ~/.ollama/models tests/chx "<681 tokens>" 1 --timing`
**[measured: 85 chunks of 8 tokens; each kernel timed alone, which overstates the small ones by their launch
overhead, so the shares are right and the sum, 78 ms a chunk, is above the 67 ms of the pipelined run]**:

| part | launches a chunk | ms a chunk | share |
|---|---|---|---|
| NVFP4 products through the matrix unit (`matvec_fp4*_batch`, 8 rows) | 336 | 53.8 | 69% |
| gated-delta work done **a token at a time**: `matvec_rows_f16` (a/b), `delta_step`, `qk_norm`, `gates`, `conv_step` | 2,309 | 19.2 | 24% |
| everything else: norms, RoPE, cache rows, attention | about 900 | 5.3 | 7% |

* The 48 delta layers run their five small kernels once a token, 2,309 launches a chunk, though the speculative
  round already has the batched forms (`conv_step3`, `delta_step3`, the a/b projections, gates and `qk_norm` over
  three rows: `docs/native.md` N4c, 18 launches a layer down to 6). The same over the chunk's rows is 288 launches.
* The weight products read 14.4 GB in about 50 ms, 270 to 290 GB/s: the 8-row matrix-unit kernel costs the same for
  1 row as for 8 (`docs/kernels.md` section 6), so it is bound by its own work per weight byte and not by the
  matrix unit. The f16 counterpart keeps that cost up to 16 rows and pays 2.3 times for 32 (`docs/kernels.md`
  section 7), with `ceil(m / 8)` activation fragments sharing each weight fragment. NVFP4 has no such kernel.
* **The roof [estimate].** A token is 2 x 26 x 10^9 multiply-adds, so Ollama's 281 tok/s is 14.6 x 10^12 a second and
  our 8-row pass is 6.6 x 10^12. Ollama is compute-bound and near what this GPU does; nobody passes it by much on a
  cold prompt. Sixteen rows at today's cost of a pass would be 16 / (0.054 + 0.008) = 258 tok/s, and 32 rows at 2.3
  times, 32 / (0.124 + 0.012) = 235: **parity on cold prefill needs the 16-row pass to cost what the 8-row one does,
  and that is not known until the kernel exists [open].**

### 8.3 The design, in four parts

**A. Several slots (the cache).** A slot is what section 7's one slot is, plus the rows of the attention caches
it needs, because another conversation will write over them:

* `ids[0, c)`, the delta and convolution snapshot at `c` (162.8 MB), and room for the K and V rows `[0, c)` of the 16
  attention layers (134 MB at `c` = 1,024 **[read: 7.7]**) and of the draft head's one layer. `--cache-slots N`, 4 by default:
  1.87 GB at most (4 x 468 MB: two snapshots of 162.8 MB and whole key and value buffers, 142.6 MB; this said
  "about 1.2 GB" before the second cut and the whole-buffer copy were counted), on a server that holds 16.7 GB.
* One slot is **resident**: its K and V rows are the engine's own, not a copy. Continuing the resident conversation
  is exactly section 7: no row is copied.
* A request picks the slot with the longest cut whose `ids[0, c)` are its first tokens and `c < n`. If that slot is
  not the resident one, the resident's rows `[0, c_res)` are copied out to its slot (they are still intact: decode
  only writes above the cut), the chosen slot's rows are copied in, its snapshot restored, and it becomes resident.
  Two copies of at most 134 MB and one of 163: under 15 ms at the 3 to 5 ms per 163 MB of 7.9 **[estimate]**.
* After the prefill passes the new cut, the snapshot goes to the slot that was hit (the conversation moved on), or on
  a miss to the least recently used one. Validity is section 7.4's rule per slot: invalid when written, valid when
  complete; a slot whose rows were never copied out is only ever read as the resident.
* **A second cut in each slot: the end of the system message.** While prefilling, a snapshot is also kept at
  the chunk-multiple below `b0`, for `b0` the index after the opener of the second message (the first
  `<|im_start|>` past token 0, its role and the newline: the end of the system message when there is one), when
  that is at least 64 tokens in. (As built: the rule does not read the role, so a conversation with no system
  message gets its second cut after its first user message, which is harmless.)
  It costs one more 163 MB copy and no rows (they are a prefix of the slot's). A request that shares the system
  message and nothing after it hits there on its first appearance: the row where Ollama pays twice.
* Held to what 7.6 holds one slot to, with the interleaved case now expecting a hit, plus: five conversations over
  four slots (the least recent one misses, the others hit), and a shared system message (the second request reuses it).

**B. Prefill twice as fast.** In order of what the profile says:

1. The delta layers' small kernels over the chunk's rows, the state carried through the tokens inside one launch
   (`conv_step` and `delta_step` over `m` rows, as `conv_step3` and `delta_step3` do for 3). 2,309 launches a
   chunk to 288. Worth up to 24% of a chunk **[estimate from the shares above]**: about 113 to 140 tok/s.
   **Measured in 8.7: 6.5%. The estimate was wrong:** it took the shares of kernels timed alone, where each small
   launch carries its overhead; in the pipelined run they overlap the weight products.
2. NVFP4 through the matrix unit for 16 rows (two activation fragments a weight fragment, `frag_rows` of
   `src/matmul.cho` taught what `frag_rows_f16` does), and the chunk raised to 16. This is the step that decides
   parity; the estimate of 8.2 says 235 to 258 tok/s if a 16-row pass costs what an 8-row one does.
   **Measured in 8.7: it costs 1.7 times as much, so 153 tok/s.**
3. Thirty-two rows only if step 2's kernel shows the cost growing slower than the rows.

Each is held to the tokens of the build before it on the golden check and `tests/prefix_check.py` (the cut becomes
a multiple of the chunk, whatever the chunk is).

**C. Keep what decode already computed.** When an answer ends, the engine holds the state after its last token,
and with thinking off the next turn's prompt starts with exactly those tokens. Section 7 throws that away and
prefills the answer again, because the state after a prefill in chunks and after a decode token by token are not
bit for bit the same (different kernels, 1e-6 apart). Keeping it makes the first token of the next turn cost only
the new user message, however long the answer was: a 500-token answer is 2 s of prefill today and would be none.
The price is the bar: "the same tokens as a server without the cache" becomes "the same tokens on the check set",
since a near tie could fall the other way. **This is a decision, not an engineering step: it is in the design as
optional, off by default (`--reuse-decode`), measured before anything is said about it.** The generated ids are
compared with the next prompt's like any other prefix, so a client that edits the answer simply misses.

**D. Decode: the round's 11 ms, and depth.** `docs/native.md` N4b and N4c measured it: Ollama's round costs about
44 ms and ours 51, of which the three-row weight kernels are 40 (at their optimum, and at Ollama's rate for the same
bytes) and the rest 11; and Ollama drafts deeper where drafts are accepted. So:

1. A depth controller: draft 1 to 4 tokens from the acceptance measured in this answer so far (N4c's table: +10 to
   +12% on code and predictable text, -5% on prose at a fixed depth of 3, hence a controller and not a constant).
2. Fewer launches in a round (about 8.7 us each, 8 ms of the 11): the fusions `docs/qwen35.md` section 5 lists and
   did not make. Worth 3 to 4 ms of 51 **[read]**; parity on prose needs 7.

Parity on prose is **not promised**: after both, the estimate is 43 to 45 tok/s against 47 to 49.

### 8.4 What to expect, and the bars

| case | today | after A | after A and B | Ollama |
|---|---|---|---|---|
| cold 743 tokens | 6.57 s | 6.57 | 2.9 to 3.2 s **[estimate]** | 2.80 s |
| warm turn, 112 new tokens | 1.25 s | 1.25 | 0.5 to 0.6 s **[estimate]** | 0.45 to 0.57 s |
| two chats taking turns | 2.7 to 3.2 s | 1.3 s **[estimate]** | 0.5 to 0.6 s | 0.33 to 0.45 s |
| shared system message, second question | 4.68 s | 0.3 s **[estimate]** | 0.2 s | 2.13 s |
| the same, third question | 4.85 s | 0.3 s | 0.2 s | 0.16 s |

The bars, on `tests/ollama_compare.py`: **A** is done when two chats taking turns and the shared system message hit
on their second request, with the tokens of a server without the cache. **B** is done at 235 tok/s on the 743-token
cold prompt, and is called parity only at 270. **Ahead of Ollama** is claimed for a row only when that row's
measured number is lower, which the estimates give for one row (the second question after a shared system
message) and, with C, for turns that follow long answers **[open: Ollama not measured there]**.

### 8.5 Not in this design

* **A context above 1,024.** The largest difference from Ollama for real use, and a kernel matter
  (`docs/kernels.md` section 4 and its later route to 16,640 positions). It changes the slots' sizes (the rows
  grow with the context, the snapshot does not) and needs its own design.
* Requests in parallel (one at a time, as section 4), and slots on disk.
* Snapshots at every turn boundary of a conversation (an edit of an old message restarts from the system cut or
  from nothing).

### 8.6 Steps

1. **A: the slots**, `--cache-slots`, the second cut, the checks. No kernel changes. **Done (8.7).**
2. **B1: the delta layers a chunk at a time. Done (8.7).**
3. **B2: the 16-row NVFP4 matrix-unit kernel**, measured alone first (`scripts/metal-kernels.sh`), then the chunk.
   **Built (8.7); its bar is not met.**
4. **D1: the depth controller.**
5. Then, each on a decision: C (`--reuse-decode`), D2 (fusions), and the context.

### 8.7 Steps 1 to 3, built and measured

**What was built.**

* **Slots** (`device/serve-main.cho`: `cache_new`, `victim`, `park`, `prefill_cached`; `device/qwen35.cho`: `kv_new`,
  `kv_copy`). As 8.3 A, with one simplification: a slot's keys and values are copied as whole buffers (34 of 4.19 MB),
  not as the rows below the cut, because a cache buffer is laid out by head and the rows of a prefix are not one range.
  A conversation that shares only the earlier cut of a slot is written to another slot, so the one it branched from
  can still continue.
* **The delta layers a chunk at a time** (`conv_step8.chx`, `delta_step8.chx` and the 16-row pair, with
  `matvec_rows_f16_m`, `gates_rows` and `qk_norm_rows` at the chunk's rows): the window and the state are carried
  through the chunk's tokens inside one launch and written once.
* **Nine to 32 rows of activations on the NVFP4 matrix-unit path** (`frag_rows_wide` in `src/matmul.cho`): the lanes,
  loads and order of `frag_rows`, with `ceil(m / 8)` fragments of activations sharing each weight fragment. Kernels of
  up to eight rows are emitted as before (no golden moved).
* **The chunk is 16 tokens, then one of 8** if that much is left before single tokens (`qwen35.mb()`, `feed_half`,
  `qwen35spec.feed`). The cache's cuts are multiples of 16.

**Held to.** `tests/prefix_check.py`: **109 checks ok** on the final build: every answer equal to the server without
the cache (greedy, and sampled with a seed), the interleaved chat now reusing its slot, five conversations over four
slots (the least recently used misses, four hit), a shared system message (the second and third questions reuse it,
the first conversation continues past it from its own slot). The new kernels against float64 references
(`tests/kernels.py`, 305 ok) and, on the Metal GPU, against the interpreter (`scripts/metal-kernels.sh`: 13 and 16
rows, 1.05e-6 and 9.2e-7). `cancho-gpu-run35` gives the tokens of the build before on three prompts of about 685
tokens (24 to 48 tokens each). `tests/serve_check.py`, `tests/chat.sh` and 458 of 458 goldens agree; `scripts/spec-check.sh` (seven prompts, speculative
against plain) and `scripts/gate.sh` pass.

**The cost of a miss** (`tests/prefix_check.py miss`, 8 requests of about 330 tokens): +20, +16 and +26 ms with the
cache in three runs, and -10 ms in the control where neither server has it (`miss control`): about 30 ms, the copy of
the resident slot's keys and values, under the bar of 100. (One full run printed +239 ms while the goldens were being
recorded on the same machine; it did not repeat.)

**Prefill** (`cancho-gpu-run35`, 689 tokens, alternating builds, four runs each within 0.5%):

| build | prefill | tok/s |
|---|---|---|
| before | 5.81 s | 119 |
| delta layers a chunk at a time | 5.43 s | 127 |
| and 16 rows (with the 8-row half chunk) | 4.51 s | 153 |

* With the weight products made no-ops (`CANCHOGPU_NOOP=matvec_fp4`) the same prefill takes 1.02 s at 8 rows and
  0.92 s at 16: **the weight products are 81% of the prefill, not the 69% of 8.2's table**, and everything else is
  1.3 to 1.5 ms a token.
* A 16-row pass of the weight products costs 88 ms against 51 for 8 rows: 1.7 times, not the 1.0 the estimate
  needed. That is 9.5 x 10^12 multiply-adds a second against 8.2 at 8 rows and Ollama's 14.6 for its whole prefill.
  The cost of this lowering is in the matrix-unit steps and the activations, not in decoding the weights, so more
  rows a pass buy little.
* The emitted kernel is sensitive to how many blocks a lane holds at once: with 4 or 8 (the 8-row kernel uses 8) a
  16-row prefill took 9 to 15 s, with 1 or 2, 4.7 s. Thirty-two rows took 11 s at 2. `frag_rows_wide` uses 2.
  **[open: why; the Metal compiler's register allocation is the suspect and was not examined.]**
  **Since examined (`docs/kernels.md` section 9): it is the array of activations the kernel itself loads ahead of
  use; loaded where they are used, every unroll costs the same and 32 rows run.**

**Against Ollama** (`tests/ollama_compare.py both`, one run, after the changes; first token):

| case | before (8.1) | now | Ollama |
|---|---|---|---|
| cold prompt, 131 / 403 / 743 tokens | 1.20 / 3.51 / 6.57 s | 1.00 / 2.90 / 5.43 s | 0.68 / 1.61 / 2.93 s |
| conversation, turns 2 to 4 | 1.24 to 1.26 s | 1.07 to 1.08 s | 0.43 to 0.58 s |
| two chats taking turns, turns 2 and 3 | 2.73 to 3.17 s | 0.74, 0.61 s | 0.33, 0.43 s |
| shared system message, second question | 4.68 s | **0.16 s** | 2.06 s |
| the same, third question | 4.85 s | 0.35 s | 0.12 s |
| decode, 256 tokens of a story | 40.2 tok/s | 42.0 tok/s | 47.3, 49.2 tok/s |

**Against the bars of 8.4.** A is met: both cases hit on their second request with the tokens of a server without
the cache. **B is not met: 153 tok/s against a bar of 235.** One row is ahead of Ollama (the second question after a
shared system message, 0.16 s against 2.06 s); every other row is behind, by 1.4 to 2.9 times.

**What this says about the rest.**

* The remaining prefill gap is the weight products' compute rate. The routes not tried: half-precision fragments
  (`simdgroup_half8x8`; the weights are 4-bit and the activations could be f16), and a product that does not go
  through per-lane `thread_elements` at all. Either is a kernel study with its own measurements before a design.
  **The study is `docs/kernels.md` section 9.** It first said the matrix unit's ceiling here is about 14 x 10^12
  and that Ollama runs at it; **that was wrong and is corrected there (9.6)**: MLX's kernels, timed alone, do 13.0
  at 16 rows and 14.8 on a long prompt. Half-precision fragments and Metal's tensor ops do not help here. This
  document's "14.6" for Ollama took 26 x 10^9 weights a token; the profile's sum is 24.3.
  **Built since (`docs/kernels.md` 9.7): the 16-row kernel after MLX's, 234 us a launch against 304, and the
  prefill above is 3.95 s, 174 tok/s, with the same tokens.** Bar B (235) is still not met.
* The tail of a prompt still costs 37 ms a token for up to 7 tokens after the half chunk (the third question above:
  0.35 s for 15 tokens, where 16 took 0.16 s). A 4-row and a 2-row set would bound it at about 150 ms.
* C (`--reuse-decode`) is now the largest lever on a warm turn that is not a kernel study: the turns above re-feed
  the answer the server itself generated.

## 9. Toward Ollama's prefill: where the time is now, and the steps (2026-11-21; steps 1 to 3 built)

> **Built: steps 1 to 3. Not built: 4 and 5.** The user's decision, the same day: 16-bit activations may be used if that is what
> reaches Ollama's rate. Marks as in section 7.

### 9.1 Where a prefill goes now [measured]

`cancho-gpu-run35`, 689 tokens, after `docs/kernels.md` 9.7: 3.90 s, 5.66 ms a token. Each family of kernels made a
no-op in turn (`CANCHOGPU_NOOP`, one run each; the outputs are wrong, the time left is what the family does not
cost):

| family | prefill without it | so it costs | ms a token |
|---|---|---|---|
| nothing removed | 3.90 s | | 5.66 |
| the NVFP4 products (`matvec_fp4`) | 0.91 s | 3.00 s | 4.35 |
| the delta layers' step (`delta_step16`) | 3.51 | 0.39 | 0.57 |
| the delta layers' two f16 projections (`matvec_rows_f16_m`) | 3.70 | 0.20 | 0.30 |
| attention (scores and weighted sum) | 3.74 | 0.16 | 0.23 |
| `rmsnorm_rows` | 3.87 | 0.04 | 0.05 |
| rope, `kv_row`, `conv_step16`, `qk_norm_rows`, `gates_rows`, `gated_norm` | 3.89 to 3.91 | nothing measurable | |
| not accounted for (host work, launch latency) | | about 0.12 | 0.17 |

`CANCHOGPU_PROFILE=2` times every launch on its own and overstates the small kernels by about two; it is not the
source of this table. **Section 8.7's guess that the 1.3 ms outside the products was per-token launches was wrong:**
the per-token kernels of an attention layer (rope, `kv_row`) cost nothing measurable, and two thirds of it is three
kernels.

Ollama: 272 to 281 tok/s, 3.6 to 3.7 ms a token for everything.

### 9.2 The steps, each with what it was measured or estimated to give

1. **The two f16 projections on the matrix unit. Built.** `matvec_rows_f16_m` took the generic lowering: 32 lanes an
   output whatever the schedule, a chain of 2,560 multiply-adds a lane, 60 us a launch at 16 rows alone. With
   `frag 1` in its Apple schedule it takes `frag_rows_f16_ks`: 12 us at 16 rows, 7 at 8, and 9 and 11 at 2 and 3
   where it was 8. **Prefill 3.72 s against 3.92, 185 tok/s, the same tokens on the three prompts of 8.7**;
   `spec-check.sh` and `serve_check.py` pass; the order of the sum changes.
2. **The delta step without threadgroup memory. Built.** The emitted `delta_step16` was the generic lowering at
   256 threads: a row's 128 columns spread over 128 threads, so each of its 96 row sums and 48 broadcasts went
   through threadgroup memory, 384 barriers, 162 us a launch alone (189 in the model). Three changes:
   * **The lowering** (`src/gen.cho`, `src/lower.cho`): in a threadgroup that is one simdgroup (`threads 32`, the
     Metal dialect) a register tile lies in its 32 lanes only, so `rowsum` and `rowmax` of one are a lane's own
     elements and `simd_sum` or `simd_max`, and a register tile of at most 32 elements is read at another index by
     `simd_shuffle`. No threadgroup memory, no barrier. Nothing else in the repository has a `threads 32` schedule,
     and no other golden moved.
   * **The schedule**: `delta_step16` and `delta_step8` are `threads 32 br 1` on Apple. That alone: 143 us.
   * **The program**: the kernel is bound by what it reads (a hand-written one took 43 us loading a token's `k` and
     `q` once and 63 loading them for each of the three blocks of the state). The two files are regenerated in
     token order, a token's `k` and `q` rows put in registers once (`zeros + load`) and borrowed by the three
     blocks. **77 us a launch alone, no barrier** [measured]. The hand-written 43 is not reached, and why is
     **[open]**.

   **Prefill 3.29 s against 3.49 (alternating builds, three runs each within 0.4%), 209 tok/s, the same tokens on
   the three prompts of 8.7.** The order of the row sums changes.

   **Found on the way: the interpreter trapped.** With one row a tile the reference run of `delta_step16` at
   `nk=16` has 2,048 instances of a 700-statement body, and the interpreter gave back an instance's floats but not
   its records: the pool ran out, which is a trap in `mem.grab`, not an error (the old program does the same under
   this schedule). `interp.execute` now gives back both. `tests/kernels.py` and every `--run` golden are
   unchanged by it.
3. **16-bit activations in the batch products. Built**, narrower than first written here: only what a weight
   product reads is f16. The chunk's and the half chunk's products read `x` as f16
   (`matvec_fp4_batch_h`, `_res_batch_h`, `_swiglu_batch_h`, `matvec_rows_f16_mh`), and the kernels that write what
   they read round to f16 with `stage f16` (`rmsnorm_rows_h`, `attn_gate_q_h`, `gated_norm_h`, and the SwiGLU
   product's own output). The residual stream, the caches, the delta states and the logits stay f32, and so do the
   single-token path and a speculative check's few-row kernels: **a token fed in a chunk and the same token fed
   alone no longer go through the same arithmetic** (they already differed in the order of the sums). An
   activation a product reads keeps 11 bits. The 16-row kernel alone: 217 us against 233 [measured], MLX's 219.
   **Prefill 3.49 s against 3.74 (alternating builds, three runs each within 0.5%), 197 tok/s, and the same
   tokens on the three prompts of 8.7, 24 each.** That the tokens are the same there says nothing about a near
   tie elsewhere.
4. **A long pass and a staged kernel.** MLX's `fp_qmm_t` does 12.1 us a row from 64 rows up against 13.6 for the
   16-row kernel with f16 activations (`docs/kernels.md` 9.6). [estimate] 0.3 s more, and it is the only step that
   needs the prefill restructured: kernels for a number of rows not known when they are compiled, the delta and
   attention layers over a whole prompt, the cache's cut. No design yet.
5. **Attention for a chunk's rows in one launch**, 0.16 s today.

[estimate] Step 5 on top of steps 1 to 3 (3.29 s): about 3.15 s, 219 tok/s. With step 4, about 2.85 s, 242. **The sum of
what is listed does not reach Ollama's 2.5 s**; the rest is in the 0.17 ms a token not accounted for and in the
products' rate between our kernel and theirs, and neither has a step here.
