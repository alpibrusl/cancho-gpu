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
     `simd_shuffle`. No threadgroup memory, no barrier. No other golden moved (`bn_fin` is `threads 32` too, which this said nothing else was, and has no such
     reduction).
   * **The schedule**: `delta_step16` and `delta_step8` are `threads 32 br 1` on Apple. That alone: 143 us.
   * **The program**: the kernel is bound by what it reads (a hand-written one took 43 us loading a token's `k` and
     `q` once and 63 loading them for each of the three blocks of the state). The two files are regenerated in
     token order, a token's `k` and `q` rows put in registers once (`zeros + load`) and borrowed by the three
     blocks. **77 us a launch alone, no barrier** [measured]. The hand-written 43 was not reached; why, and 50 us,
     are section 9.5.

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
4. **A pass of several chunks, each weight product one launch. Built** (section 9.3). What was written here first,
   a staged kernel after MLX's `fp_qmm_t` worth 0.3 s and a prefill restructured around a row count not known at
   compile time, **was wrong about the kernel**: the 16-row kernel launched once over many row tiles is as fast as
   a staged one, and the row count is the launch's grid, not the kernel's.
5. **Attention for a chunk's rows in one launch. Built.** A chunk ran `attention_scores` and `attention_pv` once a
   token: 32 launches an attention layer, of 32 threadgroups each. `attention_scores_rows` and `attention_pv_rows`
   are the same two programs with the row of the chunk as one more grid axis: one cache, a mask a row (the engine's
   `tmax`-wide masks as they are), the scores of the chunk in one `[rows * 24, t]` scratch. Two launches a layer.
   The weighted sum is split over 64 columns of `v` where the single-token kernel takes 32 (3.19 s against 3.21;
   16 columns: 3.21) [measured, three runs each]. The speculative check's two rows and the single-token path keep
   the per-token kernels. **Prefill 3.20 s against 3.30 (alternating builds, three runs a prompt, within 0.4%),
   215 tok/s, the same tokens on the three prompts of 8.7.** The programs of a row are what they were; whether
   the sums of the wider weighted sum run in the same order was not checked.

   What attention still costs, each kernel a no-op in turn: the scores 0.03 s, the weighted sum 0.07 s
   [measured, two runs each]. Every threadgroup of the weighted sum redoes its head's softmax, 4 times a head now.

The estimate written here before step 5 was built said 0.14 s for it; it gave 0.10. The one for step 4 said 0.3 s
and 2.9 s after it; it gave 0.33 and 2.87.

### 9.3 A pass: several chunks through a layer together (2026-11-22)

**What was measured first, on the kernel alone** (`n=17408 k=5120`, f16 activations, the throwaway harness of
`docs/kernels.md` 9.6; us a row):

| kernel | one launch of | us a row |
|---|---|---|
| ours (`frag_rows_split`), 16 rows | 16 rows | 13.5 to 13.7 |
| ours, 32 rows | 32 rows | 13.4 |
| ours, 16 rows a threadgroup, 8 row tiles in the launch's grid | 128 rows | **12.5 to 12.6** |
| a hand-written staged kernel after `fp_qmm_t`, 64 x 64 tile, 128 threads | 64 / 128 / 256 rows | 13.8 / 13.1 / 12.8 |
| the same, 32 x 64 tile, 64 threads | 64 / 256 rows | 14.1 / 12.6 |
| MLX (`docs/kernels.md` 9.6) | 64 / 256 / 651 rows | 12.8 / 12.2 / 13.0 |

So what a long product gains is not the staged tiles: it is the launch. The same register kernel does 12.5 us a
row once a launch has eight times the threadgroups, and the staged kernel written by hand does no better. **No new
lowering**: `matvec_fp4_pass_h`, `matvec_fp4_res_pass_h` and `matvec_fp4_swiglu_pass_h` are the three chunk
programs with the rows as a second grid axis (`grid i over n / bn, j over m / bm`), which the compiler already
takes through `frag_rows_split`.

**The pass.** `feed_pass` feeds `tiles` chunks of 16 tokens (`pt()` = 8 at most, 128 rows) through each layer
together. In a layer:

* each of the seven weight products is one launch over the pass's rows. The kernel is compiled for 128 rows and
  16 a threadgroup; a shorter pass launches it over fewer row tiles (the grid's second extent, set in the kernel's
  record before the layer loop). Nothing is compiled for a row count;
* every other kernel is the chunk's, launched once a chunk in order at an offset into the pass's buffers. The delta
  state and the convolution window go from chunk to chunk as they did; an attention layer writes the pass's keys
  and values and then attends chunk by chunk under each row's own mask.

The chunk buffers hold a pass: 8 times what they were, about 67 MB. `feed` takes as many whole chunks as are left,
up to `pt()`; the half chunk and the single tokens after them are unchanged. A row goes through the arithmetic it
went through in a chunk.

**Measured.** 689 tokens, alternating builds, three runs a prompt: **2.87 s against 3.20 (240 tok/s), the same 24
tokens on the three prompts of 8.7.** Passes of 8, 16 and 32 chunks: 2.86 to 2.87 s each; 64 (the whole prompt):
2.94. So 8.

What is left, each family a no-op in turn (one run each):

| family | costs, before the pass | after |
|---|---|---|
| the NVFP4 products | 2.64 s | 2.40 s |
| the delta step | 0.19 | 0.16 |
| attention | 0.09 | 0.08 |
| `rmsnorm_rows` | 0.05 | 0.04 |
| everything else, host and launch latency | 0.25 | 0.19 |

**Not done, not tested.** A launch over fewer row tiles than the kernel was compiled for is covered only by the
model's tokens, not by a kernel test: `metal-kernels.sh` launches a kernel over its whole grid. The other kernels
are still a launch a chunk. The speculative check and the single-token path are untouched.

**Against Ollama's 2.5 s:** 0.37 s. The products are 3.48 ms a token against the 3.3 to 3.5 its kernel would give
at MLX's measured rate, so most of what is left is outside them: the 0.47 s of everything that is not a product,
which in one pass over a prompt Ollama spends a few tenths of a millisecond a token on (`docs/kernels.md` 9.6).
No step for that is designed here.

### 9.4 What the 0.47 s outside the products is, and one step that gave nothing (2026-11-22)

**Launch and host work is 0.10 s, not most of it.** With every kernel a no-op (`CANCHOGPU_NOOP=""`: each grid is
still launched) a 689-token prefill takes 0.10 s [measured, two runs]. So of the 0.47 s outside the products
0.36 s is work the kernels do: the delta step 0.16, attention 0.08, `rmsnorm_rows` 0.05, and rope, `kv_row`, the
f16 projections, the convolution, the gates and the small norms about 0.06 together (0.01 or less each).
**Section 9.3's table called 0.19 s "host and launch latency"; 0.10 of it is, and the rest is those small
kernels' work.**

**Built and taken out: the row-grid kernels once a pass.** The norm, `qnorm`, the keys' norm, the attention gate,
the gated norm, the gates, the q/k norm and attention's two kernels all have the rows as their grid's first
extent, so each was compiled for 128 rows and launched once a pass over the rows it had, as the products are.
**2.86 s against 2.87, the same tokens on the three prompts** [measured, two runs a prompt, alternating builds]:
nothing, for thirteen more kernels and 12.6 MB of scores. Not kept. These kernels cost what they compute, and a
launch a chunk was not what they cost.

**What would move it:** the delta step (section 9.5, built); attention's weighted sum, which redoes a head's
softmax in each of its four threadgroups (0.05 s of the 0.08); the norm, 8 us a launch for 16 rows of 5,120. The
last two have no design.

### 9.5 Why the emitted delta step took 77 us and the hand-written one 43 (2026-11-22)

Section 9.2 left this open. The two kernels do the same loads and the same arithmetic; the emitted text was edited
by hand until the time moved (`delta_step16`, `nk=16`, alone, best of the harness's runs):

| emitted kernel | us |
|---|---|
| as emitted | 78 to 80 |
| row sums written to every lane, not lane 0 only | 78 |
| every local array initialised | 84 |
| `simd_broadcast_first` for `simd_shuffle(x, 0)` | 78 |
| **no `if (e < 128u)` around an element's statement** | **61 to 62** |
| that, and `simd_broadcast_first` | **48** |
| those, and the lane-0 scalars under `if (tid == 0u)` | 51 |
| those, and the four-wide loops unrolled | 48 |
| written by hand | 44 |

Two things, and the second only shows once the first is gone:

* **The bound of `owned`.** A lane owns elements `k * 32 + tid`, and the lowering guards each with `if (e < n)`.
  With `n` a multiple of the 32 lanes the guard always holds, and the Metal compiler does not see it: 17 us of 78.
  In a one-simdgroup threadgroup `gen.owned` now writes a plain block when `n % threads == 0`.
* **One value read by every lane.** A one-element register tile was read as `simd_shuffle(x, (ushort)(e / 128u))`,
  an index that is always 0. `gen.read` writes `simd_broadcast_first(x)` for a tile of one element: 13 us more.

**The emitted `delta_step16`: 50 us alone against 80** [measured]. Goldens that moved: the four of `delta_step16`
and `delta_step8`, and `bn_fin` (`threads 32`: its guards only). The same guard is in every kernel the generic
lowering emits at other thread counts, where it is not always true as written; whether it costs there was not
measured.

**Prefill 2.82 s against 2.88 (alternating builds, three runs a prompt, within 0.2%), 244 tok/s, the same tokens
on the three prompts of 8.7.** Ollama: about 2.5 s.
## 10. Deeper drafts (2026-11-22)

### 10.1 What was measured, both engines, same machine

689-token prompt of 8.7 and two variants, 128 tokens generated, greedy, engines in turn, cold:

| | prefill tok/s | generation tok/s |
|---|---|---|
| ours, plain | 232 to 245 | 27.4 to 27.6 |
| ours, `--spec` (2 drafts a round) | 219 to 232 | 38.7 to 41.3 |
| Ollama 0.40.16 | 273 to 274 | 58.3 to 62.9 |
| Ollama, a request with `logprobs` (it then does not draft **[read: `mlxrunner/speculate.go`]**) | | 27.5 to 28.6 |
| Ollama, the same prompt a third and fourth time | (cached) | 79 to 84 |

**This table does not compare like with like, found in 10.4:** on these prompts Ollama's continuation is the
prompt's sentence again and ours is new prose, so its drafts are easy and ours are not. The plain rows stand.

**Plain decoding is the same speed in both.** Everything Ollama gains in generation is its speculative decoder: a
controller that picks the number of drafts each round from the measured cost of a check and the measured acceptance
at each position, kept across requests **[read: `speculate_depth.go`]**. Its log for these runs: 42 rounds for 128
tokens at 2.1 drafts a round on the first, 25 rounds at 6.1 drafts (up to 7) and 4.1 accepted a round on the fourth.
Ours drafts 2, always: at most 3 tokens a round.

**What a round of ours costs** (`CANCHOGPU_PROFILE=1`, 41 rounds): the device 45.0 ms a round, every round; the
round 51 ms, so about 6 ms a round is the host. A plain token is 36.4 ms. Two drafts and two more rows in the check
are 9 ms together: a draft is about 3 ms, and a check of 3 rows is hardly more than a token, because a token is
bound by reading the weights and the rows after the first only add arithmetic. A prefill row in a chunk is 4.1 ms.

### 10.2 Design: a check of 8 rows

**The check is the half chunk.** 8 rows through `feed_rows` with the half chunk's kernels (set 3): [estimate] 37 to
40 ms, what a token costs. So a round drafts 7 and checks [last, d1..d7]: [estimate] 7 x 3 + 40 + 6 = 67 ms a round
whatever is accepted, against 51 for the round of 2. At the acceptance of 10.1's own run (57 of 82 drafts, 0.70 a
position) [estimate] 1 + 2.1 = 3.1 tokens a round, 46 tok/s against 47: nothing; at 0.9 a position 5.7 tokens a
round, 85 tok/s against 53. **So the depth has to follow the acceptance**, as Ollama's does; 10.3.

The rows go through the chunk's arithmetic (f16 activations into the weight products, section 9 step 3), not the
single token's or the 3-row check's: a near tie can fall differently from plain decoding.

**Keeping a prefix of the rows.** After the check, `kept` of the 8 rows are kept (1 to 8). What a row changed:

* an attention layer's cache rows, and the head's: rows past `kept` are behind the mask and are written again;
* a delta layer's state. The check reads the committed state and writes the state after 8 rows to the spare
  buffer; the committed one is not touched. All 8 kept: the two are swapped. Fewer: the step is run again for that
  layer with the rows past `kept` made no-ops, from what the check left of its inputs. A row is a no-op when its
  decay is 1 and its write strength 0, which `gates_rows` gives exactly for inputs of -1e30 (`exp(-(A softplus))`
  with softplus 0, and a sigmoid of 0): so the check keeps each delta layer's two gate inputs (48 numbers a row), its
  normalised q and k and its convolved row in buffers of their own a layer (50 MB for the 64 layers), and a rerun
  is two launches a layer (`gates_rows`, `delta_step8`), 96 in all, [estimate] 3 ms;
* a delta layer's convolution window, the last three inputs. The check writes the window after 8 rows aside. Fewer
  kept: the window is rows `kept .. kept + 2` of (old window, the check's inputs), copied by a three-row kernel
  (`rows3`) from the old window and the layer's kept inputs. Either way the result is swapped into the list the
  next position's parity reads.

Not in this design: sampling. With a sampler that is not greedy the round of 2 stays (the logits of 8 rows are
8 MB to bring back a round).

### 10.3 Design: which round

Two rounds exist, of 2 drafts and of 7. Each keeps a running mean of its milliseconds a token over the rounds it
ran (cost of the round over tokens it chose); the next round is the one with the lower mean, and every 8th round is
the other one, so that a mean cannot go stale. The means live in the engine, across requests, as Ollama's do.

### 10.4 Built and measured

`qwen35spec.round_n` is the round for any number of drafts, `commit8` keeps a prefix of a check of 8, `spec_round`
chooses. Nothing in the compiler; one new kernel (`rows3`), the 8-row output head (`matvec_fp4_batch` at 8 rows),
60 MB of buffers made on the first round.

**A round of 7 drafts: 62 to 63 ms** (19 rounds, 1,176 ms), the device's wait 42 ms of it; a round of 2: 51. The
estimate was 67. So `cost7` = 63 to `cost2` = 51, constants in `spec_round` **measured on this M4 Max and nowhere
else**: there is no clock in the round, so the cost is not measured as it runs, which Ollama's is.

**The chooser** as 10.3, with one change: the other round is a probe only while the round of 2 is ahead, every 8th
round, and a probe that changes nothing doubles the wait (64 at most). A round of 7 also says what a round of 2
would have chosen (at most 3 of its tokens), so the round of 2's mean never goes stale.

**`scripts/spec-check.sh`, 96 tokens, greedy: the same tokens as plain decoding on its 7 prompts.** Generation
against the round of 2 alone (section 9's build) and against Ollama 0.40.16 (raw prompt, temperature 0, its third
run of each prompt so that its depth is learned), tok/s:

| prompt | plain | round of 2 | now | Ollama | same text as Ollama's |
|---|---|---|---|---|---|
| The capital of France is | 27.8 | 49.8 | 69.3 | 58.0 | all of it |
| def quicksort(arr): | 27.4 | 54.1 | 80.7 | 78.4 | all of it |
| Once upon a time, ... | 27.5 | 45.0 | 45.4 | 49.2 | the first third |
| The three laws of thermodynamics, ... | 27.9 | 47.3 | 62.1 | 60.1 | the first fifth |
| Q: Why is the sky blue? | 27.5 | 51.1 | 59.4 | 55.1 | the first half |
| rare words: ... | 27.5 | 41.1 | 44.3 | 54.3 | the first third |

One run each; where the texts part the two engines are drafting different text. **Where few drafts are accepted
Ollama is ahead (49 and 54 against 45 and 44)**: its controller goes down to one draft and none, and ours has two
rounds. On the 689-token prompt of 10.1 ours writes prose its head drafts badly (71 drafts accepted in 61 rounds)
and takes 3.39 s for 128 tokens, 37.8 tok/s, where the round of 2 alone took 3.10: **slower there**, by the rounds
of 7 made before the means settle and the probes after.

**Found on the way.** A round's reruns were encoded and never sent when the round was the last: the process
aborted at exit on three of the seven prompts. `commit8` flushes after a partial keep. And the server hung on its
second request: the round's kernels were made on the first round, in the first request's memory, which the server
gives back. `serve-main` makes them at start, where it makes the batch kernels (`ensure_deep`).

**The server** (`tests/prefix_check.py`, 128 tokens a case): 67 to 73 tok/s where it was 55 to 56, the same tokens
with the cache and without.

**Not done.**
* Sampling: a request that is not greedy has the round of 2 only.
* The rows of a check of 8 go through the half chunk's arithmetic (f16 activations into the products), the round
  of 2's and plain decoding through f32. The tokens are the same on the 7 prompts; that says nothing about a near
  tie elsewhere.
* No depth between 2 and 7, none below 2 (**since built: 10.5**), and no cost measured as it runs (still so).
* `commit8`'s rerun rests on `gates_rows` giving exactly 1 and 0 for inputs of -1e30. It is checked only through
  the model's tokens (every partial keep of the 7 prompts goes through it), not by a kernel test.
* About 6 ms of a round of 2 and more of a round of 7 is the host's (encoding some 1,500 launches a round of 7).

### 10.5 The depth follows the acceptance at each position (2026-11-23)

**Why the two prompts of 10.4 where few drafts are accepted were behind** [Ollama's log, its third run of each]:
on "rare words" it made 39 rounds of 2.4 drafts on average, 57 accepted, 45 ms a round; ours made 35 rounds, 62
accepted, 62 ms a round. It chooses no more tokens a round than ours. Its rounds are cheaper, and of a depth
between ours.

**Every number of this section was measured on a day the machine ran 9% slower** (plain decoding 3.70 s for 95
tokens where 10.4 has 3.40): they compare with one another, not with 10.4's. **Corrected (11.6): the machine was
not 9% slower across the board. Two of this project's test servers had been stuck exiting since the night of 10.4,
and a round was 15% slower where a plain token was 6%; restarted, the round of this section is 46.6 + 2.1 a
draft, not 53 + 2.45, and it is 10 ms above a token and its draft, not 14.**

**What a round costs by its drafts**, the check of 8 rows whatever is drafted (a draft count forced, two prompts):
55.5 ms at 1 draft, 60.7 at 3, 70.0 at 7: **53 + 2.45 a draft.** 10.4's 51 against 63 for 2 and 7 drafts had the
check of 3 rows for the first; the check of 8 at 2 drafts is 58. Each kernel family a no-op in turn, 3 drafts
(61.3 ms a round): the NVFP4 products 49.3 (the check's and the three drafts'), the delta step 4.0 (the check's and
the rerun's), attention 1.3, every other family under 1, and 4.7 ms a round with every kernel a no-op (the host and
the launches). A plain token is 39 ms that day, all but bound by reading the weights; so a round is a token, plus
2.45 ms a draft, plus about 14 ms.

**Tried and not kept as gains** [measured, d = 7, two prompts]: sending the drafts to the device before the
check's rows are prepared (70.0 against 69.5 to 70.4 without), and letting the reruns of the layers overlap, each
layer's gates in a buffer of its own (70.0 against 70.8 to 71.6). Both are in the code; neither is worth a
millisecond.

**Built: `spec_round` picks the depth.** The engine keeps a running acceptance of the draft at each position 1 to
7, given that the ones before it were accepted (an eighth of each new round). A round of `d` drafts is expected to
choose `1 + a1 + a1 a2 + ...` tokens and costs `530 + 24 d`; the depth is the one with the most tokens for the
cost, at most one past the deepest ever drafted (whose acceptance a deeper position borrows), and every 8th round
drafts one more than that, so that the next position is measured. A greedy round's check is always the 8 rows; the
round of 2 with the check of 3 is what a request that samples has. The two rounds and the probe counter of 10.3 are
gone.

**Measured against 10.4's build, alternating, 95 tokens, ms (rounds):**

| prompt | 10.4 | now |
|---|---|---|
| def quicksort(arr): | 1,350 (19) | 1,364 (20) |
| Once upon a time, ... | 2,439 (37) | 2,365 (37) |
| The three laws of thermodynamics, ... | 1,757 (25) | 1,772 (27) |
| Q: Why is the sky blue? | 1,772 (25) | 1,698 (25) |
| rare words: ... | 2,465 (35) | 2,387 (37) |
| the 689-token prompt of 10.1, 128 tokens | 3,893 to 3,902 (61) | 3,458 to 3,465 (54) |

The same tokens as 10.4's build on each, and `spec-check.sh` holds them to plain decoding's. **The long prose
prompt, which 10.4 made slower than the round of 2 alone (3,609 to 3,624 ms that day), is now 4% faster than it.
The two short prompts where few drafts are accepted gain 3%, and stay behind Ollama**: what separates them is the
14 ms a round above a token and its drafts, where Ollama's round has about 4. **Corrected the next run of
measurements (section 11): this paragraph said no one part is most of it, with the check's products about 3 ms
above a token's. They are 9 ms above: the 49.3 ms of products was compared with a token's 39 ms whole, not with
the 32 ms of products in it.**

## 11. What a round costs above a token, kernel by kernel (2026-11-23)

Measured the same day as 10.5 (the machine 9% slower than in 10.4), 10.5's build, greedy. **Every time of 11.1 to
11.5 is from the machine in the state 11.6 describes, which slowed the multi-row kernels more than the one-row
ones: 11.6 has them again on the machine restarted, and the check's products are 5.7 ms above a token's there, not
9.3.**

### 11.1 Measured

**A round by its drafts and its check** (a draft count forced, "rare words" and "Once upon a time", two runs each
within 0.2 ms): a plain token 39.0 ms; 1 draft, check of 8 rows 55.6; 2 drafts 58.0; 3 drafts 60.4; 1 draft with
the check of 3 rows (10.3's few-row kernels) 56.6; 2 drafts with it 59.3. **The check of 3 rows is no cheaper than
the check of 8**, which 10.5 took from 10.4 (51 against 58) without measuring it the same day.

**Each kernel's GPU time, every launch in a command buffer of its own** (`CANCHOGPU_PROFILE=2`; the sum is 38.3 ms
for a token and about 58 for a round of 1 draft, so what overlaps in the real schedule is little):

| | a plain token | the check of 8 rows | the check of 3 rows |
|---|---|---|---|
| the NVFP4 products of the layers | 32.2 | 41.5 | 44.8 |
| the output head | 1.5 | 1.8 | 1.8 |
| the delta step | 0.8 | 4.7 (2.3 the check's, 2.3 the rerun's) | 2.1 |
| the gates, the two small f16 products | 1.3 | 1.8 | 1.9 |
| the norms | 2.0 | 2.3 | 2.0 |
| the convolution (and the window's copy) | 0.16 | 0.52 | 0.24 |
| rotary and cache rows (launches) | 0.14 (64) | 1.2 (519) | 0.5 (206) |
| attention | 0.25 | 0.7 | 0.7 |
| the largest logit of each row | | 0.5 | 0.3 |

A draft is 2.2 ms beside that, 1.5 of it the output head's product.

**So of the 16.6 ms a round of 1 draft costs above a token, 9.3 are the check's products and 2.2 the draft**: the matrix-unit kernels
cost 1.29 times the one-row kernels whatever the rows (2 to 8), and the few-row kernels of 10.3 cost 1.39 times.
Every greedy round of fewer than 7 drafts also runs the delta step twice: the rows past the drafts are never kept,
so the state after 8 rows is never the one committed.

### 11.2 Built: the rotary and the cache rows of a pass in one launch each

`rope_rows.chx` and `kv_rows.chx`: `rope.chx` and `kv_row.chx` with the token as the grid's second extent,
compiled for 128 tokens and launched over the rows of the pass (the kernel record's second extent, as 9.3 does).
Six launches an attention layer where a check of 8 rows had 32 and a pass of 128 rows 512. **Measured, alternating
builds: a round 0.5 ms shorter (2,369 against 2,387 ms for 95 tokens on "rare words", the same tokens on six
prompts), and a prefill of 690 tokens 3,069 ms against 3,104 (three runs each within 3 ms).** The launches
were not where the time is.

### 11.3 A check of few rows: measured on the kernel alone, not built

One product alone (17408 x 5120, 200 launches in a command buffer, best of five, a throwaway harness that is not
in the repository; each hand-written variant held to the generated kernel's checksum):

| kernel | rows | us a launch |
|---|---|---|
| `matvec_fp4_rows` (a token's) | 1 | 112 to 114 |
| `matvec_fp4_batch_h`, the matrix unit | 2, 3, 4, 8 | 143, 143, 144, 141 |
| `matvec_fp4_few`, its schedule; the best of eight schedules | 3 | 157; 148 |
| the same | 2 | 132; 125 |
| hand-written after MLX's `fp_qmv_wide` (16 lanes an output row, each row of x loaded at every weight group) | 1, 2, 3, 5 | 129, 174, 240, 439 |
| hand-written: a lane holds 2 weight rows, loads them 8 bytes at a time and x 4 values at a time | 1, 2, 3, 4 | 109, 111, 143, 182 |
| the same, 4 weight rows a lane | 1, 2, 3, 4 | 103, 110, 132, 361 |
| the same, x in f16 loaded 8 values at a time, the group's scale multiplied into the weights | 2, 3, 4, 5, 8 | 114, 130, 153, 175, 242 |
| hand-written matrix unit, eight shapes | 8 | 144 to 191 |

* **What a load of x costs is what separates these.** With the weight loads the same, the time follows the loads of
  x a weight row: one row of x for one weight row 106, two for two 111, two for one 163, three for one 222.
* **Two rows can cost what one costs (111 against 112), three rows 1.17 times, four 1.37 times**; from three rows
  each further row costs about 20 us, which is the rate of a plain multiply-add loop in registers
  (`docs/kernels.md` 9: 8.8 TFLOPS). From 4 rows the matrix unit is cheaper, and nothing tried made it cheaper than
  141 at 8 rows.
* **So a lowering of this shape would make a check of 2 or 3 rows cheaper, not the check of 8.** Reckoned from the
  table of 11.1, not measured: a round of 2 drafts about 50 ms where it is 58 (products 37.6 for 41.5, the delta
  step of 3 rows with no rerun), which on the two prompts of 10.5 where few drafts are accepted is 22.6 ms a token
  where the depth chosen today gives 23.9. Ollama's 45 ms a round (10.5) is 49 on this day's machine, 20.4 a token.
* A token's own kernel also has 7% in it by the same loads (103 to 109 against 112 to 114): plain decoding, not
  measured in the engine.

**How Ollama's round is that cheap is not explained by this.** MLX gives 2 to 5 rows to `fp_qmv_wide` and 6 to 16
to its matrix-unit kernel **[read: `mlx/backend/metal/quantized.cpp`]**; my port of the first is the slowest
kernel of the table. Whether its build runs that kernel on these products, and what its plain token spends outside
its kernels, were not measured.

### 11.4 Not done

* ~~The lowering of 11.3, and a round of 1 or 2 drafts on it.~~ Built: 11.5.
* One pass over a delta layer's state a round instead of two: the next round's step replaying the kept rows from
  the committed state (the 16-row step costs what the 8-row one does, both bound by the 6 MB of state they read and
  write). About 2.5 ms a round; the committed state is then a round behind and has to be brought up when the
  request ends.
* The largest logit only of the rows the round drafted (0.3 ms).

### 11.5 Built: the few-row product with wide loads, and the check of 3 rows for a round of 1 or 2 drafts

**The compiler had most of 11.3's kernel already**: `batched_rows` of `src/matmul.cho` has a path that loads a
group's codes as two words and the activations four at a time, taken on CUDA only (`dialect.wide_loads`). Two
changes, both to the Metal kernel alone (the CUDA goldens do not change):

* `dialect.wide_rows`: that path on Metal too, for the few-row product only. The one-row product gains nothing by
  it (111 against 112 us).
* With at most 16 accumulators a lane, the group's scale multiplies the group's sums, not every weight.

The `_few` kernels' Apple schedule is `threads 64 bn 8` (4 weight rows a simdgroup). **One product alone, as
11.3: 2 rows 110 us (was 132; 121 with the wide loads and the scale at every weight), 3 rows 125 to 128 (was
157), 4 rows 163 to 180, against the matrix unit's 141 to 144.**

**In the engine, a draft count forced, two prompts, ms a round:** 2 drafts on the check of 3 rows 52.0 (59.3
before, 58.1 on the check of 8); 1 draft on it 49.4 (56.6 before, 55.8 on the check of 8). 11.3 reckoned 50 for
the first.

**`spec_round`** gives a greedy round of 1 or 2 drafts the check of 3 rows and a deeper one the check of 8, and
weighs the depths by `470 + 25 d` up to 2 drafts and `530 + 24 d` from 3 (tenths of a millisecond). Against 10.5's
build, 95 tokens, ms (rounds), one run each, the same tokens on each:

| prompt | 10.5 | now |
|---|---|---|
| The capital of France is | 1,553 (23) | 1,591 (25) |
| def quicksort(arr): | 1,362 (20) | 1,343 (20) |
| Once upon a time, ... | 2,364 (37) | 2,285 (38) |
| The three laws of thermodynamics, ... | 1,765 (27) | 1,741 (27) |
| Q: Why is the sky blue? | 1,686 (25) | 1,664 (25) |
| rare words: ... | 2,375 (37) | 2,253 (40) |

**2% over the six, 5% on "rare words", 2% slower on the first.** Two drafts forced on every round does "Once upon a
time" in 2,134 ms, so the chooser leaves 7% there: it still goes deep on rounds that do not pay. Four variants of
its exploration (a probe every 16th round, a deeper position borrowing four fifths of the acceptance, both, and
both with a lower first guess) were each slower over the six (11,029 to 11,412 ms against 10,877), on one run each
of 95 tokens, which is too little to tune it by.

**Not done.** A round of 1 draft still runs kernels compiled for 3 rows (2 rows alone are 110 us against 125).
The chooser, as above. A request that samples has the same kernels and was not timed.

### 11.6 The slow day's cause, and the figures again on the machine restarted (2026-11-23, evening)

**What was wrong.** Two `cancho-gpu-serve` processes had been in the exiting state since 2026-11-22 23:28 and
23:37 (a build of 10.4's work that hung waiting for the GPU, killed by its test), unkillable, until the machine was
restarted. Between those times and the restart both engines were slower, and unevenly:

| | 22 Nov (before) | 23 Nov, the processes stuck | 23 Nov, restarted |
|---|---|---|---|
| ours, 95 plain tokens, "rare words" | 3.40 s | 3.70 to 3.80 s | 3.47 to 3.50 s |
| Ollama, "rare words", tok/s **[its log and its API]** | 54 to 56 | 34 | 52 to 56 |
| Ollama, ms a round of 2 drafts **[its log]** | 43 to 46 | 70 to 79 | not read |

Ollama's plain token barely moved (38 ms against 36); its drafts went from about 4 ms each to about 18, and its
depth controller stopped drafting deeper than 2 or 3. **That the stuck processes were the cause is inferred from
the times and from both engines recovering at the restart; what they held was not found.** Ollama updated itself
from 0.40.16 to 0.40.17 at the same restart, so its restarted figures are the new version's.

**What stands of 10.5 and 11.1 to 11.5:** every comparison of two of our builds made alternating (both ran in the
same state), the token-for-token checks, and the order of the kernels alone in 11.3 and 11.5 (not measured again:
the harness was a throwaway and the restart removed it). **What does not:** the times themselves, the sizes of the
gaps, and "Ollama's round is 45 ms where ours is 62".

**Again, restarted** (the same builds rebuilt from their commits; two runs each within 0.4 ms a round):

| | ms |
|---|---|
| a plain token | 36.6 |
| a round of 1, 2, 3, 7 drafts, the check of 8 rows (10.5's build) | 48.5, 50.5, 52.7, 61.2 |
| a round of 1, 2 drafts, the check of 3 rows, 10.5's kernels | 49.0, 51.5 |
| the same, 11.5's kernels | 44.5, 47.0 |

So a round on the check of 8 is 46.6 + 2.1 a draft, and 11.5's kernels take 4.5 ms off a round of 1 or 2 drafts
(11.5 measured 7 in the slow state).

**Each kernel's GPU time, as 11.1, restarted:**

| | a plain token | the check of 8 rows | the check of 3 rows, 10.5's kernels |
|---|---|---|---|
| the NVFP4 products of the layers | 31.2 | 36.9 | 39.8 |
| the delta step | 0.8 | 4.4 | 1.7 |
| the gates, the two small f16 products | 1.3 | 1.8 | 1.9 |
| the norms | 2.0 | 2.4 | 2.2 |
| rotary and cache rows | 0.16 | 1.4 | 0.5 |
| attention | 0.3 | 0.7 | 0.8 |

**The matrix-unit products are 1.18 times a token's, not 1.29, and the few-row kernels of 10.3 were 1.27 times,
not 1.39.** The products are still the largest part of what a round costs above a token (5.7 of 11.9 ms at 1
draft), and the delta step still the second (3.6).

**The six prompts, 95 tokens, ms (rounds), two runs each within 5 ms, the same tokens on every build:**

| prompt | 10.5 | 11.5 |
|---|---|---|
| The capital of France is | 1,367 (23) | 1,404 (25) |
| def quicksort(arr): | 1,198 (20) | 1,186 (20) |
| Once upon a time, ... | 2,066 (37) | 2,016 (38) |
| The three laws of thermodynamics, ... | 1,550 (27) | 1,532 (27) |
| Q: Why is the sky blue? | 1,480 (25) | 1,467 (25) |
| rare words: ... | 2,079 (37) | 2,012 (40) |

**11.5's gain is 1.3% over the six and 3% on "rare words", not 2% and 5%.** The chooser's costs (`470 + 25 d`,
`530 + 24 d`) were fitted in the slow state; with the restarted machine's (`420 + 25 d`, `466 + 21 d`) the six take
9,616 ms against 9,615, one prompt 4% faster and one 2% slower, so they were left.

**Against Ollama 0.40.17, restarted, each engine alone, greedy, through each server's chat API, the fastest of
three runs:**

| request | ours, tok/s | Ollama, tok/s |
|---|---|---|
| an LRU cache class, 500 tokens (the same text from both) | 85 | 64 |
| a binary search tree class, 500 tokens (the same text) | 99 | 67 |
| two paragraphs on a suspension bridge, 249 and 259 tokens (the texts differ) | 46 | 57 |

And "rare words" through the raw API as in 10.4: ours 47 tok/s, Ollama 52 to 56. **Ours is ahead where most
drafts are accepted and behind where few are.**

**Not done.** A test that fails when a server it started does not exit: the stuck processes went unnoticed for a
day.
