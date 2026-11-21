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

Each request starts from nothing: the conversation is prefilled again, eight tokens at a time. There is no
prefix cache (a design for one is section 7). Measured on an M4 Max (one run, the machine otherwise quiet): a short prompt's first token
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

Concurrent requests and batching them (one GPU conversation at a time), a prefix cache across turns (designed in section 7), tools,
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
cut is `c = 8 * floor(b / 8)`, a multiple of `qwen35.mb()`. Two reasons for the multiple of 8:

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
  instruction. It needs several slots.
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
4. Later, on its own design: several slots.

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
conversation (7.7). Step 3 decides the default.
