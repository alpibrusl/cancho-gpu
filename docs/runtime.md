# The runtime core: reading a model (slice 9a)

> **Status: stages 1 and 2 of 4 built and measured: GGUF headers (§2–§5)
> and safetensors headers (§7–§8), each on fixtures and on real files.**
> The BPE tokenizer and the sampler are not started (§6). Issue
> [#5](https://github.com/alpibrusl/cancho-gpu/issues/5).

Slice 9a ports the model-facing core of lex-gpu's `lex-rt` into cancho:
the part that reads a checkpoint and turns text into tokens and logits
into a token. It is pure CPU; nothing here needs a device. The Rust is the
oracle, as it is for the compiler (`docs/design.md` §1): a result is
accepted when it is **byte-identical** with what `lex-rt` produces for the
same input, and a refusal when `lex-rt` refuses the same input, or is
stricter in a way §3 lists.

Order, as the issue gives it, each stage its own pull request with its own
fixtures and gate:

1. GGUF headers: metadata and the tensor directory. *(built)*
2. Safetensors headers. *(built)*
3. The BPE tokenizer.
4. The sampler.

## 1. Why headers first, and what a header is

A GGUF file is a header and then tensor bytes. The header is the magic
`GGUF`, a version, the two counts, the metadata (key, type, value; arrays
and strings nest) and the tensor directory (name, dimensions, ggml type,
offset). The tokenizer lives in the metadata, as arrays of tens of
thousands of strings, so the header is both the smallest thing worth
getting exactly right and the bulk of what stages 3 and 4 read.

## 2. The reader

`src/gguf.cho` reads the first 32 MiB of the file (`gguf_cap`, more than
any header) straight into the text pool, as `main.cho` reads a `.lx`
source, and keeps **positions**: a key is `(start, length)`, a value is
the position after its type, a tensor's dimensions are the position of the
first. `parse` allocates only the small records naming those positions
(in `m`), so a vocabulary of 150,000 strings costs a count and a skip, not
150,000 copies. `skip` steps over any value, nested arrays included, and
is what both `parse` and the dump use.

It has no input that traps. Every length read from the file is a 64-bit
pattern, checked as a value against the bytes left (`room`) before it is
used; a metadata count is checked against the smallest record it could
describe (13 bytes), a tensor count against 24; an array of strings is
checked against its 8-byte minimum before the loop. `tests/gguf/fuzz.py`
feeds it every prefix of a header and 1,000 byte-mutated copies of it
(2,696 files): each answers a dump or `error[gguf-…]`, none traps or
hangs.

## 3. Where it is stricter than the Rust, and where it is the same

`lex-rt` accepts several things a reader of untrusted files should not.
This reader refuses them, each with a rule tag and a fixture in
`tests/gguf/reject/`, and in that direction by design: it adds refusals and removes none.
That every file it accepts `lex-rt` reads the same way is checked on the
fixtures (§5), not proved; the hostile files below are not compared with
the Rust.

| Rule | Refuses | `lex-rt` |
|---|---|---|
| `gguf-magic`, `gguf-truncated`, `gguf-value-type` | the same inputs | refuses (a string, not a tag) |
| `gguf-version` | version < 2 (as `lex-rt`) and version > 3 | accepts any version ≥ 2 |
| `gguf-duplicate` | a metadata key or tensor name twice | the later silently replaces the earlier |
| `gguf-alignment` | `general.alignment` that is not an integer in 1..1048576 | not an integer: uses 32; 0: panics |
| `gguf-block` | a tensor whose element count is not a whole number of blocks | rounds the byte count down |
| `gguf-tensor-offset` | an offset that is not a multiple of the alignment | accepts |
| `gguf-dims` | more than 8 dimensions, or a dimension or element count past 2^62 | accepts, and can overflow `usize` |
| `gguf-depth` | arrays nested more than 8 deep | recurses without a bound |
| `gguf-string` | a key or tensor name of 16 MiB or longer (a text-pool handle holds 24 bits of length) | accepts |
| `gguf-tensor-range` | `--sum` on a tensor whose bytes were not all read | `raw` fails the same way |

One difference is in the other direction and is stated, not hidden:
`lex-rt` reads strings with `from_utf8_lossy`; this keeps the file's
bytes. They agree on valid UTF-8; for invalid UTF-8 `lex-rt`'s value
contains U+FFFD and this reader's contains the original bytes (no fixture
covers it, because the dump escapes every byte past ASCII).

## 4. The dump, and how it is held to the oracle

`cancho-gpu gguf <file> [--sum]` prints a canonical text of the header
(`src/gguf_dump.cho`); `tools/gguf_dump.rs` prints the same text from
`lex-rt`'s `Gguf`. `tests/gguf.sh` compares them on every fixture:

```
gguf v3 tensors=6 kv=31 alignment=32
kv f32.third float 0x3fd5555560000000
kv str.escapes str "q\"uote \\ back\x09tab\x0a\x01 \xc3\xb1 end"
kv arr.strs array n=5 [str "a", str "bb", str "ccc", str "", ...]
tensor blk.0.ffn.weight type=Q8_0 dims=[64] bytes=68 sum=40d93799
```

* Keys and names are sorted byte-wise, because `lex-rt` holds them in a
  hash map and has no order to compare.
* A float is its binary64 bits. An f32 is widened **in integers** (the
  language has no f32 and the dump must not depend on a rounding), with
  subnormals normalised and a NaN given its quiet bit, as the hardware's
  conversion does. `lex-rt` stores both f32 and f64 as `f64`, so the dump
  cannot tell them apart and does not try.
* An array is its length and its first four elements.
* `--sum` is the 32-bit FNV-1a of each tensor's bytes. A wrong data offset
  or block size changes it, which is what makes it a test of the
  *directory*, not only of the metadata.

The goldens in `golden/gguf/` are `lex-rt`'s output, never this program's
(`scripts/regen-gguf-golden.sh <lex-gpu checkout>`, which copies the
example into the checkout for the build and removes it).

## 5. Measured

* Three fixtures (`tests/gguf/*.gguf`, made by `make_fixtures.py`) cover
  every scalar type with its extremes (`u64::MAX` as −1, `i64::MIN`, f32
  subnormals, ±0, ±inf, a quiet NaN, the largest f32), booleans, strings
  with quotes, backslashes, controls and non-ASCII, empty and nested
  arrays, every tensor type `lex-rt` knows the layout of and one it does
  not, versions 2 and 3, and alignments 32 (explicit and default) and 64.
  **All three dump byte-identically with `lex-rt`'s** (`lex-gpu`
  `851497d`).
* 17 hostile fixtures are each refused with the rule their name starts
  with and nothing written.
* 2,696 hostile files: no trap, no hang.
* **Five real models** from an Ollama store (`scripts/gguf-real.sh`):
  llama3.2:1b (147 tensors, 30 metadata keys, a 128k-entry vocabulary),
  qwen3:8b, qwen3.5:4b, qwen3.5:9b and ministral-3:3b (up to 458 tensors
  and 51 keys). The dump is **byte-identical with `lex-rt`'s on all five**,
  and takes about 10 ms for a 1.2 GB file (it reads the first 32 MiB).
  `--sum` was not run on them: it needs the tensor bytes, which are past
  that prefix.
* Not covered: a file whose header is larger than 32 MiB (none seen; it is
  refused as `gguf-truncated`), and any GGUF not from these five.

## 6. What is left of the slice

| Stage | What | Oracle |
|---|---|---|
| 3 | BPE tokenizer: vocabulary and merges from the GGUF metadata and from `tokenizer.json`, pre-tokenisation, byte fallback, special tokens, round-trip | `lex-rt/src/tokenizer.rs` |
| 4 | Sampler: greedy, temperature, top-k, top-p, with `lex-rt`'s generator | `lex-rt/src/sample.rs` |

Loading the tensors' *values* (dequantising Q8_0, Q4_K and Q6_K to what
the kernels take) belongs to the model adapters of slice 12 (issue #10),
which need the tensor directory this stage builds.

## 7. The safetensors reader (stage 2)

`src/safetensors.cho`. A file is an 8-byte little-endian length N, N bytes
of JSON, then the tensor bytes: for each tensor a dtype, a shape and a
`data_offsets` pair, and an optional `__metadata__` object of strings. Both
Qwen3.8 27B layouts on this machine have it: three 5 GB shards from
Hugging Face (`mlx-community/Qwen3.8-27B-4bit`, 2,180 tensors in all), and
Ollama's one-blob-per-tensor store (`lex-rt/src/qwen.rs`).

**The oracle is the `safetensors` library**, whose Rust reader validates a
file as the format defines it, not `lex-rt`: `qwen.rs` takes five dtypes
and checks no range. The format's checks are made here too: the length
fits the file, the header is JSON starting with `{`, every dtype is one of
the 15 of library version 0.5.3, a shape is non-negative integers, the byte range
of a tensor is exactly its element count times the dtype's size, and the
ranges, taken in order, **tile the data section**: no hole, no overlap, no
byte left after the last (`st-hole`, `st-overlap`, `st-data-short`,
`st-data-long`).

The JSON is `std.json`'s tape parser, strict (RFC 8259), into a tape in
`m`; names and metadata are decoded (escapes, surrogate pairs) into the
text pool. Nothing else is copied.

**The file's size.** cancho has no `file_size` and no positioned read
(`cancho/docs/large-files.md`, a design, nothing built), so the reader sees
the first 32 MiB. A file shorter than that is read whole and its size is
known; a longer one needs `size=<bytes>` on the command line (from `stat`)
for the checks that compare the tensors with the file's end, and without it
the dump says `data=unknown` and those two checks are not made. The
tensor bytes cannot be read at all yet; that is the gap `large-files.md`
names, and the reason this stage reads headers only.

Where it differs from the library (each has a rule tag and a fixture):

| Rule | This reader | The library (0.5.3) |
|---|---|---|
| `st-header-size` | refuses a header over 1 MiB (the tape is three words a byte) | up to 100,000,000 |
| `st-duplicate` | refuses a tensor name twice | the later replaces the earlier |

Fields in an entry that the format does not define are ignored, as the
library does (`extra-field.safetensors`).

Like the GGUF reader it has no input that traps: the header length is
checked before the header is read, and the tape is sized by
`json.tape_len`. `tests/safetensors/fuzz.py` runs every prefix of a file
and 1,000 mutated copies of its header (1,541 files): each answers a dump
or `error[st-…]`, none traps or hangs.

## 8. The dump, and how it is held to the oracle

```
safetensors header=312 tensors=4 data=224
meta format "mlx"
tensor layer.scales F16 shape=[8,1] range=[128,144]
```

`cancho-gpu safetensors <file> [size=<bytes>]` and
`tools/safetensors_dump.py` print it; names and metadata are sorted
byte-wise. The shapes and dtypes come from the library, and the byte
ranges, which it does not expose, from Python's `json` over the raw header.
`tests/safetensors.sh` (in the gate) compares them on the three fixtures
(`golden/safetensors/`, recorded from the library, version in `ORACLE`),
and each of 21 hostile fixtures is refused with the rule its name starts
with.

`tests/safetensors/differential.py` (not in the gate: it needs the library
and torch) runs the mutated files through both and compares verdicts and
dumps: on 941 files (every prefix and 400 mutations) there was no file one
reader accepted and the other refused, and the 10 both accepted dumped
identically. Mutations rarely leave a valid file, so this is evidence for
the refusals more than for the dump.

### Measured on real files

* **Five real files, byte-identical with the library**, with `size=` from
  `stat`: `Qwen2.5-0.5B-Instruct` (290 tensors) and `Qwen2.5-1.5B-Instruct`
  (338), and the three shards of `mlx-community/Qwen3.8-27B-4bit` (847, 720
  and 613 tensors; headers of 104,654, 94,306 and 80,125 bytes). Each takes
  under 10 ms, though the files are 1 to 5 GB: only the first 32 MiB is read.
* Ollama's `qwen3.8:27b-mlx` blobs: see §9.
* Not measured: reading a tensor's bytes, and any model run.
