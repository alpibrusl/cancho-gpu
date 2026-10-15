<p align="center"><img src="docs/assets/cancho-gpu-logo.png" alt="cancho-gpu" width="220"></p>

# cancho-gpu

[![ci](https://github.com/alpibrusl/cancho-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/alpibrusl/cancho-gpu/actions/workflows/ci.yml)

**A GPU kernel compiler that says what it can do.** [lex-gpu](https://github.com/alpibrusl/lex-gpu)'s `.lx` tile-kernel compiler, written in [cancho](https://github.com/alpibrusl/cancho): it checks a typed tile program (linearity, bounds, shapes, the threadgroup budget) and lowers it to **CUDA C** and **Metal**, byte-for-byte what lex-gpu's Rust emitter writes. It links nothing but libc, and `cancho authority`, checked in CI, names what it may do: read and write files, print, allocate; never the network, never foreign code. The [project page](https://alpibrusl.github.io/cancho-gpu/) has the summary.

**Status: the compiler and the device path are built; the inference runtime is under way.** CUDA has been run only against a mock driver (no machine here has an NVIDIA GPU); Metal has run on Apple Silicon. Slice 9a, reading a model, has its first stage (GGUF headers). The tokenizer, the sampler, serving, AMD and the model adapters are open issues (below).

## What you get

* **The same bytes as the Rust.** CUDA C and Metal for every emitting case of `tests/cases.txt` (140 cases: 116 emit, 24 are refused by both compilers) are byte-identical with lex-gpu's emitter, and a refusal names its rule ([`docs/rules.md`](docs/rules.md)). The goldens are recorded from the Rust, never from this compiler.
* **A reference interpreter.** `--run` interprets a kernel on the CPU and prints the same summary as the Rust's, f32 order included (14 runs).
* **A way to drive a GPU.** A C shim and a device program compile the emitted kernel with NVRTC (CUDA) or `newLibraryWithSource` (Metal), upload the interpreter's inputs, launch, and compare ([`docs/device.md`](docs/device.md)).
* **A model reader, in progress.** `cancho-gpu gguf <file>` reads a GGUF header (metadata and tensor directory) and refuses hostile bytes with a rule tag; its dump matches lex-rt's reader on every fixture and on five real models ([`docs/runtime.md`](docs/runtime.md)).
* **An authority you can read.** The compiler performs `args`, `err_write`, `fs_read`, `fs_write`, `heap`, `io_write` and nothing else; the device program is the one that calls foreign code, and says so.

How it is built, what it cost and what the language made hard: [`docs/design.md`](docs/design.md).

## Requirements

- The **cancho** compiler at the commit pinned in `cancho.toml`; the revision is part of the contract.
- Rust, to build that compiler.
- To run the tests: `python3` (the GGUF fixtures and fuzzing) and a C compiler (the device plumbing).
- Only to regenerate the goldens or run the live differential: a [lex-gpu](https://github.com/alpibrusl/lex-gpu) checkout and `cargo`.

## Quick start

```sh
git clone https://github.com/alpibrusl/cancho
git clone https://github.com/alpibrusl/cancho-gpu && cd cancho-gpu

REV=$(sed -n 's/^cancho = "\([0-9a-f]*\)".*/\1/p' cancho.toml)    # the revision CI uses
(cd ../cancho && git checkout "$REV" && cargo build --release -p cancho)
export CANCHO=$PWD/../cancho/target/release/cancho

$CANCHO build                         # build/cancho-gpu
mkdir -p out
build/cancho-gpu emit tests/lx/lex-gpu/gemm_fp4.lx out m=128 n=128 k=64
```

## Examples

**Emit a kernel** for every target the file has a schedule for:

```
$ build/cancho-gpu emit tests/lx/lex-gpu/gemm_fp4.lx out m=128 n=128 k=64
out/gemm_fp4_128_128_64_128_128_64.cu: 256 threads, grid 1x1, 36864 B of tiles
out/gemm_fp4_128_128_64_64_128_32.metal: 256 threads, grid 2x1, 12288 B of tiles
```

**Check it, and run it on the CPU:**

```
$ build/cancho-gpu check tests/lx/lex-gpu/rmsnorm.lx n=4096 eps=1e-5 --run
```

**A refusal names its rule:**

```
error[use-after-move]: `x.0` used after it was moved
```

**Read a model's header** (the same text lex-rt's reader prints; `--sum` adds a checksum of each tensor's bytes):

```
$ build/cancho-gpu gguf tests/gguf/tiny.gguf --sum
gguf v3 tensors=6 kv=31 alignment=32
kv general.architecture str "llama"
kv arr.strs array n=5 [str "a", str "bb", str "ccc", str "", ...]
tensor blk.0.ffn.weight type=Q8_0 dims=[64] bytes=68 sum=40d93799
...
```

**Run a kernel on a GPU** against the interpreter (needs a GPU, and is not part of the gate):

```sh
CANCHO=$CANCHO scripts/build-device.sh        # CUDA: build/cancho-gpu-device
CANCHO=$CANCHO scripts/build-metal-device.sh  # Metal: build/cancho-gpu-metal-device
```

**See what it is allowed to do** (checked by `tests/authority.sh`, in the gate):

```sh
$CANCHO authority --std src/*.cho
# performs: args, err_write, fs_read(""), fs_write(""), heap, io_write
# never touches: the network, signals, other programs, foreign code
```

## Checking it

```sh
CANCHO=$CANCHO scripts/gate.sh               # format, file budget, build, unit tests, goldens, refusals, device plumbing, GGUF, authority
scripts/differential.sh ../lex-gpu           # the same cases against a live build of the Rust
scripts/regen-golden.sh ../lex-gpu           # golden/ from the Rust, never from this compiler
scripts/regen-gguf-golden.sh ../lex-gpu      # golden/gguf/ from lex-rt's reader
```

`tests/lx/lex-gpu/` holds lex-gpu's own `.lx` kernels, copied at the commit in `golden/LEX_GPU`; `tests/lx/` holds programs written here for the lowering paths those do not reach (`softmax.lx` is the first kernel that needs `exp` and `log`).

## Roadmap

The epic is [cancho#251](https://github.com/alpibrusl/cancho/issues/251); each slice is an issue here.

| Slice | | State |
|---|---|---|
| 1–7 | parser, checker, CUDA and Metal emitters, interpreter | **built** |
| 8 | driving a CUDA GPU (shim, device program, mock driver) | **built**; real hardware not yet |
| 9a | runtime core: GGUF, safetensors, BPE tokenizer, sampler ([#5](https://github.com/alpibrusl/cancho-gpu/issues/5)) | GGUF headers **built**; the rest open |
| 9b | exp/log in `.lx`, transformer kernels ([#6](https://github.com/alpibrusl/cancho-gpu/issues/6)) | `exp`, `log`, `softmax` **built**; rope, attention and the KV cache are not started (the issue was closed after that first part) |
| 9c | serving: OpenAI-compatible chat endpoint ([#7](https://github.com/alpibrusl/cancho-gpu/issues/7)) | open |
| 10 | Metal device path ([#8](https://github.com/alpibrusl/cancho-gpu/issues/8)) | **built and run on Apple Silicon** |
| 11 | AMD ([#9](https://github.com/alpibrusl/cancho-gpu/issues/9)) | open |
| 12 | scale: model adapters, KV offload ([#10](https://github.com/alpibrusl/cancho-gpu/issues/10)) | open |

## Learn more

| | |
|---|---|
| [docs/design.md](docs/design.md) | how the compiler is built, what it cost, every difference from the Rust and why |
| [docs/rules.md](docs/rules.md) | every refusal rule and what triggers it |
| [docs/device.md](docs/device.md) | driving a GPU: the shim, the authority it reports, the mock, what stays unproven |
| [docs/runtime.md](docs/runtime.md) | reading a model: the GGUF reader, where it is stricter than lex-rt, how it is held to it |
| [The project page](https://alpibrusl.github.io/cancho-gpu/) | the summary |

## Layout

```
src/               the compiler: lexer, parser, elaboration, checker, lowering, interpreter, GGUF reader
device/            the device programs (CUDA, Metal): the only code that calls the shim
shim/              lexgpu.c (CUDA, over dlopen) and lexgpu-metal.mm (Metal)
tests/             goldens, refusal fixtures, the mock driver, GGUF fixtures and fuzzing
golden/            what lex-gpu's Rust writes for every case (and, in gguf/, what lex-rt dumps)
tools/             the Rust example that prints the GGUF dump
scripts/           gate, differential, golden regeneration, device builds
docs/              design, rules, device, runtime, and the project page
```

## Limitations

CUDA is verified against a mock driver, not a GPU; Metal's speed is not measured (reach and numerical agreement are). The lowering is Ada's for NVIDIA (no Hopper) and there is no AMD path yet. The GGUF reader is checked on synthetic fixtures and on five real models (headers only; it reads the first 32 MiB of a file, so tensor values are not read yet). There is no tokenizer, sampler or serving layer yet. `cancho-gpu` is a compiler and a reader; it is not an inference runtime.

## Contributing

Every change goes through `scripts/gate.sh` (the same as CI). A new lowering path gets a case in `tests/cases.txt` and a run of `scripts/differential.sh`; every refusal gets a rule tag and a fixture. Design before code, in `docs/`, with claims measured; a claim that turns out false is corrected in place. See [`CLAUDE.md`](CLAUDE.md).

## Licence

[EUPL-1.2](LICENSE).
