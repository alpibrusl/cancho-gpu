# CLAUDE.md — cancho-gpu

A compiler for the `.chx` tile-kernel language, in cancho. The language began as lex-gpu's and is defined here: [`docs/chx.md`](docs/chx.md). Read [`docs/design.md`](docs/design.md)
before changing it, and cancho's `AGENTS.md` (`cancho agent-guidelines`)
before writing cancho.

- **The gate:** `CANCHO=<pinned cancho> scripts/gate.sh` passes before
  anything is called done. The compiler is the commit in `cancho.toml`.
- **There is no Rust oracle** (it was retired: `docs/native.md` N2). What a
  kernel *computes* is held to a float64 reference written from the
  mathematics (`tests/kernels.py`, every numeric kernel family, with
  mutants that must fail it) and, on a Mac, to the GPU. `golden/` is a
  regression record of what this compiler wrote
  (`scripts/record-golden.sh`; `git diff golden/` is read as text and an
  unexplained diff is a bug), not an answer. A new lowering path gets a
  case in `tests/cases.txt`, and, if it computes numbers, a reference. The
  order of a floating-point reduction is not specified (`docs/chx.md` §4):
  never compare a sum bit for bit across implementations. The GGUF,
  sampler and tokenizer goldens come from Python tools and Hugging Face
  (`tools/`, `docs/native.md` N2f); no Rust tool is left.
- **Every refusal has a rule tag** and a fixture in `tests/reject/`
  (`tests/reject.sh` fails on a tag without one). No input reaches a trap.
- **Two programs.** `cancho-gpu` (from `cancho.toml`) links only libc;
  `cancho-gpu-device` (`scripts/build-device.sh`) links `shim/canchogpu.c`.
  Nothing reachable from `src/main.cho` may call the shim.
- **No source file over 2,000 lines** (`tests/files.sh`). Split by concern.
- **Design before code**, in `docs/`, with claims measured; a claim that
  turns out false is corrected in place.
- cancho#236: a local named like a module's function hides it. Name
  module functions so they do not collide with the locals around them.
