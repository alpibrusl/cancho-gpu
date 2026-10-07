# CLAUDE.md — cancho-gpu

lex-gpu's `.lx` compiler in cancho. Read [`docs/design.md`](docs/design.md)
before changing it, and cancho's `AGENTS.md` (`cancho agent-guidelines`)
before writing cancho.

- **The gate:** `CANCHO=<pinned cancho> scripts/gate.sh` passes before
  anything is called done. The compiler is the commit in `cancho.toml`.
- **The oracle is the Rust.** Output must be byte-identical with lex-gpu's
  emitter; `golden/` is regenerated only from the Rust
  (`scripts/regen-golden.sh`), never from this compiler. A new lowering
  path gets a case in `tests/cases.txt` and a run of
  `scripts/differential.sh` against a lex-gpu checkout.
- **Every refusal has a rule tag** and a fixture in `tests/reject/`
  (`tests/reject.sh` fails on a tag without one). No input reaches a trap.
- **Two programs.** `cancho-gpu` (from `cancho.toml`) links only libc;
  `cancho-gpu-device` (`scripts/build-device.sh`) links `shim/lexgpu.c`.
  Nothing reachable from `src/main.cho` may call the shim.
- **No source file over 2,000 lines** (`tests/files.sh`). Split by concern.
- **Design before code**, in `docs/`, with claims measured; a claim that
  turns out false is corrected in place.
- cancho#236: a local named like a module's function hides it. Name
  module functions so they do not collide with the locals around them.
