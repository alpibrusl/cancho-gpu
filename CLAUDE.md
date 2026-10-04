# CLAUDE.md — lexsys-gpu

lex-gpu's `.lx` compiler in lex-sys. Read [`docs/design.md`](docs/design.md)
before changing it, and lex-sys's `AGENTS.md` (`lex-sys agent-guidelines`)
before writing lex-sys.

- **The gate:** `LEX_SYS=<pinned lex-sys> scripts/gate.sh` passes before
  anything is called done. The compiler is the commit in `lex-sys.toml`.
- **The oracle is the Rust.** Output must be byte-identical with lex-gpu's
  emitter; `golden/` is regenerated only from the Rust
  (`scripts/regen-golden.sh`), never from this compiler. A new lowering
  path gets a case in `tests/cases.txt` and a run of
  `scripts/differential.sh` against a lex-gpu checkout.
- **Every refusal has a rule tag** and a fixture in `tests/reject/`
  (`tests/reject.sh` fails on a tag without one). No input reaches a trap.
- **No source file over 2,000 lines** (`tests/files.sh`). Split by concern.
- **Design before code**, in `docs/`, with claims measured; a claim that
  turns out false is corrected in place.
- lex-sys#236: a local named like a module's function hides it. Name
  module functions so they do not collide with the locals around them.
