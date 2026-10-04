# lexsys-gpu: design

> **Status: slices 1–6 of lex-sys#251 built and measured.** The `.lx`
> front end, checker and both emitters of
> [lex-gpu](https://github.com/alpibrusl/lex-gpu), written in lex-sys,
> produce **byte-identical** CUDA C and Metal for every case in
> `tests/cases.txt`: 122 cases, 98 that emit and 24 that both compilers
> refuse.

---

## 1. What this is, and what it is not

lex-gpu compiles a typed tile program — an `algo` that says what to
compute over tiles, and a `schedule` per target that says how big the
pieces are — to CUDA C and Metal Shading Language. Its compiler is
Rust. This repository is the same compiler in lex-sys.

It is **not** a GPU backend for lex-sys, and it does not reopen
lex-sys's `docs/gpu.md` §5 (whether a GPU dialect of lex-sys should
exist). `.lx` stays its own language; lex-sys is what its compiler is
written in. The program is text in, text out: nothing here runs on a
device.

Why it exists (lex-sys#251):

* lex-sys had no compiler-sized program written in it. This is one,
  with an external oracle — the Rust emitter — so "does it work" is a
  `diff`, not an opinion.
* It is a real asker for the language gaps the GPU discussion named
  (f32, the missing direction of `bits_of`), measured by a program that
  needs them instead of argued (§6).

## 2. The pipeline

| Stage | Rust (lex-gpu) | Here |
|---|---|---|
| Lexer | `lex_front::syntax::Lexer` | `src/lexer.ls` |
| Parser | `lex_front::syntax::Parser`, `parse` | `src/parser.ls` |
| Elaboration (`Unit::compile`) | `syntax::Algo::build_with`, `lower`, `substitute`, `affine` | `src/elab.ls` |
| Tile IR and builder | `lex_front::ir` | `src/ir.ls` |
| Checker | `lex_front::check` | `src/check.ls` |
| Target table | `lex_ir::Target` | `src/target.ls` |
| Lowering | `lex_msl::program` | `src/gen.ls`, `src/lower.ls`, `src/matmul.ls` |
| Dialects | `lex_msl::dialect` (`Msl`, `Cuda`, `Wmma`, `Simdgroup`) | `src/dialect.ls` |
| CLI | `lex-msl/examples/emit_lx.rs` | `src/main.ls` |

The ported subset is everything a `.lx` file can reach. The Rust IR has
more — futures, arrays, pipes and warp-specialised roles, the Q4/Q6/
ternary dequantisations — but those come only from the Rust builder
API; the surface has no syntax for them, so neither does this port. The
same goes for the lowering paths only they reach.

## 3. Data: one memory of `int`, one of bytes

### 3.1 Why not structs and `Vec`

A compiler is many small tables: tokens, syntax nodes, IR statements,
the storage of every value. lex-sys offers growable collections as
`res` values moved from call to call (`std.vec`), and **a struct cannot
hold a reference** (lex-sys `examples/tls_nb/gaps/t2_ref_field`), so a
function that reads five tables takes five parameters and a function
that grows one returns it. Done that way, every function in the
lowering would carry a dozen tables in and out.

Instead there is one boxed slice of `int`, `m`, with a bump allocator
(`src/mem.ls`), and one boxed slice of bytes, `t`, for text
(`src/text.ls`). Every table is a record in `m` named by its offset.
Every function takes `m` and `t` and nothing else that is mutable. The
first 64 words of `m` are fixed slots: the allocator's top, the text
pool's top, the error list.

What this costs, plainly: **the records are untyped.** A `TileTy` and a
`View` are both "an `int` that is an offset", and passing one for the
other is not a type error. Bounds checks still hold — a wrong handle
traps at the access instead of reading another record's memory — and
the goldens catch any mistake that changes output, but a typed record
reference would have caught it at compile time. That is the gap the
`t2_ref_field` note names, met by a program.

### 3.2 Strings

A string is one `int`: `start * 2^24 + length`, a handle into `t`.
Strings are appended and never changed, so a handle stays valid for the
whole run and a substring is free. The emitters build text the way the
Rust does with `format!`: `text.f2(m, t, "for (uint k = 0; k < $u; ++k)
{", a, b)` splices handles at each `$`. Arguments are evaluated before
the call, so a string under construction is never interleaved with
another, which is why the pool can be a single bump pointer.

### 3.3 Enums

Every tag is a small integer and `src/kinds.ls` names them. The orders
follow the Rust enums (`DType`, `Space`, `BinOp`, ...).

### 3.4 Memory

8M words and 64 MB of text are reserved; untouched pages are not
resident. Emitting the largest case (`gemm_fp4_res`, m=512 n=5120
k=17408) peaks at **9 MB** resident.

## 4. f32 without an f32 type

lex-sys has one float type, binary64 (lex-sys `docs/floating-point.md`
§1). The Rust computes constants in f32 — `1 / n` is folded as
`1f32 / n as f32` — and prints them with `{:?}`, and "the same program"
means the same constant in the text. So f32 rounding and f32 printing
both have to be exact (`src/f32.ls`):

* **An f32 is held as a `float` whose value is one.** Every f32 is
  exactly a binary64.
* **Rounding** a binary64 to the nearest f32 (ties to even, overflow to
  infinity, subnormals included) is done in integers on `bits_of`, and
  the result rebuilt with `math.ldexp`, which is exact for every f32.
  lex-sys has `bits_of` and nothing that builds a float from bits;
  `ldexp` is the way round it.
* **Arithmetic**: binary64 has more than `2·24 + 2` significand bits, so
  one f32 `+ − × ÷` done in binary64 and rounded once is the correctly
  rounded f32 result.
* **Printing** is Steele and White's shortest-digits algorithm — the one
  lex-sys's `std.fmt` uses for binary64 — with an f32's neighbours, ties
  rounded up as Rust's shortest mode does, then laid out as Rust's
  `{:?}` lays out an f32: positional from 1e-4 to 1e16 with at least one
  fractional digit, exponential outside it.
* **Parsing** a decimal literal is `std.json`'s correctly rounded
  conversion, after bringing the text to JSON's grammar (which forbids
  the leading zeros and bare trailing point Rust accepts).

Measured: `tests/unit_test.ls` checks 25 literals against what Rust
printed for them (recorded by running Rust), and `tests/cases.txt`
sweeps 72 `rmsnorm` cases whose `1/n` and `eps` constants cover both
layouts, subnormals, overflow and the exact-tie case `2^-12`, which
Rust prints `0.00024414063`.

**Found by the differential, not by reading:** a kernel's name carries
each whole constant as `v as usize`, and for `eps = 1e20` that is
`18446744073709551615` — `usize::MAX`, saturated. lex-sys's `int` stops
at 2^63, so the first version printed `4611686018427387904`. Five cases
of the sweep differed; `f32.count_string` now prints the exact integer
below 2^64 from two 32-bit halves, and the saturated value above it.

## 5. The IR records

As `src/ir.ls` documents them:

* TileTy `[dtype, shape, space]`; Ty is a TileTy, or `-2` for a loop
  index.
* Arg `2·var` moves, `2·var + 1` borrows.
* IdxExpr `[constant, terms]`, terms `v0, c0, v1, c1, …`.
* View `[param, offsets, shape]`.
* Op `[kind, a, b, c, d]`.
* Stmt `[st_let, dst, op]` or `[st_for, index, start, end, init, params,
  body, results]`; Block `[stmts, yields]`.

Variables are numbered in the order the Rust builder numbers them,
because the emitter names storage after them (`v12`). That makes
`fresh` order part of the contract; `ir.for_begin`/`for_end` split the
Rust's closure-taking `for_range` at exactly the points the closure ran.

## 6. What the language made hard, counted

| | |
|---|---|
| **A local shadows a module's function** (lex-sys#236) | Hit 5 times: `ir.op`, `ir.index`, `gen.lines`, `gen.d`, `dialect.entry` became unreachable from functions with a local of that name. Each fixed by renaming the function |
| **Reserved words and prelude names** | `alloc`, `val`, `res`, `defer` cannot be identifiers; `join` and `split` cannot be declared even inside a module, because the prelude's builtins share the namespace |
| **No reference in a struct** | §3.1: the whole memory model follows from it |
| **No float from bits** | §4: worked around with `ldexp`; the f32 module is ~470 lines |
| **No `mkdir`** | `emit` needs its output directory to exist; the Rust creates it |

What it did **not** make hard: recursion (the parser and the checker
are recursive), the checker's own linearity (a tile-linearity checker
written in a linear language), and speed — an emit takes 4 ms against
the Rust release build's 4–6, both dominated by starting the process.

## 7. Size

The Rust this ports is 7,514 lines in five files, of which about 5,200
are what the surface can reach -- an estimate from reading which arms
are reachable, not a measurement. The port is **7,551
lines** of lex-sys, of which about 1,460 are infrastructure the Rust
gets from its standard library: the memory and lists, the string pool
and formatting, f32. Roughly 1.2× for the compiler proper.

## 8. Differences from the Rust, on purpose

* **No schedule for either target is a refusal** (`no-schedule`). The
  Rust example writes nothing and exits 0.
* **The output directory must exist** (§6).
* **No environment knobs.** The Rust reads `LEX_VEC`, `LEX_NARROW` and
  `LEX_NO_LAZY` to measure alternatives; unset, they are the defaults,
  and the defaults are what this port does.
* **No `--run`.** The reference interpreter is slice 7 of lex-sys#251,
  and wants f32 arithmetic over whole tensors.
* **A grid beside a schedule's `chunk`, or two `grid` statements,** is a
  `grid` refusal; the Rust panics on an assertion in its builder.

## 9. How it is checked

* `tests/golden.sh`: every case of `tests/cases.txt` against
  `golden/`, which is the Rust emitter's output at the lex-gpu commit in
  `golden/LEX_GPU`, regenerated only from the Rust
  (`scripts/regen-golden.sh`).
* `scripts/differential.sh <lex-gpu>`: the same cases against a live
  build of the Rust.
* `tests/reject.sh`: one fixture per rule tag, each refused with its own
  tag and nothing written; and every tag the sources can raise has a
  fixture.
* `tests/unit_test.ls`: f32 and the string pool.

Each was broken on purpose and watched fail: a changed loop spelling in
`gen.owned` (every case differed), one byte of one golden (that case
failed), the tie rule of the f32 printer (two unit tests failed).
