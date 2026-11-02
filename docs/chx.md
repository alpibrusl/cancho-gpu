# The `.chx` language

> **Status: the specification of the language, first edition.** The language began as
> [lex-gpu](https://github.com/alpibrusl/lex-gpu)'s and this document replaces that Rust implementation
> as its definition ([`native.md`](native.md) N2). **How it was written:** §2 and §3 (the lexical
> structure and the grammar) are read off `src/lexer.cho` and `src/parser.cho` and are complete for
> what those accept; §4 and §5 (types and operations) take each operation's meaning from the places that
> already pin it: the elaborator and checker for shapes and linearity (`src/elab.cho`, `src/check.cho`),
> the float64 references of `tests/kernels.py` for what is computed. **Anything this document does not
> say, it does not specify**, and §7 lists what is open on purpose. A reader who finds the implementation
> doing something this document contradicts has found a bug in one of them.

## 1. What a `.chx` file is

One file is one **algo**: a typed tile program that says what each program instance computes from windows
of its inputs, plus one **schedule** for each target that says how it is cut (threads, tile extents). It is
compiled for a target with **constants** (the sizes, `n=4096`) into a GPU kernel: CUDA C for `nvidia-ada`, Metal
for `apple-m-series`. It can also be *interpreted* on the CPU, over tensors the caller gives, which is the
meaning the compiled kernels are held to.

## 2. Lexical structure

* White space separates tokens; `//` starts a comment that runs to the end of the line. A lone `/` divides.
* A **name** is an ASCII letter, `_` or any non-ASCII byte, followed by more of those and digits. The words `in`, `out`,
  `over`, `with`, `let`, `store`, `grid`, `yield`, `for`, `tile`, `schedule` and the operation names are recognised where the
  grammar expects them.
* A **number** is a decimal literal with an optional fraction and exponent (`16`, `0.00001`, `1e-5`). It is read to
  binary64 and, where a value is computed, rounded once to binary32.
* **Punctuation:** `( ) { } [ ] , : ; = * + - / & @`, and the two-character `->` and `..`.
* A target name may contain `-` (`apple-m-series`), which is not one token; the schedule rule joins the parts.

## 3. Grammar

```
file      := algo schedule*
algo      := 'algo' NAME [ '(' [NAME {',' NAME}] ')' ]      // the constants a caller supplies
             param+ [ 'tile' NAME {',' NAME} ] '{' block '}'
param     := ('in' | 'out') NAME ':' type
type      := DTYPE '[' dim {',' dim} ']'                      DTYPE := f16 | f32 | i8
dim       := ce                                               // a constant expression
ce        := ce ('+'|'-') ce | ce ('*'|'/') ce | '(' ce ')' | NUMBER | NAME      // * and / bind tighter

block     := { stmt } '}'
stmt      := 'let' NAME '=' expr
           | 'store' expr '->' NAME [ '[' ce {',' ce} ']' ]    // a window of an output at these offsets
           | 'grid' NAME 'over' ce {',' NAME 'over' ce}        // the program instance's coordinates
           | 'yield' expr {',' expr}                           // the value(s) of a loop body

expr      := expr ('+'|'-') expr | expr ('*'|'/') expr | atom   // * and / bind tighter than + and -
atom      := '&' NAME                                          // a borrow: the tile is read, not consumed
           | '(' expr ')'  |  NUMBER  |  NAME                 // a name is a move (it consumes the tile)
           | 'load' NAME                                       // the whole tensor
           | 'load' NAME '[' ce {',' ce} ';' dim {',' dim} ']' [space]   // a window: offsets ; extents
           | 'zeros' type [space]
           | 'mma' atom atom atom                              // acc a b
           | 'matmul' atom atom | 'matmul_nt' atom atom
           | 'dequant_fp4' atom atom atom NUMBER               // q scales g group
           | 'add_window' atom NAME '[' ce {',' ce} ']'
           | 'stage' ('f16'|'f32') atom
           | ('rowsum'|'rowmax'|'rsqrt'|'sigmoid'|'softplus'|'exp'|'log') atom
           | 'for' NAME 'in' ce '..' ce [ 'with' NAME {',' NAME} ] '{' block '}'
space     := '@' ('reg' | 'shared' | 'frag')

schedule  := 'schedule' NAME 'for' TARGET '{' { key } '}'      // NAME is the algo's; TARGET: nvidia-ada | apple-m-series
key       := 'threads' N | 'chunk' N | 'pad' N | 'warps' N N | 'frag' N | TILE N      // TILE: a name the algo declared with 'tile'
```

Commas between schedule keys are allowed and mean nothing. A schedule must set `threads`, and every `tile` the algo
declares; a target may have only one schedule; a file with no schedule for the target asked for is refused
(`no-schedule`). A call has no parentheses: `rowsum sq`, `rsqrt t`. **Anything the grammar does not derive is refused
as `syntax`** (and its message names the position, `line:col`).

## 4. Types, shapes and values

* **Element types:** `f32` (IEEE binary32), `f16` (IEEE binary16) and `i8` (a signed byte). A *tile* is a value of one
  element type and a shape of one or two dimensions, living in a **memory space**: `reg` (the default; the instance's
  registers), `shared` (the threadgroup's memory, counted against the target's budget: rule `budget`) or `frag` (the
  matrix-multiply fragment layout, for `mma`).
* **Tensors** are the algo's parameters (`in` read-only, `out` written). They are row-major; a `load` window is
  `[offsets ; extents]` and every offset must be an *affine function of the instance's coordinates and loop indices*
  that keeps the window inside the tensor on every iteration (rules `offset`, `bounds`).
* **Constants** are integers given at compile time (`n=4096`) or, for the ones that appear as values (`eps`), decimal
  numbers, rounded to binary32. Dimensions and offsets are constant expressions over the constants, `tile` extents,
  loop indices and `grid` coordinates, evaluated with integer arithmetic.
* **Broadcasting:** a binary operation takes two tiles of the same shape, **or** a 2-D left operand with a 1-D right operand
  whose length is the left's row count (each row of the left meets one value of the right), **or** a 2-D left operand with a
  right operand of shape `[1, n]` (each column meets one value). There are no other broadcasts (`shape`). A number
  literal takes the shape of the other operand: `ms + eps`, `2 * q`.
* **Conversions** are never implicit: storing a wider type into a narrower tensor, or combining dtypes, is refused
  (`narrowing`, `type`); arithmetic on `i8` is refused (`type`).

## 5. Operations

All arithmetic on `f32` is IEEE binary32 and **each operation is correctly rounded**: `+ - * /` are the exact result
rounded to nearest, ties to even, with overflow to infinity and gradual underflow. `f16` and `i8` values are rounded to
their type where they are stored into a tensor of that type or staged (`stage f16`): `f16` to nearest even; `i8` to the nearest
integer, saturated to -128..127, NaN to 0 (the tie rule is §7).

| Form | Meaning |
|---|---|
| `load x` / `load x[o ; e]` | the tensor, or the window of extents `e` at offsets `o`, as a tile of the tensor's dtype |
| `zeros T` | a tile of zeros |
| `a + b`, `a - b`, `a * b`, `a / b` | elementwise, with the broadcasts of §4, `f32` unless both are another type |
| `sigmoid`, `softplus`, `exp`, `log`, `rsqrt` | elementwise: `1/(1+e^-x)`, `ln(1+e^x)`, `e^x`, `ln x`, `1/sqrt(x)`; accuracy is §7 |
| `rowsum a`, `rowmax a` | the sum / maximum of each row of a 2-D tile: a 1-D tile of the row count |
| `matmul a b` | `a b`, `[m,k] x [k,n]` to `[m,n]` |
| `matmul_nt a b` | `a b^T`, `[m,k] x [n,k]` to `[m,n]` |
| `mma c a b` | `c + a b^T`, accumulated; `a`, `b` are 2-D `f16` tiles in `shared` memory and `c` an `f32` fragment (`frag`) tile; a fragment can be accumulated into, stored, or carried through a loop, and nothing else (`type`) |
| `dequant_fp4 q s g G` | `[n,k]` weights from `q: i8[n,k/2]` (two NVFP4 e2m1 codes a byte, low nibble first), `s: i8[n,k/G]` (one e4m3 scale per group of `G`), `g: f32[n]` (one scale per row): `e2m1(code) * e4m3(scale) * g[row]` |
| `stage T a` | a copy of `a` in `shared` memory, rounded to `T` |
| `add_window acc x[o]` | `acc` plus the window of `x` at offsets `o`, of `acc`'s shape |
| `store e -> y[o]` | write `e` into the window of `y` at `o` (the origin if no offsets), rounding to `y`'s dtype |
| `for i in a .. b with c { ... yield e }` | run the body for `i = a, a+1, ..., b-1`, with the carried tile `c` bound to the previous iteration's `yield` (to `c`'s initial value at first); the loop's value is the last yield. At most one carried tile; it must be yielded in every iteration with the same type (`carry`) |
| `grid i over e, j over f` | the instance's coordinates: the kernel is run for every `(i, j)` in `0..e x 0..f`, instances independent; at most two axes, not inside a loop (`grid`). **A kernel with no `grid` is one instance, which runs on one threadgroup: a core of the GPU.** Parallelism is only what the grid says (`native.md` N4: two products of 2 MB took 150 us without one and 10 us with) |

### Reductions: the order is not specified

`rowsum`, the contraction in `matmul` / `matmul_nt` / `mma`, and the accumulation across loop iterations are **sums of
`f32` values whose order this document does not fix.** An implementation may sum in any order and with any grouping
(sequentially, in blocks, with several accumulators, in a threadgroup tree), and a program must not depend on the bits
of a result. The CPU interpreter sums in program order today; the GPU kernels do not. What is specified is the
*mathematical* result and a tolerance: a conforming implementation's result differs from the exact sum by no more than
the rounding of a reduction of that length (the references of `tests/kernels.py` allow 1e-5 for sums of up to a few
hundred terms and 1e-4 for 4096). `rowmax` has no such freedom: a maximum is exact in any order.

## 6. Linearity: a tile is consumed exactly once

A **name used as a value moves the tile** (`let y = f x` consumes `x`); `&x` reads it without consuming it. Every tile
must be consumed exactly once on every path, and the checker refuses what breaks that: use after a move
(`use-after-move`), one operation that moves a tile and also uses it (`moved-while-borrowed`), a tile never consumed
(`leak`), a loop body that consumes a tile defined outside it (`consume-outer`), a variable out of scope (`scope`). The
other refusals, in `docs/rules.md`, are `syntax`, `unbound`, `offset`, `shape`, `carry`, `grid`, `chunk`, `no-schedule`, `type`,
`narrowing`, `bounds`, `budget` and `lowering`. **Every refusal has a rule tag and a fixture in `tests/reject/`, and refusal
is by tag**: the message is for a person and is not part of the language.

## 7. Targets, and what is open

* **Targets:** `nvidia-ada` (CUDA C) and `apple-m-series` (Metal). The schedule keys have the meaning each backend gives
  them (`threads` per threadgroup, `warps r c` the warp grid of an `mma`, `bm bn bk` tile extents, `chunk` a 1-D split, `pad` a
  threadgroup row padding, `frag` the number of 8-row tiles a simdgroup owns in a register-fragment product); the
  budgets are the target's threadgroup memory and thread limits (`docs/kernels.md`, `docs/design.md`). A schedule affects
  performance and the order of reductions, never the mathematical result.
* **Open: the accuracy of `exp`, `log`, `rsqrt`, `sigmoid` and `softplus`.** The interpreter computes them in binary64
  and rounds once to binary32; the GPU backends use their libraries' functions. This document specifies no ulp bound for
  them; the references check results to 1e-5 relative. A bound should be chosen, written here, and measured.
* **Open: NaN and infinity.** `rowmax` of a row with a NaN, and `exp` overflow, are as the implementations give them. Not specified.
* **Open: the tie rule of `i8` rounding** (`stage`/`store` to `i8`): the implementation (`src/f32.cho` `round_i8`) uses `std.math.round`; whether
  that is half away from zero or to even has not been checked against a case, and the backends' behaviour is not specified.
* **Open: an arithmetic *semantics* as a document of its own** (a formal one): this edition is prose and tables.

## 8. How an implementation is held to this

1. **The grammar and the checker:** `tests/reject/` has one fixture for every rule tag the sources can raise
   (`tests/reject.sh` fails on a tag without one, and on a fixture the compiler accepts).
2. **What is computed:** `tests/kernels.py` holds the interpreter, on both targets, to a float64 reference written from the
   mathematics, for every one of the 41 numeric kernel families in `tests/cases.txt`, and `--mutants` makes one deliberate change
   to nine of them that their references must catch. On a Mac, `scripts/metal-io.py` and `scripts/metal-kernels.sh` hold the
   GPU to the interpreter.
3. **The emitted text:** `golden/`, recorded by `scripts/record-golden.sh`, is a **regression record**, not an answer:
   it says what this compiler wrote when the other checks passed. For CUDA, which no machine here can run, it is the only
   check on the text, and it is a weak one.
4. **Not held:** the CUDA kernels have never run on a GPU (`docs/device.md`); the exact-rounding claims of §5 are
   those of the interpreter (cancho's `f32` is bit-identical to a correctly rounded operation: cancho `docs/f32.md` F1) and are
   not checked on a device.

## 9. Where this differs from the implementation it began as

The language was ported from lex-gpu's `.chx` and every case in `tests/cases.txt` first agreed with that implementation
byte for byte (218 cases, `golden/PROVENANCE`). Two differences are on purpose. The schedule key `frag` (the register-fragment
product) does not exist in the Rust. And §5 leaves the **order of a reduction unspecified**, where the Rust's `Sum for f32` fixed
program order: this edition frees it, but the interpreter still sums in program order, so nothing computed has changed yet;
`native.md` N3 is what would use the freedom.
