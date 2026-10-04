# lexsys-gpu

[lex-gpu](https://github.com/alpibrusl/lex-gpu)'s `.lx` kernel compiler,
written in [lex-sys](https://github.com/alpibrusl/lex-sys).

A `.lx` file is a typed tile program: an `algo` says what to compute,
over tiles that are each consumed exactly once, and a `schedule` per
target says how big the pieces are. This compiler checks it — linearity,
bounds, shapes, the threadgroup budget — and lowers it to **CUDA C**
(`nvidia-ada`) and **Metal** (`apple-m-series`), producing the same
bytes as lex-gpu's Rust emitter for every case in `tests/cases.txt`.

It is the epic lex-sys#251: lex-sys's first compiler-sized program, held
to an external oracle.

```sh
lex-sys build                                  # the compiler pinned in lex-sys.toml
mkdir -p out
build/lexsys-gpu emit tests/lx/lex-gpu/gemm_fp4.lx out m=128 n=128 k=64
build/lexsys-gpu check tests/lx/lex-gpu/rmsnorm.lx n=4096 eps=1e-5
```

```
out/gemm_fp4_128_128_64_128_128_64.cu: 256 threads, grid 1x1, 36864 B of tiles
out/gemm_fp4_128_128_64_64_128_32.metal: 256 threads, grid 2x1, 12288 B of tiles
```

A refusal names its rule (`docs/rules.md`):

```
error[use-after-move]: `x.0` used after it was moved
```

## Status

| | |
|---|---|
| Parser, elaboration, checker, CUDA and Metal emitters | **built**: byte-identical with the Rust on 98 emitting cases, and refuses the 24 the Rust refuses |
| CPU reference interpreter (`--run`) | not yet (lex-sys#251 slice 7) |
| Driving a GPU (`Ffi` + a C shim) | not yet (slice 8) |
| The inference runtime | not yet (slice 9) |

How it is built, what it cost and what the language made hard:
[`docs/design.md`](docs/design.md).

## Checking it

```sh
LEX_SYS=lex-sys scripts/gate.sh               # format, file budget, build, unit tests, goldens, refusals
scripts/differential.sh ../lex-gpu            # the same cases against a live build of the Rust
scripts/regen-golden.sh ../lex-gpu            # golden/ from the Rust, never from this compiler
```

`tests/lx/lex-gpu/` holds lex-gpu's own `.lx` kernels, copied at the
commit in `golden/LEX_GPU`; `tests/lx/` holds programs written here for
the lowering paths those do not reach.

Licensed under the EUPL-1.2.
