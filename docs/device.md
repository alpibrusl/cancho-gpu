# Driving a GPU from lex-sys

> **Status: built (lex-sys#251 slice 8) and verified against a mock
> driver, §7. Not yet run on a GPU** — this repository's machines have
> none. §6 says exactly what that leaves unproven.

lex-sys's `docs/gpu.md` §6 left one question open: *"can a lex-sys
program drive a GPU at all, with no new backend?"* This is the answer
for CUDA. It is a question about **reach**, not speed: the kernels are
the ones this compiler already emits; what is new is a lex-sys program
that compiles one with NVRTC, uploads inputs, launches it and reads the
result back — and compares that against the reference interpreter
(`docs/design.md` §10).

## 1. Why a C shim, and how thin

lex-sys reaches C through `extern fn` under an `Ffi` capability, with
three limits that decide the shape (lex-sys `docs/opaque-pointers.md`,
`docs/foreign-linking.md`, `examples/tls_nb/gaps/`):

1. **No floats cross.** An extern parameter is an `int`, a `bool`, a
   `c_ptr`, or a byte slice.
2. **A byte slice crosses as a pointer *and* a length**, and only safely
   as the last parameter (`g7`, `g14`).
3. **`c_ptr` names only an extern's own parameters and results** — an
   ordinary function cannot take or return one, so a handle could not be
   passed around a lex-sys program.

The CUDA driver API needs the opposite of all three:
`cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream, void **params,
void **extra)` takes an array of pointers to arguments, and `CUdeviceptr`
and `CUfunction` are values a program holds. No signature in it can be
declared from lex-sys directly.

So a shim of about 250 lines of C, `shim/lexgpu.c`, owns every handle
and hands lex-sys **integers**: a buffer is `3`, a function `0`. Every
entry point takes `int64_t`s and at most one trailing byte buffer, and
answers an `int64_t` (≥ 0 a handle or a count, < 0 a failure whose text
`lxg_error` copies out):

| Entry | Does |
|---|---|
| `lxg_open()` | `dlopen` the driver (`libcuda.so.1`) and NVRTC, `cuInit`, device 0, a context |
| `lxg_compile(src)` | NVRTC to PTX for the device's own `compute_XY`, then `cuModuleLoadData`: a module |
| `lxg_function(module, name)` | `cuModuleGetFunction`: a function |
| `lxg_alloc(bytes)` | `cuMemAlloc`: a buffer |
| `lxg_upload(buf, bytes)` / `lxg_download(buf, out)` | `cuMemcpyHtoD` / `DtoH` of a whole buffer |
| `lxg_arg(buf)` | append a buffer to the next launch's argument list |
| `lxg_launch(fn, gx, gy, threads)` | `cuLaunchKernel` with the arguments appended, then synchronise; clears the list |
| `lxg_error(out)` | the last failure's text |
| `lxg_close()` | everything freed, context destroyed |

`lxg_arg` is how the `void **params` array is built without lex-sys ever
holding a pointer: the shim keeps it.

The shim `dlopen`s rather than links, as lex-gpu's own `lex-cuda` does,
so the program builds on any machine and a missing driver is an answer
(`error[device]: …`), not a link failure. `LEXGPU_LIBCUDA` and
`LEXGPU_LIBNVRTC` name other libraries, which is what the mock (§5)
uses.

## 2. The authority

Every entry is declared under `Ffi("liblexgpu")`, and the device program
narrows its `Ffi` to exactly that, so `lex-sys authority` reports
`ffi("liblexgpu")` and every symbol by name — and `unbounded`, honestly:
lex-sys's `docs/gpu.md` §6 predicted it would "report `ffi("libcuda")`
and nothing about the device". It does, one library over. Narrowing a
GPU capability is lex-sys's `gpu.md` §3 (`Gpu(device)`), which this
measures the need for but does not build.

## 3. The program

`lexsys-gpu-device run <file.lx> [name=value ...]`, a second binary
(`device/`), so the compiler itself still links nothing but libc:

1. Parse, elaborate, check and lower for `nvidia-ada`, exactly as
   `emit` does; run the reference interpreter over the made-up inputs.
2. Open the device, compile the emitted source, find the entry.
3. Upload every parameter with the **same** inputs the interpreter used,
   encoded as the device holds them: f32 as IEEE bits, f16 as half bits,
   I8 as bytes. lex-sys has no float bit casts in either direction for
   f32 or f16; `f32.ls` computes the bit patterns in integers.
4. Launch on the program's grid with the schedule's threads.
5. Download the writable parameters, decode them, and print the same
   summary line `--run` prints, prefixed `device:`, then the worst
   difference against the interpreter.

The interpreter is the reference, as it is in lex-gpu's own tests: the
GPU sums in a different order (split-K, fragments), so equality is not
the claim. **The refusal threshold** is a difference of 2e-2 of the
output's peak magnitude -- the same number lex-gpu uses as its log-prob
tolerance against an f32 reference, borrowed rather than derived;
past it, `error[device-mismatch]`.

## 4. What is not attempted

* **Metal.** It is Objective-C (`MTLDevice`, `newLibraryWithSource`),
  reached through `objc_msgSend` with a different signature per call —
  a shim like this one, in Objective-C, built and tested only on macOS.
  None of this repository's machines is one.
* **Timing.** Reach first.
* **Hopper**: NVRTC is told the device's own architecture, but the
  lowering is the Ada one; lex-gpu picks a Hopper target from the
  compute capability and this does not.

## 5. How it is checked without a GPU

A mock driver and a mock NVRTC (`tests/mock/`, C) stand in for the real
ones. They cannot run a kernel, so they check the **plumbing** — the part
the C-to-lex-sys boundary can get wrong:

* NVRTC's "PTX" is the source itself; the mock driver writes, per
  launch, the entry it was asked for, the grid, the block, how many
  arguments arrived and each buffer's size into a log, and refuses an
  entry name the source does not define.
* Memory is host memory: an upload followed by a download must give back
  the same bytes. The mock "kernel" writes nothing, so a writable
  output reads back as zeros and the comparison against the interpreter
  must fail with `device-mismatch` — which is itself the test that the
  comparison can fail.

`tests/device.sh` runs that for every kernel of lex-gpu at test sizes and
checks the log against what the program is: entry name, grid, threads,
one argument per parameter, each buffer the parameter's size. And it
runs once with no driver at all, which must be `error[device]`, not a
trap.

## 6. What remains unproven

That the emitted CUDA *compiles* under NVRTC (lex-gpu checks this with
`scripts/cuda_check.sh` on its own emitter's output; ours is the same
bytes, §9 of `design.md`), and that a real driver accepts these calls
and the results land within the threshold. Those need a machine with a
GPU — lex-gpu's `scripts/gcp/nvidia_test.sh` provisions an L4 — and are
the next measurement, not a claim this document makes.

## 7. Measured

`tests/device.sh`, in the gate, on every one of lex-gpu's six kernels:

* The "driver" was asked for the entry the golden `.cu` defines, on the
  grid and block `emit` reports, with **one argument per parameter whose
  size is the upload's**, in order; NVRTC was asked for `compute_89`.
* The mock's zeros were refused as `device-mismatch` every time, so the
  comparison can fail.
* With no driver, `error[device]: opening the device: no CUDA driver:
  libcuda.so: cannot open shared object file: …`, exit 1.

Watched fail: binding one argument fewer than the kernel takes was
caught on all six kernels (the sizes no longer match the uploads). The
first version of the check counted arguments from the kernel's
signature and missed it; it now compares the sizes, and the shim points
unused argument slots at a zero, so a short list can never make a
reader walk into stack garbage.

The bit patterns (§3) are checked against Python's `struct` packing of
the same values (`test_ieee_bit_patterns_for_the_device`): normal,
subnormal, the largest finite, infinity and `-0.0`, for f32 and f16.

What `lex-sys authority` says about the device program — §2's
prediction, exactly:

```
UNBOUNDED: this program calls foreign code. ...
performs
    args
    err_write
    ffi("liblexgpu")    <- unbounded
    fs_read("")
    heap
    io_write
unbounded by
    liblexgpu:lxg_alloc
    liblexgpu:lxg_arg
    ...
```

and the compiler itself, unchanged: `args, err_write, fs_read(""),
fs_write(""), heap, io_write`, bounded.
