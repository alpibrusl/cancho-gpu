edition 5;
module cuda;

// The shim's entry points (`shim/lexgpu.c`, `docs/device.md` §1), each
// under `Ffi("liblexgpu")`: integers in, an integer out, and at most one
// byte buffer, last. A negative answer is a failure; `lxg_error` copies
// its text out.
extern fn lxg_open[&f](ffi: &f Ffi("liblexgpu")) -> [ffi("liblexgpu")] int;

extern fn lxg_compile[&f, &s](ffi: &f Ffi("liblexgpu"), src: &s [byte]) -> [ffi("liblexgpu")] int;

extern fn lxg_function[&f, &s](ffi: &f Ffi("liblexgpu"), handle: int, name: &s [byte]) -> [ffi("liblexgpu")] int;

extern fn lxg_alloc[&f](ffi: &f Ffi("liblexgpu"), bytes: int) -> [ffi("liblexgpu")] int;

extern fn lxg_upload[&f, &s](ffi: &f Ffi("liblexgpu"), buf: int, data: &s [byte]) -> [ffi("liblexgpu")] int;

extern fn lxg_download[&f, &s](ffi: &f Ffi("liblexgpu"), buf: int, out: &!s [byte]) -> [ffi("liblexgpu")] int;

extern fn lxg_arg[&f](ffi: &f Ffi("liblexgpu"), buf: int) -> [ffi("liblexgpu")] int;

extern fn lxg_launch[&f](ffi: &f Ffi("liblexgpu"), func: int, gx: int, gy: int, threads: int) -> [ffi("liblexgpu")] int;

extern fn lxg_error[&f, &s](ffi: &f Ffi("liblexgpu"), out: &!s [byte]) -> [ffi("liblexgpu")] int;

extern fn lxg_close[&f](ffi: &f Ffi("liblexgpu")) -> [ffi("liblexgpu")] int;
