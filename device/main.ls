edition 5;

// `lexsys-gpu-device` -- run a `.lx` kernel on a CUDA device and hold it
// to the reference interpreter (`docs/device.md`).
//
//     lexsys-gpu-device run <file.lx> [name=value ...]
//
// Compiles the program for `nvidia-ada` as `emit` does, runs the
// interpreter over the made-up inputs, then compiles the emitted source
// with NVRTC, uploads the same inputs, launches the kernel, reads the
// writable parameters back and prints both summaries and the worst
// difference. Exit 0 when every output is within 2e-2 of its peak, 1
// with `error[device]` or `error[device-mismatch]` otherwise, 2 for a
// usage mistake.
//
// A second program, so `lexsys-gpu` itself links nothing but libc; this
// one links `shim/lexgpu.c` (`scripts/build-device.sh`).

import std.io;
import mem;
import text;
import err;
import kinds;
import f32;
import ir;
import parser;
import elab;
import check;
import target;
import dialect;
import lower;
import interp;
import cli;
import cuda;

fn words() -> [] int {
    return 8000000;
}

fn bytes_cap() -> [] int {
    return 256000000;
}

fn floats_cap() -> [] int {
    return 32000000;
}

fn usage[&i](io: &!i Io) -> [err_write] int {
    io.error_all(io, "usage: lexsys-gpu-device run <file.lx> [name=value ...]\n");
    return 2;
}

// The shim's last failure, as a `device` refusal.
fn device_fail[&m, &t, &f](m: &!m [int], t: &!t [byte], ffi: &f Ffi("liblexgpu"), what: &static [byte]) -> [ffi("liblexgpu")] int {
    let top = m[mem.text_slot()];
    let n = cuda.lxg_error(ffi, t[top..top + 4096]);
    // Keep the bytes the shim wrote before the pool is used again.
    var detail = text.handle(top, 0);
    if n > 0 {
        m[mem.text_slot()] = top + n;
        detail = text.handle(top, n);
    } else {
        detail = text.lit(m, t, "no detail");
    }
    return err.fail(m, t, "device", text.f2(m, t, "$: $", text.lit(m, t, what), detail));
}

// `n` bytes at the end of the pool, reserved: a staging area for one
// tensor's bytes.
fn reserve[&m](m: &!m [int], n: int) -> [] int {
    let at = m[mem.text_slot()];
    m[mem.text_slot()] = at + n;
    return at;
}

// A tensor's bytes as the device holds them: f32 and f16 as IEEE words,
// I8 as two's-complement bytes, little-endian. Answers where they start.
fn encode[&m, &t, &x](m: &!m [int], t: &!t [byte], fm: &!x [float], g: int) -> [] int {
    let dt = m[g];
    let size = kinds.size_bytes(dt);
    let (at, n) = (m[g + 2], m[g + 3]);
    let out = reserve(m, n * size);
    var i = 0;
    while i < n {
        let v = fm[at + i];
        var word = 0;
        if dt == kinds.f32() {
            word = f32.bits32(v);
        } else if dt == kinds.f16() {
            word = f32.bits16(v);
        } else {
            word = truncate(v) & 255;
        }
        var b = 0;
        while b < size {
            t[out + i * size + b] = byte_of(word >> 8 * b & 255);
            b = b + 1;
        }
        i = i + 1;
    }
    return out;
}

fn decode[&m, &t, &x](m: &!m [int], t: &!t [byte], fm: &!x [float], g: int, from: int) -> [] int {
    let dt = m[g];
    let size = kinds.size_bytes(dt);
    let (at, n) = (m[g + 2], m[g + 3]);
    var i = 0;
    while i < n {
        var word = 0;
        var b = size - 1;
        while b >= 0 {
            word = word << 8 | int_of(t[from + i * size + b]);
            b = b - 1;
        }
        if dt == kinds.f32() {
            fm[at + i] = f32.from_bits32(word);
        } else if dt == kinds.f16() {
            fm[at + i] = f32.from_bits16(word);
        } else {
            var v = word;
            if v >= 128 {
                v = v - 256;
            }
            fm[at + i] = float_of(v);
        }
        i = i + 1;
    }
    return 0;
}

// The worst `|device - reference|` over every writable tensor, against
// each one's peak: answers the lines saying so, and records a
// `device-mismatch` refusal past 2e-2 of the peak.
fn compare[&m, &t, &x](m: &!m [int], t: &!t [byte], fm: &!x [float], prog: int, want: int, got: int) -> [] int {
    let lines = mem.list(m);
    let ps = ir.p_params(m, prog);
    var i = 0;
    while i < mem.size(m, ps) {
        let p = mem.get(m, ps, i);
        if ir.param_writable(m, p) {
            let (a, b) = (mem.get(m, want, i), mem.get(m, got, i));
            var peak = 0.0;
            var worst = 0.0;
            var nan = false;
            var j = 0;
            while j < m[a + 3] {
                let (r, d) = (fm[m[a + 2] + j], fm[m[b + 2] + j]);
                peak = f32.max(peak, f32.abs(r));
                if is_nan(d) != is_nan(r) {
                    nan = true;
                } else if !is_nan(d) {
                    worst = f32.max(worst, f32.abs(d - r));
                }
                j = j + 1;
            }
            let name = ir.param_name(m, p);
            let l = mem.of3(m, name, f32.debug(m, t, f32.round(worst)), f32.debug(m, t, peak));
            mem.push(m, lines, text.fmt(m, t, "compare: $ worst |device - reference| = $ of peak $\n", l));
            if nan || worst > 0.02 * peak && worst > 0.0 {
                err.fail(m, t, "device-mismatch", text.f3(m, t, "`$` is $ from the reference, past 2e-2 of its peak $", name, f32.debug(m, t, f32.round(worst)), f32.debug(m, t, peak)));
            }
        }
        i = i + 1;
    }
    return text.joined(m, t, lines, "");
}

// Upload every parameter, launch, and read the writable ones back into
// `got` (fresh inputs, overwritten). Answers 0, or -1 with a refusal.
fn on_device[&m, &t, &x, &f](m: &!m [int], t: &!t [byte], fm: &!x [float], ffi: &f Ffi("liblexgpu"), prog: int, l: int, got: int, sizes: int) -> [ffi("liblexgpu")] int {
    if cuda.lxg_open(ffi) < 0 {
        return device_fail(m, t, ffi, "opening the device");
    }
    let compiled_mod = cuda.lxg_compile(ffi, text.bytes(t, m[l + 1]));
    if compiled_mod < 0 {
        return device_fail(m, t, ffi, "compiling the kernel");
    }
    let func = cuda.lxg_function(ffi, compiled_mod, text.bytes(t, m[l]));
    if func < 0 {
        return device_fail(m, t, ffi, "finding the entry point");
    }
    let bufs = mem.list(m);
    var i = 0;
    while i < mem.size(m, got) {
        let g = mem.get(m, got, i);
        let bytes = m[g + 3] * kinds.size_bytes(m[g]);
        mem.push(m, sizes, bytes);
        let buf = cuda.lxg_alloc(ffi, bytes);
        if buf < 0 {
            return device_fail(m, t, ffi, "allocating a buffer");
        }
        let at = encode(m, t, fm, g);
        if cuda.lxg_upload(ffi, buf, t[at..at + bytes]) < 0 {
            return device_fail(m, t, ffi, "uploading an input");
        }
        if cuda.lxg_arg(ffi, buf) < 0 {
            return device_fail(m, t, ffi, "binding an argument");
        }
        mem.push(m, bufs, buf);
        i = i + 1;
    }
    if cuda.lxg_launch(ffi, func, m[l + 2], m[l + 3], m[l + 4]) < 0 {
        return device_fail(m, t, ffi, "launching the kernel");
    }
    let ps = ir.p_params(m, prog);
    i = 0;
    while i < mem.size(m, got) {
        if ir.param_writable(m, mem.get(m, ps, i)) {
            let g = mem.get(m, got, i);
            let bytes = m[g + 3] * kinds.size_bytes(m[g]);
            let at = reserve(m, bytes);
            if cuda.lxg_download(ffi, mem.get(m, bufs, i), t[at..at + bytes]) < 0 {
                return device_fail(m, t, ffi, "reading an output back");
            }
            decode(m, t, fm, g, at);
        }
        i = i + 1;
    }
    cuda.lxg_close(ffi);
    return 0;
}

fn run[&m, &t, &x, &i, &f, &a, &g](m: &!m [int], t: &!t [byte], fm: &!x [float], io: &!i Io, fs: &f Fs(""), args: &a Args, ffi: &g Ffi("liblexgpu")) -> [io_write, err_write, fs_read(""), args, ffi("liblexgpu")] int {
    if arg_count(args) < 3 {
        return usage(io);
    }
    if !text.is(t, text.from_bytes(m, t, arg(args, 1)), "run") {
        return usage(io);
    }
    let file = arg(args, 2);
    let top = m[mem.text_slot()];
    let got = fs_read(fs, file, t[top..len(t)]);
    if got < 0 {
        io.error_all(io, "error[io]: cannot read ");
        io.error_all(io, file);
        io.error_all(io, "\n");
        return 1;
    }
    m[mem.text_slot()] = top + got;
    let vals = cli.constants(m, t, args, 3);
    if vals < 0 {
        return cli.report(m, t, io);
    }
    let unit = parser.parse(m, t, text.handle(top, got));
    if unit < 0 {
        return cli.report(m, t, io);
    }
    let tg = target.nvidia_ada(m, t);
    let compiled = elab.compile(m, t, unit, target.name(m, tg), vals);
    if compiled < 0 {
        return cli.report(m, t, io);
    }
    let (prog, s) = (m[compiled], m[compiled + 1]);
    if check.check(m, t, prog, tg) < 0 {
        return cli.report(m, t, io);
    }
    let l = lower.lower(m, t, prog, tg, dialect.cuda(), m[s + 1], m[s + 3], m[s + 4], m[s + 5]);
    if l < 0 {
        return cli.report(m, t, io);
    }
    let want = interp.inputs(m, t, fm, prog);
    if want < 0 || interp.execute(m, t, fm, prog, want) < 0 {
        return cli.report(m, t, io);
    }
    io.write_all(io, text.bytes(t, interp.summary(m, t, fm, prog, want, text.lit(m, t, "reference"))));
    let have = interp.inputs(m, t, fm, prog);
    if have < 0 {
        return cli.report(m, t, io);
    }
    let sizes = mem.list(m);
    if on_device(m, t, fm, ffi, prog, l, have, sizes) < 0 {
        return cli.report(m, t, io);
    }
    // What was uploaded, parameter by parameter: the arguments the
    // driver should have seen, in order (tests/device.sh).
    let ps = ir.p_params(m, prog);
    var k = 0;
    while k < mem.size(m, sizes) {
        let l2 = mem.of2(m, ir.param_name(m, mem.get(m, ps, k)), text.num(m, t, mem.get(m, sizes, k)));
        io.write_all(io, text.bytes(t, text.fmt(m, t, "upload: $ $ bytes\n", l2)));
        k = k + 1;
    }
    io.write_all(io, text.bytes(t, interp.summary(m, t, fm, prog, have, text.lit(m, t, "device"))));
    io.write_all(io, text.bytes(t, compare(m, t, fm, prog, want, have)));
    if err.failed(m) {
        return cli.report(m, t, io);
    }
    return 0;
}

fn main(world: World) -> [] int {
    let Split { io, ffi, fs, heap, args, net, clock } = split(world);
    release(net);
    release(clock);
    let gpu = narrow(ffi, "liblexgpu");
    var status = 0;
    borrow mut heap as &!h in {
        var mb = box_slice(h, words(), 0);
        var tb = box_slice(h, bytes_cap(), byte_of(0));
        var fb = box_slice(h, floats_cap(), 0.0);
        borrow mut mb as &!mr in {
            borrow mut tb as &!tr in {
                borrow mut fb as &!fr in {
                    let m = contents(mr);
                    let t = contents(tr);
                    let fm = contents(fr);
                    mem.init(m);
                    borrow mut io as &!i in {
                        borrow fs as &f in {
                            borrow args as &a in {
                                borrow gpu as &g in {
                                    status = run(m, t, fm, i, f, a, g);
                                }
                            }
                        }
                    }
                }
            }
        }
        unbox_slice(h, mb);
        unbox_slice(h, tb);
        unbox_slice(h, fb);
    }
    release(heap);
    release(io);
    release(fs);
    release(args);
    release(gpu);
    return status;
}
