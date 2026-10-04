edition 5;

// `lexsys-gpu` -- lex-gpu's `.lx` compiler, in lex-sys.
//
//     lexsys-gpu emit  <file.lx> <dir> [name=value ...] [--run]
//     lexsys-gpu check <file.lx> [name=value ...] [--run]
//
// `emit` checks the program, lowers it for every target it has a
// schedule for (`nvidia-ada` to CUDA C, `apple-m-series` to Metal) and
// writes `<dir>/<kernel>.cu` and `<dir>/<kernel>.metal`; it is
// `lex-msl`'s `emit_lx` example, and `tests/golden.sh` holds the two to
// the same bytes. `check` stops after the checker. `--run` interprets
// the program on the CPU over made-up inputs first and prints what it
// writes, as `emit_lx --run` does (`docs/design.md` §10). The directory must
// exist: lex-sys has no way to make one.
//
// Exit status: 0, or 1 with every refusal on standard error as
// `error[<rule>]: <message>`, or 2 for a usage mistake.

import std.io;
import mem;
import text;
import err;
import f32;
import parser;
import elab;
import check;
import target;
import dialect;
import lower;
import interp;
import cli;

fn words() -> [] int {
    return 8000000;
}

fn bytes_cap() -> [] int {
    return 64000000;
}

// Floats for `--run`: untouched pages cost nothing.
fn floats_cap() -> [] int {
    return 32000000;
}

fn usage[&i](io: &!i Io) -> [err_write] int {
    io.error_all(io, "usage: lexsys-gpu emit <file.lx> <dir> [name=value ...] [--run]\n       lexsys-gpu check <file.lx> [name=value ...] [--run]\n");
    return 2;
}

// The kernel for one target: `[lowered, path suffix]`, or 0 when the
// file has no schedule for it, or -1.
fn one_target[&m, &t, &f](m: &!m [int], t: &!t [byte], fm: &!f [float], unit: int, tg: int, d: int, vals: int, lower_it: bool, run_it: bool) -> [] int {
    let name = target.name(m, tg);
    let scheds = m[unit + 1];
    var found = false;
    var i = 0;
    while i < mem.size(m, scheds) {
        if text.eq(t, m[mem.get(m, scheds, i)], name) {
            found = true;
        }
        i = i + 1;
    }
    if !found {
        return 0;
    }
    let compiled = elab.compile(m, t, unit, name, vals);
    if compiled < 0 {
        return 0 - 1;
    }
    let (prog, s) = (m[compiled], m[compiled + 1]);
    if check.check(m, t, prog, tg) < 0 {
        return 0 - 1;
    }
    var ran = text.empty();
    if run_it {
        ran = interp.run(m, t, fm, prog, name);
        if ran < 0 {
            return 0 - 1;
        }
    }
    if !lower_it {
        return mem.rec2(m, 0, ran);
    }
    let l = lower.lower(m, t, prog, tg, d, m[s + 1], m[s + 3], m[s + 4], m[s + 5]);
    if l < 0 {
        return 0 - 1;
    }
    return mem.rec2(m, l, ran);
}

fn run[&m, &t, &x, &i, &f, &a](m: &!m [int], t: &!t [byte], fm: &!x [float], io: &!i Io, fs: &f Fs(""), args: &a Args) -> [io_write, err_write, fs_read(""), fs_write(""), args] int {
    if arg_count(args) < 3 {
        return usage(io);
    }
    let cmd = text.from_bytes(m, t, arg(args, 1));
    let emit = text.is(t, cmd, "emit");
    if !emit && !text.is(t, cmd, "check") {
        return usage(io);
    }
    if emit && arg_count(args) < 4 {
        return usage(io);
    }
    let file = arg(args, 2);
    // The source goes straight into the pool, so names are handles into it.
    let top = m[mem.text_slot()];
    let got = fs_read(fs, file, t[top..len(t)]);
    if got < 0 {
        io.error_all(io, "error[io]: cannot read ");
        io.error_all(io, file);
        io.error_all(io, "\n");
        return 1;
    }
    m[mem.text_slot()] = top + got;
    let src = text.handle(top, got);
    var run_it = false;
    var ai = 2;
    while ai < arg_count(args) {
        if text.is(t, text.from_bytes(m, t, arg(args, ai)), "--run") {
            run_it = true;
        }
        ai = ai + 1;
    }
    var first = 3;
    if emit {
        first = 4;
    }
    let vals = cli.constants(m, t, args, first);
    if vals < 0 {
        return cli.report(m, t, io);
    }
    let unit = parser.parse(m, t, src);
    if unit < 0 {
        return cli.report(m, t, io);
    }
    var dir = text.empty();
    if emit {
        dir = text.from_bytes(m, t, arg(args, 3));
    }
    var k = 0;
    var any = false;
    while k < 2 {
        var tg = target.nvidia_ada(m, t);
        var d = dialect.cuda();
        var ext = text.lit(m, t, "cu");
        if k == 1 {
            tg = target.apple_m_series(m, t);
            d = dialect.msl();
            ext = text.lit(m, t, "metal");
        }
        let r = one_target(m, t, fm, unit, tg, d, vals, emit, run_it);
        if r < 0 {
            return cli.report(m, t, io);
        }
        if r > 0 {
            any = true;
            io.write_all(io, text.bytes(t, m[r + 1]));
            if emit {
                let l = m[r];
                let path = text.f3(m, t, "$/$.$", dir, m[l], ext);
                var wrote = 0;
                region w {
                    // The path is copied out of the pool so that writing
                    // reads two different slices.
                    let p = alloc_slice[w](text.size(path), byte_of(0));
                    var j = 0;
                    while j < text.size(path) {
                        p[j] = byte_of(text.at(t, path, j));
                        j = j + 1;
                    }
                    wrote = fs_write(fs, p, text.bytes(t, m[l + 1]));
                }
                if wrote < 0 {
                    io.error_all(io, "error[io]: cannot write ");
                    io.error_all(io, text.bytes(t, path));
                    io.error_all(io, "\n");
                    return 1;
                }
                let line = text.fmt(m, t, "$: $ threads, grid $x$, $ B of tiles\n", mem.of5(m, path, text.num(m, t, m[l + 4]), text.num(m, t, m[l + 2]), text.num(m, t, m[l + 3]), text.num(m, t, m[l + 5])));
                io.write_all(io, text.bytes(t, line));
            } else {
                let line = text.f2(m, t, "$: $ ok\n", parser.algo_name(m, unit), target.name(m, tg));
                io.write_all(io, text.bytes(t, line));
            }
        }
        k = k + 1;
    }
    if !any {
        io.error_all(io, "error[no-schedule]: the file has no schedule for nvidia-ada or apple-m-series\n");
        return 1;
    }
    return 0;
}

fn main(world: World) -> [] int {
    let Split { io, ffi, fs, heap, args, net, clock } = split(world);
    release(ffi);
    release(net);
    release(clock);
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
                                status = run(m, t, fm, i, f, a);
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
    return status;
}
