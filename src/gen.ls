edition 5;
module gen;

// The lowering's state and the helpers every op uses: `lex_msl::program::Gen`
// and its small methods, ported. The ops themselves are in `lower.ls`
// and the matrix products in `matmul.ls`.
//
// A generator is a record (the slots are named below) with four tables
// indexed by variable: where each value lives (`locs`), its lazy
// dequantisation (`dqs`), the padding of a shared tile only a matrix op
// reads (`padded`), and the parameter window of a lazy load (`views`).
//
// A location is `[kind, text, ty, geom]`:
// * `reg`   -- `text[k]` is this thread's k-th element of a distributed
//              register tile;
// * `tg`    -- `text[e]` is flat element `e` of a threadgroup tile;
// * `index` -- `text` names a loop or grid index;
// * `frag`  -- `text[i][j]` is this warp's fragment (i, j); `geom`
//              `[wm, wn, fm, fnn, atom]` says how the warps split it;
// * `lazy`  -- `text` is an expression for element `@I@`, evaluated
//              where it is read.

import mem;
import text;
import err;
import kinds;
import f32;
import ir;
import dialect;

pub fn l_reg() -> [] int {
    return 1;
}

pub fn l_tg() -> [] int {
    return 3;
}

pub fn l_index() -> [] int {
    return 4;
}

pub fn l_frag() -> [] int {
    return 5;
}

pub fn l_lazy() -> [] int {
    return 6;
}

// Access kinds: how an op reads one operand.
pub fn a_local() -> [] int {
    return 0;
}

pub fn a_shared() -> [] int {
    return 1;
}

pub fn a_lazy() -> [] int {
    return 2;
}

// --- The record

pub fn prog[&m](m: &!m [int], g: int) -> [] int {
    return m[g];
}

pub fn dia[&m](m: &!m [int], g: int) -> [] int {
    return m[g + 1];
}

pub fn threads[&m](m: &!m [int], g: int) -> [] int {
    return m[g + 2];
}

pub fn body[&m](m: &!m [int], g: int) -> [] int {
    return m[g + 3];
}

pub fn arena[&m](m: &!m [int], g: int) -> [] int {
    return m[g + 6];
}

pub fn scratch[&m](m: &!m [int], g: int) -> [] int {
    return m[g + 7];
}

pub fn staged[&m](m: &!m [int], g: int) -> [] int {
    return m[g + 8];
}

pub fn barriers[&m](m: &!m [int], g: int) -> [] int {
    return m[g + 9];
}

pub fn fp4[&m](m: &!m [int], g: int) -> [] bool {
    return m[g + 11] == 1;
}

pub fn warps_rows[&m](m: &!m [int], g: int) -> [] int {
    return m[g + 12];
}

pub fn warps_cols[&m](m: &!m [int], g: int) -> [] int {
    return m[g + 13];
}

pub fn tile_align[&m](m: &!m [int], g: int) -> [] int {
    return m[g + 14];
}

pub fn uses_mma[&m](m: &!m [int], g: int) -> [] bool {
    return m[g + 15] == 1;
}

pub fn tgt[&m](m: &!m [int], g: int) -> [] int {
    return m[g + 18];
}

pub fn set_fp4[&m](m: &!m [int], g: int) -> [] int {
    m[g + 11] = 1;
    return 0;
}

pub fn depth_in[&m](m: &!m [int], g: int) -> [] int {
    m[g + 4] = m[g + 4] + 1;
    return 0;
}

pub fn depth_out[&m](m: &!m [int], g: int) -> [] int {
    m[g + 4] = m[g + 4] - 1;
    return 0;
}

pub fn set_staged[&m](m: &!m [int], g: int, n: int) -> [] int {
    m[g + 8] = n;
    return 0;
}

// `scratch = max(scratch, n)`
pub fn need_scratch[&m](m: &!m [int], g: int, n: int) -> [] int {
    if n > m[g + 7] {
        m[g + 7] = n;
    }
    return 0;
}

pub fn new_gen[&m](m: &!m [int], p: int, d: int, threads: int, wr: int, wc: int, mma: bool, tg: int) -> [] int {
    let n = ir.vars(m, p) + 1;
    let g = mem.grab(m, 19);
    m[g] = p;
    m[g + 1] = d;
    m[g + 2] = threads;
    m[g + 3] = mem.list(m);
    m[g + 4] = 1;
    m[g + 5] = mem.grab(m, n);
    m[g + 6] = 0;
    m[g + 7] = 0;
    m[g + 8] = 0;
    m[g + 9] = 0;
    m[g + 10] = mem.grab(m, n);
    m[g + 11] = 0;
    m[g + 12] = wr;
    m[g + 13] = wc;
    if mma {
        m[g + 14] = 32;
        m[g + 15] = 1;
    } else {
        m[g + 14] = 16;
        m[g + 15] = 0;
    }
    m[g + 16] = mem.grab(m, n);
    m[g + 17] = mem.grab(m, n);
    m[g + 18] = tg;
    return g;
}

// --- Per-variable tables

pub fn set_loc[&m](m: &!m [int], g: int, v: int, l: int) -> [] int {
    m[m[g + 5] + v] = l;
    return 0;
}

// The location of `v`, or 0 if it has none.
pub fn loc_or_none[&m](m: &!m [int], g: int, v: int) -> [] int {
    return m[m[g + 5] + v];
}

pub fn loc[&m, &t](m: &!m [int], t: &!t [byte], g: int, v: int) -> [] int {
    let l = m[m[g + 5] + v];
    if l == 0 {
        return err.fail(m, t, "lowering", text.f1(m, t, "`$` has no storage", ir.name_of(m, m[g], v)));
    }
    return l;
}

pub fn mk_loc[&m](m: &!m [int], kind: int, s: int, ty: int, geom: int) -> [] int {
    return mem.rec4(m, kind, s, ty, geom);
}

pub fn set_dq[&m](m: &!m [int], g: int, v: int, dq: int) -> [] int {
    m[m[g + 10] + v] = dq;
    return 0;
}

pub fn dq_of[&m](m: &!m [int], g: int, v: int) -> [] int {
    return m[m[g + 10] + v];
}

pub fn set_padded[&m](m: &!m [int], g: int, v: int, pad: int) -> [] int {
    m[m[g + 16] + v] = pad;
    return 0;
}

pub fn padded[&m](m: &!m [int], g: int, v: int) -> [] int {
    return m[m[g + 16] + v];
}

pub fn set_view[&m](m: &!m [int], g: int, v: int, vw: int) -> [] int {
    m[m[g + 17] + v] = vw;
    return 0;
}

pub fn view_of[&m](m: &!m [int], g: int, v: int) -> [] int {
    return m[m[g + 17] + v];
}

// --- Text

pub fn line[&m, &t](m: &!m [int], t: &!t [byte], g: int, s: int) -> [] int {
    let parts = mem.list(m);
    var i = 0;
    while i < m[g + 4] {
        mem.push(m, parts, text.lit(m, t, "    "));
        i = i + 1;
    }
    mem.push(m, parts, s);
    mem.push(m, parts, text.lit(m, t, "\n"));
    mem.push(m, m[g + 3], text.joined(m, t, parts, ""));
    return 0;
}

pub fn line_lit[&m, &t](m: &!m [int], t: &!t [byte], g: int, s: &static [byte]) -> [] int {
    return line(m, t, g, text.lit(m, t, s));
}

pub fn emit_lines[&m, &t](m: &!m [int], t: &!t [byte], g: int, ls: int) -> [] int {
    var i = 0;
    while i < mem.size(m, ls) {
        line(m, t, g, mem.get(m, ls, i));
        i = i + 1;
    }
    return 0;
}

pub fn barrier[&m, &t](m: &!m [int], t: &!t [byte], g: int) -> [] int {
    m[g + 9] = m[g + 9] + 1;
    return line(m, t, g, dialect.barrier(m, t, m[g + 1]));
}

pub fn per[&m](m: &!m [int], g: int, n: int) -> [] int {
    let th = m[g + 2];
    return (n + th - 1) / th;
}

pub fn num[&m, &t](m: &!m [int], t: &!t [byte], n: int) -> [] int {
    return text.num(m, t, n);
}

// An f32 constant as C writes it: `{:?}f`, or the infinities by name.
pub fn lit_f32[&m, &t](m: &!m [int], t: &!t [byte], x: float) -> [] int {
    if x == f32.infinity() {
        return text.lit(m, t, "INFINITY");
    }
    if x == -f32.infinity() {
        return text.lit(m, t, "(-INFINITY)");
    }
    return text.cat(m, t, f32.debug(m, t, x), text.lit(m, t, "f"));
}

// `template` with every `@I@` replaced by `(idx)`.
pub fn at_index[&m, &t](m: &!m [int], t: &!t [byte], template: int, idx: int) -> [] int {
    return text.replace_lit(m, t, template, "@I@", text.f1(m, t, "($)", idx));
}

pub fn at_index_lit[&m, &t](m: &!m [int], t: &!t [byte], template: int, idx: &static [byte]) -> [] int {
    return at_index(m, t, template, text.lit(m, t, idx));
}

fn is_ident_byte(c: int) -> [] bool {
    return c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c == '_';
}

// Whether the bracket opened at `open` is closed by the last byte of `s`
// and by nothing earlier.
fn closes_at_end[&t](t: &!t [byte], s: int, open: int, o: int, c: int) -> [] bool {
    var depth = 0;
    var i = open;
    while i < text.size(s) {
        let ch = text.at(t, s, i);
        if ch == o {
            depth = depth + 1;
        } else if ch == c {
            depth = depth - 1;
            if depth == 0 {
                return i == text.size(s) - 1;
            }
        }
        i = i + 1;
    }
    return false;
}

fn trim[&t](t: &!t [byte], s: int) -> [] int {
    var a = 0;
    var b = text.size(s);
    while a < b && (text.at(t, s, a) == ' ' || text.at(t, s, a) == '\t' || text.at(t, s, a) == '\n') {
        a = a + 1;
    }
    while b > a && (text.at(t, s, b - 1) == ' ' || text.at(t, s, b - 1) == '\t' || text.at(t, s, b - 1) == '\n') {
        b = b - 1;
    }
    return text.sub(s, a, b);
}

// A lazy read's element as an lvalue template, `pN_name[..@I@..]`, when
// the read is one parameter element under nothing but conversions; -1
// for anything computed.
pub fn lvalue[&t](t: &!t [byte], template: int) -> [] int {
    var s = trim(t, template);
    var peeling = true;
    while peeling {
        let open = text.index_of(t, s, '(', 0);
        if open < 0 {
            peeling = false;
        } else {
            var ident = open > 0;
            var i = 0;
            while i < open {
                if !is_ident_byte(text.at(t, s, i)) {
                    ident = false;
                }
                i = i + 1;
            }
            if !ident || !closes_at_end(t, s, open, '(', ')') {
                peeling = false;
            } else {
                s = text.sub(s, open + 1, text.size(s) - 1);
            }
        }
    }
    let open = text.index_of(t, s, '[', 0);
    if open < 0 || open < 1 || text.at(t, s, 0) != 'p' {
        return 0 - 1;
    }
    let us = text.index_of(t, s, '_', 1);
    if us < 0 || us > open {
        return 0 - 1;
    }
    if us == 1 {
        return 0 - 1;
    }
    var i = 1;
    while i < us {
        let c = text.at(t, s, i);
        if c < '0' || c > '9' {
            return 0 - 1;
        }
        i = i + 1;
    }
    if !closes_at_end(t, s, open, '[', ']') {
        return 0 - 1;
    }
    return s;
}

// A name as a C identifier: anything that is not ASCII alphanumeric
// becomes `_`.
pub fn ident[&m, &t](m: &!m [int], t: &!t [byte], name: int) -> [] int {
    let parts = mem.list(m);
    var i = 0;
    while i < text.size(name) {
        if is_ident_byte(text.at(t, name, i)) {
            mem.push(m, parts, text.sub(name, i, i + 1));
        } else {
            mem.push(m, parts, text.lit(m, t, "_"));
        }
        i = i + 1;
    }
    return text.joined(m, t, parts, "");
}

pub fn param_ident[&m, &t](m: &!m [int], t: &!t [byte], i: int, name: int) -> [] int {
    return text.f2(m, t, "p$_$", text.num(m, t, i), ident(m, t, name));
}

pub fn vname[&m, &t](m: &!m [int], t: &!t [byte], v: int) -> [] int {
    return text.n1(m, t, "v$", v);
}

// --- Indices and addresses

// `(uint)(c + k * name + ...)`
pub fn idx[&m, &t](m: &!m [int], t: &!t [byte], g: int, e: int) -> [] int {
    let parts = mem.of1(m, text.num(m, t, ir.idx_constant(m, e)));
    var i = 0;
    while i < ir.idx_count(m, e) {
        let l = loc(m, t, g, ir.idx_var(m, e, i));
        if l < 0 {
            return 0 - 1;
        }
        if m[l] != l_index() {
            return err.fail(m, t, "lowering", text.lit(m, t, "index expression over a non-index"));
        }
        mem.push(m, parts, text.f2(m, t, "$ * $", text.num(m, t, ir.idx_coeff(m, e, i)), m[l + 1]));
        i = i + 1;
    }
    return text.f1(m, t, "(uint)($)", text.joined(m, t, parts, " + "));
}

// Device address of flat element `e` of `view`.
pub fn addr[&m, &t](m: &!m [int], t: &!t [byte], g: int, vw: int, e: int) -> [] int {
    let p = ir.param(m, m[g], ir.view_param(m, vw));
    let dims = ir.param_shape(m, p);
    let r = mem.size(m, dims);
    let vs = ir.view_shape(m, vw);
    let offs = ir.view_offsets(m, vw);
    let pid = param_ident(m, t, ir.view_param(m, vw), ir.param_name(m, p));
    // A view spanning every inner dimension is one contiguous run.
    var contiguous = true;
    var dd = 1;
    while dd < r {
        if mem.get(m, vs, dd) != mem.get(m, dims, dd) {
            contiguous = false;
        }
        dd = dd + 1;
    }
    if contiguous {
        let base = mem.list(m);
        var i = 0;
        while i < r {
            let stride = mem.product_from(m, dims, i + 1);
            let x = idx(m, t, g, mem.get(m, offs, i));
            if x < 0 {
                return 0 - 1;
            }
            mem.push(m, base, text.f2(m, t, "$ * $u", x, text.num(m, t, stride)));
            i = i + 1;
        }
        return text.f3(m, t, "$[$ + ($)]", pid, text.joined(m, t, base, " + "), e);
    }
    let terms = mem.list(m);
    var i = 0;
    while i < r {
        let inner = mem.product_from(m, vs, i + 1);
        let stride = mem.product_from(m, dims, i + 1);
        let coord = text.f3(m, t, "(($) / $u % $u)", e, text.num(m, t, inner), text.num(m, t, mem.get(m, vs, i)));
        let x = idx(m, t, g, mem.get(m, offs, i));
        if x < 0 {
            return 0 - 1;
        }
        mem.push(m, terms, text.f3(m, t, "($ + $) * $u", x, coord, text.num(m, t, stride)));
        i = i + 1;
    }
    return text.f2(m, t, "$[$]", pid, text.joined(m, t, terms, " + "));
}

pub fn addr_lit[&m, &t](m: &!m [int], t: &!t [byte], g: int, vw: int, e: &static [byte]) -> [] int {
    return addr(m, t, g, vw, text.lit(m, t, e));
}

// --- Loops over elements

// `for each element this thread owns: body`.
pub fn owned[&m, &t](m: &!m [int], t: &!t [byte], g: int, n: int, body_lines: int) -> [] int {
    line(m, t, g, text.n1(m, t, "for (uint k = 0; k < $u; ++k) {", per(m, g, n)));
    depth_in(m, g);
    line(m, t, g, text.n1(m, t, "const uint e = k * $u + tid;", m[g + 2]));
    line(m, t, g, text.n1(m, t, "if (e < $u) {", n));
    depth_in(m, g);
    emit_lines(m, t, g, body_lines);
    depth_out(m, g);
    line_lit(m, t, g, "}");
    depth_out(m, g);
    return line_lit(m, t, g, "}");
}

pub fn owned1[&m, &t](m: &!m [int], t: &!t [byte], g: int, n: int, s: int) -> [] int {
    return owned(m, t, g, n, mem.of1(m, s));
}

// `for every element, cooperatively: body`.
pub fn every[&m, &t](m: &!m [int], t: &!t [byte], g: int, n: int, s: int) -> [] int {
    line(m, t, g, text.n2(m, t, "for (uint e = tid; e < $u; e += $u) {", n, m[g + 2]));
    depth_in(m, g);
    line(m, t, g, s);
    depth_out(m, g);
    return line_lit(m, t, g, "}");
}

// --- Storage

pub fn declare_reg[&m, &t](m: &!m [int], t: &!t [byte], g: int, x: int, ty: int) -> [] int {
    let name = vname(m, t, x);
    let n = per(m, g, ir.elems(m, ty));
    line(m, t, g, text.f3(m, t, "$ $[$];", dialect.scalar(m, t, m[g + 1], ir.dtype(m, ty)), name, text.num(m, t, n)));
    set_loc(m, g, x, mk_loc(m, l_reg(), name, ty, 0));
    return name;
}

// The row length of a shared tile in memory: its columns, and the
// padding if it has any.
pub fn row_len[&m](m: &!m [int], g: int, x: int, cols: int) -> [] int {
    return cols + padded(m, g, x);
}

fn next_multiple(x: int, k: int) -> [] int {
    return (x + k - 1) / k * k;
}

pub fn declare_tg[&m, &t](m: &!m [int], t: &!t [byte], g: int, x: int, ty: int) -> [] int {
    let name = vname(m, t, x);
    let off = next_multiple(m[g + 6], m[g + 14]);
    let shape = ir.shape(m, ty);
    var bytes = ir.bytes(m, ty);
    if mem.size(m, shape) == 2 {
        bytes = mem.get(m, shape, 0) * row_len(m, g, x, mem.get(m, shape, 1)) * kinds.size_bytes(ir.dtype(m, ty));
    }
    m[g + 6] = off + bytes;
    let p = dialect.shared_ptr(m, t, m[g + 1], dialect.scalar(m, t, m[g + 1], ir.dtype(m, ty)));
    line(m, t, g, text.f4(m, t, "$ $ = ($)(arena + $);", p, name, p, text.num(m, t, off)));
    set_loc(m, g, x, mk_loc(m, l_tg(), name, ty, 0));
    return name;
}

// How the schedule's warp grid cuts a fragment tile: a geom, or -1.
pub fn frag_geom[&m, &t](m: &!m [int], t: &!t [byte], g: int, ty: int) -> [] int {
    let (wm, wn) = (m[g + 12], m[g + 13]);
    if wm < 0 {
        return err.fail(m, t, "lowering", text.lit(m, t, "a program with fragment tiles needs a schedule with `warps`"));
    }
    let a = dialect.atom(m[g + 1]);
    let shape = ir.shape(m, ty);
    if mem.size(m, shape) != 2 {
        return err.fail(m, t, "lowering", text.lit(m, t, "a fragment tile is 2-d"));
    }
    let (rows, cols) = (mem.get(m, shape, 0), mem.get(m, shape, 1));
    if rows % (wm * a) != 0 || cols % (wn * a) != 0 {
        let l = mem.list(m);
        mem.push(m, l, text.num(m, t, rows));
        mem.push(m, l, text.num(m, t, cols));
        mem.push(m, l, text.num(m, t, wm));
        mem.push(m, l, text.num(m, t, wn));
        mem.push(m, l, text.num(m, t, a));
        mem.push(m, l, text.num(m, t, a));
        return err.fail(m, t, "lowering", text.fmt(m, t, "a $x$ fragment tile does not split into $x$ warps of whole $x$ atoms", l));
    }
    return mem.rec5(m, wm, wn, rows / (wm * a), cols / (wn * a), a);
}

pub fn declare_frag[&m, &t](m: &!m [int], t: &!t [byte], g: int, x: int, ty: int, geom: int) -> [] int {
    let name = vname(m, t, x);
    line(m, t, g, dialect.frag_decl(m, t, m[g + 1], name, m[geom + 2], m[geom + 3]));
    set_loc(m, g, x, mk_loc(m, l_frag(), name, ty, geom));
    return name;
}

// Where fragment (i, j) of this warp's share of a tile sits in the
// window `vw`: answers `[address, row length]`, or -1.
pub fn frag_window[&m, &t](m: &!m [int], t: &!t [byte], g: int, ty: int, geom: int, vw: int) -> [] int {
    let p = ir.param(m, m[g], ir.view_param(m, vw));
    let dims = ir.param_shape(m, p);
    let ldm = mem.last(m, dims);
    let atom = m[geom + 4];
    var ok = ldm % dialect.store_align(m[g + 1]) == 0;
    let vs = ir.view_shape(m, vw);
    var i = 0;
    while i < mem.size(m, vs) {
        if mem.get(m, vs, i) % atom != 0 {
            ok = false;
        }
        i = i + 1;
    }
    if !ok {
        return err.fail(m, t, "lowering", text.n2(m, t, "a fragment access needs rows of a multiple of $ elements, found $", dialect.store_align(m[g + 1]), ldm));
    }
    let shape = ir.shape(m, ty);
    let rows = mem.get(m, shape, 0) / m[geom];
    let cols = mem.get(m, shape, 1) / m[geom + 1];
    let n = mem.get(m, shape, 1);
    let l = mem.list(m);
    mem.push(m, l, num(m, t, 32 * m[geom + 1]));
    mem.push(m, l, num(m, t, rows));
    mem.push(m, l, num(m, t, atom));
    mem.push(m, l, num(m, t, n));
    mem.push(m, l, num(m, t, m[geom + 1]));
    mem.push(m, l, num(m, t, cols));
    mem.push(m, l, num(m, t, atom));
    let at = text.fmt(m, t, "((tid / $u) * $u + i * $u) * $u + (tid / 32u % $u) * $u + j * $u", l);
    let a = addr(m, t, g, vw, at);
    if a < 0 {
        return 0 - 1;
    }
    return mem.rec2(m, text.f1(m, t, "&$", a), ldm);
}

// --- Operands

// The type of an operand.
pub fn arg_ty[&m, &t](m: &!m [int], t: &!t [byte], g: int, a: int) -> [] int {
    let l = loc(m, t, g, ir.var_of(a));
    if l < 0 {
        return 0 - 1;
    }
    if m[l] == l_index() {
        return err.fail(m, t, "lowering", text.lit(m, t, "operand is not a tile"));
    }
    return m[l + 2];
}

// The location of a lazy operand, or -1 if it is not one.
pub fn lazy[&m](m: &!m [int], g: int, a: int) -> [] int {
    let l = loc_or_none(m, g, ir.var_of(a));
    if l != 0 && m[l] == l_lazy() {
        return l;
    }
    return 0 - 1;
}

pub fn access[&m](m: &!m [int], kind: int, s: int, ty: int) -> [] int {
    return mem.rec3(m, kind, s, ty);
}

// Read an operand at a flat element index.
pub fn read[&m, &t](m: &!m [int], t: &!t [byte], a: int, at: int) -> [] int {
    let k = m[a];
    if k == a_local() {
        return text.f1(m, t, "float($[k])", m[a + 1]);
    }
    if k == a_shared() {
        return text.f2(m, t, "float($[$])", m[a + 1], at);
    }
    return text.f1(m, t, "float($)", at_index(m, t, m[a + 1], at));
}

pub fn read_lit[&m, &t](m: &!m [int], t: &!t [byte], a: int, at: &static [byte]) -> [] int {
    return read(m, t, a, text.lit(m, t, at));
}

// Resolve operands for an op whose output has `out_elems` elements.
// `local` says, per operand, whether it is read at the output's own
// element index. Register operands that are not local are staged into
// scratch. Answers a list of accesses, or -1.
pub fn operands[&m, &t](m: &!m [int], t: &!t [byte], g: int, args: int, local: int, out_elems: int) -> [] int {
    let out = mem.list(m);
    let stage = mem.list(m);
    var off = 0;
    var i = 0;
    while i < mem.size(m, args) {
        let a = mem.get(m, args, i);
        let l = loc(m, t, g, ir.var_of(a));
        if l < 0 {
            return 0 - 1;
        }
        let k = m[l];
        let ty = m[l + 2];
        if k == l_lazy() {
            mem.push(m, out, access(m, a_lazy(), m[l + 1], ty));
        } else if k == l_tg() {
            mem.push(m, out, access(m, a_shared(), m[l + 1], ty));
        } else if k == l_reg() {
            if mem.get(m, local, i) == 1 && ir.elems(m, ty) == out_elems {
                mem.push(m, out, access(m, a_local(), m[l + 1], ty));
            } else {
                let base = text.n1(m, t, "(scratch + $)", off);
                mem.push(m, stage, mem.rec3(m, m[l + 1], ir.elems(m, ty), off));
                off = off + ir.elems(m, ty);
                mem.push(m, out, access(m, a_shared(), base, ty));
            }
        } else {
            return err.fail(m, t, "lowering", text.lit(m, t, "operand is not a tile"));
        }
        i = i + 1;
    }
    m[g + 8] = off;
    if mem.size(m, stage) > 0 {
        need_scratch(m, g, off);
        // Nobody may still be reading the scratch from the previous op.
        barrier(m, t, g);
        i = 0;
        while i < mem.size(m, stage) {
            let s = mem.get(m, stage, i);
            owned1(m, t, g, m[s + 1], text.f2(m, t, "scratch[$ + e] = float($[k]);", text.num(m, t, m[s + 2]), m[s]));
            i = i + 1;
        }
        barrier(m, t, g);
    }
    return out;
}

pub fn flags1[&m](m: &!m [int], a: bool) -> [] int {
    var x = 0;
    if a {
        x = 1;
    }
    return mem.of1(m, x);
}

pub fn flags2[&m](m: &!m [int], a: bool, b: bool) -> [] int {
    let l = flags1(m, a);
    var y = 0;
    if b {
        y = 1;
    }
    mem.push(m, l, y);
    return l;
}
