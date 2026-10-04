edition 5;
module elab;

// A parsed algorithm, bound to one target: `Unit::compile`,
// `Algo::build_with` and the lowering under them in
// `lex_front::syntax`, ported. The constants come from the caller, the
// tile extents and the chunk from the file's schedule, and what comes
// back is the typed program.
//
// The IR has to be the *same* program the Rust builds, not an
// equivalent one -- variable numbering included, because the emitter
// names storage after it (`v12`). So the order of every `fresh` below
// follows the Rust's.
//
// Lookup tables are lists searched from the end, which is what a Rust
// `HashMap::insert` that replaces gives: the latest binding wins.

import mem;
import text;
import err;
import kinds;
import f32;
import ir;
import parser;

// --- A small association list: `[key, value]` records, newest last.

fn assoc_find[&m, &t](m: &!m [int], t: &!t [byte], l: int, key: int) -> [] int {
    var i = mem.size(m, l) - 1;
    while i >= 0 {
        let r = mem.get(m, l, i);
        if text.eq(t, m[r], key) {
            return r;
        }
        i = i - 1;
    }
    return 0 - 1;
}

fn var_space[&m](m: &!m [int], spaces: int, v: int) -> [] int {
    var i = mem.size(m, spaces) - 1;
    while i >= 0 {
        let r = mem.get(m, spaces, i);
        if m[r] == v {
            return m[r + 1];
        }
        i = i - 1;
    }
    return 0 - 1;
}

fn set_space[&m](m: &!m [int], cx: int, v: int, s: int) -> [] int {
    mem.push(m, m[cx + 2], mem.rec2(m, v, s));
    return 0;
}

// The context: `[vals, idx, spaces]` -- constants (name, f64 bits),
// grid and loop indices (name, var), and every tile's memory space.
fn cx_new[&m](m: &!m [int], vals: int, idx: int, spaces: int) -> [] int {
    return mem.rec3(m, vals, idx, spaces);
}

fn bad[&m, &t](m: &!m [int], t: &!t [byte], rule: &static [byte], msg: int) -> [] int {
    return err.fail(m, t, rule, msg);
}

// ---------------------------------------------------------------------
// Affine offsets
// ---------------------------------------------------------------------

fn scale_idx[&m](m: &!m [int], a: int, k: int) -> [] int {
    let terms = mem.list(m);
    var i = 0;
    while i < ir.idx_count(m, a) {
        mem.push(m, terms, ir.idx_var(m, a, i));
        mem.push(m, terms, ir.idx_coeff(m, a, i) * k);
        i = i + 1;
    }
    return mem.rec2(m, ir.idx_constant(m, a) * k, terms);
}

// `c + sum(coeff * index)`, which is all an offset can be.
fn affine[&m, &t](m: &!m [int], t: &!t [byte], e: int, cx: int) -> [] int {
    let k = m[e];
    if k == kinds.c_num() {
        let v = f32.from_bits(m[e + 1]);
        if !f32.is_whole(v) {
            return bad(m, t, "offset", text.f1(m, t, "$ is not an integer", f32.debug(m, t, v)));
        }
        return ir.idx_lit(m, truncate(v));
    }
    if k == kinds.c_name() {
        let s = m[e + 1];
        let ix = assoc_find(m, t, m[cx + 1], s);
        if ix >= 0 {
            return ir.idx_scaled(m, m[ix + 1], 1, 0);
        }
        let c = assoc_find(m, t, m[cx], s);
        if c >= 0 {
            let v = f32.from_bits(m[c + 1]);
            if f32.is_whole(v) {
                return ir.idx_lit(m, truncate(v));
            }
        }
        return bad(m, t, "offset", text.f1(m, t, "`$` is neither an index nor an integer constant", s));
    }
    let op = m[e + 1];
    let l = affine(m, t, m[e + 2], cx);
    if l < 0 {
        return 0 - 1;
    }
    let r = affine(m, t, m[e + 3], cx);
    if r < 0 {
        return 0 - 1;
    }
    if op == '+' || op == '-' {
        var sign = 1;
        if op == '-' {
            sign = 0 - 1;
        }
        let rs = scale_idx(m, r, sign);
        let terms = mem.copy(m, ir.idx_terms(m, l));
        var i = 0;
        while i < mem.size(m, ir.idx_terms(m, rs)) {
            mem.push(m, terms, mem.get(m, ir.idx_terms(m, rs), i));
            i = i + 1;
        }
        return mem.rec2(m, ir.idx_constant(m, l) + ir.idx_constant(m, rs), terms);
    }
    if op == '*' {
        if ir.idx_count(m, l) == 0 {
            return scale_idx(m, r, ir.idx_constant(m, l));
        }
        if ir.idx_count(m, r) == 0 {
            return scale_idx(m, l, ir.idx_constant(m, r));
        }
        return bad(m, t, "offset", text.lit(m, t, "a product of two indices is not an affine offset"));
    }
    if ir.idx_count(m, l) != 0 || ir.idx_count(m, r) != 0 {
        return bad(m, t, "offset", text.lit(m, t, "an offset divides constants only"));
    }
    let (a, b) = (ir.idx_constant(m, l), ir.idx_constant(m, r));
    if b == 0 || a % b != 0 {
        return bad(m, t, "offset", text.n2(m, t, "$ does not divide by $", a, b));
    }
    return ir.idx_lit(m, a / b);
}

// A non-negative constant: a loop bound or a grid extent.
fn constant[&m, &t](m: &!m [int], t: &!t [byte], e: int, cx: int) -> [] int {
    let a = affine(m, t, e, cx);
    if a < 0 {
        return 0 - 1;
    }
    if ir.idx_count(m, a) != 0 || ir.idx_constant(m, a) < 0 {
        return bad(m, t, "offset", text.lit(m, t, "an extent is a constant, not an index"));
    }
    return ir.idx_constant(m, a);
}

// A dimension, resolved against the constants alone.
fn dim_of[&m, &t](m: &!m [int], t: &!t [byte], d: int, vals: int) -> [] int {
    let k = m[d];
    if k == kinds.d_lit() {
        return f32.to_count(f32.from_bits(m[d + 1]));
    }
    if k == kinds.d_named() {
        let c = assoc_find(m, t, vals, m[d + 1]);
        if c < 0 {
            return bad(m, t, "unbound", text.f1(m, t, "`$` is not one of this algo's constants", m[d + 1]));
        }
        return f32.to_count(f32.from_bits(m[c + 1]));
    }
    let cx = cx_new(m, vals, mem.list(m), mem.list(m));
    return constant(m, t, m[d + 1], cx);
}

fn dims_of[&m, &t](m: &!m [int], t: &!t [byte], dims: int, vals: int) -> [] int {
    let out = mem.list(m);
    var i = 0;
    while i < mem.size(m, dims) {
        let d = dim_of(m, t, mem.get(m, dims, i), vals);
        if d < 0 {
            return 0 - 1;
        }
        mem.push(m, out, d);
        i = i + 1;
    }
    return out;
}

// A window of a parameter at affine offsets, `shape` big.
fn window[&m, &t](m: &!m [int], t: &!t [byte], param: int, at: int, shape: int, cx: int) -> [] int {
    if mem.size(m, at) != mem.size(m, shape) {
        return bad(m, t, "shape", text.n2(m, t, "$ offsets for a window of $ dimensions", mem.size(m, at), mem.size(m, shape)));
    }
    let offs = mem.list(m);
    var i = 0;
    while i < mem.size(m, at) {
        let a = affine(m, t, mem.get(m, at, i), cx);
        if a < 0 {
            return 0 - 1;
        }
        mem.push(m, offs, a);
        i = i + 1;
    }
    return ir.view(m, param, offs, mem.copy(m, shape));
}

fn whole[&m](m: &!m [int], param: int, shape: int) -> [] int {
    let offs = mem.list(m);
    var i = 0;
    while i < mem.size(m, shape) {
        mem.push(m, offs, ir.idx_lit(m, 0));
        i = i + 1;
    }
    return ir.view(m, param, offs, mem.copy(m, shape));
}

// What one grid instance sees of a parameter: the whole thing, or its
// own cut of the trailing dimension. `pid` is `[var, chunk]` or -1.
fn slice_view[&m](m: &!m [int], pid: int, param: int, shape: int) -> [] int {
    let v = whole(m, param, shape);
    if pid >= 0 {
        let offs = ir.view_offsets(m, v);
        mem.set(m, offs, mem.size(m, offs) - 1, ir.idx_scaled(m, m[pid], m[pid + 1], 0));
    }
    return v;
}

// ---------------------------------------------------------------------
// Constant substitution and folding
// ---------------------------------------------------------------------

fn num_node[&m](m: &!m [int], v: float) -> [] int {
    return parser.node(m, kinds.e_num(), bits_of(v), 0, 0, 0, 0);
}

fn is_num[&m](m: &!m [int], e: int) -> [] bool {
    return m[e] == kinds.e_num();
}

fn num_of[&m](m: &!m [int], e: int) -> [] float {
    return f32.from_bits(m[e + 1]);
}

// Replace a use of an algorithm constant with its value, and fold an
// operation on two numbers in f32 -- where the Rust computes it, and
// where "the same program" puts the same constant in the emitted text.
fn substitute[&m, &t](m: &!m [int], t: &!t [byte], e: int, vals: int) -> [] int {
    let k = m[e];
    if k == kinds.e_move() {
        let c = assoc_find(m, t, vals, m[e + 1]);
        if c >= 0 {
            return parser.node(m, kinds.e_num(), m[c + 1], 0, 0, 0, 0);
        }
        return e;
    }
    if k == kinds.e_call() {
        return parser.node(m, k, m[e + 1], substitute(m, t, m[e + 2], vals), 0, 0, 0);
    }
    if k == kinds.e_dequant_fp4() {
        let q = substitute(m, t, m[e + 1], vals);
        let s = substitute(m, t, m[e + 2], vals);
        let g = substitute(m, t, m[e + 3], vals);
        return parser.node(m, k, q, s, g, m[e + 4], 0);
    }
    if k == kinds.e_stage() {
        return parser.node(m, k, m[e + 1], substitute(m, t, m[e + 2], vals), 0, 0, 0);
    }
    if k == kinds.e_add_window() {
        return parser.node(m, k, substitute(m, t, m[e + 1], vals), m[e + 2], m[e + 3], 0, 0);
    }
    if k == kinds.e_mma() {
        let c = substitute(m, t, m[e + 1], vals);
        let a = substitute(m, t, m[e + 2], vals);
        let b = substitute(m, t, m[e + 3], vals);
        return parser.node(m, k, c, a, b, 0, 0);
    }
    if k == kinds.e_matmul() {
        let a = substitute(m, t, m[e + 2], vals);
        let b = substitute(m, t, m[e + 3], vals);
        return parser.node(m, k, m[e + 1], a, b, 0, 0);
    }
    if k == kinds.e_bin() {
        let op = m[e + 1];
        let l = substitute(m, t, m[e + 2], vals);
        let r = substitute(m, t, m[e + 3], vals);
        if is_num(m, l) && is_num(m, r) && op != kinds.max() {
            let a = f32.round(num_of(m, l));
            let b = f32.round(num_of(m, r));
            var v = 0.0;
            if op == kinds.add() {
                v = f32.add(a, b);
            } else if op == kinds.sub() {
                v = f32.sub(a, b);
            } else if op == kinds.mul() {
                v = f32.mul(a, b);
            } else {
                v = f32.div(a, b);
            }
            return num_node(m, v);
        }
        return parser.node(m, k, op, l, r, 0, 0);
    }
    return e;
}

// ---------------------------------------------------------------------
// Lowering a body
// ---------------------------------------------------------------------
//
// `st` is the elaboration's state: `[builder, params, pid]`, `params` a
// list of `[name, id, seen shape, dtype]` and `pid` `[var, chunk]` or -1.
// A lowered value is `[var, shape, dtype]`; `env` a list of
// `[name, var, shape, dtype]`.

fn value[&m](m: &!m [int], v: int, shape: int, dt: int) -> [] int {
    return mem.rec3(m, v, shape, dt);
}

fn find_param[&m, &t](m: &!m [int], t: &!t [byte], st: int, name: int) -> [] int {
    let r = assoc_find(m, t, m[st + 1], name);
    if r < 0 {
        return bad(m, t, "unbound", text.f1(m, t, "`$` is not a parameter", name));
    }
    return r;
}

// `&x` borrows; anything else moves.
fn arg_of[&m](m: &!m [int], e: int, v: int) -> [] int {
    if m[e] == kinds.e_borrow() {
        return ir.borrow_of(v);
    }
    return ir.mv(v);
}

fn sd[&m, &t](m: &!m [int], t: &!t [byte], s: int) -> [] int {
    return ir.shape_debug(m, t, s);
}

fn lower[&m, &t](m: &!m [int], t: &!t [byte], st: int, env: int, cx: int, e: int, name: int) -> [] int {
    let b = m[st];
    let k = m[e];
    let vals = m[cx];
    if k == kinds.e_load() {
        let p = find_param(m, t, st, m[e + 1]);
        if p < 0 {
            return 0 - 1;
        }
        let shape = m[p + 2];
        let dt = m[p + 3];
        let view = slice_view(m, m[st + 2], m[p + 1], shape);
        let o = ir.make_op(m, kinds.op_load(), view, ir.tile(m, dt, shape, kinds.reg()), 0, 0);
        let v = ir.emit(m, t, b, m[e + 1], o);
        return value(m, v, mem.copy(m, shape), dt);
    }
    if k == kinds.e_load_at() {
        let p = find_param(m, t, st, m[e + 1]);
        if p < 0 {
            return 0 - 1;
        }
        let dt = m[p + 3];
        let shape = dims_of(m, t, m[e + 3], vals);
        if shape < 0 {
            return 0 - 1;
        }
        let view = window(m, t, m[p + 1], m[e + 2], shape, cx);
        if view < 0 {
            return 0 - 1;
        }
        let sp = m[e + 4];
        let o = ir.make_op(m, kinds.op_load(), view, ir.tile(m, dt, shape, sp), 0, 0);
        let v = ir.emit(m, t, b, m[e + 1], o);
        set_space(m, cx, v, sp);
        return value(m, v, shape, dt);
    }
    if k == kinds.e_zeros() {
        let shape = dims_of(m, t, m[e + 2], vals);
        if shape < 0 {
            return 0 - 1;
        }
        let dt = m[e + 1];
        let sp = m[e + 3];
        let o = ir.make_op(m, kinds.op_fill(), ir.tile(m, dt, shape, sp), bits_of(0.0), 0, 0);
        let v = ir.emit(m, t, b, name, o);
        set_space(m, cx, v, sp);
        return value(m, v, shape, dt);
    }
    if k == kinds.e_dequant_fp4() {
        let q = lower(m, t, st, env, cx, m[e + 1], name);
        if q < 0 {
            return 0 - 1;
        }
        let s = lower(m, t, st, env, cx, m[e + 2], name);
        if s < 0 {
            return 0 - 1;
        }
        let g = lower(m, t, st, env, cx, m[e + 3], name);
        if g < 0 {
            return 0 - 1;
        }
        let group = m[e + 4];
        let (qs, ss, gs) = (m[q + 1], m[s + 1], m[g + 1]);
        if mem.size(m, qs) != 2 || group <= 0 {
            return bad(m, t, "shape", text.f2(m, t, "`$`: nvfp4 values are a 2-d tile, found $", name, sd(m, t, qs)));
        }
        let rows = mem.get(m, qs, 0);
        let cols = 2 * mem.get(m, qs, 1);
        let want_s = mem.of2(m, rows, cols / group);
        let want_g = mem.of1(m, rows);
        if !mem.same(m, ss, want_s) || !mem.same(m, gs, want_g) {
            let msg = text.f6(m, t, "`$`: $ values want scales $ and row scales $, found $ and $", name, sd(m, t, qs), sd(m, t, want_s), sd(m, t, want_g), sd(m, t, ss), sd(m, t, gs));
            return bad(m, t, "shape", msg);
        }
        let o = ir.make_op(m, kinds.op_dequant_fp4(), arg_of(m, m[e + 1], m[q]), arg_of(m, m[e + 2], m[s]), arg_of(m, m[e + 3], m[g]), group);
        let v = ir.emit(m, t, b, name, o);
        return value(m, v, mem.of2(m, rows, cols), kinds.f32());
    }
    if k == kinds.e_add_window() {
        let a = lower(m, t, st, env, cx, m[e + 1], name);
        if a < 0 {
            return 0 - 1;
        }
        let p = find_param(m, t, st, m[e + 2]);
        if p < 0 {
            return 0 - 1;
        }
        let view = window(m, t, m[p + 1], m[e + 3], m[a + 1], cx);
        if view < 0 {
            return 0 - 1;
        }
        let o = ir.make_op(m, kinds.op_add_window(), arg_of(m, m[e + 1], m[a]), view, 0, 0);
        let v = ir.emit(m, t, b, name, o);
        set_space(m, cx, v, kinds.frag());
        return value(m, v, m[a + 1], m[a + 2]);
    }
    if k == kinds.e_stage() {
        let a = lower(m, t, st, env, cx, m[e + 2], name);
        if a < 0 {
            return 0 - 1;
        }
        let dt = m[e + 1];
        let o = ir.make_op(m, kinds.op_stage(), arg_of(m, m[e + 2], m[a]), dt, 0, 0);
        let v = ir.emit(m, t, b, name, o);
        set_space(m, cx, v, kinds.threadgroup());
        return value(m, v, m[a + 1], dt);
    }
    if k == kinds.e_mma() {
        let c = lower(m, t, st, env, cx, m[e + 1], name);
        if c < 0 {
            return 0 - 1;
        }
        let l = lower(m, t, st, env, cx, m[e + 2], name);
        if l < 0 {
            return 0 - 1;
        }
        let r = lower(m, t, st, env, cx, m[e + 3], name);
        if r < 0 {
            return 0 - 1;
        }
        let (cs, ls, rs) = (m[c + 1], m[l + 1], m[r + 1]);
        if mem.size(m, cs) != 2 || mem.size(m, ls) != 2 || mem.size(m, rs) != 2 {
            return bad(m, t, "shape", text.f1(m, t, "`$`: mma takes 2-d tiles", name));
        }
        let ok = mem.get(m, cs, 0) == mem.get(m, ls, 0) && mem.get(m, cs, 1) == mem.get(m, rs, 0) && mem.get(m, ls, 1) == mem.get(m, rs, 1);
        if !ok {
            return bad(m, t, "shape", text.f4(m, t, "`$`: mma of $ x $^T into $", name, sd(m, t, ls), sd(m, t, rs), sd(m, t, cs)));
        }
        let o = ir.make_op(m, kinds.op_mma(), arg_of(m, m[e + 1], m[c]), arg_of(m, m[e + 2], m[l]), arg_of(m, m[e + 3], m[r]), 0);
        let v = ir.emit(m, t, b, name, o);
        set_space(m, cx, v, kinds.frag());
        return value(m, v, cs, m[c + 2]);
    }
    if k == kinds.e_matmul() {
        let nt = m[e + 1] == 1;
        let l = lower(m, t, st, env, cx, m[e + 2], name);
        if l < 0 {
            return 0 - 1;
        }
        let r = lower(m, t, st, env, cx, m[e + 3], name);
        if r < 0 {
            return 0 - 1;
        }
        let (ls, rs) = (m[l + 1], m[r + 1]);
        if mem.size(m, ls) != 2 || mem.size(m, rs) != 2 {
            return bad(m, t, "shape", text.f1(m, t, "`$`: a matmul takes 2-d tiles", name));
        }
        let (mm, kk) = (mem.get(m, ls, 0), mem.get(m, ls, 1));
        var kr = mem.get(m, rs, 0);
        var n = mem.get(m, rs, 1);
        if nt {
            kr = mem.get(m, rs, 1);
            n = mem.get(m, rs, 0);
        }
        if kk != kr {
            return bad(m, t, "shape", text.f3(m, t, "`$`: $ against $ do not share their inner dimension", name, sd(m, t, ls), sd(m, t, rs)));
        }
        var ok = kinds.op_matmul();
        if nt {
            ok = kinds.op_matmul_nt();
        }
        let o = ir.make_op(m, ok, arg_of(m, m[e + 2], m[l]), arg_of(m, m[e + 3], m[r]), kinds.f32(), 0);
        let v = ir.emit(m, t, b, name, o);
        return value(m, v, mem.of2(m, mm, n), kinds.f32());
    }
    if k == kinds.e_for() {
        return lower_for(m, t, st, env, cx, e);
    }
    if k == kinds.e_move() || k == kinds.e_borrow() {
        let r = assoc_find(m, t, env, m[e + 1]);
        if r < 0 {
            return bad(m, t, "unbound", text.f1(m, t, "`$` is not bound", m[e + 1]));
        }
        return value(m, m[r + 1], m[r + 2], m[r + 3]);
    }
    if k == kinds.e_num() {
        return bad(m, t, "shape", text.lit(m, t, "a constant needs something to take its shape from"));
    }
    if k == kinds.e_call() {
        let a = lower(m, t, st, env, cx, m[e + 2], name);
        if a < 0 {
            return 0 - 1;
        }
        let f = m[e + 1];
        let arg = arg_of(m, m[e + 2], m[a]);
        if text.is(t, f, "rowsum") || text.is(t, f, "rowmax") {
            var r = kinds.r_max();
            if text.is(t, f, "rowsum") {
                r = kinds.r_sum();
            }
            // A row reduction of `[rows, n]` gives `[rows]`.
            if mem.size(m, m[a + 1]) != 2 {
                return bad(m, t, "shape", text.f2(m, t, "`$`: a row reduction takes a 2-d tile, found $", name, sd(m, t, m[a + 1])));
            }
            let v = ir.emit(m, t, b, name, ir.make_op(m, kinds.op_row_reduce(), r, arg, 0, 0));
            return value(m, v, mem.of1(m, mem.get(m, m[a + 1], 0)), m[a + 2]);
        }
        var u = 0 - 1;
        if text.is(t, f, "rsqrt") {
            u = kinds.rsqrt();
        } else if text.is(t, f, "sigmoid") {
            u = kinds.sigmoid();
        } else if text.is(t, f, "softplus") {
            u = kinds.softplus();
        } else {
            return bad(m, t, "unbound", text.f1(m, t, "`$` is not an operation here", f));
        }
        let v = ir.emit(m, t, b, name, ir.make_op(m, kinds.op_unary(), u, arg, 0, 0));
        return value(m, v, m[a + 1], m[a + 2]);
    }
    return lower_bin(m, t, st, env, cx, e, name);
}

// A binary op. A literal on either side of a multiply is a `Scale`; on
// the right of anything else it is a `Fill` shaped like what it meets.
fn lower_bin[&m, &t](m: &!m [int], t: &!t [byte], st: int, env: int, cx: int, e: int, name: int) -> [] int {
    let b = m[st];
    let op = m[e + 1];
    let (le, re) = (m[e + 2], m[e + 3]);
    if op == kinds.mul() && is_num(m, re) {
        let l = lower(m, t, st, env, cx, le, name);
        if l < 0 {
            return 0 - 1;
        }
        let k = f32.round(num_of(m, re));
        let v = ir.emit(m, t, b, name, ir.make_op(m, kinds.op_scale(), arg_of(m, le, m[l]), bits_of(k), 0, 0));
        return value(m, v, m[l + 1], m[l + 2]);
    }
    if op == kinds.mul() && is_num(m, le) {
        let r = lower(m, t, st, env, cx, re, name);
        if r < 0 {
            return 0 - 1;
        }
        let k = f32.round(num_of(m, le));
        let v = ir.emit(m, t, b, name, ir.make_op(m, kinds.op_scale(), arg_of(m, re, m[r]), bits_of(k), 0, 0));
        return value(m, v, m[r + 1], m[r + 2]);
    }
    let l = lower(m, t, st, env, cx, le, name);
    if l < 0 {
        return 0 - 1;
    }
    let (ls, dt) = (m[l + 1], m[l + 2]);
    var r = 0;
    if is_num(m, re) {
        // Shaped like what it meets, so `ms + eps` fills a `[rows]` tile.
        let k = f32.round(num_of(m, re));
        let fill = ir.make_op(m, kinds.op_fill(), ir.tile(m, dt, ls, kinds.reg()), bits_of(k), 0, 0);
        let v = ir.emit_lit(m, t, b, "eps", fill);
        r = value(m, v, mem.copy(m, ls), dt);
    } else {
        r = lower(m, t, st, env, cx, re, name);
        if r < 0 {
            return 0 - 1;
        }
    }
    let rs = m[r + 1];
    // The checker's two broadcasts, and no others.
    let row_b = mem.size(m, ls) == 2 && mem.size(m, rs) == 1 && mem.get(m, rs, 0) == mem.get(m, ls, 0);
    let col_b = mem.size(m, ls) == 2 && mem.size(m, rs) == 2 && mem.get(m, rs, 0) == 1 && mem.get(m, rs, 1) == mem.get(m, ls, 1);
    if !mem.same(m, ls, rs) && !row_b && !col_b {
        return bad(m, t, "shape", text.f3(m, t, "`$`: $ against $, which do not broadcast", name, sd(m, t, ls), sd(m, t, rs)));
    }
    let o = ir.make_op(m, kinds.op_binary(), op, arg_of(m, le, m[l]), arg_of(m, re, m[r]), 0);
    let v = ir.emit(m, t, b, name, o);
    return value(m, v, ls, dt);
}

// `for i in a .. b with acc { ...; yield e }`.
fn lower_for[&m, &t](m: &!m [int], t: &!t [byte], st: int, env: int, cx: int, e: int) -> [] int {
    let b = m[st];
    let index = m[e + 1];
    let start = constant(m, t, m[e + 2], cx);
    if start < 0 {
        return 0 - 1;
    }
    let end = constant(m, t, m[e + 3], cx);
    if end < 0 {
        return 0 - 1;
    }
    let carry = m[e + 4];
    let init = mem.list(m);
    let tys = mem.list(m);
    let seen = mem.list(m);
    var i = 0;
    while i < mem.size(m, carry) {
        let n = mem.get(m, carry, i);
        let r = assoc_find(m, t, env, n);
        if r < 0 {
            return bad(m, t, "unbound", text.f1(m, t, "`$` is not bound", n));
        }
        let v = m[r + 1];
        mem.push(m, init, v);
        var sp = var_space(m, m[cx + 2], v);
        if sp < 0 {
            sp = kinds.reg();
        }
        mem.push(m, tys, ir.tile(m, m[r + 3], m[r + 2], sp));
        mem.push(m, seen, mem.rec3(m, m[r + 2], m[r + 3], sp));
        i = i + 1;
    }
    if mem.size(m, carry) != 1 {
        return bad(m, t, "carry", text.f2(m, t, "a loop carries one tile for now, `for $` carries $", index, text.num(m, t, mem.size(m, carry))));
    }
    let opened = ir.for_begin(m, t, b, tys);
    let params = m[opened + 1];
    let inner = mem.copy(m, env);
    i = 0;
    while i < mem.size(m, carry) {
        let p = mem.get(m, params, i);
        let s = mem.get(m, seen, i);
        mem.push(m, inner, mem.rec4(m, mem.get(m, carry, i), p, m[s], m[s + 1]));
        set_space(m, cx, p, m[s + 2]);
        i = i + 1;
    }
    let idx = mem.copy(m, m[cx + 1]);
    mem.push(m, idx, mem.rec2(m, index, m[opened]));
    let inside = cx_new(m, m[cx], idx, m[cx + 2]);
    let ys = lower_block(m, t, st, inner, inside, m[e + 5]);
    let yields = mem.list(m);
    var failed = ys < 0;
    if ys == 0 {
        failed = true;
        bad(m, t, "carry", text.f1(m, t, "the loop over `$` never yields", index));
    }
    if !failed {
        i = 0;
        while i < mem.size(m, ys) {
            mem.push(m, yields, m[mem.get(m, ys, i)]);
            i = i + 1;
        }
    }
    let outs = ir.for_end(m, t, b, opened, start, end, init, yields);
    if failed {
        return 0 - 1;
    }
    let out = mem.get(m, outs, 0);
    let s0 = mem.get(m, seen, 0);
    set_space(m, cx, out, m[s0 + 2]);
    return value(m, out, m[s0], m[s0 + 1]);
}

// Run statements in order. Answers the yielded values (a list of lowered
// values) if the block ended with a `yield`, 0 if it did not, or -1.
fn lower_block[&m, &t](m: &!m [int], t: &!t [byte], st: int, env: int, cx: int, stmts: int) -> [] int {
    let b = m[st];
    let vals = m[cx];
    var i = 0;
    while i < mem.size(m, stmts) {
        let s = mem.get(m, stmts, i);
        let k = m[s];
        if k == kinds.s_let() {
            let e = substitute(m, t, m[s + 2], vals);
            let v = lower(m, t, st, env, cx, e, m[s + 1]);
            if v < 0 {
                return 0 - 1;
            }
            mem.push(m, env, mem.rec4(m, m[s + 1], m[v], m[v + 1], m[v + 2]));
        } else if k == kinds.s_store() {
            let e = substitute(m, t, m[s + 1], vals);
            let v = lower(m, t, st, env, cx, e, text.lit(m, t, "y"));
            if v < 0 {
                return 0 - 1;
            }
            let p = find_param(m, t, st, m[s + 2]);
            if p < 0 {
                return 0 - 1;
            }
            var view = 0;
            if m[s + 3] >= 0 {
                view = window(m, t, m[p + 1], m[s + 3], m[v + 1], cx);
                if view < 0 {
                    return 0 - 1;
                }
            } else {
                view = slice_view(m, m[st + 2], m[p + 1], m[p + 2]);
            }
            ir.effect(m, b, ir.make_op(m, kinds.op_store(), ir.mv(m[v]), view, 0, 0));
        } else if k == kinds.s_grid() {
            return bad(m, t, "grid", text.lit(m, t, "`grid` goes first, outside any loop"));
        } else {
            let es = m[s + 1];
            let out = mem.list(m);
            var j = 0;
            while j < mem.size(m, es) {
                let e = substitute(m, t, mem.get(m, es, j), vals);
                let v = lower(m, t, st, env, cx, e, text.lit(m, t, "y"));
                if v < 0 {
                    return 0 - 1;
                }
                mem.push(m, out, v);
                j = j + 1;
            }
            return out;
        }
        i = i + 1;
    }
    return 0;
}

// ---------------------------------------------------------------------
// The entry points
// ---------------------------------------------------------------------

fn lookup_val[&m, &t](m: &!m [int], t: &!t [byte], vals: int, name: int) -> [] int {
    return assoc_find(m, t, vals, name);
}

// The algorithm with its constants bound. `vals` is a list of
// `[name, f64 bits]`, later entries winning; `chunk` is -1 for none.
pub fn build[&m, &t](m: &!m [int], t: &!t [byte], algo: int, vals: int, chunk: int) -> [] int {
    let name = m[algo];
    let consts = m[algo + 1];
    let tiles = m[algo + 2];
    let params = m[algo + 3];
    let body = m[algo + 4];
    let declared_consts = mem.copy(m, consts);
    var i = 0;
    while i < mem.size(m, tiles) {
        mem.push(m, declared_consts, mem.get(m, tiles, i));
        i = i + 1;
    }
    i = 0;
    while i < mem.size(m, declared_consts) {
        let c = mem.get(m, declared_consts, i);
        if lookup_val(m, t, vals, c) < 0 {
            return bad(m, t, "unbound", text.f2(m, t, "`$` needs a value for `$`", name, c));
        }
        i = i + 1;
    }
    // The suffix the Rust builders use: every constant in declaration
    // order. `rmsnorm(n)` with n = 4096 is `rmsnorm_4096`.
    var full = name;
    i = 0;
    while i < mem.size(m, declared_consts) {
        let r = lookup_val(m, t, vals, mem.get(m, declared_consts, i));
        let v = f32.from_bits(m[r + 1]);
        if f32.is_whole(v) && v >= 0.0 {
            full = text.f2(m, t, "$_$", full, f32.count_string(m, t, v));
        }
        i = i + 1;
    }
    let b = ir.builder(m, full);

    let full_shapes = mem.list(m);
    i = 0;
    while i < mem.size(m, params) {
        let s = dims_of(m, t, m[mem.get(m, params, i) + 2], vals);
        if s < 0 {
            return 0 - 1;
        }
        mem.push(m, full_shapes, s);
        i = i + 1;
    }
    var pid = 0 - 1;
    if chunk >= 0 {
        let n = mem.last(m, mem.get(m, full_shapes, 0));
        i = 0;
        while i < mem.size(m, full_shapes) {
            if mem.last(m, mem.get(m, full_shapes, i)) != n {
                let trailing = mem.list(m);
                var j = 0;
                while j < mem.size(m, full_shapes) {
                    mem.push(m, trailing, mem.last(m, mem.get(m, full_shapes, j)));
                    j = j + 1;
                }
                return bad(m, t, "chunk", text.f2(m, t, "`$`: a chunked algo needs one trailing dimension, not $", name, sd(m, t, trailing)));
            }
            i = i + 1;
        }
        if chunk == 0 || n % chunk != 0 {
            return bad(m, t, "chunk", text.f3(m, t, "`$`: chunk $ does not divide $", name, text.num(m, t, chunk), text.num(m, t, n)));
        }
        let pv = ir.grid(m, t, b, n / chunk);
        pid = mem.rec2(m, pv, chunk);
    }
    let pmap = mem.list(m);
    i = 0;
    while i < mem.size(m, params) {
        let prm = mem.get(m, params, i);
        let full_shape = mem.get(m, full_shapes, i);
        let id = ir.add_param(m, b, m[prm], m[prm + 1], full_shape, m[prm + 3] == 1);
        var seen = mem.copy(m, full_shape);
        if pid >= 0 {
            mem.set(m, seen, mem.size(m, seen) - 1, m[pid + 1]);
        }
        mem.push(m, pmap, mem.rec4(m, m[prm], id, seen, m[prm + 1]));
        i = i + 1;
    }
    let st = mem.rec3(m, b, pmap, pid);
    let env = mem.list(m);
    let idx = mem.list(m);
    let spaces = mem.list(m);
    i = 0;
    while i < mem.size(m, body) {
        let s = mem.get(m, body, i);
        if m[s] == kinds.s_grid() {
            let axes = m[s + 1];
            if mem.size(m, axes) > 2 {
                return bad(m, t, "grid", text.lit(m, t, "a grid has at most two axes"));
            }
            if ir.has_grid(m, b) {
                return bad(m, t, "grid", text.lit(m, t, "a grid is declared once, and not beside a schedule's `chunk`"));
            }
            var j = 0;
            while j < mem.size(m, axes) {
                let ax = mem.get(m, axes, j);
                let cx = cx_new(m, vals, idx, spaces);
                let n = constant(m, t, m[ax + 1], cx);
                if n < 0 {
                    return 0 - 1;
                }
                var v = 0;
                if j == 0 {
                    v = ir.grid(m, t, b, n);
                } else {
                    v = ir.grid2(m, t, b, n);
                }
                mem.push(m, idx, mem.rec2(m, m[ax], v));
                j = j + 1;
            }
        } else {
            let cx = cx_new(m, vals, idx, spaces);
            let r = lower_block(m, t, st, env, cx, mem.of1(m, s));
            if r < 0 {
                return 0 - 1;
            }
            if r > 0 {
                return bad(m, t, "carry", text.lit(m, t, "`yield` outside a loop"));
            }
        }
        i = i + 1;
    }
    return ir.finish(m, b);
}

// `Unit::compile`: the schedule for `target`, its extents added to the
// caller's constants. Answers `[program, schedule]`, or -1.
pub fn compile[&m, &t](m: &!m [int], t: &!t [byte], unit: int, target: int, args: int) -> [] int {
    let s = parser.schedule_for(m, t, unit, target);
    if s < 0 {
        return 0 - 1;
    }
    let vals = mem.copy(m, args);
    let ext = m[s + 6];
    var i = 0;
    while i < mem.size(m, ext) {
        let x = mem.get(m, ext, i);
        mem.push(m, vals, mem.rec2(m, m[x], bits_of(float_of(m[x + 1]))));
        i = i + 1;
    }
    let p = build(m, t, m[unit], vals, m[s + 2]);
    if p < 0 {
        return 0 - 1;
    }
    return mem.rec2(m, p, s);
}
