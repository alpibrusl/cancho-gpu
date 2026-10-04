edition 5;
module check;

// The type checker: `lex_front::check`, for the programs a `.lx` file
// can produce. Linearity (every tile consumed exactly once), borrows
// against moves within one op, region-scoped consumption in loop
// bodies, loop-invariant carries, shapes, implicit narrowing, bounds of
// every view over every iteration, and the peak threadgroup footprint
// against the target.
//
// What is not here is what no surface program reaches: futures, arrays,
// pipes and roles. A `.lx` file has no syntax that makes one.
//
// Diagnostics are `[kind, message]` with the Rust `Kind` as the rule
// tag in kebab case (`use-after-move`, `leak`, `consume-outer`, ...).

import mem;
import text;
import err;
import kinds;
import ir;
import target;

// The checker: `[prog, target, slots, ranges, depth, live_tg, peak,
// diags, poisoned]`. Per variable, `slots` holds `[ty, live, depth,
// present]` and `ranges` `[some, lo, hi]`.
fn new_checker[&m](m: &!m [int], prog: int, tg: int) -> [] int {
    let n = ir.vars(m, prog);
    let c = mem.grab(m, 9);
    m[c] = prog;
    m[c + 1] = tg;
    m[c + 2] = mem.grab(m, 4 * n + 4);
    m[c + 3] = mem.grab(m, 3 * n + 3);
    m[c + 4] = 0;
    m[c + 5] = 0;
    m[c + 6] = 0;
    m[c + 7] = mem.list(m);
    m[c + 8] = mem.grab(m, n + 1);
    return c;
}

fn slot[&m](m: &!m [int], c: int, v: int) -> [] int {
    return m[c + 2] + 4 * v;
}

fn rng[&m](m: &!m [int], c: int, v: int) -> [] int {
    return m[c + 3] + 3 * v;
}

fn poisoned[&m](m: &!m [int], c: int, v: int) -> [] bool {
    return m[m[c + 8] + v] == 1;
}

fn diag[&m, &t](m: &!m [int], t: &!t [byte], c: int, kind: &static [byte], msg: int) -> [] int {
    mem.push(m, m[c + 7], mem.rec2(m, text.lit(m, t, kind), msg));
    return 0 - 1;
}

fn name[&m](m: &!m [int], c: int, v: int) -> [] int {
    return ir.name_of(m, m[c], v);
}

fn bump[&m](m: &!m [int], c: int, bytes: int) -> [] int {
    m[c + 5] = m[c + 5] + bytes;
    if m[c + 5] > m[c + 6] {
        m[c + 6] = m[c + 5];
    }
    return 0;
}

fn define[&m](m: &!m [int], c: int, v: int, ty: int, scope: int) -> [] int {
    bump(m, c, ir.threadgroup_bytes(m, ty));
    let s = slot(m, c, v);
    m[s] = ty;
    m[s + 1] = 1;
    m[s + 2] = m[c + 4];
    m[s + 3] = 1;
    mem.push(m, scope, v);
    return 0;
}

// The slot of `v`, or -1 (reported, unless `v` is poisoned).
fn lookup[&m, &t](m: &!m [int], t: &!t [byte], c: int, v: int) -> [] int {
    if poisoned(m, c, v) {
        return 0 - 1;
    }
    let s = slot(m, c, v);
    if m[s + 3] == 0 {
        return diag(m, t, c, "scope", text.f1(m, t, "`$` is not in scope", name(m, c, v)));
    }
    return s;
}

// Consume `v`: answers its type, or `-3` on a refusal.
fn consume[&m, &t](m: &!m [int], t: &!t [byte], c: int, v: int) -> [] int {
    let s = lookup(m, t, c, v);
    if s < 0 {
        return 0 - 3;
    }
    let ty = m[s];
    if ty == ir.index_ty() {
        diag(m, t, c, "type", text.f1(m, t, "loop index `$` is not a value", name(m, c, v)));
        return 0 - 3;
    }
    if m[s + 1] == 0 {
        diag(m, t, c, "use-after-move", text.f1(m, t, "`$` used after it was moved", name(m, c, v)));
        return 0 - 3;
    }
    if m[s + 2] < m[c + 4] {
        diag(m, t, c, "consume-outer", text.f1(m, t, "`$` is defined outside this loop body and would be consumed on every iteration; borrow it, or carry it through the loop", name(m, c, v)));
        return 0 - 3;
    }
    m[s + 1] = 0;
    bump(m, c, 0 - ir.threadgroup_bytes(m, ty));
    return ty;
}

fn borrow_var[&m, &t](m: &!m [int], t: &!t [byte], c: int, v: int) -> [] int {
    let s = lookup(m, t, c, v);
    if s < 0 {
        return 0 - 3;
    }
    let ty = m[s];
    if ty == ir.index_ty() {
        diag(m, t, c, "type", text.f1(m, t, "loop index `$` is not a value", name(m, c, v)));
        return 0 - 3;
    }
    if m[s + 1] == 0 {
        diag(m, t, c, "use-after-move", text.f1(m, t, "`$` borrowed after it was moved", name(m, c, v)));
        return 0 - 3;
    }
    return ty;
}

// The range of an index: `[some, lo, hi]` as a record, or -1.
fn range_of[&m, &t](m: &!m [int], t: &!t [byte], c: int, v: int) -> [] int {
    let s = slot(m, c, v);
    if m[s + 3] == 0 {
        if !poisoned(m, c, v) {
            diag(m, t, c, "scope", text.f1(m, t, "index `$` is not in scope", name(m, c, v)));
        }
        return 0 - 1;
    }
    if m[s] != ir.index_ty() {
        return diag(m, t, c, "type", text.f1(m, t, "`$` is used as an index but is a value", name(m, c, v)));
    }
    return rng(m, c, v);
}

// Type every operand of one op: borrows first, then moves; a value may
// not be both. Answers a list of types, or -1.
fn args[&m, &t](m: &!m [int], t: &!t [byte], c: int, a: int) -> [] int {
    let n = mem.size(m, a);
    var i = 0;
    while i < n {
        let x = mem.get(m, a, i);
        if !ir.is_borrow(x) {
            var j = 0;
            while j < n {
                if j != i && ir.var_of(mem.get(m, a, j)) == ir.var_of(x) {
                    return diag(m, t, c, "moved-while-borrowed", text.f1(m, t, "`$` is moved by an op that also uses it", name(m, c, ir.var_of(x))));
                }
                j = j + 1;
            }
        }
        i = i + 1;
    }
    let out = mem.list(m);
    i = 0;
    while i < n {
        mem.push(m, out, 0 - 3);
        i = i + 1;
    }
    i = 0;
    while i < n {
        let x = mem.get(m, a, i);
        if ir.is_borrow(x) {
            let ty = borrow_var(m, t, c, ir.var_of(x));
            if ty == 0 - 3 {
                return 0 - 1;
            }
            mem.set(m, out, i, ty);
        }
        i = i + 1;
    }
    i = 0;
    while i < n {
        let x = mem.get(m, a, i);
        if !ir.is_borrow(x) {
            let ty = consume(m, t, c, ir.var_of(x));
            if ty == 0 - 3 {
                return 0 - 1;
            }
            mem.set(m, out, i, ty);
        }
        i = i + 1;
    }
    return out;
}

// A tile that is not a fragment, or -1.
fn tile_of[&m, &t](m: &!m [int], t: &!t [byte], c: int, ty: int, what: &static [byte]) -> [] int {
    let x = tile_any(m, t, c, ty, what);
    if x < 0 {
        return 0 - 1;
    }
    if ir.space(m, x) == kinds.frag() {
        return diag(m, t, c, "type", text.f1(m, t, "$ is a matrix fragment; it can be accumulated into with `mma`, stored, or carried through a loop, and nothing else", text.lit(m, t, what)));
    }
    return x;
}

fn tile_any[&m, &t](m: &!m [int], t: &!t [byte], c: int, ty: int, what: &static [byte]) -> [] int {
    if ty >= 0 {
        return ty;
    }
    return diag(m, t, c, "type", text.f2(m, t, "$ must be a tile, found $", text.lit(m, t, what), ir.ty_debug(m, t, ty)));
}

fn expr_range[&m, &t](m: &!m [int], t: &!t [byte], c: int, e: int) -> [] (bool, int, int) {
    var lo = ir.idx_constant(m, e);
    var hi = lo;
    var i = 0;
    while i < ir.idx_count(m, e) {
        let r = range_of(m, t, c, ir.idx_var(m, e, i));
        if r < 0 {
            return (false, 0, 0);
        }
        // An empty loop never evaluates the view; any range will do.
        if m[r] == 1 {
            let k = ir.idx_coeff(m, e, i);
            let (x, y) = (m[r + 1] * k, m[r + 2] * k);
            if x < y {
                lo = lo + x;
                hi = hi + y;
            } else {
                lo = lo + y;
                hi = hi + x;
            }
        }
        i = i + 1;
    }
    return (true, lo, hi);
}

// A view, checked: answers `[dtype, shape]`, or -1.
fn view[&m, &t](m: &!m [int], t: &!t [byte], c: int, vw: int, writing: bool) -> [] int {
    let prog = m[c];
    let pi = ir.view_param(m, vw);
    if pi < 0 || pi >= mem.size(m, ir.p_params(m, prog)) {
        return diag(m, t, c, "scope", text.n1(m, t, "parameter #$ does not exist", pi));
    }
    let p = ir.param(m, prog, pi);
    let pshape = ir.param_shape(m, p);
    if writing && !ir.param_writable(m, p) {
        return diag(m, t, c, "type", text.f1(m, t, "parameter `$` is read-only", ir.param_name(m, p)));
    }
    let offs = ir.view_offsets(m, vw);
    let shape = ir.view_shape(m, vw);
    let rank = mem.size(m, pshape);
    if mem.size(m, offs) != rank || mem.size(m, shape) != rank {
        return diag(m, t, c, "shape", text.f1(m, t, "view of `$` has the wrong rank", ir.param_name(m, p)));
    }
    var d = 0;
    while d < rank {
        let (ok, lo, hi) = expr_range(m, t, c, mem.get(m, offs, d));
        if !ok {
            return 0 - 1;
        }
        let ext = mem.get(m, shape, d);
        if lo < 0 || hi + ext > mem.get(m, pshape, d) {
            let msg = text.f5(m, t, "view of `$` dim $ spans $..$ over the loop, but the dim is $", ir.param_name(m, p), text.num(m, t, d), text.num(m, t, lo), text.num(m, t, hi + ext), text.num(m, t, mem.get(m, pshape, d)));
            return diag(m, t, c, "bounds", msg);
        }
        d = d + 1;
    }
    return mem.rec2(m, ir.param_dtype(m, p), shape);
}

fn numeric[&m, &t](m: &!m [int], t: &!t [byte], c: int, ty: int, what: &static [byte]) -> [] bool {
    if ir.dtype(m, ty) == kinds.i8() {
        diag(m, t, c, "type", text.f1(m, t, "$ is quantised (I8); `dequant` it with its scales first", text.lit(m, t, what)));
        return false;
    }
    return true;
}

fn not_narrower[&m, &t](m: &!m [int], t: &!t [byte], c: int, from: int, to: int, what: &static [byte]) -> [] bool {
    if kinds.size_bytes(to) < kinds.size_bytes(from) {
        diag(m, t, c, "narrowing", text.f3(m, t, "$ narrows $ to $ implicitly; use `convert`", text.lit(m, t, what), text.lit(m, t, kinds.dtype_name(from)), text.lit(m, t, kinds.dtype_name(to))));
        return false;
    }
    return true;
}

fn sd[&m, &t](m: &!m [int], t: &!t [byte], s: int) -> [] int {
    return ir.shape_debug(m, t, s);
}

fn reg[&m](m: &!m [int], dt: int, shape: int) -> [] int {
    return ir.tile(m, dt, shape, kinds.reg());
}

// The type of an op's result: a type, `-2`... no: answers the result
// type, `-4` for "no result", or `-1` on a refusal.
fn op_ty[&m, &t](m: &!m [int], t: &!t [byte], c: int, o: int) -> [] int {
    let k = m[o];
    if k == kinds.op_fill() {
        let tl = m[o + 1];
        if ir.space(m, tl) == kinds.global() {
            return diag(m, t, c, "type", text.lit(m, t, "tiles live in threadgroup or registers"));
        }
        return tl;
    }
    if k == kinds.op_load() {
        let tl = m[o + 2];
        let vw = view(m, t, c, m[o + 1], false);
        if vw < 0 {
            return 0 - 1;
        }
        if !mem.same(m, m[vw + 1], ir.shape(m, tl)) {
            return diag(m, t, c, "shape", text.f2(m, t, "load of $ into a $ tile", sd(m, t, m[vw + 1]), sd(m, t, ir.shape(m, tl))));
        }
        if !not_narrower(m, t, c, m[vw], ir.dtype(m, tl), "load") {
            return 0 - 1;
        }
        return tl;
    }
    if k == kinds.op_store() {
        let vw = view(m, t, c, m[o + 2], true);
        if vw < 0 {
            return 0 - 1;
        }
        let tys = args(m, t, c, mem.of1(m, m[o + 1]));
        if tys < 0 {
            return 0 - 1;
        }
        let tl = tile_any(m, t, c, mem.get(m, tys, 0), "stored value");
        if tl < 0 {
            return 0 - 1;
        }
        // A view's unit dimensions are squeezed: a `[n]` tile stores into
        // a `[n, 1]` column as well as a `[1, n]` row.
        if !mem.same(m, squeeze(m, ir.shape(m, tl)), squeeze(m, m[vw + 1])) {
            return diag(m, t, c, "shape", text.f2(m, t, "store of $ into $", sd(m, t, ir.shape(m, tl)), sd(m, t, m[vw + 1])));
        }
        if !not_narrower(m, t, c, ir.dtype(m, tl), m[vw], "store") {
            return 0 - 1;
        }
        if ir.dtype(m, tl) != m[vw] {
            return diag(m, t, c, "type", text.f2(m, t, "store of $ into $", text.lit(m, t, kinds.dtype_name(ir.dtype(m, tl))), text.lit(m, t, kinds.dtype_name(m[vw]))));
        }
        return 0 - 4;
    }
    if k == kinds.op_add_window() {
        let vw = view(m, t, c, m[o + 2], false);
        if vw < 0 {
            return 0 - 1;
        }
        let tys = args(m, t, c, mem.of1(m, m[o + 1]));
        if tys < 0 {
            return 0 - 1;
        }
        let tl = tile_any(m, t, c, mem.get(m, tys, 0), "accumulator");
        if tl < 0 {
            return 0 - 1;
        }
        if ir.space(m, tl) != kinds.frag() || ir.dtype(m, tl) != kinds.f32() || m[vw] != kinds.f32() {
            let msg = text.f3(m, t, "add_window adds an f32 window into an f32 fragment tile, found $ into $ in $", text.lit(m, t, kinds.dtype_name(m[vw])), text.lit(m, t, kinds.dtype_name(ir.dtype(m, tl))), text.lit(m, t, kinds.space_name(ir.space(m, tl))));
            return diag(m, t, c, "type", msg);
        }
        if !mem.same(m, m[vw + 1], ir.shape(m, tl)) {
            return diag(m, t, c, "shape", text.f2(m, t, "window of $ added into a $ tile", sd(m, t, m[vw + 1]), sd(m, t, ir.shape(m, tl))));
        }
        return tl;
    }
    if k == kinds.op_stage() {
        let tys = args(m, t, c, mem.of1(m, m[o + 1]));
        if tys < 0 {
            return 0 - 1;
        }
        let tl = tile_of(m, t, c, mem.get(m, tys, 0), "staged tile");
        if tl < 0 || !numeric(m, t, c, tl, "staged tile") {
            return 0 - 1;
        }
        return ir.tile(m, m[o + 2], ir.shape(m, tl), kinds.threadgroup());
    }
    if k == kinds.op_mma() {
        let tys = args(m, t, c, mem.of3(m, m[o + 1], m[o + 2], m[o + 3]));
        if tys < 0 {
            return 0 - 1;
        }
        let tc = tile_any(m, t, c, mem.get(m, tys, 0), "mma accumulator");
        if tc < 0 {
            return 0 - 1;
        }
        let ta = tile_of(m, t, c, mem.get(m, tys, 1), "mma lhs");
        if ta < 0 {
            return 0 - 1;
        }
        let tb = tile_of(m, t, c, mem.get(m, tys, 2), "mma rhs");
        if tb < 0 {
            return 0 - 1;
        }
        if ir.space(m, tc) != kinds.frag() || ir.dtype(m, tc) != kinds.f32() {
            return diag(m, t, c, "type", text.f2(m, t, "mma accumulates into an f32 fragment tile, found $ in $", text.lit(m, t, kinds.dtype_name(ir.dtype(m, tc))), text.lit(m, t, kinds.space_name(ir.space(m, tc)))));
        }
        if !mma_operand(m, t, c, ta, "lhs") || !mma_operand(m, t, c, tb, "rhs") {
            return 0 - 1;
        }
        let (cs, as_, bs) = (ir.shape(m, tc), ir.shape(m, ta), ir.shape(m, tb));
        let ok = mem.size(m, cs) == 2 && mem.get(m, cs, 0) == mem.get(m, as_, 0) && mem.get(m, cs, 1) == mem.get(m, bs, 0) && mem.get(m, as_, 1) == mem.get(m, bs, 1);
        if !ok {
            return diag(m, t, c, "shape", text.f3(m, t, "mma of $ x $^T into $", sd(m, t, as_), sd(m, t, bs), sd(m, t, cs)));
        }
        return tc;
    }
    if k == kinds.op_matmul_nt() || k == kinds.op_matmul() {
        let tys = args(m, t, c, mem.of2(m, m[o + 1], m[o + 2]));
        if tys < 0 {
            return 0 - 1;
        }
        let ta = tile_of(m, t, c, mem.get(m, tys, 0), "matmul lhs");
        if ta < 0 {
            return 0 - 1;
        }
        let tb = tile_of(m, t, c, mem.get(m, tys, 1), "matmul rhs");
        if tb < 0 {
            return 0 - 1;
        }
        if !numeric(m, t, c, ta, "matmul lhs") || !numeric(m, t, c, tb, "matmul rhs") {
            return 0 - 1;
        }
        let (as_, bs) = (ir.shape(m, ta), ir.shape(m, tb));
        if mem.size(m, as_) != 2 || mem.size(m, bs) != 2 {
            return diag(m, t, c, "shape", text.lit(m, t, "matmul operands must be 2-d"));
        }
        let mm = mem.get(m, as_, 0);
        let kk = mem.get(m, as_, 1);
        var kb = mem.get(m, bs, 0);
        var n = mem.get(m, bs, 1);
        if k == kinds.op_matmul_nt() {
            kb = mem.get(m, bs, 1);
            n = mem.get(m, bs, 0);
        }
        if kk != kb {
            return diag(m, t, c, "shape", text.f2(m, t, "matmul $ x $: inner dims disagree", sd(m, t, as_), sd(m, t, bs)));
        }
        let acc = m[o + 3];
        if !not_narrower(m, t, c, ir.dtype(m, ta), acc, "matmul accumulator") || !not_narrower(m, t, c, ir.dtype(m, tb), acc, "matmul accumulator") {
            return 0 - 1;
        }
        return reg(m, acc, mem.of2(m, mm, n));
    }
    if k == kinds.op_binary() {
        let tys = args(m, t, c, mem.of2(m, m[o + 2], m[o + 3]));
        if tys < 0 {
            return 0 - 1;
        }
        let ta = tile_of(m, t, c, mem.get(m, tys, 0), "lhs");
        if ta < 0 {
            return 0 - 1;
        }
        let tb = tile_of(m, t, c, mem.get(m, tys, 1), "rhs");
        if tb < 0 {
            return 0 - 1;
        }
        if !numeric(m, t, c, ta, "lhs") || !numeric(m, t, c, tb, "rhs") {
            return 0 - 1;
        }
        let (as_, bs) = (ir.shape(m, ta), ir.shape(m, tb));
        let row_b = mem.size(m, as_) == 2 && mem.size(m, bs) == 1 && mem.get(m, bs, 0) == mem.get(m, as_, 0);
        let col_b = mem.size(m, as_) == 2 && mem.size(m, bs) == 2 && mem.get(m, bs, 0) == 1 && mem.get(m, bs, 1) == mem.get(m, as_, 1);
        if !mem.same(m, as_, bs) && !row_b && !col_b {
            return diag(m, t, c, "shape", text.f3(m, t, "$ of $ and $", text.lit(m, t, kinds.binop_name(m[o + 1])), sd(m, t, as_), sd(m, t, bs)));
        }
        if ir.dtype(m, ta) != ir.dtype(m, tb) {
            return diag(m, t, c, "type", text.f3(m, t, "$ mixes $ and $", text.lit(m, t, kinds.binop_name(m[o + 1])), text.lit(m, t, kinds.dtype_name(ir.dtype(m, ta))), text.lit(m, t, kinds.dtype_name(ir.dtype(m, tb)))));
        }
        return reg(m, ir.dtype(m, ta), as_);
    }
    if k == kinds.op_scale() || k == kinds.op_unary() {
        var a = m[o + 1];
        if k == kinds.op_unary() {
            a = m[o + 2];
        }
        let tys = args(m, t, c, mem.of1(m, a));
        if tys < 0 {
            return 0 - 1;
        }
        let tl = tile_of(m, t, c, mem.get(m, tys, 0), "operand");
        if tl < 0 || !numeric(m, t, c, tl, "operand") {
            return 0 - 1;
        }
        return reg(m, ir.dtype(m, tl), ir.shape(m, tl));
    }
    if k == kinds.op_row_reduce() {
        let tys = args(m, t, c, mem.of1(m, m[o + 2]));
        if tys < 0 {
            return 0 - 1;
        }
        let tl = tile_of(m, t, c, mem.get(m, tys, 0), "reduce operand");
        if tl < 0 || !numeric(m, t, c, tl, "reduce operand") {
            return 0 - 1;
        }
        if mem.size(m, ir.shape(m, tl)) != 2 {
            return diag(m, t, c, "shape", text.lit(m, t, "row reduce needs a 2-d tile"));
        }
        return reg(m, ir.dtype(m, tl), mem.of1(m, mem.get(m, ir.shape(m, tl), 0)));
    }
    // DequantFp4
    let tys = args(m, t, c, mem.of3(m, m[o + 1], m[o + 2], m[o + 3]));
    if tys < 0 {
        return 0 - 1;
    }
    let tq = tile_of(m, t, c, mem.get(m, tys, 0), "nvfp4 values");
    if tq < 0 {
        return 0 - 1;
    }
    let ts = tile_of(m, t, c, mem.get(m, tys, 1), "nvfp4 scales");
    if ts < 0 {
        return 0 - 1;
    }
    let tg = tile_of(m, t, c, mem.get(m, tys, 2), "nvfp4 row scales");
    if tg < 0 {
        return 0 - 1;
    }
    if ir.dtype(m, tq) != kinds.i8() || ir.dtype(m, ts) != kinds.i8() {
        return diag(m, t, c, "type", text.lit(m, t, "nvfp4 values and FP8 scales must be I8"));
    }
    if ir.dtype(m, tg) != kinds.f32() {
        return diag(m, t, c, "type", text.lit(m, t, "nvfp4 row scales must be F32"));
    }
    let qs = ir.shape(m, tq);
    var r = 0;
    var cols = 0;
    if mem.size(m, qs) > 0 {
        r = mem.get(m, qs, 0);
    }
    if mem.size(m, qs) > 1 {
        cols = 2 * mem.get(m, qs, 1);
    }
    let group = m[o + 4];
    let ok = mem.size(m, qs) == 2 && group > 0 && cols % group == 0 && mem.same(m, ir.shape(m, ts), mem.of2(m, r, cols / max1(group))) && mem.same(m, ir.shape(m, tg), mem.of1(m, r));
    if !ok {
        let msg = text.f4(m, t, "nvfp4 of $ values with scales $ and row scales $ in groups of $", sd(m, t, qs), sd(m, t, ir.shape(m, ts)), sd(m, t, ir.shape(m, tg)), text.num(m, t, group));
        return diag(m, t, c, "shape", msg);
    }
    return reg(m, kinds.f32(), mem.of2(m, r, cols));
}

fn max1(x: int) -> [] int {
    if x < 1 {
        return 1;
    }
    return x;
}

fn mma_operand[&m, &t](m: &!m [int], t: &!t [byte], c: int, x: int, what: &static [byte]) -> [] bool {
    if ir.space(m, x) != kinds.threadgroup() || ir.dtype(m, x) != kinds.f16() || mem.size(m, ir.shape(m, x)) != 2 {
        let msg = text.f4(m, t, "mma $ is a 2-d f16 tile in threadgroup memory, found $$ in $", text.lit(m, t, what), text.lit(m, t, kinds.dtype_name(ir.dtype(m, x))), sd(m, t, ir.shape(m, x)), text.lit(m, t, kinds.space_name(ir.space(m, x))));
        diag(m, t, c, "type", msg);
        return false;
    }
    return true;
}

fn squeeze[&m](m: &!m [int], s: int) -> [] int {
    let out = mem.list(m);
    var i = 0;
    while i < mem.size(m, s) {
        if mem.get(m, s, i) != 1 {
            mem.push(m, out, mem.get(m, s, i));
        }
        i = i + 1;
    }
    return out;
}

// The values a statement defines in its enclosing block.
fn defs[&m](m: &!m [int], s: int) -> [] int {
    if m[s] == kinds.st_let() {
        if m[s + 1] >= 0 {
            return mem.of1(m, m[s + 1]);
        }
        return mem.list(m);
    }
    return m[s + 7];
}

// Check a block at the current depth; answers the types of its yields
// (a list), or -1. Every linear value it defines must be gone by its end.
fn block[&m, &t](m: &!m [int], t: &!t [byte], c: int, b: int, pre: int) -> [] int {
    let scope = mem.list(m);
    var ok = true;
    let stmts = m[b];
    var i = 0;
    while i < mem.size(m, stmts) {
        let s = mem.get(m, stmts, i);
        if stmt(m, t, c, s, scope) < 0 {
            ok = false;
            let d = defs(m, s);
            var j = 0;
            while j < mem.size(m, d) {
                m[m[c + 8] + mem.get(m, d, j)] = 1;
                j = j + 1;
            }
        }
        i = i + 1;
    }
    let yields = mem.list(m);
    i = 0;
    while i < mem.size(m, m[b + 1]) {
        let ty = consume(m, t, c, mem.get(m, m[b + 1], i));
        if ty == 0 - 3 {
            ok = false;
        } else {
            mem.push(m, yields, ty);
        }
        i = i + 1;
    }
    // After an error, what looks like a leak is usually the error's
    // shadow; report leaks only in blocks that otherwise checked.
    let clean = ok;
    i = 0;
    while i < mem.size(m, scope) {
        let v = mem.get(m, scope, i);
        let s = slot(m, c, v);
        if clean && m[s + 1] == 1 && m[s] != ir.index_ty() {
            diag(m, t, c, "leak", text.f1(m, t, "`$` is never consumed; drop it, yield it or move it into an op", name(m, c, v)));
            ok = false;
        }
        i = i + 1;
    }
    i = 0;
    while i < mem.size(m, scope) {
        let v = mem.get(m, scope, i);
        let s = slot(m, c, v);
        if m[s + 3] == 1 && m[s + 1] == 1 {
            bump(m, c, 0 - ir.threadgroup_bytes(m, m[s]));
        }
        m[s + 3] = 0;
        m[rng(m, c, v)] = 0;
        i = i + 1;
    }
    // A region's parameters belong to the body's block.
    let block_ok = ok;
    i = 0;
    while i < mem.size(m, pre) {
        let v = mem.get(m, pre, i);
        let s = slot(m, c, v);
        if m[s + 3] == 1 {
            if m[s + 1] == 1 && m[s] != ir.index_ty() && block_ok {
                diag(m, t, c, "leak", text.f1(m, t, "loop parameter `$` is neither consumed nor yielded", name(m, c, v)));
                ok = false;
            }
            if m[s + 1] == 1 {
                bump(m, c, 0 - ir.threadgroup_bytes(m, m[s]));
            }
            m[s + 3] = 0;
        }
        m[rng(m, c, v)] = 0;
        i = i + 1;
    }
    if !ok {
        return 0 - 1;
    }
    return yields;
}

fn stmt[&m, &t](m: &!m [int], t: &!t [byte], c: int, s: int, scope: int) -> [] int {
    if m[s] == kinds.st_let() {
        let ty = op_ty(m, t, c, m[s + 2]);
        if ty == 0 - 1 {
            return 0 - 1;
        }
        let dst = m[s + 1];
        if dst >= 0 && ty != 0 - 4 {
            define(m, c, dst, ty, scope);
        } else if dst >= 0 {
            return diag(m, t, c, "type", text.f1(m, t, "`$` is bound to an op with no result", name(m, c, dst)));
        } else if ty != 0 - 4 {
            return diag(m, t, c, "leak", text.lit(m, t, "op result is discarded without a drop"));
        }
        return 0;
    }
    // `[st_for, index, start, end, init, params, body, results]`
    let (index, start, end) = (m[s + 1], m[s + 2], m[s + 3]);
    let (init, params, body, results) = (m[s + 4], m[s + 5], m[s + 6], m[s + 7]);
    if mem.size(m, init) != mem.size(m, params) || mem.size(m, results) != mem.size(m, params) {
        return diag(m, t, c, "carry", text.lit(m, t, "loop carry arity mismatch"));
    }
    let carry = mem.list(m);
    var i = 0;
    while i < mem.size(m, init) {
        let got = consume(m, t, c, mem.get(m, init, i));
        if got == 0 - 3 {
            return 0 - 1;
        }
        let want = ir.declared(m, m[c], mem.get(m, params, i));
        if want == kinds.none() {
            return diag(m, t, c, "type", text.f1(m, t, "region parameter `$` has no declared type", name(m, c, mem.get(m, params, i))));
        }
        if !ir.same_ty(m, got, want) {
            return diag(m, t, c, "carry", text.f3(m, t, "loop entered with `$`: $, carry declares $", name(m, c, mem.get(m, init, i)), ir.ty_debug(m, t, got), ir.ty_debug(m, t, want)));
        }
        mem.push(m, carry, want);
        i = i + 1;
    }
    // The body, one level deeper, with its index and parameters.
    m[c + 4] = m[c + 4] + 1;
    let pre = mem.list(m);
    define(m, c, index, ir.index_ty(), pre);
    let r = rng(m, c, index);
    if start < end {
        m[r] = 1;
        m[r + 1] = start;
        m[r + 2] = end - 1;
    } else {
        m[r] = 0;
    }
    i = 0;
    while i < mem.size(m, params) {
        define(m, c, mem.get(m, params, i), mem.get(m, carry, i), pre);
        i = i + 1;
    }
    let yields = block(m, t, c, body, pre);
    m[c + 4] = m[c + 4] - 1;
    if yields < 0 {
        return 0 - 1;
    }
    var same = mem.size(m, yields) == mem.size(m, carry);
    i = 0;
    while i < mem.size(m, yields) && i < mem.size(m, carry) {
        let (y, w) = (mem.get(m, yields, i), mem.get(m, carry, i));
        if !ir.same_ty(m, y, w) {
            diag(m, t, c, "carry", text.f3(m, t, "loop yields $ for carry #$, which was $", ir.ty_debug(m, t, y), text.num(m, t, i), ir.ty_debug(m, t, w)));
            same = false;
        }
        i = i + 1;
    }
    if mem.size(m, yields) != mem.size(m, carry) {
        diag(m, t, c, "carry", text.lit(m, t, "loop yields the wrong number of values"));
    }
    if !same {
        return 0 - 1;
    }
    i = 0;
    while i < mem.size(m, results) {
        define(m, c, mem.get(m, results, i), mem.get(m, carry, i), scope);
        i = i + 1;
    }
    return 0;
}

// Check `prog` for target `tg`. Answers the peak threadgroup footprint,
// or -1 with every diagnostic recorded as a refusal.
pub fn check[&m, &t](m: &!m [int], t: &!t [byte], prog: int, tg: int) -> [] int {
    let c = new_checker(m, prog, tg);
    let top = mem.list(m);
    let pid = ir.p_pid(m, prog);
    if pid >= 0 {
        if ir.p_grid(m, prog) == 0 {
            diag(m, t, c, "shape", text.lit(m, t, "a grid needs at least one instance"));
        }
        define(m, c, pid, ir.index_ty(), top);
        let r = rng(m, c, pid);
        m[r] = 1;
        m[r + 1] = 0;
        m[r + 2] = ir.p_grid(m, prog) - 1;
    }
    let pid2 = ir.p_pid2(m, prog);
    if pid2 >= 0 {
        define(m, c, pid2, ir.index_ty(), top);
        let r = rng(m, c, pid2);
        m[r] = 1;
        m[r + 1] = 0;
        m[r + 2] = ir.p_grid2(m, prog) - 1;
    }
    block(m, t, c, ir.p_body(m, prog), mem.list(m));
    let peak = m[c + 6];
    let limit = target.max_threadgroup_bytes(m, tg);
    if peak > limit {
        diag(m, t, c, "budget", text.f3(m, t, "peak threadgroup footprint is $ B; $ allows $ B", text.num(m, t, peak), target.name(m, tg), text.num(m, t, limit)));
    }
    let diags = m[c + 7];
    if mem.size(m, diags) == 0 {
        return peak;
    }
    var i = 0;
    while i < mem.size(m, diags) {
        let d = mem.get(m, diags, i);
        mem.push(m, m[mem.err_list_slot()], d);
        m[mem.err_count_slot()] = m[mem.err_count_slot()] + 1;
        i = i + 1;
    }
    return 0 - 1;
}
