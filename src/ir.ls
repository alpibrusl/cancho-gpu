edition 5;
module ir;

// The typed tile program: `lex_front::ir`, the part a `.lx` file can
// produce. Values are SSA variables numbered from 0; every record below
// lives in `m` (`docs/design.md` §5 has the layouts).
//
// * TileTy  `[dtype, shape, space]` -- `shape` a list of extents.
// * Ty      a TileTy handle for a tile, `index_ty()` for a loop index, and
//           `kinds.none()` for "not declared".
// * Arg     `2 * var` moves the value, `2 * var + 1` borrows it.
// * IdxExpr `[constant, terms]` -- `terms` a flat list `v0, c0, v1, c1..`
//           for `constant + sum(c * v)`.
// * View    `[param, offsets, shape]` -- `offsets` a list of IdxExpr.
// * Op      `[kind, a, b, c, d]` -- `kinds.op_*`.
// * Stmt    `[st_let, dst, op]` or
//           `[st_for, index, start, end, init, params, body, results]`.
// * Block   `[stmts, yields]`.
// * Program `[name, params, body, names, declared, grid, pid, grid2, pid2]`,
//           a param being `[name, dtype, shape, writable]`.

import mem;
import text;
import kinds;

pub fn index_ty() -> [] int {
    return 0 - 2;
}

// --- TileTy

pub fn tile[&m](m: &!m [int], dtype: int, shape: int, space: int) -> [] int {
    return mem.rec3(m, dtype, shape, space);
}

pub fn dtype[&m](m: &!m [int], ty: int) -> [] int {
    return m[ty];
}

pub fn shape[&m](m: &!m [int], ty: int) -> [] int {
    return m[ty + 1];
}

pub fn space[&m](m: &!m [int], ty: int) -> [] int {
    return m[ty + 2];
}

pub fn elems[&m](m: &!m [int], ty: int) -> [] int {
    return mem.product(m, m[ty + 1]);
}

pub fn bytes[&m](m: &!m [int], ty: int) -> [] int {
    return elems(m, ty) * kinds.size_bytes(m[ty]);
}

pub fn same_tile[&m](m: &!m [int], a: int, b: int) -> [] bool {
    return m[a] == m[b] && m[a + 2] == m[b + 2] && mem.same(m, m[a + 1], m[b + 1]);
}

// Type equality: two indices, or two equal tiles.
pub fn same_ty[&m](m: &!m [int], a: int, b: int) -> [] bool {
    if a < 0 || b < 0 {
        return a == b;
    }
    return same_tile(m, a, b);
}

// Threadgroup bytes a value of this type pins while it is live.
pub fn threadgroup_bytes[&m](m: &!m [int], ty: int) -> [] int {
    if ty >= 0 && m[ty + 2] == kinds.threadgroup() {
        return bytes(m, ty);
    }
    return 0;
}

// `[a, b, c]`, as `{:?}` prints a `Vec<usize>`.
pub fn shape_debug[&m, &t](m: &!m [int], t: &!t [byte], s: int) -> [] int {
    let parts = mem.list(m);
    var i = 0;
    while i < mem.size(m, s) {
        mem.push(m, parts, text.num(m, t, mem.get(m, s, i)));
        i = i + 1;
    }
    return text.f1(m, t, "[$]", text.joined(m, t, parts, ", "));
}

// A type as the checker's messages print it.
pub fn ty_debug[&m, &t](m: &!m [int], t: &!t [byte], ty: int) -> [] int {
    if ty == index_ty() {
        return text.lit(m, t, "Index");
    }
    if ty < 0 {
        return text.lit(m, t, "None");
    }
    let d = text.lit(m, t, kinds.dtype_name(m[ty]));
    let s = shape_debug(m, t, m[ty + 1]);
    let sp = text.lit(m, t, kinds.space_name(m[ty + 2]));
    return text.f3(m, t, "Tile(TileTy { dtype: $, shape: $, space: $ })", d, s, sp);
}

// --- Arg

pub fn mv(v: int) -> [] int {
    return 2 * v;
}

pub fn borrow_of(v: int) -> [] int {
    return 2 * v + 1;
}

pub fn var_of(a: int) -> [] int {
    return a / 2;
}

pub fn is_borrow(a: int) -> [] bool {
    return a % 2 == 1;
}

// --- IdxExpr

pub fn idx_lit[&m](m: &!m [int], c: int) -> [] int {
    return mem.rec2(m, c, mem.list(m));
}

pub fn idx_scaled[&m](m: &!m [int], v: int, coeff: int, c: int) -> [] int {
    return mem.rec2(m, c, mem.of2(m, v, coeff));
}

pub fn idx_constant[&m](m: &!m [int], e: int) -> [] int {
    return m[e];
}

pub fn idx_terms[&m](m: &!m [int], e: int) -> [] int {
    return m[e + 1];
}

// Number of `(var, coeff)` terms.
pub fn idx_count[&m](m: &!m [int], e: int) -> [] int {
    return mem.size(m, m[e + 1]) / 2;
}

pub fn idx_var[&m](m: &!m [int], e: int, i: int) -> [] int {
    return mem.get(m, m[e + 1], 2 * i);
}

pub fn idx_coeff[&m](m: &!m [int], e: int, i: int) -> [] int {
    return mem.get(m, m[e + 1], 2 * i + 1);
}

// --- View

pub fn view[&m](m: &!m [int], param: int, offsets: int, shape: int) -> [] int {
    return mem.rec3(m, param, offsets, shape);
}

pub fn view_param[&m](m: &!m [int], v: int) -> [] int {
    return m[v];
}

pub fn view_offsets[&m](m: &!m [int], v: int) -> [] int {
    return m[v + 1];
}

pub fn view_shape[&m](m: &!m [int], v: int) -> [] int {
    return m[v + 2];
}

// --- Op

pub fn make_op[&m](m: &!m [int], k: int, a: int, b: int, c: int, d: int) -> [] int {
    return mem.rec5(m, k, a, b, c, d);
}

// --- Program

pub fn p_name[&m](m: &!m [int], p: int) -> [] int {
    return m[p];
}

pub fn p_params[&m](m: &!m [int], p: int) -> [] int {
    return m[p + 1];
}

pub fn p_body[&m](m: &!m [int], p: int) -> [] int {
    return m[p + 2];
}

pub fn p_names[&m](m: &!m [int], p: int) -> [] int {
    return m[p + 3];
}

pub fn p_declared[&m](m: &!m [int], p: int) -> [] int {
    return m[p + 4];
}

pub fn p_grid[&m](m: &!m [int], p: int) -> [] int {
    return m[p + 5];
}

pub fn p_pid[&m](m: &!m [int], p: int) -> [] int {
    return m[p + 6];
}

pub fn p_grid2[&m](m: &!m [int], p: int) -> [] int {
    return m[p + 7];
}

pub fn p_pid2[&m](m: &!m [int], p: int) -> [] int {
    return m[p + 8];
}

pub fn vars[&m](m: &!m [int], p: int) -> [] int {
    return mem.size(m, m[p + 3]);
}

pub fn name_of[&m](m: &!m [int], p: int, v: int) -> [] int {
    return mem.get(m, m[p + 3], v);
}

pub fn declared[&m](m: &!m [int], p: int, v: int) -> [] int {
    return mem.get(m, m[p + 4], v);
}

pub fn param[&m](m: &!m [int], p: int, i: int) -> [] int {
    return mem.get(m, m[p + 1], i);
}

pub fn param_name[&m](m: &!m [int], prm: int) -> [] int {
    return m[prm];
}

pub fn param_dtype[&m](m: &!m [int], prm: int) -> [] int {
    return m[prm + 1];
}

pub fn param_shape[&m](m: &!m [int], prm: int) -> [] int {
    return m[prm + 2];
}

pub fn param_writable[&m](m: &!m [int], prm: int) -> [] bool {
    return m[prm + 3] == 1;
}

// ---------------------------------------------------------------------
// The builder: `ir::Builder`. `[name, params, names, declared, stack,
// grid, pid, grid2, pid2]`, the stack a list of statement lists.
// ---------------------------------------------------------------------

pub fn builder[&m](m: &!m [int], name: int) -> [] int {
    let b = mem.grab(m, 9);
    m[b] = name;
    m[b + 1] = mem.list(m);
    m[b + 2] = mem.list(m);
    m[b + 3] = mem.list(m);
    m[b + 4] = mem.of1(m, mem.list(m));
    m[b + 5] = 1;
    m[b + 6] = kinds.none();
    m[b + 7] = 1;
    m[b + 8] = kinds.none();
    return b;
}

pub fn has_grid[&m](m: &!m [int], b: int) -> [] bool {
    return m[b + 6] >= 0;
}

pub fn has_grid2[&m](m: &!m [int], b: int) -> [] bool {
    return m[b + 8] >= 0;
}

// A fresh variable named `name.id`, with a declared type (or none).
pub fn fresh[&m, &t](m: &!m [int], t: &!t [byte], b: int, name: int, declared: int) -> [] int {
    let v = mem.size(m, m[b + 2]);
    mem.push(m, m[b + 2], text.f2(m, t, "$.$", name, text.num(m, t, v)));
    mem.push(m, m[b + 3], declared);
    return v;
}

pub fn fresh_lit[&m, &t](m: &!m [int], t: &!t [byte], b: int, name: &static [byte], declared: int) -> [] int {
    return fresh(m, t, b, text.lit(m, t, name), declared);
}

pub fn grid[&m, &t](m: &!m [int], t: &!t [byte], b: int, n: int) -> [] int {
    let pid = fresh_lit(m, t, b, "pid", index_ty());
    m[b + 5] = n;
    m[b + 6] = pid;
    return pid;
}

pub fn grid2[&m, &t](m: &!m [int], t: &!t [byte], b: int, n: int) -> [] int {
    let pid = fresh_lit(m, t, b, "pid2", index_ty());
    m[b + 7] = n;
    m[b + 8] = pid;
    return pid;
}

pub fn add_param[&m](m: &!m [int], b: int, name: int, dtype: int, shape: int, writable: bool) -> [] int {
    var w = 0;
    if writable {
        w = 1;
    }
    mem.push(m, m[b + 1], mem.rec4(m, name, dtype, shape, w));
    return mem.size(m, m[b + 1]) - 1;
}

fn push_stmt[&m](m: &!m [int], b: int, s: int) -> [] int {
    mem.push(m, mem.last(m, m[b + 4]), s);
    return 0;
}

// An op that produces a value.
pub fn emit[&m, &t](m: &!m [int], t: &!t [byte], b: int, name: int, o: int) -> [] int {
    let v = fresh(m, t, b, name, kinds.none());
    push_stmt(m, b, mem.rec3(m, kinds.st_let(), v, o));
    return v;
}

pub fn emit_lit[&m, &t](m: &!m [int], t: &!t [byte], b: int, name: &static [byte], o: int) -> [] int {
    return emit(m, t, b, text.lit(m, t, name), o);
}

// An op for its effect only.
pub fn effect[&m](m: &!m [int], b: int, o: int) -> [] int {
    push_stmt(m, b, mem.rec3(m, kinds.st_let(), kinds.none(), o));
    return 0;
}

// `for i in start..end` -- in two halves, because the body is built by
// the caller in between: `for_begin` makes the index and the carried
// parameters and opens a block; `for_end` closes it with the yields and
// makes the results. Answers `[index, params]`.
pub fn for_begin[&m, &t](m: &!m [int], t: &!t [byte], b: int, carry_tys: int) -> [] int {
    let i = fresh_lit(m, t, b, "i", index_ty());
    let params = mem.list(m);
    var k = 0;
    while k < mem.size(m, carry_tys) {
        mem.push(m, params, fresh_lit(m, t, b, "carry", mem.get(m, carry_tys, k)));
        k = k + 1;
    }
    mem.push(m, m[b + 4], mem.list(m));
    return mem.rec2(m, i, params);
}

// Answers the results.
pub fn for_end[&m, &t](m: &!m [int], t: &!t [byte], b: int, opened: int, start: int, end: int, init: int, yields: int) -> [] int {
    let stmts = mem.pop(m, m[b + 4]);
    let params = m[opened + 1];
    let results = mem.list(m);
    var k = 0;
    while k < mem.size(m, params) {
        mem.push(m, results, fresh_lit(m, t, b, "out", kinds.none()));
        k = k + 1;
    }
    let s = mem.grab(m, 8);
    m[s] = kinds.st_for();
    m[s + 1] = m[opened];
    m[s + 2] = start;
    m[s + 3] = end;
    m[s + 4] = init;
    m[s + 5] = params;
    m[s + 6] = mem.rec2(m, stmts, yields);
    m[s + 7] = results;
    push_stmt(m, b, s);
    return results;
}

pub fn finish[&m](m: &!m [int], b: int) -> [] int {
    let stmts = mem.pop(m, m[b + 4]);
    let p = mem.grab(m, 9);
    m[p] = m[b];
    m[p + 1] = m[b + 1];
    m[p + 2] = mem.rec2(m, stmts, mem.list(m));
    m[p + 3] = m[b + 2];
    m[p + 4] = m[b + 3];
    m[p + 5] = m[b + 5];
    m[p + 6] = m[b + 6];
    m[p + 7] = m[b + 7];
    m[p + 8] = m[b + 8];
    return p;
}
