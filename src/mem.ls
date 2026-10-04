edition 5;
module mem;

// `mem` -- the compiler's one integer memory, and the growable lists
// built in it.
//
// A compiler's tables are many and small: tokens, syntax nodes, IR
// statements, the storage of every value. lex-sys gives a growable
// collection only as a `res` value moved from call to call (`std.vec`),
// and a struct cannot hold a reference (`lex-sys`'s
// `examples/tls_nb/gaps/t2_ref_field`), so passing a dozen tables to
// every function would be a dozen parameters. Instead there is one
// boxed slice of `int`, `m`, with a bump allocator, and every table is a
// record in it named by its offset (docs/design.md §3). Nothing is ever
// freed: a compile allocates a few megabytes and exits.
//
// Every access is bounds-checked by the language, so a stale or wrong
// handle is a trap at the access, never a read of something else's
// memory.
//
// The first 64 words are fixed slots (`docs/design.md` §3.1).

// Next free word of `m`.
pub fn top_slot() -> [] int {
    return 0;
}

// Next free byte of the text pool (`text.ls`).
pub fn text_slot() -> [] int {
    return 1;
}

// Number of errors recorded (`err.ls`).
pub fn err_count_slot() -> [] int {
    return 2;
}

// The list of error messages (`err.ls`).
pub fn err_list_slot() -> [] int {
    return 3;
}

// First word the allocator hands out.
pub fn first_word() -> [] int {
    return 64;
}

// Set up a fresh memory: the slots, and the text pool starting at 0.
pub fn init[&m](m: &!m [int]) -> [] int {
    m[top_slot()] = first_word();
    m[text_slot()] = 0;
    m[err_count_slot()] = 0;
    m[err_list_slot()] = 0;
    m[err_list_slot()] = list(m);
    return 0;
}

// `n` fresh words, zeroed (the memory starts zeroed and nothing is
// reused). Running out traps at the bounds check of the first write past
// the end, which is a resource limit, not a property of the input.
pub fn grab[&m](m: &!m [int], n: int) -> [] int {
    let at = m[top_slot()];
    let next = at + n;
    m[top_slot()] = next;
    // Touch the last word so exhaustion is found here, at the allocation,
    // rather than at some later write.
    if n > 0 {
        m[next - 1] = 0;
    }
    return at;
}

// A record of `n` words.
pub fn rec[&m](m: &!m [int], n: int) -> [] int {
    return grab(m, n);
}

pub fn rec2[&m](m: &!m [int], a: int, b: int) -> [] int {
    let r = grab(m, 2);
    m[r] = a;
    m[r + 1] = b;
    return r;
}

pub fn rec3[&m](m: &!m [int], a: int, b: int, c: int) -> [] int {
    let r = grab(m, 3);
    m[r] = a;
    m[r + 1] = b;
    m[r + 2] = c;
    return r;
}

pub fn rec4[&m](m: &!m [int], a: int, b: int, c: int, d: int) -> [] int {
    let r = grab(m, 4);
    m[r] = a;
    m[r + 1] = b;
    m[r + 2] = c;
    m[r + 3] = d;
    return r;
}

pub fn rec5[&m](m: &!m [int], a: int, b: int, c: int, d: int, e: int) -> [] int {
    let r = grab(m, 5);
    m[r] = a;
    m[r + 1] = b;
    m[r + 2] = c;
    m[r + 3] = d;
    m[r + 4] = e;
    return r;
}

// ---------------------------------------------------------------------
// Lists: `[len, cap, data]`, data reallocated by doubling
// ---------------------------------------------------------------------

pub fn list[&m](m: &!m [int]) -> [] int {
    let h = grab(m, 3);
    m[h] = 0;
    m[h + 1] = 4;
    m[h + 2] = grab(m, 4);
    return h;
}

pub fn size[&m](m: &!m [int], h: int) -> [] int {
    return m[h];
}

pub fn push[&m](m: &!m [int], h: int, v: int) -> [] int {
    let n = m[h];
    if n == m[h + 1] {
        let cap = 2 * n;
        let data = grab(m, cap);
        let old = m[h + 2];
        var i = 0;
        while i < n {
            m[data + i] = m[old + i];
            i = i + 1;
        }
        m[h + 1] = cap;
        m[h + 2] = data;
    }
    m[m[h + 2] + n] = v;
    m[h] = n + 1;
    return h;
}

pub fn get[&m](m: &!m [int], h: int, i: int) -> [] int {
    if i < 0 || i >= m[h] {
        // An index past the end is the program's own mistake, never
        // the input's; trap at it rather than read the next record.
        trap();
    }
    return m[m[h + 2] + i];
}

pub fn set[&m](m: &!m [int], h: int, i: int, v: int) -> [] int {
    if i < 0 || i >= m[h] {
        trap();
    }
    m[m[h + 2] + i] = v;
    return 0;
}

pub fn last[&m](m: &!m [int], h: int) -> [] int {
    return get(m, h, m[h] - 1);
}

pub fn pop[&m](m: &!m [int], h: int) -> [] int {
    let v = last(m, h);
    m[h] = m[h] - 1;
    return v;
}

pub fn of1[&m](m: &!m [int], a: int) -> [] int {
    let h = list(m);
    push(m, h, a);
    return h;
}

pub fn of2[&m](m: &!m [int], a: int, b: int) -> [] int {
    let h = list(m);
    push(m, h, a);
    push(m, h, b);
    return h;
}

pub fn of3[&m](m: &!m [int], a: int, b: int, c: int) -> [] int {
    let h = of2(m, a, b);
    push(m, h, c);
    return h;
}

pub fn copy[&m](m: &!m [int], h: int) -> [] int {
    let out = list(m);
    var i = 0;
    while i < m[h] {
        push(m, out, get(m, h, i));
        i = i + 1;
    }
    return out;
}

// Element-wise equality of two lists of ints.
pub fn same[&m](m: &!m [int], a: int, b: int) -> [] bool {
    if m[a] != m[b] {
        return false;
    }
    var i = 0;
    while i < m[a] {
        if get(m, a, i) != get(m, b, i) {
            return false;
        }
        i = i + 1;
    }
    return true;
}

pub fn product[&m](m: &!m [int], h: int) -> [] int {
    var p = 1;
    var i = 0;
    while i < m[h] {
        p = p * get(m, h, i);
        i = i + 1;
    }
    return p;
}

// Product of the elements from `from` to the end.
pub fn product_from[&m](m: &!m [int], h: int, from: int) -> [] int {
    var p = 1;
    var i = from;
    while i < m[h] {
        p = p * get(m, h, i);
        i = i + 1;
    }
    return p;
}

pub fn contains[&m](m: &!m [int], h: int, v: int) -> [] bool {
    var i = 0;
    while i < m[h] {
        if get(m, h, i) == v {
            return true;
        }
        i = i + 1;
    }
    return false;
}

pub fn of4[&m](m: &!m [int], a: int, b: int, c: int, d: int) -> [] int {
    let h = of3(m, a, b, c);
    push(m, h, d);
    return h;
}

pub fn of5[&m](m: &!m [int], a: int, b: int, c: int, d: int, e: int) -> [] int {
    let h = of4(m, a, b, c, d);
    push(m, h, e);
    return h;
}

pub fn of6[&m](m: &!m [int], a: int, b: int, c: int, d: int, e: int, f: int) -> [] int {
    let h = of5(m, a, b, c, d, e);
    push(m, h, f);
    return h;
}

pub fn of7[&m](m: &!m [int], a: int, b: int, c: int, d: int, e: int, f: int, g: int) -> [] int {
    let h = of6(m, a, b, c, d, e, f);
    push(m, h, g);
    return h;
}
