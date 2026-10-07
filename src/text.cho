edition 5;
module text;

// `text` -- strings, as handles into one byte pool.
//
// A string is `start * 2^24 + length`: one `int`, so it fits in a list,
// a record field or a return value with nothing to free. The bytes live
// in `t`, the compiler's one byte memory, appended and never changed --
// so a handle stays valid for the whole run and a substring is free
// (`docs/design.md` §3.2).
//
// The emitters build their output by formatting, as the Rust they port
// does with `format!`: `fmt2(m, t, "for (uint k = 0; k < $u; ++k) {", a, b)`
// splices handles where the template has `$`. Every argument is a
// handle evaluated before the call, so a string under construction is
// never interleaved with another -- the reason the pool can be a single
// bump pointer.

import mem;

fn shift() -> [] int {
    return 16777216;
}

pub fn start(h: int) -> [] int {
    return h / shift();
}

pub fn size(h: int) -> [] int {
    return h % shift();
}

pub fn handle(at: int, n: int) -> [] int {
    return at * shift() + n;
}

pub fn empty() -> [] int {
    return 0;
}

// The bytes of a handle.
pub fn bytes[&t](t: &t [byte], h: int) -> [] &t [byte] {
    let a = start(h);
    return t[a..a + size(h)];
}

pub fn at[&t](t: &!t [byte], h: int, i: int) -> [] int {
    return int_of(t[start(h) + i]);
}

// Append one byte at the end of the pool.
fn put_byte[&m, &t](m: &!m [int], t: &!t [byte], b: byte) -> [] int {
    let top = m[mem.text_slot()];
    t[top] = b;
    m[mem.text_slot()] = top + 1;
    return 0;
}

// Append a static literal at the end of the pool.
fn put_lit[&m, &t](m: &!m [int], t: &!t [byte], s: &static [byte]) -> [] int {
    let top = m[mem.text_slot()];
    var i = 0;
    while i < len(s) {
        t[top + i] = s[i];
        i = i + 1;
    }
    m[mem.text_slot()] = top + len(s);
    return 0;
}

// Append a copy of an existing string at the end of the pool.
fn put_str[&m, &t](m: &!m [int], t: &!t [byte], h: int) -> [] int {
    let top = m[mem.text_slot()];
    let n = size(h);
    copy_within(t, top, start(h), n);
    m[mem.text_slot()] = top + n;
    return 0;
}

fn begin[&m](m: &!m [int]) -> [] int {
    return m[mem.text_slot()];
}

fn finish[&m](m: &!m [int], from: int) -> [] int {
    return handle(from, m[mem.text_slot()] - from);
}

// A literal, copied into the pool.
pub fn lit[&m, &t](m: &!m [int], t: &!t [byte], s: &static [byte]) -> [] int {
    let from = begin(m);
    put_lit(m, t, s);
    return finish(m, from);
}

// Bytes from anywhere (an argument, a file name), copied into the pool.
pub fn from_bytes[&m, &t, &s](m: &!m [int], t: &!t [byte], s: &s [byte]) -> [] int {
    let from = begin(m);
    var i = 0;
    while i < len(s) {
        put_byte(m, t, s[i]);
        i = i + 1;
    }
    return finish(m, from);
}

// A decimal integer.
pub fn num[&m, &t](m: &!m [int], t: &!t [byte], n: int) -> [] int {
    let from = begin(m);
    if n < 0 {
        put_byte(m, t, byte_of('-'));
    }
    var digits = 1;
    var scale = 1;
    // Count with a negative magnitude so the most negative int prints.
    var rest = n;
    if rest > 0 {
        rest = 0 - rest;
    }
    while rest / scale <= 0 - 10 {
        scale = scale * 10;
        digits = digits + 1;
    }
    while scale > 0 {
        let d = 0 - rest / scale % 10;
        put_byte(m, t, byte_of('0' + d));
        scale = scale / 10;
    }
    return finish(m, from);
}

pub fn cat[&m, &t](m: &!m [int], t: &!t [byte], a: int, b: int) -> [] int {
    let from = begin(m);
    put_str(m, t, a);
    put_str(m, t, b);
    return finish(m, from);
}

pub fn cat3[&m, &t](m: &!m [int], t: &!t [byte], a: int, b: int, c: int) -> [] int {
    let from = begin(m);
    put_str(m, t, a);
    put_str(m, t, b);
    put_str(m, t, c);
    return finish(m, from);
}

// `$` in `tmpl` is replaced by the next of `args` (a list of handles).
pub fn fmt[&m, &t](m: &!m [int], t: &!t [byte], tmpl: &static [byte], args: int) -> [] int {
    let from = begin(m);
    var next = 0;
    var i = 0;
    while i < len(tmpl) {
        if tmpl[i] == byte_of('$') {
            put_str(m, t, mem.get(m, args, next));
            next = next + 1;
        } else {
            put_byte(m, t, tmpl[i]);
        }
        i = i + 1;
    }
    return finish(m, from);
}

pub fn f1[&m, &t](m: &!m [int], t: &!t [byte], tmpl: &static [byte], a: int) -> [] int {
    return fmt(m, t, tmpl, mem.of1(m, a));
}

pub fn f2[&m, &t](m: &!m [int], t: &!t [byte], tmpl: &static [byte], a: int, b: int) -> [] int {
    return fmt(m, t, tmpl, mem.of2(m, a, b));
}

pub fn f3[&m, &t](m: &!m [int], t: &!t [byte], tmpl: &static [byte], a: int, b: int, c: int) -> [] int {
    return fmt(m, t, tmpl, mem.of3(m, a, b, c));
}

pub fn f4[&m, &t](m: &!m [int], t: &!t [byte], tmpl: &static [byte], a: int, b: int, c: int, d: int) -> [] int {
    let l = mem.of3(m, a, b, c);
    mem.push(m, l, d);
    return fmt(m, t, tmpl, l);
}

pub fn f5[&m, &t](m: &!m [int], t: &!t [byte], tmpl: &static [byte], a: int, b: int, c: int, d: int, e: int) -> [] int {
    let l = mem.of3(m, a, b, c);
    mem.push(m, l, d);
    mem.push(m, l, e);
    return fmt(m, t, tmpl, l);
}

pub fn f6[&m, &t](m: &!m [int], t: &!t [byte], tmpl: &static [byte], a: int, b: int, c: int, d: int, e: int, f: int) -> [] int {
    let l = mem.of3(m, a, b, c);
    mem.push(m, l, d);
    mem.push(m, l, e);
    mem.push(m, l, f);
    return fmt(m, t, tmpl, l);
}

// Integer conveniences: the arguments are numbers, not handles.
pub fn n1[&m, &t](m: &!m [int], t: &!t [byte], tmpl: &static [byte], a: int) -> [] int {
    return f1(m, t, tmpl, num(m, t, a));
}

pub fn n2[&m, &t](m: &!m [int], t: &!t [byte], tmpl: &static [byte], a: int, b: int) -> [] int {
    let x = num(m, t, a);
    let y = num(m, t, b);
    return f2(m, t, tmpl, x, y);
}

pub fn n3[&m, &t](m: &!m [int], t: &!t [byte], tmpl: &static [byte], a: int, b: int, c: int) -> [] int {
    let x = num(m, t, a);
    let y = num(m, t, b);
    let z = num(m, t, c);
    return f3(m, t, tmpl, x, y, z);
}

// The strings of a list, joined by `sep`.
pub fn joined[&m, &t](m: &!m [int], t: &!t [byte], parts: int, sep: &static [byte]) -> [] int {
    let from = begin(m);
    var i = 0;
    while i < mem.size(m, parts) {
        if i > 0 {
            put_lit(m, t, sep);
        }
        put_str(m, t, mem.get(m, parts, i));
        i = i + 1;
    }
    return finish(m, from);
}

pub fn eq[&t](t: &!t [byte], a: int, b: int) -> [] bool {
    if size(a) != size(b) {
        return false;
    }
    let (x, y) = (start(a), start(b));
    var i = 0;
    while i < size(a) {
        if t[x + i] != t[y + i] {
            return false;
        }
        i = i + 1;
    }
    return true;
}

pub fn is[&t](t: &!t [byte], a: int, s: &static [byte]) -> [] bool {
    if size(a) != len(s) {
        return false;
    }
    let x = start(a);
    var i = 0;
    while i < len(s) {
        if t[x + i] != s[i] {
            return false;
        }
        i = i + 1;
    }
    return true;
}

// Where `needle` first occurs in `hay` at or after `from`, or -1.
pub fn find[&t](t: &!t [byte], hay: int, needle: int, from: int) -> [] int {
    let (h, n) = (start(hay), start(needle));
    let (hn, nn) = (size(hay), size(needle));
    var i = from;
    while i + nn <= hn {
        var j = 0;
        while j < nn && t[h + i + j] == t[n + j] {
            j = j + 1;
        }
        if j == nn {
            return i;
        }
        i = i + 1;
    }
    return 0 - 1;
}

pub fn contains[&m, &t](m: &!m [int], t: &!t [byte], hay: int, needle: &static [byte]) -> [] bool {
    let n = lit(m, t, needle);
    return find(t, hay, n, 0) >= 0;
}

pub fn starts_with[&t](t: &!t [byte], s: int, p: &static [byte]) -> [] bool {
    if size(s) < len(p) {
        return false;
    }
    var i = 0;
    while i < len(p) {
        if t[start(s) + i] != p[i] {
            return false;
        }
        i = i + 1;
    }
    return true;
}

pub fn ends_with[&t](t: &!t [byte], s: int, p: &static [byte]) -> [] bool {
    let n = size(s);
    if n < len(p) {
        return false;
    }
    var i = 0;
    while i < len(p) {
        if t[start(s) + n - len(p) + i] != p[i] {
            return false;
        }
        i = i + 1;
    }
    return true;
}

// Bytes `i .. j` of a string, sharing its storage.
pub fn sub(h: int, i: int, j: int) -> [] int {
    return handle(start(h) + i, j - i);
}

// Every occurrence of `pat` replaced by `rep`.
pub fn replace[&m, &t](m: &!m [int], t: &!t [byte], s: int, pat: int, rep: int) -> [] int {
    let pieces = mem.list(m);
    var from = 0;
    var at = find(t, s, pat, 0);
    while at >= 0 {
        mem.push(m, pieces, sub(s, from, at));
        from = at + size(pat);
        at = find(t, s, pat, from);
    }
    mem.push(m, pieces, sub(s, from, size(s)));
    let out = begin(m);
    var i = 0;
    while i < mem.size(m, pieces) {
        if i > 0 {
            put_str(m, t, rep);
        }
        put_str(m, t, mem.get(m, pieces, i));
        i = i + 1;
    }
    return finish(m, out);
}

pub fn replace_lit[&m, &t](m: &!m [int], t: &!t [byte], s: int, pat: &static [byte], rep: int) -> [] int {
    let p = lit(m, t, pat);
    return replace(m, t, s, p, rep);
}

// Is every byte ASCII alphanumeric?
pub fn is_word[&t](t: &!t [byte], s: int) -> [] bool {
    if size(s) == 0 {
        return false;
    }
    var i = 0;
    while i < size(s) {
        let c = at(t, s, i);
        let ok = c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c == '_';
        if !ok {
            return false;
        }
        i = i + 1;
    }
    return true;
}

// Index of the first occurrence of byte `c` at or after `from`, or -1.
pub fn index_of[&t](t: &!t [byte], s: int, c: int, from: int) -> [] int {
    var i = from;
    while i < size(s) {
        if at(t, s, i) == c {
            return i;
        }
        i = i + 1;
    }
    return 0 - 1;
}
