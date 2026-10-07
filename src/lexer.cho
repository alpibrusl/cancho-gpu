edition 5;
module lexer;

// `.lx` text to tokens: `lex_front::syntax::Lexer`, ported.
//
// A token is a record `[kind, text, value, line, col]`: `kind` is one of
// the four below, `text` the bytes it was spelled with (a handle into the
// pool, sharing the source's storage), `value` a number's binary64 bits.

import mem;
import text;
import err;
import f32;

pub fn ident() -> [] int {
    return 0;
}

pub fn num() -> [] int {
    return 1;
}

pub fn punct() -> [] int {
    return 2;
}

pub fn end() -> [] int {
    return 3;
}

pub fn kind[&m](m: &!m [int], tok: int) -> [] int {
    return m[tok];
}

pub fn spelling[&m](m: &!m [int], tok: int) -> [] int {
    return m[tok + 1];
}

pub fn value[&m](m: &!m [int], tok: int) -> [] float {
    return f32.from_bits(m[tok + 2]);
}

pub fn line[&m](m: &!m [int], tok: int) -> [] int {
    return m[tok + 3];
}

pub fn col[&m](m: &!m [int], tok: int) -> [] int {
    return m[tok + 4];
}

fn token[&m](m: &!m [int], k: int, s: int, v: int, line: int, col: int) -> [] int {
    return mem.rec5(m, k, s, v, line, col);
}

// What `{:?}` of the Rust `Tok` prints, for messages.
pub fn show[&m, &t](m: &!m [int], t: &!t [byte], tok: int) -> [] int {
    let k = m[tok];
    if k == ident() {
        return text.f1(m, t, "Ident(\"$\")", m[tok + 1]);
    }
    if k == num() {
        return text.f1(m, t, "Num($)", f32.debug(m, t, value(m, tok)));
    }
    if k == punct() {
        return text.f1(m, t, "Punct(\"$\")", m[tok + 1]);
    }
    return text.lit(m, t, "End");
}

fn is_space(c: int) -> [] bool {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 11 || c == 12;
}

fn is_digit(c: int) -> [] bool {
    return c >= '0' && c <= '9';
}

// ASCII letters and `_`, and any byte of a multi-byte character, which
// `char::is_alphabetic` would take for the letters this lexer meets.
fn is_alpha(c: int) -> [] bool {
    return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c == '_' || c >= 128;
}

fn is_punct(c: int) -> [] bool {
    return c == '(' || c == ')' || c == '{' || c == '}' || c == '[' || c == ']' || c == ',' || c == ':' || c == ';' || c == '=' || c == '*' || c == '+' || c == '-' || c == '/' || c == '&' || c == '@';
}

// The tokens of `src` (a handle), ending with an `End`; or -1 with the
// refusal recorded.
pub fn tokens[&m, &t](m: &!m [int], t: &!t [byte], src: int) -> [] int {
    let out = mem.list(m);
    let n = text.size(src);
    var i = 0;
    var line = 1;
    var col = 1;
    while true {
        // Whitespace and `//` comments.
        var skipping = true;
        while skipping && i < n {
            let c = text.at(t, src, i);
            if is_space(c) {
                if c == '\n' {
                    line = line + 1;
                    col = 1;
                } else {
                    col = col + 1;
                }
                i = i + 1;
            } else if c == '/' {
                // Only a comment if doubled; a lone `/` divides.
                if i + 1 < n && text.at(t, src, i + 1) == '/' {
                    while i < n && text.at(t, src, i) != '\n' {
                        i = i + 1;
                    }
                    if i < n {
                        i = i + 1;
                        line = line + 1;
                        col = 1;
                    }
                } else {
                    mem.push(m, out, token(m, punct(), text.sub(src, i, i + 1), 0, line, col));
                    i = i + 1;
                    col = col + 1;
                }
            } else {
                skipping = false;
            }
        }
        if i >= n {
            mem.push(m, out, token(m, end(), text.empty(), 0, line, col));
            return out;
        }
        let c = text.at(t, src, i);
        let (tl, tc) = (line, col);
        if is_alpha(c) {
            let from = i;
            while i < n && (is_alpha(text.at(t, src, i)) || is_digit(text.at(t, src, i))) {
                // A continuation byte is part of the character before it.
                let b = text.at(t, src, i);
                if b < 128 || b >= 192 {
                    col = col + 1;
                }
                i = i + 1;
            }
            mem.push(m, out, token(m, ident(), text.sub(src, from, i), 0, tl, tc));
        } else if is_digit(c) {
            let from = i;
            var going = true;
            while going && i < n {
                let d = text.at(t, src, i);
                // `1e-5` and `0.5` both lex as one number, so a sign is
                // part of it only straight after an exponent marker.
                var prev = 0;
                if i > from {
                    prev = text.at(t, src, i - 1);
                }
                let exponent_sign = (d == '-' || d == '+') && (prev == 'e' || prev == 'E');
                // `0 .. k` and `0..k` both mean a range: a dot followed by
                // another is not part of the number.
                let range_dots = d == '.' && i + 1 < n && text.at(t, src, i + 1) == '.';
                if !range_dots && (is_digit(d) || d == '.' || d == 'e' || d == 'E' || exponent_sign) {
                    i = i + 1;
                    col = col + 1;
                } else {
                    going = false;
                }
            }
            let s = text.sub(src, from, i);
            let (ok, v) = f32.parse(m, t, s);
            if !ok {
                let at = err.pos(m, t, tl, tc);
                return err.fail(m, t, "syntax", text.f2(m, t, "$: `$` is not a number", at, s));
            }
            mem.push(m, out, token(m, num(), s, bits_of(v), tl, tc));
            // `->` and `..` are one token; everything else is one byte.
        } else if c == '-' && i + 1 < n && text.at(t, src, i + 1) == '>' {
            mem.push(m, out, token(m, punct(), text.sub(src, i, i + 2), 0, tl, tc));
            i = i + 2;
            col = col + 2;
        } else if c == '.' && i + 1 < n && text.at(t, src, i + 1) == '.' {
            mem.push(m, out, token(m, punct(), text.sub(src, i, i + 2), 0, tl, tc));
            i = i + 2;
            col = col + 2;
        } else if is_punct(c) {
            mem.push(m, out, token(m, punct(), text.sub(src, i, i + 1), 0, tl, tc));
            i = i + 1;
            col = col + 1;
        } else {
            let at = err.pos(m, t, tl, tc);
            return err.fail(m, t, "syntax", text.f2(m, t, "$: `$` means nothing here", at, text.sub(src, i, i + 1)));
        }
    }
    return out;
}
