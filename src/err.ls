edition 5;
module err;

// `err` -- refusals, each with a rule tag.
//
// The Rust this ports answers `Err(String)` and stops at the first one;
// the checker collects a list. Here every refusal is recorded as a
// `(rule, message)` pair in one list (`mem.err_list_slot`), and a caller
// that cannot go on returns `-1` after asking `failed`. The rule is a
// stable kebab-case name, so a test or a tool can match it without
// reading the sentence (`docs/rules.md`).

import mem;
import text;

pub fn fail[&m, &t](m: &!m [int], t: &!t [byte], rule: &static [byte], msg: int) -> [] int {
    let r = text.lit(m, t, rule);
    mem.push(m, m[mem.err_list_slot()], mem.rec2(m, r, msg));
    m[mem.err_count_slot()] = m[mem.err_count_slot()] + 1;
    return 0 - 1;
}

pub fn failed[&m](m: &!m [int]) -> [] bool {
    return m[mem.err_count_slot()] > 0;
}

pub fn count[&m](m: &!m [int]) -> [] int {
    return m[mem.err_count_slot()];
}

// Error `i` as `(rule, message)`.
pub fn nth[&m](m: &!m [int], i: int) -> [] (int, int) {
    let r = mem.get(m, m[mem.err_list_slot()], i);
    return (m[r], m[r + 1]);
}

// Forget every error recorded so far: for a caller that tries something
// and takes the refusal as an answer.
pub fn clear[&m](m: &!m [int]) -> [] int {
    m[mem.err_count_slot()] = 0;
    m[mem.err_list_slot()] = mem.list(m);
    return 0;
}

// `line:col`
pub fn pos[&m, &t](m: &!m [int], t: &!t [byte], line: int, col: int) -> [] int {
    return text.n2(m, t, "$:$", line, col);
}
