edition 5;
module target;

// The hardware table: `lex_ir::Target`, as data. A target is a record
// `[name, simd_width, max_threads_per_threadgroup, max_threadgroup_bytes,
//   async_copy, split_barriers]`. Only the two rows with a backend are
// named on the command line, but the table carries all four the Rust
// has, so a schedule for the others still type-checks.

import mem;
import text;

fn row[&m, &t](m: &!m [int], t: &!t [byte], name: &static [byte], simd: int, threads: int, bytes: int, async_copy: int, split: int) -> [] int {
    let r = mem.grab(m, 6);
    m[r] = text.lit(m, t, name);
    m[r + 1] = simd;
    m[r + 2] = threads;
    m[r + 3] = bytes;
    m[r + 4] = async_copy;
    m[r + 5] = split;
    return r;
}

pub fn apple_m_series[&m, &t](m: &!m [int], t: &!t [byte]) -> [] int {
    return row(m, t, "apple-m-series", 32, 1024, 32 * 1024, 0, 0);
}

pub fn nvidia_ada[&m, &t](m: &!m [int], t: &!t [byte]) -> [] int {
    return row(m, t, "nvidia-ada", 32, 1024, 99 * 1024, 1, 0);
}

pub fn nvidia_hopper[&m, &t](m: &!m [int], t: &!t [byte]) -> [] int {
    return row(m, t, "nvidia-hopper", 32, 1024, 227 * 1024, 1, 1);
}

pub fn amd_cdna3[&m, &t](m: &!m [int], t: &!t [byte]) -> [] int {
    return row(m, t, "amd-cdna3", 64, 1024, 64 * 1024, 0, 0);
}

pub fn name[&m](m: &!m [int], tg: int) -> [] int {
    return m[tg];
}

pub fn simd_width[&m](m: &!m [int], tg: int) -> [] int {
    return m[tg + 1];
}

pub fn max_threads[&m](m: &!m [int], tg: int) -> [] int {
    return m[tg + 2];
}

pub fn max_threadgroup_bytes[&m](m: &!m [int], tg: int) -> [] int {
    return m[tg + 3];
}
