edition 5;

// Unit tests for the parts that have no oracle in the goldens on their
// own: f32 rounding and printing (the expected strings are what Rust's
// `{:?}` prints for `s.parse::<f64>().unwrap() as f32`, recorded by
// running it), the f32 operations, and the string pool.
//
//     lex-sys test tests/unit_test.ls <every src/*.ls but main.ls> --std

import mem;
import text;
import f32;

fn words() -> [] int {
    return 100000;
}

fn prints_as[&m, &t](m: &!m [int], t: &!t [byte], input: &static [byte], want: &static [byte]) -> [] bool {
    let (ok, v) = f32.parse(m, t, text.lit(m, t, input));
    return ok && text.is(t, f32.debug(m, t, f32.round(v)), want);
}

fn check_printing[&m, &t](m: &!m [int], t: &!t [byte]) -> [] bool {
    var ok = true;
    ok = ok && prints_as(m, t, "0.5", "0.5");
    ok = ok && prints_as(m, t, "1e-5", "1e-5");
    ok = ok && prints_as(m, t, "0.00001", "1e-5");
    ok = ok && prints_as(m, t, "16384", "16384.0");
    ok = ok && prints_as(m, t, "0.000244140625", "0.00024414063");
    ok = ok && prints_as(m, t, "0.1", "0.1");
    ok = ok && prints_as(m, t, "3", "3.0");
    ok = ok && prints_as(m, t, "1e16", "1e16");
    ok = ok && prints_as(m, t, "123456789", "123456790.0");
    ok = ok && prints_as(m, t, "0.0001", "0.0001");
    ok = ok && prints_as(m, t, "1e-45", "1e-45");
    ok = ok && prints_as(m, t, "3.4028235e38", "3.4028235e38");
    ok = ok && prints_as(m, t, "7e-46", "0.0");
    ok = ok && prints_as(m, t, "1e39", "inf");
    ok = ok && prints_as(m, t, "2.5e-40", "2.5e-40");
    ok = ok && prints_as(m, t, "1.17549435e-38", "1.1754944e-38");
    ok = ok && prints_as(m, t, "65504", "65504.0");
    ok = ok && prints_as(m, t, "0.3333333333", "0.33333334");
    ok = ok && prints_as(m, t, "1e-4", "0.0001");
    ok = ok && prints_as(m, t, "9.999999e-5", "9.999999e-5");
    ok = ok && prints_as(m, t, "1e15", "1000000000000000.0");
    ok = ok && prints_as(m, t, "1.5e16", "1.5e16");
    ok = ok && prints_as(m, t, "4096", "4096.0");
    ok = ok && prints_as(m, t, "1e-10", "1e-10");
    ok = ok && prints_as(m, t, "100000000", "100000000.0");
    return ok;
}

fn check_arithmetic[&m, &t](m: &!m [int], t: &!t [byte]) -> [] bool {
    let one = f32.round(1.0);
    let q = f32.div(one, f32.round(4096.0));
    let third = f32.div(one, f32.round(3.0));
    let mix = f32.add(f32.mul(q, f32.round(2.0)), f32.round(0.00001));
    var ok = text.is(t, f32.debug(m, t, q), "0.00024414063");
    ok = ok && text.is(t, f32.debug(m, t, third), "0.33333334");
    ok = ok && text.is(t, f32.debug(m, t, mix), "0.00049828127");
    // A value round-trips through its bits.
    ok = ok && f32.from_bits(bits_of(third)) == third;
    ok = ok && f32.from_bits(bits_of(-2.5)) == -2.5;
    return ok;
}

fn check_parsing[&m, &t](m: &!m [int], t: &!t [byte]) -> [] bool {
    let (a, x) = f32.parse(m, t, text.lit(m, t, "007.50"));
    let (b, y) = f32.parse(m, t, text.lit(m, t, "1."));
    let (c, z) = f32.parse(m, t, text.lit(m, t, "1e"));
    let (d, w) = f32.parse(m, t, text.lit(m, t, "1.5.2"));
    return a && x == 7.5 && b && y == 1.0 && !c && !d;
}

fn check_counts[&m, &t](m: &!m [int], t: &!t [byte]) -> [] bool {
    var ok = text.is(t, f32.count_string(m, t, 4096.0), "4096");
    ok = ok && text.is(t, f32.count_string(m, t, 1e20), "18446744073709551615");
    ok = ok && text.is(t, f32.count_string(m, t, 9223372036854775808.0), "9223372036854775808");
    ok = ok && text.is(t, f32.count_string(m, t, 18446744073709549568.0), "18446744073709549568");
    return ok;
}

fn check_text[&m, &t](m: &!m [int], t: &!t [byte]) -> [] bool {
    let s = text.f2(m, t, "a$b$c", text.lit(m, t, "1"), text.num(m, t, 0 - 42));
    var ok = text.is(t, s, "a1b-42c");
    let r = text.replace_lit(m, t, text.lit(m, t, "x@I@y@I@"), "@I@", text.lit(m, t, "(e)"));
    ok = ok && text.is(t, r, "x(e)y(e)");
    ok = ok && text.contains(m, t, text.lit(m, t, "gid2 = 1"), "gid2");
    ok = ok && !text.contains(m, t, text.lit(m, t, "gid = 1"), "gid2");
    let l = mem.of3(m, text.lit(m, t, "a"), text.lit(m, t, "b"), text.lit(m, t, "c"));
    ok = ok && text.is(t, text.joined(m, t, l, ", "), "a, b, c");
    return ok;
}

fn with_memory[&h](heap: &!h Heap, which: int) -> [heap] int {
    var mb = box_slice(heap, words(), 0);
    var tb = box_slice(heap, 1000000, byte_of(0));
    var ok = false;
    borrow mut mb as &!mr in {
        borrow mut tb as &!tr in {
            let m = contents(mr);
            let t = contents(tr);
            mem.init(m);
            if which == 0 {
                ok = check_printing(m, t);
            } else if which == 1 {
                ok = check_arithmetic(m, t);
            } else if which == 2 {
                ok = check_parsing(m, t);
            } else if which == 3 {
                ok = check_counts(m, t);
            } else {
                ok = check_text(m, t);
            }
        }
    }
    unbox_slice(heap, mb);
    unbox_slice(heap, tb);
    if ok {
        return 0;
    }
    return 1;
}

fn test_f32_printing_matches_rust[&h](heap: &!h Heap) -> [heap] int {
    return with_memory(heap, 0);
}

fn test_f32_operations_round_once[&h](heap: &!h Heap) -> [heap] int {
    return with_memory(heap, 1);
}

fn test_numbers_parse_as_rust_reads_them[&h](heap: &!h Heap) -> [heap] int {
    return with_memory(heap, 2);
}

fn test_kernel_name_counts_saturate_like_usize[&h](heap: &!h Heap) -> [heap] int {
    return with_memory(heap, 3);
}

fn test_the_string_pool[&h](heap: &!h Heap) -> [heap] int {
    return with_memory(heap, 4);
}
