//! The oracle of `cancho-gpu gguf`: lex-rt's reader, printing the canonical
//! dump of `src/gguf_dump.cho` (docs/runtime.md §4).
//!
//! It is not part of lex-gpu: `scripts/regen-gguf-golden.sh` copies it into
//! `crates/lex-rt/examples/` for the build and removes it after.
//!
//!     gguf_dump <file.gguf> [--sum]

use lex_rt::gguf::{Gguf, GgmlType, Value};
use std::path::Path;

fn quoted(s: &str) -> String {
    let mut o = String::from("\"");
    for &b in s.as_bytes() {
        match b {
            b'"' | b'\\' => {
                o.push('\\');
                o.push(b as char);
            }
            0..=31 | 127..=255 => o.push_str(&format!("\\x{b:02x}")),
            _ => o.push(b as char),
        }
    }
    o.push('"');
    o
}

fn value(v: &Value) -> String {
    match v {
        Value::Int(i) => format!("int {i}"),
        Value::Float(f) => format!("float 0x{:016x}", f.to_bits()),
        Value::Bool(b) => format!("bool {b}"),
        Value::Str(s) => format!("str {}", quoted(s)),
        Value::Array(xs) => {
            let first: Vec<String> = xs.iter().take(4).map(value).collect();
            let more = if xs.len() > 4 { ", ..." } else { "" };
            format!("array n={} [{}{}]", xs.len(), first.join(", "), more)
        }
    }
}

fn type_name(t: GgmlType) -> String {
    match t {
        GgmlType::F32 => "F32".into(),
        GgmlType::F16 => "F16".into(),
        GgmlType::BF16 => "BF16".into(),
        GgmlType::Q8_0 => "Q8_0".into(),
        GgmlType::Q4_K => "Q4_K".into(),
        GgmlType::Q6_K => "Q6_K".into(),
        GgmlType::PQ2_0 => "PQ2_0".into(),
        GgmlType::PTQ1_0 => "PTQ1_0".into(),
        GgmlType::Other(x) => format!("other({x})"),
    }
}

fn block_bytes(t: GgmlType) -> Option<(usize, usize)> {
    match t {
        GgmlType::F32 => Some((1, 4)),
        GgmlType::F16 | GgmlType::BF16 => Some((1, 2)),
        GgmlType::Q8_0 => Some((32, 34)),
        GgmlType::Q4_K => Some((256, 144)),
        GgmlType::Q6_K => Some((256, 210)),
        GgmlType::PQ2_0 => Some((128, 34)),
        GgmlType::PTQ1_0 => Some((128, 28)),
        GgmlType::Other(_) => None,
    }
}

fn fnv32(b: &[u8]) -> u32 {
    let mut h: u32 = 2166136261;
    for &x in b {
        h = (h ^ x as u32).wrapping_mul(16777619);
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sum = args.iter().any(|a| a == "--sum");
    let g = match Gguf::open(Path::new(&args[1])) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    // The version is not kept by the reader; the file says it.
    let raw = std::fs::read(&args[1]).unwrap();
    let version = u32::from_le_bytes(raw[4..8].try_into().unwrap());
    let alignment = g.meta.get("general.alignment").and_then(Value::as_int).unwrap_or(32);
    println!(
        "gguf v{version} tensors={} kv={} alignment={alignment}",
        g.tensors.len(),
        g.meta.len()
    );
    let mut keys: Vec<&String> = g.meta.keys().collect();
    keys.sort();
    for k in keys {
        println!("kv {k} {}", value(&g.meta[k]));
    }
    let mut names: Vec<&String> = g.tensors.keys().collect();
    names.sort();
    for n in names {
        let t = &g.tensors[n];
        let dims: Vec<String> = t.dims.iter().map(|d| d.to_string()).collect();
        let bytes = match block_bytes(t.ty) {
            Some((per, size)) => (t.elems() / per * size).to_string(),
            None => "unsupported".into(),
        };
        let mut line = format!("tensor {n} type={} dims=[{}] bytes={bytes}", type_name(t.ty), dims.join(","));
        if sum && block_bytes(t.ty).is_some() {
            let (_, b) = g.raw(n).unwrap();
            line.push_str(&format!(" sum={:08x}", fnv32(b)));
        }
        println!("{line}");
    }
}
