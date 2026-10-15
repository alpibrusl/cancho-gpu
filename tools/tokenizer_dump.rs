//! The oracle of `cancho-gpu tokenize`: lex-rt's tokenizer, printing the
//! canonical form of `src/tokenize.cho` (docs/runtime.md §9).
//!
//! It is not part of lex-gpu: `scripts/regen-tokenizer-golden.sh` copies it
//! into `crates/lex-rt/examples/` for the build and removes it after.
//!
//!     tokenizer_dump <tokenizer.json | model.gguf> <corpus>
//!
//! The corpus is records separated by a line holding only `%%`.

use lex_rt::gguf::Gguf;
use lex_rt::tokenizer::Tokenizer;
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read(&args[1]).expect("tokenizer file");
    let tok = if raw.starts_with(b"GGUF") {
        Tokenizer::from_gguf(&Gguf::open_header(Path::new(&args[1])).expect("gguf header"))
    } else {
        Tokenizer::from_json(std::str::from_utf8(&raw).expect("utf-8 tokenizer.json"))
    };
    let tok = match tok {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let corpus = std::fs::read_to_string(&args[2]).expect("utf-8 corpus");
    println!("tokenizer ids={}", tok.len());
    for record in corpus.split("\n%%\n") {
        let ids = tok.encode(record);
        let list: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
        println!("ids {}: {}", ids.len(), list.join(" "));
        println!("dec {}", quoted(&tok.decode(&ids)));
    }
}
