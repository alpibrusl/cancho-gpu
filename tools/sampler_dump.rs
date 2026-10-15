//! The oracle of `cancho-gpu sample`: lex-rt's `Sampler`, printing the form of
//! `src/sample.cho` (docs/runtime.md §10).
//!
//! It is not part of lex-gpu: `scripts/regen-sampler-golden.sh` copies it into
//! `crates/lex-rt/examples/` for the build and removes it after.
//!
//!     sampler_dump <logits.f32> <vocab> <temperature> <top_p> <top_k> <seed>

use lex_rt::sample::Sampler;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let raw = std::fs::read(&a[1]).expect("logits file");
    let vocab: usize = a[2].parse().expect("vocab");
    // The arguments are read as binary64 and rounded, as cancho-gpu reads them.
    let temperature = a[3].parse::<f64>().expect("temperature") as f32;
    let top_p = a[4].parse::<f64>().expect("top_p") as f32;
    let top_k: usize = a[5].parse().expect("top_k");
    let seed: u64 = a[6].parse().expect("seed");
    let mut s = Sampler::new(temperature, top_p, top_k, seed);
    let words: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    for (row, logits) in words.chunks(vocab).enumerate() {
        let (ids, probs) = s.nucleus(logits);
        let pick = s.pick(logits);
        let parts: Vec<String> = ids
            .iter()
            .zip(&probs)
            .map(|(i, p)| {
                if p.is_nan() {
                    format!("{i}=nan")
                } else {
                    format!("{i}={:08x}", p.to_bits())
                }
            })
            .collect();
        println!("row {row} pick {pick} nucleus {}: {}", ids.len(), parts.join(" "));
    }
}
