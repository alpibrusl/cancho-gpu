# A hand-written flash attention in Metal, to measure a ceiling (docs/diffusion.md D16)

Not part of the build or the gate. `flash.metal` is a flash attention (online softmax, the scores never written, the output and the running maximum and sum in registers) for the edit step of Qwen-Image-2.1-Turbo: half `q`, `k`, `v` token-major, an additive float mask `[nq, t]`, 128-wide heads, four simdgroups of eight query rows each, a chunk of keys through threadgroup memory. `flash2.metal` has `RS` atom rows per simdgroup. `bench.mm` compiles one of them with function constants and macros and times it on the GPU.

    clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation bench.mm -o bench
    ./bench 2 256 512 check                  # heads, queries, keys: correctness against float64 attention (every 7th row)
    ./bench 32 4160 8320 x NOMASK BK=16      # the edit step's shape; macros: NOMASK, FASTEXP, BK, PAD
    SRC=flash2.metal NSG=4 QB=64 ./bench 32 4160 8320 x NOMASK RS=2 NSG=4 BK=16

The element layout of an 8 x 8 `simdgroup_matrix` (each lane holds (fm, fn) and (fm, fn + 1), `fm = (qid & 4) + ((lane / 2) % 4)`, `fn = (qid & 2) * 2 + (lane % 2) * 2`, `qid = lane / 4`; a row's four lanes meet with `simd_shuffle_xor` 1 and 8) is confirmed by the check (9e-5 from float64). Results and what they say: docs/diffusion.md D16.
