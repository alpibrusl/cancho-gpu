# tests/kolibri_toy.sh -- sourced by tests/kolibri_host.sh and tests/kolibri_cmd.sh: `toy`, which writes with awk, byte by byte, a three-layer GGUF with Kolibri 1.1's metadata and
# tensors (32 wide: a dense layer, a mixture of 2 experts with its own keys, a mixture that reads layer 1's keys; 2 query heads over 1 key/value head of 16; a vocabulary of 64; end tokens 5
# and 6). It needs $T, a directory to work in.
toy() { # out file, variant (none | notensor | nokv | f32q | wide | v8 | notop | w65 | noeos): the toy GGUF, and for `none` the expected files under $T/want with $T/manifest
awk -v T="$T" -v VAR="$2" '
function rb(k, i) { return (i * 37 + 11 + 101 * int(i / 61) + 53 * k) % 256 }
function tb(k, i,   j) { # byte i of tensor k: any byte, but an f16 field never has all its exponent bits set (no infinity, no NaN)
  if (TY[k] == 2) { j = i % 18; return j == 1 ? rb(k, i) - (int(rb(k, i) / 4) % 2) * 4 : rb(k, i) }
  if (TY[k] == 3) { j = i % 20; return (j == 1 || j == 3) ? rb(k, i) - (int(rb(k, i) / 4) % 2) * 4 : rb(k, i) }
  if (TY[k] == 14) { j = i % 210; return j == 209 ? rb(k, i) - (int(rb(k, i) / 4) % 2) * 4 : rb(k, i) }
  return rb(k, i)
}
function half(lo, hi,   s, e, fr) { s = hi >= 128 ? -1 : 1; e = int((hi % 128) / 4); fr = (hi % 4) * 256 + lo; return e == 0 ? s * fr / 1024 * 2^-14 : s * (1 + fr / 1024) * 2^(e - 15) }
function out(b) { printf "%d\n", b > BYTES; NB++ }
function u32(v,   i) { for (i = 0; i < 4; i++) { out(v % 256); v = int(v / 256) } }
function u64(v) { u32(v); u32(0) }
function str(s,   i) { u64(length(s)); for (i = 1; i <= length(s); i++) out(ORD[substr(s, i, 1)]) }
function kvu(name, v) { if (VAR == "nokv" && name == "kolibri1.expert_count") name = "kolibri1.expert_kount"; str(name); u32(4); u32(v) }
function add(name, ty, d0, d1, d2,   n) { # a tensor: ggml type 0 F32, 2 Q4_0, 3 Q4_1, 14 Q6_K; dims as ggml has them (the row length first)
  if (VAR == "notensor" && name == "blk.1.attn_k.weight") name = "blk.1.attn_kk.weight"
  if (VAR == "f32q" && name == "blk.0.attn_output.weight") ty = 0
  if (VAR == "wide" && name == "blk.2.attn_q.weight") { d0 = 16; d1 = d1 * 2 }
  n = d0 * (d1 ? d1 : 1) * (d2 ? d2 : 1); NAME[NT] = name; TY[NT] = ty; D0[NT] = d0; D1[NT] = d1; D2[NT] = d2; EL[NT] = n; ID[name] = NT
  SZ[NT] = ty == 0 ? 4 * n : (ty == 2 ? n / 32 * 18 : (ty == 3 ? n / 32 * 20 : n / 256 * 210)); OFF[NT] = DATA; DATA += SZ[NT]; DATA += (32 - DATA % 32) % 32; NT++
}
function want(file, kind) { W = T "/want/" file; printf "%s %s\n", file, kind > (T "/manifest") }
function copy(name, file,   k, i) { want(file, "bytes"); k = ID[name]; for (i = 0; i < SZ[k]; i++) printf "%d\n", tb(k, i) > W; close(W) }
function q4(names, n, file,   a, k, bw, b, q, i, v, any) { # the tensors names[1..n] one after another: codes, scales, and offsets where Q4_1
  want(file ".q4", "bytes"); for (a = 1; a <= n; a++) { k = ID[names[a]]; bw = TY[k] == 3 ? 20 : 18
    for (b = 0; b < EL[k] / 32; b++) { q = b * bw + bw - 16; for (i = 0; i < 32; i++) v[i] = i < 16 ? tb(k, q + i) % 16 : int(tb(k, q + i - 16) / 16); for (i = 0; i < 16; i++) printf "%d\n", v[2 * i] + 16 * v[2 * i + 1] > W } } close(W)
  want(file ".s", "bytes"); for (a = 1; a <= n; a++) { k = ID[names[a]]; bw = TY[k] == 3 ? 20 : 18; for (b = 0; b < EL[k] / 32; b++) { printf "%d\n%d\n", tb(k, b * bw), tb(k, b * bw + 1) > W } } close(W)
  any = 0; for (a = 1; a <= n; a++) if (TY[ID[names[a]]] == 3) any = 1
  if (any) { want(file ".c", "vals"); for (a = 1; a <= n; a++) { k = ID[names[a]]; for (b = 0; b < EL[k] / 32; b++) printf "%.9g\n", half(tb(k, b * 20 + 2), tb(k, b * 20 + 3)) + 8 * half(tb(k, b * 20), tb(k, b * 20 + 1)) > W } close(W) }
}
function q41(name, file) { ONE[1] = name; q4(ONE, 1, file) }
function mlp(l, suffix, pre,   a) { q41("blk." l ".ffn_gate" suffix ".weight", "m" l "." pre ".gate"); q41("blk." l ".ffn_up" suffix ".weight", "m" l "." pre ".up"); q41("blk." l ".ffn_down" suffix ".weight", "m" l "." pre ".down") }
function fillw(file, n, v, ramp,   i) { want(file, "vals"); for (i = 0; i < n; i++) printf "%s\n", ramp ? i : v > W; close(W) }
BEGIN {
  NT = 0; DATA = 0; NB = 0; BYTES = T "/bytes.txt"; for (i = 1; i < 128; i++) ORD[sprintf("%c", i)] = i
  D = 32; NQ = 2; NKV = 1; HD = 16; NE = 2; EH = 32; FF = 64; V = (VAR == "v8") ? 8 : 64; NL = 3; WIN = (VAR == "w65") ? 65 : 2
  for (l = 0; l < NL; l++) { p = "blk." l "."
    add(p "attn_norm.weight", 0, D); add(p "attn_q_norm.weight", 0, HD); add(p "post_attention_norm.weight", 0, D); add(p "ffn_norm.weight", 0, D); add(p "post_ffw_norm.weight", 0, D)
    add(p "attn_q.weight", 2, D, NQ * HD); add(p "attn_output.weight", 2, NQ * HD, D)
    if (l < 2) { add(p "attn_k_norm.weight", 0, HD); add(p "attn_k.weight", 2, D, NKV * HD); add(p "attn_v.weight", 2, D, NKV * HD) }
    if (l == 0) { add(p "ffn_gate.weight", 2, D, FF); add(p "ffn_up.weight", 2, D, FF); add(p "ffn_down.weight", 3, FF, D) }
    else { add(p "ffn_gate_inp.weight", 0, D, NE); add(p "exp_probs_b.bias", 0, NE)
      add(p "ffn_gate_shexp.weight", 2, D, EH); add(p "ffn_up_shexp.weight", 2, D, EH); add(p "ffn_down_shexp.weight", l == 1 ? 3 : 2, EH, D)
      add(p "ffn_gate_exps.weight", 2, D, EH, NE); add(p "ffn_up_exps.weight", 2, D, EH, NE); add(p "ffn_down_exps.weight", l == 1 ? 3 : 2, EH, D, NE) }
  }
  add("token_embd.weight", 2, D, V); add("output.weight", 14, D, V); add("output_norm.weight", 0, D)
  # the header
  out(71); out(71); out(85); out(70); u32(3); u64(NT); u64(VAR == "notop" || VAR == "noeos" ? 17 : 18)
  kvu("kolibri1.block_count", NL); kvu("kolibri1.leading_dense_block_count", 1); kvu("kolibri1.embedding_length", D); kvu("kolibri1.expert_count", NE); kvu("kolibri1.expert_used_count", 1)
  kvu("kolibri1.expert_feed_forward_length", EH); kvu("kolibri1.feed_forward_length", FF); kvu("kolibri1.attention.head_count", NQ); kvu("kolibri1.attention.head_count_kv", NKV)
  kvu("general.sampling.top_k", 2); kvu("kolibri1.attention.key_length", HD); kvu("kolibri1.attention.sliding_window", WIN); if (VAR != "notop") kvu("kolibri1.attention.indexer.top_k", 0); kvu("general.alignment", 32)
  if (VAR != "noeos") kvu("tokenizer.ggml.eos_token_id", 5); kvu("tokenizer.ggml.padding_token_id", 6)
  str("kolibri1.attention.kv_source_map"); u32(9); u32(5); u64(NL); u32(0); u32(1); u32(1)
  str("kolibri1.rope.freq_base"); u32(6); out(0); out(64); out(28); out(70)       # 10000 as binary32
  for (k = 0; k < NT; k++) { str(NAME[k]); nd = 1 + (D1[k] > 0) + (D2[k] > 0); u32(nd); u64(D0[k]); if (D1[k]) u64(D1[k]); if (D2[k]) u64(D2[k]); u32(TY[k]); u64(OFF[k]) }
  while (NB % 32) out(0)
  for (k = 0; k < NT; k++) { for (i = 0; i < SZ[k]; i++) out(tb(k, i)); while (NB % 32) out(0) }
  if (VAR != "none") exit
  # what `weights` writes
  fillw("ones_d.f32", D, 1); fillw("ones_h.f32", EH, 1); fillw("ones_f.f32", FF, 1); fillw("ones_qkv.f32", (NQ + 2 * NKV) * HD, 1); fillw("ones_q.f32", NQ * HD, 1); fillw("one64.f32", HD / 2, 1)
  fillw("zero64.f32", HD / 2, 0); fillw("ar.f32", NE, 0, 1); fillw("sl.f32", 1, 0, 1)
  n = 0; for (k = 1; k < 8; k++) { ev[n] = k * 2^-9; od[n++] = k % 2 } for (e = -6; e < 8; e++) for (j = 0; j < 8; j++) { ev[n] = (1 + j / 8) * 2^e; od[n++] = j % 2 } for (j = 0; j < 7; j++) { ev[n] = (1 + j / 8) * 256; od[n++] = j % 2 }
  want("e4m3.f32", "vals"); for (i = 0; i < 125; i++) printf "%.17g\n", (ev[i] + ev[i + 1]) / 2 > W; for (i = 0; i < 125; i++) printf "%.17g\n", ev[i + 1] - ev[i] > W; for (i = 0; i < 125; i++) printf "%d\n", od[i] > W; close(W)
  copy("output_norm.weight", "fnorm.f32")
  want("head.f32", "vals"); k = ID["output.weight"]
  for (n = 0; n < V * D; n++) { o = int(n / 256) * 210; h = int((n % 256) / 128); r = n % 128; kq = int(r / 32); l = r % 32; qb = tb(k, o + h * 64 + l + (kq % 2 ? 32 : 0)); nib = kq < 2 ? qb % 16 : int(qb / 16); top = int(tb(k, o + 128 + h * 32 + l) / 4^kq) % 4
    s = tb(k, o + 192 + h * 8 + int(l / 16) + 2 * kq); if (s > 127) s -= 256; printf "%.9g\n", half(tb(k, o + 208), tb(k, o + 209)) * s * (nib + 16 * top - 32) > W } close(W)
  q41("token_embd.weight", "emb"); want("rank.f32", "vals"); for (i = 0; i < V; i++) printf "%d\n", V - i > W; close(W); fillw("zmk.f32", V, 0); fillw("one1.f32", 1, 1)
  # for the sampler: the file says the 2 largest, made 24 chunks
  fillw("ciota.f32", V / 64, 0, 1); fillw("cslot.f32", 24, 0, 1); fillw("czero.f32", 24 * 64, 0)
  want("ctri.f32", "vals"); for (s = 0; s < 24; s++) for (c = 0; c < 24; c++) printf "%d\n", (c <= s) > W; close(W)
  want("tri64.f32", "vals"); for (s = 0; s < 64; s++) for (c = 0; c < 64; c++) printf "%d\n", (c <= s) > W; close(W)
  for (l = 0; l < NL; l++) { p = "blk." l "."
    copy(p "attn_norm.weight", "a" l ".norm.f32"); copy(p "attn_q_norm.weight", "a" l ".qn.f32"); copy(p "post_attention_norm.weight", "a" l ".pn.f32"); copy(p "ffn_norm.weight", "m" l ".ffn_norm.f32"); copy(p "post_ffw_norm.weight", "m" l ".post_norm.f32")
    if (l < 2) { copy(p "attn_k_norm.weight", "a" l ".kn.f32"); THREE[1] = p "attn_q.weight"; THREE[2] = p "attn_k.weight"; THREE[3] = p "attn_v.weight"; q4(THREE, 3, "a" l ".qkv") } else q41(p "attn_q.weight", "a" l ".q")
    q41(p "attn_output.weight", "a" l ".o")
    if (l == 0) mlp(l, "", "dn"); else { copy(p "ffn_gate_inp.weight", "m" l ".route.f32"); copy(p "exp_probs_b.bias", "m" l ".bias.f32"); mlp(l, "_shexp", "sh"); mlp(l, "_exps", "ex") }
  }
  # what `inputs` writes for the ids 3 0 7, five steps and caches of 64 positions (after the line `inputs` of the manifest)
  printf "inputs -\n" > (T "/manifest"); NP = 3; NS = 5; split("3 0 7", IDS, " "); k = ID["token_embd.weight"]
  want("x0.f32", "vals"); for (s = 1; s <= NS; s++) for (b = 0; b < D / 32; b++) { o = IDS[s] * (D / 32) * 18 + b * 18; for (i = 0; i < 32; i++) { c = i < 16 ? tb(k, o + 2 + i) % 16 : int(tb(k, o + 2 + i - 16) / 16); printf "%.9g\n", (s <= NP ? (c - 8) * half(tb(k, o), tb(k, o + 1)) : 0) > W } } close(W)
  fillw("id0.f32", 1, 0)
  # the uniform numbers of seed 7, which `draws` prints (not a file of `inputs`, so not in the manifest): x becomes (1103515245 x + 12345) mod 2^31 (in pieces: the product does not fit a
  # double), the upper 24 bits over 2^24
  W = T "/want_draws.txt"; xx = 7; for (s = 0; s < NS; s++) { xh = int(xx / 65536); xl = xx % 65536; xx = ((((1103515245 * xh) % 2147483648) * 65536) % 2147483648 + (1103515245 * xl) % 2147483648 + 12345) % 2147483648; printf "%.24f\n", int(xx / 128) / 16777216 > W } close(W)
  want("pos.f32", "vals"); for (s = 0; s < NS; s++) printf "%d\n", s > W; close(W)
  want("iota.f32", "vals"); for (c = 0; c < 64; c++) printf "%d\n", c > W; close(W)
  want("maskw.f32", "vals"); for (s = 0; s < 64; s++) for (c = 0; c < 64; c++) printf "%s\n", ((c <= s && c > s - WIN) ? 0 : -1e9) > W; close(W)
  want("cos.f32", "vals"); for (s = 0; s < NS; s++) for (c = 0; c < HD / 2; c++) printf "%.9g\n", cos(s / 10000 ^ (c / (HD / 2))) > W; close(W)
  want("sin.f32", "vals"); for (s = 0; s < NS; s++) for (c = 0; c < HD / 2; c++) printf "%.9g\n", sin(s / 10000 ^ (c / (HD / 2))) > W; close(W)
}'
awk '{ printf "\\%03o", $1; if (NR % 512 == 0) printf "\n" } END { printf "\n" }' "$T/bytes.txt" | while IFS= read -r line; do printf "$line"; done > "$1"
rm "$T/bytes.txt"
}
