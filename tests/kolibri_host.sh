#!/bin/sh
# tests/kolibri_host.sh -- the host side of Kolibri 1.1's step (tests/kolibri_gpu.cho, docs/kolibri.md K5) without the model: it builds, and on a toy GGUF written here byte by byte (three layers 32
# wide: a dense one, a mixture of 2 experts with its own keys, a mixture that reads layer 1's keys; 2 query heads over 1 key/value head of 16; a vocabulary of 64) every file `weights` and `inputs`
# write is what awk's arithmetic of the formats says: the copied tensors, the repacked codes and the scales to the byte, the Q4_1 offsets, the decoded Q6_K head, the embedding rows and the tables
# as numbers; the list of files is exactly the expected one. Then `argmax`, and the refusals, each with its rule and status 1 (2 for usage), none a trap.
set -eu
cd "$(dirname "$0")/.."
C="${CANCHO:-cancho}"
CANCHO="$C" scripts/build-kolibri.sh
B=build/kolibri_gpu
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
fail=0
: > "$T/fails"   # a line a failure of the two big checks, so that an `ok` is said only of a part that had none
toy() { # out file, variant (none | notensor | nokv | f32q | wide | v8): the toy GGUF, and for `none` the expected files under $T/want with $T/manifest
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
  D = 32; NQ = 2; NKV = 1; HD = 16; NE = 2; EH = 32; FF = 64; V = (VAR == "v8") ? 8 : 64; NL = 3; WIN = 2
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
  out(71); out(71); out(85); out(70); u32(3); u64(NT); u64(14)
  kvu("kolibri1.block_count", NL); kvu("kolibri1.leading_dense_block_count", 1); kvu("kolibri1.embedding_length", D); kvu("kolibri1.expert_count", NE); kvu("kolibri1.expert_used_count", 1)
  kvu("kolibri1.expert_feed_forward_length", EH); kvu("kolibri1.feed_forward_length", FF); kvu("kolibri1.attention.head_count", NQ); kvu("kolibri1.attention.head_count_kv", NKV)
  kvu("kolibri1.attention.key_length", HD); kvu("kolibri1.attention.sliding_window", WIN); kvu("general.alignment", 32)
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
  for (l = 0; l < NL; l++) { p = "blk." l "."
    copy(p "attn_norm.weight", "a" l ".norm.f32"); copy(p "attn_q_norm.weight", "a" l ".qn.f32"); copy(p "post_attention_norm.weight", "a" l ".pn.f32"); copy(p "ffn_norm.weight", "m" l ".ffn_norm.f32"); copy(p "post_ffw_norm.weight", "m" l ".post_norm.f32")
    if (l < 2) { copy(p "attn_k_norm.weight", "a" l ".kn.f32"); THREE[1] = p "attn_q.weight"; THREE[2] = p "attn_k.weight"; THREE[3] = p "attn_v.weight"; q4(THREE, 3, "a" l ".qkv") } else q41(p "attn_q.weight", "a" l ".q")
    q41(p "attn_output.weight", "a" l ".o")
    if (l == 0) mlp(l, "", "dn"); else { copy(p "ffn_gate_inp.weight", "m" l ".route.f32"); copy(p "exp_probs_b.bias", "m" l ".bias.f32"); mlp(l, "_shexp", "sh"); mlp(l, "_exps", "ex") }
  }
  # what `inputs` writes for the ids 3 0 7, five steps and caches of 64 positions (after the line `inputs` of the manifest)
  printf "inputs -\n" > (T "/manifest"); NP = 3; NS = 5; split("3 0 7", IDS, " "); k = ID["token_embd.weight"]
  want("x0.f32", "vals"); for (s = 1; s <= NS; s++) for (b = 0; b < D / 32; b++) { o = IDS[s] * (D / 32) * 18 + b * 18; for (i = 0; i < 32; i++) { c = i < 16 ? tb(k, o + 2 + i) % 16 : int(tb(k, o + 2 + i - 16) / 16); printf "%.9g\n", (s <= NP ? (c - 8) * half(tb(k, o), tb(k, o + 1)) : 0) > W } } close(W)
  want("idtab.f32", "vals"); for (s = 1; s <= NS; s++) printf "%d\n", (s <= NP ? IDS[s] : 0) > W; close(W)
  want("mtab.f32", "vals"); for (s = 1; s <= NS; s++) printf "%d\n", (s <= NP) > W; close(W)
  want("accsel.f32", "vals"); for (s = 0; s < NS; s++) for (c = 0; c < NS; c++) printf "%d\n", (c == s) > W; close(W)
  want("acckeep.f32", "vals"); for (s = 0; s < NS; s++) for (c = 0; c < NS; c++) printf "%d\n", (c != s) > W; close(W)
  fillw("acc0.f32", NS, 0); fillw("id0.f32", 1, 0)
  want("sel.f32", "vals"); for (s = 0; s < NS; s++) for (c = 0; c < 64; c++) printf "%d\n", (c == s) > W; close(W)
  want("keep.f32", "vals"); for (s = 0; s < NS; s++) for (c = 0; c < 64; c++) printf "%d\n", (c != s) > W; close(W)
  want("mask.f32", "vals"); for (s = 0; s < NS; s++) for (c = 0; c < 64; c++) printf "%s\n", ((c <= s && c > s - WIN) ? 0 : -1e9) > W; close(W)
  want("maskg.f32", "vals"); for (s = 0; s < NS; s++) for (c = 0; c < 64; c++) printf "%s\n", (c <= s ? 0 : -1e9) > W; close(W)
  want("cos.f32", "vals"); for (s = 0; s < NS; s++) for (c = 0; c < HD / 2; c++) printf "%.9g\n", cos(s / 10000 ^ (c / (HD / 2))) > W; close(W)
  want("sin.f32", "vals"); for (s = 0; s < NS; s++) for (c = 0; c < HD / 2; c++) printf "%.9g\n", sin(s / 10000 ^ (c / (HD / 2))) > W; close(W)
  want("zcache.f32", "vals"); for (c = 0; c < NKV * HD * 64; c++) printf "0\n" > W; close(W)
}'
awk '{ printf "\\%03o", $1; if (NR % 512 == 0) printf "\n" } END { printf "\n" }' "$T/bytes.txt" | while IFS= read -r line; do printf "$line"; done > "$1"
rm "$T/bytes.txt"
}
mkdir "$T/want" "$T/w" "$T/x"
toy "$T/toy.gguf" none
check() { # dir, from manifest line, to manifest line (exclusive): every file as wanted, and no other file in dir
  sed -n "$2,$(($3 - 1))p" "$T/manifest" > "$T/part"
  while read -r file kind; do
    if [ ! -f "$1/$file" ]; then echo "FAIL $file is not written"; fail=1; echo x >> "$T/fails"; continue; fi
    if [ "$kind" = bytes ]; then od -A n -v -t u1 "$1/$file" | tr -s ' \n' '\n\n' | grep -v '^$' > "$T/got.txt" || true; tol=0; else od -A n -v -t fF "$1/$file" | tr -s ' \n' '\n\n' | grep -v '^$' > "$T/got.txt" || true; tol=2e-6; fi
    r=$(paste "$T/got.txt" "$T/want/$file" | awk -v tol="$tol" -v n="$(wc -l < "$T/want/$file")" 'NF != 2 { bad = "a line of " NF " fields"; exit } { d = $1 - $2; if (d < 0) d = -d; s = ($2 < 0 ? -$2 : $2); if (d > tol * s) { bad = "value " NR ": " $1 ", wanted " $2; exit } } END { if (bad != "") print bad; else if (NR != n) print NR " values, wanted " n; else print "ok" }')
    if [ "$r" != ok ]; then echo "FAIL $file: $r"; fail=1; echo x >> "$T/fails"; fi
  done < "$T/part"
  if [ "$(ls "$1" | sort | tr '\n' ' ')" != "$(cut -d' ' -f1 "$T/part" | sort | tr '\n' ' ')" ]; then echo "FAIL the files of $1 are not the expected ones: $(ls "$1" | sort | tr '\n' ' ')"; fail=1; echo x >> "$T/fails"; fi
}
split_at=$(grep -n '^inputs -$' "$T/manifest" | cut -d: -f1)
total=$(wc -l < "$T/manifest")
before=$(cat "$T/fails")
if "$B" weights "$T/toy.gguf" "$T/w" > "$T/out.txt" 2> "$T/err.txt"; then check "$T/w" 1 "$split_at"; else echo "FAIL weights: $(head -c 300 "$T/err.txt")"; fail=1; echo x >> "$T/fails"; fi
if [ "$(tr '\n' ';' < "$T/out.txt")" != "layer 0: its own keys, a dense MLP;layer 1: its own keys, a mixture of experts;layer 2: reads another layer's keys, a mixture of experts;" ]; then echo "FAIL weights says: $(cat "$T/out.txt")"; fail=1; echo x >> "$T/fails"; fi
[ "$(cat "$T/fails")" = "$before" ] && echo "ok   weights: $((split_at - 1)) files of a three-layer toy, each as awk has it, and no other"
before=$(cat "$T/fails")
printf '3 0\n7\n' > "$T/ids.txt"
if "$B" inputs "$T/toy.gguf" "$T/ids.txt" "$T/x" 64 5 > "$T/out.txt" 2> "$T/err.txt"; then check "$T/x" $((split_at + 1)) $((total + 1)); else echo "FAIL inputs: $(head -c 300 "$T/err.txt")"; fail=1; echo x >> "$T/fails"; fi
[ "$(cat "$T/out.txt")" = "steps 5 prompt 3" ] || { echo "FAIL inputs says: $(cat "$T/out.txt")"; fail=1; echo x >> "$T/fails"; }
[ "$(cat "$T/fails")" = "$before" ] && echo "ok   inputs: the embedding rows of three ids, two steps past them, and the thirteen tables"
# argmax: the largest of the decoded head by awk, the first of equal values, a file of one value
want=$(awk '{ if (NR == 1 || $1 > m) { m = $1; i = NR - 1 } } END { print i }' "$T/want/head.f32")
got=$("$B" argmax "$T/w/head.f32" 2>&1 | cut -d' ' -f1-2)
if [ "$got" = "argmax $want" ]; then echo "ok   argmax of the head: $want"; else echo "FAIL argmax of the head: $got, wanted $want"; fail=1; fi
got=$("$B" argmax "$T/w/ones_d.f32" 2>&1)
if [ "$got" = "argmax 0 value 1.0 margin 0.0" ]; then echo "ok   argmax of equal values is the first"; else echo "FAIL argmax of equal values: $got"; fail=1; fi
got=$("$B" argmax "$T/w/sl.f32" 2>&1)
if [ "$got" = "argmax 0 value 0.0 margin none" ]; then echo "ok   argmax of one value"; else echo "FAIL argmax of one value: $got"; fail=1; fi
got=$("$B" ids "$T/x/idtab.f32" 1 2>&1)
if [ "$got" = "0 7 0 0" ]; then echo "ok   ids from an index on"; else echo "FAIL ids: $got"; fail=1; fi
refuse() { # name, rule, status, the command
  name=$1; rule=$2; want=$3; shift 3
  status=0; "$@" > "$T/out.txt" 2> "$T/err.txt" || status=$?
  if [ "$status" -eq "$want" ] && grep -q "^error\[$rule\]" "$T/err.txt"; then echo "ok   refused ($rule): $name"; else echo "FAIL $name: status $status, wanted $want and error[$rule]: $(head -c 300 "$T/err.txt")"; fail=1; fi
}
mkdir "$T/r"
refuse "no command" kolibri-usage 2 "$B"
refuse "an argument too many" kolibri-usage 2 "$B" weights "$T/toy.gguf" "$T/r" more
refuse "a file that is not there" kolibri-read 1 "$B" weights "$T/none.gguf" "$T/r"
printf 'not a GGUF file at all, but long enough to have a header' > "$T/bad.gguf"
refuse "a file that is not a GGUF" gguf-magic 1 "$B" weights "$T/bad.gguf" "$T/r"
head -c 1000 "$T/toy.gguf" > "$T/cut.gguf"
refuse "a header cut short" gguf-truncated 1 "$B" weights "$T/cut.gguf" "$T/r"
head -c $(($(wc -c < "$T/toy.gguf") - 600)) "$T/toy.gguf" > "$T/cut2.gguf"
refuse "the data cut short" kolibri-read 1 "$B" weights "$T/cut2.gguf" "$T/r"
toy "$T/v.gguf" notensor
refuse "a tensor that is not there" kolibri-tensor 1 "$B" weights "$T/v.gguf" "$T/r"
toy "$T/v.gguf" nokv
refuse "a count missing from the metadata" kolibri-meta 1 "$B" weights "$T/v.gguf" "$T/r"
toy "$T/v.gguf" f32q
refuse "a matrix that is F32 where 4-bit is read" kolibri-tensor 1 "$B" weights "$T/v.gguf" "$T/r"
toy "$T/v.gguf" wide
refuse "a 4-bit matrix whose rows are not a multiple of 32" kolibri-tensor 1 "$B" weights "$T/v.gguf" "$T/r"
refuse "a directory that is not there" kolibri-write 1 "$B" weights "$T/toy.gguf" "$T/nowhere"
printf '3 x 7\n' > "$T/bad.txt"
refuse "an id that is not a number" kolibri-ids 1 "$B" inputs "$T/toy.gguf" "$T/bad.txt" "$T/r" 64
printf '3 64\n' > "$T/bad.txt"
refuse "an id past the vocabulary" kolibri-ids 1 "$B" inputs "$T/toy.gguf" "$T/bad.txt" "$T/r" 64
: > "$T/bad.txt"
refuse "no ids" kolibri-ids 1 "$B" inputs "$T/toy.gguf" "$T/bad.txt" "$T/r" 64
refuse "positions that are not a multiple of 64" kolibri-usage 1 "$B" inputs "$T/toy.gguf" "$T/ids.txt" "$T/r" 100
refuse "positions that are not a number" kolibri-usage 1 "$B" inputs "$T/toy.gguf" "$T/ids.txt" "$T/r" 6x
awk 'BEGIN { for (i = 0; i < 65; i++) printf "1 " }' > "$T/bad.txt"
refuse "more ids than positions" kolibri-ids 1 "$B" inputs "$T/toy.gguf" "$T/bad.txt" "$T/r" 64
refuse "fewer steps than ids" kolibri-usage 1 "$B" inputs "$T/toy.gguf" "$T/ids.txt" "$T/r" 64 2
refuse "more steps than positions" kolibri-usage 1 "$B" inputs "$T/toy.gguf" "$T/ids.txt" "$T/r" 64 65
toy "$T/v.gguf" v8
refuse "an embedding that is not a multiple of 64 rows" kolibri-tensor 1 "$B" weights "$T/v.gguf" "$T/r"
refuse "ids from past the file's end" kolibri-read 1 "$B" ids "$T/x/idtab.f32" 6
refuse "ids of a file whose values are not whole" kolibri-read 1 "$B" ids "$T/x/cos.f32" 0
printf 'abc' > "$T/bad.f32"
refuse "an f32 file of three bytes" kolibri-read 1 "$B" argmax "$T/bad.f32"
: > "$T/bad.f32"
refuse "an empty f32 file" kolibri-read 1 "$B" argmax "$T/bad.f32"
printf '\000\000\300\177' > "$T/bad.f32"
refuse "a value that is not a number" kolibri-read 1 "$B" argmax "$T/bad.f32"
[ "$fail" -eq 0 ] && echo "kolibri_host: ok"
exit $fail
