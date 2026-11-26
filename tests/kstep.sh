#!/bin/sh
# tests/kstep.sh [session binary] -- models/kolibri_step.model (docs/kolibri.md K4) as a small model, on the CPU session (the gate) or, given its binary, the Metal one (by hand): ten layers 64 wide
# with 4 query heads over 2 key/value heads of 128, 64 experts of 32 of which 2 are taken, a head of 2,048, every file text, four steps. Layers 0 and 1 have the dense MLP, 4 is indexed (no positions,
# its keys and values rounded to NVFP4) and 9 reads layer 4's caches; the mask's window is 2 keys, so it bites from the third step. Held to awk's double arithmetic of the definition: the row after
# the last layer at every step, and after the last step the row after every layer and the logits.
# Then the same model generating (GEN=1): from a prompt of two ids, the id each step chooses, awk's arg max over awk's logits; and sampling (SAMP=1): the id each step draws, awk's sampler of
# the same definition with the same four uniform numbers.
# Then (sections 15 and 19.2) a window that slides and caches in chunks: the same ten layers for 14 steps with chunks of 4 keys, a window of 3 of which 4 are scanned and caches of 16
# positions, against the same awk, which has no caches at all: the row after the last layer at every step, the row after every layer and the logits at the last. Then seeks back to
# any step, each exact: a cache is a roll of positions, and a step run again reads what it read.
# First fp4_round.chx alone on values that sit on its thresholds, where tests/kernels.py's made-up table and values never are: ties go to even, the scale stays in [2^-9, 448].
# KSTEP_MODEL=<file> runs another model file, KSTEP_CHX=<directory> other kernels: the mutants of the model and of its kernels, by hand.
set -eu
cd "$(dirname "$0")/.."
B="${1:-build/cancho-gpu-session}"
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
fail=0
CHX=${KSTEP_CHX:-$(pwd)/tests/chx}
mkdir "$T/w" "$T/x" "$T/xl"
awk -v T="$T" '
function rne(v,   f, d) { f = int(v); d = v - f; if (d > 0.5 || (d == 0.5 && f % 2 == 1)) f++; return f }   # v >= 0, to the nearest whole number, ties to even
function e4m3(x,   e, p) { if (x < 2^-9) x = 2^-9; if (x > 448) x = 448; e = -9; p = 2^-9; while (p * 2 <= x) { p *= 2; e++ } if (e < -6) return rne(x / 2^-9) * 2^-9; x = rne(x / p * 8) / 8 * p; return x > 448 ? 448 : x }
function e2m1(m,   st) { if (m > 6) m = 6; st = m < 2 ? 0.5 : (m < 4 ? 1 : 2); return rne(m / st) * st }
function fp4(v, sc,   m) { m = v / sc; return m < 0 ? -e2m1(-m) * sc : e2m1(m) * sc }
BEGIN {
  # the table, from the list of E4M3 values: thresholds, increments, and a 1 where the lower neighbour is odd (a tie goes up)
  n = 0; for (k = 1; k < 8; k++) { v[n] = k * 2^-9; od[n++] = k % 2 }
  for (e = -6; e < 8; e++) for (j = 0; j < 8; j++) { v[n] = (1 + j / 8) * 2^e; od[n++] = j % 2 }
  for (j = 0; j < 7; j++) { v[n] = (1 + j / 8) * 256; od[n++] = j % 2 }
  f = T "/w/e4m3.f32"; for (i = 0; i < 125; i++) printf "%.17g ", (v[i] + v[i + 1]) / 2 > f; for (i = 0; i < 125; i++) printf "%.17g ", v[i + 1] - v[i] > f; for (i = 0; i < 125; i++) printf "%d ", od[i] > f; printf "\n" > f; close(f)
  # five groups of 16: every magnitude threshold at scale 1; a scale on a threshold that goes down (1.0625) and one that goes up (1.1875); a group under the smallest scale; one over the largest
  split("6 0.25 -0.25 0.75 -0.75 1.25 1.75 -1.75 2.5 -2.5 3.5 -3.5 5 -5 0.24 0", g0, " ")
  split("6.375 -1 0.3 0.76 1.26 1.74 2.4 2.6 3.4 3.6 4.9 5.1 0 -0.5 -6.375 0.74", g1, " ")
  split("7.125 -7.125 1 0.3125 0.9375 1.5625 2.1875 3.125 4.375 6.25 6.5 -0.3125 0 2 3 5", g2, " ")
  split("0.001 -0.001 0.0005 0.0004 0.0015 0 0.00075 -0.0002 0.00048828125 0.000244140625 0.0001 0 0 0 0 0", g3, " ")
  split("5000 -448 224 112 0 1 -1 672 1120 1568 2240 448 896 1344 1792 2688", g4, " ")
  for (i = 1; i <= 16; i++) { x[i - 1] = g0[i]; x[15 + i] = g1[i]; x[31 + i] = g2[i]; x[47 + i] = g3[i]; x[63 + i] = g4[i] }
  for (g = 0; g < 5; g++) { mx = 0; for (i = 0; i < 16; i++) { a = x[g * 16 + i]; if (a < 0) a = -a; if (a > mx) mx = a } sc = e4m3(mx / 6); sca[g] = sc
    for (i = 0; i < 16; i++) { printf "%.17g ", x[g * 16 + i] > (T "/rx.txt"); printf "%.9g\n", fp4(x[g * 16 + i], sc) > (T "/rwant.txt") } }
  printf "%.10g %.10g %.10g %.10g %.10g\n", sca[0], sca[1], sca[2], sca[3], sca[4] > (T "/rscales.txt")
}'
printf 'program r %s/fp4_round.chx r=1 n=80\nresident r.x %s/rx.txt\nresident r.tb %s/w/e4m3.f32\nanswer r.y\n' "$CHX" "$T" "$T" > "$T/r.session"
printf '\n' | "$B" "$T/r.session" 2>&1 | head -c 20000 | tr -s ' \n' '\n\n' | grep -v '^$' > "$T/rgot.txt" || true
same() { # name, got, want: every value within 2e-5 of the larger of 1 and the wanted value, or exactly (a third argument of 0)
  out=$(paste "$2" "$3" | awk -v tol="${4:-2e-5}" -v n="$(wc -l < "$3")" 'NF != 2 { bad = "a line of " NF " fields: " $0; exit } { d = $1 - $2; if (d < 0) d = -d; s = ($2 < 0 ? -$2 : $2); if (s < 1) s = 1; if (d > tol * s) { bad = "value " NR ": " $1 ", wanted " $2; exit } } END { if (bad != "") print bad; else if (NR != n) print NR " values, wanted " n; else print "ok" }')
  if [ "$out" = ok ]; then echo "ok   $1"; else echo "FAIL $1: $out"; fail=1; fi
}
if [ "$(cat "$T/rscales.txt")" = "1 1 1.25 0.001953125 448" ]; then echo "ok   the reference's scales: 1, 1 (a tie that goes down), 1.25 (one that goes up), 2^-9, 448"; else echo "FAIL the reference's scales: $(cat "$T/rscales.txt")"; fail=1; fi
same "fp4_round on its thresholds: 80 values, exactly" "$T/rgot.txt" "$T/rwant.txt" 0
# two mutants tests/kernels.py cannot see (its made-up table is not E4M3's, so no value there reaches the largest magnitude, and nothing sits on a threshold): the largest magnitude 5, and a tie at 0.75 going down
mutant() { # name, sed expression
  mkdir "$T/chx"; cp tests/chx/fp4_round.chx "$T/chx/"; sed "$2" tests/chx/fp4_round.chx > "$T/chx/fp4_round.chx"
  if cmp -s tests/chx/fp4_round.chx "$T/chx/fp4_round.chx"; then echo "FAIL the mutant's text was not found in fp4_round.chx ($1)"; fail=1; fi
  printf 'program r %s/fp4_round.chx r=1 n=80\nresident r.x %s/rx.txt\nresident r.tb %s/w/e4m3.f32\nanswer r.y\n' "$T/chx" "$T" "$T" > "$T/m.session"
  printf '\n' | "$B" "$T/m.session" 2>&1 | head -c 20000 | tr -s ' \n' '\n\n' | grep -v '^$' > "$T/mgot.txt" || true
  bad=$(paste "$T/mgot.txt" "$T/rwant.txt" | awk 'NF != 2 { print "did not run"; exit } $1 - $2 != 0 { n++ } END { print n + 0 }')
  case "$bad" in 0) echo "FAIL the mutant ($1) survives"; fail=1 ;; *[!0-9]*) echo "FAIL the mutant ($1) did not run: $(head -c 200 "$T/mgot.txt")"; fail=1 ;; *) echo "ok   the mutant ($1) is killed: $bad of 80 values differ" ;; esac
  rm -r "$T/chx"
}
mutant "the largest magnitude is 5" 's|let w7  = s7 \* 2.0|let w7  = s7 * 1.0|'
mutant "a value at 0.75 goes down" 's|let b2  = &ng + 0.75|let b2  = \&ng + 0.7500001|'

D=64; NQ=4; NKV=2; NE=64; EH=32; K=2; NS=4; WIN=2; NL=10; V=2048; KC=24; TOPK=5; INVT=1.5; PC=0.4; NSL=14; WINL=3; TWL=4; TL=16
awk -v D=$D -v NQ=$NQ -v NKV=$NKV -v NE=$NE -v EH=$EH -v K=$K -v NS=$NS -v WIN=$WIN -v NL=$NL -v V=$V -v KCH=$KC -v TOPK=$TOPK -v INVT=$INVT -v PC=$PC -v W="$T/w" -v X="$T/x" -v XL="$T/xl" -v NSL=$NSL -v WINL=$WINL -v TWL=$TWL -v TL=$TL '
function rne(v,   f, d) { f = int(v); d = v - f; if (d > 0.5 || (d == 0.5 && f % 2 == 1)) f++; return f }
function e4m3(x,   e, p) { if (x < 2^-9) x = 2^-9; if (x > 448) x = 448; e = -9; p = 2^-9; while (p * 2 <= x) { p *= 2; e++ } if (e < -6) return rne(x / 2^-9) * 2^-9; x = rne(x / p * 8) / 8 * p; return x > 448 ? 448 : x }
function e2m1(m,   st) { if (m > 6) m = 6; st = m < 2 ? 0.5 : (m < 4 ? 1 : 2); return rne(m / st) * st }
function fp4(v, sc,   m) { m = v / sc; return m < 0 ? -e2m1(-m) * sc : e2m1(m) * sc }
function byte(m, i) { return (37 * i + 101 * int(i / 61) + 11 + 53 * m) % 256 }
function sc(m, g) { return SC[(g * 5 + 3 * int(g / 7) + m) % 16 + 1] / 8 }
function off(m, g) { return ((g * 7 + int(g / 5) + m) % 9 - 4) / 32 }
function wv(m, cols, row, c, q41,   b, code, g) { b = byte(m, row * cols / 2 + int(c / 2)); code = (c % 2 == 0) ? b % 16 : int(b / 16); g = row * cols / 32 + int(c / 32); return (code - 8) * sc(m, g) + (q41 ? off(m, g) : 0) }
function emit(name, m, rows, cols, q41,   i, b, f) {
  f = W "/" name ".q4"; for (i = 0; i < rows * cols / 2; i++) { b = byte(m, i); printf "%d ", (b > 127 ? b - 256 : b) > f } printf "\n" > f; close(f)
  f = W "/" name ".s"; for (i = 0; i < rows * cols / 32; i++) printf "%s ", sc(m, i) > f; printf "\n" > f; close(f)
  if (q41) { f = W "/" name ".c"; for (i = 0; i < rows * cols / 32; i++) printf "%s ", off(m, i) > f; printf "\n" > f; close(f) }
}
function nw(m, c) { return 0.5 + ((c * 3 + m * 5) % 7) / 8 }     # a norm weight
function vec(name, m, n,   c, f) { f = W "/" name; for (c = 0; c < n; c++) printf "%s ", nw(m, c) > f; printf "\n" > f; close(f) }
function ones(name, n, v,   c, f) { f = W "/" name; for (c = 0; c < n; c++) printf "%s ", v > f; printf "\n" > f; close(f) }
function rms(src, m, n, dst,   c, ss, rs) { ss = 0; for (c = 0; c < n; c++) ss += src[c] * src[c]; rs = 1 / sqrt(ss / n + 0.000001); for (c = 0; c < n; c++) dst[c] = src[c] * rs * nw(m, c) }
function silu(v) { return v / (1 + exp(-v)) }
function mlp(mg, mu, md, base, h, q41,   r, c, g, u) { # y[] = down(silu(gate n2) * (up n2)), the matrices rows from block base
  for (r = 0; r < h; r++) { g = 0; u = 0; for (c = 0; c < D; c++) { g += wv(mg, D, base * h + r, c, 0) * n2[c]; u += wv(mu, D, base * h + r, c, 0) * n2[c] } act[r] = silu(g) * u }
  for (r = 0; r < D; r++) { y[r] = 0; for (c = 0; c < h; c++) y[r] += wv(md, h, base * D + r, c, q41) * act[c] }
}
function head128(src, o, m, rot, s, dst, d0,   c, ss, rs, a) { # a head of 128 at src[o ..]: over its root mean square, times the weight m, rotated by step s where rot
  ss = 0; for (c = 0; c < 128; c++) ss += src[o + c] * src[o + c]; rs = 1 / sqrt(ss / 128 + 0.000001); for (c = 0; c < 128; c++) a[c] = src[o + c] * rs * nw(m, c)
  for (c = 0; c < 64; c++) { if (rot) { dst[d0 + c] = a[c] * CS[s, c] - a[64 + c] * SN[s, c]; dst[d0 + 64 + c] = a[64 + c] * CS[s, c] + a[c] * SN[s, c] } else { dst[d0 + c] = a[c]; dst[d0 + 64 + c] = a[64 + c] } }
}
function round16(a, n,   g, i, mx, v, s) { for (g = 0; g < n / 16; g++) { mx = 0; for (i = 0; i < 16; i++) { v = a[g * 16 + i]; if (v < 0) v = -v; if (v > mx) mx = v } s = e4m3(mx / 6); for (i = 0; i < 16; i++) a[g * 16 + i] = fp4(a[g * 16 + i], s) } }
function layer(l, s,   own, idx, src, m0, rows, r, c, h, t, t0, kh, g, qh, mxs, z, j, e, best, used, bi, rsrc, f) {
  idx = (l % 5 == 4); own = (!idx || l == 4 || l == 19 || l == 24 || l == 29); src = own ? l : ((l == 49 || l < 29) ? 4 : 29); m0 = 20 * l
  rms(x, m0 + 1, D, n1)
  rows = (own ? NQ + 2 * NKV : NQ) * 128
  for (r = 0; r < rows; r++) { yq[r] = 0; for (c = 0; c < D; c++) yq[r] += wv(m0 + 1, D, r, c, 0) * n1[c] }
  for (h = 0; h < NQ; h++) head128(yq, h * 128, m0 + 2, !idx, s, q, h * 128)
  if (own) {
    for (h = 0; h < NKV; h++) { head128(yq, (NQ + h) * 128, m0 + 3, !idx, s, kn, h * 128); for (c = 0; c < 128; c++) vn[h * 128 + c] = yq[(NQ + NKV + h) * 128 + c] }
    if (idx) { round16(kn, NKV * 128); round16(vn, NKV * 128) }
    for (c = 0; c < NKV * 128; c++) { KC[l, s, c] = kn[c]; VC[l, s, c] = vn[c] }
  }
  t0 = idx ? 0 : (s - WIN + 1 > 0 ? s - WIN + 1 : 0); g = NQ / NKV
  for (qh = 0; qh < NQ; qh++) { kh = int(qh / g); mxs = -1e30
    for (t = t0; t <= s; t++) { z = 0; for (c = 0; c < 128; c++) z += q[qh * 128 + c] * KC[src, t, kh * 128 + c]; p[t] = z / sqrt(128); if (p[t] > mxs) mxs = p[t] }
    z = 0; for (t = t0; t <= s; t++) { p[t] = exp(p[t] - mxs); z += p[t] }
    for (c = 0; c < 128; c++) { o[qh * 128 + c] = 0; for (t = t0; t <= s; t++) o[qh * 128 + c] += p[t] / z * VC[src, t, kh * 128 + c] } }
  for (r = 0; r < D; r++) { av[r] = 0; for (c = 0; c < NQ * 128; c++) av[r] += wv(m0 + 2, NQ * 128, r, c, 0) * o[c] }
  rms(av, m0 + 4, D, an); for (c = 0; c < D; c++) mid[c] = x[c] + an[c]
  rms(mid, m0 + 5, D, n2)
  if (l < 2) { mlp(m0 + 3, m0 + 4, m0 + 5, 0, D, 1); for (r = 0; r < D; r++) mm[r] = y[r] }
  else {
    for (e = 0; e < NE; e++) { lg[e] = 0; for (c = 0; c < D; c++) lg[e] += rt(l, e, c) * n2[c]; zz[e] = lg[e] + rb(l, e) }
    mlp(m0 + 6, m0 + 7, m0 + 8, 0, EH, l < 6); for (r = 0; r < D; r++) mm[r] = y[r]
    for (j = 0; j < K; j++) { best = -1; for (e = 0; e < NE; e++) if (!(e in used) && (best < 0 || zz[e] > zz[best])) best = e; used[best] = 1
      mlp(m0 + 9, m0 + 10, m0 + 11, best, EH, l < 6); z = 1 / (1 + exp(-lg[best])); for (r = 0; r < D; r++) mm[r] += z * y[r] }
  }
  rms(mm, m0 + 6, D, an); for (c = 0; c < D; c++) x[c] = mid[c] + an[c]
}
function rt(l, e, c) { return ((e * 31 + c * 17 + l * 7) % 23 - 11) / 64 }
function rb(l, e) { return ((e * 13 + l * 3) % 11 - 5) / 16 + e / 4096 }
BEGIN {
  split("0.5 0.25 1.5 3 0.125 2 0.75 1 4 0.375 6 0.0625 1.25 0.5 3.5 2.5", SC, " ")
  ones("ones_d.f32", D, 1); ones("ones_h.f32", EH, 1); ones("ones_f.f32", D, 1); ones("ones_qkv.f32", (NQ + 2 * NKV) * 128, 1); ones("ones_q.f32", NQ * 128, 1); ones("one64.f32", 64, 1); ones("zero64.f32", 64, 0)
  for (e = 0; e < NE; e++) printf "%d ", e > (W "/ar.f32"); for (j = 0; j < K; j++) printf "%d ", j > (W "/sl.f32")
  for (l = 0; l < NL; l++) { m0 = 20 * l; idx = (l % 5 == 4); own = (!idx || l == 4)
    vec("a" l ".norm.f32", m0 + 1, D); vec("a" l ".qn.f32", m0 + 2, 128); vec("a" l ".pn.f32", m0 + 4, D); vec("m" l ".ffn_norm.f32", m0 + 5, D); vec("m" l ".post_norm.f32", m0 + 6, D)
    if (own) { vec("a" l ".kn.f32", m0 + 3, 128); emit("a" l ".qkv", m0 + 1, (NQ + 2 * NKV) * 128, D, 0) } else emit("a" l ".q", m0 + 1, NQ * 128, D, 0)
    emit("a" l ".o", m0 + 2, D, NQ * 128, 0)
    if (l < 2) { emit("m" l ".dn.gate", m0 + 3, D, D, 0); emit("m" l ".dn.up", m0 + 4, D, D, 0); emit("m" l ".dn.down", m0 + 5, D, D, 1) }
    else {
      f = W "/m" l ".route.f32"; for (e = 0; e < NE; e++) for (c = 0; c < D; c++) printf "%s ", rt(l, e, c) > f; close(f); f = W "/m" l ".bias.f32"; for (e = 0; e < NE; e++) printf "%s ", rb(l, e) > f; close(f)
      emit("m" l ".sh.gate", m0 + 6, EH, D, 0); emit("m" l ".sh.up", m0 + 7, EH, D, 0); emit("m" l ".sh.down", m0 + 8, D, EH, l < 6)
      emit("m" l ".ex.gate", m0 + 9, NE * EH, D, 0); emit("m" l ".ex.up", m0 + 10, NE * EH, D, 0); emit("m" l ".ex.down", m0 + 11, NE * D, EH, l < 6)
    }
  }
  vec("fnorm.f32", 999, D); f = W "/head.f32"; for (r = 0; r < V; r++) for (c = 0; c < D; c++) printf "%s ", hd(r, c) > f; close(f)
  # (the E4M3 table is the one the first part of this test wrote.) The run: rows, positions, masks.
  for (s = 0; s < NS; s++) {
    for (c = 0; c < D; c++) { X0[s, c] = ((c * 37 + s * 23 + 11) % 29 - 14) / 8; printf "%s ", X0[s, c] > (X "/x0.f32") }
    for (c = 0; c < 64; c++) { fr = s / (10000 ^ (c / 64)); CS[s, c] = cos(fr); SN[s, c] = sin(fr); printf "%.17g ", CS[s, c] > (X "/cos.f32"); printf "%.17g ", SN[s, c] > (X "/sin.f32") }
    printf "%d ", s > (X "/pos.f32")
  }
  # the caches are rolls (section 19.2): the numbers of the positions, and the mask of a sliding layer over its window, a row a step while the window is not full
  for (t = 0; t < 64; t++) { printf "%d ", t > (X "/iota.f32"); for (c = 0; c < 64; c++) printf "%s ", ((c <= t && c > t - WIN) ? 0 : -1e9) > (X "/maskw.f32") }
  for (s = 0; s < NS; s++) { for (c = 0; c < D; c++) x[c] = X0[s, c]
    for (l = 0; l < NL; l++) { layer(l, s); if (s == NS - 1) for (c = 0; c < D; c++) printf "%.9g\n", x[c] > (X "/want_out" l ".txt") }
    for (c = 0; c < D; c++) printf "%.9g\n", x[c] > (X "/want_ans.txt") }
  rms(x, 999, D, fnv); for (r = 0; r < V; r++) { z = 0; for (c = 0; c < D; c++) z += hd(r, c) * fnv[c]; printf "%.9g\n", z > (X "/want_logits.txt") }
  # GEN=1 (section 13): the same model from a prompt of two ids; the row of a step is the row its id has in the 4-bit embedding matrix, and past the prompt the id is the arg max of the step before
  emit("emb", 998, V, D, 0); ones("one1.f32", 1, 1); ones("zmk.f32", V, 0); for (r = 0; r < V; r++) printf "%d ", V - r > (W "/rank.f32")
  PID[0] = 5; PID[1] = 77; printf "0 " > (X "/id0.f32")
  # a request a step: a step, its id and whether the id is the one of the request (the prompt) or the one the step before chose (section 17, S1); a second prompt for the conversations after one another
  for (s = 0; s < NS; s++) { printf "1 %d %d\n", (s < 2 ? PID[s] : 0), (s < 2) > (X "/req_gen.txt"); printf "1 %d %d\n", (s < 2 ? 300 - 291 * s : 0), (s < 2) > (X "/req_b.txt") }
  for (s = 0; s < NS; s++) { id = s < 2 ? PID[s] : nxt; for (c = 0; c < D; c++) x[c] = wv(998, D, id, c, 0)
    for (l = 0; l < NL; l++) layer(l, s)
    rms(x, 999, D, fnv); nxt = 0; best = -1e30; for (r = 0; r < V; r++) { z = 0; for (c = 0; c < D; c++) z += hd(r, c) * fnv[c]; if (z > best) { best = z; nxt = r } }
    printf "%d\n", nxt > (X "/want_ids.txt") }
  # SAMP=1 (section 14): the same, the id drawn. The KCH chunks of 64 with the largest maxima, of their candidates those with fewer than TOPK larger ones, of those the ones whose running mass from
  # the smallest is over PC of the whole, and a uniform number through the cumulative sums in the gathered order.
  for (r = 0; r < V / 64; r++) printf "%d ", r > (W "/ciota.f32"); for (r = 0; r < KCH; r++) { printf "%d ", r > (W "/cslot.f32"); for (c = 0; c < KCH; c++) printf "%d ", (c <= r) > (W "/ctri.f32") }
  for (r = 0; r < KCH * 64; r++) printf "0 " > (W "/czero.f32"); for (r = 0; r < 64; r++) for (c = 0; c < 64; c++) printf "%d ", (c <= r) > (W "/tri64.f32")
  # the uniform numbers: three given, and the last chosen by draw() to fall inside the mass of a candidate top-p dropped (see there)
  split("0.03125 0.5 0.96875 -1", UU, " ")
  for (s = 0; s < NS; s++) { id = s < 2 ? PID[s] : nxt; for (c = 0; c < D; c++) x[c] = wv(998, D, id, c, 0)
    for (l = 0; l < NL; l++) layer(l, s)
    rms(x, 999, D, fnv); am = 0; for (r = 0; r < V; r++) { z = 0; for (c = 0; c < D; c++) z += hd(r, c) * fnv[c]; LG[r] = z; if (z > LG[am]) am = r }
    nxt = draw(UU[s + 1]); printf "%d\n", nxt > (X "/want_draws.txt"); if (s == NS - 1) for (r = 0; r < KCH; r++) { printf "%.9g\n", CU[r] > (X "/want_cum.txt"); printf "%d\n", CN[r] > (X "/want_cnt.txt") }
    printf "1 %d %d %.17g\n", (s < 2 ? PID[s] : 0), (s < 2), (UU[s + 1] < 0 ? ULAST : UU[s + 1]) > (X "/req_samp.txt"); printf "%d %d %d %d %s\n", nxt, am, K1, K2, DROPW > (X "/draw_stats.txt") }
  # once more with no top-p (PC 0), the 4 largest and the same numbers: there the largest are what is drawn from, and a sampler that keeps one more shows. With top-p it did not, and not with 5
  # either: the large logits of this head come in equal pairs, the fifth and sixth are equal, and 5 already keeps both (which is the rule: an entry equal to the last kept stays)
  PC = 0; TOPK = 4; UU[NS] = ULAST
  for (s = 0; s < NS; s++) { id = s < 2 ? PID[s] : nxt; for (c = 0; c < D; c++) x[c] = wv(998, D, id, c, 0)
    for (l = 0; l < NL; l++) layer(l, s)
    rms(x, 999, D, fnv); for (r = 0; r < V; r++) { z = 0; for (c = 0; c < D; c++) z += hd(r, c) * fnv[c]; LG[r] = z }
    nxt = draw(UU[s + 1]); printf "%d\n", nxt > (X "/want_draws_k.txt"); if (s == NS - 1) for (r = 0; r < KCH; r++) printf "%.9g\n", CU[r] > (X "/want_cum_k.txt") }
  # section 15: NSL steps with a window of WINL keys. The tables are for the ring of TWL slots of a sliding layer (slot t holds the last position ps <= s with ps % TWL == t, if there is one, and is
  # seen when that position is in the window) and the TL positions of an indexed one; layer() above knows neither a ring nor a chunk, only the window
  WIN = WINL
  for (s = 0; s < NSL; s++) {
    for (c = 0; c < D; c++) { X0[s, c] = ((c * 37 + s * 23 + 11) % 29 - 14) / 8; printf "%s ", X0[s, c] > (XL "/x0.f32") }
    for (c = 0; c < 64; c++) { fr = s / (10000 ^ (c / 64)); CS[s, c] = cos(fr); SN[s, c] = sin(fr); printf "%.17g ", CS[s, c] > (XL "/cos.f32"); printf "%.17g ", SN[s, c] > (XL "/sin.f32") }
    printf "%d ", s > (XL "/pos.f32")
  }
  for (t = 0; t < TL; t++) printf "%d ", t > (XL "/iota.f32"); for (t = 0; t < TWL; t++) for (c = 0; c < TWL; c++) printf "%s ", ((c <= t && c > t - WIN) ? 0 : -1e9) > (XL "/maskw.f32")
  for (s = 0; s < NSL; s++) { for (c = 0; c < D; c++) x[c] = X0[s, c]
    for (l = 0; l < NL; l++) { layer(l, s); if (s == NSL - 1) for (c = 0; c < D; c++) printf "%.9g\n", x[c] > (XL "/want_out" l ".txt") }
    for (c = 0; c < D; c++) printf "%.9g\n", x[c] > (XL "/want_ans.txt") }
  rms(x, 999, D, fnv); for (r = 0; r < V; r++) { z = 0; for (c = 0; c < D; c++) z += hd(r, c) * fnv[c]; printf "%.9g\n", z > (XL "/want_logits.txt") }
}
function draw(uu,   dj, lk, q, c, r, best, used, n, j, i, done, gmax, nk, mm, mmax, tot, t, rs, tl, lc, col) {
  for (q = 0; q < V / 64; q++) { cmx[q] = LG[q * 64]; for (c = 1; c < 64; c++) if (LG[q * 64 + c] > cmx[q]) cmx[q] = LG[q * 64 + c] }
  for (r = 0; r < KCH; r++) { best = -1; for (q = 0; q < V / 64; q++) if (!(q in used) && (best < 0 || cmx[q] > cmx[best])) best = q; used[best] = 1; ORD[r] = best }
  n = KCH * 64; for (j = 0; j < n; j++) { Y[j] = LG[ORD[int(j / 64)] * 64 + j % 64] * INVT; Z[j] = -1e9 }
  K1 = 0; gmax = -1e30; for (j = 0; j < n; j++) { done = 0; for (i = 0; i < n && done < TOPK; i++) if (Y[i] > Y[j]) done++; if (done < TOPK) { Z[j] = Y[j]; KL[K1++] = j; if (Y[j] > gmax) gmax = Y[j] } }
  mmax = 0; for (i = 0; i < K1; i++) { mm = 0; for (j = 0; j < K1; j++) if (Z[KL[j]] <= Z[KL[i]]) mm += exp(Z[KL[j]] - gmax); M[i] = mm; if (mm > mmax) mmax = mm }
  K2 = 0; for (r = 0; r < KCH; r++) CM[r] = 0; for (c = 0; c < n; c++) E[c] = 0
  for (i = 0; i < K1; i++) if (M[i] > PC * mmax) { E[KL[i]] = exp(Z[KL[i]] - gmax); CM[int(KL[i] / 64)] += E[KL[i]]; K2++ }
  tot = 0; for (r = 0; r < KCH; r++) { tot += CM[r]; CU[r] = tot }
  # uu < 0: the number is chosen so that the target falls in the middle of where the largest candidate top-p dropped would lie, had it been kept (one that lies before some kept candidate in the gathered order: past the last there is no target): a step that counts columns in the candidates
  # before top-p then stops at that column, and the right one does not. A multiple of 1/4096, exact in f32. (A mutant of the model file that did this drew the same ids until this was here.)
  if (uu < 0) { lk = -1; for (i = 0; i < K1; i++) if (E[KL[i]] > 0 && KL[i] > lk) lk = KL[i]; dj = -1; for (i = 0; i < K1; i++) if (E[KL[i]] == 0 && KL[i] < lk && (dj < 0 || Z[KL[i]] > Z[dj])) dj = KL[i]
    if (dj < 0) uu = 0.3125; else { r = int(dj / 64); lc = (r > 0 ? CU[r - 1] : 0); for (c = r * 64; c < dj; c++) lc += E[c]; uu = int((lc + exp(Z[dj] - gmax) / 2) / tot * 4096 + 0.5) / 4096; if (uu >= 1) uu = 4095 / 4096 }
    ULAST = uu; DROPW = (dj < 0 ? 0 : exp(Z[dj] - gmax) / tot) }
  t = uu * tot; rs = 0; for (r = 0; r < KCH; r++) if (CU[r] < t) rs++
  for (r = 0; r < KCH; r++) { tl = CM[r] - CU[r] + t; lc = 0; CN[r] = 0; for (c = 0; c < 64; c++) { lc += E[r * 64 + c]; if (lc < tl) CN[r]++ } }
  return ORD[rs] * 64 + CN[rs]
}
# the head: rows 2k and 2k + 1 are near each other, so that large logits come in pairs inside a chunk of 64 (a candidate top-p drops can then lie before one it keeps)
function hd(r, c) { return ((int(r / 2) * 29 + c * 13) % 31 - 15) / 32 + ((r * 7) % 5 - 2) * ((c * 3 + r) % 7 - 3) / 64 }'
RX="$T/x"; RN=$NS; RT=64; RTW=64; RTS=64   # a run's tables, its steps, an indexed layer's positions, a sliding layer's slots, the keys a chunk
REQ=""; CK=1   # a file of request lines (none: RN requests of a step), and whether a checkpoint follows them
run() { # the model file's extra arguments
  { if [ -n "$REQ" ]; then cat "$REQ"; else i=0; while [ $i -lt $RN ]; do echo 1; i=$((i + 1)); done; fi; [ $CK -eq 0 ] || echo checkpoint; } | "$B" "${KSTEP_MODEL:-models/kolibri_step.model}" CHX="$CHX" W="$T/w" X="$RX" D=$D NQ=$NQ NKV=$NKV NE=$NE EH=$EH K=$K FF=$D V=$V BN=32 T=$RT TW=$RTW TS=$RTS NS=$RN L0=0 NL=$NL HEAD=1 SAVE=1 ANS=1 F32= F16= I8= "$@" > "$T/out.txt" 2> "$T/err.txt" || true
  head -n $RN "$T/out.txt" | head -c 200000 | tr -s ' \n' '\n\n' | grep -v '^$' > "$T/got_ans.txt" || true
}
lines() { sed -n "$1,$2p" "$T/out.txt" | tr -s ' \n' '\n\n' | grep -v '^$' > "$3" || true; }   # the answers of requests $1 to $2 of the last run
col() { od -A n -v -t fF "$1" | tr -s ' \n' '\n\n' | grep -v '^$' > "$2" || true; }   # a save is binary f32 whatever the session's files are
run
if [ ! -s "$T/got_ans.txt" ]; then echo "FAIL the session answered nothing: $(head -c 300 "$T/err.txt")"; fail=1; fi
same "the row after the last layer at each of $NS steps" "$T/got_ans.txt" "$T/x/want_ans.txt"
l=0; while [ $l -lt $NL ]; do
  if [ -f "$T/x/out$l.f32" ]; then col "$T/x/out$l.f32" "$T/got.txt"; else : > "$T/got.txt"; fi
  case $l in 0) what="rotary, a dense MLP" ;; 4) what="indexed, its keys and values rounded" ;; 9) what="indexed, reading layer 4's keys and values" ;; *) what="" ;; esac
  if [ -n "$what" ]; then same "the row after layer $l ($what) at the last step" "$T/got.txt" "$T/x/want_out$l.txt"; else same "the row after layer $l at the last step" "$T/got.txt" "$T/x/want_out$l.txt" > "$T/q.txt"; grep FAIL "$T/q.txt" || true; fi
  l=$((l + 1))
done
if [ -f "$T/x/logits.f32" ]; then col "$T/x/logits.f32" "$T/got.txt"; else : > "$T/got.txt"; fi
same "the logits at the last step" "$T/got.txt" "$T/x/want_logits.txt"
# the step feeding itself: the id each of the four steps chose. A request is a step, its id and its mask (section 17, S1)
REQ="$T/x/req_gen.txt"; CK=0
run GEN=1 SAVE=0
same "generating from two ids: the id each step chose, exactly" "$T/got_ans.txt" "$T/x/want_ids.txt" 0
# one session, conversations one after another. First another prompt in a session of its own, for what it answers
REQ="$T/x/req_b.txt"; run GEN=1 SAVE=0
cp "$T/got_ans.txt" "$T/want_b.txt"
if cmp -s "$T/want_b.txt" "$T/x/want_ids.txt" || [ "$(wc -l < "$T/want_b.txt" | tr -d ' ')" -ne "$NS" ]; then echo "FAIL the second prompt's ids are the first's, or not $NS: a session that ignores a request's id would pass"; fail=1; fi
# then: the first prompt, seek 0, the first prompt again, seek 0, the second prompt, and back to its third step. The step after a seek is given its id (the one the second step chose): what
# the session carries is the id of the last step it ran, not of the step before the one sought
{ cat "$T/x/req_gen.txt"; echo "seek 0"; cat "$T/x/req_gen.txt"; echo "seek 0"; cat "$T/x/req_b.txt"; echo "seek 2"; echo "1 $(sed -n 2p "$T/want_b.txt") 1"; sed -n '4,$p' "$T/x/req_b.txt"; } > "$T/req_conv.txt"
REQ="$T/req_conv.txt"; run GEN=1 SAVE=0
lines $((NS + 2)) $((2 * NS + 1)) "$T/got.txt"
same "the same prompt again in one session, after seek 0: the same ids" "$T/got.txt" "$T/x/want_ids.txt" 0
lines $((2 * NS + 3)) $((3 * NS + 2)) "$T/got.txt"
same "another prompt after it: the ids of a session that starts with it" "$T/got.txt" "$T/want_b.txt" 0
lines $((3 * NS + 4)) $((4 * NS + 1)) "$T/got.txt"
sed -n '3,$p' "$T/want_b.txt" > "$T/want.txt"
same "back to its third step (seek 2), the step given its id: the same ids from there" "$T/got.txt" "$T/want.txt" 0
REQ="$T/x/req_samp.txt"; CK=1
# the four ids must differ and lie in more than one 64-entry chunk of the vocabulary, not all in the first, or a step that ignores its id, or an arg max that looks at the first chunk only, would pass
run GEN=1 SAMP=1 SAVE=1 KC=$KC TOPK=$TOPK CB=16 PC=$PC INVT=$INVT
same "sampling from two ids (temperature 2/3, the 5 largest, top-p 0.6): the id each step drew, exactly" "$T/got_ans.txt" "$T/x/want_draws.txt" 0
# four draws tell little of a distribution (four mutants of the model file drew the same four): so also, at the last step, the gathered chunks' cumulative masses and every chunk's count of columns before the target
if [ -f "$T/x/cum.f32" ]; then col "$T/x/cum.f32" "$T/got.txt"; else : > "$T/got.txt"; fi
same "the cumulative masses of the 24 gathered chunks at the last step" "$T/got.txt" "$T/x/want_cum.txt"
if [ -f "$T/x/cnt.f32" ]; then col "$T/x/cnt.f32" "$T/got.txt"; else : > "$T/got.txt"; fi
same "the columns before the target in each gathered chunk, exactly" "$T/got.txt" "$T/x/want_cnt.txt" 0
run GEN=1 SAMP=1 SAVE=1 KC=$KC TOPK=4 CB=16 PC=0.0 INVT=$INVT
same "sampling with no top-p and the 4 largest: the id each step drew, exactly" "$T/got_ans.txt" "$T/x/want_draws_k.txt" 0
if [ -f "$T/x/cum.f32" ]; then col "$T/x/cum.f32" "$T/got.txt"; else : > "$T/got.txt"; fi
same "its cumulative masses at the last step" "$T/got.txt" "$T/x/want_cum_k.txt"
# the draws must not all be the arg max, top-p must drop something somewhere and keep more than one somewhere, and both at the last step, whose masses are compared: or a sampler that takes the largest, or one with no top-p, would pass
if [ "$(awk '$1 != $2' "$T/x/draw_stats.txt" | wc -l)" -eq 0 ] || [ "$(awk '$4 < $3' "$T/x/draw_stats.txt" | wc -l)" -eq 0 ] || [ "$(awk '$4 > 1' "$T/x/draw_stats.txt" | wc -l)" -eq 0 ] || [ "$(tail -1 "$T/x/draw_stats.txt" | awk '{ print ($4 < $3 && $4 > 1 && $5 > 0.002) }')" != 1 ]; then echo "FAIL the reference's draws do not exercise the sampler (id, arg max, kept by top-k, kept by top-p, the dropped candidate's share of the mass): $(tr '\n' ';' < "$T/x/draw_stats.txt")"; fail=1; fi
if [ "$(sort -u "$T/x/want_ids.txt" | wc -l)" -lt 2 ] || [ "$(awk '{ print int($1 / 64) }' "$T/x/want_ids.txt" | sort -u | wc -l)" -lt 2 ] || [ "$(awk '$1 >= 64' "$T/x/want_ids.txt" | wc -l)" -eq 0 ]; then echo "FAIL the reference's four ids do not differ or are all in one chunk: $(tr '\n' ' ' < "$T/x/want_ids.txt")"; fail=1; fi
# a window that slides and caches in chunks (sections 15 and 19.2): 14 steps, so a sliding layer's window of 4 scanned keys has moved ten times and an indexed layer's 14 keys lie in four chunks of 4
RX="$T/xl"; RN=$NSL; RT=$TL; RTW=$TWL; RTS=$TWL; REQ=""; CK=1
run
if [ ! -s "$T/got_ans.txt" ]; then echo "FAIL the session answered nothing with caches that wrap: $(head -c 300 "$T/err.txt")"; fail=1; fi
same "a window of $WINL keys scanned as $TWL, caches of $TL positions in chunks of $TWL: the row after the last layer at each of $NSL steps" "$T/got_ans.txt" "$T/xl/want_ans.txt"
l=0; while [ $l -lt $NL ]; do
  if [ -f "$T/xl/out$l.f32" ]; then col "$T/xl/out$l.f32" "$T/got.txt"; else : > "$T/got.txt"; fi
  same "the row after layer $l at the last of them" "$T/got.txt" "$T/xl/want_out$l.txt" > "$T/q.txt"; grep FAIL "$T/q.txt" && fail=1
  l=$((l + 1))
done
if [ -f "$T/xl/logits.f32" ]; then col "$T/xl/logits.f32" "$T/got.txt"; else : > "$T/got.txt"; fi
same "the logits at the last of them" "$T/got.txt" "$T/xl/want_logits.txt"
# seek over caches that are rolls (section 19.2). After $NSL steps every position is in its block, so a step runs again as it ran from any of them: the last two, the last three (where a
# ring of $TWL slots had lost the window's first key and the answer was another: section 17.6), from the middle, and the whole run from 0
{ i=0; while [ $i -lt $NSL ]; do echo 1; i=$((i + 1)); done; echo "seek $((NSL - 2))"; echo 2; echo "seek $((NSL - 3))"; echo 1; echo "seek 5"; echo 1; echo 1; echo "seek 0"; i=0; while [ $i -lt $NSL ]; do echo 1; i=$((i + 1)); done; } > "$T/req_ring.txt"
REQ="$T/req_ring.txt"; CK=0; run SAVE=0
lines $NSL $NSL "$T/want.txt"; lines $((NSL + 2)) $((NSL + 2)) "$T/got.txt"
same "back two steps, both in one request: the last step's row again, exactly" "$T/got.txt" "$T/want.txt" 0
lines $((NSL - 2)) $((NSL - 2)) "$T/want.txt"; lines $((NSL + 4)) $((NSL + 4)) "$T/got.txt"
same "back three steps, past a window of $WINL: that step's row again, exactly" "$T/got.txt" "$T/want.txt" 0
lines 6 7 "$T/want.txt"; lines $((NSL + 6)) $((NSL + 7)) "$T/got.txt"
same "back to step 5: its row and the next again, exactly" "$T/got.txt" "$T/want.txt" 0
lines 1 $NSL "$T/want.txt"; lines $((NSL + 9)) $((2 * NSL + 8)) "$T/got.txt"
same "back to 0 after it: the $NSL steps again, exactly" "$T/got.txt" "$T/want.txt" 0
[ -s "$T/got.txt" ] || { echo "FAIL the seeks answered nothing: $(head -c 300 "$T/err.txt")"; fail=1; }
[ "$fail" -eq 0 ] && echo "kstep: ok"
exit $fail
