#!/bin/sh
# tests/moe.sh [session binary] -- models/kolibri_mlp.model (docs/kolibri.md K3) as a small layer, on the CPU session (the gate) or, given its binary, the Metal one (by hand): 64 wide, 128 experts
# of 64 of which 3 are taken, every file text. The answer is held to awk's double arithmetic of the same definition: the norm, the router, a stable sort from the largest, the shared expert and the
# three with sigmoid of their logits, the norm after and the residual. Two pairs of router rows are equal, one pair across the 64-entry chunks and one at the third place, so the three are right only if ties go to
# the smaller index. Then the same with the `down` matrices in Q4_1's form, and the dense MLP of layers 0 and 1.
# The codes and scales are not periodic in a block of the tables, or every expert would be the same matrix. MOE_MODEL=<file> runs another model file: the mutants of the model, by hand.
set -eu
cd "$(dirname "$0")/.."
B="${1:-build/cancho-gpu-session}"
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
fail=0
CHX=$(pwd)/tests/chx
D=64; NE=128; EH=64; K=3
files() { # moe, q41: the layer's files and want.txt (MOE: K indices, K weights, then the row)
awk -v D=$D -v NE=$NE -v EH=$EH -v K=$K -v MOE="$1" -v Q41="$2" -v T="$T" '
function byte(m, i) { return (37 * i + 101 * int(i / 61) + 11 + 53 * m) % 256 }
function sc(m, g) { return SC[(g * 5 + 3 * int(g / 7) + m) % 16 + 1] / 8 }
function off(m, g) { return ((g * 7 + int(g / 5) + m) % 9 - 4) / 32 }
function wv(m, cols, row, c, q41,   b, code, g) { b = byte(m, row * cols / 2 + int(c / 2)); code = (c % 2 == 0) ? b % 16 : int(b / 16); g = row * cols / 32 + int(c / 32); return (code - 8) * sc(m, g) + (q41 ? off(m, g) : 0) }
function emit(name, m, rows, cols, q41,   i, b, f) {
  f = T "/" name ".q4"; for (i = 0; i < rows * cols / 2; i++) { b = byte(m, i); printf "%d ", (b > 127 ? b - 256 : b) > f } printf "\n" > f; close(f)
  f = T "/" name ".s"; for (i = 0; i < rows * cols / 32; i++) printf "%s ", sc(m, i) > f; printf "\n" > f; close(f)
  if (q41) { f = T "/" name ".c"; for (i = 0; i < rows * cols / 32; i++) printf "%s ", off(m, i) > f; printf "\n" > f; close(f) }
}
function silu(v) { return v / (1 + exp(-v)) }
function mlp(mg, mu, md, base, h, q41,   r, c, g, u) { # y[] = down(silu(gate n) * (up n)), the matrices rows from block base
  for (r = 0; r < h; r++) { g = 0; u = 0; for (c = 0; c < D; c++) { g += wv(mg, D, base * h + r, c, 0) * n[c]; u += wv(mu, D, base * h + r, c, 0) * n[c] } a[r] = silu(g) * u }
  for (r = 0; r < D; r++) { y[r] = 0; for (c = 0; c < h; c++) y[r] += wv(md, h, base * D + r, c, q41) * a[c] }
}
BEGIN {
  split("0.5 0.25 1.5 3 0.125 2 0.75 1 4 0.375 6 0.0625 1.25 0.5 3.5 2.5", SC, " ")
  ss = 0; for (c = 0; c < D; c++) { x[c] = ((c * 37 + 11) % 29 - 14) / 8; w1[c] = 0.5 + (c % 7) / 8; w2[c] = 0.25 + (c % 5) / 4; ss += x[c] * x[c]; printf "%s ", x[c] > (T "/x.f32"); printf "%s ", w1[c] > (T "/ffn_norm.f32"); printf "%s ", w2[c] > (T "/post_norm.f32"); printf "1 " > (T "/ones_d.f32") }
  rs = 1 / sqrt(ss / D + 0.000001); for (c = 0; c < D; c++) n[c] = x[c] * rs * w1[c]
  h = MOE ? EH : D; for (c = 0; c < h; c++) printf "1 " > (T "/ones_h.f32")
  if (MOE) {
    for (e = 0; e < NE; e++) { src = (e == 70) ? 5 : ((e == 100) ? 20 : e); lg[e] = 0; for (c = 0; c < D; c++) { v = ((src * 31 + c * 17) % 23 - 11) / 64; lg[e] += v * n[c]; printf "%s ", v > (T "/route.f32") }
      bi = ((src * 13) % 11 - 5) / 16; if (src == 5) bi = 40; if (src == 20) bi = 30; z[e] = lg[e] + bi; printf "%s ", bi > (T "/bias.f32"); printf "%d ", e > (T "/ar.f32") }
    for (j = 0; j < K; j++) printf "%d ", j > (T "/sl.f32")
    emit("sh.gate", 1, EH, D, 0); emit("sh.up", 2, EH, D, 0); emit("sh.down", 3, D, EH, Q41); emit("ex.gate", 4, NE * EH, D, 0); emit("ex.up", 5, NE * EH, D, 0); emit("ex.down", 6, NE * D, EH, Q41)
    mlp(1, 2, 3, 0, EH, Q41); for (r = 0; r < D; r++) m[r] = y[r]
    for (j = 0; j < K; j++) { best = -1; for (e = 0; e < NE; e++) if (!(e in used) && (best < 0 || z[e] > z[best])) best = e; used[best] = 1; sel[j] = best; wt[j] = 1 / (1 + exp(-lg[best]))
      mlp(4, 5, 6, best, EH, Q41); for (r = 0; r < D; r++) m[r] += wt[j] * y[r] }
    for (j = 0; j < K; j++) printf "%d\n", sel[j] > (T "/want.txt"); for (j = 0; j < K; j++) printf "%.9g\n", wt[j] > (T "/want.txt")
  } else { emit("dn.gate", 1, D, D, 0); emit("dn.up", 2, D, D, 0); emit("dn.down", 3, D, D, Q41); mlp(1, 2, 3, 0, D, Q41); for (r = 0; r < D; r++) m[r] = y[r] }
  ss = 0; for (r = 0; r < D; r++) ss += m[r] * m[r]; rs = 1 / sqrt(ss / D + 0.000001)
  for (r = 0; r < D; r++) printf "%.9g\n", x[r] + m[r] * rs * w2[r] > (T "/want.txt")
}'
}
run() { # moe, q41
  printf '\n' | "$B" "${MOE_MODEL:-models/kolibri_mlp.model}" CHX="$CHX" W="$T" D=$D NE=$NE EH=$EH K=$K FF=$D BN=32 MOE="$1" Q41="$2" SAVE=0 ANS=1 F32= F16= I8= 2>&1 | head -c 20000 | tr -s ' \n' '\n\n' | grep -v '^$' > "$T/got.txt" || true
}
near() { # name, moe: the K indices exactly, the rest within 2e-5 of the larger of 1 and the value
  out=$(paste "$T/got.txt" "$T/want.txt" | awk -v K="$2" -v n="$(wc -l < "$T/want.txt")" 'NF != 2 { bad = "a line of " NF " fields: " $0; exit } { d = $1 - $2; if (d < 0) d = -d; s = ($2 < 0 ? -$2 : $2); if (s < 1) s = 1; if (NR <= K ? d != 0 : d > 2e-5 * s) { bad = "value " NR ": " $1 ", wanted " $2; exit } } END { if (bad != "") print bad; else if (NR != n) print NR " values, wanted " n; else print "ok" }')
  if [ "$out" = ok ]; then echo "ok   $1"; else echo "FAIL $1: $out"; fail=1; fi
}
files 1 0; run 1 0
if [ "$(head -3 "$T/want.txt" | tr '\n' ' ')" = "5 70 20 " ]; then echo "ok   the reference's three: 5, its equal 70, and 20 before its equal 100"; else echo "FAIL the reference's three: $(head -3 "$T/want.txt" | tr '\n' ' ')"; fail=1; fi
near "a mixture-of-experts layer: the three experts, their weights and the row" $K
# the mutant tests/kernels.py cannot see (made-up logits are never equal): ties not broken, moe_rank as samp_rank
mkdir "$T/chx"; cp tests/chx/*.chx "$T/chx/"; sed 's|let ea = step di|let ea = di * 0.0|' tests/chx/moe_rank.chx > "$T/chx/moe_rank.chx"
if cmp -s tests/chx/moe_rank.chx "$T/chx/moe_rank.chx"; then echo "FAIL the mutant's text was not found in moe_rank.chx"; fail=1; fi
keep=$CHX; CHX="$T/chx"; run 1 0; CHX=$keep
three=$(head -3 "$T/got.txt" | tr '\n' ' ')
case "$three" in
  "5e0 7e1 2e1 ") echo "FAIL the mutant (ties not broken) survives"; fail=1 ;;
  *[!0-9e.\ -]*) echo "FAIL the mutant (ties not broken) did not run: $(head -c 200 "$T/got.txt")"; fail=1 ;;
  *) echo "ok   the mutant (ties not broken) is killed: it takes $three" ;;
esac
files 1 1; run 1 1
near "the same with the down matrices in Q4_1's form" $K
files 0 1; run 0 1
near "the dense MLP, its down matrix in Q4_1's form" 0
files 0 0; run 0 0
near "the dense MLP in Q4_0" 0
[ "$fail" -eq 0 ] && echo "moe: ok"
exit $fail
