#!/bin/sh
# tests/q4.sh -- `dequant_q4` (docs/kolibri.md K1) on codes that do not repeat. The interpreter's made-up bytes for an i8 parameter are 0x30 + i % 8, the same in every tile of 16 bytes, so
# tests/kernels.py cannot tell a kernel that reads every tile's codes from the first tile. Here the bytes are (37 i + 11) mod 256, the scales and the rows' scales powers of two and their sums, and
# the answer is exact: every value of dq_q4.chx's output, and matvec_q4_rows.chx's product with a row of small whole numbers, against awk's arithmetic.
set -eu
cd "$(dirname "$0")/.."
G="${CANCHO_GPU:-build/cancho-gpu}"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fail=0
n=8; k=128
awk -v n=$n -v k=$k 'BEGIN {
  split("0.5 0.25 1.5 3 0.125 2 0.75 1 4 0.375 6 0.0625 1.25 0.5 3.5 2.5", sc, " "); split("1 2 0.5 4", rw, " ");
  for (i = 0; i < n * k / 2; i++) { b = (37 * i + 11) % 256; byte[i] = b; printf "%d ", (b > 127 ? b - 256 : b) > "'"$T"'/wq.txt" }
  for (i = 0; i < n * k / 32; i++) printf "%s ", sc[(i * 5) % 16 + 1] > "'"$T"'/ws.txt";
  for (i = 0; i < n; i++) printf "%s ", rw[i % 4 + 1] > "'"$T"'/wg.txt";
  for (c = 0; c < k; c++) { x[c] = (c * 7) % 5 - 2; printf "%d ", x[c] > "'"$T"'/x.txt" }
  for (r = 0; r < n; r++) { y = 0;
    for (c = 0; c < k; c++) { b = byte[r * k / 2 + int(c / 2)]; code = (c % 2 == 0) ? b % 16 : int(b / 16); v = (code - 8) * sc[((r * k / 32 + int(c / 32)) * 5) % 16 + 1] * rw[r % 4 + 1]; printf "%.9g\n", v > "'"$T"'/w.want"; y += v * x[c] }
    printf "%.9g\n", y > "'"$T"'/y.want" }
}'
same() { # name, got file (numbers in any layout), wanted file (a number a line)
  if tr -s ' \n' '\n\n' < "$2" | grep -v '^$' | awk '{printf "%.9g\n", $1}' | cmp -s - "$3"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi
}
"$G" run tests/chx/dq_q4.chx n=$n k=$k bn=2 bk=32 --target apple-m-series --in wq "$T/wq.txt" --in ws "$T/ws.txt" --in wg "$T/wg.txt" --out w > "$T/w.got"
same "dq_q4: 1,024 values of codes that do not repeat, exactly" "$T/w.got" "$T/w.want"
"$G" run tests/chx/dq_q4.chx n=$n k=$k bn=4 bk=64 --target nvidia-ada --in wq "$T/wq.txt" --in ws "$T/ws.txt" --in wg "$T/wg.txt" --out w > "$T/w2.got"
same "dq_q4, other tiles and the other target: the same values" "$T/w2.got" "$T/w.want"
"$G" run tests/chx/matvec_q4_rows.chx n=$n k=$k --target apple-m-series --in x "$T/x.txt" --in wq "$T/wq.txt" --in ws "$T/ws.txt" --in wg "$T/wg.txt" --out y > "$T/y.got"
same "matvec_q4_rows: the product, exactly" "$T/y.got" "$T/y.want"
# the mutant tests/kernels.py cannot kill: every tile's codes read from the first tile
sed 's|let q = load wq\[i \* bn, j \* (bk / 2) ; bn, bk / 2\]|let q = load wq[i * bn, 0 ; bn, bk / 2]|' tests/chx/dq_q4.chx > "$T/mut.chx"
if cmp -s tests/chx/dq_q4.chx "$T/mut.chx"; then echo "FAIL the mutant's text was not found in dq_q4.chx"; fail=1; fi
"$G" run "$T/mut.chx" n=$n k=$k bn=2 bk=32 --target apple-m-series --in wq "$T/wq.txt" --in ws "$T/ws.txt" --in wg "$T/wg.txt" --out w > "$T/wm.got"
if tr -s ' \n' '\n\n' < "$T/wm.got" | grep -v '^$' | awk '{printf "%.9g\n", $1}' | cmp -s - "$T/w.want"; then echo "FAIL the mutant (every tile's codes from the first) survives"; fail=1; else echo "ok   the mutant (every tile's codes from the first) is killed"; fi
[ "$fail" -eq 0 ] && echo "q4: ok"
exit $fail
