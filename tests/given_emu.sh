#!/bin/sh
# tests/given_emu.sh -- emitted kernels run on the emulators with inputs that are given, where the cases of tests/cases.txt run on inputs one generator makes (its f32 are
# under a half either way and its bytes 48 to 55, so an index it makes is always 0 and a byte never negative). Each run is held to the interpreter by the device program
# itself; here the difference must be nothing, or for a product no more than 1e-5 of the peak, and where awk can say the answer it does.
#  1. pick_block.chx (docs/kolibri.md section 26): blocks picked by numbers that are in range, past it on either side, and between two whole numbers;
#  2. matvec_i8_rows.chx (section 25): codes over the whole range of a signed byte, a scale a group, a scale a row;
#  3. expert2_gu.chx, expert2_down.chx, expert2_down41.chx: a token's experts picked out of tables whose every expert is other numbers.
# CUDA's emulator, then HIP's. Needs scripts/build-emu.sh and scripts/build-device.sh.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd -P)
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
EMU=$ROOT/build/emu
fail=0
[ -x build/cancho-gpu-device ] && [ -f "$EMU/libcuda_emu.so" ] || { echo "FAIL given_emu: build/cancho-gpu-device or the emulators are not built"; exit 1; }
dev() { # backend, then the device program's arguments: its output in $T/out.txt
  b=$1; shift
  if [ "$b" = hip ]; then CANCHOGPU_LIBHIP=$EMU/libamdhip64_emu.so CANCHOGPU_LIBHIPRTC=$EMU/libhiprtc_emu.so CANCHOGPU_EMU_CACHE=$EMU/cache build/cancho-gpu-device run "$@" > "$T/out.txt" 2>&1
  else CANCHOGPU_LIBCUDA=$EMU/libcuda_emu.so CANCHOGPU_LIBNVRTC=$EMU/libnvrtc_emu.so CANCHOGPU_EMU_CACHE=$EMU/cache build/cancho-gpu-device run "$@" > "$T/out.txt" 2>&1; fi
}
# the device agrees with the interpreter to `tol` of the peak on every output, with no finding of the emulator
agrees() { # name, tol
  if grep -q '^compare: ' "$T/out.txt" && ! grep -qE '^(emu-race|emu-warn|error)' "$T/out.txt" && [ -z "$(sed -n 's/^compare: .* = \(.*\) of peak \(.*\)$/\1 \2/p' "$T/out.txt" | awk -v tol="$2" '$2 > 0 && $1 / $2 > tol { print "x" }')" ]; then echo "ok   $1"; else echo "FAIL $1: $(grep -E '^(compare|emu-|error)' "$T/out.txt" | head -3 | tr '\n' ';' | cut -c1-300)"; fail=1; fi
}
awk 'BEGIN { for (i = 0; i < 40; i++) printf "%d ", i; print "" }' > "$T/w.txt"
block() { awk -v b="$1" 'BEGIN { for (i = 0; i < 8; i++) printf "%d.0 ", b * 8 + i }'; }
for be in cuda hip; do
  # --- 1. pick_block: 5 blocks of 2 rows of 4, block b the numbers 8 b .. 8 b + 7
  for c in "3 1 3 1" "0 4 0 4" "9 -2 4 0" "2.7 0.2 2 0" "4.5 -0.5 4 0"; do
    set -- $c
    echo "$1 $2" > "$T/sel.txt"
    dev $be tests/chx/pick_block.chx ne=5 n=2 k=4 --in w "$T/w.txt" --in sel "$T/sel.txt" --out y
    got=$(tail -1 "$T/out.txt")
    if [ "$got" = "$(block "$3")$(block "$4" | sed 's/ $//')" ]; then echo "ok   $be: pick_block by $1 and $2: blocks $3 and $4"; else echo "FAIL $be: pick_block by $1 and $2: \`$(printf '%s' "$got" | cut -c1-120)\`"; fail=1; fi
    agrees "$be: and the interpreter says the same" 0
  done
  # --- 2. byte codes of either sign
  awk 'BEGIN { for (r = 0; r < 8; r++) for (c = 0; c < 32; c++) printf "%d ", (r * 37 + c * 61 + 5) % 256 - 128; print "" }' > "$T/wq.txt"
  awk 'BEGIN { for (r = 0; r < 8; r++) for (g = 0; g < 2; g++) printf "%s ", (1 + (r * 3 + g * 5) % 7) / 64; print "" }' > "$T/ws.txt"
  awk 'BEGIN { for (r = 0; r < 8; r++) printf "%s ", 1 + r / 4; print "" }' > "$T/wg.txt"
  awk 'BEGIN { for (c = 0; c < 32; c++) printf "%s ", ((c * 11 + 3) % 17 - 8) / 8; print "" }' > "$T/x.txt"
  dev $be tests/chx/matvec_i8_rows.chx n=8 k=32 --in x "$T/x.txt" --in wq "$T/wq.txt" --in ws "$T/ws.txt" --in wg "$T/wg.txt" --out y
  agrees "$be: matvec_i8_rows with codes from -128 to 127 is the interpreter's" 0.00001
  want=$(awk 'BEGIN { for (r = 0; r < 8; r++) { z = 0; for (c = 0; c < 32; c++) z += (((c * 11 + 3) % 17 - 8) / 8) * ((r * 37 + c * 61 + 5) % 256 - 128) * ((1 + (r * 3 + int(c / 16) * 5) % 7) / 64) * (1 + r / 4); printf "%.3f ", z } }')
  got=$(tail -1 "$T/out.txt" | awk '{ for (i = 1; i <= NF; i++) printf "%.3f ", $i }')
  if [ "$got" = "$want" ] && grep -q -- "-" "$T/wq.txt"; then echo "ok   $be: and awk's sums of code times scale times row scale"; else echo "FAIL $be: matvec_i8_rows: \`$got\`, wanted \`$want\`"; fail=1; fi
  # --- 3. a token's two experts out of tables of eight, every expert other bytes: the token is row 1 of 4, its experts sel[2] and sel[3]
  awk 'BEGIN { for (r = 0; r < 256; r++) for (c = 0; c < 32; c++) printf "%d ", (int(r / 32) * 53 + r * 7 + c * 29) % 256 - 128; print "" }' > "$T/gq.txt"
  awk 'BEGIN { for (r = 0; r < 256; r++) for (c = 0; c < 32; c++) printf "%d ", (int(r / 32) * 91 + r * 3 + c * 13 + 7) % 256 - 128; print "" }' > "$T/uq.txt"
  nums() { awk -v n="$1" -v a="$2" -v b="$3" -v d="$4" 'BEGIN { for (i = 0; i < n; i++) printf "%s ", ((i * a + 3) % b - int(b / 2)) / d; print "" }'; }   # n numbers of a spread, a file
  nums 256 7 23 16 > "$T/xr.txt"; nums 512 5 9 64 > "$T/gs.txt"; nums 512 11 7 64 > "$T/us.txt"; nums 32 3 5 4 > "$T/wg32.txt"
  GU="--in x $T/xr.txt --in gq $T/gq.txt --in gs $T/gs.txt --in uq $T/uq.txt --in us $T/us.txt --in wg $T/wg32.txt"
  echo "1 2 6 3 5 0 7 4" > "$T/sel8.txt"
  dev $be tests/chx/expert2_gu.chx r=4 b=1 n=32 k=64 ne=8 $GU --in sel "$T/sel8.txt" --out y
  agrees "$be: expert2_gu with experts 6 and 3 of eight is the interpreter's" 0.00001
  cp "$T/out.txt" "$T/picked.txt"
  echo "1 2 3 6 5 0 7 4" > "$T/sel8.txt"
  dev $be tests/chx/expert2_gu.chx r=4 b=1 n=32 k=64 ne=8 $GU --in sel "$T/sel8.txt" --out y
  if [ "$(tail -1 "$T/out.txt" | awk '{ for (i = 33; i <= 64; i++) printf "%s ", $i; for (i = 1; i <= 32; i++) printf "%s ", $i }')" = "$(tail -1 "$T/picked.txt" | awk '{ for (i = 1; i <= 64; i++) printf "%s ", $i }')" ] && [ "$(tail -1 "$T/out.txt")" != "$(tail -1 "$T/picked.txt")" ]; then echo "ok   $be: with the two experts in the other order its two rows are in the other order"; else echo "FAIL $be: expert2_gu does not follow its picks"; fail=1; fi
  awk 'BEGIN { for (r = 0; r < 512; r++) for (c = 0; c < 16; c++) printf "%d ", (int(r / 64) * 67 + r * 5 + c * 23) % 256 - 128; print "" }' > "$T/dq.txt"
  echo "1 2 6 3 5 0 7 4" > "$T/sel8.txt"
  nums 64 7 23 16 > "$T/a.txt"; nums 512 5 9 64 > "$T/ds.txt"; nums 512 13 11 32 > "$T/dc.txt"; nums 64 3 5 4 > "$T/wg64.txt"; nums 8 3 7 8 > "$T/wt.txt"; nums 256 5 19 16 > "$T/m.txt"; nums 2 1 3 2 > "$T/s2.txt"
  for k in expert2_down expert2_down41; do
    DN="--in a $T/a.txt --in dq $T/dq.txt --in ds $T/ds.txt --in wg $T/wg64.txt --in wt $T/wt.txt --in m $T/m.txt"
    [ $k = expert2_down ] || DN="$DN --in s $T/s2.txt --in dc $T/dc.txt"
    dev $be tests/chx/$k.chx r=4 b=1 n=64 k=32 ne=8 $DN --in sel "$T/sel8.txt" --out mo
    agrees "$be: $k with experts 6 and 3 of eight is the interpreter's" 0.00001
    cp "$T/out.txt" "$T/picked.txt"
    echo "1 2 6 5 5 0 7 4" > "$T/sel8b.txt"
    dev $be tests/chx/$k.chx r=4 b=1 n=64 k=32 ne=8 $DN --in sel "$T/sel8b.txt" --out mo
    if [ "$(tail -1 "$T/out.txt")" != "$(tail -1 "$T/picked.txt")" ]; then echo "ok   $be: and another second expert gives another row"; else echo "FAIL $be: $k does not follow its second pick"; fail=1; fi
  done
done
if [ $fail -ne 0 ]; then echo "FAIL given_emu"; exit 1; fi
echo "given_emu: ok"
