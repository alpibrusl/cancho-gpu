#!/bin/sh
# media/png.cho against PIL, through goldens (docs/png-cho.md): every fixture in tests/png/ decodes to the width, height and RGB bytes PIL gave (the .sha file holds
# `W H sha256-of-RGB`), the refusals say what they mean, and an encoded picture decodes to what went in. No Python runs here: the goldens were made once, with PIL.
#
#   CANCHO=<compiler> sh tests/png_diff.sh
set -e
cd "$(dirname "$0")/.."
R=$(pwd -P)
C="${CANCHO:-cancho}"
mkdir -p build
"$C" build tests/png_tool.cho media/png.cho media/image.cho studio/files.cho --std -o build/png_tool 2>&1 | grep -v "^ld: warning" || true
test -x build/png_tool
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
fail=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2', wanted '$3'"; fail=1; fi; }
for f in tests/png/*.sha; do
  n=$(basename "$f" .sha)
  out=$(build/png_tool decode "$R/tests/png/$n.png" "$T/o.rgb" 2>&1) || out="refused: $out"
  h=$(shasum -a 256 "$T/o.rgb" 2>/dev/null | cut -d' ' -f1)
  check "decode $n" "$out $h" "$(cat "$f")"
  # an encoded copy decodes to the same bytes
  w=$(cut -d' ' -f1 "$f"); ht=$(cut -d' ' -f2 "$f")
  build/png_tool encode "$w" "$ht" "$T/o.rgb" "$T/e.png"
  build/png_tool decode "$T/e.png" "$T/e.rgb" >/dev/null
  check "roundtrip $n" "$(shasum -a 256 "$T/e.rgb" | cut -d' ' -f1)" "$h"
  rm -f "$T/o.rgb" "$T/e.rgb" "$T/e.png"
done
# the picture covering W x H and cropped at the centre, as planar singles: the bytes tools/media.py picture_f32 writes (media/image.cho, Pillow's LANCZOS)
while read -r n w ht want; do
  build/png_tool cover "$R/tests/png/$n.png" "$w" "$ht" "$T/c.f32"
  check "cover $n ${w}x$ht" "$(shasum -a 256 "$T/c.f32" | cut -d' ' -f1)" "$want"
done < tests/png/cover.list
refuse() { # file, message
  out=$(build/png_tool decode "$R/tests/png/$1.png" "$T/o.rgb" 2>&1) && code=0 || code=$?
  check "refuse $1" "$code $out" "1 png: $2"
}
refuse bad_sig "not a PNG file"
refuse empty "not a PNG file"
refuse bad_crc "damaged PNG (chunk length or checksum)"
refuse trunc "damaged PNG (chunk length or checksum)"
refuse gray16 "unsupported PNG (only 8-bit, not interlaced, at most 16384 x 16384)"
printf 'abc' > "$T/s.rgb"
out=$(build/png_tool encode 2 2 "$T/s.rgb" "$T/s.png" 2>&1) && code=0 || code=$?
check "encode wrong size" "$code $out" "1 png: bad size"
check "encode golden" "$(build/png_tool encode 3 2 "$R/tests/png/golden3x2.rgb" "$T/g.png"; shasum -a 256 "$T/g.png" | cut -d' ' -f1)" "$(cat tests/png/golden3x2.png.sha256)"
[ $fail = 0 ] && echo "png_diff: ok"
exit $fail
