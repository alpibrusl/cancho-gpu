#!/bin/sh
# Download the public-domain (CC0) photograph that the image-editing tests use, and check it: nothing is committed, and no photograph of a user is ever a test input (docs/diffusion.md D9).
#
#     scripts/fetch-test-photo.sh DIR        writes DIR/berlin_park.jpg
#
# "People hanging out in James-Simon-Park at dusk, Berlin, 2017" by DimiTalen (own work, 2017-07-29), Wikimedia Commons, CC0 1.0 Universal Public Domain Dedication.
# File page: https://commons.wikimedia.org/wiki/File:People_hanging_out_in_James-Simon-Park_at_dusk,_Berlin,_2017.jpg   5,877 x 3,907 px, 17,285,200 bytes.
set -eu
DIR=${1:?usage: fetch-test-photo.sh DIR}
SHA=9e3f2bf9af2ed0da629fd6db90f8b005125df438bf5521146e04fc72aa4efc92
mkdir -p "$DIR"
OUT="$DIR/berlin_park.jpg"
if [ ! -f "$OUT" ]; then
  curl -sS -L -A "cancho-gpu-tests/1.0 (image-editing test photo)" -o "$OUT" \
    "https://upload.wikimedia.org/wikipedia/commons/8/83/People_hanging_out_in_James-Simon-Park_at_dusk%2C_Berlin%2C_2017.jpg"
fi
GOT=$(shasum -a 256 "$OUT" | cut -d' ' -f1)
[ "$GOT" = "$SHA" ] || { echo "fetch-test-photo: checksum mismatch ($GOT)" >&2; rm -f "$OUT"; exit 1; }
echo "$OUT"
