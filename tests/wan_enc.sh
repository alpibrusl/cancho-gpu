#!/bin/sh
# tests/wan_enc.sh -- the half and single encoders of the Wan VAE encoder's generator (tests/wan_enc/codec.cho, docs/wan-vae-encoder.md) held to known bit patterns, and one mutant that must fail:
# ties rounded up instead of to even. Needs no GPU and no checkpoint.
set -eu
cd "$(dirname "$0")/.."
C="${CANCHO:-cancho}"
mkdir -p build
$C build tests/wan_enc/codec_check.cho tests/wan_enc/codec.cho studio/files.cho --std -o build/codec_check 2>&1 | grep -v "^ld: warning" || true
build/codec_check
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
sed 's/m10 \& 1 == 1/true/' tests/wan_enc/codec.cho > "$T/codec.cho"
if cmp -s "$T/codec.cho" tests/wan_enc/codec.cho; then echo "FAIL the mutant changed nothing"; exit 1; fi
$C build tests/wan_enc/codec_check.cho "$T/codec.cho" studio/files.cho --std -o "$T/mutant" 2>&1 | grep -v "^ld: warning" || true
if "$T/mutant" >/dev/null 2>&1; then echo "FAIL the mutant (ties rounded up) survived"; exit 1; fi
echo "ok   wan_enc codec: known patterns pass, the tie mutant is refused"
