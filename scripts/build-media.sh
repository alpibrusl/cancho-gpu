#!/bin/sh
# Build build/media from media/*.cho (docs/media-cho.md): CANCHO=<pinned cancho> scripts/build-media.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p build
"${CANCHO:-cancho}" build media/media.cho media/shared.cho media/video.cho media/transcribe.cho media/speak.cho media/clip.cho media/negative.cho media/image.cho media/png.cho media/pathx.cho media/stage.cho media/rewriter.cho studio/files.cho --std -o build/media 2>&1 | grep -v "^ld: warning" || true
test -x build/media
