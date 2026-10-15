#!/bin/sh
# Record scripts/demo.sh with asciinema, render it with agg and encode docs/demo.mp4, docs/demo.webm
# and docs/demo-poster.png with ffmpeg. The commands in the session are run for real on this machine:
# the compiler and the 27B runner must be built (scripts/build-run35.sh, `cancho build`) and the model
# must be in Ollama's store. Needs asciinema, agg, ffmpeg.
#
#     scripts/make-demo.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
TERM=xterm-256color asciinema rec --overwrite --quiet --cols 120 --rows 26 --title "cancho-gpu" -c "bash scripts/demo.sh" docs/demo.cast
# No idle time is cut: the pauses are the machine's (loading the weights takes what it takes).
agg --theme github-dark --font-size 18 --idle-time-limit 30 --no-loop --last-frame-duration 4 docs/demo.cast "$WORK/demo.gif"
ffmpeg -y -loglevel error -i "$WORK/demo.gif" -movflags +faststart -pix_fmt yuv420p \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -crf 22 docs/demo.mp4
ffmpeg -y -loglevel error -i "$WORK/demo.gif" -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
  -c:v libvpx-vp9 -crf 36 -b:v 0 docs/demo.webm
# The poster: the last frame, where the whole session is on screen.
ffmpeg -y -loglevel error -sseof -2 -i docs/demo.mp4 -frames:v 1 docs/demo-poster.png
ls -l docs/demo.cast docs/demo.mp4 docs/demo.webm docs/demo-poster.png
