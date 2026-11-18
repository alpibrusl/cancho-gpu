#!/bin/sh
# build/media (media/media.cho) against tools/media.py: the same refusals, word for word, and the same exit status (docs/media-cho.md, slice A).
#
#   CANCHO=<compiler> [PYMEDIA=<scripts/media of a checkout that has build/cancho-gpu-metal-session>] sh tests/media_diff.sh
#
# tools/media.py asks for the Metal session before it checks anything, so PYMEDIA must be a checkout where it is built (default: this one).
# A case that Python would carry on into a GPU job (an accepted size) is not run through it: only the cancho side is checked, against numbers worked by hand.
set -e
cd "$(dirname "$0")/.."
C="${CANCHO:-cancho}"
PY="${PYMEDIA:-scripts/media}"
CANCHO="$C" sh scripts/build-media.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fail=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2', wanted '$3'"; fail=1; fi; }
HAVE_PY=1
if [ -z "${PYMEDIA:-}" ] && [ ! -e build/cancho-gpu-metal-session ]; then HAVE_PY=0; echo "note: no Metal session built here and no PYMEDIA: the comparison with tools/media.py is skipped"; fi
same() { # name, cancho args (as one line for display), then run both
  name=$1; shift
  if [ $HAVE_PY = 0 ]; then return 0; fi
  c_out=$(build/media "$@" 2>"$T/c.err") && c_code=0 || c_code=$?
  case $1$2 in checkvideo) shift 2; set -- video "PROMPT" -o "$T/o" --size "$1" --frames "$2" $( [ "$3" = 1 ] && echo --force ) ;; checkimage) shift 2; set -- image "$2" -o "$T/o" --size "$1" ;; esac
  $PY "$@" >"$T/p.out" 2>"$T/p.err" && p_code=0 || p_code=$?
  check "$name: exit" "$c_code" "$p_code"
  check "$name: stderr" "$(cat "$T/c.err")" "$(grep '^media:' "$T/p.err")"
}
same "image 641x512"        check image 641x512 "a fox"
same "image 32x512"         check image 32x512 "a fox"
same "image 1056x1056"      check image 1056x1056 "a fox"
same "image empty prompt"   check image 512x512 "   "
same "video 641x352"        check video 641x352 17 0
same "video 640x0"          check video 640x0 17 0
same "video 1280x704 65 frames"  check video 1280x704 65 0
same "video many tokens"    check video 640x352 200 0
# accepted: worked by hand. 17 frames: lf = (16 + 3) / 4 + 1 = 5; tokens = 5 * 11 * 20 = 1100
check "video 640x352 17"   "$(build/media check video 640x352 17 0)" "ok frames 17 latent 5 tokens 1100"
check "video 640x352 16 rounds up" "$(build/media check video 640x352 16 0)" "$(printf 'media: 16 frames requested; the model makes 1 + 4k, so 17\nok frames 17 latent 5 tokens 1100')"
check "video 1x frame"     "$(build/media check video 640x352 1 0)" "ok frames 1 latent 1 tokens 220"
check "image ok"           "$(build/media check image 1024x1024 "a fox")" ok
if [ $HAVE_PY = 1 ]; then check "models" "$(build/media models)" "$($PY models)"; fi
A=$($C authority media/media.cho media/pathx.cho media/stage.cho media/rewriter.cho studio/files.cho --std 2>&1)
check "authority: no network, no signals" "$(echo "$A" | grep -c -E '^    (the network|signals)$')" 2
check "authority: programs only beneath a directory" "$(echo "$A" | grep -c -E '^    (exec_under)$')" 1
check "authority: no exec of a path" "$(echo "$A" | grep -c -E 'exec\(')" 0
exit $fail
