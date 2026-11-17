#!/bin/sh
# build/media's `image` and `edit` against tools/media.py, with the stage replaced by a program that prints its arguments (docs/media-cho.md section 4): the same stage argument list, the same prompt.txt, the same
# refusals (message and exit status). No GPU, no weights: a temporary tree plays the repository's root, the Hugging Face cache and the weights.
#
#   CANCHO=<compiler> sh tests/media_stage.sh
set -e
cd "$(dirname "$0")/.."
REPO=$(pwd)
C="${CANCHO:-cancho}"
CANCHO="$C" sh scripts/build-media.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
R=$T/root; H=$T/home; K=$T/cache; W=$T/cwd
mkdir -p "$T/cwd"; W=$(cd "$T/cwd" && pwd -P); mkdir -p "$R/scripts" "$R/build" "$R/tools" "$H/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/abc/transformer" "$K/qwen-edit-dit" "$K/qwen-edit-text" "$T/w/dit" "$W"
: >"$R/build/cancho-gpu-metal-session"
cp tools/media.py "$R/tools/media.py"
cat >"$R/scripts/stage" <<'EOS'
#!/bin/sh
for a in "$@"; do echo "STAGE $a"; done
EOS
chmod +x "$R/scripts/stage"
mkdir -p "$R/studio/rewrite"
for k in video edit edit2; do echo "system $k" >"$R/studio/rewrite/$k.txt"; done
cat >"$R/build/rewrite_cli" <<'EOS'
#!/bin/sh
case "$*" in *FAILSOFT*) echo "code 3 the model is not there"; exit 3 ;; esac
printf 'code 0 ok\nREWRITTEN %s\n' "$*"
EOS
chmod +x "$R/build/rewrite_cli"
: >"$W/a.png"; : >"$W/b.png"; : >"$W/c.png"
fail=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1"; echo "  got:    $2"; echo "  wanted: $3"; fail=1; fi; }
# Python's list, one STAGE line per argument (the interpreter is not one)
py() { (cd "$W" && HOME=$H CANCHO_MEDIA_CACHE=$K python3 -c 'import sys,subprocess,runpy
real=subprocess.run
subprocess.run=lambda cmd,cwd=None,**k:real(cmd,cwd=cwd,**k) if cmd[0].endswith("rewrite_cli") else [print("STAGE "+c) for c in cmd[1:]] and None or type("R",(),{"returncode":0})()
sys.argv=["media"]+sys.argv[1:]
runpy.run_path("'"$R"'/tools/media.py",run_name="__main__")' "$@"); }
cho() { (cd "$W" && "$REPO/build/media" --root "$R" --cwd "$W" --home "$H" --py /bin/true --cache "$K" "$@"); }
both() { # name, args...
  name=$1; shift
  rm -rf "$W/o"
  p_out=$(py "$@" 2>"$T/p.err") && p_code=0 || p_code=$?
  p_prompt=$(cat "$W/o/prompt.txt" 2>/dev/null || true); rm -rf "$W/o"
  c_out=$(cho "$@" 2>"$T/c.err") && c_code=0 || c_code=$?
  c_prompt=$(cat "$W/o/prompt.txt" 2>/dev/null || true)
  check "$name: exit" "$c_code" "$p_code"
  check "$name: stage arguments" "$(echo "$c_out" | grep '^STAGE ')" "$(echo "$p_out" | grep '^STAGE ')"
  check "$name: done line" "$(echo "$c_out" | grep '^done:' | sed "s#$W/##")" "$(echo "$p_out" | grep '^done:' | sed "s#$W/##")"
  check "$name: prompt.txt" "$c_prompt" "$p_prompt"
  check "$name: errors" "$(grep -v '^$' "$T/c.err" | sed "s#$T#TMP#g")" "$(grep -v '^$' "$T/p.err" | sed "s#$T#TMP#g")"
}
refuse() { # name, args...: both refuse the same way
  name=$1; shift
  py "$@" >/dev/null 2>"$T/p.err" && p_code=0 || p_code=$?
  cho "$@" >/dev/null 2>"$T/c.err" && c_code=0 || c_code=$?
  check "$name: exit" "$c_code" "$p_code"
  check "$name: message" "$(grep -v '^$' "$T/c.err" | sed "s#$T#TMP#g")" "$(grep -v '^$' "$T/p.err" | sed "s#$T#TMP#g")"
}
rw() { # name, args...: stdout, errors and exit status of `rewrite` (no GPU)
  name=$1; shift
  p_out=$(py "$@" 2>"$T/p.err") && p_code=0 || p_code=$?
  c_out=$(cho "$@" 2>"$T/c.err") && c_code=0 || c_code=$?
  check "$name: exit" "$c_code" "$p_code"
  check "$name: stdout" "$(echo "$c_out" | sed "s#$T#TMP#g")" "$(echo "$p_out" | sed "s#$T#TMP#g")"
  check "$name: errors" "$(grep -v '^$' "$T/c.err" | sed "s#$T#TMP#g")" "$(grep -v '^$' "$T/p.err" | sed "s#$T#TMP#g")"
}
both "image"                  image "a red fox" -o o
both "image sized"            image "a red fox" -o o --size 640X480 --seed 7 --steps 4
both "image weights flags"    image "a red fox" -o o --weights "$T/w/dit" --text-weights "$T/w/dit"
both "edit"                   edit a.png "make it night" -o o
both "edit two pictures"      edit a.png "combine them" -o o --also b.png
both "edit three, area"       edit a.png "all three" -o o --also b.png --also c.png --condition-area 400000 --seed 3 --steps 6
both "edit rewritten"         edit a.png "make it night" -o o --rewrite
both "edit rewritten, two"    edit a.png "combine them" -o o --also b.png --rewrite --rewrite-model m1
both "edit rewrite fails soft" edit a.png "FAILSOFT make it night" -o o --rewrite
refuse "edit rewrite 3 pictures" edit a.png "x" -o o --also b.png --also c.png --rewrite
rw "rewrite edit"             rewrite edit "make it night" --image a.png
rw "rewrite edit two"         rewrite edit "combine" --image a.png --also b.png --rewrite-model m2
rw "rewrite video"            rewrite video "a fox runs"
rw "rewrite fails soft"       rewrite video "FAILSOFT a fox runs"
rw "rewrite edit no image"    rewrite edit "x"
rw "rewrite video image"      rewrite video "x" --image a.png
rw "rewrite missing image"    rewrite edit "x" --image nothere.png
rw "rewrite also on video"    rewrite video "x" --also b.png
rw "rewrite three"            rewrite edit "x" --image a.png --also b.png --also c.png
rw "rewrite missing also"     rewrite edit "x" --image a.png --also nothere.png
rw "rewrite empty"            rewrite video "   "
refuse "image bad size"       image "a fox" -o o --size 641x512
refuse "image empty prompt"   image "  " -o o
refuse "image no weights"     image "a fox" -o o --weights "$T/nothere"
refuse "image bad model"      image "a fox" -o o --model Nope
refuse "edit missing picture" edit nothere.png "x" -o o
refuse "edit missing also"    edit a.png "x" -o o --also nothere.png
refuse "edit area too big"    edit a.png "x" -o o --also b.png --also c.png --condition-area 1048576
rm -rf "$H/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/abc/transformer"
refuse "image no snapshot"    image "a fox" -o o
mv "$R/build/rewrite_cli" "$R/build/rewrite_cli.off"
rw "rewrite no cli"           rewrite video "a fox"
refuse "edit rewrite no cli"  edit a.png "x" -o o --rewrite
mv "$R/build/rewrite_cli.off" "$R/build/rewrite_cli"
rm -rf "$R/build/cancho-gpu-metal-session"
refuse "image no session"     image "a fox" -o o
# a failing stage: build/media reports it as the Python does and exits 1
mkdir -p "$H/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/abc/transformer"; : >"$R/build/cancho-gpu-metal-session"
printf '#!/bin/sh\necho oops >&2\nexit 3\n' >"$R/scripts/stage"
c_err=$(cho image "a fox" -o o 2>&1 >/dev/null) && c_code=0 || c_code=$?
check "failing stage: exit" "$c_code" 1
check "failing stage: message" "$(echo "$c_err" | grep -c 'text to image (GPU, three stages) failed (exit 3)')" 1
check "failing stage: stage's errors are shown" "$(echo "$c_err" | grep -c oops)" 1
exit $fail
