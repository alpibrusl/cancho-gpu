#!/bin/sh
# build/media's `video` (from a prompt) against tools/media.py, with the stage replaced by a program that prints its arguments (docs/media-cho.md section 4): the same stage argument list, the same prompt.txt, the same
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
mkdir -p "$T/cwd"; W=$(cd "$T/cwd" && pwd -P); mkdir -p "$R/scripts" "$R/build" "$R/tools" "$H/.cache/huggingface/hub/models--FastVideo--FastWan2.2-TI2V-5B-FullAttn-Diffusers/snapshots/abc/transformer" "$H/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/abc/vae" "$K/wan-f16" "$T/w/dit" "$T/bin" "$W"
: >"$R/build/cancho-gpu-metal-session"
cp tools/media.py "$R/tools/media.py"
cat >"$R/scripts/ffmpeg" <<'EOS'
#!/bin/sh
for a in "$@"; do echo "STAGE $a"; done
EOS
cp "$R/scripts/ffmpeg" "$T/bin/ffmpeg"; chmod +x "$R/scripts/ffmpeg" "$T/bin/ffmpeg"
cat >"$R/scripts/stage" <<'EOS'
#!/bin/sh
for a in "$@"; do echo "STAGE $a"; done
EOS
chmod +x "$R/scripts/stage"
mkdir -p "$H/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/abc/transformer" "$T/w/base"
: >"$T/w/base/config.json"
: >"$R/build/wan_vae_enc_gpu"
cp "$R/scripts/stage" "$R/scripts/enc"; cp "$R/scripts/stage" "$R/scripts/enc-session"
cp tests/png/gradient1024.png "$T/cwd/fox.png"; cp tests/png/filter4_ties.png "$T/cwd/small.png"; cp tests/png/gray.png "$T/cwd/gray.png"
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
py() { (cd "$W" && PATH=$T/bin:$PATH HOME=${HOME2:-$H} CANCHO_MEDIA_CACHE=$K python3 -c 'import sys,subprocess,runpy
real=subprocess.run
subprocess.run=lambda cmd,cwd=None,**k:real(cmd,cwd=cwd,**k) if cmd[0].endswith("rewrite_cli") else [print("STAGE "+c) for c in cmd[1:]] and None or type("R",(),{"returncode":0})()
sys.argv=["media"]+sys.argv[1:]
runpy.run_path("'"$R"'/tools/media.py",run_name="__main__")' "$@"); }
cho() { (cd "$W" && "$REPO/build/media" --root "$R" --cwd "$W" --home "${HOME2:-$H}" --py /bin/true --cache "$K" --ffmpeg ${FREEKB:+--free-kb $FREEKB} "$@"); }
norm() { sed "s#$T#TMP#g; s#: [0-9]* s\$##; /^STAGE -c\$/d; s#^STAGE printf .* \(/[^ ]*\)\$#STAGE \1#"; }
both() { # name, args...: the whole output (timings dropped), exit status, prompt.txt and errors
  name=$1; shift
  rm -rf "$W/o"
  p_out=$(py "$@" 2>"$T/p.err") && p_code=0 || p_code=$?
  p_prompt=$(cat "$W/o/prompt.txt" 2>/dev/null || true); p_orig=$(cat "$W/o/prompt.orig.txt" 2>/dev/null || true); p_pic=$(cksum "$W/o/work/picture.f32" 2>/dev/null | cut -d" " -f1,2 || true); rm -rf "$W/o"
  c_out=$(cho "$@" 2>"$T/c.err") && c_code=0 || c_code=$?
  c_prompt=$(cat "$W/o/prompt.txt" 2>/dev/null || true); c_orig=$(cat "$W/o/prompt.orig.txt" 2>/dev/null || true); c_pic=$(cksum "$W/o/work/picture.f32" 2>/dev/null | cut -d" " -f1,2 || true)
  check "$name: picture floats" "$c_pic" "$p_pic"
  check "$name: exit" "$c_code" "$p_code"
  check "$name: output" "$(printf "%s\n" "$c_out" | norm)" "$(printf "%s\n" "$p_out" | norm)"
  check "$name: prompt.txt" "$c_prompt" "$p_prompt"
  check "$name: prompt.orig.txt" "$c_orig" "$p_orig"
  check "$name: errors" "$(grep -v '^$' "$T/c.err" | norm)" "$(grep -v '^$' "$T/p.err" | norm)"
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
both "video"                  video "a red fox" -o o
both "video sized"            video "a red fox" -o o --size 512x288 --frames 9 --seed 3 --fps 8
both "video 20 frames"        video "a red fox" -o o --frames 20
both "video weights flag"     video "a red fox" -o o --weights "$T/w/dit"
both "video wdir flag"        video "a red fox" -o o --wdir "$T/w/dit"
both "video rewritten"        video "a red fox" -o o --rewrite
both "video rewrite fails soft" video "FAILSOFT a red fox" -o o --rewrite --rewrite-model m1
both "video forced"           video "a fox" -o o --size 1280x704 --frames 33 --force
both "image"                  video "a fox runs" -o o --image fox.png
both "image small, nonsquare" video "a fox runs" -o o --image small.png --size 512x288 --frames 9 --seed 3 --fps 8
both "image grayscale"        video "a fox runs" -o o --image gray.png --size 384x256 --frames 5
both "image steps, guidance"  video "a fox runs" -o o --image small.png --steps 12 --guidance 4
both "image guidance 4.50"    video "a fox runs" -o o --image small.png --guidance 4.50
both "image no negative"      video "a fox runs" -o o --image small.png --negative ""
both "image own negative"     video "a fox runs" -o o --image small.png --negative "blurry, dark"
both "image base weights"     video "a fox runs" -o o --image small.png --weights "$T/w/base"
both "image wdir flag"        video "a fox runs" -o o --image small.png --wdir "$T/w/base"
both "image explicit model"   video "a fox runs" -o o --image small.png --model WAN2.2-TI2V-5B-BASE
refuse "image missing"        video "a fox" -o o --image nothere.png
refuse "image one frame"      video "a fox" -o o --image small.png --frames 1
refuse "image zero steps"     video "a fox" -o o --image small.png --steps 0
refuse "image weak guidance"  video "a fox" -o o --image small.png --guidance 0.5
refuse "image distilled model" video "a fox" -o o --image small.png --model fastwan2.2-ti2v-5b
refuse "image rewrite"        video "a fox" -o o --image small.png --rewrite
refuse "image weights no config" video "a fox" -o o --image small.png --weights "$T/w/dit"
refuse "image bad size"       video "a fox" -o o --image small.png --size 641x352
refuse "image too many tokens" video "a fox" -o o --image small.png --size 1280x704 --frames 33
refuse "video bad size"       video "a fox" -o o --size 641x352
refuse "video too many tokens" video "a fox" -o o --size 1280x704 --frames 33
refuse "video too many pixels" video "a fox" -o o --size 1024x576 --frames 25
refuse "video bad model"      video "a fox" -o o --model Nope
refuse "video base model"     video "a fox" -o o --model wan2.2-ti2v-5b-base
refuse "video segments"       video "a fox" -o o --segments 2
refuse "video then"           video "a fox" -o o --then "more"
refuse "video no-correct"     video "a fox" -o o --no-correct
mkdir -p "$T/empty"
HOME2=$T/empty refuse "video no weights" video "a fox" -o o
HOME2=$T/empty refuse "image no base model" video "a fox" -o o --image small.png
unset HOME2
# the disk check: only the cancho side can be told how much is free (Python asks the machine)
FREEKB=1000 cho video "a fox" -o o >/dev/null 2>"$T/c.err" && c_code=0 || c_code=$?
check "video low disk: exit" "$c_code" 2
check "video low disk: message" "$(grep -v '^$' "$T/c.err" | norm)" "media: the f16 weight files (10 GB) are written to TMP/cache/wan-f16 on first use and only 0 GB are free"
: >"$K/wan-f16/proj_out.f16"
FREEKB=1000 cho video "a fox" -o o >/dev/null 2>&1 && c_code=0 || c_code=$?
check "video low disk but the weight files exist: runs" "$c_code" 0
[ $fail = 0 ] && echo "media_video: ok"
exit $fail
