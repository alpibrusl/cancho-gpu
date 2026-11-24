#!/bin/sh
# build/media's `transcribe` (media/transcribe.cho, docs/audio.md section 16) with its stages replaced by programs that log their arguments and print the lines the real stages print. No GPU, no checkpoint:
# a temporary tree plays the repository's root, the Hugging Face cache and the weights. There is no Python of this command, so the expected lists are written here.
#
#   CANCHO=<compiler> sh tests/media_transcribe.sh
set -e
cd "$(dirname "$0")/.."
REPO=$(pwd)
C="${CANCHO:-cancho}"
CANCHO="$C" sh scripts/build-media.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/cwd"; W=$(cd "$T/cwd" && pwd -P); T=$(dirname "$W")
R=$T/root; H=$T/home; K=$T/cache
SNAPDIR=$H/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/abc
mkdir -p "$R/scripts" "$R/build" "$SNAPDIR" "$K/qwen3-asr-text-f16" "$T/ctl" "$T/empty"
for p in cancho-gpu-metal-session qwen3_asr_front qwen3_asr_audio_gpu qwen3_asr_text_gpu qwen3_asr_step_gpu; do : >"$R/build/$p"; done
: >"$K/qwen3-asr-text-f16/embt.f16"
: >"$W/clip.wav"
# the stages: every argument on a line of the log; then what the real program prints (tests/qwen3_asr_*.cho), unless a control file says to fail or to say nothing
cat >"$R/scripts/asr" <<EOS
#!/bin/sh
echo "asr \$*" >>"$T/log"
if [ -f "$T/ctl/fail" ] && [ "\$1 \$2" = "\$(cat "$T/ctl/fail")" ]; then echo "the stage's own line"; echo "more" >&2; exit 104; fi
if [ -f "$T/ctl/mute" ] && [ "\$1 \$2" = "\$(cat "$T/ctl/mute")" ]; then echo "nothing to read"; exit 0; fi
case "\$1 \$2" in
  "front mel") echo "frames \$(cat "$T/ctl/frames") rate 24000 samples 389760" ;;
  "audio prepare") echo "BP=1630 SP=256 tokens 211" ;;
  "front prompt") echo "prompt \$(cat "$T/ctl/prompt")" ;;
  "text inputs") echo "SP=256 tokens 225" ;;
  "step inputs") echo "T=384 NS=114 rows 211" ;;
  "front answer") echo "English"; echo "The quick brown fox." ;;
  *) echo "weights written" ;;
esac
EOS
cat >"$R/scripts/asr-session" <<EOS
#!/bin/sh
echo "asr-session \$*" >>"$T/log"
if [ -f "$T/ctl/fail" ] && [ "session" = "\$(cat "$T/ctl/fail")" ]; then echo "error[budget]: too much" >&2; exit 1; fi
EOS
chmod +x "$R/scripts/asr" "$R/scripts/asr-session"
fail=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1"; echo "  got:    $2"; echo "  wanted: $3"; fail=1; fi; }
cho() { (cd "$W" && "$REPO/build/media" --root "$R" --cwd "$W" --home "${HOME2:-$H}" --py /bin/true --cache "$K" ${FREEKB:+--free-kb $FREEKB} "$@"); }
norm() { sed "s#$T#TMP#g; s#: [0-9]*\.[0-9][0-9] s\$##; s# in [0-9]*\.[0-9][0-9] s\$##"; }
fresh() { rm -rf "$W/o" "$T/log" "$T/ctl/fail" "$T/ctl/mute"; echo 1624 >"$T/ctl/frames"; echo 226 >"$T/ctl/prompt"; }
run() { fresh; [ -n "${CTL:-}" ] && echo "$CTL" >"$T/ctl/$CTLNAME"; out=$(cho "$@" 2>"$T/err") && code=0 || code=$?; }
SN=TMP/home/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/abc
stages() { # the answer's steps, the positions: the log of a whole job
cat <<EOL
asr front mel TMP/cwd/clip.wav TMP/cwd/o/work/clip.mel.f32
asr audio prepare $SN TMP/cache/qwen3-asr-audio-f16 TMP/cwd/o/work TMP/cwd/o/work/clip.mel.f32 1624
asr-session 1 TMP/root/models/qwen3_asr_audio.model CHX=TMP/root/tests/chx CONV3=TMP/root/examples/cnn/conv3.chx WORK=TMP/cwd/o/work WDIR=TMP/cache/qwen3-asr-audio-f16 BP=1630 SP=256
asr front prompt 211 TMP/cwd/o/work/prompt.txt TMP/cwd/o/work/prefill.txt
asr text inputs TMP/cache/qwen3-asr-text-f16 TMP/cwd/o/work TMP/cwd/o/work/prefill.txt TMP/cwd/o/work/rows.f32
asr-session 1 TMP/root/models/qwen3_asr_text.model CHX=TMP/root/tests/chx WORK=TMP/cwd/o/work WDIR=TMP/cache/qwen3-asr-text-f16 SP=256 PRE=1 T=$2
asr step inputs TMP/cache/qwen3-asr-text-f16 TMP/cwd/o/work TMP/cwd/o/work/prompt.txt TMP/cwd/o/work/rows.f32 $1 $2 225
asr-session $1 TMP/root/models/qwen3_asr_step.model CHX=TMP/root/tests/chx QW=TMP/root/examples/qwen TR=TMP/root/examples/train WORK=TMP/cwd/o/work WDIR=TMP/cache/qwen3-asr-text-f16 T=$2 NS=$1 PRE=1
asr front answer $SN/vocab.json TMP/cwd/o/work/ids_out.f32 0
EOL
}
run transcribe clip.wav -o o
check "a clip: exit" "$code" 0
check "a clip: the stages (16.24 s: 113 tokens of budget, 384 positions)" "$(norm <"$T/log")" "$(stages 114 384)"
check "a clip: transcript.txt" "$(cat "$W/o/transcript.txt")" "English
The quick brown fox."
check "a clip: output" "$(printf '%s\n' "$out" | norm)" "== mel
== mel
== audio tower's inputs
== audio tower's inputs
== audio tower (GPU)
== audio tower (GPU)
== prompt
== prompt
== prompt's inputs
== prompt's inputs
== prompt in one pass (GPU)
== prompt in one pass (GPU)
== answer's inputs
== answer's inputs
== answer (GPU)
== answer (GPU)
== answer as text
== answer as text
English
The quick brown fox.
done: TMP/cwd/o (transcript.txt)"
check "a clip: no errors" "$(cat "$T/err")" ""
run transcribe -o o --most 40 -- clip.wav
check "--most 40, the clip after --: exit" "$code" 0
check "--most 40: the stages (41 steps, 320 positions)" "$(norm <"$T/log")" "$(stages 41 320)"
run transcribe "$W/clip.wav" --output "$W/o"
check "absolute paths, --output: the stages" "$(norm <"$T/log")" "$(stages 114 384)"
# the language model's weights: written when their last file is not there
rm "$K/qwen3-asr-text-f16/embt.f16"
run transcribe clip.wav -o o
check "no weights yet: the two programs that write them run first" "$(norm <"$T/log" | head -2)" "asr text weights $SN TMP/cache/qwen3-asr-text-f16
asr step weights $SN TMP/cache/qwen3-asr-text-f16"
check "no weights yet: then the nine stages" "$(norm <"$T/log" | tail -9)" "$(stages 114 384)"
FREEKB=1000 run transcribe clip.wav -o o
check "no weights and a full disk: exit" "$code" 2
check "no weights and a full disk: message" "$(norm <"$T/err")" "media: the f16 weight files (7 GB) are written to TMP/cache/qwen3-asr-text-f16 on first use and only 0 GB are free"
check "no weights and a full disk: no stage ran" "$(cat "$T/log" 2>/dev/null)" ""
: >"$K/qwen3-asr-text-f16/embt.f16"
FREEKB=1000 run transcribe clip.wav -o o
check "a full disk but the weights are there: runs" "$code" 0
# refusals before any stage
refuse() { name=$1; want=$2; shift 2; run "$@"; check "$name: exit" "$code" 2; check "$name: message" "$(norm <"$T/err")" "$want"; check "$name: no stage ran" "$(cat "$T/log" 2>/dev/null)" ""; }
refuse "no output" "media: the following arguments are required: clip, -o/--output" transcribe clip.wav
refuse "no clip" "media: the following arguments are required: clip, -o/--output" transcribe -o o
refuse "two clips" "media: unrecognized arguments" transcribe clip.wav clip.wav -o o
refuse "--most 0" "media: --most is the answer's budget of tokens, 1 to 4096" transcribe clip.wav -o o --most 0
refuse "--most 5000" "media: --most is the answer's budget of tokens, 1 to 4096" transcribe clip.wav -o o --most 5000
refuse "--most x" "media: argument --most: expected a value (a whole number where a number is asked)" transcribe clip.wav -o o --most x
refuse "no such clip" "media: no such clip: nothere.wav" transcribe nothere.wav -o o
HOME2=$T/empty refuse "no checkpoint" "media: weights not in the Hugging Face cache: \`hf download Qwen/Qwen3-ASR-1.7B\`" transcribe clip.wav -o o
unset HOME2
rm "$R/build/qwen3_asr_step_gpu"
refuse "a host program not built" "media: the host programs of the recognition are missing (build/qwen3_asr_front, _audio_gpu, _text_gpu, _step_gpu): build them with \`CANCHO=<pinned cancho> scripts/build-asr.sh\`" transcribe clip.wav -o o
: >"$R/build/qwen3_asr_step_gpu"; rm "$R/build/cancho-gpu-metal-session"
refuse "no session" "media: build/cancho-gpu-metal-session is missing: build it with \`CANCHO=<pinned cancho> scripts/build-metal-session.sh\` (cancho.toml names the compiler)" transcribe clip.wav -o o
: >"$R/build/cancho-gpu-metal-session"
# refusals of a stage: status 1, the stage's own line passed on, nothing after it
stop() { name=$1; ctl=$2; val=$3; want=$4; lines=$5; shift 5; CTLNAME=$ctl CTL=$val run "$@"; check "$name: exit" "$code" 1; check "$name: message" "$(norm <"$T/err")" "$want"; check "$name: the stages that ran" "$(wc -l <"$T/log" | tr -d ' ')" "$lines"; [ -e "$W/o/transcript.txt" ] && { echo "FAIL $name: a transcript was written"; fail=1; }; return 0; }
stop "the mel refuses the file" fail "front mel" "more
media: mel failed (exit 104): the stage's own line" 1 transcribe clip.wav -o o
stop "the tower's inputs fail" fail "audio prepare" "more
media: audio tower's inputs failed (exit 104): the stage's own line" 2 transcribe clip.wav -o o
stop "a session fails" fail "session" "error[budget]: too much
media: audio tower (GPU) failed (exit 1)" 3 transcribe clip.wav -o o
stop "the mel prints no frames" mute "front mel" "media: mel: the stage did not print \`frames \` and its number" 1 transcribe clip.wav -o o
stop "the tower's inputs print no numbers" mute "audio prepare" "media: audio tower's inputs: the stage did not print \`BP=\` and its number" 2 transcribe clip.wav -o o
stop "the prompt prints no length" mute "front prompt" "media: prompt: the stage did not print \`prompt \` and its number" 4 transcribe clip.wav -o o
stop "the answer has no end" fail "front answer" "more
media: answer as text failed (exit 104): the stage's own line
media: the answer had a budget of 113 tokens; if it did not end in them, pass --most with more" 9 transcribe clip.wav -o o
stop "a clip of 152.01 s" frames 15201 "media: the clip is 152 s; the longest that was transcribed to the reference's words is 152 s (docs/audio.md section 17)" 1 transcribe clip.wav -o o
stop "a clip of 400 s" frames 40000 "media: the clip is 400 s; the longest that was transcribed to the reference's words is 152 s (docs/audio.md section 17)" 1 transcribe clip.wav -o o
stop "more positions than the session holds" prompt 2000 "media: the prompt (2000 tokens) and the answer's budget (2100) need 4160 positions and the answer's session holds 4032: pass a smaller --most" 4 transcribe clip.wav -o o --most 2100
CTLNAME=prompt CTL=2000 run transcribe clip.wav -o o --most 2000
check "4032 positions: runs" "$code" 0
check "4032 positions: the steps' session" "$(grep -c 'T=4032 NS=2001 PRE=1' "$T/log")" 1
CTLNAME=frames CTL=15200 run transcribe clip.wav -o o
check "a clip of 152 s: runs, 928 tokens of budget" "$code $(grep -c ' NS=929 ' "$T/log")" "0 1"
# the wrappers
out=$(CANCHO_MEDIA_IMPL=python CANCHO_MEDIA_CACHE=$K scripts/media transcribe clip.wav -o "$W/o" 2>&1) && code=0 || code=$?
check "scripts/media with CANCHO_MEDIA_IMPL=python: refused, there is no Python of it" "$code $out" "2 media: transcribe is build/media's alone (there is no Python of it): build it with \`CANCHO=<pinned cancho> scripts/build-media.sh\`, and leave CANCHO_MEDIA_IMPL unset"
out=$(scripts/asr nothing 2>&1) && code=0 || code=$?
check "scripts/asr with another program's name" "$code $out" "100 asr: the first argument is front, audio, text or step"
[ $fail = 0 ] && echo "media_transcribe: ok"
exit $fail
