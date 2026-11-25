#!/bin/sh
# build/media's `speak` (media/speak.cho, docs/audio.md section 28) with its stages replaced by programs that log their arguments and print the lines the real stages print. No GPU, no checkpoint: a
# temporary tree plays the repository's root, the Hugging Face cache and the weights. There is no Python of this command, so the expected lists are written here.
#
#   CANCHO=<compiler> sh tests/media_speak.sh
set -e
cd "$(dirname "$0")/.."
REPO=$(pwd)
C="${CANCHO:-cancho}"
CANCHO="$C" sh scripts/build-media.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/cwd"; W=$(cd "$T/cwd" && pwd -P); T=$(dirname "$W")
R=$T/root; H=$T/home; K=$T/cache
SNAPDIR=$H/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/abc
mkdir -p "$R/scripts" "$R/build" "$SNAPDIR" "$K/qwen3-tts-tables" "$K/qwen3-tts-step-f16" "$K/qwen3-tts-decoder-f32" "$T/ctl" "$T/empty"
for p in cancho-gpu-metal-session cancho-gpu qwen3_tts_front qwen3_tts_lm_gpu qwen3_tts_step_gpu qwen3_tts_decoder_gpu; do : >"$R/build/$p"; done
weights() { : >"$K/qwen3-tts-tables/tokenizer.json"; : >"$K/qwen3-tts-step-f16/sumt.f16"; : >"$K/qwen3-tts-decoder-f32/fc_b.f32"; }
weights
# the stages: every argument on a line of the log; then what the real program prints (tests/qwen3_tts_*.cho), unless a control file says to fail or to say nothing
cat >"$R/scripts/tts" <<EOS
#!/bin/sh
echo "tts \$*" >>"$T/log"
if [ -f "$T/ctl/fail" ] && { [ "\$1 \$2" = "\$(cat "$T/ctl/fail")" ] || [ "\$1" = "\$(cat "$T/ctl/fail")" ]; }; then echo "the stage's own line"; echo "more" >&2; exit \$(cat "$T/ctl/status"); fi
if [ -f "$T/ctl/mute" ] && [ "\$1 \$2" = "\$(cat "$T/ctl/mute")" ]; then echo "nothing to read"; exit 0; fi
case "\$1 \$2" in
  "front prompt") echo "text \$(cat "$T/ctl/tokens") language 2050 speaker 3061" ;;
  "lm inputs") echo "SP=64 rows 25 prefix \$(cat "$T/ctl/prefix") frames 0" ;;
  "step inputs") echo "T=192 NS=177 prefix 25 frames 153" ;;
  "step codes") echo "frames \$(cat "$T/ctl/frames")" ;;
  "decoder prepare") echo "SP=128 frames \$6 samples \$((1920 * \$6))" ;;
  "front pcm") echo "pcm \$6" ;;
  "front wav") echo "samples \$((1920 * \$(cat "$T/ctl/frames")))" ;;
  tokens*) ;;
  *) echo "weights: written" ;;
esac
EOS
cat >"$R/scripts/tts-session" <<EOS
#!/bin/sh
echo "tts-session \$*" >>"$T/log"
if [ -f "$T/ctl/fail" ] && [ "session" = "\$(cat "$T/ctl/fail")" ]; then echo "error[budget]: too much" >&2; exit 1; fi
[ "\$1" = until ] && echo "steps 92"
exit 0
EOS
chmod +x "$R/scripts/tts" "$R/scripts/tts-session"
fail=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1"; echo "  got:    $2"; echo "  wanted: $3"; fail=1; fi; }
cho() { (cd "$W" && "$REPO/build/media" --root "$R" --cwd "$W" --home "${HOME2:-$H}" --py /bin/true --cache "$K" ${FREEKB:+--free-kb $FREEKB} "$@"); }
norm() { sed "s#$T#TMP#g; s#: [0-9]*\.[0-9][0-9] s\$##; s# in [0-9]*\.[0-9][0-9] s\$##"; }
fresh() { rm -rf "$W/o" "$T/log" "$T/ctl/fail" "$T/ctl/mute"; echo 14 >"$T/ctl/tokens"; echo 25 >"$T/ctl/prefix"; echo 67 >"$T/ctl/frames"; echo 104 >"$T/ctl/status"; }
run() { fresh; [ -n "${CTL:-}" ] && echo "$CTL" >"$T/ctl/$CTLNAME"; [ -n "${CTL2:-}" ] && echo "$CTL2" >"$T/ctl/$CTLNAME2"; out=$(cho "$@" 2>"$T/err") && code=0 || code=$?; unset CTL CTLNAME CTL2 CTLNAME2 FREEKB; }
SN=TMP/home/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/abc
WK=TMP/cwd/o/work
front() { # the voice, the language, the seed, the steps, the positions: the stages up to the codes
cat <<EOL
tts tokens TMP/cache/qwen3-tts-tables/tokenizer.json $WK/text.txt $WK/tok.txt
tts front prompt $WK/tok.txt $1 $2 $WK/ids.txt
tts lm inputs talker TMP/cache/qwen3-tts-tables $WK/lm $WK/ids.txt - 2050 3061
tts step inputs $WK/step $WK/lm/x0.f32 $WK/lm/pad.f32 25 $4 $5 seed $3 -
tts-session until 2.151e3 $4 TMP/root/models/qwen3_tts_step.model CHX=TMP/root/tests/chx QW=TMP/root/examples/qwen TR=TMP/root/examples/train WORK=$WK/step WDIR=TMP/cache/qwen3-tts-step-f16 T=$5 NS=$4
tts step codes $WK/step 25 $WK/codes.txt
EOL
}
piece() { # the first frame, the frames, the samples skipped and taken, new or more: one piece of the waveform
cat <<EOL
tts decoder prepare TMP/cache/qwen3-tts-decoder-f32 $WK/dec $WK/codes.txt $1 $2
tts-session 1 TMP/root/models/qwen3_tts_decoder.model CHX=TMP/root/tests/chx WORK=$WK/dec WDIR=TMP/cache/qwen3-tts-decoder-f32 SP=128
tts front pcm $WK/speech.pcm $WK/dec/wave.f32 $3 $4 $5
EOL
}
last="tts front wav $WK/speech.pcm TMP/cwd/o/speech.wav"
run speak "The quick brown fox." -o o
check "a sentence: exit" "$code" 0
check "a sentence: the stages (14 tokens: 152 frames of budget, 177 steps, 192 positions; 67 frames in one piece)" "$(norm <"$T/log")" "$(front ryan auto 1 177 192; piece 0 67 0 128640 new; echo "$last")"
check "a sentence: the text as a file" "$(cat "$W/o/work/text.txt")" "The quick brown fox."
check "a sentence: output" "$(printf '%s\n' "$out" | norm | grep -v '^== ')" "done: TMP/cwd/o (speech.wav, 5.36 s of speech)"
check "a sentence: a line a stage, twice" "$(printf '%s\n' "$out" | grep -c '^== ')" 20
check "a sentence: no errors" "$(cat "$T/err")" ""
run speak -o o --voice Serena --language German --seed 7 --most 30 -- "Guten Morgen."
check "a voice, a language, a seed, --most 30, the text after --: the stages (55 steps, 64 positions)" "$(norm <"$T/log")" "$(front serena german 7 55 64; piece 0 67 0 128640 new; echo "$last")"
run speak "The quick brown fox." --output "$W/o"
check "--output, an absolute path: the stages" "$(norm <"$T/log")" "$(front ryan auto 1 177 192; piece 0 67 0 128640 new; echo "$last")"
# a long clip in the reference's pieces: 300 frames, each after the first with the 25 before it, whose 48,000 samples are skipped
CTLNAME=frames CTL=620 run speak "The quick brown fox." -o o
check "620 frames: three pieces" "$(norm <"$T/log" | tail -10)" "$(piece 0 300 0 576000 new; piece 275 325 48000 576000 more; piece 575 45 48000 38400 more; echo "$last")"
CTLNAME=frames CTL=300 run speak "The quick brown fox." -o o
check "300 frames: one piece" "$(norm <"$T/log" | tail -4)" "$(piece 0 300 0 576000 new; echo "$last")"
CTLNAME=frames CTL=301 run speak "The quick brown fox." -o o
check "301 frames: a second piece of one frame after 25 of context" "$(norm <"$T/log" | tail -7)" "$(piece 0 300 0 576000 new; piece 275 26 48000 1920 more; echo "$last")"
# the budget: 8 frames a token and 40, no more than the 1,024 steps of a request with the prefix
CTLNAME=tokens CTL=110 run speak "The quick brown fox." -o o
check "110 tokens: 920 frames of budget, 945 steps, 960 positions" "$(grep -c ' 945 960 seed 1 -$' "$T/log") $(grep -c 'until 2.151e3 945 .* T=960 NS=945$' "$T/log")" "1 1"
CTLNAME=tokens CTL=150 run speak "The quick brown fox." -o o
check "150 tokens: the budget stops at what a request holds, 1024 steps" "$code $(grep -c 'until 2.151e3 1024 .* T=1024 NS=1024$' "$T/log")" "0 1"
# the weights: each program runs when the last file it writes is not there
rm "$K/qwen3-tts-tables/tokenizer.json" "$K/qwen3-tts-step-f16/sumt.f16" "$K/qwen3-tts-decoder-f32/fc_b.f32"
run speak "The quick brown fox." -o o
check "no weights yet: the four programs that write them run first" "$(norm <"$T/log" | head -4)" "tts lm tables $SN/model.safetensors TMP/cache/qwen3-tts-tables
tts front tokenizer $SN TMP/cache/qwen3-tts-tables/tokenizer.json
tts step weights $SN/model.safetensors TMP/cache/qwen3-tts-step-f16
tts decoder weights $SN/speech_tokenizer/model.safetensors TMP/cache/qwen3-tts-decoder-f32"
check "no weights yet: then the ten stages" "$(norm <"$T/log" | tail -10)" "$(front ryan auto 1 177 192; piece 0 67 0 128640 new; echo "$last")"
FREEKB=1000 run speak "The quick brown fox." -o o
check "no weights and a full disk: exit" "$code" 2
check "no weights and a full disk: message" "$(norm <"$T/err")" "media: the weight files (4.3 GB) are written under TMP/cache on first use and only 0 GB are free"
check "no weights and a full disk: no stage ran" "$(cat "$T/log" 2>/dev/null)" ""
weights; rm "$K/qwen3-tts-decoder-f32/fc_b.f32"
run speak "The quick brown fox." -o o
check "the decoder's weights alone are missing: its program alone runs first" "$(norm <"$T/log" | head -2)" "tts decoder weights $SN/speech_tokenizer/model.safetensors TMP/cache/qwen3-tts-decoder-f32
tts tokens TMP/cache/qwen3-tts-tables/tokenizer.json $WK/text.txt $WK/tok.txt"
weights
FREEKB=1000 run speak "The quick brown fox." -o o
check "a full disk but the weights are there: runs" "$code" 0
# refusals before any stage
refuse() { name=$1; want=$2; shift 2; run "$@"; check "$name: exit" "$code" 2; check "$name: message" "$(norm <"$T/err")" "$want"; check "$name: no stage ran" "$(cat "$T/log" 2>/dev/null)" ""; }
refuse "no output" "media: the following arguments are required: text, -o/--output" speak "A text."
refuse "no text" "media: the following arguments are required: text, -o/--output" speak -o o
refuse "two texts" "media: unrecognized arguments" speak "One." "Two." -o o
refuse "a text of spaces" "media: the text is empty" speak "  " -o o
refuse "--most 0" "media: --most is the budget of frames (12.5 a second of speech), 1 to 1000" speak "A text." -o o --most 0
refuse "--most 1001" "media: --most is the budget of frames (12.5 a second of speech), 1 to 1000" speak "A text." -o o --most 1001
refuse "--most x" "media: argument --most: expected a value (a whole number where a number is asked)" speak "A text." -o o --most x
refuse "--seed 0" "media: --seed is 1 to 2147483646" speak "A text." -o o --seed 0
refuse "--voice with no name" "media: argument --voice: expected a value (a whole number where a number is asked)" speak "A text." -o o --voice
HOME2=$T/empty refuse "no checkpoint" "media: weights not in the Hugging Face cache: \`hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice\`" speak "A text." -o o
unset HOME2
rm "$R/build/qwen3_tts_step_gpu"
refuse "a host program not built" "media: the host programs of the speech are missing (build/qwen3_tts_front, _lm_gpu, _step_gpu, _decoder_gpu): build them with \`CANCHO=<pinned cancho> scripts/build-tts.sh\`" speak "A text." -o o
: >"$R/build/qwen3_tts_step_gpu"; rm "$R/build/cancho-gpu-metal-session"
refuse "no session" "media: build/cancho-gpu-metal-session is missing: build it with \`CANCHO=<pinned cancho> scripts/build-metal-session.sh\` (cancho.toml names the compiler)" speak "A text." -o o
: >"$R/build/cancho-gpu-metal-session"; rm "$R/build/cancho-gpu"
refuse "no tokenizer" "media: build/cancho-gpu is missing (the tokenizer is its \`tokenize\`): build it with \`CANCHO=<pinned cancho> scripts/build.sh\`" speak "A text." -o o
: >"$R/build/cancho-gpu"
# refusals of a stage: status 1, the stage's own line passed on, nothing after it
stop() { name=$1; ctl=$2; val=$3; want=$4; lines=$5; shift 5; CTLNAME=$ctl CTL=$val run "$@"; check "$name: exit" "$code" 1; check "$name: message" "$(norm <"$T/err")" "$want"; check "$name: the stages that ran" "$(wc -l <"$T/log" | tr -d ' ')" "$lines"; check "$name: no wav" "$([ -e "$W/o/speech.wav" ] && echo there || echo none)" "none"; }
stop "the tokenizer fails" fail "tokens" "more
media: tokens failed (exit 104): the stage's own line" 1 speak "A text." -o o
stop "the prompt refuses the voice" fail "front prompt" "more
media: prompt failed (exit 104): the stage's own line" 2 speak "A text." -o o --voice bob
stop "the prompt prints no numbers" mute "front prompt" "media: prompt: the stage did not print \`text \` and its number" 2 speak "A text." -o o
stop "the prefix prints no length" mute "lm inputs" "media: prefix: the stage did not print \`prefix \` and its number" 3 speak "A text." -o o
stop "the frames' session fails" fail "session" "error[budget]: too much
media: frames (GPU) failed (exit 1)" 5 speak "A text." -o o
stop "no end token in the budget" fail "step codes" "more
media: codes failed (exit 104): the stage's own line
media: the speech had a budget of 152 frames; if it did not end in them, pass --most with more" 6 speak "A text." -o o
CTLNAME2=tokens CTL2=150 stop "no end token in a whole request" fail "step codes" "more
media: codes failed (exit 104): the stage's own line
media: the speech had a budget of 999 frames, the most a request holds; if it did not end in them, the text is too long for one request" 6 speak "A text." -o o
stop "--most past what a request holds with its prefix" prefix 25 "media: the prefix (25 rows) and the budget of frames (1000) are more than the 1024 steps a request holds: a shorter text, or a smaller --most" 3 speak "A text." -o o --most 1000
stop "no frame before the end" frames 0 "media: the generation ended before its first frame: another --seed may not" 6 speak "A text." -o o
stop "the decoder's inputs fail" fail "decoder prepare" "more
media: decoder's inputs failed (exit 104): the stage's own line" 7 speak "A text." -o o
# the wrappers
out=$(CANCHO_MEDIA_IMPL=python CANCHO_MEDIA_CACHE=$K scripts/media speak "A text." -o "$W/o" 2>&1) && code=0 || code=$?
check "scripts/media with CANCHO_MEDIA_IMPL=python: refused, there is no Python of it" "$code $out" "2 media: speak is build/media's alone (there is no Python of it): build it with \`CANCHO=<pinned cancho> scripts/build-media.sh\`, and leave CANCHO_MEDIA_IMPL unset"
out=$(scripts/tts nothing 2>&1) && code=0 || code=$?
check "scripts/tts with another program's name" "$code $out" "100 tts: the first argument is front, lm, step, decoder or tokens"
out=$(scripts/tts tokens a b 2>&1) && code=0 || code=$?
check "scripts/tts tokens with too few arguments" "$code $out" "100 tts: tokens TOKENIZER TEXT OUT"
out=$(scripts/tts-session until 2.151e3 2>&1) && code=0 || code=$?
check "scripts/tts-session until with too few arguments" "$code $out" "100 tts-session: until ANSWER MOST MODEL [NAME=value]..."
[ $fail = 0 ] && echo "media_speak: ok"
exit $fail
