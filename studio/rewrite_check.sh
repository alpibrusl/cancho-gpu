#!/bin/sh
# Checks of the prompt-rewriting step (docs/studio-rewrite.md, slice R1): the worker, a stand-in for Ollama (`studio/rewrite_stub.cho`) and a fake `media` that prints its arguments. No GPU, no model.
#
#   CANCHO=<compiler> PK=<cancho checkout>/packages sh studio/rewrite_check.sh
#
# The programs the checks run are built from copies of the sources in which two literals are replaced by `sed`: the address's port (11434 -> 18160, so that the real Ollama on this machine is never reached)
# and the deadline (600 s -> 2 s, so that the slow stand-in is a test of seconds). Nothing else differs; the real worker is built from the real files (check 1).
set -e
cd "$(dirname "$0")/.."
PK="${PK:-../cancho/packages}"
C="${CANCHO:-cancho}"
T=$(mktemp -d)
trap 'kill $SP 2>/dev/null || true; rm -rf "$T"' EXIT
mkdir -p build "$T/src"
fail=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2', wanted '$3'"; fail=1; fi; }
build() { # out, sources...
  out=$1; shift
  $C build "$@" --std -o "$out" 2>&1 | grep -v "^ld: warning" || true
}

# 1. the real worker, built from the real files, says what it may reach
build build/studio_worker studio/worker.cho studio/files.cho studio/rewrite_step.cho studio/rewrite.cho
A=$($C authority studio/worker.cho studio/files.cho studio/rewrite_step.cho studio/rewrite.cho --std 2>&1)
check "authority: the worker's only network row is 127.0.0.1:11434" "$(echo "$A" | grep -c 'net_out("127.0.0.1:11434")')" 1
check "authority: ...and there is no other net row" "$(echo "$A" | grep 'net_' | grep -vc '127.0.0.1:11434')" 0
check "authority: no foreign code" "$(echo "$A" | grep -ci 'foreign code unbounded')" 0

# 2. test builds: the port and the deadline replaced, nothing else
for f in worker.cho rewrite_step.cho rewrite.cho; do sed -e 's/11434/18160/g' -e 's/model, pdir, 600)/model, pdir, 2)/' studio/$f > "$T/src/$f"; done
cp studio/files.cho "$T/src/"
build "$T/worker" "$T/src/worker.cho" "$T/src/files.cho" "$T/src/rewrite_step.cho" "$T/src/rewrite.cho"
build build/rewrite_stub studio/rewrite_stub.cho studio/files.cho "$PK"/http-server/server.cho
for f in worker.cho rewrite_step.cho rewrite.cho; do check "test build of $f differs from the real file only by the port and the deadline" "$(sed -e 's/18160/11434/g' -e 's/model, pdir, 2)/model, pdir, 600)/' "$T/src/$f" | cmp - studio/$f && echo same)" same; done

# 3. a fake media: one line per argument
printf '#!/bin/sh\nfor a in "$@"; do printf "[%%s]\\n" "$a"; done\n' > "$T/media"; chmod +x "$T/media"
P="$PWD/studio/rewrite"
# a synthetic picture: a PNG's first bytes and noise (it is only ever read as bytes)
(printf '\211PNG\r\n\032\n'; head -c 2000 /dev/urandom) > "$T/pic.png"

# stub NAME MODE: start the stand-in
stub() { build/rewrite_stub 18160 "$T/last.json" "$1" >/dev/null 2>&1 & SP=$!; sleep 0.7; }
unstub() { kill $SP 2>/dev/null; wait $SP 2>/dev/null || true; }
# job DIR ID ARGS...: the job's argument file (NUL after each entry)
job() { d=$1; id=$2; shift 2; : > "$d/$id.argv"; for a in "$@"; do printf '%s\0' "$a" >> "$d/$id.argv"; done; echo "$id" >> "$d/queue"; }
newdir() { D=$(mktemp -d "$T/d.XXXX"); }
lastarg() { grep '^\[' "$1/$2.log" | tail -1; }
note() { grep '^studio: prompt' "$1/$2.log" | head -1; }
W="$T/worker"
run() { "$W" "$T/media" /Users/test "$D" --rewriter test-model "$P" 2>/dev/null; }

# 4. rewritten
newdir; stub ok
job "$D" 1 video -o "$D/out/1" -- "a fox"; echo on > "$D/1.rewrite"
cp "$D/1.argv" "$T/1.argv.before"
run
check "ok: the job ran with the rewritten prompt" "$(lastarg "$D" 1)" "[A stub prompt.]"
check "ok: the state is done" "$(cat "$D/1.state")" done
check "ok: the log says who rewrote it" "$(note "$D" 1)" "studio: prompt rewritten by test-model"
check "ok: the original is kept" "$(cat "$D/1.prompt.orig")" "a fox"
check "ok: the new prompt is kept" "$(cat "$D/1.prompt.new")" "A stub prompt."
check "ok: <id>.argv is not changed" "$(cmp "$D/1.argv" "$T/1.argv.before" && echo same)" same
check "ok: the request names the model" "$(grep -o '"model":"[a-z-]*"' "$T/last.json")" '"model":"test-model"'
check "ok: the request asks for no streaming, no thinking, unloading" "$(grep -o '"stream":false,"think":false,"keep_alive":0' "$T/last.json")" '"stream":false,"think":false,"keep_alive":0'
check "ok: the request has no tools" "$(grep -c '"tools"' "$T/last.json")" 0
check "ok: a video sends no picture" "$(grep -c '"images"' "$T/last.json")" 0
check "ok: the system prompt sent is the video one" "$(grep -c 'text-to-video model' "$T/last.json")" 1
unstub

# 5. an edit sends the picture, and the picture's path is not sent
newdir; stub ok
job "$D" 1 edit -o "$D/out/1" -- "$T/pic.png" "make it look nicer"; echo on > "$D/1.rewrite"
run
check "edit: rewritten, the picture's path and the seed flag untouched" "$(grep '^\[' "$D/1.log" | tr '\n' ' ')" "[edit] [-o] [$D/out/1] [--] [$T/pic.png] [A stub prompt.] "
check "edit: the request carries one picture" "$(grep -o '"images":\["' "$T/last.json" | wc -l | tr -d ' ')" 1
check "edit: the system prompt sent is the edit one" "$(grep -c 'image-editing model' "$T/last.json")" 1
check "edit: the picture's path was not sent" "$(grep -c "$T/pic.png" "$T/last.json")" 0
check "edit: the prompt in the request is the person's" "$(grep -c 'make it look nicer' "$T/last.json")" 1
unstub

# 6. the prompt is data all the way: quotes, backslash, a command substitution, UTF-8
newdir; stub echo
job "$D" 1 video -o "$D/out/1" -- 'say "hi"; $(touch '"$T"'/PWNED) \back é ñ'; echo on > "$D/1.rewrite"
run
check "data: the rewritten prompt is one argument with every byte intact" "$(lastarg "$D" 1)" "[say \"hi\"; \$(touch $T/PWNED) \\back é ñ]"
check "data: nothing was executed" "$(ls "$T/PWNED" 2>/dev/null | wc -l | tr -d ' ')" 0
unstub

# 7. not asked, or no rewriter: the original, no call
newdir; stub ok; rm -f "$T/last.json"
job "$D" 1 video -o "$D/out/1" -- "a fox"
run
check "off: no <id>.rewrite: the original prompt" "$(lastarg "$D" 1)" "[a fox]"
check "off: ...no note in the log" "$(note "$D" 1 | wc -l | tr -d ' ')" 0
check "off: ...no request was made" "$(ls "$T/last.json" 2>/dev/null | wc -l | tr -d ' ')" 0
job "$D" 2 video -o "$D/out/2" -- "a boat"; echo on > "$D/2.rewrite"
"$T/worker" "$T/media" /Users/test "$D" 2>/dev/null
check "off: a worker with no --rewriter ignores <id>.rewrite" "$(lastarg "$D" 2)" "[a boat]"
check "off: ...and made no request" "$(ls "$T/last.json" 2>/dev/null | wc -l | tr -d ' ')" 0
unstub

# 8. every failure of the rewriter falls open: the job runs with the original prompt and the log says why
for m in err500:"the rewriter answered with an error" badjson:"the answer was not what was expected" nomessage:"the answer was not what was expected" empty:"the text was empty, too long or not clean" long:"the text was empty, too long or not clean" control:"the text was empty, too long or not clean" slow:"the rewriter did not answer in time"; do
  mode=${m%%:*}; why=${m#*:}
  newdir; stub "$mode"
  job "$D" 1 video -o "$D/out/1" -- "a fox"; echo on > "$D/1.rewrite"
  run
  check "$mode: the job ran with the original prompt and finished" "$(lastarg "$D" 1) $(cat "$D/1.state")" "[a fox] done"
  check "$mode: the log says why" "$(note "$D" 1)" "studio: prompt not rewritten: $why"
  check "$mode: no rewritten prompt was kept" "$(ls "$D"/1.prompt.new 2>/dev/null | wc -l | tr -d ' ')" 0
  unstub
done
newdir
job "$D" 1 video -o "$D/out/1" -- "a fox"; echo on > "$D/1.rewrite"
run
check "refused: nothing listening: original prompt, done" "$(lastarg "$D" 1) $(cat "$D/1.state")" "[a fox] done"
check "refused: ...the log says why" "$(note "$D" 1)" "studio: prompt not rewritten: could not connect to the rewriter"

# 9. a model's text is cleaned: quotes and a code fence are taken off
newdir; stub quotes
job "$D" 1 video -o "$D/out/1" -- "a fox"; echo on > "$D/1.rewrite"; run
check "quotes: one pair of quotes round the answer is removed" "$(lastarg "$D" 1)" "[a quoted prompt]"
unstub; newdir; stub fence
job "$D" 1 video -o "$D/out/1" -- "a fox"; echo on > "$D/1.rewrite"; run
check "fence: a code fence round the answer is removed" "$(lastarg "$D" 1)" "[fenced prompt]"
unstub

# 10. what the worker refuses to guess
newdir; stub ok
job "$D" 1 video -o "$D/out/1" "a fox"; echo on > "$D/1.rewrite"; run
check "shape: a job with no -- is not rewritten, and runs" "$(note "$D" 1) $(cat "$D/1.state")" "studio: prompt not rewritten: the job's arguments are not the shape the rewriter expects done"
job "$D" 2 edit -o "$D/out/2" -- "a fox"; echo on > "$D/2.rewrite"; run
check "shape: an edit with no picture entry is not rewritten" "$(note "$D" 2)" "studio: prompt not rewritten: the job's arguments are not the shape the rewriter expects"
job "$D" 3 edit -o "$D/out/3" -- "$T/missing.png" "make it nicer"; echo on > "$D/3.rewrite"; run
check "picture: one that cannot be read is not rewritten, and the job runs" "$(note "$D" 3) $(cat "$D/3.state")" "studio: prompt not rewritten: the picture could not be read or is over 16 MiB done"
mkdir "$T/noprompts"
job "$D" 4 video -o "$D/out/4" -- "a fox"; echo on > "$D/4.rewrite"; "$W" "$T/media" /Users/test "$D" --rewriter test-model "$T/noprompts" 2>/dev/null
check "system prompt: a missing file is not rewritten, and the job runs" "$(note "$D" 4) $(cat "$D/4.state")" "studio: prompt not rewritten: no system prompt for this kind of job done"
unstub

# 10b. two pictures (the command-line rewriter, `media --rewrite` with `--also`): a test build of rewrite_cli with the port replaced, against the stand-in
sed -e 's/11434/18160/g' studio/rewrite_cli.cho > "$T/src/rewrite_cli.cho"
build "$T/cli" "$T/src/rewrite_cli.cho" "$T/src/rewrite.cho" "$T/src/files.cho"
printf 'AAA' > "$T/a.bin"; printf 'BBB' > "$T/b.bin"; printf 'sys' > "$T/sys.txt"
two() { "$T/cli" ask m "$T/sys.txt" "p" "$1" 5 "$2"; }
stub echo
two "$T/a.bin" "$T/b.bin" > "$T/o2.txt" || true
check "two pictures: the request carries both, in order (image1, image2)" "$(grep -o '"images":\[[^]]*\]' "$T/last.json")" '"images":["QUFB","QkJC"]'
check "two pictures: the answer is the prompt" "$(sed -n 1p "$T/o2.txt")" "code 0 ok"
"$T/cli" ask m "$T/sys.txt" "p" "$T/a.bin" 5 > /dev/null || true
check "one picture: the request carries one" "$(grep -o '"images":\[[^]]*\]' "$T/last.json")" '"images":["QUFB"]'
out=$(two "$T/a.bin" "$T/nosuch" 2>&1) && st=0 || st=$?
check "two pictures: an unreadable second picture is refused (102)" "$st" 102
out=$(two - "$T/b.bin" 2>&1) && st=0 || st=$?
check "two pictures: a second picture without a first is refused (102)" "$st" 102
unstub
# the mutant: the second picture is dropped
mkdir -p "$T/m2"; cp "$T/src/rewrite.cho" "$T/src/rewrite_cli.cho" "$T/src/files.cho" "$T/m2/"
python3 - "$T/m2/rewrite.cho" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
a = "        if len(picture2) > 0 {\n            w = json.put_string(heap, w, picture2);"
assert a in s
open(p, "w").write(s.replace(a, "        if false {\n            w = json.put_string(heap, w, picture2);", 1))
PY
build "$T/m2/cli" "$T/m2/rewrite_cli.cho" "$T/m2/rewrite.cho" "$T/m2/files.cho"
stub echo
"$T/m2/cli" ask m "$T/sys.txt" "p" "$T/a.bin" 5 "$T/b.bin" > /dev/null || true
if [ "$(grep -o '"images":\[[^]]*\]' "$T/last.json")" != '"images":["QUFB","QkJC"]' ]; then echo "ok   mutant the second picture is dropped: killed"; else echo "FAIL mutant the second picture is dropped: survived"; fail=1; fi
unstub

# 11. mutants of the step, each of which one check above must refuse
mutant() { # name file sed-expression check-description
  mkdir -p "$T/m"; rm -f "$T/m"/*
  for f in worker.cho rewrite_step.cho rewrite.cho files.cho; do cp "$T/src/$f" "$T/m/$f"; done
  sed -e "$3" "$T/m/$2" > "$T/m/$2.new" && mv "$T/m/$2.new" "$T/m/$2"
  if cmp -s "$T/m/$2" "$T/src/$2"; then echo "FAIL mutant $1: the edit changed nothing"; fail=1; return; fi
  out=$($C build "$T/m/worker.cho" "$T/m/files.cho" "$T/m/rewrite_step.cho" "$T/m/rewrite.cho" --std -o "$T/m/worker" 2>&1 | grep -v "^ld: warning" || true)
  if [ ! -x "$T/m/worker" ]; then echo "ok   mutant $1: killed (it does not build)"; return; fi
  newdir; killed=0
  for mode in control long err500 ok; do
    stub $mode
    job "$D" 1 video -o "$D/out/1" -- "a fox"; echo on > "$D/1.rewrite"
    "$T/m/worker" "$T/media" /Users/test "$D" --rewriter test-model "$P" 2>/dev/null || true
    got="$(lastarg "$D" 1)"; orig=0; [ -f "$D/1.prompt.orig" ] && orig=1
    case "$mode" in
      control|long|err500) [ "$got" != "[a fox]" ] && killed=1 ;;
      ok) [ "$got" != "[A stub prompt.]" ] && killed=1; [ $orig = 0 ] && killed=1 ;;
    esac
    unstub
    newdir
  done
  if [ $killed = 1 ]; then echo "ok   mutant $1: killed"; else echo "FAIL mutant $1: survived"; fail=1; fi
}
mutant "an unclean answer is accepted" rewrite.cho 's/if c < 32 || c == 127 {/if false {/'
mutant "the length bound is skipped" rewrite.cho 's/if bad || count < 1 || count > max_prompt() {/if bad || count < 1 {/'
mutant "a failure still uses the rewritten list" worker.cho 's/if rcode == 0 {/if rcode >= 0 {/'
mutant "the original is not kept" rewrite_step.cho 's|files.write_all(fs, buffer.bytes(opb), list\[pa..pb\]);|let unused = 0;|'
exit $fail
