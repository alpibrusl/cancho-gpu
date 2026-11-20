#!/bin/sh
# Builds the studio's programs and checks the worker against a fake script (no GPU): the arguments reach the script as data, its output lands in `log`, `state` says
# done or failed N, a missing argv file does nothing. CANCHO names the compiler; PK the packages directory of a cancho checkout.
set -e
cd "$(dirname "$0")/.."
PK="${PK:-../cancho/packages}"
C="${CANCHO:-cancho}"
mkdir -p build
$C build studio/studio.cho studio/files.cho studio/jobs.cho studio/results.cho studio/auth.cho studio/range.cho studio/upload.cho studio/compare.cho studio/sets.cho studio/cases.cho studio/gallery.cho studio/sessions.cho studio/library.cho studio/picks.cho studio/batch.cho "$PK"/http-server/server.cho --std -o build/studio 2>&1 | grep -v "^ld: warning" || true
$C build studio/worker.cho studio/files.cho studio/rewrite_step.cho studio/rewrite.cho --std -o build/studio_worker 2>&1 | grep -v "^ld: warning" || true
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
printf '#!/bin/sh\necho "args: $1 | $2 | home=$HOME"\n[ "$1" = fail ] && exit 7\nexit 0\n' > "$T/fake.sh"
chmod +x "$T/fake.sh"
printf 'video\0a prompt; with $(shell) chars\0' > "$T/1.argv"
printf 'fail\0x\0' > "$T/2.argv"
printf 'video\0again\0' > "$T/4.argv"
printf 'old\n' > "$T/4.state"
printf '1\n2\n3\n../etc\n4\n1234567890\n' > "$T/queue"
fail=0
# a server is up when its log says so (a fixed sleep failed when the machine was busy: a compile elsewhere took the CPU for seconds)
waitfor() { n=0; until [ "$(cat "$1" 2>/dev/null)" = "$2" ] || [ $n -ge 100 ]; do sleep 0.2; n=$((n + 1)); done; }
waitup() { for f in "$@"; do n=0; until grep -q listening "$f" 2>/dev/null || [ $n -ge 150 ]; do sleep 0.2; n=$((n + 1)); done; done; }
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2', wanted '$3'"; fail=1; fi; }
build/studio_worker "$T/fake.sh" /Users/test "$T"; check "exit 0" "$?" 0
check "job 1 done" "$(cat "$T/1.state")" done
check "metacharacters stay data" "$(cat "$T/1.log")" 'args: video | a prompt; with $(shell) chars | home=/Users/test'
check "job 2 failed 7" "$(cat "$T/2.state")" "failed 7"
check "job 3 (no argv file): no log" "$(ls "$T/3.log" 2>/dev/null | wc -l | tr -d ' ')" 0
check "job 4 had a state: not run again" "$(cat "$T/4.state")" old
check "an id that is not digits is not a path" "$(ls "$T" | grep -c etc)" 0
build/studio_worker "$T/fake.sh" /Users/test "$T"; check "second drain: nothing new, still 0" "$?" 0
# the errors of a job are in its log (they went to /dev/null: a refused job had an empty log and no reason)
E=$(mktemp -d)
printf '#!/bin/sh\necho "out: $1"\necho "media: unrecognized arguments" >&2\n[ "$1" = shout ] && exit 2\nexit 0\n' > "$E/fake.sh"; chmod +x "$E/fake.sh"
printf 'shout\0x\0' > "$E/1.argv"; printf 'quiet\0x\0' > "$E/2.argv"; printf '1\n2\n' > "$E/queue"
build/studio_worker "$E/fake.sh" /Users/test "$E"
check "errors: a refused job is failed 2" "$(cat "$E/1.state")" "failed 2"
check "errors: its log has what it printed, then its errors under stderr:" "$(cat "$E/1.log" | tr '\n' '|')" "out: shout||stderr:|media: unrecognized arguments|"
check "errors: a job that ended well keeps its errors in the log too" "$(cat "$E/2.log" | tr '\n' '|')" "out: quiet||stderr:|media: unrecognized arguments|"
printf '#!/bin/sh\necho fine\n' > "$E/silent.sh"; chmod +x "$E/silent.sh"; printf 'a\0' > "$E/3.argv"; printf '3\n' > "$E/queue2"; mv "$E/queue2" "$E/queue"
build/studio_worker "$E/silent.sh" /Users/test "$E"
check "errors: a job with no errors has a log as before" "$(cat "$E/3.log")" "fine"
rm -rf "$E"
check "job 1 not run twice (log unchanged)" "$(wc -l < "$T/1.log" | tr -d ' ')" 1
rc=0; build/studio_worker 2>/dev/null || rc=$?; check "usage: exit 2" "$rc" 2
rc=0; build/studio_worker "$T/fake.sh" /Users/test relative/dir 2>/dev/null || rc=$?; check "relative data dir refused" "$rc" 3

# The script is started beneath its directory (`exec_spawn_under`), so the report says `exec_under`, not `exec("")` (any program anywhere). Measured before: `exec("")`.
if [ -n "${CANCHO_AUTH:-1}" ]; then
  A=$($C authority studio/worker.cho studio/files.cho studio/rewrite_step.cho studio/rewrite.cho --std 2>&1)
  check "exec: the report names exec_under" "$(echo "$A" | grep -c '^    exec_under$')" 1
  check "exec: ...and no longer exec(\"\")" "$(echo "$A" | grep -c 'exec("")')" 0
fi
rc=0; build/studio_worker relative.sh /Users/test "$T" 2>/dev/null || rc=$?; check "exec: a script with no directory refused: exit 3" "$rc" 3
rc=0; build/studio_worker /no/such/dir/fake.sh /Users/test "$T" 2>/dev/null || rc=$?; check "exec: a script in a directory that is not there: exit 3" "$rc" 3
rc=0; build/studio_worker "$T/.." /Users/test "$T" 2>/dev/null || rc=$?; check "exec: a name that is .. refused: exit 3" "$rc" 3
rc=0; build/studio_worker "$T/" /Users/test "$T" 2>/dev/null || rc=$?; check "exec: an empty name refused: exit 3" "$rc" 3
V=$(mktemp -d); mkdir "$V/bin" "$V/data"
printf '#!/bin/sh\npwd -P\necho "arg0=$0"\n' > "$V/bin/where.sh"; chmod +x "$V/bin/where.sh"
printf 'x\0' > "$V/data/1.argv"; printf '1\n' > "$V/data/queue"
rc=0; build/studio_worker "$V/bin/where.sh" /Users/test "$V/data" || rc=$?
check "exec: a script in another directory runs" "$(cat "$V/data/1.state")" done
check "exec: ...started in its own directory" "$(head -1 "$V/data/1.log")" "$(cd "$V/bin" && pwd -P)"
check "exec: ...by its name" "$(tail -1 "$V/data/1.log")" "arg0=where.sh"
rm -rf "$V"

# Tidying `out/<id>` after a job that is done: work files go, results, prompt and frames/ stay; a failed job keeps everything; a link is left alone.
M=$(mktemp -d)
printf '#!/bin/sh\n[ "$1" = fail ] && exit 7\nexit 0\n' > "$M/fake.sh"; chmod +x "$M/fake.sh"
for id in 7 8; do
  mkdir -p "$M/out/$id/frames"
  for f in video.mp4 video.gif edited.png image.png prompt.txt ci_s.f32 ctx.f16 text.npy video.npy; do printf 'x' > "$M/out/$id/$f"; done
  printf 'x' > "$M/out/$id/frames/f000.png"
  mkdir "$M/out/$id/vae"; printf 'x' > "$M/out/$id/vae/cv1.f16"; printf 'x' > "$M/out/$id/vae/bi2.f32"
done
printf 'secret' > "$M/secret.txt"; ln -s "$M/secret.txt" "$M/out/7/link"
printf 'video\0ok\0' > "$M/7.argv"; printf 'fail\0x\0' > "$M/8.argv"; printf '7\n8\n' > "$M/queue"
build/studio_worker "$M/fake.sh" /Users/test "$M"
check "tidy: work files removed" "$(ls "$M/out/7" | tr '\n' ' ')" 'edited.png frames image.png link prompt.txt vae video.gif video.mp4 '
check "tidy: frames/ untouched" "$(ls "$M/out/7/frames")" f000.png
check "tidy: a subdirectory of work files is emptied" "$(ls "$M/out/7/vae" | wc -l | tr -d ' ')" 0
check "tidy: a link is not followed or removed" "$(cat "$M/secret.txt")" secret
check "tidy: the log says how many" "$(tail -1 "$M/7.log")" "studio: removed 6 work files from out/7"
check "tidy: a failed job keeps everything" "$(ls "$M/out/8" | wc -l | tr -d ' ')" 11
rm -rf "$M"

# The worker in a loop (`--every`): a job that arrives later is run; TERM while it waits stops it at once (exit 0); TERM during a job lets that job end first.
L=$(mktemp -d)
printf '#!/bin/sh\nif [ "$1" = slow ]; then sleep 2; fi\necho "ran $1"\nexit 0\n' > "$L/fake.sh"
chmod +x "$L/fake.sh"
printf 'video\0one\0' > "$L/1.argv"; printf '1\n' > "$L/queue"
build/studio_worker "$L/fake.sh" /Users/test "$L" --every 30 &
WP=$!
waitfor "$L/1.state" done
check "loop: the job in the queue ran" "$(cat "$L/1.state")" done
printf 'video\0two\0' > "$L/2.argv"; printf '1\n2\n' > "$L/queue"
sleep 0.5
check "loop: waiting (30 s), a later job is not run yet" "$(ls "$L/2.state" 2>/dev/null | wc -l | tr -d ' ')" 0
kill -TERM $WP
rc=0; wait $WP || rc=$?
check "loop: TERM while waiting: exit 0" "$rc" 0
build/studio_worker "$L/fake.sh" /Users/test "$L" --every 1 &
WP=$!
waitfor "$L/2.state" done
check "loop: a job that arrived while it was stopped runs on the next drain" "$(cat "$L/2.state")" done
printf 'slow\0three\0' > "$L/3.argv"; printf '1\n2\n3\n' > "$L/queue"
waitfor "$L/3.state" running
check "loop: slow job started" "$(cat "$L/3.state")" running
kill -TERM $WP
rc=0; wait $WP || rc=$?
check "loop: TERM during a job: exit 0" "$rc" 0
check "loop: ...and the job ended first" "$(cat "$L/3.state")" done
check "loop: its output was kept" "$(cat "$L/3.log")" "ran slow"
rc=0; build/studio_worker "$L/fake.sh" /Users/test "$L" --every 0 2>/dev/null || rc=$?; check "loop: --every 0 is usage (exit 2)" "$rc" 2
rc=0; build/studio_worker "$L/fake.sh" /Users/test "$L" --every abc 2>/dev/null || rc=$?; check "loop: --every abc is usage (exit 2)" "$rc" 2
rm -rf "$L"

# The page (studio/ui/studio.cui): checked and replayed with cancho-ui when UI names its binary; built into a site directory for the server below.
mkdir -p "$T/site"
if [ -n "${UI:-}" ]; then
  check "ui: cancho-ui check" "$($UI check studio/ui/studio.cui >/dev/null 2>&1; echo $?)" 0
  check "ui: replay trace passes" "$($UI test studio/ui/studio.cui studio/ui/studio.trace.json 2>&1 | tail -1 | cut -c1-4)" PASS
  $UI build studio/ui/studio.cui --page -o "$T/site" --name cancho-studio >/dev/null 2>&1
else
  echo "skip ui checks (set UI to the cancho-ui binary)"
  printf '<!doctype html><title>fake page</title>' > "$T/site/index.html"; printf 'console.log(1)' > "$T/site/app.js"
fi

# One worker at a time (`worker.lock`): a second is refused with exit 4 and runs nothing, while the first is in a job and while it waits; SIGTERM frees the lock; SIGKILL does too, and the job it left `running`
# becomes `failed 255` and is not run again.
K=$(mktemp -d)
printf '#!/bin/sh\nif [ "$1" = slow ]; then sleep 2; fi\necho "ran $1"\nexit 0\n' > "$K/fake.sh"; chmod +x "$K/fake.sh"
printf 'slow\0a\0' > "$K/1.argv"; printf 'video\0b\0' > "$K/2.argv"; printf '1\n2\n' > "$K/queue"
build/studio_worker "$K/fake.sh" /Users/test "$K" --every 30 &
WA=$!
waitfor "$K/1.state" running
check "lock: the first worker is in job 1" "$(cat "$K/1.state")" running
rc=0; build/studio_worker "$K/fake.sh" /Users/test "$K" 2>"$K/err" || rc=$?
check "lock: a second worker, once, during a job: exit 4" "$rc" 4
check "lock: ...and says why" "$(grep -c 'another worker holds worker.lock' "$K/err")" 1
rc=0; build/studio_worker "$K/fake.sh" /Users/test "$K" --every 5 2>/dev/null || rc=$?
check "lock: a second worker, looping: exit 4" "$rc" 4
waitfor "$K/2.state" done
check "lock: the first worker finished jobs 1 and 2 alone" "$(cat "$K/1.state") $(cat "$K/2.state")" "done done"
check "lock: job 1 ran once (one line in its log)" "$(grep -c 'ran slow' "$K/1.log")" 1
rc=0; build/studio_worker "$K/fake.sh" /Users/test "$K" 2>/dev/null || rc=$?
check "lock: a second worker while the first waits: exit 4" "$rc" 4
kill -TERM $WA; rc=0; wait $WA || rc=$?
check "lock: SIGTERM ends the first worker, exit 0" "$rc" 0
rc=0; build/studio_worker "$K/fake.sh" /Users/test "$K" 2>/dev/null || rc=$?
check "lock: once it is gone the lock is free: exit 0" "$rc" 0
printf 'slow\0c\0' > "$K/3.argv"; printf 'video\0d\0' > "$K/4.argv"; printf '1\n2\n3\n4\n' > "$K/queue"
build/studio_worker "$K/fake.sh" /Users/test "$K" --every 30 &
WB=$!
waitfor "$K/3.state" running
check "lock: a worker in job 3" "$(cat "$K/3.state")" running
kill -KILL $WB; wait $WB 2>/dev/null || true
sleep 3
rc=0; build/studio_worker "$K/fake.sh" /Users/test "$K" 2>/dev/null || rc=$?
check "lock: SIGKILL freed the lock: exit 0" "$rc" 0
check "lock: the job it left running is failed 255" "$(cat "$K/3.state")" "failed 255"
check "lock: ...and was not run again" "$(grep -c 'ran slow' "$K/3.log")" 1
check "lock: the job it had not started ran" "$(cat "$K/4.state")" done
rm -rf "$K"

# The server, the queue and the worker together, with a fake `media` that prints its arguments (port 18091).
printf '#!/bin/sh\nfor a in "$@"; do echo "[$a]"; done\n[ "$1" = edit ] && exit 3\nexit 0\n' > "$T/media"
chmod +x "$T/media"
mkdir "$T/data"
build/studio 18091 "$T/data" >"$T/s1.log" 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null; rm -rf "$T"' EXIT
waitup "$T/s1.log"
TOK1=$(grep -o 'token=[0-9a-f]*' "$T/s1.log" | cut -d= -f2); TOK=$TOK1
# every request carries the token (the way a script would: a bearer header); a POST says it is JSON
c() { curl -s -H "Authorization: Bearer $TOK" "$@"; }
post() { c -w ' %{http_code}' -XPOST -H 'Content-Type: application/json' localhost:18091/api/jobs -d "$1"; }
check "POST a video job" "$(post '{"kind":"video","prompt":"-a fox; $(rm -rf /) \"q\"","frames":9,"size":"640x352","seed":3}')" '{"id":1} 202'
check "POST an edit job" "$(post '{"kind":"edit","prompt":"sunny","image":"/tmp/p.jpg"}')" '{"id":2} 202'
check "empty prompt refused" "$(post '{"kind":"video","prompt":""}')" '{"error":"prompt is a string of 1 to 1000 bytes"} 422'
check "an edit without image refused" "$(post '{"kind":"edit","prompt":"x"}' | sed 's/:.*//')" '{"error"'
check "frames over 129 refused" "$(post '{"kind":"video","prompt":"x","frames":500}' | tail -c 4)" ' 422'
check "bad size refused" "$(post '{"kind":"video","prompt":"x","size":"64x3520000"}' | tail -c 4)" ' 422'
check "unknown kind refused" "$(post '{"kind":"nope","prompt":"x"}')" '{"error":"kind is video, edit or image"} 422'
check "not JSON refused" "$(post 'not json' | tail -c 4)" ' 422'
check "queued" "$(c localhost:18091/api/jobs/1)" '{"id":1,"state":"queued"}'
check "unknown job" "$(c -w ' %{http_code}' localhost:18091/api/jobs/99)" '{"error":"no such job"} 404'
check "id that is not digits" "$(c -o /dev/null -w '%{http_code}' localhost:18091/api/jobs/1a)" 422
check "queue file" "$(tr '\n' ' ' < "$T/data/queue")" '1 2 '
check "argv: the prompt is one entry after --" "$(tr '\0' '\n' < "$T/data/1.argv" | tail -1)" '-a fox; $(rm -rf /) "q"'
build/studio_worker "$T/media" /Users/test "$T/data"
check "job 1 done" "$(c localhost:18091/api/jobs/1)" '{"id":1,"state":"done"}'
check "job 2 failed 3" "$(c localhost:18091/api/jobs/2)" '{"id":2,"state":"failed 3"}'
check "the list: newest first, state, kind, prompt (a quote in the prompt is escaped)" "$(c localhost:18091/api/jobs)" '{"running":0,"queued":0,"jobs":[{"id":2,"state":"failed 3","kind":"edit","prompt":"sunny"},{"id":1,"state":"done","kind":"video","prompt":"-a fox; $(rm -rf /) \"q\""}]}'
check "the list is JSON" "$(c -o /dev/null -w '%{content_type}' localhost:18091/api/jobs)" 'application/json'
check "job 1 got its arguments" "$(grep -c . "$T/data/1.log")" 11
# The results page and the files (job 1 done with a fake result, job 2 failed, job 3 queued).
mkdir -p "$T/data/out/1"
printf '\211PNG\r\n\032\n\0\0binary\377\0' > "$T/data/out/1/video.mp4"
cp "$T/data/out/1/video.mp4" "$T/data/secret.bin"
printf '3 queued\n' >/dev/null
printf 'video\0q\0' > "$T/data/3.argv"
c localhost:18091/jobs/1 > "$T/page1"
check "page of a done job: state" "$(grep -c 'state: done' "$T/page1")" 1
check "page of a done job: the video" "$(grep -c 'src=/files/1/video.mp4' "$T/page1")" 1
check "page of a done job: no picture it does not have" "$(grep -c 'edited.png\|video.gif' "$T/page1")" 0
check "page of a failed job: no result" "$(c localhost:18091/jobs/2 | grep -c 'state: failed 3.*\(video\|edited\)')" 0
check "page of a queued job refreshes" "$(c localhost:18091/jobs/3 | grep -c 'refresh')" 1
check "page of an unknown job" "$(c -o /dev/null -w '%{http_code}' localhost:18091/jobs/99)" 404
check "page of a bad id" "$(c -o /dev/null -w '%{http_code}' localhost:18091/jobs/1a)" 422
c -D "$T/hdr" localhost:18091/files/1/video.mp4 > "$T/got.bin"
check "file: the bytes arrive unchanged" "$(cmp "$T/got.bin" "$T/data/out/1/video.mp4" && echo same)" same
check "file: media type" "$(grep -ci '^content-type: video/mp4' "$T/hdr")" 1
check "file: the log is the worker's output" "$(c localhost:18091/files/1/log | head -1)" '[video]'
head -c 6000000 /dev/urandom > "$T/data/out/1/video.gif"
check "file: 6 MB arrive unchanged (streamed)" "$(c localhost:18091/files/1/video.gif | cmp - "$T/data/out/1/video.gif" && echo same)" same
check "file: Content-Length is the size" "$(c -D - -o /dev/null localhost:18091/files/1/video.gif | tr -d '\r' | grep -i '^content-length' | awk '{print $2}')" 6000000
# Range (what Safari asks of a video): pieces of video.gif (6000000 random bytes) compared with `tail -c | head -c`.
G="$T/data/out/1/video.gif"; N=6000000
rg() { c -H "Range: $1" localhost:18091/files/1/video.gif; }
piece() { tail -c +$(( $1 + 1 )) "$G" | head -c "$2"; }
same() { rg "$1" > "$T/got"; piece "$2" "$3" > "$T/want"; if cmp -s "$T/got" "$T/want"; then echo same; else echo different; fi; }
hdr() { c -D - -o /dev/null -H "Range: $1" localhost:18091/files/1/video.gif | tr -d '\r'; }
check "range: a plain GET says Accept-Ranges" "$(c -D - -o /dev/null localhost:18091/files/1/video.gif | tr -d '\r' | grep -ci '^accept-ranges: bytes')" 1
check "range: bytes=0-1 is 206" "$(hdr bytes=0-1 | head -1 | cut -d' ' -f2)" 206
check "range: ...with Content-Range" "$(hdr bytes=0-1 | grep -i '^content-range' | cut -d' ' -f2-)" "bytes 0-1/6000000"
check "range: ...and Content-Length 2" "$(hdr bytes=0-1 | grep -i '^content-length' | cut -d' ' -f2)" 2
check "range: bytes=0-1 is the first two bytes" "$(same bytes=0-1 0 2)" same
check "range: bytes=1000-1999" "$(same bytes=1000-1999 1000 1000)" same
check "range: across the 64 KiB piece boundary" "$(same bytes=65530-65545 65530 16)" same
check "range: bytes=5990000- runs to the end" "$(same bytes=5990000- 5990000 10000)" same
check "range: bytes=-5 is the last five" "$(same bytes=-5 5999995 5)" same
check "range: a suffix longer than the file is the file" "$(rg bytes=-99999999 | wc -c | tr -d ' ')" $N
check "range: a last byte past the end is cut" "$(same bytes=5999990-99999999 5999990 10)" same
check "range: ...and says where it ends" "$(hdr bytes=5999990-99999999 | grep -i '^content-range' | cut -d' ' -f2-)" "bytes 5999990-5999999/6000000"
check "range: a megabyte from the middle" "$(same bytes=2000000-2999999 2000000 1000000)" same
check "range: starting at the end: 416" "$(hdr bytes=6000000- | head -1 | cut -d' ' -f2)" 416
check "range: ...with Content-Range */N" "$(hdr bytes=6000000- | grep -i '^content-range' | cut -d' ' -f2-)" "bytes */6000000"
check "range: the last 0 bytes: 416" "$(hdr bytes=-0 | head -1 | cut -d' ' -f2)" 416
check "range: two ranges: the whole file, 200" "$(hdr bytes=0-0,5-9 | head -1 | cut -d' ' -f2)" 200
check "range: not a range: 200" "$(hdr bytes=abc | head -1 | cut -d' ' -f2)" 200
check "range: last before first: 200" "$(hdr bytes=9-3 | head -1 | cut -d' ' -f2)" 200
check "range: another unit: 200" "$(hdr items=0-1 | head -1 | cut -d' ' -f2)" 200
check "range: a number too long to be a size: 200" "$(hdr bytes=0-99999999999999 | head -1 | cut -d' ' -f2)" 200
# Several files at once: six downloads of the 6 MB file in parallel, each compared.
PARS=""
for k in 1 2 3 4 5 6; do (c localhost:18091/files/1/video.gif | cmp - "$G" > "$T/par$k" 2>&1; echo $? >> "$T/par$k") & PARS="$PARS $!"; done
for q in $PARS; do wait $q; done
check "streams: six at once, each whole" "$(cat "$T"/par? | tr -d '\n')" 000000
check "streams: the server serves after" "$(c localhost:18091/api/jobs/1)" '{"id":1,"state":"done"}'
head -c 40000000 /dev/urandom > "$T/data/out/1/video.gif"
check "file: 40 MB arrive unchanged" "$(c localhost:18091/files/1/video.gif | cmp - "$T/data/out/1/video.gif" && echo same)" same
check "file: two in a row on one connection" "$(c localhost:18091/files/1/video.gif localhost:18091/files/1/video.gif | wc -c | tr -d ' ')" 80000000
check "the server serves after streaming" "$(c localhost:18091/api/jobs/1)" '{"id":1,"state":"done"}'
head -c 6000000 /dev/urandom > "$T/data/out/1/video.gif"
head -c 150000 /dev/urandom > "$T/data/out/1/edited.png"
check "file: 150 KB arrive unchanged" "$(c localhost:18091/files/1/edited.png | cmp - "$T/data/out/1/edited.png" && echo same)" same
rm "$T/data/out/1/edited.png"
check "file: a name not on the list" "$(c -o /dev/null -w '%{http_code}' localhost:18091/files/1/secret.bin)" 404
check "file: a path is not a name" "$(c --path-as-is -o /dev/null -w '%{http_code}' 'localhost:18091/files/1/../secret.bin')" 404
check "file: a file the job does not have" "$(c -o /dev/null -w '%{http_code}' localhost:18091/files/1/edited.png)" 404
check "file: bad id" "$(c -o /dev/null -w '%{http_code}' localhost:18091/files/1a/log)" 422
# The page and the list, from a server with a site directory (port 18094).
build/studio 18094 "$T/data" "$T/site" >"$T/s2.log" 2>&1 &
SP2=$!
trap 'kill $SP $SP2 2>/dev/null; rm -rf "$T"' EXIT
waitup "$T/s2.log"
TOK2=$(grep -o 'token=[0-9a-f]*' "$T/s2.log" | cut -d= -f2); TOK=$TOK2
check "site: / is index.html, with the stylesheet, the header and the frame of recent jobs added and nothing else changed" "$(c localhost:18094/ | sed 's#<link rel=stylesheet href=/style.css>##; s#<header class=top>.*</header>##; s#<div class=tabs>\(<a [^>]*>[^<]*</a>\)*</div>##; s#<iframe class=recent src=/recent title="Recent jobs"></iframe>##' | cmp - "$T/site/index.html" && echo same)" same
check "site: /app.js is app.js, a script" "$(c -D - -o /dev/null localhost:18094/app.js | tr -d '\r' | grep -ci '^content-type: text/javascript')" 1
check "site: app.js bytes unchanged" "$(c localhost:18094/app.js | cmp - "$T/site/app.js" && echo same)" same
TOK=$TOK1
check "site: no site, / is the stub" "$(c localhost:18091/ | grep -c 'Slice 1')" 1
TOK=$TOK1 post '{"kind":"video","prompt":"third"}' >/dev/null
check "jobs: newest first" "$(TOK=$TOK2 c localhost:18094/jobs | grep -o 'class=idn>#[0-9]*' | tr '\n' ' ')" 'class=idn>#3 class=idn>#2 class=idn>#1 '
check "jobs: states" "$(TOK=$TOK2 c localhost:18094/jobs | grep -o 'class=\"badge [a-z]*\"' | tr '\n' ' ')" 'class="badge queued" class="badge failed" class="badge done" '
check "jobs: a finished video's card draws the first frame of its clip" "$(TOK=$TOK2 c localhost:18094/jobs | grep -c '<video muted playsinline preload=metadata src=/files/1/video.mp4#t=0.1></video>')" 1
# The OpenAPI document (studio/openapi.json): served as it is, and held to what the server answers. Each row is a request and the status measured; the document must list that status for that path.
cp studio/openapi.json "$T/site/openapi.json"; TOK=$TOK2
check "openapi: served as the file, as JSON" "$(c localhost:18094/openapi.json | cmp - studio/openapi.json && echo same)" same
check "openapi: without a site directory: 404" "$(TOK=$TOK1 c -o /dev/null -w '%{http_code}' localhost:18091/openapi.json)" 404
check "openapi: needs the token like any route" "$(curl -s -o /dev/null -w '%{http_code}' localhost:18094/openapi.json)" 401
doc_lists() { awk -v p="    \"$1\": {" -v s="\"$2\":" 'index($0,p)==1{on=1;next} on&&/^    "\//{on=0} on&&index($0,s){found=1} END{exit !found}' studio/openapi.json && echo yes || echo no; }
TOK=$TOK2
[ -f "$T/data/1.log" ] || echo streamed-bytes > "$T/data/1.log"
row() { # path-in-document, method, url, curl args...
  d=$1; m=$2; url=$3; shift 3
  got=$(c -o /dev/null -w '%{http_code}' -X "$m" "$@" "localhost:18094$url")
  check "openapi: $m $url ($got) is listed under $d" "$(doc_lists "$d" "$got")" yes
}
row /api/models GET /api/models
row /api/jobs GET /api/jobs
check "openapi: 202 on POST /api/jobs is listed (queued by the checks above)" "$(doc_lists /api/jobs 202)" yes
row /api/jobs POST /api/jobs -H 'Content-Type: application/json' -d '{"kind":"nope"}'
row /api/jobs POST /api/jobs -H 'Content-Type: text/plain' -d '{}'
row '/api/jobs/{id}' GET /api/jobs/1
row '/api/jobs/{id}' GET /api/jobs/99999
row '/api/jobs/{id}' GET /api/jobs/x
row /jobs GET /jobs
row /recent GET /recent
row '/jobs/{id}' GET /jobs/1
row '/jobs/{id}' GET /jobs/99999
row '/jobs/{id}' GET /jobs/x
row '/files/{id}/{name}' GET /files/1/log
row '/files/{id}/{name}' GET /files/1/log -H 'Range: bytes=0-3'
row '/files/{id}/{name}' GET /files/1/log -H 'Range: bytes=99999-'
row '/files/{id}/{name}' GET /files/1/nothing
row '/files/{id}/{name}' GET /files/x/log
row '/sets/{id}' GET /sets/1
row '/sets/{id}' GET /sets/x
row /app.js GET /app.js
row /style.css GET /style.css
row / GET /
row /openapi.json GET /openapi.json
row / GET / -H 'Host: rebinding.example:18094'
row / GET '/?token=00000000000000000000000000000000'
check "openapi: 302 on / is listed" "$(doc_lists / 302)" yes
check "openapi: 500 on POST /api/jobs is listed (not reached by a request here)" "$(doc_lists /api/jobs 500)" yes
# The other way: every path the server matches is in the document.
for p in $(grep -o 'equal(path, "/[a-z./]*")' studio/studio.cho | grep -o '"/[a-z./]*"' | tr -d '"' | grep -v '^/$' | sort -u); do
  check "openapi: the server's route $p is in the document" "$(grep -c "^    \"$p\": {" studio/openapi.json)" 1
done
for p in /api/jobs/ /jobs/ /files/ /compare/ /sets/; do
  check "openapi: the server's prefix route $p is in the document" "$(grep -c "^    \"$p" studio/openapi.json)" 1
done
if command -v jq >/dev/null 2>&1; then
  check "openapi: the file is JSON" "$(jq -e .openapi studio/openapi.json)" '"3.1.0"'
  check "openapi: every \$ref has a target" "$(jq -r '[.. | .["$ref"]? | select(.)] | unique[]' studio/openapi.json | sed 's|^#/||; s|/|.|g' | while read -r r; do jq -e ".$r" studio/openapi.json >/dev/null || echo "missing $r"; done | wc -l | tr -d ' ')" 0
fi

# Editing a picture chosen in the browser: GET/POST /edit (studio/upload.cho), a server of its own (port 18096) so the other tests' queue is not changed. The picture is made here (a JPEG's first bytes, then random ones): no photo of anyone.
U=$(mktemp -d)
(printf '\377\330\377\340'; head -c 3000000 /dev/urandom) > "$U/pic.jpg"
(printf '\211PNG\r\n\032\n'; head -c 2000 /dev/urandom) > "$U/pic.png"
(printf 'RIFF\0\0\0\0WEBPVP8 '; head -c 2000 /dev/urandom) > "$U/pic.webp"
printf 'a text file named as a picture' > "$U/fake.jpg"
mkdir "$U/data"
build/studio 18096 "$U/data" >"$U/s.log" 2>&1 &
SP3=$!
trap 'kill $SP $SP2 $SP3 2>/dev/null; rm -rf "$T" "$U"' EXIT
waitup "$U/s.log"
TOK3=$(grep -o 'token=[0-9a-f]*' "$U/s.log" | cut -d= -f2)
e() { curl -s -H "Authorization: Bearer $TOK3" "$@"; }
check "edit: the form" "$(e localhost:18096/edit | grep -c 'enctype=multipart/form-data')" 1
check "edit: the form takes a picture and has no script" "$(e localhost:18096/edit | grep -ci '<script')" 0
check "edit: a JPEG of 3 MB: 302 to the job" "$(e -o /dev/null -w '%{http_code} %{redirect_url}' --form-string 'prompt=sunny; "quoted" \back é' -F seed=7 -F "image=@$U/pic.jpg" localhost:18096/edit)" "302 http://localhost:18096/jobs/1"
check "edit: the picture is kept byte for byte" "$(cmp "$U/data/upload-1.jpg" "$U/pic.jpg" && echo same)" same
check "edit: the job's arguments (prompt as data, seed, the kept picture after --)" "$(tr '\0' '|' < "$U/data/1.argv" | sed "s|$U/data|D|g")" 'edit|-o|D/out/1|--seed|7|--|D/upload-1.jpg|sunny; "quoted" \back é|'
check "edit: the job is in the queue" "$(cat "$U/data/queue")" 1
check "edit: its page" "$(e -o /dev/null -w '%{http_code}' localhost:18096/jobs/1)" 200
check "edit: a PNG" "$(e -o /dev/null -w '%{http_code}' -F prompt=x -F "image=@$U/pic.png" localhost:18096/edit)" 302
check "edit: ...kept as .png" "$(ls "$U/data" | grep -c '^upload-2.png$')" 1
check "edit: a WebP" "$(e -o /dev/null -w '%{http_code}' -F prompt=x -F "image=@$U/pic.webp;type=image/webp" localhost:18096/edit)" 302
check "edit: ...kept as .webp" "$(ls "$U/data" | grep -c '^upload-3.webp$')" 1
check "edit: no seed is fine, the argument list has none" "$(tr '\0' ' ' < "$U/data/2.argv" | grep -c -- '--seed')" 0
N=$(wc -l < "$U/data/queue" | tr -d ' ')
check "edit: a text file named .jpg: 422" "$(e -w ' %{http_code}' -F prompt=x -F "image=@$U/fake.jpg;type=image/jpeg" localhost:18096/edit)" '{"error":"the picture is not a JPEG, PNG or WebP file"} 422'
check "edit: no prompt: 422" "$(e -o /dev/null -w '%{http_code}' -F "image=@$U/pic.jpg" localhost:18096/edit)" 422
check "edit: no picture: 422" "$(e -o /dev/null -w '%{http_code}' -F prompt=x localhost:18096/edit)" 422
check "edit: a bad seed: 422" "$(e -o /dev/null -w '%{http_code}' -F prompt=x -F seed=12ab -F "image=@$U/pic.jpg" localhost:18096/edit)" 422
check "edit: a prompt of 1001 bytes: 422" "$(e -o /dev/null -w '%{http_code}' --form-string "prompt=$(head -c 1001 /dev/zero | tr '\0' a)" -F "image=@$U/pic.jpg" localhost:18096/edit)" 422
check "edit: a prompt with a control character: 422" "$(e -o /dev/null -w '%{http_code}' --form-string "prompt=$(printf 'a\001b')" -F "image=@$U/pic.jpg" localhost:18096/edit)" 422
check "edit: JSON to /edit: 422" "$(e -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' -d '{}' localhost:18096/edit)" 422
check "edit: ...and none of those queued a job or kept a picture" "$(wc -l < "$U/data/queue" | tr -d ' ') $(ls "$U/data" | grep -c '^upload-')" "$N 3"
head -c 17000000 /dev/zero > "$U/huge.bin"
check "edit: over 16 MiB: 413" "$(e -o /dev/null -w '%{http_code}' -F prompt=x -F "image=@$U/huge.bin" localhost:18096/edit)" 413
check "edit: no token: 401" "$(curl -s -o /dev/null -w '%{http_code}' -F prompt=x -F "image=@$U/pic.jpg" localhost:18096/edit)" 401
check "edit: another site's form (Origin): 403" "$(e -o /dev/null -w '%{http_code}' -H 'Origin: http://evil.example' -F prompt=x -F "image=@$U/pic.jpg" localhost:18096/edit)" 403
check "edit: a rebinding Host: 403" "$(e -o /dev/null -w '%{http_code}' -H 'Host: rebinding.example:18096' -F prompt=x -F "image=@$U/pic.jpg" localhost:18096/edit)" 403
check "edit: multipart is not taken anywhere else (POST /api/jobs): 415" "$(e -o /dev/null -w '%{http_code}' -F prompt=x localhost:18096/api/jobs)" 415
check "edit: the server still answers after the uploads" "$(e -o /dev/null -w '%{http_code}' localhost:18096/api/models)" 200
kill $SP3 2>/dev/null; rm -rf "$U"
rc=0; build/studio 18095 "$T/data" relative 2>/dev/null || rc=$?; check "relative site dir refused" "$rc" 2
# Who may talk to the server (studio/auth.cho): the token, the host, the origin, the content type. Server 1, port 18091. `u` is a request without the token.
u() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
check "auth: no token: 401" "$(u localhost:18091/api/models)" 401
check "auth: wrong token: 401" "$(u -H 'Authorization: Bearer 00000000000000000000000000000000' localhost:18091/api/models)" 401
check "auth: a token of another length: 401" "$(u -H 'Authorization: Bearer abc' localhost:18091/api/models)" 401
check "auth: the page needs it too: 401" "$(u localhost:18091/)" 401
check "auth: the results need it: 401" "$(u localhost:18091/jobs/1)" 401
check "auth: a POST without it: 401, and nothing queued" "$(u -XPOST -H 'Content-Type: application/json' localhost:18091/api/jobs -d '{"kind":"video","prompt":"x"}') $(wc -l < "$T/data/queue" | tr -d ' ')" "401 3"
check "auth: bearer works" "$(u -H "Authorization: Bearer $TOK1" localhost:18091/api/models)" 200
check "auth: the cookie works" "$(u -H "Cookie: a=b; studio=$TOK1; c=d" localhost:18091/api/models)" 200
check "auth: a cookie with the wrong value: 401" "$(u -H "Cookie: studio=${TOK1}x" localhost:18091/api/models)" 401
check "auth: the token is 32 hex characters" "$(printf %s "$TOK1" | grep -cE '^[0-9a-f]{32}$')" 1
check "auth: the printed URL is the loopback one" "$(grep -c "^open http://127.0.0.1:18091/?token=$TOK1\$" "$T/s1.log")" 1
check "auth: the URL logs in: 302 to /" "$(curl -s -o /dev/null -w '%{http_code} %{redirect_url}' "localhost:18091/?token=$TOK1")" "302 http://localhost:18091/"
curl -s -D "$T/login" -o /dev/null "localhost:18091/?token=$TOK1"
check "auth: ...with a cookie HttpOnly and SameSite=Strict" "$(tr -d '\r' < "$T/login" | grep -i '^set-cookie' | grep -c "studio=$TOK1; HttpOnly; SameSite=Strict; Path=/")" 1
check "auth: a wrong token in the URL: 401" "$(u 'localhost:18091/?token=00000000000000000000000000000000')" 401
check "auth: the URL token is taken on / only" "$(u "localhost:18091/api/models?token=$TOK1")" 401
check "auth: the cookie logs in the page" "$(u -H "Cookie: studio=$TOK1" localhost:18091/)" 200
check "auth: a rebinding Host: 403" "$(u -H "Authorization: Bearer $TOK1" -H 'Host: rebinding.example:18091' localhost:18091/api/models)" 403
check "auth: the right host on the wrong port: 403" "$(u -H "Authorization: Bearer $TOK1" -H 'Host: localhost:9' localhost:18091/api/models)" 403
check "auth: 127.0.0.1:port is accepted" "$(u -H "Authorization: Bearer $TOK1" -H 'Host: 127.0.0.1:18091' localhost:18091/api/models)" 200
check "auth: localhost:port with trailing text: 403" "$(u -H "Authorization: Bearer $TOK1" -H 'Host: localhost:18091.evil.example' localhost:18091/api/models)" 403
check "auth: a foreign Origin: 403" "$(u -H "Authorization: Bearer $TOK1" -H 'Origin: http://evil.example' localhost:18091/api/models)" 403
check "auth: Origin null: 403" "$(u -H "Authorization: Bearer $TOK1" -H 'Origin: null' localhost:18091/api/models)" 403
check "auth: Origin http://localhost.evil.example:18091: 403" "$(u -H "Authorization: Bearer $TOK1" -H 'Origin: http://localhost.evil.example:18091' localhost:18091/api/models)" 403
check "auth: the page's own Origin is accepted" "$(u -H "Authorization: Bearer $TOK1" -H 'Origin: http://localhost:18091' localhost:18091/api/models)" 200
# The attack that was measured on the unguarded server (2026-11-16: 202 and a job in the queue): another site's page POSTs text/plain with the user's own cookie.
Q=$(wc -l < "$T/data/queue" | tr -d ' ')
check "attack: foreign Origin, text/plain, cookie: 403" "$(u -XPOST -H 'Origin: http://evil.example' -H 'Content-Type: text/plain' -H "Cookie: studio=$TOK1" localhost:18091/api/jobs -d '{"kind":"edit","prompt":"x","image":"/etc/hosts"}')" 403
check "attack: ...and no job was queued" "$(wc -l < "$T/data/queue" | tr -d ' ')" "$Q"
check "auth: same origin, text/plain: 415" "$(u -XPOST -H 'Content-Type: text/plain' -H "Cookie: studio=$TOK1" localhost:18091/api/jobs -d '{"kind":"video","prompt":"x"}')" 415
check "auth: no Content-Type on a POST: 415" "$(u -XPOST -H 'Content-Type:' -H "Cookie: studio=$TOK1" localhost:18091/api/jobs -d '{"kind":"video","prompt":"x"}')" 415
check "auth: two servers, two tokens" "$([ "$TOK1" != "$TOK2" ] && echo different)" different
rc=0; build/studio 18092 relative 2>/dev/null || rc=$?; check "relative data dir refused" "$rc" 2

# The rewrite option (docs/studio-rewrite.md, R3): the server writes `<id>.rewrite` (on or off) before the job is queued; the worker reads it. Servers on 18097 (default off) and 18098 (`--rewrite on`).
R=$(mktemp -d); mkdir "$R/off" "$R/on"
(printf '\211PNG\r\n\032\n'; head -c 2000 /dev/urandom) > "$R/pic.png"
build/studio 18097 "$R/off" >"$R/off.log" 2>&1 &
SP4=$!
build/studio 18098 "$R/on" --rewrite on >"$R/on.log" 2>&1 &
SP5=$!
trap 'kill $SP $SP2 $SP3 $SP4 $SP5 2>/dev/null; rm -rf "$T" "$U" "$R"' EXIT
waitup "$R/off.log" "$R/on.log"
TOKOFF=$(grep -o 'token=[0-9a-f]*' "$R/off.log" | cut -d= -f2); TOKON=$(grep -o 'token=[0-9a-f]*' "$R/on.log" | cut -d= -f2)
po() { curl -s -w ' %{http_code}' -H "Authorization: Bearer $TOKOFF" -XPOST -H 'Content-Type: application/json' localhost:18097/api/jobs -d "$1"; }
pn() { curl -s -w ' %{http_code}' -H "Authorization: Bearer $TOKON" -XPOST -H 'Content-Type: application/json' localhost:18098/api/jobs -d "$1"; }
check "rewrite: a server that starts with no flag writes off" "$(po '{"kind":"video","prompt":"a"}' >/dev/null; cat "$R/off/1.rewrite")" off
check "rewrite: ...and the job says on when it asks" "$(po '{"kind":"video","prompt":"b","rewrite":"on"}' >/dev/null; cat "$R/off/2.rewrite")" on
check "rewrite: a job can ask off" "$(po '{"kind":"video","prompt":"c","rewrite":"off"}' >/dev/null; cat "$R/off/3.rewrite")" off
check "rewrite: with --rewrite on the default is on" "$(pn '{"kind":"video","prompt":"a"}' >/dev/null; cat "$R/on/1.rewrite")" on
check "rewrite: ...and a job can still say off" "$(pn '{"kind":"video","prompt":"b","rewrite":"off"}' >/dev/null; cat "$R/on/2.rewrite")" off
check "rewrite: every queued job has its .rewrite file (two jobs, two lines, two files)" "$(wc -l < "$R/on/queue" | tr -d ' ') $(ls "$R/on"/*.rewrite | wc -l | tr -d ' ')" "2 2"
check "rewrite: a value that is not on or off: 422" "$(po '{"kind":"video","prompt":"d","rewrite":"maybe"}')" '{"error":"rewrite is on, off or ab"} 422'
check "rewrite: a value that is not a string: 422" "$(po '{"kind":"video","prompt":"d","rewrite":true}' | tail -c 4)" ' 422'
check "rewrite: a refused job leaves no file and no queue line" "$(ls "$R/off"/*.rewrite | wc -l | tr -d ' ') $(wc -l < "$R/off/queue" | tr -d ' ')" "3 3"
e2() { curl -s -w ' %{http_code}' -H "Authorization: Bearer $TOKOFF" "$@" localhost:18097/edit; }
check "rewrite: /edit takes the field (on)" "$(e2 -F prompt=x -F rewrite=on -F "image=@$R/pic.png" >/dev/null; cat "$R/off/4.rewrite")" on
check "rewrite: /edit without the field takes the server's default" "$(e2 -F prompt=x -F "image=@$R/pic.png" >/dev/null; cat "$R/off/5.rewrite")" off
check "rewrite: /edit with a value that is not on or off: 422" "$(e2 -F prompt=x -F rewrite=perhaps -F "image=@$R/pic.png")" '{"error":"rewrite is on, off or ab"} 422'
check "rewrite: the /edit form offers it" "$(curl -s -H "Authorization: Bearer $TOKOFF" localhost:18097/edit | grep -c 'name=rewrite')" 1
rc=0; build/studio 18099 "$R/off" --rewrite maybe 2>/dev/null || rc=$?; check "rewrite: a flag value that is not on or off refuses to start" "$rc" 2
rc=0; build/studio 18099 "$R/off" --rewrite 2>/dev/null || rc=$?; check "rewrite: a flag with no value is not a site directory" "$rc" 2
# the job's page: what a model wrote is shown escaped
printf '%s' '<script>alert(1)</script> & "q" '"'"'s'"'" > "$R/off/2.prompt.orig"; printf '%s' '<b>new</b> & more' > "$R/off/2.prompt.new"; printf 'done\n' > "$R/off/2.state"
pg() { curl -s -H "Authorization: Bearer $TOKOFF" localhost:18097/jobs/$1; }
check "page: the original prompt is shown, escaped" "$(pg 2 | grep -c 'Prompt as written</h2><p>&lt;script&gt;alert(1)&lt;/script&gt; &amp; &quot;q&quot; &#39;s')" 1
check "page: ...no raw tag from the model is in the page" "$(pg 2 | grep -c '<script>\|<b>new')" 0
check "page: the rewritten prompt is shown, escaped" "$(pg 2 | grep -c 'Rewritten by the local model</h2><p>&lt;b&gt;new&lt;/b&gt; &amp; more')" 1
printf 'done\n' > "$R/off/2.state"; printf 'failed 1\n' > "$R/off/3.state"; printf 'on\n' > "$R/off/3.rewrite"
check "page: a job that asked and finished with no rewritten prompt says so" "$(pg 3 | grep -c 'the prompt was not rewritten (see the log)')" 1
check "page: a job that did not ask says nothing of it" "$(pg 1 | grep -c 'rewritten')" 0
check "page: a queued job that asked does not say it was not rewritten yet" "$(printf 'on\n' > "$R/off/1.rewrite"; pg 1 | grep -c 'was not rewritten')" 0

# The blind comparison (docs/studio-rewrite.md, R4), on the server of 18097 (default off). A pair is two jobs of one request, with the rewrite off and on, and `<first>.pair` = `<second> <flip>`.
pj() { po "$1" | sed 's/ [0-9][0-9][0-9]$//'; }
hdr() { echo "Authorization: Bearer $TOKOFF"; }
cget() { curl -s -w ' %{http_code}' -H "$(hdr)" "localhost:18097$1"; }
cpost() { curl -s -w ' %{http_code}' -H "$(hdr)" -XPOST -H 'Content-Type: application/x-www-form-urlencoded' -d "$2" "localhost:18097$1"; }
finish() { for j in "$@"; do mkdir -p "$R/off/out/$j"; printf 'fake' > "$R/off/out/$j/video.mp4"; echo done > "$R/off/$j.state"; done; }
A=$(pj '{"kind":"video","prompt":"a pair","seed":3,"rewrite":"ab"}'); A1=$(echo "$A" | jq -r .id); A2=$(echo "$A" | jq -r .pair)
check "pair: the answer holds both ids, consecutive" "$A2" "$((A1 + 1))"
check "pair: the first has the rewrite off, the second on" "$(cat "$R/off/$A1.rewrite") $(cat "$R/off/$A2.rewrite")" "off on"
check "pair: the two argument lists differ only in their output directory" "$(tr '\0' '|' < "$R/off/$A1.argv" | sed 's#out/[0-9]*#out/X#') = $(tr '\0' '|' < "$R/off/$A2.argv" | sed 's#out/[0-9]*#out/X#')" "$(tr '\0' '|' < "$R/off/$A1.argv" | sed 's#out/[0-9]*#out/X#') = $(tr '\0' '|' < "$R/off/$A1.argv" | sed 's#out/[0-9]*#out/X#')"
check "pair: ...and the seed and the prompt are in both" "$(tr '\0' ' ' < "$R/off/$A2.argv" | sed 's#out/[0-9]*#out/X#')" "video -o $R/off/out/X --seed 3 -- a pair "
check "pair: the record is the second id and a bit" "$(cat "$R/off/$A1.pair" | tr -d '\n' | grep -c "^$A2 [01]\$")" 1
check "pair: both ids are in the queue" "$(tail -2 "$R/off/queue" | tr '\n' ' ')" "$A1 $A2 "
flips=""; for n in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do r=$(pj '{"kind":"video","prompt":"coin","rewrite":"ab"}'); flips="$flips$(cut -d' ' -f2 "$R/off/$(echo "$r" | jq -r .id).pair")"; done
check "pair: the order is random (both values in 30 pairs)" "$(echo "$flips" | grep -c 0) $(echo "$flips" | grep -c 1)" "1 1"
check "pair: a refused pair makes no job and no pair file" "$(Q=$(wc -l < "$R/off/queue"); po '{"kind":"video","prompt":"","rewrite":"ab"}' >/dev/null; echo "$(wc -l < "$R/off/queue" | tr -d ' ') $Q" | awk '{print ($1==$2)?"same":"changed"}')" same
check "pair: /compare of a job that is not a pair: 404" "$(cget /compare/$((A2 + 100)) | tail -c 4)" " 404"
check "pair: ...and of an id that is not digits: 422" "$(cget /compare/x1 | tail -c 4)" " 422"
# before the jobs finish
check "wait: the page says the jobs are not done and offers no choice" "$(cget /compare/$A1 | grep -c 'name=choice')" 0
check "wait: ...and a choice is refused: 409" "$(cpost /compare/$A1 choice=left | tail -c 4)" " 409"
# when both are done: still blind
finish $A1 $A2
P=$(cget /compare/$A1)
check "blind: once both are done the page offers the three choices" "$(echo "$P" | grep -o 'name=choice value=[a-z]*' | tr '\n' ' ')" "name=choice value=left name=choice value=tie name=choice value=right "
check "blind: ...and says neither the prompts nor which result is which" "$(echo "$P" | grep -ci 'as written:\|rewritten:\|result used\|does not count')" 0
check "blind: ...the results are the two videos, one on each side" "$(echo "$P" | grep -o '<h2>[a-z]*</h2>' | tr '\n' ' ')" "<h2>left</h2> <h2>right</h2> "
# the choice
check "choice: a word that is not one of the three: 422" "$(cpost /compare/$A1 choice=best | tail -c 4)" " 422"
check "choice: ...and an empty body: 422" "$(cpost /compare/$A1 '' | tail -c 4)" " 422"
check "choice: left is recorded and the person is sent back to the page" "$(curl -s -o /dev/null -w '%{http_code} %{redirect_url}' -H "$(hdr)" -XPOST -H 'Content-Type: application/x-www-form-urlencoded' -d choice=left localhost:18097/compare/$A1)" "302 http://localhost:18097/compare/$A1"
check "choice: the file says left" "$(cat "$R/off/$A1.verdict")" left
check "choice: a second choice is refused and changes nothing: 409" "$(cpost /compare/$A1 choice=right | tail -c 4) $(cat "$R/off/$A1.verdict")" " 409 left"
FLIP=$(cut -d' ' -f2 "$R/off/$A1.pair")
check "reveal: the page says which side had the prompt as written" "$(cget /compare/$A1 | grep -c "The left result used $([ $FLIP = 0 ] && echo 'the prompt as written' || echo 'the rewritten prompt')")" 1
check "reveal: a pair whose second job was not really rewritten says it does not count" "$(cget /compare/$A1 | grep -c 'This pair does not count')" 1
check "reveal: the prompt as written is shown" "$(cget /compare/$A1 | grep -c 'as written: a pair')" 1
# a form from another site, no token, a JSON post
check "gate: another site's form (Origin): 403" "$(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -H 'Origin: http://evil.example' -XPOST -H 'Content-Type: application/x-www-form-urlencoded' -d choice=tie localhost:18097/compare/$A1)" 403
check "gate: no token: 401" "$(curl -s -o /dev/null -w '%{http_code}' -XPOST -H 'Content-Type: application/x-www-form-urlencoded' -d choice=tie localhost:18097/compare/$A1)" 401
check "gate: JSON to /compare/N passes the gate and the handler refuses the body: 422" "$(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -XPOST -H 'Content-Type: application/json' -d '{}' localhost:18097/compare/$A1)" 422
check "gate: a form is not taken on another route (POST /api/jobs): 415" "$(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -XPOST -H 'Content-Type: application/x-www-form-urlencoded' -d choice=tie localhost:18097/api/jobs)" 415
# the tally, from pairs whose outcomes are known: four video pairs and one edit pair
mk() { # kind prompt -> first id (prints "first second")
  if [ "$1" = edit ]; then r=$(pj "{\"kind\":\"edit\",\"prompt\":\"$2\",\"image\":\"/tmp/p.png\",\"rewrite\":\"ab\"}"); else r=$(pj "{\"kind\":\"video\",\"prompt\":\"$2\",\"rewrite\":\"ab\"}"); fi
  echo "$(echo "$r" | jq -r .id) $(echo "$r" | jq -r .pair)"
}
rewritten_side() { f=$(cut -d' ' -f2 "$R/off/$1.pair"); [ "$f" = 1 ] && echo left || echo right; }
original_side() { f=$(cut -d' ' -f2 "$R/off/$1.pair"); [ "$f" = 1 ] && echo right || echo left; }
set -- $(mk video w1); W1=$1; W1B=$2; finish $W1 $W1B; echo "a rewritten one" > "$R/off/$W1B.prompt.new"; cpost /compare/$W1 "choice=$(rewritten_side $W1)" >/dev/null
set -- $(mk video w2); W2=$1; W2B=$2; finish $W2 $W2B; echo "another rewritten" > "$R/off/$W2B.prompt.new"; cpost /compare/$W2 "choice=$(rewritten_side $W2)" >/dev/null
set -- $(mk video l1); L1=$1; L1B=$2; finish $L1 $L1B; echo "rewritten again" > "$R/off/$L1B.prompt.new"; cpost /compare/$L1 "choice=$(original_side $L1)" >/dev/null
set -- $(mk video t1); T1=$1; T1B=$2; finish $T1 $T1B; echo "rewritten tie" > "$R/off/$T1B.prompt.new"; cpost /compare/$T1 choice=tie >/dev/null
set -- $(mk video v1); V1=$1; V1B=$2; finish $V1 $V1B; cpost /compare/$V1 choice=left >/dev/null
set -- $(mk edit e1); E1=$1; E1B=$2; finish $E1 $E1B; echo "edit rewritten" > "$R/off/$E1B.prompt.new"; cpost /compare/$E1 "choice=$(rewritten_side $E1)" >/dev/null
IDX=$(cget /compare)
check "tally: video, counted by who won (2 for the rewritten, 1 for the prompt as written, 1 tie; the 1 earlier pair did not count)" "$(echo "$IDX" | sed 's/<[^>]*>/ /g' | grep -o 'video: the rewritten prompt won [0-9]*, the prompt as written won [0-9]*, ties [0-9]*')" "video: the rewritten prompt won 2, the prompt as written won 1, ties 1"
check "tally: edit" "$(echo "$IDX" | sed 's/<[^>]*>/ /g' | grep -o 'edit: the rewritten prompt won [0-9]*, the prompt as written won [0-9]*, ties [0-9]*')" "edit: the rewritten prompt won 1, the prompt as written won 0, ties 0"
check "tally: pairs chosen without a rewritten prompt are not counted (the first pair and v1)" "$(echo "$IDX" | grep -o 'not counted: [0-9]*')" "not counted: 2"
check "tally: every pair is a link" "$(echo "$IDX" | grep -c "href=/compare/$W1>")" 1
check "tally: the page of one of the rewritten wins reveals it" "$(cget /compare/$W1 | grep -c 'rewritten: a rewritten one')" 1
# the /edit form makes a pair too, and sends the person to the pair
EP=$(curl -s -o /dev/null -w '%{redirect_url}' -H "$(hdr)" -F prompt=x -F rewrite=ab -F "image=@$R/pic.png" localhost:18097/edit); EID=${EP##*/}
check "edit: rewrite=ab makes a pair and goes to /compare/N" "$(echo "$EP" | sed 's#.*/compare/[0-9]*#ok#')" ok
PIC1=$(tr '\0' '\n' < "$R/off/$EID.argv" | grep '/upload-'); PIC2=$(tr '\0' '\n' < "$R/off/$((EID + 1)).argv" | grep '/upload-')
check "edit: ...and both jobs of the pair name the same uploaded picture" "$([ -n "$PIC1" ] && [ "$PIC1" = "$PIC2" ] && echo same)" same
check "edit: ...with the rewrite off on the first and on on the second" "$(cat "$R/off/$EID.rewrite") $(cat "$R/off/$((EID + 1)).rewrite")" "off on"
check "edit: the form offers ab" "$(curl -s -H "$(hdr)" localhost:18097/edit | grep -c 'value=ab')" 1
# Variations (docs/studio-variations.md, V1): one request, N jobs, the seeds s .. s+N-1, a set file written after the jobs, a page of the set.
Vq=$(wc -l < "$R/off/queue" | tr -d ' ')
VR=$(pj '{"kind":"video","prompt":"fox","seed":5,"variations":3}'); VF=$(echo "$VR" | jq -r .id)
check "variations: the answer holds the three ids, consecutive, and the set's address" "$(echo "$VR" | jq -c '[.ids, .set]')" "[[$VF,$((VF + 1)),$((VF + 2))],\"/sets/$VF\"]"
check "variations: the seeds are 5, 6 and 7, one in each argument list" "$(for k in 0 1 2; do tr '\0' ' ' < "$R/off/$((VF + k)).argv" | sed 's#.*--seed \([0-9]*\) .*#\1#'; echo; done | tr '\n' ' ')" "5 6 7 "
check "variations: ...and nothing else differs (prompt, kind)" "$(for k in 0 1 2; do tr '\0' ' ' < "$R/off/$((VF + k)).argv" | sed 's#out/[0-9]*#out/X#; s#--seed [0-9]*#--seed S#'; done | sort -u | wc -l | tr -d ' ')" 1
check "variations: the set file is the ids, then the first seed" "$(tr '\n' '|' < "$R/off/$VF.set")" "$VF $((VF + 1)) $((VF + 2))|5|"
check "variations: all three are in the queue, in order" "$(tail -3 "$R/off/queue" | tr '\n' ' ')" "$VF $((VF + 1)) $((VF + 2)) "
VD=$(pj '{"kind":"video","prompt":"no seed","variations":2}'); VDF=$(echo "$VD" | jq -r .id)
check "variations: with no seed they start at 1" "$(for k in 0 1; do tr '\0' ' ' < "$R/off/$((VDF + k)).argv" | sed 's#.*--seed \([0-9]*\) .*#\1#'; echo; done | tr '\n' ' ')" "1 2 "
Vq2=$(wc -l < "$R/off/queue" | tr -d ' ')
for bad in '"variations":1' '"variations":9' '"variations":"3"' '"variations":2.5' '"variations":0' '"variations":-2'; do
  check "variations: $bad is refused: 422" "$(po "{\"kind\":\"video\",\"prompt\":\"x\",$bad}" | tail -c 4)" " 422"
done
check "variations: with rewrite ab is refused: 422" "$(po '{"kind":"video","prompt":"x","variations":2,"rewrite":"ab"}' | tail -c 4)" " 422"
check "variations: a seed whose last variation would pass 2147483647 is refused: 422" "$(po '{"kind":"video","prompt":"x","variations":3,"seed":2147483646}' | tail -c 4)" " 422"
check "variations: ...and the largest that fits is accepted" "$(po '{"kind":"video","prompt":"x","variations":3,"seed":2147483645}' | tail -c 4)" " 202"
check "variations: a bad request makes no job (the prompt is empty)" "$(po '{"kind":"video","prompt":"","variations":4}' | tail -c 4) $(wc -l < "$R/off/queue" | tr -d ' ')" " 422 $((Vq2 + 3))"
check "variations: the refusals above queued nothing (6 + 3 + the accepted three)" "$(wc -l < "$R/off/queue" | tr -d ' ')" "$((Vq2 + 3))"
check "variations: the page shows each variation with its seed" "$(cget /sets/$VF | grep -o 'variation [0-9], seed [0-9]*' | tr '\n' ';')" "variation 1, seed 5;variation 2, seed 6;variation 3, seed 7;"
check "variations: while the jobs wait the page refreshes and says so" "$(cget /sets/$VF | grep -c 'http-equiv=refresh') $(cget /sets/$VF | grep -o '<p>queued</p>' | wc -l | tr -d ' ')" "1 3"
finish $VF $((VF + 1)) $((VF + 2))
check "variations: when all are done, three videos and no refresh" "$(cget /sets/$VF | grep -o '<video' | wc -l | tr -d ' ') $(cget /sets/$VF | grep -c 'http-equiv=refresh')" "3 0"
echo failed 3 > "$R/off/$((VF + 1)).state"; rm -f "$R/off/out/$((VF + 1))/video.mp4"
check "variations: one failed shows as it is and the others stay" "$(cget /sets/$VF | grep -o '<p>failed 3</p>' | wc -l | tr -d ' ') $(cget /sets/$VF | grep -o '<video' | wc -l | tr -d ' ')" "1 2"
check "variations: /sets of a job that is not the first of a set: 404" "$(cget /sets/$((VF + 1)) | tail -c 4)" " 404"
check "variations: /sets of an id that is not digits: 422" "$(cget /sets/x1 | tail -c 4)" " 422"

check "style: /style.css is served as CSS" "$(curl -s -D - -o /dev/null -H "$(hdr)" localhost:18097/style.css | tr -d '\r' | grep -i '^content-type' | tr 'A-Z' 'a-z')" "content-type: text/css; charset=utf-8"
check "style: ...and holds the card and the grid the pages use" "$(curl -s -H "$(hdr)" localhost:18097/style.css | grep -o '\.cell{\|\.grid{\|color-scheme:dark\|--accent:#F2B544' | sort -u | tr '\n' ' ')" "--accent:#F2B544 .cell{ .grid{ color-scheme:dark "
check "style: ...and needs the token: 401" "$(curl -s -o /dev/null -w '%{http_code}' localhost:18097/style.css)" 401
check "style: the set page links it and is mobile-ready" "$(cget /sets/$VF | grep -c 'href=/style.css') $(cget /sets/$VF | grep -c 'name=viewport')" "1 1"
check "style: every page the server writes links it (jobs, a job, a set, compare index, the edit form)" "$(for u in /jobs /jobs/$VF /sets/$VF /compare /edit; do cget $u | grep -c 'href=/style.css'; done | tr '\n' ' ')" "1 1 1 1 1 "
check "variations: /sets with no token: 401" "$(curl -s -o /dev/null -w '%{http_code}' localhost:18097/sets/$VF)" 401
# The CLI's parameters in the API (steps, fps, model), the prompt on the pages, and the /edit form's variations (docs/studio-variations.md, section 8).
argv_of() { tr '\0' ' ' < "$R/off/$1.argv" | sed 's#out/[0-9]*#out/X#'; }
Pq=$(wc -l < "$R/off/queue" | tr -d ' ')
PE=$(pj '{"kind":"edit","prompt":"param edit","image":"/tmp/p.png","seed":2,"steps":4,"model":"qwen-image-2.1-turbo"}'); PEI=$(echo "$PE" | jq -r .id)
check "params: an edit with steps and model reaches media as --model and --steps" "$(argv_of $PEI)" "edit -o $R/off/out/X --seed 2 --model qwen-image-2.1-turbo --steps 4 -- /tmp/p.png param edit "
PV=$(pj '{"kind":"video","prompt":"param video","fps":24,"model":"fastwan2.2-ti2v-5b"}'); PVI=$(echo "$PV" | jq -r .id)
check "params: a video with fps and model reaches media as --model and --fps" "$(argv_of $PVI)" "video -o $R/off/out/X --model fastwan2.2-ti2v-5b --fps 24 -- param video "
Pq2=$(wc -l < "$R/off/queue" | tr -d ' ')
for bad in '"kind":"video","prompt":"x","steps":4' '"kind":"edit","prompt":"x","image":"/tmp/p.png","fps":24' '"kind":"edit","prompt":"x","image":"/tmp/p.png","steps":0' '"kind":"edit","prompt":"x","image":"/tmp/p.png","steps":51' '"kind":"edit","prompt":"x","image":"/tmp/p.png","steps":"4"' '"kind":"video","prompt":"x","fps":61' '"kind":"video","prompt":"x","fps":0' '"kind":"video","prompt":"x","model":"../x"' '"kind":"video","prompt":"x","model":"Fast Wan"' '"kind":"video","prompt":"x","model":""' '"kind":"video","prompt":"x","model":5'; do
  check "params: refused 422: $bad" "$(po "{$bad}" | tail -c 4)" " 422"
done
check "params: the refusals queued nothing" "$(wc -l < "$R/off/queue" | tr -d ' ')" "$Pq2"
PH=$(pj '{"kind":"video","prompt":"a <b>fox</b> & \"snow\""}'); PHI=$(echo "$PH" | jq -r .id)
check "prompt: a job page shows the prompt, escaped, even with no rewrite" "$(cget /jobs/$PHI | grep -c 'Prompt as written</h2><p>a &lt;b&gt;fox&lt;/b&gt; &amp; &quot;snow&quot;')" 1
check "prompt: a set page shows it once" "$(cget /sets/$VF | grep -o 'prompt: fox' | wc -l | tr -d ' ')" 1
check "prompt: a rewritten job still shows its prompt once (the pair's first job)" "$(cget /jobs/$A1 | grep -o 'Prompt as written' | wc -l | tr -d ' ')" 1
check "edit form: it offers variations, steps and model" "$(curl -s -H "$(hdr)" localhost:18097/edit | grep -o 'name=variations\|name=steps\|name=model' | sort | tr '\n' ' ')" "name=model name=steps name=variations "
EV=$(curl -s -o /dev/null -w '%{redirect_url}' -H "$(hdr)" -F prompt=vary -F variations=3 -F steps=4 -F model=qwen-image-2.1-turbo -F "image=@$R/pic.png" localhost:18097/edit); EVI=${EV##*/}
check "edit form: variations=3 goes to /sets/N" "$(echo "$EV" | sed 's#.*/sets/[0-9]*#ok#')" ok
check "edit form: ...three jobs with the seeds 1, 2, 3 and the steps and model in each" "$(for k in 0 1 2; do argv_of $((EVI + k)) | sed 's#.*--seed \([0-9]*\) --model \([^ ]*\) --steps \([0-9]*\) .*#\1/\2/\3#'; echo; done | tr '\n' ' ')" "1/qwen-image-2.1-turbo/4 2/qwen-image-2.1-turbo/4 3/qwen-image-2.1-turbo/4 "
check "edit form: variations=1 is one job and goes to /jobs/N" "$(curl -s -o /dev/null -w '%{redirect_url}' -H "$(hdr)" -F prompt=one -F variations=1 -F "image=@$R/pic.png" localhost:18097/edit | sed 's#.*/jobs/[0-9]*#ok#')" ok
check "edit form: variations=9 is refused: 422" "$(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -F prompt=nine -F variations=9 -F "image=@$R/pic.png" localhost:18097/edit)" 422
check "edit form: steps that are not digits are refused: 422" "$(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -F prompt=s -F steps=4x -F "image=@$R/pic.png" localhost:18097/edit)" 422
# The design applied (docs/studio-design.md): one header on every page, the queue's status in it, the job page's parameters, the cards of /jobs.
for pair in "/jobs:jobs" "/edit:edit" "/compare:compare" "/sets/$VF:jobs" "/jobs/$PEI:jobs"; do
  u=${pair%%:*}; k=${pair##*:}; case $k in jobs) href=/jobs; place=/gallery;; edit) href=/edit; place=/;; compare) href=/compare; place=/gallery;; esac
  check "design: $u has the header and marks $k as the open tab, in its place (Create or Library)" "$(cget $u | grep -c "class=brand") $(cget $u | grep -o '<a href=[^>]*aria-current=page>' | tr '\n' ' ')" "1 <a href=$place aria-current=page> <a href=$href aria-current=page> "
done
check "design: the status of an idle GPU with one waiting job" "$(TOK=$TOK2 c localhost:18094/jobs | grep -o 'GPU idle[^<]*')" "GPU idle &middot; 1 queued"
mkdir -p "$R/off/out/$PEI"; printf 'x' > "$R/off/out/$PEI/edited.png"; echo running > "$R/off/$PEI.state"
check "design: with a job running the status says busy and counts it" "$(cget /jobs | grep -o 'GPU busy &middot; 1 running &middot; [0-9]* queued' | sed 's/[0-9]* queued/N queued/')" "GPU busy &middot; 1 running &middot; N queued"
check "design: a running job's card has the moving bar and the running badge" "$(cget /jobs | grep -o 'class=\"badge running\"\|class=bar' | tr '\n' ' ')" 'class="badge running" class=bar '
printf '== lowering\nsteps 8, mu 1.2, timesteps []\nstep 0: t 1000.0\nstep 2: t 900.0, 9 s wall\n' > "$R/off/$PEI.log"
check "progress: a running edit shows its stage and the steps done, and the bar is that far" "$(cget /jobs | grep -o 'class=det style=\"width:[0-9]*%\"\|lowering &middot; step [0-9]* of [0-9]*' | tr '\n' ' ')" 'class=det style="width:37%" lowering &middot; step 3 of 8 '
printf '== sampling\nstep 5 of 10: sigma 0.5 -> 0.4, 3 s\n' > "$R/off/$PEI.log"
check "progress: a video's \"step i of N\" is read too" "$(cget /jobs | grep -o 'class=det style=\"width:[0-9]*%\"\|sampling &middot; step [0-9]* of [0-9]*' | tr '\n' ' ')" 'class=det style="width:50%" sampling &middot; step 5 of 10 '
printf 'no stage lines here\n' > "$R/off/$PEI.log"
check "progress: a log with none keeps the moving bar and no stage" "$(cget /jobs | grep -c 'class=det')" 0
check "recent: the frame's page has no header, refreshes itself, opens cards in the window, and shows the running card" "$(cget /recent | grep -c 'class=brand') $(cget /recent | grep -c 'http-equiv=refresh content=5><base target=_top>') $(cget /recent | grep -c 'class=\"job running\"')" "0 1 1"
check "recent: it is the same cards as /jobs (the ids, newest first)" "$(cget /recent | grep -o 'class=idn>#[0-9]*' | head -3 | tr '\n' ' ')" "$(cget /jobs | grep -o 'class=idn>#[0-9]*' | head -3 | tr '\n' ' ')"
rm -f "$R/off/$PEI.log"
check "design: the job page's parameters are its kind and its options, in order" "$(cget /jobs/$PEI | grep -o '<dt>[a-z]*</dt><dd>[^<]*</dd>' | sed 's/<dt>//; s#</dt><dd># #; s#</dd>##' | tr '\n' ';')" "kind edit;seed 2;model qwen-image-2.1-turbo;steps 4;rewrite off;"
echo done > "$R/off/$PEI.state"
check "design: a finished edit is its own thumbnail on /jobs" "$(cget /jobs | grep -c "loading=lazy src=/files/$PEI/edited.png")" 1
check "design: a job page names its files that exist, and the log" "$(cget /jobs/$PEI | grep -o '<li><a href=/files/[0-9]*/[a-z.]*' | sed 's#.*/files/[0-9]*/##' | tr '\n' ' ')" "edited.png log "
check "design: a failed job's page says so instead of a result" "$(cget /jobs/$((VF + 1)) | grep -c 'class=alert role=alert')" 1
check "design: the edit form has the drop zone and the instruction as a text area" "$(curl -s -H "$(hdr)" localhost:18097/edit | grep -c 'name=image type=file accept=image/jpeg,image/png,image/webp></label>.*<textarea name=prompt')" 1
check "design: the stylesheet names the font stack with the system font as fallback and loads nothing" "$(curl -s -H "$(hdr)" localhost:18097/style.css | grep -c "IBM Plex Sans',system-ui") $(curl -s -H "$(hdr)" localhost:18097/style.css | grep -c '@import\|url(')" "1 0"
# Text to image (docs/text-to-image.md): the kind `image`.
Iq=$(wc -l < "$R/off/queue" | tr -d ' ')
IM=$(pj '{"kind":"image","prompt":"a lighthouse","size":"512x512","seed":3,"steps":4,"model":"qwen-image-2.1-turbo"}'); IMI=$(echo "$IM" | jq -r .id)
check "image: the arguments media gets" "$(argv_of $IMI)" "image -o $R/off/out/X --size 512x512 --seed 3 --model qwen-image-2.1-turbo --steps 4 -- a lighthouse "
IP=$(pj '{"kind":"image","prompt":"plain"}'); IPI=$(echo "$IP" | jq -r .id)
check "image: with nothing else, only the kind, the output and the prompt" "$(argv_of $IPI)" "image -o $R/off/out/X -- plain "
Iq2=$(wc -l < "$R/off/queue" | tr -d ' ')
for bad in '"frames":17' '"fps":24' '"size":"500x512"' '"size":"1056x1056"' '"size":"2048x2048"' '"size":"32x32"' '"steps":51' '"steps":0' '"rewrite":"on"' '"rewrite":"ab"'; do
  check "image: refused 422: $bad" "$(po "{\"kind\":\"image\",\"prompt\":\"x\",$bad}" | tail -c 4)" " 422"
done
check "image: the refusals queued nothing" "$(wc -l < "$R/off/queue" | tr -d ' ')" "$Iq2"
check "image: a size at the limit is accepted (1024x1024, 512x1024, 96x96)" "$(for z in 1024x1024 512x1024 96x96; do po "{\"kind\":\"image\",\"prompt\":\"x\",\"size\":\"$z\"}" | tail -c 4; done | tr '\n' ' ')" " 202  202  202"
check "image: rewrite off is accepted" "$(po '{"kind":"image","prompt":"x","rewrite":"off"}' | tail -c 4)" " 202"
IW=$(pn '{"kind":"image","prompt":"no rewrite on an image"}' | sed 's/ [0-9][0-9][0-9]$//'); IWI=$(echo "$IW" | jq -r .id)
check "image: the server's rewrite default (on) does not apply to an image" "$(tr -d '\n' < "$R/on/$IWI.rewrite")" off
IV=$(pj '{"kind":"image","prompt":"vary an image","variations":2,"size":"512x512"}'); IVF=$(echo "$IV" | jq -r .id)
check "image: variations make a set of images with the seeds 1 and 2" "$(for k in 0 1; do argv_of $((IVF + k)) | sed 's#.*--seed \([0-9]*\) .*#\1#'; echo; done | tr '\n' ' ') $(echo "$IV" | jq -r '.set')" "1 2  /sets/$IVF"
mkdir -p "$R/off/out/$IMI"; printf 'PNGDATA' > "$R/off/out/$IMI/image.png"; echo done > "$R/off/$IMI.state"
check "image: a finished image shows on its page" "$(cget /jobs/$IMI | grep -c "alt=\"the generated picture\" src=/files/$IMI/image.png")" 1
check "image: ...is served as a PNG" "$(curl -s -D - -o /dev/null -H "$(hdr)" localhost:18097/files/$IMI/image.png | tr -d '\r' | grep -i '^content-type' | tr 'A-Z' 'a-z')" "content-type: image/png"
check "image: ...and is its own thumbnail on /jobs" "$(cget /jobs | grep -c "loading=lazy src=/files/$IMI/image.png")" 1
check "image: ...and the page lists the file" "$(cget /jobs/$IMI | grep -o '<li><a href=/files/[0-9]*/[a-z.]*' | sed 's#.*/files/[0-9]*/##' | tr '\n' ' ')" "image.png log "
mkdir -p "$R/off/out/$IVF"; printf 'PNGDATA' > "$R/off/out/$IVF/image.png"; echo done > "$R/off/$IVF.state"
check "image: a set's cell shows the image" "$(cget /sets/$IVF | grep -c "alt=\"variation\" src=/files/$IVF/image.png")" 1
check "image: the OpenAPI document names the kind and the file" "$(jq -r '.components.schemas.NewJob.properties.kind.enum | join(",")' studio/openapi.json) $(jq -r '.paths["/files/{id}/{name}"].get.parameters[1].schema.enum | index("image.png")' studio/openapi.json)" "video,edit,image 1"
# The cancho-ui page of `/` served with the design (docs/studio-design.md section 6): the stylesheet linked after the page's own style and the header after <body>; a page without those tags is served as it is.
if [ -n "${UI:-}" ]; then
  PG=$(TOK=$TOK2 c localhost:18094/)
  check "main page: the stylesheet is linked once, after the page's own style and before </head>" "$(echo "$PG" | tr '\n' ' ' | grep -c '</style> *<link rel=stylesheet href=/style.css></head>')" 1
  check "main page: the header is there once and marks Generate, in Create, as the open tab" "$(echo "$PG" | grep -c 'class=brand') $(echo "$PG" | grep -o 'aria-current=page>[A-Za-z ]*' | tr '\n' ' ')" "1 aria-current=page>Create aria-current=page>Generate "
  check "main page: the app is untouched (its root and its script are where they were)" "$(echo "$PG" | grep -c 'id="root"') $(echo "$PG" | grep -c 'src="./app.js"')" "1 1"
  check "main page: the header comes before the app's root" "$(echo "$PG" | tr '\n' ' ' | grep -c '<body><header class=top>.*</header>.*<div id="root"')" 1
  check "main page: the frame of recent jobs is once, after the app and before </body>" "$(echo "$PG" | tr '\n' ' ' | grep -c '<script type="module" src="./app.js"></script> *<iframe class=recent src=/recent title="Recent jobs"></iframe></body>')" 1
else
  check "main page: a page with no <head> and <body> tags is served as it is" "$(TOK=$TOK2 c localhost:18094/)" '<!doctype html><title>fake page</title>'
fi
check "main page: the stylesheet has the rules for the page's fields" "$(TOK=$TOK2 c localhost:18094/style.css | grep -o '#root \.cui-field{\|#root>\.cui-col{' | sort -u | tr '\n' ' ')" "#root .cui-field{ #root>.cui-col{ "
check "narrow screens: the stylesheet has one column and 16 px gutters under 600 px" "$(TOK=$TOK2 c localhost:18094/style.css | grep -c '@media(max-width:600px){header.top{padding:12px 16px}.*\.grid{grid-template-columns:1fr}')" 1

# Several pictures (docs/two-images.md): an edit with a second picture, in the API and in the /edit form.
Tq=$(wc -l < "$R/off/queue" | tr -d ' ')
TW=$(pj '{"kind":"edit","prompt":"combine them","image":"/tmp/a.png","also":"/tmp/b.png","condition_area":1048576,"seed":3}'); TWI=$(echo "$TW" | jq -r .id)
check "two pictures: the arguments media gets" "$(argv_of $TWI)" "edit -o $R/off/out/X --seed 3 --also /tmp/b.png --condition-area 1048576 -- /tmp/a.png combine them "
TW2=$(pj '{"kind":"edit","prompt":"combine them","image":"/tmp/a.png","also":"/tmp/b.png"}'); TW2I=$(echo "$TW2" | jq -r .id)
check "two pictures: without condition_area media's default applies (none passed)" "$(argv_of $TW2I)" "edit -o $R/off/out/X --also /tmp/b.png -- /tmp/a.png combine them "
Tq2=$(wc -l < "$R/off/queue" | tr -d ' ')
for bad in '"kind":"video","prompt":"x","also":"/tmp/b.png"' '"kind":"image","prompt":"x","also":"/tmp/b.png"' '"kind":"edit","prompt":"x","image":"/tmp/a.png","also":""' '"kind":"edit","prompt":"x","image":"/tmp/a.png","also":5' '"kind":"edit","prompt":"x","image":"/tmp/a.png","condition_area":589824' '"kind":"edit","prompt":"x","image":"/tmp/a.png","also":"/tmp/b.png","condition_area":65535' '"kind":"edit","prompt":"x","image":"/tmp/a.png","also":"/tmp/b.png","condition_area":1048577' '"kind":"edit","prompt":"x","image":"/tmp/a.png","also":"/tmp/b.png","condition_area":"768"'; do
  check "two pictures: refused 422: $bad" "$(po "{$bad}" | tail -c 4)" " 422"
done
check "two pictures: the refusals queued nothing" "$(wc -l < "$R/off/queue" | tr -d ' ')" "$Tq2"
check "two pictures: the job page lists the second picture and the size among the parameters" "$(cget /jobs/$TWI | grep -o '<dt>[a-z-]*</dt><dd>[^<]*</dd>' | sed 's/<dt>//; s#</dt><dd># #; s#</dd>##' | tr '\n' ';')" "kind edit;seed 3;also /tmp/b.png;condition-area 1048576;rewrite off;"
# A video that starts from a picture (docs/video-image.md): kind video with image
Vq=$(wc -l < "$R/off/queue" | tr -d ' ')
VI=$(pj '{"kind":"video","prompt":"the fox runs","image":"/tmp/fox.png","frames":17,"size":"640x352","seed":3,"steps":20}'); VII=$(echo "$VI" | jq -r .id)
check "image to video: the arguments media gets" "$(argv_of $VII)" "video -o $R/off/out/X --frames 17 --size 640x352 --seed 3 --steps 20 --image /tmp/fox.png -- the fox runs "
Vq2=$(wc -l < "$R/off/queue" | tr -d ' ')
for bad in '"kind":"image","prompt":"x","image":"/tmp/fox.png"' '"kind":"video","prompt":"x","image":5' '"kind":"video","prompt":"x","image":"/"' '"kind":"video","prompt":"x","image":"/tmp/fox.png","frames":1' '"kind":"video","prompt":"x","image":"/tmp/fox.png","rewrite":"on"' '"kind":"video","prompt":"x","steps":4'; do
  check "image to video: refused 422: $bad" "$(po "{$bad}" | tail -c 4)" " 422"
done
check "image to video: the refusals queued nothing" "$(wc -l < "$R/off/queue" | tr -d ' ')" "$Vq2"
check "image to video: the job page lists the picture among the parameters" "$(cget /jobs/$VII | grep -o '<dt>image</dt><dd>[^<]*</dd>')" "<dt>image</dt><dd>/tmp/fox.png</dd>"
check "image to video: the OpenAPI document says a video can take image" "$(jq -r '.components.schemas.NewJob.properties.image.description' studio/openapi.json | grep -c 'video')" 1
# the form
printf '\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00' > "$R/p1.jpg"; head -c 4000 /dev/urandom >> "$R/p1.jpg"
printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' > "$R/p2.png"; head -c 4000 /dev/urandom >> "$R/p2.png"
FM=$(curl -s -o /dev/null -w '%{redirect_url}' -H "$(hdr)" -F prompt=both -F "image=@$R/p1.jpg" -F "image2=@$R/p2.png" -F condition_area=1048576 localhost:18097/edit); FMI=${FM##*/}
check "edit form: two pictures make one job" "$(echo "$FM" | sed 's#.*/jobs/[0-9]*#ok#')" ok
check "edit form: ...and its arguments name both uploaded pictures, the second as -2, and the size" "$(argv_of $FMI | sed "s#$R/off#D#g")" "edit -o D/out/X --also D/upload-$FMI-2.png --condition-area 1048576 -- D/upload-$FMI.jpg both "
check "edit form: both pictures are on disk, nothing else" "$(ls "$R/off" | grep "^upload-$FMI" | tr '\n' ' ')" "upload-$FMI-2.png upload-$FMI.jpg "
check "edit form: no second picture (an empty file field) is one picture" "$(curl -s -o /dev/null -w '%{redirect_url}' -H "$(hdr)" -F prompt=one -F "image=@$R/p1.jpg" -F "image2=@/dev/null" localhost:18097/edit | sed 's#.*/jobs/[0-9]*#ok#')" ok
Fq=$(wc -l < "$R/off/queue" | tr -d ' '); Fu=$(ls "$R/off" | grep -c '^upload-')
check "edit form: a second file that is not a picture is refused: 422" "$(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -F prompt=bad -F "image=@$R/p1.jpg" -F "image2=@/etc/hosts" localhost:18097/edit)" 422
check "edit form: a condition_area that is not digits is refused: 422" "$(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -F prompt=bad -F "image=@$R/p1.jpg" -F "image2=@$R/p2.png" -F condition_area=big localhost:18097/edit)" 422
check "edit form: it offers the second picture and the size" "$(curl -s -H "$(hdr)" localhost:18097/edit | grep -o 'name=image2\|name=condition_area' | sort | tr '\n' ' ')" "name=condition_area name=image2 "
check "edit form: the two refused requests queued no job and saved no picture (the checks come before the files)" "$(wc -l < "$R/off/queue" | tr -d ' ') $(ls "$R/off" | grep -c '^upload-')" "$Fq $Fu"
# the pictures the person gave, on the job's page and from /files (studio/results.cho: input1.*, input2.*)
FT=$(curl -s -o /dev/null -D - -H "$(hdr)" localhost:18097/files/$FMI/input1.jpg | tr -d '\r' | grep -i '^content-type' | tr 'A-Z' 'a-z' | tr -d ' ')
check "inputs: the first picture is served unchanged, as image/jpeg" "$(curl -s -H "$(hdr)" localhost:18097/files/$FMI/input1.jpg | cmp - "$R/p1.jpg" && echo same) $FT" "same content-type:image/jpeg"
FT2=$(curl -s -o /dev/null -D - -H "$(hdr)" localhost:18097/files/$FMI/input2.png | tr -d '\r' | grep -i '^content-type' | tr 'A-Z' 'a-z' | tr -d ' ')
check "inputs: the second picture is served unchanged, as image/png" "$(curl -s -H "$(hdr)" localhost:18097/files/$FMI/input2.png | cmp - "$R/p2.png" && echo same) $FT2" "same content-type:image/png"
check "inputs: a type the upload did not have is 404 (input1.png, input2.webp)" "$(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" localhost:18097/files/$FMI/input1.png) $(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" localhost:18097/files/$FMI/input2.webp)" "404 404"
check "inputs: names off the list are 404 (input3.png, input1, a path)" "$(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" localhost:18097/files/$FMI/input3.png) $(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" localhost:18097/files/$FMI/input1) $(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" --path-as-is localhost:18097/files/$FMI/..%2Fupload-$FMI.jpg)" "404 404 404"
check "inputs: a job with no upload has no input (404)" "$(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" localhost:18097/files/$PEI/input1.jpg)" 404
check "inputs: the page of a job still waiting shows the two pictures and no result" "$(cget /jobs/$FMI | grep -o '<figure><a href=/files/[0-9]*/[a-z0-9.]*' | sed 's#.*/files/[0-9]*/##' | tr '\n' ' ')" "input1.jpg input2.png "
printf 'done\n' > "$R/off/$FMI.state"; mkdir -p "$R/off/out/$FMI"; head -c 3000 /dev/urandom > "$R/off/out/$FMI/edited.png"
check "inputs: a finished edit shows the first picture, the second and the result, in that order" "$(cget /jobs/$FMI | grep -o '<figure><a href=/files/[0-9]*/[a-z0-9.]*' | sed 's#.*/files/[0-9]*/##' | tr '\n' ' ')" "input1.jpg input2.png edited.png "
check "inputs: ...captioned Picture 1, Picture 2 and Result, and the result is not shown twice" "$(cget /jobs/$FMI | grep -o '<figcaption>[A-Za-z0-9 ]*</figcaption>' | sed 's#</*figcaption>##g' | tr '\n' ',') $(cget /jobs/$FMI | grep -c '<img alt=\"the edited picture\" src=')" "Picture 1,Picture 2,Result, 0"
check "inputs: a job made by the API (no upload) shows its result alone, as before" "$(cget /jobs/$PEI | grep -c 'class=trio') $(cget /jobs/$PEI | grep -c '<img alt=\"the edited picture\" src=')" "0 1"
check "inputs: the OpenAPI document lists the six names" "$(jq -r '.paths["/files/{id}/{name}"].get.parameters[] | select(.name=="name") | .schema.enum | map(select(startswith("input"))) | length' studio/openapi.json)" 6
# the cases page (studio/cases.cho): six forms that post to /edit with two pictures and a prepared instruction
CS=$(curl -s -H "$(hdr)" localhost:18097/cases)
check "cases: six forms, each to /edit as multipart" "$(echo "$CS" | grep -o '<form id=[a-z]* method=post action=/edit enctype=multipart/form-data>' | wc -l | tr -d ' ')" 6
check "cases: the six cases and the two batch forms ask for image and the instruction, the six cases for image2 too (and no script)" "$(echo "$CS" | grep -o 'name=image type=file\|name=image2 type=file\|<textarea name=prompt' | sort | uniq -c | tr -s ' ' | tr '\n' ';') $(echo "$CS" | grep -ci '<script')" " 8 <textarea name=prompt; 8 name=image type=file; 6 name=image2 type=file; 0"
check "cases: the instructions of the six cases and the two batch forms name image1 and image2" "$(echo "$CS" | grep -o '<textarea[^>]*>[^<]*</textarea>' | grep 'image1' | grep -c 'image2')" 8
check "cases: the header has the place and marks it open" "$(echo "$CS" | grep -o '<a href=/cases aria-current=page>Cases</a>' | wc -l | tr -d ' ')" 1
check "cases: the other pages link to it" "$(curl -s -H "$(hdr)" localhost:18097/edit | grep -c 'href=/cases')" 1
check "cases: each form (the six and the two batch ones) offers the optional sizes" "$(echo "$CS" | grep -o 'name=note maxlength=200' | wc -l | tr -d ' ')" 8
NF=$(curl -s -o /dev/null -w '%{redirect_url}' -H "$(hdr)" -F prompt=place -F "note=the sofa is 2.2 m wide" -F "image=@$R/p1.jpg" -F "image2=@$R/p2.png" localhost:18097/edit); NFI=${NF##*/}
check "cases: the sizes are added to the instruction as one argument" "$(argv_of $NFI | sed 's#.*\.jpg ##')" "place Real sizes: the sofa is 2.2 m wide. "
check "cases: sizes of 201 bytes are refused: 422" "$(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -F prompt=x --form-string "note=$(head -c 201 /dev/zero | tr '\0' a)" -F "image=@$R/p1.jpg" -F "image2=@$R/p2.png" localhost:18097/edit)" 422
check "cases: an instruction and sizes over 1000 bytes together are refused: 422" "$(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" --form-string "prompt=$(head -c 900 /dev/zero | tr '\0' a)" --form-string "note=$(head -c 150 /dev/zero | tr '\0' b)" -F "image=@$R/p1.jpg" -F "image2=@$R/p2.png" localhost:18097/edit)" 422
check "cases: sizes with a control character are refused: 422" "$(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -F prompt=x --form-string "note=$(printf 'a\001b')" -F "image=@$R/p1.jpg" -F "image2=@$R/p2.png" localhost:18097/edit)" 422
check "cases: only the try-on and the room cards carry a tip, above the instruction" "$(echo "$CS" | grep -o '<form id=[a-z]*\|class=tip\|<label>The instruction' | tr '\n' ' ' | sed 's#<form id=##g; s#class=tip#TIP#g; s#<label>The instruction#INSTR#g')" "tryon INSTR room INSTR tryon TIP INSTR room TIP INSTR object INSTR scene INSTR hair INSTR style INSTR "
check "cases: the try-on tip says to name the garments" "$(echo "$CS" | grep -o '<form id=tryon method.*' | grep -o 'class=tip[^>]*>[^<]*' | grep -c 'Say which garments to swap')" 1
check "cases: the room tip says to name what to replace" "$(echo "$CS" | grep -o '<form id=room method.*' | grep -o 'class=tip[^>]*>[^<]*' | head -1 | grep -c 'name what to replace')" 1
check "cases: a tip is outside the textarea, so the instruction posted is unchanged" "$(echo "$CS" | grep -o '<textarea[^>]*>[^<]*</textarea>' | grep -c 'Tip:')" 0
TRY=$(echo "$CS" | grep -o '<form id=tryon method.*' | grep -o '<textarea[^>]*>[^<]*' | head -1 | sed 's/<textarea[^>]*>//')
CF=$(curl -s -o /dev/null -w '%{redirect_url}' -H "$(hdr)" --form-string "prompt=$TRY" -F "image=@$R/p1.jpg" -F "image2=@$R/p2.png" localhost:18097/edit); CFI=${CF##*/}
check "cases: the try-on form's own instruction and two pictures make a two-picture edit" "$(argv_of $CFI | sed "s#$R/off#D#g" | cut -c1-110)" "edit -o D/out/X --also D/upload-$CFI-2.png -- D/upload-$CFI.jpg Dress the person in image1 in the clothing shown"
check "two pictures: the OpenAPI document names also and condition_area in the job and image2 in the form" "$(jq -r '.components.schemas.NewJob.properties | [has("also"), has("condition_area")] | join(",")' studio/openapi.json) $(grep -c '"image2"' studio/openapi.json)" "true,true 2"
# editing sessions (docs/studio-sessions.md): the prompt again, a result as the next input, .from and .root
D=$R/off
ED() { curl -s -o /dev/null -w '%{redirect_url}' -H "$(hdr)" "$@" localhost:18097/edit; }
SX=$(ED --form-string 'prompt=make it <b>red</b> & "bold"' -F seed=7 -F steps=4 -F model=qwen-image-2.1-turbo -F "image=@$R/pic.png"); SX=${SX##*/}
check "session: a job made by the form is its own session (.root) and its picture is an upload (.from)" "$(cat "$D/$SX.root") $(cat "$D/$SX.from")" "$SX upload"
AG=$(curl -s -H "$(hdr)" "localhost:18097/edit?again=$SX")
check "again: the prompt comes back, escaped, in the text area" "$(echo "$AG" | grep -o '<textarea[^>]*>[^<]*' | sed 's/<textarea[^>]*>//')" 'make it &lt;b&gt;red&lt;/b&gt; &amp; &quot;bold&quot;'
check "again: ...with the seed, the steps and the model" "$(echo "$AG" | grep -o 'name=seed[^>]*\|name=steps[^>]*\|name=model[^>]*' | grep -o 'value=[^ >]*' | tr '\n' ' ')" 'value=7 value=4 value="qwen-image-2.1-turbo" '
check "again: ...and a picture is still asked for (an upload or a choice from the gallery)" "$(echo "$AG" | grep -c 'name=image type=file') $(echo "$AG" | grep -c 'name=pick1')" "1 1"
check "again: an id that is not digits, or a job that does not exist, is ignored (the plain form)" "$(for q in again=abc again=999999 'again=1;2' from=abc from=999999; do curl -s -w '%{http_code}' -H "$(hdr)" "localhost:18097/edit?$q" | grep -c 'value=7'; done | tr '\n' ' ')" "0 0 0 0 0 "
NR=$(curl -s -H "$(hdr)" "localhost:18097/edit?from=$SX"); check "refine: a job that is not done is not offered as a picture (the plain form)" "$(echo "$NR" | grep -c 'name=from')" 0
check "refine: the job page offers neither refine nor a video's prompt, but the prompt again" "$(curl -s -H "$(hdr)" localhost:18097/jobs/$SX | grep -o '/edit?from=[0-9]*\|/edit?again=[0-9]*' | tr '\n' ' ')" "/edit?again=$SX "
mkdir -p "$D/out/$SX"; printf 'png-bytes' > "$D/out/$SX/edited.png"; echo done > "$D/$SX.state"
check "refine: a done job offers both on its page" "$(curl -s -H "$(hdr)" localhost:18097/jobs/$SX | grep -o '/edit?from=[0-9]*\|/edit?again=[0-9]*' | tr '\n' ' ')" "/edit?from=$SX /edit?again=$SX "
FR=$(curl -s -H "$(hdr)" "localhost:18097/edit?from=$SX")
check "refine: the form shows the result as the first picture and asks for no upload of it" "$(echo "$FR" | grep -c "src=/files/$SX/edited.png") $(echo "$FR" | grep -c 'name=from value='"$SX") $(echo "$FR" | grep -c 'name=image type=file') $(echo "$FR" | grep -c 'name=image2 type=file')" "1 1 0 1"
BEF=$(ls "$D" | wc -l | tr -d ' ')
RF=$(ED --form-string 'prompt=now blue' -F from=$SX); RFI=${RF##*/}
check "refine: a job is made and the person goes to it" "$(echo "$RF" | sed 's#.*/jobs/[0-9]*#ok#')" ok
check "refine: its first picture is the earlier result, named by the server, and nothing was copied" "$(argv_of $RFI | sed "s#$D#D#g") $(ls "$D" | grep -c "^upload-$RFI")" "edit -o D/out/X -- D/out/$SX/edited.png now blue  0"
check "refine: .from names the job and the file, .root is the session of the job it came from" "$(cat "$D/$RFI.from") $(cat "$D/$RFI.root")" "$SX edited.png $SX"
mkdir -p "$D/out/$RFI"; printf 'png2' > "$D/out/$RFI/edited.png"; echo done > "$D/$RFI.state"
R3=$(ED --form-string 'prompt=and green' -F from=$RFI); R3=${R3##*/}
check "refine: a third round stays in the first session" "$(cat "$D/$R3.root") $(cat "$D/$R3.from")" "$SX $RFI edited.png"
check "refine: the page of the refined job shows the earlier result as its first picture" "$(curl -s -H "$(hdr)" localhost:18097/jobs/$RFI | grep -c "src=/files/$SX/edited.png")" 1
check "refine: with a second picture uploaded, both are inputs and .from has two lines" "$(RT=$(ED --form-string 'prompt=combine' -F from=$SX -F "image2=@$R/pic.png"); RT=${RT##*/}; echo "$(wc -l < "$D/$RT.from" | tr -d ' ') $(argv_of $RT | grep -c -- '--also')")" "2 1"
VR=$(ED --form-string 'prompt=more' -F from=$SX -F variations=2); VRI=${VR##*/}
check "refine: variations are all in the session, with the same source" "$(cat "$D/$VRI.root") $(cat "$D/$((VRI + 1)).root") $(cat "$D/$((VRI + 1)).from")" "$SX $SX $SX edited.png"
check "refine: rewrite=ab (a pair) records both jobs" "$(PB=$(ED --form-string 'prompt=pair' -F from=$SX -F rewrite=ab); PB=${PB##*/}; echo "$(cat "$D/$PB.root") $(cat "$D/$((PB + 1)).root")")" "$SX $SX"
NOT=$(ED --form-string 'prompt=x' -F "image=@$R/pic.png"); NOT=${NOT##*/}
AFTER=$(ls "$D" | wc -l | tr -d ' ')
rf() { curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -F prompt=x "$@" localhost:18097/edit; }
mkdir -p "$D/out/$NOT"; echo failed > "$D/$NOT.state"
BEFR=$(ls "$D" | wc -l | tr -d ' ')
check "refine: refused (422): not digits, no such job, a failed job, a picture as well as from" "$(rf -F from=abc) $(rf -F from=999999) $(rf -F from=$NOT) $(rf -F from=$SX -F "image=@$R/pic.png")" "422 422 422 422"
check "refine: ...and none of those left a file behind" "$(ls "$D" | wc -l | tr -d ' ')" "$BEFR"
check "refine: a path is never taken from the client (from=../x is refused)" "$(rf -F 'from=../x')" 422
# the gallery (docs/studio-sessions.md, section 9): every picture, with a tag from the file it is
GI=$(po '{"kind":"image","prompt":"a harbour crane"}' | grep -o '"id":[0-9]*' | grep -o '[0-9]*'); mkdir -p "$D/out/$GI"; printf 'png' > "$D/out/$GI/image.png"; echo done > "$D/$GI.state"
printf 'png' > "$D/out/$NOT/edited.png"   # a failed job's file is never listed
gal() { # tag: the tiles of all pages carrying that tag, by counting their badges
  pg=1; n=0
  while :; do
    body=$(curl -s -H "$(hdr)" "localhost:18097/gallery?tag=$1&page=$pg")
    n=$((n + $(echo "$body" | grep -o "class=\"badge $1\"" | wc -l)))
    echo "$body" | grep -q 'Older</a>' || break
    pg=$((pg + 1))
  done
  echo $n
}
want_made() { c=0; for dd in "$D"/out/*/; do j=$(basename "$dd"); [ -f "$dd/$1" ] && [ "$(cat "$D/$j.state" 2>/dev/null)" = done ] && c=$((c + 1)); done; echo $c; }
check "gallery: uploaded tiles are the uploaded files" "$(gal uploaded)" "$(ls "$D" | grep -c '^upload-')"
check "gallery: edited tiles are the done jobs' edited.png (a failed job's file is not listed)" "$(gal edited)" "$(want_made edited.png)"
check "gallery: generated tiles are the done jobs' image.png" "$(gal generated)" "$(want_made image.png)"
check "gallery: ...and there is one generated picture, with its job's prompt" "$(curl -s -H "$(hdr)" 'localhost:18097/gallery?tag=generated' | grep -o 'class=prompt>a harbour crane' | wc -l | tr -d ' ')" 1
check "gallery: a result tile offers Refine, an uploaded tile does not" "$(curl -s -H "$(hdr)" 'localhost:18097/gallery?tag=edited' | grep -o 'Refine</a>' | wc -l | tr -d ' ') $(curl -s -H "$(hdr)" 'localhost:18097/gallery?tag=uploaded' | grep -o 'Refine</a>' | wc -l | tr -d ' ')" "$(want_made edited.png) 0"
check "gallery: a refined job's first picture (an earlier result) is not listed as an upload" "$(ls "$D" | grep -c "^upload-$RFI")" 0
check "gallery: a tile's picture is the file the job page serves" "$(curl -s -H "$(hdr)" 'localhost:18097/gallery?tag=generated' | grep -o "src=/files/$GI/image.png")" "src=/files/$GI/image.png"
check "gallery: the other pages link to it, and it marks itself open" "$(curl -s -H "$(hdr)" localhost:18097/edit | grep -c 'href=/gallery') $(curl -s -H "$(hdr)" localhost:18097/gallery | grep -c 'href=/gallery aria-current=page')" "1 1"
check "gallery: refused (422): an unknown tag, a page that is not digits, a page of 5 digits" "$(for q in tag=nope page=x page=12345 'tag=edited&page=1a'; do curl -s -o /dev/null -w '%{http_code} ' -H "$(hdr)" "localhost:18097/gallery?$q"; done)" "422 422 422 422 "
check "gallery: a page past the end is empty, not an error" "$(curl -s -w ' %{http_code}' -H "$(hdr)" 'localhost:18097/gallery?page=9999' | grep -o 'No pictures here yet\.\|[0-9]*$' | tr '\n' ' ')" "No pictures here yet. 200 "
check "gallery: no token: 401" "$(curl -s -o /dev/null -w '%{http_code}' localhost:18097/gallery)" 401
for k in $(seq 1 50); do ED --form-string "prompt=fill $k" -F "image=@$R/pic.png" >/dev/null; done
P1=$(curl -s -H "$(hdr)" 'localhost:18097/gallery?tag=uploaded'); P2=$(curl -s -H "$(hdr)" 'localhost:18097/gallery?tag=uploaded&page=2')
check "gallery: a page holds 48 tiles and offers Older, the next page offers Newer" "$(echo "$P1" | grep -o 'class=\"badge uploaded\"' | wc -l | tr -d ' ') $(echo "$P1" | grep -c 'Older</a>') $(echo "$P2" | grep -c 'Newer</a>')" "48 1 1"
check "gallery: the pages together hold every upload once" "$(gal uploaded)" "$(ls "$D" | grep -c '^upload-')"
check "gallery: newest first (the last upload leads the first page)" "$(echo "$P1" | grep -o 'src=/files/[0-9]*/input1' | head -1)" "src=/files/$(ls "$D" | grep -o '^upload-[0-9]*' | grep -o '[0-9]*' | sort -n | tail -1)/input1"
# sessions (docs/studio-sessions.md, section 10.2)
SL=$(cget /sessions | sed 's/ [0-9][0-9][0-9]$//')
SP=$(cget /sessions/$SX | sed 's/ [0-9][0-9][0-9]$//')
want_rounds=$(grep -lx "$SX" "$D"/*.root | wc -l | tr -d ' ')
check "sessions: the list has a card for the first job's session, with the number of rounds (the jobs with that root)" "$(echo "$SL" | grep -o "badge>[0-9]* rounds</span><span class=idn>session $SX<" | grep -o '[0-9]* rounds')" "$want_rounds rounds"
check "sessions: an edit made without a root (through the API) is a session of one round (its page; the list shows 60 and 50 uploads came before)" "$(cget /sessions/$E1 | grep -o 'Round [0-9]* &middot;\|[0-9]*$' | tr '\n' ' ')" "Round 1 &middot; 200 "
check "sessions: the list shows at most 60 sessions" "$(echo "$SL" | grep -o '<a class=job href=/sessions/' | wc -l | tr -d ' ')" 60
check "sessions: a video job is not a session" "$(echo "$SL" | grep -c "href=/sessions/$W2>") $(cget /sessions/$W2 | grep -o '[0-9]*$')" "0 404"
check "sessions: refused: not digits 422, no such session 404, no token 401" "$(cget /sessions/abc | grep -o '[0-9]*$') $(cget /sessions/999999 | grep -o '[0-9]*$') $(curl -s -o /dev/null -w '%{http_code}' localhost:18097/sessions/$SX)" "422 404 401"
check "sessions: the page lists the rounds oldest first, with the refine indented under the round it came from" "$(echo "$SP" | grep -o "margin-left:[0-9]*px\"><div class=row><h2><a href=/jobs/$SX>Round 1\|margin-left:[0-9]*px\"><div class=row><h2><a href=/jobs/$RFI>Round [0-9]*\|margin-left:[0-9]*px\"><div class=row><h2><a href=/jobs/$R3>Round [0-9]*" | sed 's/"><div.*>Round / /' | tr '\n' ';')" "margin-left:0px 1;margin-left:24px 2;margin-left:48px 3;"
check "sessions: a round shows its pictures and its result (the refine's first picture is the earlier result)" "$(echo "$SP" | grep -c "src=/files/$SX/edited.png") $(echo "$SP" | grep -c "src=/files/$RFI/edited.png")" "1 1"
check "sessions: the refine form offers the finished results, the newest checked, and posts to /edit" "$(echo "$SP" | grep -o 'name=from value=[0-9]*\( checked\)\?>' | tr '\n' ';') $(echo "$SP" | grep -c 'action=/edit enctype=multipart/form-data')" "name=from value=$SX>;name=from value=$RFI checked>; 1"
check "sessions: the header marks the Library and its Sessions tab" "$(echo "$SP" | grep -o '<a href=[^>]*aria-current=page>' | tr '\n' ' ')" "<a href=/gallery aria-current=page> <a href=/sessions aria-current=page> "
check "sessions: the other pages link to the list (the Library's tabs)" "$(cget /gallery | grep -c 'href=/sessions')" 1
# the library and the picker (docs/studio-sessions.md, sections 10.3 and 10.4)
GP() { curl -s -o /dev/null -w '%{http_code} %{redirect_url}' -H "$(hdr)" "$@" localhost:18097/gallery; }
check "library: a picture is kept and the person goes back to the gallery" "$(GP -F "image=@$R/pic.png" | sed 's#//[^/]*/#//H/#')" "302 http://H/gallery"
check "library: the file and its line in the index" "$(cmp "$D/lib-1.png" "$R/pic.png" && echo same) $(cat "$D/lib.index")" "same 1 png"
check "library: a JPEG gets .jpg and the next number (by the first bytes, not the name)" "$(GP -F "image=@$R/p1.jpg;filename=whatever.png" >/dev/null; ls "$D" | grep '^lib-' | tr '\n' ' ')" "lib-1.png lib-2.jpg "
check "library: refused: not a picture 422, no picture 422, JSON (passes the gate, the handler refuses) 422, no token 401, a foreign page 403" "$(printf 'text, not a picture, long enough' > "$R/t.txt"; GP -F "image=@$R/t.txt" | cut -c1-3) $(GP -F other=x | cut -c1-3) $(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -H 'Content-Type: application/json' -XPOST -d '{}' localhost:18097/gallery) $(curl -s -o /dev/null -w '%{http_code}' -F "image=@$R/pic.png" localhost:18097/gallery) $(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -H 'Origin: http://evil.example' -F "image=@$R/pic.png" localhost:18097/gallery)" "422 422 422 401 403"
check "library: ...and none of those kept a file" "$(ls "$D" | grep -c '^lib-')" 2
check "library: it is served as a picture, byte for byte" "$(curl -s -H "$(hdr)" localhost:18097/files/L1/picture.png | cmp - "$R/pic.png" && echo same) $(curl -s -D - -o /dev/null -H "$(hdr)" localhost:18097/files/L2/picture.jpg | tr -d '\r' | grep -i '^content-type' | tr 'A-Z' 'a-z')" "same content-type: image/jpeg"
check "library: wrong extension, no such picture, another name, a bad id: 404 404 404 422" "$(for u in L1/picture.jpg L99/picture.png L1/evil Lx/picture.png; do curl -s -o /dev/null -w '%{http_code} ' -H "$(hdr)" localhost:18097/files/$u; done)" "404 404 404 422 "
GL=$(curl -s -H "$(hdr)" 'localhost:18097/gallery?tag=uploaded')
check "library: the gallery lists it, tagged uploaded, first, with Edit this" "$(echo "$GL" | grep -o 'src=/files/L[0-9]*/picture\.[a-z]*' | tr '\n' ' ') $(echo "$GL" | grep -o '/edit?pick1=l/L[0-9]*' | tr '\n' ' ')" "src=/files/L2/picture.jpg src=/files/L1/picture.png  /edit?pick1=l/L2 /edit?pick1=l/L1 "
check "library: the gallery has the form to add one" "$(echo "$GL" | grep -c 'action=/gallery enctype=multipart/form-data')" 1
LASTUP=$(ls "$D" | grep -o '^upload-[0-9]*\.png' | grep -o '[0-9]*' | sort -n | tail -1)
check "picker: the edit form offers the library's and the newest job's pictures, for both slots" "$(curl -s -H "$(hdr)" localhost:18097/edit | grep -o 'name=pick[12] value=\"\(l/L1\|j/'"$LASTUP"'/input1.png\)\"' | sort | uniq -c | tr -s ' ' | tr '\n' ';')" " 1 name=pick1 value=\"j/$LASTUP/input1.png\"; 1 name=pick1 value=\"l/L1\"; 1 name=pick2 value=\"j/$LASTUP/input1.png\"; 1 name=pick2 value=\"l/L1\";"
check "picker: the file inputs are no longer required, and the choice is a closed details" "$(curl -s -H "$(hdr)" localhost:18097/edit | grep -c 'type=file[^>]* required') $(curl -s -H "$(hdr)" localhost:18097/edit | grep -o '<summary>or choose from the gallery</summary>' | wc -l | tr -d ' ')" "0 2"
check "picker: each case has two pickers (12), each batch form one for its first picture (2 more)" "$(curl -s -H "$(hdr)" localhost:18097/cases | grep -o '<summary>or choose from the gallery</summary>' | wc -l | tr -d ' ') $(curl -s -H "$(hdr)" localhost:18097/cases | grep -c 'type=file[^>]* required')" "14 0"
BEF=$(ls "$D" | wc -l | tr -d ' ')
PL=$(ED --form-string 'prompt=from the library' -F pick1=l/L1); PLI=${PL##*/}
check "picker: a library picture is the first picture: its path in the job, nothing copied" "$(argv_of $PLI | sed "s#$D#D#g") $(ls "$D" | grep -c "^upload-$PLI")" "edit -o D/out/X -- D/lib-1.png from the library  0"
check "picker: ...and .from names it, a library picture starts a session" "$(cat "$D/$PLI.from") / $(cat "$D/$PLI.root")" "L1 picture.png / $PLI"
check "picker: the job's page shows it as the first picture" "$(curl -s -H "$(hdr)" localhost:18097/jobs/$PLI | grep -o 'src=/files/L1/picture.png\|Picture 1 (from the gallery)' | tr '\n' ' ')" "src=/files/L1/picture.png Picture 1 (from the gallery) "
PR2=$(ED --form-string 'prompt=a result chosen' -F pick1=j/$SX/edited.png); PR2I=${PR2##*/}
check "picker: a job's result chosen stays in that session, like from" "$(cat "$D/$PR2I.from") / $(cat "$D/$PR2I.root")" "$SX edited.png / $SX"
PU=$(ED --form-string 'prompt=an upload chosen' -F pick1=j/$SX/input1.png); PUI=${PU##*/}
check "picker: a picture a job was given can be chosen (its upload file, a new session)" "$(argv_of $PUI | sed "s#$D#D#g") / $(cat "$D/$PUI.from") / $(cat "$D/$PUI.root" | sed "s/^$PUI$/self/")" "edit -o D/out/X -- D/upload-$SX.png an upload chosen  / $SX input1.png / self"
PT=$(ED --form-string 'prompt=two chosen' -F pick1=l/L1 -F pick2=l/L2); PTI=${PT##*/}
check "picker: both slots by choice: --also is the second, and .from has two lines" "$(argv_of $PTI | sed "s#$D#D#g" | cut -c1-60) / $(wc -l < "$D/$PTI.from" | tr -d ' ')" "edit -o D/out/X --also D/lib-2.jpg -- D/lib-1.png two chosen / 2"
PW=$(ED --form-string 'prompt=upload wins' -F "image=@$R/pic.png" -F pick1=l/L1); PWI=${PW##*/}
check "picker: an upload of the same slot wins over a choice" "$(argv_of $PWI | sed "s#$D#D#g" | cut -c1-50) / $(cat "$D/$PWI.from")" "edit -o D/out/X -- D/upload-$PWI.png upload wins  / upload"
AFT=$(ls "$D" | wc -l | tr -d ' ')
pk() { curl -s -o /dev/null -w '%{http_code} ' -H "$(hdr)" -F prompt=x "$@" localhost:18097/edit; }
check "picker: refused (422): a failed job's picture, a name outside the list, a path, a library picture that is not there, a video job's, junk, and neither upload nor choice" "$(pk -F pick1=j/$NOT/edited.png; pk -F pick1=j/$SX/input3.png; pk -F 'pick1=j/../x/edited.png'; pk -F pick1=l/L999; pk -F pick1=j/$W2/edited.png; pk -F pick1=junk; pk -F pick2=l/L999 -F pick1=l/L1; curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -F prompt=x localhost:18097/edit)" "422 422 422 422 422 422 422 422"
check "picker: ...and none of those left a file behind" "$(ls "$D" | wc -l | tr -d ' ')" "$AFT"
FP=$(curl -s -H "$(hdr)" 'localhost:18097/edit?pick1=l/L1')
check "picker: /edit?pick1= shows the picture as the first picture and carries the choice; a bad choice is the plain form" "$(echo "$FP" | grep -o 'src=/files/L1/picture.png\|name=pick1 value=\"l/L1\">' | tr '\n' ' ') $(curl -s -H "$(hdr)" 'localhost:18097/edit?pick1=l/L999' | grep -c 'type=hidden name=pick1')" "src=/files/L1/picture.png name=pick1 value=\"l/L1\"> src=/files/L1/picture.png  0"
check "picker: the OpenAPI document names pick1, pick2 and POST /gallery" "$(jq -r '[.components.schemas // {} | tostring | length > 0] | .[0]' studio/openapi.json >/dev/null; grep -c '"pick1"' studio/openapi.json) $(grep -c '"pick2"' studio/openapi.json) $(jq -r '.paths["/gallery"] | has("post")' studio/openapi.json)" "2 2 true"
# a new line in a text area (a browser sends CR LF): accepted, written as one space (the refine form of a session failed with "the prompt has no control characters")
NLP=$(ED --form-string "prompt=$(printf 'line one\r\nline two\r\n\r\n\tline three\r\n')" -F from=$SX); NLI=${NLP##*/}
check "newlines: a prompt typed over several lines is accepted and each run of CR, LF and tab is one space (no trailing one)" "$(argv_of $NLI | sed "s#$D#D#g")" "edit -o D/out/X -- D/out/$SX/edited.png line one line two line three "
check "newlines: ...also in the note of a case, and in a new upload's prompt" "$(NN=$(ED --form-string "prompt=$(printf 'a\nb')" --form-string "note=$(printf 'the sofa\r\nis 2 m')" -F "image=@$R/pic.png"); argv_of ${NN##*/} | sed 's#.*\.png ##')" "a b Real sizes: the sofa is 2 m. "
check "newlines: a prompt of only new lines is refused (422) and any other control character still is" "$(curl -s -o /dev/null -w '%{http_code} ' -H "$(hdr)" --form-string "prompt=$(printf '\r\n')" -F from=$SX localhost:18097/edit; curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" --form-string "prompt=$(printf 'a\001b')" -F from=$SX localhost:18097/edit)" "422 422"
check "newlines: the session page's refine form (a text area) posts a prompt over lines and refines" "$(RN=$(curl -s -o /dev/null -w '%{redirect_url}' -H "$(hdr)" --form-string "prompt=$(printf 'warmer\r\nand brighter')" -F from=$RFI -F seed= localhost:18097/edit); argv_of ${RN##*/} | sed 's#.*\.png ##')" "warmer and brighter "
# one picture and several others, one edit each (docs/studio-sessions.md, section 14)
BT() { curl -s -o /dev/null -w '%{http_code} %{redirect_url}' -H "$(hdr)" "$@" localhost:18097/batch; }
Q0=$(wc -l < "$D/queue" | tr -d ' ')
BR=$(BT --form-string 'prompt=try these' -F seed=5 -F "image=@$R/pic.png" -F "image2=@$R/p1.jpg" -F "image3=@$R/p2.png" -F "image4=@$R/pic.png"); BF=${BR##*/}
check "batch: three garments make three jobs and go to the session of the first" "$(echo "$BR" | sed 's#//[^/]*/#//H/#; s#/sessions/[0-9]*#/sessions/FIRST#') $(( $(wc -l < "$D/queue" | tr -d ' ') - Q0 ))" "302 http://H/sessions/FIRST 3"
check "batch: each job is the first picture and its own garment, the same seed and instruction (the first picture is kept once)" "$(for j in $BF $((BF + 1)) $((BF + 2)); do argv_of $j | sed "s#$D#D#g"; echo; done | tr '\n' '|') $(ls "$D" | grep -c "^upload-$BF\.png")" "edit -o D/out/X --seed 5 --also D/upload-$BF-2.jpg -- D/upload-$BF.png try these |edit -o D/out/X --seed 5 --also D/upload-$((BF + 1))-2.png -- D/upload-$BF.png try these |edit -o D/out/X --seed 5 --also D/upload-$((BF + 2))-2.png -- D/upload-$BF.png try these | 1"
check "batch: one session, and .from names the first picture then the garment" "$(for j in $BF $((BF + 1)) $((BF + 2)); do printf '%s:%s:%s ' "$(cat "$D/$j.root")" "$(head -1 "$D/$j.from" | tr ' ' _)" "$(tail -1 "$D/$j.from")"; done)" "$BF:upload:upload $BF:${BF}_input1.png:upload $BF:${BF}_input1.png:upload "
BS=$(cget /sessions/$BF | sed 's/ [0-9][0-9][0-9]$//')
check "batch: the session page shows three rounds, the first picture each time, and three garments" "$(echo "$BS" | grep -o 'Round [0-9]* &middot;' | tr '\n' ' ') $(echo "$BS" | grep -o "src=/files/$BF/input1.png" | wc -l | tr -d ' ') $(echo "$BS" | grep -o 'src=/files/[0-9]*/input2\.[a-z]*' | wc -l | tr -d ' ')" "Round 1 &middot; Round 2 &middot; Round 3 &middot;  3 3"
QB=$(wc -l < "$D/queue" | tr -d ' ')
BC=$(BT --form-string "prompt=$(printf 'two\r\nlines')" -F pick1=l/L1 -F "image2=@$R/p1.jpg" -F pick3=j/$SX/edited.png -F pick4=l/L2); BCF=${BC##*/}
check "batch: the first picture and the others can be chosen from the gallery (no copy of a chosen one)" "$(argv_of $BCF | sed "s#$D#D#g") | $(argv_of $((BCF + 1)) | sed "s#$D#D#g") | $(argv_of $((BCF + 2)) | sed "s#$D#D#g") | $(ls "$D" | grep -c "^upload-$BCF\.") $(( $(wc -l < "$D/queue" | tr -d ' ') - QB ))" "edit -o D/out/X --also D/upload-$BCF-2.jpg -- D/lib-1.png two lines  | edit -o D/out/X --also D/out/$SX/edited.png -- D/lib-1.png two lines  | edit -o D/out/X --also D/lib-2.jpg -- D/lib-1.png two lines  | 0 3"
check "batch: ...with .from naming each chosen picture" "$(cat "$D/$BCF.from" "$D/$((BCF + 1)).from" "$D/$((BCF + 2)).from" | tr ' \n' '_|')" "L1_picture.png|upload|L1_picture.png|${SX}_edited.png|L1_picture.png|L2_picture.jpg|"
AFB=$(ls "$D" | wc -l | tr -d ' '); QA=$(wc -l < "$D/queue" | tr -d ' ')
check "batch: refused (422): no first picture, no other picture, a garment that is not a picture, a bad choice, an empty prompt" "$(BT -F prompt=x -F "image2=@$R/p1.jpg" | cut -c1-3) $(BT -F prompt=x -F "image=@$R/pic.png" | cut -c1-3) $(BT -F prompt=x -F "image=@$R/pic.png" -F "image2=@$R/t.txt" | cut -c1-3) $(BT -F prompt=x -F "image=@$R/pic.png" -F pick2=j/$NOT/edited.png | cut -c1-3) $(BT -F prompt= -F "image=@$R/pic.png" -F "image2=@$R/p1.jpg" | cut -c1-3)" "422 422 422 422 422"
check "batch: ...and a good garment next to a bad one makes nothing (all or none); none of those left a file or a job" "$(BT -F prompt=x -F "image=@$R/pic.png" -F "image2=@$R/p1.jpg" -F "image3=@$R/t.txt" | cut -c1-3) $(ls "$D" | wc -l | tr -d ' ') $(wc -l < "$D/queue" | tr -d ' ')" "422 $AFB $QA"
check "batch: JSON passes the gate and is refused (422), no token 401, a foreign page 403" "$(curl -s -o /dev/null -w '%{http_code} ' -H "$(hdr)" -H 'Content-Type: application/json' -XPOST -d '{}' localhost:18097/batch; curl -s -o /dev/null -w '%{http_code} ' -F prompt=x localhost:18097/batch; curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -H 'Origin: http://evil.example' -F prompt=x localhost:18097/batch)" "422 401 403"
CB=$(curl -s -H "$(hdr)" localhost:18097/cases)
check "batch: Cases has the two forms, each posting to /batch with a first picture, a grid to tick and a multiple-file input" "$(echo "$CB" | grep -o '<form id=[a-z-]*-many method=post action=/batch enctype=multipart/form-data>' | wc -l | tr -d ' ') $(echo "$CB" | grep -o 'name=images type=file multiple' | wc -l | tr -d ' ') $(echo "$CB" | grep -o '<summary>tick the pictures to use, from the gallery</summary>' | wc -l | tr -d ' ') $(echo "$CB" | grep -o 'type=checkbox name=pickm' | wc -l | tr -d ' ' | sed 's/^[1-9][0-9]*$/some/')" "2 2 2 some"
check "batch: the instruction of each is the case's own" "$(echo "$CB" | grep -o '<form id=tryon-many.*' | grep -o '<textarea[^>]*>[^<]*' | head -1 | grep -c 'Dress the person in image1') $(echo "$CB" | grep -o '<form id=room-many.*' | grep -o '<textarea[^>]*>[^<]*' | head -1 | grep -c 'Redecorate the room in image1')" "1 1"
check "batch: the OpenAPI document has POST /batch" "$(jq -r '.paths["/batch"] | has("post")' studio/openapi.json)" true
# descriptions, search, and the grid with several files (docs/studio-sessions.md, section 15)
LB() { curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" "$@" localhost:18097/gallery; }
LD() { LB --form-string "description=$2" -F "image=@$1"; }
check "description: a picture is added with a description, kept as lib-N.txt before its line in the index" "$(LD "$R/pic.png" 'Green quilted jacket, size M') $(cat "$D/lib-3.txt") $(tail -1 "$D/lib.index")" "302 Green quilted jacket, size M 3 png"
check "description: ...a JPEG with another, one with markup, one with none" "$(LD "$R/p1.jpg" 'blue denim jeans') $(LD "$R/pic.png" 'red <b>coat</b> & "hat"') $(LB -F "image=@$R/pic.png") $(ls "$D" | grep -c '^lib-[0-9]*\.txt$')" "302 302 302 3"
check "description: more than 200 bytes, or a control character, is refused (422) and nothing is kept" "$(LD "$R/pic.png" "$(head -c 201 /dev/zero | tr '\0' a)") $(LD "$R/pic.png" "$(printf 'a\001b')") $(ls "$D" | grep -c '^lib-')" "422 422 9"
GD=$(curl -s -H "$(hdr)" 'localhost:18097/gallery?tag=uploaded')
check "description: the library tile shows it, escaped; a picture without one says it was added" "$(echo "$GD" | grep -c 'Green quilted jacket, size M') $(echo "$GD" | grep -c 'red &lt;b&gt;coat&lt;/b&gt; &amp; &quot;hat&quot;') $(echo "$GD" | grep -c 'red <b>coat'; true) $(echo "$GD" | grep -c 'Added to the library')" "1 1 0 1"
sq() { curl -s -H "$(hdr)" "localhost:18097/gallery?$1"; }
check "search: ?q= finds a library picture by its description, ignoring case, with + or %20 for a space" "$(for q in jacket JACKET 'quilted+jacket' 'quilted%20Jacket'; do sq "q=$q" | grep -o 'src=/files/L[0-9]*/picture.png' | tr '\n' ' '; echo; done | sort -u | tr '\n' '|')" "src=/files/L3/picture.png |"
check "search: ...and a job's picture by the instruction it was made with" "$(sq 'q=fill%2017' | grep -o 'badge uploaded\"' | wc -l | tr -d ' ') $(sq 'q=fill%2017' | grep -o 'job 1[0-9]*</span>' | wc -l | tr -d ' ') $(sq 'q=fill%2017' | grep -c 'fill 17')" "1 1 1"
check "search: no match is an empty page, not an error; the words are escaped; the form keeps them and the tag" "$(sq 'q=zzzznothing' | grep -c 'No pictures here yet') $(sq 'q=%3Cscript%3Ex' | grep -c '<script>x') $(sq 'q=%3Cscript%3Ex' | grep -c 'value=\"&lt;script&gt;x\"') $(sq 'tag=edited&q=bold' | grep -c 'name=tag value=edited')" "1 0 1 1"
SF1=$(sq 'q=fill'); SF2=$(sq 'q=fill&page=2')
check "search: a search with many hits pages, and the links keep the words (and the tag)" "$(echo "$SF1" | grep -o 'class=\"badge uploaded\"' | wc -l | tr -d ' ') $(echo "$SF1" | grep -o 'href=\"/gallery?q=fill&amp;page=2\"' | wc -l | tr -d ' ') $(echo "$SF2" | grep -o 'class=\"badge uploaded\"' | wc -l | tr -d ' ') $(echo "$SF2" | grep -o 'href=\"/gallery?q=fill&amp;page=1\"' | wc -l | tr -d ' ')" "48 1 2 1"
check "search: the pickers of /cases take the same words (a description finds the library's jacket and not the jeans)" "$(curl -s -H "$(hdr)" 'localhost:18097/cases?q=jacket' | grep -o 'value=\"l/L[0-9]*\"' | sort -u | tr '\n' ' ') $(curl -s -H "$(hdr)" 'localhost:18097/cases?q=jacket' | grep -c 'name=q maxlength=100 value=\"jacket\"')" 'value="l/L3"  1'
# the grid: several ticked and several files in one request
Q1=$(wc -l < "$D/queue" | tr -d ' ')
GR=$(BT --form-string 'prompt=grid' -F "image=@$R/pic.png" -F pickm=l/L3 -F pickm=l/L4 -F "images=@$R/p1.jpg" -F "images=@$R/p2.png"); GRF=${GR##*/}
check "grid: two ticked and two files make four jobs, the ticked first, in one session" "$(for j in $GRF $((GRF + 1)) $((GRF + 2)) $((GRF + 3)); do argv_of $j | sed "s#$D#D#g; s#.*--also ##; s# -- .*##"; echo; done | tr '\n' '|') $(( $(wc -l < "$D/queue" | tr -d ' ') - Q1 )) $(cat "$D/$((GRF + 3)).root")" "D/lib-3.png|D/lib-4.jpg|D/upload-$((GRF + 2))-2.jpg|D/upload-$((GRF + 3))-2.png| 4 $GRF"
check "grid: .from names each picture" "$(cat "$D/$GRF.from" "$D/$((GRF + 1)).from" "$D/$((GRF + 3)).from" | tr ' \n' '_|')" "upload|L3_picture.png|${GRF}_input1.png|L4_picture.jpg|${GRF}_input1.png|upload|"
Q2=$(wc -l < "$D/queue" | tr -d ' '); AF2=$(ls "$D" | wc -l | tr -d ' ')
check "grid: an empty file input (a browser sends one when no file is chosen) adds nothing; one tick alone makes one job" "$(BT --form-string 'prompt=one' -F "image=@$R/pic.png" -F pickm=l/L3 -F "images=@/dev/null" | cut -c1-3) $(( $(wc -l < "$D/queue" | tr -d ' ') - Q2 ))" "302 1"
Q3=$(wc -l < "$D/queue" | tr -d ' '); AF3=$(ls "$D" | wc -l | tr -d ' ')
check "grid: six pictures are refused (422), and so is a bad tick next to good ones; nothing is made" "$(BT --form-string 'prompt=six' -F "image=@$R/pic.png" -F pickm=l/L3 -F pickm=l/L4 -F pickm=l/L1 -F "images=@$R/p1.jpg" -F "images=@$R/p2.png" -F "images=@$R/pic.png" | cut -c1-3) $(BT --form-string 'prompt=bad' -F "image=@$R/pic.png" -F pickm=l/L3 -F pickm=junk | cut -c1-3) $(( $(wc -l < "$D/queue" | tr -d ' ') - Q3 )) $(( $(ls "$D" | wc -l | tr -d ' ') - AF3 ))" "422 422 0 0"
check "grid: the OpenAPI document has q, description, pickm and images" "$(grep -c '"name": "q"' studio/openapi.json) $(grep -c '"description": { "type": "string", "maxLength": 200' studio/openapi.json) $(grep -c '"pickm"' studio/openapi.json) $(grep -c '"images": { "type": "string", "format": "binary"' studio/openapi.json)" "2 1 1 1"
# a description the describer wrote says (auto); one a person typed does not (docs/studio-sessions.md, section 16)
: > "$D/lib-3.auto"
check "auto: a description written by the describer says (auto), a typed one does not" "$(curl -s -H "$(hdr)" 'localhost:18097/gallery?q=quilted' | grep -o '(auto)' | wc -l | tr -d ' ') $(curl -s -H "$(hdr)" 'localhost:18097/gallery?q=denim' | grep -o '(auto)' | wc -l | tr -d ' ')" "1 0"
# synonyms and words in any order (docs/studio-sessions.md, section 17)
lastlib() { tail -1 "$D/lib.index" | cut -d' ' -f1; }
SYB=$(LD "$R/pic.png" 'blue bicycle with a wicker basket' >/dev/null; lastlib); SYS=$(LD "$R/pic.png" 'beige sneakers against a white wall' >/dev/null; lastlib); SYC=$(LD "$R/pic.png" 'wooden dining table and a chandelier' >/dev/null; lastlib); SYF=$(LD "$R/pic.png" 'grey sofa and a rug' >/dev/null; lastlib); SYD=$(LD "$R/pic.png" 'woman in a red dress' >/dev/null; lastlib)
libhits() { sq "q=$(printf '%s' "$1" | sed 's/ /%20/g')" | grep -o 'src=/files/L[0-9]*/picture' | grep -o 'L[0-9]*' | tr '\n' ' '; }
check "synonyms: bike finds the bicycle, shoes the sneakers, lamp the chandelier, couch the sofa" "$(libhits bike)/$(libhits shoes)/$(libhits lamp)/$(libhits couch)" "L$SYB /L$SYS /L$SYC /L$SYF "
check "synonyms: ...either way round, plural or not, in capitals, and with other members of the group" "$(libhits bicycle)/$(libhits Sneaker)/$(libhits TRAINERS)/$(libhits sofas)/$(libhits chandeliers)" "L$SYB /L$SYS /L$SYS /L$SYF /L$SYC "
check "synonyms: a word of no group is only itself, and a synonym does not find an unrelated picture" "$(libhits zebra)/$(libhits bike | grep -c "L$SYF")/$(libhits lamp | grep -c "L$SYB")" "/0/0"
check "synonyms: jeans and denim find the denim jeans, and a jacket is not a coat (the list holds true synonyms only)" "$(libhits denim)$(libhits jeans) $(libhits coat)" "L4 L4  L5 "
SYH=$(LD "$R/pic.png" 'that landscape with a chat room' >/dev/null; lastlib); SYW=$(LD "$R/pic.png" 'a woolly hat on a bench' >/dev/null; lastlib)
check "synonyms: a synonym is a whole word (beanie finds both pictures that have the whole word hat, and not the hat inside that or the cap inside landscape); a word you type still matches inside words, as before" "$(libhits beanie)/$(libhits beanie | grep -c "L$SYH")/$(libhits hat | grep -c "L$SYH")" "L$SYW L5 /0/1"
check "words: several words in any order each have to be there (and an order that was a phrase still works)" "$(libhits 'jacket green')/$(libhits 'dress red')/$(libhits 'red dress')/$(libhits 'green bicycle')/$(libhits 'sofa rug')" "L3 /L$SYD /L$SYD //L$SYF "
check "words: spaces around and between words do not matter" "$(libhits '++sofa++rug++')/$(libhits 'sofa%20%20rug')" "L$SYF /L$SYF "
check "synonyms: the search of a job's picture by its instruction uses them too (an instruction with fill finds fills; padded with synonyms it finds nothing else)" "$(sq 'q=fill%2017' | grep -o 'badge uploaded\"' | wc -l | tr -d ' ') $(sq 'q=bicycle%20fill' | grep -o 'badge uploaded\"' | wc -l | tr -d ' ')" "1 0"
check "synonyms: /cases pickers use them (bike shows the bicycle, and not the sofa)" "$(curl -s -H "$(hdr)" 'localhost:18097/cases?q=bike' | grep -o "value=\"l/L$SYB\"" | head -1) $(curl -s -H "$(hdr)" 'localhost:18097/cases?q=bike' | grep -c "value=\"l/L$SYF\"")" "value=\"l/L$SYB\" 0"
# deleting a library picture (docs/studio-sessions.md, section 18)
DL() { curl -s -o /dev/null -w '%{http_code} %{redirect_url}' -H "$(hdr)" -XPOST -H 'Content-Type: application/x-www-form-urlencoded' "$@"; }
LD "$R/pic.png" 'a picture to delete' >/dev/null; DX=$(lastlib)
: > "$D/lib-$DX.auto"
check "delete: refused while a job is queued (409), and nothing is removed" "$(DL localhost:18097/gallery/delete/$DX | cut -c1-3) $(ls "$D" | grep -c "^lib-$DX\.")" "409 3"
mv "$D/queue" "$D/queue.hold"
check "delete: the page offers it for a library picture only (once for each, behind a confirmation)" "$(sq "q=to%20delete" | grep -c "action=/gallery/delete/$DX ") $(sq 'tag=generated' | grep -c 'gallery/delete/')" "1 0"
check "delete: the picture, its description and its marks go; the person returns to the gallery" "$(DL localhost:18097/gallery/delete/$DX | sed 's#//[^/]*/#//H/#') $(ls "$D" | grep "^lib-$DX\." | tr '\n' ' ')" "302 http://H/gallery lib-$DX.tried "
check "delete: it is gone from the gallery, the picker and the file route; the index keeps its line, so the number is not given again" "$(sq 'q=to%20delete' | grep -c "L$DX/") $(curl -s -H "$(hdr)" localhost:18097/cases | grep -c "l/L$DX\"") $(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" localhost:18097/files/L$DX/picture.png) $(grep -c "^$DX " "$D/lib.index") $(LD "$R/pic.png" next >/dev/null; lastlib | awk -v n=$DX '{print $1 - n}')" "0 0 404 1 1"
check "delete: a number twice 404, no such 404, not digits 422, GET 404, JSON 415 or 404, no token 401, a foreign page 403" "$(DL localhost:18097/gallery/delete/$DX | cut -c1-3) $(DL localhost:18097/gallery/delete/9999 | cut -c1-3) $(DL localhost:18097/gallery/delete/x1 | cut -c1-3) $(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" localhost:18097/gallery/delete/$DX) $(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -XPOST -H 'Content-Type: text/plain' localhost:18097/gallery/delete/1) $(curl -s -o /dev/null -w '%{http_code}' -XPOST -H 'Content-Type: application/x-www-form-urlencoded' localhost:18097/gallery/delete/1) $(DL -H 'Origin: http://evil.example' localhost:18097/gallery/delete/1 | cut -c1-3)" "404 404 422 404 415 401 403"
mv "$D/queue.hold" "$D/queue"
exit $fail
