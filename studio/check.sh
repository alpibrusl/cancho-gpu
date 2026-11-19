#!/bin/sh
# Builds the studio's programs and checks the worker against a fake script (no GPU): the arguments reach the script as data, its output lands in `log`, `state` says
# done or failed N, a missing argv file does nothing. CANCHO names the compiler; PK the packages directory of a cancho checkout.
set -e
cd "$(dirname "$0")/.."
PK="${PK:-../cancho/packages}"
C="${CANCHO:-cancho}"
mkdir -p build
$C build studio/studio.cho studio/files.cho studio/jobs.cho studio/results.cho studio/auth.cho studio/range.cho studio/upload.cho studio/compare.cho studio/sets.cho studio/cases.cho "$PK"/http-server/server.cho --std -o build/studio 2>&1 | grep -v "^ld: warning" || true
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
check "site: / is index.html, with the stylesheet, the header and the frame of recent jobs added and nothing else changed" "$(c localhost:18094/ | sed 's#<link rel=stylesheet href=/style.css>##; s#<header class=top>.*</header>##; s#<iframe class=recent src=/recent title="Recent jobs"></iframe>##' | cmp - "$T/site/index.html" && echo same)" same
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
  u=${pair%%:*}; k=${pair##*:}; case $k in jobs) href=/jobs;; edit) href=/edit;; compare) href=/compare;; esac
  check "design: $u has the header and marks $k as the open place" "$(cget $u | grep -c "class=brand") $(cget $u | grep -o '<a href=[^>]*aria-current=page>' | tr '\n' ' ')" "1 <a href=$href aria-current=page> "
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
check "design: the edit form has the drop zone and the instruction as a text area" "$(curl -s -H "$(hdr)" localhost:18097/edit | grep -c 'type=file accept=image/jpeg,image/png,image/webp required.*<textarea name=prompt')" 1
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
  check "main page: the header is there once and marks Studio as the open place" "$(echo "$PG" | grep -c 'class=brand') $(echo "$PG" | grep -o 'aria-current=page>[A-Za-z ]*' | tr '\n' ' ')" "1 aria-current=page>Studio "
  check "main page: the app is untouched (its root and its script are where they were)" "$(echo "$PG" | grep -c 'id="root"') $(echo "$PG" | grep -c 'src="./app.js"')" "1 1"
  check "main page: the header comes before the app's root" "$(echo "$PG" | tr '\n' ' ' | grep -c '<body><header class=top>.*</header> *<div id="root"')" 1
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
# the cases page (studio/cases.cho): six forms that post to /edit with two pictures and a prepared instruction
CS=$(curl -s -H "$(hdr)" localhost:18097/cases)
check "cases: six forms, each to /edit as multipart" "$(echo "$CS" | grep -o '<form id=[a-z]* method=post action=/edit enctype=multipart/form-data>' | wc -l | tr -d ' ')" 6
check "cases: each asks for image, image2 and the instruction (and no script)" "$(echo "$CS" | grep -o 'name=image type=file\|name=image2 type=file\|<textarea name=prompt' | sort | uniq -c | tr -s ' ' | tr '\n' ';') $(echo "$CS" | grep -ci '<script')" " 6 <textarea name=prompt; 6 name=image type=file; 6 name=image2 type=file; 0"
check "cases: the instructions name image1 and image2" "$(echo "$CS" | grep -o '<textarea[^>]*>[^<]*</textarea>' | grep 'image1' | grep -c 'image2')" 6
check "cases: the header has the place and marks it open" "$(echo "$CS" | grep -o '<a href=/cases aria-current=page>Cases</a>' | wc -l | tr -d ' ')" 1
check "cases: the other pages link to it" "$(curl -s -H "$(hdr)" localhost:18097/edit | grep -c 'href=/cases')" 1
check "cases: each form offers the optional sizes" "$(echo "$CS" | grep -o 'name=note maxlength=200' | wc -l | tr -d ' ')" 6
NF=$(curl -s -o /dev/null -w '%{redirect_url}' -H "$(hdr)" -F prompt=place -F "note=the sofa is 2.2 m wide" -F "image=@$R/p1.jpg" -F "image2=@$R/p2.png" localhost:18097/edit); NFI=${NF##*/}
check "cases: the sizes are added to the instruction as one argument" "$(argv_of $NFI | sed 's#.*\.jpg ##')" "place Real sizes: the sofa is 2.2 m wide. "
check "cases: sizes of 201 bytes are refused: 422" "$(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -F prompt=x --form-string "note=$(head -c 201 /dev/zero | tr '\0' a)" -F "image=@$R/p1.jpg" -F "image2=@$R/p2.png" localhost:18097/edit)" 422
check "cases: an instruction and sizes over 1000 bytes together are refused: 422" "$(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" --form-string "prompt=$(head -c 900 /dev/zero | tr '\0' a)" --form-string "note=$(head -c 150 /dev/zero | tr '\0' b)" -F "image=@$R/p1.jpg" -F "image2=@$R/p2.png" localhost:18097/edit)" 422
check "cases: sizes with a control character are refused: 422" "$(curl -s -o /dev/null -w '%{http_code}' -H "$(hdr)" -F prompt=x --form-string "note=$(printf 'a\001b')" -F "image=@$R/p1.jpg" -F "image2=@$R/p2.png" localhost:18097/edit)" 422
check "cases: only the try-on and the room cards carry a tip, above the instruction" "$(echo "$CS" | grep -o '<form id=[a-z]*\|class=tip\|<label>The instruction' | tr '\n' ' ' | sed 's#<form id=##g; s#class=tip#TIP#g; s#<label>The instruction#INSTR#g')" "tryon TIP INSTR room TIP INSTR object INSTR scene INSTR hair INSTR style INSTR "
check "cases: the try-on tip says to name the garments" "$(echo "$CS" | grep -o '<form id=tryon.*' | grep -o 'class=tip[^>]*>[^<]*' | grep -c 'Say which garments to swap')" 1
check "cases: the room tip says to name what to replace" "$(echo "$CS" | grep -o '<form id=room.*' | grep -o 'class=tip[^>]*>[^<]*' | head -1 | grep -c 'name what to replace')" 1
check "cases: a tip is outside the textarea, so the instruction posted is unchanged" "$(echo "$CS" | grep -o '<textarea[^>]*>[^<]*</textarea>' | grep -c 'Tip:')" 0
TRY=$(echo "$CS" | grep -o '<form id=tryon.*' | grep -o '<textarea[^>]*>[^<]*' | head -1 | sed 's/<textarea[^>]*>//')
CF=$(curl -s -o /dev/null -w '%{redirect_url}' -H "$(hdr)" --form-string "prompt=$TRY" -F "image=@$R/p1.jpg" -F "image2=@$R/p2.png" localhost:18097/edit); CFI=${CF##*/}
check "cases: the try-on form's own instruction and two pictures make a two-picture edit" "$(argv_of $CFI | sed "s#$R/off#D#g" | cut -c1-110)" "edit -o D/out/X --also D/upload-$CFI-2.png -- D/upload-$CFI.jpg Dress the person in image1 in the clothing shown"
check "two pictures: the OpenAPI document names also and condition_area in the job and image2 in the form" "$(jq -r '.components.schemas.NewJob.properties | [has("also"), has("condition_area")] | join(",")' studio/openapi.json) $(grep -c '"image2"' studio/openapi.json)" "true,true 1"
exit $fail
