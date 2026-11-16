#!/bin/sh
# Builds the studio's programs and checks the worker against a fake script (no GPU): the arguments reach the script as data, its output lands in `log`, `state` says
# done or failed N, a missing argv file does nothing. CANCHO names the compiler; PK the packages directory of a cancho checkout.
set -e
cd "$(dirname "$0")/.."
PK="${PK:-../cancho/packages}"
C="${CANCHO:-cancho}"
mkdir -p build
$C build studio/studio.cho studio/files.cho studio/jobs.cho studio/results.cho studio/auth.cho studio/range.cho studio/upload.cho "$PK"/http-server/server.cho --std -o build/studio 2>&1 | grep -v "^ld: warning" || true
$C build studio/worker.cho studio/files.cho --std -o build/studio_worker 2>&1 | grep -v "^ld: warning" || true
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
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2', wanted '$3'"; fail=1; fi; }
build/studio_worker "$T/fake.sh" /Users/test "$T"; check "exit 0" "$?" 0
check "job 1 done" "$(cat "$T/1.state")" done
check "metacharacters stay data" "$(cat "$T/1.log")" 'args: video | a prompt; with $(shell) chars | home=/Users/test'
check "job 2 failed 7" "$(cat "$T/2.state")" "failed 7"
check "job 3 (no argv file): no log" "$(ls "$T/3.log" 2>/dev/null | wc -l | tr -d ' ')" 0
check "job 4 had a state: not run again" "$(cat "$T/4.state")" old
check "an id that is not digits is not a path" "$(ls "$T" | grep -c etc)" 0
build/studio_worker "$T/fake.sh" /Users/test "$T"; check "second drain: nothing new, still 0" "$?" 0
check "job 1 not run twice (log unchanged)" "$(wc -l < "$T/1.log" | tr -d ' ')" 1
rc=0; build/studio_worker 2>/dev/null || rc=$?; check "usage: exit 2" "$rc" 2
rc=0; build/studio_worker "$T/fake.sh" /Users/test relative/dir 2>/dev/null || rc=$?; check "relative data dir refused" "$rc" 3

# The script is started beneath its directory (`exec_spawn_under`), so the report says `exec_under`, not `exec("")` (any program anywhere). Measured before: `exec("")`.
if [ -n "${CANCHO_AUTH:-1}" ]; then
  A=$($C authority studio/worker.cho studio/files.cho --std 2>&1)
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
  for f in video.mp4 video.gif edited.png prompt.txt ci_s.f32 ctx.f16 text.npy video.npy; do printf 'x' > "$M/out/$id/$f"; done
  printf 'x' > "$M/out/$id/frames/f000.png"
  mkdir "$M/out/$id/vae"; printf 'x' > "$M/out/$id/vae/cv1.f16"; printf 'x' > "$M/out/$id/vae/bi2.f32"
done
printf 'secret' > "$M/secret.txt"; ln -s "$M/secret.txt" "$M/out/7/link"
printf 'video\0ok\0' > "$M/7.argv"; printf 'fail\0x\0' > "$M/8.argv"; printf '7\n8\n' > "$M/queue"
build/studio_worker "$M/fake.sh" /Users/test "$M"
check "tidy: work files removed" "$(ls "$M/out/7" | tr '\n' ' ')" 'edited.png frames link prompt.txt vae video.gif video.mp4 '
check "tidy: frames/ untouched" "$(ls "$M/out/7/frames")" f000.png
check "tidy: a subdirectory of work files is emptied" "$(ls "$M/out/7/vae" | wc -l | tr -d ' ')" 0
check "tidy: a link is not followed or removed" "$(cat "$M/secret.txt")" secret
check "tidy: the log says how many" "$(tail -1 "$M/7.log")" "studio: removed 6 work files from out/7"
check "tidy: a failed job keeps everything" "$(ls "$M/out/8" | wc -l | tr -d ' ')" 10
rm -rf "$M"

# The worker in a loop (`--every`): a job that arrives later is run; TERM while it waits stops it at once (exit 0); TERM during a job lets that job end first.
L=$(mktemp -d)
printf '#!/bin/sh\nif [ "$1" = slow ]; then sleep 2; fi\necho "ran $1"\nexit 0\n' > "$L/fake.sh"
chmod +x "$L/fake.sh"
printf 'video\0one\0' > "$L/1.argv"; printf '1\n' > "$L/queue"
build/studio_worker "$L/fake.sh" /Users/test "$L" --every 30 &
WP=$!
sleep 1
check "loop: the job in the queue ran" "$(cat "$L/1.state")" done
printf 'video\0two\0' > "$L/2.argv"; printf '1\n2\n' > "$L/queue"
sleep 0.5
check "loop: waiting (30 s), a later job is not run yet" "$(ls "$L/2.state" 2>/dev/null | wc -l | tr -d ' ')" 0
kill -TERM $WP
rc=0; wait $WP || rc=$?
check "loop: TERM while waiting: exit 0" "$rc" 0
build/studio_worker "$L/fake.sh" /Users/test "$L" --every 1 &
WP=$!
sleep 1.5
check "loop: a job that arrived while it was stopped runs on the next drain" "$(cat "$L/2.state")" done
printf 'slow\0three\0' > "$L/3.argv"; printf '1\n2\n3\n' > "$L/queue"
sleep 1.2
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
sleep 1
check "lock: the first worker is in job 1" "$(cat "$K/1.state")" running
rc=0; build/studio_worker "$K/fake.sh" /Users/test "$K" 2>"$K/err" || rc=$?
check "lock: a second worker, once, during a job: exit 4" "$rc" 4
check "lock: ...and says why" "$(grep -c 'another worker holds worker.lock' "$K/err")" 1
rc=0; build/studio_worker "$K/fake.sh" /Users/test "$K" --every 5 2>/dev/null || rc=$?
check "lock: a second worker, looping: exit 4" "$rc" 4
sleep 2
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
sleep 1
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
sleep 1
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
check "unknown kind refused" "$(post '{"kind":"nope","prompt":"x"}')" '{"error":"kind is video or edit"} 422'
check "not JSON refused" "$(post 'not json' | tail -c 4)" ' 422'
check "queued" "$(c localhost:18091/api/jobs/1)" '{"id":1,"state":"queued"}'
check "unknown job" "$(c -w ' %{http_code}' localhost:18091/api/jobs/99)" '{"error":"no such job"} 404'
check "id that is not digits" "$(c -o /dev/null -w '%{http_code}' localhost:18091/api/jobs/1a)" 422
check "queue file" "$(tr '\n' ' ' < "$T/data/queue")" '1 2 '
check "argv: the prompt is one entry after --" "$(tr '\0' '\n' < "$T/data/1.argv" | tail -1)" '-a fox; $(rm -rf /) "q"'
build/studio_worker "$T/media" /Users/test "$T/data"
check "job 1 done" "$(c localhost:18091/api/jobs/1)" '{"id":1,"state":"done"}'
check "job 2 failed 3" "$(c localhost:18091/api/jobs/2)" '{"id":2,"state":"failed 3"}'
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
sleep 1
TOK2=$(grep -o 'token=[0-9a-f]*' "$T/s2.log" | cut -d= -f2); TOK=$TOK2
check "site: / is index.html" "$(c localhost:18094/ | cmp - "$T/site/index.html" && echo same)" same
check "site: /app.js is app.js, a script" "$(c -D - -o /dev/null localhost:18094/app.js | tr -d '\r' | grep -ci '^content-type: text/javascript')" 1
check "site: app.js bytes unchanged" "$(c localhost:18094/app.js | cmp - "$T/site/app.js" && echo same)" same
TOK=$TOK1
check "site: no site, / is the stub" "$(c localhost:18091/ | grep -c 'Slice 1')" 1
TOK=$TOK1 post '{"kind":"video","prompt":"third"}' >/dev/null
check "jobs: newest first" "$(TOK=$TOK2 c localhost:18094/jobs | grep -o 'job [0-9]*' | tr '\n' ' ')" 'job 3 job 2 job 1 '
check "jobs: states" "$(TOK=$TOK2 c localhost:18094/jobs | grep -o '</a>: [a-z0-9 ]*' | tr '\n' ' ')" '</a>: queued </a>: failed 3 </a>: done '
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
check "openapi: 202 on POST /api/jobs is listed (queued by the checks above)" "$(doc_lists /api/jobs 202)" yes
row /api/jobs POST /api/jobs -H 'Content-Type: application/json' -d '{"kind":"nope"}'
row /api/jobs POST /api/jobs -H 'Content-Type: text/plain' -d '{}'
row '/api/jobs/{id}' GET /api/jobs/1
row '/api/jobs/{id}' GET /api/jobs/99999
row '/api/jobs/{id}' GET /api/jobs/x
row /jobs GET /jobs
row '/jobs/{id}' GET /jobs/1
row '/jobs/{id}' GET /jobs/99999
row '/jobs/{id}' GET /jobs/x
row '/files/{id}/{name}' GET /files/1/log
row '/files/{id}/{name}' GET /files/1/log -H 'Range: bytes=0-3'
row '/files/{id}/{name}' GET /files/1/log -H 'Range: bytes=99999-'
row '/files/{id}/{name}' GET /files/1/nothing
row '/files/{id}/{name}' GET /files/x/log
row /app.js GET /app.js
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
for p in /api/jobs/ /jobs/ /files/; do
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
sleep 1
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
exit $fail
