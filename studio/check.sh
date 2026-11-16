#!/bin/sh
# Builds the studio's programs and checks the worker against a fake script (no GPU): the arguments reach the script as data, its output lands in `log`, `state` says
# done or failed N, a missing argv file does nothing. CANCHO names the compiler; PK the packages directory of a cancho checkout.
set -e
cd "$(dirname "$0")/.."
PK="${PK:-../cancho/packages}"
C="${CANCHO:-cancho}"
mkdir -p build
$C build studio/studio.cho studio/files.cho studio/jobs.cho studio/results.cho "$PK"/http-server/server.cho --std -o build/studio 2>&1 | grep -v "^ld: warning" || true
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

# The server, the queue and the worker together, with a fake `media` that prints its arguments (port 18091).
printf '#!/bin/sh\nfor a in "$@"; do echo "[$a]"; done\n[ "$1" = edit ] && exit 3\nexit 0\n' > "$T/media"
chmod +x "$T/media"
mkdir "$T/data"
build/studio 18091 "$T/data" >/dev/null 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null; rm -rf "$T"' EXIT
sleep 1
post() { curl -s -w ' %{http_code}' -XPOST localhost:18091/api/jobs -d "$1"; }
check "POST a video job" "$(post '{"kind":"video","prompt":"-a fox; $(rm -rf /) \"q\"","frames":9,"size":"640x352","seed":3}')" '{"id":1} 202'
check "POST an edit job" "$(post '{"kind":"edit","prompt":"sunny","image":"/tmp/p.jpg"}')" '{"id":2} 202'
check "empty prompt refused" "$(post '{"kind":"video","prompt":""}')" '{"error":"prompt is a string of 1 to 1000 bytes"} 422'
check "an edit without image refused" "$(post '{"kind":"edit","prompt":"x"}' | sed 's/:.*//')" '{"error"'
check "frames over 129 refused" "$(post '{"kind":"video","prompt":"x","frames":500}' | tail -c 4)" ' 422'
check "bad size refused" "$(post '{"kind":"video","prompt":"x","size":"64x3520000"}' | tail -c 4)" ' 422'
check "unknown kind refused" "$(post '{"kind":"nope","prompt":"x"}')" '{"error":"kind is video or edit"} 422'
check "not JSON refused" "$(post 'not json' | tail -c 4)" ' 422'
check "queued" "$(curl -s localhost:18091/api/jobs/1)" '{"id":1,"state":"queued"}'
check "unknown job" "$(curl -s -w ' %{http_code}' localhost:18091/api/jobs/99)" '{"error":"no such job"} 404'
check "id that is not digits" "$(curl -s -o /dev/null -w '%{http_code}' localhost:18091/api/jobs/1a)" 422
check "queue file" "$(tr '\n' ' ' < "$T/data/queue")" '1 2 '
check "argv: the prompt is one entry after --" "$(tr '\0' '\n' < "$T/data/1.argv" | tail -1)" '-a fox; $(rm -rf /) "q"'
build/studio_worker "$T/media" /Users/test "$T/data"
check "job 1 done" "$(curl -s localhost:18091/api/jobs/1)" '{"id":1,"state":"done"}'
check "job 2 failed 3" "$(curl -s localhost:18091/api/jobs/2)" '{"id":2,"state":"failed 3"}'
check "job 1 got its arguments" "$(grep -c . "$T/data/1.log")" 11
# The results page and the files (job 1 done with a fake result, job 2 failed, job 3 queued).
mkdir -p "$T/data/out/1"
printf '\211PNG\r\n\032\n\0\0binary\377\0' > "$T/data/out/1/video.mp4"
cp "$T/data/out/1/video.mp4" "$T/data/secret.bin"
printf '3 queued\n' >/dev/null
printf 'video\0q\0' > "$T/data/3.argv"
curl -s localhost:18091/jobs/1 > "$T/page1"
check "page of a done job: state" "$(grep -c 'state: done' "$T/page1")" 1
check "page of a done job: the video" "$(grep -c 'src=/files/1/video.mp4' "$T/page1")" 1
check "page of a done job: no picture it does not have" "$(grep -c 'edited.png\|video.gif' "$T/page1")" 0
check "page of a failed job: no result" "$(curl -s localhost:18091/jobs/2 | grep -c 'state: failed 3.*\(video\|edited\)')" 0
check "page of a queued job refreshes" "$(curl -s localhost:18091/jobs/3 | grep -c 'refresh')" 1
check "page of an unknown job" "$(curl -s -o /dev/null -w '%{http_code}' localhost:18091/jobs/99)" 404
check "page of a bad id" "$(curl -s -o /dev/null -w '%{http_code}' localhost:18091/jobs/1a)" 422
curl -s -D "$T/hdr" localhost:18091/files/1/video.mp4 > "$T/got.bin"
check "file: the bytes arrive unchanged" "$(cmp "$T/got.bin" "$T/data/out/1/video.mp4" && echo same)" same
check "file: media type" "$(grep -ci '^content-type: video/mp4' "$T/hdr")" 1
check "file: the log is the worker's output" "$(curl -s localhost:18091/files/1/log | head -1)" '[video]'
head -c 6000000 /dev/urandom > "$T/data/out/1/video.gif"
check "file: 6 MB is refused, not sent cut" "$(curl -s -o /dev/null -w '%{http_code}' localhost:18091/files/1/video.gif)" 413
head -c 150000 /dev/urandom > "$T/data/out/1/edited.png"
check "file: 150 KB arrive unchanged" "$(curl -s localhost:18091/files/1/edited.png | cmp - "$T/data/out/1/edited.png" && echo same)" same
rm "$T/data/out/1/edited.png"
check "file: a name not on the list" "$(curl -s -o /dev/null -w '%{http_code}' localhost:18091/files/1/secret.bin)" 404
check "file: a path is not a name" "$(curl -s --path-as-is -o /dev/null -w '%{http_code}' 'localhost:18091/files/1/../secret.bin')" 404
check "file: a file the job does not have" "$(curl -s -o /dev/null -w '%{http_code}' localhost:18091/files/1/edited.png)" 404
check "file: bad id" "$(curl -s -o /dev/null -w '%{http_code}' localhost:18091/files/1a/log)" 422
rc=0; build/studio 18092 relative 2>/dev/null || rc=$?; check "relative data dir refused" "$rc" 2
exit $fail
