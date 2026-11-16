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
check "file: 6 MB arrive unchanged (streamed)" "$(curl -s localhost:18091/files/1/video.gif | cmp - "$T/data/out/1/video.gif" && echo same)" same
check "file: Content-Length is the size" "$(curl -s -D - -o /dev/null localhost:18091/files/1/video.gif | tr -d '\r' | grep -i '^content-length' | awk '{print $2}')" 6000000
head -c 40000000 /dev/urandom > "$T/data/out/1/video.gif"
check "file: 40 MB arrive unchanged" "$(curl -s localhost:18091/files/1/video.gif | cmp - "$T/data/out/1/video.gif" && echo same)" same
check "file: two in a row on one connection" "$(curl -s localhost:18091/files/1/video.gif localhost:18091/files/1/video.gif | wc -c | tr -d ' ')" 80000000
check "the server serves after streaming" "$(curl -s localhost:18091/api/jobs/1)" '{"id":1,"state":"done"}'
head -c 6000000 /dev/urandom > "$T/data/out/1/video.gif"
head -c 150000 /dev/urandom > "$T/data/out/1/edited.png"
check "file: 150 KB arrive unchanged" "$(curl -s localhost:18091/files/1/edited.png | cmp - "$T/data/out/1/edited.png" && echo same)" same
rm "$T/data/out/1/edited.png"
check "file: a name not on the list" "$(curl -s -o /dev/null -w '%{http_code}' localhost:18091/files/1/secret.bin)" 404
check "file: a path is not a name" "$(curl -s --path-as-is -o /dev/null -w '%{http_code}' 'localhost:18091/files/1/../secret.bin')" 404
check "file: a file the job does not have" "$(curl -s -o /dev/null -w '%{http_code}' localhost:18091/files/1/edited.png)" 404
check "file: bad id" "$(curl -s -o /dev/null -w '%{http_code}' localhost:18091/files/1a/log)" 422
rc=0; build/studio 18092 relative 2>/dev/null || rc=$?; check "relative data dir refused" "$rc" 2
exit $fail
