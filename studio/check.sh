#!/bin/sh
# Builds the studio's programs and checks the worker against a fake script (no GPU): the arguments reach the script as data, its output lands in `log`, `state` says
# done or failed N, a missing argv file does nothing. CANCHO names the compiler; PK the packages directory of a cancho checkout.
set -e
cd "$(dirname "$0")/.."
PK="${PK:-../cancho/packages}"
C="${CANCHO:-cancho}"
mkdir -p build
$C build studio/studio.cho "$PK"/http-server/server.cho --std -o build/studio 2>&1 | grep -v "^ld: warning" || true
$C build studio/worker.cho studio/files.cho --std -o build/studio_worker 2>&1 | grep -v "^ld: warning" || true
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
printf '#!/bin/sh\necho "args: $1 | $2 | home=$HOME"\n[ "$1" = fail ] && exit 7\nexit 0\n' > "$T/fake.sh"
chmod +x "$T/fake.sh"
mkdir "$T/ok" "$T/bad" "$T/nofile"
printf 'video\0a prompt; with $(shell) chars\0' > "$T/ok/argv"
printf 'fail\0x\0' > "$T/bad/argv"
fail=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2', wanted '$3'"; fail=1; fi; }
build/studio_worker "$T/fake.sh" /Users/test "$T/ok"; check "exit 0" "$?" 0
check "state done" "$(cat "$T/ok/state")" done
check "metacharacters stay data" "$(cat "$T/ok/log")" 'args: video | a prompt; with $(shell) chars | home=/Users/test'
rc=0; build/studio_worker "$T/fake.sh" /Users/test "$T/bad" || rc=$?; check "exit 7" "$rc" 7
check "state failed 7" "$(cat "$T/bad/state")" "failed 7"
rc=0; build/studio_worker "$T/fake.sh" /Users/test "$T/nofile" || rc=$?; check "no argv: exit 3" "$rc" 3
check "no argv: nothing written" "$(ls "$T/nofile" | wc -l | tr -d ' ')" 0
rc=0; build/studio_worker 2>/dev/null || rc=$?; check "usage: exit 2" "$rc" 2
rc=0; build/studio_worker "$T/fake.sh" /Users/test relative/dir 2>/dev/null || rc=$?; check "relative job dir refused" "$rc" 3
exit $fail
