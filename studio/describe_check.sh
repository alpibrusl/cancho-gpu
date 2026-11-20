#!/bin/sh
# Checks of the picture describer (docs/studio-sessions.md, section 16): `studio/describer.cho`, the stand-in for Ollama of `studio/rewrite_stub.cho`, no model, no GPU.
#
#   CANCHO=<compiler> PK=<cancho checkout>/packages sh studio/describe_check.sh
#
# As the rewriter's checks do: the programs the checks run are built from copies of the sources in which two literals are replaced by `sed` (the address's port 11434 -> 18160, so that the real
# Ollama on this machine is never reached, and the deadline 180 s -> 2 s), and the real describer is built from the real files for its authority report.
set -e
cd "$(dirname "$0")/.."
PK="${PK:-../cancho/packages}"
C="${CANCHO:-cancho}"
T=$(mktemp -d)
trap 'kill $SP $DP 2>/dev/null || true; rm -rf "$T"' EXIT
SP=; DP=
mkdir -p build "$T/src"
fail=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2', wanted '$3'"; fail=1; fi; }
build() { out=$1; shift; $C build "$@" --std -o "$out" 2>&1 | grep -v "^ld: warning" || true; }

# 1. the real describer says what it may do
A=$($C authority studio/describer.cho studio/files.cho studio/rewrite.cho studio/results.cho --std 2>&1)
check "authority: the only network row is 127.0.0.1:11434" "$(echo "$A" | grep -c 'net_out("127.0.0.1:11434")') $(echo "$A" | grep 'net_' | grep -vc '127.0.0.1:11434')" "1 0"
check "authority: no other programs, no foreign code, no signals" "$(echo "$A" | grep -c 'other programs') $(echo "$A" | grep -ci 'foreign code unbounded') $(echo "$A" | sed -n '/^performs/,/^never/p' | grep -c signals)" "1 0 0"

# 2. test builds
for f in describer.cho rewrite.cho; do sed -e 's/11434/18160/g' -e 's/"", 180)/"", 2)/' studio/$f > "$T/src/$f"; done
cp studio/files.cho studio/results.cho "$T/src/"
build "$T/describer" "$T/src/describer.cho" "$T/src/files.cho" "$T/src/rewrite.cho" "$T/src/results.cho"
build build/rewrite_stub studio/rewrite_stub.cho studio/files.cho "$PK"/http-server/server.cho
for f in describer.cho rewrite.cho; do check "test build of $f differs from the real file only by the port and the deadline" "$(sed -e 's/18160/11434/g' -e 's/"", 2)/"", 180)/' "$T/src/$f" | cmp - studio/$f && echo same)" same; done

# helpers
(printf '\211PNG\r\n\032\n'; head -c 2000 /dev/urandom) > "$T/pic.png"
(printf '\377\330\377\340'; head -c 2000 /dev/urandom) > "$T/pic.jpg"
stub() { build/rewrite_stub 18160 "$T/last.json" "$1" >/dev/null 2>&1 & SP=$!; sleep 0.7; }
unstub() { kill $SP 2>/dev/null; wait $SP 2>/dev/null || true; SP=; }
newdir() { D=$(mktemp -d "$T/d.XXXX"); : > "$D/queue"; : > "$D/lib.index"; n=0; }
addpic() { n=$((n + 1)); if [ "$1" = jpg ]; then cp "$T/pic.jpg" "$D/lib-$n.jpg"; else cp "$T/pic.png" "$D/lib-$n.png"; fi; echo "$n $1" >> "$D/lib.index"; }
describe_for() { "$T/describer" "$D" test-model --every 1 >/dev/null 2>&1 & DP=$!; sleep "$1"; kill $DP 2>/dev/null; wait $DP 2>/dev/null || true; DP=; }
files() { ls "$D" | grep "^lib-$1\.\(txt\|auto\|tried\)$" | tr '\n' ' '; }

# 3. described, and what was sent
newdir; addpic png; addpic jpg; echo 'my own words' > "$D/lib-3.txt"; addpic png
stub ok; describe_for 3
check "ok: an undescribed picture gets the answer as lib-N.txt, marked auto" "$(cat "$D/lib-1.txt") / $(cat "$D/lib-2.txt") / $(files 1)$(files 2)" "A stub prompt. / A stub prompt. / lib-1.auto lib-1.txt lib-2.auto lib-2.txt "
check "ok: a description a person typed is never touched" "$(cat "$D/lib-3.txt") / $(files 3)" "my own words / lib-3.txt "
check "ok: the request names the model, sends a picture, asks for no streaming or thinking, and has no tools" "$(grep -o '"model":"test-model"' "$T/last.json") $(grep -c '"images"' "$T/last.json") $(grep -o '"stream":false,"think":false,"keep_alive":0' "$T/last.json") $(grep -c '"tools"' "$T/last.json")" '"model":"test-model" 1 "stream":false,"think":false,"keep_alive":0 0'
check "ok: the question asks for plain comma-separated words, with no adjectives and no verbs, at most 20 words" "$(grep -c 'plain comma-separated words for a search index' "$T/last.json") $(grep -c 'no adjectives like serene or beautiful, no verbs, at most 20 words' "$T/last.json")" "1 1"
check "ok: the temperature is 0, so the same picture gets the same words" "$(grep -o '"temperature":[0-9.]*' "$T/last.json" | sed 's/0\.0$/0/')" '"temperature":0'
unstub

# 4. at most three a pass, the rest on the next
newdir; for k in 1 2 3 4 5; do addpic png; done
stub ok; "$T/describer" "$D" test-model --every 30 >/dev/null 2>&1 & DP=$!; sleep 2; kill $DP 2>/dev/null; wait $DP 2>/dev/null || true; DP=; first=$(ls "$D" | grep -c '\.txt$')
describe_for 3; all=$(ls "$D" | grep -c '\.txt$')
check "ok: three at most in a pass (a long interval: one pass), the rest in the next" "$first $all" "3 5"
unstub

# 5. a job in the queue: the GPU is the job's
newdir; addpic png; echo 1 > "$D/queue"; printf 'edit\0--\0x\0y\0' > "$D/1.argv"
stub ok; describe_for 3
check "wait: nothing is described while a job is queued or running" "$(files 1)" ""
echo done > "$D/1.state"; describe_for 3
check "wait: ...and it is once the queue is empty" "$(cat "$D/lib-1.txt")" "A stub prompt."
unstub

# 6. cut to 200 bytes
newdir; addpic png; stub mid; describe_for 3
len=$(wc -c < "$D/lib-1.txt" | tr -d ' ')
check "cut: an answer of 300 bytes becomes at most 200, ends at a whole word and holds only words" "$([ "$len" -le 200 ] && [ "$len" -ge 140 ] && echo ok) $(grep -c '^\(red \)*red$' "$D/lib-1.txt")" "ok 1"
unstub

# 7. an answer that cannot be used is marked tried and never asked again; one that fails for now is asked again
newdir; addpic png; stub empty; describe_for 3
check "tried: an empty answer: no description, marked tried" "$(files 1)" "lib-1.tried "
unstub; stub ok; describe_for 3
check "tried: ...and not asked again (a good stand-in later changes nothing)" "$(files 1)" "lib-1.tried "
unstub
newdir; addpic png; stub long; describe_for 3
check "tried: an answer over 1,000 bytes: marked tried" "$(files 1)" "lib-1.tried "
unstub
newdir; addpic png; stub err500; describe_for 3
check "retry: an error from Ollama writes nothing" "$(files 1)" ""
unstub; stub ok; describe_for 3
check "retry: ...and the next pass, with Ollama well, describes it" "$(cat "$D/lib-1.txt")" "A stub prompt."
unstub
newdir; addpic png; describe_for 3
check "retry: Ollama not there at all writes nothing" "$(files 1)" ""
newdir; addpic png; stub slow; describe_for 6
check "retry: no answer in time writes nothing" "$(files 1)" ""
unstub

# 8. a picture that is gone or too small is marked tried; a bad index line is skipped
newdir; addpic png; rm "$D/lib-1.png"; echo 'x png' >> "$D/lib.index"; stub ok; describe_for 3
check "gone: a listed picture whose file is gone is marked tried, a bad line is skipped" "$(files 1) $(ls "$D" | grep -c '^lib-x')" "lib-1.tried  0"
unstub

# 10. the group of a picture (docs/studio-sessions.md, section 19)
kind() { cat "$D/lib-$1.kind" 2>/dev/null; }
newdir; addpic png; addpic jpg; echo 'my own words' > "$D/lib-2.txt"
stub garment; describe_for 3
check "kind: a group's name with a capital and a full stop is the group; a typed description does not stop it being sorted; the description is untouched" "$(kind 1) $(kind 2) / $(cat "$D/lib-2.txt")" "garment garment / my own words"
unstub
newdir; addpic png; stub rooms; describe_for 3
check "kind: the first word of a list counts" "$(kind 1)" "room"
unstub
newdir; addpic png; stub plural; describe_for 3
check "kind: a word that is not in the list (a plural) is other" "$(kind 1)" "other"
unstub
newdir; addpic png; stub ok; describe_for 3
check "kind: an answer that is no group (words) is other" "$(kind 1)" "other"
check "kind: the question in the program holds the closed list and the rule for a person wearing clothes (the words measured on eight pictures)" "$(grep -c 'one word from this list and nothing else: person, garment, furniture, room, place, object, animal, other' studio/describer.cho) $(grep -c 'Use person when a person is the main subject, even when they wear clothes' studio/describer.cho)" "1 1"
unstub
newdir; addpic png; stub empty; describe_for 3
check "kind: an empty answer: no group, marked kindtried, and not asked again" "$(ls "$D" | grep -c '^lib-1.kind$') $(ls "$D" | grep -c '^lib-1.kindtried$')" "0 1"
unstub; stub garment; describe_for 3
check "kind: ...a good stand-in later changes nothing" "$(ls "$D" | grep -c '^lib-1.kind$')" "0"
unstub
newdir; addpic png; stub err500; describe_for 3
check "kind: an error from Ollama writes nothing, neither a group nor a mark" "$(ls "$D" | grep -c '^lib-1.kind')" "0"
unstub; stub garment; describe_for 3
check "kind: ...and the next pass sorts it" "$(kind 1)" "garment"
unstub
newdir; addpic png; echo room > "$D/lib-1.kind"; stub garment; describe_for 3
check "kind: a picture that has a group keeps it" "$(kind 1)" "room"
unstub
newdir; addpic png; rm "$D/lib-1.png"; stub garment; describe_for 3
check "kind: a picture whose file is gone is not sorted" "$(ls "$D" | grep -c '^lib-1.kind')" "0"
unstub

# 9. a mutant: the typed description is overwritten
mkdir -p "$T/mut"; sed -e 's/exists(heap, fs, dir, number, ".txt") == false \&\& //' "$T/src/describer.cho" > "$T/mut/describer.cho"; cp "$T/src/files.cho" "$T/src/rewrite.cho" "$T/src/results.cho" "$T/mut/"
build "$T/mut/d" "$T/mut/describer.cho" "$T/mut/files.cho" "$T/mut/rewrite.cho" "$T/mut/results.cho"
newdir; addpic png; echo 'typed' > "$D/lib-1.txt"; stub ok; "$T/mut/d" "$D" test-model --every 1 >/dev/null 2>&1 & DP=$!; sleep 3; kill $DP 2>/dev/null; wait $DP 2>/dev/null || true; DP=
check "mutant: a describer that no longer looks for an existing description overwrites it (so the check above can fail)" "$(cat "$D/lib-1.txt")" "A stub prompt."
unstub
exit $fail
