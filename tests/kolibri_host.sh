#!/bin/sh
# tests/kolibri_host.sh -- the host side of Kolibri 1.1's step (tests/kolibri_gpu.cho, docs/kolibri.md K5) without the model: it builds, and on a toy GGUF written here byte by byte (three layers 32
# wide: a dense one, a mixture of 2 experts with its own keys, a mixture that reads layer 1's keys; 2 query heads over 1 key/value head of 16; a vocabulary of 64) every file `weights` and `inputs`
# write is what awk's arithmetic of the formats says: the copied tensors, the repacked codes and the scales to the byte, the Q4_1 offsets, the decoded Q6_K head, the embedding rows and the tables
# as numbers; the list of files is exactly the expected one. Then `inputs` for a longer run and longer caches (docs/kolibri.md section 19.2), `draws`, `argmax`, and the refusals, each with its rule and status 1 (2 for usage), none a trap.
set -eu
cd "$(dirname "$0")/.."
C="${CANCHO:-cancho}"
# KOLIBRI_GPU=<binary> tests another build of the program and builds nothing: the mutants, by hand
if [ -z "${KOLIBRI_GPU:-}" ]; then CANCHO="$C" scripts/build-kolibri.sh; fi
B=${KOLIBRI_GPU:-build/kolibri_gpu}
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
fail=0
: > "$T/fails"   # a line a failure of the two big checks, so that an `ok` is said only of a part that had none
. tests/kolibri_toy.sh
mkdir "$T/want" "$T/w" "$T/x"
toy "$T/toy.gguf" none
check() { # dir, from manifest line, to manifest line (exclusive): every file as wanted, and no other file in dir
  sed -n "$2,$(($3 - 1))p" "$T/manifest" > "$T/part"
  while read -r file kind; do
    if [ ! -f "$1/$file" ]; then echo "FAIL $file is not written"; fail=1; echo x >> "$T/fails"; continue; fi
    if [ "$kind" = bytes ]; then od -A n -v -t u1 "$1/$file" | tr -s ' \n' '\n\n' | grep -v '^$' > "$T/got.txt" || true; tol=0; else od -A n -v -t fF "$1/$file" | tr -s ' \n' '\n\n' | grep -v '^$' > "$T/got.txt" || true; tol=2e-6; fi
    r=$(paste "$T/got.txt" "$T/want/$file" | awk -v tol="$tol" -v n="$(wc -l < "$T/want/$file")" 'NF != 2 { bad = "a line of " NF " fields"; exit } { d = $1 - $2; if (d < 0) d = -d; s = ($2 < 0 ? -$2 : $2); if (d > tol * s) { bad = "value " NR ": " $1 ", wanted " $2; exit } } END { if (bad != "") print bad; else if (NR != n) print NR " values, wanted " n; else print "ok" }')
    if [ "$r" != ok ]; then echo "FAIL $file: $r"; fail=1; echo x >> "$T/fails"; fi
  done < "$T/part"
  if [ "$(ls "$1" | sort | tr '\n' ' ')" != "$(cut -d' ' -f1 "$T/part" | sort | tr '\n' ' ')" ]; then echo "FAIL the files of $1 are not the expected ones: $(ls "$1" | sort | tr '\n' ' ')"; fail=1; echo x >> "$T/fails"; fi
}
split_at=$(grep -n '^inputs -$' "$T/manifest" | cut -d: -f1)
total=$(wc -l < "$T/manifest")
before=$(cat "$T/fails")
if "$B" weights "$T/toy.gguf" "$T/w" > "$T/out.txt" 2> "$T/err.txt"; then check "$T/w" 1 "$split_at"; else echo "FAIL weights: $(head -c 300 "$T/err.txt")"; fail=1; echo x >> "$T/fails"; fi
if [ "$(tr '\n' ';' < "$T/out.txt")" != "layer 0: its own keys, a dense MLP;layer 1: its own keys, a mixture of experts;layer 2: reads another layer's keys, a mixture of experts;" ]; then echo "FAIL weights says: $(cat "$T/out.txt")"; fail=1; echo x >> "$T/fails"; fi
[ "$(cat "$T/fails")" = "$before" ] && echo "ok   weights: $((split_at - 1)) files of a three-layer toy, each as awk has it, and no other"
before=$(cat "$T/fails")
printf '3 0\n7\n' > "$T/ids.txt"
if "$B" inputs "$T/toy.gguf" "$T/ids.txt" "$T/x" 64 5 > "$T/out.txt" 2> "$T/err.txt"; then check "$T/x" $((split_at + 1)) $((total + 1)); else echo "FAIL inputs: $(head -c 300 "$T/err.txt")"; fail=1; echo x >> "$T/fails"; fi
[ "$(cat "$T/out.txt")" = "steps 5 prompt 3" ] || { echo "FAIL inputs says: $(cat "$T/out.txt")"; fail=1; echo x >> "$T/fails"; }
[ "$(cat "$T/fails")" = "$before" ] && echo "ok   inputs: the embedding rows of three ids, two steps past them, and the six tables"
# draws: a seed's uniform numbers a line each, every decimal of a number of 24 bits over 2^24; awk prints the same 24 decimals of its own arithmetic
if [ "$("$B" draws 7 5 2> "$T/err.txt")" = "$(cat "$T/want_draws.txt")" ]; then echo "ok   draws: the five numbers of seed 7, to the last decimal"; else echo "FAIL draws 7 5: $("$B" draws 7 5 2>&1 | head -c 300), wanted $(tr '\n' ' ' < "$T/want_draws.txt")"; fail=1; fi
if [ "$("$B" draws 7 3)" = "$("$B" draws 7 5 | head -3)" ] && [ "$("$B" draws 8 3)" != "$("$B" draws 7 3)" ] && [ -z "$("$B" draws 7 0)" ] && [ "$("$B" draws 0 2 | wc -l | tr -d ' ')" = 2 ]; then echo "ok   draws: a shorter run is the start of a longer one, another seed is other numbers, none is none"; else echo "FAIL draws: a run's start, another seed or a count of 0"; fail=1; fi
# a longer run with longer caches: 136 steps (the most an indexed layer of this toy sees whole) and 192 positions. What a step needs of the positions is a number a step and a number a
# position; a sliding layer's mask is the same 64 rows whatever the run (docs/kolibri.md section 19.2)
mkdir "$T/xl"
nums() { od -A n -v -t fF "$1" | tr -s ' \n' '\n\n' | grep -v '^$'; }
if "$B" inputs "$T/toy.gguf" "$T/ids.txt" "$T/xl" 192 136 > "$T/out.txt" 2> "$T/err.txt" && cmp -s "$T/xl/maskw.f32" "$T/x/maskw.f32" && [ "$(nums "$T/xl/pos.f32" | awk '$1 != NR - 1 { bad = 1 } END { print NR, bad + 0 }')" = "136 0" ] && [ "$(nums "$T/xl/iota.f32" | awk '$1 != NR - 1 { bad = 1 } END { print NR, bad + 0 }')" = "192 0" ] && [ "$(wc -c < "$T/xl/cos.f32" | tr -d ' ')" = $((136 * 8 * 4)) ] && [ "$(ls "$T/xl" | sort | tr '\n' ' ')" = "$(ls "$T/x" | sort | tr '\n' ' ')" ]; then
  echo "ok   inputs for 136 steps and 192 positions: the steps' numbers, the positions', the same mask of 64 rows, and no file of steps times positions"
else echo "FAIL inputs for 136 steps and 192 positions: $(head -c 300 "$T/err.txt") $(ls -l "$T/xl" | awk '{ print $5, $9 }' | tr '\n' ' ')"; fail=1; fi
# argmax: the largest of the decoded head by awk, the first of equal values, a file of one value
want=$(awk '{ if (NR == 1 || $1 > m) { m = $1; i = NR - 1 } } END { print i }' "$T/want/head.f32")
got=$("$B" argmax "$T/w/head.f32" 2>&1 | cut -d' ' -f1-2)
if [ "$got" = "argmax $want" ]; then echo "ok   argmax of the head: $want"; else echo "FAIL argmax of the head: $got, wanted $want"; fail=1; fi
got=$("$B" argmax "$T/w/ones_d.f32" 2>&1)
if [ "$got" = "argmax 0 value 1.0 margin 0.0" ]; then echo "ok   argmax of equal values is the first"; else echo "FAIL argmax of equal values: $got"; fail=1; fi
got=$("$B" argmax "$T/w/sl.f32" 2>&1)
if [ "$got" = "argmax 0 value 0.0 margin none" ]; then echo "ok   argmax of one value"; else echo "FAIL argmax of one value: $got"; fail=1; fi
refuse() { # name, rule, status, the command
  name=$1; rule=$2; want=$3; shift 3
  status=0; "$@" > "$T/out.txt" 2> "$T/err.txt" || status=$?
  if [ "$status" -eq "$want" ] && grep -q "^error\[$rule\]" "$T/err.txt"; then echo "ok   refused ($rule): $name"; else echo "FAIL $name: status $status, wanted $want and error[$rule]: $(head -c 300 "$T/err.txt")"; fail=1; fi
}
mkdir "$T/r"
refuse "no command" kolibri-usage 2 "$B"
refuse "an argument too many" kolibri-usage 2 "$B" weights "$T/toy.gguf" "$T/r" more
refuse "a file that is not there" kolibri-read 1 "$B" weights "$T/none.gguf" "$T/r"
printf 'not a GGUF file at all, but long enough to have a header' > "$T/bad.gguf"
refuse "a file that is not a GGUF" gguf-magic 1 "$B" weights "$T/bad.gguf" "$T/r"
head -c 1000 "$T/toy.gguf" > "$T/cut.gguf"
refuse "a header cut short" gguf-truncated 1 "$B" weights "$T/cut.gguf" "$T/r"
head -c $(($(wc -c < "$T/toy.gguf") - 600)) "$T/toy.gguf" > "$T/cut2.gguf"
refuse "the data cut short" kolibri-read 1 "$B" weights "$T/cut2.gguf" "$T/r"
toy "$T/v.gguf" notensor
refuse "a tensor that is not there" kolibri-tensor 1 "$B" weights "$T/v.gguf" "$T/r"
toy "$T/v.gguf" nokv
refuse "a count missing from the metadata" kolibri-meta 1 "$B" weights "$T/v.gguf" "$T/r"
toy "$T/v.gguf" f32q
refuse "a matrix that is F32 where 4-bit is read" kolibri-tensor 1 "$B" weights "$T/v.gguf" "$T/r"
toy "$T/v.gguf" wide
refuse "a 4-bit matrix whose rows are not a multiple of 32" kolibri-tensor 1 "$B" weights "$T/v.gguf" "$T/r"
refuse "a directory that is not there" kolibri-write 1 "$B" weights "$T/toy.gguf" "$T/nowhere"
printf '3 x 7\n' > "$T/bad.txt"
refuse "an id that is not a number" kolibri-ids 1 "$B" inputs "$T/toy.gguf" "$T/bad.txt" "$T/r" 64
printf '3 64\n' > "$T/bad.txt"
refuse "an id past the vocabulary" kolibri-ids 1 "$B" inputs "$T/toy.gguf" "$T/bad.txt" "$T/r" 64
: > "$T/bad.txt"
refuse "no ids" kolibri-ids 1 "$B" inputs "$T/toy.gguf" "$T/bad.txt" "$T/r" 64
refuse "positions that are not a multiple of 64" kolibri-usage 1 "$B" inputs "$T/toy.gguf" "$T/ids.txt" "$T/r" 100
refuse "positions that are not a number" kolibri-usage 1 "$B" inputs "$T/toy.gguf" "$T/ids.txt" "$T/r" 6x
awk 'BEGIN { for (i = 0; i < 65; i++) printf "1 " }' > "$T/bad.txt"
refuse "more ids than positions" kolibri-ids 1 "$B" inputs "$T/toy.gguf" "$T/bad.txt" "$T/r" 64
refuse "fewer steps than ids" kolibri-usage 1 "$B" inputs "$T/toy.gguf" "$T/ids.txt" "$T/r" 64 2
refuse "an argument after the steps" kolibri-usage 2 "$B" inputs "$T/toy.gguf" "$T/ids.txt" "$T/r" 64 5 64
toy "$T/v.gguf" w65
refuse "fewer positions than a sliding layer's window rounded up (65 keys, 128 scanned)" kolibri-usage 1 "$B" inputs "$T/v.gguf" "$T/ids.txt" "$T/r" 64 5
refuse "more steps than positions" kolibri-usage 1 "$B" inputs "$T/toy.gguf" "$T/ids.txt" "$T/r" 64 65
refuse "a step more than an indexed layer sees whole" kolibri-context 1 "$B" inputs "$T/toy.gguf" "$T/ids.txt" "$T/r" 192 137
toy "$T/v.gguf" notop
refuse "no indexer.top_k in the metadata" kolibri-meta 1 "$B" inputs "$T/v.gguf" "$T/ids.txt" "$T/r" 64
# a window of 65 keys is scanned as 128, not 64: the mask is 128 rows of 128, and its last row has the 65 keys
toy "$T/v.gguf" w65
if "$B" inputs "$T/v.gguf" "$T/ids.txt" "$T/r" 128 5 > "$T/out.txt" 2> "$T/err.txt" && [ "$(wc -c < "$T/r/maskw.f32" | tr -d ' ')" = 65536 ] && [ "$(nums "$T/r/maskw.f32" | awk 'NR > 127 * 128 && $1 == 0' | wc -l | tr -d ' ')" = 65 ]; then echo "ok   a window of 65 keys: 128 scanned, 65 of them seen once the window is full"; else echo "FAIL a window of 65 keys: $(head -c 200 "$T/err.txt")"; fail=1; fi
toy "$T/v.gguf" v8
refuse "an embedding that is not a multiple of 64 rows" kolibri-tensor 1 "$B" weights "$T/v.gguf" "$T/r"
printf 'abc' > "$T/bad.f32"
refuse "an f32 file of three bytes" kolibri-read 1 "$B" argmax "$T/bad.f32"
: > "$T/bad.f32"
refuse "an empty f32 file" kolibri-read 1 "$B" argmax "$T/bad.f32"
printf '\000\000\300\177' > "$T/bad.f32"
refuse "a value that is not a number" kolibri-read 1 "$B" argmax "$T/bad.f32"
[ "$fail" -eq 0 ] && echo "kolibri_host: ok"
exit $fail
