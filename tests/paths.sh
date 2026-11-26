#!/bin/sh
# A path the file verbs stop the program on is a refusal, not a crash (docs/runtime.md section 11). cancho's `Fs` traps on a path with two dots in a row
# anywhere in it and on one of 4,096 bytes or more: the program dies of an illegal instruction (status 132) with nothing said. Every open of a path a caller
# gave therefore asks `fs_covers` first (`fileio.covers`). Here each such open is reached with a path that has `..` in it, and one with a path of 5,000 bytes:
# the program must end with a status under 128 and a first line on standard error that carries a rule tag.
#
#     CANCHO=<pinned cancho> tests/paths.sh
#
# PATHS_BUILD names another directory of programs (the mutants: a guard taken out must fail a case here), and then nothing is built. The Metal programs are
# tried where they are built (a Mac) and passed over, said, where they are not.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
B=${PATHS_BUILD:-$ROOT/build}
if [ -z "${PATHS_BUILD:-}" ]; then
  (cd "$ROOT" && CANCHO="${CANCHO:-cancho}" scripts/build-kolibri.sh > /dev/null && CANCHO="${CANCHO:-cancho}" scripts/build-qsw.sh > /dev/null && CANCHO="${CANCHO:-cancho}" scripts/build-session.sh > /dev/null && CANCHO="${CANCHO:-cancho}" scripts/build-device.sh > /dev/null) || { echo "FAIL paths: build"; exit 1; }
fi
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
fail=0
n=0
mkdir -p "$T/a" "$T/w" "$T/o" "$T/x" "$T/want"
P=$T/a/../gone
# 5,000 bytes of path with no two dots in a row
LONG=$T/$(awk 'BEGIN { for (i = 0; i < 500; i++) printf "aaaaaaaaa/"; }')f

# name, tag pattern, the command: a status under 128, and a first line of standard error with the tag
refused() {
  name=$1; tag=$2; shift 2
  n=$((n + 1))
  "$@" > "$T/out" 2> "$T/err" < /dev/null
  rc=$?
  if [ "$rc" -ge 128 ] || [ "$rc" -eq 0 ]; then echo "FAIL $name: status $rc"; fail=1; return; fi
  if ! head -1 "$T/err" | grep -q -e "$tag"; then echo "FAIL $name: status $rc, but the refusal is \`$(head -1 "$T/err" | cut -c1-120)\`, not $tag"; fail=1; return; fi
  echo "ok   $name"
}
# name, the command: it must succeed (the fixture is good, so the case beside it is refused for its path and nothing else)
works() {
  name=$1; shift
  n=$((n + 1))
  if "$@" > "$T/out" 2> "$T/err" < /dev/null; then echo "ok   $name"; else echo "FAIL $name: status $?: $(head -1 "$T/err" | cut -c1-120)"; fail=1; fi
}

# --- cancho-gpu
G=$B/cancho-gpu
K=$ROOT/tests/chx/bwd_dx.chx
printf '1 2\n3 4\n' > "$T/dy"
printf '1 0 2\n0 1 3\n' > "$T/w.txt"
printf '1 2 3\n' > "$T/ids"
works "cancho-gpu run: the fixture" "$G" run "$K" r=2 n=2 k=3 --target nvidia-ada --in dy "$T/dy" --in w "$T/w.txt" --out dx
works "cancho-gpu emit: the fixture" "$G" emit "$K" "$T/o" r=2 n=2 k=3
for c in emit check run gguf model-expand; do
  refused "cancho-gpu $c: a file with .." "error\[io\]" "$G" $c "$P" "$T/o"
done
for c in safetensors chat; do
  refused "cancho-gpu $c: a file with .." "error\[io\]" "$G" $c "$P"
done
refused "cancho-gpu emit: a file of 5,000 bytes of path" "error\[io\]" "$G" emit "$LONG" "$T/o"
refused "cancho-gpu emit: a directory with .." "error\[io\]" "$G" emit "$K" "$T/a/../o" r=2 n=2 k=3
refused "cancho-gpu run: an input with .." "error\[io\]" "$G" run "$K" r=2 n=2 k=3 --target nvidia-ada --in dy "$T/a/../dy" --in w "$T/w.txt" --out dx
refused "cancho-gpu tokenize: a tokenizer with .." "error\[io\]" "$G" tokenize "$P" "$T/ids"
refused "cancho-gpu detokenize: a tokenizer with .." "error\[io\]" "$G" detokenize "$P" "$T/ids"
refused "cancho-gpu sample: logits with .." "error\[io\]" "$G" sample "$P" 4 1 1 1 1

# --- cancho-gpu-session: the session file, and the files it names
S=$B/cancho-gpu-session
printf '1 2\n3 4\n' > "$T/dy"
sess() { printf 'program b %s r=2 n=2 k=3\nresident b.dy %s\nresident b.w %s\nanswer b.dx\n' "$1" "$2" "$3" > "$T/s.session"; }
sess "$K" "$T/dy" "$T/w.txt"
n=$((n + 1))
if [ "$(printf '\n' | "$S" "$T/s.session" 2> "$T/err" | tr -s ' \n' ' ')" = "1e0 2e0 8e0 3e0 4e0 1.8e1 " ]; then echo "ok   cancho-gpu-session: the fixture"; else echo "FAIL cancho-gpu-session: the fixture: $(head -1 "$T/err")"; fail=1; fi
refused "cancho-gpu-session: a session file with .." "error\[io\]" "$S" "$P"
sess "$T/a/../k.chx" "$T/dy" "$T/w.txt"
refused "cancho-gpu-session: a program with .." "error\[session\]" "$S" "$T/s.session"
sess "$K" "$T/a/../dy" "$T/w.txt"
refused "cancho-gpu-session: a resident with .." "error\[session\]" "$S" "$T/s.session"
printf 'program b %s r=2 n=2 k=3\nresident b.w %s\ntable b.dy %s 1\nanswer b.dx\n' "$K" "$T/w.txt" "$T/a/../dy" > "$T/s.session"
refused "cancho-gpu-session: a table with .." "error\[" "$S" "$T/s.session"

# --- the device programs: the kernel file they are given
if [ -x "$B/cancho-gpu-device" ]; then refused "cancho-gpu-device: a file with .." "error\[io\]" "$B/cancho-gpu-device" run "$P"; else echo "FAIL cancho-gpu-device is not built"; fail=1; fi
for d in cancho-gpu-metal-device; do
  if [ -x "$B/$d" ]; then refused "$d: a file with .." "error\[io\]" "$B/$d" run "$P"; else echo "     $d is not built here: not tried"; fi
done
if [ -x "$B/cancho-gpu-metal-session" ]; then refused "cancho-gpu-metal-session: a session file with .." "error\[io\]" "$B/cancho-gpu-metal-session" "$P"; else echo "     cancho-gpu-metal-session is not built here: not tried"; fi

# --- kolibri_gpu
KG=$B/kolibri_gpu
. "$ROOT/tests/kolibri_toy.sh"
toy "$T/toy.gguf" none
printf 'hi' > "$T/user.txt"
printf '1 2 3\n' > "$T/kids.txt"
works "kolibri_gpu weights: the fixture" "$KG" weights "$T/toy.gguf" "$T/w"
refused "kolibri_gpu prompt: a text with .." "error\[kolibri-read\]" "$KG" prompt "$P" none
refused "kolibri_gpu prompt: a system text with .." "error\[kolibri-read\]" "$KG" prompt "$T/user.txt" none "$P"
refused "kolibri_gpu ends: a file with .." "error\[kolibri-read\]" "$KG" ends "$P"
refused "kolibri_gpu say: a file with .." "error\[kolibri-read\]" "$KG" say "$P"
refused "kolibri_gpu weights: a directory with .." "error\[kolibri-write\]" "$KG" weights "$T/toy.gguf" "$T/a/../w"
refused "kolibri_gpu inputs: an ids file with .." "error\[kolibri-" "$KG" inputs "$T/toy.gguf" "$T/a/../kids.txt" "$T/x" 64 3
refused "kolibri_gpu inputs: a directory with .." "error\[kolibri-write\]" "$KG" inputs "$T/toy.gguf" "$T/kids.txt" "$T/a/../x" 64 3

# --- qsw: a model of one layer, 64 wide, with files of zeros
Q=$B/qsw
printf 'layers 1\nd 64\nh 2\nhkv 2\ndh 32\nff 64\nv 16\nt 64\nnb 1\ntied 1\nr 16\nsteps 1\neps 4517329193108106637\ntheta 4666723172467343360\n' > "$T/c.txt"
mkdir -p "$T/qw" "$T/qo"
head -c 2048 /dev/zero > "$T/qw/emb.f16"
head -c 256 /dev/zero > "$T/tok"
head -c 114688 /dev/zero > "$T/ad"
head -c 90112 /dev/zero > "$T/gr"
qw() { cfg=$1; wd=$2; od=$3; tk=$4; shift 4; "$Q" write --config "$cfg" --wdir "$wd" --out "$od" --root "$ROOT" --tokens "$tk" --adapters "$T/ad" "$@"; }
works "qsw write: the fixture" qw "$T/c.txt" "$T/qw" "$T/qo" "$T/tok"
works "qsw split: the fixture" "$Q" split --config "$T/c.txt" --grads "$T/gr" --out "$T/qo/split"
refused "qsw write: a configuration with .." "qsw: qsw\.input" qw "$T/a/../c.txt" "$T/qw" "$T/qo" "$T/tok"
refused "qsw write: tokens with .." "qsw: qsw\.input" qw "$T/c.txt" "$T/qw" "$T/qo" "$T/a/../tok"
refused "qsw write: weights with .." "qsw: qsw\." qw "$T/c.txt" "$T/a/../qw" "$T/qo" "$T/tok"
refused "qsw write: an output directory with .." "qsw: qsw\." qw "$T/c.txt" "$T/qw" "$T/a/../qo" "$T/tok"
refused "qsw write: a root with .." "qsw: qsw\.kernel" "$Q" write --config "$T/c.txt" --wdir "$T/qw" --out "$T/qo" --root "$ROOT/tests/.." --tokens "$T/tok" --adapters "$T/ad"
refused "qsw split: gradients with .." "qsw: qsw\.input" "$Q" split --config "$T/c.txt" --grads "$T/a/../gr" --out "$T/qo/split"
refused "qsw split: an output with .." "qsw: qsw\.write" "$Q" split --config "$T/c.txt" --grads "$T/gr" --out "$T/a/../split"

if [ $fail -ne 0 ]; then echo "FAIL paths"; exit 1; fi
echo "paths: ok ($n checks)"
