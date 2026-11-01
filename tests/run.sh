#!/bin/sh
# Gates 3 and 4 of docs/run.md: `cancho-gpu run` gives the known answer on real
# inputs, and every refusal of its section 3 has its rule tag, exit status 1 and
# nothing on standard output. A refusal that has no case here is not tested:
# keep the list in step with the section.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${CANCHO_GPU:-$ROOT/build/cancho-gpu}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
K=$ROOT/tests/chx/bwd_dx.chx
fail=0

# dy [2,2] = 1 2 / 3 4, w [2,3] = 1 0 2 / 0 1 3, so dx = dy w = 1 2 8 / 3 4 18
printf '1 2\n3 4\n' > "$W/dy"
printf '1 0 2\n0 1 3\n' > "$W/w"
printf '1 2 3\n' > "$W/three"
printf '1 2 3 x\n' > "$W/nan"
printf '1 2 3 4 5\n' > "$W/five"
printf -- '1 +2\n3 4\n' > "$W/plus"

# a copy of bwd_dx with a schedule for apple-m-series only
grep -v 'nvidia-ada' "$K" > "$W/one.chx"

ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; fail=1; }

# Gate 3: real inputs, both targets, the known answer (a negative sign, an exponent).
for tg in nvidia-ada apple-m-series; do
  got=$("$BIN" run "$K" r=2 n=2 k=3 --target $tg --in dy "$W/dy" --in w "$W/w" --out dx 2>&1) || got="exit $?: $got"
  if [ "$got" = "1.0 2.0 8.0 3.0 4.0 18.0" ]; then ok "known answer [$tg]"; else bad "known answer [$tg]: $got"; fi
done
printf -- '-1.5e0 2\n3 4\n' > "$W/neg"
got=$("$BIN" run "$K" r=2 n=2 k=3 --target nvidia-ada --in dy "$W/neg" --in w "$W/w" --out dx 2>&1) || got="exit $?: $got"
if [ "$got" = "-1.5 2.0 3.0 3.0 4.0 18.0" ]; then ok "a signed exponent is read"; else bad "signed exponent: $got"; fi
# two --out: one line each, in the order given
got=$("$BIN" run "$K" r=2 n=2 k=3 --target nvidia-ada --in dy "$W/dy" --in w "$W/w" --out dx --out dx 2>&1) || got="exit $?: $got"
want="1.0 2.0 8.0 3.0 4.0 18.0
1.0 2.0 8.0 3.0 4.0 18.0"
if [ "$got" = "$want" ]; then ok "two --out, a line each"; else bad "two --out: $got"; fi

# Gate 4: each refusal. refuse <rule> <label> <args...>
refuse() {
  rule=$1; label=$2; shift 2
  status=0
  "$BIN" run "$@" >"$W/out" 2>"$W/err" || status=$?
  if [ "$status" -ne 1 ]; then bad "$label: exit $status, expected 1"
  elif ! grep -q "^error\[$rule\]" "$W/err"; then bad "$label: expected error[$rule], got: $(head -1 "$W/err")"
  elif [ -s "$W/out" ]; then bad "$label: refused but printed $(head -c 40 "$W/out")"
  else ok "refused: $label ($rule)"; fi
}
T="--target nvidia-ada"
refuse usage "a read-only parameter not given" "$K" r=2 n=2 k=3 $T --in dy "$W/dy" --out dx
refuse usage "no --out" "$K" r=2 n=2 k=3 $T --in dy "$W/dy" --in w "$W/w"
refuse usage "too few numbers" "$K" r=2 n=2 k=3 $T --in dy "$W/three" --in w "$W/w" --out dx
refuse usage "too many numbers" "$K" r=2 n=2 k=3 $T --in dy "$W/five" --in w "$W/w" --out dx
refuse usage "a token that is not a number" "$K" r=2 n=2 k=3 $T --in dy "$W/nan" --in w "$W/w" --out dx
refuse usage "an explicit plus sign" "$K" r=2 n=2 k=3 $T --in dy "$W/plus" --in w "$W/w" --out dx
refuse usage "--in names an unknown parameter" "$K" r=2 n=2 k=3 $T --in nosuch "$W/dy" --in dy "$W/dy" --in w "$W/w" --out dx
refuse usage "--in names a writable parameter" "$K" r=2 n=2 k=3 $T --in dx "$W/dy" --in dy "$W/dy" --in w "$W/w" --out dx
refuse usage "--in given twice" "$K" r=2 n=2 k=3 $T --in dy "$W/dy" --in dy "$W/dy" --in w "$W/w" --out dx
refuse usage "--out names a read-only parameter" "$K" r=2 n=2 k=3 $T --in dy "$W/dy" --in w "$W/w" --out dy
refuse usage "--out names an unknown parameter" "$K" r=2 n=2 k=3 $T --in dy "$W/dy" --in w "$W/w" --out nosuch
refuse usage "--in without its file" "$K" r=2 n=2 k=3 $T --in dy
refuse usage "--out without its name" "$K" r=2 n=2 k=3 $T --in dy "$W/dy" --in w "$W/w" --out
refuse usage "--target without its value" "$K" r=2 n=2 k=3 --target
refuse usage "an unknown target" "$K" r=2 n=2 k=3 --target gfx --in dy "$W/dy" --in w "$W/w" --out dx
refuse usage "two schedules and no --target" "$K" r=2 n=2 k=3 --in dy "$W/dy" --in w "$W/w" --out dx
refuse usage "a constant that is not name=value" "$K" r=2 n=2 k=3 junk $T --in dy "$W/dy" --in w "$W/w" --out dx
refuse io "an input file that does not exist" "$K" r=2 n=2 k=3 $T --in dy "$W/absent" --in w "$W/w" --out dx
refuse io "a source file that does not exist" "$W/absent.chx" $T
refuse no-schedule "a target the file has no schedule for" "$W/one.chx" r=2 n=2 k=3 --target nvidia-ada --in dy "$W/dy" --in w "$W/w" --out dx
exit $fail
