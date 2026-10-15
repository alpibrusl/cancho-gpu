#!/bin/sh
# Compare `cancho-gpu tokenize` with lex-rt's tokenizer on real tokenizers
# and real text (not in the gate: the tokenizers are tens of megabytes and
# the oracle takes about a minute to load the biggest).
#
#     scripts/tokenizer-real.sh <lex-gpu checkout> <tokenizer.json | model.gguf>...
#
# The oracle is a *release* build (a debug build is ten times slower):
#     cargo build --release -p lex-rt --example tokenizer_dump
# after copying tools/tokenizer_dump.rs to crates/lex-rt/examples/.
#
# The corpus is made here: prose and source from the repositories next to
# this one, cut into records of up to 3,000 characters at line ends, three
# pieces of very long text, and 3,000 random records over the scripts and
# marks the split and NFC care about (tests/tokenizer/make_fixtures.py,
# another seed). A tokenizer both programs refuse counts as agreeing.
set -eu
LEX_GPU=${1:?usage: tokenizer-real.sh <lex-gpu checkout> <tokenizer>...}
shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ORACLE=$LEX_GPU/target/release/examples/tokenizer_dump
[ -x "$ORACLE" ] || { echo "no release oracle at $ORACLE"; exit 2; }
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
python3 - "$ROOT" "$WORK/corpus.txt" <<'PY'
import glob, importlib.util, os, sys
root, out = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("fx", os.path.join(root, "tests", "tokenizer", "make_fixtures.py"))
fx = importlib.util.module_from_spec(spec); spec.loader.exec_module(fx)
files = sorted(glob.glob(os.path.join(root, "..", "cancho", "docs", "*.md")))[:80]
files += sorted(glob.glob(os.path.join(root, "..", "lex-gpu", "crates", "lex-rt", "src", "*.rs")))
files += sorted(glob.glob(os.path.join(root, "docs", "*.md"))) + sorted(glob.glob(os.path.join(root, "src", "*.cho")))[:12]
records = []
for path in files:
    text = open(path, encoding="utf-8", errors="replace").read().replace("\n%%\n", "\n")
    chunk = ""
    for line in text.split("\n"):
        if len(chunk) + len(line) > 3000 and chunk:
            records.append(chunk); chunk = ""
        chunk += line + "\n"
    if chunk:
        records.append(chunk)
longest = sorted(records, key=len)[-3:]
records += ["".join(longest) * 3]
random = fx.random_corpus(3000, seed=99).decode("utf-8")
open(out, "wb").write(("\n%%\n".join(records) + "\n%%\n" + random).encode("utf-8"))
print(f"{len(records)} text records and 3000 random ones, {os.path.getsize(out)} bytes")
PY
fail=0
for tok in "$@"; do
  a=0; b=0
  "$ROOT/build/cancho-gpu" tokenize "$tok" "$WORK/corpus.txt" > "$WORK/a" 2>"$WORK/a.err" || a=$?
  "$ORACLE" "$tok" "$WORK/corpus.txt" > "$WORK/b" 2>"$WORK/b.err" || b=$?
  name=$(echo "$tok" | sed 's|.*/models--||; s|/snapshots/[0-9a-f]*||')
  if [ "$a" != 0 ] && [ "$b" != 0 ]; then
    echo "both refuse  $name ($(head -1 "$WORK/a.err" | cut -c1-70))"
  elif [ "$a" = 0 ] && [ "$b" = 0 ] && cmp -s "$WORK/a" "$WORK/b"; then
    echo "identical    $name ($(head -1 "$WORK/a"), $(grep -c '^ids' "$WORK/a") records, $(grep '^ids' "$WORK/a" | cut -d' ' -f2 | tr -d ':' | paste -sd+ - | bc) tokens)"
  else
    echo "DIFFERENT    $name (cancho exit $a, oracle exit $b)"; fail=1
  fi
done
exit $fail
