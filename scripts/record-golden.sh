#!/bin/sh
# Record golden/ from this compiler for every case of tests/cases.txt: the files it writes in golden/<case>/ (plus RUN.txt for a
# `--run` case, the lines the interpreter printed), or golden/<case>/REFUSED beside any files it wrote before refusing (exit 1).
#
#     scripts/record-golden.sh
#
# These are regression goldens, not an oracle (docs/native.md N2): they say what this compiler wrote, and `git diff golden/` after
# a change is the thing to read. A diff nobody can explain is a bug; one that is explained (a new lowering, a better schedule) is
# committed with the change that causes it. What a kernel *computes* is held to float64 references by tests/kernels.py, which
# does not depend on these files.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${CANCHO_GPU:-$ROOT/build/cancho-gpu}
# Only the directories of this file's cases are rebuilt: the other suites' goldens (gguf, safetensors, tokenizer, sampler) have
# their own scripts. A case removed from tests/cases.txt leaves its directory to be deleted by hand.
while read -r file rest; do
  case "$file" in ''|'#'*) continue ;; esac
  dir=$ROOT/golden/$("$ROOT/scripts/case-name.sh" "$file" $rest)
  rm -rf "$dir"
  mkdir -p "$dir"
  status=0
  # shellcheck disable=SC2086
  "$BIN" emit "$ROOT/$file" "$dir" $rest >"$dir/.stdout" 2>/dev/null || status=$?
  case " $rest " in *" --run "*) grep -v "^$dir/" "$dir/.stdout" > "$dir/RUN.txt" || true ;; esac
  rm -f "$dir/.stdout"
  if [ "$status" -eq 1 ]; then
    touch "$dir/REFUSED"
  elif [ "$status" -ne 0 ]; then
    echo "$file $rest: exit $status (a trap or a crash is a bug, not a golden)"; exit 1
  fi
done < "$ROOT/tests/cases.txt"
echo "golden/ recorded from this compiler: review with git diff golden/"
