#!/bin/sh
# The gate: everything that must pass before a change is called done.
#
#     LEX_SYS=<path to the pinned lex-sys> scripts/gate.sh
#
# Canonical formatting, the file budget, the build, the unit tests, the
# goldens (byte-identical with lex-gpu's Rust emitter) and the refusal
# fixtures. The live differential against a lex-gpu checkout
# (scripts/differential.sh) is not in it: it needs cargo and lex-gpu,
# and the goldens are its recorded answer.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LEX_SYS=${LEX_SYS:-lex-sys}
cd "$ROOT"
"$LEX_SYS" fmt --check src tests
tests/files.sh
"$LEX_SYS" build
"$LEX_SYS" test
tests/golden.sh
tests/reject.sh
echo "gate: ok"
