#!/bin/sh
# Builds the session writer in cancho (tools/qsw, docs/qsw.md) into build/qsw, or into $1.   CANCHO=path/to/cancho scripts/build-qsw.sh
set -eu
cd "$(dirname "$0")/.."
C="${CANCHO:-cancho}"
mkdir -p build
"$C" build tools/qsw/rd.cho tools/qsw/enc.cho tools/qsw/tpl.cho tools/qsw/cfg.cho tools/qsw/kern.cho tools/qsw/data.cho tools/qsw/session.cho tools/qsw/split.cho tools/qsw/main.cho --std -o "${1:-build/qsw}" 2>&1 | grep -v "ld: warning" || true
[ -x "${1:-build/qsw}" ] || { echo "FAIL build qsw"; exit 1; }
"$C" build tools/qsw/rd.cho tools/qsw/enc.cho tools/qsw/probe.cho --std -o "$(dirname "${1:-build/qsw}")/qsw_probe" 2>&1 | grep -v "ld: warning" || true
