#!/bin/sh
# What `cancho-gpu` may do is part of its contract (README, "Authority"):
# it reads and writes files (by path, or by offset through an open handle), prints, and allocates. It never touches the
# network, signals, other programs or foreign code, and the GGUF reader
# (src/gguf*.cho) added no capability. `cancho authority` says so; this
# holds the report to it, so a change that widens it fails the gate.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CANCHO=${CANCHO:-cancho}
cd "$ROOT"
"$CANCHO" authority --std src/*.cho > "${TMPDIR:-/tmp}/cancho-gpu-authority.$$" 2>&1
trap 'rm -f "${TMPDIR:-/tmp}/cancho-gpu-authority.$$"' EXIT
report=$(sed -n '/^performs/,/^provably pure/p' "${TMPDIR:-/tmp}/cancho-gpu-authority.$$" | sed '$d')
want='performs
    args
    err_write
    file_read
    fs_read("")
    fs_write("")
    heap
    io_write
never touches
    the network
    signals
    other programs
    foreign code'
if [ "$report" != "$want" ]; then
  echo "authority changed:"; echo "$report"; exit 1
fi
echo "authority: args, err_write, file_read, fs_read, fs_write, heap, io_write; no network, no foreign code"
