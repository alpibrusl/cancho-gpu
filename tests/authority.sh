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
# `cancho-gpu-session` (session/main.cho) reads standard input, which `cancho-gpu` may not. It also writes files, for the `save` directives of its session file
# on the request `checkpoint` (docs/resident.md section 19): that `fs_write` and `file_write` were added to its report is the owner's decision (docs/resident.md section 19,
# "The authority"), and a change to this list is made the same way.
"$CANCHO" authority --std $(ls src/*.cho | grep -v '/main\.cho$') session/main.cho > "${TMPDIR:-/tmp}/cancho-gpu-session-authority.$$" 2>&1
trap 'rm -f "${TMPDIR:-/tmp}/cancho-gpu-authority.$$" "${TMPDIR:-/tmp}/cancho-gpu-session-authority.$$"' EXIT
report=$(sed -n '/^performs/,/^provably pure/p' "${TMPDIR:-/tmp}/cancho-gpu-session-authority.$$" | sed '$d')
want='performs
    args
    err_write
    file_read
    file_write
    fs_read("")
    fs_write("")
    heap
    io_read
    io_write
never touches
    the network
    signals
    other programs
    foreign code'
if [ "$report" != "$want" ]; then
  echo "the session program's authority changed:"; echo "$report"; exit 1
fi
echo "authority: args, err_write, file_read, fs_read, fs_write, heap, io_write; no network, no foreign code"
echo "authority of cancho-gpu-session: args, err_write, file_read, file_write, fs_read, fs_write (for save, on checkpoint; owner-approved), heap, io_read, io_write; no network, no foreign code"
