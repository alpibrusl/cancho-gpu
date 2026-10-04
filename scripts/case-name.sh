#!/bin/sh
# A case's directory name: the file's name without `.lx`, then its
# constants, joined by `_` -- `rmsnorm_n=4096_eps=1e-5`.
name=$(basename "$1" .lx)
shift
for a in "$@"; do name="${name}_$a"; done
echo "$name"
