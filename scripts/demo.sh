#!/bin/bash
# The terminal session behind docs/demo.mp4. Every command is run for real, on the machine it
# is recorded on; only the typing is simulated. Record, render and encode with scripts/make-demo.sh.
#
#     scripts/demo.sh        # needs a built compiler, the 27B runner and the model in Ollama's store
set -u
export TERM=${TERM:-xterm-256color}
cd "$(dirname "$0")/.."
cancho_gpu() { build/cancho-gpu "$@"; }
cancho_gpu_run35() { build/cancho-gpu-run35 "$@"; }
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

say() {            # a comment, typed
  printf '\033[2m'; type_text "# $1"; printf '\033[0m'
}
type_text() {
  local s=$1 i
  for ((i = 0; i < ${#s}; i++)); do printf '%s' "${s:i:1}"; sleep 0.028; done
  printf '\n'
}
run() {            # a command, typed and then run
  printf '\033[1;32m$\033[0m '; type_text "$1"; sleep 0.4
  eval "$2"
  sleep "${3:-2.5}"
  echo
}

clear
say "cancho-gpu: typed tile kernels, one source for CUDA C and Metal"
sleep 1.2
echo
run "cat swiglu.chx" "cat tests/chx/swiglu.chx" 4
run "cancho-gpu emit swiglu.chx out n=17408" "cancho_gpu emit tests/chx/swiglu.chx $OUT n=17408 | sed \"s#$OUT/##\"" 2.5
clear
say "what it writes for Apple GPUs (the CUDA C is written next to it)"
sleep 1
run "sed -n '/kernel void/,/^}/p' out/swiglu_17408.metal | head -22" "sed -n '/kernel void/,/^}/p' $OUT/swiglu_17408.metal | head -22" 4
clear
say "a wrong program is refused, with the rule it broke"
sleep 1
run "cat leak.chx" "sed -n '3,12p' tests/reject/leak.chx" 3
run "cancho-gpu check leak.chx n=8" "cancho_gpu check tests/reject/leak.chx n=8" 3.5
clear
say "and the kernels run a 27B model on this Mac's GPU"
sleep 1
run "cancho-gpu-run35 ~/.ollama/models kernels \"Q: Why is the sky blue?\\nA:\" 64 --stream --timing" \
    "cancho_gpu_run35 \$HOME/.ollama/models tests/chx \"\$(printf 'Q: Why is the sky blue?\nA:')\" 64 --stream --timing 2>&1 | grep -v '^tokens'" 6
