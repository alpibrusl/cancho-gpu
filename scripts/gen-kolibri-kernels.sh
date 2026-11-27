#!/bin/sh
# scripts/gen-kolibri-kernels.sh [DIR] -- the kernels of Kolibri's pass that have as many parameters as a token has experts (K) or a pass has tokens (B), which a kernel file
# cannot say (docs/kolibri.md section 20): written here for K = 2 and 6 and B = 4 and 16, the gate's small model and Kolibri 1.1, into DIR (tests/chx).
#     expertK_gu.chx      the K experts' gate and up products of one row and the activation, a launch
#     expertK_down.chx    their down products (Q4_0), weighed and added to the row of the shared expert's output, a launch
#     expertK_down41.chx  the same where `down` is Q4_1 (layers 2 to 5): the offsets' product with the activations' group sums added
#     rows_catB.chx       B rows as one matrix
# tests/kstep.sh holds the files in tests/chx to what this script writes.
set -eu
D=${1:-$(cd "$(dirname "$0")/.." && pwd -P)/tests/chx}
sched() { printf '\nschedule %s for apple-m-series { threads %s%s }\nschedule %s for nvidia-ada     { threads %s%s }\nschedule %s for amd-cdna3      { threads %s%s }\n' "$1" "$2" "$3" "$1" "$2" "$3" "$1" "$2" "$3"; }
for K in 2 6; do
  {
    echo "// Written by scripts/gen-kolibri-kernels.sh: do not edit. The $K experts a token takes, each expert's gate and up products and the activation, as one launch (docs/kolibri.md section 20):"
    echo "// y[e] = silu(x ge^T) * (x ue^T) with Q4_0 weights (matvec_q4_rows.chx's product), x row b of r rows. The $((2 * K)) matrices are tables the session picks by the router's choice."
    echo "algo expert${K}_gu(r, b, n, k)"
    echo "  in  x:   f32[r, k]"
    e=0; while [ $e -lt $K ]; do echo "  in  gq${e}: i8[n, k / 2]"; echo "  in  gs${e}: f16[n, k / 32]"; echo "  in  uq${e}: i8[n, k / 2]"; echo "  in  us${e}: f16[n, k / 32]"; e=$((e + 1)); done
    echo "  in  wg:  f32[n]"
    echo "  out y:   f32[${K}, n]"
    echo "  tile bn"
    echo "{"
    echo "  grid i over n / bn"
    e=0; while [ $e -lt $K ]; do
      cat <<EOF
  let x${e} = load x[b, 0 ; 1, k]
  let a${e} = load gq${e}[i * bn, 0 ; bn, k / 2]
  let b${e} = load gs${e}[i * bn, 0 ; bn, k / 32]
  let c${e} = load wg[i * bn ; bn]
  let d${e} = load uq${e}[i * bn, 0 ; bn, k / 2]
  let e${e} = load us${e}[i * bn, 0 ; bn, k / 32]
  let f${e} = load wg[i * bn ; bn]
  let w${e} = dequant_q4 a${e} b${e} c${e} 32
  let v${e} = dequant_q4 d${e} e${e} f${e} 32
  let g${e} = matmul_nt &x${e} w${e}
  let u${e} = matmul_nt x${e} v${e}
  let s${e} = sigmoid &g${e}
  let t${e} = g${e} * s${e}
  store t${e} * u${e} -> y[${e}, i * bn]
EOF
      e=$((e + 1)); done
    echo "}"
    sched "expert${K}_gu" 256 " bn 8"
  } > "$D/expert${K}_gu.chx"
  for form in 0 41; do
    name="expert${K}_down"; [ $form = 0 ] || name="expert${K}_down41"
    {
      if [ $form = 0 ]; then
        echo "// Written by scripts/gen-kolibri-kernels.sh: do not edit. The $K experts' down products of one token, weighed by the router's weights and added to the token's row of the shared"
        echo "// expert's output, as one launch (docs/kolibri.md section 20): mo = m[b] + sum_e wt[b K + e] (a[e] de^T), Q4_0 weights. a is expert${K}_gu.chx's output."
      else
        echo "// Written by scripts/gen-kolibri-kernels.sh: do not edit. expert${K}_down.chx where \`down\` is Q4_1 (docs/kolibri.md section 10: code d + m is (code - 8) d + (m + 8 d)): to each"
        echo "// expert's Q4_0 product is added s[e] ce^T, s the sums of a[e] over its groups of 32 (group_sum.chx) and ce the group offsets m + 8 d the host writes."
      fi
      echo "algo ${name}(r, b, n, k)"
      echo "  in  a:   f32[${K}, k]"
      [ $form = 0 ] || echo "  in  s:   f32[${K}, k / 32]"
      e=0; while [ $e -lt $K ]; do echo "  in  dq${e}: i8[n, k / 2]"; echo "  in  ds${e}: f16[n, k / 32]"; [ $form = 0 ] || echo "  in  dc${e}: f32[n, k / 32]"; e=$((e + 1)); done
      echo "  in  wg:  f32[n]"
      echo "  in  wt:  f32[r * ${K}, 1]"
      echo "  in  m:   f32[r, n]"
      echo "  out mo:  f32[1, n]"
      echo "  tile bn"
      echo "{"
      echo "  grid i over n / bn"
      echo "  let acc0 = load m[b, i * bn ; 1, bn]"
      e=0; while [ $e -lt $K ]; do
        cat <<EOF
  let x${e} = load a[${e}, 0 ; 1, k]
  let q${e} = load dq${e}[i * bn, 0 ; bn, k / 2]
  let h${e} = load ds${e}[i * bn, 0 ; bn, k / 32]
  let g${e} = load wg[i * bn ; bn]
  let w${e} = dequant_q4 q${e} h${e} g${e} 32
EOF
        if [ $form = 0 ]; then echo "  let p${e} = matmul_nt x${e} w${e}"; else
          cat <<EOF
  let y${e} = matmul_nt x${e} w${e}
  let z${e} = load s[${e}, 0 ; 1, k / 32]
  let o${e} = load dc${e}[i * bn, 0 ; bn, k / 32]
  let p${e} = y${e} + matmul_nt z${e} o${e}
EOF
        fi
        cat <<EOF
  let c${e} = load wt[b * ${K} + ${e}, 0 ; 1, 1]
  let r${e} = p${e} * c${e}
  let acc$((e + 1)) = acc${e} + r${e}
EOF
        e=$((e + 1)); done
      echo "  store acc${K} -> mo[0, i * bn]"
      echo "}"
      sched "$name" 256 " bn 8"
    } > "$D/$name.chx"
  done
done
for B in 4 16; do
  {
    echo "// Written by scripts/gen-kolibri-kernels.sh: do not edit. $B rows of n as one matrix (docs/kolibri.md section 20): what each of a pass's tokens computed alone (its embedding row,"
    echo "// its experts' sum), for the kernels that take the pass's rows together. bn columns a threadgroup."
    echo "algo rows_cat${B}(n, bn)"
    b=0; while [ $b -lt $B ]; do echo "  in  y${b}: f32[1, n]"; b=$((b + 1)); done
    echo "  out o:  f32[${B}, n]"
    echo "{"
    echo "  grid c over n / bn"
    b=0; while [ $b -lt $B ]; do echo "  let t${b} = load y${b}[0, c * bn ; 1, bn]"; echo "  store t${b} -> o[${b}, c * bn]"; b=$((b + 1)); done
    echo "}"
    sched "rows_cat${B}" 128 ""
  } > "$D/rows_cat${B}.chx"
done
