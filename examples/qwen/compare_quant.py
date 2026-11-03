#!/usr/bin/env python3
"""The forward exits of roadmap M7.5 (docs/resident.md section 27): the forward pass of a 4-bit base in a Metal session against the same weights dequantised on the host and run through the unchanged f16 session,
and against transformers (float32 on the CPU) on those dequantised weights.

    python3 examples/qwen/compare_quant.py QWEIGHTS HFDIR [--fake FWEIGHTS] [--nb 8] [--reps 3] [--metal build/cancho-gpu-metal-session]

QWEIGHTS: a weights directory with `.q/.s/.g` matrices (convert.py --quant, quant.py pack); HFDIR: hf_reference.py's answers on the same weights (`hf_reference.py HFDIR --base QWEIGHTS`); FWEIGHTS: the dequantised
f16 directory (quant.py dequant), when the check against the f16 session is wanted. Prints the loss of the fixed batch from each, the arg max agreement and the time of the forward pass."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import data  # noqa: E402
import qwen_ref as R  # noqa: E402
import qwen_session as S  # noqa: E402

arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


def forward(binary, work, c, wdir, tok, reps, label):
    sess = S.write(os.path.join(work, label), c, wdir, tok, acts=os.path.join(work, label, "acts.bin"))
    ss = S.Session(binary, sess)
    loss, am, first = ss.forward(c.n)
    times = [ss.forward(c.n)[2] for _ in range(reps)]
    ss.checkpoint()
    ss.close()
    return R.mean_loss(loss, c), loss, am, sorted(times)[len(times) // 2] * 1000, first * 1000


def main():
    qdir, hdir = sys.argv[1], sys.argv[2]
    nb, reps = int(arg("--nb", 8)), int(arg("--reps", 3))
    binary = arg("--metal", os.path.join(ROOT, "build", "cancho-gpu-metal-session"))
    c = R.config_of(qdir, 512, nb)
    tok = data.fixed_batch(c.nb, c.t)
    assert (np.load(os.path.join(hdir, "tokens.npy")) == tok).all(), "the reference was made on another batch"
    work = os.path.join(os.path.dirname(os.path.abspath(hdir)), "compare_quant_" + os.path.basename(os.path.normpath(qdir)))  # no ".." in a path: the session's Fs traps on it
    os.makedirs(work, exist_ok=True)
    hl = float(np.load(os.path.join(hdir, "loss.npy"))[0])
    href = np.load(os.path.join(hdir, "argmax.npy"))
    margin = np.load(os.path.join(hdir, "margin.npy"))
    ql, qrows, qam, qms, qfirst = forward(binary, work, c, qdir, tok, reps, "quant")
    print(f"4-bit base, session (dequantise kernel + f16 products): loss {ql:.6f}; forward {qms:.1f} ms ({c.n / qms * 1000:.0f} tokens a second; first request {qfirst:.0f} ms)")
    print(f"transformers float32 on the dequantised weights:        loss {hl:.6f}; difference {ql - hl:+.2e} (exit: within 2e-3)")
    print(f"arg max equal to transformers' at {(qam == href).mean() * 100:.2f}% of {c.n} positions ({int((qam != href).sum())} differ; the gap between transformers' two largest logits there: "
          f"{[round(float(x), 4) for x in margin[qam != href]][:12]})")
    if "--fake" in sys.argv:
        fdir = arg("--fake", None)
        fl, frows, fam, fms, ffirst = forward(binary, work, c, fdir, tok, reps, "fake")
        print(f"dequantised on the host, f16 session (the specification of the 4-bit base): loss {fl:.6f}; forward {fms:.1f} ms")
        print(f"4-bit session against it: loss {ql - fl:+.2e} (exit: within 2e-3); largest per-position difference {np.abs(qrows - frows).max():.2e}; arg max equal at {(qam == fam).mean() * 100:.2f}%; forward time {qms / fms:.3f} times")


if __name__ == "__main__":
    main()
