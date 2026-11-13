"""D3 of docs/diffusion.md: the eight denoising steps of Qwen-Image-2.1-Turbo in float64, from D0's initial latents and sigmas: every step is the transformer of tools/dit_step_ref.py (front end, 32 blocks,
back end) and the flow-matching Euler update latents <- latents + (sigma_next - sigma) * prediction, the prediction being the image tokens' rows (the last 256 of 277) of the output.

    python tools/dit_traj_ref.py GOLDEN_DIR OUTDIR [--text TEXT.npy]

--text gives the text embeddings [21, 4096] (D6: the float64 encoder's, in place of torch's BF16 ones).

Writes OUTDIR/latents_f64.npy ([8, 256, 64]: the latents after each step) and OUTDIR/pred_f64.npy ([8, 256, 64]: each step's prediction), and prints, per step, the difference from torch's BF16 latents (D0's `latents_N`).
It takes about 3 to 6 minutes a step (the weights are read again for each step: 32 blocks in float64 do not fit in memory twice).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dit_block_ref as ref  # noqa: E402
import dit_step_ref as sr  # noqa: E402


def main():
    gold, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    ld = lambda n: np.load(os.path.join(gold, n + ".npy")).astype(np.float64)
    sigmas, timesteps = ld("sigmas"), ld("timesteps")
    lat = ld("step0_in")[0]
    text = np.load(sys.argv[sys.argv.index("--text") + 1]).astype(np.float64) if "--text" in sys.argv else None
    _, _, _, rope, tmask = ref.goldens(gold)
    lats, preds = [], []
    for i in range(len(timesteps)):
        t0 = time.time()
        x, temb, mod = sr.front(gold, lat, timesteps[i] / 1000.0, text)
        for b in range(32):
            w = {k: ref.load64("transformer_blocks.%d.%s" % (b, v)) for k, v in ref.BLOCK0.items()}
            x = ref.block(x, w, mod, rope, tmask)
        pred = sr.back(x, temb, tmask)[-256:]
        lat = lat + (sigmas[i + 1] - sigmas[i]) * pred
        lats.append(lat.copy())
        preds.append(pred.copy())
        want = ld("latents_%d" % i)[0]
        d = np.abs(lat - want)
        print("step %d (sigma %.6f -> %.6f): latents rms %.4g; against torch BF16: mean abs %.4g, max abs %.4g  (%.0f s)" % (i, sigmas[i], sigmas[i + 1], np.sqrt((lat ** 2).mean()), d.mean(), d.max(), time.time() - t0), flush=True)
        np.save(os.path.join(out, "latents_f64.npy"), np.stack(lats))
        np.save(os.path.join(out, "pred_f64.npy"), np.stack(preds))


if __name__ == "__main__":
    main()
