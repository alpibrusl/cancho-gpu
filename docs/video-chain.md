# Longer videos by chaining clips: what was tried (measured 2026-11-17)

> **Status: experiments, nothing built into `media`.** Claims are **[measured]** (run here, one picture, one prompt, one seed each, M4 Max) or **[read]** (papers, through summaries: the first part of each page was verbatim, the rest an extracted summary). The scripts were scratch files, not committed: the commands are described, not stored.

## 1. Question

`media video --image` makes 9 to 17 frames and degrades after about 8 (docs/video-image.md section 7). Can a longer video be made by carrying a clip's last frame into the next clip?

## 2. What the papers say [read]

Drift is error accumulation: the model is trained on clean context and generates from its own output (SVI, FramePack, LongLive, Self-Forcing++, Deep Forcing all say so); the symptoms named are blur, colour shift and over-saturation. **Every fix in these papers is training or cache management** (SVI: a LoRA on Wan2.1 I2V 14B that recycles the model's own errors; LongLive and Self-Forcing++: training on long self-rollouts, Wan2.1 1.3B; FramePack: context compression and planning from both ends, LoRA on HunyuanVideo 13B; Deep Forcing: training-free, but a cache of attention keys on a Self-Forcing model). **None uses colour correction or a different sampler shift**; those two come from community workflows only.

## 3. Setup [measured]

Start picture: the model-made fox (640 x 352). Segment 1: "the fox bolts out of the snow and runs toward the camera"; later ones: "the fox keeps running through the snow toward the camera". Base model, 30 steps, guidance 3.5, Wan's negative prompt, seeds 3, 4, 5, ... One segment of 9 frames takes about 52 s, of 17 about 78 s.

## 4. Results

| chain | frames | what happened |
|---|---|---|
| 3 x 9 frames, the last frame carried | 25 | the scene holds, the fox **does not run** (9 frames give it too little); mean red 127 to 110, saturation 18.5 to 26.4 |
| same, sampler shift 12 | 25 | **no measurable difference** from shift 5 (end 111/116/120, saturation 24.8) |
| 3 x 9, a global colour match (mean and spread per channel) of the carried frame to the picture | 25 | colour held (end 125/125/127, saturation 20.4) but a **jump at the first seam** (step 8.9 against 3.3): the one frame was corrected by about 11 levels at once |
| **3 x 17, the last frame carried** | 49 | **the fox runs and the next segment continues the run**; mean red falls to 99, saturation to 33; from about frame 36 the background smears and the tail gets a green patch |
| 3 x 17, **ramped** correction (frame j of n gets j/(n-1) of the correction; the last, fully matched, is carried) | 49 | the same motion; segment ends return to red 126-127 (sagging to about 121 inside a segment), saturation 26.2; no visible jump; **the smearing in the third segment remains** |
| latent chaining: segment k+1 holds two latent frames fixed (frame 4(lf-2) of the previous video re-encoded, plus its last latent frame as it is); `--cond` extended to K frames in a scratch edit, since reverted | 53 | **collapse**: mean red 127 to 75, saturation 18 to 46, colour garbage from about frame 44, seam steps 26 to 41 (median 12). The conditioned frames themselves reproduce the previous segment (relative rms 4 to 14%): what fails is what the model does after them |
| 6 x 13 frames, each starting from the previous segment's frame 8 (single-frame condition), the 5 overlapping frames cross-faded, ramped correction | 53 | **collapse from the second segment** (garbage from about frame 16): the carried frame 8 of the first segment had an artefact (a purple blob at the top) and the model amplified it |

## 5. Reading

* **What works**: carrying the last frame of a 17-frame clip, with the ramped correction. Two segments (about 33 frames, 2 s) are clean; the third degrades. [measured, one picture]
* **What does not**: the shift (no effect), the correction applied to one frame (a jump), several latent frames held fixed (the model was trained with one conditioned first frame; the rest of its input is not what it saw), and an earlier carried frame (early frames of a clip carry the model's artefacts; the **last** frame of a 17-frame clip was clean in every run here).
* **Most of the colour drift happens inside a segment**, not across the chain: the red value falls from 127 to about 117 within the first four generated frames of the first segment.
* **Not tried**: a first-picture anchor for later segments (the papers' sink and FramePack's first-frame target need a model trained for it), segments longer than 17 frames (the token limit of `media video`), other pictures, prompts and seeds, UniPC, a tuned correction strength.

## 6. Built (2026-11-17)

`media video --image PIC --segments N [--then PROMPT]... [--no-correct]` (docs/cli.md): the 17-frame chain with the ramped correction of section 4, limited to two clips unless `--force`. The correction is in `tools/media.py` (`ramp_correct`; PIL and numpy, no new file). **Run once**: 2 x 17 frames of the fox, 33 frames, 3 min 1 s; the first clip equals the scratch chain's (0.4 of 255 at frame 16), the second runs the fox on (up to 8.6 of 255 from the scratch chain's second clip, which began from a frame with a 0.4 difference: the GPU is not bit-for-bit repeatable). Refusals checked: no `--image`, `--segments 0`, `--segments 3` without `--force`, more `--then` than clips. Not checked: `--no-correct`, `--force` with three clips, a picture other than the fox, the studio (it does not offer chaining).
