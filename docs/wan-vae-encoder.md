# The Wan VAE encoder, in cancho (design, 2026-11-16)

> **Status: the encoder for one frame is built and measured (sections 8 to 10): E1 the float64 reference, E2 and E3 the GPU session; the sampler's use of its latent is section 11 (it holds frame 0; the distilled model does not continue from it) and section 12 (the base model's sampler does, and is held to torch).** This document began as a design before any code and says, in place, where building corrected it. Issue [#174](https://github.com/alpibrusl/cancho-gpu/issues/174) (image to video) needs the VAE *encoder* of `Wan2.2-TI2V-5B`; only the decoder exists. Claims are tagged **[read]** (from the source or the checkpoint, not run), **[measured]** (run, with the command) or **[estimate]**. The owner chose, on 2026-11-16, to build it **native in cancho** (no new Python, no new Rust): the session as a `.model` file, the weights written by a cancho command, the float64 reference a cancho program. This is the first of a chain; each slice is its own PR and holds a measured result before the next starts.

## 1. What the encoder is

**[read]** `autoencoder_kl_wan.py` of diffusers 0.35 in the venv of the first image edits (`WanEncoder3d`, `WanResidualDownBlock`, `AvgDown3D`, `WanResample`, `_encode`), and the checkpoint's `config.json`; **[measured]** the tensor names and shapes with `build/cancho-gpu safetensors .../vae/diffusion_pytorch_model.safetensors` (196 tensors in the file, decoder included).

Config: `base_dim` 160, `dim_mult` [1, 2, 4, 4], `num_res_blocks` 2, `z_dim` 48, `patch_size` 2, `temperal_downsample` [false, true, true], `is_residual` true, no attention at any scale outside the middle block. Compression 4 (time) x 16 x 16 (space): the patch (2) times three spatial halvings (2 x 2 x 2).

The function, for a clip `x` of `[3, T, H, W]` with `T = 1 + 4 k` and `H`, `W` multiples of 16:

| step | what | weights in the checkpoint |
|---|---|---|
| patchify 2 | `[3, T, H, W]` to `[12, T, H/2, W/2]`: **[read, `patchify` line 921: `permute(0, 1, 6, 4, 2, 3, 5)`]** channel `c`, row offset `p`, column offset `q` becomes channel `4 c + 2 q + p` (the column offset is the slower one) | none |
| `conv_in` | causal 3 x 3 x 3, 12 to 160 | `encoder.conv_in` |
| down block 0 | 2 residual blocks 160 to 160; spatial halving (`downsample2d`) | `down_blocks.0.resnets.{0,1}`, `down_blocks.0.downsampler.resample.1` |
| down block 1 | 2 residual blocks 160 to 320 (the first has a 1 x 1 x 1 `conv_shortcut`); spatial **and temporal** halving (`downsample3d`) | `down_blocks.1.*`, `downsampler.time_conv` |
| down block 2 | 2 residual blocks 320 to 640; `downsample3d` | `down_blocks.2.*` |
| down block 3 | 2 residual blocks 640 to 640; no downsampling | `down_blocks.3.resnets` |
| each down block | **adds** `AvgDown3D(input)`: no weights; the block's input is regrouped (`factor_t`, `factor_s`) and **averaged** over groups of channels so that it has the output's shape | none |
| mid block | residual block, attention (one head of 640 channels, per frame, RMS norm in front, `to_qkv` and `proj` 1 x 1), residual block | `mid_block.*` |
| head | RMS norm, SiLU, causal 3 x 3 x 3 `conv_out` 640 to 96 | `encoder.norm_out`, `encoder.conv_out` |
| `quant_conv` | 1 x 1 x 1, 96 to 96 | `quant_conv` (outside `encoder.`) |
| posterior | the first 48 channels are the mean, the last 48 the log-variance | none |

The image-to-video pipeline takes the **mean** and normalises it **[read, `pipeline_wan_i2v.py` lines 428 to 462]**: `retrieve_latents(vae.encode(video_condition), sample_mode="argmax")`, then `(mean - latents_mean) * (1 / latents_std)` with the 48 constants of the config. And, in the `expand_timesteps` branch that FastWan2.2-TI2V-5B uses, **`video_condition = image`: the clip is one frame** (`T = 1`, `T' = 1`; the branches that pad with zero frames are the other model family's). A video encoder is not needed by #174.

## 2. What is the same as the decoder, and what is new

**Same [read, from `tests/wan_vae_gpu.py` and `tools/wan_vae_ref.py`]**: the causal 3 x 3 x 3 convolution (`conv3t.chx`: the two frames before the clip are the guard's zero rows), the 1 x 1 x 1 convolutions, the RMS norm and SiLU kernels, the residual block, the per-frame attention (`vae_scores_f` and its kin), the bordered-image row layout. The decoder needed nine new kernels (docs/diffusion.md V2); this encoder reuses most of those and adds the following.

**New:**

1. **A strided spatial convolution**: `ZeroPad2d((0, 1, 0, 1))` (one zero column on the **right**, one zero row at the **bottom**) then a 3 x 3 convolution of stride 2, per frame. The layout has no stride: the kernel must read the bordered image at even positions and write a half-size image. *[read]*
2. **A temporal convolution of stride 2** (`time_conv`, kernel (3, 1, 1), no padding) with the **first frame passed through**: from the model's cache logic **[read]** the whole-clip function is: output frame 0 = input frame 0 (no convolution, not even a bias), output frame `j >= 1` = `time_conv(x[2j-2], x[2j-1], x[2j])`. The clip goes from `1 + 4 k` to `1 + 2 k` frames. This is the mirror of the decoder's `upsample3d` rule, and the decoder's lesson applies: **the cache does something a plain causal convolution does not**, so the oracle must be written from the whole-clip reading and the claim "chunked equals whole" must be *checked*, not assumed (the decoder's two documented traps were exactly this).
3. **`AvgDown3D`**: front-pad time with zero frames to a multiple of `factor_t`, group the `factor_t x factor_s x factor_s` neighbours into channels in the order `(c, ft, fh, fw)`, then average consecutive groups of `group_size = in * factor / out` channels. A fixed linear map, no weights. **[read, worked by hand]** with `in = 160`, `out = 320`, `factor_t = 2`, `factor_s = 2` (block 1; `group_size` 4): output channel `2 c + ft` is the mean of the 2 x 2 spatial block of input channel `c` at time slot `ft`. **For one frame (`T = 1`) the front pad makes slot 0 the zero frame and slot 1 the picture**: output channel `2 c` is zero and `2 c + 1` is the 2 x 2 mean of channel `c`. So the temporal part of `AvgDown3D` is **not** skippable for one frame (an earlier draft of this section said the temporal kernels could be skipped altogether; that was wrong for this one).
4. **The tail**: `quant_conv` and the mean/normalisation are a 1 x 1 convolution and a per-channel affine map; for one frame at the video's latent size they may be done on the host.
5. **Patchify** is a host rearrangement.

## 3. The reference (slice E1): float64, in cancho

A cancho program, `tests/wan_vae_enc_ref.cho` (cancho's `f64`, whole-clip, the same function as the model's chunked loop), reading the checkpoint with `src/safetensors.cho` (a header reader today: **a small F32/BF16 to f64 tensor reader has to be written beside it**; the VAE file is F32). Written from the table in section 1; its output is a plain file of f64 values.

**What it does not yet have, and the claim the owner must accept or change:** the decoder's reference was held to torch float32 (8.2e-7 to 1.5e-6 of the rms **[measured, docs/diffusion.md V2]**). A cancho reference that no other implementation has seen is a reading of the source, not a check of it. The options are (a) one run of `AutoencoderKLWan.encode` by hand producing a data file of the reference tensors for the small shapes, committed as data, no script; (b) accepting the float64 reference as the specification with the risk stated. **This design proposes (a)** and does not assume it.

Small shapes, as for the decoder: 1, 5 and 9 frames at 32 x 32 and 32 x 64 pixels (every code path runs: first-frame passthrough, both temporal halvings, all three spatial ones).

## 4. The session (slice E2) and the weights (slice E3)

`models/wan_vae_encoder.model`, a model file like `models/qwen_image_vae_encoder.model` (docs/resident.md section 31), expanded by `cancho-gpu model-expand`. The weights as f16 files in the layouts the kernels read (`[O, 27 C]` tap-major for the 3 x 3 x 3 convolutions, as `conv_w3` does for the 2-D encoder) by a `cancho-gpu` subcommand written in cancho, reading the safetensors and writing the files: the Python of `tests/wan_vae_gpu.py` is replaced, not copied.

**[estimate]** the work, for a 17-frame 640 x 352 clip (not needed by #174, which encodes one frame): about 6.6 T multiply-adds, 2.65 T of them in the two 160-channel residual blocks at the full patched resolution. The decoder's measured time for a clip of the same size is 9.5 to 10.8 s (docs/diffusion.md V2), with a heavier channel count; the same order is expected. For the one frame #174 needs, a quarter of a second to a few seconds would be a guess without a measurement, and none is made here.

## 5. The checks written before the first run

* **GPU against the float64 reference:** the bound is the decoder's, **5e-3 of the output's rms**, written now, before any run (the decoder reached 2.9e-4 to 3.8e-4). Reported as the number measured, not as the bound.
* **Mutants that the comparison must refuse** (each run once, to a red result, then reverted): `time_conv` applied to the first frame (the passthrough lost); `ZeroPad2d` on the left and top instead of the right and bottom; `AvgDown3D` without its front pad of time; the log-variance half taken as the mean; the strided convolution reading even positions shifted by one; the attention's keys including the border (as the decoder's `border-keys`).
* **Whole equals chunked:** the oracle is whole-clip; a second small program runs the model's chunked loop (1 frame, then 4 at a time, with the cache as the source defines it) at one shape and compares. That the two agree is **[to measure]**, the decoder's was.

## 6. Slices

| slice | what | held to |
|---|---|---|
| E0 | this document | review |
| E1 | the f64 tensor reader and the reference, at small shapes; the torch comparison data if the owner accepts (a) | the float64 against the reference data (if any); structural checks (shapes, first-frame passthrough, `AvgDown3D` on a hand example) |
| ~~E2~~ (as planned; see the next row) | the weight-writing command and the new kernels (strided spatial conv, temporal stride-2, average-down) with unit checks each | each against a small host computation in cancho |
| ~~E3~~ (as planned; see the next row) | `models/wan_vae_encoder.model` and the whole encoder, on the GPU | float64 reference, bound 5e-3, the mutants |
| E2, E3 | **built as one program** (section 10): the session and its files written by `tests/wan_vae_enc_gpu.cho`; no `.model` file, no new kernel | the float64 reference and torch: `rel` 3.5e-4 to 4.9e-4 against the bound 5e-3; four mutants refused |
| E4 | `media video --image`: not this chain (issue #174's remaining items: per-token timestep, mixing, the CLI) | |

## 7. Not decided

* ~~Whether the comparison with torch is accepted~~: decided (a), done once; see "The torch reference data".
* Whether the encoder is built for whole clips (video to video, not asked for) or **only for one frame** (all #174 needs). Proposed: build it for the clip layout the decoder uses, test at 1, 5 and 9 frames, and use it at one; the `time_conv` kernel is not needed for a chunk of one frame, so E2 can defer it (the first frame passes through both `downsample3d` blocks without it). That makes the encoder for one frame nearly spatial-only: strided spatial convolutions, the same residual blocks, the same attention, and `AvgDown3D` with its time pad (the zero slot), but no `time_conv`: **[read]** the first frame is passed through, not convolved in time. This is the smallest thing that unblocks #174 and is the proposal for E2/E3's first target; the temporal kernels follow when a video encoder is wanted.

## 8. The torch reference data (measured, 2026-11-16)

One run, by hand, of `AutoencoderKLWan.encode` (torch 2.15.0, diffusers 0.44.0.dev0, float32, CPU, the checkpoint of `Wan-AI/Wan2.2-TI2V-5B-Diffusers` from the local cache, offline). **No script is committed**: the program was given on the standard input of the venv's Python and its only product is the data in `tests/data/wan_vae_enc/` (30 raw little-endian float32 files, 5.6 MB, and `manifest.json` with each shape). It did, for each case: draw an input with `torch.randn` from a seeded generator (seeds 1, 2, 3) clamped to [-1, 1] (**synthetic noise, not a photograph**); call `vae.encode(x).latent_dist` (the model's own chunked loop with its feature cache); keep the outputs of `conv_in`, of each of the four down blocks, of the mid block and of `quant_conv` through forward hooks (the chunks' outputs concatenated in time), and the posterior's `mean` and `logvar`.

| case | input `[1, 3, T, H, W]` | after `conv_in` | `down0` | `down1` | `down2` | `down3` = `mid` | `mean` (and `logvar`) |
|---|---|---|---|---|---|---|---|
| a | 1 x 32 x 32 | 160 x 1 x 16 x 16 | 160 x 1 x 8 x 8 | 320 x 1 x 4 x 4 | 640 x 1 x 2 x 2 | 640 x 1 x 2 x 2 | 48 x 1 x 2 x 2 |
| b | 5 x 32 x 32 | 160 x 5 x 16 x 16 | 160 x 5 x 8 x 8 | 320 x **3** x 4 x 4 | 640 x **2** x 2 x 2 | 640 x 2 x 2 x 2 | 48 x 2 x 2 x 2 |
| c | 9 x 32 x 64 | 160 x 9 x 16 x 32 | 160 x 9 x 8 x 16 | 320 x **5** x 4 x 8 | 640 x **3** x 2 x 4 | 640 x 3 x 2 x 4 | 48 x 3 x 2 x 4 |

* **[measured]** The time axis goes `1 + 4 k` to `1 + 2 k` at `down1` and to `1 + k` at `down2` (5 to 3 to 2; 9 to 5 to 3), as section 2 reads it; `down0` and `down3` leave it alone.
* **[measured]** The rms of the mean is 0.446, 0.467 and 0.499 on the three cases (a check against a degenerate reference: none is zero or constant).
* The intermediate tensors are in the data so that a mismatch can be located to a block, not only seen at the end. `quant` has 96 channels (mean then log-variance); `mean` is its first 48.
* **Not measured:** that these are the *only* correct values (the same code that is the specification produced them); that the tensors have no hidden state across the three runs (a fresh `encode` each, cache cleared by the model).

## 9. E1 results (measured, 2026-11-16): the float64 reference, in cancho

`tests/wan_vae_enc_ref.cho` with `tests/wan_enc/st.cho` (an F32 tensor reader over the safetensors header, `std.json`'s tape, bounds checked before any read) and `tests/wan_enc/ops.cho` (the operations in `float`, binary64: a general convolution with time padding in front, spatial padding on either side and strides, the RMS norm, SiLU, `AvgDown3D`, the per-frame attention, patchify). No Python and no Rust. It runs the encoder over the **whole clip** and compares, stage by stage, with the torch data of section 8. Build and run (not in `scripts/gate.sh`: it needs the checkpoint in the Hugging Face cache):

```
cancho build tests/wan_vae_enc_ref.cho tests/wan_enc/st.cho tests/wan_enc/ops.cho tests/wan_enc/codec.cho studio/files.cho --std -o build/wan_vae_enc_ref
build/wan_vae_enc_ref <vae/diffusion_pytorch_model.safetensors> tests/data/wan_vae_enc a 1 32 32
build/wan_vae_enc_ref ...                                       tests/data/wan_vae_enc b 5 32 32
build/wan_vae_enc_ref ...                                       tests/data/wan_vae_enc c 9 32 64
```

* **[measured] The reference equals torch float32 at every stage of every case.** `rel` is `rms(difference) / rms(torch)`. The bound was written in the program before the first run: **1e-5** (about 7 times the worst the decoder's float64 reference was from torch, 1.5e-6). The worst stage of each case: **a** (1 frame, 32 x 32) 1.25e-6 (`down2`), **b** (5 frames) 1.44e-6 (`down2`), **c** (9 frames, 32 x 64) **1.68e-6** (`down2`); the posterior's mean 1.0e-6, 1.3e-6 and 1.6e-6. Exit status 0 on all three; 2 s, 8 s and 29 s (compiler's native code, one thread, plain loops, the checkpoint read per layer).
* **[measured] Whole clip equals the model's chunked loop** at 1, 5 and 9 frames (the "to measure" of section 5): torch ran the chunked loop with its feature cache; this program never chunks. The two agree at the sizes above, including the first-frame passthrough of `time_conv` (the time axis 5 to 3 to 2 and 9 to 5 to 3 is reproduced: a mismatch in the frame count would be a `cannot compare`, not a small error).
* **[measured] Five mutants are refused** (an optional last argument selects one; each run once on the case that exercises it): 1 the spatial downsample's zero pad on the left and top instead of the right and bottom (**7 stages** refuse; first `down0`, rel 0.46); 2 `time_conv` also applied to the first frame, a plain causal stride-2 convolution (case b, 6 stages; first `down1`, rel 0.34); 3 `AvgDown3D` grouping channels in the order (c, row, column, time slot) instead of (c, time slot, row, column) (6 stages; first `down1`, rel 0.36); 4 the log-variance half taken as the mean (the `mean` stage only, rel 40); 5 patchify with the row and column offsets exchanged (8 stages; first `conv_in`, rel 0.96). The smallest refused `rel` is 0.29 against 1.7e-6 unmutated. **A first version of mutant 3 did not fail and is not a mutant:** it exchanged the row and column inside the 2 x 2 block, and the mean over a block does not depend on their order (the same kind of equivalent mutant as the decoder's fourth).
* **Validity of the checks.** The reference and the torch data share one specification (diffusers' source) and no code; they agree to float32 precision on three clip shapes through all 32 convolutions of the encoder (one `conv_in`, 16 in the eight residual blocks, 2 shortcuts, 3 spatial downsamples, 2 `time_conv`, 4 in the middle block's residual blocks, the attention's two, `conv_out` and `quant_conv`), the attention and the three kinds of downsampling, which would not happen by accident, and the mutants show the comparison is not blind.
* **Not shown:** nothing above 9 frames or 32 x 64 pixels (**[estimate, not run]** one frame at 640 x 352 is about 220 times the work of case a, whose 2 s would become minutes in this program); the posterior's log-variance is not compared (the image-to-video pipeline uses the mean); the latent normalisation (`(mean - latents_mean) / latents_std`) is not part of this program; and the program is not in the gate (it needs the 1.4 GB checkpoint).

## 10. E2 and E3 built (measured, 2026-11-16): the encoder for one frame, on the GPU

**What the design predicted and what was true.** Section 7 proposed building the encoder for one frame first, "nearly spatial-only", with the `time_conv` kernel deferred. That held, and **more than the design expected is shared with the Qwen-Image VAE encoder** (`tests/vae_enc_gpu.py`, the same family of VAE): **no new kernel was needed**. Section 2 listed a strided spatial convolution, a stride-2 temporal convolution and `AvgDown3D` as new; for one frame they are `vae_s2d` plus a stride-1 convolution of 4c channels, nothing (the first frame passes through `time_conv`), and `vae_avg2` plus a 0/1 matrix. **Corrected in place:** section 4 and the slice table said the session would be a `models/wan_vae_encoder.model` file expanded by `cancho-gpu model-expand` with the weights written by a `cancho-gpu` subcommand. **What was built instead is one cancho program that writes the session text and every file it names** (`tests/wan_vae_enc_gpu.cho`, a port of the generator of `tests/vae_enc_gpu.py`): the counters that tie a `.model` file to its weight files are then made in one place, not two. There is no `.model` file for this encoder.

**Why one frame is a 2-D network (proved by the implementation, checked by the comparison).** A causal 3 x 3 x 3 convolution pads two zero frames in front of the clip, so on a single frame only its last time tap meets the picture: every 3 x 3 x 3 convolution is the 3 x 3 convolution of `w[:, :, 2]`. `time_conv` is not applied to the first frame. `AvgDown3D`'s time pad puts the picture in time slot 1 and zeros in slot 0: for the two blocks that halve time, output channel `2c` is zero and `2c + 1` is the 2 x 2 mean of input channel `c`; block 0 averages 2 x 2 per channel; block 3 adds its input as it is.

**What exists.** `tests/wan_vae_enc_gpu.cho` (with `tests/wan_enc/st.cho` and `tests/wan_enc/codec.cho`): `prepare <vae.safetensors> <work dir> <repo dir> <input.f32 | -> <H> <W> [<mutant>]` writes `<work>/enc.session` (101 operations) and the files it names (the weights as halves: 138 MB at 640 x 352; norms and biases as singles; the masks; the input rows: the picture patchified by 2 into 12 channels padded to 32); the session is run by `build/cancho-gpu-metal-session`; `compare <work dir> <reference mean> <H> <W>` holds the posterior's mean, the first 48 of the 96 channels of the last convolution, to a `[48, H/16, W/16]` reference; `latent <config.json> <work dir> <H> <W> <out>` writes the same mean normalised as the image-to-video pipeline does, `(mean - latents_mean) * (1 / latents_std)` (**[measured]** against torch's mean normalised by hand with the 48 constants of the config: largest difference 9.5e-4 over 192 values of rms 0.95, the GPU's own error amplified by `1 / std`). **cancho has no `f16`**: `wenc_codec.f16_bits` rounds a binary64 value to the nearest half, ties to even, with subnormals and overflow; `tests/wan_enc/codec_check.cho` holds it to 26 known bit patterns of the format and `tests/wan_enc.sh` (**in `scripts/gate.sh`**, no GPU or checkpoint needed) also runs a mutant, ties rounded up, that must fail.

**[measured] The GPU against torch and against the float64 reference.** The bound, **5e-3 of the rms of the mean**, is the decoder's and was written in the program before the first run; `rel` is `rms(difference) / rms(reference)` over the 48-channel mean.

| picture | reference | `rel` | largest difference |
|---|---|---|---|
| 32 x 32, torch's input (case a of section 8) | **torch float32** | **3.5e-4** | 4.7e-4 |
| 64 x 96, synthetic | float64 (`wan_vae_enc_ref`, 5 s) | 4.1e-4 | 6.2e-4 |
| 128 x 192 | float64 (20 s) | 4.9e-4 | 7.2e-4 |
| 192 x 256 | float64 (39 s) | 4.5e-4 | 6.9e-4 |
| **640 x 352** (the video's size) | float64 (about 172 s, from the files' times) | **4.6e-4** | 7.7e-4 |

The first row is the one that matters most: **the GPU encoder, never tuned, equals what torch's `AutoencoderKLWan.encode` gives** (3.5e-4 against the bound 5e-3; the decoder's GPU against its float64 was 2.9e-4 to 3.8e-4). The other rows hold it to the float64 reference of this repository, which itself equals torch to 1.7e-6 (section 9), at the sizes of a real video frame.

**[measured] Four mutants are refused** (on the 32 x 32 case, against torch): the stride-2 weights losing a tap, **`rel` 0.106**; the shortcut taking every channel's mean, the pad frame forgotten, **0.200**; the attention's border rows not masked, **0.195**; the log-variance half read as the mean, **40.3**. Unmutated: 3.5e-4. The smallest refused is 300 times the unmutated error.

**[measured] Time.** The session at 640 x 352 (a picture of 320 x 176 positions after the patch, 56,320 rows before the first block) ran in **about 0.125 s** a run, process start, compile of the kernels and upload of the 138 MB of weights included (three runs: 0.135, 0.126, 0.125 s) on this machine's Metal GPU; `prepare` takes about 1 s to write the files. **[estimate, reasoning only]** the work is about 0.4 T multiply-adds (the section 4 estimate for 17 frames divided by the 17 frames' worth), so about 0.8 TFLOP in 0.1 s is the order of what this chip's f16 matrix units give; **not compared with torch's time here.**

**What this does not show**

* **One frame only, which is what `video_condition = image` needs (#174).** A clip of more than one frame is not encoded by the GPU path; the float64 reference does (sections 3 and 9), the kernels for the temporal downsample are not built here.
* **No real photograph went through it**: the pictures are the torch run's noise and a synthetic smooth-plus-noise pattern; the comparison does not depend on content, but a photograph was not tried.
* **The comparison at 640 x 352 is to the float64 reference, not to torch** (torch's data are the three small clips of section 8). ~~Torch float32 at that size was not run.~~ **Corrected by section 11: it was, on a natural picture (a frame of the model's own text-to-video run), and the GPU's normalised latent is 9.6e-4 of the rms from torch float32's** (largest difference 7.1e-3; bound 5e-3).
* **`#174` is not finished by this.** What remained: the denoiser's per-token timestep, the mixing of the first frame in the sampler, and `media video --image`. The first two are section 11 (they run; the result is not what the issue hoped for); `media video --image` is not built.
* The session is run by hand (`printf '1\ncheckpoint\n' | build/cancho-gpu-metal-session ...`); `media` does not call it.
* Not in `scripts/gate.sh` except the codec check: the encoder needs a Mac, the checkpoint and `build/cancho-gpu-metal-session`.

**Commands**

```
cancho build tests/wan_vae_enc_gpu.cho tests/wan_enc/st.cho tests/wan_enc/codec.cho studio/files.cho --std -o build/wan_vae_enc_gpu
build/wan_vae_enc_gpu prepare <vae/diffusion_pytorch_model.safetensors> <work dir> $PWD <input.f32 | -> <H> <W>
printf '1\ncheckpoint\n' | build/cancho-gpu-metal-session <work dir>/enc.session
build/wan_vae_enc_gpu compare <work dir> <reference mean> <H> <W>
build/wan_vae_enc_gpu latent <vae/config.json> <work dir> <H> <W> <out.f32>
```

## 11. The first frame in the sampler (measured, 2026-11-16): it runs, frame 0 holds, and the video does not follow it

**What was done.** `tests/wan_video_gpu.py --cond COND.f32` (the file `latent` writes, `[48, H/16, W/16]` normalised singles). Two changes, both host-side:

* **The per-token timestep was already in the denoiser.** Issue #174 said the script "appears to use one scalar timestep (not verified)": **verified, the session (`models/wan_dit.model`) has had two modulation rows and a per-token selector since V1**, held to torch with the conditioning frame at t = 0 (V0, V1b: 7.0e-4 of the rms for the whole model). What the video script did was write the *same* row twice and a selector of all ones. Now row 1 is the t = 0 row (`time_embed` of 0, in every block's six vectors and in the head's shift and scale) and the selector is 0 on the first frame's tokens (`S = F x (H/2)(W/2)`, frame-major: the first `(H/2)(W/2)` tokens). No kernel, no model file changed.
* **The mixing** `(1 - mask) * cond + mask * latents`, mask 0 on frame 0, applied to the initial noise, to every `x0` and to every re-noised latent (so before each call and after the last), as `pipeline_wan_i2v.py` does in the `expand_timesteps` branch **[read, diffusers]**.

**What was read and does not support it.** FastVideo's `DmdDenoisingStage` (`fastvideo/pipelines/basic/wan/stages/dmd.py`), the sampler of the distilled weights this repository runs, **has no first-frame mask, no per-token timestep and no mixing**: it concatenates the image latent onto the **channels** of the input (a model with more than 48 input channels) and gives one scalar timestep. TI2V-5B's transformer has 48, so that path is not this model's. Whether the DMD-distilled FastWan2.2-TI2V-5B was ever trained to continue a clean first frame is **not read from anywhere**; diffusers' mixing is what the issue asked for and what is built, and it was not checked against FastVideo's own TI2V path (not read: only `dmd.py` was).

**[measured] Test**, 17 frames of 640 x 352 (a latent of 48 x 5 x 22 x 40), seed 1, the fox prompt of V4a, three steps. The picture is **frame 0 of the text-to-video run with the same seed and prompt** (a picture the model itself made: nothing of anyone's; an earlier try with the synthetic noise pattern of section 10 was useless, a VAE of 16 times compression cannot hold per-pixel noise), encoded by this section's GPU encoder (`prepare` with that picture as `input.f32`, `latent`).

| quantity | text to video (control) | with the first frame |
|---|---|---|
| latent frame 0 against the condition | 0.57 of the rms | **0** |
| decoded frame 0 against the picture | 0 (it is the picture) | 0.0434 |
| encode then decode of the picture alone, against it | | **0.0434** |
| decoded frame 0 against that round trip | | **0** |
| frame 1 against frame 0 | 0.142 | **0.455** |
| frame 4 against frame 0 | 0.339 | **0.654** |
| frame 8 / frame 16 against frame 0 | 0.46 / 0.60 | **0.734 / 0.776** |

Relative rms differences (the script prints the last four rows). **Frame 0 is exactly the encoder's picture decoded (the 4.3% is the VAE's own round trip), and the mixing holds. But the video does not continue from it:** frame 1 is 3.2 times further from frame 0 than in the text-to-video run, and **looked at, frame 0 is the blurred fox seen from behind and frame 1 a sharper fox facing the camera**: a cut, in the same colours and scene. **This is the opposite of what the issue wanted, and it is measured once, on one picture, one prompt, one seed.**

**[measured] Mutants.** (A) the selector all ones, frame 0's tokens at the sampled timestep: the mixing holds (frame 0 exact) and the continuation is **worse**, 0.436 / 0.772 / 0.819 / 0.851: the t = 0 row helps a little (frame 4: 0.654 against 0.772), which is a **weak** detector, one number and no threshold written before the run. (B) the mixing only on the initial noise, not after each step: frame 0 off the condition by **1.08** of its rms and decoded 0.70 from the picture: refused strongly.

**What others did (searched 2026-11-16, read, not run).** `Blaizzy/mlx-video` (`mlx_video/models/wan_2/generate.py`, `i2v_utils.py`), a Wan2.2 port for Apple silicon, does the TI2V-5B image-to-video **as this section does**: mask 0 on frame 0 and 1 elsewhere, the latents initialised `(1 - mask) * z + mask * noise`, the timestep per token (`mask * t`, frame 0 at 0), and `latents = (1 - mask) * z + mask * latents` **after each scheduler step**, nothing after the loop. So the mixing and the per-token timestep here agree with an independent implementation of the **base** model (40 steps, guidance 5; not the distilled one). Two things differ from this section: it blends after the scheduler step and not also before the first call (the same thing, once the initial latents are mixed), and it has no re-noising step (not DMD). A community conversion of the distilled weights, `lBroth/FastWan2.2-TI2V-5B-MLX`, calls itself image to video with "3 denoise steps (sigmas 1.0, 0.757, 0.522, 0.0, guide=1, renoise)"; its sampler code is not in the repository and **how it handles the first frame was not found** (the `videoboom` project it links was not read). FastVideo's own distilled-model page is text to video. **Nothing found says the distilled model continues a clean first frame well, or badly.**

**[measured] The encoder against torch at the video's size.** The pipeline's own preprocessing and `AutoencoderKLWan.encode` (float32, CPU, posterior mean, normalised as above) on the same picture: the GPU encoder's 48 x 22 x 40 latent is **9.6e-4 of the rms** from it (largest difference 7.1e-3). **A first comparison gave 1.6e-2, three times the bound, and was the picture's 8-bit quantisation**: torch had been given the picture as an 8-bit PNG (relative change 2.4e-3 in the pixels) and the GPU the unquantised floats; the latent is 7 times more sensitive than the pixels. With the GPU given the same quantised picture the difference is the 9.6e-4. This closes the first limit above (torch at 640 x 352, a natural picture); one picture.

**[attempted, and not usable] A torch reference for the whole pipeline.** `WanImageToVideoPipeline` of diffusers 0.44 with the base model's five transformer shards (4 of them downloaded for this, 14.7 GB), 20 steps, guidance 5, the same picture, prompt, size and seed, text embeddings passed in (the venv has no `ftfy`), the transformer on the Metal GPU and the VAE on the CPU (no `conv3d` on MPS). Its frame 0 is the picture's round trip, **0.0434 from the picture, the same number as this repository's**; its frames after the first two are **noise of saturated colours**, in bfloat16 and in float32 alike. **The same harness run as text to video (no picture) is noise too, with guidance 5 and with guidance 1**: so this harness is wrong or the regime (17 frames, 640 x 352, 20 steps, MPS) is outside the model's, and **its image-to-video numbers (frame 1 against frame 0 0.41, frame 4 0.75, 16 0.95) are not evidence of anything**: they are not to be quoted as "the base model does no better". **Tried afterwards, none explained it:** the whole pipeline on the **CPU** in float32 (4 minutes for 17 frames, 20 steps, guidance 5): **the same numbers as on the Metal GPU to three digits** (frame 1 0.411, frame 4 0.754, 8 0.815, 16 0.950; MPS 0.41, 0.754, 0.815, 0.951), so MPS is not the cause; the five transformer shards' SHA-256 against their blob names: all match; the prompt encoded by the pipeline itself (its missing `ftfy` replaced by the identity) instead of my embeddings: still noise (0.431, 0.814, 0.835, 0.961). **Then the model's own default of 50 steps** (CPU, 8.5 minutes, the pipeline's text): **the reference becomes usable, not good**: frame 1 against frame 0 **0.365**, frame 4 0.70, frame 8 0.84, frame 16 0.92; looked at, **the same fox, from behind, continues in place through the first frames and the later frames degrade into artefacts**. So the 20-step noise was the step count, at this size. **Against it, this repository's distilled run (0.455, 0.654, 0.734, 0.776) cuts to a different, sharper fox facing the camera at frame 1**: the base model continues the picture where the distilled path does not, on the same picture, prompt and seed. That is one picture and one seed, and the two use different weights, schedulers and step counts, so it does **not** separate "the distilled model cannot do this" from "this sampler's adaptation of the mixing is wrong"; it does say the encoder's latent is a condition the base model follows. **Left (none tried):** the regime (the base model at 17 frames, 640 x 352, 20 steps of UniPC with `flow_shift` 5 is far from its 121 frames of 1280 x 704, 50 steps; the distilled weights made coherent videos at this size, which argues against a size limit alone), and the versions here (diffusers 0.44 dev, torch 2.15). The script is a throwaway (not committed: no new Python); its results are in the PR's description, not in the repository.

**What this does not show, and the causes that are open (none was tested).** (1) The distilled model may simply not do image conditioning this way (DMD distils a text-to-video model; the base model's I2V behaviour is in the 50-step pipeline, whose weights past block 6 are not in the cache here). (2) The three steps start from noise at t = 1000 with only frame 0 clean; a sampler that handles the first frame differently (or fewer re-noising steps) was not tried. (3) A defect of mine in the host arithmetic that the tests above do not see; the per-token path itself is held by V0/V1b, the encoder by section 10, so the likeliest place is the combination, which has **no torch reference at all** (the issue's last item, `WanImageToVideoPipeline` at small size, is not done: that needs the base weights). **The right next step is a reference that works** (the attempt above does not), not a command line over this.

**Commands** (one GPU job at a time; the encoder's, then the video's)

```
build/wan_vae_enc_gpu prepare <vae/diffusion_pytorch_model.safetensors> <work> $PWD <picture.f32 [3,H,W] in [-1,1]> 352 640
printf '1\ncheckpoint\n' | build/cancho-gpu-metal-session <work>/enc.session
build/wan_vae_enc_gpu latent <vae/config.json> <work> 352 640 <cond.f32>
python tests/wan_video_gpu.py <out> --weights <FastWan transformer dir> --text <text.npy> --frames 5 --size 22x40 --seed 1 --cond <cond.f32>
```

## 12. How it is done elsewhere, and the base model's sampler (measured, 2026-11-16)

**What was read (2026-11-16, not run).** The question was why the distilled path of section 11 cuts to another scene and what the published way to do image to video with Wan2.2 TI2V-5B is.
* **FastVideo** (`hao-ai-lab/FastVideo`, main): the **non-distilled** TI2V path (`WanDenoisingStage`) is what section 11 built: the first frame is VAE-encoded (`WanFirstFrameEncodingStage`), `latents = (1 - mask) * first_frame + mask * latents` before the loop, the timestep per token `mask[0][:, ::2, ::2] * step`, and the same mix after each scheduler step. The **DMD** stage (`DmdDenoisingStage`, `stages/dmd.py`) has **none of it**: a scalar timestep, no mask, no mix; its image-input pipeline (`WanImageToVideoDmdPipeline`) concatenates a 4-channel mask and the latent on the channels (the 14B layout, which a 48-channel model cannot take). The model card of `FastWan2.2-TI2V-5B-FullAttn-Diffusers` shows text to video only, and says it was "trained using only DMD" with a data-free, simulated forward. **No code was found that combines the DMD sampler with a TI2V first frame.** (Files read: `fastvideo/pipelines/basic/wan/{wan_pipeline,wan_i2v_dmd_pipeline,wan_dmd_pipeline}.py`, `stages/dmd.py`, `stages/denoising.py`, `fastvideo/models/wan/pipeline_config.py`; not read: the cookbook, `WanFirstFrameEncodingStage`, the transformer's use of `expand_timesteps`.)
* **ComfyUI** (`comfy_extras/nodes_wan.py` `Wan22ImageToVideoLatent`, `comfy/samplers.py`, the node's page): the start image is encoded into the first latent frames, the noise mask is 0 there and 1 elsewhere, and the sampler overwrites the known region every step (`out = out * denoise_mask + latent_image * (1 - denoise_mask)`). That the 5B model keeps the known frame clean and at timestep 0 is from a summary of `model_base.py`, not verbatim. Wan's own pages give no sampling defaults for the 5B; **LightX2V's Lightning distillation is A14B only, its "Wan2.2-TI2V-5B-4steps" item unchecked**; no post or issue about a first frame followed by a cut was found (forum and issue search was not done).
* **Reading:** what section 11 built is what everyone does for the base model, and **the distilled weights were not trained for it**: the cut is the model, not the mixing. The way to continue a picture is the base model, with its own sampler.

**What was built.** `tests/wan_video_gpu.py --base`: the base model's weights (`--weights` its transformer directory, `--wdir` a directory for their f16 files: 14.7 GB of float32 shards downloaded for this, converted once), flow matching with **Euler steps over diffusers' `FlowMatchEulerDiscreteScheduler(shift=5)` sigmas** (`--steps`, default 30; **[measured] the formula matches the scheduler's `set_timesteps` sigmas to 3e-8** for 20 and 30 steps: the shift is applied twice), and **classifier-free guidance** (`--guidance`, default 5): two transformer calls a step with the text and with the empty prompt's umT5 states (`tools/wan_text.py "" NEG.npy`), the text entering the session through the 60 buffers `b*_yg.a`, `b*_zg.a` that are reloaded when it changes; the first frame as in section 11 (`--cond`, the per-token timestep, the mix after every step); `--noise` starts from a given noise. The official default scheduler is UniPC; **Euler is what is built** (UniPC is not).

**[measured] The same picture, prompt, seed and noise, 17 frames of 640 x 352, 30 steps, guidance 5** (the picture is frame 0 of the section 11 text-to-video run; the condition is the GPU encoder's latent of its 8-bit version, 9.6e-4 from torch's):
| | frame 1 against frame 0 | 4 | 8 | 16 |
|---|---|---|---|---|
| distilled, 3 steps (section 11) | 0.455 | 0.654 | 0.734 | 0.776 |
| **base, this GPU path** | **0.307** | 0.629 | 0.815 | 0.862 |
| base, torch float32 on the CPU, the same Euler and noise | 0.318 | 0.627 | 0.800 | 0.857 |
| base, torch, UniPC, 50 steps (section 11) | 0.365 | 0.700 | 0.839 | 0.921 |
Looked at, **frame 1 is the same fox, moving**, as in torch's (the distilled run's frame 1 was another fox facing the camera); from the fourth frame on both the GPU and torch degrade into artefacts of saturated colour: **that is the base model at 17 frames, 640 x 352 and 30 steps, torch does the same**, not this path (its trained size is 121 frames of 1280 x 704). Time: 30 steps with guidance 2.2 s a step on the GPU, 66 s for the sampler, the decoder 11 s; torch float32 on the CPU 363 s.
**[measured] GPU against torch, relative rms over the whole video (frame 1, 4, 16)**: **0.28 (0.16, 0.29, 0.34)** with torch given the same text embeddings as the GPU. **The floor**, torch against itself with the text embeddings of its own encoder (float32) against the GPU's (bf16 on MPS): **0.34 (0.21, 0.34, 0.40)**. The GPU is **no further from torch than torch is from itself under a rounding of the text**: the thirty steps amplify a perturbation of about 1e-3, so a per-pixel comparison of a whole run says only that the two are the same video, not how exact the arithmetic is; that is held by the pieces (the transformer against torch, docs/diffusion.md V1b, 7.0e-4 at 72 tokens; the encoder, section 10; the sigmas, 3e-8). **Not measured:** the transformer's velocity against torch's at the first step of this setup (per-token timesteps with 1,100 tokens and the base weights).

**What this does not show.** One picture, prompt and seed; torch's and this path's artefacts after frame 4 were looked at, not scored; no number for how well frame 1 to 4 continue the picture except the table; the base weights' f16 files were converted by the script on the first run (the transformer's f16 error was not measured on these weights); UniPC, the official sampler, is not implemented; **`media video --image` is not built**: the pieces are run by hand (below), and `media` does not know the base model, its weights directory or the negative text.

**Commands** (one GPU job at a time)
```
build/wan_vae_enc_gpu prepare <vae.safetensors> <work> $PWD <picture.f32 [3,H,W] in [-1,1]> 352 640
printf '1\ncheckpoint\n' | build/cancho-gpu-metal-session <work>/enc.session
build/wan_vae_enc_gpu latent <vae/config.json> <work> 352 640 <cond.f32>
python tools/wan_text.py "<prompt>" text.npy ; python tools/wan_text.py "" neg.npy
python tests/wan_video_gpu.py <out> --weights <BASE transformer dir> --wdir <f16 dir for the base> --text text.npy --neg neg.npy \
    --frames 5 --size 22x40 --seed 1 --cond <cond.f32> --base --steps 30 --guidance 5
```
