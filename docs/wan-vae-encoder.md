# The Wan VAE encoder, in cancho (design, 2026-11-16)

> **Status: the encoder for one frame is built and measured (sections 8 to 10): E1 the float64 reference, E2 and E3 the GPU session; what #174 still needs is not here (the end of section 10).** This document began as a design before any code and says, in place, where building corrected it. Issue [#174](https://github.com/alpibrusl/cancho-gpu/issues/174) (image to video) needs the VAE *encoder* of `Wan2.2-TI2V-5B`; only the decoder exists. Claims are tagged **[read]** (from the source or the checkpoint, not run), **[measured]** (run, with the command) or **[estimate]**. The owner chose, on 2026-11-16, to build it **native in cancho** (no new Python, no new Rust): the session as a `.model` file, the weights written by a cancho command, the float64 reference a cancho program. This is the first of a chain; each slice is its own PR and holds a measured result before the next starts.

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
* **The comparison at 640 x 352 is to the float64 reference, not to torch** (torch's data are the three small clips of section 8). Torch float32 at that size was not run.
* **`#174` is not finished by this.** What remains: the denoiser's per-token timestep, the mixing of the first frame in the sampler, and `media video --image`. This encoder gives the first frame's 48 normalised latent channels; nothing yet consumes them.
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
