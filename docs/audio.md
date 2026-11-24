# Audio: recognition, then speech, then music (design, 2026-11-24; the two reference runs made, nothing built)

What was **read** (the Hub's API, configuration files, the headers of the safetensors files, the models' reference code, this repository's sources) is marked **[read]**; arithmetic or judgment on it **[reckoned]**. **Sections 1 to 6 were written before anything was run or any weights downloaded** (only the first bytes of each safetensors file, which hold the names and shapes of its tensors); section 7 is the two reference runs, made the same day once the downloads were agreed, and is **[measured]**. It replaces `docs/diffusion.md` 6.2, which found no usable model, and follows `docs/hf-survey.md`, which named these two on 2026-11-14.

The order, as asked: **A. recognition** with `Qwen/Qwen3-ASR-1.7B`, **S. speech** with `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, **M. music**, not chosen yet.

## 1. The Hub on 2026-11-24 **[read]**

The 15 trending models of `automatic-speech-recognition`, `text-to-speech` and `text-to-audio`, from the listing only (date, licence, gating, downloads). Ungated with a permissive licence:

| task | model | created | licence | downloads |
|---|---|---|---|---|
| recognition | **`Qwen/Qwen3-ASR-1.7B`** | 2026-01 | Apache-2.0 | 1.67 M |
| | `openai/whisper-large-v3-turbo`, `whisper-large-v3` | 2024-10, 2023-11 | MIT, Apache-2.0 | 4.5 M, 3.2 M |
| | `Reson8/Rumble-2-Flash` | 2026-10 | Apache-2.0 | 3 k |
| speech | `hexgrad/Kokoro-82M` | 2024-12 | Apache-2.0 | 13.5 M |
| | **`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`** | 2026-01 | Apache-2.0 | 1.05 M |
| | `Zyphra/ZONOS3-Mini`, `n4ze3m/tiny-voca`, `KittenML/kitten-tts-2-mini` | 2026-11 | Apache-2.0 | 176 k, 12 k, 11 k |
| music, sound | `ACE-Step/Ace-Step1.5` | 2026-01 | MIT | 107 k |
| | `OpenMOSS-Team/MOSS-SoundEffect-v2.0` | 2026-05 | Apache-2.0 | 8 k |

Left out: gated or under a licence of their own (every Stable Audio model, `fishaudio/drama-3`, NVIDIA's Nemotron transcriber); non-commercial (`m-a-p/YuE2-3B`, Reverb-TTS); no licence tag in the listing (`k2-fsa/OmniVoice`, `MiniMaxAI/MiniMax-Music3`: their cards were not read). CC-BY-4.0, usable with attribution: the Parakeet models, `FermionResearch/Phonon-2`, `google/magenta-realtime-2`.

**Why the two Qwen models [reckoned]:** both are a Qwen3 language model of 28 layers with something in front of it or behind it, and a Qwen language model is what this repository runs most (`docs/decode.md`, `docs/serving.md`). The same 28 layers serve both, so the second costs less than the first.

## 2. A. Recognition: `Qwen/Qwen3-ASR-1.7B`

Revision `7278e1e7`, two safetensors files, 4.70 GB, 708 tensors, 2,349 M values, **all BF16** (exact in f16). The code read is `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py` of `github.com/QwenLM/Qwen3-ASR`.

### 2.1 What it is **[read]**

* **Features:** 16 kHz mono, Whisper's log-mel spectrogram, 128 bins, `n_fft` 400, hop 160: 100 frames a second.
* **The audio tower** (`thinker.audio_tower`, 318 M values):
  * the frames are cut in pieces of 100 (one second; the last may be shorter), each a `[1, 128, 100]` picture;
  * three 3 x 3 convolutions of stride 2, padding 1, each followed by GELU (erf): 1 to 480 channels, then 480 to 480 twice. 128 x 100 becomes 16 x 13; a full second is **13 tokens**;
  * `conv_out`, a product without bias from 480 x 16 = 7,680 to 1,024, a column of the picture a token; plus sinusoidal positions that **restart at 0 in every piece**;
  * 24 layers of 1,024 wide, 16 heads of 64, feed-forward 4,096 with GELU, layer norms before each half (with weight and bias), biases on every product: Whisper's encoder layer;
  * **attention in windows of 104 tokens (8 s)**: a token sees only its own window (`n_window_infer` 800 frames);
  * `ln_post`, `proj1` (1,024 to 1,024), GELU, `proj2` (1,024 to 2,048).
* **The language model** (`thinker.model`, 28 layers, 2,031 M values with the embedding and the head): hidden 2,048, 16 query heads over 8 key/value heads of 128, an RMS norm with a weight of 128 on q and on k of each head, no bias anywhere, SwiGLU of 6,144, RMS norms of eps 1e-6, rotary positions of theta 1e6 (the three position rows of its "mrope" are equal for text and audio, so it is the plain rotary embedding), `lm_head` stored beside the embedding (151,936 x 2,048 each).
* **The prompt:** `<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>` + one `<|audio_pad|>` a token of audio + `<|audio_end|><|im_end|>\n<|im_start|>assistant\n`; the tower's rows replace the embeddings of the pads. The answer is `language English<asr_text>the words`, greedy, ended by `<|im_end|>` or `<|endoftext|>`. 30 s of audio is 390 audio tokens and about 15 of prompt.

**A doubt found in the code [read]:** the tower's `forward` computes the windows (`cu_seqlens`) and hands them to each layer, but the mask that would apply them (`_prepare_attention_mask`) is not called there: with any attention but `flash_attention_2` the layers would attend across windows. For a clip of 8 s or less there is one window and no doubt; **the first gates use such a clip**, and which behaviour the published numbers have is to be found when the reference runs (A4).

### 2.2 Each piece against what exists **[read from `tests/chx/`, `models/`; the "no new kernel" lines are reckoned until A1 and A2 run]**

| piece | how | new? |
|---|---|---|
| reading a wav, resampling to 16 kHz | host, in cancho (`media/`, beside `png.cho`) | new host code; 16-bit PCM wav first, other containers through `ffmpeg` as `video` makes its mp4 |
| log-mel | **first on the host in cancho** (a 400-point transform a frame is 400 x 201 products: 8 M a second of audio); as two products on the GPU (`docs/roadmap-models.md` G15) only if the host's time shows | new host code |
| 3 x 3 stride-2 convolutions | `vae_s2d` (space to depth) + `conv3` + bias + GELU, the VAE encoder's way (`models/qwen_image_vae_encoder.model`); 480 channels are a multiple of 32; the first convolution's 4 input channels are padded to 16 | **[corrected the same day, before A1 was written] two small kernels**: the bias, the erf GELU and the border mask in one, f32 out (`vit_bias_gelu_erf` writes f16 and has no mask; `vae_s2d` reads f32), and the gather of a time column's 16 x 480 values into a row for `conv_out`. The pieces of a second are laid side by side in one picture, 104 columns each (100 and 4 of zeros), so that 104, 52, 26, 13 stay whole and the zeros between pieces are each convolution's padding |
| `conv_out`, `proj1`, `proj2`, the layers' products | `gemm_mma` + `vae_bias` | none |
| the tower's layers | `vit_ln`, the attention as umT5's (`dit_scores_mma`, `dit_stats_o`, `dit_pv_mma`) with the window mask from the host and **1/sqrt(64) = 1/8 folded into `q_proj`'s weight and bias, exactly**; `vit_bias_gelu_erf` | reckoned none |
| the language model, a whole pass | `rmsnorm_rows`; `gemm_mma`; q and k normed per head by `rmsnorm_rows` over rows of 128 (as `tests/dit_block.py` holds it); `dit_rope16`; **the 8 key/value heads written twice by the host in the weight files** (16 x 128 rows of `k_proj` and `v_proj`), so the attention kernels see 16 heads; a causal mask; `dit_swiglu` | reckoned none |
| the tokeniser | `src/bpe.cho`, `src/tokenize.cho` (Qwen's BPE) with this model's `vocab.json`, `merges.txt` and added tokens | to be checked: the audio tokens are added tokens |
| the next token | the last row's logits (`gemm_mma` over 151,936 rows of the head), arg max on the host | none |
| a token a step with caches | `docs/decode.md`'s step is written for Qwen2.5-1.5B's 12 + 2 heads (`dec_post`); `device/qwen35.cho`'s for NVFP4 weights | A3, a design of its own once A2 gives the transcript |

### 2.3 Stages, each a gate

* **A0. The reference.** The model's own package (`qwen-asr`, torch) run once on a fixed clip of 8 s or less: the mel, the tower's rows, the logits of each generated position and the transcript, kept as files. No Python enters the repository for it (the rule of 2026-11-09): the run is by hand, its outputs are the fixture.
* **A1. The tower** as a model file (`models/qwen3_asr_audio.model`) from the reference's mel: its 2,048-wide rows against the reference's. **Bound, set now:** every row nearer float32's than the same package in BF16 on MPS is, the measure V3 used for umT5 (`docs/diffusion.md`); mutants: the windows' mask dropped (on a clip longer than 8 s), the positions not restarting.
* **A2. The transcript, slowly.** The language model as a model file over the whole sequence (prompt, audio rows, the tokens so far), one pass a token, no cache. **Bound: the same tokens as the reference's greedy answer for the clip, and the arg max equal at every position when teacher-forced.** Time is reported and not bounded: a pass over 200 rows is 28 layers of products of at most 2,048 x 6,144, **[reckoned]** tens of milliseconds, so a hundred tokens are seconds.
* **A3. A token a step**, with key and value caches and the prefill in one pass: the design of `docs/decode.md` at this model's shapes. Written when A2 passes; the bar is real time on 30 s clips with room to spare, and the figure against the reference on MPS.
* **A4. `scripts/media transcribe CLIP -o DIR`**: host mel in cancho against the reference's mel, long clips (the 8 s windows, the reference's cutting of audio over 20 minutes), the refusals (not a wav, no audio, too long), each with a rule tag.

### 2.4 Not known

The reference's accuracy on MPS in BF16 (A1's yardstick); whether the tower attends across windows in the published numbers (2.1); the tokeniser's handling of the audio tokens; the forced aligner (timestamps) is another model and is not in this design; streaming is not in it either.

## 3. S. Speech: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`

Revision `0c0e3051`: `model.safetensors` 3.83 GB (404 tensors, 1,917 M values, BF16) and `speech_tokenizer/model.safetensors` 0.68 GB (496 tensors, 171 M values, **F32**). The code read is `qwen_tts/core/models/modeling_qwen3_tts.py` and `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` of `github.com/QwenLM/Qwen3-TTS`. Nine preset voices, ten languages with Spanish; no reference audio is needed (cloning is the `Base` models, with a speaker encoder that is not in this checkpoint's path).

### 3.1 What it is **[read]**

* **The talker:** the same 28 layers as 2.1's language model (2,048, 16 over 8 heads of 128, SwiGLU 6,144), over a vocabulary of 3,072 codec entries (2,048 codes, then the control, language and voice entries). The text goes through the text embedding (151,936 x 2,048) and `text_projection` (two products with a SiLU); **a frame's input is the sum of its 16 code embeddings plus the embedding of the next text token** (the text is fed alongside the frames, then a pad embedding).
* **A frame is 16 codes of 2,048, 12.5 frames a second.** The talker's head gives the first; **the code predictor** (5 layers of 1,024 wide, the same kind of layer, a product from 2,048 down to 1,024 in front) gives the other 15 one after another, each with a head and an embedding of its own, from the talker's last hidden row and the codes so far.
* **Sampling**, not arg max: temperature 0.9, top-k 50, repetition penalty 1.05, for the talker and for the predictor.
* **The codec's decoder** (the `decoder.*` tensors of the tokenizer; its encoder is for cloning and is not needed):
  * the 16 codes looked up in their codebooks (256 wide, `embedding_sum / cluster_usage`) and summed after two 1 x 1 products to 512;
  * a causal 1-D convolution of 3 taps to 1,024; **8 transformer layers** of 512 wide (16 heads of 64, rotary, RMS norms, a learned scale on each residual, attention over the last 72 positions only);
  * two stages that double the rate: a transposed convolution of 2 taps and stride 2, then a ConvNeXt block (a depthwise causal convolution of 7 taps, a layer norm, two products with GELU);
  * a convolution of 7 taps to 1,536 channels, then four blocks that multiply the rate by 8, 5, 4 and 3 and halve the channels (1,536 to 96): **the Snake activation `x + sin^2(a x) / b`** with a learned `a` and `b` a channel, a transposed convolution of 2r taps and stride r, and three residual units (Snake, a causal convolution of 7 taps with dilation 1, 3 or 9, Snake, a 1 x 1 convolution);
  * Snake, a convolution of 7 taps to one channel, clamped to [-1, 1]: **24,000 samples a second**, 1,920 a frame.

### 3.2 What the language and the kernels lack **[read from `docs/chx.md`, `src/elab.cho`]**

| needed | state |
|---|---|
| **`sin`** (Snake) | **not in the language**: its elementwise functions are `rsqrt`, `exp`, `log`, `sigmoid`, `softplus`, `relu`, `step`, `tanh`. A new operation: parser, checker, interpreter, both lowerings, and its accuracy stated as `docs/chx.md` section 7 does for the others |
| a 1-D causal convolution of 7 taps with a dilation | not built. As `conv3` is a loop of `mma` over the taps of shifted row windows (`docs/roadmap-models.md` G11), over one axis; the causal padding is rows of zeros in front |
| a transposed 1-D convolution of stride r and 2r taps | not built. **[reckoned]** r convolutions of 2 taps, one for each output phase, each a product of the same input rows with a slice of the weight: no new operation, a kernel and an interleaving store |
| a depthwise convolution of 7 taps | `conv_step.chx` is one of 4 taps for one token; the general one is elementwise products of shifted rows with a `[7, c]` table |
| a sampler with top-k and a repetition penalty | `docs/sampler.md`'s kernels, to be read against these settings |
| 16 sequential draws a frame | the caches of A3, for two models (the talker and the predictor) |

### 3.3 Stages

* **S0. The reference**, as A0: one sentence, one voice, a fixed seed; the 16 x T codes and the waveform kept.
* **S1. The codec's decoder from the reference's codes**, the part with everything new and no sampling: `sin` first (the operation, its float64 test, its mutants), then the three convolutions, then the model file. **Bound, set now: the waveform within 1e-3 of full scale of the reference's float32 at every sample, and the kernels held to float64 as the others are.**
* **S2. The talker and the predictor, teacher-forced** on the reference's codes: the logits of every draw. Bound: as A2's, the arg max equal wherever the reference's top two are clear.
* **S3. Generation**, with the device's sampler. A sampled sequence cannot equal torch's; **the check is S2's logits and the predictor's draws given the reference's uniforms if they can be fed, else a listening check and A's recogniser on the result** (the two halves of this document test each other).
* **S4. `scripts/media speak "TEXT" -o DIR --voice NAME --language L`**, and the real-time factor.

### 3.4 Size **[reckoned]**

A second of speech is 12.5 steps of the talker and 187.5 of the predictor. At `docs/decode.md`'s 11 ms a step for 28 layers (8 sequences, a model of 1.5B) the talker is about 0.14 s a second of audio; the predictor's 5 layers at a fifth of that are about 0.4 s. **Real time is within reach only with caches; without them (the form of A2) S3 is a correctness run of minutes a sentence.** The decoder is 24,000 x 96 values a second at its widest: small.

## 4. M. Music: not chosen

`ACE-Step/Ace-Step1.5` (MIT) and `OpenMOSS-Team/MOSS-SoundEffect-v2.0` (Apache-2.0) are the two ungated candidates of section 1; **neither's configuration was read.** The Stable Audio models are gated: someone must accept their licence on the Hub. Whatever is chosen will need an audio autoencoder's decoder, so `sin` and the 1-D convolutions of S1 come first either way.

## 5. What needs a yes before it happens

* **Downloading the weights**: 4.70 GB for A, 4.51 GB for S (about 86 GB were free on 2026-11-14, `docs/hf-survey.md`).
* **Installing the two reference packages** (`qwen-asr`, `qwen-tts`) in the benchmark environment (`~/.cache/cancho-media/bench-venv`), for A0 and S0 only.
* **A clip to transcribe**: any recording of 8 s or less whose words are known; one made by S0 would do for both.

## 6. What this document does not claim

That any kernel named in 2.2 fits without change; that the transcript will equal the reference's; any time. The tensor names and shapes are the files'; the forward passes are the reference code's as read, not as run.

## 7. A0 and S0: the two reference runs **[measured, 2026-11-24, M4 Max]**

Both by hand, with scripts that are not in the repository; their outputs are the fixtures, in `~/.cache/cancho-media/audio-ref/`. **The two packages could not share an environment, nor the benchmark's:** `qwen-asr` pins transformers 4.57.6, `qwen-tts` 4.57.3, and the benchmark environment has 5.24.0; each has its own (`~/.cache/cancho-media/qwen-asr-venv`, `qwen-tts-venv`, torch 2.15.0, 1.2 GB each). Both run offline from the local snapshot's path (given a repository name, `qwen-tts` asks the Hub).

* **S0.** "The quick brown fox jumps over the lazy dog near the quiet river.", English, the voice Ryan, `torch.manual_seed(1)`, float32 on the CPU: **67 frames of 16 codes** (values 1 to 2,041), **128,640 samples at 24 kHz, 5.36 s** (1,920 a frame, as read), peak 0.415, generated in 8.5 s. Kept: `s0_codes.npy` `[67, 16]`, `s0_wav_f32.npy`, `s0.wav` (16-bit).
* **A0, on that clip** (resampled to 16 kHz by the package): **the transcript is the sentence, to the letter, in float32 on the CPU and in BF16 on MPS** (1.7 and 1.6 s a transcription, the model loaded). The mel is `[128, 536]`; the tower gives **70 rows** (5 seconds of 13 and a last piece of 36 frames, 5 rows: `_get_feat_extract_output_lengths`); the prompt is 85 ids; the answer 18: `language English<asr_text>The quick brown fox jumps over the lazy dog near the quiet river.<|im_end|>`. Teacher-forced, the arg max equals the generated id at 18 of 18 positions. Kept: the mel, the tower's rows in both precisions, the prompt's ids, the sequence, the 19 rows of logits.
* **A1's yardstick:** the package's tower rows in BF16 on MPS are **1.59e-4 mean absolute error** from its float32 rows (the rows' rms is 0.0187; the worst element 9.0e-3).
* The clip is under 8 s: one attention window, so 2.1's doubt is not met yet.
