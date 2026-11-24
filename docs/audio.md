# Audio: recognition, then speech, then music (design, 2026-11-24; the reference runs made; A1 and A2 built: the reference clip is transcribed on the GPU, token for token; A3 and A3b built: 11.5 ms a token, the prompt in one pass; A4a built: two clips from their wav files to the reference's words)

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
  * **attention in windows of 104 tokens (8 s)**: a token sees only its own window (`n_window_infer` 800 frames) **[corrected 2026-11-24, section 15: that is what the configuration says; the reference as it runs attends across the windows, and so does the session]**;
  * `ln_post`, `proj1` (1,024 to 1,024), GELU, `proj2` (1,024 to 2,048).
* **The language model** (`thinker.model`, 28 layers, 2,031 M values with the embedding and the head): hidden 2,048, 16 query heads over 8 key/value heads of 128, an RMS norm with a weight of 128 on q and on k of each head, no bias anywhere, SwiGLU of 6,144, RMS norms of eps 1e-6, rotary positions of theta 1e6 (the three position rows of its "mrope" are equal for text and audio, so it is the plain rotary embedding), `lm_head` stored beside the embedding (151,936 x 2,048 each).
* **The prompt:** `<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>` + one `<|audio_pad|>` a token of audio + `<|audio_end|><|im_end|>\n<|im_start|>assistant\n`; the tower's rows replace the embeddings of the pads. The answer is `language English<asr_text>the words`, greedy, ended by `<|im_end|>` or `<|endoftext|>`. 30 s of audio is 390 audio tokens and about 15 of prompt.

**A doubt found in the code [read]:** the tower's `forward` computes the windows (`cu_seqlens`) and hands them to each layer, but the mask that would apply them (`_prepare_attention_mask`) is not called there: with any attention but `flash_attention_2` the layers would attend across windows. For a clip of 8 s or less there is one window and no doubt; **the first gates use such a clip**, and which behaviour the published numbers have is to be found when the reference runs (A4). **[Found, section 15: across the windows.]**

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
| a transposed 1-D convolution of stride r and 2r taps | not built. **[reckoned]** r convolutions of 2 taps, one for each output phase, each a product of the same input rows with a slice of the weight: no new operation, a kernel and an interleaving store. **[corrected with S1b, section 20: one convolution of 2 taps to r times the outputs, and no interleaving store: the buffer read with another shape]** |
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

## 8. A1: the audio tower on the GPU **[measured, 2026-11-24, M4 Max]**

`models/qwen3_asr_audio.model` (144 lines, 1,689 lines expanded for the clip), its host side `tests/qwen3_asr_audio_gpu.cho` (cancho: the weights, the inputs, the comparison; no Python), two kernels (`conv_bias_gelu_erf`, `conv_cols`), and BF16 in the cancho safetensors reader (`tests/wan_enc/st.cho`, `load_bf16`).

```sh
B=build/qwen3_asr_audio_gpu; A=~/.cache/cancho-media/audio-ref; W=~/.cache/cancho-media/qwen3-asr-audio-f16; K=<a work directory>
$B prepare <the snapshot's directory> $W $K $A/a0_mel.f32 536        # prints BP=10 SP=128 tokens 70
printf '1\ncheckpoint\n' | build/cancho-gpu-metal-session models/qwen3_asr_audio.model CHX=$PWD/tests/chx CONV3=$PWD/examples/cnn/conv3.chx WORK=$K WDIR=$W BP=10 SP=128
$B compare $K $A/a0_f32_tower.f32 $A/a0_bf16mps_tower.f32 70
```

| the 70 rows of the clip against the reference's float32 (their rms 1.87e-2) | mean abs error | worst element | rows nearer than the yardstick's |
|---|---|---|---|
| **the GPU session (f16 operands, f32 sums)** | **7.98e-6** | 2.7e-4 | **70 of 70** |
| the yardstick: the reference in BF16 on MPS | 1.59e-4 | 9.0e-3 | |
| mutant 1: the positions do not restart in every piece | 8.8e-3 | | 0 |
| mutant 2: the time axis's taps laid out as padding (0, 1) | 1.1e-2 | | 0 |
| mutant 3: q not scaled by 1/8 | 1.0e-2 | | 0 |
| mutant 4: the zeros between pieces not put back after the first convolution | 7.1e-4 | | 0 |

* **The bound of 2.3 is met at the first run: every row is nearer float32's than BF16's is, 20 times nearer on average.** The four mutants fail with no row nearer.
* **Time:** `prepare` 1.3 s (630 MB of weights written from the two BF16 shards, the 5 MB of inputs); the session 0.21 s a run, three runs, with its compile and upload; `compare` under 0.1 s.
* **Held in the gate without a GPU or the checkpoint:** the two kernels against float64 with six mutants (`tests/vit_pieces.py`), their goldens, the model file's expansion (`tests/model.sh`), and `tests/qwen3_asr_audio.sh`: the host program builds, refuses with the status it documents (9 cases) and writes the session's inputs at their sizes for a clip of one second.
* **What 2.2 got wrong:** "reckoned none" for the convolutions; two small kernels were needed (corrected there). The rest of the tower is existing kernels: `vae_s2d`, `conv3` and `conv3_k16`, `gemm_mma`, `vae_bias`, `vit_ln`, `vit_bias_gelu_erf`, `dit_scores_bias`, `dit_stats`, `dit_pv`, `dit_to_f16`, `dit_add`.
* **Not done:** a clip longer than 8 s (more than one attention window: 2.1's doubt, and the mutant "the windows' mask dropped" that 2.3 named, which needs one); the mel is the reference's, not the host's (A4); the BF16 reader has no test of its own in the gate (it is exercised only when the checkpoint is there); one clip.

## 9. A2: the transcript, a whole pass a token **[measured, 2026-11-24, M4 Max]**

`models/qwen3_asr_text.model` (119 lines, 1,973 expanded): the 28 layers, the final norm and the head over the whole sequence, the Qwen-Image transformer's layer without its modulation. **No new kernel**: `rmsnorm_rows` (the stream's norms, and q and k of each head as rows of 128), `gemm_mma`, `dit_rope16`, `dit_scores_mma`, `dit_stats_o`, `dit_pv_mma`, `dit_to_f16`, `dit_swiglu`, `dit_add`. Its host side is `tests/qwen3_asr_text_gpu.cho` (cancho); what it shares with A1's is `tests/qwen_asr/host.cho`. The audio rows are A1's, from the GPU: the two sessions in a row.

```sh
B=build/qwen3_asr_text_gpu; A=~/.cache/cancho-media/audio-ref; W=~/.cache/cancho-media/qwen3-asr-text-f16; K=<a work directory>
$B weights <the snapshot's directory> $W                                  # once: 4.0 GB in 7 s
$B inputs $W $K $A/a0_seq.txt <A1's work directory>/rows.f32              # prints SP=128 tokens 103
printf '1\ncheckpoint\n' | build/cancho-gpu-metal-session models/qwen3_asr_text.model CHX=$PWD/tests/chx WORK=$K WDIR=$W SP=128
$B check $K $A/a0_seq.txt 85 $A/a0_logits.f32
tests/qwen3_asr_greedy.sh $W <another work directory> $A/a0_prompt.txt <A1's work directory>/rows.f32
```

| | arg max equal to the next id, of 18 | logits against the reference's float32 (largest 50.05) |
|---|---|---|
| **teacher-forced on the reference's sequence (85 of prompt and audio, 18 of answer)** | **18** | worst difference 0.123, smallest cosine of a row 0.999974 |
| mutant 1: no causal mask | 4 | worst 35.2 |
| mutant 2: every row at position 0 | 0 | worst 42.1 |
| mutant 3: query head h reads key/value head h % 8, not h / 2 | 0 | |

* **The greedy answer from the prompt alone, one pass a token, is the reference's 18 ids:** `language English<asr_text>The quick brown fox jumps over the lazy dog near the quiet river.<|im_end|>`. Both halves of 2.3's bound are met: the same tokens, and the arg max equal at every position teacher-forced.
* **Time:** a pass 0.93 s with its compile and the upload of 4 GB; the greedy loop 11.3 s for 18 tokens (0.63 s a token: a new process, the inputs and the 78 MB of logits to the host each time). 2.3 reckoned "tens of milliseconds" a pass for the products; the pass alone was not timed apart from the process. This is the form to check, not to use: A3 is the cached step.
* **Held in the gate without a GPU or the checkpoint:** the model file's expansion (`tests/model.sh`) and `tests/qwen3_asr_text.sh`: the host program builds, refuses with its documented statuses (16 cases), and from a made-up embedding table writes a pass's inputs at their sizes with the right bytes (the halves 1.0, -2.0 and 2^-24 decoded, the audio row in its place, the causal mask).
* **Not done:** the ids are the reference's: the prompt is not built or tokenised here, nor the answer turned back into text (A4, with `src/bpe.cho`); one clip, 5.4 s, English; the logits' 0.123 of 50 was not compared with what the reference's own BF16 gives; `tests/qwen3_asr_greedy.sh` is a shell loop and has no test of its own.

## 10. A3 (design, 2026-11-24, written before its code): a token a step, with caches

A2 runs the whole sequence for every token: 0.63 s a token. A3 runs one row a step against caches of the keys and values, as `docs/decode.md` does for Qwen2.5-1.5B. **[read, `tools/qsw/decode.cho` and the kernels]** That step's kernels take the head counts as constants (`kv_write_b`, `attention_scores_b`, `attention_pv_b`: `nkv`, `g`, `d`), except `dec_post`, written for 12 + 2 heads with biases and no norms; the generator (`qsw decode`) refuses any other shape, feeds token ids only, and samples by temperature.

**One sequence, one model file** (`models/qwen3_asr_step.model`), not an extension of `qsw`: the audio rows are not token embeddings, the answer is an arg max, and a model file with `table` and `carry` lines says the step in 130 lines where the generator would need a third mode.

**A step, for the row at position p:**

| piece | kernel | new? |
|---|---|---|
| the row: the prompt's own row while it lasts (a table of rows the host writes: text embeddings and A1's audio rows), else the embedding of the last arg max | `onehot_h` and the product with the transposed table (`matvec_rows_f16h`, as `docs/decode.md`), then **`row_mix`**: `m x_table + (1 - m) x_embedding` | `row_mix`: `ids_mix` for a row of 2,048 |
| the stream's norm, f16 out | `rms_h` | |
| q, k and v in one product, 4,096 wide (16 + 8 + 8 heads of 128; the host stacks the three weights) | `matvec_rows_f16h` | |
| the RMS norm of each q and k head with its weight, the rotary rotation at p; v as it is | **`dec_post_n`** | `dec_post` with norms in place of biases, 32 slots |
| the new key and value rows into the caches (8 heads, `tmax` positions) | `kv_write_b`, `carry` | |
| scores (scaled there by 1/sqrt(128)) and the weighted sum, 2 query heads a key/value head | `attention_scores_b`, `attention_pv_b` | |
| o, the residual, the norm, gate and up in one product of 12,288, SwiGLU, down, the residual | `cast_h`, `matvec_rows_f16h`, `res_add`, `rms_h`, `swiglu_h` | |
| the final norm and the head: 151,936 logits | `rms_h`, `matvec_rows_f16h` | |
| **the arg max, on the device**: the row's maximum (`samp_max`, `samp_gmax`, with a mask of zeros), then **`amax_cand`**: `(V - j) (1 - step(max - l_j))` for column j (0-based) **[corrected when built: the design had `step(l_j - max + tiny)`, which the Metal GPU gave 0 at the maximum too; see section 11]**, whose maximum (the same two kernels again) is `V + 1 - id` of the **first** maximal column, as torch's arg max; **`amax_id`** turns it into the id | `amax_cand`, `amax_id` |
| the id into slot j of the request's answer, and carried to the next step's one-hot | `tok_acc`, `carry` | |

**Four small kernels**, each a few lines, each to be held to float64 with mutants as the others are. **Tables, a block a step, written by the host:** the cos and sin of p, the cache's `sel` and `keep`, the attention's mask (positions up to p), the prompt's rows and their flag, the answer's `sel` and `keep`.

**The prompt goes through the same step, a row a step**, audio rows included: no second session, no import of caches, and every row of the sequence is computed by the same kernels. It costs a step a row: 85 for the reference clip, about 405 for 30 s of audio. **A prefill in one pass (A2's session exporting its keys and values) is A3b, and is built only if the measured step makes the prompt cost more than a second for 30 s of audio.**

**Sizes [reckoned]:** the caches are 28 layers x 2 x 8 heads x `tmax` x 128 singles: 117 MB at `tmax` 512. The weights are A2's 4 GB, restacked (q, k, v as one file a layer, gate and up as one; the 8 key/value heads once, not twice). A step reads every weight once: 4 GB of halves, which at the 11 ms a step `docs/decode.md` measured for 8 sequences of a 1.5B is **about 15 ms a step, unverified**.

**Checks, written before the code:**

1. **The four kernels against float64**, with mutants: for `dec_post_n` the norm's weight left out, q's weight used for k, the rotation's sign; for `amax_cand` and `amax_id` the last maximal column taken for the first; for `row_mix` the flag ignored.
2. **The reference clip, teacher-forced** (all 103 rows from the table, the ids being the reference's): the arg max equal to the next id at 18 of 18 positions, as A2's bound; the ids the device wrote are those arg maxes. **And greedy** (the 85 prompt rows from the table, then the device's own ids for 24 steps): the reference's 18 ids, then whatever follows the end token, which the host cuts.
3. **Mutants of the step, each to fail check 2:** the cache's `keep` read as `sel` (the row not written), the mask a position short (a row does not see itself), the rotary table a position late **[corrected when run: that one cannot fail, the rotation being relative, and it survived; it is 'every step at position 0' now]**.
4. **Time:** the milliseconds of a step, from requests of 16 and 64 steps in one session, three times; the whole clip (85 + 18 steps) from a cold process. Bar: the clip in under 3 s where A2's loop took 11.3 s, and no bar on the step beyond reporting it against the 15 ms reckoned.
5. `scripts/gate.sh`, with the host program's refusals and inputs held without a GPU as A1's and A2's are.

**Not in A3:** sampling (speech needs it: S2), more than one sequence, the prefill pass (A3b, on the condition above), the prompt's tokens (A4). **A tie of two maximal logits** would be resolved as torch does, to the first; it was the reason for `V + 1 - j`.

## 11. A3: a token a step **[measured, 2026-11-24, M4 Max]**

`models/qwen3_asr_step.model` (131 lines, 1,526 expanded), the four kernels of section 10 (`row_mix`, `dec_post_n`, `amax_cand`, `amax_id`) and the host side `tests/qwen3_asr_step_gpu.cho` (cancho). The prompt's rows go through the same step; the arg max is on the device.

```sh
B=build/qwen3_asr_step_gpu; A=~/.cache/cancho-media/audio-ref; W=~/.cache/cancho-media/qwen3-asr-text-f16; K=<a work directory>
$B weights <the snapshot's directory> $W                                  # once, after A2's: 2.3 GB more in 5 s
$B inputs $W $K $A/a0_prompt.txt <A1's work directory>/rows.f32 109 128 0 # the prompt's 85 rows and 24 steps of answer, from position 0 (the last argument came with A3b); prints T=128 NS=109 rows 85
printf '109\ncheckpoint\n' | build/cancho-gpu-metal-session models/qwen3_asr_step.model CHX=$PWD/tests/chx QW=$PWD/examples/qwen TR=$PWD/examples/train WORK=$K WDIR=$W T=128 NS=109
$B ids $K 84                                                              # the ids chosen from the prompt's last row on
```

| | the id chosen is the next id, of 18 |
|---|---|
| **teacher-forced** (the 103 rows of the reference's sequence from the table) | **18** |
| **greedy** (the 85 rows of the prompt, then the device's own ids), three cold runs | **the reference's 18 ids**, each time, then what follows the end token (a newline, `<|endoftext|>`, a new turn), which the host cuts |
| mutant 1: the caches' `sel` zeros (a row never written) | 0 |
| mutant 2: the mask a position short | 1 |
| mutant 3: every step at position 0 | 0 |

**Time:**

| | |
|---|---|
| a step, in one session: requests of 16 and of 64 steps, three times each | 11.8, 11.5, 11.6 ms and 11.6, 11.4, 11.4 ms a step (reckoned 15) |
| the first step, with the session's compile and the upload of 6.3 GB | 0.47 s |
| **the clip from a cold process: 85 rows of prompt and 24 steps** | **1.85, 1.81, 1.81 s** (A2's loop: 11.3 s for 18 tokens; the bar was 3 s); the tables are written in 0.02 s |

* **30 s of audio would be about 405 rows of prompt: 4.7 s at this step [reckoned from the step, not run].** That is over the second section 10 set as the condition, so **A3b, the prefill in one pass, is to be built.** A2's pass over 128 rows took 0.93 s with its upload.
* **Two things the design got wrong, both corrected there:**
  * `step(l - max + 1e-30)` for "is this column the maximum": on the Metal GPU it was 0 everywhere, the maximum included (the first run chose no id at all; the interpreter and the kernel's GPU comparison on random data did not show it). The kernel now takes `1 - step(max - l)`, which is exact and needs no small number.
  * the mutant "the rotary table a position late" survived with 18 of 18: a shift of every position changes no rotation between two rows. Replaced by "every step at position 0".
* **Held in the gate without a GPU or the checkpoint:** the four kernels against float64 (`tests/kernels.py`), their goldens, **[corrected 2026-11-24, with A4: this line said the seven mutants were in the gate; the gate runs `tests/kernels.py` without `--mutants`, as it did before these kernels, so the mutants were run by hand: all seven killed, after three first drafts that died of a refusal to compile were replaced]** the model file's expansion, and `tests/qwen3_asr_step.sh`: the host program builds, refuses with its documented statuses, writes the tables of four steps with the right sizes and bytes, and reads a made-up answer of the device.
* **Not done:** the step's time is from the host's clock around a request, not a profile of its kernels; one clip; the weights directory now holds some matrices twice (A2's files and the stacked ones: 6.3 GB for a 4.7 GB checkpoint); a tie of two maximal logits was not made to happen on the model; the ids past the end token are computed and thrown away (a request is a fixed number of steps).

## 12. A3b (design, 2026-11-24, written before its code): the prompt in one pass

Section 11 left the prompt at a step a row: 4.7 s for 30 s of audio. **[measured before this was written]** A2's session over 448 rows (a made-up sequence of 404 ids, the head included) takes **0.165 s a pass, four passes, once it is loaded; 1.20 s with its load.** 405 steps take 4.7 s. So the prompt goes through A2's pass and only the answer through steps.

* **The pass, without its head** (`models/qwen3_asr_text.model` with `PRE=1`): the rows 0 to P - 2 of the prompt. For each layer the keys (normed and rotated, as the scores read them) and the values are already there as halves, token-major, 16 heads of which each pair is one key/value head. **One kernel, `kv_export`**, writes them as the step's caches: singles, head-major, `[8 * tmax, 128]`, position t of head h from row `16 t + 2 h`. 56 saves, 0.5 MB each at 128 positions.
* **The steps** (`models/qwen3_asr_step.model` with `PRE=1`) start from those files in place of zeros, at the prompt's last row, P - 1, which comes from the table as before and gives the first id of the answer. The host's tables take the position of their first step.
* **Not the same numbers as a prompt fed by steps:** the pass's keys and values are rounded to f16 (they are what A2 attends over) where a step's rows are f32. The bound is the reference's ids, not equality with section 11's caches.
* **Two sessions, two loads** (the pass's weights are 4 GB, the step's 6.3 GB, stacked differently). One session for both would need the pass's products to read the stacked matrices, which is a different pass; not done here.

**Checks, written before the code:**

1. `kv_export` against float64, with mutants: the row of another head read, the position written a place late.
2. **The reference clip:** the pass over the prompt's first 84 rows, then 25 steps from row 84: the reference's 18 ids, greedy; and teacher-forced (the answer's rows from the table) 18 of 18. **A mutant: the steps started from zeros** (the caches not read) must fail.
3. **A long prompt** (the made-up 404 ids, caches of 512 positions): the ids of 16 steps after the pass against the ids of the same 16 steps after 404 steps. Reported, not bounded: the sequence means nothing, so its next tokens may be near ties.
4. **Time, from cold processes, three times:** the clip (pass, export, steps), and the long prompt with 100 steps of answer, against the same by steps alone. Bar: the long prompt under 3.5 s where steps alone are reckoned at 5.9 s.
5. `scripts/gate.sh`.

## 13. A3b: the prompt in one pass **[measured, 2026-11-24, M4 Max]**

`kv_export` (one kernel), `PRE=1` in `models/qwen3_asr_text.model` (no head; each layer's keys and values written as caches and saved) and in `models/qwen3_asr_step.model` (the caches start from those files), and a first position for the step's tables (`tests/qwen3_asr_step_gpu.cho inputs ... <first>`, `check ... [<first>]`).

```sh
BT=build/qwen3_asr_text_gpu; BS=build/qwen3_asr_step_gpu; K=<a work directory>     # P = the prompt's rows; the pass takes the first P - 1
$BT inputs $W $K <the prompt's ids less the last> <audio rows>                    # prints SP
printf '1\ncheckpoint\n' | build/cancho-gpu-metal-session models/qwen3_asr_text.model CHX=$PWD/tests/chx WORK=$K WDIR=$W SP=<SP> PRE=1 T=<positions>
$BS inputs $W $K <the prompt's ids> <audio rows> <steps> <positions> <P - 1>
printf '<steps>\ncheckpoint\n' | build/cancho-gpu-metal-session models/qwen3_asr_step.model CHX=$PWD/tests/chx QW=$PWD/examples/qwen TR=$PWD/examples/train WORK=$K WDIR=$W T=<positions> NS=<steps> PRE=1
$BS ids $K 0
```

| the reference clip: the pass over 84 rows, then steps from row 84 | |
|---|---|
| teacher-forced: the id chosen is the next id | **18 of 18** |
| greedy, three cold runs | **the reference's 18 ids** each time |
| mutant: the steps start from zeros (`PRE=0` in the step) | 9 of 18 |
| **time, cold, all of it** | **1.35, 1.31, 1.32 s** (section 11, by steps alone: 1.81 to 1.85): the pass with its load and 56 saves 0.50, the 25 steps with their load 0.80 to 0.83, the host's inputs 0.02 |

| a long prompt: the made-up 404 ids, caches of 512 positions, 100 steps of answer; three cold runs | |
|---|---|
| by steps alone (504 steps) | 6.76, 6.66, 6.67 s (reckoned 5.9) |
| **the pass over 403 rows, then 101 steps** | **2.87, 2.49, 2.43 s** (bar: 3.5): the pass with its load and saves 1.05, 0.69, 0.64; the steps with their load 1.79, 1.76, 1.76 |
| the first 16 ids after the prompt, the two ways | the same 16, each run |

* **The bar is met and the two ways agree** on the long prompt too, though its keys and values are rounded to f16 one way and not the other; on a sequence that means nothing this was reported, not required.
* **Where the 2.4 s go now:** two loads (about 0.5 s each: 4 GB and 6.3 GB), a pass of 0.17 s, and 100 steps at about 12.5 ms with caches of 512 positions (11.5 at 128). The answer's steps are the larger part of a long clip.
* **Held in the gate:** `kv_export` against float64 (`tests/kernels.py`; **[corrected with A4]** its two mutants are not in the gate, which runs that test without `--mutants`: run by hand after the PR was opened, both killed), its goldens, the two `PRE=1` expansions (`tests/model.sh`), and in `tests/qwen3_asr_step.sh` the tables from a first position and the check from one.
* **Not done:** one session for the pass and the steps (two loads, and the weights twice on disk); the long prompt is made up, so nothing says its answer is right, only that the two ways give the same; the clip is still the only real one.

## 14. A4 (design, 2026-11-24, written before its code): from a wav file to words

After A3b the sessions transcribe the reference clip, but from the reference's mel and the reference's token ids. A4 is what is left between a file and a line of text, in two parts.

**[read, before this was written]**

* **The prompt needs no tokeniser.** With no context it is 9 fixed ids (`<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>`), one `<|audio_pad|>` (151676) a tower row, and 6 fixed ids (`<|audio_end|><|im_end|>\n<|im_start|>assistant\n`): read off A0's 85 ids, 9 + 70 + 6. A context (the system text) would need the tokeniser and is not in A4.
* **The answer is `language <name>` `<asr_text>` (151704) the words, then `<|im_end|>` (151645).** Cut by id; the words are ordinary byte-level BPE tokens.
* **The checkpoint has no `tokenizer.json`** (it ships `vocab.json` and `merges.txt`), which `src/bpe.cho` reads. Made by hand with the package, a `tokenizer.json` decodes A0's answer to the letter with `cancho-gpu detokenize`; but decoding needs only `vocab.json` (id to token) and the byte-level table of GPT-2, so **the answer is decoded from `vocab.json` in cancho**, and nothing is made by hand.
* **The mel is Whisper's** (`transformers`' `WhisperFeatureExtractor`, numpy): the samples reflected 200 each side, frames of 400 every 160 under a periodic Hann window, the power of the 201 bins, 128 triangular filters on the Slaney mel scale from 0 to 8,000 Hz with Slaney's norm, `log10` floored at 1e-10, the last frame dropped, floored again at the clip's maximum less 8, then `(x + 4) / 4`. The clip is not padded to 30 s.
* **The package resamples to 16 kHz with librosa**, whose filter is not reproduced here.

### A4a: the host pieces, each against the reference

| piece | where | check |
|---|---|---|
| a wav file: RIFF, PCM of 16 bits or float of 32, any number of channels (their mean), any rate | `tests/qwen_asr/wav.cho` | fixtures in the gate: a mono and a stereo file of a few samples, a truncated one, a format it does not read: each a refusal with its status, none a crash |
| to 16 kHz when the rate is another: a windowed-sinc filter (Kaiser, stated in the code), in float64 | the same | **the mel of the 24 kHz reference clip resampled here against the reference's (resampled by librosa): reported, not bounded** (two filters); and the transcript below |
| the mel | `tests/qwen_asr/mel.cho`, float64 on the host | **from the reference's own 16 kHz samples: within 2e-4 of the reference's mel at every value** (it computes in float64 and keeps float32; the values are of order 1). Mutants: the window left out, the last frame kept, the floor at the maximum less 8 left out |
| the prompt's ids from the number of tower rows | the host program of A3 | equal to A0's 85 ids |
| the answer's ids to `language` and text | `tests/qwen_asr/answer.cho` | A0's 18 ids give `English` and the sentence, to the letter; an id that is not in the vocabulary, an answer with no `<asr_text>`: refusals |
| **a clip longer than 8 s** | a second reference run by hand (S0 then A0, three sentences) | **which of the two the reference's tower does, attention in windows of 104 rows or across them (2.1's doubt): the GPU's rows with each mask against the reference's float32 rows. The one that meets A1's bound is the one built; the other must fail it.** Then the transcript, end to end, with the host's own mel |

**The whole of A4a's bound: the two reference clips, from their wav files to their words, through the host's mel, A1, the prompt, A3b and the answer, give the reference's transcripts to the letter.**

**Time is reported for the mel** (a direct transform is 400 x 201 products a frame, 100 frames a second: 8 M a second of audio, in cancho on one core). If 30 s costs more than half a second it becomes two products on the GPU (`docs/roadmap-models.md` G15), not in A4a.

### A4b: `scripts/media transcribe CLIP -o DIR`

The stages as the media program runs its others (`media/`, the stage runner), the sessions given their two lines on their input; the weights written on first use, as umT5's are; `DIR/transcript.txt` and the text on the output. Refusals with the media program's rule tags: no such file, not a wav it reads, no audio, longer than the caches allow, the checkpoint not in the cache, no Metal session. Designed in detail when A4a is in, since its stages are A4a's programs.

**Not in A4:** a context or a forced language (they need the tokeniser on the prompt's side), timestamps (another model), streaming, clips over the 20 minutes the reference cuts at, any container but wav.

## 15. A4a: from a wav file to words **[measured, 2026-11-24, M4 Max]**

`tests/qwen3_asr_front.cho` and its modules in `tests/qwen_asr/` (`wav.cho`, `mel.cho`, `answer.cho`; cancho, no Python), `tests/qwen3_asr_transcribe.sh` (the stages in a row; it needs the GPU and the weights), and a second reference clip made by hand as the first was: three sentences, **16.24 s**, 1,624 frames of mel, 211 tower rows, a prompt of 226 ids, an answer of 62 (`~/.cache/cancho-media/audio-ref/s1.*`, `a1_*`).

**The doubt of 2.1 is settled: the reference attends across the 8 s windows.** On the 16 s clip, from the reference's mel:

| the audio tower's 211 rows against the reference's float32 | mean abs error | rows nearer than the yardstick's (BF16 on MPS: 1.28e-4) |
|---|---|---|
| attention over the whole clip | **6.8e-6** | **211 of 211** |
| attention in windows of 104 rows, as the configuration reads | 3.0e-3 | 0 |

The package on `transformers` without flash-attention never applies the windows it computes. The session does what the reference does; the windows are mutant 5 of `tests/qwen3_asr_audio_gpu.cho`, which fails on this clip (and cannot on the first, which is one window). Whether the model was trained with the windows, and so whether the reference is right, is not something these runs say.

| piece | against the reference | |
|---|---|---|
| the mel from the reference's own 16 kHz samples (bound 2e-4) | largest difference **1.9e-5** and 1.7e-5 on the two clips (mean 1.8e-7) | met |
| the mel from the 24 kHz files, resampled here | largest 0.16 and 0.15, mean 1.6e-3 and 1.3e-3: another filter than librosa's | reported |
| the prompt's ids for 70 and for 211 rows | the reference's 85 and 226 ids | equal |
| the answer's ids as text, from `vocab.json` | `English` and both transcripts | to the letter |
| **the two clips from their 24 kHz wav files, every stage the host's and the sessions'** | **the reference's two transcripts** | **to the letter** |

**Time, cold, every process new, three runs** (`TIMES=1 tests/qwen3_asr_transcribe.sh`):

| | all | the mel (with the resampling) | the tower | the prompt's pass | the answer |
|---|---|---|---|---|---|
| the 16.2 s clip, 65 steps of answer (it needs 62) | **2.63, 2.61, 2.61 s** | 0.27 | 0.26 to 0.27 | 0.55 to 0.56 | 1.32 |
| the 5.4 s clip, 24 steps | 1.85, 1.86, 1.85 s | | | | |

For scale, the reference package transcribes the 16 s clip in 3.7 s in float32 on the CPU and 1.6 s in BF16 on MPS **with its model already loaded**; its load was not timed, so the two are not compared here.

* **The mel alone** is 0.13 s for the 16 s clip from 16 kHz samples: about 0.25 s for 30 s, under the half second that would have sent it to the GPU. The first resampler took 1.6 s for the same clip (a Bessel series of 60 terms at each of 26 M taps); the window is a table now and the mel moved by 9.5e-7.
* **The answer is a fixed number of steps** (the script's third argument, 192 if not given): a clip whose answer is shorter pays for the rest (3.0 s at 192). **[corrected with A4b, section 16]** This said the media command would ask for steps until the end token; a session cannot end on a value, so the command fixes a budget from the clip's length.
* **Held in the gate without a GPU or the checkpoint** (`tests/qwen3_asr_front.sh`, 41 checks): the mel within 2e-4 of a fixture the reference's feature extractor made (`tests/qwen_asr/fixture`: 0.2 s of two tones and noise, its second half 100 dB down so that the floor is at work; 4.7e-6), three mutants of the mel far from it, wav files read (stereo PCM of 16 bits, float of 32, a rate of 8 kHz) and refused (8 bits, 4 kHz, an extensible format with no sub-format, a data chunk longer than the file, under 400 samples, not RIFF, four bytes, no file), the prompt's ids, and answers decoded from a made-up vocabulary, with their refusals (no `<asr_text>`, no end, an id with no token, a token outside the alphabet).
* **Not done:** the resampler's numbers have no test of their own (only that 8 kHz gives twice the samples); the fixture and the two clips are the reference's by hand, with scripts that are not in the repository; two clips, one voice, English, both synthetic speech from S0; no clip over 16 s, so nothing past 256 rows of tower or 448 of prompt was transcribed for real; PCM of 24 bits and every container but wav are refused; the tower's weights are now written once and found by one file's presence, which a half-written directory would fool.

## 16. A4b (design, 2026-11-24, written before its code): `scripts/media transcribe CLIP -o DIR`

A4a's stages are run by a shell script that is not in the gate. A4b runs them from the media program (`media/`, cancho), as the image and the video are run, and holds the command in the gate without a GPU.

**[read, before this was written]**

* **The media program can start only a program of `scripts/`** (`media/stage.cho`: the directory is opened once and the child is started under it). The picture encoder has `scripts/enc` and `scripts/enc-session` for that. Recognition gets the same two: `scripts/asr WHICH ARGS...` (the four host programs of A1 to A4a: `front`, `audio`, `text`, `step`) and `scripts/asr-session STEPS MODEL NAME=value...` (the Metal session, given its two lines).
* **The stages pass numbers to one another** (the mel's frames, the tower's `BP` and `SP` and its rows, the prompt's length, the pass's `SP`): the script reads them with `sed`. The media program reads them from what each stage prints (`stage.capture`), so a stage's printed line is its interface; the gate test holds those lines.
* **A session runs the number of steps it is given** and cannot end on a value it computes. **[corrected here: section 15 said the media command "is to ask for steps until the end token"; it cannot, and does not.]** The answer therefore has a budget of steps fixed before it starts.
* **There is no Python `transcribe`** and none is written (the media program's other commands are checked against `tools/media.py`; this one has nothing to be checked against but its own stages).

**The command.** `scripts/media transcribe CLIP -o DIR [--most N]`: `DIR/transcript.txt` (the language on the first line, the text on the second), the same two lines on the output after the stages' lines, the intermediate files in `DIR/work` (which `CANCHO_MEDIA_CLEAN_WORK=1` removes, as for the others). The wrapper takes the GPU lock as for an image. With no `build/media`, or with `CANCHO_MEDIA_IMPL=python`, the wrapper refuses: there is no other implementation.

**The stages**, each a line `== label` and its time, to a hundredth of a second (the whole job is two or three seconds; the other commands print whole seconds):

| | program | reads from what it prints |
|---|---|---|
| the language model's weights, on first use (6.3 GB) | `asr text weights`, `asr step weights` | |
| the mel | `asr front mel CLIP work/clip.mel.f32` | `frames F` |
| the tower's inputs (and its weights on first use, 0.6 GB) | `asr audio prepare SNAPSHOT WEIGHTS work work/clip.mel.f32 F` | `BP=.. SP=.. tokens R` |
| the audio tower | `asr-session 1 models/qwen3_asr_audio.model ...` | |
| the prompt | `asr front prompt R work/prompt.txt work/prefill.txt` | `prompt P` |
| the pass's inputs | `asr text inputs WEIGHTS work work/prefill.txt work/rows.f32` | `SP=S` |
| the prompt in one pass | `asr-session 1 models/qwen3_asr_text.model ... PRE=1 T=..` | |
| the steps' inputs | `asr step inputs WEIGHTS work work/prompt.txt work/rows.f32 N+1 T P-1` | |
| the answer | `asr-session N+1 models/qwen3_asr_step.model ... PRE=1` | |
| the answer as text | `asr front answer SNAPSHOT/vocab.json work/ids_out.f32 0` | the two lines |

**The budget of steps** is `--most N`, or 6 for each second of audio plus 16 **[reckoned: the two clips' answers are 3.4 and 3.8 tokens a second of audio, the language's two tokens and the end included]**. An answer that has not ended in its budget is refused (`the answer did not end in N tokens: pass --most`), not written cut. A step is 11.5 to 12.5 ms (section 13), so the 16 s clip pays for about 50 steps it does not use: 0.6 s **[reckoned]**.

**The longest clip.** **[done, section 17: 152 s; the clips made were 38, 76, 151 and 303 s]** A4a transcribed nothing over 16 s. Before the command has a bound, clips of about 38 s and 76 s (the two reference clips one after another, by hand) are transcribed by the reference and by the stages; **the command refuses a clip longer than the longest that gave the reference's words**, and the refusal says so.

**Refusals** (`media: ...`, status 2, before any stage): no `-o` or no clip; a `--most` that is not 1 to 4096; no such file; `build/cancho-gpu-metal-session`, or one of the four host programs, not built (with the command that builds them: `scripts/build-asr.sh`, new); the checkpoint not in the Hugging Face cache; under 8 GB free when the language model's weights are still to be written. From a stage (status 1, the stage's own line passed on): a file that is not a wav it reads, a clip under 25 ms, a clip over the bound, an answer with no end.

**Checks.**

* **In the gate, no GPU and no checkpoint** (`tests/media_transcribe.sh`): a temporary root whose `scripts/asr` and `scripts/asr-session` print their arguments and the lines the real stages print; the argument list of every stage for a clip of 1,624 frames, `transcript.txt`, the budget from the frames and from `--most`, the weights stages run when the weights are not there and not when they are, every refusal above with its status, a stage that fails and one whose line cannot be read. The stages' real lines are held by the four host tests already in the gate.
* **By hand, on the GPU: the two reference clips through `scripts/media transcribe` give the reference's transcripts to the letter**, and the time of the command beside the script's at the same number of steps.

**Not in A4b:** a context, a forced language, any container but wav, a `transcribe` entry in `media models` (that text is compared with `tools/media.py models`, which has no such command), the studio.

## 17. A4b: `scripts/media transcribe` **[measured, 2026-11-24, M4 Max]**

`media/transcribe.cho` (cancho; the command of `build/media`), `scripts/asr` and `scripts/asr-session` (the two programs it starts), `scripts/build-asr.sh` (the four host programs), `tests/media_transcribe.sh` (in the gate).

```sh
scripts/media transcribe clip.wav -o out/clip        # out/clip/transcript.txt: the language, then the text
```

**The bound of section 16 is met: the two reference clips give the reference's transcripts to the letter**, from their 24 kHz wav files.

| cold, every process new, three runs | `scripts/media transcribe` | the script of A4a at the same number of steps |
|---|---|---|
| the 16.2 s clip, a budget of 113 tokens (the answer is 62) | **3.03, 2.99, 3.07 s** | 3.30, 3.35, 3.36 s |
| the 5.4 s clip, a budget of 48 (the answer is 18) | 1.96, 1.93, 1.94 s | 2.22, 2.22, 2.22 s |

The script's own timing costs it about 0.3 s (a Python started for each lap), so the two are the same stages at the same price. Of the 16 s clip's 3.07 s: the mel 0.27, the tower 0.25, the prompt's pass 0.58, the answer 1.91, the six host steps 0.06 together. **The budget costs 0.4 s over A4a's 2.61 s at 65 steps** (reckoned 0.6): the price of a session that cannot stop at the end token.

**Longer clips**, made by hand of the two reference clips one after another (the long, the short, the long: 37.8 s, and that 2, 4 and 8 times), transcribed by the reference in float32 on the CPU:

| clip | prompt, budget, positions | from the reference's own 16 kHz samples | from the 24 kHz file, resampled here | time (24 kHz file) |
|---|---|---|---|---|
| 37.8 s | 507, 243, 768 | **the reference's transcript to the letter** | the reference's but for a comma (`gone, and` where it has `gone and`, twice) | 6.4 s |
| 75.7 s | 999, 470, 1,472 | **to the letter** | the same comma, four times | 11.1 s |
| 151.4 s | 1,983, 924, 2,944 | **to the letter** | not the reference's: see below | 23.1 s (20.9 from 16 kHz) |
| 302.7 s | | | refused by the session: `error[budget]`, 46,720 bytes of thread-group memory for 5,824 positions where 32,768 are allowed | |

* **The stages are the reference's to the letter up to 151 s; the resampler is not librosa's and moves a near tie.** The comma is the only difference at 38 and 76 s. On its own (section 15) the long clip's sentence has the comma in the reference too; inside the longer clip the reference drops it and these samples keep it.
* **At 151 s the reference itself leaves out 63 words** (one short sentence and one whole long clip after it, 4 of the speech's 28 sentences: 2,010 characters where the speech has 2,343). From the reference's samples the command leaves out the same 63 words, to the letter; from samples resampled here it writes all of them (its transcript is the 76 s one twice). So the text of a clip this long depends on rounding in the samples, in the reference as here, and neither is "the model's answer" more than the other. The clips repeat the same sentences, which is the worst case for this; no long clip of ordinary speech was tried.
* **The answer's session holds 4,032 positions**: its lowering needs 8 bytes of thread-group memory a position and 128 more (33,920 at 4,224, 32,896 at 4,096, measured as refusals), and the positions are a multiple of 64. At 13 rows and 6 tokens of budget a second that is about 3.5 minutes; the command refuses before the session does, with the numbers.
* **The command refuses a clip over 152 s**: 151.4 s is the longest that was compared, and its answer is already not stable.
* **A step costs more with more positions**: 16.5 s for 925 steps at 2,944 positions is 17.8 ms a step, where 128 positions cost 11.5 (section 11). The answer is 60 to 80% of a long clip's time.
* **The mel with its resampling is 2.4 s for 151 s from 24 kHz** (1.1 s from 16 kHz): 0.48 s for 30 s, and 0.22 s from 16 kHz: under the half second for 30 s that section 14 set for sending it to the GPU, narrowly when it resamples. Left on the host.

**Held in the gate without a GPU or the checkpoint** (`tests/media_transcribe.sh`, stages that log their arguments and print the real stages' lines): the nine argument lists of a job, the budget and the positions from the frames and from `--most`, the clip after `--`, the two weights programs run first when `embt.f16` is not there and the disk check before them, ten refusals before any stage and the disk check (status 2, no stage run), ten after one (status 1, the stage's own line passed on, no transcript written: a file the mel refuses, a stage that fails, a session that fails, a stage whose line has no number, an answer with no end, a clip over 152 s, more positions than 4,032), the two bounds met exactly (152.00 s, 4,032 positions), and the wrapper's refusal when asked for the Python.

**Not done:** a session that ends at the end token (the budget is paid in full, and an answer longer than it is refused, not continued); one session for the pass and the steps; the comma: a resampler that gives librosa's samples, or a statement of which filter the model was trained behind; clips of ordinary speech, other voices, other languages; any clip between 152 s and the reference's 20 minutes (the reference cuts long audio into pieces itself; that is not built); a context, a forced language, timestamps; `media models` does not list the command; the studio does not offer it; the weights' directory is trusted when its last file is there.

## 18. S1 (design, 2026-11-24, written before its code): the codec's decoder, in three parts

Section 3.3 set S1's bound: the waveform within 1e-3 of full scale of the reference's float32 at every sample, from the reference's codes. S1 is everything new in S (an operation, three kinds of convolution, a transformer with a window) and is built in three parts, each with its own check, so that a wrong waveform at the end has few places to come from.

**[read, before this was written]** `SnakeBeta.forward` of the reference: `x + sin^2(x e^alpha) / (e^beta + 1e-9)`, `alpha` and `beta` a channel. The two exponentials are of the weights, not of the activations: the host writes `a = e^alpha` and `ib = 1 / (e^beta + 1e-9)` once, in float64 rounded to f32, and the kernel needs `sin` and nothing else new.

### S1a: `sin`, and the Snake activation

* **`sin` is an elementwise operation of the kernel language**, beside `tanh`: the parser's word, the elaborator's kind, the interpreter (binary64 `sin` rounded once to binary32, as `exp` and `tanh` are), and both lowerings (`sinf` on CUDA, `precise::sin` on Metal). No new rule tag: a misuse is the refusals an operation already has.
* **Accuracy, stated as `docs/chx.md` section 7 states the others': open, and measured.** The float64 reference holds the interpreter to 1e-5 relative; on the Metal GPU, `sin` of arguments over [-64, 64) (`tests/chx/sin_scaled.chx`: `sin(128 x)`, x in [-0.5, 0.5)) is compared with the interpreter and the worst difference reported. What arguments the decoder's Snakes see is measured in S1c, and if they pass 64 the range here is widened. **[done with S1b, section 21: they reach 95, and the test is `sin(256 x)`, over [-128, 128)]**
* **`tests/chx/snake.chx`**: `y = x + sin^2(x a) ib` for s rows (positions) of c channels, `a` and `ib` a channel.
* **Held:** both kernels against float64 (`tests/kernels.py`), with mutants (`sin` a `tanh`; the square left out; `ib` left out), goldens for both targets (`tests/cases.txt`), and the Metal run by hand (`scripts/metal-kernels.sh`).

### S1b: the three convolutions

A causal 1-D convolution of k taps with a dilation (the taps a loop of products over shifted row windows, rows of zeros in front); a transposed one of stride r and 2r taps (r products, one an output phase, interleaved); a depthwise one of 7 taps (elementwise products of shifted rows with a `[7, c]` table). Each against float64 at a small shape, and against the reference's own layer at its real shape from tensors hooked by hand. Designed in detail when S1a is in.

### S1c: the decoder

The codebook sums, the 8 transformer layers with their window of 72, the two ConvNeXt stages, the four Snake blocks, the model file and the session; the reference's activations hooked at each stage by hand, so that the first stage past the bound is named. **The bound of 3.3 is S1c's.**

## 19. S1a: `sin` and the Snake activation **[measured, 2026-11-24, M4 Max]**

`sin` is an operation of the kernel language (`docs/chx.md` sections 3, 4 and 7; `src/parser.cho`, `elab.cho`, `kinds.cho`, `interp.cho`, `f32.cho`, `lower.cho`, `dialect.cho`: seven small additions beside `tanh`'s), and `tests/chx/snake.chx` is the decoder's activation, `x + sin^2(x a) ib`.

| | against | |
|---|---|---|
| `sin(128 x)`, the interpreter, both targets | float64 (`tests/kernels.py`) | within its 1e-5 |
| `snake`, the interpreter, both targets | float64 | within its 1e-5 |
| `sin(128 x)` on the Metal GPU, 65,536 arguments over [-64, 64) **[with S1b: `sin(256 x)`, over [-128, 128), the same worst difference]** | the interpreter | **worst difference 1.2e-7** (a unit in the last place at 1) |
| `snake` on the Metal GPU, 64 x 1,536 (the decoder's widest) and 24,000 x 96 (a second of audio at its narrowest) | the interpreter | worst 6.0e-8 of a peak of 0.53 |

* **Mutants, all killed by numbers** (run by hand; the gate runs `tests/kernels.py` without `--mutants`): `sin` a `tanh`; the argument scaled by 127; `x sin` where the square of the sine is; the periodic part subtracted; `a` added where it multiplies. A first mutant (the square left out) stopped on a refusal, not on a number (a tile left unconsumed), and was replaced.
* **In the gate:** the two float64 references, and the goldens of five cases for both targets (`tests/cases.txt`).
* **Not done:** the CUDA lowering writes `sinf` and was not run on a device (there is none here; the gate's emulator stage runs it on the host); arguments past 64 in magnitude; what arguments the decoder's Snakes see (S1c); the inputs of `snake`'s test are all under 0.5 in magnitude, so its sine is never past a quarter period there: the range is `sin_scaled`'s to hold.

## 20. S1b (design, 2026-11-24): the decoder's convolutions

**[read: `CausalConvNet`, `CausalTransConvNet`, `ConvNeXtBlock` and the decoder's `__init__` of the reference]** Every convolution of the decoder has stride 1 or is transposed. A position is a row and a channel a column, as everywhere in the sessions.

| the reference's layer | where | as a kernel |
|---|---|---|
| `CausalConvNet(cin, cout, k, dilation)`: zeros in front for `(k - 1) dilation` positions, then `Conv1d` | `pre_conv` (512 to 1,024, 3 taps), the decoder's first (1,024 to 1,536, 7 taps) and last (96 to 1, 7 taps), each residual unit's first (7 taps, dilation 1, 3 or 9) and second (1 tap) | **`tests/chx/conv1d_f.chx`**: `y[t, co] = sum_j sum_ci w[co, j cin + ci] x[t + j dl, ci]` over an input with its rows of zeros in front, the taps a loop of matrix-unit products over shifted row windows (as `conv3.chx`). The bias is the next kernel's |
| `CausalTransConvNet(cin, cout, 2r, r)`: `ConvTranspose1d` of 2r taps and stride r, the last r outputs cut | the four blocks (r = 8, 5, 4, 3) | **the same kernel, no new one.** The output at `u r + ph` (`ph < r`) is `x[u] W[ph] + x[u - 1] W[ph + r]`: a convolution of 2 taps to `r cout` outputs, tap 0 the weights `W[.., ph + r]` and tap 1 `W[.., ph]`, phase outermost in the output's columns. A row of `[s, r cout]` is then r rows of `[s r, cout]`: the interleaving is the buffer read with another shape. **[corrected: 3.2 reckoned "r convolutions of 2 taps, a kernel and an interleaving store"]** |
| `CausalTransConvNet(c, c, r, r)`: taps equal to the stride | the two rate-doubling stages (r = 2) | the same with one tap: a product to `r c` outputs |
| the ConvNeXt block's depthwise `CausalConvNet(c, c, 7, groups=c)` | the two stages (c = 1,024) | **`tests/chx/dwconv_f.chx`**: `y[t, ch] = b[ch] + sum_j w[j, ch] x[t + j, ch]`, elementwise over shifted rows |

**Both operands of the products are f32.** The codec's weights are F32 in the checkpoint (3), unlike the language models' BF16, so rounding them to f16 would be a change to the model (4.9e-4 of a weight at worst), and the decoder is some thirty convolutions deep against a bound of 1e-3 of full scale. The matrix unit takes f32 operands on Metal since V3c (`docs/diffusion.md` section 6); on CUDA it is refused (`lowering`), so `conv1d_f` is Metal's alone, as `gemm_mma_f` is.

**Checks.**

* **Against float64, in the gate** (`tests/kernels.py`): `conv1d_f` with 3 taps and a dilation of 2; **the transposed convolution written as the reference writes it** (every input position adds its 2r weighted taps into the output, the last r cut) against `conv1d_f` with 2 taps and the weights laid out as above, read with the other shape; `dwconv_f`. Mutants: a tap's window not shifted by the dilation, the taps' weights in one tap's columns, the two taps of the transposed layout exchanged (a mutant of the reference's reading, since the kernel is the same), the depthwise bias left out, its taps read from one row.
* **On the Metal GPU against the interpreter**, at the decoder's shapes (by hand, `scripts/metal-kernels.sh`).
* **Against the reference's own layers, by hand:** the input and output of one layer of each kind hooked from the reference on S0's codes, the checkpoint's weights laid out by a script that is not in the repository, the kernel run on those tensors. This checks the layout and the padding against torch, which the float64 references (written from the same reading) cannot. The cancho host that writes the weights is S1c's, held there by the waveform.

## 21. S1b: the decoder's convolutions **[measured, 2026-11-24, M4 Max]**

`tests/chx/conv1d_f.chx` (the causal convolution with a dilation, and the transposed one as section 20 reads it) and `tests/chx/dwconv_f.chx` (the depthwise one with its bias). Two kernels for the five kinds of layer.

**Against the reference's own layers on S0's codes**, by hand: the layer's input and output hooked from the reference (whose decoder alone, given S0's codes, gives S0's waveform exactly), the checkpoint's weights laid out by a script that is not in the repository, the kernel run in the interpreter on those tensors, the reference's bias taken off its output:

| the reference's layer | as | compared | largest difference | the output's peak |
|---|---|---|---|---|
| `pre_conv`, 512 to 1,024, 3 taps | `conv1d_f k=3` | 67 positions, 128 of the outputs | 1.0e-6 | 0.92 |
| the decoder's first, 1,024 to 1,536, 7 taps | `conv1d_f k=7` | 268 positions, 64 of the outputs | 3.0e-5 | 13.4 |
| a residual unit's, 96 channels, 7 taps, dilation 9 | `conv1d_f k=7 dl=9` | the first 6,400 positions, all outputs | 4.3e-6 | 9.0 |
| the decoder's last, 96 to 1, 7 taps | `conv1d_f k=7`, 31 rows of zero weights | the first 6,400 positions | 1.5e-8 | 0.072 |
| transposed, 768 to 384, stride 5, 10 taps | `conv1d_f k=2`, read as 5 rows a position | 640 positions (3,200 rows), 64 of the outputs | 1.0e-6 | 1.21 |
| transposed, 192 to 96, stride 3, 6 taps | the same | 640 positions (1,920 rows), all outputs | 3.8e-7 | 0.61 |
| transposed, 1,024 to 1,024, stride 2, 2 taps | `conv1d_f k=1`, read as 2 rows a position | 67 positions (134 rows), 256 of the outputs | 1.0e-6 | 1.32 |
| depthwise, 1,024 channels, 7 taps, with its bias | `dwconv_f` | 134 positions, all channels | 2.5e-6 | 26.6 |

Every one is within 2.3e-6 of its peak. Where only some outputs are compared it is because the interpreter reads a tensor as text of at most 16 MiB: the weights of those outputs are whole (every input channel, every tap); a later position cannot change an earlier one in a causal layer, so a prefix is the layer's own output. **Not compared: the transposed convolution of stride 8 (1,536 to 768) and of stride 4**, whose layout is the one of strides 5 and 3.

| on the Metal GPU against the interpreter, the test's own inputs | worst difference, of the peak |
|---|---|
| `conv1d_f` 512 to 1,024, 3 taps, 64 positions | 2.0e-6 |
| `conv1d_f` 1,024 to 1,536, 7 taps, 256 positions | 3.5e-6 |
| `conv1d_f` 768 to 1,920 (the transposed of stride 5), 2,048 positions | 1.5e-6 |
| `conv1d_f` 96 to 96, 7 taps, dilation 9, 2,432 positions | 1.0e-6 |
| `dwconv_f` 1,024 channels, 128 positions | 1.3e-7 |

* **The Snakes' arguments reach 95** (`x e^alpha`, in the fifth block's; the largest activation entering one is 135), measured on S0's codes: past the 64 that S1a's test covered. `tests/chx/sin_scaled.chx` is now `sin(256 x)`, over [-128, 128), with the same worst difference on the GPU (1.2e-7); section 19's table is marked.
* **Against float64, in the gate:** `conv1d_f` with 3 taps and a dilation of 2; the transposed convolution of stride 3 written as the reference computes it, against `conv1d_f` with 2 taps; `dwconv_f`. **Mutants, run by hand, all killed by numbers:** a tap shifted by one position where the dilation is 2; every tap's weights read from the first tap; the first tap left out; the depthwise bias left out; its taps' weights from the first; its taps all reading the position itself; and, of the reference's reading, the two taps of the transposed layout exchanged.
* **Goldens** for seven cases; `conv1d_f` is refused on CUDA (`lowering`: an f32 operand of the matrix unit), as designed, and its goldens there are the refusal.
* **Not done:** the kernels on the GPU with the reference's tensors (the device runner takes the test's inputs only; real tensors on the GPU are S1c's session); the bias of `conv1d_f` (S1c's kernel); the rows of zeros in front are the caller's to provide, and nothing here checks that a session's buffers have them; the widest transposed layer (6,144 outputs of 3,072 inputs) does not fit the interpreter, so it has a golden and no numeric run; no time was measured.
