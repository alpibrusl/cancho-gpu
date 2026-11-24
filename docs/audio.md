# Audio: recognition, then speech, then music (design, 2026-11-24; the reference runs made; A1 and A2 built: the reference clip is transcribed on the GPU, token for token; A3 and A3b built: 11.5 ms a token, the prompt in one pass)

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
* **Held in the gate without a GPU or the checkpoint:** the four kernels against float64 with seven mutants (`tests/kernels.py`; three first drafts of those mutants died of a refusal to compile and were replaced by ones that die on the numbers), their goldens, the model file's expansion, and `tests/qwen3_asr_step.sh`: the host program builds, refuses with its documented statuses, writes the tables of four steps with the right sizes and bytes, and reads a made-up answer of the device.
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
* **Held in the gate:** `kv_export` against float64 with two mutants (`tests/kernels.py`), its goldens, the two `PRE=1` expansions (`tests/model.sh`), and in `tests/qwen3_asr_step.sh` the tables from a first position and the check from one.
* **Not done:** one session for the pass and the steps (two loads, and the weights twice on disk); the long prompt is made up, so nothing says its answer is right, only that the two ways give the same; the clip is still the only real one.
