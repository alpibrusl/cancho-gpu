# Which models this machine can run: a survey of the Hugging Face Hub (2026-11-14)

What was **read** (from the Hub's API, model cards and configuration files, on 2026-11-14, by the method below) is marked **[read]**; what is arithmetic on it or a judgment is marked **[estimate]** or **[judgment]**. Nothing here was run on the GPU.

## The machine **[measured]**

Apple M4 Max, 64 GB of unified memory. Metal's `recommendedMaxWorkingSetSize` is **55.7 GB** and `maxBufferLength` **41.7 GB**; the largest sessions of this repository peaked at 24.7 GB (the 32-block transformer with its activations aliased) and, for the text encoder, 15.7 GB of weights. Disk: 86 GB free after clearing 58 GB of scratch.

**A fit rule [estimate]:** the weights as they would be run (f16: 2 bytes a parameter; NVFP4: about 0.55, the 27B's measured file) plus the activations (a few GB to tens of GB for a diffusion transformer at 4,000 to 30,000 tokens) must stay under about 50 GB, and one tensor under 41.7 GB.

## Method and its limits

The Hub's API, `pipeline_tag` in {text-to-video, image-to-video, text-to-image, image-to-image, text-to-speech, text-to-audio, any-to-any, image-text-to-text, text-generation, automatic-speech-recognition}, the 25 most downloaded and the 25 most liked of each, kept if created or updated since 2025-03 and not an obvious repackaging (names with gguf, mlx, awq, gptq, fp8, int4, lora, unsloth ...); then the nine most downloaded of each tag sized by summing the weight files of the repository (`?blobs=true`). **Limits:** downloads and likes are popularity, not quality; the size is of the repository as published (often bf16 or float32, sometimes several variants, sometimes ComfyUI bundles that repeat several models); a licence of "other" was not read; the pipeline tag is the uploader's.

## Candidates **[read]** (size of the repository as published; params from the safetensors metadata when it has them)

| Kind | Model | Params | Repository | Licence | Notes |
|---|---|---|---|---|---|
| video | Wan-AI/Wan2.2-TI2V-5B-Diffusers | 5.0 B | 34.2 GB (umT5-XXL 11.4, transformer 20.0 as float32, VAE 2.8) | Apache-2.0 | public |
| video | **FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers** | 5.0 B (bf16) | 24.2 GB | Apache-2.0 | DMD-distilled from the one above, **3 steps** |
| video | Wan-AI/Wan2.1-T2V-1.3B-Diffusers | 1.4 B | 28.9 GB | Apache-2.0 | |
| video | Wan-AI/Wan2.2-T2V-A14B-Diffusers, I2V-A14B | 14.3 B | 126.2 GB | Apache-2.0 | two 14 B experts |
| video | Lightricks/LTX-2 / 2.3 / 2.5 | 18.9 B | 156 to 314 GB | "other" (2.5 gated) | |
| video | MiniMax-H3 and its repackagings | 33 B | 74 to 498 GB | Apache-2.0 / other | |
| image | **Tongyi-MAI/Z-Image-Turbo** | 6 B | 32.8 GB (float32) | Apache-2.0 | 5,601 likes, **8 steps, no guidance** |
| image | black-forest-labs/FLUX.2-klein-4B | 3.9 B | 23.7 GB | Apache-2.0 | |
| image edit | Qwen/Qwen-Image-Edit-2509 / 2511 | 20.4 B | 57.7 GB | Apache-2.0 | |
| image | Comfy-Org/Qwen-Image-2.1 | | 122.8 GB (bundle) | other | the model of D0 to D12 |
| speech | **Qwen/Qwen3-TTS-12Hz-0.6B-Base, 1.7B-CustomVoice** | 0.9 B, 1.9 B | 2.5 GB, 4.5 GB | Apache-2.0 | ten languages **including Spanish** |
| speech | openbmb/VoxCPM2 | 2.3 B | 5.0 GB | Apache-2.0 | |
| speech | k2-fsa/OmniVoice | 0.6 B | 3.3 GB | not tagged | |
| recognition | Qwen/Qwen3-ASR-1.7B | 2.3 B | 4.7 GB | Apache-2.0 | |
| recognition | nvidia/parakeet-tdt-0.6b-v3 | 0.6 B | 2.5 GB | CC-BY-4.0 | |
| music | ACE-Step/Ace-Step1.5 | | 10.0 GB | MIT | |
| audio | stabilityai/stable-audio-3-small-music, -sfx, -medium | 0.6 B, 0.6 B, 2.3 B | 3.5, 3.5, 10.4 GB | "other", **gated** | |
| LLM | Qwen/Qwen3.8-27B | 27.8 B | 55.6 GB | Apache-2.0 | runs here (NVFP4, 27 tokens/s) |
| LLM | google/gemma-4-31B-it, 26B-A4B-it, 12B-it | 31.3 B, 25.8 B (MoE, about 4 B active), 12 B | 62.5, 51.6, 23.9 GB | Apache-2.0 | |
| LLM | nvidia/Qwen3.6-35B-A3B-NVFP4 | 18.7 B stored | 23.4 GB | Apache-2.0 | MoE, about 3 B active, already NVFP4 |
| LLM | openai/gpt-oss-20b | 20.9 B (MoE) | 41.3 GB | Apache-2.0 | |
| omni | Qwen/Qwen3-Omni-30B-A3B-Instruct | 35.3 B | 70.5 GB | other | |

## Facts read from the configurations of the candidates that matter

* **FastWan2.2-TI2V-5B-FullAttn**: a finetune of Wan2.2-TI2V-5B **with an identical transformer configuration** (every field compared: 30 layers, 24 heads of 128, `ffn_dim` 14,336, patch (1, 2, 2), 48 channels, `text_dim` 4,096), trained "using only DMD"; **3 steps** (`--dmd-denoising-steps 1000,757,522`); trained at 121 x 704 x 1280; guidance not stated on the card; text encoder `UMT5EncoderModel`, VAE `AutoencoderKLWan`, scheduler `UniPCMultistepScheduler`. Whether its weights differ from the base's was not checked.
* **Z-Image-Turbo**: a single-stream DiT (S3-DiT) of 30 layers, 30 heads, width 3,840, patch 2, 16 latent channels; the text encoder is a **Qwen3** causal LM (36 layers, hidden 2,560, 8 KV heads, head 128), text and image tokens "concatenated at the sequence level"; a KL VAE with four levels (128, 256, 512, 512) and 16 channels (its `_name_or_path` is flux-dev; the card does not say it is FLUX's); 8 NFEs, no guidance; its relation to Qwen-Image is not stated.
* **Qwen3-TTS**: an autoregressive Qwen3-family language model over **16 codebooks of a 12.5 Hz codec** (2,048 entries each; the first semantic, 15 acoustic), the residual codebooks generated by a multi-token-prediction module; the codec is a separate repository (`Qwen/Qwen3-TTS-Tokenizer-12Hz`) whose decoder is "a lightweight causal ConvNet"; the Qwen text tokenizer, no phonemizer; cloning (the Base models) needs a reference audio and its transcript, through a speaker encoder; the nine preset voices of CustomVoice are Chinese, English, Japanese and Korean speakers who can speak any of the ten languages. Backbone dimensions, the codec's sample rate and sizes are not stated on the pages read.

## What would be worth building, in order **[judgment]**

1. **Video: FastWan2.2-TI2V-5B.** It fits (transformer about 10 GB in bf16, text encoder 11.4, VAE 2.8), it is Apache-2.0 and public, its architecture is the one `docs/diffusion.md` section 6 designs, and **three steps instead of 50 with guidance would cut a 5-second 720p video from about an hour to a few minutes** (3 calls of about 55 s: an estimate from the shapes and this repository's measured rates). Needs: the transformer block with 3-D rotary positions and cross-attention, the attention in blocks of queries, the causal 3-D VAE as taps of the existing 2-D convolution, and umT5-XXL's encoder.
2. **Speech: Qwen3-TTS (0.6 B and 1.7 B).** Small (2.5 to 4.5 GB), Apache-2.0, Spanish among its ten languages, and its backbone is a Qwen3 language model, the family the decode runners and the training sessions of this repository already run. Needs: the backbone's runner at its dimensions (not stated on the pages), the multi-token-prediction module, the codebook embeddings, and the codec's causal convolution decoder (1-D causal and probably transposed convolutions, which the language does not have; a `sin` if it uses the Snake activation, which is not known). With Qwen3-ASR-1.7B (Apache-2.0, same family) it would make a speech loop.
3. **Image: Z-Image-Turbo.** 8 steps, no guidance, a Qwen3 text encoder that is the same code path as the text encoder of D9c at other dimensions (a model file with other constants), a single-stream DiT close to this repository's joint-attention blocks, and a KL VAE whose norms are group norms (a new kernel: the existing VAE kernels use a per-channel RMS norm).
4. **LLMs with few active parameters** (Qwen3.6-35B-A3B in NVFP4 at 23.4 GB, gemma-4-26B-A4B): decode would be far faster than the dense 27B, but a mixture of experts needs a gather by an index chosen by the data (issue #34, the index-valued load), which the language does not have.
5. **Not worth it here:** MiniMax-H3, LTX-2.x (19 B and gated or "other"), FLUX.2-dev (32 B), Qwen-Image-Edit-2509 (20 B: 41 GB in f16 with a 16 GB text encoder does not fit one session; NVFP4 for a diffusion transformer is not built), Qwen3-Omni (70 GB).
