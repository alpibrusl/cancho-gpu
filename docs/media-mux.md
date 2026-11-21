# Writing video and audio files in cancho (the part of ffmpeg this repository needs)

Status: **design, nothing built.** Written 2026-11-15. Claims are tagged **[read]** (from a source file named here), **[estimate]** or **[not checked]**.

## 1. Why

`scripts/media video` writes `video.mp4` by running the external `ffmpeg` (`tools/media.py`, `cmd_video`), and the video models here make no sound (docs/cli.md). The aim is a cancho program that turns what this repository produces (frames from the VAE decoder, later audio from a speech or sound model) into a file a browser and QuickTime play, with no call to another program. It is **not** an ffmpeg: it reads no foreign formats, filters nothing, and encodes only what is listed in section 3.

## 2. What cancho already has [read, `std/` of the compiler at the pinned commit and `src/ckpt.cho`]

* `std/flate.cho`: **inflate and `deflate`** (line 1354: `deflate(heap, src) -> Buffer`) and `std/crc.cho`: `crc32_*`. So a **PNG writer** needs only the chunk framing.
* `std/buffer.cho`: a growable byte `Buffer` (`push`, `append`, `push_nat`, `write`); `std/bytes.cho`: `store_le32` (only the **little-endian** store: MP4 is big-endian, so `store_be16/32/64` are to be written, ten lines).
* `src/ckpt.cho`: the write path of a binary file under the session's authority (`write_all`, a temporary file then `finish`), the model for ours.
* Not in `std/` [not checked beyond a listing and greps]: a bit writer (needed from M3), an FFT or DCT, any codec.

## 3. Scope, in stages; each stage ends in something that plays

| Stage | What | Why it is separate | Oracle (a test input or comparison only, never a runtime dependency) |
|---|---|---|---|
| **M1** | **MP4 muxer** (ISO BMFF) for: one H.264 video track from *already encoded* samples (Annex B to AVCC, `avcC` built from the stream's SPS and PPS, `stss` for key frames, no B-frames so no `ctts`), one audio track of PCM (`sowt`/`lpcm`) or AAC samples (`esds`); `moov` before `mdat` (fast start). Also **WAV** and **PNG** writers. | Container errors are the commonest way to get a file that plays nowhere; testing it needs a known-good bitstream, which an external encoder gives, so it comes before our own codec. | `ffprobe -show_streams -show_packets` must equal the streams' parameters and every packet's offset, size, time and key flag; `ffmpeg` decoding our remux must give the **same frames** (md5 of the decoded raw video) as decoding the source stream. |
| **M2** | **H.264 encoder, key frames only** (Baseline/High intra, CAVLC, 4 x 4 and 16 x 16 intra prediction, integer transform, a fixed quantiser): our own frames into samples for M1's muxer. The transform and quantisation are matrix products and go on the GPU (`mma`); the entropy coding is serial, on the CPU. | First step with a codec: browser-playable video, files 5 to 10 times larger than x264's **[estimate]**. | Decoding by `ffmpeg` of our stream against the encoder's own reconstruction (an encoder carries its decoder's output): equal frame by frame, and PSNR against the source frames. |
| **M3** | **AAC-LC encoder** (MDCT, scalefactor bands, a fixed bitrate, no psychoacoustic model beyond a band energy rule) and a bit writer. | Audio at a sensible size. | Decoded by `ffmpeg`: SNR against the input PCM, and a mutant set below. |
| **M4** | (optional) inter prediction (P frames), CABAC, rate control. | Only if the file size of M2 matters. Not designed here. | |

Not in any stage: reading arbitrary video or audio, resampling beyond a simple linear one, filters, subtitles, streaming containers (HLS, fragmented MP4), HDR, colour management beyond BT.709 limited range 4:2:0 for the video.

## 4. M1 in detail

**Input and output of the program.** `cancho-media-mux OUT.mp4 --video H264_ANNEXB --fps N [--audio WAV]`: it reads an Annex B H.264 file (from the oracle, in M1's tests; from M2 later) and a 16-bit PCM WAV, and writes OUT.mp4. Authority: reads the inputs, writes `OUT.mp4` through a temporary file renamed on success (`ckpt.cho`'s pattern), no network, no foreign code; it is a binary of its own in `cancho.toml` so that the session's `fs_write` authority (owner-approved, `tests/authority.sh`) is not widened.

**Boxes written** (big-endian, 32-bit sizes, `co64` only above 4 GiB): `ftyp` (`isom`, `iso2`, `avc1`, `mp41`); `moov` > `mvhd`, one `trak` per stream > `tkhd`, `mdia` > `mdhd`, `hdlr` (`vide`/`soun`), `minf` > `vmhd`/`smhd`, `dinf` > `dref` > `url `, `stbl` > `stsd` (`avc1` > `avcC`, or `sowt`, or `mp4a` > `esds`), `stts`, `stsc`, `stsz`, `stco`, and for video `stss`; then `mdat`. Timescale 90,000 for video, the sample rate for audio. Interleaving: the video sample then the audio samples up to its end time, so one chunk per sample (simple, larger tables, valid).

**What is hard, so measured first.** (1) The chunk offsets (`stco`) depend on the size of `moov`, which depends on their count: computed in two passes (sizes first, offsets second), and a mutant checks it. (2) The `avcC` must carry the SPS and PPS with the right profile bytes and a 4-byte length size, and the samples must use 4-byte lengths, not start codes. (3) Audio and video durations differ by up to a frame: the track durations are written as they are and the movie's is the longest; the edit list is not used, so playback starts at zero for both.

**Tests.**
* Pure part, in `cancho test`: the box writer against **expected bytes** written by hand for a three-sample toy (not produced by the writer or an external tool), and big-endian stores at the edges (0, 2^31, 2^32 - 1).
* With `ffmpeg`/`ffprobe` present (skipped without them, like the numpy tests of `scripts/gate.sh`): a 2-second test stream made by the oracle, remuxed, then compared as above, **and ten mutants, each of which must fail the comparison**: an `stco` off by the `moov` size; `stsz` one byte short for a sample; a missing `stss` (seek test: `ffmpeg -ss` lands on a non-key frame); `stts` with the wrong delta; `avcC` with a 2-byte length size; SPS and PPS swapped; `mdhd` timescale wrong; audio chunk offsets shifted by one sample; `moov` after `mdat` with unchanged offsets; `hdlr` types exchanged. A mutant that survives is a hole in the test, as in `tests/vae_pieces.py`.
* By eye, once: the file played in QuickTime and a browser's `<video>` (**not checkable from here: only a person with those players can say it**); recorded as such.

**Claims to measure before writing any, not assumed.** The time and memory of remuxing 33 frames at 1280 x 704 and one minute of 48 kHz stereo PCM (expected: milliseconds and the file's size). That the file is **valid** is what the oracle comparison says; that it **plays everywhere** is what no test here says.

## 5. M2 and M3 in outline (to be designed in full when M1 is merged)

* **M2.** Input: the decoder's frames as 8-bit RGB (a new kernel writes u8 planes in the session, replacing the host's PNG loop), a colour conversion kernel (BT.709 limited range, 4:2:0 by 2 x 2 averaging), then per 16 x 16 macroblock: intra mode decision (the sum of absolute differences of the nine 4 x 4 modes, a kernel), the 4 x 4 integer transform and quantisation as products, CAVLC on the host. A first version uses one fixed QP and the DC-only 16 x 16 mode and states the file size it gives. Held to: the encoder's reconstruction equals an independent decode; PSNR above a bound **written before the run**.
* **M3.** The MDCT as a matrix product (a long block of 2,048 samples; short blocks and window switching left out, with the pre-echo that implies, stated), bands, a uniform quantiser and Huffman codebooks of the AAC standard (the tables are data, read from the specification by hand, a source of typing errors: held by decoding every table's code once). No claim of transparent quality: the test is SNR and a person's listening.
* **Audio generation** is a separate model (docs/diffusion.md 6.2) and does not depend on these stages: M1 takes any WAV.

## 6. Risks, stated

* **[estimate]** M1 is the least risky stage and does not by itself replace ffmpeg in `tools/media.py`: until M2 the program muxes a stream someone else encoded. **Only M2 removes the dependency on `ffmpeg` for video**; M3 for audio.
* The H.264 specification is read, not built from memory: the tables of CAVLC and the intra predictors are long and a wrong entry gives a stream that decodes to the wrong picture, not an error. The reconstruction-equality test is what catches it, and it only catches what the test pictures exercise.
* The browser and QuickTime are stricter than ffmpeg (which accepts many broken files); the oracle is therefore necessary and **not sufficient**.
* `tools/media.py` and the other drivers are Python; this design does not move them, only the file writing.

## 7. Decisions needed

1. Binary of its own (`cancho-media-mux`, this design) or a subcommand of `cancho-gpu`? Chosen here: its own, for the authority.
2. Is browser-playable video required, or is QuickTime/VLC enough? If the latter, M-JPEG in a MOV replaces M2 and is about a third of the work **[estimate]**.
3. AAC or Opus for audio: AAC is chosen (the container support is universal; Opus's CELT/SILK is the harder codec).
