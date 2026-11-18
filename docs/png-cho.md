# A PNG codec in cancho (2026-11-18)

> **Status: built (`media/png.cho`): decode and encode, differential against PIL through goldens.** Resize, the colour statistics, the GIF and the `video` command itself are the next slices. Part of `docs/media-cho.md` / `docs/python-removal.md` phase 2. Claims are **[measured]** unless tagged.

## 1. Why

`media video --image` and `--segments` read and write PNG frames, resize and crop a picture, and move per-channel statistics; cancho's `std` has no PNG, zlib or image codec (`docs/media-cho.md` §1). This is the codec, written in cancho, so `video` can stop needing PIL.

## 2. What it does

* **Decode** (`png.decode(heap, bytes) -> (status, w, h, rgb)`): an inflate (stored, fixed and dynamic Huffman blocks, the canonical decoder of zlib's `puff.c`), the five row filters, and the conversion to 8-bit RGB as `Image.open(p).convert("RGB")` does it: gray, gray+alpha, palette, RGB, RGBA; bit depths 1, 2, 4 and 8 for gray and palette, 8 for the rest; alpha is dropped, not composited (as PIL does). Chunk checksums, the zlib header and the Adler-32 are checked.
* **Encode** (`png.encode(heap, w, h, rgb)`): 8-bit RGB, filter 0 on every row, zlib **stored** blocks (no compression): a 1024 x 1024 picture is 3.1 MB. Every reader accepts it; compressing is a later choice if the size ever matters (the frames go to ffmpeg and a GIF, not to the network).
* **Refused, with a message** (`png.message`): not a PNG, damaged (chunk length or checksum, zlib header, compressed data, data that does not fill the size, a row filter above 4), and **unsupported**: 16-bit, interlaced, a bit depth other than the above for RGB/alpha types, a side over 16384. No input reaches a panic: every index is checked against its length first.

## 3. Checked

`tests/png_diff.sh` (in the gate) decodes the 22 pictures of `tests/png/` and compares `W H sha256(RGB)` with what PIL 12.2 gave when the goldens were made (the gate itself runs no Python): PIL at compression levels 0, 1, 6 and 9, noise, blocks, RGBA, gray, gray+alpha, 1-bit, palettes of 2, 4, 16 and 256 colours, 1 x 1, 300 x 2, and each filter type forced by hand, with a Paeth tie-breaking case. It also decodes five damaged or unsupported files and checks their messages and exit status, decodes then encodes then decodes every picture (the bytes must come back), and compares the encoder's output for a 3 x 2 picture with a golden hash (also read by PIL once, equal). **Mutants:** two Paeth tie-break changes fail 1 and 2 checks.

**Speed:** a 1024 x 1024 gradient decodes in 0.014 s and encodes in 0.011 s (a debug compiler, native code).

## 4. Resize and crop (`media/image.cho`)

`image.resize` is Pillow's LANCZOS, coefficient for coefficient (`Resample.c`: support 3, weights normalised then rounded to 22 fractional bits, a horizontal pass then a vertical, each rounded and clipped to a byte). `image.cover` is `picture_f32`'s geometry (scale to cover, Python's round-half-even for the size, centre crop) and `image.planar_f32` writes `[3, h, w]` little-endian singles of `v / 127.5 - 1` in single precision. **Checked:** `tests/png/cover.list` holds 15 cases whose sha256 is what `tools/media.py picture_f32` wrote (shrinking, enlarging, square, wide, 1 x 1, palette and RGBA sources, 1024 x 1024 to 128 x 96 and 1100 x 700), all **byte-identical**; mutants (the fixed-point scale, the rounding term, round-half-even instead of half-up) fail 2, 11 and 1 checks. A 1024 x 1024 picture to 1100 x 700 takes 0.06 s.

## 5. Colour statistics and the ramped correction (`image.stats`, `image.ramp`)

`stats` is `frame_stats` of `tools/media.py`: the per-channel mean and standard deviation as numpy computes them (the sums down the rows in order, the root of the mean squared difference); `ramp` is `ramp_correct`'s arithmetic for one frame, with the result cut (not rounded) to a byte as `astype(np.uint8)` does. **Checked** (`tests/png/stats.list`, `ramp.list`): the six statistics of seven pictures are **bit for bit** what numpy gave (compared as the bits of each double), and eight ramped frames (first, middle and last of chains of 2 to 5, other pictures as the reference, RGBA and palette sources, a 1024 x 1024 one) are **byte-identical** to the ones `ramp_correct` writes. **Mutants:** rounding instead of cutting fails 8 checks, the ramp's `n - 1` replaced by `n` fails 6; changing the `1e-6` guard to `1e-5` is **not** caught (its effect is far below one level on these pictures), so that constant is checked by the arithmetic being otherwise exact, not by a test.

## 6. Not here

The GIF (stays `ffmpeg`), and the `video` command itself: the next slices.
