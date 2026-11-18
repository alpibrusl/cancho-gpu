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

## 4. Not here

Resize (LANCZOS), centre crop, the per-channel mean and spread, the ramped correction (they are float work on the RGB bytes), the GIF (stays `ffmpeg`), and the `video` command: the next slices.
