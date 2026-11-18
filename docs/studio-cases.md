# Studio cases: ready-made two-picture edits (2026-11-18)

> **Status: built (`/cases`), one case run for real (try-on); the other five are written, not run.** Claims are **[measured]** unless tagged.

## 1. What it is

A two-picture edit (`docs/two-images.md`) needs the right instruction: which element comes from which picture, what to change, what to keep. `/cases` writes that instruction for six common uses, so the person gives two pictures and presses a button. Each case is a form that posts to `/edit` (`studio/upload.cho`) with `image`, `image2`, `prompt` and an optional seed; **no new server logic**, so the same refusals, the same size limits and the same job apply. The instruction is in a text box and can be changed. The pictures are named `image1` and `image2`, as the model's template names them.

| Case | image1 | image2 |
|---|---|---|
| Try clothes on a person | the person | the clothes |
| Redecorate a room | the room | a style (another room, a mood board, a piece of furniture) |
| Put an object in a room | the room | the object |
| Put a person in another place | the person | the place |
| Try a hairstyle | the person | someone with the hair wanted |
| Use the style of another picture | the picture to restyle | a picture in the style |

## 2. Checked

`studio/check.sh` (414 ok, 0 failed): the page has six forms, each to `/edit` as multipart, each with `image`, `image2` and the instruction, no script, every instruction names both pictures, the header marks the place and the other pages link to it; submitting the try-on form with its own instruction and two files makes a two-picture edit (`--also`); the OpenAPI document has `/cases`.

## 3. Run for real: try-on

Pictures made by `media image` (no photograph of anyone): a woman with short dark hair in a white T-shirt and jeans (768 x 1024, seed 1), and a green quilted jacket laid flat (768 x 768, seed 1). `media edit person.png <the try-on instruction> --also jacket.png --seed 1`: **98 s** at the default condition area. **Looked at:** the person wears the jacket, zipped, with its colour, quilting and pockets; the face, hair, jeans and background are those of the first picture. **Seen as less good:** the result is 1024 x 1024 where the person was 768 x 1024 (the edit model's output size, not the picture's), so the framing changed; one run, one seed.

## 4. Not shown

The other five cases have not been run; whether they work as well as try-on is open (a room is a harder case than a jacket: the layout has to hold). One pair of pictures made by the same model is the easiest input; photographs of real people and rooms (shadows, odd poses, cluttered rooms) are not tested. The model does not always keep a pattern or a face exactly, which the page says.
