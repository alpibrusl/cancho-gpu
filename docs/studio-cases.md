# Studio cases: ready-made two-picture edits (2026-11-18)

> **Status: built (`/cases`), two cases run for real (try-on; an object in a room, with and without real sizes); the other four are written, not run.** Claims are **[measured]** unless tagged.

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

## 4. Real sizes and "keep it as it is" (2026-11-18)

Each case has an optional **Real sizes** box (up to 200 bytes; the instruction and the sizes together at most 1000), added to the instruction as `Real sizes: ...` (`note` in `/edit`, `studio/upload.cho`), and the try-on and object instructions now say not to resize or reshape the garment or object. **A hint, not a measurement:** the model has no metres; it matches scale by eye.

**Measured once** (a room 1024 x 768 and a two-seat sofa, both made by `media image`; the object case; seed 1; the sofa in the first run, same instruction): without sizes the sofa spans about 585 px of the 1024; with `Real sizes: the sofa is 1.4 m wide and 0.8 m deep, the door is 2 m high.` about 550 px. By the door (about 650 px for 2 m) 1.4 m would be about 455 px, so the hint moved the sofa **the right way and about a third of the way to what was asked** (about 1.7 m instead of 1.8 m, against 1.4). The sofa kept its shape, colour and proportions in both. **One pair, one seed**: no more than a sign that the words are read. **Seen as less good, in both:** the edit's output was 1024 x 1024 for a 1024 x 768 room, so the room is taller than the original's proportions.

**Not possible here:** a true fit (a garment against body measurements, a sofa against a wall's length) needs geometry the model does not have: a depth or pose estimate, or a drawing at scale, which is a project of its own.

## 5. The pictures you gave, beside the result (2026-11-19)

**Asked for:** "in the results of jobs I can see the three pictures". A two-picture edit's page (`/jobs/N`) showed only the result; the two pictures it was made from were on disk (`upload-N.*`, `upload-N-2.*`, kept by `POST /edit`) but nowhere on the page, so a result could not be judged against what went in.

**What it does now:** an edit made through the form shows a row **Picture 1, Picture 2, Result** (the result joins the row when the job is done; while it waits or runs, the two inputs show alone; a failed job shows its inputs and the alert). Each is a link to the full file. A job with one picture shows two figures; a job made by `POST /api/jobs` has no upload and its page is as before (its `image` is a path on the machine, and a path from a request is never served). The files come from `GET /files/N/NAME` with six new names in the closed list, `input1.jpg|png|webp` and `input2.jpg|png|webp`, each served from the upload of that name (`input2.png` is `upload-N-2.png`) with its own type; no other name reaches a path, so `input3.png`, `input1` and `..%2F...` are 404. `studio/openapi.json` lists them.

**Checked:** `studio/check.sh` has 10 new checks (434 ok; the run on unmodified `main` has 424 ok): both pictures arrive byte for byte with `image/jpeg` and `image/png`, a type the upload did not have (`input1.png`, `input2.webp`), a name off the list and a job with no upload are 404, the page of a waiting job lists the two inputs, the page of a finished job lists input1, input2 and the result in that order with the captions `Picture 1`, `Picture 2`, `Result` and no second copy of the result, a job made by the API shows its result alone, and the OpenAPI document lists the six names. **One check fails the same way with and without this change** (`image: a size at the limit is accepted`, whitespace in its expected text, `' 202 202 202'` against `' 202  202  202'`); it is not touched here.

**Not done:** the pictures are not on the list page (`/jobs`: one thumbnail per job, the result) or on the comparison pages; they stay on the studio's disk as long as the data directory does (nothing deletes `upload-N.*`), which a person with private pictures should know.

## 6. Not shown

The other four cases have not been run; whether they work as well as try-on is open (a room is a harder case than a jacket: the layout has to hold). One pair of pictures made by the same model is the easiest input; photographs of real people and rooms (shadows, odd poses, cluttered rooms) are not tested. The model does not always keep a pattern or a face exactly, which the page says.

## A tip when the standard instruction may do nothing (2026-11-19)

**What was seen.** Two pictures of ordinary clothing, a woman in a pink long-sleeved top and floral pyjama trousers (image1) and a woman in a white t-shirt and black jeans (image2), made the standard try-on instruction return **image1 unchanged** (re-rendered a little sharper, the pyjamas still on): once, and again with another seed. Naming the garments, "replace the pink long-sleeved top with the white short-sleeved t-shirt worn in image2, and replace the floral pyjama trousers with the black skinny jeans with zip pockets worn in image2, keep the face, hair, skin, pose, bare feet and background", gave the wanted picture on the first try: the white t-shirt and the black zip-pocket jeans, the same face, hair, pose and bare feet. Other pairs worked with the standard text (a quilted jacket on a white t-shirt, a tracksuit on a white top), so the naming is not always needed; a matching set was the case where it was. **One pair of pictures, one run of each: a measured observation, not a rule.**

**The room has the same shape.** A real photograph of a furnished living room (a blue sofa, a white chair, plants, a TV unit) redecorated with another living room as the style, with the standard room instruction, kept the blue sofa and added the new pieces (a cream sofa, an oval table, a rug, a chandelier): two sofas. Not repeated with named furniture, so the room tip is advice from the try-on result, not a measurement.

**What the page does.** The try-on and room cards show a tip above the instruction: for a person who already wears a matching set, say which garments to swap, with an example; for a furnished room, say what to replace. The tip is outside the textarea, so the instruction posted is the one in the box. The other four cards have none (not measured). A form that builds the sentence from two fields ("what she wears now", "what to put on") would need `/edit` to compose the instruction on the server; it is not done.

**Checked:** `studio/check.sh` (4 new checks): the tip is on exactly the try-on and room cards, above the instruction; it says to name the garments, or what to replace; it is not inside any textarea.
