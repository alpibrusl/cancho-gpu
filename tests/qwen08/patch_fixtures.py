#!/usr/bin/env python3
"""Patch fixtures for `tests/qwen08_load.sh` (T11, docs/weight-space.md 14.13; nothing is committed).

    python3 tests/qwen08/patch_fixtures.py <dir written by fixtures.py>

Writes `<dir>/patch/ok.patch` (the toy checkpoint `ok` to `edited`), `<dir>/patch/big.patch` (`big` to
`big-edited`) and `<dir>/patch/reject/<rule>[-variant].patch`, one file a refusal of `src/qwen08_patch.cho`
or `qwen08_load.open_patch`, named for the rule it must be refused with. The malformed ones are written
with `w1_patch.encode` (so a CRC-32 is valid where the fixture means to get past it) from the valid
records, one thing changed.
"""
import copy
import os
import struct
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "examples", "weightspace"))
import w1_diff as D  # noqa: E402
import w1_patch as W  # noqa: E402

P = "model.language_model."


def reseal(body):
    """`body` (everything but the trailer) with its CRC-32 appended."""
    return body + struct.pack("<I", zlib.crc32(body))


def main(fx):
    out = os.path.join(fx, "patch")
    rej = os.path.join(out, "reject")
    os.makedirs(rej, exist_ok=True)
    recs = W.make(D.read_checkpoint(os.path.join(fx, "ok")), D.read_checkpoint(os.path.join(fx, "edited")))
    good = W.encode(recs)
    with open(os.path.join(out, "ok.patch"), "wb") as f:
        f.write(good)
    brecs = W.make(D.read_checkpoint(os.path.join(fx, "big")), D.read_checkpoint(os.path.join(fx, "big-edited")))
    with open(os.path.join(out, "big.patch"), "wb") as f:
        f.write(W.encode(brecs))

    def put(name, data):
        with open(os.path.join(rej, name + ".patch"), "wb") as f:
            f.write(data)

    def variant(index=0, **kw):
        rs = copy.deepcopy(recs)
        r = rs[index]
        for k, v in kw.items():
            r[k] = v
        return rs

    # The file as a file.
    put("q08-patch-truncated", good[:10])
    put("q08-patch-truncated-cut", reseal(good[:-4][:-9]))
    put("q08-patch-truncated-names", reseal(good[:16 + 3]))
    put("q08-patch-magic", b"CHOPATCX" + good[8:])
    put("q08-patch-version", good[:8] + struct.pack("<I", 2) + good[12:])
    put("q08-patch-checksum", good[:-6] + bytes([good[-6] ^ 1]) + good[-5:])
    put("q08-patch-checksum-trailer", good[:-1] + bytes([good[-1] ^ 0x80]))
    put("q08-patch-format-zero-tensors", reseal(good[:12] + struct.pack("<I", 0)))
    put("q08-patch-format-many-tensors", reseal(good[:12] + struct.pack("<I", 65) + good[16:-4]))
    put("q08-patch-format-trailing", reseal(good[:-4] + b"\0\0\0\0"))
    # Inside a tensor (offsets from the first tensor's start at 16: name length, name padded, rows, cols, ncols, 4 hashes).
    nl = len(recs[0]["name"].encode())
    pad = -nl % 4
    q = 16 + 4 + nl + pad

    def body_with(at, fmt, v):
        b = bytearray(good[:-4])
        struct.pack_into(fmt, b, at, v)
        return reseal(bytes(b))

    put("q08-patch-format-name-zero", body_with(16, "<I", 0))
    put("q08-patch-format-name-long", body_with(16, "<I", 256))
    put("q08-patch-format-rows-zero", body_with(q, "<I", 0))
    put("q08-patch-format-cols-huge", body_with(q + 4, "<I", 1 << 25))
    put("q08-patch-format-ncols-zero", body_with(q + 8, "<I", 0))
    if pad:
        b = bytearray(good[:-4])
        b[16 + 4 + nl] = 1
        put("q08-patch-format-name-padding", reseal(bytes(b)))
    put("q08-patch-range-ncols", body_with(q + 8, "<I", recs[0]["cols"] + 1))
    put("q08-patch-range-index", W.encode(variant(cols_idx=[0, 5, 6, recs[0]["cols"]])))
    put("q08-patch-range-index-huge", W.encode(variant(cols_idx=[0, 5, 6, 4294967295])))
    put("q08-patch-duplicate", W.encode(variant(cols_idx=[0, 5, 5, 15])))
    put("q08-patch-order", W.encode(variant(cols_idx=[0, 6, 5, 15])))
    put("q08-patch-twice", W.encode([recs[0], recs[0]]))
    # Against the checkpoint.
    put("q08-patch-unknown-name", W.encode(variant(name=P + "layers.1.mlp.no_proj.weight")))
    put("q08-patch-unknown-visual", W.encode(variant(name="model.visual.patch_embed.weight")))
    put("q08-patch-unknown-f32", W.encode(variant(name=P + "layers.0.linear_attn.A_log")))
    import numpy as np
    put("q08-patch-shape", W.encode(variant(rows=recs[0]["cols"], cols=recs[0]["rows"], cols_idx=[0, 1, 2, 3], values=np.zeros((4, recs[0]["cols"]), dtype="<u2"))))
    put("q08-patch-base", W.encode(variant(base_hash=(recs[0]["base_hash"][0] ^ 1, recs[0]["base_hash"][1]))))
    put("q08-patch-base-adler", W.encode(variant(base_hash=(recs[0]["base_hash"][0], recs[0]["base_hash"][1] ^ 1))))
    put("q08-patch-base-second", W.encode([recs[0]] + [dict(recs[1], base_hash=(recs[1]["base_hash"][0], recs[1]["base_hash"][1] ^ 4))]))
    put("q08-patch-result", W.encode(variant(edited_hash=(recs[0]["edited_hash"][0] ^ 1, recs[0]["edited_hash"][1]))))
    wrong = recs[0]["values"].copy()
    wrong[0, 0] ^= 1
    put("q08-patch-result-value", W.encode(variant(values=wrong)))


if __name__ == "__main__":
    main(sys.argv[1])
