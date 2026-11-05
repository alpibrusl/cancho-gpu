#!/usr/bin/env python3
"""examples/weightspace/w1_patch.py (T11, docs/weight-space.md 14.12): the patch writer and its independent reader, on the toy
checkpoints of tests/qwen08/fixtures.py. No model, no GPU, numpy only. The loader's side (tests/qwen08_load.sh) is held against this
reader on the same bytes.

    python3 tests/w1_patch.py
"""
import contextlib
import io
import os
import random
import struct
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "examples", "weightspace"))
sys.path.insert(0, os.path.join(HERE, "qwen08"))
import fixtures  # noqa: E402
import w1_diff as D  # noqa: E402
import w1_patch as W  # noqa: E402

fails = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        fails.append(what)


def refused(rule, fn, *a):
    try:
        fn(*a)
    except W.PatchError as e:
        return e.rule == rule
    return False


def main():
    with tempfile.TemporaryDirectory() as d:
        fixtures.main(d)
        base, edited = D.read_checkpoint(f"{d}/ok"), D.read_checkpoint(f"{d}/edited")
        recs = W.make(base, edited)
        blob = W.encode(recs)
        check([r["name"].split(".")[-3] + "." + r["name"].split(".")[-2] for r in recs] == ["mlp.down_proj"] * 2, "make finds exactly the two tensors the toy edit changed")
        check(sum(len(r["cols_idx"]) for r in recs) == 5 and [r["cols_idx"] for r in recs] == [[0, 5, 6, 15], [7]], "columns found: 0, 5, 6, 15 and 7")
        dec = W.decode(blob)
        check(W.encode(dec) == blob and len(dec) == 2, "decode then encode gives the bytes back")
        rows, ok = W.check(base, blob, edited)
        check(ok and len(rows) == 57 and sum(1 for _, p, _ in rows if p) == 2, "base + patch equals edited in all 57 tensors by sha256")
        # The row left alone inside a replaced column is written with its own value, so the result is still exact.
        t = [r for r in dec if r["name"].endswith("layers.1.mlp.down_proj.weight")][0]
        before = base[t["name"]].unsigned()
        check((t["values"][:, 2] == before[2, t["cols_idx"]]).all(), "an entry the edit left alone inside a replaced column keeps its value in the patch")
        # A patch for another base, or a changed value, is found by check().
        check(refused("q08-patch-base", W.check, edited, blob, edited), "the patch on the edited checkpoint is refused (q08-patch-base)")
        bad = [dict(r) for r in recs]
        v = bad[0]["values"].copy()
        v[0, 0] ^= 1
        bad[0]["values"] = v
        rows2, ok2 = W.check(base, W.encode(bad), edited)
        check(not ok2 and [n for n, _, e in rows2 if not e] == [bad[0]["name"]], "a changed value is found, in that tensor only")
        # make refuses what a patch cannot say.
        check(refused("q08-patch-format", W.make, base, base), "identical checkpoints: nothing to patch")
        check(refused("q08-patch-unknown", W.make, base, D.read_checkpoint(f"{d}/reject/q08-missing")), "checkpoints with other tensors are refused")
        ed2 = dict(edited)
        a_log = [n for n in base if n.endswith("linear_attn.A_log")][0]
        raw = bytearray(bytes(edited[a_log].raw))
        raw[0] ^= 1
        ed2[a_log] = D.Tensor(a_log, "F32", edited[a_log].shape, memoryview(bytes(raw)), "x")
        check(refused("q08-patch-unknown", W.make, base, ed2), "a differing F32 tensor is refused: a patch holds BF16 matrices")
        # Files on disk, through the command line.
        out = f"{d}/t.patch"
        with contextlib.redirect_stdout(io.StringIO()):
            rc = W.main(["make", f"{d}/ok", f"{d}/edited", "-o", out, "--json", f"{d}/t.json"])
            rc2 = W.main(["inspect", out])
            rc3 = W.main(["check", f"{d}/ok", out, f"{d}/edited"])
        check((rc, rc2, rc3) == (0, 0, 0) and open(out, "rb").read() == blob, "make, inspect and check run, and write the same bytes")
        with contextlib.redirect_stderr(io.StringIO()):
            rc4 = W.main(["check", f"{d}/edited", out, f"{d}/edited"])
        check(rc4 == 1, "check of a patch on the wrong base exits 1")
        # The reader never raises anything but PatchError.
        rnd = random.Random(7)
        odd, n = 0, 0
        for _ in range(3000):
            b = bytearray(blob)
            for _ in range(1 + rnd.randrange(3)):
                b[rnd.randrange(len(b))] = rnd.randrange(256)
            cases = [bytes(b)] + [bytes(b[:-4]) + struct.pack("<I", zlib.crc32(bytes(b[:-4])))] + [blob[:rnd.randrange(len(blob))]]
            for c in cases:
                n += 1
                try:
                    W.decode(c)
                except W.PatchError:
                    pass
                except Exception as e:  # noqa: BLE001
                    odd += 1
                    print("decode raised", type(e).__name__, e)
        check(odd == 0, f"decode of {n} hostile patches raises only PatchError")
    print("FAILED" if fails else "all w1_patch tests pass")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
