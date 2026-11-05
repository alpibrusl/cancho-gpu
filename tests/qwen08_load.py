#!/usr/bin/env python3
"""Check C2 and the refusals of the Qwen3.5-0.8B weight loader (docs/weight-space.md
section 14, ticket T5), against `cancho-gpu-qwen08-load`.

    qwen08_load.py ok     <tool> <fixtures dir>       # toy checkpoint: every tensor, f32 and f16
    qwen08_load.py reject <tool> <fixtures dir>       # each refusal, with its rule tag
    qwen08_load.py fuzz   <tool> <fixtures dir> [n]   # hostile headers and configs never trap
    qwen08_load.py model  <tool> <model dir> f32|f16 [<patch> <edited dir>]  # the real checkpoint (numpy; skips when absent);
                                                    # with a patch: <model dir> + patch against the file in <edited dir>

The tool prints, for every tensor it put on the device and read back, `name dtype
values bytes adler32=<n> changed=<n>`. This script knows none of the loader's code:
it parses the safetensors header itself, widens bf16 by a shift (f32) or with
IEEE half conversion (f16: Python's `struct` `e`, numpy's `astype(float16)` on the
real model, both round to nearest even), and compares Adler-32 of the device bytes
(`zlib.adler32`) and the number of values the narrowing changed.
"""
import json
import os
import random
import re
import struct
import subprocess
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "examples", "weightspace"))
import w1_patch  # noqa: E402

PREFIX = "model.language_model."


def header_of(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


def weights_file(d):
    for name in ("model.safetensors", "model.safetensors-00001-of-00001.safetensors"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise SystemExit(f"no weights in {d}")


def text_tensors(d):
    """[(name, dtype, elems, absolute offset)] of the text tower, in the order the tool prints
    them: embedding, layers 0..n-1 (each layer's tensors in the loader's order), final norm.
    The order is read back from the tool's lines, so only the set is taken from the file."""
    h, data = header_of(weights_file(d))
    out = {}
    for name, e in h.items():
        if name.startswith(PREFIX):
            n = 1
            for x in e["shape"]:
                n *= x
            out[name] = (e["dtype"], n, data + e["data_offsets"][0])
    return out


def run_tool(tool, d, mode, patch=None):
    r = subprocess.run([tool, d, mode] + (["--patch", patch] if patch else []), capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


# ---- the oracle, pure Python (toy checkpoint) and numpy (the real one) ----

def py_bf16_to_f16(b):
    """(half bits, changed) for the bf16 pattern b, by IEEE half conversion."""
    sign = b >> 15
    f = struct.unpack("<f", struct.pack("<I", b << 16))[0]
    if f != f:
        h = 0x7E00 | sign << 15
        return h, (sign << 31 | 0x7FC00000) != b << 16
    if abs(f) >= 65520.0:
        h = 0x7C00 | sign << 15
        return h, f not in (float("inf"), float("-inf"))
    h = struct.unpack("<H", struct.pack("<e", f))[0]
    return h, struct.unpack("<e", struct.pack("<H", h))[0] != f


def py_expected(raw, dtype, mode):
    if dtype == "F32":
        return raw, 0, zlib.adler32(raw)
    vals = struct.unpack(f"<{len(raw) // 2}H", raw)
    if mode == "f32":
        out = b"".join(b"\0\0" + struct.pack("<H", v) for v in vals)
        return out, 0, zlib.adler32(out)
    hs, changed = [], 0
    for v in vals:
        h, c = py_bf16_to_f16(v)
        hs.append(h)
        changed += c
    out = struct.pack(f"<{len(hs)}H", *hs)
    return out, changed, zlib.adler32(out)


def parse_lines(out):
    rows, total = {}, None
    for line in out.splitlines():
        p = line.split()
        if p[0] == "total":
            total = dict(x.split("=") for x in p[1:])
        else:
            rows[p[0]] = (p[1], int(p[2]), int(p[3]), int(p[4].split("=")[1]), int(p[5].split("=")[1]))
    return rows, total


def check_ok(tool, fx, run="ok", oracle="ok", patch=None):
    """The tool on directory `run` (with `patch`) against the oracle computed from the file of directory `oracle`."""
    d = os.path.join(fx, oracle)
    want = text_tensors(d)
    h, data = header_of(weights_file(d))
    blob = open(weights_file(d), "rb").read()
    bad = 0
    for mode in ("f32", "f16"):
        rc, out, err = run_tool(tool, os.path.join(fx, run), mode, patch)
        if rc != 0:
            print(f"ok/{mode}: exit {rc}: {err[:200]}")
            return 1
        rows, total = parse_lines(out)
        if set(rows) != set(want):
            print(f"ok/{mode}: tensors differ: {sorted(set(rows) ^ set(want))[:4]}")
            bad += 1
        changed_all = 0
        for name, (dtype, elems, off) in want.items():
            raw = blob[off:off + elems * (4 if dtype == "F32" else 2)]
            exp, ch, ad = py_expected(raw, dtype, mode)
            got = rows.get(name)
            if got != (dtype, elems, len(exp), ad, ch):
                print(f"ok/{mode}: {name}: tool {got}, oracle {(dtype, elems, len(exp), ad, ch)}")
                bad += 1
            changed_all += ch
        if total is None or int(total["changed"]) != changed_all or int(total["tensors"]) != len(want):
            print(f"ok/{mode}: total line {total} against {len(want)} tensors, {changed_all} changed")
            bad += 1
        print(f"{run}{' + patch' if patch else ''}/{mode}: {len(want)} tensors equal the oracle of {oracle} (adler32 of the device bytes), {changed_all} values changed")
    # What is not loaded is not loaded.
    if any(not n.startswith(PREFIX) for n in rows):
        print("ok: a tensor outside the text tower was loaded")
        bad += 1
    return 1 if bad else 0


def check_reject(tool, fx):
    rules = set()
    for root, _, files in os.walk(os.path.join(ROOT, "src")):
        for f in files:
            if f.startswith("qwen08") and f.endswith(".cho"):
                rules |= set(re.findall(r'"(q08-[a-z0-9-]+)"', open(os.path.join(root, f)).read()))
    for root, _, files in os.walk(os.path.join(ROOT, "device")):
        for f in files:
            if f.startswith("qwen08") and f.endswith(".cho"):
                rules |= set(re.findall(r'"(q08-[a-z0-9-]+)"', open(os.path.join(root, f)).read()))
    # Refusals that need a failing device, a file changing under the reader or a patch.
    exempt = {"q08-device": "needs a device that fails (the mock does not)",
              "q08-verify": "fires only if the device differs from the file, which is what C2 exists to find",
              "q08-read": "a read that fails or runs short after the header was checked against the file's size",
              "q08-engine": "the engine's, not the loader's: held by tests/qwen08_engine.sh",
              "q08-range": "the engine's: held by tests/qwen08_engine.sh",
              "q08-tmax": "the engine's: held by tests/qwen08_engine.sh"}
    bad, seen, n = 0, set(), 0
    rd = os.path.join(fx, "reject")
    for name in sorted(os.listdir(rd)):
        rule = name
        for r in sorted(rules | {"st-truncated"}, key=len, reverse=True):
            if name == r or name.startswith(r + "-"):
                rule = r
                break
        n += 1
        seen.add(rule)
        r = subprocess.run([tool, os.path.join(rd, name), "f32"], capture_output=True, text=True)
        if r.returncode != 1 or not r.stderr.startswith(f"error[{rule}]") or r.stdout:
            print(f"{name}: exit {r.returncode}, expected error[{rule}], got {r.stderr[:120]!r} (stdout {len(r.stdout)} bytes)")
            bad += 1
    # The patch's refusals (T11): each fixture of patch_fixtures.py, and a decoder written apart (w1_patch.py) that must
    # give the same rule for the file-level ones.
    structural = {"q08-patch-truncated", "q08-patch-magic", "q08-patch-version", "q08-patch-checksum", "q08-patch-format",
                  "q08-patch-range", "q08-patch-duplicate", "q08-patch-order", "q08-patch-twice"}
    prd = os.path.join(fx, "patch", "reject")
    for name in sorted(os.listdir(prd)):
        rule = name[:-len(".patch")]
        for r in sorted(rules, key=len, reverse=True):
            if rule == r or rule.startswith(r + "-"):
                rule = r
                break
        n += 1
        seen.add(rule)
        path = os.path.join(prd, name)
        r = subprocess.run([tool, os.path.join(fx, "ok"), "f32", "--patch", path], capture_output=True, text=True)
        if r.returncode != 1 or not r.stderr.startswith(f"error[{rule}]") or r.stdout:
            print(f"{name}: exit {r.returncode}, expected error[{rule}], got {r.stderr[:120]!r} (stdout {len(r.stdout)} bytes)")
            bad += 1
        try:
            w1_patch.decode(open(path, "rb").read())
            got = None
        except w1_patch.PatchError as e:
            got = e.rule
        if (got or "") != (rule if rule in structural else ""):
            print(f"{name}: the Python decoder says {got}, the fixture is for {rule}")
            bad += 1
    for what, argv in (("a missing file", ["/no/such/file.patch"]), ("a directory", [fx])):
        n += 1
        seen.add("q08-patch-io")
        r = subprocess.run([tool, os.path.join(fx, "ok"), "f32", "--patch"] + argv, capture_output=True, text=True)
        if r.returncode != 1 or not r.stderr.startswith("error[q08-patch-io]") or r.stdout:
            print(f"--patch {what}: exit {r.returncode}: {r.stderr[:120]!r}")
            bad += 1
    # A patch is for one base: the same patch on the base it was not made from (the edited toy) is refused.
    r = subprocess.run([tool, os.path.join(fx, "edited"), "f32", "--patch", os.path.join(fx, "patch", "ok.patch")], capture_output=True, text=True)
    if r.returncode != 1 or not r.stderr.startswith("error[q08-patch-base]") or r.stdout:
        print(f"patch on the edited checkpoint: exit {r.returncode}: {r.stderr[:120]!r}")
        bad += 1
    # --patch without its file is a usage mistake.
    r = subprocess.run([tool, os.path.join(fx, "ok"), "f32", "--patch"], capture_output=True, text=True)
    if r.returncode != 2:
        print(f"--patch without a file: exit {r.returncode}, expected 2")
        bad += 1
    # Usage mistakes are exit 2.
    for argv in ([], ["x"], ["x", "f64"]):
        r = subprocess.run([tool] + argv, capture_output=True, text=True)
        if r.returncode != 2:
            print(f"usage {argv}: exit {r.returncode}, expected 2")
            bad += 1
    for r in sorted(rules - seen - set(exempt)):
        print(f"rule `{r}` has no fixture in tests/qwen08/fixtures.py")
        bad += 1
    print(f"{n} refusal fixtures (the patch's among them) and the usage errors checked; {len(exempt)} rules need a fault the tests cannot make: {', '.join(sorted(exempt))}")
    return 1 if bad else 0


def check_fuzz(tool, fx, count):
    src = os.path.join(fx, "ok")
    st = open(weights_file(src), "rb").read()
    cfg = open(os.path.join(src, "config.json"), "rb").read()
    hdr = 8 + struct.unpack("<Q", st[:8])[0]
    rnd = random.Random(20261105)
    picks = [0, 1, 34, 45, 91, 93, 123, 125, 127, 128, 255]
    cases = []
    for n in range(0, hdr + 40, 7):
        cases.append((st[:n], cfg))
    for n in range(0, len(cfg), 5):
        cases.append((st, cfg[:n]))
    for _ in range(count):
        b = bytearray(st)
        for _ in range(1 + rnd.randrange(4)):
            b[rnd.randrange(hdr)] = picks[rnd.randrange(len(picks))] if rnd.random() < 0.8 else rnd.randrange(256)
        cases.append((bytes(b), cfg))
    for _ in range(count):
        c = bytearray(cfg)
        for _ in range(1 + rnd.randrange(3)):
            c[rnd.randrange(len(c))] = picks[rnd.randrange(len(picks))] if rnd.random() < 0.8 else rnd.randrange(256)
        cases.append((st, bytes(c)))
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        for i, (s, c) in enumerate(cases):
            open(os.path.join(d, "model.safetensors"), "wb").write(s)
            open(os.path.join(d, "config.json"), "wb").write(c)
            try:
                r = subprocess.run([tool, d, "f16" if i % 2 else "f32"], capture_output=True, timeout=20)
            except subprocess.TimeoutExpired:
                print(f"case {i}: timed out")
                bad += 1
                continue
            e = r.stderr.decode(errors="replace")
            if r.returncode == 0:
                continue
            if r.returncode != 1 or not (e.startswith("error[q08-") or e.startswith("error[st-")):
                print(f"case {i}: exit {r.returncode}: {e[:120]!r}")
                bad += 1
    print(f"{len(cases)} hostile checkpoints, {bad} bad")
    return 1 if bad else 0


def check_patch(tool, fx):
    """C10 on the toy checkpoints: base + patch is the edited checkpoint, tensor by tensor, f32 and f16, and the
    patch changed what it should (and only that)."""
    bad = 0
    for run, oracle, patch in (("ok", "edited", "ok.patch"), ("big", "big-edited", "big.patch")):
        bad += check_ok(tool, fx, run, oracle, os.path.join(fx, "patch", patch))
        for mode in ("f32", "f16"):
            _, plain, _ = run_tool(tool, os.path.join(fx, run), mode)
            rc, patched, err = run_tool(tool, os.path.join(fx, run), mode, os.path.join(fx, "patch", patch))
            if rc != 0:
                print(f"{run}/{mode}: the tool refused the valid patch: {err[:200]}")
                bad += 1
                continue
            a, ta = parse_lines(plain)
            b, tb = parse_lines(patched)
            moved = sorted(k for k in a if a[k] != b.get(k))
            want = [k for k in a if k.endswith("mlp.down_proj.weight") and ("layers.1." in k or "layers.3." in k)] if run == "ok" else [PREFIX + "embed_tokens.weight"]
            if moved != sorted(want):
                print(f"{run}/{mode}: the patch moved {moved}, it should have moved {sorted(want)}")
                bad += 1
            if tb.get("patched") != str(len(want)) or ta.get("patched") != "0":
                print(f"{run}/{mode}: total lines say patched={tb.get('patched')} / {ta.get('patched')}")
                bad += 1
    print("patch: only the tensors it names move" if not bad else "patch: FAILED")
    return 1 if bad else 0


def check_patch_fuzz(tool, fx, count):
    """Hostile patches never trap: every prefix of the toy patch, byte mutations with the CRC-32 left wrong and
    with it recomputed (so the layout, index and hash checks run), and the loader agrees with the Python decoder."""
    good = open(os.path.join(fx, "patch", "ok.patch"), "rb").read()
    rnd = random.Random(20261105)
    cases = [good[:n] for n in range(0, len(good), 3)] + [good[:n] + good[n + 1:] for n in range(16, len(good) - 4, 11)]
    picks = [0, 1, 2, 255, 128, 127, 4]
    for _ in range(count):
        b = bytearray(good)
        for _ in range(1 + rnd.randrange(3)):
            b[rnd.randrange(len(b))] = picks[rnd.randrange(len(picks))] if rnd.random() < 0.6 else rnd.randrange(256)
        cases.append(bytes(b))
        body = bytearray(b[:-4])
        cases.append(bytes(body) + struct.pack("<I", zlib.crc32(bytes(body))))
        # a field set to a hostile count, the CRC-32 valid
        for at in range(16, len(body) - 3, 4):
            if rnd.random() < 0.08:
                c = bytearray(good[:-4])
                struct.pack_into("<I", c, at, [0, 1, 0xFFFFFFFF, 0x7FFFFFFF, 1 << 24, (1 << 24) + 1, 65536][rnd.randrange(7)])
                cases.append(bytes(c) + struct.pack("<I", zlib.crc32(bytes(c))))
    bad, ok, refused = 0, 0, 0
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.patch")
        for i, blob in enumerate(cases):
            open(path, "wb").write(blob)
            try:
                r = subprocess.run([tool, os.path.join(fx, "ok"), "f32" if i % 2 else "f16", "--patch", path], capture_output=True, timeout=20)
            except subprocess.TimeoutExpired:
                print(f"case {i}: timed out")
                bad += 1
                continue
            e = r.stderr.decode(errors="replace")
            try:
                w1_patch.decode(blob)
                py = None
            except w1_patch.PatchError as x:
                py = x.rule
            if r.returncode == 0:
                ok += 1
                if py:
                    print(f"case {i}: loaded, but the Python decoder refuses it ({py})")
                    bad += 1
                continue
            refused += 1
            if r.returncode != 1 or not e.startswith("error[q08-patch-"):
                print(f"case {i}: exit {r.returncode}: {e[:120]!r}")
                bad += 1
            elif py and not e.startswith(f"error[{py}]"):
                print(f"case {i}: the loader says {e[:60]!r}, the Python decoder {py}")
                bad += 1
    print(f"{len(cases)} hostile patches ({ok} still valid, {refused} refused), {bad} bad")
    return 1 if bad else 0


def check_model(tool, d, mode, patch=None, oracle=None):
    if not os.path.isdir(d) or not any(f.startswith("model.safetensors") for f in os.listdir(d)):
        print(f"skip: no checkpoint at {d}")
        return 0
    import numpy as np
    od = oracle or d
    want = text_tensors(od)
    path = weights_file(od)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    rc, out, err = run_tool(tool, d, mode, patch)
    if rc != 0:
        print(f"{d} {mode}: exit {rc}: {err[:300]}")
        return 1
    rows, total = parse_lines(out)
    bad = 0
    if set(rows) != set(want):
        print(f"tensors differ: {sorted(set(rows) ^ set(want))[:4]}")
        bad += 1
    changed_all, exact_all, step = 0, 0, 1 << 24
    sub_nonzero = flush = bf16_total = 0
    for name, (dtype, elems, off) in want.items():
        fw = 4 if dtype == "F32" else 2
        ad, ch, nbytes = 1, 0, 0
        for a in range(0, elems, step):
            n = min(step, elems - a)
            raw = np.asarray(mm[off + fw * a: off + fw * (a + n)])
            if dtype == "F32":
                dev = raw
            else:
                u32 = raw.view("<u2").astype(np.uint32) << np.uint32(16)
                if mode == "f32":
                    dev = u32.astype("<u4").view(np.uint8)
                else:
                    f = u32.view(np.float32)
                    h = f.astype(np.float16)
                    assert not np.isnan(f).any(), "a NaN weight: the oracle's count is not defined"
                    ch += int(np.count_nonzero(h.astype(np.float32) != f))
                    dev = h.view(np.uint8)
                    af = np.abs(f)
                    bf16_total += n
                    sub_nonzero += int(np.count_nonzero((af > 0) & (af < 2.0 ** -14)))
                    flush += int(np.count_nonzero((af > 0) & (h == 0)))
            ad = zlib.adler32(dev.tobytes(), ad)
            nbytes += dev.size
        got = rows.get(name)
        exp = (dtype, elems, nbytes, ad, ch)
        if got != exp:
            print(f"{name}: tool {got}, oracle {exp}")
            bad += 1
        changed_all += ch
        exact_all += 1
    if total is None or int(total["changed"]) != changed_all:
        print(f"total line {total} against {changed_all} changed")
        bad += 1
    if patch and (total is None or int(total.get("patched", -1)) < 1):
        print(f"total line {total}: no tensor was patched")
        bad += 1
    msg = f"{os.path.basename(d.rstrip('/'))}{' + ' + os.path.basename(patch) + ' against ' + os.path.basename(od.rstrip('/')) if patch else ''} {mode}: {exact_all} tensors, device bytes equal the oracle's (adler32), {changed_all} values differ from the bf16 originals"
    if mode == "f16":
        msg += f"; of {bf16_total} bf16 weights {sub_nonzero} are nonzero below 2^-14, {flush} narrow to zero"
    print(msg)
    return 1 if bad else 0


def main():
    mode, tool = sys.argv[1], sys.argv[2]
    if mode == "ok":
        return check_ok(tool, sys.argv[3])
    if mode == "reject":
        return check_reject(tool, sys.argv[3])
    if mode == "fuzz":
        return check_fuzz(tool, sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 400)
    if mode == "model":
        return check_model(tool, sys.argv[3], sys.argv[4], *(sys.argv[5:7] if len(sys.argv) > 6 else ()))
    if mode == "patch":
        return check_patch(tool, sys.argv[3])
    if mode == "patch-fuzz":
        return check_patch_fuzz(tool, sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 200)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
