#!/usr/bin/env python3
"""docs/weight-space.md section 13 (W1): the weight diff, the grading and the verdict of examples/weightspace/w1*.py, on tiny
synthetic checkpoints (no model, no download, no GPU).

    python3 tests/w1.py

  1. A hand-written safetensors file (bfloat16, float32, int32) is read back; a planted rank-1 edit on 4 columns of one tensor is
     reported as that tensor alone, with 4 changed columns, rank 1 at tolerance 0.1, and the entries, rows and norms the test counts
     independently; identical checkpoints report nothing; a shape change, an added and a removed tensor, a NaN, and a sharded
     directory with the Hugging Face file name are reported, not raised.
  2. Malformed files (truncated, a header longer than the file, not JSON, offsets outside the file, a wrong byte count, an unknown
     dtype, overlapping tensors, an empty file, a missing path, an empty directory) raise ReadError and nothing else; 400 random
     single-byte corruptions of a valid file raise nothing but ReadError; the command line exits 2 with no traceback.
  3. The published-edit expectations pass on a checkpoint built to match them and fail on each planted deviation.
  4. Grading (nested braces, commas, fractions, no answer, a cut-off \\boxed{), the bootstrap (reproducible, contains the mean), the
     rule (reproduces / partial / not reproduced, and the 20-point and half-width boundaries are strict), the run loop with a fake
     generator (resume, a log line cut off by a crash), the problem file reader and the `--gpu-free` refusal.
  5. What scripts/gcp/w1_cloud.sh relies on: `behave` in growing chunks (--n 4, then 8, then 10) over a directory that holds a partial
     run (a batch missing, a line cut off) finishes it with the same seeds as one uninterrupted run and re-generates nothing it has;
     `status` and `verdict` read the jsonl alone, agree with the in-process rule, refuse an incomplete directory, and say whether
     the cap check is asked."""
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, os.path.join(ROOT, "examples", "weightspace"))
try:
    import numpy as np
except ImportError:
    print("skip: numpy is not installed")
    sys.exit(0)
import w1  # noqa: E402
import w1_behave as B  # noqa: E402
import w1_diff as D  # noqa: E402

fails = 0


def expect(name, ok, detail=""):
    global fails
    print("%s %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "  " + str(detail)))
    fails += 0 if ok else 1


def bf16(a):
    """float32 array -> uint16 bit patterns, round to nearest even."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_f(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


def write_st(path, tensors, raw_header=None):
    """tensors: {name: (dtype, ndarray of that dtype's storage)}; bf16 storage is uint16."""
    header, blobs, off = {}, [], 0
    for name, (dt, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = raw_header if raw_header is not None else json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def raises_read_error(f):
    try:
        f()
    except D.ReadError:
        return True
    except BaseException as e:  # noqa: BLE001
        return "raised " + type(e).__name__ + ": " + str(e)
    return "raised nothing"


tmp = tempfile.mkdtemp(prefix="w1_test_")
rng = np.random.default_rng(7)
DOWN = "model.language_model.layers.3.mlp.down_proj.weight"
OTHER = "model.language_model.layers.3.mlp.up_proj.weight"


def make_base():
    return {DOWN: ("BF16", bf16(rng.standard_normal((16, 40)))),
            OTHER: ("BF16", bf16(rng.standard_normal((40, 16)))),
            "norm.weight": ("F32", rng.standard_normal(16).astype(np.float32)),
            "ids": ("I32", np.arange(10, dtype=np.int32))}


base = make_base()
COLS = [3, 11, 12, 30]
u, v = rng.standard_normal(16), rng.standard_normal(4)
edited = dict(base)
w = bf16_f(base[DOWN][1]).copy()
w[:, COLS] += 5.0 * np.outer(u, v).astype(np.float32)
edited[DOWN] = ("BF16", bf16(w))
write_st(f"{tmp}/base.safetensors", base)
write_st(f"{tmp}/edited.safetensors", edited)

# ---- 1
rep = D.diff_checkpoints(D.read_checkpoint(f"{tmp}/base.safetensors"), D.read_checkpoint(f"{tmp}/edited.safetensors"))
mask = base[DOWN][1] != edited[DOWN][1]
expect("1a only the edited tensor is reported", [c["name"] for c in rep["changed"]] == [DOWN] and rep["unchanged"] == 3, rep["changed"])
c = rep["changed"][0]
expect("1b entries, rows and columns equal the test's own count", c["changed_entries"] == int(mask.sum()) and c["changed_columns"] == 4
       and c["columns"] == COLS and c["changed_rows"] == int(mask.any(1).sum()), c)
d = bf16_f(edited[DOWN][1]).astype(np.float64) - bf16_f(base[DOWN][1]).astype(np.float64)
expect("1c delta norm and relative change equal numpy's", abs(c["delta_norm"] - np.linalg.norm(d)) < 1e-9 and
       abs(c["relative_change"] - np.linalg.norm(d) / np.linalg.norm(bf16_f(base[DOWN][1]).astype(np.float64))) < 1e-9, c)
expect("1d the rank-1 plant is rank 1 at 0.1 and carries the energy", c["rank_at"]["0.1"] == 1 and c["top1_energy"] > 0.99
       and c["energy_in_changed_columns"] == 1.0 and c["support_shape"][1] == 4, c)
expect("1e sha256 is of the raw bytes", c["edited_sha256"] == hashlib.sha256(edited[DOWN][1].tobytes()).hexdigest()
       and c["base_sha256"] == hashlib.sha256(base[DOWN][1].tobytes()).hexdigest())
expect("1f the text rendering runs and names the tensor", DOWN in D.render(rep) and "rank" in D.render(rep))
# a rank-2 plant is not called rank 1
w2 = bf16_f(base[DOWN][1]).copy()
w2[:, COLS] += 5.0 * np.outer(u, v).astype(np.float32) + 5.0 * np.outer(rng.standard_normal(16), rng.standard_normal(4)).astype(np.float32)
write_st(f"{tmp}/e2.safetensors", {**base, DOWN: ("BF16", bf16(w2))})
r2 = D.diff_checkpoints(D.read_checkpoint(f"{tmp}/base.safetensors"), D.read_checkpoint(f"{tmp}/e2.safetensors"))["changed"][0]
expect("1g a rank-2 plant reports rank 2 (the rank test can fail)", r2["rank_at"]["0.1"] == 2, r2["rank_at"])
same = D.diff_checkpoints(D.read_checkpoint(f"{tmp}/base.safetensors"), D.read_checkpoint(f"{tmp}/base.safetensors"))
expect("1h identical checkpoints: nothing changed", same["changed"] == [] and same["changed_weights"] == 0 and same["unchanged"] == 4)
odd = {**{k: x for k, x in base.items() if k != "ids"}, DOWN: ("BF16", bf16(rng.standard_normal((16, 41)))), "extra": ("F32", np.zeros(3, np.float32))}
write_st(f"{tmp}/odd.safetensors", odd)
ro = D.diff_checkpoints(D.read_checkpoint(f"{tmp}/base.safetensors"), D.read_checkpoint(f"{tmp}/odd.safetensors"))
expect("1i a reshaped, an added and a removed tensor are reported", [m["name"] for m in ro["mismatched"]] == [DOWN]
       and ro["only_in_edited"] == ["extra"] and ro["only_in_base"] == ["ids"], ro)
nanw = bf16_f(base[DOWN][1]).copy()
nanw[0, 0] = np.nan
write_st(f"{tmp}/nan.safetensors", {**base, DOWN: ("BF16", bf16(nanw))})
rn = D.diff_checkpoints(D.read_checkpoint(f"{tmp}/base.safetensors"), D.read_checkpoint(f"{tmp}/nan.safetensors"))["changed"][0]
expect("1j a NaN is reported, not computed on", "numeric" in rn and "delta_norm" not in rn, rn)
os.makedirs(f"{tmp}/shards_b"), os.makedirs(f"{tmp}/shards_e")
write_st(f"{tmp}/shards_b/model.safetensors-00001-of-00002.safetensors", {DOWN: base[DOWN]})
write_st(f"{tmp}/shards_b/model.safetensors-00002-of-00002.safetensors", {k: x for k, x in base.items() if k != DOWN})
write_st(f"{tmp}/shards_e/model.safetensors-00001-of-00001.safetensors", edited)
rs = D.diff_checkpoints(D.read_checkpoint(f"{tmp}/shards_b"), D.read_checkpoint(f"{tmp}/shards_e"))
expect("1k two shards against one file, Hugging Face names", [c["name"] for c in rs["changed"]] == [DOWN], rs["changed"])
write_st(f"{tmp}/dup.safetensors", {OTHER: base[OTHER]})
os.makedirs(f"{tmp}/dupdir")
write_st(f"{tmp}/dupdir/a.safetensors", {OTHER: base[OTHER]})
write_st(f"{tmp}/dupdir/b.safetensors", {OTHER: base[OTHER]})
expect("1l a tensor in two shards is refused", raises_read_error(lambda: D.read_checkpoint(f"{tmp}/dupdir")) is True)
for dt, arr in (("I32", np.array([1, 2, 3], np.int32)), ("F16", np.array([1, 2, 3], np.float16)), ("F64", np.array([1, 2, 3], np.float64))):
    write_st(f"{tmp}/t_{dt}.safetensors", {"x": (dt, arr)})
    write_st(f"{tmp}/t2_{dt}.safetensors", {"x": (dt, arr + arr.dtype.type(1))})
    cr = D.diff_checkpoints(D.read_checkpoint(f"{tmp}/t_{dt}.safetensors"), D.read_checkpoint(f"{tmp}/t2_{dt}.safetensors"))["changed"][0]
    expect(f"1m {dt}: delta norm of +1 on 3 values is sqrt(3)", abs(cr["delta_norm"] - 3 ** 0.5) < 1e-9, cr)

# ---- 2
good = open(f"{tmp}/base.safetensors", "rb").read()


def bad_file(name, data):
    p = f"{tmp}/{name}.safetensors"
    open(p, "wb").write(data)
    return p


def hdr(h):
    b = json.dumps(h).encode()
    return struct.pack("<Q", len(b)) + b


cases = {
    "truncated": good[:len(good) - 50],
    "header longer than file": struct.pack("<Q", 10 ** 9) + b"{}",
    "header absurd": struct.pack("<Q", 2 ** 63) + b"{}",
    "not json": struct.pack("<Q", 5) + b"[[[[[",
    "not utf-8": struct.pack("<Q", 2) + b"\xff\xfe",
    "header is a list": hdr([1, 2]),
    "offsets outside": hdr({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}) + b"\0" * 4,
    "wrong byte count": hdr({"x": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}) + b"\0" * 8,
    "unknown dtype": hdr({"x": {"dtype": "F7", "shape": [1], "data_offsets": [0, 1]}}) + b"\0",
    "dtype is a list": hdr({"x": {"dtype": ["F32"], "shape": [1], "data_offsets": [0, 4]}}) + b"\0" * 4,
    "negative shape": hdr({"x": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}}) + b"\0" * 4,
    "offsets are strings": hdr({"x": {"dtype": "F32", "shape": [1], "data_offsets": ["0", "4"]}}) + b"\0" * 4,
    "missing field": hdr({"x": {"dtype": "F32"}}),
    "entry not an object": hdr({"x": 3}),
    "overlap": hdr({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}, "y": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}}) + b"\0" * 12,
    "empty file": b"",
    "seven bytes": b"1234567",
}
for name, data in cases.items():
    r = raises_read_error(lambda p=bad_file(name.replace(" ", "_"), data): D.read_file(p))
    expect(f"2a malformed ({name}) raises ReadError", r is True, r)
expect("2b a missing path", raises_read_error(lambda: D.read_checkpoint(f"{tmp}/nowhere")) is True)
os.makedirs(f"{tmp}/emptydir")
expect("2c a directory with no checkpoint", raises_read_error(lambda: D.read_checkpoint(f"{tmp}/emptydir")) is True)
okz = D.read_file(bad_file("zero_tensors", hdr({"__metadata__": {"format": "pt"}})))
expect("2d a file with only metadata is a valid empty checkpoint", okz == {})
fz = D.read_file(bad_file("zero_size", hdr({"x": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}})))
expect("2e a zero-size tensor reads", fz["x"].size == 0 and fz["x"].as_float64().size == 0)
bad, kinds = 0, set()
fr = np.random.default_rng(11)
for i in range(400):
    b = bytearray(good)
    for _ in range(1 + i % 3):
        b[int(fr.integers(0, 8 + 400 if i % 2 else len(b)))] = int(fr.integers(0, 256))
    p = bad_file("fuzz", bytes(b))
    try:
        t = D.read_file(p)
        D.diff_checkpoints(t, D.read_checkpoint(f"{tmp}/edited.safetensors"))
    except D.ReadError:
        kinds.add("ReadError")
    except BaseException as e:  # noqa: BLE001
        bad += 1
        kinds.add(type(e).__name__)
expect("2f 400 random corruptions raise nothing but ReadError", bad == 0, kinds)
script = os.path.join(ROOT, "examples", "weightspace", "w1.py")


def run(*a):
    return subprocess.run([sys.executable, script, *a], capture_output=True, text=True)


r = run("diff", f"{tmp}/truncated.safetensors", f"{tmp}/edited.safetensors")
expect("2g the command line: a malformed file exits 2, no traceback", r.returncode == 2 and "Traceback" not in r.stderr and "error:" in r.stderr, (r.returncode, r.stderr))
r = run("diff", f"{tmp}/nowhere", f"{tmp}/edited.safetensors")
expect("2h the command line: a missing path exits 2", r.returncode == 2 and "Traceback" not in r.stderr, (r.returncode, r.stderr))
r = run("diff", f"{tmp}/base.safetensors", f"{tmp}/edited.safetensors", "--json", f"{tmp}/rep.json")
expect("2i the command line: a diff exits 0, prints the tensor and writes the json", r.returncode == 0 and DOWN in r.stdout
       and json.load(open(f"{tmp}/rep.json"))["changed"][0]["name"] == DOWN, (r.returncode, r.stderr))
r = run("diff", f"{tmp}/base.safetensors", f"{tmp}/edited.safetensors", "--w1-expect")
expect("2j --w1-expect on a checkpoint that is not the published edit exits 1 and says FAIL", r.returncode == 1 and "FAIL" in r.stdout, (r.returncode, r.stdout[-300:]))

# ---- 3
pub_c = [c for c in rep["changed"]][0]
saved = dict(w1.EXPECT)
w1.EXPECT.clear()
w1.EXPECT[DOWN] = {k: pub_c[k] for k in ("changed_entries", "changed_columns", "changed_rows", "shape", "base_sha256", "edited_sha256")}
expect("3a a checkpoint built to match the expectations passes", all(ok for _, ok, _ in w1.check_expect(rep)), w1.check_expect(rep))
for key, val in (("changed_entries", pub_c["changed_entries"] + 1), ("changed_columns", 5), ("edited_sha256", "0" * 64), ("base_sha256", "0" * 64)):
    w1.EXPECT[DOWN] = {**w1.EXPECT[DOWN], key: val}
    res = w1.check_expect(rep)
    expect(f"3b a wrong published {key} fails (and only that)", [w for w, ok, _ in res if not ok] == [f"{DOWN} {key}"], res)
    w1.EXPECT[DOWN] = {k: pub_c[k] for k in ("changed_entries", "changed_columns", "changed_rows", "shape", "base_sha256", "edited_sha256")}
write_st(f"{tmp}/e3.safetensors", {**edited, OTHER: ("BF16", bf16(rng.standard_normal((40, 16))))})
res = w1.check_expect(D.diff_checkpoints(D.read_checkpoint(f"{tmp}/base.safetensors"), D.read_checkpoint(f"{tmp}/e3.safetensors")))
expect("3c a second changed tensor fails the 'exactly the published' check", not res[0][1], res[0])
w1.EXPECT.clear()
w1.EXPECT.update(saved)
res = w1.check_expect(same)
expect("3d the real expectations on identical checkpoints: fails, does not raise", not all(ok for _, ok, _ in res) and len(res) > 2)
expect("3e the real expectations name two tensors with 64 columns", len(saved) == 2 and all(e["changed_columns"] == 64 for e in saved.values()))

# ---- 4
expect("4a boxed", B.grade("so \\boxed{18}.", "18") == ("18", True))
expect("4b the last boxed wins", B.grade("\\boxed{5} then \\boxed{7}", "7")[1] and not B.grade("\\boxed{5} then \\boxed{7}", "5")[1])
expect("4c nested braces and a comma", B.last_boxed("\\boxed{\\frac{1}{2}} and \\boxed{1,234}") == "1,234" and B.grade("\\boxed{\\frac{3}{4}}", "0.75")[1])
expect("4d dollars, units and percent", B.grade("\\boxed{\\$1,200}", "1200")[1] and B.grade("\\boxed{12 \\text{ cm}}", "12")[1] and B.grade("\\boxed{50\\%}", "50")[1])
expect("4e no boxed is wrong; a wrong number is wrong", B.grade("the answer is 4", "4") == (None, False) and not B.grade("\\boxed{5}", "4")[1])
expect("4f a cut-off \\boxed{ falls back to the one before it", B.last_boxed("\\boxed{3} then \\boxed{4 + ") == "3")
expect("4g junk in the box is wrong, not an error", B.grade("\\boxed{abc}", "4") == ("abc", False) and B.grade("\\boxed{\\frac{1}{0}}", "4")[1] is False)
v = B.bootstrap([1, 0, 0, 1, 0] * 40)
expect("4h bootstrap is reproducible and contains the mean", v == B.bootstrap([1, 0, 0, 1, 0] * 40) and v[1] <= v[0] <= v[2] and abs(v[0] - 0.4) < 1e-12 and v[2] - v[1] > 0.05, v)
expect("4i bootstrap of nothing is refused", raises_read_error(lambda: B.bootstrap([])) is not True)


def recs(correct, hit):
    return [{"correct": c, "hit_limit": h} for c, h in zip(correct, hit)]


n = 200
base_r = recs([i < 28 for i in range(n)], [i < 186 for i in range(n)])        # 14% accuracy, 93% limit
good_r = recs([i < 134 for i in range(n)], [i < 66 for i in range(n)])        # 67%, 33%
vv = B.verdict(base_r, good_r)
expect("4j the post's numbers reproduce", vv["verdict"] == "reproduces" and vv["accuracy_rule_holds"] and vv["limit_rule_holds"], vv)
acc_only = recs([i < 134 for i in range(n)], [i < 186 for i in range(n)])
lim_only = recs([i < 28 for i in range(n)], [i < 66 for i in range(n)])
expect("4k accuracy alone and limit alone are partial", B.verdict(base_r, acc_only)["verdict"] == "partial" and B.verdict(base_r, lim_only)["verdict"] == "partial")
expect("4l an unchanged model is not reproduced", B.verdict(base_r, base_r)["verdict"].startswith("not reproduced"))
exact20 = recs([i < 28 for i in range(n)], [i < 146 for i in range(n)])      # limit rate 73%: a drop of exactly 20 points
expect("4m a drop of exactly 20 points does not hold (strict)", B.verdict(base_r, exact20)["limit_drop_points"] == 20.0 and not B.verdict(base_r, exact20)["limit_rule_holds"])
half = B.verdict(base_r, base_r)["base_half_width"]
k = int(np.ceil(half * n)) + 1
expect("4n a gain no larger than the half-width does not hold", not B.verdict(base_r, recs([i < 28 + int(half * n) for i in range(n)], [True] * n))["accuracy_rule_holds"]
       and B.verdict(base_r, recs([i < 28 + k for i in range(n)], [True] * n))["accuracy_rule_holds"], half)
expect("4o runs over different problem sets are refused", raises_read_error(lambda: B.verdict(base_r, good_r[:5])) is not True and raises_read_error(lambda: B.verdict([], [])) is not True)
expect("4p the verdict renders", "reproduces" in B.render_verdict(vv, "x"))

problems = [{"id": i, "question": f"q{i}", "gold": str(i)} for i in range(5)]
calls = []


def fake_gen(prompts, max_new, seed):
    calls.append((list(prompts), max_new, seed))
    out = []
    for p in prompts:
        i = int(p.split("\n")[0][1:])
        out.append((f"thinking \\boxed{{{i if i % 2 == 0 else 99}}}", 10 + i, i == 4))
    return out


log = f"{tmp}/run.jsonl"
rs = B.run_behaviour(fake_gen, problems, "base", 100, 5, 2, log)
expect("4q the run loop grades, batches and logs", [r["correct"] for r in rs] == [True, False, True, False, True] and len(calls) == 3
       and calls[0][0][0].endswith(B.SUFFIX) and calls[0][2] == 5 and calls[1][2] == 7 and [r["hit_limit"] for r in rs] == [False] * 4 + [True]
       and len(open(log).read().splitlines()) == 5, (calls, rs))
calls.clear()
lines = open(log).read().splitlines()
open(log, "w").write("\n".join(lines[:3]) + "\n" + lines[3][:20])    # a crash in the middle of the fourth line
rs2 = B.run_behaviour(fake_gen, problems, "base", 100, 5, 2, log)
expect("4r a rerun skips the finished problems and redoes the cut-off line", len(calls) == 1 and len(calls[0][0]) == 2 and [r["id"] for r in rs2] == list(range(5))
       and [r["correct"] for r in rs2] == [r["correct"] for r in rs], calls)
calls.clear()
B.run_behaviour(fake_gen, problems, "edited", 100, 5, 2, log)
B.run_behaviour(fake_gen, problems, "base", 200, 5, 2, log)
expect("4s another model tag or another cap does not reuse the log", len(calls) == 6, len(calls))
expect("4t a generator that returns the wrong count is refused", raises_read_error(lambda: B.run_behaviour(lambda p, m, s: [], problems, "z", 100, 0, 2)) is not True)

jl = f"{tmp}/gsm.jsonl"
open(jl, "w").write("\n".join(json.dumps({"question": f"Q{i}?", "answer": f"work\n#### {i * 1000:,}"}) for i in range(10)) + "\n")
ps = B.load_problems(jl, 4)
expect("4u problems: first n, gold after #### with commas removed", [p["gold"] for p in ps] == ["0", "1000", "2000", "3000"] and ps[0]["id"] == 0)
open(f"{tmp}/badrows.jsonl", "w").write('{"question": 3}\n')
for name, f_, nn in (("a row without question/answer strings", f"{tmp}/badrows.jsonl", 1), ("a missing file", f"{tmp}/none.jsonl", 1),
                     ("fewer problems than asked", jl, 11), ("invalid json", bad_file("notjson", b"{{{"), 1)):
    try:
        B.load_problems(f_, nn)
        r = "raised nothing"
    except B.BehaveError:
        r = True
    except BaseException as e:  # noqa: BLE001
        r = type(e).__name__
    expect(f"4v problems: {name} raises BehaveError", r is True, r)
r = run("behave", "--base", f"{tmp}/base.safetensors", "--edited", f"{tmp}/edited.safetensors", "--data", jl, "--out", f"{tmp}/out")
expect("4w behave without --gpu-free refuses (exit 2) and writes nothing", r.returncode == 2 and "--gpu-free" in r.stderr and not os.path.exists(f"{tmp}/out"), (r.returncode, r.stderr))
r = run("behave", "--base", f"{tmp}/base.safetensors", "--edited", f"{tmp}/edited.safetensors", "--data", jl, "--out", f"{tmp}/out", "--gpu-free", "--n", "3")
expect("4x behave on a checkpoint that is not the published edit refuses (exit 1) before loading anything", r.returncode == 1 and "published edit" in r.stderr and not os.path.exists(f"{tmp}/out"), (r.returncode, r.stderr))

# ---- 5: chunks, a partial directory, status and verdict from the jsonl
chunk_calls = []


def chunk_gen_for(path):
    def g(prompts, max_new, seed):
        chunk_calls.append((os.path.basename(path), len(prompts), seed))
        return fake_gen(prompts, max_new, seed)
    return g


saved_pick = w1.pick_generator
w1.pick_generator = lambda a, model_dir: chunk_gen_for(model_dir)
d1, d2 = f"{tmp}/mono", f"{tmp}/chunks"
jl5 = f"{tmp}/gsm5.jsonl"
open(jl5, "w").write("\n".join(json.dumps({"question": f"q{i}", "answer": f"#### {i}"}) for i in range(10)) + "\n")


def behave_main(out, n, quiet=True):
    args = ["behave", "--base", "base_ckpt", "--edited", "edited_ckpt", "--data", jl5, "--out", out, "--gpu-free", "--no-verify",
            "--n", str(n), "--max-new", "100", "--batch", "2", "--seed", "5"]
    if quiet:
        so, sys.stdout = sys.stdout, open(os.devnull, "w")
    try:
        return w1.main(args)
    finally:
        if quiet:
            sys.stdout.close()
            sys.stdout = so


behave_main(d1, 10)
mono = [c for c in chunk_calls]
chunk_calls.clear()
for n_ in (4, 8):
    behave_main(d2, n_)
# a crash in the middle of the base run's next chunk: whole batches written, then a line cut off
lines = open(f"{d2}/base_100.jsonl").read().splitlines()
open(f"{d2}/base_100.jsonl", "w").write("\n".join(lines[:6]) + "\n" + lines[6][:15])
chunk_calls.clear()
behave_main(d2, 10)
expect("5a a partial directory resumes: only the cut-off batch and the new problems are generated, with the uninterrupted run's seeds",
       [c for c in chunk_calls if c[0] == "base_ckpt"] == [("base_ckpt", 2, 5 + 6), ("base_ckpt", 2, 5 + 8)] and [c for c in chunk_calls if c[0] == "edited_ckpt"] == [("edited_ckpt", 2, 5 + 8)], chunk_calls)
expect("5b the records of the chunked run equal the uninterrupted run's",
       all(sorted(json.loads(x)["id"] for x in open(f"{d2}/{t}_100.jsonl").read().splitlines() if x.endswith("}")) == list(range(10)) for t in ("base", "edited"))
       and all(sorted(r["correct"] for r in B.load_done(f"{d1}/{t}_100.jsonl", t, 100).values()) == sorted(r["correct"] for r in B.load_done(f"{d2}/{t}_100.jsonl", t, 100).values()) for t in ("base", "edited")))
expect("5c the seeds of the chunked run are the uninterrupted run's", sorted({c[2] for c in chunk_calls if c[0] == "edited_ckpt"}) == [13] and sorted(c[2] for c in mono if c[0] == "base_ckpt") == [5, 7, 9, 11, 13])
w1.pick_generator = saved_pick
st = lambda d, n_, mx="100": run("status", "--dir", d, "--n", str(n_), "--max-new", mx)  # noqa: E731
r = st(d2, 10)
expect("5d status: complete directory exits 0", r.returncode == 0 and "base 10/10 edited 10/10 complete" in r.stdout, (r.returncode, r.stdout, r.stderr))
r = st(d2, 11)
expect("5e status: asking for more than was run exits 1 and says how much", r.returncode == 1 and "base 10/11 edited 10/11 incomplete" in r.stdout, (r.returncode, r.stdout))
r = st(d2, 10, "200")
expect("5f status: another cap is not the same run", r.returncode == 1 and "base 0/10" in r.stdout, r.stdout)
r = st(f"{tmp}/nowhere", 3)
expect("5g status: a missing directory is empty, not an error", r.returncode == 1 and "base 0/3" in r.stdout and "Traceback" not in r.stderr, (r.returncode, r.stderr))
r = run("verdict", "--dir", d2, "--n", "10", "--max-new", "100", "--json", f"{tmp}/v.json")
inproc = B.verdict([B.load_done(f"{d2}/base_100.jsonl", "base", 100)[i] for i in range(10)], [B.load_done(f"{d2}/edited_100.jsonl", "edited", 100)[i] for i in range(10)])
vj = json.load(open(f"{tmp}/v.json")) if os.path.isfile(f"{tmp}/v.json") else {}
expect("5h verdict from the jsonl alone equals the in-process rule and writes the json",
       r.returncode == 0 and "W1 verdict (cap 100): " + inproc["verdict"] in r.stdout and vj.get("verdict") == inproc["verdict"] and vj.get("accuracy_gain") == inproc["accuracy_gain"], (r.returncode, r.stdout, r.stderr))
expect("5i the same model twice is not reproduced, so the cap check is not asked", "not asked" in r.stdout, r.stdout)
r = run("verdict", "--dir", d2, "--n", "11", "--max-new", "100")
expect("5j verdict on an incomplete directory exits 1 and says so", r.returncode == 1 and "no verdict" in r.stderr and "Traceback" not in r.stderr, (r.returncode, r.stderr))
v_re = {"verdict": "reproduces"}
v_pa = {"verdict": "partial"}
v_no = {"verdict": "not reproduced at this protocol"}
expect("5k the cap check is asked after reproduces and partial, not after not reproduced, and is itself not asked",
       "ASKED" in w1.cap_check_line(v_re, 8192) and "ASKED" in w1.cap_check_line(v_pa, 8192) and "not asked" in w1.cap_check_line(v_no, 8192)
       and "ASKED" not in w1.cap_check_line(v_re, 16384), "")
# a real verdict from planted logs: base 14% / 93% limit, edited 67% / 33%
os.makedirs(f"{tmp}/plant")
for tag, rr in (("base", base_r), ("edited", good_r)):
    with open(f"{tmp}/plant/{tag}_8192.jsonl", "w") as f:
        for i, x in enumerate(rr):
            f.write(json.dumps({"model": tag, "id": i, "max_new": 8192, **x}) + "\n")
r = run("verdict", "--dir", f"{tmp}/plant")
expect("5l planted logs at the main protocol: reproduces, and the cap check is asked", r.returncode == 0 and "W1 verdict (cap 8192): reproduces" in r.stdout and "ASKED" in r.stdout, r.stdout)

print("w1: ok" if not fails else f"w1: {fails} FAILED")
sys.exit(1 if fails else 0)
