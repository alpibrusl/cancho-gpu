#!/usr/bin/env python3
"""docs/weight-space.md section 14, ticket T9: `w1.py behave --backend cancho` against a fake engine (tests/w1c/fake_engine.py, the
protocol of device/w1-main.cho): the Python half of the backend (the subprocess kept alive, the job line, the reply, the records, the
seeds, resume after a cut-off line, a dead or confused engine) with no GPU and no model. The driver itself is held by tests/w1_driver.sh
(its front door, against the mock Metal driver) and, on a Mac with the checkpoints, by the smoke run recorded in docs/weight-space.md 14.12."""
import json
import os
import sys
import tempfile

try:
    import numpy  # noqa: F401
except ImportError:
    print("skip: numpy is not installed")
    sys.exit(0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "weightspace"))
import w1  # noqa: E402
import w1_behave as B  # noqa: E402
import w1_cancho as C  # noqa: E402

FAKE = os.path.join(ROOT, "tests", "w1c", "fake_engine.py")
fails = 0


def expect(name, ok, detail=""):
    global fails
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f"  ({detail})"))
    fails += not ok


def raises(f, exc=B.BehaveError):
    try:
        f()
    except exc as e:
        return str(e)
    return None


tmp = tempfile.mkdtemp()
logf = f"{tmp}/fake.log"
os.environ["FAKE_LOG"] = logf


def jobs():
    return [x for x in (json.loads(l) for l in open(logf).read().splitlines()) if "id" in x]


def argvs():
    return [x["argv"] for x in (json.loads(l) for l in open(logf).read().splitlines()) if "argv" in x]


def reset():
    if os.path.exists(logf):
        os.remove(logf)


problems = [{"id": i, "question": f"q{i}", "gold": str(i)} for i in range(6)]
problems[1]["question"] = "q1|STOP\nwith a newline, a \"quote\" and a backslash \\ and 世界"
problems[3]["question"] = "q3|CUT"
cmd = [FAKE, "model", "kernels", "f16", "576", "--temperature", "0.6"]

# 1. the command line: the protocol's numbers come from w1_behave.SAMPLING
c = C.command_for("bin", "m", "k", "f16", 576)
expect("1a the driver is given the protocol's sampling numbers, from w1_behave.SAMPLING",
       c[5:] == ["--temperature", "0.6", "--top-p", "0.95", "--top-k", "20", "--presence", "1.5"] and c[4] == "576" and os.path.isabs(c[0]) and os.path.isabs(c[2]), c)
expect("1b tmax holds the cap and a prompt, in multiples of 64", C.default_tmax(8192) == 8704 and C.default_tmax(64) == 576 and C.default_tmax(16384) == 16896, "")

# 2. generation, grading, records, seeds (batch 1, as w1.py forces for this backend)
reset()
log = f"{tmp}/run.jsonl"
gen = C.cancho_generator([FAKE, "m", "k", "f16", "576"])
rs = B.run_behaviour(gen, problems, "base", 8, 100, 1, log)
gen.close()
J = jobs()
expect("2a one job a problem, seed = --seed + the problem's id, the cap as given", [(j["id"], j["seed"], j["max_new"]) for j in J] == [(i, 100 + i, 8) for i in range(6)], J)
expect("2b the question reaches the engine intact on one line (newline, quote, backslash, non-ASCII) with the protocol's suffix",
       J[1]["question"] == B.build_question(problems[1]["question"]) and J[0]["question"].endswith(B.SUFFIX), J[1])
expect("2c the records are w1_behave's: the text decoded from hex, graded by its last \\boxed{}, n_tokens counts the stop id, hit_limit",
       [r["correct"] for r in rs] == [True, False, True, False, True, False] and rs[1]["n_tokens"] == 3 and not rs[1]["hit_limit"]
       and rs[3]["n_tokens"] == 8 and rs[3]["hit_limit"] and rs[0]["n_tokens"] == 6 and "héllo 世界" in rs[0]["text"]
       and set(rs[0]) == {"model", "id", "max_new", "gold", "pred", "correct", "n_tokens", "hit_limit", "text"}, rs[0])
expect("2d the log is the jsonl run_behaviour writes", len(open(log).read().splitlines()) == 6 and B.load_done(log, "base", 8).keys() == set(range(6)), "")
reset()
gen = C.cancho_generator([FAKE, "m", "k", "f16", "576"])
rs_b = B.run_behaviour(gen, problems, "base", 8, 100, 1)
gen.close()
expect("2e the same seeds give the same answers (a rerun of a fresh process)", [(r["text"], r["n_tokens"], r["correct"]) for r in rs_b] == [(r["text"], r["n_tokens"], r["correct"]) for r in rs])
stats = []
gen = C.cancho_generator([FAKE, "m", "k", "f16", "576"], stats)
B.run_behaviour(gen, problems[:2], "base", 8, 100, 1)
gen.close()
expect("2f per-job statistics are kept for the caller's throughput line", [s_["new"] for s_ in stats] == [6, 3] and all(s_["gen_ms"] == 3 and s_["prompt_tokens"] == 7 for s_ in stats), stats)
gen = C.cancho_generator([FAKE, "m", "k", "f16", "576"])
expect("2g more than one prompt at a time is refused (batched decode is out of scope)", raises(lambda: gen(["a", "b"], 8, 0)) is not None, "")
gen.close()

# 3. resume: a line cut off by a crash is redone, finished problems are not, and the seeds are the uninterrupted run's
reset()
lines = open(log).read().splitlines()
open(log, "w").write("\n".join(lines[:3]) + "\n" + lines[3][:25])
gen = C.cancho_generator([FAKE, "m", "k", "f16", "576"])
rs2 = B.run_behaviour(gen, problems, "base", 8, 100, 1, log)
gen.close()
J = jobs()
expect("3a only the cut-off problem and the unfinished ones run again: problems 3, 4, 5 with seeds 103, 104, 105 (the job counter restarts with the process, the seed does not)", [j["seed"] for j in J] == [103, 104, 105], J)
expect("3b the resumed records equal the uninterrupted run's, and the log has one line a problem",
       [(r["id"], r["text"], r["correct"]) for r in rs2] == [(r["id"], r["text"], r["correct"]) for r in rs] and len([l for l in open(log).read().splitlines() if l.endswith("}")]) == 6, "")

# 4. a dead, refused or confused engine is a BehaveError that carries the engine's own message
reset()
os.environ["FAKE_NOREADY"] = "1"
m = raises(lambda: C.cancho_generator([FAKE, "m", "k", "f16", "576"]))
del os.environ["FAKE_NOREADY"]
expect("4a an engine that does not load: the error says so and carries its message", m is not None and "q08-io" in m and "ready" in m, m)
os.environ["FAKE_DIE_AFTER"] = "2"
gen = C.cancho_generator([FAKE, "m", "k", "f16", "576"])
m = raises(lambda: B.run_behaviour(gen, problems, "base", 8, 100, 1))
del os.environ["FAKE_DIE_AFTER"]
expect("4b an engine that refuses a job (q08-range) stops the run with its message", m is not None and "q08-range" in m, m)
m = raises(lambda: C.cancho_generator([f"{tmp}/no-such-binary", "m"]))
expect("4c a missing binary is a BehaveError, not a traceback", m is not None and "cannot start" in m, m)

# 5. w1.py behave --backend cancho, end to end on the fake: --batch is ignored, both models run, the verdict is written
reset()
jl = f"{tmp}/gsm.jsonl"
open(jl, "w").write("\n".join(json.dumps({"question": f"q{i}", "answer": f"#### {i}"}) for i in range(6)) + "\n")
out = f"{tmp}/out"
so, sys.stdout = sys.stdout, open(os.devnull, "w")
try:
    rc = w1.main(["behave", "--base", f"{tmp}/mb", "--edited", f"{tmp}/me", "--data", jl, "--out", out, "--gpu-free", "--no-verify", "--n", "6",
                  "--max-new", "8", "--batch", "4", "--seed", "100", "--backend", "cancho", "--cancho-bin", FAKE, "--kernels", "kdir", "--mode", "f32", "--tmax", "640"])
finally:
    sys.stdout.close()
    sys.stdout = so
v = json.load(open(f"{out}/verdict_8.json"))
A = argvs()
expect("5a behave runs both models in an engine process each, with the model dir, kernel dir, mode, tmax and sampling asked for", rc == 0 and len(A) == 2 and A[0][0].endswith("mb") and A[1][0].endswith("me") and [a[2:4] for a in A] == [["f32", "640"]] * 2 and A[0][1].endswith("kdir") and A[0][4:] == ["--temperature", "0.6", "--top-p", "0.95", "--top-k", "20", "--presence", "1.5"], A)
expect("5b --batch is ignored: every problem is its own job with seed 100 + id, for both models", [(j["id"], j["seed"]) for j in jobs()] == [(i, 100 + i) for i in range(6)] * 2, jobs())
expect("5c the verdict records the backend, a batch of 1, the sampling and the seed model",
       v["protocol"]["backend"] == "cancho" and v["protocol"]["batch"] == 1 and v["protocol"]["presence_penalty"] == 1.5 and v["protocol"]["mode"] == "f32" and "splitmix64" in v["protocol"]["seed_model"], v["protocol"])
expect("5d the jsonl files are the ones status and verdict read", all(B.load_done(f"{out}/{t}_8.jsonl", t, 8).keys() == set(range(6)) for t in ("base", "edited")), "")

print("w1_cancho: ok" if not fails else f"w1_cancho: {fails} FAILED")
sys.exit(1 if fails else 0)
