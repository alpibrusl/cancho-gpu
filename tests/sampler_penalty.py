#!/usr/bin/env python3
"""The presence penalty of `cancho-gpu sample` (docs/weight-space.md 13.3, 14.6 C9; ticket T8).

    sampler_penalty.py <cancho-gpu>

Semantics held (the Hugging Face / vLLM definition, as examples/weightspace/w1_behave.py applies it): a token that has appeared
at least once in the GENERATION has the penalty subtracted from its raw logit, once; then temperature, top-k and top-p, in
that order (transformers merges a processor given to `generate` ahead of its temperature, top-k and top-p warpers).

(a) 1,000 random rows, a random seen set each (duplicates, out-of-range ids, empty sets, ties built so that a penalised logit
    equals an unpenalised one in float32), against tools/sampler_dump.py (numpy float32): picks and nucleus ids identical,
    probabilities within 8 ulps (`exp` is the one place the two differ: docs/runtime.md section 10). A mutant (the oracle with
    the penalty ignored) must differ, so the check can fail.
(b) the distribution: two streams of 100,000 draws (200,000 in all) from one row with a seen set, at temperature 0.6, top-p 0.95, top-k 20, penalty 1.5, against
    the exact distribution (float64 numpy; and, when transformers is installed, the processors of transformers on the same row
    in float32): the support equal, and chi-square not rejected at 0.001, with the seed fixed here so the test cannot flake.
Needs numpy; exits 0 with a note without it.
"""
import math
import os
import subprocess
import sys
import tempfile

try:
    import numpy as np
except ImportError:
    print("skip sampler_penalty.py: numpy is not installed")
    sys.exit(0)

BIN = sys.argv[1]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp()
fail = 0


def say(ok, msg):
    global fail
    print(("ok   " if ok else "FAIL ") + msg)
    fail += not ok


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def parse(text):
    rows = []
    for line in text.splitlines():
        head, _, rest = line.partition(": ")
        pick = int(head.split()[3])
        ids = [(int(u.partition("=")[0]), u.partition("=")[2]) for u in rest.split()]
        rows.append((pick, ids))
    return rows


def compare(a, b, limit=8):
    """(ok, worst ulps, bit-identical, total)"""
    if len(a) != len(b):
        return False, 0, 0, 0
    worst = same = total = 0
    for (pa, ia), (pb, ib) in zip(a, b):
        if pa != pb or [i for i, _ in ia] != [i for i, _ in ib]:
            return False, worst, same, total
        for (_, x), (_, y) in zip(ia, ib):
            d = abs(int(x, 16) - int(y, 16))
            worst, same, total = max(worst, d), same + (d == 0), total + 1
    return worst <= limit, worst, same, total


# ---------------------------------------------------------------- (a)
rng = np.random.default_rng(20261105)
VOCAB, ROWS = 300, 1000
logits = rng.normal(0, 2.5, (ROWS, VOCAB)).astype("<f4")
seen_lines = []
for r in range(ROWS):
    if r % 7 == 0:
        seen_lines.append("")
        continue
    ids = rng.integers(0, VOCAB, rng.integers(1, 40)).tolist()
    if r % 11 == 0:
        ids += ids[:3] + [VOCAB + 5, 10 ** 6]  # duplicates, ids outside the vocabulary
    if r % 5 == 0:
        # ties: an unseen token whose logit is exactly the seen token's penalised logit, in float32
        s = ids[0]
        u = [i for i in range(VOCAB) if i not in ids][0]
        logits[r, u] = np.float32(logits[r, s]) - np.float32(1.5)
    seen_lines.append(" ".join(map(str, ids)))
lf, sf = os.path.join(TMP, "a.f32"), os.path.join(TMP, "a-seen.txt")
logits.tofile(lf)
open(sf, "w").write("\n".join(seen_lines) + "\n")
args = [lf, str(VOCAB), "0.6", "0.95", "20", "77", "1.5", sf]
ours = run([BIN, "sample", *args])
theirs = run([sys.executable, os.path.join(ROOT, "tools", "sampler_dump.py"), *args])
say(ours.returncode == 0 and theirs.returncode == 0, "(a) cancho-gpu and the oracle run on 1,000 rows with penalty 1.5")
ok, worst, same, total = compare(parse(ours.stdout), parse(theirs.stdout))
say(ok, f"(a) picks and nucleus ids identical on 1,000 rows, {total} probabilities within 8 ulps of numpy ({same} bit-identical, worst {worst})")
# The penalty itself: for every row, the ids of the nucleus are those of numpy's float32 `scores - penalty * seen` ordering.
pen = np.float32(1.5)
exact = 0
for r, (pick, ids) in enumerate(parse(ours.stdout)):
    sc = logits[r].copy()
    for i in {int(w) for w in seen_lines[r].split() if int(w) < VOCAB}:
        sc[i] = sc[i] - pen
    order = sorted(range(VOCAB), key=lambda i: (-float(sc[i]), i))[: len(ids)]
    exact += order == [i for i, _ in ids]
say(exact == ROWS, f"(a) the nucleus is the head of numpy's penalised order in {exact} of {ROWS} rows")
nopen = run([sys.executable, os.path.join(ROOT, "tools", "sampler_dump.py"), lf, str(VOCAB), "0.6", "0.95", "20", "77"])
say(not compare(parse(ours.stdout), parse(nopen.stdout))[0], "(a) mutant: the oracle without the penalty differs, so the check can fail")
# Penalty 0 and an empty seen file are the plain sampler.
open(sf, "w").write("")
zero = run([BIN, "sample", lf, str(VOCAB), "0.6", "0.95", "20", "77", "1.5", sf])
say(compare(parse(zero.stdout), parse(nopen.stdout))[0], "(a) an empty seen set is the unpenalised sampler")

# ---------------------------------------------------------------- (b)
V, N = 128, 100000  # per run; two runs (the CLI keeps every row it prints in memory: 200,000 rows exhaust it), seeds 20261105 and 20261106
row = rng.normal(0, 1.0, V).astype("<f4")
row[:12] += np.linspace(6, 3, 12).astype("<f4")  # a head, so that top-k and top-p both cut
seen = [0, 1, 2, 5, 40, 100, 100, 7]  # some of the head, some not
T, P, K, PEN = 0.6, 0.95, 20, 1.5
sc = row.astype(np.float64)
for i in set(seen):
    sc[i] -= PEN
sc = sc / T
order = np.argsort(-sc, kind="stable")[:K]
e = np.exp(sc[order] - sc[order[0]])
p = e / e.sum()
cut = int(np.searchsorted(np.cumsum(p), P, side="left")) + 1  # smallest prefix with mass >= top-p
support = [int(i) for i in order[:cut]]
expect = p[:cut] / p[:cut].sum()
gap = abs(np.cumsum(p)[cut - 1] - P)
say(gap > 1e-3 and 1 < cut < K, f"(b) the row's top-p boundary is clear of 0.95 (by {gap:.4f}) and cuts inside the top-{K} ({cut} ids)")
try:
    import torch
    import transformers as TR
    s32 = torch.tensor(row)[None].clone()
    for i in set(seen):
        s32[0, i] -= PEN
    s32 = TR.TemperatureLogitsWarper(T)(None, s32)
    s32 = TR.TopKLogitsWarper(K)(None, s32)
    s32 = TR.TopPLogitsWarper(P)(None, s32)
    hf = torch.softmax(s32, -1)[0].double().numpy()
    hs = [int(i) for i in np.nonzero(hf)[0]]
    say(sorted(hs) == sorted(support), f"(b) transformers' processors (penalty, temperature, top-k, top-p) keep the same {len(hs)} ids as the float64 reference")
    say(float(np.abs(hf[support] - expect).max()) < 1e-6, f"(b) and their probabilities match it (max difference {float(np.abs(hf[support] - expect).max()):.1e})")
except ImportError:
    print("note (b): transformers not installed; the float64 reference alone")
lf = os.path.join(TMP, "b.f32")
np.tile(row, (N, 1)).tofile(lf)
sf = os.path.join(TMP, "b-seen.txt")
open(sf, "w").write((" ".join(map(str, seen)) + "\n") * N)
rows = []
for seed in ("20261105", "20261106"):
    out = run([BIN, "sample", lf, str(V), str(T), str(P), str(K), seed, str(PEN), sf])
    say(out.returncode == 0, f"(b) {N} draws, seed {seed}")
    rows += parse(out.stdout)
N = len(rows)
say(all([i for i, _ in ids] == support for _, ids in rows[:50]) and len({tuple(i for i, _ in ids) for _, ids in rows[:2000]}) == 1,
    f"(b) cancho's nucleus ids equal the reference's support: {support}")
counts = {i: 0 for i in support}
outside = 0
for pick, _ in rows:
    if pick in counts:
        counts[pick] += 1
    else:
        outside += 1
say(outside == 0, "(b) every draw is inside the support")
exp_counts = expect * N
chi2 = float(sum((counts[i] - c) ** 2 / c for i, c in zip(support, exp_counts)))
df = len(support) - 1
# Wilson-Hilferty: the 0.999 quantile of chi-square with df degrees of freedom (z = 3.0902)
crit = df * (1 - 2 / (9 * df) + 3.0902 * math.sqrt(2 / (9 * df))) ** 3
say(chi2 < crit, f"(b) chi-square {chi2:.1f} < {crit:.1f} (df {df}, level 0.001, seed fixed; min expected count {exp_counts.min():.0f})")
# A mutant distribution (no penalty) must be rejected by the same statistic.
sc0 = row.astype(np.float64) / T
o0 = np.argsort(-sc0, kind="stable")[:K]
p0 = np.exp(sc0[o0] - sc0[o0[0]])
p0 /= p0.sum()
c0 = int(np.searchsorted(np.cumsum(p0), P, side="left")) + 1
e0 = {int(i): float(x) / float(p0[:c0].sum()) * N for i, x in zip(o0[:c0], p0[:c0])}
bad = float(sum((counts.get(i, 0) - c) ** 2 / c for i, c in e0.items()))
say(bad > crit * 5, f"(b) mutant: the unpenalised distribution is rejected (chi-square {bad:.0f})")
sys.exit(1 if fail else 0)
