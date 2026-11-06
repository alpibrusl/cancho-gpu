#!/usr/bin/env python3
r"""`cancho-gpu detokenize` against Hugging Face `tokenizers` (`Tokenizer.decode`, special tokens kept), on the real Qwen2.5 tokenizer.json found in the Hugging Face cache.

    detokenize_hf.py [build/cancho-gpu]

Records: 400 random id sequences (any id of the vocabulary, so tokens that end in the middle of a UTF-8 character, added tokens and unprintable bytes occur), the ids of 60 real
sentences and code lines, the empty record; and the refusals (an id past the vocabulary, a non-number). Skipped (exit 0) when `tokenizers` or the tokenizer is missing."""
import glob, os, random, subprocess, sys, tempfile

BIN = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "build/cancho-gpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
paths = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*/tokenizer.json")))
try:
    from tokenizers import Tokenizer
except ImportError:
    print("skip detokenize_hf: the `tokenizers` library is not installed"); sys.exit(0)
if not paths:
    print("skip detokenize_hf: no Qwen2.5 tokenizer.json in the Hugging Face cache"); sys.exit(0)


def quoted(s):
    # the canonical quoting of tools/tokenizer_dump.py (which runs its main on import, so it is copied)
    out = ['"']
    for b in s.encode("utf-8"):
        if b in (0x22, 0x5C):
            out.append("\\" + chr(b))
        elif b < 32 or b >= 127:
            out.append("\\x%02x" % b)
        else:
            out.append(chr(b))
    out.append('"')
    return "".join(out)

tk = Tokenizer.from_file(paths[0])
n = tk.get_vocab_size()
rng = random.Random(11)
records = [[rng.randrange(n) for _ in range(rng.randrange(1, 40))] for _ in range(400)]
sentences = ["def f(x):\n    return x + 1\n", "Hello, world! 你好，世界 🙂", "```cancho\nedition 7;\nfn f(n: int) -> [] int { return n; }\n```", "<|im_start|>assistant\n<|im_end|>"] * 15
records += [tk.encode(s).ids for s in sentences] + [[]]
d = tempfile.mkdtemp(prefix="detok_")
open(d + "/ids", "w").write("".join(" ".join(map(str, r)) + "\n" for r in records))
got = subprocess.run([BIN, "detokenize", paths[0], d + "/ids"], capture_output=True, text=True)
want = ["dec " + quoted(tk.decode(r, skip_special_tokens=False)) for r in records]
lines = got.stdout.splitlines()
bad = [i for i, (a, b) in enumerate(zip(lines, want)) if a != b]
ok = got.returncode == 0 and len(lines) == len(want) and not bad
print(("ok   " if ok else "FAIL ") + f"{len(records)} records (random ids, real text, the empty record): {len(bad)} differ from Hugging Face's decode" + (f", first {bad[0]}: {lines[bad[0]][:80]!r} vs {want[bad[0]][:80]!r}" if bad else ""))
fails = 0 if ok else 1
for name, text in (("an id past the vocabulary", f"1 2 {n + 5}\n"), ("a non-number", "1 x\n")):
    open(d + "/bad", "w").write(text)
    r = subprocess.run([BIN, "detokenize", paths[0], d + "/bad"], capture_output=True, text=True)
    good = r.returncode == 1 and "tok-id" in r.stderr
    print(("ok   " if good else "FAIL ") + f"refusal tok-id: {name}"); fails += 0 if good else 1
sys.exit(1 if fails else 0)
