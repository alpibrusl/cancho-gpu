#!/usr/bin/env python3
r"""docs/weight-space.md section 14, ticket T2 and check C8: the W1 prompt of Qwen3.5-0.8B.

    chat_w1.py [build/cancho-gpu]

`cancho-gpu chat <request.json> qwen35-08b-w1` (src/chat_w1.cho) renders the one request W1 makes, a single user message with
thinking on, with the model's own chat template. This holds it, byte for byte, to Hugging Face's
`apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)` on the 200 W1 prompts (GSM8K test problems 0 to 199,
the question built by examples/weightspace/w1_behave.py), then holds `cancho-gpu tokenize` on those 200 prompts to Hugging Face's
ids and decode, and checks that the renderer refuses everything it does not render. Skipped (exit 0, with the reason) when
`transformers`, `pyarrow`, the model's directory or the GSM8K parquet is missing; set W1_MODEL_DIR / W1_GSM8K to point at them.
"""
import json
import os
import subprocess
import sys
import tempfile

BIN = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "build/cancho-gpu")
HOME = os.path.expanduser("~")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("W1_MODEL_DIR", HOME + "/models/qwen35-08b-base")
GSM8K = os.environ.get("W1_GSM8K", HOME + "/models/gsm8k/main/test-00000-of-00001.parquet")
N = 200

sys.path.insert(0, os.path.join(ROOT, "examples", "weightspace"))


def skip(why):
    print(f"skip chat_w1: {why}")
    sys.exit(0)


for p, what in ((MODEL + "/tokenizer.json", "the Qwen3.5-0.8B tokenizer"), (MODEL + "/chat_template.jinja", "its chat template"), (GSM8K, "the GSM8K parquet")):
    if not os.path.exists(p):
        skip(f"{what} is not at {p}")
try:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    import w1_behave
except ImportError as e:
    skip(f"{e.name or e} is not installed")

hf = AutoTokenizer.from_pretrained(MODEL)
questions = [w1_behave.build_question(q) for q in pq.read_table(GSM8K).column("question").to_pylist()[:N]]
assert len(questions) == N
bad = 0


def unquote(line):
    """The bytes of a `dec "..."` line: `\\"` and `\\\\` are the character, `\\xNN` a byte."""
    body, out, i = line[len('dec "'):-1], bytearray(), 0
    while i < len(body):
        if body[i] == "\\" and body[i + 1] == "x":
            out.append(int(body[i + 2:i + 4], 16))
            i += 4
        elif body[i] == "\\":
            out += body[i + 1].encode()
            i += 2
        else:
            out += body[i].encode()
            i += 1
    return bytes(out)


def render(req, extra=()):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(req, f, ensure_ascii=False)
    try:
        return subprocess.run([BIN, "chat", f.name, "qwen35-08b-w1", *extra], capture_output=True)
    finally:
        os.remove(f.name)


def want(q):
    return hf.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True)


# 1. the 200 prompts, byte for byte
prompts, differ = [], []
for i, q in enumerate(questions):
    r = render({"messages": [{"role": "user", "content": q}]})
    expected = want(q)
    prompts.append(expected)
    if r.returncode != 0 or r.stdout != expected.encode("utf-8"):
        differ.append(i)
print(f"{'ok  ' if not differ else 'FAIL'} W1 prompt: {N} prompts, " + ("byte-equal to apply_chat_template" if not differ else f"{len(differ)} differ from apply_chat_template: {differ[:10]}"))
bad += bool(differ)
print(f"     shape: {prompts[0]!r}"[:150])

# 2. edge cases of the template itself, against HF: untrimmed text, parts, escapes, non-ASCII, enable_thinking spelled out
edges = ["  padded \n", "tab\tand \"quotes\" \\ back\\slash", "Ünïcode ✓ 日本語 😀", "line1\n\n\nline2", "<|im_end|> injected <think>"]
edge_bad = []
for e in edges:
    for req in ({"messages": [{"role": "user", "content": e}]}, {"messages": [{"role": "user", "content": [{"type": "text", "text": e}]}], "enable_thinking": True}):
        r = render(req)
        if r.returncode != 0 or r.stdout.decode("utf-8") != want(e):
            edge_bad.append(e)
print(f"{'ok  ' if not edge_bad else 'FAIL'} template edge cases: {len(edges)} texts as a string and as parts" + ("" if not edge_bad else f", differ: {edge_bad}"))
bad += bool(edge_bad)

# 3. refusals: everything but the one request is refused by name, nothing on stdout
user = {"role": "user", "content": "q"}
refusals = {
    "system message": {"messages": [{"role": "system", "content": "s"}, user]},
    "two messages": {"messages": [user, {"role": "assistant", "content": "a"}]},
    "assistant only": {"messages": [{"role": "assistant", "content": "a"}]},
    "no messages": {"messages": []},
    "thinking off": {"messages": [user], "enable_thinking": False},
    "tools": {"messages": [user], "tools": [{"type": "function"}]},
    "image": {"messages": [{"role": "user", "content": [{"type": "image"}]}]},
}
for name, req in refusals.items():
    r = render(req)
    good = r.returncode == 1 and r.stdout == b"" and b"chat-" in r.stderr
    print(f"{'ok  ' if good else 'FAIL'} refuses {name}" + ("" if good else f" (exit {r.returncode}, stderr {r.stderr[:80]!r})"))
    bad += not good
r = subprocess.run([BIN, "chat", os.devnull, "no-such-template"], capture_output=True)
good = r.returncode != 0 and r.stdout == b""
print(f"{'ok  ' if good else 'FAIL'} refuses an unknown template name")
bad += not good

# 4. the tokenizer half of C8: ids and decode of the rendered prompts, against `tokenizers`
tok = hf.backend_tokenizer
with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
    f.write("\n%%\n".join(prompts).encode("utf-8"))
try:
    r = subprocess.run([BIN, "tokenize", MODEL + "/tokenizer.json", f.name], capture_output=True, text=True)
finally:
    os.remove(f.name)
lines = [l for l in r.stdout.split("\n") if l.startswith("ids ")]
decs = [l for l in r.stdout.split("\n") if l.startswith("dec ")]
id_bad, dec_bad = [], []
if r.returncode != 0 or len(lines) != N:
    print(f"FAIL tokenize: exit {r.returncode}, {len(lines)} records: {r.stderr[:100]}")
    bad += 1
else:
    for i, p in enumerate(prompts):
        ids = [int(x) for x in lines[i].split(": ", 1)[1].split()]
        theirs = tok.encode(p, add_special_tokens=False).ids
        if ids != theirs:
            id_bad.append(i)
        if unquote(decs[i]) != tok.decode(theirs, skip_special_tokens=False).encode("utf-8") or unquote(decs[i]) != p.encode("utf-8"):
            dec_bad.append(i)
    print(f"{'ok  ' if not id_bad else 'FAIL'} ids: {N} prompts, " + ("equal to Hugging Face tokenizers" if not id_bad else f"{len(id_bad)} differ: {id_bad[:10]}"))
    print(f"{'ok  ' if not dec_bad else 'FAIL'} decode: {N} prompts, " + ("cancho's decode equals Hugging Face's and the prompt" if not dec_bad else f"{len(dec_bad)} differ: {dec_bad[:10]}"))
    bad += bool(id_bad) + bool(dec_bad)
sys.exit(1 if bad else 0)
