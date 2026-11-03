#!/usr/bin/env python3
"""M8 training pairs made from cancho sources and the real compiler (docs/roadmap-models.md section 7).

Two kinds, both in the exact prompt/reply format cancho-code's ollama agent uses (src/prompt.cho `compose`; the reply is the complete file):

  instruction pair  a top-level function with a doc comment: the brief is the comment, the signature is the function's; the prompt is the task block cancho-code writes for a unit
                    (UNIT / FUNCTION / SIGNATURE / DESCRIPTION / INSTRUCTIONS) around the scaffold (`edition 7;` and the imports the function uses); the reply is the scaffold with the function in it.
                    Kept only when `cancho check --std` accepts the reply (a function that calls its file's other items does not stand alone and is dropped).
  fix pair          an accepted reply broken by one systematic operator (BREAKERS), refused by the real compiler: the prompt is cancho-code's own (the task, the refusals as `feedback_text`
                    prints them with the source line and a caret, the broken file between `---` rules), the reply is the unbroken file. 70% carry the task, 30% are the bare fix prompt.

`feedback_text` and `compose` are ports of src/prompt.cho; tests/m8_pairs.sh (cancho-code's own prompt for the pilot run, byte for byte) holds them. Pure Python standard library; the compiler is run as a subprocess."""
import json
import os
import random
import re
import subprocess
import tempfile

MAIN_RULE = "Reply with the complete file, with the task done, and nothing else: no markdown fences, no explanation.\n"
FIX_ONLY = "The cancho program below is refused by its compiler. Reply with the complete corrected file and nothing else: no markdown fences, no explanation.\n"
INSTRUCTIONS = "Add this function to project.cho, keeping every other function as it is. The file has no main. Do not write tests or examples of your own: the examples above are run for you."


def feedback_text(source, feedback_json):
    """src/prompt.cho `feedback_text`: one line per refusal (at most eight) with the source line and a caret under the column; the first 1,500 bytes as they are when the JSON is not the checker's."""
    try:
        d = json.loads(feedback_json)
        out = []
        lines = source.split("\n")
        for r in d["refused"][:8]:
            line, col = r["position"]["line"], r["position"]["column"]
            out.append("- line %d, column %d: %s (rule %s)\n" % (line, col, r["message"], r["rule"]))
            if 1 <= line <= len(lines):
                out.append("    %s\n    %s^\n" % (lines[line - 1], " " * (col - 1)))
        return "".join(out)
    except (ValueError, KeyError, TypeError):
        return feedback_json.encode()[:1500].decode(errors="ignore")


def compose(source, task="", feedback=""):
    """src/prompt.cho `compose` for the complete-file mode, without hint, memory or facts."""
    p = ""
    job = task.strip()
    if job:
        p += "Your task, for the cancho program below:\n" + job + "\n\n" + MAIN_RULE
    else:
        p += FIX_ONLY
    fb = feedback_text(source, feedback) if feedback else ""
    if fb:
        p += "The current output of the compiler and of the checks on the file:\n" + fb
    return p + "---\n" + source + "---\n"


def task_block(unit, fn, sig, description):
    return "UNIT: %s\nFUNCTION: %s\nSIGNATURE: %s\nDESCRIPTION: %s\nINSTRUCTIONS: %s" % (unit, fn, sig, description, INSTRUCTIONS)


MAIN_CHO = ("edition 7;\n\nfn main(world: World) -> [] int {\n    let Split { io, ffi, fs, heap, args, net, clock, signals, exec } = split(world);\n    release(io);\n    release(ffi);\n"
            "    release(fs);\n    release(heap);\n    release(args);\n    release(net);\n    release(clock);\n    release(signals);\n    release(exec);\n    return 0;\n}\n")


def check(cancho, text, workdir):
    """The checker's answer for `text` as project.cho next to the driver's fixed main.cho (what `project` runs: `cancho check main.cho project.cho --std --output json`): (parsed, raw JSON),
    parsed None when the compiler did not answer."""
    path = os.path.join(workdir, "project.cho")
    open(path, "w").write(text)
    open(os.path.join(workdir, "main.cho"), "w").write(MAIN_CHO)
    try:
        p = subprocess.run([cancho, "check", os.path.join(workdir, "main.cho"), path, "--std", "--output", "json"], capture_output=True, text=True, timeout=60)
        return json.loads(p.stdout), p.stdout
    except (subprocess.TimeoutExpired, ValueError):
        return None, ""


# ---- functions out of a source file ---------------------------------------------------------------------------------------------------------------------------------------

FN_RE = re.compile(r"^(pub )?fn ([A-Za-z_][A-Za-z0-9_]*)\b")


def body_end(lines, i):
    """The index of the line where the brace block that opens on lines[i] (first `{` outside strings, chars and comments) closes, or -1."""
    depth, opened = 0, False
    for j in range(i, len(lines)):
        s = lines[j]
        k = 0
        while k < len(s):
            c = s[k]
            if c == "/" and s[k:k + 2] == "//":
                break
            if c in "\"'":
                q = c
                k += 1
                while k < len(s) and s[k] != q:
                    k += 2 if s[k] == "\\" else 1
            elif c == "{":
                depth += 1
                opened = True
            elif c == "}":
                depth -= 1
                if opened and depth == 0:
                    return j
            k += 1
        if j - i > 400:
            return -1
    return -1


def functions(text):
    """[(name, signature, doc comment, source text of the function)] for the documented top-level functions of a .cho file whose signature fits one line."""
    lines = text.split("\n")
    out = []
    for i, ln in enumerate(lines):
        m = FN_RE.match(ln)
        if not m or "{" not in ln or "->" not in ln:
            continue
        j = i - 1
        doc = []
        while j >= 0 and lines[j].startswith("//"):
            doc.append(lines[j][2:].strip())
            j -= 1
        doc.reverse()
        desc = " ".join(d for d in doc if d and not set(d) <= set("-=~*"))
        e = body_end(lines, i)
        if e < 0 or len(desc) < 24 or "//~" in desc:
            continue
        sig = ln[:ln.rindex("{")].strip()
        sig = sig[4:] if sig.startswith("pub ") else sig
        out.append((m.group(2), sig, desc, "\n".join(lines[i:e + 1])))
    return out


def scaffold_for(fn_text, file_text):
    """`edition 7;` and the `import std.X;` lines of the file whose module the function text names."""
    imps = []
    for m in re.finditer(r"^import (std\.[a-z_0-9]+);", file_text, flags=re.M):
        mod = m.group(1).split(".")[-1]
        if re.search(r"\b%s\." % mod, fn_text):
            imps.append("import %s;" % m.group(1))
    return "edition 7;\n" + ("\n".join(imps) + "\n" if imps else "")


def strip_pub(fn_text):
    return fn_text[4:] if fn_text.startswith("pub ") else fn_text


# ---- breakers ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _sub_at(text, pattern, repl, rng, flags=0):
    ms = list(re.finditer(pattern, text, flags))
    if not ms:
        return None
    m = ms[rng.randrange(len(ms))]
    return text[:m.start()] + (repl(m) if callable(repl) else repl) + text[m.end():]


def _idents(text):
    body = text.split("{", 1)[-1]
    return sorted(set(re.findall(r"\b(?:let|var) ([a-z_][a-z0-9_]*)", body)))


def _rename_use(text, rng):
    names = _idents(text)
    if not names:
        return None
    n = names[rng.randrange(len(names))]
    uses = [m for m in re.finditer(r"\b%s\b" % re.escape(n), text)]
    if len(uses) < 2:
        return None
    m = uses[-1]
    return text[:m.start()] + n + "_" + text[m.end():]


BREAKERS = {
    "drop-effect-row": lambda t, r: _sub_at(t, r"-> \[[a-z_, ]*\] ", "-> ", r),
    "wrong-return-type": lambda t, r: _sub_at(t, r"(-> \[[a-z_, ]*\] )int\b", lambda m: m.group(1) + "bool", r),
    "unknown-name": _rename_use,
    "missing-semicolon": lambda t, r: _sub_at(t, r";(?=\n)", "", r),
    "let-for-var": lambda t, r: _sub_at(t, r"\bvar (?=[a-z_])", "let ", r),
    "drop-return": lambda t, r: _sub_at(t, r"\breturn ", "", r),
    "drop-import": lambda t, r: _sub_at(t, r"^import [^\n]*\n", "", r, re.M),
    "param-type": lambda t, r: _sub_at(t, r"(\(|, )([a-z_]+): int\b", lambda m: m.group(1) + m.group(2) + ": bool", r),
    "drop-close-brace": lambda t, r: _sub_at(t, r"\n\}(?=\n|$)", "", r),
    "misspell-keyword": lambda t, r: _sub_at(t, r"\b(while|return|else)\b", lambda m: {"while": "whilst", "return": "give", "else": "otherwise"}[m.group(1)], r),
    "drop-edition": lambda t, r: _sub_at(t, r"^edition \d+;\n", "", r, re.M),
    "assign-op": lambda t, r: _sub_at(t, r"(?<![=!<>])==(?!=)", " = ", r),
}


def fix_pairs(cancho, unit, scaffold, fn_text, task, rng, workdir, per_base=3):
    """Fix pairs for one accepted reply `scaffold + fn_text`: [(prompt, reply, operator, [rule tags])]."""
    good = scaffold + "\n" + fn_text + "\n"
    out = []
    ops = sorted(BREAKERS)
    rng.shuffle(ops)
    for op in ops:
        if len(out) >= per_base:
            break
        broken = BREAKERS[op](good, rng)
        if broken is None or broken == good:
            continue
        refused, raw = check(cancho, broken, workdir)
        if not refused or not refused.get("refused"):
            continue
        tags = [r["rule"] for r in refused["refused"]]
        prompt = compose(broken, task if rng.random() < 0.7 else "", raw)
        out.append((prompt, good, op, tags))
    return out
