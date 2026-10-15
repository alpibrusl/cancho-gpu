#!/usr/bin/env python3
"""Put the text of docs/demo.cast into docs/index.html as the video's transcript
(the <pre id="transcript"> element), so the page and the recording cannot disagree.

    scripts/page-transcript.py
"""
import html
import json
import re
import sys

cast = open("docs/demo.cast").read().splitlines()
out = "".join(json.loads(l)[2] for l in cast[1:] if json.loads(l)[1] == "o")
# The screen is cleared between steps: show the steps one after another.
out = re.sub(r"\x1b\[H\x1b\[2J|\x1b\[3J|\x1b\[H\x1b\[J|\x1b\[2J\x1b\[H", "\n\x00\n", out)
out = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", out).replace("\r\n", "\n").replace("\r", "")
out = re.sub(r"\n*\x00\n*", "\n\n", out).strip("\n")
# `agg` and the typing leave nothing else; drop the stray blank runs.
out = re.sub(r"\n{3,}", "\n\n", out)
page = open("docs/index.html").read()
new = re.sub(r'(<pre id="transcript">).*?(</pre>)', lambda m: m.group(1) + html.escape(out, quote=False) + m.group(2), page, count=1, flags=re.S)
if new == page and "<pre id=\"transcript\">" not in page:
    sys.exit("no transcript element in docs/index.html")
open("docs/index.html", "w").write(new)
print(len(out), "characters of transcript")
