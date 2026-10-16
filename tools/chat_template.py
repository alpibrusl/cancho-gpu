#!/usr/bin/env python3
"""The oracle for `cancho-gpu chat`: render a chat-completions request with the model's own chat template
(`chat_template` of its tokenizer_config.json, found in Ollama's store) through jinja2, the way Hugging Face
does (an immutable sandbox, trim_blocks, lstrip_blocks, `raise_exception`).

    chat_template.py <tokenizer_config.json> <request.json>

Prints the prompt (no trailing newline), or exits 1 when the template refuses the request. The template's
variables come from the request as `docs/serving.md` says: `enable_thinking` (a bool), `think` (a bool, or an
effort that also turns thinking on), `reasoning_effort` (low, medium or high; high is the template's xhigh);
thinking is off when the request names none of them.
"""
import json
import sys

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

cfg = json.load(open(sys.argv[1]))
req = json.load(open(sys.argv[2]))


def raise_exception(message):
    raise TemplateError(message)


env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.globals["raise_exception"] = raise_exception
env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
template = env.from_string(cfg["chat_template"])

think = False
effort = "xhigh"  # the template's default when thinking is on and none is named
if isinstance(req.get("enable_thinking"), bool):
    think = req["enable_thinking"]
if isinstance(req.get("think"), bool):
    think = req["think"]
elif isinstance(req.get("think"), str):
    think, effort = True, req["think"]
if isinstance(req.get("reasoning_effort"), str):
    effort = req["reasoning_effort"]
    if "enable_thinking" not in req and "think" not in req:
        think = True
if effort == "high":
    effort = "xhigh"

try:
    out = template.render(messages=req["messages"], add_generation_prompt=True, enable_thinking=think, reasoning_effort=effort)
except TemplateError as e:
    print(f"template refused: {e}", file=sys.stderr)
    sys.exit(1)
sys.stdout.write(out)
