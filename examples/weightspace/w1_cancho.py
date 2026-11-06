"""docs/weight-space.md section 14, ticket T9: the W1 generation backend over cancho-gpu's own engine.

`cancho_generator` returns a `gen(prompts, max_new, seed) -> [(text, n_tokens, hit_limit)]` for `w1_behave.run_behaviour`, backed by
`build/cancho-gpu-w1` (device/w1-main.cho, scripts/build-w1.sh): the Qwen3.5-0.8B engine on Metal, loaded once and kept alive as a
subprocess, one job a line on its standard input. The sampler is cancho's own (src/sampler.cho), run on the host on the logits row read
back after every token: temperature, top-k, top-p, and the presence penalty over the generated tokens (not the prompt), the numbers
of `w1_behave.SAMPLING`.

What "the same seeds" means here (and what it does not). `run_behaviour` calls `gen` with seed = `--seed` + the index of the chunk's first
problem. Under this backend a chunk is always one problem (`--batch` is ignored: batched decode is out of scope), so problem i is
generated with seed `--seed` + i, whatever was resumed or skipped before it. The driver seeds that job's stream with
`sampler.new(.., seed)`: splitmix64 of the seed into a xorshift64* state, one draw per generated token. That is not
`torch.manual_seed`'s stream (and not the transformers backend's, where eight rows share one stream): the same seed number gives the
same draws only between runs of this backend, never between backends. The two backends are two independent samples of the same
distribution, so a comparison between them is statistical (the bootstrap interval), not a token-by-token match. Within this backend a
run is reproducible bit for bit on the same machine, engine and weight path, and the draw does not depend on the order or on resuming.
The stop ids are 248044 and 248046; the text leaves the stop id out and `n_tokens` counts it, as the transformers backend does."""
import json
import os
import subprocess

import w1_behave as B

STOPS = (248044, 248046)
PROMPT_ROOM = 512   # tokens of room for the prompt when --tmax is not given (the longest of GSM8K's first 200 is far below)


def default_tmax(max_new):
    """The smallest multiple of 64 that holds a W1 prompt and `max_new` new tokens."""
    return -(-(max_new + PROMPT_ROOM) // 64) * 64


class Engine:
    """A `cancho-gpu-w1` process. `command` is its argv up to and including the sampling flags (a fake in the tests)."""

    def __init__(self, command):
        self.command = list(command)
        try:
            self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            raise B.BehaveError(f"cannot start {self.command[0]}: {e}")
        self.jobs = 0
        first = self._line()
        if first != "ready":
            raise self._died(f"expected 'ready', got {first!r}")

    def _line(self):
        raw = self.proc.stdout.readline()
        return raw.decode("utf-8", "replace").rstrip("\n") if raw else None

    def _died(self, why):
        self.proc.kill()
        err = self.proc.stderr.read().decode("utf-8", "replace").strip()
        self.proc.wait()
        return B.BehaveError(f"{os.path.basename(self.command[0])}: {why}" + (f"; its error: {err}" if err else f"; exit status {self.proc.returncode}"))

    def generate(self, question, max_new, seed):
        """(text, n_tokens, hit_limit, info) of one conversation."""
        jid = self.jobs
        self.jobs += 1
        req = json.dumps({"messages": [{"role": "user", "content": question}]}, ensure_ascii=False)
        if "\n" in req or "\r" in req:
            raise B.BehaveError("a request must be one line")
        try:
            self.proc.stdin.write(f"{jid} {int(seed)} {int(max_new)} {req}\n".encode("utf-8"))
            self.proc.stdin.flush()
        except OSError as e:
            raise self._died(f"its input closed ({e})")
        toks = None
        while True:
            line = self._line()
            if line is None:
                raise self._died("it ended before the job finished")
            head = line.split(" ", 2)[0]
            if head == "tokens":
                toks = [int(x) for x in (line.split(" ", 2) + [""])[2].split(",") if x]
            elif head == "gen":
                break
        f = line.split(" ")
        kv = {k: v for k, _, v in (x.partition("=") for x in f[2:])}
        if f[1] != str(jid) or toks is None or int(kv["new"]) != len(toks):
            raise self._died(f"a reply that does not belong to job {jid}: {line[:120]!r}")
        eos = int(kv["eos"])
        if (eos in STOPS) != (len(toks) > 0 and toks[-1] in STOPS) or (int(kv["hit_limit"]) == 1) == (eos in STOPS):
            raise self._died(f"inconsistent stop fields in {line[:120]!r}")
        text = bytes.fromhex(kv["text"]).decode("utf-8", "replace")
        info = {"prompt_tokens": int(kv["prompt"]), "gen_ms": int(kv["gen_ms"]), "tokens": toks}
        return text, int(kv["new"]), int(kv["hit_limit"]) == 1, info

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        for s in (self.proc.stdout, self.proc.stderr):
            s.close()


def command_for(binary, model_dir, kernels, mode, tmax, sampling=None):
    s = sampling or B.SAMPLING
    # absolute and normalised: a kernel directory spelled with `..` makes the driver trap (found by T9's smoke; the path is the engine's, not the sampler's)
    return [os.path.abspath(binary), os.path.abspath(model_dir), os.path.abspath(kernels), mode, str(tmax), "--temperature", repr(s["temperature"]), "--top-p", repr(s["top_p"]),
            "--top-k", str(s["top_k"]), "--presence", repr(s["presence_penalty"])]


def cancho_generator(command, stats=None):
    """A `gen` over a kept-alive `cancho-gpu-w1` (`command` from `command_for`). It takes one prompt at a time; `.close()` ends the
    process. `stats`, if a list, gets {prompt_tokens, new, gen_ms} for every job (the throughput the caller reports)."""
    eng = Engine(command)

    def gen(questions, max_new, seed):
        if len(questions) != 1:
            raise B.BehaveError("the cancho backend decodes one conversation at a time (--batch 1)")
        text, n, hit, info = eng.generate(questions[0], max_new, seed)
        if stats is not None:
            stats.append({"prompt_tokens": info["prompt_tokens"], "new": n, "gen_ms": info["gen_ms"]})
        return [(text, n, hit)]

    gen.close = eng.close
    gen.engine = eng
    return gen
