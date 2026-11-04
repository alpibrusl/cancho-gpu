#!/usr/bin/env python3
"""The harness of docs/weight-space.md section 4: the only tools a writer agent has, and the log of every call.

    python3 examples/weightspace/harness.py init RUN_DIR --budget B [--init zeros|random] [--spec chain|none] [--backend session|numpy] [--session PATH] [--timeout S]
    python3 examples/weightspace/harness.py RUN_DIR describe
    python3 examples/weightspace/harness.py RUN_DIR show TENSOR START COUNT
    python3 examples/weightspace/harness.py RUN_DIR write TENSOR SOURCE.py       (SOURCE.py is run in the sandbox; `-` reads standard input)
    python3 examples/weightspace/harness.py RUN_DIR forward N_POSES
    python3 examples/weightspace/harness.py RUN_DIR status
  any call also takes --tokens-in N --tokens-out M (what the driver of the agent says the exchange cost; logged as given).

Four tools (`Harness.describe/show/write/forward`):
  * `describe()`  the architecture, the tensors and the task statement; free.
  * `show(tensor, start, count)`  the stored weights as text; free.
  * `write(tensor, source)`  runs the source in the sandbox (sandbox.py: a restricted subprocess, NOT a security boundary) and stores the f32 tensor it defines; free.
  * `forward(n_poses)`  COUNTED: n_poses (1..256) fresh held-in poses, drawn by the harness from OS entropy, run through the network by the cancho-gpu CPU session (the forward the trained arms use),
    and ONLY the scalar mean Euclidean position error in millimetres comes back (never a per-pose value, never the poses, never the seed). A budget of B calls is fixed at `init`; the count is kept in
    RUN_DIR/state.json *before* the pass runs; calls beyond B are refused (`forward.budget`) and logged. The held-out sets (task.sets) are not reachable from here.

The agent's python (the sandbox) has no cancho-gpu binary, no run directory, no network: it reaches the session only through `forward`. Every call, with the full source of every `write`, its result and any token
counts the driver supplies, is one line of RUN_DIR/log.jsonl. Compliance with "the model may not train" is by review (review.py), not by construction: this harness cannot stop a program from computing a
table in numpy and fitting weights to it, it can only record that it did."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sandbox  # noqa: E402
import task  # noqa: E402
import trainer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
MAX_POSES = 256
SHOW_CAP = 2000
DEFAULT_SESSION = os.environ.get("CANCHO_GPU_SESSION", os.path.join(ROOT, "build", "cancho-gpu-session"))


class Refused(Exception):
    pass


class Harness:
    def __init__(self, run_dir, budget=None, init="zeros", spec="chain", backend="session", session=None, timeout=20.0, draw_seed=None, os_sandbox=None):
        """Opens RUN_DIR; creates it (config, state, initial weights) when `budget` is given and it is new. `draw_seed` fixes forward's draws (tests only: a real run leaves it None)."""
        self.dir = os.path.abspath(run_dir)
        cfg_path = os.path.join(self.dir, "config.json")
        if budget is not None and not os.path.exists(cfg_path):
            os.makedirs(os.path.join(self.dir, "weights"), exist_ok=True)
            cfg = {"budget": budget, "init": init, "spec": spec, "backend": backend, "session": os.path.abspath(session or DEFAULT_SESSION), "timeout": timeout}
            json.dump(cfg, open(cfg_path, "w"), indent=1)
            self._save_state({"used": 0, "writes": 0, "calls": 0})
            p = task.zero_params() if init == "zeros" else task.init_params()
            for n in task.NAMES:
                self._store(n, p[n])
            self._log("init", cfg, {"ok": True})
        self.cfg = json.load(open(cfg_path))
        self.draw_seed = draw_seed
        self.os_sandbox = (os.environ.get("WS_NO_OS_SANDBOX") is None) if os_sandbox is None else os_sandbox      # WS_NO_OS_SANDBOX=1: the audit hook alone (what Linux has)
        self.timeout = self.cfg["timeout"]

    # -- state and log ---------------------------------------------------------------------------------------------------------------------------------------------------------
    def _save_state(self, st):
        tmp = os.path.join(self.dir, "state.json.tmp")
        json.dump(st, open(tmp, "w"))
        os.replace(tmp, os.path.join(self.dir, "state.json"))

    def _state(self):
        return json.load(open(os.path.join(self.dir, "state.json")))

    def _log(self, call, args, result, tokens=None, extra=None):
        st = self._state() if os.path.exists(os.path.join(self.dir, "state.json")) else {"calls": 0, "used": 0, "writes": 0}
        st["calls"] += 1
        self._save_state(st)
        rec = {"seq": st["calls"], "time": time.time(), "call": call, "args": args, "result": result}
        if tokens:
            rec["tokens"] = tokens
        if extra:
            rec["private"] = extra          # for the reviewer only (e.g. the seed of a forward's draw); never returned to the agent
        with open(os.path.join(self.dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _wpath(self, name):
        return os.path.join(self.dir, "weights", name + ".f32")

    def _store(self, name, arr):
        np.asarray(arr, dtype="<f4").reshape(-1).tofile(self._wpath(name))

    def params(self):
        return {n: np.fromfile(self._wpath(n), dtype="<f4").reshape(task.SHAPES[n]) for n in task.NAMES}

    # -- the tools -------------------------------------------------------------------------------------------------------------------------------------------------------------
    def describe(self, tokens=None):
        text = _describe(self.cfg)
        self._log("describe", {}, {"ok": True}, tokens)
        return text

    def show(self, tensor, start=0, count=SHOW_CAP, tokens=None):
        if tensor not in task.SHAPES:
            return self._refuse("show", {"tensor": tensor}, "show.tensor", "unknown tensor %r; the tensors are %s" % (tensor, ", ".join(task.NAMES)), tokens)
        flat = self.params()[tensor].reshape(-1)
        start, count = max(0, int(start)), max(0, min(int(count), SHOW_CAP))
        vals = flat[start:start + count]
        text = "%s shape %s (%d values), values %d..%d, row-major:\n%s" % (tensor, list(task.SHAPES[tensor]), flat.size, start, start + len(vals) - 1, " ".join("%.9g" % v for v in vals))
        self._log("show", {"tensor": tensor, "start": start, "count": count}, {"ok": True, "values": len(vals)}, tokens)
        return text

    def write(self, tensor, source, tokens=None):
        if tensor not in task.SHAPES:
            return self._refuse("write", {"tensor": tensor, "source": source}, "write.tensor", "unknown tensor %r; the tensors are %s" % (tensor, ", ".join(task.NAMES)), tokens)
        st = self._state()
        st["writes"] += 1
        self._save_state(st)
        work = os.path.join(self.dir, "scratch", "%04d" % st["writes"])
        private = [self.dir, os.path.join(ROOT, "build"), os.path.dirname(self.cfg["session"])]
        r = sandbox.run(source, task.SHAPES[tensor], work, current={n: v for n, v in self.params().items()}, timeout=self.timeout, private=private, os_sandbox=self.os_sandbox)
        res = {"ok": r.ok, "tensor": tensor, "shape": list(task.SHAPES[tensor])}
        if r.ok:
            arr = r.array.astype(np.float32)
            if not np.isfinite(arr).all():
                res.update(ok=False, rule="write.finite", error="the tensor has NaN or infinite values; nothing was stored")
            else:
                self._store(tensor, arr)
        else:
            res.update(rule=r.rule or "write.error", error=r.error)
        res["output"] = r.output
        self._log("write", {"tensor": tensor, "source": source}, res, tokens)
        return res

    def forward(self, n_poses, tokens=None):
        args = {"n_poses": n_poses}
        if not isinstance(n_poses, int) or isinstance(n_poses, bool) or not 1 <= n_poses <= MAX_POSES:
            return self._refuse("forward", args, "forward.n", "n_poses must be an integer from 1 to %d" % MAX_POSES, tokens)
        st = self._state()
        if st["used"] >= self.cfg["budget"]:
            return self._refuse("forward", args, "forward.budget", "the budget of %d forward passes is spent" % self.cfg["budget"], tokens, used=st["used"], left=0)
        st["used"] += 1                              # counted before the pass runs: a crash buys nothing
        self._save_state(st)
        seed = self.draw_seed if self.draw_seed is not None else int.from_bytes(os.urandom(8), "little")
        q = task.sample_held_in(n_poses, np.random.default_rng([seed, st["used"]]))
        p = self.params()
        if self.cfg["backend"] == "session":
            if not os.path.exists(self.cfg["session"]):
                raise RuntimeError("the session binary %s does not exist (CANCHO=... scripts/build-session.sh)" % self.cfg["session"])
            pred = trainer.run_forward(self.cfg["session"], os.path.join(self.dir, "_fwd"), p, q.astype(np.float32), rows=MAX_POSES)
        else:
            pred = task.forward_np(p, q.astype(np.float32))
        err = 1000.0 * np.linalg.norm(pred - task.fk(q), axis=1)
        left = self.cfg["budget"] - st["used"]
        if not np.isfinite(err).all():
            res = {"ok": False, "rule": "forward.finite", "error": "the network produced NaN or infinite outputs; the pass is spent", "passes_used": st["used"], "passes_left": left}
        else:
            res = {"ok": True, "score_mm": float(err.mean()), "n_poses": n_poses, "passes_used": st["used"], "passes_left": left}
        self._log("forward", args, res, tokens, extra={"seed": [seed, st["used"]], "backend": self.cfg["backend"]})
        return res

    def status(self):
        st = self._state()
        return {"budget": self.cfg["budget"], "passes_used": st["used"], "passes_left": self.cfg["budget"] - st["used"], "writes": st["writes"]}

    def _refuse(self, call, args, rule, msg, tokens, **more):
        res = {"ok": False, "rule": rule, "error": msg, **more}
        self._log(call, args, res, tokens)
        return res


def _describe(cfg):
    lines = ["TASK. Write the weights of a small neural network that computes the forward kinematics of a 5-joint robot arm (the SO-101): five joint angles in radians -> the (x, y, z) position in metres",
             "of the gripper frame in the base frame. You may not train: you may reason, and use numpy for analysis, but no weight may be found by fitting, searching or optimising against computed outputs of the",
             "target function (least squares, scipy/torch/autograd, a loop that adjusts a parameter to lower an error you computed). Setting a weight by a closed formula from the structure of the problem is allowed,",
             "and so is tabulating a function at fixed points. Compliance is reviewed from the log of your code; borderline code is reported as borderline.",
             "",
             "NETWORK. input x [5] (radians) -> h1 = tanh(W1 x + b1) [64] -> h2 = tanh(W2 h1 + b2) [64] -> y = W3 h2 + b3 [3] (metres). Tensors (row-major, float32): " +
             ", ".join("%s%s" % (n, list(task.SHAPES[n])) for n in task.NAMES) + ". W has shape [out, in]. 4,611 weights in all.",
             "INITIAL WEIGHTS. " + ("all zero." if cfg["init"] == "zeros" else "the trainer's random initialisation (Glorot-uniform weights, zero biases)."),
             "",
             "TOOLS. show(tensor, start, count): the stored weights as text. write(tensor, source): runs your python (numpy only; no network, no files outside a scratch directory, no subprocesses, "
             "a time limit of %.0f s) and stores the array `tensor` it defines (its shape is in `SHAPE`; `CURRENT` holds the stored tensors; `np` is numpy). forward(n_poses): scores the CURRENT network on n_poses "
             "(1..%d) fresh random poses and returns only the mean Euclidean position error in millimetres. You have %d forward calls in all; calls beyond that are refused. Nothing else tells you how good "
             "the network is." % (cfg["timeout"], MAX_POSES, cfg["budget"]), ""]
    if cfg["spec"] == "chain":
        lines += ["KINEMATICS (the URDF, so101_new_calib). The pose of the gripper frame is the product, along the chain below, of one 4x4 transform per joint: a joint's frame is its parent's, moved by xyz, turned by "
                  "R = Rz(yaw) Ry(pitch) Rx(roll) with rpy = (roll, pitch, yaw), then turned about its own z axis by the joint angle q (none for the last, fixed joint). The position is the translation of the product. "
                  "Joint angles are in radians; limits [lower, upper] in radians; poses are scored inside these limits.",
                  "  name                  xyz (m)                                        rpy (rad)                                    limits"]
        for name, xyz, rpy, lo, hi in task.CHAIN:
            lines.append("  %-20s  %-45s  %-43s  %s" % (name, xyz, rpy, "fixed" if lo is None else "[%s, %s]" % (lo, hi)))
    else:
        lines.append("No kinematic data is given beyond the joint limits: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll in [%s]." % ", ".join(str(list(map(float, l))) for l in task.LIMITS))
    return "\n".join(lines)


def _tokens(argv):
    t = {}
    for k, key in (("--tokens-in", "in"), ("--tokens-out", "out")):
        if k in argv:
            i = argv.index(k)
            t[key] = int(argv[i + 1])
            del argv[i:i + 2]
    return t or None


def main(argv):
    if argv and argv[0] == "init":
        run, rest = argv[1], argv[2:]
        o = lambda k, d: type(d)(rest[rest.index(k) + 1]) if k in rest else d
        Harness(run, budget=int(o("--budget", 0)), init=o("--init", "zeros"), spec=o("--spec", "chain"), backend=o("--backend", "session"), session=o("--session", ""), timeout=float(o("--timeout", 20.0)))
        print("initialised", run)
        return 0
    run, call, rest = argv[0], argv[1], argv[2:]
    tokens = _tokens(rest)
    h = Harness(run)
    if call == "describe":
        print(h.describe(tokens))
    elif call == "show":
        out = h.show(rest[0], int(rest[1]), int(rest[2]), tokens)
        print(out if isinstance(out, str) else json.dumps(out))
    elif call == "write":
        src = sys.stdin.read() if rest[1] == "-" else open(rest[1]).read()
        print(json.dumps(h.write(rest[0], src, tokens), indent=1))
    elif call == "forward":
        print(json.dumps(h.forward(int(rest[0]), tokens)))
    elif call == "status":
        print(json.dumps(h.status()))
    else:
        print("usage: see the module docstring", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
