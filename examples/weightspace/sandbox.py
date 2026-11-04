#!/usr/bin/env python3
"""The sandbox of the harness's `write` (docs/weight-space.md section 4): a restricted subprocess that runs the source an agent wrote and gives back one numpy array.

    run(source, shape, workdir, current=None, timeout=20.0, private=(), os_sandbox=True) -> Result

WHAT IT IS. A fresh `python -I` process (no site customisation, no environment but a few variables, working directory = a scratch directory made for this call, a new process group) that
  1. sets resource limits (CPU seconds, the size of any file it writes, address space where the OS honours it), and the parent kills it after `timeout` seconds of wall time (rule `sandbox.time`);
  2. preloads numpy and installs a PEP 578 audit hook that refuses, with a rule tag in the message: any socket event (`sandbox.network`), any process creation, `exec` or `os.system` (`sandbox.process`),
     any ctypes use (`sandbox.native`), the import of modules that would reach outside or train (scipy, torch, jax, tensorflow, sklearn, subprocess, socket, ctypes, multiprocessing, ... `sandbox.import`),
     and any file open, listing, removal, rename, copy or chdir whose path is not under the scratch directory or the Python installation (read-only) (`sandbox.file`);
  3. on macOS, when `sandbox-exec` works, also runs the child under a kernel profile that denies all network, all file writes outside the scratch directory, and reads of `private` paths
     (the run directory, `build/`, the session binary).

WHAT IT IS NOT. It is NOT A SECURITY BOUNDARY; it is a speed bump for an honest agent that wants to be told "no". The audit hook is inside the process it guards. Things it does not stop, said so that nobody
relies on them (tests/weightspace.py pins the first two as known behaviour):
  * `os.stat`, `os.path.exists` and friends are not audited by Python: a program can learn WHETHER a path outside exists (not read it) -- on Linux, where only the hook guards;
  * it reads the whole Python installation (the standard library and site-packages are readable: numpy needs them);
  * it can burn CPU for the whole time limit and print up to the output cap;
  * anything that reaches the OS without a Python audit event (a C extension we did not block that opens files itself, an already-loaded extension module) is not seen; on Linux nothing else catches it;
  * the memory limit is not enforced on every OS; a program can exhaust RAM before the time limit;
  * the numpy it runs is the parent's: a numpy bug or a side channel (timing) is not addressed.
Compliance with "the model may not train" is not this module's job at all: it is by review (review.py, section 4)."""
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import numpy as np

OUTPUT_CAP = 4000          # bytes of the child's stdout/stderr returned
BLOCKED_IMPORTS = ("scipy", "torch", "jax", "jaxlib", "tensorflow", "sklearn", "autograd", "cvxpy", "socket", "ssl", "urllib", "http", "ftplib", "smtplib", "telnetlib",
                   "subprocess", "_posixsubprocess", "_socket", "_ssl", "ctypes", "_ctypes", "multiprocessing", "_multiprocessing", "asyncio", "socketserver", "xmlrpc", "webbrowser", "requests", "httpx", "pty", "shutil", "tempfile", "sqlite3", "mmap")
# audit events: (indices of path arguments, mutates)
PATH_EVENTS = {"os.listdir": ((0,), False), "os.scandir": ((0,), False), "os.chdir": ((0,), False), "os.mkdir": ((0,), True), "os.remove": ((0,), True), "os.rmdir": ((0,), True),
               "os.rename": ((0, 1), True), "os.chmod": ((0,), True), "os.chown": ((0,), True), "os.truncate": ((0,), True), "os.symlink": ((0, 1), True), "os.link": ((0, 1), True),
               "os.utime": ((0,), True), "os.chroot": ((0,), True), "glob.glob": ((0,), False), "shutil.copyfile": ((0, 1), True), "shutil.rmtree": ((0,), True), "shutil.move": ((0, 1), True)}
PROCESS_EVENTS = ("subprocess.Popen", "os.system", "os.exec", "os.fork", "os.forkpty", "os.posix_spawn", "os.spawn", "os.startfile", "pty.spawn", "os.kill", "os.killpg")


@dataclass
class Result:
    ok: bool
    array: object = None          # float64 ndarray when ok
    rule: str = ""                # a sandbox.* tag when the sandbox refused, "" otherwise
    error: str = ""               # the tail of the child's traceback / the parent's reason
    output: str = ""              # what the program printed (capped)


CHILD = r'''
import json, os, resource, sys
workdir, shape_json, cpu, fsize = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
roots_json, blocked_json, events_json = sys.argv[5], sys.argv[6], sys.argv[7]
os.chdir(workdir)
for lim, val in ((resource.RLIMIT_CPU, cpu), (resource.RLIMIT_FSIZE, fsize), (resource.RLIMIT_CORE, 0)):
    try:
        resource.setrlimit(lim, (val, val))
    except (ValueError, OSError):
        pass
try:
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
except (ValueError, OSError):
    pass
import numpy as np
import numpy.linalg, numpy.random, numpy.fft, numpy.polynomial  # loaded now: after the hook a lazy submodule import would read files
import math, itertools, functools, fractions, decimal, random, collections, typing, re, textwrap, cmath, statistics, time, operator, dataclasses  # what a program of pure arithmetic reads
shape = tuple(json.loads(shape_json))
code = open("code.py").read()
cur = {k: np.array(v) for k, v in np.load("current.npz").items()}
outf = open("out.bin", "wb")
roots = [os.path.realpath(r) for r in json.loads(roots_json)] + [os.path.dirname(os.path.dirname(np.__file__))]
blocked = tuple(json.loads(blocked_json))
events = {k: (tuple(v[0]), v[1]) for k, v in json.loads(events_json).items()}
work = os.path.realpath(workdir)
for name in list(sys.modules):
    if name.split(".")[0] in blocked and name.split(".")[0] not in ("re",):
        del sys.modules[name]       # numpy may have imported some; a later `import` of them must be seen by the hook (numpy keeps its own references)
NULLS = ("/dev/null", "/dev/urandom", "/dev/random", "/dev/zero")

def under(path, base):
    return path == base or path.startswith(base.rstrip("/") + "/")

def check(path, write):
    if isinstance(path, int):
        return
    try:
        p = os.path.realpath(os.fsdecode(path))
    except (TypeError, ValueError):
        raise PermissionError("sandbox.file: unreadable path")
    if p in NULLS:
        return
    if under(p, work):
        return
    if not write and any(under(p, r) for r in roots):
        return
    raise PermissionError("sandbox.file: %s of %r is outside the scratch directory" % ("write" if write else "access", os.fsdecode(path)))

def hook(event, args):
    if event == "open":
        mode, flags = args[1], args[2]
        w = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
        check(args[0], bool(w))
    elif event.startswith("socket.") or event.startswith("urllib.Request"):
        raise PermissionError("sandbox.network: %s" % event)
    elif event in events:
        idx, mutates = events[event]
        for i in idx:
            if i < len(args) and args[i] is not None:
                check(args[i], mutates)
    elif event.startswith("ctypes.") or event in ("os.dlopen", "mmap.__new__"):
        raise PermissionError("sandbox.native: %s" % event)
    elif event.startswith("subprocess.") or event in PROCESS:
        raise PermissionError("sandbox.process: %s" % event)
    elif event == "import":
        top = str(args[0]).split(".")[0]
        if top in blocked:
            raise ImportError("sandbox.import: %s may not be imported (no network, no processes, no training libraries)" % args[0])

PROCESS = set(json.loads(sys.argv[8]))
sys.addaudithook(hook)
ns = {"np": np, "numpy": np, "SHAPE": shape, "CURRENT": cur, "__name__": "__write__"}
exec(compile(code, "<write>", "exec"), ns)
if "tensor" not in ns:
    raise SystemExit("the program must define `tensor`, a numpy array of shape %s" % (shape,))
a = np.asarray(ns["tensor"], dtype=np.float64)
if a.shape != shape:
    raise SystemExit("`tensor` has shape %s, wanted %s" % (a.shape, shape))
outf.write(a.tobytes())
outf.close()
'''


def _probe_sandbox_exec():
    if sys.platform != "darwin" or not os.path.exists("/usr/bin/sandbox-exec"):
        return False
    try:
        return subprocess.run(["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


_OS_SANDBOX = None


def os_sandbox_available():
    global _OS_SANDBOX
    if _OS_SANDBOX is None:
        _OS_SANDBOX = _probe_sandbox_exec()
    return _OS_SANDBOX


def _profile(workdir, private):
    q = lambda p: json.dumps(os.path.realpath(p))
    lines = ["(version 1)", "(allow default)", "(deny network*)", "(deny file-write*)",
             "(allow file-write* (subpath %s) (literal \"/dev/null\") (literal \"/dev/dtracehelper\"))" % q(workdir)]
    for p in private:
        lines.append("(deny file-read* (subpath %s))" % q(p))
    lines.append("(allow file-read* (subpath %s))" % q(workdir))       # last match wins: the scratch directory is inside the run directory
    return "\n".join(lines)


def _read_roots():
    import sysconfig
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix}
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        roots.add(sysconfig.get_paths()[key])
    return sorted(roots)


def run(source, shape, workdir, current=None, timeout=20.0, private=(), os_sandbox=True):
    """Runs `source` (it must define `tensor`, shape `shape`; `np`, `SHAPE` and `CURRENT`, the stored tensors, are in scope) in `workdir` (created if missing; nothing else is readable or writable)."""
    os.makedirs(workdir, exist_ok=True)
    workdir = os.path.realpath(workdir)
    open(os.path.join(workdir, "code.py"), "w").write(source)
    np.savez(os.path.join(workdir, "current.npz"), **(current or {"_": np.zeros(1)}))
    out = os.path.join(workdir, "out.bin")
    if os.path.exists(out):
        os.remove(out)
    cpu = int(timeout) + 2
    child = os.path.join(workdir, "_child.py")
    open(child, "w").write(CHILD)
    argv = [sys.executable, "-I", "-B", child, workdir, json.dumps(list(shape)), str(cpu), str(64 << 20), json.dumps(_read_roots()), json.dumps(list(BLOCKED_IMPORTS)),
            json.dumps({k: [list(v[0]), v[1]] for k, v in PATH_EVENTS.items()}), json.dumps(list(PROCESS_EVENTS))]
    env = {"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
    if os_sandbox and os_sandbox_available():
        argv = ["/usr/bin/sandbox-exec", "-p", _profile(workdir, private)] + argv
    try:
        proc = subprocess.run(argv, cwd=workdir, env=env, capture_output=True, timeout=timeout, start_new_session=True)
    except subprocess.TimeoutExpired as e:
        return Result(False, rule="sandbox.time", error="the program ran for more than %.0f s and was killed" % timeout, output=_cap(e.stdout))
    text = _cap(proc.stdout)
    err = proc.stderr.decode("utf-8", "replace")
    if proc.returncode != 0:
        m = re.search(r"sandbox\.[a-z]+", err)
        if proc.returncode in (-24, 152) or "CPU time limit" in err:
            m = re.search(r"(sandbox\.time)", "sandbox.time")
        return Result(False, rule=m.group(0) if m else "", error=err.strip()[-OUTPUT_CAP:], output=text)
    raw = open(out, "rb").read() if os.path.exists(out) else b""
    n = int(np.prod(shape))
    if len(raw) != 8 * n:
        return Result(False, error="no tensor came back", output=text)
    return Result(True, array=np.frombuffer(raw, dtype=np.float64).reshape(shape).copy(), output=text)


def _cap(b):
    return (b or b"").decode("utf-8", "replace")[-OUTPUT_CAP:]


if __name__ == "__main__":
    # python3 examples/weightspace/sandbox.py FILE.py SHAPE   -- run a program in the sandbox by hand
    r = run(open(sys.argv[1]).read(), json.loads(sys.argv[2]), tempfile.mkdtemp(prefix="ws_sandbox_"))
    print(r.ok, r.rule, r.error, r.output)
