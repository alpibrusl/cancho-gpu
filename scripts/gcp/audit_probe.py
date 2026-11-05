#!/usr/bin/env python3
"""Step `sandbox_linux` of docs/cloud.md: why does tests/weightspace.py's check
"a network call is refused" fail on Linux?  (Result there: rule None, ok True, the
program ran and stored its tensor.)  This tries the SAME network calls the test
makes, bare and under examples/weightspace/sandbox.py, and prints the exact audit
events (PEP 578) seen, so the gap can be read from the log.

    python3 scripts/gcp/audit_probe.py

Part 1, BARE: this process, with a recording audit hook; each snippet is exec'd.
Part 2, SANDBOXED: the sandbox's own child program (sandbox.CHILD) with a recording
hook installed before the sandbox's own, so the events that reached it are listed
beside what it decided. Exit 0 always: it is a diagnostic; read the output."""
import os
import platform
import socket
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "examples", "weightspace"))
import sandbox  # noqa: E402


def snippets(port):
    return [
        ("import socket; socket.create_connection", "import socket\ns = socket.create_connection(('127.0.0.1', %d))\ntensor = np.zeros(3)\n" % port),
        ("urllib.request.urlopen", "import urllib.request\nurllib.request.urlopen('http://127.0.0.1:%d/')\ntensor = np.zeros(3)\n" % port),
        ("importlib.import_module('_socket')", "import importlib\nimportlib.import_module('_socket')\ntensor = np.zeros(3)\n"),
    ]


def main():
    print("python", sys.version.replace("\n", " "), "|", platform.platform())
    print("sys.addaudithook:", hasattr(sys, "addaudithook"), "| blocked imports:", " ".join(sandbox.BLOCKED_IMPORTS[:12]), "...")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(5)
    port = listener.getsockname()[1]

    print("\n=== 1. BARE: events a plain process sees")
    seen = []

    def bare(event, args):
        if event.startswith(("socket.", "urllib")) or (event == "import" and "socket" in repr(args)):
            seen.append((event, repr(args)[:100]))

    sys.addaudithook(bare)
    socket.setdefaulttimeout(3)      # urlopen waits for an HTTP answer the listener never gives
    import numpy as np
    for label, code in snippets(port):
        seen.clear()
        status = "ran"
        try:
            exec(compile(code, "<probe>", "exec"), {"np": np})
        except BaseException as e:  # noqa: BLE001
            status = "raised %s: %s" % (type(e).__name__, str(e)[:80])
        print(f"- {label}: {status}")
        for e, a in seen:
            print(f"    event {e} {a}")
    print("socket already in sys.modules before the probe:", "socket" in sys.modules, "| _socket:", "_socket" in sys.modules)

    listener.settimeout(0.5)
    n = 0
    try:
        while True:
            listener.accept()
            n += 1
    except socket.timeout:
        pass
    print(f"connections the BARE part made to the listener (expected: it is bare): {n}")

    print("\n=== 2. SANDBOXED: sandbox.run, with a recording hook in the child")
    rec = (
        "_ev = open('events.log', 'w')\n"
        "def rec(event, args):\n"
        "    if event.startswith(('socket.', 'urllib', 'ctypes', 'subprocess', 'os.system', 'os.fork')) or (event == 'import' and str(args[0]).split('.')[0] in blocked):\n"
        "        _ev.write('%s %s\\n' % (event, repr(args)[:100])); _ev.flush()\n"
        "sys.addaudithook(rec)\n"
    )
    saved = sandbox.CHILD
    sandbox.CHILD = saved.replace("sys.addaudithook(hook)", rec + "sys.addaudithook(hook)")
    assert sandbox.CHILD != saved, "the child program changed: the probe's hook point is gone"
    try:
        for label, code in snippets(port):
            work = tempfile.mkdtemp(prefix="ws_probe_")
            r = sandbox.run(code, (3,), work, timeout=8.0, os_sandbox=True)
            print(f"- {label}: ok={r.ok} rule={r.rule!r} error={r.error[-200:]!r} output={r.output!r}")
            log = os.path.join(work, "events.log")
            events = open(log).read().strip().splitlines() if os.path.exists(log) else ["(no events.log: the hook point was not reached)"]
            print("    events the child's hook saw:" if events else "    no watched events at all")
            for line in events[:12]:
                print("     ", line)
    finally:
        sandbox.CHILD = saved
    listener.settimeout(1.0)
    try:
        listener.accept()
        print("\nA CONNECTION FROM THE SANDBOX REACHED THE LISTENER (the sandbox did not stop it)")
    except socket.timeout:
        print("\nnothing from the sandbox connected to the listener")
    return 0


if __name__ == "__main__":
    sys.exit(main())
