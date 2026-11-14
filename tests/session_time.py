"""Time the requests of a session program one at a time (docs/diffusion.md D6b): `timed(binary, session, n)` starts it, sends n request lines "1", and returns the seconds each answer took, measured from the write of the line to the arrival of the
answer line. The first includes everything the session does before it can answer (opening the device, compiling, uploading the resident files); the others are the steady state with the weights on the device. Answer a small tensor (a probe) so
that the answer's size is not what is timed."""
import subprocess
import time


def timed(binary, session, n, line="1\n"):
    p = subprocess.Popen([binary, session], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        p.stdin.write(line)
        p.stdin.flush()
        out = p.stdout.readline()
        times.append(time.perf_counter() - t0)
        if not out:
            err = p.stderr.read()
            raise SystemExit("the session stopped: " + err[:600])
    p.stdin.close()
    p.wait()
    return times
