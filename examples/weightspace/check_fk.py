#!/usr/bin/env python3
"""The numpy forward kinematics (task.fk) against cancho-robot's `src/kinematics.cho`, the program the differential tests of that repository hold to placo.

    python3 examples/weightspace/check_fk.py CANCHO_ROBOT_DIR CANCHO [N]       builds tests/kinematics_driver.cho with CANCHO, runs N random poses (default 2000) and the zero pose

cancho-robot is read, never written: the driver is built into a temporary directory. Prints the largest position difference in metres (kinematics.cho takes degrees; task.fk radians).
`--record FILE` writes the poses and the driver's positions as tests/weightspace/fk_fixture.json (what the gate compares task.fk with, so the gate needs no other checkout)."""
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task  # noqa: E402


def bits(x):
    return struct.unpack("<q", struct.pack("<d", x))[0]


def unbits(n):
    return struct.unpack("<d", struct.pack("<q", n))[0]


def driver_positions(robot, cancho, deg):
    exe = os.path.join(tempfile.mkdtemp(), "kinematics_driver")
    src = [os.path.join(robot, p) for p in ("tests/kinematics_driver.cho", "src/kinematics.cho", "src/so101.cho")]
    subprocess.run([cancho, "build", "--std", *src, "-o", exe], check=True, capture_output=True)
    lines = "".join("fk " + " ".join(str(bits(float(x))) for x in row) + "\n" for row in deg)
    out = subprocess.run([exe], input=lines, capture_output=True, text=True, check=True).stdout.splitlines()
    pose = np.array([[unbits(int(v)) for v in line.split()] for line in out]).reshape(-1, 4, 4)
    return pose[:, :3, 3]


def main():
    robot, cancho = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 2000
    rng = np.random.default_rng(7)
    q = np.concatenate([np.zeros((1, 5)), rng.uniform(task.LIMITS[:, 0], task.LIMITS[:, 1], size=(n, 5)),
                        rng.uniform(-3.5, 3.5, size=(n // 4, 5))])      # the zero pose, the box, and beyond the box
    got = driver_positions(robot, cancho, np.degrees(q))
    # kinematics.cho multiplies by pi/180 itself: feed it degrees that round-trip to q within 1 ulp, so compare to the radians the driver really used
    used = np.array([[float(x) * (3.141592653589793 / 180.0) for x in row] for row in np.degrees(q)])
    want = task.fk(used)
    d = np.abs(got - want).max()
    print("check_fk: %d poses (zero, %d in the box, %d beyond it); largest |difference| of a position component: %.3e m" % (len(q), n, n // 4, d))
    print("zero pose: cancho %s, numpy %s" % (got[0], want[0]))
    if "--record" in sys.argv:
        path = sys.argv[sys.argv.index("--record") + 1]
        k = np.concatenate([[0], np.arange(1, 25)])
        json.dump({"source": "cancho-robot src/kinematics.cho via tests/kinematics_driver.cho", "radians": used[k].tolist(), "position": got[k].tolist()}, open(path, "w"))
        print("recorded", len(k), "poses in", path)
    sys.exit(0 if d < 1e-9 else 1)


if __name__ == "__main__":
    main()
