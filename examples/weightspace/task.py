"""The W2 task (docs/weight-space.md section 7): the SO-101 forward kinematics, five joint angles (radians) -> the gripper frame's position (x, y, z, metres), as a 5 -> 64 -> 64 -> 3 tanh network.

Everything here is float64 numpy and is the *reference* (the scorer's truth); the chain is `so101_chain.py`, copied from the URDF cancho-robot generates `src/kinematics.cho` from, and held to that
program by `check_fk.py` (when the checkout is present) and by recorded outputs of it in `tests/weightspace/fk_fixture.json` (always).

The pose space and its split [choice, fixed on 2026-11-04 before any arm was run; docs/roadmap-models.md section 8.3's split]:
  * the box is the URDF's limits of the five moving joints;
  * the BAND is every pose with `shoulder_pan` in [PAN_HOLE] or `wrist_flex` in its upper third [WF_HIGH, limit] (the "extreme third" is read as one end of the range: the third of the range that is at the top);
  * the HELD-IN region is the box minus the band. `forward` (the harness) and every trained arm draw from it, never from the band; the *interior* held-out set is fresh draws from it, the *band* set
    is fresh draws from the band."""
import numpy as np

from so101_chain import CHAIN

LIMITS = np.array([[c[3], c[4]] for c in CHAIN[:5]])          # [5, 2] radians
PAN_HOLE = (0.64, 1.28)                                       # radians, an interval of shoulder_pan, a sixth of its range
WF_HIGH = LIMITS[3, 1] / 3.0                                  # wrist_flex above this is the extreme third
DIMS = (5, 64, 64, 3)
NAMES = ("w1", "b1", "w2", "b2", "w3", "b3")
SHAPES = {"w1": (64, 5), "b1": (64,), "w2": (64, 64), "b2": (64,), "w3": (3, 64), "b3": (3,)}
N_PARAMS = sum(int(np.prod(s)) for s in SHAPES.values())      # 5 -> 64 -> 64 -> 3: 4,611

# fixed seeds (numpy SeedSequence entropy). The two held-out sets are made only by `evaluate.py`; nothing else reads these two.
SEED_INTERIOR = 20261104001
SEED_BAND = 20261104002
SEED_TRAIN = 20261104003                                      # the trained arms' samples (a prefix of one stream, so G(1e3) is inside G(1e4))
SEED_INIT = 20261104004                                       # the trainer's random initialisation (R, and where every G starts)
SEED_BOOT = 20261104                                          # the bootstrap's (docs/weight-space.md section 6, 7)


def _rot(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def fk(q):
    """Positions [n, 3] (metres) of gripper_frame_link in base_link's frame for joint angles q [n, 5] (radians), binary64."""
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    n = q.shape[0]
    t = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    for k, (_, xyz, rpy, _, _) in enumerate(CHAIN):
        e = np.zeros((n, 4, 4))
        e[:, :3, :3] = _rot(rpy)
        e[:, :3, 3] = xyz
        e[:, 3, 3] = 1.0
        if k < 5:
            c, s = np.cos(q[:, k]), np.sin(q[:, k])
            rz = np.zeros((n, 4, 4))
            rz[:, 0, 0], rz[:, 0, 1], rz[:, 1, 0], rz[:, 1, 1], rz[:, 2, 2], rz[:, 3, 3] = c, -s, s, c, 1.0, 1.0
            e = e @ rz
        t = t @ e
    return t[:, :3, 3]


def in_band(q):
    q = np.atleast_2d(q)
    return ((q[:, 0] >= PAN_HOLE[0]) & (q[:, 0] <= PAN_HOLE[1])) | (q[:, 3] >= WF_HIGH)


def _draw(rng, n, want_band):
    out = np.empty((0, 5))
    while len(out) < n:
        q = rng.uniform(LIMITS[:, 0], LIMITS[:, 1], size=(max(2 * n, 64), 5))
        out = np.concatenate([out, q[in_band(q) == want_band]])
    return out[:n]


def sample_held_in(n, rng):
    """n poses [n, 5] uniform in the box minus the band."""
    return _draw(rng, n, False)


def sample_band(n, rng):
    """n poses [n, 5] uniform in the band."""
    return _draw(rng, n, True)


def train_set(n):
    """The first n poses of the trained arms' stream and their positions (a prefix: the arms nest)."""
    q = sample_held_in(1_000_000, np.random.default_rng(SEED_TRAIN))[:n]
    return q, fk(q)


def sets():
    """(interior, band): 2,000 poses each, fixed seeds. Only the evaluator calls this."""
    return (sample_held_in(2000, np.random.default_rng(SEED_INTERIOR)), sample_band(2000, np.random.default_rng(SEED_BAND)))


def unflatten(flat):
    out, i = {}, 0
    for name in NAMES:
        n = int(np.prod(SHAPES[name]))
        out[name] = np.asarray(flat[i:i + n], dtype=np.float64).reshape(SHAPES[name])
        i += n
    return out


def flatten(p):
    return np.concatenate([np.asarray(p[n], dtype=np.float64).reshape(-1) for n in NAMES])


def init_params(seed=SEED_INIT):
    """The trainer's random initialisation (examples/train/deep_session.init: Glorot-uniform weights, zero biases), float32."""
    rng = np.random.default_rng(seed)
    p = {}
    for i, (k, n) in enumerate(zip(DIMS[:-1], DIMS[1:]), 1):
        lim = np.sqrt(6.0 / (k + n))
        p["w%d" % i] = rng.uniform(-lim, lim, size=(n, k)).astype(np.float32)
        p["b%d" % i] = np.zeros(n, dtype=np.float32)
    return p


def zero_params():
    return {n: np.zeros(SHAPES[n], dtype=np.float32) for n in NAMES}


def forward_np(p, x):
    """The network in binary64 from (float32) parameters: tanh, tanh, linear. [n, 5] -> [n, 3]."""
    p = {k: np.asarray(v, dtype=np.float64) for k, v in p.items()}
    h = np.tanh(np.asarray(x, dtype=np.float64) @ p["w1"].T + p["b1"])
    h = np.tanh(h @ p["w2"].T + p["b2"])
    return h @ p["w3"].T + p["b3"]


def errors_mm(p, q):
    """Per-pose Euclidean position error of the network, millimetres."""
    return 1000.0 * np.linalg.norm(forward_np(p, q) - fk(q), axis=1)


def bootstrap_ci(values, resamples=10_000, seed=SEED_BOOT):
    """(mean, lo, hi): the mean and its 95% percentile bootstrap interval over `resamples` resamples of the values, numpy default_rng(seed)."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(resamples, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
