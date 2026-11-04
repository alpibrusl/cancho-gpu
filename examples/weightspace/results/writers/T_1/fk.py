import numpy as np

_J = [
    ((0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),
    ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0)),
    ((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),
    ((-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),
    ((5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),
]
_TIP = ((-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0))


def _rot(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _fixed(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = _rot(rpy)
    T[:3, 3] = xyz
    return T


_A = [_fixed(*j) for j in _J]
_TIPT = _fixed(*_TIP)


def fk(q):
    q = np.asarray(q, dtype=np.float64)
    n = q.shape[0]
    T = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    for i in range(5):
        c, s = np.cos(q[:, i]), np.sin(q[:, i])
        Rz = np.zeros((n, 4, 4))
        Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s
        Rz[:, 1, 0] = s; Rz[:, 1, 1] = c
        Rz[:, 2, 2] = 1; Rz[:, 3, 3] = 1
        T = T @ (_A[i] @ Rz)
    T = T @ _TIPT
    return T[:, :3, 3]
