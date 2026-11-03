#!/usr/bin/env python3
"""The gated delta rule of Qwen3.5/3.8's linear-attention layers, in binary64 numpy: the specification of docs/roadmap-models.md M10.

Three things, each held to something by tests/gdn.py:

  * `recurrent` : the definition, one token at a time (S = S g; S += (v - S^T k) beta k^T ... as HF's torch_recurrent_gated_delta_rule).
  * `chunk_fwd` : the chunked ("WY") form HF uses in torch_chunk_gated_delta_rule, chunk size C = 64, with the triangular solve (I + Ls)^-1 of a
                  chunk done three ways (`solve=`): forward substitution (HF's loop), the exact inverse, and repeated squaring of the nilpotent
                  matrix (six matrix products at C = 64: the form a matrix unit can run).
  * `chunk_bwd` : the hand-derived backward of `chunk_fwd` (reverse over chunks, carrying dS), and `layer_fwd` / `layer_bwd` : a whole
                  Qwen3_5GatedDeltaNet layer (projections, the causal convolution and silu, l2norm, gates, the rule, the gated norm, out_proj)
                  with every gradient. Checked against finite differences (numpy only) and against torch autograd through HF's code in
                  binary64 (examples/qwen35/hf_check.py; needs torch and transformers).

Shapes: one sequence, [T, heads, dim] for the layer; [H, T, dim] inside the rule. `mut=` names a deliberate fault (tests/gdn.py requires every
one to be caught); None is the specification."""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def l2norm_fwd(x, eps=1e-6):
    r = 1.0 / np.sqrt((x * x).sum(-1, keepdims=True) + eps)
    return x * r, r


def l2norm_bwd(x, r, dy):
    return r * dy - x * r**3 * (x * dy).sum(-1, keepdims=True)


def recurrent(q, k, v, g, beta, S0=None):
    """HF's recurrent form. q, k: [H, T, dk] (q NOT yet scaled), v: [H, T, dv], g, beta: [H, T]. Returns o [H, T, dv], S [H, dk, dv]."""
    H, T, dk = k.shape
    dv = v.shape[-1]
    S = np.zeros((H, dk, dv)) if S0 is None else S0.copy()
    o = np.zeros((H, T, dv))
    q = q * dk**-0.5
    for t in range(T):
        S = S * np.exp(g[:, t])[:, None, None]
        kv = np.einsum("hkv,hk->hv", S, k[:, t])
        delta = (v[:, t] - kv) * beta[:, t, None]
        S = S + k[:, t, :, None] * delta[:, None, :]
        o[:, t] = np.einsum("hkv,hk->hv", S, q[:, t])
    return o, S


def solve_unit_lower(Ls, how, mut=None):
    """(I + Ls)^-1 for strictly lower Ls [..., C, C]."""
    C = Ls.shape[-1]
    I = np.eye(C)
    if how == "inv":
        return np.linalg.inv(I + Ls)
    if how == "loop":  # HF: row i of (I + Ls)^-1 - I by forward substitution
        a = -Ls.copy()
        for i in range(1, C):
            row = a[..., i, :i].copy()
            sub = a[..., :i, :i].copy()
            a[..., i, :i] = row + (row[..., :, None] * sub).sum(-2)
        return a + I
    if how == "neumann":  # (I + N)^-1 = (I - N)(I + N^2)(I + N^4)... , N^C = 0
        N = Ls
        out = I - N
        p = N
        steps = 1
        while 2**steps < C:
            p = p @ p
            out = out @ (I + p)
            steps += 1
        return out
    raise ValueError(how)


def tril(x, k=0):
    C = x.shape[-1]
    return np.where(np.tril(np.ones((C, C), bool), k), x, 0.0)


def chunk_fwd(q, k, v, g, beta, C=64, solve="loop", S0=None, mut=None):
    """HF's chunked form. Returns o [H, T, dv], S_final, cache."""
    H, T, dk = k.shape
    dv = v.shape[-1]
    pad = (C - T % C) % C
    P = lambda x: np.concatenate([x, np.zeros(x.shape[:1] + (pad,) + x.shape[2:])], 1)
    scale = dk**-0.5
    qs = P(q) * scale
    kp, vp, gp, bp = P(k), P(v), P(g), P(beta)
    N = (T + pad) // C
    R = lambda x: x.reshape(H, N, C, *x.shape[2:])
    qs, kp, vp, gp, bp = R(qs), R(kp), R(vp), R(gp), R(bp)
    gc = np.cumsum(gp, -1)
    low = np.tril(np.ones((C, C), bool))
    diff = gc[..., :, None] - gc[..., None, :]
    D = np.where(low, np.exp(np.where(low, diff, 0.0)), 0.0)
    if mut == "no_decay":
        D = np.where(low, 1.0, 0.0)
    kb = kp * bp[..., None]
    vb = vp * bp[..., None]
    Ls = tril(kb @ kp.swapaxes(-1, -2) * D, -1)
    if mut == "solve_transposed":
        Ls = Ls.swapaxes(-1, -2)
        M = solve_unit_lower(Ls, solve, mut).swapaxes(-1, -2)
    else:
        M = solve_unit_lower(Ls, solve, mut)
    E = np.exp(gc)[..., None]
    u = M @ vb
    w = M @ (kb * E)
    S = np.zeros((H, dk, dv)) if S0 is None else S0.copy()
    out = np.zeros((H, N, C, dv))
    states = []
    for i in range(N):
        states.append(S.copy())
        Aqk = tril(qs[:, i] @ kp[:, i].swapaxes(-1, -2) * D[:, i])
        vnew = u[:, i] - w[:, i] @ S
        out[:, i] = (qs[:, i] * E[:, i]) @ S + Aqk @ vnew
        gl = gc[:, i, -1]
        Kd = kp[:, i] * np.exp(gl[:, None] - gc[:, i])[..., None]
        S = S * np.exp(gl)[:, None, None] + Kd.swapaxes(-1, -2) @ vnew
    cache = dict(qs=qs, k=kp, v=vp, beta=bp, gc=gc, D=D, M=M, u=u, w=w, states=states, C=C, N=N, T=T, scale=scale)
    return out.reshape(H, N * C, dv)[:, :T], S, cache


def chunk_bwd(cache, do, dSfinal=None, mut=None):
    """Gradients of chunk_fwd: dq, dk, dv, dg, dbeta ([H, T, *]) and dS0."""
    qs, k, v, beta, gc, D, M, u, w = (cache[n] for n in ("qs", "k", "v", "beta", "gc", "D", "M", "u", "w"))
    C, N, T, scale = cache["C"], cache["N"], cache["T"], cache["scale"]
    H, _, _, dk = k.shape
    dv = v.shape[-1]
    pad = N * C - T
    do = np.concatenate([do, np.zeros((H, pad, dv))], 1).reshape(H, N, C, dv)
    dq = np.zeros_like(qs)
    dkk = np.zeros_like(k)
    du = np.zeros_like(u)
    dw = np.zeros_like(w)
    dgc = np.zeros_like(gc)
    dD = np.zeros_like(D)
    dS = np.zeros((H, dk, dv)) if dSfinal is None else dSfinal.copy()
    E = np.exp(gc)[..., None]
    for i in reversed(range(N)):
        S = cache["states"][i]
        gl = gc[:, i, -1]
        eg = np.exp(gc[:, i])[..., None]
        el = np.exp(gl)[:, None, None]
        dec = np.exp(gl[:, None] - gc[:, i])[..., None]
        Kd = k[:, i] * dec
        vnew = u[:, i] - w[:, i] @ S
        qk = qs[:, i] @ k[:, i].swapaxes(-1, -2)
        Aqk = tril(qk * D[:, i])
        d_o = do[:, i]
        dvnew = Aqk.swapaxes(-1, -2) @ d_o + Kd @ dS
        dAqk = tril(d_o @ vnew.swapaxes(-1, -2))
        dqg = d_o @ S.swapaxes(-1, -2)
        dSin = dS * el + (qs[:, i] * eg).swapaxes(-1, -2) @ d_o - w[:, i].swapaxes(-1, -2) @ dvnew
        dw[:, i] = -dvnew @ S.swapaxes(-1, -2)
        du[:, i] = dvnew
        dKd = vnew @ dS.swapaxes(-1, -2)
        t_kd = (dKd * k[:, i]).sum(-1) * dec[..., 0]  # [H, C]
        dgc[:, i] += -t_kd
        dgc[:, i, -1] += t_kd.sum(-1) + (dS * S).sum((-1, -2)) * el[:, 0, 0]
        dkk[:, i] += dKd * dec
        dgc[:, i] += (dqg * qs[:, i]).sum(-1) * eg[..., 0]
        dq[:, i] += dqg * eg
        dQK = dAqk * D[:, i]
        dD[:, i] += dAqk * qk
        dq[:, i] += dQK @ k[:, i]
        dkk[:, i] += dQK.swapaxes(-1, -2) @ qs[:, i]
        dS = dSin
    kb = k * beta[..., None]
    vb = v * beta[..., None]
    kbE = kb * E
    dM = du @ vb.swapaxes(-1, -2) + dw @ kbE.swapaxes(-1, -2)
    dvb = M.swapaxes(-1, -2) @ du
    dkbE = M.swapaxes(-1, -2) @ dw
    dkb = dkbE * E
    dgc += (dkbE * kb).sum(-1) * E[..., 0]
    dLs = tril(-(M.swapaxes(-1, -2) @ dM @ M.swapaxes(-1, -2)), -1)
    if mut == "dls_unmasked":
        dLs = -(M.swapaxes(-1, -2) @ dM @ M.swapaxes(-1, -2))
    dKBD = dLs * D
    dkb += dKBD @ k
    dkk += dKBD.swapaxes(-1, -2) @ kb
    dD += dLs * (kb @ k.swapaxes(-1, -2))
    X = dD * D
    sgn = 1.0 if mut != "gate_sign" else -1.0
    dgc += X.sum(-1) - sgn * X.sum(-2)
    dbeta = (dkb * k).sum(-1) + (dvb * v).sum(-1)
    dkk += dkb * beta[..., None]
    dv = dvb * beta[..., None]
    dg = np.flip(np.cumsum(np.flip(dgc, -1), -1), -1)
    ro = lambda x: x.reshape(H, N * C, *x.shape[3:])[:, :T]
    return dict(dq=ro(dq) * scale, dk=ro(dkk), dv=ro(dv), dg=ro(dg.reshape(H, N, C, 1))[..., 0], dbeta=ro(dbeta.reshape(H, N, C, 1))[..., 0], dS0=dS)


def r16(x):
    """round to binary16 (what `stage f16` / an f16 tensor does to an mma operand)"""
    return x.astype(np.float16).astype(np.float64)


def mm16(a, b):
    """a matrix-unit product: f16 operands, f32 accumulation"""
    return (r16(a).astype(np.float32) @ r16(b).astype(np.float32)).astype(np.float64)


def chunk_fwd_mp(q, k, v, g, beta, C=64, solve="loop"):
    """The forward as the launches of examples/qwen35/probes/ would run it: every operand of a product rounded to f16, accumulation in f32, the
    solve by five squarings (neumann_mma_step), the scan by scan_mma_a / _bo / _bu, the decay mask and the rest elementwise in f32. Returns o [H, T, dv]
    (measured against chunk_fwd in tests/gdn.py and docs/roadmap-models.md M10)."""
    H, T, dk = k.shape
    dv = v.shape[-1]
    pad = (C - T % C) % C
    P = lambda x: np.concatenate([x, np.zeros(x.shape[:1] + (pad,) + x.shape[2:])], 1)
    qs, kp, vp, gp, bp = P(q) * dk**-0.5, P(k), P(v), P(g), P(beta)
    N = (T + pad) // C
    R = lambda x: x.reshape(H, N, C, *x.shape[2:])
    qs, kp, vp, gp, bp = R(qs), R(kp), R(vp), R(gp), R(bp)
    f32 = lambda x: x.astype(np.float32).astype(np.float64)
    gc = f32(np.cumsum(gp, -1))
    low = np.tril(np.ones((C, C), bool))
    D = np.where(low, f32(np.exp(np.minimum(gc[..., :, None] - gc[..., None, :], 0.0))), 0.0)
    kb, vb = kp * bp[..., None], vp * bp[..., None]
    Ls = tril(f32(mm16(kb, kp.swapaxes(-1, -2)) * D), -1)
    # (I + Ls)^-1 by forward substitution in f32 (solve_fwdsub.chx with nv = 64: HF's loop, stable). Repeated squaring is NOT usable: its powers of Ls grow
    # binomially (2e8 at key cosine 0.5) and cancel; tests/gdn.py measures it in f32 and f16
    if solve == "loop":
        M = solve_unit_lower(Ls.astype(np.float32).astype(np.float64), "loop")
    else:  # the squaring form on the matrix unit: Pm <- Pm Pm, Pi <- Pi + Pi Pm, all products on f16 operands (neumann_mma_step.chx)
        Pm, Pi, steps = Ls, np.eye(C) - Ls, 1
        while 2**steps < C:
            Pm = mm16(Pm, Pm)
            Pi = Pi + mm16(Pi, Pm)
            steps += 1
        M = Pi
    E = np.exp(gc)[..., None]
    u = mm16(M, vb)
    w = mm16(M, kb * E)
    S = np.zeros((H, dk, dv))
    out = np.zeros((H, N, C, dv))
    for i in range(N):
        Aqk = tril(f32(mm16(qs[:, i], kp[:, i].swapaxes(-1, -2)) * D[:, i]))
        vnew = u[:, i] - mm16(w[:, i], S)
        out[:, i] = mm16(qs[:, i] * E[:, i], S) + mm16(Aqk, vnew)
        gl = gc[:, i, -1]
        Kd = kp[:, i] * np.exp(gl[:, None] - gc[:, i])[..., None]
        S = f32(S * np.exp(gl)[:, None, None]) + mm16(Kd.swapaxes(-1, -2), vnew)
    return out.reshape(H, N * C, dv)[:, :T]


# ----------------------------------------------------------------------------------------------------------------------------------------
# one whole Qwen3_5GatedDeltaNet layer
# ----------------------------------------------------------------------------------------------------------------------------------------

def make_params(rng, hidden, nk, nv, dk, dv, kern=4, scale=0.3):
    key_dim, val_dim = nk * dk, nv * dv
    cd = 2 * key_dim + val_dim
    n = lambda *s: rng.normal(size=s) * scale
    return dict(
        Wqkv=n(cd, hidden), Wz=n(val_dim, hidden), Wb=n(nv, hidden), Wa=n(nv, hidden), conv=n(cd, kern),
        A_log=np.log(rng.uniform(0.5, 4.0, size=nv)), dt_bias=n(nv) + 0.5, norm=1.0 + 0.2 * n(dv), Wout=n(hidden, val_dim),
        dims=(nk, nv, dk, dv, kern),
    )


def layer_fwd(p, x, C=64, solve="loop", mut=None):
    nk, nv, dk, dv, kern = p["dims"]
    T = x.shape[0]
    mixed = x @ p["Wqkv"].T
    z = (x @ p["Wz"].T).reshape(T, nv, dv)
    b = x @ p["Wb"].T
    a = x @ p["Wa"].T
    pad = np.concatenate([np.zeros((kern - 1, mixed.shape[1])), mixed], 0)
    pc = sum(p["conv"][:, j][None, :] * pad[j:j + T] for j in range(kern))
    act = silu(pc)
    kd, vd = nk * dk, nv * dv
    q0, k0, v0 = act[:, :kd].reshape(T, nk, dk), act[:, kd:2 * kd].reshape(T, nk, dk), act[:, 2 * kd:].reshape(T, nv, dv)
    qn, rq = l2norm_fwd(q0)
    kn, rk = l2norm_fwd(k0)
    rep = nv // nk
    qh, kh = np.repeat(qn, rep, 1), np.repeat(kn, rep, 1)
    beta = sigmoid(b)
    sp = softplus(a + p["dt_bias"])
    g = -np.exp(p["A_log"]) * sp
    tr = lambda t: np.ascontiguousarray(t.transpose(1, 0, 2)) if t.ndim == 3 else np.ascontiguousarray(t.T)
    o, S, cache = chunk_fwd(tr(qh), tr(kh), tr(v0), tr(g), tr(beta), C, solve, mut=mut)
    o = o.transpose(1, 0, 2)  # [T, nv, dv]
    r = 1.0 / np.sqrt((o * o).mean(-1, keepdims=True) + 1e-6)
    n = o * r
    sz = silu(z)
    h = p["norm"] * n * sz
    y = h.reshape(T, -1) @ p["Wout"].T
    saved = dict(x=x, z=z, b=b, a=a, mixed=mixed, pad=pad, pc=pc, q0=q0, k0=k0, v0=v0, rq=rq, rk=rk, qn=qn, kn=kn, beta=beta, sp=sp,
                 g=g, o=o, r=r, n=n, sz=sz, h=h, cache=cache, dims=p["dims"])
    return y, saved


def layer_bwd(p, s, dy, mut=None):
    nk, nv, dk, dv, kern = p["dims"]
    x = s["x"]
    T = x.shape[0]
    rep = nv // nk
    dh = (dy @ p["Wout"]).reshape(T, nv, dv)
    gr = {"Wout": dy.T @ s["h"].reshape(T, -1)}
    z = s["z"]
    sg = sigmoid(z)
    dsz = dh * p["norm"] * s["n"]
    dz = dsz * (sg * (1 + z * (1 - sg)))
    gr["norm"] = (dh * s["sz"] * s["n"]).sum((0, 1))
    dn = dh * p["norm"] * s["sz"]
    do = s["r"] * dn - s["o"] * s["r"] ** 3 * (s["o"] * dn).mean(-1, keepdims=True)
    tr = lambda t: np.ascontiguousarray(t.transpose(1, 0, 2)) if t.ndim == 3 else np.ascontiguousarray(t.T)
    cg = chunk_bwd(s["cache"], tr(do), mut=mut)
    back = lambda t: t.transpose(1, 0, 2)
    dqh, dkh, dv0 = back(cg["dq"]), back(cg["dk"]), back(cg["dv"])
    dg, dbeta = cg["dg"].T, cg["dbeta"].T
    dqn = dqh.reshape(T, nk, rep, dk).sum(2)
    dkn = dkh.reshape(T, nk, rep, dk).sum(2)
    dq0 = l2norm_bwd(s["q0"], s["rq"], dqn)
    dk0 = l2norm_bwd(s["k0"], s["rk"], dkn)
    dact = np.concatenate([dq0.reshape(T, -1), dk0.reshape(T, -1), dv0.reshape(T, -1)], 1)
    pc = s["pc"]
    sc = sigmoid(pc)
    dpc = dact * (sc * (1 + pc * (1 - sc)))
    gr["conv"] = np.stack([(dpc * s["pad"][j:j + T]).sum(0) for j in range(kern)], 1)
    dmixed = np.zeros_like(s["mixed"])
    for j in range(kern):
        # pc[t] += conv[:, j] * pad[t + j], pad[i] = mixed[i - (kern - 1)]
        lo = kern - 1 - j
        dmixed[: T - lo] += p["conv"][:, j][None, :] * dpc[lo:]
    b, a = s["b"], s["a"]
    db = dbeta * s["beta"] * (1 - s["beta"])
    E = np.exp(p["A_log"])
    da = dg * (-E) * sigmoid(a + p["dt_bias"])
    gr["dt_bias"] = da.sum(0)
    gr["A_log"] = (dg * (-E) * s["sp"]).sum(0)
    gr["Wqkv"] = dmixed.T @ x
    gr["Wz"] = dz.reshape(T, -1).T @ x
    gr["Wb"] = db.T @ x
    gr["Wa"] = da.T @ x
    dx = dmixed @ p["Wqkv"] + dz.reshape(T, -1) @ p["Wz"] + db @ p["Wb"] + da @ p["Wa"]
    return dx, gr
