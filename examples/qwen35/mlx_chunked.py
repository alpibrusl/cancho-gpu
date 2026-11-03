"""The chunked (WY) gated delta rule written in MLX ops, a port of transformers' torch_chunk_gated_delta_rule (examples/qwen35/hf_check.py holds gdn_ref.py to the original):
what an MLX user would write to get a differentiable, chunk-parallel layer WITHOUT a custom kernel. The 64 x 64 unit-lower solve is mx.linalg.inv (CPU stream; MLX 0.31.2 has no gradient for it, so the
gradient -M^T dM M^T is attached with mx.custom_function); the sequence is cut into chunks of 64 and only the 8 (at T = 512) chunk steps are a Python loop. q is pre-scaled by mlx-lm's own rms_norm
form (qwen3_5.GatedDeltaNet), so no 1/sqrt(Dk) here. Used by mlx_tiny.py --chunked; its equality with mlx-lm's `gated_delta_ops` is checked in main()."""
import mlx.core as mx
import mlx.nn as nn


@mx.custom_function
def _inv(A):
    return mx.linalg.inv(A, stream=mx.cpu)


@_inv.vjp
def _inv_vjp(primals, cotangent, output):
    # mx.linalg.inv has no gradient in mlx 0.31.2 ("[Primitive::vjp] Not implemented for Inverse"): d(A^-1) = -A^-1 dA A^-1, so dA = -M^T dM M^T
    Mt = output.swapaxes(-1, -2)
    return -(Mt @ cotangent @ Mt)


def chunked_gated_delta(q, k, v, g, beta, C=64):
    """q, k: [B, T, Hk, Dk] (q pre-scaled), v: [B, T, Hv, Dv], g: [B, T, Hv] the DECAY exp(log-decay), beta: [B, T, Hv]. Returns y [B, T, Hv, Dv]."""
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if Hv // Hk > 1:
        q = mx.repeat(q, Hv // Hk, -2)
        k = mx.repeat(k, Hv // Hk, -2)
    q, k, v = [x.transpose(0, 2, 1, 3).astype(mx.float32) for x in (q, k, v)]  # [B, H, T, D]
    beta = beta.transpose(0, 2, 1).astype(mx.float32)
    gl = mx.log(g.transpose(0, 2, 1).astype(mx.float32))  # [B, H, T]
    pad = (C - T % C) % C
    if pad:
        q, k, v = [mx.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)]) for x in (q, k, v)]
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad)])
        gl = mx.pad(gl, [(0, 0), (0, 0), (0, pad)])
    N = (T + pad) // C
    R = lambda x: x.reshape(B, Hv, N, C, *x.shape[3:])
    q, k, v = R(q), R(k), R(v)
    beta, gl = beta.reshape(B, Hv, N, C), gl.reshape(B, Hv, N, C)
    gc = mx.cumsum(gl, axis=-1)
    low = mx.tril(mx.ones((C, C), dtype=mx.bool_))
    D = mx.where(low, mx.exp(mx.where(low, gc[..., :, None] - gc[..., None, :], 0.0)), 0.0)
    kb, vb = k * beta[..., None], v * beta[..., None]
    Ls = mx.tril(kb @ k.swapaxes(-1, -2) * D, k=-1)
    M = _inv(mx.eye(C) + Ls)
    u = M @ vb
    w = M @ (kb * mx.exp(gc)[..., None])
    S = mx.zeros((B, Hv, Dk, Dv))
    outs = []
    for i in range(N):
        qi, ki = q[:, :, i], k[:, :, i]
        Aqk = mx.tril(qi @ ki.swapaxes(-1, -2) * D[:, :, i])
        vnew = u[:, :, i] - w[:, :, i] @ S
        outs.append((qi * mx.exp(gc[:, :, i])[..., None]) @ S + Aqk @ vnew)
        gend = gc[:, :, i, -1]
        S = S * mx.exp(gend)[..., None, None] + (ki * mx.exp(gend[..., None] - gc[:, :, i])[..., None]).swapaxes(-1, -2) @ vnew
    y = mx.stack(outs, axis=2).reshape(B, Hv, N * C, Dv)[:, :, :T]
    return y.transpose(0, 2, 1, 3), S


def gated_delta_update_chunked(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
    """Drop-in for mlx_lm.models.gated_delta.gated_delta_update (training only: no state in, none used)."""
    from mlx_lm.models.gated_delta import compute_g
    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    y, S = chunked_gated_delta(q, k, v, g, beta)
    return y.astype(q.dtype), S.transpose(0, 1, 3, 2)


def main():
    from mlx_lm.models.gated_delta import gated_delta_ops, compute_g
    mx.random.seed(0)
    B, T, Hk, Hv, Dk, Dv = 1, 150, 4, 12, 32, 32
    q = mx.random.normal((B, T, Hk, Dk)); k = mx.random.normal((B, T, Hk, Dk)); v = mx.random.normal((B, T, Hv, Dv))
    q = q / mx.sqrt((q * q).sum(-1, keepdims=True)) / Dk**0.5
    k = k / mx.sqrt((k * k).sum(-1, keepdims=True))
    a = mx.random.normal((B, T, Hv)) * 0.5; b = mx.random.normal((B, T, Hv))
    A_log = mx.log(mx.random.uniform(0.5, 4.0, (Hv,))); dt_bias = mx.random.normal((Hv,)) * 0.3
    y0, s0 = gated_delta_ops(q, k, v, compute_g(A_log, a, dt_bias), mx.sigmoid(b))
    y1, s1 = gated_delta_update_chunked(q, k, v, a, b, A_log, dt_bias)
    print("chunked MLX port vs mlx-lm gated_delta_ops: max |dy|", float(mx.abs(y0 - y1).max()), "max |dS|", float(mx.abs(s0 - s1).max()))
    # gradients agree too
    # and through the key (the solve's gradient)
    fk0 = lambda k_: (gated_delta_ops(q, k_, v, compute_g(A_log, a, dt_bias), mx.sigmoid(b))[0] ** 2).sum()
    fk1 = lambda k_: (gated_delta_update_chunked(q, k_, v, a, b, A_log, dt_bias)[0] ** 2).sum()
    gk0, gk1 = mx.grad(fk0)(k), mx.grad(fk1)(k)
    print("gradient wrt k (through the solve): max |d|", float(mx.abs(gk0 - gk1).max()), "scale", float(mx.abs(gk0).max()))
    f0 = lambda q_, v_: (gated_delta_ops(q_, k, v_, compute_g(A_log, a, dt_bias), mx.sigmoid(b))[0] ** 2).sum()
    f1 = lambda q_, v_: (gated_delta_update_chunked(q_, k, v_, a, b, A_log, dt_bias)[0] ** 2).sum()
    g0 = mx.grad(f0, argnums=(0, 1))(q, v); g1 = mx.grad(f1, argnums=(0, 1))(q, v)
    print("gradients wrt q, v: max |d|", float(mx.abs(g0[0] - g1[0]).max()), float(mx.abs(g0[1] - g1[1]).max()), "scale", float(mx.abs(g0[1]).max()))


if __name__ == "__main__":
    main()
