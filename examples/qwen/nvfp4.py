"""NVFP4 as `dequant_fp4` reads it (docs/resident.md sections 25 and 26): the encoder, the packing into the three files of a quantised matrix, and the decoder.

A matrix w [n, k] becomes `q` (uint8 [n, k / 2]: two E2M1 codes a byte, the low nibble the first of the pair), `s` (uint8 [n, k / G]: an E4M3 scale per G values) and `g` (f32 [n]: a scale per row); the value
of w[i, j] is e2m1(code) * e4m3(scale of its group) * g[i]. numpy, binary64. The encoder works on blocks of rows (the row scale is per row, so a block is exact) so that a 7B model's biggest matrix
(37,888 x 3,584) is not held in binary64 several times over."""
import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # codes 0..7; code + 8 is the negative (the interpreter's FP4_V)


def e4m3_table():
    """All 126 non-negative finite OCP e4m3fn values with their byte (exponent 4 bits, bias 7, mantissa 3 bits, subnormals at exponent 0, 0x7F is NaN)."""
    vals = []
    for b in range(0x7F):
        e, m = (b >> 3) & 15, b & 7
        vals.append(m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
    return np.array(vals)


E4M3 = e4m3_table()


def round_e4m3(x):
    """The byte (0..126) of the e4m3 value nearest to x >= 0 (ties to the lower one, which can differ from ties-to-even only on exact midpoints) and the value."""
    i = np.clip(np.searchsorted(E4M3, x), 1, len(E4M3) - 1)
    lo, hi = E4M3[i - 1], E4M3[i]
    i = np.where(x - lo <= hi - x, i - 1, i)
    return i, E4M3[i]


def round_codebook(x, book):
    """The index of the nearest entry of the sorted `book` for every element of x."""
    mid = (book[1:] + book[:-1]) / 2
    return np.searchsorted(mid, x)



def groups(w, g):
    n, k = w.shape
    assert k % g == 0, (k, g)
    return w.reshape(n, k // g, g)


def nvfp4(w, g=16):
    """(dequantised, codes [n, k] in 0..15, scale bytes [n, k/g], row scale [n]) in the format of `dequant_fp4`: e2m1(code) * e4m3(scale) * row. The row scale is amax / (6 * 448), so that the
    largest group's scale is at most 448; a group's scale is its amax / 6 over the row scale, rounded to e4m3."""
    gw = groups(w, g)
    row = np.abs(w).max(1) / (6 * 448)
    row = np.where(row == 0, 1.0, row).astype(np.float32).astype(np.float64)
    amax = np.abs(gw).max(2)
    sb, sv = round_e4m3(amax / 6 / row[:, None])
    eff = sv * row[:, None]
    z = np.where(eff[:, :, None] > 0, gw / np.where(eff > 0, eff, 1)[:, :, None], 0.0)
    mag = round_codebook(np.abs(z), E2M1)
    code = mag + 8 * (z < 0)
    deq = np.sign(z) * E2M1[mag] * eff[:, :, None]
    return deq.reshape(w.shape), code.reshape(w.shape), sb, row


def pack_nvfp4(w, g=16, swap=False):
    """The three arrays `dequant_fp4` reads, as the interpreter's text files want them: q i8[n, k/2] (two codes a byte, the low nibble the first of the pair), s i8[n, k/g], row f32[n]; and the
    dequantised matrix the packing stands for."""
    deq, code, sb, row = nvfp4(w, g)
    byte = (code[:, 1::2] | (code[:, 0::2] << 4)) if swap else (code[:, 0::2] | (code[:, 1::2] << 4))  # swap: the nibbles in the wrong order (the mutant of `validate`)
    return byte.astype(np.uint8).astype(np.int8), sb.astype(np.uint8).astype(np.int8), row, deq



def unpack_rows(q, s, g, G):
    """The inverse of `pack_nvfp4`, from the bytes (not from the encoder's own dequantisation): q uint8 [n, k/2], s uint8 [n, k/G], g f32 [n] -> float64 [n, k]. A scale byte of 0x7F or 0xFF (NaN in
    e4m3) is not written by the encoder and is refused."""
    n, kh = q.shape
    k = 2 * kh
    assert s.shape == (n, k // G) and g.shape == (n,), (q.shape, s.shape, g.shape, G)
    sb = s.astype(np.int64)
    if ((sb & 0x7F) == 0x7F).any():
        raise ValueError("an e4m3 scale byte is NaN")
    sign = np.where(sb >= 128, -1.0, 1.0)
    scale = sign * E4M3[sb & 0x7F] * g.astype(np.float64)[:, None]
    codes = np.empty((n, k), dtype=np.int64)
    codes[:, 0::2] = q & 15
    codes[:, 1::2] = q >> 4
    mag = E2M1[codes & 7] * np.where(codes >= 8, -1.0, 1.0)
    return (mag.reshape(n, k // G, G) * scale[:, :, None]).reshape(n, k)


def pack_rows(w, G=16, block=2048):
    """(q, s, g) of the matrix w [n, k] (any float array), encoded `block` rows at a time. q and s are uint8, g is float32."""
    n, k = w.shape
    q = np.empty((n, k // 2), dtype=np.uint8)
    s = np.empty((n, k // G), dtype=np.uint8)
    g = np.empty(n, dtype=np.float32)
    for i in range(0, n, block):
        qi, si, gi, _ = pack_nvfp4(np.asarray(w[i:i + block], dtype=np.float64), G)
        q[i:i + block], s[i:i + block], g[i:i + block] = qi.view(np.uint8), si.view(np.uint8), gi
    return q, s, g
