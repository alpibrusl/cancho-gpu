"""The learning-rate schedule as a construct (docs/optimizer.md section 2), and the table rows the AdamW kernels read.

    lr = warmup_cosine(peak=1e-3, warmup=50, total=1000, floor=1e-4)
    lr(t)                       # t = 1 .. total, the rate of optimiser step t

Equal to `SequentialLR([LinearLR, CosineAnnealingLR], milestones=[warmup])` of torch (tests/optim.py holds it to the real schedulers):
step t <= warmup gives peak t / warmup; step t > warmup gives floor + (peak - floor)(1 + cos(pi (t - warmup - 1) / (total - warmup))) / 2.
"""
import math


def warmup_cosine(peak, warmup, total, floor=0.0):
    assert 1 <= warmup < total and 0 <= floor <= peak

    def lr(t):
        if t <= warmup:
            return peak * t / warmup
        return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * (t - warmup - 1) / (total - warmup)))

    return lr


def constant(peak):
    return lambda t: peak


def rows(steps, accum, lr, wd, b1=0.9, b2=0.999):
    """The Adam table of one parameter group: a row [rate, isq, dec, apply, 1 - apply] for each of `steps * accum` repetitions (binary64 values).
    Repetition j belongs to optimiser step j // accum + 1 and applies the update when it is the last of its window."""
    out = []
    for j in range(steps * accum):
        t = j // accum + 1
        a = 1.0 if j % accum == accum - 1 else 0.0
        out.append([lr(t) / (1.0 - b1 ** t), 1.0 / math.sqrt(1.0 - b2 ** t), 1.0 - lr(t) * wd, a, 1.0 - a])
    return out


def keep(steps, accum):
    """The accumulator's table: 0 on the first repetition of a window, 1 after."""
    return [0.0 if j % accum == 0 else 1.0 for j in range(steps * accum)]
