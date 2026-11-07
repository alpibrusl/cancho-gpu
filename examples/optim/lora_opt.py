"""The G6 optimiser path of the Qwen LoRA session (docs/optimizer.md section 4): gradient accumulation, global-norm clipping, decay following the schedule and a table row per micro-step,
in place of `adamw_l` of qwen_train.write. `write` appends the directives; qwen_train calls it when given `opt=`. The adapters are the matrices of one decay group (`wd` is the decay of
both A and B: a LoRA's weight decay applies to every adapter matrix, as peft's usual AdamW over `lora_*` parameters does).

The fused B of a group has off-diagonal blocks that are not parameters (`mB`, 0 there): the mask multiplies the gradient as it is accumulated (gacc_m), so the norm and the moments never see it."""
import os

import numpy as np

import schedule as S

HERE = os.path.dirname(os.path.abspath(__file__))


def block(n, cap=128):
    return next(b for b in range(min(n, cap), 0, -1) if n % b == 0)


def wire(L, p, res_names, gs, scale, b1, b2, eps, steps, lr_fn, first_step, opt, link_of, f16_users):
    """`res_names`: [(name, size, tag)] for every adapter matrix (name `aA00_qkv`, tag the file suffix `A00_qkv`, ab 'A' or 'B' is name[1]); `link_of[name]`: the program output holding its raw gradient;
    `f16_users[name]`: where its f16 copy is carried. `opt`: accum, max_norm (None: no clip), wd. `scale`: alpha / r, on the B matrices' gradient and f16 copy."""
    accum, max_norm, wd = opt.get("accum", 1), opt.get("max_norm"), opt.get("wd", 0.0)
    reps = steps * accum
    prog = lambda name, kern, **kw: L.append(f"program {name} {os.path.join(HERE, kern)} " + " ".join(f"{k}={v}" for k, v in kw.items()))
    np.asarray(S.keep(steps, accum), dtype="<f4").tofile(p("o_keep"))
    np.asarray([0.0], dtype="<f4").tofile(p("o_zero1"))
    np.asarray(S.rows(steps, accum, lr_fn, wd, b1, b2, first_step), dtype="<f4").reshape(-1).tofile(p("o_sched"))
    gsof = lambda nm: gs * (scale if nm[1] == "B" else 1.0)
    for nm, n, tag in res_names:
        np.zeros(n, dtype="<f4").tofile(p("o_z" + tag))
        prog("acc_" + nm, "gacc_m.chx", n=n, bn=block(n), sc=repr(1.0 / accum))
        L.append(f"resident acc_{nm}.acc {p('o_z' + tag)} f32")
        L.append(f"resident acc_{nm}.mk {p('m' + tag)} f32")
        L.append(f"table acc_{nm}.k {p('o_keep')} {reps} f32")
        L.append(f"link {link_of[nm]} -> acc_{nm}.grad")
        L.append(f"carry acc_{nm}.o -> acc_{nm}.acc")
    prev = None
    for nm, n, tag in res_names:
        prog("sq_" + nm, "gradsq.chx", n=n, bn=block(n), gs=repr(gsof(nm)))
        L.append(f"resident sq_{nm}.acc {p('o_zero1')} f32" if prev is None else f"link sq_{prev}.o -> sq_{nm}.acc")
        L.append(f"link acc_{nm}.o -> sq_{nm}.grad")
        prev = nm
    prog("clip", "clip_scale.chx", maxn=repr(float(max_norm)) if max_norm is not None else "1.0e30")
    L.append(f"link sq_{prev}.o -> clip.ss")
    for nm, n, tag in res_names:
        prog(nm, "adamw_gl.chx", n=n, bn=block(n), gs=repr(gsof(nm)), b1=repr(b1), ob1=repr(1.0 - b1), b2=repr(b2), ob2=repr(1.0 - b2), eps=repr(eps), osc=repr(scale if nm[1] == "B" else 1.0))
        L.append(f"resident {nm}.w {p(tag)} f32")
        L.append(f"resident {nm}.m {p('zm' + tag)} f32")
        L.append(f"resident {nm}.v {p('zv' + tag)} f32")
        L.append(f"table {nm}.s {p('o_sched')} {reps} f32")
        L.append(f"link acc_{nm}.o -> {nm}.grad")
        L.append(f"link clip.cs -> {nm}.cs")
        for x in "wmv":
            L.append(f"carry {nm}.{x}o -> {nm}.{x}")
        for u in f16_users[nm]:
            L.append(f"carry {nm}.wh -> {u}")
