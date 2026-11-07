"""The G6 optimiser path of the GPT session (docs/optimizer.md section 4): gradient accumulation, global-norm clipping, a decay group and a schedule per step, in place of
`adamw_h` / `adam` of gpt_session.write. `write` appends the directives; gpt_session calls it when given `opt=`."""
import os

import numpy as np

import schedule as S

HERE = os.path.dirname(os.path.abspath(__file__))


def block(n, cap=128):
    return next(b for b in range(min(n, cap), 0, -1) if n % b == 0)


def wire(L, p, mats, vecs, sizes, grad_of, f16_users, f32_users, gs, b1, b2, eps, steps, lr_fn, opt):
    """`opt`: accum (micro-steps a step), max_norm (None for no clip), wd (decay of the matrices; vectors are not decayed, as torch's usual groups)."""
    accum, max_norm, wd = opt.get("accum", 1), opt.get("max_norm"), opt.get("wd", 0.0)
    reps = steps * accum
    names = mats + vecs
    prog = lambda name, kern, **kw: L.append(f"program {name} {os.path.join(HERE, kern)} " + " ".join(f"{k}={v}" for k, v in kw.items()))

    def put(path, values):
        np.asarray(values, dtype="<f4").reshape(-1).tofile(path)

    put(p("o_keep"), S.keep(steps, accum))
    put(p("o_zero1"), [0.0])
    put(p("o_sched_m"), S.rows(steps, accum, lr_fn, wd, b1, b2))
    put(p("o_sched_v"), S.rows(steps, accum, lr_fn, 0.0, b1, b2))
    for name in names:
        n = sizes[name]
        prog("acc_" + name, "gacc.chx", n=n, bn=block(n), sc=repr(1.0 / accum))
        L.append(f"resident acc_{name}.acc {p('z' + name)} f32")
        L.append(f"table acc_{name}.k {p('o_keep')} {reps} f32")
        L.append(f"link {grad_of[name]} -> acc_{name}.grad")
        L.append(f"carry acc_{name}.o -> acc_{name}.acc")
    prev = None
    for name in names:
        prog("sq_" + name, "gradsq.chx", n=sizes[name], bn=block(sizes[name]), gs=repr(gs))
        L.append(f"resident sq_{name}.acc {p('o_zero1')} f32" if prev is None else f"link sq_{prev}.o -> sq_{name}.acc")
        L.append(f"link acc_{name}.o -> sq_{name}.grad")
        prev = name
    prog("clip", "clip_scale.chx", maxn=repr(float(max_norm)) if max_norm is not None else "1.0e30")
    L.append(f"link sq_{prev}.o -> clip.ss")
    for name in names:
        prog("a_" + name, "adamw_gh.chx" if name in mats else "adamw_g.chx", n=sizes[name], bn=block(sizes[name]), gs=repr(gs), b1=repr(b1), ob1=repr(1.0 - b1), b2=repr(b2), ob2=repr(1.0 - b2), eps=repr(eps))
        L.append(f"resident a_{name}.w {p(name)} f32")
        L.append(f"resident a_{name}.m {p('z' + name)} f32")
        L.append(f"resident a_{name}.v {p('z' + name)} f32")
        L.append(f"table a_{name}.s {p('o_sched_m' if name in mats else 'o_sched_v')} {reps} f32")
        L.append(f"link acc_{name}.o -> a_{name}.grad")
        L.append(f"link clip.cs -> a_{name}.cs")
        for x in "wmv":
            L.append(f"carry a_{name}.{x}o -> a_{name}.{x}")
    for name in mats:
        for u in f16_users[name]:
            L.append(f"carry a_{name}.wh -> {u}.w")
    for name in vecs:
        for u in f32_users[name]:
            L.append(f"carry a_{name}.wo -> {u}")
