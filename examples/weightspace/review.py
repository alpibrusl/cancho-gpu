#!/usr/bin/env python3
"""The review checklist of docs/weight-space.md section 4: a static check over the code a run logged, for the ways a weight can be *found* instead of *written*.

    python3 examples/weightspace/review.py RUN_DIR [--json]        exit 1 if any write is VIOLATING, 0 otherwise (borderline cases are printed, never dropped)

The rule (fixed in the design): a run VIOLATES if any parameter is set by solving or searching for it against computed outputs of the target (`lstsq`, `scipy.optimize`, autograd, a loop that adjusts a
parameter to reduce a computed error). A closed formula from the structure of the problem, or a function tabulated at fixed points, is allowed. Three verdicts per piece of code:
  violating   a pattern that is the rule's own examples: a least-squares / optimiser / autograd call or import, or a loop that updates a variable by a gradient-like step or keeps the best of random trials;
  borderline  a pattern that is legitimate in closed-form mathematics and is also how a fit is written (`pinv`, `solve`, `inv`, `svd`, `eig`, `qr`, `cholesky`, an iterate-until-converged loop, sampling
              together with linear algebra, dynamic code that cannot be read statically, reading data files): reported with its line, for a person to decide;
  clean       none of the above.
THIS IS A HEURISTIC, NOT A PROOF. It reads the code, it does not run it: a fit written with names it does not know, built from the primitives it does not list, or from code assembled at run time, passes as
clean (it flags `eval`/`exec`/`getattr` with a computed name/`__import__` as borderline for that reason). It is the first of two filters: its output is read by a person, who has the code.
`disabled` (a set of rule names) exists so the tests can show that each rule is what catches its planted violator."""
import ast
import json
import os
import re
import sys

FIT_CALLS = {"lstsq", "lsq_linear", "least_squares", "leastsq", "curve_fit", "minimize", "minimize_scalar", "fmin", "fmin_bfgs", "fmin_l_bfgs_b", "fmin_powell", "fmin_cg", "nnls",
             "polyfit", "basinhopping", "differential_evolution", "dual_annealing", "shgo", "brute", "root", "fsolve", "newton_krylov", "backward", "value_and_grad", "grad", "jacrev", "jacfwd", "fit"}
LINALG_CALLS = {"pinv", "solve", "inv", "svd", "eig", "eigh", "qr", "cholesky", "lu_solve", "matrix_power", "cho_solve", "solve_triangular"}
FIT_MODULES = {"scipy.optimize", "torch", "jax", "autograd", "tensorflow", "sklearn", "cvxpy", "optax", "scipy.linalg.lstsq"}
DYNAMIC = {"eval", "exec", "compile", "__import__", "importlib", "getattr", "setattr", "globals", "locals", "vars"}
SAMPLING = {"uniform", "normal", "randn", "rand", "random", "default_rng", "RandomState", "meshgrid", "linspace", "choice", "randint", "standard_normal", "mgrid"}
UPDATE_NAMES = re.compile(r"(grad|learning|\blr\b|^lr$|_lr$|momentum|adam|loss|resid|err|step_size|stepsize|delta_w|dw\b)", re.I)
FORWARD_NAMES = re.compile(r"(fk|forward|kinemat|predict|evaluate|net|model|score|error|err|loss|cost|objective)", re.I)
RULES = ("fit-call", "fit-import", "update-loop", "random-search", "linalg", "iterate", "sampling+linalg", "dynamic", "file-read")


def _tail(node):
    while isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else None


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}


def review_source(source, disabled=()):
    """[(verdict, rule, line, note)] for one program. A program that does not parse is borderline (nothing could be checked)."""
    out = []
    add = lambda verdict, rule, node, note: out.append((verdict, rule, getattr(node, "lineno", 0), note)) if rule not in disabled else None
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [("borderline", "unparsable", e.lineno or 0, "the code does not parse, so it was not checked: %s" % e.msg)]
    sampling = linalg_nodes = None
    sampling_nodes, linalg_nodes = [], []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [(node.module or "") + "." + a.name for a in node.names] + [node.module or ""]
            for m in mods:
                if any(m == f or m.startswith(f + ".") for f in FIT_MODULES):
                    add("violating", "fit-import", node, "imports %s (an optimiser, autograd or learning library)" % m)
        elif isinstance(node, ast.Call):
            t = _tail(node.func)
            if t in FIT_CALLS:
                add("violating", "fit-call", node, "calls %s(...), a fitting / optimising / differentiating routine" % t)
            elif t in LINALG_CALLS:
                add("borderline", "linalg", node, "calls %s(...): fine for a closed form (inverting a rotation), a fit if applied to computed target outputs" % t)
                linalg_nodes.append(node)
            elif t in DYNAMIC and not (t == "getattr" and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)):
                add("borderline", "dynamic", node, "%s(...) builds code or names at run time, which this check cannot read" % t)
            elif t == "open" or (t in ("load", "loadtxt", "fromfile", "genfromtxt", "read_csv", "memmap") and "np" in _names(node.func) | {"np"}):
                add("borderline", "file-read", node, "reads a file (%s): the sandbox refuses anything outside its scratch directory, but the attempt is recorded" % t)
            if t in SAMPLING:
                sampling_nodes.append(node)
        elif isinstance(node, (ast.For, ast.While)):
            _review_loop(node, add)
    if sampling_nodes and linalg_nodes:
        add("borderline", "sampling+linalg", linalg_nodes[0], "the program draws samples (line %d) and applies linear algebra: the shape of a regression" % sampling_nodes[0].lineno)
    return out


def _assigned(node):
    """Names a statement list updates in place (`x -= ..`, `x[i] += ..`, `x = x - ..`) with the node of the update."""
    res = []
    for n in ast.walk(node):
        if isinstance(n, ast.AugAssign) and isinstance(n.op, (ast.Sub, ast.Add)):
            tgt = n.target.value if isinstance(n.target, ast.Subscript) else n.target
            if isinstance(tgt, ast.Name):
                res.append((tgt.id, n, _names(n.value)))
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, (ast.Sub, ast.Add)):
            t = n.targets[0]
            tgt = t.value if isinstance(t, ast.Subscript) else t
            if isinstance(tgt, ast.Name) and tgt.id in _names(n.value.left):
                res.append((tgt.id, n, _names(n.value.right)))
    return res


def _review_loop(loop, add):
    body = ast.Module(body=loop.body, type_ignores=[])
    for name, upd, used in _assigned(body):
        if UPDATE_NAMES.search(name) is None and any(UPDATE_NAMES.search(u) for u in used if u != name):
            add("violating", "update-loop", upd, "a loop updates `%s` by a step built from %s: a parameter adjusted by a gradient-like rule" % (name, ", ".join(sorted(u for u in used if UPDATE_NAMES.search(u)))))
            return
    # keep-the-best / hill climbing: a loop that evaluates a forward/error, compares, and keeps a candidate
    names = _names(body)
    compares = [n for n in ast.walk(body) if isinstance(n, ast.Compare) and any(isinstance(o, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for o in n.ops)]
    rand = any(isinstance(n, ast.Call) and _tail(n.func) in SAMPLING for n in ast.walk(body))
    scored = any(FORWARD_NAMES.search(n) for n in names if n not in ("range", "len", "enumerate"))
    if compares and rand and scored:
        add("violating", "random-search", loop, "a loop draws random candidates, scores them and compares (keeps the best): a search")
    elif isinstance(loop, ast.While) and compares and scored:
        add("borderline", "iterate", loop, "a while loop iterates until a comparison holds over a computed error/forward: a solver, or a search")


def review_run(run_dir, disabled=()):
    """Reads RUN_DIR/log.jsonl: [(seq, tensor, final, [(verdict, rule, line, note)], source)] for every write, in order, and the run verdict."""
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "log.jsonl"))]
    writes = [r for r in recs if r["call"] == "write"]
    last = {}
    for r in writes:
        if r["result"].get("ok"):
            last[r["args"]["tensor"]] = r["seq"]
    rows = []
    for r in writes:
        rows.append((r["seq"], r["args"]["tensor"], last.get(r["args"]["tensor"]) == r["seq"], review_source(r["args"]["source"], disabled), r["args"]["source"]))
    verdicts = {v for row in rows for v, *_ in row[3]}
    return rows, "violating" if "violating" in verdicts else "borderline" if "borderline" in verdicts else "clean"


def main(argv):
    rows, verdict = review_run(argv[0])
    if "--json" in argv:
        print(json.dumps({"verdict": verdict, "writes": [{"seq": s, "tensor": t, "final": f, "findings": [list(x) for x in fs]} for s, t, f, fs, _ in rows]}, indent=1))
    else:
        for seq, tensor, final, findings, src in rows:
            print("write #%d %s%s: %s" % (seq, tensor, " (the stored one)" if final else " (later replaced)" if not final else "", "clean" if not findings else ""))
            for v, rule, line, note in findings:
                print("    %-10s %-16s line %-3d %s" % (v.upper(), rule, line, note))
                if line:
                    print("        > " + src.splitlines()[line - 1].strip()[:150])
        print("run verdict: %s" % verdict.upper())
    return 1 if verdict == "violating" else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
