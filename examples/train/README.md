# Running kernels on inputs you give them, and training with them

One example for each use of `cancho-gpu run` (`docs/run.md`). `tests/run.sh` and `tests/kernels.py` run the first three on every gate;
`scripts/check_train.py` runs the last (it needs a built cancho-opt, so it is not in `scripts/gate.sh`). Outputs below are what the commands printed.

Inputs are plain text: numbers separated by white space, row-major, exactly the parameter's element count.

## 1. A kernel on your numbers

`dy` is `1 2 / 3 4` and `w` is `1 0 2 / 0 1 3`, so `dx = dy w` is `1 2 8 / 3 4 18`:

```
$ printf '1 2\n3 4\n' > dy; printf '1 0 2\n0 1 3\n' > w
$ build/cancho-gpu run tests/chx/bwd_dx.chx r=2 n=2 k=3 --target nvidia-ada --in dy dy --in w w --out dx
1.0 2.0 8.0 3.0 4.0 18.0
```

`--target` is required when the file has a schedule for both targets (the two cut a reduction differently, so the last digit can differ).

## 2. The whole output, to check a kernel against a reference

`check --run` prints a mean, a peak and four values. `--out` prints every element, so a test can compare all of them: `tests/kernels.py` does
that for the backward kernels against a Python reference that is itself checked against finite differences, and finds the largest difference
(2e-6 for a batch of 256). Several `--out` print one line each, in the order given.

## 3. A refusal says what is wrong, and prints nothing

```
$ build/cancho-gpu run tests/chx/bwd_dx.chx r=2 n=2 k=3 --target nvidia-ada --in dy dy --out dx
error[usage]: the parameter `w` is read by the kernel: give it with --in
$ printf '1 2 3\n' > three
$ build/cancho-gpu run tests/chx/bwd_dx.chx r=2 n=2 k=3 --target nvidia-ada --in dy three --in w w --out dx
error[usage]: the file for `dy` has 3 numbers; the parameter holds 4
```

There is no fall-back to the interpreter's made-up numbers: a kernel run on them by mistake looks exactly like a kernel run on real ones.
Exit status is 1; the other refusals (an unknown or doubled name, a token that is not a number, an unreadable file) are in `tests/run.sh`.

## 4. A model trained with kernels as the objective

`lsq_objective.py` is an objective for cancho-opt's `opt`: it computes the loss and gradient of a least-squares fit with
`lsq_residual.chx` (forward) and `tests/chx/bwd_dw.chx` (backward), calling `cancho-gpu run` twice, and BFGS drives it.
`opt` starts the program by the name it is given without searching `PATH`, and gives it no environment but `PATH` and `HOME`, so the
interpreter and the two paths are absolute:

```
$ opt minimize --method bfgs --x0 0,0,0 -- $(command -v python3) examples/train/lsq_objective.py $PWD/build/cancho-gpu $PWD
{"method":"bfgs","x":[1.5065308617772533,-2.005166007077459,0.4572115109128056],"fun":0.00028100085112224424,"jac":[0.000001015418125,4.36324625e-7,-0.000001142791],"nfev":12,"njev":12,"nit":9,"converged":true,"message":"Optimization terminated successfully."}
```

Twelve evaluations, within 1.1e-5 of the normal-equations solution (`[1.50652, -2.00517, 0.45722]`), 28 to 36 ms an evaluation over three runs on the machine it
was measured on. The transposes `bwd_dw` needs are supplied by the program, because the surface has no `A^T B` product
(`docs/training.md` section 4). Not shown: anything on a GPU, a network of more than 32 parameters (`opt`'s limit), or a large batch.

```
$ OPT=<cancho-opt>/build/opt python3 scripts/check_train.py
```

It also flips the gradient's sign and requires the fit to fail.
