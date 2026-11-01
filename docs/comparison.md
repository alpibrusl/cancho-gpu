# Compared with OpenNN

> **Status: measured here, on a CPU, for a forward pass only.** Nothing in this
> document is about a GPU, about training speed, or about a model larger than the
> one below. Marks: **[measured]** (with the command), **[read]** (from the other
> project's source or documents, not run) and **[open]**. The rules this follows
> (differences and not a score, a column for where this project is worse, no claim
> about speed that was not measured here) are cancho-www's
> `docs/package-structure.md` section 2.5.

[OpenNN](https://github.com/Artelnics/opennn) (Artelnics) is a C++ neural-network
library: layers, losses, optimisers (Adam, SGD, quasi-Newton, Levenberg-Marquardt **[read: `opennn/training`]**), a CPU path and a CUDA path. cancho-gpu is a
compiler for typed tile kernels, with a model runner and, since
[`run.md`](run.md), a way to run a kernel on inputs you give it. They overlap in one
place, running a small network forward, and nowhere else.

## 1. What was compared

A two-layer network, `y = sigmoid(w2 . sigmoid(W1 x + b1) + b2)`, with 28 inputs and
64 hidden units, the shape of the first layers of OpenNN's own `dense` benchmark
**[read: `benchmarks/README.md`]**, over a batch of 1 and of 256 rows. Both sides read
the same weights and inputs from the same text files, and a binary64 reference
computes the same network.

* **OpenNN:** `bench/opennn/forward_opennn.cpp`, two `opennn::Dense` layers with a
  sigmoid, `Network::calculate_outputs`, timed in process after 20 dropped calls.
* **cancho-gpu:** `bench/opennn/dense_sigmoid.chx` run twice through
  `cancho-gpu run` (once per layer), on the **CPU interpreter**.

The data are multiples of 1/1024, so every number is exact in f32 and in the text.

## 2. Measured

One machine: 4 vCPU Intel Xeon (a cloud container), Linux 6.18, g++ 13.3, cmake 3.28,
load average about 0.7 at the start of each run, nothing else running.
Three invocations of `python3 bench/opennn/compare.py --opennn build/forward_opennn
--gpu build/cancho-gpu --reps 500 --runs 7`; the table gives the range of the three.
**[measured]**

| | batch 1 | batch 256 |
|---|---|---|
| largest error of OpenNN against binary64 | 2.9e-9 | 4.3e-8 |
| largest error of cancho-gpu against binary64 | 6.3e-8 | 1.35e-7 |
| largest difference between the two | 6.0e-8 | 1.36e-7 |
| OpenNN `calculate_outputs`, 1 thread, median of 500 calls | 2.5 us | 31.2 to 31.5 us |
| OpenNN, 4 threads, median of 500 calls | 2.4 to 3.9 us | 31.1 to 32.0 us |
| cancho-gpu, two `cancho-gpu run` processes, CPU interpreter, median of 7 | 4 ms | 161 to 164 ms |

**Numerics.** Both are within 1.4e-7 of binary64 on outputs near 0.5, where one f32
unit in the last place is 6e-8; they agree with each other to the same bound. That is
what two f32 evaluations of one network with different summation orders look like. It
is not evidence that either is the better numerics, and nothing here says how either
behaves on a trained network. OpenNN's smaller error is a measurement and has no
explanation in this document **[open]**.

**Timing: no ratio is claimed.** The two timing rows measure different things. OpenNN's
is a warm in-process call to optimised C++. The cancho-gpu row is the wall time of two
processes, each starting, parsing the kernel, checking it and interpreting it with
every f32 operation emulated in integers (`design.md` section 10), so it is
dominated by the start-up and the emulation. The interpreter is the reference the GPU
output is held to, not a performance path. **How fast cancho-gpu runs this network on
a GPU is not measured here**; the device programs do not take input yet
([`run.md`](run.md) section 5), and the CUDA path has run only against a mock driver.

## 3. Differences

| | OpenNN | cancho-gpu | Where cancho-gpu is worse |
|---|---|---|---|
| **What it is** | a library of layers (dense, LSTM, CNN, transformer, others), losses and optimisers **[read: its `opennn/network`, `benchmarks/`]** | a compiler for tile kernels with its own checker, plus runners for two language models | no layer library, no losses, no optimiser: a network is written kernel by kernel |
| **Training** | training and inference throughput are benchmarked against PyTorch **[read: `benchmarks/README.md`]**; not run here | four backward kernels, a least-squares fit through cancho-opt's BFGS on the CPU interpreter (12 evaluations, `run.md` section 6) | no autograd, no mini-batch loop, at most 32 parameters through `opt`, nothing on a GPU: [`training.md`](training.md) |
| **Activations** | ReLU, GELU, softmax, identity, sigmoid and others **[read: `activation_operator`, `dense_layer.cpp`]** | `sigmoid`, `softplus`, `exp`, `log` | no ReLU: the surface has no `max` (`training.md` section 4) |
| **GPU** | CUDA with cuDNN **[read: `CMakeLists.txt`]**; not run here, no GPU | Metal has run on Apple Silicon; CUDA against a mock driver only | CUDA has never run on hardware; on Linux and NVIDIA nothing is verified |
| **Refusals** | C++ exceptions with a message; measured here: a wrong-size parameter vector throws `runtime_error` naming both sizes | every refusal is `error[<rule>]` with a stable rule, an exit status and no output | (not a weakness: listed because it is a difference) |
| **What the program may do** | an ordinary C++ library | the compiler's authority report is committed and CI fails if it widens | |
| **Getting the weights in** | the parameter buffer is padded to 64-byte tensors: 8 parameters occupy a 32-float buffer, and `set_parameters` takes the padded buffer **[measured: a probe, see section 5]**; per-layer views give each tensor | a text file per tensor, exactly the parameter's size | |
| **Build** | C++20, OpenMP, Eigen 5.0.1, zlib and libjpeg-turbo fetched at configure time **[read]**; one download was refused here, section 5 | the cancho compiler at one pinned commit, links only libc | a Rust toolchain is needed to build that compiler |
| **Licence** | AGPL-3.0-only **[read: file headers]** | EUPL-1.2 | |

## 4. What was not measured

* Any GPU, for either side: OpenNN's CUDA path, cancho-gpu on Metal or CUDA for this
  network.
* Training, for either side, speed or quality.
* Larger or other shapes: OpenNN's own `dense` benchmark is 28 inputs, two layers of
  1024 and a sigmoid output over a real dataset; this is 28, 64 and 1, on made-up numbers.
* Other machines, other compilers, `-march` other than `native`, OpenNN with TBB or
  oneDNN (neither was found here), energy, memory.
* Whether OpenNN's numbers change with the patch of section 5. The dense forward path
  does not use the files removed, but that was not tested.

## 5. The OpenNN build, and one patch

OpenNN `ad4dd07` (version 26.0.3), configured with
`-DCMAKE_BUILD_TYPE=Release -DOpenNN_DISABLE_CUDA=ON -DOpenNN_WITH_JPEG=OFF
-DOpenNN_INSTALL=OFF -DOpenNN_BUILD_TESTS=OFF -DOpenNN_BUILD_EXAMPLES=OFF
-DOpenNN_BUILD_VISION=OFF`; its defaults were kept otherwise (`NATIVE` CPU target,
link-time optimisation on, OpenMP). The test program is compiled with `-O3
-march=native -fopenmp` (`bench/opennn/build.sh`).

**The patch.** `OpenNN_WITH_JPEG` does not exist upstream. This session's network
policy refused the download of the libjpeg-turbo source (HTTP 403 from the proxy), and
OpenNN fetches it at configure time for its image loaders. The six-line patch
(`bench/opennn/opennn-no-jpeg.patch`) adds the option, default ON, and when it is OFF
skips that download and four image-loading sources
(`image_processing`, `image_dataset`, `yolo_dataset`, `yolo_dataset_io`). Nothing else
of OpenNN was changed.

**The probe.** To learn the parameter layout, a one-layer network with an identity
activation had its parameter buffer set to each unit vector in turn. It reported 8
parameters (`get_parameters_number`) and a 32-float buffer; the bias started at float 0
and the weights, `[inputs, outputs]` row-major, at float 16. The harness then reads
each layer's parameter views and checks their sizes at run time instead of assuming
these offsets, and the agreement with binary64 above is the check that the weights
landed where the forward pass reads them.

## 6. Reproducing

```
git clone https://github.com/Artelnics/opennn && cd opennn && git checkout ad4dd07
patch -p1 < ../cancho-gpu/bench/opennn/opennn-no-jpeg.patch   # only if libjpeg-turbo cannot be fetched
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenNN_DISABLE_CUDA=ON -DOpenNN_WITH_JPEG=OFF \
  -DOpenNN_INSTALL=OFF -DOpenNN_BUILD_TESTS=OFF -DOpenNN_BUILD_EXAMPLES=OFF -DOpenNN_BUILD_VISION=OFF
cmake --build build --target opennn --parallel 4
cd ../cancho-gpu
OPENNN_SRC=../opennn OPENNN_BUILD=../opennn/build bench/opennn/build.sh
python3 bench/opennn/compare.py --opennn build/forward_opennn --gpu build/cancho-gpu --reps 500 --runs 7
```

It writes `bench/opennn/results.json` with the machine and the numbers of the run.
