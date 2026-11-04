// cuda_emu.h -- the CUDA the emitter writes, run on the CPU (docs/device.md
// section 5b). Included, through cuda_fp16.h and mma.h, in place of the
// toolkit's headers when tests/emu/libnvrtc_emu.c compiles a kernel.
//
// A kernel is a C++ function that every thread of a block calls. Here a
// thread is a fiber (ucontext) on one OS thread, and a block runs to
// completion before the next starts. Fibers switch only where CUDA itself
// synchronises -- __syncthreads, a warp shuffle, a wmma call -- so between two
// of those points a thread runs alone; what an unsynchronised program does
// is decided by the ORDER the scheduler picks (ascending in schedule 0,
// shuffled by a seed in schedule 1) and by what shared memory held before it
// was written (0xFF in schedule 0, 0x00 in schedule 1). Two schedules that
// leave different bytes in global memory are a race or an uninitialised read
// (libcuda_emu.c compares them).
//
// What it checks, and what it cannot, is docs/device.md section 5b.
#ifndef CUDA_EMU_H
#define CUDA_EMU_H

#define _XOPEN_SOURCE 700
#define _DARWIN_C_SOURCE 1
#define _GNU_SOURCE 1
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <utility>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <type_traits>
#include <ucontext.h>
#include <unistd.h>

#include "emu_abi.h"

#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"

typedef unsigned int uint_emu_unused_;

// ---- the qualifiers ------------------------------------------------------
#define __global__ __attribute__((visibility("default")))
#define __device__
#define __host__
#define __forceinline__ inline __attribute__((always_inline))
#define __constant__ static const
#define __align__(n) __attribute__((aligned(n)))
#ifdef __APPLE__
#define EMU_SMEM_SECTION __attribute__((section("__DATA,emu_smem")))
#else
#define EMU_SMEM_SECTION __attribute__((section("emu_smem")))
#endif
// Static __shared__: one copy per process, which is one per block because
// blocks run one at a time. It lives in its own section so the launcher can
// size it (the per-block shared memory) and poison it between blocks.
#define __shared__ static EMU_SMEM_SECTION

// ---- vector types and the builtin variables -------------------------------
// A vector load or store through a pointer that is not aligned to the vector
// is an error on a GPU ("misaligned address"); on the CPU it is silently
// fine, and clang's alignment sanitizer does not look at aggregate copies,
// so the copies here look themselves.
[[noreturn]] void emu_misaligned(const void *p, unsigned align);
#define EMU_VEC(NAME, ALIGN, ...) \
    struct alignas(ALIGN) NAME { \
        __VA_ARGS__; \
        NAME() = default; \
        NAME(const NAME &o) { if ((uintptr_t)&o % ALIGN) emu_misaligned(&o, ALIGN); memcpy((void *)this, &o, sizeof *this); } \
        NAME &operator=(const NAME &o) { \
            if ((uintptr_t)&o % ALIGN) emu_misaligned(&o, ALIGN); \
            if ((uintptr_t)this % ALIGN) emu_misaligned(this, ALIGN); \
            memcpy((void *)this, &o, sizeof *this); return *this; } \
    }
EMU_VEC(float2, 8, float x, y);
EMU_VEC(float4, 16, float x, y, z, w);
EMU_VEC(uint2, 8, unsigned x, y);
EMU_VEC(uint4, 16, unsigned x, y, z, w);
EMU_VEC(int2, 8, int x, y);
EMU_VEC(int4, 16, int x, y, z, w);
struct uint3 { unsigned x, y, z; };
struct dim3 { unsigned x, y, z; };

static uint3 threadIdx, blockIdx;
static dim3 blockDim, gridDim;
static const int warpSize = 32;

// ---- half precision -------------------------------------------------------
// IEEE binary16, round to nearest even, as cuda_fp16.h converts.
struct __half {
    _Float16 h;
    __half() = default;
    template <class T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    __half(T v) : h((_Float16)v) {}
    operator float() const { return (float)h; }
};
struct __half2 {
    __half x, y;
};
static inline __half2 __floats2half2_rn(float a, float b) {
    __half2 r;
    r.x = __half(a);
    r.y = __half(b);
    return r;
}
static inline float2 __half22float2(__half2 v) { float2 r; r.x = (float)v.x; r.y = (float)v.y; return r; }
static inline float __half2float(__half v) { return (float)v; }
static inline __half __float2half(float v) { return __half(v); }

// ---- scalar builtins ------------------------------------------------------
static inline float __int_as_float(int v) { float f; memcpy(&f, &v, 4); return f; }
static inline int __float_as_int(float v) { int i; memcpy(&i, &v, 4); return i; }
static inline unsigned __float_as_uint(float v) { unsigned i; memcpy(&i, &v, 4); return i; }
static inline float __uint_as_float(unsigned v) { float f; memcpy(&f, &v, 4); return f; }
static inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
template <class A, class B>
static inline typename std::common_type<A, B>::type min(A a, B b) {
    typedef typename std::common_type<A, B>::type C;
    C x = (C)a, y = (C)b;
    return y < x ? y : x;
}
template <class A, class B>
static inline typename std::common_type<A, B>::type max(A a, B b) {
    typedef typename std::common_type<A, B>::type C;
    C x = (C)a, y = (C)b;
    return y > x ? y : x;
}

namespace emu {

// ---- the scheduler --------------------------------------------------------
enum State { RUN, BAR, WARP, DONE };

struct Fiber;
typedef void (*CollectiveFn)(Fiber **lanes, int n);

struct Fiber {
    ucontext_t ctx;
    State st;
    int site;          // the __syncthreads line a fiber waits at
    CollectiveFn run;  // the warp collective it waits at
    unsigned mask;
    uint64_t key;      // what the collective's lanes must agree on
    uint64_t v;        // a value in, a result out
    int a1, a2;
    void *p0, *p1, *p2;
    int lane;
};

static Fiber *F;
static unsigned N;
static int cur;
static ucontext_t sched;
static EmuInfo *info;
static void (*g_entry)(void **);
static void **g_args;
static uint64_t rng_state;
static unsigned warns_seen;

static uint64_t rnd() {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void warn(unsigned kind, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void warn(unsigned kind, const char *fmt, ...) {
    if (warns_seen & (1u << kind)) {
        return;
    }
    warns_seen |= 1u << kind;
    char buf[400];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    fprintf(stderr, "emu-warn: block (%u,%u) thread %d: %s\n", blockIdx.x, blockIdx.y, cur, buf);
}

[[noreturn]] static void fail(int code, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
[[noreturn]] static void fail(int code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(info->msg, sizeof info->msg, fmt, ap);
    va_end(ap);
    info->code = code;
    info->blk_x = blockIdx.x;
    info->blk_y = blockIdx.y;
    info->tid = (unsigned)cur;
    _exit(3);
}

}  // namespace emu
[[noreturn]] void emu_misaligned(const void *p, unsigned align) {
    emu::fail(16, "misaligned address: a %u-byte vector access at %p (CUDA requires natural alignment)", align, p);
}
namespace emu {

static void to_sched() { swapcontext(&F[cur].ctx, &sched); }

// __syncthreads(): wait until every live thread of the block is here.
static inline void barrier(int site) {
    F[cur].st = BAR;
    F[cur].site = site;
    to_sched();
}

// A warp collective: park until every lane the mask names has arrived with
// the same call; `run` then does the work for the whole group at once (so
// no lane runs ahead and changes memory the others still have to read).
static inline void collective(CollectiveFn run, unsigned mask, uint64_t key) {
    Fiber &f = F[cur];
    f.st = WARP;
    f.run = run;
    f.mask = mask;
    f.key = key;
    to_sched();
}

static void fiber_main() {
    g_entry(g_args);
    F[cur].st = DONE;
}

// Make every group of lanes whose collective is complete run it.
static bool release_warps() {
    bool any = false;
    for (unsigned w = 0; w * 32 < N; w++) {
        unsigned base = w * 32, lanes = N - base < 32 ? N - base : 32;
        for (;;) {
            bool progress = false;
            for (unsigned l = 0; l < lanes; l++) {
                Fiber &f = F[base + l];
                if (f.st != WARP) {
                    continue;
                }
                unsigned mask = f.mask;
                if (!(mask >> l & 1)) {
                    cur = (int)(base + l);
                    fail(11, "a thread called a warp collective with a mask (0x%08x) that does not name its own lane %u", mask, l);
                }
                Fiber *group[32];
                int n = 0;
                bool ready = true;
                for (unsigned m = 0; m < 32; m++) {
                    if (!(mask >> m & 1)) {
                        continue;
                    }
                    if (m >= lanes) {
                        cur = (int)(base + l);
                        warn(0, "the mask 0x%08x names lane %u but the block's last warp has %u lanes (undefined in CUDA)", mask, m, lanes);
                        continue;
                    }
                    Fiber &g = F[base + m];
                    if (g.st == DONE) {
                        cur = (int)(base + l);
                        warn(1, "the mask 0x%08x names lane %u, which has exited (undefined in CUDA)", mask, m);
                        continue;
                    }
                    if (g.st != WARP || g.run != f.run || g.mask != mask) {
                        ready = false;
                        break;
                    }
                    if (g.key != f.key) {
                        cur = (int)(base + l);
                        fail(12, "the lanes of one warp collective disagree on its arguments (lane %u and lane %u)", l, m);
                    }
                    group[n++] = &g;
                }
                if (!ready) {
                    continue;
                }
                cur = (int)(base + l);
                f.run(group, n);
                for (int i = 0; i < n; i++) {
                    group[i]->st = RUN;
                }
                progress = any = true;
            }
            if (!progress) {
                break;
            }
        }
    }
    return any;
}

static const char *state_name(State s) {
    return s == RUN ? "running" : s == BAR ? "at __syncthreads" : s == WARP ? "at a warp collective" : "exited";
}

static void deadlock() {
    int bar = 0, warp = 0, done = 0;
    int first_bar = -1, first_warp = -1;
    for (unsigned i = 0; i < N; i++) {
        if (F[i].st == BAR) {
            bar++;
            if (first_bar < 0) first_bar = (int)i;
        } else if (F[i].st == WARP) {
            warp++;
            if (first_warp < 0) first_warp = (int)i;
        } else {
            done++;
        }
    }
    cur = first_warp >= 0 ? first_warp : first_bar;
    fail(10, "deadlock: %d threads wait at __syncthreads (first: thread %d, line %d), %d at a warp collective (first: thread %d), %d exited",
         bar, first_bar, first_bar >= 0 ? F[first_bar].site : 0, warp, first_warp, done);
}

static void run_block(unsigned nthreads, int shuffle, char *stacks, size_t stack_size) {
    N = nthreads;
    for (unsigned i = 0; i < N; i++) {
        Fiber &f = F[i];
        memset(&f, 0, sizeof f);
        f.lane = (int)(i % 32);
        f.st = RUN;
        getcontext(&f.ctx);
        f.ctx.uc_stack.ss_sp = stacks + i * stack_size;
        f.ctx.uc_stack.ss_size = stack_size;
        f.ctx.uc_link = &sched;
        makecontext(&f.ctx, fiber_main, 0);
    }
    unsigned *order = (unsigned *)malloc(N * sizeof *order);
    unsigned done_warned = 0;
    for (;;) {
        unsigned n = 0;
        for (unsigned i = 0; i < N; i++) {
            if (F[i].st == RUN) {
                order[n++] = i;
            }
        }
        if (shuffle) {
            for (unsigned i = n; i > 1; i--) {
                unsigned j = (unsigned)(rnd() % i);
                unsigned t = order[i - 1];
                order[i - 1] = order[j];
                order[j] = t;
            }
        }
        for (unsigned k = 0; k < n; k++) {
            cur = (int)order[k];
            threadIdx.x = order[k];
            info->tid = order[k];
            swapcontext(&sched, &F[cur].ctx);
        }
        unsigned live = 0, at_bar = 0;
        for (unsigned i = 0; i < N; i++) {
            live += F[i].st != DONE;
            at_bar += F[i].st == BAR;
        }
        if (live == 0) {
            break;
        }
        if (at_bar == live) {
            int site = -1;
            bool mixed = false;
            for (unsigned i = 0; i < N; i++) {
                if (F[i].st == BAR) {
                    if (site < 0) site = F[i].site;
                    else if (F[i].site != site) mixed = true;
                }
            }
            cur = 0;
            if (mixed) {
                warn(2, "threads reached __syncthreads at different lines (%d and another): undefined in CUDA when the barrier is in divergent code", site);
            }
            if (live != N && !done_warned) {
                done_warned = 1;
                warn(3, "__syncthreads with %u of %u threads already exited (hardware lets the barrier through; the guide calls divergent barriers undefined)", N - live, N);
            }
            for (unsigned i = 0; i < N; i++) {
                if (F[i].st == BAR) F[i].st = RUN;
            }
            continue;
        }
        if (release_warps()) {
            continue;
        }
        bool any_run = false;
        for (unsigned i = 0; i < N; i++) any_run |= F[i].st == RUN;
        if (!any_run) {
            deadlock();
        }
    }
    free(order);
}

// ---- crashes --------------------------------------------------------------
static void on_signal(int sig, siginfo_t *si, void *) {
    if (info->code == 0) {
        info->code = 20 + sig;
        info->fault_addr = (uint64_t)(uintptr_t)si->si_addr;
        info->blk_x = blockIdx.x;
        info->blk_y = blockIdx.y;
        info->tid = (unsigned)cur;
    }
    _exit(4);
}

}  // namespace emu

// ---- warp shuffles --------------------------------------------------------
namespace emu {
enum ShflKind { DOWN, UP, XOR, IDX };
template <ShflKind K>
static void shfl_run(Fiber **g, int n) {
    // Values were deposited in v; results go back in v after all are read.
    uint64_t out[32];
    for (int i = 0; i < n; i++) {
        Fiber &f = *g[i];
        int w = f.a2;
        int l = f.lane;
        int seg = l / w * w;
        int src = l;
        int d = f.a1;
        if (K == DOWN) { src = l + d; if (src >= seg + w) src = l; }
        else if (K == UP) { src = l - d; if (src < seg) src = l; }
        else if (K == XOR) { src = l ^ d; if (src >= seg + w) src = l; }
        else { src = seg + (d % w); }
        uint64_t v = f.v;
        bool found = false;
        for (int j = 0; j < n; j++) {
            if (g[j]->lane == src) { v = g[j]->v; found = true; break; }
        }
        if (!found) {
            warn(4, "a shuffle read lane %d, which is not in its mask (undefined in CUDA)", src);
        }
        out[i] = v;
    }
    for (int i = 0; i < n; i++) g[i]->v = out[i];
}
template <ShflKind K, class T>
static T shfl(unsigned mask, T v, int arg, int width) {
    static_assert(sizeof(T) <= 8, "shuffle of a wide value");
    if (width < 1 || width > 32 || (width & (width - 1))) {
        fail(13, "shuffle width %d is not a power of two up to 32", width);
    }
    Fiber &f = F[cur];
    f.v = 0;
    memcpy(&f.v, &v, sizeof(T));
    f.a1 = arg;
    f.a2 = width;
    collective(shfl_run<K>, mask, ((uint64_t)width << 32) ^ (uint64_t)K);
    T r;
    memcpy(&r, &F[cur].v, sizeof(T));
    return r;
}
}  // namespace emu
template <class T> static inline T __shfl_down_sync(unsigned m, T v, unsigned d, int w = 32) { return emu::shfl<emu::DOWN, T>(m, v, (int)d, w); }
template <class T> static inline T __shfl_up_sync(unsigned m, T v, unsigned d, int w = 32) { return emu::shfl<emu::UP, T>(m, v, (int)d, w); }
template <class T> static inline T __shfl_xor_sync(unsigned m, T v, int d, int w = 32) { return emu::shfl<emu::XOR, T>(m, v, d, w); }
template <class T> static inline T __shfl_sync(unsigned m, T v, int s, int w = 32) { return emu::shfl<emu::IDX, T>(m, v, s, w); }
#define __syncthreads() ::emu::barrier(__LINE__)
static inline void __syncwarp(unsigned = 0xffffffffu) {}

// ---- wmma -----------------------------------------------------------------
// Fragments hold their logical matrix (the hardware scatters it across the
// lanes; which lane holds what is not observable through the API). load and
// mma and store are collective: all lanes of the warp must arrive with the
// same arguments, which the rendezvous checks, and the work is done once.
namespace nvcuda {
namespace wmma {
struct row_major {};
struct col_major {};
struct matrix_a {};
struct matrix_b {};
struct accumulator {};
enum layout_t { mem_row_major, mem_col_major };

template <class Use, int M, int N, int K, class T, class L = void>
struct fragment {
    static const int rows = std::is_same<Use, matrix_b>::value ? K : M;
    static const int cols = std::is_same<Use, matrix_a>::value ? K : N;
    // The hardware spreads the tile over the lanes (8 floats a lane for 16x16);
    // here every lane holds all of it, so elementwise code (x[t] += y[t]) is
    // right and code that depends on which lane holds which element is not.
    static const int num_elements = rows * cols;
    float x[rows * cols];
};

template <class F>
static void fill_fragment(F &f, float v) {
    for (int i = 0; i < F::rows * F::cols; i++) f.x[i] = v;
}

}  // namespace wmma
}  // namespace nvcuda

namespace emu {
template <class F>
static void wmma_check_mem(const void *p, int ld, size_t elem, const char *what) {
    if (((uintptr_t)p & 31) != 0) {
        fail(14, "wmma %s: the pointer %p is not 256-bit (32-byte) aligned, which wmma requires", what, p);
    }
    if ((size_t)ld * elem % 16 != 0) {
        fail(15, "wmma %s: ldm %d is not a multiple of 16 bytes (8 halves or 4 floats), which wmma requires", what, ld);
    }
}

// Lanes of a warp that have arrived at one wmma call. `n` lanes in the group;
// the group array `g` holds them. Copy the leader's result into the others.
template <class F, class Elem, bool COL>
static void wmma_load_run(Fiber **g, int n) {
    F *lead = (F *)g[0]->p0;
    const Elem *src = (const Elem *)g[0]->p1;
    int ld = g[0]->a1;
    for (int r = 0; r < F::rows; r++)
        for (int c = 0; c < F::cols; c++)
            lead->x[r * F::cols + c] = (float)(COL ? src[(size_t)c * ld + r] : src[(size_t)r * ld + c]);
    for (int i = 1; i < n; i++) memcpy(g[i]->p0, lead, sizeof(F));
}

template <class F, class Elem, bool COL>
static void wmma_load(F &f, const Elem *p, unsigned ldm) {
    wmma_check_mem<F>(p, (int)ldm, sizeof(Elem), "load_matrix_sync");
    Fiber &me = emu::F[emu::cur];
    me.p0 = &f;
    me.p1 = (void *)p;
    me.a1 = (int)ldm;
    collective(wmma_load_run<F, Elem, COL>, 0xffffffffu, (uint64_t)(uintptr_t)p * 31 + ldm + (COL ? 7 : 0));
}

template <class F, class Elem, bool COL>
static void wmma_store_run(Fiber **g, int) {
    const F *f = (const F *)g[0]->p0;
    Elem *dst = (Elem *)g[0]->p1;
    int ld = g[0]->a1;
    for (int r = 0; r < F::rows; r++)
        for (int c = 0; c < F::cols; c++) {
            Elem v = (Elem)f->x[r * F::cols + c];
            if (COL) dst[(size_t)c * ld + r] = v; else dst[(size_t)r * ld + c] = v;
        }
}

template <class F, class Elem, bool COL>
static void wmma_store(Elem *p, const F &f, unsigned ldm) {
    wmma_check_mem<F>(p, (int)ldm, sizeof(Elem), "store_matrix_sync");
    Fiber &me = emu::F[emu::cur];
    me.p0 = (void *)&f;
    me.p1 = p;
    me.a1 = (int)ldm;
    collective(wmma_store_run<F, Elem, COL>, 0xffffffffu, (uint64_t)(uintptr_t)p * 31 + ldm + (COL ? 7 : 0));
}

template <class D, class A, class B, class C>
static void wmma_mma_run(Fiber **g, int n) {
    // Every lane holds the same tiles (they were loaded and filled by
    // collectives, or by lane-independent code), so the product is computed
    // once, from the first lane's fragments, and handed to every lane.
    const A *a = (const A *)g[0]->p1;
    const B *b = (const B *)g[0]->p2;
    const C *c = (const C *)(uintptr_t)g[0]->v;
    D out;
    for (int r = 0; r < D::rows; r++)
        for (int col = 0; col < D::cols; col++) {
            float acc = c->x[r * C::cols + col];
            for (int k = 0; k < A::cols; k++) acc = fmaf(a->x[r * A::cols + k], b->x[k * B::cols + col], acc);
            out.x[r * D::cols + col] = acc;
        }
    for (int i = 0; i < n; i++) *(D *)g[i]->p0 = out;
}
}  // namespace emu

namespace nvcuda {
namespace wmma {
// matrix_a / matrix_b: the layout is the fragment's.
template <int M, int N, int K, class T, class L>
static void load_matrix_sync(fragment<matrix_a, M, N, K, T, L> &f, const T *p, unsigned ldm) {
    emu::wmma_load<fragment<matrix_a, M, N, K, T, L>, T, std::is_same<L, col_major>::value>(f, p, ldm);
}
template <int M, int N, int K, class T, class L>
static void load_matrix_sync(fragment<matrix_b, M, N, K, T, L> &f, const T *p, unsigned ldm) {
    emu::wmma_load<fragment<matrix_b, M, N, K, T, L>, T, std::is_same<L, col_major>::value>(f, p, ldm);
}
template <int M, int N, int K, class T>
static void load_matrix_sync(fragment<accumulator, M, N, K, T> &f, const T *p, unsigned ldm, layout_t l) {
    if (l == mem_col_major) emu::wmma_load<fragment<accumulator, M, N, K, T>, T, true>(f, p, ldm);
    else emu::wmma_load<fragment<accumulator, M, N, K, T>, T, false>(f, p, ldm);
}
template <int M, int N, int K, class T>
static void store_matrix_sync(T *p, const fragment<accumulator, M, N, K, T> &f, unsigned ldm, layout_t l) {
    if (l == mem_col_major) emu::wmma_store<fragment<accumulator, M, N, K, T>, T, true>(p, f, ldm);
    else emu::wmma_store<fragment<accumulator, M, N, K, T>, T, false>(p, f, ldm);
}
template <int M, int N, int K, class T, class LA, class LB>
static void mma_sync(fragment<accumulator, M, N, K, T> &d, const fragment<matrix_a, M, N, K, __half, LA> &a,
                     const fragment<matrix_b, M, N, K, __half, LB> &b, const fragment<accumulator, M, N, K, T> &c) {
    typedef fragment<accumulator, M, N, K, T> D;
    typedef fragment<matrix_a, M, N, K, __half, LA> A;
    typedef fragment<matrix_b, M, N, K, __half, LB> B;
    emu::Fiber &me = emu::F[emu::cur];
    me.p0 = &d;
    me.p1 = (void *)&a;
    me.p2 = (void *)&b;
    me.v = (uint64_t)(uintptr_t)&c;
    emu::collective(emu::wmma_mma_run<D, A, B, D>, 0xffffffffu, 0x4d4d41ull);
}
}  // namespace wmma
}  // namespace nvcuda

// ---- the launch -----------------------------------------------------------
#ifdef __APPLE__
extern char emu_smem_start __asm("section$start$__DATA$emu_smem");
extern char emu_smem_end __asm("section$end$__DATA$emu_smem");
#else
extern char __start_emu_smem[], __stop_emu_smem[];
#endif
static inline char *emu_smem_begin() {
#ifdef __APPLE__
    return &emu_smem_start;
#else
    return __start_emu_smem;
#endif
}
static inline size_t emu_smem_size() {
#ifdef __APPLE__
    return (size_t)(&emu_smem_end - &emu_smem_start);
#else
    return (size_t)(__stop_emu_smem - __start_emu_smem);
#endif
}

// The kernel's entry as a function of the launch's argument array: every
// parameter the emitter writes is a pointer, passed as cuLaunchKernel does,
// by the address of the value.
template <class... P, size_t... I>
static void emu_call_impl(void (*f)(P...), void **a, std::index_sequence<I...>) {
    f(*(P *)a[I]...);
}
template <class... P>
static void emu_call(void (*f)(P...), void **a) {
    emu_call_impl(f, a, std::index_sequence_for<P...>{});
}
template <class... P>
static int emu_nparams(void (*)(P...)) { return (int)sizeof...(P); }

#define EMU_ENTRY(name) \
    extern "C" __attribute__((visibility("default"))) void __emu_entry_##name(void **a) { emu_call(&name, a); } \
    extern "C" __attribute__((visibility("default"))) int __emu_nparams_##name() { return emu_nparams(&name); }

extern "C" __attribute__((visibility("default"))) size_t __emu_smem_bytes() { return emu_smem_size(); }

// Run the whole grid in this process (the caller forked). Never returns on
// a fault: it records it in `info` and exits.
extern "C" __attribute__((visibility("default"))) int __emu_run(void (*entry)(void **), void **args, const EmuLaunch *L) {
    using namespace emu;
    info = L->info;
    g_entry = entry;
    g_args = args;
    rng_state = L->seed;
    stack_t ss;
    ss.ss_size = 1 << 16;
    ss.ss_sp = mmap(0, ss.ss_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    ss.ss_flags = 0;
    sigaltstack(&ss, 0);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_signal;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    int sigs[] = {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT, SIGTRAP};
    for (int s : sigs) sigaction(s, &sa, 0);
    unsigned nthreads = L->bx;
    size_t stack_size = 1 << 19;
    char *stacks = (char *)mmap(0, stack_size * nthreads, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    F = (Fiber *)calloc(nthreads, sizeof(Fiber));
    if (stacks == MAP_FAILED || !F) {
        snprintf(info->msg, sizeof info->msg, "out of memory for %u fiber stacks", nthreads);
        info->code = 5;
        _exit(3);
    }
    blockDim = dim3{L->bx, 1, 1};
    gridDim = dim3{L->gx, L->gy, 1};
    threadIdx = uint3{0, 0, 0};
    unsigned total = L->gx * L->gy;
    unsigned *border = (unsigned *)malloc(total * sizeof *border);
    for (unsigned i = 0; i < total; i++) border[i] = i;
    if (L->shuffle) {
        for (unsigned i = total; i > 1; i--) {
            unsigned j = (unsigned)(rnd() % i);
            unsigned t = border[i - 1];
            border[i - 1] = border[j];
            border[j] = t;
        }
    }
    for (unsigned k = 0; k < total; k++) {
        unsigned b = border[k];
        blockIdx = uint3{b % L->gx, b / L->gx, 0};
        memset(emu_smem_begin(), L->poison, emu_smem_size());
        run_block(nthreads, L->shuffle, stacks, stack_size);
    }
    return 0;
}

#endif
