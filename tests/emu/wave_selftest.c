/* The emulator's own width, tested without the compiler (tests/hip-emu.sh):
 * one hand-written HIP kernel that sums a wave with a 64-lane shuffle tree
 * and hands the sum to every lane, run through libhiprtc_emu and
 * libamdhip64_emu at 64 lanes and again at 32.
 *
 *     wave_selftest <libamdhip64_emu.so> <libhiprtc_emu.so>
 *     wave_selftest <libamdhip64.so> <libhiprtc.so> real
 *
 * With `real` the libraries are AMD's and the kernel runs once, on the card
 * (docs/amd.md slice H6): it prints the `warpSize` the card reports and the
 * sums, and passes when they are a 64-lane wave's or a 32-lane wave's, which
 * is whether a real shuffle is the one the emulator models. WAVE_HIPRTC_OPT
 * is one option handed to hipRTC (an include path, an architecture).
 *
 * At 64 every lane of a wave must hold that wave's sum. At 32 the same text
 * is a different program (the first step reaches past the wave and adds a
 * lane to itself), and the sums must be the ones that program gives: if both
 * widths gave one answer the width would not be reaching the kernel. */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *SRC =
    "#include <hip/hip_fp16.h>\n"
    "extern \"C\" __global__ void wave_sum(const float* x, float* y, float* w)\n"
    "{\n"
    "    const unsigned tid = threadIdx.x;\n"
    "    float s = x[tid];\n"
    "    for (unsigned d = 32u; d > 0; d /= 2) s = s + __shfl_down_sync(0xffffffffffffffffull, s, d);\n"
    "    y[tid] = __shfl_sync(0xffffffffffffffffull, s, 0);\n"
    "    w[tid] = (float)warpSize;\n"
    "}\n";

#define N 128
static void *hip, *rtc;
static void *sym(void *lib, const char *name) {
    void *p = dlsym(lib, name);
    if (!p) {
        fprintf(stderr, "wave_selftest: no %s\n", name);
        exit(2);
    }
    return p;
}

static int run(const char *warp, float *y, float *w) {
    int (*create)(void **, const char *, const char *, int, const char **, const char **) = sym(rtc, "hiprtcCreateProgram");
    int (*compile)(void *, int, const char **) = sym(rtc, "hiprtcCompileProgram");
    int (*logsize)(void *, size_t *) = sym(rtc, "hiprtcGetProgramLogSize");
    int (*getlog)(void *, char *) = sym(rtc, "hiprtcGetProgramLog");
    int (*codesize)(void *, size_t *) = sym(rtc, "hiprtcGetCodeSize");
    int (*code)(void *, char *) = sym(rtc, "hiprtcGetCode");
    int (*load)(void **, const void *) = sym(hip, "hipModuleLoadData");
    int (*getfn)(void **, void *, const char *) = sym(hip, "hipModuleGetFunction");
    int (*alloc)(void **, size_t) = sym(hip, "hipMalloc");
    int (*htod)(void *, const void *, size_t) = sym(hip, "hipMemcpyHtoD");
    int (*dtoh)(void *, void *, size_t) = sym(hip, "hipMemcpyDtoH");
    int (*launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **) =
        sym(hip, "hipModuleLaunchKernel");
    const char *(*errstr)(int) = sym(hip, "hipGetErrorString");
    if (warp) {
        setenv("CANCHOGPU_EMU_WARP", warp, 1);
    } else {
        unsetenv("CANCHOGPU_EMU_WARP");
    }
    void *prog = NULL, *mod = NULL, *fn = NULL;
    create(&prog, SRC, "kernel.hip", 0, NULL, NULL);
    const char *opt = getenv("WAVE_HIPRTC_OPT");
    if (compile(prog, opt ? 1 : 0, &opt)) {
        size_t n = 0;
        logsize(prog, &n);
        char *log = calloc(1, n + 1);
        getlog(prog, log);
        fprintf(stderr, "wave_selftest: the kernel did not compile: %s\n", log);
        return 1;
    }
    size_t n = 0;
    codesize(prog, &n);
    char *image = calloc(1, n + 1);
    code(prog, image);
    if (load(&mod, image) || getfn(&fn, mod, "wave_sum")) {
        fprintf(stderr, "wave_selftest: the module did not load\n");
        return 1;
    }
    float x[N];
    for (int i = 0; i < N; i++) {
        x[i] = (float)(i + 1);
    }
    void *dx, *dy, *dw;
    alloc(&dx, sizeof x);
    alloc(&dy, sizeof x);
    alloc(&dw, sizeof x);
    htod(dx, x, sizeof x);
    void *params[64];
    static void *none;
    for (int i = 0; i < 64; i++) {
        params[i] = &none;
    }
    params[0] = &dx;
    params[1] = &dy;
    params[2] = &dw;
    int r = launch(fn, 1, 1, 1, N, 1, 1, 0, NULL, params, NULL);
    if (r) {
        fprintf(stderr, "wave_selftest: the launch failed: %s\n", errstr(r));
        return 1;
    }
    dtoh(y, dy, sizeof x);
    dtoh(w, dw, sizeof x);
    return 0;
}

int main(int argc, char **argv) {
    int real = argc == 4 && !strcmp(argv[3], "real");
    if (argc != 3 && !real) {
        fprintf(stderr, "usage: wave_selftest <libamdhip64_emu.so> <libhiprtc_emu.so>\n");
        return 2;
    }
    hip = dlopen(argv[1], RTLD_NOW);
    rtc = dlopen(argv[2], RTLD_NOW);
    if (!hip || !rtc) {
        fprintf(stderr, "wave_selftest: %s\n", dlerror());
        return 2;
    }
    float y[N], w[N];
    int bad = 0;
    if (real) {
        int (*init)(unsigned) = dlsym(hip, "hipInit");
        int (*dev)(int) = dlsym(hip, "hipSetDevice");
        if ((init && init(0)) || (dev && dev(0))) {
            fprintf(stderr, "wave_selftest: the device did not open\n");
            return 1;
        }
        if (run(NULL, y, w)) {
            return 1;
        }
        printf("warpSize %g; thread 0 holds %g, thread 32 holds %g, thread 64 holds %g\n", w[0], y[0], y[32], y[64]);
        if (w[0] == 64.0f && y[0] == 2080.0f && y[32] == 2080.0f && y[64] == 6176.0f) {
            printf("a 64-lane wave, shuffled as the emulator models it\n");
            return 0;
        }
        if (w[0] == 32.0f && y[0] == 1056.0f && y[32] == 3104.0f && y[64] == 5152.0f) {
            printf("a 32-lane wave, shuffled as the emulator models it at 32\n");
            return 0;
        }
        printf("NOT what the emulator models at either width\n");
        return 1;
    }
    /* 64 lanes (the default of the HIP build): 1+...+64 = 2080, 65+...+128 = 6176. */
    if (run(NULL, y, w)) {
        return 1;
    }
    for (int i = 0; i < N; i++) {
        float want = i < 64 ? 2080.0f : 6176.0f;
        if (y[i] != want || w[i] != 64.0f) {
            fprintf(stderr, "wave_selftest: at 64 lanes thread %d holds %g (warpSize %g), not %g\n", i, y[i], w[i], want);
            bad = 1;
            break;
        }
    }
    /* 32 lanes: the step of 32 finds no lane and returns the lane's own value, so a wave's sum is
     * doubled: 2 * (32 * first + 496) for a wave starting at `first`. */
    if (run("32", y, w)) {
        return 1;
    }
    for (int i = 0; i < N; i++) {
        float first = (float)(i / 32 * 32 + 1);
        float want = 2.0f * (32.0f * first + 496.0f);
        if (y[i] != want || w[i] != 32.0f) {
            fprintf(stderr, "wave_selftest: at 32 lanes thread %d holds %g (warpSize %g), not %g\n", i, y[i], w[i], want);
            bad = 1;
            break;
        }
    }
    if (!bad) {
        printf("the emulator's wave: 64 lanes sum as one wave, 32 as two\n");
    }
    return bad;
}
