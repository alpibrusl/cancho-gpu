/* canchogpu -- the CUDA driver and NVRTC, as cancho can call them.
 *
 * docs/device.md is the design. cancho's `extern fn` passes 64-bit
 * integers and, as the last parameter, a byte slice as a pointer and a
 * length; it cannot pass a float, and a `c_ptr` cannot travel through
 * its ordinary functions. So everything the driver hands out stays here,
 * in tables, and cancho holds indices: every entry point takes
 * `int64_t`s and at most one trailing `(const char *, int64_t)` buffer,
 * and answers an `int64_t` -- a handle or a count when >= 0, a failure
 * when < 0, whose text `cgpu_error` copies out.
 *
 * The driver and NVRTC are `dlopen`ed, so
 * this builds and links anywhere and a machine without them gets an
 * answer instead of a link error. CANCHOGPU_LIBCUDA and CANCHOGPU_LIBNVRTC
 * name other libraries (the mock in tests/mock/ does).
 *
 * The same entry points drive AMD's HIP (docs/amd.md section 3.4): its
 * runtime and hipRTC have the driver's and NVRTC's shapes, so their
 * functions are loaded into the same slots under their own names
 * (`hip_open`) and the bookkeeping below does not know which it holds.
 * Where the shapes differ it asks `g.hip`: no context (HIP's is
 * deprecated), no architecture option (hipRTC takes the device's), the
 * error text as a result and not through a pointer. A device pointer is
 * `void *` there and a 64-bit integer here; both travel as one integer
 * register on x86-64 and AArch64, which is every machine either runs on.
 * CANCHOGPU_LIBHIP and CANCHOGPU_LIBHIPRTC name the libraries.
 *
 * Read from ROCm's documentation, checked against the mock, never run
 * against the real libraries (docs/amd.md section 1).
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef unsigned long long CUdeviceptr;
typedef int nvrtcResult;
typedef void *nvrtcProgram;

#define MAX_MODULES 64
#define MAX_FUNCTIONS 256
#define MAX_BUFFERS 256
#define MAX_ARGS 64
/* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR */
#define ATTR_CC_MAJOR 75
#define ATTR_CC_MINOR 76

static struct {
    void *cu_lib, *rtc_lib;
    CUresult (*cuInit)(unsigned);
    CUresult (*cuDeviceGet)(CUdevice *, int);
    CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice);
    CUresult (*cuCtxCreate_v2)(CUcontext *, unsigned, CUdevice);
    CUresult (*cuCtxDestroy_v2)(CUcontext);
    CUresult (*cuModuleLoadData)(CUmodule *, const void *);
    CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*cuMemAlloc_v2)(CUdeviceptr *, size_t);
    CUresult (*cuMemFree_v2)(CUdeviceptr);
    CUresult (*cuMemcpyHtoD_v2)(CUdeviceptr, const void *, size_t);
    CUresult (*cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t);
    CUresult (*cuCtxSynchronize)(void);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                               unsigned, unsigned, void *, void **, void **);
    CUresult (*cuGetErrorString)(CUresult, const char **);
    nvrtcResult (*nvrtcCreateProgram)(nvrtcProgram *, const char *, const char *, int,
                                      const char *const *, const char *const *);
    nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram, int, const char *const *);
    nvrtcResult (*nvrtcGetPTXSize)(nvrtcProgram, size_t *);
    nvrtcResult (*nvrtcGetPTX)(nvrtcProgram, char *);
    nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram, size_t *);
    nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram, char *);
    nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram *);
    /* HIP only. */
    int hip, probed;
    CUresult (*hipSetDevice)(int);
    const char *(*hipGetErrorString)(CUresult);
    CUcontext ctx;
    char arch[32];
    CUmodule modules[MAX_MODULES];
    int64_t n_modules;
    CUfunction functions[MAX_FUNCTIONS];
    int64_t n_functions;
    CUdeviceptr buffers[MAX_BUFFERS];
    int64_t sizes[MAX_BUFFERS];
    int64_t n_buffers;
    CUdeviceptr args[MAX_ARGS];
    int64_t n_args;
    char error[4096];
} g;

static int64_t fail(const char *what, const char *detail) {
    snprintf(g.error, sizeof g.error, "%s%s%s", what, detail ? ": " : "", detail ? detail : "");
    return -1;
}

static int64_t cu_fail(const char *what, CUresult r) {
    const char *s = NULL;
    if (g.hip && g.hipGetErrorString) {
        s = g.hipGetErrorString(r);
    } else if (g.cuGetErrorString) {
        g.cuGetErrorString(r, &s);
    }
    char detail[256];
    snprintf(detail, sizeof detail, "%s (%s %d)", s ? s : "unknown error", g.hip ? "hipError_t" : "CUresult", r);
    return fail(what, detail);
}

/* A NUL-terminated copy of a cancho byte slice. */
static char *cstr(const char *p, int64_t n) {
    char *s = malloc((size_t)n + 1);
    if (s) {
        memcpy(s, p, (size_t)n);
        s[n] = 0;
    }
    return s;
}

static void *open_first(const char *env, const char *const *names) {
    const char *over = getenv(env);
    if (over) {
        return dlopen(over, RTLD_NOW | RTLD_LOCAL);
    }
    for (; *names; names++) {
        void *h = dlopen(*names, RTLD_NOW | RTLD_LOCAL);
        if (h) {
            return h;
        }
    }
    return NULL;
}

#define SYM(lib, name)                                                      \
    do {                                                                    \
        *(void **)(&g.name) = dlsym(g.lib, #name);                         \
        if (!g.name) {                                                      \
            return fail("missing symbol", #name);                           \
        }                                                                   \
    } while (0)

/* A slot filled from a symbol of another name: HIP's, into the driver's. */
#define SYM_AS(lib, slot, name)                                             \
    do {                                                                    \
        *(void **)(&g.slot) = dlsym(g.lib, name);                           \
        if (!g.slot) {                                                      \
            return fail("missing symbol", name);                            \
        }                                                                   \
    } while (0)

static const char *const cu_names[] = {"libcuda.so.1", "libcuda.so", NULL};
static const char *const hip_names[] = {"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6",
                                        "/opt/rocm/lib/libamdhip64.so", NULL};

/* Which runtime this process drives, decided once and without starting it:
 * HIP when CANCHOGPU_LIBHIP names one; CUDA when CANCHOGPU_LIBCUDA does, or
 * when a CUDA driver opens; else HIP if its runtime opens; else CUDA, whose
 * absence `cgpu_open` then reports. */
static void probe(void) {
    if (g.probed) {
        return;
    }
    g.probed = 1;
    if (getenv("CANCHOGPU_LIBHIP")) {
        g.hip = 1;
        g.cu_lib = open_first("CANCHOGPU_LIBHIP", hip_names);
        return;
    }
    g.cu_lib = open_first("CANCHOGPU_LIBCUDA", cu_names);
    if (!g.cu_lib && !getenv("CANCHOGPU_LIBCUDA")) {
        void *h = open_first("CANCHOGPU_LIBHIP", hip_names);
        if (h) {
            g.hip = 1;
            g.cu_lib = h;
        } else {
            /* Leave the CUDA driver's reason as the last dlerror. */
            dlopen(cu_names[0], RTLD_NOW | RTLD_LOCAL);
        }
    }
}

/* 1 when the device is HIP's, 0 when it is CUDA's. Opens nothing but the library. */
int64_t cgpu_backend(void) {
    probe();
    return g.hip;
}

static int64_t hip_open(void) {
    static const char *const rtc_names[] = {"libhiprtc.so", "libhiprtc.so.7", "/opt/rocm/lib/libhiprtc.so",
                                            "libamdhip64.so", NULL};
    if (!g.cu_lib) {
        return fail("no HIP runtime", dlerror());
    }
    g.rtc_lib = open_first("CANCHOGPU_LIBHIPRTC", rtc_names);
    if (!g.rtc_lib) {
        return fail("no hipRTC", dlerror());
    }
    SYM_AS(cu_lib, cuInit, "hipInit");
    SYM_AS(cu_lib, hipSetDevice, "hipSetDevice");
    SYM_AS(cu_lib, cuModuleLoadData, "hipModuleLoadData");
    SYM_AS(cu_lib, cuModuleGetFunction, "hipModuleGetFunction");
    SYM_AS(cu_lib, cuMemAlloc_v2, "hipMalloc");
    SYM_AS(cu_lib, cuMemFree_v2, "hipFree");
    SYM_AS(cu_lib, cuMemcpyHtoD_v2, "hipMemcpyHtoD");
    SYM_AS(cu_lib, cuMemcpyDtoH_v2, "hipMemcpyDtoH");
    SYM_AS(cu_lib, cuCtxSynchronize, "hipDeviceSynchronize");
    SYM_AS(cu_lib, cuLaunchKernel, "hipModuleLaunchKernel");
    SYM_AS(cu_lib, hipGetErrorString, "hipGetErrorString");
    SYM_AS(rtc_lib, nvrtcCreateProgram, "hiprtcCreateProgram");
    SYM_AS(rtc_lib, nvrtcCompileProgram, "hiprtcCompileProgram");
    SYM_AS(rtc_lib, nvrtcGetPTXSize, "hiprtcGetCodeSize");
    SYM_AS(rtc_lib, nvrtcGetPTX, "hiprtcGetCode");
    SYM_AS(rtc_lib, nvrtcGetProgramLogSize, "hiprtcGetProgramLogSize");
    SYM_AS(rtc_lib, nvrtcGetProgramLog, "hiprtcGetProgramLog");
    SYM_AS(rtc_lib, nvrtcDestroyProgram, "hiprtcDestroyProgram");
    CUresult r = g.cuInit(0);
    if (r) {
        return cu_fail("hipInit", r);
    }
    if ((r = g.hipSetDevice(0))) {
        return cu_fail("hipSetDevice", r);
    }
    /* No context to hold: a mark that the device is open. */
    g.ctx = (CUcontext)&g;
    return 0;
}

int64_t cgpu_open(void) {
    static const char *const rtc_names[] = {"libnvrtc.so", "libnvrtc.so.12",
                                            "/usr/local/cuda/lib64/libnvrtc.so", NULL};
    if (g.ctx) {
        return 0;
    }
    probe();
    if (g.hip) {
        return hip_open();
    }
    if (!g.cu_lib) {
        return fail("no CUDA driver", dlerror());
    }
    g.rtc_lib = open_first("CANCHOGPU_LIBNVRTC", rtc_names);
    if (!g.rtc_lib) {
        return fail("no NVRTC", dlerror());
    }
    SYM(cu_lib, cuInit);
    SYM(cu_lib, cuDeviceGet);
    SYM(cu_lib, cuDeviceGetAttribute);
    SYM(cu_lib, cuCtxCreate_v2);
    SYM(cu_lib, cuCtxDestroy_v2);
    SYM(cu_lib, cuModuleLoadData);
    SYM(cu_lib, cuModuleGetFunction);
    SYM(cu_lib, cuMemAlloc_v2);
    SYM(cu_lib, cuMemFree_v2);
    SYM(cu_lib, cuMemcpyHtoD_v2);
    SYM(cu_lib, cuMemcpyDtoH_v2);
    SYM(cu_lib, cuCtxSynchronize);
    SYM(cu_lib, cuLaunchKernel);
    SYM(cu_lib, cuGetErrorString);
    SYM(rtc_lib, nvrtcCreateProgram);
    SYM(rtc_lib, nvrtcCompileProgram);
    SYM(rtc_lib, nvrtcGetPTXSize);
    SYM(rtc_lib, nvrtcGetPTX);
    SYM(rtc_lib, nvrtcGetProgramLogSize);
    SYM(rtc_lib, nvrtcGetProgramLog);
    SYM(rtc_lib, nvrtcDestroyProgram);
    CUresult r = g.cuInit(0);
    if (r) {
        return cu_fail("cuInit", r);
    }
    CUdevice dev = 0;
    if ((r = g.cuDeviceGet(&dev, 0))) {
        return cu_fail("cuDeviceGet", r);
    }
    int major = 0, minor = 0;
    if ((r = g.cuDeviceGetAttribute(&major, ATTR_CC_MAJOR, dev)) ||
        (r = g.cuDeviceGetAttribute(&minor, ATTR_CC_MINOR, dev))) {
        return cu_fail("compute capability", r);
    }
    snprintf(g.arch, sizeof g.arch, "compute_%d%d", major, minor);
    if ((r = g.cuCtxCreate_v2(&g.ctx, 0, dev))) {
        g.ctx = NULL;
        return cu_fail("cuCtxCreate", r);
    }
    return 0;
}

/* NVRTC has no include path of its own; cuda_fp16.h is in the toolkit's. The same for hipRTC and
 * hip/hip_fp16.h under ROCm's root. */
static int include_flags(char flags[][512], int max) {
    const char *cu_roots[] = {getenv("CUDA_HOME"), getenv("CUDA_PATH"), "/usr/local/cuda", "/usr", NULL};
    const char *hip_roots[] = {getenv("ROCM_PATH"), getenv("HIP_PATH"), "/opt/rocm", "/usr", NULL};
    const char **roots = g.hip ? hip_roots : cu_roots;
    int n = 0;
    for (int i = 0; i < 4 && n < max; i++) {
        if (!roots[i]) {
            continue;
        }
        char probe[512];
        snprintf(probe, sizeof probe, "%s/include/%s", roots[i], g.hip ? "hip/hip_fp16.h" : "cuda_fp16.h");
        FILE *f = fopen(probe, "r");
        if (f) {
            fclose(f);
            snprintf(flags[n++], 512, "-I%s/include", roots[i]);
        }
    }
    return n;
}

int64_t cgpu_compile(const char *src_p, int64_t src_n) {
    if (!g.ctx) {
        return fail("cgpu_compile", "the device is not open");
    }
    if (g.n_modules == MAX_MODULES) {
        return fail("cgpu_compile", "too many modules");
    }
    char *src = cstr(src_p, src_n);
    if (!src) {
        return fail("cgpu_compile", "out of memory");
    }
    nvrtcProgram prog = NULL;
    if (g.nvrtcCreateProgram(&prog, src, g.hip ? "kernel.hip" : "kernel.cu", 0, NULL, NULL)) {
        free(src);
        return fail(g.hip ? "hiprtcCreateProgram" : "nvrtcCreateProgram", NULL);
    }
    /* NVRTC is told the device's architecture. hipRTC takes the device's own, and is told one only
     * when CANCHOGPU_HIP_ARCH names it (a machine with no supported GPU needs that). */
    char arch[64];
    int n_opt = 1;
    if (!g.hip) {
        snprintf(arch, sizeof arch, "--gpu-architecture=%s", g.arch);
    } else if (getenv("CANCHOGPU_HIP_ARCH")) {
        snprintf(arch, sizeof arch, "--gpu-architecture=%s", getenv("CANCHOGPU_HIP_ARCH"));
    } else {
        n_opt = 0;
    }
    char incs[4][512];
    int n_inc = include_flags(incs, 4);
    const char *opts[5] = {arch};
    for (int i = 0; i < n_inc; i++) {
        opts[n_opt + i] = incs[i];
    }
    if (g.nvrtcCompileProgram(prog, n_opt + n_inc, opts)) {
        size_t n = 0;
        g.nvrtcGetProgramLogSize(prog, &n);
        char *log = malloc(n + 1);
        if (log) {
            g.nvrtcGetProgramLog(prog, log);
            log[n] = 0;
        }
        fail("the kernel does not compile", log ? log : "");
        free(log);
        g.nvrtcDestroyProgram(&prog);
        free(src);
        return -1;
    }
    size_t n = 0;
    g.nvrtcGetPTXSize(prog, &n);
    char *ptx = malloc(n + 1);
    if (!ptx) {
        g.nvrtcDestroyProgram(&prog);
        free(src);
        return fail("cgpu_compile", "out of memory");
    }
    g.nvrtcGetPTX(prog, ptx);
    ptx[n] = 0;
    g.nvrtcDestroyProgram(&prog);
    free(src);
    CUmodule mod = NULL;
    CUresult r = g.cuModuleLoadData(&mod, ptx);
    free(ptx);
    if (r) {
        return cu_fail(g.hip ? "hipModuleLoadData" : "cuModuleLoadData", r);
    }
    g.modules[g.n_modules] = mod;
    return g.n_modules++;
}

int64_t cgpu_function(int64_t module, const char *name_p, int64_t name_n) {
    if (module < 0 || module >= g.n_modules) {
        return fail("cgpu_function", "no such module");
    }
    if (g.n_functions == MAX_FUNCTIONS) {
        return fail("cgpu_function", "too many functions");
    }
    char *name = cstr(name_p, name_n);
    if (!name) {
        return fail("cgpu_function", "out of memory");
    }
    CUfunction f = NULL;
    CUresult r = g.cuModuleGetFunction(&f, g.modules[module], name);
    free(name);
    if (r) {
        return cu_fail(g.hip ? "hipModuleGetFunction" : "cuModuleGetFunction", r);
    }
    g.functions[g.n_functions] = f;
    return g.n_functions++;
}

int64_t cgpu_alloc(int64_t bytes) {
    if (!g.ctx) {
        return fail("cgpu_alloc", "the device is not open");
    }
    if (g.n_buffers == MAX_BUFFERS || bytes <= 0) {
        return fail("cgpu_alloc", bytes <= 0 ? "an empty buffer" : "too many buffers");
    }
    CUdeviceptr p = 0;
    CUresult r = g.cuMemAlloc_v2(&p, (size_t)bytes);
    if (r) {
        return cu_fail(g.hip ? "hipMalloc" : "cuMemAlloc", r);
    }
    g.buffers[g.n_buffers] = p;
    g.sizes[g.n_buffers] = bytes;
    return g.n_buffers++;
}

static int64_t buffer_ok(int64_t buf, int64_t n, const char *what) {
    if (buf < 0 || buf >= g.n_buffers) {
        return fail(what, "no such buffer");
    }
    if (n != g.sizes[buf]) {
        return fail(what, "the bytes are not the buffer's size");
    }
    return 0;
}

int64_t cgpu_upload(int64_t buf, const char *p, int64_t n) {
    if (buffer_ok(buf, n, "cgpu_upload")) {
        return -1;
    }
    CUresult r = g.cuMemcpyHtoD_v2(g.buffers[buf], p, (size_t)n);
    return r ? cu_fail("cuMemcpyHtoD", r) : n;
}

int64_t cgpu_download(int64_t buf, char *p, int64_t n) {
    if (buffer_ok(buf, n, "cgpu_download")) {
        return -1;
    }
    CUresult r = g.cuMemcpyDtoH_v2(p, g.buffers[buf], (size_t)n);
    return r ? cu_fail("cuMemcpyDtoH", r) : n;
}

int64_t cgpu_arg(int64_t buf) {
    if (buf < 0 || buf >= g.n_buffers) {
        return fail("cgpu_arg", "no such buffer");
    }
    if (g.n_args == MAX_ARGS) {
        return fail("cgpu_arg", "too many arguments");
    }
    g.args[g.n_args] = g.buffers[buf];
    return g.n_args++;
}

int64_t cgpu_launch(int64_t fn, int64_t gx, int64_t gy, int64_t threads) {
    if (fn < 0 || fn >= g.n_functions) {
        g.n_args = 0;
        return fail("cgpu_launch", "no such function");
    }
    /* Slots past the arguments point at a zero, so a reader that
     * expects more than were bound (the mock counts from the kernel's
     * signature) reads a null buffer instead of stack garbage. */
    static CUdeviceptr none = 0;
    void *params[MAX_ARGS];
    for (int64_t i = 0; i < MAX_ARGS; i++) {
        params[i] = i < g.n_args ? (void *)&g.args[i] : (void *)&none;
    }
    CUresult r;
    if (g.hip && getenv("CANCHOGPU_HIP_ARGS") && !strcmp(getenv("CANCHOGPU_HIP_ARGS"), "extra")) {
        /* The other way a launch takes its arguments (CUDA's too): one packed buffer, here of pointers
         * only, named through `extra`. For a hipModuleLaunchKernel that refuses `kernelParams`, which
         * one page of ROCm's documentation says and no run here has shown (docs/amd.md section 1). */
        size_t bytes = (size_t)g.n_args * sizeof g.args[0];
        void *extra[] = {(void *)1 /* HIP_LAUNCH_PARAM_BUFFER_POINTER */, g.args,
                         (void *)2 /* HIP_LAUNCH_PARAM_BUFFER_SIZE */, &bytes, (void *)0 /* _END */};
        r = g.cuLaunchKernel(g.functions[fn], (unsigned)gx, (unsigned)gy, 1, (unsigned)threads, 1, 1, 0,
                             NULL, NULL, extra);
    } else {
        r = g.cuLaunchKernel(g.functions[fn], (unsigned)gx, (unsigned)gy, 1, (unsigned)threads, 1, 1, 0,
                             NULL, params, NULL);
    }
    g.n_args = 0;
    if (r) {
        return cu_fail(g.hip ? "hipModuleLaunchKernel" : "cuLaunchKernel", r);
    }
    r = g.cuCtxSynchronize();
    return r ? cu_fail("cuCtxSynchronize", r) : 0;
}

int64_t cgpu_error(char *out, int64_t n) {
    int64_t len = (int64_t)strlen(g.error);
    if (len > n) {
        len = n;
    }
    memcpy(out, g.error, (size_t)len);
    return len;
}

int64_t cgpu_close(void) {
    if (!g.ctx) {
        return 0;
    }
    for (int64_t i = 0; i < g.n_buffers; i++) {
        g.cuMemFree_v2(g.buffers[i]);
    }
    g.n_buffers = g.n_functions = g.n_modules = g.n_args = 0;
    if (!g.hip) {
        g.cuCtxDestroy_v2(g.ctx);
    }
    g.ctx = NULL;
    return 0;
}
