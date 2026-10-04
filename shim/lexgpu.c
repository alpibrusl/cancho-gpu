/* lexgpu -- the CUDA driver and NVRTC, as lex-sys can call them.
 *
 * docs/device.md is the design. lex-sys's `extern fn` passes 64-bit
 * integers and, as the last parameter, a byte slice as a pointer and a
 * length; it cannot pass a float, and a `c_ptr` cannot travel through
 * its ordinary functions. So everything the driver hands out stays here,
 * in tables, and lex-sys holds indices: every entry point takes
 * `int64_t`s and at most one trailing `(const char *, int64_t)` buffer,
 * and answers an `int64_t` -- a handle or a count when >= 0, a failure
 * when < 0, whose text `lxg_error` copies out.
 *
 * The driver and NVRTC are `dlopen`ed, as lex-gpu's `lex-cuda` does, so
 * this builds and links anywhere and a machine without them gets an
 * answer instead of a link error. LEXGPU_LIBCUDA and LEXGPU_LIBNVRTC
 * name other libraries (the mock in tests/mock/ does).
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
    if (g.cuGetErrorString) {
        g.cuGetErrorString(r, &s);
    }
    char detail[256];
    snprintf(detail, sizeof detail, "%s (CUresult %d)", s ? s : "unknown error", r);
    return fail(what, detail);
}

/* A NUL-terminated copy of a lex-sys byte slice. */
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

int64_t lxg_open(void) {
    static const char *const cu_names[] = {"libcuda.so.1", "libcuda.so", NULL};
    static const char *const rtc_names[] = {"libnvrtc.so", "libnvrtc.so.12",
                                            "/usr/local/cuda/lib64/libnvrtc.so", NULL};
    if (g.ctx) {
        return 0;
    }
    g.cu_lib = open_first("LEXGPU_LIBCUDA", cu_names);
    if (!g.cu_lib) {
        return fail("no CUDA driver", dlerror());
    }
    g.rtc_lib = open_first("LEXGPU_LIBNVRTC", rtc_names);
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

/* NVRTC has no include path of its own; cuda_fp16.h is in the toolkit's. */
static int include_flags(char flags[][512], int max) {
    const char *roots[] = {getenv("CUDA_HOME"), getenv("CUDA_PATH"), "/usr/local/cuda", "/usr", NULL};
    int n = 0;
    for (int i = 0; i < 4 && n < max; i++) {
        if (!roots[i]) {
            continue;
        }
        char probe[512];
        snprintf(probe, sizeof probe, "%s/include/cuda_fp16.h", roots[i]);
        FILE *f = fopen(probe, "r");
        if (f) {
            fclose(f);
            snprintf(flags[n++], 512, "-I%s/include", roots[i]);
        }
    }
    return n;
}

int64_t lxg_compile(const char *src_p, int64_t src_n) {
    if (!g.ctx) {
        return fail("lxg_compile", "the device is not open");
    }
    if (g.n_modules == MAX_MODULES) {
        return fail("lxg_compile", "too many modules");
    }
    char *src = cstr(src_p, src_n);
    if (!src) {
        return fail("lxg_compile", "out of memory");
    }
    nvrtcProgram prog = NULL;
    if (g.nvrtcCreateProgram(&prog, src, "kernel.cu", 0, NULL, NULL)) {
        free(src);
        return fail("nvrtcCreateProgram", NULL);
    }
    char arch[64];
    snprintf(arch, sizeof arch, "--gpu-architecture=%s", g.arch);
    char incs[4][512];
    int n_inc = include_flags(incs, 4);
    const char *opts[5] = {arch};
    for (int i = 0; i < n_inc; i++) {
        opts[1 + i] = incs[i];
    }
    if (g.nvrtcCompileProgram(prog, 1 + n_inc, opts)) {
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
        return fail("lxg_compile", "out of memory");
    }
    g.nvrtcGetPTX(prog, ptx);
    ptx[n] = 0;
    g.nvrtcDestroyProgram(&prog);
    free(src);
    CUmodule mod = NULL;
    CUresult r = g.cuModuleLoadData(&mod, ptx);
    free(ptx);
    if (r) {
        return cu_fail("cuModuleLoadData", r);
    }
    g.modules[g.n_modules] = mod;
    return g.n_modules++;
}

int64_t lxg_function(int64_t module, const char *name_p, int64_t name_n) {
    if (module < 0 || module >= g.n_modules) {
        return fail("lxg_function", "no such module");
    }
    if (g.n_functions == MAX_FUNCTIONS) {
        return fail("lxg_function", "too many functions");
    }
    char *name = cstr(name_p, name_n);
    if (!name) {
        return fail("lxg_function", "out of memory");
    }
    CUfunction f = NULL;
    CUresult r = g.cuModuleGetFunction(&f, g.modules[module], name);
    free(name);
    if (r) {
        return cu_fail("cuModuleGetFunction", r);
    }
    g.functions[g.n_functions] = f;
    return g.n_functions++;
}

int64_t lxg_alloc(int64_t bytes) {
    if (!g.ctx) {
        return fail("lxg_alloc", "the device is not open");
    }
    if (g.n_buffers == MAX_BUFFERS || bytes <= 0) {
        return fail("lxg_alloc", bytes <= 0 ? "an empty buffer" : "too many buffers");
    }
    CUdeviceptr p = 0;
    CUresult r = g.cuMemAlloc_v2(&p, (size_t)bytes);
    if (r) {
        return cu_fail("cuMemAlloc", r);
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

int64_t lxg_upload(int64_t buf, const char *p, int64_t n) {
    if (buffer_ok(buf, n, "lxg_upload")) {
        return -1;
    }
    CUresult r = g.cuMemcpyHtoD_v2(g.buffers[buf], p, (size_t)n);
    return r ? cu_fail("cuMemcpyHtoD", r) : n;
}

int64_t lxg_download(int64_t buf, char *p, int64_t n) {
    if (buffer_ok(buf, n, "lxg_download")) {
        return -1;
    }
    CUresult r = g.cuMemcpyDtoH_v2(p, g.buffers[buf], (size_t)n);
    return r ? cu_fail("cuMemcpyDtoH", r) : n;
}

int64_t lxg_arg(int64_t buf) {
    if (buf < 0 || buf >= g.n_buffers) {
        return fail("lxg_arg", "no such buffer");
    }
    if (g.n_args == MAX_ARGS) {
        return fail("lxg_arg", "too many arguments");
    }
    g.args[g.n_args] = g.buffers[buf];
    return g.n_args++;
}

int64_t lxg_launch(int64_t fn, int64_t gx, int64_t gy, int64_t threads) {
    if (fn < 0 || fn >= g.n_functions) {
        g.n_args = 0;
        return fail("lxg_launch", "no such function");
    }
    /* Slots past the arguments point at a zero, so a reader that
     * expects more than were bound (the mock counts from the kernel's
     * signature) reads a null buffer instead of stack garbage. */
    static CUdeviceptr none = 0;
    void *params[MAX_ARGS];
    for (int64_t i = 0; i < MAX_ARGS; i++) {
        params[i] = i < g.n_args ? (void *)&g.args[i] : (void *)&none;
    }
    CUresult r = g.cuLaunchKernel(g.functions[fn], (unsigned)gx, (unsigned)gy, 1, (unsigned)threads,
                                  1, 1, 0, NULL, params, NULL);
    g.n_args = 0;
    if (r) {
        return cu_fail("cuLaunchKernel", r);
    }
    r = g.cuCtxSynchronize();
    return r ? cu_fail("cuCtxSynchronize", r) : 0;
}

int64_t lxg_error(char *out, int64_t n) {
    int64_t len = (int64_t)strlen(g.error);
    if (len > n) {
        len = n;
    }
    memcpy(out, g.error, (size_t)len);
    return len;
}

int64_t lxg_close(void) {
    if (!g.ctx) {
        return 0;
    }
    for (int64_t i = 0; i < g.n_buffers; i++) {
        g.cuMemFree_v2(g.buffers[i]);
    }
    g.n_buffers = g.n_functions = g.n_modules = g.n_args = 0;
    g.cuCtxDestroy_v2(g.ctx);
    g.ctx = NULL;
    return 0;
}
