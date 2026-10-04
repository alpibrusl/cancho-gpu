/* A stand-in for the CUDA driver, for tests/device.sh (docs/device.md
 * section 5). It cannot run a kernel; it checks the plumbing. Memory is
 * host memory, so an upload then a download gives the bytes back; a
 * launch writes nothing and appends one line to $LEXGPU_MOCK_LOG:
 *
 *     launch <entry> grid <gx>x<gy> block <threads> args <n> bytes <b0>,<b1>,...
 *
 * reading exactly as many arguments as the entry declares (counted from
 * its signature in the loaded "PTX", which is the source -- see
 * libnvrtc_mock.c), and an entry the source does not define is refused,
 * as the real driver would.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;

struct fn { char name[256]; int params; };
struct block { void *p; size_t n; };
static struct block blocks[1024];
static int n_blocks;

CUresult cuInit(unsigned f) { (void)f; return 0; }
CUresult cuDeviceGet(int *d, int i) { *d = i; return 0; }
CUresult cuDeviceGetAttribute(int *v, int attr, int d) {
    (void)d;
    *v = attr == 75 ? 8 : 9;
    return 0;
}
CUresult cuCtxCreate_v2(void **c, unsigned f, int d) { (void)f; (void)d; *c = (void *)1; return 0; }
CUresult cuCtxDestroy_v2(void *c) { (void)c; return 0; }
CUresult cuModuleLoadData(void **m, const void *ptx) { *m = strdup((const char *)ptx); return 0; }
CUresult cuModuleGetFunction(void **f, void *m, const char *name) {
    char want[300];
    snprintf(want, sizeof want, "__global__ void %s(", name);
    if (!strstr((const char *)m, want)) {
        return 500; /* CUDA_ERROR_NOT_FOUND */
    }
    struct fn *x = calloc(1, sizeof *x);
    snprintf(x->name, sizeof x->name, "%s", name);
    /* How many parameters the entry declares: the commas of its list. */
    const char *open = strstr((const char *)m, want) + strlen(want);
    const char *close = strchr(open, ')');
    x->params = 1;
    for (const char *c = open; c < close; c++) {
        x->params += *c == ',';
    }
    *f = x;
    return 0;
}
CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t n) {
    void *q = calloc(1, n);
    blocks[n_blocks].p = q;
    blocks[n_blocks++].n = n;
    *p = (CUdeviceptr)(size_t)q;
    return 0;
}
CUresult cuMemFree_v2(CUdeviceptr p) { free((void *)(size_t)p); return 0; }
CUresult cuMemcpyHtoD_v2(CUdeviceptr d, const void *s, size_t n) { memcpy((void *)(size_t)d, s, n); return 0; }
CUresult cuMemcpyDtoH_v2(void *d, CUdeviceptr s, size_t n) { memcpy(d, (void *)(size_t)s, n); return 0; }
CUresult cuCtxSynchronize(void) { return 0; }
CUresult cuGetErrorString(CUresult r, const char **s) { *s = r == 500 ? "CUDA_ERROR_NOT_FOUND" : "mock error"; return 0; }

static size_t size_of(CUdeviceptr p) {
    for (int i = 0; i < n_blocks; i++) {
        if ((CUdeviceptr)(size_t)blocks[i].p == p) {
            return blocks[i].n;
        }
    }
    return 0;
}

CUresult cuLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                        unsigned bz, unsigned shared, void *stream, void **params, void **extra) {
    (void)gz; (void)by; (void)bz; (void)shared; (void)stream; (void)extra;
    const char *path = getenv("LEXGPU_MOCK_LOG");
    FILE *log = path ? fopen(path, "a") : NULL;
    if (!log) {
        return 1;
    }
    int n = ((struct fn *)f)->params;
    fprintf(log, "launch %s grid %ux%u block %u args %d bytes ", ((struct fn *)f)->name, gx, gy, bx, n);
    for (int i = 0; i < n; i++) {
        fprintf(log, "%s%zu", i ? "," : "", size_of(*(CUdeviceptr *)params[i]));
    }
    fprintf(log, "\n");
    fclose(log);
    return 0;
}
