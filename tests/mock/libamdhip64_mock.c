/* A stand-in for AMD's HIP runtime, for tests/device.sh (docs/amd.md section 3.5; the CUDA one is libcuda_mock.c, docs/device.md
 * section 5). It cannot run a kernel; it checks the plumbing. Memory is host memory, so an upload then a download gives the
 * bytes back; a launch writes nothing and appends to $CANCHOGPU_MOCK_LOG:
 *
 *     launch <entry> grid <gx>x<gy> block <threads> args <n> bytes <b0>,<b1>,...
 *     sums <h0>,<h1>,...      (FNV-1a 64 of each argument buffer, hex)
 *     args-by params | extra  (which of hipModuleLaunchKernel's two ways carried them)
 *
 * reading exactly as many arguments as the entry declares (counted from its signature in the loaded "code", which is the source
 * -- see libhiprtc_mock.c). An entry the source does not define is refused, and so is a launch that names its arguments both
 * ways or neither, or whose packed buffer is not one pointer an argument.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int hipError_t;

struct fn { char name[256]; int params; };
struct block { void *p; size_t n; };
static struct block blocks[1024];
static int n_blocks;

hipError_t hipInit(unsigned f) { (void)f; return 0; }
hipError_t hipSetDevice(int d) { return d == 0 ? 0 : 101; }
hipError_t hipModuleLoadData(void **m, const void *image) { *m = strdup((const char *)image); return 0; }
hipError_t hipModuleGetFunction(void **f, void *m, const char *name) {
    char want[300];
    snprintf(want, sizeof want, "__global__ void %s(", name);
    if (!strstr((const char *)m, want)) {
        return 500; /* hipErrorNotFound */
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
hipError_t hipMalloc(void **p, size_t n) {
    void *q = calloc(1, n);
    blocks[n_blocks].p = q;
    blocks[n_blocks++].n = n;
    *p = q;
    return 0;
}
hipError_t hipFree(void *p) { free(p); return 0; }
hipError_t hipMemcpyHtoD(void *d, const void *s, size_t n) { memcpy(d, s, n); return 0; }
hipError_t hipMemcpyDtoH(void *d, void *s, size_t n) { memcpy(d, s, n); return 0; }
hipError_t hipDeviceSynchronize(void) { return 0; }
const char *hipGetErrorString(hipError_t r) { return r == 500 ? "hipErrorNotFound" : r == 1 ? "hipErrorInvalidValue" : "mock error"; }

static size_t size_of(void *p) {
    for (int i = 0; i < n_blocks; i++) {
        if (blocks[i].p == p) {
            return blocks[i].n;
        }
    }
    return 0;
}

hipError_t hipModuleLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                                 unsigned shared, void *stream, void **params, void **extra) {
    (void)gz; (void)by; (void)bz; (void)shared; (void)stream;
    const char *path = getenv("CANCHOGPU_MOCK_LOG");
    FILE *log = path ? fopen(path, "a") : NULL;
    if (!log) {
        return 1;
    }
    int n = ((struct fn *)f)->params;
    void *args[64];
    if ((params != NULL) == (extra != NULL) || n > 64) {
        fclose(log);
        return 1;
    }
    if (params) {
        for (int i = 0; i < n; i++) {
            args[i] = *(void **)params[i];
        }
    } else {
        /* {BUFFER_POINTER, buffer, BUFFER_SIZE, &bytes, END} */
        if (extra[0] != (void *)1 || extra[2] != (void *)2 || extra[4] != (void *)0 ||
            *(size_t *)extra[3] != (size_t)n * sizeof(void *)) {
            fclose(log);
            return 1;
        }
        memcpy(args, extra[1], (size_t)n * sizeof(void *));
    }
    fprintf(log, "launch %s grid %ux%u block %u args %d bytes ", ((struct fn *)f)->name, gx, gy, bx, n);
    for (int i = 0; i < n; i++) {
        fprintf(log, "%s%zu", i ? "," : "", size_of(args[i]));
    }
    fprintf(log, "\nsums ");
    for (int i = 0; i < n; i++) {
        const unsigned char *b = args[i];
        unsigned long long h = 14695981039346656037ULL;
        for (size_t k = 0; k < size_of(args[i]); k++) {
            h = (h ^ b[k]) * 1099511628211ULL;
        }
        fprintf(log, "%s%016llx", i ? "," : "", h);
    }
    fprintf(log, "\nargs-by %s\n", params ? "params" : "extra");
    fclose(log);
    return 0;
}
