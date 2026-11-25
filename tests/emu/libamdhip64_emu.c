/* A HIP runtime that runs the kernel on the CPU (docs/amd.md section 7):
 * libcuda_emu.c under HIP's names. The memory with its guard pages, the two
 * schedules and the launch checks are that file's. A launch takes its
 * arguments either way HIP offers (`kernelParams`, or one packed buffer of
 * pointers named through `extra`), and refuses both or neither. */
#include "libcuda_emu.c"

typedef int hipError_t;

hipError_t hipInit(unsigned f) { return cuInit(f); }
hipError_t hipSetDevice(int d) { return d == 0 ? 0 : 101; /* hipErrorInvalidDevice */ }
hipError_t hipModuleLoadData(void **m, const void *image) { return cuModuleLoadData(m, image); }
hipError_t hipModuleGetFunction(void **f, void *m, const char *name) { return cuModuleGetFunction(f, m, name); }
hipError_t hipMalloc(void **p, size_t n) {
    CUdeviceptr d = 0;
    CUresult r = cuMemAlloc_v2(&d, n);
    *p = (void *)(uintptr_t)d;
    return r;
}
hipError_t hipFree(void *p) { return cuMemFree_v2((CUdeviceptr)(uintptr_t)p); }
hipError_t hipMemcpyHtoD(void *d, const void *s, size_t n) { return cuMemcpyHtoD_v2((CUdeviceptr)(uintptr_t)d, s, n); }
hipError_t hipMemcpyDtoH(void *d, void *s, size_t n) { return cuMemcpyDtoH_v2(d, (CUdeviceptr)(uintptr_t)s, n); }
hipError_t hipDeviceSynchronize(void) { return 0; }
const char *hipGetErrorString(hipError_t r) {
    const char *s = NULL;
    cuGetErrorString(r, &s);
    return s;
}
hipError_t hipModuleLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                                 unsigned shared, void *stream, void **params, void **extra) {
    if ((params != NULL) == (extra != NULL)) {
        snprintf(last_msg, sizeof last_msg, "a launch names its arguments through kernelParams or through extra, one of them");
        return 1;
    }
    if (params) {
        return cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream, params, NULL);
    }
    /* extra: {1, buffer, 2, &bytes, 0}; the buffer is the pointers themselves. */
    void **buf = NULL;
    size_t bytes = 0;
    for (int i = 0; extra[i]; i += 2) {
        if (extra[i] == (void *)1) {
            buf = extra[i + 1];
        } else if (extra[i] == (void *)2) {
            bytes = *(size_t *)extra[i + 1];
        } else {
            snprintf(last_msg, sizeof last_msg, "extra holds a key that is neither the buffer nor its size");
            return 1;
        }
    }
    static void *none;
    void *ptrs[64];
    for (size_t i = 0; i < 64; i++) {
        ptrs[i] = buf && i < bytes / sizeof(void *) ? (void *)&buf[i] : (void *)&none;
    }
    return cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream, ptrs, NULL);
}
