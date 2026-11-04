/* A CUDA driver that runs the kernel on the CPU (docs/device.md section 5b).
 * Pairs with libnvrtc_emu.c: the "PTX" it is handed is the path of a shared
 * object holding the kernel and cuda_emu.h's scheduler.
 *
 *   * Device memory is host memory, one MAP_SHARED mapping per allocation
 *     with an inaccessible page on each side and the buffer flush against the
 *     upper one: a read or write one byte past the end faults, and the fault
 *     is reported with the buffer it overran. Slack before the buffer holds a
 *     canary, checked after the launch.
 *   * A launch runs in a forked child (so a crash, a deadlock or a trap is
 *     an error, not the end of the test) and runs TWICE: schedule 0 in order
 *     with shared memory poisoned 0xFF, schedule 1 with blocks and threads
 *     shuffled by the seed and shared memory poisoned 0x00. The device memory
 *     of both must be bit for bit equal; if not, that is a race or a read of
 *     uninitialised shared memory, and the seed is printed.
 *   * The launch is checked as the driver would: block size at most 1024,
 *     grid y at most 65535, no dynamic shared memory (nothing to ask for it).
 *
 *     CANCHOGPU_EMU_SEED      the shuffle seed (default 1)
 *     CANCHOGPU_EMU_PASSES    1 or 2 (default 2)
 *     CANCHOGPU_EMU_TIMEOUT   seconds per pass before it is a hang (default 600)
 *
 * Emulator findings are lines on standard error starting `emu-warn:` (legal
 * on hardware, undefined in the guide) or `emu-race:`.
 */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include <dlfcn.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "emu_abi.h"

typedef int CUresult;
typedef unsigned long long CUdeviceptr;

#define EMU_ERR 719 /* CUDA_ERROR_LAUNCH_FAILED */
#define CANARY 0xA5

struct fn {
    char name[256];
    void (*entry)(void **);
    int (*run)(void (*)(void **), void **, const EmuLaunch *);
    size_t smem;
};
struct alloc {
    char *map;
    size_t map_len;
    char *data;     /* start of the usable pages */
    size_t data_len;
    char *ptr;      /* the buffer */
    size_t size;
    size_t padded;  /* size rounded up to 16 */
    int live;
};
static struct alloc allocs[1024];
static int n_allocs;
static char last_msg[600];

static size_t page(void) { return (size_t)sysconf(_SC_PAGESIZE); }

CUresult cuInit(unsigned f) { (void)f; return 0; }
CUresult cuDeviceGet(int *d, int i) { *d = i; return 0; }
CUresult cuDeviceGetAttribute(int *v, int attr, int d) {
    (void)d;
    *v = attr == 75 ? 8 : 9; /* sm_89 */
    return 0;
}
CUresult cuCtxCreate_v2(void **c, unsigned f, int d) { (void)f; (void)d; *c = (void *)1; return 0; }
CUresult cuCtxDestroy_v2(void *c) { (void)c; return 0; }
CUresult cuCtxSynchronize(void) { return 0; }

CUresult cuModuleLoadData(void **m, const void *ptx) {
    void *lib = dlopen((const char *)ptx, RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        snprintf(last_msg, sizeof last_msg, "dlopen: %s", dlerror());
        return 218; /* CUDA_ERROR_INVALID_PTX */
    }
    *m = lib;
    return 0;
}
CUresult cuModuleGetFunction(void **f, void *m, const char *name) {
    char sym[300];
    snprintf(sym, sizeof sym, "__emu_entry_%s", name);
    void *entry = dlsym(m, sym);
    void *run = dlsym(m, "__emu_run");
    if (!entry || !run) {
        return 500; /* CUDA_ERROR_NOT_FOUND */
    }
    struct fn *x = calloc(1, sizeof *x);
    snprintf(x->name, sizeof x->name, "%s", name);
    x->entry = (void (*)(void **))entry;
    x->run = (int (*)(void (*)(void **), void **, const EmuLaunch *))run;
    size_t (*smem)(void) = (size_t (*)(void))dlsym(m, "__emu_smem_bytes");
    x->smem = smem ? smem() : 0;
    *f = x;
    return 0;
}

CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t n) {
    if (n_allocs == 1024) {
        return 2; /* CUDA_ERROR_OUT_OF_MEMORY */
    }
    size_t pg = page();
    size_t padded = (n + 15) / 16 * 16;
    if (padded == 0) {
        padded = 16;
    }
    size_t data_len = (padded + pg - 1) / pg * pg;
    size_t len = data_len + 2 * pg;
    char *map = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
    if (map == MAP_FAILED) {
        return 2;
    }
    mprotect(map, pg, PROT_NONE);
    mprotect(map + pg + data_len, pg, PROT_NONE);
    struct alloc *a = &allocs[n_allocs++];
    a->map = map;
    a->map_len = len;
    a->data = map + pg;
    a->data_len = data_len;
    a->ptr = a->data + (data_len - padded);
    a->size = n;
    a->padded = padded;
    a->live = 1;
    memset(a->data, CANARY, data_len);
    memset(a->ptr, 0, padded);
    *p = (CUdeviceptr)(size_t)a->ptr;
    return 0;
}
static struct alloc *find(CUdeviceptr p, size_t n) {
    for (int i = 0; i < n_allocs; i++) {
        struct alloc *a = &allocs[i];
        if (a->live && p >= (CUdeviceptr)(size_t)a->ptr && p + n <= (CUdeviceptr)(size_t)a->ptr + a->size) {
            return a;
        }
    }
    return NULL;
}
CUresult cuMemFree_v2(CUdeviceptr p) {
    for (int i = 0; i < n_allocs; i++) {
        if (allocs[i].live && (CUdeviceptr)(size_t)allocs[i].ptr == p) {
            munmap(allocs[i].map, allocs[i].map_len);
            allocs[i].live = 0;
            return 0;
        }
    }
    return 1;
}
CUresult cuMemcpyHtoD_v2(CUdeviceptr d, const void *s, size_t n) {
    if (n && !find(d, n)) {
        snprintf(last_msg, sizeof last_msg, "cuMemcpyHtoD outside any allocation");
        return 1;
    }
    memcpy((void *)(size_t)d, s, n);
    return 0;
}
CUresult cuMemcpyDtoH_v2(void *d, CUdeviceptr s, size_t n) {
    if (n && !find(s, n)) {
        snprintf(last_msg, sizeof last_msg, "cuMemcpyDtoH outside any allocation");
        return 1;
    }
    memcpy(d, (void *)(size_t)s, n);
    return 0;
}
CUresult cuGetErrorString(CUresult r, const char **s) {
    static char buf[600];
    if (r == EMU_ERR || r == 218 || (r == 1 && last_msg[0])) {
        snprintf(buf, sizeof buf, "%s", last_msg);
        buf[200] = 0; /* the shim copies at most 255 */
        *s = buf;
    } else {
        *s = r == 500 ? "CUDA_ERROR_NOT_FOUND" : r == 1 ? "CUDA_ERROR_INVALID_VALUE" : "emulator error";
    }
    return 0;
}

/* ---- running a launch -------------------------------------------------- */

static const char *describe(int i) {
    static char b[64];
    snprintf(b, sizeof b, "buffer #%d (%zu bytes)", i, allocs[i].size);
    return b;
}

/* Where a faulting address is, relative to the buffers. */
static void locate(uint64_t addr, char *out, size_t cap) {
    for (int i = 0; i < n_allocs; i++) {
        struct alloc *a = &allocs[i];
        if (!a->live) continue;
        uint64_t end = (uint64_t)(size_t)a->ptr + a->size;
        if (addr >= end && addr < (uint64_t)(size_t)a->data + a->data_len + page()) {
            snprintf(out, cap, "address %#llx is %llu bytes past the end of %s", (unsigned long long)addr, (unsigned long long)(addr - end), describe(i));
            return;
        }
        if (addr < (uint64_t)(size_t)a->data && addr + page() >= (uint64_t)(size_t)a->map) {
            snprintf(out, cap, "address %#llx is before %s", (unsigned long long)addr, describe(i));
            return;
        }
    }
    snprintf(out, cap, "address %#llx is not in any buffer", (unsigned long long)addr);
}

static int run_once(struct fn *f, void **params, unsigned gx, unsigned gy, unsigned bx, uint64_t seed, int shuffle, int poison, int timeout) {
    EmuInfo *info = mmap(NULL, sizeof *info, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
    char tmpl[] = "/tmp/emu-err-XXXXXX";
    const char *td = getenv("TMPDIR");
    char path[600];
    if (td) {
        snprintf(path, sizeof path, "%s/emu-err-XXXXXX", td);
    } else {
        snprintf(path, sizeof path, "%s", tmpl);
    }
    int efd = mkstemp(path);
    fflush(NULL);
    pid_t pid = fork();
    if (pid < 0) {
        snprintf(last_msg, sizeof last_msg, "fork failed");
        return EMU_ERR;
    }
    if (pid == 0) {
        if (efd >= 0) {
            dup2(efd, 2);
        }
        alarm((unsigned)timeout);
        EmuLaunch L = {gx, gy, bx, seed, shuffle, poison, info};
        f->run(f->entry, params, &L);
        _exit(0);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    /* The child's standard error: the emulator's own warnings, forwarded. */
    char err[4096] = {0};
    if (efd >= 0) {
        ssize_t got = pread(efd, err, sizeof err - 1, 0);
        if (got > 0) {
            fputs(err, stderr);
        }
        close(efd);
        unlink(path);
    }
    int rc = 0;
    last_msg[0] = 0;
    if (info->code) {
        int c = info->code;
        char where[200];
        snprintf(where, sizeof where, "block (%u,%u) thread %u", info->blk_x, info->blk_y, info->tid);
        if (c >= 20 && c < 40) {
            int sig = c - 20;
            if (sig == SIGSEGV || sig == SIGBUS) {
                char loc[200];
                locate(info->fault_addr, loc, sizeof loc);
                snprintf(last_msg, sizeof last_msg, "illegal memory access in %s: %s", where, loc);
            } else if (sig == SIGTRAP || sig == SIGILL) {
                snprintf(last_msg, sizeof last_msg, "undefined behaviour trap (the alignment sanitizer: a misaligned access, which a GPU refuses) in %s", where);
            } else {
                snprintf(last_msg, sizeof last_msg, "signal %d in %s", sig, where);
            }
        } else {
            snprintf(last_msg, sizeof last_msg, "%s (%s)", info->msg, where);
        }
        rc = EMU_ERR;
    } else if (WIFSIGNALED(st)) {
        snprintf(last_msg, sizeof last_msg, WTERMSIG(st) == SIGALRM ? "no progress in %d s (a hang)" : "the kernel was killed by signal %d", WTERMSIG(st) == SIGALRM ? timeout : WTERMSIG(st));
        rc = EMU_ERR;
    } else if (WEXITSTATUS(st) != 0) {
        snprintf(last_msg, sizeof last_msg, "the emulator exited %d", WEXITSTATUS(st));
        rc = EMU_ERR;
    }
    munmap(info, sizeof *info);
    return rc;
}

/* The canaries on both sides of every buffer must have survived. */
static int check_canaries(void) {
    for (int i = 0; i < n_allocs; i++) {
        struct alloc *a = &allocs[i];
        if (!a->live) continue;
        for (char *p = a->data; p < a->ptr; p++) {
            if ((unsigned char)*p != CANARY) {
                snprintf(last_msg, sizeof last_msg, "a write %zd bytes before %s", a->ptr - p, describe(i));
                return EMU_ERR;
            }
        }
        for (char *p = a->ptr + a->size; p < a->ptr + a->padded; p++) {
            if ((unsigned char)*p != CANARY && a->size != a->padded) {
                /* The tail pad starts zeroed (see cuMemAlloc); only bytes past the size that changed count. */
                if (*p != 0) {
                    snprintf(last_msg, sizeof last_msg, "a write %zd bytes past the end of %s", p - (a->ptr + a->size) + 1, describe(i));
                    return EMU_ERR;
                }
            }
        }
    }
    return 0;
}

CUresult cuLaunchKernel(void *fp, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                        unsigned shared, void *stream, void **params, void **extra) {
    struct fn *f = fp;
    (void)stream; (void)extra;
    if (bx == 0 || bx > 1024 || by != 1 || bz != 1) {
        snprintf(last_msg, sizeof last_msg, "block %ux%ux%u: at most 1024 threads, and this launch is one-dimensional", bx, by, bz);
        return 1;
    }
    if (gx == 0 || gy == 0 || gz != 1 || gy > 65535) {
        snprintf(last_msg, sizeof last_msg, "grid %ux%ux%u: y is at most 65535, and z is 1 here", gx, gy, gz);
        return 1;
    }
    if (shared != 0) {
        snprintf(last_msg, sizeof last_msg, "dynamic shared memory is not emulated");
        return 1;
    }
    const char *e;
    uint64_t seed = (e = getenv("CANCHOGPU_EMU_SEED")) ? strtoull(e, NULL, 10) : 1;
    int passes = (e = getenv("CANCHOGPU_EMU_PASSES")) ? atoi(e) : 2;
    int timeout = (e = getenv("CANCHOGPU_EMU_TIMEOUT")) ? atoi(e) : 600;
    if (passes < 1) passes = 1;
    if (passes > 2) passes = 2;
    /* Snapshot, so the second schedule starts from what the first did. */
    char *snap[1024] = {0}, *first[1024] = {0};
    for (int i = 0; i < n_allocs; i++) {
        if (allocs[i].live) {
            snap[i] = malloc(allocs[i].size + 1);
            memcpy(snap[i], allocs[i].ptr, allocs[i].size);
        }
    }
    CUresult rc = 0;
    for (int pass = 0; pass < passes && !rc; pass++) {
        if (pass == 1) {
            for (int i = 0; i < n_allocs; i++) {
                if (allocs[i].live) {
                    first[i] = malloc(allocs[i].size + 1);
                    memcpy(first[i], allocs[i].ptr, allocs[i].size);
                    memcpy(allocs[i].ptr, snap[i], allocs[i].size);
                }
            }
        }
        rc = run_once(f, params, gx, gy, bx, seed, pass, pass == 0 ? 0xFF : 0x00, timeout);
        if (!rc) {
            rc = check_canaries();
        }
    }
    if (!rc && passes == 2) {
        for (int i = 0; i < n_allocs && !rc; i++) {
            if (!allocs[i].live) continue;
            size_t bad = 0, firstbad = 0;
            for (size_t k = 0; k < allocs[i].size; k++) {
                if (first[i][k] != allocs[i].ptr[k]) {
                    if (!bad) firstbad = k;
                    bad++;
                }
            }
            if (bad) {
                fprintf(stderr, "emu-race: %s differs between schedule 0 (in order, shared memory 0xFF) and schedule 1 (seed %llu, shuffled, shared memory 0x00): %zu bytes, the first at offset %zu; a race or an uninitialised __shared__ read (CANCHOGPU_EMU_SEED=%llu reproduces)\n",
                        describe(i), (unsigned long long)seed, bad, firstbad, (unsigned long long)seed);
                /* Leave schedule 0's answer, as a deterministic run would. */
            }
            memcpy(allocs[i].ptr, first[i], allocs[i].size);
        }
    }
    for (int i = 0; i < 1024; i++) {
        free(snap[i]);
        free(first[i]);
    }
    return rc;
}
