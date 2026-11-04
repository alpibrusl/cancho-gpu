/* An NVRTC that compiles for the CPU (docs/device.md section 5b). The CUDA
 * source goes to the system C++ compiler with tests/emu/include/ in front of
 * the toolkit's headers (cuda_fp16.h and mma.h are cuda_emu.h); the "PTX" it
 * hands back is the path of the shared object, which libcuda_emu.c dlopens.
 *
 * Errors come back in NVRTC's log format (`kernel.cu(12): error: ...`), and
 * what ptxas would refuse -- a static __shared__ over 48 KB -- is refused
 * here too, at compile time, as the real one does.
 *
 *     CANCHOGPU_EMU_CXX       the C++ compiler (default c++)
 *     CANCHOGPU_EMU_INCLUDE   tests/emu/include (default: baked in at build)
 *     CANCHOGPU_EMU_CACHE     where compiled kernels are kept, keyed by a
 *                             hash of the source and the emulator headers
 *     CANCHOGPU_EMU_MUTATE    `from=>to`: every `from` in the source becomes
 *                             `to` before compiling -- how tests/cuda-emu.sh
 *                             shows the emulator can fail on the emitted CUDA
 */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef EMU_INCLUDE_DIR
#define EMU_INCLUDE_DIR "tests/emu/include"
#endif

typedef int nvrtcResult;
struct prog {
    char *src;
    char *log;
    char *so;
};

static unsigned long long fnv(unsigned long long h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ b[i]) * 1099511628211ULL;
    }
    return h;
}

static unsigned long long fnv_file(unsigned long long h, const char *path) {
    FILE *f = fopen(path, "rb");
    char buf[4096];
    size_t n;
    if (!f) {
        return h;
    }
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) {
        h = fnv(h, buf, n);
    }
    fclose(f);
    return h;
}

nvrtcResult nvrtcCreateProgram(void **p, const char *src, const char *name, int n, const char *const *h,
                               const char *const *i) {
    (void)name; (void)n; (void)h; (void)i;
    struct prog *x = calloc(1, sizeof *x);
    if (!x) {
        return 1;
    }
    x->src = strdup(src);
    const char *mut = getenv("CANCHOGPU_EMU_MUTATE");
    const char *arrow = mut ? strstr(mut, "=>") : NULL;
    if (arrow && arrow > mut) {
        size_t fl = (size_t)(arrow - mut), tl = strlen(arrow + 2);
        char *out = malloc(strlen(src) * (tl > fl ? tl / fl + 1 : 1) + 1);
        char *w = out;
        for (const char *r = src; *r;) {
            if (strncmp(r, mut, fl) == 0) {
                memcpy(w, arrow + 2, tl);
                w += tl;
                r += fl;
            } else {
                *w++ = *r++;
            }
        }
        *w = 0;
        free(x->src);
        x->src = out;
    }
    x->log = strdup("");
    *p = x;
    return 0;
}

static void say(struct prog *x, const char *line) {
    size_t a = strlen(x->log), b = strlen(line);
    x->log = realloc(x->log, a + b + 2);
    memcpy(x->log + a, line, b);
    x->log[a + b] = '\n';
    x->log[a + b + 1] = 0;
}

/* Every `__global__ void NAME(` of the source. */
static char *entries(const char *src) {
    size_t cap = 256, len = 0;
    char *out = malloc(cap);
    out[0] = 0;
    const char *key = "__global__ void ";
    for (const char *p = src; (p = strstr(p, key)); p += strlen(key)) {
        const char *s = p + strlen(key);
        const char *e = s;
        while ((*e >= 'a' && *e <= 'z') || (*e >= 'A' && *e <= 'Z') || (*e >= '0' && *e <= '9') || *e == '_') {
            e++;
        }
        if (e == s || len + (size_t)(e - s) + 32 > cap) {
            cap = cap * 2 + (size_t)(e - s) + 32;
            out = realloc(out, cap);
        }
        len += (size_t)sprintf(out + len, "EMU_ENTRY(%.*s)\n", (int)(e - s), s);
    }
    return out;
}

/* clang's `path:LINE:COL: error: text` as NVRTC's `kernel.cu(LINE): error: text`. */
static void translate(struct prog *x, const char *raw) {
    char *copy = strdup(raw);
    for (char *line = strtok(copy, "\n"); line; line = strtok(NULL, "\n")) {
        const char *e = strstr(line, ": error: ");
        const char *w = strstr(line, ": warning: ");
        const char *at = e ? e : w;
        if (!at) {
            if (strstr(line, "fatal error:") || strstr(line, "ld: ") || strstr(line, "error:")) {
                say(x, line);
            }
            continue;
        }
        /* walk back over `:COL` and `:LINE` */
        const char *c = at;
        int colons = 0;
        while (c > line && colons < 2) {
            c--;
            if (*c == ':') {
                colons++;
            }
        }
        int ln = 0;
        if (colons == 2) {
            ln = atoi(c + 1);
        }
        char out[1024];
        snprintf(out, sizeof out, "kernel.cu(%d): %s", ln, at + 2);
        say(x, out);
    }
    free(copy);
}

nvrtcResult nvrtcCompileProgram(void *p, int n, const char *const *opts) {
    struct prog *x = p;
    for (int i = 0; i < n; i++) {
        const char *log = getenv("CANCHOGPU_MOCK_LOG");
        FILE *f = log ? fopen(log, "a") : NULL;
        if (f) {
            fprintf(f, "%s%s%s", i ? " " : "compile ", opts[i], i == n - 1 ? "\n" : "");
            fclose(f);
        }
    }
    const char *inc = getenv("CANCHOGPU_EMU_INCLUDE");
    if (!inc) {
        inc = EMU_INCLUDE_DIR;
    }
    const char *cxx = getenv("CANCHOGPU_EMU_CXX");
    if (!cxx) {
        cxx = "c++";
    }
    const char *tmp = getenv("TMPDIR");
    char dir[600];
    const char *cache = getenv("CANCHOGPU_EMU_CACHE");
    if (cache) {
        snprintf(dir, sizeof dir, "%s", cache);
    } else {
        snprintf(dir, sizeof dir, "%s/cancho-cuda-emu", tmp ? tmp : "/tmp");
    }
    mkdir(dir, 0700);
    char *ents = entries(x->src);
    /* The source, the generated entries, the headers and the compiler's name
     * decide the object; the same ones give the same file. */
    unsigned long long h = 14695981039346656037ULL;
    h = fnv(h, x->src, strlen(x->src));
    h = fnv(h, ents, strlen(ents));
    h = fnv(h, cxx, strlen(cxx));
    char path[900];
    snprintf(path, sizeof path, "%s/cuda_emu.h", inc);
    /* include/ holds forwarding headers; the real one is its parent's. */
    snprintf(path, sizeof path, "%s/../cuda_emu.h", inc);
    h = fnv_file(h, path);
    snprintf(path, sizeof path, "%s/../emu_abi.h", inc);
    h = fnv_file(h, path);
    char so[900], cu[900], logf[900];
    snprintf(so, sizeof so, "%s/k%016llx.so", dir, h);
    snprintf(cu, sizeof cu, "%s/k%016llx.cu", dir, h);
    snprintf(logf, sizeof logf, "%s/k%016llx.log.%d", dir, h, (int)getpid());
    if (access(so, R_OK) != 0) {
        FILE *f = fopen(cu, "w");
        if (!f) {
            say(x, "kernel.cu(0): error: cannot write the emulator's scratch directory");
            free(ents);
            return 6;
        }
        fputs(x->src, f);
        fputs("\n", f);
        fputs(ents, f);
        fclose(f);
        char cmd[3000];
        char tmp_so[950];
        snprintf(tmp_so, sizeof tmp_so, "%s.%d", so, (int)getpid());
        /* --fmad=true is NVRTC's default: a*b+c contracts to an fma. The
         * alignment check traps (no runtime library to load) and the launcher
         * reports it as a fault. */
        snprintf(cmd, sizeof cmd,
                 "%s -std=c++17 -O1 -g0 -fno-strict-aliasing -ffp-contract=fast -fno-math-errno -shared -fPIC "
                 "-fsanitize=alignment -fsanitize-trap=alignment -Wno-everything -I'%s' -x c++ '%s' -o '%s' >'%s' 2>&1",
                 cxx, inc, cu, tmp_so, logf);
        int rc = system(cmd);
        FILE *lf = fopen(logf, "r");
        char *raw = calloc(1, 65536);
        if (lf) {
            size_t got = fread(raw, 1, 65535, lf);
            raw[got] = 0;
            fclose(lf);
        }
        unlink(logf);
        if (rc != 0) {
            translate(x, raw);
            if (x->log[0] == 0) {
                say(x, "kernel.cu(0): error: the C++ compiler failed");
                say(x, raw);
            }
            free(raw);
            free(ents);
            unlink(tmp_so);
            return 6;
        }
        free(raw);
        if (rename(tmp_so, so) != 0) {
            say(x, "kernel.cu(0): error: cannot place the compiled kernel");
            free(ents);
            return 6;
        }
    }
    free(ents);
    /* What ptxas refuses: static __shared__ over 48 KB. */
    void *lib = dlopen(so, RTLD_NOW | RTLD_LOCAL);
    if (lib) {
        size_t (*bytes)(void) = (size_t (*)(void))dlsym(lib, "__emu_smem_bytes");
        size_t used = bytes ? bytes() : 0;
        dlclose(lib);
        if (used > 48 * 1024) {
            char msg[300];
            snprintf(msg, sizeof msg, "ptxas error   : Entry function uses too much shared data (0x%zx bytes, 0xc000 max)", used);
            say(x, msg);
            return 6;
        }
    }
    x->so = strdup(so);
    return 0;
}

nvrtcResult nvrtcGetPTXSize(void *p, size_t *n) {
    struct prog *x = p;
    *n = x->so ? strlen(x->so) + 1 : 0;
    return 0;
}
nvrtcResult nvrtcGetPTX(void *p, char *out) {
    struct prog *x = p;
    if (x->so) {
        memcpy(out, x->so, strlen(x->so) + 1);
    }
    return 0;
}
nvrtcResult nvrtcGetProgramLogSize(void *p, size_t *n) { *n = strlen(((struct prog *)p)->log); return 0; }
nvrtcResult nvrtcGetProgramLog(void *p, char *out) {
    struct prog *x = p;
    memcpy(out, x->log, strlen(x->log));
    return 0;
}
nvrtcResult nvrtcDestroyProgram(void **p) {
    struct prog *x = *p;
    free(x->src);
    free(x->log);
    free(x->so);
    free(x);
    return 0;
}
